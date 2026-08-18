"""Independent heavy-validation stages for the chi-vs-nbar notebook.

Each public function can be called from its own notebook cell.  Missing QPT
points are cached independently, so long calculations can be interrupted and
resumed without rerunning the other validation stages.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qp

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - minimal environments use plain iteration
    def tqdm(iterable, **_kwargs):
        return iterable

import drive_amplitude_calibration as amplitude_calibration
import drive_calibration_qpt_analysis as qpt_analysis
import coherent_limit_comparison as coherent_bridge
import kirchhoff_paper_infidelity as kirchhoff_paper
import ms_gate_functions as mg
import model_specific_hxx_calibration as hxx_calibration
import phonon_xx_angle_analysis as phonon_angle


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    output = Path(config.get("OUTPUT_DIR", "results/chi_error_element_fit"))
    advanced = output / "advanced_publication_validation"
    drive = advanced / "hxx_drive_amplitude_calibration"
    control = advanced / "physical_control_validation"
    robustness = advanced / "parameter_robustness"
    paths = {
        "output": output,
        "advanced": advanced,
        "cptp": advanced / "cptp_projection",
        "generator": advanced / "error_generator",
        "drive": drive,
        "drive_qpt": drive / "qpt_cache",
        "fock_angle": advanced / "fock_resolved_xx_angle",
        "kirchhoff_paper": drive / "kirchhoff_paper_infidelity",
        "kirchhoff_qpt": drive / "kirchhoff_thermal_qubit_qpt",
        "kirchhoff_coherent_compare": drive / "kirchhoff_coherent_limit_comparison",
        "kirchhoff_model_calibration": drive / "model_specific_hxx_zero_calibration",
        "control": control,
        "robustness": robustness,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _simulation_params(config: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(config["SIMULATION_PARAMS"])
    params["parallel_workers"] = int(
        config.get("FAST_PROCESS_WORKERS", params.get("parallel_workers", 4))
    )
    return params


def _convention(config: Mapping[str, Any]) -> str:
    return str(config.get("ERROR_CHANNEL_CONVENTION", "undo_before_actual"))


def _cptp_options(config: Mapping[str, Any]) -> tuple[float, int]:
    return (
        float(config.get("CPTP_TOLERANCE", 1e-11)),
        int(config.get("CPTP_MAX_ITERATIONS", 5000)),
    )


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _legacy_nbar_stem(n_bar: float) -> str:
    return str(float(n_bar)).replace("-", "m").replace(".", "p")


def _compact_nbar_stem(n_bar: float) -> str:
    return f"{float(n_bar):.12g}".replace("-", "m").replace(".", "p")


def _condition_stem(name: str) -> str:
    return str(name).replace(".", "p").replace("-", "m")


def _load_generator(paths: Mapping[str, Path]) -> pd.DataFrame:
    path = paths["generator"] / "error_generator_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the cached/core workflow before this stage."
        )
    return pd.read_csv(path).sort_values("n_bar").reset_index(drop=True)


def _load_cptp_chi(paths: Mapping[str, Path], n_bar: float) -> np.ndarray:
    path = paths["cptp"] / f"cptp_chi_nbar_{_legacy_nbar_stem(n_bar)}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing baseline CPTP cache for n_bar={n_bar:g}: {path}"
        )
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["chi_trace_normalized"], dtype=complex)


def _extract(chi, config: Mapping[str, Any]) -> dict[str, Any]:
    tolerance, max_iterations = _cptp_options(config)
    return qpt_analysis.extract_xx_generator_observables(
        chi,
        cptp_tolerance=tolerance,
        cptp_max_iterations=max_iterations,
    )


def _drive_cache_path(
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
    n_bar: float,
    iteration: int,
    amplitude: float,
) -> Path:
    params = _simulation_params(config)
    signature_keys = [
        "delta",
        "rho0",
        "time_points",
        "t_gate_phys",
        "heating_rate_phys",
        "dephasing_rate_phys",
        "T2_star",
        "rayleigh_rate_phys",
        "raman_rate_phys",
        "eta",
        "use_full_order",
    ]
    payload = {
        "n_bar": float(n_bar),
        "iteration": int(iteration),
        "amplitude": float(amplitude),
        "parameters": {key: params[key] for key in signature_keys},
        "convention": _convention(config),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=_json_safe).encode("utf-8")
    ).hexdigest()[:12]
    return paths["drive_qpt"] / (
        f"hxx_feedback_i{int(iteration):02d}__nbar_"
        f"{_compact_nbar_stem(n_bar)}__{digest}.npz"
    )


def _resolve_drive_cache_path(
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
    n_bar: float,
    iteration: int,
    amplitude: float,
) -> Path:
    """Find a drive cache even if CSV float round-tripping changed its hash.

    The historical filename digest contains the full-precision amplitude.
    Reading the baseline generator through CSV can perturb the final binary
    digit and consequently produce a different digest for the same physical
    point.  Metadata matching makes cache lookup insensitive to that harmless
    serialization difference.
    """

    exact_path = _drive_cache_path(
        paths, config, n_bar, iteration, amplitude
    )
    if exact_path.exists():
        return exact_path
    pattern = (
        f"hxx_feedback_i{int(iteration):02d}__nbar_"
        f"{_compact_nbar_stem(n_bar)}__*.npz"
    )
    for candidate in sorted(paths["drive_qpt"].glob(pattern)):
        try:
            with np.load(candidate, allow_pickle=False) as data:
                cached_n_bar = float(np.asarray(data["n_bar"]).item())
                metadata = json.loads(
                    str(np.asarray(data["metadata_json"]).item())
                )
            cached_amplitude = float(metadata["A_calibrated"])
            cached_iteration = int(metadata.get("iteration", iteration))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            np.isclose(cached_n_bar, n_bar, rtol=0.0, atol=1e-12)
            and cached_iteration == int(iteration)
            and np.isclose(
                cached_amplitude, amplitude, rtol=1e-12, atol=1e-14
            )
        ):
            return candidate
    return exact_path


def _baseline_drive_rows(
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
    generator: pd.DataFrame,
    nbar_values: Iterable[float],
) -> pd.DataFrame:
    rows = []
    for n_bar in sorted({float(value) for value in nbar_values}):
        match = generator.loc[np.isclose(generator["n_bar"], n_bar)]
        if match.empty:
            continue
        baseline_chi = _load_cptp_chi(paths, n_bar)
        observables = _extract(baseline_chi, config)
        rows.append({
            "n_bar": n_bar,
            "h_XX_before_rad_per_gate": float(
                match.iloc[0]["h_XX_rad_per_gate"]
            ),
            "gamma_XX_before_per_gate": float(
                match.iloc[0]["gamma_XX_per_gate"]
            ),
            "average_infidelity_before": observables["average_infidelity"],
            "abs_chi_II_XX_before": observables["abs_chi_II_XX"],
            "chi_XX_XX_before": observables["chi_XX_XX"],
        })
    return pd.DataFrame(rows)


def _plot_drive_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    if frame.empty:
        return
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    panels = [
        (
            axes[0, 0],
            "h_XX_before_rad_per_gate",
            "h_XX_after_rad_per_gate",
            r"Hamiltonian $h_{XX}$",
            False,
        ),
        (
            axes[0, 1],
            "gamma_XX_before_per_gate",
            "gamma_XX_after_per_gate",
            r"Stochastic $\gamma_{XX}$",
            True,
        ),
        (
            axes[1, 0],
            "average_infidelity_before",
            "average_infidelity_after",
            "Average infidelity",
            True,
        ),
    ]
    for axis, before, after, title, log_scale in panels:
        axis.plot(frame["n_bar"], frame[before], "o-", label="before")
        axis.plot(frame["n_bar"], frame[after], "o-", label="after")
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.28)
        axis.legend()
    axes[1, 1].plot(frame["n_bar"], frame["A_factor"], "o-")
    axes[1, 1].set_title(r"Validated drive factor $A/A_0$")
    axes[1, 1].grid(True, alpha=0.28)
    for axis in axes.flat:
        axis.set_xlabel(r"Mean phonon number $\bar n$")
    figure.suptitle("Full-Hamiltonian QPT after hXX drive calibration", y=1.01)
    figure.tight_layout()
    figure.savefig(
        output_dir / "hxx_drive_calibration_before_after.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "hxx_drive_calibration_before_after.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def _refresh_drive_completion_artifacts(
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
    summary: pd.DataFrame,
) -> dict[str, Any]:
    """Synchronize the publication checklist with the current drive summary."""

    expected_nbars = [
        float(value)
        for value in config.get(
            "HXX_DRIVE_CALIBRATION_NBARS",
            summary.get("n_bar", pd.Series(dtype=float)).tolist(),
        )
    ]
    completed = 0
    converged = 0
    if not summary.empty:
        for n_bar in expected_nbars:
            match = summary.loc[
                np.isclose(summary["n_bar"].astype(float), n_bar)
            ]
            if match.empty:
                continue
            completed += 1
            if bool(match.iloc[-1].get("h_XX_converged", False)):
                converged += 1

    expected = len(expected_nbars)
    tolerance = float(config.get("HXX_CONVERGENCE_TOL_RAD", 2e-3))
    row = {
        "check": "hXX-derived drive calibration re-QPT",
        "status": "complete" if completed >= expected else "pending",
        "result": (
            f"{completed}/{expected} temperatures re-QPT; "
            f"{converged} with |h_XX| <= {tolerance:.1e} rad/gate"
        ),
    }

    checklist_path = paths["advanced"] / "advanced_publication_checklist.csv"
    if checklist_path.exists():
        checklist = pd.read_csv(checklist_path)
        mask = checklist["check"] == row["check"]
        if mask.any():
            for key in ("status", "result"):
                checklist.loc[mask, key] = row[key]
        else:
            checklist = pd.concat(
                [checklist, pd.DataFrame([row])], ignore_index=True
            )
        checklist.to_csv(checklist_path, index=False)
    else:
        checklist = pd.DataFrame([row])

    manifest_path = paths["advanced"] / "advanced_publication_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = manifest.setdefault("qpt_completion", {})
        completion.update({
            "hxx_drive_calibration_temperatures": int(completed),
            "hxx_drive_calibration_expected": int(expected),
            "hxx_drive_calibration_converged": int(converged),
        })
        manifest["checklist"] = checklist.to_dict(orient="records")
        manifest["generated_at_timezone"] = pd.Timestamp.now(
            tz="Asia/Tokyo"
        ).isoformat()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "completed_count": completed,
        "expected_count": expected,
        "converged_count": converged,
        "checklist_row": row,
        "checklist_path": checklist_path,
        "manifest_path": manifest_path,
    }


def run_drive_feedback_stage(
    config: Mapping[str, Any],
    nbar_values: Iterable[float],
    *,
    run_qpt: bool = False,
    force_recompute: bool = False,
    max_feedback_iterations: int | None = None,
) -> dict[str, Any]:
    """Run/load the temperature-dependent hXX amplitude-feedback QPT stage."""

    paths = _paths(config)
    params = _simulation_params(config)
    generator = _load_generator(paths)
    selected_nbars = [float(value) for value in nbar_values]
    target_angle = float(config.get("TARGET_XX_ANGLE_RAD", np.pi / 4.0))
    max_iterations = int(
        max_feedback_iterations
        if max_feedback_iterations is not None
        else config.get("HXX_MAX_FEEDBACK_ITERATIONS", 2)
    )
    convergence_tolerance = float(
        config.get("HXX_CONVERGENCE_TOL_RAD", 2e-3)
    )
    final_summary_path = (
        paths["drive"] / "hxx_drive_calibration_final_summary.csv"
    )
    iterations_path = paths["drive"] / "hxx_drive_feedback_qpt_iterations.csv"
    if not run_qpt and not force_recompute and final_summary_path.exists():
        cached_summary = pd.read_csv(final_summary_path).sort_values(
            "n_bar"
        ).reset_index(drop=True)
        has_every_requested_point = all(
            np.isclose(
                cached_summary["n_bar"].astype(float), n_bar
            ).any()
            for n_bar in selected_nbars
        )
        if has_every_requested_point:
            cached_iterations = (
                pd.read_csv(iterations_path)
                if iterations_path.exists()
                else pd.DataFrame()
            )
            status = {
                "requested_nbars": selected_nbars,
                "completed_nbars": sorted(set(selected_nbars)),
                "completed_count": len(set(selected_nbars)),
                "expected_count": len(selected_nbars),
                "pending": [],
                "run_qpt": False,
                "loaded_final_summary_cache": True,
            }
            completion = _refresh_drive_completion_artifacts(
                paths, config, cached_summary
            )
            status["publication_completion"] = completion
            return {
                "iterations": cached_iterations,
                "summary": cached_summary,
                "status": status,
            }
    max_amplitude_factor = float(config.get("HXX_MAX_AMPLITUDE_FACTOR", 1.6))
    base_amplitude = float(np.asarray(params["A"]))
    rows = []
    pending = []

    for n_bar in selected_nbars:
        baseline = generator.loc[np.isclose(generator["n_bar"], n_bar)]
        if baseline.empty:
            pending.append({"n_bar": n_bar, "reason": "missing_generator"})
            continue
        current_amplitude = base_amplitude
        current_h_xx = float(baseline.iloc[0]["h_XX_rad_per_gate"])

        for iteration in range(1, max_iterations + 1):
            next_amplitude = amplitude_calibration.amplitude_update_from_hxx(
                current_amplitude,
                current_h_xx,
                target_angle,
            )
            amplitude_factor = next_amplitude / base_amplitude
            if amplitude_factor > max_amplitude_factor:
                raise ValueError(
                    f"n_bar={n_bar:g}: A/A0={amplitude_factor:.4f} exceeds "
                    f"HXX_MAX_AMPLITUDE_FACTOR={max_amplitude_factor:.4f}"
                )
            cache_path = _resolve_drive_cache_path(
                paths,
                config,
                n_bar,
                iteration,
                next_amplitude,
            )
            if force_recompute or not cache_path.exists():
                if not run_qpt:
                    pending.append({
                        "n_bar": n_bar,
                        "iteration": iteration,
                        "A_calibrated": next_amplitude,
                        "A_factor": amplitude_factor,
                        "reason": "qpt_cache_missing",
                    })
                    break
                started = time.perf_counter()
                result = qpt_analysis.calculate_error_channel_batch(
                    [n_bar],
                    params,
                    {
                        "A": next_amplitude,
                        "parallel_workers": int(
                            config.get("FAST_PROCESS_WORKERS", 4)
                        ),
                        "show_progress": False,
                    },
                    convention=_convention(config),
                )[0]
                result["metadata"].update({
                    "calibration_method": "hxx_quadratic_feedback",
                    "iteration": iteration,
                    "input_h_XX_rad_per_gate": current_h_xx,
                    "A_baseline": base_amplitude,
                    "A_calibrated": next_amplitude,
                    "A_factor": amplitude_factor,
                    "elapsed_seconds": time.perf_counter() - started,
                })
                qpt_analysis.save_qpt_point(
                    cache_path,
                    n_bar,
                    f"hxx_feedback_iteration_{iteration}",
                    result["chi"],
                    result["metadata"],
                )

            with np.load(cache_path, allow_pickle=False) as data:
                calibrated_chi = np.asarray(
                    data["chi_trace_normalized"], dtype=complex
                )
            observables = _extract(calibrated_chi, config)
            rows.append({
                "n_bar": n_bar,
                "iteration": iteration,
                "input_h_XX_rad_per_gate": current_h_xx,
                "A_calibrated": next_amplitude,
                "A_factor": amplitude_factor,
                "cache_path": str(cache_path),
                **{
                    key: value
                    for key, value in observables.items()
                    if key != "projected_chi"
                },
            })
            current_amplitude = next_amplitude
            current_h_xx = observables["h_XX_rad_per_gate"]
            if abs(current_h_xx) <= convergence_tolerance:
                break

    existing = (
        pd.read_csv(iterations_path)
        if iterations_path.exists() and not force_recompute
        else pd.DataFrame()
    )
    iterations = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    if not iterations.empty:
        iterations = (
            iterations.sort_values(["n_bar", "iteration"])
            .drop_duplicates(["n_bar", "iteration"], keep="last")
            .reset_index(drop=True)
        )
        iterations.to_csv(iterations_path, index=False)

    if iterations.empty:
        summary = pd.DataFrame()
    else:
        latest = (
            iterations.sort_values(["n_bar", "iteration"])
            .groupby("n_bar", as_index=False)
            .tail(1)
            .rename(columns={
                "h_XX_rad_per_gate": "h_XX_after_rad_per_gate",
                "gamma_XX_per_gate": "gamma_XX_after_per_gate",
                "average_infidelity": "average_infidelity_after",
                "abs_chi_II_XX": "abs_chi_II_XX_after",
                "chi_XX_XX": "chi_XX_XX_after",
            })
        )
        baseline = _baseline_drive_rows(
            paths,
            config,
            generator,
            latest["n_bar"],
        )
        summary = (
            latest.merge(baseline, on="n_bar", how="left")
            .sort_values("n_bar")
            .reset_index(drop=True)
        )
        summary["abs_h_XX_reduction_factor"] = (
            np.abs(summary["h_XX_before_rad_per_gate"])
            / np.maximum(np.abs(summary["h_XX_after_rad_per_gate"]), 1e-15)
        )
        summary["infidelity_reduction_factor"] = (
            summary["average_infidelity_before"]
            / np.maximum(summary["average_infidelity_after"], 1e-15)
        )
        summary["h_XX_converged"] = (
            np.abs(summary["h_XX_after_rad_per_gate"])
            <= convergence_tolerance
        )
        summary.to_csv(final_summary_path, index=False)
        _plot_drive_summary(summary, paths["drive"])

    selected_completed = sorted(
        set(selected_nbars)
        & set(iterations.get("n_bar", pd.Series(dtype=float)).astype(float))
    )
    status = {
        "requested_nbars": selected_nbars,
        "completed_nbars": selected_completed,
        "completed_count": len(selected_completed),
        "expected_count": len(selected_nbars),
        "pending": pending,
        "run_qpt": bool(run_qpt),
    }
    completion = _refresh_drive_completion_artifacts(
        paths, config, summary
    )
    status["publication_completion"] = completion
    return {"iterations": iterations, "summary": summary, "status": status}


def run_fock_xx_angle_stage(
    config: Mapping[str, Any],
    reference_nbars: Iterable[float],
    *,
    thermal_tail_tolerance: float = 1e-5,
    max_fock_n: int | None = None,
    plot_max_fock_n: int = 80,
    phonon_buffer: int = 24,
    force_recompute: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Resolve the XX angle by initial Fock number and thermally average it.

    This is a noise-free conditional-branch calculation, not another QPT
    sweep.  It therefore isolates the thermal Hamiltonian mechanism and then
    compares its prediction with the existing noisy full-Hamiltonian QPT.
    """

    paths = _paths(config)
    params = _simulation_params(config)
    target_angle = float(config.get("TARGET_XX_ANGLE_RAD", np.pi / 4.0))
    selected_nbars = sorted({float(value) for value in reference_nbars})
    if not selected_nbars:
        raise ValueError("reference_nbars must contain at least one value")
    if min(selected_nbars) < 0.0:
        raise ValueError("reference_nbars must be non-negative")
    if max_fock_n is None:
        resolved_max_fock_n = max(
            phonon_angle.required_fock_cutoff(
                n_bar, thermal_tail_tolerance
            )
            for n_bar in selected_nbars
        )
    else:
        resolved_max_fock_n = int(max_fock_n)

    drive_path = paths["drive"] / "hxx_drive_calibration_final_summary.csv"
    if not drive_path.exists():
        raise FileNotFoundError(
            f"Missing {drive_path}. Run the drive calibration cell first."
        )
    drive_summary = pd.read_csv(drive_path)
    missing_drive_nbars = [
        n_bar
        for n_bar in selected_nbars
        if not np.isclose(
            drive_summary["n_bar"].astype(float), n_bar
        ).any()
    ]
    if missing_drive_nbars:
        # The main workflow may have overwritten the summary with only the
        # exact-hash hits.  Rebuild it from all existing metadata-matched QPT
        # caches before rejecting the Fock analysis request.
        rebuilt = run_drive_feedback_stage(
            config,
            config.get(
                "HXX_DRIVE_CALIBRATION_NBARS", selected_nbars
            ),
            run_qpt=False,
            force_recompute=False,
        )
        drive_summary = rebuilt["summary"]
    conditions = [{
        "condition": "baseline",
        "label": "baseline rectangular",
        "calibration_n_bar": np.nan,
        "amplitude": float(np.asarray(params["A"])),
    }]
    for n_bar in selected_nbars:
        match = drive_summary.loc[
            np.isclose(drive_summary["n_bar"].astype(float), n_bar)
        ]
        if match.empty:
            raise ValueError(
                f"No completed drive-calibration row for n_bar={n_bar:g}"
            )
        conditions.append({
            "condition": f"drive_calibrated_nbar_{_compact_nbar_stem(n_bar)}",
            "label": rf"$h_{{XX}}$ calibrated at $\bar n={n_bar:g}$",
            "calibration_n_bar": n_bar,
            "amplitude": float(match.iloc[-1]["A_calibrated"]),
        })

    cache_dir = paths["fock_angle"] / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    curve_frames = []
    iterator = tqdm(
        conditions,
        desc="Fock-resolved XX curves",
        unit="drive",
        disable=not show_progress,
    )
    for condition in iterator:
        signature = phonon_angle.fock_curve_signature(
            params,
            amplitude=condition["amplitude"],
            max_fock_n=resolved_max_fock_n,
            phonon_buffer=phonon_buffer,
            target_xx_angle_rad=target_angle,
        )
        cache_path = cache_dir / (
            f"{condition['condition']}__{signature}.npz"
        )
        if force_recompute or not cache_path.exists():
            curve = phonon_angle.calculate_fock_resolved_xx_angles(
                params,
                amplitude=condition["amplitude"],
                max_fock_n=resolved_max_fock_n,
                phonon_buffer=phonon_buffer,
                target_xx_angle_rad=target_angle,
            )
            phonon_angle.save_fock_curve(cache_path, curve)
        else:
            curve = phonon_angle.load_fock_curve(cache_path)
        curve = curve.assign(
            condition=condition["condition"],
            label=condition["label"],
            calibration_n_bar=condition["calibration_n_bar"],
            amplitude=condition["amplitude"],
            cache_path=str(cache_path),
        )
        curve_frames.append(curve)

    curves = pd.concat(curve_frames, ignore_index=True)
    curves_path = paths["fock_angle"] / "fock_resolved_xx_angle_curves.csv"
    curves.to_csv(curves_path, index=False)

    generator = _load_generator(paths)
    summary_rows = []
    for condition in conditions:
        condition_curve = curves.loc[
            curves["condition"] == condition["condition"]
        ]
        for thermal_n_bar in selected_nbars:
            prediction = phonon_angle.summarize_thermal_xx_angle(
                condition_curve, thermal_n_bar,
                target_xx_angle_rad=target_angle,
            )
            is_baseline = condition["condition"] == "baseline"
            is_matched = is_baseline or np.isclose(
                float(condition["calibration_n_bar"]), thermal_n_bar
            )
            qpt_h_xx = np.nan
            qpt_gamma_xx = np.nan
            qpt_infidelity = np.nan
            if is_baseline:
                generator_match = generator.loc[
                    np.isclose(generator["n_bar"].astype(float), thermal_n_bar)
                ]
                if not generator_match.empty:
                    qpt_h_xx = float(
                        generator_match.iloc[-1]["h_XX_rad_per_gate"]
                    )
                    qpt_gamma_xx = float(
                        generator_match.iloc[-1]["gamma_XX_per_gate"]
                    )
                    baseline_chi = _load_cptp_chi(
                        paths, thermal_n_bar
                    )
                    labels = [
                        label for label, _ in mg.pauli_labels_and_weights()
                    ]
                    qpt_infidelity = 4.0 / 5.0 * (
                        1.0
                        - float(
                            np.real(
                                baseline_chi[
                                    labels.index("II"), labels.index("II")
                                ]
                            )
                        )
                    )
            elif is_matched:
                calibrated_match = drive_summary.loc[
                    np.isclose(
                        drive_summary["n_bar"].astype(float), thermal_n_bar
                    )
                ]
                if not calibrated_match.empty:
                    calibrated_row = calibrated_match.iloc[-1]
                    qpt_h_xx = float(
                        calibrated_row["h_XX_after_rad_per_gate"]
                    )
                    qpt_gamma_xx = float(
                        calibrated_row["gamma_XX_after_per_gate"]
                    )
                    qpt_infidelity = float(
                        calibrated_row["average_infidelity_after"]
                    )
            summary_rows.append({
                "condition": condition["condition"],
                "label": condition["label"],
                "calibration_n_bar": condition["calibration_n_bar"],
                "amplitude": condition["amplitude"],
                "is_matched_calibration": bool(is_matched),
                **prediction,
                "qpt_h_XX_rad_per_gate": qpt_h_xx,
                "qpt_gamma_XX_per_gate": qpt_gamma_xx,
                "qpt_average_infidelity": qpt_infidelity,
            })

    summary = pd.DataFrame(summary_rows)
    summary_path = paths["fock_angle"] / "thermal_xx_angle_summary.csv"
    summary.to_csv(summary_path, index=False)
    figure_path = phonon_angle.plot_fock_angle_comparison(
        curves,
        summary,
        output_dir=paths["fock_angle"],
        plot_max_fock_n=plot_max_fock_n,
        target_xx_angle_rad=target_angle,
    )
    matched = summary.loc[summary["is_matched_calibration"]].copy()
    return {
        "curves": curves,
        "summary": summary,
        "matched_summary": matched,
        "figure_path": figure_path,
        "curves_path": curves_path,
        "summary_path": summary_path,
        "status": {
            "reference_nbars": selected_nbars,
            "max_fock_n": resolved_max_fock_n,
            "thermal_tail_tolerance": float(thermal_tail_tolerance),
            "condition_count": len(conditions),
        },
    }


def run_kirchhoff_direct_comparison_stage(
    config: Mapping[str, Any],
    *,
    mode_frequency_hz: float | None = None,
    reference_k: float | None = None,
    loop_number: float = 1.0,
) -> dict[str, Any]:
    """Compare hXX-inferred drive factors with one fixed Kirchhoff K,L point."""

    paths = _paths(config)
    params = _simulation_params(config)
    if reference_k is None:
        if mode_frequency_hz is None:
            raise ValueError(
                "Set mode_frequency_hz or reference_k for a direct Kirchhoff comparison"
            )
        reference_k = float(mode_frequency_hz) * float(params["t_gate_phys"])
    reference_k = float(reference_k)
    reference_l = reference_k - float(loop_number)
    ratios = amplitude_calibration.kirchhoff_renormalization_ratios(
        reference_k,
        reference_l,
        float(params["eta"]),
    )

    prediction_path = paths["drive"] / "hxx_drive_amplitude_prediction.csv"
    if prediction_path.exists():
        prediction = pd.read_csv(prediction_path)
    else:
        generator = _load_generator(paths)
        base_amplitude = float(np.asarray(params["A"]))
        target = float(config.get("TARGET_XX_ANGLE_RAD", np.pi / 4.0))
        prediction = pd.DataFrame({
            "n_bar": generator["n_bar"].astype(float),
            "h_XX_baseline_rad_per_gate": generator["h_XX_rad_per_gate"],
        })
        prediction["A_baseline"] = base_amplitude
        prediction["A_hxx_first_update"] = [
            amplitude_calibration.amplitude_update_from_hxx(
                base_amplitude,
                value,
                target,
            )
            for value in prediction["h_XX_baseline_rad_per_gate"]
        ]
        prediction["A_hxx_factor"] = (
            prediction["A_hxx_first_update"] / base_amplitude
        )

    comparison = prediction.copy()
    comparison["kirchhoff_K"] = reference_k
    comparison["kirchhoff_L"] = reference_l
    comparison["kirchhoff_omega_2_over_omega_ld"] = ratios[
        "omega_2_over_omega_ld"
    ]
    comparison["kirchhoff_omega_4_over_omega_ld"] = ratios[
        "omega_4_over_omega_ld"
    ]
    comparison["hxx_minus_kirchhoff_4"] = (
        comparison["A_hxx_factor"] - ratios["omega_4_over_omega_ld"]
    )
    comparison["relative_difference_to_kirchhoff_4"] = (
        comparison["hxx_minus_kirchhoff_4"]
        / ratios["omega_4_over_omega_ld"]
    )
    comparison_path = paths["drive"] / "kirchhoff_direct_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    axis.plot(
        comparison["n_bar"],
        comparison["A_hxx_factor"],
        linewidth=2.4,
        label=r"$h_{XX}$ feedback",
    )
    axis.axhline(
        ratios["omega_2_over_omega_ld"],
        linestyle=":",
        linewidth=2.0,
        label="Kirchhoff Eq. (35)",
    )
    axis.axhline(
        ratios["omega_4_over_omega_ld"],
        linestyle="--",
        linewidth=2.0,
        label="Kirchhoff Eq. (41)",
    )
    axis.set_xlabel(r"Mean phonon number $\bar n$")
    axis.set_ylabel("Drive-amplitude renormalization factor")
    axis.set_title(
        f"Direct Kirchhoff comparison: K={reference_k:.4g}, L={reference_l:.4g}"
    )
    axis.grid(True, alpha=0.28)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        paths["drive"] / "kirchhoff_direct_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        paths["drive"] / "kirchhoff_direct_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    return {
        "comparison": comparison,
        "reference": {
            "K": reference_k,
            "L": reference_l,
            "eta": float(params["eta"]),
            **ratios,
        },
        "path": comparison_path,
    }


def _kirchhoff_paper_cache_path(
    output_dir: Path,
    payload: Mapping[str, Any],
    omega_per_second: float,
) -> Path:
    cache_payload = {
        **dict(payload),
        "omega_per_second": float(omega_per_second),
        "implementation": "kirchhoff_paper_v1",
    }
    digest = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, default=_json_safe).encode("utf-8")
    ).hexdigest()[:14]
    omega_stem = _condition_stem(f"{float(omega_per_second) / 1e6:.9f}")
    return output_dir / "propagator_cache" / f"omega_mhz_{omega_stem}__{digest}.npz"


def _kirchhoff_landmark_label(
    omega_per_second: float,
    landmarks: Mapping[str, float],
) -> str:
    labels = {"omega_ld": "Omega_LD", "omega_2": "Omega_2", "omega_4": "Omega_4"}
    for key, value in landmarks.items():
        if np.isclose(float(omega_per_second), float(value), rtol=1e-11, atol=1e-7):
            return labels[key]
    return "sweep"


def _plot_kirchhoff_paper_infidelity(
    summary: pd.DataFrame,
    reference: pd.DataFrame,
    landmarks: Mapping[str, float],
    output_dir: Path,
) -> Path | None:
    if summary.empty:
        return None
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5))
    reference_nbar = float(
        summary.loc[(summary["n_bar"] - 0.02).abs().idxmin(), "n_bar"]
    )
    amplitude_curve = summary.loc[
        np.isclose(summary["n_bar"].astype(float), reference_nbar)
    ].sort_values("omega_per_second")
    omega_mhz = amplitude_curve["omega_per_second"] / 1e6
    axes[0, 0].semilogy(
        omega_mhz,
        amplitude_curve["paper_average_infidelity"],
        marker="o",
        label="numerical Eq. (B6)",
    )
    axes[0, 1].semilogy(
        omega_mhz,
        amplitude_curve["paper_bell_infidelity"],
        marker="o",
        color="#D55E00",
        label="numerical Bell fidelity",
    )
    colors = {"omega_ld": "#777777", "omega_2": "#E69F00", "omega_4": "#CC79A7"}
    display_labels = {
        "omega_ld": r"$\Omega_{\rm LD}$",
        "omega_2": r"$\Omega_2$",
        "omega_4": r"$\Omega_4$",
    }
    for key, value in landmarks.items():
        for axis in axes[0]:
            axis.axvline(
                float(value) / 1e6,
                color=colors[key],
                linestyle="--" if key != "omega_ld" else ":",
                alpha=0.9,
                label=display_labels[key],
            )
    if not reference.empty:
        for _, row in reference.iterrows():
            axes[0, 0].scatter(
                row["omega_per_second"] / 1e6,
                row["paper_reported_average_infidelity"],
                marker="x",
                s=90,
                linewidths=2.2,
                color="black",
                zorder=5,
            )
            axes[0, 1].scatter(
                row["omega_per_second"] / 1e6,
                row["paper_reported_bell_infidelity"],
                marker="x",
                s=90,
                linewidths=2.2,
                color="black",
                zorder=5,
            )
    for axis, title in [
        (axes[0, 0], "Paper average-overlap infidelity"),
        (axes[0, 1], "Paper Bell-state infidelity"),
    ]:
        axis.set_xlabel(r"Drive amplitude $\Omega$ [MHz, paper convention]")
        axis.set_ylabel("Infidelity")
        axis.set_title(title + rf" at $\bar n={reference_nbar:g}$")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)

    for landmark in ("Omega_2", "Omega_4"):
        curve = summary.loc[summary["landmark"] == landmark].sort_values("n_bar")
        if curve.empty:
            continue
        axes[1, 0].loglog(
            np.maximum(curve["n_bar"], 1e-6),
            curve["paper_average_infidelity"],
            marker="o",
            label=r"$\Omega_2$" if landmark == "Omega_2" else r"$\Omega_4$",
        )
    axes[1, 0].set_xlabel(r"Mean phonon number $\bar n$")
    axes[1, 0].set_ylabel("Paper average infidelity")
    axes[1, 0].set_title("Thermal dependence from the same propagator")
    axes[1, 0].grid(True, which="both", alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    if reference.empty:
        axes[1, 1].text(0.5, 0.5, "Omega_2/Omega_4 points pending", ha="center")
        axes[1, 1].set_axis_off()
    else:
        x_positions = np.arange(len(reference))
        width = 0.36
        axes[1, 1].bar(
            x_positions - width / 2,
            reference["simulated_average_infidelity"],
            width,
            label="simulation",
        )
        axes[1, 1].bar(
            x_positions + width / 2,
            reference["paper_reported_average_infidelity"],
            width,
            label="paper",
        )
        axes[1, 1].set_xticks(x_positions)
        axes[1, 1].set_xticklabels(reference["landmark"])
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_ylabel("Average infidelity")
        axes[1, 1].set_title(r"Reference check at $\bar n=0.02$")
        axes[1, 1].grid(True, which="both", axis="y", alpha=0.25)
        axes[1, 1].legend(fontsize=8)

    figure.suptitle(
        "Kirchhoff et al. PRX Quantum 6, 010328 (2025): numerical reproduction",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "kirchhoff_paper_infidelity_reproduction.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    figure.savefig(
        output_dir / "kirchhoff_paper_infidelity_reproduction.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    return figure_path


def run_kirchhoff_paper_infidelity_stage(
    config: Mapping[str, Any],
    *,
    K: float = 28.0,
    L: float | None = None,
    k_minus_l: float = 3.0,
    mode_frequency_hz: float = 1e6,
    gate_duration_seconds: float | None = None,
    eta: float = 0.18,
    nbar_values=(0.02,),
    omega_mhz_values=None,
    phonon_dim: int = 8,
    sideband_cutoff: int = 3,
    solver: str = "adaptive_dop853",
    trotter_steps: int | None = None,
    trotter_step_scale: float = 1.0,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
    parallel_workers: int = 1,
    run_simulation: bool = False,
    force_recompute: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Reproduce the paper's U_num infidelity with its own Hamiltonian/metric.

    The default parameters are those of Figs. 2, 3 and 5.  Propagators are
    cached by drive amplitude and can be reweighted for new nbar values without
    rerunning time evolution.
    """

    paths = _paths(config)
    output_dir = paths["kirchhoff_paper"]
    cache_dir = output_dir / "propagator_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    K = float(K)
    if L is None:
        L = K - float(k_minus_l)
    L = float(L)
    mode_frequency_hz = float(mode_frequency_hz)
    if gate_duration_seconds is None:
        gate_duration_seconds = K / mode_frequency_hz
    gate_duration_seconds = float(gate_duration_seconds)
    if not np.isclose(
        mode_frequency_hz * gate_duration_seconds, K, rtol=2e-10, atol=2e-10
    ):
        raise ValueError("K must equal mode_frequency_hz * gate_duration_seconds")

    model = kirchhoff_paper.build_paper_model(
        K=K,
        L=L,
        eta=eta,
        phonon_dim=phonon_dim,
        sideband_cutoff=sideband_cutoff,
    )
    landmarks = kirchhoff_paper.drive_landmarks(
        K=K,
        L=L,
        gate_duration_seconds=gate_duration_seconds,
        eta=eta,
    )
    if omega_mhz_values is None:
        omega_mhz_values = np.linspace(1.04, 1.14, 11)
    requested_omega = [float(value) * 1e6 for value in omega_mhz_values]
    omega_values = kirchhoff_paper.unique_drive_values(requested_omega, landmarks)
    nbar_values = sorted({float(value) for value in nbar_values})
    if not nbar_values:
        raise ValueError("nbar_values cannot be empty")

    payload = {
        "K": K,
        "L": L,
        "mode_frequency_hz": mode_frequency_hz,
        "gate_duration_seconds": gate_duration_seconds,
        "eta": float(eta),
        "phonon_dim": int(phonon_dim),
        "sideband_cutoff": int(sideband_cutoff),
        "solver": str(solver),
        "trotter_steps": trotter_steps,
        "trotter_step_scale": float(trotter_step_scale),
        "relative_tolerance": float(relative_tolerance),
        "absolute_tolerance": float(absolute_tolerance),
    }
    cache_paths = {
        omega: _kirchhoff_paper_cache_path(output_dir, payload, omega)
        for omega in omega_values
    }
    missing = [
        omega
        for omega in omega_values
        if force_recompute or not cache_paths[omega].exists()
    ]

    def compute_and_cache(omega):
        start = time.perf_counter()
        propagator, metadata = kirchhoff_paper.propagate(
            model,
            omega_per_second=omega,
            gate_duration_seconds=gate_duration_seconds,
            solver=solver,
            number_of_steps=trotter_steps,
            step_scale=trotter_step_scale,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        metadata = {
            **metadata,
            "omega_per_second": float(omega),
            "wall_time_seconds": float(time.perf_counter() - start),
        }
        path = cache_paths[omega]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            propagator=propagator,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return omega

    newly_computed = 0
    if missing and run_simulation:
        workers = max(1, min(int(parallel_workers), len(missing)))
        if workers == 1:
            for omega in tqdm(
                missing,
                desc="8.1 Kirchhoff paper infidelity",
                unit="drive",
                disable=not show_progress,
            ):
                compute_and_cache(omega)
                newly_computed += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(compute_and_cache, omega): omega
                    for omega in missing
                }
                iterator = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="8.1 Kirchhoff paper infidelity",
                    unit="drive",
                    disable=not show_progress,
                )
                for future in iterator:
                    future.result()
                    newly_computed += 1

    rows = []
    metadata_rows = []
    for omega in omega_values:
        path = cache_paths[omega]
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            propagator = np.asarray(data["propagator"], dtype=complex)
            metadata = json.loads(str(data["metadata_json"].item()))
        metadata_rows.append({
            "omega_per_second": float(omega),
            "omega_mhz": float(omega) / 1e6,
            "landmark": _kirchhoff_landmark_label(omega, landmarks),
            "cache_path": str(path),
            **metadata,
        })
        for n_bar in nbar_values:
            rows.append({
                "omega_per_second": float(omega),
                "omega_mhz": float(omega) / 1e6,
                "landmark": _kirchhoff_landmark_label(omega, landmarks),
                "cache_path": str(path),
                **kirchhoff_paper.paper_fidelities(
                    propagator, model, n_bar=n_bar
                ),
            })
    summary = pd.DataFrame(rows)
    metadata_frame = pd.DataFrame(metadata_rows)
    if not summary.empty:
        summary = summary.sort_values(["n_bar", "omega_per_second"])
    summary_path = output_dir / "kirchhoff_paper_infidelity_summary.csv"
    metadata_path = output_dir / "kirchhoff_paper_propagator_metadata.csv"
    summary.to_csv(summary_path, index=False)
    metadata_frame.to_csv(metadata_path, index=False)

    paper_reported = {
        "Omega_2": {"average": 0.67e-3, "bell": 1.3e-3},
        "Omega_4": {"average": 0.24e-3, "bell": 0.43e-3},
    }
    reference_rows = []
    for landmark, reported in paper_reported.items():
        landmark_key = "omega_2" if landmark == "Omega_2" else "omega_4"
        omega = float(landmarks[landmark_key])
        path = cache_paths[omega]
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            propagator = np.asarray(data["propagator"], dtype=complex)
        simulated = kirchhoff_paper.paper_fidelities(
            propagator, model, n_bar=0.02
        )
        reference_rows.append({
            "landmark": landmark,
            "omega_per_second": omega,
            "omega_mhz": omega / 1e6,
            "simulated_average_infidelity": simulated[
                "paper_average_infidelity"
            ],
            "paper_reported_average_infidelity": reported["average"],
            "relative_average_difference": (
                simulated["paper_average_infidelity"] / reported["average"] - 1.0
            ),
            "simulated_bell_infidelity": simulated["paper_bell_infidelity"],
            "paper_reported_bell_infidelity": reported["bell"],
            "relative_bell_difference": (
                simulated["paper_bell_infidelity"] / reported["bell"] - 1.0
            ),
        })
    reference = pd.DataFrame(reference_rows)
    reference_path = output_dir / "kirchhoff_paper_reference_comparison.csv"
    reference.to_csv(reference_path, index=False)
    figure_path = _plot_kirchhoff_paper_infidelity(
        summary, reference, landmarks, output_dir
    )

    pending = [omega for omega in omega_values if not cache_paths[omega].exists()]
    return {
        "summary": summary,
        "propagator_metadata": metadata_frame,
        "reference_comparison": reference,
        "landmarks": {
            key: value / 1e6 for key, value in landmarks.items()
        },
        "figure_path": figure_path,
        "summary_path": summary_path,
        "reference_path": reference_path,
        "output_dir": output_dir,
        "status": {
            "completed_drive_points": len(omega_values) - len(pending),
            "expected_drive_points": len(omega_values),
            "newly_computed": newly_computed,
            "pending_drive_points": len(pending),
            "run_simulation": bool(run_simulation),
            "solver": str(solver),
            "parallel_workers": int(parallel_workers),
            "K": K,
            "L": L,
            "eta": float(eta),
            "phonon_dim": int(phonon_dim),
            "sideband_cutoff": int(sideband_cutoff),
            "gate_duration_seconds": gate_duration_seconds,
            "paper_fidelity_definition": "Appendix B Eq. (B6) gate-overlap amplitude",
        },
        "pending": pending,
    }


def _kirchhoff_thermal_qpt_cache_path(
    output_dir: Path,
    *,
    source_propagator_path: Path,
    omega_per_second: float,
    n_bar: float,
    convention: str,
    cptp_tolerance: float,
    cptp_max_iterations: int,
) -> Path:
    payload = {
        "source_propagator_cache": source_propagator_path.name,
        "omega_per_second": float(omega_per_second),
        "n_bar": float(n_bar),
        "convention": str(convention),
        "cptp_tolerance": float(cptp_tolerance),
        "cptp_max_iterations": int(cptp_max_iterations),
        "implementation": "kirchhoff_thermal_qubit_qpt_v1",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:14]
    return output_dir / "channel_cache" / (
        f"omega_mhz_{_condition_stem(f'{float(omega_per_second) / 1e6:.9f}')}"
        f"__nbar_{_compact_nbar_stem(n_bar)}__{digest}.npz"
    )


def _plot_kirchhoff_thermal_qpt(
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if summary.empty:
        return None
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5))
    reference_nbar = float(
        summary.loc[(summary["n_bar"] - 0.02).abs().idxmin(), "n_bar"]
    )
    amplitude = summary.loc[
        np.isclose(summary["n_bar"].astype(float), reference_nbar)
    ].sort_values("omega_per_second")
    axes[0, 0].semilogy(
        amplitude["omega_mhz"],
        np.maximum(amplitude["average_infidelity"], 1e-16),
        marker="o",
        label=r"standard channel $1-F_{\rm avg}$",
    )
    axes[0, 0].semilogy(
        amplitude["omega_mhz"],
        np.maximum(amplitude["paper_average_infidelity"], 1e-16),
        marker="s",
        linestyle="--",
        label="paper Eq. (B6)",
    )
    axes[0, 0].set_xlabel(r"Drive amplitude $\Omega$ [MHz]")
    axes[0, 0].set_ylabel("Infidelity")
    axes[0, 0].set_title(rf"Metric comparison at $\bar n={reference_nbar:g}$")
    axes[0, 0].grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    landmark_labels = ["Omega_LD", "Omega_2", "Omega_4"]
    display_labels = {
        "Omega_LD": r"$\Omega_{\rm LD}$",
        "Omega_2": r"$\Omega_2$",
        "Omega_4": r"$\Omega_4$",
    }
    colors = {
        "Omega_LD": "#777777",
        "Omega_2": "#E69F00",
        "Omega_4": "#CC79A7",
    }
    for landmark in landmark_labels:
        curve = summary.loc[summary["landmark"] == landmark].sort_values("n_bar")
        if curve.empty:
            continue
        axes[0, 1].plot(
            curve["n_bar"],
            curve["h_XX_rad_per_gate"],
            marker="o",
            color=colors[landmark],
            label=display_labels[landmark],
        )
        axes[1, 0].loglog(
            curve["n_bar"],
            np.maximum(curve["gamma_XX_per_gate"], 1e-18),
            marker="o",
            color=colors[landmark],
            label=display_labels[landmark],
        )
        axes[1, 1].loglog(
            curve["n_bar"],
            np.maximum(curve["average_infidelity"], 1e-16),
            marker="o",
            color=colors[landmark],
            label=display_labels[landmark],
        )

    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel(r"Mean phonon number $\bar n$")
    axes[0, 1].set_ylabel(r"$h_{XX}$ [rad/gate]")
    axes[0, 1].set_title("Coherent XX error after YY-to-XX basis mapping")
    axes[0, 1].grid(True, which="both", alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_xlabel(r"Mean phonon number $\bar n$")
    axes[1, 0].set_ylabel(r"$\gamma_{XX}$ [1/gate]")
    axes[1, 0].set_title("Stochastic XX component from the reduced channel")
    axes[1, 0].grid(True, which="both", alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_xlabel(r"Mean phonon number $\bar n$")
    axes[1, 1].set_ylabel(r"Standard channel $1-F_{\rm avg}$")
    axes[1, 1].set_title("Thermal channel infidelity")
    axes[1, 1].grid(True, which="both", alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    figure.suptitle(
        "Kirchhoff propagator → thermal qubit channel → QPT/error generator",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "kirchhoff_thermal_qubit_qpt.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    figure.savefig(
        output_dir / "kirchhoff_thermal_qubit_qpt.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    return figure_path


def run_kirchhoff_thermal_qubit_qpt_stage(
    config: Mapping[str, Any],
    paper_result: Mapping[str, Any],
    *,
    nbar_values=None,
    convention: str | None = None,
    force_recompute: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    r"""Convert cached Kirchhoff propagators to thermal qubit QPT channels.

    For each thermal input, this stage forms
    ``K_mn=sqrt(p_n)<m|U|n>``, maps the paper's YY basis to XX by local Z
    rotations, removes the ideal XX gate, and calls the same CPTP projection
    and ``log(PTM)`` Pauli-generator decomposition used by the drive-QPT stages.
    No Hamiltonian propagation is performed here.
    """

    paths = _paths(config)
    output_dir = paths["kirchhoff_qpt"]
    cache_dir = output_dir / "channel_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    convention = _convention(config) if convention is None else str(convention)
    if convention not in {"undo_before_actual", "undo_after_actual"}:
        raise ValueError(
            "convention must be 'undo_before_actual' or 'undo_after_actual'"
        )
    cptp_tolerance, cptp_max_iterations = _cptp_options(config)

    propagator_metadata = paper_result.get("propagator_metadata")
    if propagator_metadata is None or len(propagator_metadata) == 0:
        raise FileNotFoundError(
            "Cell 8.1 has no completed propagator cache. Run/load Cell 8.1 first."
        )
    propagator_metadata = pd.DataFrame(propagator_metadata).copy()
    required_columns = {
        "omega_per_second", "omega_mhz", "landmark", "cache_path"
    }
    missing_columns = required_columns - set(propagator_metadata.columns)
    if missing_columns:
        raise ValueError(
            "paper_result propagator metadata is missing: "
            + ", ".join(sorted(missing_columns))
        )
    propagator_metadata = (
        propagator_metadata.sort_values("omega_per_second")
        .drop_duplicates("omega_per_second", keep="last")
        .reset_index(drop=True)
    )

    paper_status = dict(paper_result.get("status", {}))
    model = kirchhoff_paper.build_paper_model(
        K=float(paper_status["K"]),
        L=float(paper_status["L"]),
        eta=float(paper_status["eta"]),
        phonon_dim=int(paper_status["phonon_dim"]),
        sideband_cutoff=int(paper_status["sideband_cutoff"]),
    )
    if nbar_values is None:
        paper_summary = pd.DataFrame(paper_result.get("summary", []))
        if paper_summary.empty:
            nbar_values = [0.02]
        else:
            nbar_values = paper_summary["n_bar"].tolist()
    nbar_values = sorted({float(value) for value in nbar_values})
    if not nbar_values or any(value < 0.0 for value in nbar_values):
        raise ValueError("nbar_values must contain nonnegative values")

    task_rows = []
    missing_propagators = []
    for _, source_row in propagator_metadata.iterrows():
        source_path = Path(source_row["cache_path"])
        if not source_path.exists():
            missing_propagators.append(source_path)
            continue
        for n_bar in nbar_values:
            cache_path = _kirchhoff_thermal_qpt_cache_path(
                output_dir,
                source_propagator_path=source_path,
                omega_per_second=float(source_row["omega_per_second"]),
                n_bar=n_bar,
                convention=convention,
                cptp_tolerance=cptp_tolerance,
                cptp_max_iterations=cptp_max_iterations,
            )
            task_rows.append((source_row, n_bar, source_path, cache_path))

    summary_rows = []
    coefficient_rows = []
    chi_component_rows = []
    newly_computed = 0
    loaded_propagator_path = None
    loaded_propagator = None
    iterator = tqdm(
        task_rows,
        desc="8.2 Kirchhoff thermal QPT",
        unit="channel",
        disable=not show_progress,
    )
    pauli_labels = [label for label, _ in mg.pauli_labels_and_weights()]
    for source_row, n_bar, source_path, cache_path in iterator:
        if force_recompute or not cache_path.exists():
            if loaded_propagator_path != source_path:
                with np.load(source_path, allow_pickle=False) as source_data:
                    loaded_propagator = np.asarray(
                        source_data["propagator"], dtype=complex
                    )
                loaded_propagator_path = source_path
            _, error_super, channel_metadata = (
                kirchhoff_paper.kirchhoff_xx_error_channel(
                    loaded_propagator,
                    model,
                    n_bar=n_bar,
                    convention=convention,
                )
            )
            chi_raw = np.asarray(qp.to_chi(error_super).full(), dtype=complex)
            raw_trace = np.trace(chi_raw)
            chi = chi_raw / raw_trace
            physicality = mg.choi_physicality_metrics(error_super)
            decomposition = qpt_analysis.extract_pauli_generator_observables(
                chi,
                cptp_tolerance=cptp_tolerance,
                cptp_max_iterations=cptp_max_iterations,
            )
            projected_chi = decomposition.pop("projected_chi")
            hamiltonian = decomposition.pop(
                "hamiltonian_coefficients_rad_per_gate"
            )
            dissipator = decomposition.pop("pauli_dissipator_rates_per_gate")
            paper_metrics = kirchhoff_paper.paper_fidelities(
                loaded_propagator, model, n_bar=n_bar
            )
            row = {
                "omega_per_second": float(source_row["omega_per_second"]),
                "omega_mhz": float(source_row["omega_mhz"]),
                "landmark": str(source_row["landmark"]),
                "n_bar": float(n_bar),
                "h_XX_rad_per_gate": float(hamiltonian["XX"]),
                "gamma_XX_per_gate": float(dissipator["XX"]),
                **decomposition,
                "paper_average_infidelity": float(
                    paper_metrics["paper_average_infidelity"]
                ),
                "paper_bell_infidelity": float(
                    paper_metrics["paper_bell_infidelity"]
                ),
                "raw_chi_trace_real": float(np.real(raw_trace)),
                "raw_chi_trace_imag": float(np.imag(raw_trace)),
                "raw_cp_pass": bool(physicality["cp_pass"]),
                "raw_tp_pass": bool(physicality["tp_pass"]),
                "raw_min_choi_eigenvalue": float(
                    physicality["min_choi_eigenvalue"]
                ),
                "raw_tp_frobenius_error": float(
                    physicality["tp_frobenius_error"]
                ),
                "chi_cptp_projection_frobenius_shift": float(
                    np.linalg.norm(projected_chi - chi)
                ),
                **channel_metadata,
                "source_propagator_cache": str(source_path),
                "qpt_cache_path": str(cache_path),
            }
            temporary_path = cache_path.with_name(
                cache_path.stem + ".tmp.npz"
            )
            np.savez_compressed(
                temporary_path,
                chi_trace_normalized_raw=chi,
                chi_trace_normalized_projected=projected_chi,
                summary_json=np.asarray(
                    json.dumps(row, sort_keys=True, default=_json_safe)
                ),
                hamiltonian_json=np.asarray(
                    json.dumps(hamiltonian, sort_keys=True)
                ),
                dissipator_json=np.asarray(
                    json.dumps(dissipator, sort_keys=True)
                ),
            )
            temporary_path.replace(cache_path)
            newly_computed += 1

        with np.load(cache_path, allow_pickle=False) as cached:
            projected_chi = np.asarray(
                cached["chi_trace_normalized_projected"], dtype=complex
            )
            row = json.loads(str(cached["summary_json"].item()))
            hamiltonian = json.loads(str(cached["hamiltonian_json"].item()))
            dissipator = json.loads(str(cached["dissipator_json"].item()))
        summary_rows.append(row)
        for pauli in pauli_labels[1:]:
            coefficient_rows.append({
                "omega_per_second": float(row["omega_per_second"]),
                "omega_mhz": float(row["omega_mhz"]),
                "landmark": row["landmark"],
                "n_bar": float(row["n_bar"]),
                "pauli": pauli,
                "hamiltonian_coefficient_rad_per_gate": float(
                    hamiltonian[pauli]
                ),
                "pauli_dissipator_rate_per_gate": float(dissipator[pauli]),
            })
        for row_index, row_pauli in enumerate(pauli_labels):
            for column_index in range(row_index, len(pauli_labels)):
                value = projected_chi[row_index, column_index]
                chi_component_rows.append({
                    "omega_per_second": float(row["omega_per_second"]),
                    "omega_mhz": float(row["omega_mhz"]),
                    "landmark": row["landmark"],
                    "n_bar": float(row["n_bar"]),
                    "component": f"{row_pauli},{pauli_labels[column_index]}",
                    "real": float(np.real(value)),
                    "imag": float(np.imag(value)),
                    "abs": float(abs(value)),
                })

    summary = pd.DataFrame(summary_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    chi_components = pd.DataFrame(chi_component_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["n_bar", "omega_per_second"]
        ).reset_index(drop=True)
    summary_path = output_dir / "kirchhoff_thermal_qubit_qpt_summary.csv"
    coefficient_path = output_dir / "kirchhoff_error_generator_coefficients.csv"
    chi_component_path = output_dir / "kirchhoff_qpt_chi_components.csv"
    summary.to_csv(summary_path, index=False)
    coefficients.to_csv(coefficient_path, index=False)
    chi_components.to_csv(chi_component_path, index=False)
    figure_path = _plot_kirchhoff_thermal_qpt(summary, output_dir)

    expected_points = len(propagator_metadata) * len(nbar_values)
    completed_points = len(summary)
    status = {
        "completed_channel_points": int(completed_points),
        "expected_channel_points": int(expected_points),
        "newly_computed": int(newly_computed),
        "pending_channel_points": int(expected_points - completed_points),
        "missing_propagator_caches": int(len(missing_propagators)),
        "error_channel_convention": convention,
        "yy_to_xx_basis_mapping": "Rz(-pi/2) tensor Rz(-pi/2)",
        "qpt_pipeline": "CPTP projection -> PTM -> log(PTM) -> Pauli generator",
    }
    if not summary.empty:
        status.update({
            "raw_cp_pass": int(summary["raw_cp_pass"].astype(bool).sum()),
            "raw_tp_pass": int(summary["raw_tp_pass"].astype(bool).sum()),
            "max_kraus_completeness_error": float(
                summary["kraus_completeness_frobenius_error"].max()
            ),
            "max_cptp_projection_shift": float(
                summary["chi_cptp_projection_frobenius_shift"].max()
            ),
        })
    return {
        "summary": summary,
        "generator_coefficients": coefficients,
        "chi_components": chi_components,
        "figure_path": figure_path,
        "summary_path": summary_path,
        "coefficient_path": coefficient_path,
        "chi_component_path": chi_component_path,
        "output_dir": output_dir,
        "status": status,
        "missing_propagators": missing_propagators,
    }


def _coherent_limit_own_cache_path(
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    model_name: str,
    omega_per_second: float,
) -> Path:
    cache_payload = {
        **dict(payload),
        "model_name": str(model_name),
        "omega_per_second": float(omega_per_second),
        "implementation": "coherent_limit_bridge_v1",
    }
    digest = hashlib.sha256(
        json.dumps(
            cache_payload, sort_keys=True, default=_json_safe
        ).encode("utf-8")
    ).hexdigest()[:14]
    return output_dir / "own_propagator_cache" / (
        f"{_condition_stem(model_name)}__omega_mhz_"
        f"{_condition_stem(f'{float(omega_per_second) / 1e6:.9f}')}"
        f"__{digest}.npz"
    )


def _plot_coherent_limit_comparison(
    model_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if model_summary.empty or comparison.empty:
        return None
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5))
    preferred_landmark = (
        "Omega_4"
        if "Omega_4" in set(model_summary["landmark"])
        else str(model_summary["landmark"].iloc[0])
    )
    line_styles = {
        "Kirchhoff_full_carrier": ("black", "o", "-"),
        "own_lamb_dicke": ("#0072B2", "s", "--"),
        "own_eta2_corrected": ("#D55E00", "^", "-."),
    }
    display_labels = {
        "Kirchhoff_full_carrier": "Kirchhoff full carrier-sideband",
        "own_lamb_dicke": "repository first-sideband (LD)",
        "own_eta2_corrected": r"repository first-sideband ($\eta^2$ corrected)",
    }
    for model_name, curve in model_summary.loc[
        model_summary["landmark"] == preferred_landmark
    ].groupby("model"):
        curve = curve.sort_values("n_bar")
        color, marker, linestyle = line_styles.get(
            model_name, (None, "o", "-")
        )
        label = display_labels.get(model_name, model_name)
        axes[0, 0].loglog(
            curve["n_bar"],
            np.maximum(curve["average_infidelity"], 1e-16),
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
        axes[0, 1].plot(
            curve["n_bar"],
            curve["h_XX_rad_per_gate"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
        axes[1, 0].loglog(
            curve["n_bar"],
            np.maximum(curve["gamma_XX_per_gate"], 1e-18),
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )

    axes[0, 0].set_xlabel(r"Mean phonon number $\bar n$")
    axes[0, 0].set_ylabel(r"Standard channel $1-F_{\rm avg}$")
    axes[0, 0].set_title(f"Coherent-limit infidelity at {preferred_landmark}")
    axes[0, 0].grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel(r"Mean phonon number $\bar n$")
    axes[0, 1].set_ylabel(r"$h_{XX}$ [rad/gate]")
    axes[0, 1].set_title(f"Coherent XX error at {preferred_landmark}")
    axes[0, 1].grid(True, which="both", alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_xlabel(r"Mean phonon number $\bar n$")
    axes[1, 0].set_ylabel(r"$\gamma_{XX}$ [1/gate]")
    axes[1, 0].set_title(f"Stochastic XX error at {preferred_landmark}")
    axes[1, 0].grid(True, which="both", alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    for (model_name, landmark), curve in comparison.groupby(
        ["model", "landmark"]
    ):
        curve = curve.sort_values("n_bar")
        color, marker, _ = line_styles.get(model_name, (None, "o", "-"))
        axes[1, 1].loglog(
            curve["n_bar"],
            np.maximum(curve["error_ptm_relative_difference"], 1e-16),
            color=color,
            marker=marker,
            linestyle={
                "Omega_LD": ":",
                "Omega_2": "--",
                "Omega_4": "-",
            }.get(landmark, "-"),
            label=(
                display_labels.get(model_name, model_name)
                + f" / {landmark}"
            ),
        )
    axes[1, 1].set_xlabel(r"Mean phonon number $\bar n$")
    axes[1, 1].set_ylabel(r"Relative $\|(R-I)_{\rm own}-(R-I)_{\rm K}\|_F$")
    axes[1, 1].set_title("Error-channel PTM mismatch")
    axes[1, 1].grid(True, which="both", alpha=0.25)
    axes[1, 1].legend(fontsize=7, ncol=2)

    figure.suptitle(
        "Coherent-limit benchmark: Kirchhoff full model vs repository effective model",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "kirchhoff_vs_repository_coherent_limit.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    figure.savefig(
        output_dir / "kirchhoff_vs_repository_coherent_limit.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    return figure_path


def run_kirchhoff_coherent_limit_comparison_stage(
    config: Mapping[str, Any],
    paper_result: Mapping[str, Any],
    *,
    nbar_values=(0.02, 0.1, 0.3, 1.0),
    landmarks=("Omega_LD", "Omega_2", "Omega_4"),
    model_variants=None,
    t_gate_sim: float = 1.0,
    time_points: int = 501,
    solver_method: str = "vern9",
    solver_max_step: float | None = None,
    solver_atol: float = 1e-11,
    solver_rtol: float = 1e-9,
    run_own_propagation: bool = False,
    force_recompute: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Benchmark the repository's coherent MS model against Kirchhoff.

    Both models use the same ``K, L, eta, T, n_bar``, phonon cutoff, ideal XX
    target, and error-channel convention.  All collapse/noise rates are absent.
    The comparison intentionally retains the model hierarchy: Kirchhoff keeps
    carrier and multiple sidebands, while the repository model keeps only the
    effective first sideband, optionally with its eta-squared correction.
    """

    paths = _paths(config)
    output_dir = paths["kirchhoff_coherent_compare"]
    own_cache_dir = output_dir / "own_propagator_cache"
    own_cache_dir.mkdir(parents=True, exist_ok=True)
    convention = _convention(config)
    if model_variants is None:
        model_variants = {
            "own_lamb_dicke": False,
            "own_eta2_corrected": True,
        }
    model_variants = {
        str(name): bool(use_full_order)
        for name, use_full_order in dict(model_variants).items()
    }
    if not model_variants:
        raise ValueError("model_variants cannot be empty")
    nbar_values = sorted({float(value) for value in nbar_values})
    if not nbar_values or any(value < 0.0 for value in nbar_values):
        raise ValueError("nbar_values must contain nonnegative values")
    requested_landmarks = tuple(str(value) for value in landmarks)

    paper_status = dict(paper_result.get("status", {}))
    required_status = {
        "K", "L", "eta", "phonon_dim", "sideband_cutoff",
        "gate_duration_seconds",
    }
    missing_status = required_status - set(paper_status)
    if missing_status:
        raise ValueError(
            "paper_result status is missing: "
            + ", ".join(sorted(missing_status))
        )
    K = float(paper_status["K"])
    L = float(paper_status["L"])
    eta = float(paper_status["eta"])
    phonon_dim = int(paper_status["phonon_dim"])
    gate_duration_seconds = float(paper_status["gate_duration_seconds"])
    paper_model = kirchhoff_paper.build_paper_model(
        K=K,
        L=L,
        eta=eta,
        phonon_dim=phonon_dim,
        sideband_cutoff=int(paper_status["sideband_cutoff"]),
    )
    paper_metadata = pd.DataFrame(
        paper_result.get("propagator_metadata", [])
    )
    if paper_metadata.empty:
        raise FileNotFoundError(
            "Cell 8.1 has no completed propagator cache. Run/load it first."
        )
    paper_metadata = paper_metadata.loc[
        paper_metadata["landmark"].isin(requested_landmarks)
    ].copy()
    found_landmarks = set(paper_metadata["landmark"])
    missing_landmarks = set(requested_landmarks) - found_landmarks
    if missing_landmarks:
        raise FileNotFoundError(
            "Missing Kirchhoff landmark propagators: "
            + ", ".join(sorted(missing_landmarks))
        )
    paper_metadata = (
        paper_metadata.sort_values("omega_per_second")
        .drop_duplicates("landmark", keep="last")
        .reset_index(drop=True)
    )

    detuning_sim = coherent_bridge.effective_detuning(
        K, L, t_gate_sim=t_gate_sim
    )
    cache_payload = {
        "K": K,
        "L": L,
        "eta": eta,
        "phonon_dim": phonon_dim,
        "gate_duration_seconds": gate_duration_seconds,
        "t_gate_sim": float(t_gate_sim),
        "detuning_sim": float(detuning_sim),
        "time_points": int(time_points),
        "solver_method": str(solver_method),
        "solver_max_step": solver_max_step,
        "solver_atol": float(solver_atol),
        "solver_rtol": float(solver_rtol),
    }
    own_tasks = []
    for model_name, use_full_order in model_variants.items():
        for _, paper_row in paper_metadata.iterrows():
            omega = float(paper_row["omega_per_second"])
            amplitude = coherent_bridge.carrier_to_effective_sideband_amplitude(
                omega,
                eta=eta,
                gate_duration_seconds=gate_duration_seconds,
                t_gate_sim=t_gate_sim,
            )
            path = _coherent_limit_own_cache_path(
                output_dir,
                {**cache_payload, "use_full_order": use_full_order},
                model_name=model_name,
                omega_per_second=omega,
            )
            own_tasks.append({
                "model": model_name,
                "use_full_order": use_full_order,
                "landmark": str(paper_row["landmark"]),
                "omega_per_second": omega,
                "omega_mhz": omega / 1e6,
                "effective_amplitude_sim": amplitude,
                "cache_path": path,
            })
    missing_own = [
        task for task in own_tasks
        if force_recompute or not task["cache_path"].exists()
    ]
    newly_computed = 0
    if missing_own and run_own_propagation:
        for task in tqdm(
            missing_own,
            desc="8.3 repository coherent propagators",
            unit="propagator",
            disable=not show_progress,
        ):
            start = time.perf_counter()
            propagator, metadata = mg.coherent_ms_propagator(
                task["effective_amplitude_sim"],
                detuning_sim,
                rho0=0.0,
                phonon_dim=phonon_dim,
                eta=eta,
                use_full_order=task["use_full_order"],
                time_points=time_points,
                t_gate_sim=t_gate_sim,
                solver_method=solver_method,
                solver_max_step=solver_max_step,
                solver_atol=solver_atol,
                solver_rtol=solver_rtol,
            )
            metadata = {
                **metadata,
                "model": task["model"],
                "landmark": task["landmark"],
                "omega_per_second": task["omega_per_second"],
                "wall_time_seconds": float(time.perf_counter() - start),
            }
            temporary_path = task["cache_path"].with_name(
                task["cache_path"].stem + ".tmp.npz"
            )
            np.savez_compressed(
                temporary_path,
                propagator=np.asarray(propagator.full(), dtype=complex),
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, default=_json_safe)
                ),
            )
            temporary_path.replace(task["cache_path"])
            newly_computed += 1

    def analyze_error_channel(
        error_super,
        *,
        model_name,
        model_scope,
        landmark,
        omega,
        n_bar,
        extra_metadata,
    ):
        observables = qpt_analysis.extract_pauli_generator_from_superoperator(
            error_super
        )
        ptm = np.asarray(observables.pop("ptm"), dtype=complex)
        chi = np.asarray(
            observables.pop("chi_trace_normalized"), dtype=complex
        )
        hamiltonian = observables.pop(
            "hamiltonian_coefficients_rad_per_gate"
        )
        dissipator = observables.pop("pauli_dissipator_rates_per_gate")
        physicality = mg.choi_physicality_metrics(error_super)
        row = {
            "model": model_name,
            "model_scope": model_scope,
            "landmark": landmark,
            "omega_per_second": float(omega),
            "omega_mhz": float(omega) / 1e6,
            "n_bar": float(n_bar),
            "h_XX_rad_per_gate": float(hamiltonian["XX"]),
            "gamma_XX_per_gate": float(dissipator["XX"]),
            **observables,
            "cp_pass": bool(physicality["cp_pass"]),
            "tp_pass": bool(physicality["tp_pass"]),
            "min_choi_eigenvalue": float(
                physicality["min_choi_eigenvalue"]
            ),
            "tp_frobenius_error": float(
                physicality["tp_frobenius_error"]
            ),
            **extra_metadata,
        }
        return row, ptm, chi, hamiltonian, dissipator

    model_rows = []
    coefficient_rows = []
    analysis_by_key = {}
    paper_propagators = {}
    for _, paper_row in paper_metadata.iterrows():
        source_path = Path(paper_row["cache_path"])
        if not source_path.exists():
            continue
        with np.load(source_path, allow_pickle=False) as data:
            paper_propagators[str(paper_row["landmark"])] = np.asarray(
                data["propagator"], dtype=complex
            )

    pauli_labels = [label for label, _ in mg.pauli_labels_and_weights()][1:]
    for _, paper_row in paper_metadata.iterrows():
        landmark = str(paper_row["landmark"])
        if landmark not in paper_propagators:
            continue
        omega = float(paper_row["omega_per_second"])
        propagator = paper_propagators[landmark]
        for n_bar in nbar_values:
            _, error_super, thermal_metadata = (
                kirchhoff_paper.kirchhoff_xx_error_channel(
                    propagator,
                    paper_model,
                    n_bar=n_bar,
                    convention=convention,
                )
            )
            row, ptm, chi, hamiltonian, dissipator = analyze_error_channel(
                error_super,
                model_name="Kirchhoff_full_carrier",
                model_scope="carrier_plus_sidebands",
                landmark=landmark,
                omega=omega,
                n_bar=n_bar,
                extra_metadata={
                    **thermal_metadata,
                    "effective_amplitude_sim": np.nan,
                    "detuning_sim": np.nan,
                    "source_cache": str(paper_row["cache_path"]),
                },
            )
            key = ("Kirchhoff_full_carrier", landmark, float(n_bar))
            analysis_by_key[key] = {
                "row": row,
                "ptm": ptm,
                "chi": chi,
                "h": hamiltonian,
                "gamma": dissipator,
            }
            model_rows.append(row)
            for pauli in pauli_labels:
                coefficient_rows.append({
                    "model": row["model"],
                    "landmark": landmark,
                    "omega_mhz": omega / 1e6,
                    "n_bar": float(n_bar),
                    "pauli": pauli,
                    "hamiltonian_coefficient_rad_per_gate": float(
                        hamiltonian[pauli]
                    ),
                    "pauli_dissipator_rate_per_gate": float(
                        dissipator[pauli]
                    ),
                })

    completed_own_propagators = 0
    for task in own_tasks:
        path = task["cache_path"]
        if not path.exists():
            continue
        completed_own_propagators += 1
        with np.load(path, allow_pickle=False) as data:
            propagator = np.asarray(data["propagator"], dtype=complex)
            propagation_metadata = json.loads(
                str(data["metadata_json"].item())
            )
        for n_bar in nbar_values:
            _, error_super, thermal_metadata = coherent_bridge.own_xx_error_channel(
                propagator,
                phonon_dim=phonon_dim,
                n_bar=n_bar,
                convention=convention,
            )
            row, ptm, chi, hamiltonian, dissipator = analyze_error_channel(
                error_super,
                model_name=task["model"],
                model_scope=(
                    "first_sideband_eta2_corrected"
                    if task["use_full_order"]
                    else "first_sideband_lamb_dicke"
                ),
                landmark=task["landmark"],
                omega=task["omega_per_second"],
                n_bar=n_bar,
                extra_metadata={
                    **thermal_metadata,
                    "effective_amplitude_sim": float(
                        task["effective_amplitude_sim"]
                    ),
                    "detuning_sim": float(detuning_sim),
                    "source_cache": str(path),
                    "propagator_unitarity_frobenius_error": float(
                        propagation_metadata["unitarity_frobenius_error"]
                    ),
                },
            )
            key = (task["model"], task["landmark"], float(n_bar))
            analysis_by_key[key] = {
                "row": row,
                "ptm": ptm,
                "chi": chi,
                "h": hamiltonian,
                "gamma": dissipator,
            }
            model_rows.append(row)
            for pauli in pauli_labels:
                coefficient_rows.append({
                    "model": row["model"],
                    "landmark": task["landmark"],
                    "omega_mhz": task["omega_mhz"],
                    "n_bar": float(n_bar),
                    "pauli": pauli,
                    "hamiltonian_coefficient_rad_per_gate": float(
                        hamiltonian[pauli]
                    ),
                    "pauli_dissipator_rate_per_gate": float(
                        dissipator[pauli]
                    ),
                })

    comparison_rows = []
    coefficient_difference_rows = []
    identity_ptm = np.eye(16)
    for model_name in model_variants:
        for landmark in requested_landmarks:
            for n_bar in nbar_values:
                paper_key = (
                    "Kirchhoff_full_carrier", landmark, float(n_bar)
                )
                own_key = (model_name, landmark, float(n_bar))
                if paper_key not in analysis_by_key or own_key not in analysis_by_key:
                    continue
                paper_data = analysis_by_key[paper_key]
                own_data = analysis_by_key[own_key]
                paper_row = paper_data["row"]
                own_row = own_data["row"]
                ptm_difference = float(
                    np.linalg.norm(own_data["ptm"] - paper_data["ptm"])
                )
                error_ptm_difference = float(
                    np.linalg.norm(
                        (own_data["ptm"] - identity_ptm)
                        - (paper_data["ptm"] - identity_ptm)
                    )
                )
                paper_error_norm = float(
                    np.linalg.norm(paper_data["ptm"] - identity_ptm)
                )
                comparison_rows.append({
                    "model": model_name,
                    "landmark": landmark,
                    "n_bar": float(n_bar),
                    "omega_mhz": float(own_row["omega_mhz"]),
                    "effective_amplitude_sim": float(
                        own_row["effective_amplitude_sim"]
                    ),
                    "detuning_sim": float(detuning_sim),
                    "ptm_frobenius_difference": ptm_difference,
                    "error_ptm_frobenius_difference": error_ptm_difference,
                    "error_ptm_relative_difference": (
                        error_ptm_difference / max(paper_error_norm, 1e-15)
                    ),
                    "own_average_infidelity": float(
                        own_row["average_infidelity"]
                    ),
                    "kirchhoff_average_infidelity": float(
                        paper_row["average_infidelity"]
                    ),
                    "delta_average_infidelity": float(
                        own_row["average_infidelity"]
                        - paper_row["average_infidelity"]
                    ),
                    "own_h_XX_rad_per_gate": float(
                        own_row["h_XX_rad_per_gate"]
                    ),
                    "kirchhoff_h_XX_rad_per_gate": float(
                        paper_row["h_XX_rad_per_gate"]
                    ),
                    "delta_h_XX_rad_per_gate": float(
                        own_row["h_XX_rad_per_gate"]
                        - paper_row["h_XX_rad_per_gate"]
                    ),
                    "own_gamma_XX_per_gate": float(
                        own_row["gamma_XX_per_gate"]
                    ),
                    "kirchhoff_gamma_XX_per_gate": float(
                        paper_row["gamma_XX_per_gate"]
                    ),
                    "delta_gamma_XX_per_gate": float(
                        own_row["gamma_XX_per_gate"]
                        - paper_row["gamma_XX_per_gate"]
                    ),
                })
                for pauli in pauli_labels:
                    coefficient_difference_rows.append({
                        "model": model_name,
                        "landmark": landmark,
                        "n_bar": float(n_bar),
                        "pauli": pauli,
                        "delta_hamiltonian_coefficient_rad_per_gate": float(
                            own_data["h"][pauli] - paper_data["h"][pauli]
                        ),
                        "delta_pauli_dissipator_rate_per_gate": float(
                            own_data["gamma"][pauli]
                            - paper_data["gamma"][pauli]
                        ),
                    })

    model_summary = pd.DataFrame(model_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    comparison = pd.DataFrame(comparison_rows)
    coefficient_differences = pd.DataFrame(coefficient_difference_rows)
    if not model_summary.empty:
        model_summary = model_summary.sort_values(
            ["model", "landmark", "n_bar"]
        ).reset_index(drop=True)
    if not comparison.empty:
        comparison = comparison.sort_values(
            ["model", "landmark", "n_bar"]
        ).reset_index(drop=True)

    model_summary_path = output_dir / "coherent_limit_model_summary.csv"
    comparison_path = output_dir / "coherent_limit_direct_comparison.csv"
    coefficient_path = output_dir / "coherent_limit_generator_coefficients.csv"
    coefficient_difference_path = (
        output_dir / "coherent_limit_generator_coefficient_differences.csv"
    )
    model_summary.to_csv(model_summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    coefficients.to_csv(coefficient_path, index=False)
    coefficient_differences.to_csv(coefficient_difference_path, index=False)
    figure_path = _plot_coherent_limit_comparison(
        model_summary, comparison, output_dir
    )

    expected_own_propagators = len(own_tasks)
    status = {
        "completed_own_propagators": int(completed_own_propagators),
        "expected_own_propagators": int(expected_own_propagators),
        "newly_computed": int(newly_computed),
        "pending_own_propagators": int(
            expected_own_propagators - completed_own_propagators
        ),
        "completed_comparison_points": int(len(comparison)),
        "expected_comparison_points": int(
            len(model_variants) * len(requested_landmarks) * len(nbar_values)
        ),
        "generator_path": "error superoperator -> PTM -> log(PTM)",
        "chi_role": "diagnostic output only; no chi round-trip",
        "drive_mapping": "A_sim = eta * Omega * T / (2 * t_gate_sim)",
        "detuning_mapping": "delta_sim = 2*pi*(K-L)/t_gate_sim",
        "noise_rates": "all zero (coherent limit)",
        "K": K,
        "L": L,
        "eta": eta,
        "phonon_dim": phonon_dim,
        "gate_duration_seconds": gate_duration_seconds,
        "model_variants": model_variants,
    }
    if not model_summary.empty:
        status.update({
            "cp_pass": int(model_summary["cp_pass"].astype(bool).sum()),
            "tp_pass": int(model_summary["tp_pass"].astype(bool).sum()),
            "total_analyzed_channels": int(len(model_summary)),
        })
    return {
        "model_summary": model_summary,
        "comparison": comparison,
        "generator_coefficients": coefficients,
        "generator_coefficient_differences": coefficient_differences,
        "figure_path": figure_path,
        "model_summary_path": model_summary_path,
        "comparison_path": comparison_path,
        "coefficient_path": coefficient_path,
        "coefficient_difference_path": coefficient_difference_path,
        "output_dir": output_dir,
        "status": status,
        "pending": [task for task in own_tasks if not task["cache_path"].exists()],
    }


def _model_hxx_calibration_cache_path(
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    model_name: str,
    omega_per_second: float,
) -> Path:
    cache_payload = {
        **dict(payload),
        "model_name": str(model_name),
        "omega_per_second": float(omega_per_second),
        "implementation": "model_specific_hxx_zero_v1",
    }
    digest = hashlib.sha256(
        json.dumps(
            cache_payload, sort_keys=True, default=_json_safe
        ).encode("utf-8")
    ).hexdigest()[:14]
    return output_dir / "propagator_cache" / (
        f"{_condition_stem(model_name)}__omega_mhz_"
        f"{_condition_stem(f'{float(omega_per_second) / 1e6:.9f}')}"
        f"__{digest}.npz"
    )


def _plot_model_specific_hxx_calibration(
    summary: pd.DataFrame,
    tolerance_rad: float,
    output_dir: Path,
) -> Path | None:
    if summary.empty:
        return None
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5))
    styles = {
        "Kirchhoff_full_carrier": ("black", "o", "-"),
        "own_lamb_dicke": ("#0072B2", "s", "--"),
        "own_eta2_corrected": ("#D55E00", "^", "-."),
    }
    labels = {
        "Kirchhoff_full_carrier": "Kirchhoff full carrier-sideband",
        "own_lamb_dicke": "repository first-sideband (LD)",
        "own_eta2_corrected": r"repository first-sideband ($\eta^2$ corrected)",
    }
    for model_name, curve in summary.groupby("model"):
        curve = curve.sort_values("n_bar")
        color, marker, linestyle = styles.get(model_name, (None, "o", "-"))
        label = labels.get(model_name, model_name)
        axes[0, 0].semilogx(
            curve["n_bar"],
            curve["omega_calibrated_mhz"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
        axes[0, 1].loglog(
            curve["n_bar"],
            np.maximum(np.abs(curve["h_XX_rad_per_gate"]), 1e-18),
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
        axes[1, 0].loglog(
            curve["n_bar"],
            np.maximum(curve["gamma_XX_per_gate"], 1e-18),
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
        axes[1, 1].loglog(
            curve["n_bar"],
            np.maximum(curve["average_infidelity"], 1e-18),
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )

    axes[0, 0].set_xlabel(r"Mean phonon number $\bar n$")
    axes[0, 0].set_ylabel(r"Calibrated carrier-equivalent $\Omega$ [MHz]")
    axes[0, 0].set_title(r"Model-specific solution of $h_{XX}(\Omega)=0$")
    axes[0, 0].grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].axhline(
        float(tolerance_rad),
        color="#777777",
        linestyle=":",
        label="calibration tolerance",
    )
    axes[0, 1].set_xlabel(r"Mean phonon number $\bar n$")
    axes[0, 1].set_ylabel(r"Residual $|h_{XX}|$ [rad/gate]")
    axes[0, 1].set_title("Residual coherent XX error")
    axes[0, 1].grid(True, which="both", alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_xlabel(r"Mean phonon number $\bar n$")
    axes[1, 0].set_ylabel(r"Calibrated $\gamma_{XX}$ [1/gate]")
    axes[1, 0].set_title("Structural stochastic XX error after angle calibration")
    axes[1, 0].grid(True, which="both", alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_xlabel(r"Mean phonon number $\bar n$")
    axes[1, 1].set_ylabel(r"Calibrated $1-F_{\rm avg}$")
    axes[1, 1].set_title("Residual channel infidelity after angle calibration")
    axes[1, 1].grid(True, which="both", alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    figure.suptitle(
        "Separating XX-angle miscalibration from Hamiltonian-structure error",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "model_specific_hxx_zero_calibration.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    figure.savefig(
        output_dir / "model_specific_hxx_zero_calibration.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    return figure_path


def run_model_specific_hxx_zero_calibration_stage(
    config: Mapping[str, Any],
    paper_result: Mapping[str, Any],
    coherent_result: Mapping[str, Any],
    *,
    nbar_values=(0.02, 0.1, 0.3, 1.0),
    model_variants=None,
    omega_bounds_mhz=(1.04, 1.18),
    hxx_tolerance_rad: float = 2e-4,
    max_root_iterations: int = 3,
    t_gate_sim: float = 1.0,
    own_time_points: int = 501,
    own_solver_method: str = "vern9",
    own_solver_max_step: float | None = None,
    paper_solver: str = "adaptive_dop853",
    paper_relative_tolerance: float = 1e-9,
    paper_absolute_tolerance: float = 1e-11,
    run_calibration_propagation: bool = False,
    force_recompute: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Independently set hXX=0 in each model, then compare residual errors."""

    paths = _paths(config)
    output_dir = paths["kirchhoff_model_calibration"]
    cache_dir = output_dir / "propagator_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    convention = _convention(config)
    nbar_values = sorted({float(value) for value in nbar_values})
    if not nbar_values or any(value < 0.0 for value in nbar_values):
        raise ValueError("nbar_values must contain nonnegative values")
    lower_omega = float(omega_bounds_mhz[0]) * 1e6
    upper_omega = float(omega_bounds_mhz[1]) * 1e6
    if not 0.0 < lower_omega < upper_omega:
        raise ValueError("omega_bounds_mhz must be an increasing positive pair")

    paper_status = dict(paper_result.get("status", {}))
    required_status = {
        "K", "L", "eta", "phonon_dim", "sideband_cutoff",
        "gate_duration_seconds",
    }
    missing_status = required_status - set(paper_status)
    if missing_status:
        raise ValueError(
            "paper_result status is missing: "
            + ", ".join(sorted(missing_status))
        )
    K = float(paper_status["K"])
    L = float(paper_status["L"])
    eta = float(paper_status["eta"])
    phonon_dim = int(paper_status["phonon_dim"])
    sideband_cutoff = int(paper_status["sideband_cutoff"])
    gate_duration_seconds = float(paper_status["gate_duration_seconds"])
    paper_model = kirchhoff_paper.build_paper_model(
        K=K,
        L=L,
        eta=eta,
        phonon_dim=phonon_dim,
        sideband_cutoff=sideband_cutoff,
    )
    if model_variants is None:
        model_variants = dict(
            coherent_result.get("status", {}).get("model_variants", {})
        )
    model_variants = {
        str(name): bool(use_full_order)
        for name, use_full_order in dict(model_variants).items()
    }
    if not model_variants:
        raise ValueError("Cell 8.3 model variants are required")
    models = {"Kirchhoff_full_carrier": None, **model_variants}
    detuning_sim = coherent_bridge.effective_detuning(
        K, L, t_gate_sim=t_gate_sim
    )
    omega_4 = float(
        kirchhoff_paper.drive_landmarks(
            K=K,
            L=L,
            gate_duration_seconds=gate_duration_seconds,
            eta=eta,
        )["omega_4"]
    )

    payload = {
        "K": K,
        "L": L,
        "eta": eta,
        "phonon_dim": phonon_dim,
        "sideband_cutoff": sideband_cutoff,
        "gate_duration_seconds": gate_duration_seconds,
        "t_gate_sim": float(t_gate_sim),
        "detuning_sim": float(detuning_sim),
        "own_time_points": int(own_time_points),
        "own_solver_method": str(own_solver_method),
        "own_solver_max_step": own_solver_max_step,
        "paper_solver": str(paper_solver),
        "paper_relative_tolerance": float(paper_relative_tolerance),
        "paper_absolute_tolerance": float(paper_absolute_tolerance),
    }

    source_paths = {model: {} for model in models}
    paper_metadata = pd.DataFrame(
        paper_result.get("propagator_metadata", [])
    )
    if paper_metadata.empty:
        raise FileNotFoundError("Cell 8.1 propagator metadata is required")
    for _, row in paper_metadata.iterrows():
        path = Path(row["cache_path"])
        if path.exists():
            source_paths["Kirchhoff_full_carrier"][
                round(float(row["omega_per_second"]), 9)
            ] = path

    coherent_summary = pd.DataFrame(
        coherent_result.get("model_summary", [])
    )
    if coherent_summary.empty:
        raise FileNotFoundError("Cell 8.3 model summary is required")
    for _, row in coherent_summary.drop_duplicates(
        ["model", "omega_per_second"]
    ).iterrows():
        model_name = str(row["model"])
        if model_name not in source_paths or model_name == "Kirchhoff_full_carrier":
            continue
        path = Path(row["source_cache"])
        if path.exists():
            source_paths[model_name][
                round(float(row["omega_per_second"]), 9)
            ] = path

    if not force_recompute:
        for path in sorted(cache_dir.glob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata_json"].item()))
                model_name = str(metadata["model"])
                omega = float(metadata["omega_per_second"])
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if model_name in source_paths:
                source_paths[model_name][round(omega, 9)] = path

    propagator_memory = {}
    analysis_memory = {}
    evaluation_rows = []
    pending_requests = []
    newly_computed = 0

    def ensure_propagator(model_name, omega):
        nonlocal newly_computed
        omega = float(omega)
        key = round(omega, 9)
        existing = source_paths[model_name].get(key)
        if existing is not None and existing.exists():
            return existing
        use_full_order = models[model_name]
        cache_path = _model_hxx_calibration_cache_path(
            output_dir,
            {**payload, "use_full_order": use_full_order},
            model_name=model_name,
            omega_per_second=omega,
        )
        if cache_path.exists() and not force_recompute:
            source_paths[model_name][key] = cache_path
            return cache_path
        if not run_calibration_propagation:
            request = (model_name, omega)
            if request not in pending_requests:
                pending_requests.append(request)
            return None

        start = time.perf_counter()
        if model_name == "Kirchhoff_full_carrier":
            propagator, metadata = kirchhoff_paper.propagate(
                paper_model,
                omega_per_second=omega,
                gate_duration_seconds=gate_duration_seconds,
                solver=paper_solver,
                relative_tolerance=paper_relative_tolerance,
                absolute_tolerance=paper_absolute_tolerance,
            )
        else:
            amplitude = coherent_bridge.carrier_to_effective_sideband_amplitude(
                omega,
                eta=eta,
                gate_duration_seconds=gate_duration_seconds,
                t_gate_sim=t_gate_sim,
            )
            propagator_qobj, metadata = mg.coherent_ms_propagator(
                amplitude,
                detuning_sim,
                rho0=0.0,
                phonon_dim=phonon_dim,
                eta=eta,
                use_full_order=bool(use_full_order),
                time_points=own_time_points,
                t_gate_sim=t_gate_sim,
                solver_method=own_solver_method,
                solver_max_step=own_solver_max_step,
            )
            propagator = np.asarray(propagator_qobj.full(), dtype=complex)
            metadata["effective_amplitude_sim"] = float(amplitude)
        metadata = {
            **metadata,
            "model": model_name,
            "omega_per_second": omega,
            "wall_time_seconds": float(time.perf_counter() - start),
        }
        temporary_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
        np.savez_compressed(
            temporary_path,
            propagator=np.asarray(propagator, dtype=complex),
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, default=_json_safe)
            ),
        )
        temporary_path.replace(cache_path)
        source_paths[model_name][key] = cache_path
        newly_computed += 1
        return cache_path

    def evaluate(model_name, omega, n_bar):
        omega = float(omega)
        n_bar = float(n_bar)
        memory_key = (model_name, round(omega, 9), round(n_bar, 12))
        if memory_key in analysis_memory:
            return analysis_memory[memory_key]
        path = ensure_propagator(model_name, omega)
        if path is None:
            return None
        if path not in propagator_memory:
            with np.load(path, allow_pickle=False) as data:
                propagator_memory[path] = np.asarray(
                    data["propagator"], dtype=complex
                )
        propagator = propagator_memory[path]
        if model_name == "Kirchhoff_full_carrier":
            _, error_super, thermal_metadata = (
                kirchhoff_paper.kirchhoff_xx_error_channel(
                    propagator,
                    paper_model,
                    n_bar=n_bar,
                    convention=convention,
                )
            )
            effective_amplitude = np.nan
        else:
            _, error_super, thermal_metadata = coherent_bridge.own_xx_error_channel(
                propagator,
                phonon_dim=phonon_dim,
                n_bar=n_bar,
                convention=convention,
            )
            effective_amplitude = (
                coherent_bridge.carrier_to_effective_sideband_amplitude(
                    omega,
                    eta=eta,
                    gate_duration_seconds=gate_duration_seconds,
                    t_gate_sim=t_gate_sim,
                )
            )
        observables = qpt_analysis.extract_pauli_generator_from_superoperator(
            error_super
        )
        ptm = np.asarray(observables.pop("ptm"), dtype=complex)
        chi = np.asarray(
            observables.pop("chi_trace_normalized"), dtype=complex
        )
        hamiltonian = observables.pop(
            "hamiltonian_coefficients_rad_per_gate"
        )
        dissipator = observables.pop("pauli_dissipator_rates_per_gate")
        physicality = mg.choi_physicality_metrics(error_super)
        result = {
            "model": model_name,
            "n_bar": n_bar,
            "omega_per_second": omega,
            "omega_mhz": omega / 1e6,
            "effective_amplitude_sim": float(effective_amplitude),
            "h_XX_rad_per_gate": float(hamiltonian["XX"]),
            "gamma_XX_per_gate": float(dissipator["XX"]),
            **observables,
            "thermal_tail_mass": float(thermal_metadata["thermal_tail_mass"]),
            "cp_pass": bool(physicality["cp_pass"]),
            "tp_pass": bool(physicality["tp_pass"]),
            "tp_frobenius_error": float(physicality["tp_frobenius_error"]),
            "source_cache": str(path),
            "_ptm": ptm,
            "_chi": chi,
            "_h": hamiltonian,
            "_gamma": dissipator,
        }
        analysis_memory[memory_key] = result
        evaluation_rows.append({
            key: value
            for key, value in result.items()
            if not key.startswith("_")
        })
        return result

    calibration_rows = []
    calibrated_data = {}
    root_iterator = [
        (model_name, n_bar)
        for model_name in models
        for n_bar in nbar_values
    ]
    for model_name, n_bar in tqdm(
        root_iterator,
        desc="8.4 model-specific hXX=0",
        unit="root",
        disable=not show_progress,
    ):
        seed_omegas = sorted(source_paths[model_name])
        root = hxx_calibration.find_hxx_zero(
            lambda omega, m=model_name, n=n_bar: evaluate(m, omega, n),
            seed_omegas,
            lower_bound=lower_omega,
            upper_bound=upper_omega,
            tolerance_rad=hxx_tolerance_rad,
            max_iterations=max_root_iterations,
        )
        best = root["best"]
        if best is None:
            continue
        bracket = root["bracket"]
        baseline_match = coherent_summary.loc[
            (coherent_summary["model"] == model_name)
            & (coherent_summary["landmark"] == "Omega_4")
            & np.isclose(coherent_summary["n_bar"].astype(float), n_bar)
        ]
        baseline = baseline_match.iloc[-1] if not baseline_match.empty else None
        row = {
            key: value
            for key, value in best.items()
            if not key.startswith("_")
        }
        row.update({
            "omega_calibrated_mhz": float(best["omega_mhz"]),
            "omega_factor_vs_omega4": float(
                best["omega_per_second"] / omega_4
            ),
            "calibration_converged": bool(root["converged"]),
            "calibration_pending": bool(root["pending"]),
            "calibration_iterations": int(root["iterations"]),
            "calibration_evaluated_points": int(root["evaluated_points"]),
            "bracket_lower_mhz": (
                np.nan if bracket is None else float(bracket[0]) / 1e6
            ),
            "bracket_upper_mhz": (
                np.nan if bracket is None else float(bracket[1]) / 1e6
            ),
            "precalibration_omega4_h_XX": (
                np.nan if baseline is None else float(baseline["h_XX_rad_per_gate"])
            ),
            "precalibration_omega4_gamma_XX": (
                np.nan if baseline is None else float(baseline["gamma_XX_per_gate"])
            ),
            "precalibration_omega4_infidelity": (
                np.nan if baseline is None else float(baseline["average_infidelity"])
            ),
        })
        calibration_rows.append(row)
        if root["converged"]:
            calibrated_data[(model_name, n_bar)] = best

    summary = pd.DataFrame(calibration_rows)
    if not summary.empty:
        summary = summary.sort_values(["model", "n_bar"]).reset_index(drop=True)
    history = pd.DataFrame(evaluation_rows)
    if not history.empty:
        history = history.sort_values(
            ["model", "n_bar", "omega_per_second"]
        ).reset_index(drop=True)

    coefficient_rows = []
    comparison_rows = []
    coefficient_difference_rows = []
    pauli_labels = [label for label, _ in mg.pauli_labels_and_weights()][1:]
    for (model_name, n_bar), data in calibrated_data.items():
        for pauli in pauli_labels:
            coefficient_rows.append({
                "model": model_name,
                "n_bar": n_bar,
                "omega_calibrated_mhz": float(data["omega_mhz"]),
                "pauli": pauli,
                "hamiltonian_coefficient_rad_per_gate": float(
                    data["_h"][pauli]
                ),
                "pauli_dissipator_rate_per_gate": float(
                    data["_gamma"][pauli]
                ),
            })
    for model_name in model_variants:
        for n_bar in nbar_values:
            paper_key = ("Kirchhoff_full_carrier", n_bar)
            own_key = (model_name, n_bar)
            if paper_key not in calibrated_data or own_key not in calibrated_data:
                continue
            paper_data = calibrated_data[paper_key]
            own_data = calibrated_data[own_key]
            ptm_difference = float(
                np.linalg.norm(own_data["_ptm"] - paper_data["_ptm"])
            )
            chi_difference = float(
                np.linalg.norm(own_data["_chi"] - paper_data["_chi"])
            )
            comparison_rows.append({
                "model": model_name,
                "n_bar": n_bar,
                "own_omega_calibrated_mhz": float(own_data["omega_mhz"]),
                "kirchhoff_omega_calibrated_mhz": float(
                    paper_data["omega_mhz"]
                ),
                "delta_calibrated_omega_mhz": float(
                    own_data["omega_mhz"] - paper_data["omega_mhz"]
                ),
                "own_h_XX_residual": float(own_data["h_XX_rad_per_gate"]),
                "kirchhoff_h_XX_residual": float(
                    paper_data["h_XX_rad_per_gate"]
                ),
                "own_gamma_XX_per_gate": float(own_data["gamma_XX_per_gate"]),
                "kirchhoff_gamma_XX_per_gate": float(
                    paper_data["gamma_XX_per_gate"]
                ),
                "delta_gamma_XX_per_gate": float(
                    own_data["gamma_XX_per_gate"]
                    - paper_data["gamma_XX_per_gate"]
                ),
                "relative_gamma_XX_difference": float(
                    (
                        own_data["gamma_XX_per_gate"]
                        - paper_data["gamma_XX_per_gate"]
                    ) / max(abs(paper_data["gamma_XX_per_gate"]), 1e-15)
                ),
                "own_average_infidelity": float(own_data["average_infidelity"]),
                "kirchhoff_average_infidelity": float(
                    paper_data["average_infidelity"]
                ),
                "delta_average_infidelity": float(
                    own_data["average_infidelity"]
                    - paper_data["average_infidelity"]
                ),
                "relative_infidelity_difference": float(
                    (
                        own_data["average_infidelity"]
                        - paper_data["average_infidelity"]
                    ) / max(abs(paper_data["average_infidelity"]), 1e-15)
                ),
                "calibrated_ptm_frobenius_difference": ptm_difference,
                "calibrated_chi_frobenius_difference": chi_difference,
            })
            for pauli in pauli_labels:
                coefficient_difference_rows.append({
                    "model": model_name,
                    "n_bar": n_bar,
                    "pauli": pauli,
                    "delta_hamiltonian_coefficient_rad_per_gate": float(
                        own_data["_h"][pauli] - paper_data["_h"][pauli]
                    ),
                    "delta_pauli_dissipator_rate_per_gate": float(
                        own_data["_gamma"][pauli]
                        - paper_data["_gamma"][pauli]
                    ),
                })

    comparison = pd.DataFrame(comparison_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficient_differences = pd.DataFrame(coefficient_difference_rows)
    summary_path = output_dir / "model_specific_hxx_zero_summary.csv"
    history_path = output_dir / "model_specific_hxx_calibration_history.csv"
    comparison_path = output_dir / "postcalibration_structural_comparison.csv"
    coefficient_path = output_dir / "postcalibration_generator_coefficients.csv"
    coefficient_difference_path = (
        output_dir / "postcalibration_generator_coefficient_differences.csv"
    )
    summary.to_csv(summary_path, index=False)
    history.to_csv(history_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    coefficients.to_csv(coefficient_path, index=False)
    coefficient_differences.to_csv(coefficient_difference_path, index=False)
    figure_path = _plot_model_specific_hxx_calibration(
        summary, hxx_tolerance_rad, output_dir
    )

    expected_roots = len(models) * len(nbar_values)
    converged_roots = int(
        summary.get("calibration_converged", pd.Series(dtype=bool))
        .astype(bool)
        .sum()
    )
    return {
        "summary": summary,
        "history": history,
        "comparison": comparison,
        "generator_coefficients": coefficients,
        "generator_coefficient_differences": coefficient_differences,
        "figure_path": figure_path,
        "summary_path": summary_path,
        "history_path": history_path,
        "comparison_path": comparison_path,
        "coefficient_path": coefficient_path,
        "coefficient_difference_path": coefficient_difference_path,
        "output_dir": output_dir,
        "status": {
            "converged_roots": converged_roots,
            "expected_roots": int(expected_roots),
            "pending_roots": int(expected_roots - converged_roots),
            "newly_computed_propagators": int(newly_computed),
            "pending_propagator_requests": int(len(pending_requests)),
            "hxx_tolerance_rad": float(hxx_tolerance_rad),
            "omega_bounds_mhz": [lower_omega / 1e6, upper_omega / 1e6],
            "generator_path": "error superoperator -> PTM -> log(PTM)",
            "comparison_meaning": (
                "hXX individually nulled; residual gamma/infidelity are structural"
            ),
            "model_variants": model_variants,
        },
        "pending_requests": pending_requests,
    }


def _normalized_control_envelope(shape_name: str, number_of_points: int):
    phase = np.linspace(0.0, 1.0, int(number_of_points))
    if shape_name == "sin2":
        envelope = np.sin(np.pi * phase) ** 2
    elif shape_name == "blackman":
        envelope = np.blackman(int(number_of_points))
    else:
        raise ValueError(f"Unknown pulse shape: {shape_name}")
    rms = float(np.sqrt(np.mean(envelope**2)))
    if rms <= 0.0:
        raise ValueError("Pulse RMS must be positive")
    return envelope / rms


def _control_candidates(
    params: Mapping[str, Any],
    amplitude_factors: Iterable[float],
    gate_time_factors: Iterable[float],
    detuning_factors: Iterable[float],
    pulse_shapes: Iterable[str],
) -> list[dict[str, Any]]:
    base_gate_time_sim = 2.0 * np.pi / abs(float(params["delta"]))
    candidates = []
    for factor in amplitude_factors:
        if np.isclose(factor, 1.0):
            continue
        candidates.append({
            "name": f"amplitude_{float(factor):.3f}",
            "kind": "amplitude",
            "factor": float(factor),
            "overrides": {"A": float(params["A"]) * float(factor)},
        })
    for factor in gate_time_factors:
        candidates.append({
            "name": f"gate_time_{float(factor):.3f}",
            "kind": "gate_time",
            "factor": float(factor),
            "overrides": {
                "t_gate_sim": base_gate_time_sim * float(factor),
                "t_gate_phys": float(params["t_gate_phys"]) * float(factor),
            },
        })
    for factor in detuning_factors:
        candidates.append({
            "name": f"detuning_{float(factor):.3f}",
            "kind": "detuning",
            "factor": float(factor),
            "overrides": {
                "delta": float(params["delta"]) * float(factor),
                "t_gate_sim": base_gate_time_sim / float(factor),
                "t_gate_phys": float(params["t_gate_phys"]) / float(factor),
            },
        })
    for shape in pulse_shapes:
        envelope = _normalized_control_envelope(
            shape, int(params["time_points"])
        )
        candidates.append({
            "name": f"pulse_{shape}_rms",
            "kind": "pulse",
            "factor": 1.0,
            "overrides": {
                "A": float(params["A"]) * envelope,
                "t_gate_sim": base_gate_time_sim,
            },
        })
    return candidates


def _control_observables(chi, config: Mapping[str, Any]) -> dict[str, float]:
    labels = [label for label, _ in mg.pauli_labels_and_weights()]
    ii = labels.index("II")
    xx = labels.index("XX")
    ix = labels.index("IX")
    xi = labels.index("XI")
    projected = _extract(chi, config)
    projected_chi = projected.pop("projected_chi")
    ii_value = float(max(np.real(projected_chi[ii, ii]), 1e-15))
    coherent_weight = float(abs(projected_chi[ii, xx]) ** 2 / ii_value)
    chi_xx = float(np.real(projected_chi[xx, xx]))
    chi_ix = float(np.real(projected_chi[ix, ix]))
    chi_xi = float(np.real(projected_chi[xi, xi]))
    return {
        **projected,
        "chi_II_II": ii_value,
        "chi_IX_IX": chi_ix,
        "chi_XI_XI": chi_xi,
        "coherent_XX_equivalent_weight": coherent_weight,
        "control_score": chi_xx + coherent_weight + chi_ix + chi_xi,
    }


def _simple_qpt_cache_path(directory: Path, condition: str, n_bar: float) -> Path:
    return directory / (
        f"{_condition_stem(condition)}__nbar_{_legacy_nbar_stem(n_bar)}.npz"
    )


def run_physical_control_stage(
    config: Mapping[str, Any],
    nbar_values: Iterable[float],
    *,
    run_qpt: bool = False,
    force_recompute: bool = False,
    show_progress: bool = True,
    amplitude_factors=(0.95, 1.00, 1.05, 1.10, 1.15, 1.20),
    gate_time_factors=(0.97, 1.03),
    detuning_factors=(0.97, 1.03),
    pulse_shapes=("sin2", "blackman"),
) -> dict[str, Any]:
    """Compare amplitude, duration, detuning and smooth-pulse controls."""

    paths = _paths(config)
    params = _simulation_params(config)
    selected_nbars = [float(value) for value in nbar_values]
    candidates = _control_candidates(
        params,
        amplitude_factors,
        gate_time_factors,
        detuning_factors,
        pulse_shapes,
    )
    pending = []

    candidate_iterator = tqdm(
        candidates,
        desc="Physical controls",
        unit="control",
        disable=not show_progress,
    )
    for candidate in candidate_iterator:
        if show_progress and hasattr(candidate_iterator, "set_postfix_str"):
            candidate_iterator.set_postfix_str(candidate["name"])
        missing = [
            n_bar
            for n_bar in selected_nbars
            if force_recompute
            or not _simple_qpt_cache_path(
                paths["control"], candidate["name"], n_bar
            ).exists()
        ]
        if missing and run_qpt:
            overrides = dict(candidate["overrides"])
            overrides.update({
                "parallel_workers": int(config.get("FAST_PROCESS_WORKERS", 4)),
                # ms_gate_functions displays evolution-level progress for the
                # five nbar points within the current control candidate.
                "show_progress": bool(show_progress),
            })
            for result in qpt_analysis.calculate_error_channel_batch(
                missing,
                params,
                overrides,
                convention=_convention(config),
            ):
                result["metadata"].update({
                    "candidate": candidate["name"],
                    "kind": candidate["kind"],
                    "factor": candidate["factor"],
                })
                qpt_analysis.save_qpt_point(
                    _simple_qpt_cache_path(
                        paths["control"], candidate["name"], result["n_bar"]
                    ),
                    result["n_bar"],
                    candidate["name"],
                    result["chi"],
                    result["metadata"],
                )
        elif missing:
            pending.extend(
                {"candidate": candidate["name"], "n_bar": n_bar}
                for n_bar in missing
            )

    rows = []
    for n_bar in selected_nbars:
        baseline_chi = _load_cptp_chi(paths, n_bar)
        rows.append({
            "n_bar": n_bar,
            "candidate": "baseline",
            "kind": "baseline",
            "factor": 1.0,
            **_control_observables(baseline_chi, config),
        })
        for candidate in candidates:
            path = _simple_qpt_cache_path(
                paths["control"], candidate["name"], n_bar
            )
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                chi = np.asarray(data["chi_trace_normalized"], dtype=complex)
            rows.append({
                "n_bar": n_bar,
                "candidate": candidate["name"],
                "kind": candidate["kind"],
                "factor": candidate["factor"],
                **_control_observables(chi, config),
            })
    summary = pd.DataFrame(rows).sort_values(["n_bar", "candidate"])
    summary.to_csv(paths["control"] / "physical_control_qpt_summary.csv", index=False)
    nonbaseline = summary.loc[summary["candidate"] != "baseline"]
    best = (
        summary.sort_values("control_score")
        .groupby("n_bar", as_index=False)
        .first()
    )
    best.to_csv(paths["control"] / "best_physical_control_by_nbar.csv", index=False)

    if not nonbaseline.empty:
        figure, axis = plt.subplots(figsize=(10.0, 6.0))
        for candidate, group in summary.groupby("candidate"):
            axis.semilogy(
                group["n_bar"],
                group["control_score"],
                marker="o",
                label=candidate,
            )
        axis.set_xlabel(r"Mean phonon number $\bar n$")
        axis.set_ylabel("CPTP χ-based control score")
        axis.set_title("Full-Hamiltonian physical-control comparison")
        axis.grid(True, which="both", alpha=0.28)
        axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        figure.savefig(
            paths["control"] / "physical_control_qpt_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        figure.savefig(
            paths["control"] / "physical_control_qpt_comparison.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)

    expected = len(candidates) * len(selected_nbars)
    completed = len(nonbaseline)
    return {
        "summary": summary,
        "best": best,
        "candidates": pd.DataFrame(candidates).drop(columns="overrides"),
        "status": {
            "completed_points": completed,
            "expected_points": expected,
            "pending": pending,
            "run_qpt": bool(run_qpt),
        },
    }


def run_physical_control_screening_report(
    config: Mapping[str, Any],
    nbar_values: Iterable[float],
) -> dict[str, Any]:
    """Create a read-only report from completed physical-control screening.

    No QPT is run here.  The report shows every candidate, baseline-normalized
    ratios, and separate winners for coherent-angle suppression, physical
    infidelity, and the chi-based aggregate control score.
    """

    paths = _paths(config)
    summary_path = paths["control"] / "physical_control_qpt_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run the physical-control cell first."
        )
    requested_nbars = sorted({float(value) for value in nbar_values})
    if not requested_nbars:
        raise ValueError("nbar_values must contain at least one value")
    full_summary = pd.read_csv(summary_path)
    selected_parts = [
        full_summary.loc[
            np.isclose(full_summary["n_bar"].astype(float), n_bar)
        ]
        for n_bar in requested_nbars
    ]
    missing_nbars = [
        n_bar
        for n_bar, part in zip(requested_nbars, selected_parts)
        if part.empty
    ]
    if missing_nbars:
        raise ValueError(
            "Physical-control summary is missing n_bar="
            + ", ".join(f"{value:g}" for value in missing_nbars)
        )
    screening = pd.concat(selected_parts, ignore_index=True).copy()
    screening["abs_h_XX_rad_per_gate"] = np.abs(
        screening["h_XX_rad_per_gate"].astype(float)
    )
    baseline = screening.loc[
        screening["candidate"] == "baseline",
        [
            "n_bar",
            "abs_h_XX_rad_per_gate",
            "gamma_XX_per_gate",
            "average_infidelity",
            "control_score",
        ],
    ].rename(columns={
        "abs_h_XX_rad_per_gate": "baseline_abs_h_XX_rad_per_gate",
        "gamma_XX_per_gate": "baseline_gamma_XX_per_gate",
        "average_infidelity": "baseline_average_infidelity",
        "control_score": "baseline_control_score",
    })
    if baseline["n_bar"].nunique() != len(requested_nbars):
        raise ValueError("Each requested n_bar must have exactly one baseline row")
    screening = screening.merge(baseline, on="n_bar", how="left")
    screening["abs_h_XX_ratio_to_baseline"] = (
        screening["abs_h_XX_rad_per_gate"]
        / np.maximum(screening["baseline_abs_h_XX_rad_per_gate"], 1e-15)
    )
    screening["gamma_XX_ratio_to_baseline"] = (
        screening["gamma_XX_per_gate"]
        / np.maximum(screening["baseline_gamma_XX_per_gate"], 1e-15)
    )
    screening["infidelity_ratio_to_baseline"] = (
        screening["average_infidelity"]
        / np.maximum(screening["baseline_average_infidelity"], 1e-15)
    )
    screening["control_score_ratio_to_baseline"] = (
        screening["control_score"]
        / np.maximum(screening["baseline_control_score"], 1e-15)
    )
    screening["generator_fit_warning"] = (
        screening["gamma_nnls_residual"].astype(float) > 0.2
    ) | (
        screening["generator_imaginary_frobenius_norm"].astype(float) > 1e-6
    )

    winner_rows = []
    for n_bar, group in screening.groupby("n_bar", sort=True):
        baseline_row = group.loc[group["candidate"] == "baseline"].iloc[0]
        best_h = group.loc[group["abs_h_XX_rad_per_gate"].idxmin()]
        best_infidelity = group.loc[group["average_infidelity"].idxmin()]
        best_score = group.loc[group["control_score"].idxmin()]
        pulse_rows = group.loc[group["kind"] == "pulse"]
        winner_rows.append({
            "n_bar": float(n_bar),
            "best_abs_h_XX_candidate": best_h["candidate"],
            "best_abs_h_XX_rad_per_gate": best_h[
                "abs_h_XX_rad_per_gate"
            ],
            "h_XX_reduction_factor": (
                baseline_row["abs_h_XX_rad_per_gate"]
                / max(float(best_h["abs_h_XX_rad_per_gate"]), 1e-15)
            ),
            "best_infidelity_candidate": best_infidelity["candidate"],
            "best_average_infidelity": best_infidelity[
                "average_infidelity"
            ],
            "infidelity_improvement_factor": (
                baseline_row["average_infidelity"]
                / max(float(best_infidelity["average_infidelity"]), 1e-15)
            ),
            "best_infidelity_abs_h_XX_ratio": best_infidelity[
                "abs_h_XX_ratio_to_baseline"
            ],
            "best_infidelity_gamma_XX_ratio": best_infidelity[
                "gamma_XX_ratio_to_baseline"
            ],
            "best_control_score_candidate": best_score["candidate"],
            "control_score_improvement_factor": (
                baseline_row["control_score"]
                / max(float(best_score["control_score"]), 1e-15)
            ),
            "best_pulse_infidelity": (
                float(pulse_rows["average_infidelity"].min())
                if not pulse_rows.empty else np.nan
            ),
        })
    winners = pd.DataFrame(winner_rows)

    report_dir = paths["control"] / "screening_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    screening_path = report_dir / "physical_control_screening_selected.csv"
    winners_path = report_dir / "physical_control_screening_winners.csv"
    screening.to_csv(screening_path, index=False)
    winners.to_csv(winners_path, index=False)

    candidates = screening[
        ["candidate", "kind", "factor"]
    ].drop_duplicates().sort_values(["kind", "factor", "candidate"])
    color_maps = {
        "amplitude": plt.get_cmap("Blues"),
        "detuning": plt.get_cmap("Oranges"),
        "gate_time": plt.get_cmap("Greens"),
        "pulse": plt.get_cmap("Reds"),
    }
    styles = {"baseline": {"color": "black", "linewidth": 3.0}}
    for kind, kind_rows in candidates.loc[
        candidates["kind"] != "baseline"
    ].groupby("kind", sort=False):
        values = np.linspace(0.45, 0.9, len(kind_rows))
        for shade, (_, row) in zip(values, kind_rows.iterrows()):
            styles[str(row["candidate"])] = {
                "color": color_maps.get(kind, plt.get_cmap("Purples"))(shade),
                "linewidth": 1.8,
                "linestyle": "--" if kind == "pulse" else "-",
            }

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))
    panels = [
        (
            axes[0, 0], "abs_h_XX_rad_per_gate",
            r"$|h_{XX}|$ [rad/gate]", True,
        ),
        (
            axes[0, 1], "gamma_XX_per_gate",
            r"$\gamma_{XX}$ [/gate]", True,
        ),
        (
            axes[1, 0], "average_infidelity",
            r"Physical average infidelity $1-F_{\rm avg}$", True,
        ),
        (
            axes[1, 1], "infidelity_ratio_to_baseline",
            "Infidelity / baseline", True,
        ),
    ]
    for candidate, group in screening.groupby("candidate", sort=False):
        group = group.sort_values("n_bar")
        style = styles.get(str(candidate), {})
        label = str(candidate).replace("_", " ")
        for axis, metric, _, log_scale in panels:
            values = group[metric].to_numpy(dtype=float)
            if log_scale:
                values = np.maximum(values, 1e-8)
            axis.plot(
                group["n_bar"], values, marker="o", markersize=4.5,
                label=label, **style,
            )
    for axis, _, ylabel, log_scale in panels:
        axis.set_xlabel(r"Mean phonon number $\bar n$")
        axis.set_ylabel(ylabel)
        axis.set_xticks(requested_nbars)
        if log_scale:
            axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[1, 1].axhline(1.0, color="black", linestyle=":", linewidth=1.4)
    figure.suptitle(
        "Physical-control screening: full-Hamiltonian QPT",
        y=0.995,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=4, fontsize=8,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.text(
        0.5, 0.09,
        "Dashed pulse curves use RMS-normalized, angle-uncalibrated pulses; "
        "their hXX/gamma generator values are not a fair calibrated comparison.",
        ha="center", fontsize=9, color="#8B0000",
    )
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.97))
    figure_path = report_dir / "physical_control_screening_overview.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    figure.savefig(
        report_dir / "physical_control_screening_overview.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    candidate_count = screening["candidate"].nunique()
    expected_points = len(requested_nbars) * candidate_count
    return {
        "screening": screening,
        "winners": winners,
        "figure_path": figure_path,
        "screening_path": screening_path,
        "winners_path": winners_path,
        "status": {
            "nbar_values": requested_nbars,
            "candidate_count_including_baseline": int(candidate_count),
            "completed_points": int(len(screening)),
            "expected_points": int(expected_points),
            "generator_warning_points": int(
                screening["generator_fit_warning"].sum()
            ),
        },
    }


def _quadratic_screening_optimum(
    frame: pd.DataFrame,
    *,
    metric: str = "average_infidelity",
) -> dict[str, float]:
    """Estimate a bounded continuous optimum from a completed 1-D scan."""

    points = (
        frame[["factor", metric]]
        .dropna()
        .groupby("factor", as_index=False)[metric]
        .mean()
        .sort_values("factor")
    )
    if points.empty:
        raise ValueError("The screening slice has no finite optimization points")
    discrete = points.loc[points[metric].idxmin()]
    optimum = float(discrete["factor"])
    fit_used = False
    if len(points) >= 3:
        distances = np.abs(points["factor"].to_numpy(float) - optimum)
        local = points.iloc[np.argsort(distances)[:3]].sort_values("factor")
        x = local["factor"].to_numpy(float)
        y = local[metric].to_numpy(float)
        if len(np.unique(x)) == 3:
            quadratic = np.polyfit(x, y, 2)
            if quadratic[0] > 0.0:
                vertex = -quadratic[1] / (2.0 * quadratic[0])
                optimum = float(np.clip(vertex, points["factor"].min(), points["factor"].max()))
                fit_used = True
    return {
        "factor": optimum,
        "discrete_best_factor": float(discrete["factor"]),
        "discrete_best_metric": float(discrete[metric]),
        "quadratic_fit_used": bool(fit_used),
        "scan_factor_min": float(points["factor"].min()),
        "scan_factor_max": float(points["factor"].max()),
    }


def _fair_pulse_envelope(shape: str, number_of_points: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, int(number_of_points))
    if shape == "sin2":
        envelope = np.sin(np.pi * u) ** 2
    elif shape == "blackman":
        envelope = 0.42 - 0.5 * np.cos(2.0 * np.pi * u) + 0.08 * np.cos(
            4.0 * np.pi * u
        )
    else:
        raise ValueError(f"Unknown fair-comparison pulse shape: {shape}")
    return np.maximum(envelope, 0.0)


def _closed_pulse_geometric_calibration(
    shape: str,
    *,
    gate_time_sim: float,
    closure_cycles: float,
    target_xx_angle_rad: float,
    integration_points: int = 4001,
) -> dict[str, float]:
    """Calibrate ideal-LD closure and geometric phase for one pulse shape."""

    time_grid = np.linspace(0.0, float(gate_time_sim), int(integration_points))
    envelope = _fair_pulse_envelope(shape, len(time_grid))
    detuning = 2.0 * np.pi * float(closure_cycles) / float(gate_time_sim)
    phase = detuning * time_grid
    inner = np.zeros_like(time_grid, dtype=complex)
    inner[1:] = np.cumsum(
        0.5
        * (
            envelope[1:] * np.exp(-1j * phase[1:])
            + envelope[:-1] * np.exp(-1j * phase[:-1])
        )
        * np.diff(time_grid)
    )
    geometric_integral = float(
        np.trapz(
            np.imag(envelope * np.exp(1j * phase) * inner),
            time_grid,
        )
    )
    if geometric_integral <= 0.0:
        raise ValueError(
            f"Pulse {shape} has non-positive geometric phase at "
            f"closure_cycles={closure_cycles:g}"
        )
    peak_amplitude = float(
        np.sqrt(float(target_xx_angle_rad) / (2.0 * geometric_integral))
    )
    closure = np.trapz(
        peak_amplitude * envelope * np.exp(1j * phase), time_grid
    )
    normalization = max(
        float(np.trapz(np.abs(peak_amplitude * envelope), time_grid)),
        1e-15,
    )
    return {
        "shape": shape,
        "closure_cycles": float(closure_cycles),
        "detuning": detuning,
        "ideal_peak_amplitude": peak_amplitude,
        "ideal_rms_amplitude": float(
            np.sqrt(np.mean((peak_amplitude * envelope) ** 2))
        ),
        "geometric_integral_unit_amplitude": geometric_integral,
        "relative_closure_residual": float(abs(closure) / normalization),
    }


def _fair_control_cache_path(
    directory: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> Path:
    params = _simulation_params(config)
    signature_keys = [
        "time_points", "t_gate_phys", "heating_rate_phys",
        "dephasing_rate_phys", "T2_star", "rayleigh_rate_phys",
        "raman_rate_phys", "eta", "use_full_order",
    ]
    payload = {
        "condition": plan["condition"],
        "n_bar": float(plan["n_bar"]),
        "amplitude_peak": float(plan["amplitude_peak"]),
        "detuning": float(plan["detuning"]),
        "t_gate_sim": float(plan["t_gate_sim"]),
        "t_gate_phys": float(plan["t_gate_phys"]),
        "shape": plan.get("shape", "rectangular"),
        "scattering_scales_with_intensity": bool(
            plan["scattering_scales_with_intensity"]
        ),
        "parameters": {key: params.get(key) for key in signature_keys},
        "convention": _convention(config),
        "calibration_version": "fair_control_v1",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=_json_safe).encode("utf-8")
    ).hexdigest()[:14]
    return directory / (
        f"{_condition_stem(plan['condition'])}__nbar_"
        f"{_compact_nbar_stem(plan['n_bar'])}__{digest}.npz"
    )


def _plot_fair_control_comparison(
    comparison: pd.DataFrame,
    output_dir: Path,
    nbar_values: Iterable[float],
) -> Path | None:
    if comparison.empty:
        return None
    frame = comparison.copy()
    baseline = frame.loc[frame["condition"] == "baseline", [
        "n_bar", "h_XX_rad_per_gate", "gamma_XX_per_gate",
        "average_infidelity",
    ]].rename(columns={
        "h_XX_rad_per_gate": "baseline_h_XX_rad_per_gate",
        "gamma_XX_per_gate": "baseline_gamma_XX_per_gate",
        "average_infidelity": "baseline_average_infidelity",
    })
    frame = frame.merge(baseline, on="n_bar", how="left")
    frame["abs_h_XX_rad_per_gate"] = np.abs(frame["h_XX_rad_per_gate"])
    frame["infidelity_ratio_to_baseline"] = (
        frame["average_infidelity"]
        / np.maximum(frame["baseline_average_infidelity"], 1e-15)
    )
    colors = {
        "baseline": "black",
        "rectangular_A_infidelity": "#0072B2",
        "drive_detuning_joint": "#E69F00",
        "gate_time_closed_opt": "#009E73",
        "pulse_sin2_calibrated": "#CC79A7",
        "pulse_blackman_calibrated": "#D55E00",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5))
    panels = [
        (axes[0, 0], "abs_h_XX_rad_per_gate", r"$|h_{XX}|$ [rad/gate]"),
        (axes[0, 1], "gamma_XX_per_gate", r"$\gamma_{XX}$ [/gate]"),
        (
            axes[1, 0], "average_infidelity",
            r"Physical average infidelity $1-F_{\rm avg}$",
        ),
        (
            axes[1, 1], "infidelity_ratio_to_baseline",
            "Infidelity / baseline",
        ),
    ]
    for condition, group in frame.groupby("condition", sort=False):
        group = group.sort_values("n_bar")
        for axis, metric, _ in panels:
            axis.plot(
                group["n_bar"],
                np.maximum(group[metric].to_numpy(float), 1e-9),
                marker="o",
                linewidth=2.8 if condition == "baseline" else 2.0,
                label=str(condition).replace("_", " "),
                color=colors.get(str(condition)),
            )
    for axis, _, ylabel in panels:
        axis.set_xlabel(r"Mean phonon number $\bar n$")
        axis.set_ylabel(ylabel)
        axis.set_xticks(list(nbar_values))
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[1, 1].axhline(1.0, color="black", linestyle=":", linewidth=1.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=8,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.suptitle(
        "Fair calibrated-control comparison: full-Hamiltonian QPT", y=0.995
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "fair_calibrated_control_comparison.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    figure.savefig(
        output_dir / "fair_calibrated_control_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    return figure_path


def run_fair_control_comparison_stage(
    config: Mapping[str, Any],
    nbar_values: Iterable[float],
    *,
    run_qpt: bool = False,
    force_recompute: bool = False,
    show_progress: bool = True,
    gate_time_factors=(0.97, 1.03),
    pulse_closure_cycles: Mapping[str, float] | None = None,
    scattering_scales_with_intensity: bool = True,
) -> dict[str, Any]:
    """Calibrate each control family fairly and validate it by cached QPT.

    Calibration protocol:
      * rectangular A: local quadratic minimum of the completed physical
        infidelity scan;
      * drive+detuning: the rectangular optimum and detuning scan optimum are
        combined while preserving one closed phase-space loop;
      * gate time: every requested duration is paired with delta*T=2*pi and
        angle-preserving A, then the best physical-infidelity QPT is selected;
      * sin2/Blackman: analytic ideal-LD closure and geometric phase, followed
        by the measured temperature-dependent hXX amplitude renormalization.

    All final metrics come from full-Hamiltonian QPT; the inner calibration
    does not use uncalibrated RMS pulse normalization.
    """

    paths = _paths(config)
    params = _simulation_params(config)
    selected_nbars = sorted({float(value) for value in nbar_values})
    if not selected_nbars:
        raise ValueError("nbar_values must contain at least one value")
    screening_path = paths["control"] / "physical_control_qpt_summary.csv"
    if not screening_path.exists():
        raise FileNotFoundError(
            f"Missing {screening_path}. Run the screening cell first."
        )
    screening = pd.read_csv(screening_path)
    drive_path = paths["drive"] / "hxx_drive_calibration_final_summary.csv"
    drive = pd.read_csv(drive_path) if drive_path.exists() else pd.DataFrame()

    base_amplitude = float(np.asarray(params["A"]))
    base_delta = float(np.asarray(params["delta"]))
    base_gate_time_sim = float(
        params.get("t_gate_sim", 2.0 * np.pi / abs(base_delta))
    )
    base_gate_time_phys = float(params["t_gate_phys"])
    target_angle = float(config.get("TARGET_XX_ANGLE_RAD", np.pi / 4.0))
    closure_cycles = {"sin2": 2.0, "blackman": 3.0}
    if pulse_closure_cycles is not None:
        closure_cycles.update({
            str(key): float(value)
            for key, value in pulse_closure_cycles.items()
        })
    pulse_calibrations = {
        shape: _closed_pulse_geometric_calibration(
            shape,
            gate_time_sim=base_gate_time_sim,
            closure_cycles=closure_cycles[shape],
            target_xx_angle_rad=target_angle,
        )
        for shape in ("sin2", "blackman")
    }

    specs = []
    plan_rows = []
    for n_bar in selected_nbars:
        group = screening.loc[
            np.isclose(screening["n_bar"].astype(float), n_bar)
        ].copy()
        if group.empty:
            raise ValueError(f"Screening is missing n_bar={n_bar:g}")
        amplitude_scan = group.loc[
            group["kind"].isin(["baseline", "amplitude"])
        ]
        detuning_scan = group.loc[
            group["kind"].isin(["baseline", "detuning"])
        ]
        amplitude_opt = _quadratic_screening_optimum(amplitude_scan)
        detuning_opt = _quadratic_screening_optimum(detuning_scan)
        drive_match = (
            drive.loc[np.isclose(drive["n_bar"].astype(float), n_bar)]
            if not drive.empty else pd.DataFrame()
        )
        thermal_angle_factor = (
            float(drive_match.iloc[-1]["A_factor"])
            if not drive_match.empty else amplitude_opt["factor"]
        )

        def register(
            condition: str,
            family: str,
            *,
            amplitude_peak: float,
            detuning: float,
            t_gate_sim: float,
            t_gate_phys: float,
            amplitude_waveform: np.ndarray | None = None,
            shape: str = "rectangular",
            extra: Mapping[str, Any] | None = None,
        ) -> None:
            plan = {
                "n_bar": n_bar,
                "condition": condition,
                "family": family,
                "shape": shape,
                "amplitude_peak": float(amplitude_peak),
                "amplitude_factor": float(amplitude_peak / base_amplitude),
                "detuning": float(detuning),
                "detuning_factor": float(detuning / base_delta),
                "t_gate_sim": float(t_gate_sim),
                "t_gate_phys": float(t_gate_phys),
                "gate_time_factor": float(t_gate_phys / base_gate_time_phys),
                "scattering_scales_with_intensity": bool(
                    scattering_scales_with_intensity
                ),
                "rectangular_infidelity_seed_factor": amplitude_opt["factor"],
                "detuning_infidelity_seed_factor": detuning_opt["factor"],
                "thermal_hXX_angle_factor": thermal_angle_factor,
            }
            if extra:
                plan.update(dict(extra))
            overrides = {
                "A": (
                    np.asarray(amplitude_waveform, dtype=float)
                    if amplitude_waveform is not None else float(amplitude_peak)
                ),
                "delta": float(detuning),
                "t_gate_sim": float(t_gate_sim),
                "t_gate_phys": float(t_gate_phys),
                "laser_scattering_scales_with_intensity": bool(
                    scattering_scales_with_intensity
                ),
                "scattering_reference_amplitude": base_amplitude,
                "parallel_workers": int(config.get("FAST_PROCESS_WORKERS", 4)),
                "show_progress": bool(show_progress),
            }
            specs.append({"plan": plan, "overrides": overrides})
            plan_rows.append(plan)

        rectangular_amplitude = base_amplitude * amplitude_opt["factor"]
        register(
            "rectangular_A_infidelity",
            "rectangular",
            amplitude_peak=rectangular_amplitude,
            detuning=base_delta,
            t_gate_sim=base_gate_time_sim,
            t_gate_phys=base_gate_time_phys,
            extra={
                "calibration_method": "quadratic_physical_infidelity_screening",
                **{
                    f"amplitude_{key}": value
                    for key, value in amplitude_opt.items()
                },
            },
        )

        joint_delta_factor = detuning_opt["factor"]
        joint_delta = base_delta * joint_delta_factor
        joint_amplitude = (
            base_amplitude * amplitude_opt["factor"] * joint_delta_factor
        )
        register(
            "drive_detuning_joint",
            "drive_detuning",
            amplitude_peak=joint_amplitude,
            detuning=joint_delta,
            t_gate_sim=base_gate_time_sim / joint_delta_factor,
            t_gate_phys=base_gate_time_phys / joint_delta_factor,
            extra={
                "calibration_method": (
                    "quadratic_infidelity_seeds_with_closed_loop_rescaling"
                ),
                **{
                    f"detuning_{key}": value
                    for key, value in detuning_opt.items()
                },
            },
        )

        for gate_factor in gate_time_factors:
            gate_factor = float(gate_factor)
            register(
                f"gate_time_closed_{gate_factor:.3f}",
                "gate_time_closed",
                amplitude_peak=(
                    base_amplitude * amplitude_opt["factor"] / gate_factor
                ),
                detuning=base_delta / gate_factor,
                t_gate_sim=base_gate_time_sim * gate_factor,
                t_gate_phys=base_gate_time_phys * gate_factor,
                extra={
                    "calibration_method": "delta_T_2pi_with_angle_rescaling",
                    "closure_delta_times_T": 2.0 * np.pi,
                },
            )

        for shape, calibration in pulse_calibrations.items():
            peak = calibration["ideal_peak_amplitude"] * thermal_angle_factor
            envelope = _fair_pulse_envelope(shape, int(params["time_points"]))
            register(
                f"pulse_{shape}_calibrated",
                "pulse",
                amplitude_peak=peak,
                amplitude_waveform=peak * envelope,
                detuning=calibration["detuning"],
                t_gate_sim=base_gate_time_sim,
                t_gate_phys=base_gate_time_phys,
                shape=shape,
                extra={
                    "calibration_method": (
                        "ideal_LD_closed_geometric_phase_plus_thermal_hXX"
                    ),
                    **{
                        f"pulse_{key}": value
                        for key, value in calibration.items()
                        if key != "shape"
                    },
                },
            )

    output_dir = paths["control"] / "fair_calibrated_comparison"
    cache_dir = output_dir / "qpt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    calibration_plan = pd.DataFrame(plan_rows).sort_values(
        ["n_bar", "condition"]
    ).reset_index(drop=True)
    calibration_plan_path = output_dir / "fair_control_calibration_plan.csv"
    calibration_plan.to_csv(calibration_plan_path, index=False)

    pending = []
    newly_computed = 0
    rows = []
    iterator = tqdm(
        specs,
        desc="Fair calibrated-control QPT",
        unit="point",
        disable=not show_progress,
    )
    for spec in iterator:
        plan = spec["plan"]
        cache_path = _fair_control_cache_path(
            cache_dir, config, plan
        )
        if force_recompute or not cache_path.exists():
            if not run_qpt:
                pending.append({
                    "n_bar": plan["n_bar"],
                    "condition": plan["condition"],
                    "cache_path": str(cache_path),
                })
                continue
            result = qpt_analysis.calculate_error_channel_batch(
                [plan["n_bar"]],
                params,
                spec["overrides"],
                convention=_convention(config),
            )[0]
            result["metadata"].update({
                key: value
                for key, value in plan.items()
                if isinstance(value, (str, int, float, bool, np.generic))
            })
            qpt_analysis.save_qpt_point(
                cache_path,
                plan["n_bar"],
                plan["condition"],
                result["chi"],
                result["metadata"],
            )
            newly_computed += 1
        with np.load(cache_path, allow_pickle=False) as data:
            chi = np.asarray(data["chi_trace_normalized"], dtype=complex)
        rows.append({
            **plan,
            "cache_path": str(cache_path),
            **_control_observables(chi, config),
        })

    qpt_summary = pd.DataFrame(rows)
    qpt_summary_path = output_dir / "fair_control_qpt_summary.csv"
    qpt_summary.to_csv(qpt_summary_path, index=False)

    baseline_rows = screening.loc[
        screening["candidate"] == "baseline"
    ].copy()
    baseline_rows = pd.concat([
        baseline_rows.loc[
            np.isclose(baseline_rows["n_bar"].astype(float), n_bar)
        ].tail(1)
        for n_bar in selected_nbars
    ], ignore_index=True)
    baseline_rows["condition"] = "baseline"
    baseline_rows["family"] = "baseline"

    selected_rows = []
    if not qpt_summary.empty:
        for n_bar, group in qpt_summary.groupby("n_bar", sort=True):
            for condition in [
                "rectangular_A_infidelity",
                "drive_detuning_joint",
                "pulse_sin2_calibrated",
                "pulse_blackman_calibrated",
            ]:
                match = group.loc[group["condition"] == condition]
                if not match.empty:
                    selected_rows.append(match.iloc[-1].to_dict())
            gate_rows = group.loc[group["family"] == "gate_time_closed"]
            if not gate_rows.empty:
                best_gate = gate_rows.loc[
                    gate_rows["average_infidelity"].idxmin()
                ].copy()
                best_gate["source_condition"] = best_gate["condition"]
                best_gate["condition"] = "gate_time_closed_opt"
                selected_rows.append(best_gate.to_dict())
    selected = pd.concat(
        [baseline_rows, pd.DataFrame(selected_rows)],
        ignore_index=True,
        sort=False,
    )
    selected_path = output_dir / "fair_control_selected_comparison.csv"
    selected.to_csv(selected_path, index=False)
    figure_path = _plot_fair_control_comparison(
        selected, output_dir, selected_nbars
    )

    return {
        "calibration_plan": calibration_plan,
        "qpt_summary": qpt_summary,
        "selected_comparison": selected,
        "figure_path": figure_path,
        "calibration_plan_path": calibration_plan_path,
        "qpt_summary_path": qpt_summary_path,
        "selected_path": selected_path,
        "status": {
            "nbar_values": selected_nbars,
            "calibrated_qpt_points_expected": len(specs),
            "calibrated_qpt_points_completed": len(qpt_summary),
            "newly_computed": newly_computed,
            "pending": pending,
            "run_qpt": bool(run_qpt),
            "scattering_scales_with_intensity": bool(
                scattering_scales_with_intensity
            ),
        },
    }


def _load_qpt_cache_with_metadata(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        chi = np.asarray(data["chi_trace_normalized"], dtype=complex)
        metadata = {}
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"].item()))
    return chi, metadata


def _publication_control_resource_metrics(
    *,
    shape: str,
    amplitude_peak: float,
    base_amplitude: float,
    gate_time_factor: float,
    integration_points: int = 4001,
) -> dict[str, float]:
    if shape == "rectangular":
        envelope = np.ones(int(integration_points), dtype=float)
    else:
        envelope = _fair_pulse_envelope(shape, int(integration_points))
    normalized = amplitude_peak * envelope / base_amplitude
    return {
        "peak_amplitude_factor": float(amplitude_peak / base_amplitude),
        "peak_power_factor": float((amplitude_peak / base_amplitude) ** 2),
        "pulse_energy_factor": float(
            np.mean(normalized**2) * float(gate_time_factor)
        ),
    }


def _publication_resource_eligible(
    metrics: Mapping[str, float],
    *,
    resource_mode: str,
    max_peak_power_factor: float,
    max_pulse_energy_factor: float,
) -> bool:
    mode = str(resource_mode).lower()
    if mode not in {"peak_power", "pulse_energy", "both"}:
        raise ValueError(
            "resource_mode must be 'peak_power', 'pulse_energy', or 'both'"
        )
    peak_ok = metrics["peak_power_factor"] <= max_peak_power_factor * (
        1.0 + 1e-12
    )
    energy_ok = metrics["pulse_energy_factor"] <= max_pulse_energy_factor * (
        1.0 + 1e-12
    )
    return bool(
        (mode == "peak_power" and peak_ok)
        or (mode == "pulse_energy" and energy_ok)
        or (mode == "both" and peak_ok and energy_ok)
    )


def _publication_control_cache_path(
    directory: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> Path:
    params = _simulation_params(config)
    parameter_keys = [
        "heating_rate_phys", "dephasing_rate_phys", "T2_star",
        "rayleigh_rate_phys", "raman_rate_phys", "eta", "use_full_order",
        "laser_intensity_fluctuation", "laser_detuning_fluctuation",
        "laser_rotation_angle_fluctuation", "laser_noise_samples",
    ]
    payload = {
        key: plan.get(key)
        for key in [
            "validation_stage", "n_bar", "condition", "shape",
            "amplitude_peak", "detuning", "t_gate_sim", "t_gate_phys",
            "time_points", "solver_max_step", "phonon_dim_override",
            "scattering_scales_with_intensity",
        ]
    }
    payload["parameters"] = {
        key: params.get(key) for key in parameter_keys
    }
    payload["convention"] = _convention(config)
    payload["validation_version"] = "publication_control_v1"
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=_json_safe).encode("utf-8")
    ).hexdigest()[:14]
    return directory / (
        f"{_condition_stem(plan['validation_stage'])}__"
        f"{_condition_stem(plan['condition'])}__nbar_"
        f"{_compact_nbar_stem(plan['n_bar'])}__{digest}.npz"
    )


def _publication_control_spec(
    *,
    plan: Mapping[str, Any],
    params: Mapping[str, Any],
    base_amplitude: float,
    show_progress: bool,
    reuse_cache_path: Path | None = None,
) -> dict[str, Any]:
    resolved = dict(plan)
    time_points = int(resolved.get("time_points", params["time_points"]))
    shape = str(resolved.get("shape", "rectangular"))
    amplitude_peak = float(resolved["amplitude_peak"])
    if shape == "rectangular":
        amplitude = amplitude_peak
    else:
        amplitude = amplitude_peak * _fair_pulse_envelope(shape, time_points)
    overrides = {
        "A": amplitude,
        "delta": float(resolved["detuning"]),
        "t_gate_sim": float(resolved["t_gate_sim"]),
        "t_gate_phys": float(resolved["t_gate_phys"]),
        "time_points": time_points,
        "laser_scattering_scales_with_intensity": bool(
            resolved.get("scattering_scales_with_intensity", True)
        ),
        "scattering_reference_amplitude": float(base_amplitude),
        "parallel_workers": int(resolved["parallel_workers"]),
        "show_progress": bool(show_progress),
    }
    solver_max_step = resolved.get("solver_max_step")
    if solver_max_step is not None and np.isfinite(float(solver_max_step)):
        overrides["solver_max_step"] = float(solver_max_step)
    phonon_dim_override = resolved.get("phonon_dim_override")
    if phonon_dim_override is not None and np.isfinite(
        float(phonon_dim_override)
    ):
        overrides["phonon_dim_override"] = int(phonon_dim_override)
    return {
        "plan": resolved,
        "overrides": overrides,
        "reuse_cache_path": reuse_cache_path,
    }


def _run_publication_control_specs(
    specs: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    cache_dir: Path,
    run_qpt: bool,
    force_recompute: bool,
    show_progress: bool,
    description: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    params = _simulation_params(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pending = []
    newly_computed = 0
    iterator = tqdm(
        specs,
        desc=description,
        unit="point",
        disable=not show_progress,
    )
    for spec in iterator:
        plan = spec["plan"]
        reuse_path = spec.get("reuse_cache_path")
        if reuse_path is not None:
            reuse_path = Path(reuse_path)
        cache_path = _publication_control_cache_path(cache_dir, config, plan)
        load_path = (
            reuse_path
            if reuse_path is not None and reuse_path.exists() and not force_recompute
            else cache_path
        )
        if force_recompute or not load_path.exists():
            load_path = cache_path
            if not run_qpt:
                pending.append({
                    "validation_stage": plan["validation_stage"],
                    "n_bar": float(plan["n_bar"]),
                    "condition": plan["condition"],
                    "cache_path": str(cache_path),
                })
                continue
            result = qpt_analysis.calculate_error_channel_batch(
                [float(plan["n_bar"])],
                params,
                spec["overrides"],
                convention=_convention(config),
            )[0]
            result["metadata"].update({
                key: value
                for key, value in plan.items()
                if isinstance(value, (str, int, float, bool, np.generic))
            })
            qpt_analysis.save_qpt_point(
                cache_path,
                float(plan["n_bar"]),
                str(plan["condition"]),
                result["chi"],
                result["metadata"],
            )
            newly_computed += 1
        chi, metadata = _load_qpt_cache_with_metadata(load_path)
        rows.append({
            **plan,
            "cache_path": str(load_path),
            "phonon_dim_actual": metadata.get("phonon_dim", np.nan),
            "raw_cp_pass": metadata.get("cp_pass", np.nan),
            "raw_tp_pass": metadata.get("tp_pass", np.nan),
            "raw_min_choi_eigenvalue": metadata.get(
                "min_choi_eigenvalue", np.nan
            ),
            "raw_tp_frobenius_error": metadata.get(
                "tp_frobenius_error", np.nan
            ),
            **_control_observables(chi, config),
        })
    return pd.DataFrame(rows), pending, newly_computed


def _interpolate_control_seed(
    frame: pd.DataFrame,
    n_bar: float,
    *,
    column: str = "amplitude_peak",
) -> float:
    points = frame[["n_bar", column]].dropna().sort_values("n_bar")
    if points.empty:
        raise ValueError(f"No finite {column} values are available for interpolation")
    return float(np.interp(
        float(n_bar),
        points["n_bar"].to_numpy(float),
        points[column].to_numpy(float),
    ))


def _estimate_infidelity_crossover(curve: pd.DataFrame) -> float | None:
    points = curve[["n_bar", "shaped_to_rectangular_infidelity_ratio"]].dropna()
    points = points.sort_values("n_bar")
    x = points["n_bar"].to_numpy(float)
    y = points["shaped_to_rectangular_infidelity_ratio"].to_numpy(float) - 1.0
    for index in range(len(points) - 1):
        if y[index] == 0.0:
            return float(x[index])
        if y[index] * y[index + 1] < 0.0:
            fraction = -y[index] / (y[index + 1] - y[index])
            return float(x[index] + fraction * (x[index + 1] - x[index]))
    if len(points) and y[-1] == 0.0:
        return float(x[-1])
    return None


def _plot_publication_control_validation(
    fairness_selected: pd.DataFrame,
    convergence: pd.DataFrame,
    crossover_curve: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if fairness_selected.empty and convergence.empty and crossover_curve.empty:
        return None
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5))
    if not fairness_selected.empty:
        for condition, group in fairness_selected.groupby("condition"):
            axes[0, 0].semilogy(
                group["n_bar"], group["average_infidelity"], marker="o",
                label=str(condition).replace("_", " "),
            )
            axes[0, 1].plot(
                group["n_bar"], group["peak_power_factor"], marker="o",
                label=str(condition).replace("_", " "),
            )
        axes[0, 0].set_ylabel(r"$1-F_{\rm avg}$")
        axes[0, 0].set_title("Pulse-specific local amplitude calibration")
        axes[0, 1].set_ylabel(r"Peak power / baseline $A_0^2$")
        axes[0, 1].set_title("Common resource accounting")
        for axis in axes[0]:
            axis.set_xlabel(r"Mean phonon number $\bar n$")
            axis.grid(True, which="both", alpha=0.25)
            axis.legend(fontsize=8)
    else:
        axes[0, 0].text(0.5, 0.5, "Fairness QPT pending", ha="center")
        axes[0, 1].text(0.5, 0.5, "Resource scan pending", ha="center")

    if not convergence.empty and "relative_infidelity_to_high_resolution" in convergence:
        for condition, group in convergence.groupby("condition"):
            group = group.sort_values("convergence_level_order")
            axes[1, 0].plot(
                group["convergence_level_order"],
                group["relative_infidelity_to_high_resolution"],
                marker="o", label=str(condition).replace("_", " "),
            )
        axes[1, 0].axhline(1.0, color="black", linestyle=":")
        axes[1, 0].set_xticks([0, 1, 2, 3])
        axes[1, 0].set_xticklabels(
            ["base", "cutoff", "time", "both"], rotation=15
        )
        axes[1, 0].set_ylabel("Infidelity / high-resolution")
        axes[1, 0].set_title(r"$\bar n=20$ numerical convergence")
        axes[1, 0].grid(True, alpha=0.25)
        axes[1, 0].legend(fontsize=7, ncol=2)
    else:
        axes[1, 0].text(0.5, 0.5, "Convergence QPT pending", ha="center")

    if not crossover_curve.empty:
        axes[1, 1].plot(
            crossover_curve["n_bar"],
            crossover_curve["shaped_to_rectangular_infidelity_ratio"],
            marker="o", color="#D55E00",
        )
        axes[1, 1].axhline(1.0, color="black", linestyle=":")
        axes[1, 1].set_xlabel(r"Mean phonon number $\bar n$")
        axes[1, 1].set_ylabel("Best shaped / rectangular infidelity")
        axes[1, 1].set_title("Coherent-to-stochastic crossover")
        axes[1, 1].grid(True, alpha=0.25)
    else:
        axes[1, 1].text(0.5, 0.5, "Crossover QPT pending", ha="center")
    figure.suptitle("Publication control validation", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "publication_control_validation.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    figure.savefig(
        output_dir / "publication_control_validation.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    return figure_path


def run_publication_control_validation_stage(
    config: Mapping[str, Any],
    *,
    fairness_nbars=(0.01, 2.0, 4.0, 10.0, 20.0),
    amplitude_scale_factors=(0.98, 1.0, 1.02),
    resource_mode: str = "peak_power",
    max_peak_power_factor: float = 16.0,
    max_pulse_energy_factor: float = 5.0,
    convergence_nbar: float = 20.0,
    phonon_cutoff_factor: float = 1.15,
    time_points_factor: float = 2.0,
    crossover_nbars=(12.0, 16.0),
    crossover_shape: str = "auto",
    run_fairness_qpt: bool = False,
    run_convergence_qpt: bool = False,
    run_crossover_qpt: bool = False,
    force_recompute: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run the three final publication checks after the 9.2 screening.

    The stage reuses the 9.2 center points, caches every new QPT separately,
    and keeps the three expensive sub-stages independently switchable.
    """

    paths = _paths(config)
    params = _simulation_params(config)
    base_amplitude = float(np.asarray(params["A"]))
    base_delta = float(np.asarray(params["delta"]))
    base_gate_time_sim = float(
        params.get("t_gate_sim", 2.0 * np.pi / abs(base_delta))
    )
    base_gate_time_phys = float(params["t_gate_phys"])
    workers = int(config.get("FAST_PROCESS_WORKERS", 4))
    output_dir = paths["control"] / "publication_control_validation"
    cache_dir = output_dir / "qpt_cache"
    fair_path = (
        paths["control"] / "fair_calibrated_comparison"
        / "fair_control_qpt_summary.csv"
    )
    if not fair_path.exists():
        raise FileNotFoundError(f"Missing {fair_path}. Run cell 9.2 first.")
    fair_qpt = pd.read_csv(fair_path)

    fairness_nbars = sorted({float(value) for value in fairness_nbars})
    amplitude_scale_factors = sorted({
        float(value) for value in amplitude_scale_factors
    })
    if 1.0 not in amplitude_scale_factors:
        raise ValueError("amplitude_scale_factors must include 1.0")
    if min(amplitude_scale_factors) <= 0.0:
        raise ValueError("amplitude_scale_factors must be positive")
    if phonon_cutoff_factor <= 1.0 or time_points_factor <= 1.0:
        raise ValueError("convergence factors must be greater than 1")

    fairness_specs = []
    fairness_plan_rows = []
    for n_bar in fairness_nbars:
        for shape in ("sin2", "blackman"):
            seed_match = fair_qpt.loc[
                np.isclose(fair_qpt["n_bar"].astype(float), n_bar)
                & (fair_qpt["condition"] == f"pulse_{shape}_calibrated")
            ]
            if seed_match.empty:
                raise ValueError(
                    f"9.2 is missing pulse_{shape}_calibrated at n_bar={n_bar:g}"
                )
            seed = seed_match.iloc[-1]
            for scale in amplitude_scale_factors:
                amplitude_peak = float(seed["amplitude_peak"]) * scale
                resources = _publication_control_resource_metrics(
                    shape=shape,
                    amplitude_peak=amplitude_peak,
                    base_amplitude=base_amplitude,
                    gate_time_factor=float(seed["gate_time_factor"]),
                )
                eligible = _publication_resource_eligible(
                    resources,
                    resource_mode=resource_mode,
                    max_peak_power_factor=float(max_peak_power_factor),
                    max_pulse_energy_factor=float(max_pulse_energy_factor),
                )
                plan = {
                    "validation_stage": "pulse_fairness",
                    "n_bar": n_bar,
                    "condition": f"pulse_{shape}_local_A_{scale:.3f}",
                    "family": "pulse",
                    "shape": shape,
                    "amplitude_scale_from_9p2": scale,
                    "amplitude_peak": amplitude_peak,
                    "detuning": float(seed["detuning"]),
                    "t_gate_sim": float(seed["t_gate_sim"]),
                    "t_gate_phys": float(seed["t_gate_phys"]),
                    "gate_time_factor": float(seed["gate_time_factor"]),
                    "time_points": int(params["time_points"]),
                    "solver_max_step": np.nan,
                    "phonon_dim_override": np.nan,
                    "parallel_workers": workers,
                    "scattering_scales_with_intensity": True,
                    "resource_mode": resource_mode,
                    "resource_eligible": eligible,
                    "max_peak_power_factor": float(max_peak_power_factor),
                    "max_pulse_energy_factor": float(max_pulse_energy_factor),
                    "ideal_closure_residual": float(
                        seed.get("pulse_relative_closure_residual", np.nan)
                    ),
                    **resources,
                }
                fairness_plan_rows.append(plan)
                if not eligible:
                    continue
                reuse_path = None
                if np.isclose(scale, 1.0):
                    reuse_path = Path(str(seed["cache_path"]))
                fairness_specs.append(_publication_control_spec(
                    plan=plan,
                    params=params,
                    base_amplitude=base_amplitude,
                    show_progress=show_progress,
                    reuse_cache_path=reuse_path,
                ))

    fairness_scan, fairness_pending, fairness_new = (
        _run_publication_control_specs(
            fairness_specs,
            config=config,
            cache_dir=cache_dir / "pulse_fairness",
            run_qpt=run_fairness_qpt,
            force_recompute=force_recompute,
            show_progress=show_progress,
            description="9.3 pulse fairness",
        )
    )
    fairness_plan = pd.DataFrame(fairness_plan_rows)
    fairness_selected_rows = []
    if not fairness_scan.empty:
        for (n_bar, shape), group in fairness_scan.groupby(
            ["n_bar", "shape"], sort=True
        ):
            best = group.loc[group["average_infidelity"].idxmin()].copy()
            best["condition"] = f"pulse_{shape}_A_optimized"
            best["local_optimum_at_boundary"] = bool(
                np.isclose(
                    best["amplitude_scale_from_9p2"],
                    min(amplitude_scale_factors),
                )
                or np.isclose(
                    best["amplitude_scale_from_9p2"],
                    max(amplitude_scale_factors),
                )
            )
            fairness_selected_rows.append(best.to_dict())
    fairness_selected = pd.DataFrame(fairness_selected_rows)
    rectangular_reference = fair_qpt.loc[
        fair_qpt["condition"] == "rectangular_A_infidelity"
    ].copy()
    rectangular_reference = rectangular_reference.loc[
        rectangular_reference["n_bar"].astype(float).isin(fairness_nbars)
    ]
    if not rectangular_reference.empty:
        rectangular_reference["condition"] = "rectangular_A_optimized"
        rectangular_reference["shape"] = "rectangular"
        resource_rows = [
            _publication_control_resource_metrics(
                shape="rectangular",
                amplitude_peak=float(row["amplitude_peak"]),
                base_amplitude=base_amplitude,
                gate_time_factor=float(row["gate_time_factor"]),
            )
            for _, row in rectangular_reference.iterrows()
        ]
        for key in [
            "peak_amplitude_factor", "peak_power_factor", "pulse_energy_factor"
        ]:
            rectangular_reference[key] = [row[key] for row in resource_rows]
        fairness_selected = pd.concat(
            [rectangular_reference, fairness_selected],
            ignore_index=True,
            sort=False,
        )

    # High-temperature convergence: validate all six 9.2 control points, not
    # only the four that fail the raw CP threshold at n_bar=20.
    convergence_nbar = float(convergence_nbar)
    high_rows = fair_qpt.loc[
        np.isclose(fair_qpt["n_bar"].astype(float), convergence_nbar)
    ].copy()
    if len(high_rows) != 6:
        raise ValueError(
            f"Expected six 9.2 controls at n_bar={convergence_nbar:g}, "
            f"found {len(high_rows)}"
        )
    convergence_specs = []
    high_time_points = int(math.ceil(
        int(params["time_points"]) * float(time_points_factor)
    ))
    level_definitions = [
        ("base", 0, False, False),
        ("cutoff_high", 1, True, False),
        ("time_high", 2, False, True),
        ("both_high", 3, True, True),
    ]
    for _, seed in high_rows.iterrows():
        _, seed_metadata = _load_qpt_cache_with_metadata(
            Path(str(seed["cache_path"]))
        )
        base_phonon_dim = int(seed_metadata["phonon_dim"])
        high_phonon_dim = int(math.ceil(
            base_phonon_dim * float(phonon_cutoff_factor)
        ))
        for level, order, use_high_cutoff, use_high_time in level_definitions:
            time_points = (
                high_time_points if use_high_time else int(params["time_points"])
            )
            solver_max_step = (
                float(seed["t_gate_sim"]) / max(time_points - 1, 1)
                if use_high_time else np.nan
            )
            phonon_override = high_phonon_dim if use_high_cutoff else np.nan
            resources = _publication_control_resource_metrics(
                shape=str(seed["shape"]),
                amplitude_peak=float(seed["amplitude_peak"]),
                base_amplitude=base_amplitude,
                gate_time_factor=float(seed["gate_time_factor"]),
            )
            plan = {
                "validation_stage": "nbar20_convergence",
                "n_bar": convergence_nbar,
                "condition": str(seed["condition"]),
                "family": str(seed["family"]),
                "shape": str(seed["shape"]),
                "amplitude_peak": float(seed["amplitude_peak"]),
                "detuning": float(seed["detuning"]),
                "t_gate_sim": float(seed["t_gate_sim"]),
                "t_gate_phys": float(seed["t_gate_phys"]),
                "gate_time_factor": float(seed["gate_time_factor"]),
                "time_points": time_points,
                "solver_max_step": solver_max_step,
                "phonon_dim_override": phonon_override,
                "parallel_workers": workers,
                "scattering_scales_with_intensity": True,
                "convergence_level": level,
                "convergence_level_order": order,
                "base_phonon_dim": base_phonon_dim,
                "high_phonon_dim": high_phonon_dim,
                "phonon_cutoff_factor": float(phonon_cutoff_factor),
                "time_points_factor": float(time_points_factor),
                **resources,
            }
            reuse_path = (
                Path(str(seed["cache_path"])) if level == "base" else None
            )
            convergence_specs.append(_publication_control_spec(
                plan=plan,
                params=params,
                base_amplitude=base_amplitude,
                show_progress=show_progress,
                reuse_cache_path=reuse_path,
            ))
    convergence_scan, convergence_pending, convergence_new = (
        _run_publication_control_specs(
            convergence_specs,
            config=config,
            cache_dir=cache_dir / "nbar20_convergence",
            run_qpt=run_convergence_qpt,
            force_recompute=force_recompute,
            show_progress=show_progress,
            description="9.3 nbar=20 convergence",
        )
    )
    convergence_comparison = convergence_scan.copy()
    if not convergence_comparison.empty:
        reference = convergence_comparison.loc[
            convergence_comparison["convergence_level"] == "both_high",
            [
                "condition", "average_infidelity", "h_XX_rad_per_gate",
                "gamma_XX_per_gate", "raw_min_choi_eigenvalue",
            ],
        ].rename(columns={
            "average_infidelity": "reference_average_infidelity",
            "h_XX_rad_per_gate": "reference_h_XX_rad_per_gate",
            "gamma_XX_per_gate": "reference_gamma_XX_per_gate",
            "raw_min_choi_eigenvalue": "reference_raw_min_choi_eigenvalue",
        })
        convergence_comparison = convergence_comparison.merge(
            reference, on="condition", how="left"
        )
        convergence_comparison["relative_infidelity_to_high_resolution"] = (
            convergence_comparison["average_infidelity"]
            / convergence_comparison["reference_average_infidelity"]
        )
        convergence_comparison["relative_abs_h_to_high_resolution"] = (
            np.abs(convergence_comparison["h_XX_rad_per_gate"])
            / np.maximum(
                np.abs(convergence_comparison["reference_h_XX_rad_per_gate"]),
                1e-15,
            )
        )
        convergence_comparison["relative_gamma_to_high_resolution"] = (
            convergence_comparison["gamma_XX_per_gate"]
            / np.maximum(
                convergence_comparison["reference_gamma_XX_per_gate"], 1e-15
            )
        )

    # Crossover refinement at n_bar=12,16.  The shape is selected from the
    # high-temperature 9.2 data unless explicitly fixed in the notebook.
    if str(crossover_shape).lower() == "auto":
        shape_candidates = fair_qpt.loc[
            fair_qpt["shape"].isin(["sin2", "blackman"])
            & (fair_qpt["n_bar"].astype(float) >= 10.0)
        ]
        best_shape = str(
            shape_candidates.groupby("shape")["average_infidelity"].mean().idxmin()
        )
    else:
        best_shape = str(crossover_shape).lower()
    if best_shape not in {"sin2", "blackman"}:
        raise ValueError("crossover_shape must be 'auto', 'sin2', or 'blackman'")

    shaped_seed_frame = fairness_selected.loc[
        fairness_selected.get("shape", pd.Series(dtype=str)) == best_shape
    ].copy()
    if shaped_seed_frame.empty:
        shaped_seed_frame = fair_qpt.loc[
            fair_qpt["shape"] == best_shape
        ].copy()
    rectangular_seed_frame = fair_qpt.loc[
        fair_qpt["condition"] == "rectangular_A_infidelity"
    ].copy()
    crossover_specs = []
    for n_bar in sorted({float(value) for value in crossover_nbars}):
        seeds = [
            (
                "rectangular",
                "rectangular",
                _interpolate_control_seed(rectangular_seed_frame, n_bar),
                base_delta,
                base_gate_time_sim,
                base_gate_time_phys,
            ),
            (
                f"pulse_{best_shape}",
                best_shape,
                _interpolate_control_seed(shaped_seed_frame, n_bar),
                _interpolate_control_seed(
                    shaped_seed_frame, n_bar, column="detuning"
                ),
                _interpolate_control_seed(
                    shaped_seed_frame, n_bar, column="t_gate_sim"
                ),
                _interpolate_control_seed(
                    shaped_seed_frame, n_bar, column="t_gate_phys"
                ),
            ),
        ]
        for family, shape, seed_amplitude, detuning, t_sim, t_phys in seeds:
            for scale in amplitude_scale_factors:
                amplitude_peak = seed_amplitude * scale
                gate_factor = t_phys / base_gate_time_phys
                resources = _publication_control_resource_metrics(
                    shape=shape,
                    amplitude_peak=amplitude_peak,
                    base_amplitude=base_amplitude,
                    gate_time_factor=gate_factor,
                )
                eligible = _publication_resource_eligible(
                    resources,
                    resource_mode=resource_mode,
                    max_peak_power_factor=float(max_peak_power_factor),
                    max_pulse_energy_factor=float(max_pulse_energy_factor),
                )
                if not eligible:
                    continue
                plan = {
                    "validation_stage": "crossover_refinement",
                    "n_bar": n_bar,
                    "condition": f"{family}_local_A_{scale:.3f}",
                    "family": family,
                    "shape": shape,
                    "amplitude_scale_from_seed": scale,
                    "amplitude_peak": amplitude_peak,
                    "detuning": detuning,
                    "t_gate_sim": t_sim,
                    "t_gate_phys": t_phys,
                    "gate_time_factor": gate_factor,
                    "time_points": int(params["time_points"]),
                    "solver_max_step": np.nan,
                    "phonon_dim_override": np.nan,
                    "parallel_workers": workers,
                    "scattering_scales_with_intensity": True,
                    "resource_mode": resource_mode,
                    "resource_eligible": True,
                    **resources,
                }
                crossover_specs.append(_publication_control_spec(
                    plan=plan,
                    params=params,
                    base_amplitude=base_amplitude,
                    show_progress=show_progress,
                ))
    crossover_scan, crossover_pending, crossover_new = (
        _run_publication_control_specs(
            crossover_specs,
            config=config,
            cache_dir=cache_dir / "crossover_refinement",
            run_qpt=run_crossover_qpt,
            force_recompute=force_recompute,
            show_progress=show_progress,
            description="9.3 crossover nbar=12,16",
        )
    )
    crossover_selected_rows = []
    if not crossover_scan.empty:
        for (n_bar, family), group in crossover_scan.groupby(
            ["n_bar", "family"], sort=True
        ):
            best = group.loc[group["average_infidelity"].idxmin()].copy()
            best["local_optimum_at_boundary"] = bool(
                np.isclose(
                    best["amplitude_scale_from_seed"],
                    min(amplitude_scale_factors),
                )
                or np.isclose(
                    best["amplitude_scale_from_seed"],
                    max(amplitude_scale_factors),
                )
            )
            crossover_selected_rows.append(best.to_dict())
    crossover_selected = pd.DataFrame(crossover_selected_rows)

    trend_rows = []
    for n_bar in [10.0, 20.0]:
        rect = rectangular_seed_frame.loc[
            np.isclose(rectangular_seed_frame["n_bar"].astype(float), n_bar)
        ]
        shaped = fairness_selected.loc[
            np.isclose(fairness_selected["n_bar"].astype(float), n_bar)
            & (fairness_selected.get("shape", pd.Series(dtype=str)) == best_shape)
        ]
        if shaped.empty:
            shaped = fair_qpt.loc[
                np.isclose(fair_qpt["n_bar"].astype(float), n_bar)
                & (fair_qpt["shape"] == best_shape)
            ]
        if not rect.empty and not shaped.empty:
            trend_rows.append({
                "n_bar": n_bar,
                "rectangular_infidelity": float(rect.iloc[-1]["average_infidelity"]),
                "shaped_infidelity": float(shaped.iloc[-1]["average_infidelity"]),
                "shaped_to_rectangular_infidelity_ratio": float(
                    shaped.iloc[-1]["average_infidelity"]
                    / rect.iloc[-1]["average_infidelity"]
                ),
                "shaped_abs_h_XX": abs(float(
                    shaped.iloc[-1]["h_XX_rad_per_gate"]
                )),
                "shaped_gamma_XX": float(shaped.iloc[-1]["gamma_XX_per_gate"]),
                "source": "existing_fairness",
            })
    if not crossover_selected.empty:
        for n_bar, group in crossover_selected.groupby("n_bar", sort=True):
            rect = group.loc[group["family"] == "rectangular"]
            shaped = group.loc[group["family"] == f"pulse_{best_shape}"]
            if not rect.empty and not shaped.empty:
                trend_rows.append({
                    "n_bar": float(n_bar),
                    "rectangular_infidelity": float(
                        rect.iloc[-1]["average_infidelity"]
                    ),
                    "shaped_infidelity": float(
                        shaped.iloc[-1]["average_infidelity"]
                    ),
                    "shaped_to_rectangular_infidelity_ratio": float(
                        shaped.iloc[-1]["average_infidelity"]
                        / rect.iloc[-1]["average_infidelity"]
                    ),
                    "shaped_abs_h_XX": abs(float(
                        shaped.iloc[-1]["h_XX_rad_per_gate"]
                    )),
                    "shaped_gamma_XX": float(
                        shaped.iloc[-1]["gamma_XX_per_gate"]
                    ),
                    "source": "crossover_refinement",
                })
    crossover_curve = pd.DataFrame(trend_rows)
    if not crossover_curve.empty:
        crossover_curve = crossover_curve.sort_values("n_bar").drop_duplicates(
            "n_bar", keep="last"
        )
    crossover_estimate = _estimate_infidelity_crossover(crossover_curve)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "fairness_plan": fairness_plan,
        "fairness_scan": fairness_scan,
        "fairness_selected": fairness_selected,
        "convergence_scan": convergence_scan,
        "convergence_comparison": convergence_comparison,
        "crossover_scan": crossover_scan,
        "crossover_selected": crossover_selected,
        "crossover_curve": crossover_curve,
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    figure_path = _plot_publication_control_validation(
        fairness_selected,
        convergence_comparison,
        crossover_curve,
        output_dir,
    )
    status = {
        "fairness": {
            "expected_eligible_points": len(fairness_specs),
            "completed_points": len(fairness_scan),
            "newly_computed": fairness_new,
            "pending_points": len(fairness_pending),
            "run_qpt": bool(run_fairness_qpt),
            "boundary_optima": int(
                sum(
                    value is True or isinstance(value, np.bool_) and bool(value)
                    for value in fairness_selected.get(
                        "local_optimum_at_boundary", pd.Series(dtype=object)
                    )
                )
            ),
        },
        "convergence": {
            "expected_points": len(convergence_specs),
            "completed_points": len(convergence_scan),
            "newly_computed": convergence_new,
            "pending_points": len(convergence_pending),
            "run_qpt": bool(run_convergence_qpt),
            "phonon_cutoff_factor": float(phonon_cutoff_factor),
            "time_points_factor": float(time_points_factor),
        },
        "crossover": {
            "shape": best_shape,
            "expected_points": len(crossover_specs),
            "completed_points": len(crossover_scan),
            "newly_computed": crossover_new,
            "pending_points": len(crossover_pending),
            "run_qpt": bool(run_crossover_qpt),
            "estimated_n_bar": crossover_estimate,
        },
        "resource_constraint": {
            "mode": resource_mode,
            "max_peak_power_factor": float(max_peak_power_factor),
            "max_pulse_energy_factor": float(max_pulse_energy_factor),
        },
    }
    return {
        **tables,
        "figure_path": figure_path,
        "output_dir": output_dir,
        "status": status,
        "pending": {
            "fairness": fairness_pending,
            "convergence": convergence_pending,
            "crossover": crossover_pending,
        },
    }


def _robustness_conditions(
    params: Mapping[str, Any],
    eta_factors: Iterable[float],
    a_over_delta_factors: Iterable[float],
    gate_time_factors: Iterable[float],
    motional_dephasing_factors: Iterable[float],
) -> list[dict[str, Any]]:
    base_gate_time_sim = 2.0 * np.pi / abs(float(params["delta"]))
    conditions = []
    for factor in eta_factors:
        factor = float(factor)
        conditions.append({
            "name": f"eta_{factor:.2f}",
            "parameter": "eta",
            "factor": factor,
            "overrides": {"eta": float(params["eta"]) * factor},
        })
    for factor in a_over_delta_factors:
        factor = float(factor)
        conditions.append({
            "name": f"A_over_delta_{factor:.2f}",
            "parameter": "A_over_delta",
            "factor": factor,
            "overrides": {"A": float(params["A"]) * factor},
        })
    for factor in gate_time_factors:
        factor = float(factor)
        conditions.append({
            "name": f"gate_time_{factor:.2f}",
            "parameter": "gate_time",
            "factor": factor,
            "overrides": {
                "t_gate_sim": base_gate_time_sim * factor,
                "t_gate_phys": float(params["t_gate_phys"]) * factor,
            },
        })
    for factor in motional_dephasing_factors:
        factor = float(factor)
        conditions.append({
            "name": f"motional_dephasing_{factor:.2f}",
            "parameter": "motional_dephasing_rate",
            "factor": factor,
            "overrides": {
                "dephasing_rate_phys": float(params["dephasing_rate_phys"])
                * factor
            },
        })
    return conditions


def run_parameter_robustness_stage(
    config: Mapping[str, Any],
    nbar_values: Iterable[float],
    *,
    run_qpt: bool = False,
    force_recompute: bool = False,
    show_progress: bool = True,
    eta_factors=(0.8, 1.2),
    a_over_delta_factors=(0.9, 1.1),
    gate_time_factors=(0.95, 1.05),
    motional_dephasing_factors=(0.0, 0.5, 2.0),
) -> dict[str, Any]:
    """Run/load eta, A/delta, gate-time and motional-dephasing variations."""

    paths = _paths(config)
    params = _simulation_params(config)
    generator = _load_generator(paths)
    selected_nbars = [float(value) for value in nbar_values]
    conditions = _robustness_conditions(
        params,
        eta_factors,
        a_over_delta_factors,
        gate_time_factors,
        motional_dephasing_factors,
    )
    pending = []

    condition_iterator = tqdm(
        conditions,
        desc="Parameter robustness",
        unit="condition",
        disable=not show_progress,
    )
    for condition in condition_iterator:
        if show_progress and hasattr(condition_iterator, "set_postfix_str"):
            condition_iterator.set_postfix_str(condition["name"])
        missing = [
            n_bar
            for n_bar in selected_nbars
            if force_recompute
            or not _simple_qpt_cache_path(
                paths["robustness"], condition["name"], n_bar
            ).exists()
        ]
        if missing and run_qpt:
            overrides = dict(condition["overrides"])
            overrides.update({
                "parallel_workers": int(config.get("FAST_PROCESS_WORKERS", 4)),
                # ms_gate_functions displays evolution-level progress for the
                # five nbar points within the current robustness condition.
                "show_progress": bool(show_progress),
            })
            for result in qpt_analysis.calculate_error_channel_batch(
                missing,
                params,
                overrides,
                convention=_convention(config),
            ):
                result["metadata"].update({
                    "condition": condition["name"],
                    "parameter": condition["parameter"],
                    "factor": condition["factor"],
                })
                qpt_analysis.save_qpt_point(
                    _simple_qpt_cache_path(
                        paths["robustness"], condition["name"], result["n_bar"]
                    ),
                    result["n_bar"],
                    condition["name"],
                    result["chi"],
                    result["metadata"],
                )
        elif missing:
            pending.extend(
                {"condition": condition["name"], "n_bar": n_bar}
                for n_bar in missing
            )

    rows = []
    for n_bar in selected_nbars:
        baseline_generator = generator.loc[np.isclose(generator["n_bar"], n_bar)]
        if baseline_generator.empty:
            continue
        baseline_chi = _load_cptp_chi(paths, n_bar)
        baseline_obs = _extract(baseline_chi, config)
        rows.append({
            "n_bar": n_bar,
            "condition": "baseline",
            "parameter": "baseline",
            "factor": 1.0,
            "h_XX_rad_per_gate": float(
                baseline_generator.iloc[0]["h_XX_rad_per_gate"]
            ),
            "gamma_XX_per_gate": float(
                baseline_generator.iloc[0]["gamma_XX_per_gate"]
            ),
            "average_infidelity": baseline_obs["average_infidelity"],
        })
        for condition in conditions:
            path = _simple_qpt_cache_path(
                paths["robustness"], condition["name"], n_bar
            )
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                chi = np.asarray(data["chi_trace_normalized"], dtype=complex)
            obs = _extract(chi, config)
            rows.append({
                "n_bar": n_bar,
                "condition": condition["name"],
                "parameter": condition["parameter"],
                "factor": condition["factor"],
                "h_XX_rad_per_gate": obs["h_XX_rad_per_gate"],
                "gamma_XX_per_gate": obs["gamma_XX_per_gate"],
                "average_infidelity": obs["average_infidelity"],
            })
    summary = pd.DataFrame(rows).sort_values(["n_bar", "condition"])
    summary.to_csv(paths["robustness"] / "parameter_robustness_summary.csv", index=False)

    nonbaseline = summary.loc[summary["condition"] != "baseline"]
    if not nonbaseline.empty:
        figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
        for axis, metric, title in [
            (axes[0], "h_XX_rad_per_gate", r"$h_{XX}$"),
            (axes[1], "gamma_XX_per_gate", r"$\gamma_{XX}$"),
            (axes[2], "average_infidelity", "Average infidelity"),
        ]:
            for condition, group in summary.groupby("condition"):
                axis.plot(
                    group["n_bar"], group[metric], marker="o", label=condition
                )
            axis.set_title(title)
            axis.set_xlabel(r"Mean phonon number $\bar n$")
            axis.grid(True, alpha=0.28)
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
        axes[-1].legend(fontsize=7, bbox_to_anchor=(1.03, 1.0), loc="upper left")
        figure.suptitle("Parameter robustness of the calibrated error structure")
        figure.tight_layout()
        figure.savefig(
            paths["robustness"] / "parameter_robustness.png",
            dpi=300,
            bbox_inches="tight",
        )
        figure.savefig(
            paths["robustness"] / "parameter_robustness.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)

    expected = len(conditions) * len(selected_nbars)
    completed = len(nonbaseline)
    return {
        "summary": summary,
        "conditions": pd.DataFrame(conditions).drop(columns="overrides"),
        "status": {
            "completed_points": completed,
            "expected_points": expected,
            "pending": pending,
            "run_qpt": bool(run_qpt),
        },
    }
