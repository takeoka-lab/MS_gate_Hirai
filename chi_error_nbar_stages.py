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
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - minimal environments use plain iteration
    def tqdm(iterable, **_kwargs):
        return iterable

import drive_amplitude_calibration as amplitude_calibration
import drive_calibration_qpt_analysis as qpt_analysis
import ms_gate_functions as mg
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
