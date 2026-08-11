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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drive_amplitude_calibration as amplitude_calibration
import drive_calibration_qpt_analysis as qpt_analysis
import ms_gate_functions as mg


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
            cache_path = _drive_cache_path(
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

    iterations_path = paths["drive"] / "hxx_drive_feedback_qpt_iterations.csv"
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
        summary.to_csv(
            paths["drive"] / "hxx_drive_calibration_final_summary.csv",
            index=False,
        )
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
    return {"iterations": iterations, "summary": summary, "status": status}


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

    for candidate in candidates:
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
                "show_progress": False,
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

    for condition in conditions:
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
                "show_progress": False,
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
