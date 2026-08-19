"""Noise-free diagnostics for the MS-gate nbar/rate-sweep notebook.

The routines in this module extend ``noise_rate_nbar_sweep.ipynb`` with five
diagnostics of the all-four-noises-off reference:

1. full-order versus first-order Lamb--Dicke QPT,
2. Fock-resolved XX angle and forced/null coherence,
3. one-step hXX drive-amplitude correction and re-QPT,
4. phase-space closure decomposition, and
5. phonon-cutoff/time-grid convergence.

Every expensive calculation is opt-in and resumable.  Physical settings are
stored in a manifest so cached data cannot silently be mixed across models.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drive_amplitude_calibration as drive_calibration
import drive_calibration_qpt_analysis as qpt_analysis
import laser_pulse_optimization as pulse_analysis
import ms_gate_functions as mg
import phonon_xx_angle_analysis as fock_analysis


TARGET_XX_ANGLE = np.pi / 4.0
RUNTIME_ONLY_PARAMETER_KEYS = {"parallel_workers", "show_progress"}


def _json_safe(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _scientific_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    selected = dict(parameters)
    for key in RUNTIME_ONLY_PARAMETER_KEYS:
        selected.pop(key, None)
    selected.pop("n_bar_list", None)
    return _json_safe(selected)


def _manifest_payload(
    analysis: str,
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float],
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "analysis": str(analysis),
        "base_parameters": _scientific_parameters(base_parameters),
        "nbar_values": [float(value) for value in nbar_values],
        "settings": _json_safe(dict(settings or {})),
        "version": 1,
    }


def _ensure_manifest(
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    execute: bool,
    resume: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    current = _json_safe(dict(payload))
    if resume and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != current:
            raise RuntimeError(
                f"Cached {payload['analysis']} configuration differs from the "
                f"current physical settings. Use a new output directory or "
                f"set resume=False: {path}"
            )
    if execute or not path.exists():
        path.write_text(
            json.dumps(current, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return path


def _atomic_save_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def _nbar_stem(n_bar: float) -> str:
    return f"{float(n_bar):.12g}".replace("-", "m").replace(".", "p")


def _complete_nbar_grid(frame: pd.DataFrame, nbar_values: Iterable[float]) -> bool:
    if frame.empty or "nbar" not in frame:
        return False
    completed = np.sort(frame["nbar"].astype(float).unique())
    requested = np.sort(np.asarray(list(nbar_values), dtype=float))
    return len(completed) == len(requested) and np.allclose(completed, requested)


def noise_free_parameters(
    base_parameters: Mapping[str, Any],
    *,
    nbar_values: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Return a copy with the four swept dissipative noises disabled."""

    parameters = dict(base_parameters)
    if nbar_values is not None:
        parameters["n_bar_list"] = [float(value) for value in nbar_values]
    parameters.update(
        {
            "heating_rate_phys": 0.0,
            "dephasing_rate_phys": 0.0,
            "T2_star": np.inf,
            "rayleigh_rate_phys": 0.0,
            "raman_rate_phys": 0.0,
        }
    )
    return parameters


def _infidelity_rows(result: Mapping[str, Any], model: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": str(model),
            "nbar": np.asarray(result["parameters"]["n_bar_list"], dtype=float),
            "F_avg": np.asarray(result["f_avg_list"], dtype=float),
            "infidelity": np.asarray(result["infidelity_list"], dtype=float),
        }
    )


def run_full_vs_ld_comparison(
    *,
    output_dir: str | Path,
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float],
    full_order_summary: pd.DataFrame | None = None,
    execute: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Compare all-noise-zero full-order QPT with first-order LD QPT."""

    output_dir = Path(output_dir)
    nbar_values = tuple(float(value) for value in nbar_values)
    payload = _manifest_payload(
        "full_vs_lamb_dicke",
        base_parameters,
        nbar_values,
        {"target_xx_angle_rad": TARGET_XX_ANGLE},
    )
    _ensure_manifest(output_dir, payload, execute=execute, resume=resume)
    ld_path = output_dir / "lamb_dicke_qpt_summary.csv"
    comparison_path = output_dir / "full_vs_lamb_dicke_summary.csv"

    if resume and ld_path.exists():
        ld_summary = pd.read_csv(ld_path)
    else:
        ld_summary = pd.DataFrame()

    if execute and not _complete_nbar_grid(ld_summary, nbar_values):
        parameters = noise_free_parameters(
            base_parameters, nbar_values=nbar_values
        )
        parameters["use_full_order"] = False
        result = mg.run_infidelity_analysis(show_plot=False, **parameters)
        ld_summary = _infidelity_rows(result, "lamb_dicke_first_order")
        _atomic_save_csv(ld_summary, ld_path)

    full_summary = pd.DataFrame()
    if full_order_summary is not None and not full_order_summary.empty:
        full_summary = (
            full_order_summary[["nbar", "F_avg", "infidelity"]]
            .copy()
            .assign(model="full_order")
        )

    combined = pd.concat([full_summary, ld_summary], ignore_index=True)
    comparison = pd.DataFrame()
    if not full_summary.empty and _complete_nbar_grid(ld_summary, nbar_values):
        comparison = full_summary[["nbar", "infidelity"]].merge(
            ld_summary[["nbar", "infidelity"]],
            on="nbar",
            suffixes=("_full_order", "_lamb_dicke"),
        )
        comparison["delta_infidelity_full_minus_ld"] = (
            comparison["infidelity_full_order"]
            - comparison["infidelity_lamb_dicke"]
        )
        comparison["higher_order_fraction_of_full"] = (
            comparison["delta_infidelity_full_minus_ld"]
            / np.maximum(comparison["infidelity_full_order"], 1e-18)
        )
        _atomic_save_csv(comparison, comparison_path)

    figure_path = output_dir / "full_vs_lamb_dicke.png"
    if not combined.empty:
        figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
        style = {
            "full_order": ("o-", "#D55E00", "Full order"),
            "lamb_dicke_first_order": ("s--", "#0072B2", "LD first order"),
        }
        for model, group in combined.groupby("model", sort=False):
            fmt, color, label = style.get(model, ("o-", None, model))
            group = group.sort_values("nbar")
            axes[0].semilogy(
                group["nbar"], group["infidelity"], fmt,
                color=color, linewidth=2.0, label=label,
            )
        axes[0].set_xlabel(r"Mean phonon number $\bar n$")
        axes[0].set_ylabel(r"Average infidelity $1-F_{avg}$")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend()
        if not comparison.empty:
            axes[1].plot(
                comparison["nbar"],
                comparison["delta_infidelity_full_minus_ld"],
                "o-", color="#CC79A7", linewidth=2.0,
            )
        axes[1].axhline(0.0, color="black", linewidth=1.0)
        axes[1].set_xlabel(r"Mean phonon number $\bar n$")
        axes[1].set_ylabel(r"$I_{full}-I_{LD1}$")
        axes[1].grid(True, alpha=0.3)
        figure.suptitle("Noise-free full-order vs Lamb-Dicke QPT")
        figure.tight_layout()
        figure.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(figure)

    return {
        "combined": combined,
        "comparison": comparison,
        "ld_summary": ld_summary,
        "figure_path": figure_path if figure_path.exists() else None,
        "output_dir": output_dir,
    }


def run_fock_resolved_analysis(
    *,
    output_dir: str | Path,
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float],
    execute: bool = False,
    resume: bool = True,
    thermal_tail_tolerance: float = 1e-5,
    max_fock_n: int | None = None,
    phonon_buffer: int = 24,
) -> dict[str, Any]:
    """Calculate/load the full-order Fock-conditioned XX coherence curve."""

    output_dir = Path(output_dir)
    nbar_values = tuple(float(value) for value in nbar_values)
    if max_fock_n is None:
        max_fock_n = max(
            fock_analysis.required_fock_cutoff(value, thermal_tail_tolerance)
            for value in nbar_values
        )
    max_fock_n = int(max_fock_n)
    settings = {
        "thermal_tail_tolerance": float(thermal_tail_tolerance),
        "max_fock_n": max_fock_n,
        "phonon_buffer": int(phonon_buffer),
        "target_xx_angle_rad": TARGET_XX_ANGLE,
    }
    payload = _manifest_payload(
        "fock_resolved_full_order", base_parameters, nbar_values, settings
    )
    _ensure_manifest(output_dir, payload, execute=execute, resume=resume)
    curve_path = output_dir / "fock_resolved_curve.npz"
    curve_csv_path = output_dir / "fock_resolved_curve.csv"
    summary_path = output_dir / "fock_thermal_summary.csv"

    parameters = noise_free_parameters(base_parameters)
    parameters["use_full_order"] = True
    if resume and curve_path.exists():
        curve = fock_analysis.load_fock_curve(curve_path)
    elif execute:
        curve = fock_analysis.calculate_fock_resolved_xx_angles(
            parameters,
            max_fock_n=max_fock_n,
            phonon_buffer=int(phonon_buffer),
            target_xx_angle_rad=TARGET_XX_ANGLE,
        )
        fock_analysis.save_fock_curve(curve_path, curve)
        _atomic_save_csv(curve, curve_csv_path)
    else:
        curve = pd.DataFrame()

    if not curve.empty:
        if not curve_csv_path.exists():
            _atomic_save_csv(curve, curve_csv_path)
        rows = []
        for n_bar in nbar_values:
            row = fock_analysis.summarize_thermal_xx_angle(
                curve, n_bar, target_xx_angle_rad=TARGET_XX_ANGLE
            )
            row["residual_motion_gamma_per_gate"] = max(
                0.0,
                row["gamma_XX_with_residual_motion_per_gate"]
                - row["gamma_XX_phase_dispersion_per_gate"],
            )
            rows.append(row)
        thermal_summary = pd.DataFrame(rows)
        _atomic_save_csv(thermal_summary, summary_path)
    elif resume and summary_path.exists():
        thermal_summary = pd.read_csv(summary_path)
    else:
        thermal_summary = pd.DataFrame()

    figure_path = output_dir / "fock_resolved_diagnostics.png"
    if not curve.empty and not thermal_summary.empty:
        figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
        axes[0, 0].plot(curve["fock_n"], curve["theta_xx_rad"], linewidth=2.0)
        axes[0, 0].axhline(
            TARGET_XX_ANGLE, color="black", linestyle="--", label=r"target $\pi/4$"
        )
        axes[0, 0].set_ylabel(r"$\theta_{XX}(n)$ [rad]")
        axes[0, 0].legend()
        axes[0, 1].semilogy(
            curve["fock_n"],
            np.maximum(1.0 - curve["coherence_abs"], 1e-18),
            linewidth=2.0,
        )
        axes[0, 1].set_ylabel(r"Closure loss $1-|c_n|$")
        axes[1, 0].plot(
            thermal_summary["thermal_n_bar"],
            thermal_summary["predicted_h_XX_rad_per_gate"],
            "o-", linewidth=2.0,
        )
        axes[1, 0].axhline(0.0, color="black", linewidth=1.0)
        axes[1, 0].set_ylabel(r"Predicted $h_{XX}$ [rad/gate]")
        axes[1, 1].semilogy(
            thermal_summary["thermal_n_bar"],
            np.maximum(
                thermal_summary["gamma_XX_phase_dispersion_per_gate"], 1e-18
            ),
            "o-", label="Fock-angle dispersion",
        )
        axes[1, 1].semilogy(
            thermal_summary["thermal_n_bar"],
            np.maximum(
                thermal_summary["residual_motion_gamma_per_gate"], 1e-18
            ),
            "s--", label="Residual motion",
        )
        axes[1, 1].set_ylabel(r"Equivalent $\gamma_{XX}$ / gate")
        axes[1, 1].legend()
        for axis in axes[0]:
            axis.set_xlabel("Initial Fock number n")
        for axis in axes[1]:
            axis.set_xlabel(r"Mean phonon number $\bar n$")
        for axis in axes.ravel():
            axis.grid(True, which="both", alpha=0.3)
        figure.suptitle("Noise-free Fock-resolved full-order diagnostics")
        figure.tight_layout()
        figure.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(figure)

    return {
        "curve": curve,
        "thermal_summary": thermal_summary,
        "max_fock_n": max_fock_n,
        "figure_path": figure_path if figure_path.exists() else None,
        "output_dir": output_dir,
    }


def _load_cached_chi(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["chi_trace_normalized"], dtype=complex)


def _calculate_and_save_qpt_point(
    path: Path,
    *,
    n_bar: float,
    condition: str,
    parameters: Mapping[str, Any],
    amplitude: float,
    convention: str,
) -> np.ndarray:
    point = qpt_analysis.calculate_error_channel_batch(
        [float(n_bar)],
        parameters,
        {"A": float(amplitude)},
        convention=convention,
    )[0]
    qpt_analysis.save_qpt_point(
        path,
        n_bar=float(n_bar),
        condition_name=condition,
        chi=point["chi"],
        metadata={**point["metadata"], "A": float(amplitude)},
    )
    return np.asarray(point["chi"], dtype=complex)


def run_xx_angle_correction_comparison(
    *,
    output_dir: str | Path,
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float],
    execute: bool = False,
    resume: bool = True,
    convention: str = "undo_before_actual",
) -> dict[str, Any]:
    """Apply one QPT-feedback hXX amplitude correction at each nbar."""

    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    nbar_values = tuple(float(value) for value in nbar_values)
    payload = _manifest_payload(
        "xx_angle_correction",
        base_parameters,
        nbar_values,
        {"target_xx_angle_rad": TARGET_XX_ANGLE, "convention": convention},
    )
    _ensure_manifest(output_dir, payload, execute=execute, resume=resume)
    summary_path = output_dir / "xx_angle_correction_summary.csv"
    parameters = noise_free_parameters(base_parameters)
    parameters["use_full_order"] = True
    baseline_amplitude = float(np.asarray(parameters["A"]))
    rows = []
    pending = []

    for n_bar in nbar_values:
        stem = _nbar_stem(n_bar)
        baseline_path = cache_dir / f"baseline_nbar_{stem}.npz"
        if not (resume and baseline_path.exists()):
            if not execute:
                pending.append({"nbar": n_bar, "reason": "baseline cache missing"})
                continue
            _calculate_and_save_qpt_point(
                baseline_path,
                n_bar=n_bar,
                condition="all_noise_zero_before_xx_correction",
                parameters=parameters,
                amplitude=baseline_amplitude,
                convention=convention,
            )
        baseline_chi = _load_cached_chi(baseline_path)
        before = qpt_analysis.extract_xx_generator_observables(baseline_chi)
        corrected_amplitude = drive_calibration.amplitude_update_from_hxx(
            baseline_amplitude,
            before["h_XX_rad_per_gate"],
            TARGET_XX_ANGLE,
        )
        corrected_path = cache_dir / f"corrected_nbar_{stem}.npz"
        if not (resume and corrected_path.exists()):
            if not execute:
                pending.append({"nbar": n_bar, "reason": "corrected cache missing"})
                continue
            _calculate_and_save_qpt_point(
                corrected_path,
                n_bar=n_bar,
                condition="all_noise_zero_after_xx_correction",
                parameters=parameters,
                amplitude=corrected_amplitude,
                convention=convention,
            )
        corrected_chi = _load_cached_chi(corrected_path)
        after = qpt_analysis.extract_xx_generator_observables(corrected_chi)
        rows.append(
            {
                "nbar": n_bar,
                "A_before": baseline_amplitude,
                "A_after": corrected_amplitude,
                "A_factor": corrected_amplitude / baseline_amplitude,
                "h_XX_before_rad_per_gate": before["h_XX_rad_per_gate"],
                "h_XX_after_rad_per_gate": after["h_XX_rad_per_gate"],
                "theta_XX_before_rad": TARGET_XX_ANGLE
                - before["h_XX_rad_per_gate"],
                "theta_XX_after_rad": TARGET_XX_ANGLE
                - after["h_XX_rad_per_gate"],
                "infidelity_before": before["average_infidelity"],
                "infidelity_after": after["average_infidelity"],
                "infidelity_reduction": before["average_infidelity"]
                - after["average_infidelity"],
                "gamma_XX_before_per_gate": before["gamma_XX_per_gate"],
                "gamma_XX_after_per_gate": after["gamma_XX_per_gate"],
            }
        )
        _atomic_save_csv(pd.DataFrame(rows).sort_values("nbar"), summary_path)

    if rows:
        summary = pd.DataFrame(rows).sort_values("nbar").reset_index(drop=True)
    elif resume and summary_path.exists():
        summary = pd.read_csv(summary_path).sort_values("nbar").reset_index(drop=True)
    else:
        summary = pd.DataFrame()

    figure_path = output_dir / "xx_angle_correction_before_after.png"
    if not summary.empty:
        figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
        axes[0].plot(
            summary["nbar"], summary["h_XX_before_rad_per_gate"],
            "o-", label="before",
        )
        axes[0].plot(
            summary["nbar"], summary["h_XX_after_rad_per_gate"],
            "s--", label="after",
        )
        axes[0].axhline(0.0, color="black", linewidth=1.0)
        axes[0].set_ylabel(r"$h_{XX}$ [rad/gate]")
        axes[1].semilogy(
            summary["nbar"], summary["infidelity_before"],
            "o-", label="before",
        )
        axes[1].semilogy(
            summary["nbar"], summary["infidelity_after"],
            "s--", label="after",
        )
        axes[1].set_ylabel(r"Average infidelity $1-F_{avg}$")
        axes[2].plot(summary["nbar"], summary["A_factor"], "o-")
        axes[2].axhline(1.0, color="black", linewidth=1.0)
        axes[2].set_ylabel(r"Corrected amplitude $A/A_0$")
        for axis in axes:
            axis.set_xlabel(r"Mean phonon number $\bar n$")
            axis.grid(True, which="both", alpha=0.3)
            if axis is not axes[2]:
                axis.legend()
        figure.suptitle("Noise-free XX-angle correction: before and after")
        figure.tight_layout()
        figure.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(figure)

    return {
        "summary": summary,
        "pending": pd.DataFrame(pending),
        "figure_path": figure_path if figure_path.exists() else None,
        "output_dir": output_dir,
    }


def build_phase_space_closure_analysis(
    *,
    output_dir: str | Path,
    base_parameters: Mapping[str, Any],
    fock_curve: pd.DataFrame,
    fock_thermal_summary: pd.DataFrame,
    all_noise_zero_summary: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Separate first-order closure, Fock-angle dispersion, and motion loss."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if fock_curve.empty or fock_thermal_summary.empty:
        return {
            "summary": pd.DataFrame(),
            "figure_path": None,
            "output_dir": output_dir,
        }

    parameters = noise_free_parameters(base_parameters)
    detuning = np.asarray(parameters["delta"], dtype=float)
    configured_gate_time = parameters.get("t_gate_sim")
    if configured_gate_time is None:
        if detuning.ndim != 0 or float(detuning) == 0.0:
            raise ValueError("t_gate_sim is required for waveform detuning")
        gate_time = 2.0 * np.pi / abs(float(detuning))
    else:
        gate_time = float(configured_gate_time)
    time_points = int(parameters.get("time_points", 500))
    time = np.linspace(0.0, gate_time, time_points)
    amplitude = np.asarray(parameters["A"], dtype=float)
    if amplitude.ndim == 0:
        amplitude = np.full_like(time, float(amplitude))
    detuning_values = np.asarray(parameters["delta"], dtype=float)
    if detuning_values.ndim == 0:
        detuning_values = np.full_like(time, float(detuning_values))
    magnus = pulse_analysis.ms_magnus_metrics(time, amplitude, detuning_values)
    phase_space_drive = amplitude * np.exp(1j * magnus["phase"])
    displacement_trajectory = np.zeros(len(time), dtype=complex)
    displacement_trajectory[1:] = -1j * np.cumsum(
        0.5
        * (phase_space_drive[1:] + phase_space_drive[:-1])
        * np.diff(time)
    )
    trajectory = pd.DataFrame(
        {
            "time_sim": time,
            "alpha_real_per_spin_eigenvalue": np.real(displacement_trajectory),
            "alpha_imag_per_spin_eigenvalue": np.imag(displacement_trajectory),
            "alpha_abs_per_spin_eigenvalue": np.abs(displacement_trajectory),
        }
    )
    _atomic_save_csv(trajectory, output_dir / "phase_space_trajectory.csv")

    summary = fock_thermal_summary.copy()
    summary["first_order_displacement_abs"] = magnus["displacement_abs"]
    summary["first_order_forced_branch_displacement_abs"] = (
        2.0 * magnus["displacement_abs"]
    )
    summary["first_order_xx_angle_rad"] = magnus["xx_angle"]
    summary["residual_motion_gamma_per_gate"] = np.maximum(
        summary["gamma_XX_with_residual_motion_per_gate"]
        - summary["gamma_XX_phase_dispersion_per_gate"],
        0.0,
    )
    summary["thermal_closure_loss"] = (
        1.0 - summary["mean_forced_null_coherence_abs"]
    )
    if all_noise_zero_summary is not None and not all_noise_zero_summary.empty:
        summary = summary.merge(
            all_noise_zero_summary[["nbar", "infidelity"]].rename(
                columns={
                    "nbar": "thermal_n_bar",
                    "infidelity": "all_noise_zero_infidelity",
                }
            ),
            on="thermal_n_bar",
            how="left",
        )
    summary_path = output_dir / "phase_space_closure_summary.csv"
    _atomic_save_csv(summary, summary_path)

    figure_path = output_dir / "phase_space_closure.png"
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))
    axes[0, 0].plot(
        trajectory["alpha_real_per_spin_eigenvalue"],
        trajectory["alpha_imag_per_spin_eigenvalue"],
        linewidth=2.0,
    )
    axes[0, 0].scatter([0.0], [0.0], marker="o", color="black", label="start")
    axes[0, 0].scatter(
        [trajectory["alpha_real_per_spin_eigenvalue"].iloc[-1]],
        [trajectory["alpha_imag_per_spin_eigenvalue"].iloc[-1]],
        marker="x", color="#D55E00", s=60, label="end",
    )
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].set_xlabel(r"Re $\alpha(t)$")
    axes[0, 0].set_ylabel(r"Im $\alpha(t)$")
    axes[0, 0].legend()
    axes[0, 1].semilogy(
        fock_curve["fock_n"],
        np.maximum(1.0 - fock_curve["coherence_abs"], 1e-18),
        linewidth=2.0,
    )
    axes[0, 1].set_xlabel("Initial Fock number n")
    axes[0, 1].set_ylabel(r"Full-order closure loss $1-|c_n|$")
    axes[1, 0].semilogy(
        summary["thermal_n_bar"],
        np.maximum(summary["gamma_XX_phase_dispersion_per_gate"], 1e-18),
        "o-", label="angle dispersion",
    )
    axes[1, 0].semilogy(
        summary["thermal_n_bar"],
        np.maximum(summary["residual_motion_gamma_per_gate"], 1e-18),
        "s--", label="residual motion",
    )
    axes[1, 0].set_xlabel(r"Mean phonon number $\bar n$")
    axes[1, 0].set_ylabel(r"Equivalent $\gamma_{XX}$ / gate")
    axes[1, 0].legend()
    axes[1, 1].plot(
        summary["thermal_n_bar"], summary["thermal_closure_loss"], "o-"
    )
    axes[1, 1].set_xlabel(r"Mean phonon number $\bar n$")
    axes[1, 1].set_ylabel(r"Thermal closure loss $1-|\langle c_n\rangle|$")
    for axis in axes.ravel():
        axis.grid(True, which="both", alpha=0.3)
    figure.suptitle(
        "Phase-space closure diagnostics "
        f"(first-order |alpha(T)|={magnus['displacement_abs']:.3e})"
    )
    figure.tight_layout()
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    return {
        "summary": summary,
        "trajectory": trajectory,
        "magnus": magnus,
        "figure_path": figure_path,
        "output_dir": output_dir,
    }


def _estimate_alpha_max_for_cutoff(parameters: Mapping[str, Any]) -> float:
    amplitude = np.asarray(parameters["A"], dtype=float)
    detuning = np.abs(np.asarray(parameters["delta"], dtype=float))
    if float(np.min(detuning)) <= 0.0:
        raise ValueError("detuning magnitude must be positive")
    return 2.0 * float(np.max(np.abs(amplitude))) / float(np.min(detuning))


def _single_qpt_infidelity(parameters: Mapping[str, Any], n_bar: float) -> float:
    point_parameters = dict(parameters)
    point_parameters["n_bar_list"] = [float(n_bar)]
    result = mg.run_infidelity_analysis(show_plot=False, **point_parameters)
    return float(result["infidelity_list"][0])


def run_numerical_convergence(
    *,
    output_dir: str | Path,
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float] = (0.01, 4.0),
    execute: bool = False,
    resume: bool = True,
    phonon_dim_factors: Iterable[float] = (1.0, 1.25, 1.5),
    time_point_factors: Iterable[float] = (1.0, 1.5, 2.0),
) -> dict[str, Any]:
    """Check all-noise-zero QPT convergence in cutoff and time resolution."""

    output_dir = Path(output_dir)
    nbar_values = tuple(float(value) for value in nbar_values)
    phonon_dim_factors = tuple(float(value) for value in phonon_dim_factors)
    time_point_factors = tuple(float(value) for value in time_point_factors)
    settings = {
        "phonon_dim_factors": phonon_dim_factors,
        "time_point_factors": time_point_factors,
    }
    payload = _manifest_payload(
        "numerical_convergence", base_parameters, nbar_values, settings
    )
    _ensure_manifest(output_dir, payload, execute=execute, resume=resume)
    summary_path = output_dir / "numerical_convergence_summary.csv"
    if resume and summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.DataFrame()

    parameters = noise_free_parameters(base_parameters)
    parameters["use_full_order"] = True
    base_time_points = int(parameters.get("time_points", 500))
    detuning = np.asarray(parameters["delta"], dtype=float)
    configured_gate_time = parameters.get("t_gate_sim")
    if configured_gate_time is None:
        if detuning.ndim != 0 or float(detuning) == 0.0:
            raise ValueError("t_gate_sim is required for waveform detuning")
        gate_time = 2.0 * np.pi / abs(float(detuning))
    else:
        gate_time = float(configured_gate_time)
    alpha_max = _estimate_alpha_max_for_cutoff(parameters)

    requested_points = []
    for n_bar in nbar_values:
        auto_dim = int(mg.estimate_phonon_dim(n_bar, alpha_max))
        dimensions = sorted(
            {max(2, int(np.ceil(auto_dim * factor))) for factor in phonon_dim_factors}
        )
        reference_dim = max(dimensions)
        for phonon_dim in dimensions:
            requested_points.append(
                {
                    "nbar": n_bar,
                    "sweep": "phonon_cutoff",
                    "phonon_dim": phonon_dim,
                    "time_points": base_time_points,
                    "solver_max_step": np.nan,
                    "auto_phonon_dim": auto_dim,
                }
            )
        for factor in time_point_factors:
            time_points = max(2, int(round(base_time_points * factor)))
            requested_points.append(
                {
                    "nbar": n_bar,
                    "sweep": "time_grid",
                    "phonon_dim": reference_dim,
                    "time_points": time_points,
                    "solver_max_step": gate_time / (time_points - 1),
                    "auto_phonon_dim": auto_dim,
                }
            )

    for point in requested_points:
        if not summary.empty:
            mask = (
                np.isclose(summary["nbar"].astype(float), point["nbar"])
                & summary["sweep"].eq(point["sweep"])
                & summary["phonon_dim"].astype(int).eq(point["phonon_dim"])
                & summary["time_points"].astype(int).eq(point["time_points"])
            )
            if mask.any():
                continue
        if not execute:
            continue
        point_parameters = dict(parameters)
        point_parameters.update(
            {
                "phonon_dim_override": int(point["phonon_dim"]),
                "time_points": int(point["time_points"]),
                "solver_max_step": (
                    None
                    if not np.isfinite(point["solver_max_step"])
                    else float(point["solver_max_step"])
                ),
            }
        )
        infidelity = _single_qpt_infidelity(
            point_parameters, float(point["nbar"])
        )
        row = {**point, "infidelity": infidelity}
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
        summary = summary.sort_values(
            ["nbar", "sweep", "phonon_dim", "time_points"]
        ).reset_index(drop=True)
        _atomic_save_csv(summary, summary_path)

    analyzed = summary.copy()
    if not analyzed.empty:
        references = (
            analyzed.sort_values(["phonon_dim", "time_points"])
            .groupby("nbar", as_index=False)
            .tail(1)[["nbar", "infidelity"]]
            .rename(columns={"infidelity": "reference_infidelity"})
        )
        analyzed = analyzed.merge(references, on="nbar", how="left")
        analyzed["abs_delta_to_reference"] = np.abs(
            analyzed["infidelity"] - analyzed["reference_infidelity"]
        )
        analyzed["relative_delta_to_reference"] = (
            analyzed["abs_delta_to_reference"]
            / np.maximum(np.abs(analyzed["reference_infidelity"]), 1e-18)
        )
        _atomic_save_csv(analyzed, output_dir / "numerical_convergence_analyzed.csv")

    figure_path = output_dir / "numerical_convergence.png"
    if not analyzed.empty:
        figure, axes = plt.subplots(
            len(nbar_values), 2,
            figsize=(11.0, 3.6 * len(nbar_values)),
            squeeze=False,
        )
        for row_index, n_bar in enumerate(nbar_values):
            subset = analyzed[np.isclose(analyzed["nbar"], n_bar)]
            cutoff = subset[subset["sweep"] == "phonon_cutoff"].sort_values(
                "phonon_dim"
            )
            time_grid = subset[subset["sweep"] == "time_grid"].sort_values(
                "time_points"
            )
            axes[row_index, 0].plot(
                cutoff["phonon_dim"], cutoff["infidelity"], "o-"
            )
            axes[row_index, 1].plot(
                time_grid["time_points"], time_grid["infidelity"], "o-"
            )
            axes[row_index, 0].set_ylabel(
                rf"Infidelity ($\bar n={n_bar:g}$)"
            )
            axes[row_index, 0].set_xlabel("Phonon dimension")
            axes[row_index, 1].set_xlabel("Time points / max-step resolution")
            for axis in axes[row_index]:
                axis.grid(True, alpha=0.3)
        figure.suptitle("Noise-free numerical convergence")
        figure.tight_layout()
        figure.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(figure)

    return {
        "summary": analyzed,
        "raw_summary": summary,
        "requested_points": pd.DataFrame(requested_points),
        "figure_path": figure_path if figure_path.exists() else None,
        "output_dir": output_dir,
    }


__all__ = [
    "TARGET_XX_ANGLE",
    "noise_free_parameters",
    "run_full_vs_ld_comparison",
    "run_fock_resolved_analysis",
    "run_xx_angle_correction_comparison",
    "build_phase_space_closure_analysis",
    "run_numerical_convergence",
]
