"""Smooth-detuning versus standard laser-driven MS-gate analysis.

This module contains the reusable pulse generation, calibration, QPT sweep,
metric extraction, persistence, and plotting code used by
``smooth_detuning_laser_MS_comparison.ipynb``.  Times are in seconds and all
Hamiltonian frequencies are angular frequencies in rad/s.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd
import qutip as qp
from scipy.optimize import minimize_scalar

import laser_pulse_optimization as lpo
import ms_gate_functions as mg


matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_XX_ANGLE = np.pi / 4
DEFAULT_NBAR_VALUES = (0.01, 1.0, 2.0, 3.0, 4.0)
DEFAULT_ABLATION_CASES = (
    "none",
    "heating_only",
    "motional_dephasing_only",
    "all_motional",
    "all_noise",
)
SPIN_OPTICAL_INDIVIDUAL_CASES = (
    "spin_dephasing_only",
    "photon_scattering_only",
)
BASE_NOISE = {
    "heating_rate_phys": 10.0,
    "dephasing_rate_phys": 18.0,
    "T2_star": 0.3,
    "rayleigh_rate_phys": 3.0,
    "raman_rate_phys": 1.0,
}
INDEPENDENT_NOISES = (
    "heating",
    "motional_dephasing",
    "spin_dephasing",
    "rayleigh",
    "raman",
)
PAULI_LABELS_AND_WEIGHTS = mg.pauli_labels_and_weights()
PAULI_LABELS = [label for label, _ in PAULI_LABELS_AND_WEIGHTS]
PAULI_WEIGHTS = np.asarray([weight for _, weight in PAULI_LABELS_AND_WEIGHTS])


@dataclass(frozen=True)
class SmoothDetuningConfigSI:
    """Five-segment Hughes smooth-detuning pulse in SI units."""

    j: int = 3
    delta_max_hz: float = 400.0e3
    delta_min_hz: float = 21.7e3
    tau_g_s: float = 5.0e-6
    tau_d_s: float = 100.0e-6
    t_hold_s: float = 15.8e-6
    omega_plateau_hz: float = 6.0e3

    @property
    def gate_time_s(self) -> float:
        return 2 * self.tau_g_s + 2 * self.tau_d_s + self.t_hold_s


@dataclass(frozen=True)
class StandardGateConfigSI:
    """Current constant-detuning gate parameters in SI units."""

    gate_time_s: float = 100.0e-6
    detuning_hz: float = 10.0e3
    effective_sideband_hz: float = 2.5e3


@dataclass(frozen=True)
class NbarSweepConfig:
    """Numerical settings for the direct standard/smooth QPT comparison."""

    nbar_values: tuple[float, ...] = DEFAULT_NBAR_VALUES
    noise_case: str = "all_noise"
    scattering_model: str = "constant_rate"
    standard_time_points: int = 501
    smooth_time_points: int = 1501
    standard_solver_max_step_s: float = 0.2e-6
    smooth_solver_max_step_s: float = 0.05e-6
    parallel_workers: int = 1
    random_seed: int = 1234
    show_progress: bool = True
    use_full_order: bool = True

    def __post_init__(self) -> None:
        if not self.nbar_values:
            raise ValueError("nbar_values must not be empty.")
        if any(float(value) < 0 for value in self.nbar_values):
            raise ValueError("nbar_values must be non-negative.")
        if tuple(sorted(set(self.nbar_values))) != tuple(self.nbar_values):
            raise ValueError("nbar_values must be unique and increasing.")
        if self.scattering_model not in {"constant_rate", "intensity_scaled"}:
            raise ValueError(
                "scattering_model must be 'constant_rate' or 'intensity_scaled'."
            )
        if self.standard_time_points < 2 or self.smooth_time_points < 2:
            raise ValueError("time-point counts must be at least two.")
        if self.standard_solver_max_step_s <= 0 or self.smooth_solver_max_step_s <= 0:
            raise ValueError("solver max steps must be positive.")
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be at least one.")
        noise_parameters(self.noise_case)


def _smooth_detuning_down(u, config: SmoothDetuningConfigSI):
    """Return the downward detuning sweep without fractional powers of negatives."""
    u = np.asarray(u, dtype=float)
    if config.j <= 0:
        raise ValueError("j must be positive.")
    if config.delta_max_hz == 0 or config.delta_min_hz == 0:
        raise ValueError("delta_max_hz and delta_min_hz must be non-zero.")
    if np.sign(config.delta_max_hz) != np.sign(config.delta_min_hz):
        raise ValueError("delta_max_hz and delta_min_hz must have the same sign.")

    delta_max = 2 * np.pi * abs(config.delta_max_hz)
    delta_min = 2 * np.pi * abs(config.delta_min_hz)
    g = u / 2 - config.tau_d_s / (4 * np.pi) * np.sin(
        2 * np.pi * u / config.tau_d_s
    )
    D = delta_max ** (-config.j) + (2 / config.tau_d_s) * (
        delta_min ** (-config.j) - delta_max ** (-config.j)
    ) * g
    return np.sign(config.delta_max_hz) * D ** (-1 / config.j)


def build_smooth_pulse_si(
    config: SmoothDetuningConfigSI,
    time_points: int = 4001,
    detuning_offset_hz: float = 0.0,
):
    """Build the five-segment smooth-detuning pulse.

    The paper's ``Omega_g`` maps to the current simulator coefficient
    ``A = Omega_g / 2``.  Positive detuning is used by default because it gives
    the repository's existing ``exp(+i pi XX/4)`` convention.
    """
    if int(time_points) < 2:
        raise ValueError("time_points must be at least two.")
    time = np.linspace(0.0, config.gate_time_s, int(time_points))
    omega = np.empty_like(time)
    detuning = np.empty_like(time)
    omega0 = 2 * np.pi * config.omega_plateau_hz
    delta_max = 2 * np.pi * config.delta_max_hz
    delta_min = 2 * np.pi * config.delta_min_hz
    tau_g, tau_d, hold = config.tau_g_s, config.tau_d_s, config.t_hold_s

    segment_1 = time <= tau_g
    local_time = time[segment_1]
    omega[segment_1] = omega0 * np.sin(np.pi * local_time / (2 * tau_g)) ** 2
    detuning[segment_1] = delta_max

    segment_2 = (time > tau_g) & (time <= tau_g + tau_d)
    local_time = time[segment_2] - tau_g
    omega[segment_2] = omega0
    detuning[segment_2] = _smooth_detuning_down(local_time, config)

    segment_3 = (time > tau_g + tau_d) & (time <= tau_g + tau_d + hold)
    omega[segment_3] = omega0
    detuning[segment_3] = delta_min

    segment_4 = (time > tau_g + tau_d + hold) & (
        time <= tau_g + 2 * tau_d + hold
    )
    local_time = time[segment_4] - (tau_g + tau_d + hold)
    omega[segment_4] = omega0
    detuning[segment_4] = _smooth_detuning_down(tau_d - local_time, config)

    segment_5 = time > tau_g + 2 * tau_d + hold
    local_time = time[segment_5] - (tau_g + 2 * tau_d + hold)
    omega[segment_5] = omega0 * np.cos(np.pi * local_time / (2 * tau_g)) ** 2
    detuning[segment_5] = delta_max

    detuning += 2 * np.pi * float(detuning_offset_hz)
    return {
        "time_s": time,
        "omega_rad_s": omega,
        "A_rad_s": omega / 2,
        "delta_rad_s": detuning,
    }


def build_standard_pulse_si(
    config: Optional[StandardGateConfigSI] = None,
    time_points: int = 1001,
    detuning_offset_hz: float = 0.0,
):
    """Build the repository's current constant-detuning standard pulse."""
    config = StandardGateConfigSI() if config is None else config
    if int(time_points) < 2:
        raise ValueError("time_points must be at least two.")
    time = np.linspace(0.0, config.gate_time_s, int(time_points))
    amplitude = np.full_like(time, 2 * np.pi * config.effective_sideband_hz)
    detuning = np.full_like(
        time,
        2 * np.pi * (config.detuning_hz + float(detuning_offset_hz)),
    )
    return {
        "time_s": time,
        "omega_rad_s": 2 * amplitude,
        "A_rad_s": amplitude,
        "delta_rad_s": detuning,
    }


def integrated_phase(time, detuning):
    """Return Phi(t)=integral_0^t delta(s) ds by trapezoidal integration."""
    time = np.asarray(time, dtype=float)
    detuning = np.asarray(detuning, dtype=float)
    if time.shape != detuning.shape or time.ndim != 1:
        raise ValueError("time and detuning must be one-dimensional arrays of equal shape.")
    phase = np.zeros_like(time)
    phase[1:] = np.cumsum(
        0.5 * (detuning[1:] + detuning[:-1]) * np.diff(time)
    )
    return phase


def pulse_diagnostics(pulse_type: str, pulse):
    """Calculate phase-space and integrated-intensity diagnostics."""
    time = np.asarray(pulse["time_s"], dtype=float)
    amplitude = np.asarray(pulse["A_rad_s"], dtype=float)
    omega = np.asarray(pulse["omega_rad_s"], dtype=float)
    detuning = np.asarray(pulse["delta_rad_s"], dtype=float)
    phase = integrated_phase(time, detuning)
    alpha = np.zeros_like(time, dtype=complex)
    alpha[1:] = -1j * np.cumsum(
        0.5
        * (
            amplitude[1:] * np.exp(1j * phase[1:])
            + amplitude[:-1] * np.exp(1j * phase[:-1])
        )
        * np.diff(time)
    )
    magnus = lpo.ms_magnus_metrics(time, amplitude, detuning)
    return {
        "pulse_type": pulse_type,
        "gate_time_s": float(time[-1]),
        "alpha_final_abs": float(abs(alpha[-1])),
        "alpha_max_abs": float(np.max(abs(alpha))),
        "alpha2_integral": float(np.trapz(abs(alpha) ** 2, time)),
        "xx_angle_rad": float(magnus["xx_angle"]),
        "entangling_angle_error": float(magnus["xx_angle"] - TARGET_XX_ANGLE),
        "integrated_omega2": float(np.trapz(omega**2, time)),
        "time_s": time,
        "alpha": alpha,
    }


def validate_smooth_waveform(
    config: SmoothDetuningConfigSI,
    time_points: int = 8001,
):
    """Run the inexpensive waveform boundary and phase-convention tests."""
    pulse = build_smooth_pulse_si(config, time_points=time_points)
    time = pulse["time_s"]
    omega = pulse["omega_rad_s"]
    detuning = pulse["delta_rad_s"]
    delta_max = 2 * np.pi * config.delta_max_hz
    delta_min = 2 * np.pi * config.delta_min_hz

    assert abs(omega[0]) < 1e-8 and abs(omega[-1]) < 1e-8
    assert np.isclose(detuning[0], delta_max)
    assert np.isclose(detuning[-1], delta_max)

    boundaries = (
        config.tau_g_s,
        config.tau_g_s + config.tau_d_s,
        config.tau_g_s + config.tau_d_s + config.t_hold_s,
        config.tau_g_s + 2 * config.tau_d_s + config.t_hold_s,
    )
    for boundary in boundaries:
        index = int(np.argmin(abs(time - boundary)))
        assert abs(omega[index + 1] - omega[index - 1]) < 0.02 * max(omega)
        assert abs(detuning[index + 1] - detuning[index - 1]) < 0.02 * max(
            abs(detuning)
        )
    for boundary in boundaries[1:3]:
        index = int(np.argmin(abs(time - boundary)))
        assert np.isclose(detuning[index], delta_min, rtol=2e-3)

    phase = integrated_phase(time, detuning)
    recovered_detuning = np.gradient(phase, time)
    phase_error = float(
        np.max(abs(recovered_detuning[2:-2] - detuning[2:-2]))
        / max(abs(detuning))
    )
    assert phase_error < 5e-3

    omega_slope = np.gradient(omega, time)
    endpoint_slope_fraction = float(
        max(abs(omega_slope[[0, -1]])) / max(abs(omega_slope))
    )
    assert endpoint_slope_fraction < 1e-2

    constant_config = replace(config, delta_min_hz=config.delta_max_hz)
    constant_detuning = build_smooth_pulse_si(
        constant_config,
        time_points=1001,
    )["delta_rad_s"]
    assert np.allclose(constant_detuning, delta_max)
    return {
        "relative_phase_derivative_error": phase_error,
        "endpoint_omega_slope_fraction": endpoint_slope_fraction,
        "gate_time_s": float(time[-1]),
    }


def calibrate_delta_min(
    config: Optional[SmoothDetuningConfigSI] = None,
    bounds_hz=(8e3, 80e3),
    search_time_points: int = 3001,
    diagnostic_time_points: int = 8001,
):
    """Calibrate only delta_min in the noise-free first-order Magnus model."""
    config = SmoothDetuningConfigSI() if config is None else config

    def objective(delta_min_hz):
        candidate = replace(config, delta_min_hz=float(delta_min_hz))
        pulse = build_smooth_pulse_si(candidate, time_points=search_time_points)
        fidelity = lpo.analytic_ms_average_gate_fidelity(
            pulse["time_s"],
            pulse["A_rad_s"],
            pulse["delta_rad_s"],
            n_bar=0.01,
            target_xx_angle=TARGET_XX_ANGLE,
        )
        return 1.0 - fidelity

    result = minimize_scalar(
        objective,
        bounds=bounds_hz,
        method="bounded",
        options={"xatol": 0.1},
    )
    calibrated = replace(config, delta_min_hz=float(result.x))
    pulse = build_smooth_pulse_si(calibrated, time_points=diagnostic_time_points)
    diagnostics = pulse_diagnostics("smooth_detuning", pulse)
    fidelity = lpo.analytic_ms_average_gate_fidelity(
        pulse["time_s"],
        pulse["A_rad_s"],
        pulse["delta_rad_s"],
        n_bar=0.01,
        target_xx_angle=TARGET_XX_ANGLE,
    )
    calibration = {
        "delta_min_hz": calibrated.delta_min_hz,
        "F_avg_noise_free_analytic": float(fidelity),
        "xx_angle_rad": diagnostics["xx_angle_rad"],
        "entangling_angle_error_rad": diagnostics["entangling_angle_error"],
        "alpha_final_abs": diagnostics["alpha_final_abs"],
        "alpha_max_abs": diagnostics["alpha_max_abs"],
        "alpha2_integral": diagnostics["alpha2_integral"],
        "gate_time_s": diagnostics["gate_time_s"],
        "integrated_omega2": diagnostics["integrated_omega2"],
    }
    if calibration["F_avg_noise_free_analytic"] <= 0.9999:
        raise RuntimeError("Noise-free smooth-pulse calibration did not reach 0.9999.")
    return calibrated, calibration


def noise_parameters(case: str):
    """Map an ablation label to the unchanged repository noise parameters."""
    if case == "none":
        active = set()
    elif case == "heating_only":
        active = {"heating"}
    elif case == "motional_dephasing_only":
        active = {"motional_dephasing"}
    elif case == "spin_dephasing_only":
        active = {"spin_dephasing"}
    elif case == "rayleigh_only":
        active = {"rayleigh"}
    elif case == "raman_only":
        active = {"raman"}
    elif case == "photon_scattering_only":
        active = {"rayleigh", "raman"}
    elif case == "all_motional":
        active = {"heating", "motional_dephasing"}
    elif case == "all_spin_and_optical":
        active = {"spin_dephasing", "rayleigh", "raman"}
    elif case == "all_noise":
        active = set(INDEPENDENT_NOISES)
    elif case.startswith("all_except_"):
        removed = case.removeprefix("all_except_")
        if removed not in INDEPENDENT_NOISES:
            raise ValueError(f"Unknown noise case: {case}")
        active = set(INDEPENDENT_NOISES) - {removed}
    else:
        raise ValueError(f"Unknown noise case: {case}")

    return {
        "heating_rate_phys": (
            BASE_NOISE["heating_rate_phys"] if "heating" in active else 0.0
        ),
        "dephasing_rate_phys": (
            BASE_NOISE["dephasing_rate_phys"]
            if "motional_dephasing" in active
            else 0.0
        ),
        "T2_star": BASE_NOISE["T2_star"] if "spin_dephasing" in active else 1e99,
        "rayleigh_rate_phys": (
            BASE_NOISE["rayleigh_rate_phys"] if "rayleigh" in active else 0.0
        ),
        "raman_rate_phys": (
            BASE_NOISE["raman_rate_phys"] if "raman" in active else 0.0
        ),
    }


def analytic_nbar_preview(
    nbar_values=DEFAULT_NBAR_VALUES,
    smooth_config: Optional[SmoothDetuningConfigSI] = None,
    standard_config: Optional[StandardGateConfigSI] = None,
):
    """Return a fast, explicitly noise-free first-order preview."""
    smooth_config = (
        calibrate_delta_min()[0] if smooth_config is None else smooth_config
    )
    standard_config = StandardGateConfigSI() if standard_config is None else standard_config
    pulses = {
        "standard": build_standard_pulse_si(standard_config, time_points=4001),
        "smooth_detuning": build_smooth_pulse_si(smooth_config, time_points=8001),
    }
    rows = []
    for pulse_type, pulse in pulses.items():
        for nbar in nbar_values:
            fidelity = lpo.analytic_ms_average_gate_fidelity(
                pulse["time_s"],
                pulse["A_rad_s"],
                pulse["delta_rad_s"],
                n_bar=float(nbar),
                target_xx_angle=TARGET_XX_ANGLE,
            )
            rows.append(
                {
                    "pulse_type": pulse_type,
                    "nbar": float(nbar),
                    "F_avg": float(fidelity),
                    "infidelity": float(1.0 - fidelity),
                    "model": "first_order_noise_free_preview",
                }
            )
    return pd.DataFrame(rows)


def _trace_normalize(matrix):
    array = matrix.full() if hasattr(matrix, "full") else np.asarray(matrix, complex)
    return array / np.trace(array)


def _safe_ratio(numerator, denominator):
    return np.nan if abs(denominator) < 1e-15 else float(numerator / denominator)


def _pulse_and_solver_settings(
    pulse_type: str,
    smooth_config: SmoothDetuningConfigSI,
    standard_config: StandardGateConfigSI,
    sweep_config: NbarSweepConfig,
):
    if pulse_type == "standard":
        pulse = build_standard_pulse_si(
            standard_config,
            time_points=sweep_config.standard_time_points,
        )
        max_step = sweep_config.standard_solver_max_step_s
    elif pulse_type == "smooth_detuning":
        pulse = build_smooth_pulse_si(
            smooth_config,
            time_points=sweep_config.smooth_time_points,
        )
        max_step = sweep_config.smooth_solver_max_step_s
    else:
        raise ValueError(f"Unknown pulse_type: {pulse_type}")
    return pulse, max_step


def _qpt_parameters(
    pulse,
    max_step: float,
    sweep_config: NbarSweepConfig,
    standard_config: StandardGateConfigSI,
):
    time = pulse["time_s"]
    return {
        "A": pulse["A_rad_s"],
        "delta": pulse["delta_rad_s"],
        "rho0": 0.0,
        "n_bar_list": [float(value) for value in sweep_config.nbar_values],
        "time_points": len(time),
        "t_gate_sim": float(time[-1]),
        "t_gate_phys": float(time[-1]),
        "eta": 0.1,
        "use_full_order": sweep_config.use_full_order,
        "show_progress": sweep_config.show_progress,
        "parallel_workers": sweep_config.parallel_workers,
        "solver_max_step": max_step,
        "laser_noise_samples": 1,
        "laser_noise_seed": sweep_config.random_seed,
        "laser_intensity_fluctuation": 0.0,
        "laser_detuning_fluctuation": 0.0,
        "laser_rotation_angle_fluctuation": 0.0,
        "laser_scattering_scales_with_intensity": (
            sweep_config.scattering_model == "intensity_scaled"
        ),
        "scattering_reference_amplitude": (
            2 * np.pi * standard_config.effective_sideband_hz
        ),
        **noise_parameters(sweep_config.noise_case),
    }


def _extract_qpt_rows(
    pulse_type,
    pulse,
    channel,
    error,
    sweep_config,
    run_signature,
    channel_dir,
):
    diagnostics = pulse_diagnostics(pulse_type, pulse)
    ideal_chi = qp.to_chi(qp.to_super(mg.ideal_ms_gate(phi=TARGET_XX_ANGLE)))
    rows = []
    pauli_rows = []
    channel_dir.mkdir(parents=True, exist_ok=True)

    for index, nbar in enumerate(sweep_config.nbar_values):
        chi_qobj = channel["chi_qobj_list"][index]
        superoperator = channel["S_qobj_list"][index]
        chi = _trace_normalize(chi_qobj)
        error_chi = _trace_normalize(error["error_chi_qobj_list"][index])
        error_ptm = np.asarray(error["error_ptm_list"][index], complex)
        process_fidelity = float(np.real(qp.process_fidelity(chi_qobj, ideal_chi)))
        average_fidelity = (4 * process_fidelity + 1) / 5
        probabilities = np.real(np.diag(error_chi)).copy()
        probabilities /= probabilities.sum()
        offdiagonal = chi - np.diag(np.diag(chi))
        error_block = error_chi[1:, 1:]
        error_block_offdiagonal = error_block - np.diag(np.diag(error_block))
        pauli_ptm = np.diag(np.diag(error_ptm))
        physicality = mg.choi_physicality_metrics(superoperator)
        choi = qp.to_choi(superoperator).full()

        rows.append(
            {
                "run_signature": run_signature,
                "pulse_type": pulse_type,
                "noise_case": sweep_config.noise_case,
                "scattering_model": sweep_config.scattering_model,
                "nbar": float(nbar),
                "gate_time_s": diagnostics["gate_time_s"],
                "F_avg": average_fidelity,
                "infidelity": 1 - average_fidelity,
                "p_err": 1 - probabilities[0],
                "p_weight1": float(probabilities[PAULI_WEIGHTS == 1].sum()),
                "p_weight2": float(probabilities[PAULI_WEIGHTS == 2].sum()),
                "offdiag_ratio": _safe_ratio(
                    np.linalg.norm(offdiagonal), np.linalg.norm(chi)
                ),
                "error_offdiag_ratio": _safe_ratio(
                    np.linalg.norm(error_block_offdiagonal), np.linalg.norm(error_chi)
                ),
                "exact_pauli_distance": _safe_ratio(
                    np.linalg.norm(error_ptm - pauli_ptm), np.linalg.norm(error_ptm)
                ),
                "alpha_final_abs": diagnostics["alpha_final_abs"],
                "alpha_max_abs": diagnostics["alpha_max_abs"],
                "alpha2_integral": diagnostics["alpha2_integral"],
                "entangling_angle_error": diagnostics["entangling_angle_error"],
                "integrated_omega2": diagnostics["integrated_omega2"],
                "choi_min_eigenvalue": physicality["min_choi_eigenvalue"],
                "tp_error": physicality["tp_frobenius_error"],
                "hp_error": float(np.linalg.norm(choi - choi.conj().T)),
            }
        )
        pauli_rows.extend(
            {
                "run_signature": run_signature,
                "pulse_type": pulse_type,
                "noise_case": sweep_config.noise_case,
                "scattering_model": sweep_config.scattering_model,
                "nbar": float(nbar),
                "pauli": label,
                "probability": float(probability),
            }
            for label, probability in zip(PAULI_LABELS[1:], probabilities[1:])
        )
        safe_nbar = str(nbar).replace(".", "p")
        np.savez_compressed(
            channel_dir
            / f"{run_signature[:12]}__{pulse_type}__nbar_{safe_nbar}.npz",
            chi=chi,
            error_chi=error_chi,
            error_ptm=error_ptm,
            probabilities=probabilities,
        )
    return pd.DataFrame(rows), pd.DataFrame(pauli_rows)


def _configuration_payload(
    smooth_config,
    standard_config,
    sweep_config,
    calibration,
):
    payload = {
        "smooth_detuning": asdict(smooth_config),
        "standard": asdict(standard_config),
        "sweep": asdict(sweep_config),
        "noise": BASE_NOISE,
        "calibration": calibration,
        "paper_reference_detuning_hz": {
            "delta_max_hz": -400.0e3,
            "delta_min_hz": -21.7e3,
        },
        "simulator_sign_flip": True,
        "hamiltonian_amplitude_mapping": "A = paper Omega_g / 2",
    }
    signature_source = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["run_signature"] = sha256(signature_source.encode("utf-8")).hexdigest()
    return payload


def add_infidelity_comparison(summary):
    """Add standard-minus-smooth comparison columns to long-format results."""
    if summary.empty:
        return summary.copy()
    result = summary.drop(
        columns=["standard_infidelity", "delta_infidelity", "improvement_factor"],
        errors="ignore",
    )
    keys = ["run_signature", "noise_case", "scattering_model", "nbar"]
    standard = result[result.pulse_type == "standard"][
        keys + ["infidelity"]
    ].rename(columns={"infidelity": "standard_infidelity"})
    result = result.merge(standard, on=keys, how="left")
    result["delta_infidelity"] = result.infidelity - result.standard_infidelity
    result["improvement_factor"] = np.where(
        abs(result.infidelity) < 1e-15,
        np.nan,
        result.standard_infidelity / result.infidelity,
    )
    return result


def plot_infidelity_comparison(summary, output_path):
    """Plot standard and smooth infidelity on the same nbar axis."""
    if summary.empty:
        raise ValueError("summary must contain completed QPT results.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for pulse_type in ("standard", "smooth_detuning"):
        group = summary[summary.pulse_type == pulse_type].sort_values("nbar")
        if group.empty:
            continue
        axis.semilogy(
            group.nbar,
            group.infidelity,
            "o-",
            linewidth=2,
            markersize=6,
            label=pulse_type,
        )
    axis.set(
        xlabel=r"Initial mean phonon number $\bar{n}$",
        ylabel=r"Infidelity $1-F_{\mathrm{avg}}$",
        xticks=sorted(summary.nbar.unique()),
    )
    axis.grid(True, which="both", alpha=0.28)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
    return output_path


def plot_analytic_preview(preview, output_path):
    """Plot the fast noise-free model without presenting it as a QPT result."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for pulse_type, group in preview.groupby("pulse_type"):
        group = group.sort_values("nbar")
        axis.semilogy(group.nbar, group.infidelity, "o-", label=pulse_type)
    axis.set(
        xlabel=r"Initial mean phonon number $\bar{n}$",
        ylabel=r"Analytic noise-free infidelity",
        xticks=sorted(preview.nbar.unique()),
        title="First-order preview (not full QPT)",
    )
    axis.grid(True, which="both", alpha=0.28)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
    return output_path


def residual_spin_motion_infidelity(
    nbar_values=DEFAULT_NBAR_VALUES,
    smooth_config: Optional[SmoothDetuningConfigSI] = None,
    standard_config: Optional[StandardGateConfigSI] = None,
):
    """Isolate the first-order error caused by final residual displacement.

    The actual pulse displacement is retained while the geometric phase is set
    exactly to the target value.  This separates final residual spin-motion
    entanglement from entangling-angle error.  It does not include higher-order
    Lamb-Dicke/Debye-Waller effects, which remain part of the noise-free QPT
    floor.
    """
    smooth_config = (
        calibrate_delta_min()[0] if smooth_config is None else smooth_config
    )
    standard_config = StandardGateConfigSI() if standard_config is None else standard_config
    pulses = {
        "standard": build_standard_pulse_si(standard_config, time_points=4001),
        "smooth_detuning": build_smooth_pulse_si(smooth_config, time_points=8001),
    }
    rows = []
    for pulse_type, pulse in pulses.items():
        metrics = lpo.ms_magnus_metrics(
            pulse["time_s"], pulse["A_rad_s"], pulse["delta_rad_s"]
        )
        for nbar in nbar_values:
            residual_channel = lpo.thermal_ms_superoperator(
                displacement=metrics["displacement"],
                geometric_phase=TARGET_XX_ANGLE / 2,
                n_bar=float(nbar),
            )
            residual_fidelity = lpo.average_gate_fidelity_from_superoperator(
                residual_channel,
                target_xx_angle=TARGET_XX_ANGLE,
            )
            angle_only_channel = lpo.thermal_ms_superoperator(
                displacement=0.0,
                geometric_phase=metrics["geometric_phase"],
                n_bar=float(nbar),
            )
            angle_fidelity = lpo.average_gate_fidelity_from_superoperator(
                angle_only_channel,
                target_xx_angle=TARGET_XX_ANGLE,
            )
            rows.append(
                {
                    "pulse_type": pulse_type,
                    "nbar": float(nbar),
                    "alpha_final_abs": float(abs(metrics["displacement"])),
                    "residual_spin_motion_infidelity": float(
                        1.0 - residual_fidelity
                    ),
                    "entangling_angle_infidelity": float(1.0 - angle_fidelity),
                    "model": "first_order_component_isolation",
                }
            )
    return pd.DataFrame(rows)


def _load_compatible_all_noise_summary(source_dir, sweep_config):
    source_dir = Path(source_dir)
    config_path = source_dir / "config_used.yaml"
    summary_path = source_dir / "summary.csv"
    if not config_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Completed all-noise results were not found in {source_dir}."
        )
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    source_sweep = source_config.get("sweep", {})
    comparisons = {
        "nbar_values": list(sweep_config.nbar_values),
        "noise_case": "all_noise",
        "scattering_model": sweep_config.scattering_model,
        "standard_time_points": sweep_config.standard_time_points,
        "smooth_time_points": sweep_config.smooth_time_points,
        "standard_solver_max_step_s": sweep_config.standard_solver_max_step_s,
        "smooth_solver_max_step_s": sweep_config.smooth_solver_max_step_s,
        "use_full_order": sweep_config.use_full_order,
    }
    mismatches = [
        key for key, expected in comparisons.items() if source_sweep.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "The all-noise source is incompatible for: " + ", ".join(mismatches)
        )
    summary = pd.read_csv(summary_path)
    summary = summary[
        (summary.noise_case == "all_noise")
        & (summary.scattering_model == sweep_config.scattering_model)
    ].copy()
    required = {
        (pulse_type, round(float(nbar), 12))
        for pulse_type in ("standard", "smooth_detuning")
        for nbar in sweep_config.nbar_values
    }
    found = {
        (row.pulse_type, round(float(row.nbar), 12))
        for row in summary.itertuples()
    }
    if not required.issubset(found):
        raise ValueError("The all-noise source does not contain all requested conditions.")
    return summary


def build_noise_attribution(ablation_summary):
    """Build a reference-based, exactly summing attribution of smooth advantage.

    The decomposition is not an assertion that Lindblad errors add linearly.
    It explicitly retains the heating/dephasing interaction and places the
    difference between all-noise and all-motional in a spin/optical/cross-term
    remainder.
    """
    required_cases = set(DEFAULT_ABLATION_CASES)
    if ablation_summary.empty:
        return pd.DataFrame()
    rows = []
    for nbar, group in ablation_summary.groupby("nbar"):
        table = group.pivot_table(
            index="noise_case",
            columns="pulse_type",
            values="infidelity",
            aggfunc="first",
        )
        if not required_cases.issubset(table.index):
            continue
        if not {"standard", "smooth_detuning"}.issubset(table.columns):
            continue

        def advantage(case):
            return float(table.loc[case, "standard"] - table.loc[case, "smooth_detuning"])

        coherent_floor = advantage("none")
        heating = advantage("heating_only") - coherent_floor
        motional_dephasing = advantage("motional_dephasing_only") - coherent_floor
        motional_interaction = (
            advantage("all_motional")
            - coherent_floor
            - heating
            - motional_dephasing
        )
        spin_optical_remainder = advantage("all_noise") - advantage("all_motional")
        total = advantage("all_noise")
        rows.extend(
            {
                "nbar": float(nbar),
                "mechanism": mechanism,
                "smooth_advantage": float(value),
                "definition": definition,
            }
            for mechanism, value, definition in (
                (
                    "noise_free_qpt_floor",
                    coherent_floor,
                    "I_standard(none) - I_smooth(none)",
                ),
                (
                    "heating_increment",
                    heating,
                    "difference of heating-only increments above none",
                ),
                (
                    "motional_dephasing_increment",
                    motional_dephasing,
                    "difference of motional-dephasing-only increments above none",
                ),
                (
                    "motional_nonadditive_interaction",
                    motional_interaction,
                    "all-motional remainder after none and single-noise increments",
                ),
                (
                    "spin_optical_and_cross_remainder",
                    spin_optical_remainder,
                    "all-noise advantage minus all-motional advantage",
                ),
                (
                    "total_all_noise_advantage",
                    total,
                    "I_standard(all_noise) - I_smooth(all_noise)",
                ),
            )
        )
    return pd.DataFrame(rows)


def plot_noise_ablation(ablation_summary, output_path):
    """Plot standard/smooth infidelity for every completed ablation case."""
    cases = [
        case
        for case in DEFAULT_ABLATION_CASES
        if case in set(ablation_summary.noise_case)
    ]
    if not cases:
        raise ValueError("No ablation rows are available to plot.")
    columns = 2
    rows = int(np.ceil(len(cases) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(11, 4 * rows), squeeze=False)
    for axis, case in zip(axes.flat, cases):
        data = ablation_summary[ablation_summary.noise_case == case]
        for pulse_type, group in data.groupby("pulse_type"):
            group = group.sort_values("nbar")
            axis.semilogy(group.nbar, group.infidelity, "o-", label=pulse_type)
        axis.set(
            title=case,
            xlabel=r"Initial mean phonon number $\bar{n}$",
            ylabel=r"Infidelity $1-F_{\mathrm{avg}}$",
            xticks=sorted(data.nbar.unique()),
        )
        axis.grid(True, which="both", alpha=0.28)
        axis.legend()
    for axis in axes.flat[len(cases) :]:
        axis.set_visible(False)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
    return output_path


def plot_noise_penalties(ablation_summary, output_path):
    """Plot noise-induced increments relative to each pulse's noise-free QPT."""
    cases = ("heating_only", "motional_dephasing_only", "all_motional")
    noise_free = ablation_summary[ablation_summary.noise_case == "none"]
    if noise_free.empty:
        raise ValueError("Noise-free QPT rows are required for penalty plots.")
    keys = ["pulse_type", "nbar"]
    baseline = noise_free[keys + ["infidelity"]].rename(
        columns={"infidelity": "noise_free_infidelity"}
    )
    selected = ablation_summary[ablation_summary.noise_case.isin(cases)].merge(
        baseline,
        on=keys,
        how="left",
    )
    selected["noise_penalty"] = selected.infidelity - selected.noise_free_infidelity
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for axis, case in zip(axes, cases):
        data = selected[selected.noise_case == case]
        for pulse_type, group in data.groupby("pulse_type"):
            group = group.sort_values("nbar")
            axis.plot(group.nbar, group.noise_penalty, "o-", label=pulse_type)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(title=case, xlabel=r"$\bar{n}$", xticks=sorted(data.nbar.unique()))
        axis.grid(alpha=0.28)
        axis.legend()
    axes[0].set_ylabel("Infidelity increment above noise-free QPT")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
    return output_path


def plot_noise_attribution(attribution, output_path):
    """Plot the signed components that sum to the all-noise smooth advantage."""
    components = (
        "noise_free_qpt_floor",
        "heating_increment",
        "motional_dephasing_increment",
        "motional_nonadditive_interaction",
        "spin_optical_and_cross_remainder",
    )
    table = attribution[attribution.mechanism.isin(components)].pivot_table(
        index="nbar",
        columns="mechanism",
        values="smooth_advantage",
        aggfunc="first",
    )
    if table.empty:
        raise ValueError("Complete ablation results are required for attribution.")
    table = table.reindex(columns=components, fill_value=0.0).sort_index()
    x = np.arange(len(table))
    positive_bottom = np.zeros(len(table))
    negative_bottom = np.zeros(len(table))
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for component in components:
        values = table[component].to_numpy(float)
        bottom = np.where(values >= 0, positive_bottom, negative_bottom)
        axis.bar(x, values, bottom=bottom, label=component)
        positive_bottom += np.where(values >= 0, values, 0.0)
        negative_bottom += np.where(values < 0, values, 0.0)
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set(
        xticks=x,
        xticklabels=[f"{value:g}" for value in table.index],
        xlabel=r"Initial mean phonon number $\bar{n}$",
        ylabel=r"Smooth advantage $I_{std}-I_{smooth}$",
    )
    axis.grid(axis="y", alpha=0.28)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
    return output_path


def plot_residual_spin_motion(residual_summary, output_path):
    """Plot the alpha(T)-only analytic infidelity for both pulse protocols."""
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for pulse_type, group in residual_summary.groupby("pulse_type"):
        group = group.sort_values("nbar")
        axis.semilogy(
            group.nbar,
            group.residual_spin_motion_infidelity,
            "o-",
            label=pulse_type,
        )
    axis.set(
        xlabel=r"Initial mean phonon number $\bar{n}$",
        ylabel="Residual-displacement infidelity",
        xticks=sorted(residual_summary.nbar.unique()),
        title=r"First-order residual spin-motion component ($\alpha(T)$ only)",
    )
    axis.set_yscale("symlog", linthresh=1e-12)
    axis.grid(True, which="both", alpha=0.28)
    axis.legend()
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
    return output_path


def run_noise_ablation(
    output_dir="results/smooth_detuning_noise_ablation",
    sweep_config: Optional[NbarSweepConfig] = None,
    noise_cases=DEFAULT_ABLATION_CASES,
    all_noise_source_dir="results/smooth_detuning_nbar_sweep",
    execute_qpt: bool = True,
    resume: bool = True,
):
    """Run/resume the motional ablations and build a mechanism attribution."""
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    condition_dir = output_dir / "conditions"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    condition_dir.mkdir(parents=True, exist_ok=True)
    sweep_config = NbarSweepConfig() if sweep_config is None else sweep_config
    noise_cases = tuple(noise_cases)
    unsupported = set(noise_cases) - set(DEFAULT_ABLATION_CASES)
    if unsupported:
        raise ValueError(f"Unsupported ablation cases: {sorted(unsupported)}")

    calibrated_smooth, calibration = calibrate_delta_min()
    residual = residual_spin_motion_infidelity(
        sweep_config.nbar_values,
        smooth_config=calibrated_smooth,
    )
    residual.to_csv(output_dir / "residual_spin_motion.csv", index=False)
    plot_residual_spin_motion(
        residual,
        figure_dir / "residual_spin_motion_infidelity.png",
    )

    summaries = []
    for noise_case in noise_cases:
        if noise_case == "all_noise":
            summaries.append(
                _load_compatible_all_noise_summary(
                    all_noise_source_dir,
                    sweep_config,
                )
            )
            continue
        condition_config = replace(sweep_config, noise_case=noise_case)
        result = run_nbar_infidelity_sweep(
            output_dir=condition_dir / noise_case,
            smooth_config=calibrated_smooth,
            sweep_config=condition_config,
            execute_qpt=execute_qpt,
            resume=resume,
        )
        if not result["summary"].empty:
            summaries.append(result["summary"])

    ablation = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if not ablation.empty:
        ablation = ablation.drop_duplicates(
            subset=["pulse_type", "noise_case", "scattering_model", "nbar"],
            keep="last",
        ).sort_values(["noise_case", "pulse_type", "nbar"])
        ablation.to_csv(output_dir / "ablation_summary.csv", index=False)
        plot_noise_ablation(ablation, figure_dir / "noise_ablation.png")

    attribution = build_noise_attribution(ablation)
    if not attribution.empty:
        attribution.to_csv(output_dir / "mechanism_attribution.csv", index=False)
        plot_noise_penalties(ablation, figure_dir / "motional_noise_penalties.png")
        plot_noise_attribution(
            attribution,
            figure_dir / "mechanism_attribution.png",
        )

    config_payload = {
        "nbar_values": list(sweep_config.nbar_values),
        "noise_cases": list(noise_cases),
        "base_sweep": asdict(sweep_config),
        "all_noise_source_dir": str(all_noise_source_dir),
        "calibration": calibration,
        "attribution_caveat": (
            "Reference-based decomposition; nonadditive interactions are explicit, "
            "and the spin/optical remainder is not a pure single-noise contribution."
        ),
    }
    (output_dir / "config_used.yaml").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )
    return {
        "ablation_summary": ablation,
        "attribution": attribution,
        "residual_spin_motion": residual,
        "output_dir": output_dir,
    }


def run_nbar_infidelity_sweep(
    output_dir="results/smooth_detuning_nbar_sweep",
    smooth_config: Optional[SmoothDetuningConfigSI] = None,
    standard_config: Optional[StandardGateConfigSI] = None,
    sweep_config: Optional[NbarSweepConfig] = None,
    execute_qpt: bool = True,
    resume: bool = True,
):
    """Run or resume the five-point nbar comparison and save all outputs.

    Results from a different configuration are never silently reused.  Set
    ``resume=False`` to intentionally replace an earlier configuration in the
    same output directory.
    """
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    channel_dir = output_dir / "channels"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    channel_dir.mkdir(parents=True, exist_ok=True)

    smooth_config = SmoothDetuningConfigSI() if smooth_config is None else smooth_config
    standard_config = StandardGateConfigSI() if standard_config is None else standard_config
    sweep_config = NbarSweepConfig() if sweep_config is None else sweep_config
    validate_smooth_waveform(smooth_config)
    calibrated_smooth, calibration = calibrate_delta_min(smooth_config)
    payload = _configuration_payload(
        calibrated_smooth,
        standard_config,
        sweep_config,
        calibration,
    )
    run_signature = payload["run_signature"]

    config_path = output_dir / "config_used.yaml"
    calibration_path = output_dir / "calibration.json"
    summary_path = output_dir / "summary.csv"
    pauli_path = output_dir / "pauli_probabilities.csv"
    preview_path = output_dir / "analytic_preview.csv"
    existing_config = None
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        resume
        and summary_path.exists()
        and existing_config is not None
        and existing_config.get("run_signature") != run_signature
    ):
        raise ValueError(
            "Existing results use a different configuration. Use resume=False "
            "or choose a new output directory."
        )

    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    preview = analytic_nbar_preview(
        sweep_config.nbar_values,
        smooth_config=calibrated_smooth,
        standard_config=standard_config,
    )
    preview.to_csv(preview_path, index=False)
    plot_analytic_preview(preview, figure_dir / "analytic_preview_infidelity_vs_nbar.png")

    if resume and summary_path.exists():
        summary = pd.read_csv(summary_path)
        pauli = pd.read_csv(pauli_path) if pauli_path.exists() else pd.DataFrame()
    else:
        summary = pd.DataFrame()
        pauli = pd.DataFrame()

    if not execute_qpt:
        if not summary.empty:
            summary = add_infidelity_comparison(summary)
            plot_infidelity_comparison(
                summary,
                figure_dir / "infidelity_vs_nbar.png",
            )
        return {
            "summary": summary,
            "analytic_preview": preview,
            "calibration": calibration,
            "smooth_config": calibrated_smooth,
            "output_dir": output_dir,
            "run_signature": run_signature,
        }

    completed = set()
    if not summary.empty and "run_signature" in summary:
        matching = summary[summary.run_signature == run_signature]
        completed = {
            (row.pulse_type, round(float(row.nbar), 12))
            for row in matching.itertuples()
        }

    for pulse_type in ("standard", "smooth_detuning"):
        pulse, max_step = _pulse_and_solver_settings(
            pulse_type,
            calibrated_smooth,
            standard_config,
            sweep_config,
        )
        for nbar in sweep_config.nbar_values:
            condition_key = (pulse_type, round(float(nbar), 12))
            if condition_key in completed:
                continue
            condition_config = replace(
                sweep_config,
                nbar_values=(float(nbar),),
            )
            print(
                f"Running {pulse_type}: nbar={nbar}, "
                f"noise={sweep_config.noise_case}"
            )
            channel = mg.generate_chi_matrices(
                show_summary=False,
                **_qpt_parameters(
                    pulse,
                    max_step,
                    condition_config,
                    standard_config,
                ),
            )
            error = mg.generate_error_channel_matrices(
                channel_result=channel,
                convention="undo_before_actual",
                phi=TARGET_XX_ANGLE,
                show_summary=False,
            )
            new_summary, new_pauli = _extract_qpt_rows(
                pulse_type,
                pulse,
                channel,
                error,
                condition_config,
                run_signature,
                channel_dir,
            )
            if not summary.empty:
                summary = summary[
                    ~(
                        (summary.run_signature == run_signature)
                        & (summary.pulse_type == pulse_type)
                        & np.isclose(summary.nbar.astype(float), float(nbar))
                    )
                ]
            if not pauli.empty:
                pauli = pauli[
                    ~(
                        (pauli.run_signature == run_signature)
                        & (pauli.pulse_type == pulse_type)
                        & np.isclose(pauli.nbar.astype(float), float(nbar))
                    )
                ]
            summary = pd.concat([summary, new_summary], ignore_index=True)
            pauli = pd.concat([pauli, new_pauli], ignore_index=True)
            summary.to_csv(summary_path, index=False)
            pauli.to_csv(pauli_path, index=False)

    summary = add_infidelity_comparison(summary)
    summary = summary.sort_values(["pulse_type", "nbar"]).reset_index(drop=True)
    summary.to_csv(summary_path, index=False)
    plot_infidelity_comparison(summary, figure_dir / "infidelity_vs_nbar.png")
    return {
        "summary": summary,
        "analytic_preview": preview,
        "calibration": calibration,
        "smooth_config": calibrated_smooth,
        "output_dir": output_dir,
        "run_signature": run_signature,
    }


def build_spin_optical_individual_errors(
    individual_summary,
    baseline_summary,
    noise_cases=SPIN_OPTICAL_INDIVIDUAL_CASES,
):
    """Subtract the noise-free QPT floor from spin/optical-only infidelities.

    ``photon_scattering_only`` means that the existing Rayleigh and Raman
    channels are enabled together.  The returned error is an incremental QPT
    infidelity, not a rate and not an additive decomposition of ``all_noise``.
    """
    required_columns = {"pulse_type", "noise_case", "nbar", "infidelity"}
    individual_summary = pd.DataFrame(individual_summary).copy()
    baseline_summary = pd.DataFrame(baseline_summary).copy()
    for name, table in (
        ("individual_summary", individual_summary),
        ("baseline_summary", baseline_summary),
    ):
        missing_columns = required_columns - set(table.columns)
        if missing_columns:
            raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")

    noise_cases = tuple(noise_cases)
    unsupported = set(noise_cases) - set(SPIN_OPTICAL_INDIVIDUAL_CASES)
    if unsupported:
        raise ValueError(
            f"Unsupported spin/optical individual cases: {sorted(unsupported)}"
        )
    individual_summary = individual_summary[
        individual_summary.noise_case.isin(noise_cases)
    ]
    baseline_summary = baseline_summary[baseline_summary.noise_case == "none"]
    if individual_summary.empty or baseline_summary.empty:
        return pd.DataFrame()

    individual_wide = individual_summary.pivot_table(
        index=["noise_case", "nbar"],
        columns="pulse_type",
        values="infidelity",
        aggfunc="first",
    )
    baseline_wide = baseline_summary.pivot_table(
        index="nbar",
        columns="pulse_type",
        values="infidelity",
        aggfunc="first",
    )
    required_pulses = {"standard", "smooth_detuning"}
    if not required_pulses.issubset(individual_wide.columns):
        raise ValueError("Individual-noise results require both pulse types.")
    if not required_pulses.issubset(baseline_wide.columns):
        raise ValueError("Noise-free baseline requires both pulse types.")

    nbar_values = tuple(sorted(float(value) for value in baseline_wide.index))
    expected = {
        (case, round(float(nbar), 12))
        for case in noise_cases
        for nbar in nbar_values
    }
    found = {
        (str(case), round(float(nbar), 12))
        for case, nbar in individual_wide.index
    }
    missing_conditions = sorted(expected - found)
    if missing_conditions:
        raise ValueError(
            "Individual-noise QPT is incomplete; missing conditions: "
            f"{missing_conditions}"
        )

    individual_wide = individual_wide.rename(
        columns={
            "standard": "standard_noise_only",
            "smooth_detuning": "smooth_detuning_noise_only",
        }
    ).reset_index()
    baseline_wide = baseline_wide.rename(
        columns={
            "standard": "standard_none",
            "smooth_detuning": "smooth_detuning_none",
        }
    ).reset_index()
    errors = individual_wide.merge(
        baseline_wide,
        on="nbar",
        how="left",
        validate="many_to_one",
    )
    errors["standard_individual_error"] = (
        errors.standard_noise_only - errors.standard_none
    )
    errors["smooth_detuning_individual_error"] = (
        errors.smooth_detuning_noise_only - errors.smooth_detuning_none
    )
    errors["smooth_error_reduction"] = (
        errors.standard_individual_error
        - errors.smooth_detuning_individual_error
    )
    denominator = errors.smooth_detuning_individual_error.to_numpy(float)
    numerator = errors.standard_individual_error.to_numpy(float)
    errors["standard_over_smooth_error"] = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=np.abs(denominator) > 1e-15,
    )
    case_order = {case: index for index, case in enumerate(noise_cases)}
    errors["_case_order"] = errors.noise_case.map(case_order)
    errors = errors.sort_values(["_case_order", "nbar"]).drop(
        columns="_case_order"
    )
    return errors.reset_index(drop=True)


def plot_spin_optical_individual_errors(individual_errors, output_path):
    """Plot noise-only QPT increments above the matching no-noise floor."""
    individual_errors = pd.DataFrame(individual_errors)
    if individual_errors.empty:
        raise ValueError("No spin/optical individual-error rows are available.")
    cases = [
        case
        for case in SPIN_OPTICAL_INDIVIDUAL_CASES
        if case in set(individual_errors.noise_case)
    ]
    figure, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4.2))
    axes = np.atleast_1d(axes)
    for axis, case in zip(axes, cases):
        data = individual_errors[individual_errors.noise_case == case].sort_values(
            "nbar"
        )
        axis.plot(
            data.nbar,
            data.standard_individual_error,
            "o-",
            color="tab:orange",
            label="standard",
        )
        axis.plot(
            data.nbar,
            data.smooth_detuning_individual_error,
            "o-",
            color="tab:blue",
            label="smooth_detuning",
        )
        plotted = data[
            ["standard_individual_error", "smooth_detuning_individual_error"]
        ].to_numpy(float)
        if np.all(plotted > 0):
            axis.set_yscale("log")
        else:
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        title = (
            "spin dephasing only"
            if case == "spin_dephasing_only"
            else "photon scattering only\n(Rayleigh + Raman)"
        )
        axis.set(
            title=title,
            xlabel=r"Initial mean phonon number $\bar{n}$",
            ylabel=r"Individual error $I_{\mathrm{only}}-I_{\mathrm{none}}$",
            xticks=sorted(data.nbar.unique()),
        )
        axis.grid(True, which="both", alpha=0.28)
        axis.legend()
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
    return output_path


def plot_spin_optical_noise_only_infidelity(individual_summary, output_path):
    """Plot the raw QPT Infidelity for each spin/optical-only condition."""
    individual_summary = pd.DataFrame(individual_summary).copy()
    required_columns = {"pulse_type", "noise_case", "nbar", "infidelity"}
    missing_columns = required_columns - set(individual_summary.columns)
    if missing_columns:
        raise ValueError(
            "individual_summary is missing columns: "
            f"{sorted(missing_columns)}"
        )
    individual_summary = individual_summary[
        individual_summary.noise_case.isin(SPIN_OPTICAL_INDIVIDUAL_CASES)
    ]
    if individual_summary.empty:
        raise ValueError("No spin/optical noise-only QPT rows are available.")

    cases = [
        case
        for case in SPIN_OPTICAL_INDIVIDUAL_CASES
        if case in set(individual_summary.noise_case)
    ]
    figure, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4.2))
    axes = np.atleast_1d(axes)
    for axis, case in zip(axes, cases):
        data = individual_summary[individual_summary.noise_case == case]
        for pulse_type in ("standard", "smooth_detuning"):
            pulse_data = data[data.pulse_type == pulse_type].sort_values("nbar")
            if pulse_data.empty:
                continue
            axis.semilogy(
                pulse_data.nbar,
                pulse_data.infidelity,
                "o-",
                color={
                    "standard": "tab:orange",
                    "smooth_detuning": "tab:blue",
                }[pulse_type],
                label=pulse_type,
            )
        title = (
            "spin dephasing only"
            if case == "spin_dephasing_only"
            else "photon scattering only\n(Rayleigh + Raman)"
        )
        axis.set(
            title=title,
            xlabel=r"Initial mean phonon number $\bar{n}$",
            ylabel=r"Infidelity $1-F_{\mathrm{avg}}$",
            xticks=sorted(data.nbar.unique()),
        )
        axis.grid(True, which="both", alpha=0.28)
        axis.legend()
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
    return output_path


def run_spin_optical_individual_ablation(
    output_dir="results/smooth_detuning_noise_ablation",
    baseline_summary=None,
    sweep_config: Optional[NbarSweepConfig] = None,
    noise_cases=SPIN_OPTICAL_INDIVIDUAL_CASES,
    execute_qpt: bool = True,
    resume: bool = True,
):
    """Run/resume spin-dephasing and total-photon-scattering QPT sweeps."""
    output_dir = Path(output_dir)
    condition_dir = output_dir / "conditions"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    sweep_config = NbarSweepConfig() if sweep_config is None else sweep_config
    noise_cases = tuple(noise_cases)
    unsupported = set(noise_cases) - set(SPIN_OPTICAL_INDIVIDUAL_CASES)
    if unsupported:
        raise ValueError(
            f"Unsupported spin/optical individual cases: {sorted(unsupported)}"
        )

    summaries = []
    for noise_case in noise_cases:
        condition_config = replace(sweep_config, noise_case=noise_case)
        result = run_nbar_infidelity_sweep(
            output_dir=condition_dir / noise_case,
            sweep_config=condition_config,
            execute_qpt=execute_qpt,
            resume=resume,
        )
        summary = result["summary"]
        if not summary.empty:
            summaries.append(summary)
    individual_summary = (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    )
    individual_summary.to_csv(
        output_dir / "spin_optical_individual_summary.csv",
        index=False,
    )

    if baseline_summary is None:
        baseline_path = condition_dir / "none" / "summary.csv"
        baseline_summary = (
            pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
        )
    individual_errors = pd.DataFrame()
    if not individual_summary.empty and not pd.DataFrame(baseline_summary).empty:
        try:
            individual_errors = build_spin_optical_individual_errors(
                individual_summary,
                baseline_summary,
                noise_cases=noise_cases,
            )
        except ValueError:
            if execute_qpt:
                raise
    individual_errors.to_csv(
        output_dir / "spin_optical_individual_errors.csv",
        index=False,
    )
    if not individual_errors.empty:
        plot_spin_optical_individual_errors(
            individual_errors,
            figure_dir / "spin_optical_individual_errors.png",
        )

    metadata = {
        "noise_cases": list(noise_cases),
        "nbar_values": list(sweep_config.nbar_values),
        "scattering_model": sweep_config.scattering_model,
        "spin_dephasing_T2_star_s": BASE_NOISE["T2_star"],
        "rayleigh_rate_per_s": BASE_NOISE["rayleigh_rate_phys"],
        "raman_rate_per_s": BASE_NOISE["raman_rate_phys"],
        "individual_error_definition": "I(noise_only) - I(none)",
        "caveat": (
            "Single-noise increments do not include cross terms and need not sum "
            "to the all-noise infidelity."
        ),
    }
    (output_dir / "spin_optical_individual_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return {
        "individual_summary": individual_summary,
        "individual_errors": individual_errors,
        "output_dir": output_dir,
    }


__all__ = [
    "BASE_NOISE",
    "DEFAULT_ABLATION_CASES",
    "DEFAULT_NBAR_VALUES",
    "SPIN_OPTICAL_INDIVIDUAL_CASES",
    "NbarSweepConfig",
    "SmoothDetuningConfigSI",
    "StandardGateConfigSI",
    "add_infidelity_comparison",
    "analytic_nbar_preview",
    "build_noise_attribution",
    "build_spin_optical_individual_errors",
    "build_smooth_pulse_si",
    "build_standard_pulse_si",
    "calibrate_delta_min",
    "integrated_phase",
    "noise_parameters",
    "plot_analytic_preview",
    "plot_infidelity_comparison",
    "plot_noise_ablation",
    "plot_noise_attribution",
    "plot_noise_penalties",
    "plot_residual_spin_motion",
    "plot_spin_optical_individual_errors",
    "plot_spin_optical_noise_only_infidelity",
    "pulse_diagnostics",
    "residual_spin_motion_infidelity",
    "run_noise_ablation",
    "run_nbar_infidelity_sweep",
    "run_spin_optical_individual_ablation",
    "validate_smooth_waveform",
]
