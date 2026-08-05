"""Laser-amplitude pulse optimization and chi-matrix impact analysis for MS gates.

The optimizer uses the exact Magnus solution of the first-order, single-mode MS
Hamiltonian as a fast search model.  Candidate pulses are then verified with the
full spin-motion/QPT machinery in :mod:`ms_gate_functions`, including the
configured Lamb-Dicke correction and Lindblad noise sources.

All pulse times and angular frequencies use the simulator's dimensionless units.
``t_gate_phys`` remains the conversion used for physical decoherence rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import qutip as qp
from scipy.linalg import expm
from scipy.optimize import differential_evolution, minimize

import ms_gate_functions as mg


# This module only saves figures; a non-interactive backend also makes command-
# line and multiprocessing verification reliable on macOS.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LaserPulseSearchConfig:
    """Configuration for a symmetric laser-amplitude pulse search."""

    duration: float
    detuning: float
    initial_amplitude: float
    amplitude_max: float
    control_points: int = 9
    zero_endpoints: bool = False
    time_points: int = 501
    n_bar_values: tuple[float, ...] = (0.01, 1.0, 4.0)
    detuning_offsets: tuple[float, ...] = (0.0,)
    intensity_scales: tuple[float, ...] = (1.0,)
    target_xx_angle: float = np.pi / 4
    nominal_weight: float = 200.0
    worst_case_weight: float = 0.5
    smoothness_weight: float = 1e-5
    power_weight: float = 0.0
    maxiter: int = 80
    popsize: int = 10
    seed: int = 1234
    polish: bool = True

    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError("duration must be positive.")
        if self.detuning == 0:
            raise ValueError("detuning must be non-zero.")
        if self.initial_amplitude < 0:
            raise ValueError("initial_amplitude must be non-negative.")
        if self.amplitude_max <= 0:
            raise ValueError("amplitude_max must be positive.")
        if self.initial_amplitude > self.amplitude_max:
            raise ValueError("initial_amplitude cannot exceed amplitude_max.")
        if self.control_points < 5 or self.control_points % 2 == 0:
            raise ValueError("control_points must be an odd integer of at least 5.")
        if self.time_points < self.control_points:
            raise ValueError("time_points must be at least control_points.")
        if not self.n_bar_values:
            raise ValueError("n_bar_values must not be empty.")
        if any(n_bar < 0 for n_bar in self.n_bar_values):
            raise ValueError("n_bar_values must be non-negative.")
        if not self.detuning_offsets:
            raise ValueError("detuning_offsets must not be empty.")
        if not self.intensity_scales:
            raise ValueError("intensity_scales must not be empty.")
        if any(scale <= 0 for scale in self.intensity_scales):
            raise ValueError("intensity_scales must be positive.")
        if self.nominal_weight < 0 or self.worst_case_weight < 0:
            raise ValueError("fidelity weights must be non-negative.")
        if self.smoothness_weight < 0 or self.power_weight < 0:
            raise ValueError("regularization weights must be non-negative.")
        if self.maxiter < 1 or self.popsize < 1:
            raise ValueError("maxiter and popsize must be positive.")


def _as_time_grid(time):
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("time must be a one-dimensional array with at least two points.")
    if not np.all(np.diff(time) > 0):
        raise ValueError("time must be strictly increasing.")
    return time


def _as_waveform(name, values, time):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full(time.shape, float(values), dtype=float)
    if values.shape != time.shape:
        raise ValueError(f"{name} must have shape {time.shape}; got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _cumulative_trapezoid(values, time):
    result = np.zeros_like(time, dtype=float)
    result[1:] = np.cumsum(
        0.5 * (values[1:] + values[:-1]) * np.diff(time)
    )
    return result


def build_symmetric_laser_amplitude(
    free_amplitudes,
    time,
    control_points,
    zero_endpoints=False,
):
    """Build a smooth-ended, time-symmetric amplitude waveform.

    ``free_amplitudes`` specifies the first half of the nodes including the
    centre node.  The pulse is mirrored around the gate midpoint and linearly
    interpolated on ``time``.  Set ``zero_endpoints=True`` when the simulated
    gate window must explicitly include optical turn-on and turn-off ramps.
    """
    time = _as_time_grid(time)
    if control_points < 5 or control_points % 2 == 0:
        raise ValueError("control_points must be an odd integer of at least 5.")

    free_amplitudes = np.asarray(free_amplitudes, dtype=float)
    if zero_endpoints:
        expected_free = (control_points - 1) // 2
    else:
        expected_free = (control_points + 1) // 2
    if free_amplitudes.shape != (expected_free,):
        raise ValueError(
            f"free_amplitudes must have shape ({expected_free},); "
            f"got {free_amplitudes.shape}."
        )

    if zero_endpoints:
        node_amplitudes = np.concatenate(
            ([0.0], free_amplitudes, free_amplitudes[-2::-1], [0.0])
        )
    else:
        node_amplitudes = np.concatenate(
            (free_amplitudes, free_amplitudes[-2::-1])
        )
    node_times = np.linspace(time[0], time[-1], control_points)
    amplitude = np.interp(time, node_times, node_amplitudes)
    return amplitude, node_times, node_amplitudes


def ms_magnus_metrics(time, amplitude, detuning):
    """Return residual displacement and geometric phase for an MS pulse.

    The Hamiltonian convention is the one used in ``ms_gate_functions``:

        H = A(t) S (a exp(-i Phi(t)) + a^dag exp(i Phi(t))).

    The ideal gate ``exp(+i pi XX / 4)`` therefore corresponds to a geometric
    phase of ``pi/8`` multiplying ``S^2``.
    """
    time = _as_time_grid(time)
    amplitude = _as_waveform("amplitude", amplitude, time)
    detuning = _as_waveform("detuning", detuning, time)

    phase = _cumulative_trapezoid(detuning, time)
    exp_phase = np.exp(1j * phase)
    displacement = -1j * np.trapz(amplitude * exp_phase, time)

    cumulative_cos = _cumulative_trapezoid(amplitude * np.cos(phase), time)
    cumulative_sin = _cumulative_trapezoid(amplitude * np.sin(phase), time)
    geometric_integrand = amplitude * (
        np.sin(phase) * cumulative_cos - np.cos(phase) * cumulative_sin
    )
    geometric_phase = float(np.trapz(geometric_integrand, time))

    return {
        "displacement": displacement,
        "displacement_abs": float(abs(displacement)),
        "geometric_phase": geometric_phase,
        "xx_angle": 2.0 * geometric_phase,
        "phase": phase,
    }


def _collective_spin_x():
    identity = np.eye(2, dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    return np.kron(sigma_x, identity) + np.kron(identity, sigma_x)


def _ideal_xx_unitary(target_xx_angle):
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    return expm(1j * float(target_xx_angle) * np.kron(sigma_x, sigma_x))


def thermal_ms_superoperator(
    displacement,
    geometric_phase,
    n_bar,
):
    """Return the reduced two-qubit superoperator after tracing thermal motion."""
    if n_bar < 0:
        raise ValueError("n_bar must be non-negative.")

    collective_spin = _collective_spin_x()
    spin_values, spin_vectors = np.linalg.eigh(collective_spin)
    delta_spin = spin_values[:, None] - spin_values[None, :]
    delta_spin_squared = spin_values[:, None] ** 2 - spin_values[None, :] ** 2
    thermal_factor = np.exp(
        -0.5
        * (2.0 * float(n_bar) + 1.0)
        * abs(displacement) ** 2
        * delta_spin**2
    )
    phase_factor = np.exp(1j * float(geometric_phase) * delta_spin_squared)
    eigenbasis_multiplier = thermal_factor * phase_factor

    dimension = 4
    superoperator = np.zeros((dimension**2, dimension**2), dtype=complex)
    for col in range(dimension):
        for row in range(dimension):
            basis_operator = np.zeros((dimension, dimension), dtype=complex)
            basis_operator[row, col] = 1.0
            in_eigenbasis = spin_vectors.conj().T @ basis_operator @ spin_vectors
            out_eigenbasis = eigenbasis_multiplier * in_eigenbasis
            output_operator = spin_vectors @ out_eigenbasis @ spin_vectors.conj().T
            vector_index = row + col * dimension
            superoperator[:, vector_index] = output_operator.reshape(-1, order="F")
    return superoperator


def average_gate_fidelity_from_superoperator(
    superoperator,
    target_xx_angle=np.pi / 4,
):
    """Return average gate fidelity relative to ``exp(i angle XX)``."""
    superoperator = np.asarray(superoperator, dtype=complex)
    if superoperator.shape != (16, 16):
        raise ValueError("superoperator must have shape (16, 16).")
    ideal_unitary = _ideal_xx_unitary(target_xx_angle)
    ideal_superoperator = np.kron(ideal_unitary.conj(), ideal_unitary)
    dimension = 4
    entanglement_fidelity = np.trace(
        ideal_superoperator.conj().T @ superoperator
    ) / dimension**2
    average_fidelity = (
        dimension * float(np.real(entanglement_fidelity)) + 1.0
    ) / (dimension + 1.0)
    return float(np.clip(average_fidelity, 0.0, 1.0))


def analytic_ms_average_gate_fidelity(
    time,
    amplitude,
    detuning,
    n_bar,
    target_xx_angle=np.pi / 4,
):
    """Fast first-order MS average gate fidelity for a thermal motional state."""
    metrics = ms_magnus_metrics(time, amplitude, detuning)
    superoperator = thermal_ms_superoperator(
        metrics["displacement"],
        metrics["geometric_phase"],
        n_bar,
    )
    return average_gate_fidelity_from_superoperator(
        superoperator,
        target_xx_angle=target_xx_angle,
    )


def evaluate_pulse_robustness(
    time,
    amplitude,
    detuning,
    n_bar_values,
    detuning_offsets=(0.0,),
    intensity_scales=(1.0,),
    target_xx_angle=np.pi / 4,
):
    """Evaluate analytic fidelity over temperature and quasi-static errors."""
    time = _as_time_grid(time)
    amplitude = _as_waveform("amplitude", amplitude, time)
    detuning = _as_waveform("detuning", detuning, time)

    rows = []
    for n_bar in n_bar_values:
        for detuning_offset in detuning_offsets:
            for intensity_scale in intensity_scales:
                fidelity = analytic_ms_average_gate_fidelity(
                    time,
                    amplitude * float(intensity_scale),
                    detuning + float(detuning_offset),
                    n_bar=float(n_bar),
                    target_xx_angle=target_xx_angle,
                )
                rows.append(
                    {
                        "n_bar": float(n_bar),
                        "detuning_offset": float(detuning_offset),
                        "intensity_scale": float(intensity_scale),
                        "fidelity": fidelity,
                        "infidelity": 1.0 - fidelity,
                    }
                )
    return pd.DataFrame(rows)


def _pulse_objective(free_amplitudes, config, time, return_details=False):
    amplitude, node_times, node_amplitudes = build_symmetric_laser_amplitude(
        free_amplitudes,
        time,
        config.control_points,
        zero_endpoints=config.zero_endpoints,
    )
    robustness = evaluate_pulse_robustness(
        time,
        amplitude,
        config.detuning,
        config.n_bar_values,
        detuning_offsets=config.detuning_offsets,
        intensity_scales=config.intensity_scales,
        target_xx_angle=config.target_xx_angle,
    )
    mean_infidelity = float(robustness["infidelity"].mean())
    worst_infidelity = float(robustness["infidelity"].max())
    nominal_mask = np.isclose(robustness["detuning_offset"], 0.0) & np.isclose(
        robustness["intensity_scale"],
        1.0,
    )
    if nominal_mask.any():
        nominal_infidelity = float(
            robustness.loc[nominal_mask, "infidelity"].mean()
        )
    else:
        nominal_infidelity = mean_infidelity

    amplitude_scale = max(config.amplitude_max, 1e-15)
    second_difference = np.diff(node_amplitudes, n=2)
    smoothness_penalty = float(
        np.mean((second_difference / amplitude_scale) ** 2)
    )
    power_penalty = float(
        np.trapz((amplitude / amplitude_scale) ** 2, time)
        / (time[-1] - time[0])
    )
    cost = (
        mean_infidelity
        + config.nominal_weight * nominal_infidelity
        + config.worst_case_weight * worst_infidelity
        + config.smoothness_weight * smoothness_penalty
        + config.power_weight * power_penalty
    )

    if return_details:
        return {
            "cost": cost,
            "mean_infidelity": mean_infidelity,
            "nominal_infidelity": nominal_infidelity,
            "worst_infidelity": worst_infidelity,
            "smoothness_penalty": smoothness_penalty,
            "power_penalty": power_penalty,
            "time": time,
            "amplitude": amplitude,
            "node_times": node_times,
            "node_amplitudes": node_amplitudes,
            "robustness": robustness,
            "magnus": ms_magnus_metrics(time, amplitude, config.detuning),
        }
    return cost


def optimize_laser_amplitude_pulse(config):
    """Optimize a symmetric laser-amplitude pulse for robust gate fidelity."""
    if not isinstance(config, LaserPulseSearchConfig):
        raise TypeError("config must be a LaserPulseSearchConfig.")

    time = np.linspace(0.0, config.duration, config.time_points)
    if config.zero_endpoints:
        free_count = (config.control_points - 1) // 2
    else:
        free_count = (config.control_points + 1) // 2
    bounds = [(0.0, config.amplitude_max)] * free_count
    initial_parameters = np.full(free_count, config.initial_amplitude)

    global_result = differential_evolution(
        lambda values: _pulse_objective(values, config, time),
        bounds=bounds,
        seed=config.seed,
        maxiter=config.maxiter,
        popsize=config.popsize,
        polish=False,
        workers=1,
        updating="immediate",
        x0=initial_parameters,
    )

    final_result = global_result
    if config.polish:
        local_result = minimize(
            lambda values: _pulse_objective(values, config, time),
            x0=global_result.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max(50, config.maxiter)},
        )
        if local_result.fun <= global_result.fun:
            final_result = local_result

    optimized = _pulse_objective(
        final_result.x,
        config,
        time,
        return_details=True,
    )
    baseline_amplitude = np.full_like(time, config.initial_amplitude)
    baseline_robustness = evaluate_pulse_robustness(
        time,
        baseline_amplitude,
        config.detuning,
        config.n_bar_values,
        detuning_offsets=config.detuning_offsets,
        intensity_scales=config.intensity_scales,
        target_xx_angle=config.target_xx_angle,
    )
    baseline_magnus = ms_magnus_metrics(
        time,
        baseline_amplitude,
        config.detuning,
    )

    optimized_robustness = optimized["robustness"].copy()
    baseline_robustness = baseline_robustness.copy()
    optimized_robustness["pulse"] = "optimized"
    baseline_robustness["pulse"] = "baseline"
    robustness = pd.concat(
        [baseline_robustness, optimized_robustness],
        ignore_index=True,
    )

    summary = pd.DataFrame(
        [
            {
                "pulse": "baseline",
                "mean_infidelity": baseline_robustness["infidelity"].mean(),
                "nominal_infidelity": baseline_robustness.loc[
                    np.isclose(baseline_robustness["detuning_offset"], 0.0)
                    & np.isclose(baseline_robustness["intensity_scale"], 1.0),
                    "infidelity",
                ].mean(),
                "worst_infidelity": baseline_robustness["infidelity"].max(),
                "nominal_displacement_abs": baseline_magnus["displacement_abs"],
                "nominal_geometric_phase": baseline_magnus["geometric_phase"],
                "nominal_xx_angle": baseline_magnus["xx_angle"],
                "peak_amplitude": float(np.max(np.abs(baseline_amplitude))),
                "rms_amplitude": float(np.sqrt(np.mean(baseline_amplitude**2))),
            },
            {
                "pulse": "optimized",
                "mean_infidelity": optimized["mean_infidelity"],
                "nominal_infidelity": optimized["nominal_infidelity"],
                "worst_infidelity": optimized["worst_infidelity"],
                "nominal_displacement_abs": optimized["magnus"][
                    "displacement_abs"
                ],
                "nominal_geometric_phase": optimized["magnus"][
                    "geometric_phase"
                ],
                "nominal_xx_angle": optimized["magnus"]["xx_angle"],
                "peak_amplitude": float(np.max(np.abs(optimized["amplitude"]))),
                "rms_amplitude": float(
                    np.sqrt(np.mean(optimized["amplitude"] ** 2))
                ),
            },
        ]
    )

    pulse_table = pd.DataFrame(
        {
            "time": time,
            "baseline_amplitude": baseline_amplitude,
            "optimized_amplitude": optimized["amplitude"],
            "detuning": np.full_like(time, config.detuning),
        }
    )

    return {
        "config": config,
        "optimizer_success": bool(final_result.success),
        "optimizer_message": str(final_result.message),
        "optimizer_cost": float(final_result.fun),
        "optimizer_parameters": np.asarray(final_result.x, dtype=float),
        "time": time,
        "baseline_amplitude": baseline_amplitude,
        "optimized_amplitude": optimized["amplitude"],
        "control_node_times": optimized["node_times"],
        "control_node_amplitudes": optimized["node_amplitudes"],
        "baseline_magnus": baseline_magnus,
        "optimized_magnus": optimized["magnus"],
        "summary": summary,
        "robustness": robustness,
        "pulse_table": pulse_table,
        "scipy_result": final_result,
    }


def _full_qpt_fidelities(channel_result, target_xx_angle=np.pi / 4):
    ideal_unitary = mg.ideal_ms_gate(phi=target_xx_angle)
    ideal_chi = qp.to_chi(qp.to_super(ideal_unitary))
    dimension = 4
    rows = []
    for data, chi in zip(
        channel_result["results_list"],
        channel_result["chi_qobj_list"],
    ):
        process_fidelity = float(np.real(qp.process_fidelity(chi, ideal_chi)))
        average_fidelity = (
            dimension * process_fidelity + 1.0
        ) / (dimension + 1.0)
        rows.append(
            {
                "n_bar": float(data["n_bar"]),
                "process_fidelity": process_fidelity,
                "fidelity": average_fidelity,
                "infidelity": 1.0 - average_fidelity,
            }
        )
    return pd.DataFrame(rows)


def _trace_normalize(matrix):
    matrix = matrix.full() if hasattr(matrix, "full") else np.asarray(matrix)
    trace = np.trace(matrix)
    if abs(trace) < 1e-15:
        raise ValueError("chi trace is too close to zero for normalization.")
    return np.asarray(matrix, dtype=complex) / trace


def _offdiagonal_norm(matrix):
    return float(np.linalg.norm(matrix - np.diag(np.diag(matrix)), ord="fro"))


def _lookup_by_n_bar(mapping, n_bar):
    for key, value in mapping.items():
        if np.isclose(float(key), float(n_bar), atol=1e-12, rtol=0.0):
            return value
    raise KeyError(f"n_bar={n_bar} was not found.")


def _top_chi_changes(
    baseline_chi,
    optimized_chi,
    n_bar,
    channel,
    top_k,
):
    labels = [label for label, _ in mg.pauli_labels_and_weights()]
    difference = optimized_chi - baseline_chi
    flat_order = np.argsort(np.abs(difference).ravel())[::-1][:top_k]
    rows = []
    for flat_index in flat_order:
        row, col = np.unravel_index(flat_index, difference.shape)
        value = difference[row, col]
        rows.append(
            {
                "n_bar": float(n_bar),
                "channel": channel,
                "row": row,
                "col": col,
                "row_pauli": labels[row],
                "col_pauli": labels[col],
                "delta_real": float(np.real(value)),
                "delta_imag": float(np.imag(value)),
                "delta_abs": float(abs(value)),
                "baseline_abs": float(abs(baseline_chi[row, col])),
                "optimized_abs": float(abs(optimized_chi[row, col])),
            }
        )
    return rows


def verify_optimized_pulse_with_full_qpt(
    optimization_result,
    simulation_parameters,
    output_dir=None,
    top_k=20,
):
    """Run baseline/optimized full QPT and compare exact and error chi matrices."""
    config = optimization_result["config"]
    time = np.asarray(optimization_result["time"], dtype=float)
    baseline_amplitude = np.asarray(
        optimization_result["baseline_amplitude"], dtype=float
    )
    optimized_amplitude = np.asarray(
        optimization_result["optimized_amplitude"], dtype=float
    )

    parameters = dict(simulation_parameters)
    allowed_simulation_keys = set(
        mg.run_ms_gate_simulation.__code__.co_varnames[
            : mg.run_ms_gate_simulation.__code__.co_argcount
        ]
    )
    parameters = {
        key: value
        for key, value in parameters.items()
        if key in allowed_simulation_keys and key != "A"
    }
    parameters.update(
        {
            "delta": config.detuning,
            "rho0": parameters.get("rho0", 0.0),
            "time_points": len(time),
            "t_gate_sim": config.duration,
            "show_progress": parameters.get("show_progress", False),
            "parallel_workers": parameters.get("parallel_workers", 1),
        }
    )

    if parameters.get("laser_scattering_scales_with_intensity", False):
        parameters.setdefault(
            "scattering_reference_amplitude",
            config.initial_amplitude,
        )

    baseline_result = mg.generate_chi_matrices(
        A=baseline_amplitude,
        show_summary=False,
        **parameters,
    )
    optimized_result = mg.generate_chi_matrices(
        A=optimized_amplitude,
        show_summary=False,
        **parameters,
    )
    baseline_error = mg.generate_error_channel_matrices(
        channel_result=baseline_result,
        phi=config.target_xx_angle,
        convention="undo_before_actual",
        show_summary=False,
    )
    optimized_error = mg.generate_error_channel_matrices(
        channel_result=optimized_result,
        phi=config.target_xx_angle,
        convention="undo_before_actual",
        show_summary=False,
    )

    baseline_fidelity = _full_qpt_fidelities(
        baseline_result,
        target_xx_angle=config.target_xx_angle,
    ).set_index("n_bar")
    optimized_fidelity = _full_qpt_fidelities(
        optimized_result,
        target_xx_angle=config.target_xx_angle,
    ).set_index("n_bar")

    summary_rows = []
    top_change_rows = []
    chi_data = {}
    for n_bar in baseline_result["parameters"]["n_bar_list"]:
        baseline_exact_chi = _trace_normalize(
            _lookup_by_n_bar(baseline_result["chi_by_n_bar"], n_bar)
        )
        optimized_exact_chi = _trace_normalize(
            _lookup_by_n_bar(optimized_result["chi_by_n_bar"], n_bar)
        )
        baseline_error_chi = _trace_normalize(
            _lookup_by_n_bar(baseline_error["error_chi_by_n_bar"], n_bar)
        )
        optimized_error_chi = _trace_normalize(
            _lookup_by_n_bar(optimized_error["error_chi_by_n_bar"], n_bar)
        )

        baseline_row = baseline_fidelity.loc[float(n_bar)]
        optimized_row = optimized_fidelity.loc[float(n_bar)]
        summary_rows.append(
            {
                "n_bar": float(n_bar),
                "baseline_fidelity": float(baseline_row["fidelity"]),
                "optimized_fidelity": float(optimized_row["fidelity"]),
                "baseline_infidelity": float(baseline_row["infidelity"]),
                "optimized_infidelity": float(optimized_row["infidelity"]),
                "infidelity_improvement": float(
                    baseline_row["infidelity"] - optimized_row["infidelity"]
                ),
                "exact_chi_difference_fro": float(
                    np.linalg.norm(
                        optimized_exact_chi - baseline_exact_chi,
                        ord="fro",
                    )
                ),
                "error_chi_difference_fro": float(
                    np.linalg.norm(
                        optimized_error_chi - baseline_error_chi,
                        ord="fro",
                    )
                ),
                "baseline_error_chi_offdiag_norm": _offdiagonal_norm(
                    baseline_error_chi
                ),
                "optimized_error_chi_offdiag_norm": _offdiagonal_norm(
                    optimized_error_chi
                ),
                "baseline_pauli_error_probability": float(
                    1.0 - np.real(baseline_error_chi[0, 0])
                ),
                "optimized_pauli_error_probability": float(
                    1.0 - np.real(optimized_error_chi[0, 0])
                ),
            }
        )
        top_change_rows.extend(
            _top_chi_changes(
                baseline_exact_chi,
                optimized_exact_chi,
                n_bar,
                "exact",
                top_k,
            )
        )
        top_change_rows.extend(
            _top_chi_changes(
                baseline_error_chi,
                optimized_error_chi,
                n_bar,
                "post_gate_error",
                top_k,
            )
        )
        chi_data[float(n_bar)] = {
            "baseline_exact": baseline_exact_chi,
            "optimized_exact": optimized_exact_chi,
            "delta_exact": optimized_exact_chi - baseline_exact_chi,
            "baseline_error": baseline_error_chi,
            "optimized_error": optimized_error_chi,
            "delta_error": optimized_error_chi - baseline_error_chi,
        }

    summary = pd.DataFrame(summary_rows)
    top_changes = pd.DataFrame(top_change_rows)
    verification = {
        "optimization_result": optimization_result,
        "baseline_result": baseline_result,
        "optimized_result": optimized_result,
        "baseline_error_result": baseline_error,
        "optimized_error_result": optimized_error,
        "summary": summary,
        "top_chi_changes": top_changes,
        "chi_by_n_bar": chi_data,
    }

    if output_dir is not None:
        save_laser_pulse_optimization_outputs(verification, output_dir)
    return verification


def _safe_value(value):
    return str(value).replace("-", "m").replace(".", "p")


def save_laser_pulse_optimization_outputs(verification, output_dir):
    """Save pulse, fidelity, and chi-difference tables/figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimization = verification["optimization_result"]

    optimization["pulse_table"].to_csv(
        output_dir / "laser_pulse_waveforms.csv",
        index=False,
    )
    optimization["summary"].to_csv(
        output_dir / "analytic_fidelity_summary.csv",
        index=False,
    )
    optimization["robustness"].to_csv(
        output_dir / "analytic_robustness_sweep.csv",
        index=False,
    )
    verification["summary"].to_csv(
        output_dir / "full_qpt_fidelity_chi_summary.csv",
        index=False,
    )
    verification["top_chi_changes"].to_csv(
        output_dir / "top_chi_matrix_changes.csv",
        index=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(
        optimization["time"],
        optimization["baseline_amplitude"],
        label="Baseline",
    )
    axes[0].plot(
        optimization["time"],
        optimization["optimized_amplitude"],
        label="Optimized",
    )
    axes[0].scatter(
        optimization["control_node_times"],
        optimization["control_node_amplitudes"],
        s=18,
        color="black",
        zorder=3,
        label="Control nodes",
    )
    axes[0].set_xlabel("Simulation time")
    axes[0].set_ylabel("Sideband coupling A(t)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    summary = verification["summary"]
    axes[1].semilogy(
        summary["n_bar"],
        summary["baseline_infidelity"],
        "o-",
        label="Baseline",
    )
    axes[1].semilogy(
        summary["n_bar"],
        summary["optimized_infidelity"],
        "o-",
        label="Optimized",
    )
    axes[1].set_xlabel(r"Mean phonon number $\bar{n}$")
    axes[1].set_ylabel(r"Full-QPT infidelity $1-F_{avg}$")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "laser_pulse_and_fidelity_comparison.png", dpi=250)
    fig.savefig(output_dir / "laser_pulse_and_fidelity_comparison.pdf")
    plt.close(fig)

    labels = [label for label, _ in mg.pauli_labels_and_weights()]
    for n_bar, chi_data in verification["chi_by_n_bar"].items():
        fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.5))
        panels = (
            ("baseline_exact", "Baseline exact |chi|"),
            ("optimized_exact", "Optimized exact |chi|"),
            ("delta_exact", "Delta exact |chi|"),
            ("baseline_error", "Baseline error |chi|"),
            ("optimized_error", "Optimized error |chi|"),
            ("delta_error", "Delta error |chi|"),
        )
        for ax, (key, title) in zip(axes.flat, panels):
            image = ax.imshow(np.abs(chi_data[key]), origin="lower", aspect="auto")
            ax.set_title(title)
            ax.set_xticks(range(16), labels=labels, rotation=90, fontsize=6)
            ax.set_yticks(range(16), labels=labels, fontsize=6)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(rf"Laser-pulse impact on chi, $\bar{{n}}={n_bar}$")
        fig.tight_layout()
        stem = f"chi_impact_nbar_{_safe_value(n_bar)}"
        fig.savefig(output_dir / f"{stem}.png", dpi=250)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)

    return output_dir


__all__ = [
    "LaserPulseSearchConfig",
    "build_symmetric_laser_amplitude",
    "ms_magnus_metrics",
    "thermal_ms_superoperator",
    "average_gate_fidelity_from_superoperator",
    "analytic_ms_average_gate_fidelity",
    "evaluate_pulse_robustness",
    "optimize_laser_amplitude_pulse",
    "verify_optimized_pulse_with_full_qpt",
    "save_laser_pulse_optimization_outputs",
]
