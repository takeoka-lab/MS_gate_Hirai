"""Fock-resolved XX-angle analysis for the full-order MS Hamiltonian.

The MS Hamiltonian is diagonal in the collective-X spin basis.  For a fixed
initial Fock state ``|n>`` the relative coherence between the forced
``S_x=+2`` branch and a null ``S_x=0`` branch is therefore

    c_n = <n| U_{S_x=+2}(T) |n>.

Its phase gives the Fock-conditioned XX angle ``theta_n = arg(c_n) / 2`` and
its magnitude diagnoses residual spin-motion entanglement.  Thermal averaging
of ``c_n`` predicts both the coherent XX offset and the stochastic-XX-equivalent
decay without rerunning a 16-input-state QPT for every Fock number.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qp


def _json_safe(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _control_values(name: str, values: Any, time_grid: np.ndarray) -> np.ndarray:
    if callable(values):
        values = values(time_grid)
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(time_grid.shape, float(array), dtype=float)
    if array.shape != time_grid.shape:
        raise ValueError(
            f"{name} must be scalar or have shape {time_grid.shape}; "
            f"got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _integrated_phase(detuning: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    phase = np.zeros_like(time_grid, dtype=float)
    phase[1:] = np.cumsum(
        0.5 * (detuning[1:] + detuning[:-1]) * np.diff(time_grid)
    )
    return phase


def required_fock_cutoff(n_bar: float, tail_tolerance: float = 1e-5) -> int:
    """Return the largest Fock index needed for a thermal-tail tolerance.

    For a thermal state, the probability above ``n_max`` is exactly
    ``(n_bar / (1+n_bar))**(n_max+1)``.
    """

    n_bar = float(n_bar)
    tail_tolerance = float(tail_tolerance)
    if n_bar < 0:
        raise ValueError("n_bar must be non-negative.")
    if not 0 < tail_tolerance < 1:
        raise ValueError("tail_tolerance must lie strictly between 0 and 1.")
    if n_bar == 0:
        return 0
    ratio = n_bar / (1.0 + n_bar)
    return max(0, int(np.ceil(np.log(tail_tolerance) / np.log(ratio) - 1.0)))


def thermal_fock_probabilities(
    n_bar: float,
    max_fock_n: int,
) -> tuple[np.ndarray, float]:
    """Return normalized truncated thermal weights and retained probability."""

    n_bar = float(n_bar)
    max_fock_n = int(max_fock_n)
    if n_bar < 0:
        raise ValueError("n_bar must be non-negative.")
    if max_fock_n < 0:
        raise ValueError("max_fock_n must be non-negative.")
    if n_bar == 0:
        weights = np.zeros(max_fock_n + 1, dtype=float)
        weights[0] = 1.0
        return weights, 1.0
    ratio = n_bar / (1.0 + n_bar)
    weights = (1.0 - ratio) * ratio ** np.arange(max_fock_n + 1)
    retained_mass = float(np.sum(weights))
    return weights / retained_mass, retained_mass


def _gate_time_simulation_units(params: Mapping[str, Any]) -> float:
    configured = params.get("t_gate_sim")
    if configured is not None:
        value = float(configured)
        if value <= 0:
            raise ValueError("t_gate_sim must be positive.")
        return value
    detuning = np.asarray(params["delta"], dtype=float)
    if detuning.ndim != 0 or float(detuning) == 0:
        raise ValueError(
            "t_gate_sim is required for a time-dependent or zero detuning."
        )
    return 2.0 * np.pi / abs(float(detuning))


def fock_curve_signature(
    simulation_params: Mapping[str, Any],
    *,
    amplitude: Any,
    max_fock_n: int,
    phonon_buffer: int,
    target_xx_angle_rad: float = np.pi / 4.0,
) -> str:
    """Return a stable cache signature for one Fock-angle curve."""

    keys = [
        "delta",
        "rho0",
        "time_points",
        "t_gate_sim",
        "eta",
        "use_full_order",
        "solver_max_step",
    ]
    payload = {
        "parameters": {
            key: simulation_params.get(key)
            for key in keys
        },
        "amplitude": amplitude,
        "max_fock_n": int(max_fock_n),
        "phonon_buffer": int(phonon_buffer),
        "target_xx_angle_rad": float(target_xx_angle_rad),
        "method": "forced_null_coherence_v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=_json_safe).encode("utf-8")
    ).hexdigest()[:16]


def calculate_fock_resolved_xx_angles(
    simulation_params: Mapping[str, Any],
    *,
    amplitude: Any | None = None,
    max_fock_n: int,
    phonon_buffer: int = 24,
    target_xx_angle_rad: float = np.pi / 4.0,
) -> pd.DataFrame:
    """Calculate ``theta_n`` directly from the noise-free full Hamiltonian.

    The returned coherence magnitude includes residual spin-motion
    entanglement for the selected Fock state.  Lindblad and quasi-static noise
    are deliberately excluded so that thermal Hamiltonian effects are isolated.
    """

    params = dict(simulation_params)
    max_fock_n = int(max_fock_n)
    phonon_buffer = int(phonon_buffer)
    if max_fock_n < 0:
        raise ValueError("max_fock_n must be non-negative.")
    if phonon_buffer < 2:
        raise ValueError("phonon_buffer must be at least 2.")
    time_points = int(params.get("time_points", 500))
    if time_points < 2:
        raise ValueError("time_points must be at least 2.")

    gate_time = _gate_time_simulation_units(params)
    time_grid = np.linspace(0.0, gate_time, time_points)
    amplitude_values = _control_values(
        "A", params["A"] if amplitude is None else amplitude, time_grid
    )
    detuning_values = _control_values("delta", params["delta"], time_grid)
    rho_values = _control_values("rho0", params.get("rho0", 0.0), time_grid)
    if not np.allclose(rho_values, rho_values[0], atol=1e-12, rtol=0.0):
        raise ValueError(
            "Fock-resolved conditional-branch analysis requires a fixed spin axis."
        )

    phonon_dim = max_fock_n + phonon_buffer + 1
    annihilation = qp.destroy(phonon_dim)
    number = annihilation.dag() * annihilation
    if bool(params.get("use_full_order", True)):
        eta = float(params.get("eta", 0.1))
        op_minus = annihilation - (
            eta**2 / 2.0
        ) * (number + qp.qeye(phonon_dim)) * annihilation
    else:
        op_minus = annihilation

    phase = _integrated_phase(detuning_values, time_grid)
    # The forced collective-X branch has eigenvalue +2; the null branch has 0.
    forced_amplitude = 2.0 * amplitude_values
    coefficient_minus = qp.coefficient(
        forced_amplitude * np.exp(-1j * phase),
        tlist=time_grid,
        order=1,
    )
    coefficient_plus = qp.coefficient(
        forced_amplitude * np.exp(1j * phase),
        tlist=time_grid,
        order=1,
    )
    hamiltonian = (
        op_minus * coefficient_minus
        + op_minus.dag() * coefficient_plus
    )
    options = {
        "progress_bar": None,
        "atol": 1e-12,
        "rtol": 1e-9,
        "store_final_state": True,
        "store_states": False,
    }
    solver_max_step = params.get("solver_max_step")
    if solver_max_step is not None:
        options["max_step"] = float(solver_max_step)
    solver = qp.SESolver(hamiltonian, options=options)

    coherences = []
    for fock_n in range(max_fock_n + 1):
        initial = qp.basis(phonon_dim, fock_n)
        result = solver.run(initial, time_grid, e_ops=[])
        final_state = result.final_state
        coherences.append(complex(initial.dag() * final_state))

    coherences = np.asarray(coherences, dtype=complex)
    theta = 0.5 * np.unwrap(np.angle(coherences))
    return pd.DataFrame({
        "fock_n": np.arange(max_fock_n + 1, dtype=int),
        "coherence_real": np.real(coherences),
        "coherence_imag": np.imag(coherences),
        "coherence_abs": np.abs(coherences),
        "theta_xx_rad": theta,
        "theta_error_rad": float(target_xx_angle_rad) - theta,
    })


def save_fock_curve(path: str | Path, curve: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **{column: curve[column].to_numpy() for column in curve.columns},
    )
    temporary.replace(path)


def load_fock_curve(path: str | Path) -> pd.DataFrame:
    with np.load(Path(path), allow_pickle=False) as data:
        return pd.DataFrame({column: data[column] for column in data.files})


def summarize_thermal_xx_angle(
    curve: pd.DataFrame,
    n_bar: float,
    *,
    target_xx_angle_rad: float = np.pi / 4.0,
) -> dict[str, float]:
    """Thermally average a Fock curve into coherent and stochastic metrics."""

    expected_n = np.arange(len(curve), dtype=int)
    if not np.array_equal(curve["fock_n"].to_numpy(dtype=int), expected_n):
        raise ValueError("curve must contain contiguous Fock numbers starting at 0.")
    weights, retained_mass = thermal_fock_probabilities(
        n_bar, len(curve) - 1
    )
    coherence = (
        curve["coherence_real"].to_numpy(float)
        + 1j * curve["coherence_imag"].to_numpy(float)
    )
    theta = curve["theta_xx_rad"].to_numpy(float)
    mean_coherence = complex(np.dot(weights, coherence))
    phase_only_coherence = complex(np.dot(weights, np.exp(2j * theta)))
    linear_mean_theta = float(np.dot(weights, theta))
    raw_theta = 0.5 * float(np.angle(mean_coherence))
    # XX angles are periodic modulo pi.  Select the branch nearest the
    # unwrapped linear mean so high-n curves remain continuous.
    effective_theta = raw_theta + np.pi * round(
        (linear_mean_theta - raw_theta) / np.pi
    )
    angle_variance = float(
        np.dot(weights, (theta - linear_mean_theta) ** 2)
    )
    total_gamma = max(
        0.0, -0.5 * float(np.log(max(abs(mean_coherence), 1e-300)))
    )
    phase_gamma = max(
        0.0,
        -0.5 * float(np.log(max(abs(phase_only_coherence), 1e-300))),
    )
    return {
        "thermal_n_bar": float(n_bar),
        "retained_thermal_mass": retained_mass,
        "thermal_tail_mass": 1.0 - retained_mass,
        "effective_theta_xx_rad": effective_theta,
        "predicted_h_XX_rad_per_gate": (
            float(target_xx_angle_rad) - effective_theta
        ),
        "theta_xx_linear_mean_rad": linear_mean_theta,
        "theta_xx_variance_rad2": angle_variance,
        "theta_xx_std_rad": float(np.sqrt(max(angle_variance, 0.0))),
        "mean_forced_null_coherence_abs": float(abs(mean_coherence)),
        "gamma_XX_phase_dispersion_per_gate": phase_gamma,
        "gamma_XX_with_residual_motion_per_gate": total_gamma,
    }


def plot_fock_angle_comparison(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    output_dir: str | Path,
    plot_max_fock_n: int = 80,
    target_xx_angle_rad: float = np.pi / 4.0,
) -> Path:
    """Plot Fock curves and matched thermal predictions."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.5))
    visible = curves.loc[curves["fock_n"] <= int(plot_max_fock_n)]
    for condition, group in visible.groupby("condition", sort=False):
        label = str(group.iloc[0]["label"])
        linewidth = 2.6 if condition == "baseline" else 1.7
        axes[0, 0].plot(
            group["fock_n"], group["theta_xx_rad"],
            linewidth=linewidth, label=label,
        )
        axes[0, 1].plot(
            group["fock_n"], group["coherence_abs"],
            linewidth=linewidth, label=label,
        )
    axes[0, 0].axhline(
        float(target_xx_angle_rad), color="black", linestyle="--",
        linewidth=1.2, label=r"target $\pi/4$",
    )
    axes[0, 0].set_ylabel(r"Fock-conditioned $\theta_n$ [rad]")
    axes[0, 1].set_ylabel(r"Forced-null coherence $|c_n|$")
    for axis in axes[0]:
        axis.set_xlabel("Initial Fock number n")
        axis.grid(True, alpha=0.28)
    axes[0, 0].legend(fontsize=8, ncol=2)

    baseline = summary.loc[summary["condition"] == "baseline"].sort_values(
        "thermal_n_bar"
    )
    matched = summary.loc[
        (summary["condition"] != "baseline")
        & summary["is_matched_calibration"]
    ].sort_values("thermal_n_bar")
    panels = [
        (axes[1, 0], "predicted_h_XX_rad_per_gate", r"Predicted $h_{XX}$"),
        (
            axes[1, 1],
            "gamma_XX_with_residual_motion_per_gate",
            r"Intrinsic $\gamma_{XX}$ equivalent",
        ),
    ]
    for axis, metric, title in panels:
        axis.plot(
            baseline["thermal_n_bar"], baseline[metric], "o-",
            color="black", label="baseline: Fock prediction",
        )
        axis.plot(
            matched["thermal_n_bar"], matched[metric], "s-",
            color="#0072B2", label="drive calibrated: Fock prediction",
        )
        qpt_metric = (
            "qpt_h_XX_rad_per_gate"
            if metric == "predicted_h_XX_rad_per_gate"
            else "qpt_gamma_XX_per_gate"
        )
        qpt_rows = pd.concat([baseline, matched], ignore_index=True).dropna(
            subset=[qpt_metric]
        )
        if not qpt_rows.empty:
            axis.scatter(
                qpt_rows["thermal_n_bar"], qpt_rows[qpt_metric],
                marker="x", s=65, color="#D55E00", label="full noisy QPT",
                zorder=4,
            )
        axis.set_title(title)
        axis.set_xlabel(r"Mean phonon number $\bar n$")
        axis.grid(True, alpha=0.28)
        axis.legend(fontsize=8)
    axes[1, 1].set_yscale("log")
    figure.suptitle(
        "Fock-resolved XX angle and thermal angle-dispersion prediction",
        y=1.01,
    )
    figure.tight_layout()
    png_path = output_dir / "fock_resolved_xx_angle.png"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(
        output_dir / "fock_resolved_xx_angle.pdf", bbox_inches="tight"
    )
    plt.close(figure)
    return png_path
