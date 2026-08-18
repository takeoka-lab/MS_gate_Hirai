"""Numerical reproduction of Kirchhoff--Wilhelm--Motzoi MS-gate fidelity.

This module implements the one-mode, two-qubit interaction-picture model in
Eqs. (14) and (17)--(19) of PRX Quantum 6, 010328 (2025).  It deliberately
keeps the paper's notation and fidelity convention separate from the QPT
average gate fidelity used elsewhere in this repository.

The paper's ``average fidelity`` is the thermally weighted gate-overlap
amplitude in Appendix B, Eq. (B6), not the standard channel average fidelity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import qutip as qp
import scipy.integrate
import scipy.linalg
import scipy.special

import drive_amplitude_calibration as drive_formula


TWO_QUBIT_OPERATOR_DIMS = [[2, 2], [2, 2]]


@dataclass(frozen=True)
class KirchhoffPaperModel:
    """Fixed matrices and frequencies for the paper's sideband Hamiltonian."""

    K: float
    L: float
    eta: float
    phonon_dim: int
    sideband_cutoff: int
    frequencies: np.ndarray
    operators: np.ndarray
    target_qubit_unitary: np.ndarray


def _collective_spin_operators() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    identity = np.eye(2, dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return tuple(
        0.5 * (np.kron(identity, sigma) + np.kron(sigma, identity))
        for sigma in (sigma_x, sigma_y, sigma_z)
    )


def _sideband_operator(
    eta: float,
    phonon_dim: int,
    sideband_order: int,
) -> np.ndarray:
    """Return A_m from Eq. (19), truncated only in the Fock basis."""

    result = np.zeros((phonon_dim, phonon_dim), dtype=complex)
    m = int(sideband_order)
    order = abs(m)
    eta_squared = float(eta) ** 2
    gaussian = math.exp(-0.5 * eta_squared)
    for column in range(phonon_dim):
        row = column + m
        if row < 0 or row >= phonon_dim:
            continue
        lower = min(row, column)
        upper = max(row, column)
        log_factorial_ratio = (
            scipy.special.gammaln(lower + 1.0)
            - scipy.special.gammaln(upper + 1.0)
        )
        result[row, column] = (
            gaussian
            * (1j * float(eta)) ** order
            * math.exp(0.5 * log_factorial_ratio)
            * scipy.special.eval_genlaguerre(lower, order, eta_squared)
        )
    return result


def build_paper_model(
    *,
    K: float = 28.0,
    L: float = 25.0,
    eta: float = 0.18,
    phonon_dim: int = 8,
    sideband_cutoff: int = 3,
) -> KirchhoffPaperModel:
    """Build the rectangular-pulse model used for Figs. 2, 3, and 5."""

    K = float(K)
    L = float(L)
    eta = float(eta)
    phonon_dim = int(phonon_dim)
    sideband_cutoff = int(sideband_cutoff)
    if not (K > L > 0.0):
        raise ValueError("Kirchhoff parameters must satisfy K > L > 0")
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if phonon_dim < 2:
        raise ValueError("phonon_dim must be at least 2")
    if sideband_cutoff < 1:
        raise ValueError("sideband_cutoff must be positive")

    collective_x, collective_y, _ = _collective_spin_operators()
    frequencies = []
    operators = []
    for m in range(-sideband_cutoff, sideband_cutoff + 1):
        sideband = _sideband_operator(eta, phonon_dim, m)
        collective_m = collective_x if m % 2 == 0 else 1j * collective_y
        total_operator = np.kron(sideband, collective_m)
        for mu in (-1, 1):
            frequencies.append(m * K + mu * L)
            operators.append(total_operator)

    # Appendix B: U_target = exp(+i pi/2 J_y^2).
    target = scipy.linalg.expm(1j * (np.pi / 2.0) * (collective_y @ collective_y))
    model = KirchhoffPaperModel(
        K=K,
        L=L,
        eta=eta,
        phonon_dim=phonon_dim,
        sideband_cutoff=sideband_cutoff,
        frequencies=np.asarray(frequencies, dtype=float),
        operators=np.asarray(operators, dtype=complex),
        target_qubit_unitary=target,
    )

    # Eq. (17) is Hermitian only after the +/- sidebands are combined.
    for tau in (0.0, 0.137, 0.611):
        base = dimensionless_hamiltonian(model, tau, omega_times_gate=1.0)
        relative_nonhermiticity = np.linalg.norm(base - base.conj().T) / max(
            np.linalg.norm(base), 1.0
        )
        if relative_nonhermiticity > 2e-12:
            raise RuntimeError(
                "Kirchhoff sideband Hamiltonian failed its Hermiticity check: "
                f"{relative_nonhermiticity:.3e}"
            )
    return model


def dimensionless_hamiltonian(
    model: KirchhoffPaperModel,
    tau: float,
    *,
    omega_times_gate: float,
) -> np.ndarray:
    """Return T H(t)/hbar at dimensionless time tau=t/T."""

    phases = np.exp(2j * np.pi * model.frequencies * float(tau))
    hamiltonian = float(omega_times_gate) * np.einsum(
        "k,kij->ij", phases, model.operators, optimize=True
    )
    # Remove round-off anti-Hermitian components before exponentiation.
    return 0.5 * (hamiltonian + hamiltonian.conj().T)


def automatic_trotter_steps(model: KirchhoffPaperModel, scale: float = 1.0) -> int:
    """Return the Eq. (43) step count from Delta t=1/(10 f_max)."""

    if float(scale) <= 0.0:
        raise ValueError("trotter step scale must be positive")
    maximum_index = float(np.max(np.abs(model.frequencies)))
    return max(1, int(math.ceil(float(scale) * 10.0 * 2.0 * np.pi * maximum_index)))


def propagate_paper_trotter(
    model: KirchhoffPaperModel,
    *,
    omega_per_second: float,
    gate_duration_seconds: float,
    number_of_steps: int | None = None,
    step_scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Propagate with the left-endpoint product in the paper's Eq. (43)."""

    if number_of_steps is None:
        number_of_steps = automatic_trotter_steps(model, step_scale)
    number_of_steps = int(number_of_steps)
    if number_of_steps < 1:
        raise ValueError("number_of_steps must be positive")
    omega_times_gate = float(omega_per_second) * float(gate_duration_seconds)
    delta_tau = 1.0 / number_of_steps
    dimension = model.phonon_dim * 4
    propagator = np.eye(dimension, dtype=complex)
    for step in range(number_of_steps):
        tau = step * delta_tau
        hamiltonian = dimensionless_hamiltonian(
            model, tau, omega_times_gate=omega_times_gate
        )
        propagator = scipy.linalg.expm(-1j * hamiltonian * delta_tau) @ propagator
    unitarity_error = float(
        np.linalg.norm(propagator.conj().T @ propagator - np.eye(dimension))
    )
    return propagator, {
        "solver": "paper_trotter_left_endpoint",
        "number_of_steps": number_of_steps,
        "step_scale": float(step_scale),
        "unitarity_frobenius_error": unitarity_error,
    }


def propagate_adaptive_ode(
    model: KirchhoffPaperModel,
    *,
    omega_per_second: float,
    gate_duration_seconds: float,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
    maximum_step: float | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Faster high-accuracy reference propagation of the same Hamiltonian."""

    omega_times_gate = float(omega_per_second) * float(gate_duration_seconds)
    dimension = model.phonon_dim * 4
    if maximum_step is None:
        maximum_step = 1.0 / (
            20.0 * float(np.max(np.abs(model.frequencies)))
        )

    def right_hand_side(tau, flattened):
        unitary = flattened.reshape((dimension, dimension))
        hamiltonian = dimensionless_hamiltonian(
            model, tau, omega_times_gate=omega_times_gate
        )
        return (-1j * hamiltonian @ unitary).reshape(-1)

    solution = scipy.integrate.solve_ivp(
        right_hand_side,
        (0.0, 1.0),
        np.eye(dimension, dtype=complex).reshape(-1),
        method="DOP853",
        rtol=float(relative_tolerance),
        atol=float(absolute_tolerance),
        max_step=float(maximum_step),
    )
    if not solution.success:
        raise RuntimeError(f"Kirchhoff ODE propagation failed: {solution.message}")
    propagator = solution.y[:, -1].reshape((dimension, dimension))
    unitarity_error = float(
        np.linalg.norm(propagator.conj().T @ propagator - np.eye(dimension))
    )
    return propagator, {
        "solver": "adaptive_dop853",
        "number_of_steps": int(len(solution.t) - 1),
        "function_evaluations": int(solution.nfev),
        "relative_tolerance": float(relative_tolerance),
        "absolute_tolerance": float(absolute_tolerance),
        "maximum_step": float(maximum_step),
        "unitarity_frobenius_error": unitarity_error,
    }


def propagate(
    model: KirchhoffPaperModel,
    *,
    omega_per_second: float,
    gate_duration_seconds: float,
    solver: str = "paper_trotter",
    number_of_steps: int | None = None,
    step_scale: float = 1.0,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Propagate one drive point with either paper-Trotter or adaptive ODE."""

    normalized_solver = str(solver).strip().lower()
    if normalized_solver in {"paper_trotter", "trotter"}:
        return propagate_paper_trotter(
            model,
            omega_per_second=omega_per_second,
            gate_duration_seconds=gate_duration_seconds,
            number_of_steps=number_of_steps,
            step_scale=step_scale,
        )
    if normalized_solver in {"adaptive_dop853", "dop853", "ode"}:
        return propagate_adaptive_ode(
            model,
            omega_per_second=omega_per_second,
            gate_duration_seconds=gate_duration_seconds,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    raise ValueError("solver must be 'paper_trotter' or 'adaptive_dop853'")


def thermal_weights(phonon_dim: int, n_bar: float) -> tuple[np.ndarray, float]:
    """Return normalized truncated Boltzmann weights and omitted tail mass."""

    n_bar = float(n_bar)
    if n_bar < 0.0:
        raise ValueError("n_bar must be nonnegative")
    ratio = n_bar / (n_bar + 1.0)
    indices = np.arange(int(phonon_dim), dtype=float)
    raw = (1.0 - ratio) * ratio**indices
    tail_mass = float(ratio ** int(phonon_dim))
    return raw / np.sum(raw), tail_mass


def yy_to_xx_local_rotation() -> np.ndarray:
    """Return the local Z rotation C satisfying C (Y⊗Y) C† = X⊗X."""

    # Rz(-pi/2) Y Rz(-pi/2)† = X for each qubit.
    single_qubit = np.diag(
        [np.exp(1j * np.pi / 4.0), np.exp(-1j * np.pi / 4.0)]
    )
    return np.kron(single_qubit, single_qubit)


def thermal_reduced_qubit_channel(
    propagator: np.ndarray,
    model: KirchhoffPaperModel,
    *,
    n_bar: float,
) -> tuple[qp.Qobj, dict[str, float | int]]:
    r"""Trace out motion after a thermal input and return the qubit channel.

    With normalized truncated thermal weights ``p_n``, the Kraus operators are

    .. math:: K_{mn}=\sqrt{p_n}\langle m|U(T)|n\rangle.

    The input/output ordering in this module is motion ⊗ two-qubit, so every
    Fock-space matrix element is a contiguous 4-by-4 qubit block.
    """

    dimension = model.phonon_dim * 4
    propagator = np.asarray(propagator, dtype=complex)
    if propagator.shape != (dimension, dimension):
        raise ValueError(
            f"propagator shape {propagator.shape} does not match {dimension}"
        )

    weights, tail_mass = thermal_weights(model.phonon_dim, n_bar)
    blocks = propagator.reshape(model.phonon_dim, 4, model.phonon_dim, 4)
    kraus_matrices = []
    for output_fock in range(model.phonon_dim):
        for input_fock, weight in enumerate(weights):
            kraus_matrices.append(
                math.sqrt(float(weight))
                * blocks[output_fock, :, input_fock, :]
            )
    completeness = sum(
        matrix.conj().T @ matrix for matrix in kraus_matrices
    )
    kraus_qobjs = [
        qp.Qobj(matrix, dims=TWO_QUBIT_OPERATOR_DIMS)
        for matrix in kraus_matrices
    ]
    channel = qp.kraus_to_super(kraus_qobjs)
    return channel, {
        "n_bar": float(n_bar),
        "thermal_tail_mass": float(tail_mass),
        "kraus_count": int(len(kraus_matrices)),
        "kraus_completeness_frobenius_error": float(
            np.linalg.norm(completeness - np.eye(4))
        ),
    }


def kirchhoff_xx_error_channel(
    propagator: np.ndarray,
    model: KirchhoffPaperModel,
    *,
    n_bar: float,
    convention: str = "undo_before_actual",
) -> tuple[qp.Qobj, qp.Qobj, dict[str, float | int | str]]:
    """Map the paper's YY channel to XX and remove the ideal MS gate.

    ``actual_xx`` is related to the thermally reduced paper channel by the
    local basis change ``C E(C† rho C) C†``.  The returned error channel follows
    the same convention names as :func:`ms_gate_functions.remove_ideal_gate_from_channel`.
    """

    actual_yy, metadata = thermal_reduced_qubit_channel(
        propagator, model, n_bar=n_bar
    )
    rotation = qp.Qobj(
        yy_to_xx_local_rotation(), dims=TWO_QUBIT_OPERATOR_DIMS
    )
    rotation_super = qp.to_super(rotation)
    actual_xx = rotation_super * actual_yy * qp.to_super(rotation.dag())

    rotated_target = rotation * qp.Qobj(
        model.target_qubit_unitary, dims=TWO_QUBIT_OPERATOR_DIMS
    ) * rotation.dag()
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    ideal_xx = qp.Qobj(
        scipy.linalg.expm(1j * (np.pi / 4.0) * np.kron(sigma_x, sigma_x)),
        dims=TWO_QUBIT_OPERATOR_DIMS,
    )
    target_superoperator_error = float(
        np.linalg.norm((qp.to_super(rotated_target) - qp.to_super(ideal_xx)).full())
    )
    ideal_inverse_super = qp.to_super(ideal_xx.dag())
    if convention == "undo_before_actual":
        error_channel = actual_xx * ideal_inverse_super
    elif convention == "undo_after_actual":
        error_channel = ideal_inverse_super * actual_xx
    else:
        raise ValueError(
            "convention must be 'undo_before_actual' or 'undo_after_actual'"
        )
    return actual_xx, error_channel, {
        **metadata,
        "error_channel_convention": str(convention),
        "yy_to_xx_target_superoperator_frobenius_error": (
            target_superoperator_error
        ),
    }


def paper_fidelities(
    propagator: np.ndarray,
    model: KirchhoffPaperModel,
    *,
    n_bar: float,
) -> dict[str, float]:
    """Evaluate Appendix-B average-overlap and Bell-state fidelities."""

    weights, tail_mass = thermal_weights(model.phonon_dim, n_bar)
    dimension = model.phonon_dim * 4
    propagator = np.asarray(propagator, dtype=complex)
    if propagator.shape != (dimension, dimension):
        raise ValueError(
            f"propagator shape {propagator.shape} does not match {dimension}"
        )

    target_dagger_total = np.kron(
        np.eye(model.phonon_dim, dtype=complex),
        model.target_qubit_unitary.conj().T,
    )
    overlap_operator = propagator @ target_dagger_total
    weighted_qubit_block = np.zeros((4, 4), dtype=complex)
    for n, weight in enumerate(weights):
        block = overlap_operator[4 * n : 4 * (n + 1), 4 * n : 4 * (n + 1)]
        weighted_qubit_block += weight * block
    average_overlap_fidelity = float(abs(np.trace(weighted_qubit_block)) / 4.0)

    initial_qubit = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    target_qubit = model.target_qubit_unitary @ initial_qubit
    bell_fidelity = 0.0
    for n, weight in enumerate(weights):
        initial = np.zeros(dimension, dtype=complex)
        initial[4 * n : 4 * (n + 1)] = initial_qubit
        final = propagator @ initial
        reshaped = final.reshape(model.phonon_dim, 4)
        target_amplitudes = reshaped @ target_qubit.conj()
        bell_fidelity += float(weight * np.vdot(target_amplitudes, target_amplitudes).real)

    return {
        "n_bar": float(n_bar),
        "paper_average_overlap_fidelity": average_overlap_fidelity,
        "paper_average_infidelity": max(0.0, 1.0 - average_overlap_fidelity),
        "paper_bell_fidelity": bell_fidelity,
        "paper_bell_infidelity": max(0.0, 1.0 - bell_fidelity),
        "thermal_tail_mass": tail_mass,
    }


def drive_landmarks(
    *,
    K: float,
    L: float,
    gate_duration_seconds: float,
    eta: float,
) -> dict[str, float]:
    """Return the paper's Eqs. (32), (35), and (41) drive amplitudes."""

    return {
        "omega_ld": drive_formula.kirchhoff_omega_ld(
            K, L, gate_duration_seconds, eta
        ),
        "omega_2": drive_formula.kirchhoff_omega_second_order(
            K, L, gate_duration_seconds, eta
        ),
        "omega_4": drive_formula.kirchhoff_omega_fourth_order(
            K, L, gate_duration_seconds, eta
        ),
    }


def unique_drive_values(
    requested_values: Iterable[float],
    landmarks: dict[str, float],
) -> list[float]:
    """Include exact analytic landmarks in an arbitrary drive sweep."""

    values = [float(value) for value in requested_values]
    values.extend(float(value) for value in landmarks.values())
    if not values or any(value <= 0.0 or not np.isfinite(value) for value in values):
        raise ValueError("all drive amplitudes must be finite and positive")
    return sorted({round(value, 12) for value in values})
