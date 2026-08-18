"""Bridge Kirchhoff's carrier model to this repository's effective MS model."""

from __future__ import annotations

import math

import numpy as np
import qutip as qp

import kirchhoff_paper_infidelity as kirchhoff_paper
import ms_gate_functions as mg


TWO_QUBIT_OPERATOR_DIMS = [[2, 2], [2, 2]]


def carrier_to_effective_sideband_amplitude(
    omega_per_second: float,
    *,
    eta: float,
    gate_duration_seconds: float,
    t_gate_sim: float = 1.0,
) -> float:
    r"""Map the paper's carrier ``Omega`` to the repository's scalar ``A``.

    The resonant first-sideband term has coefficient ``eta*Omega*J`` while
    the repository Hamiltonian uses ``A*(sigma_1+sigma_2)=2*A*J``.  After
    rescaling physical time to ``t_gate_sim``, this gives

    ``A_sim = eta*Omega*T/(2*t_gate_sim)``.
    """

    omega_per_second = float(omega_per_second)
    eta = float(eta)
    gate_duration_seconds = float(gate_duration_seconds)
    t_gate_sim = float(t_gate_sim)
    if omega_per_second <= 0.0 or eta <= 0.0:
        raise ValueError("omega_per_second and eta must be positive")
    if gate_duration_seconds <= 0.0 or t_gate_sim <= 0.0:
        raise ValueError("gate durations must be positive")
    return (
        eta
        * omega_per_second
        * gate_duration_seconds
        / (2.0 * t_gate_sim)
    )


def effective_detuning(
    K: float,
    L: float,
    *,
    t_gate_sim: float = 1.0,
) -> float:
    """Return the repository's angular sideband detuning for ``K-L`` loops."""

    K = float(K)
    L = float(L)
    t_gate_sim = float(t_gate_sim)
    if K <= L or L <= 0.0 or t_gate_sim <= 0.0:
        raise ValueError("require K > L > 0 and t_gate_sim > 0")
    return 2.0 * np.pi * (K - L) / t_gate_sim


def thermal_qubit_channel_from_own_propagator(
    propagator,
    *,
    phonon_dim: int,
    n_bar: float,
) -> tuple[qp.Qobj, dict[str, float | int]]:
    """Trace motion from a qubits⊗motion propagator using thermal Kraus blocks."""

    phonon_dim = int(phonon_dim)
    matrix = (
        np.asarray(propagator.full(), dtype=complex)
        if isinstance(propagator, qp.Qobj)
        else np.asarray(propagator, dtype=complex)
    )
    dimension = 4 * phonon_dim
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"propagator shape {matrix.shape} does not match {dimension}"
        )
    weights, tail_mass = kirchhoff_paper.thermal_weights(phonon_dim, n_bar)
    blocks = matrix.reshape(4, phonon_dim, 4, phonon_dim)
    kraus_matrices = []
    for output_fock in range(phonon_dim):
        for input_fock, weight in enumerate(weights):
            kraus_matrices.append(
                math.sqrt(float(weight))
                * blocks[:, output_fock, :, input_fock]
            )
    completeness = sum(
        matrix.conj().T @ matrix for matrix in kraus_matrices
    )
    channel = qp.kraus_to_super([
        qp.Qobj(matrix, dims=TWO_QUBIT_OPERATOR_DIMS)
        for matrix in kraus_matrices
    ])
    return channel, {
        "n_bar": float(n_bar),
        "thermal_tail_mass": float(tail_mass),
        "kraus_count": int(len(kraus_matrices)),
        "kraus_completeness_frobenius_error": float(
            np.linalg.norm(completeness - np.eye(4))
        ),
    }


def own_xx_error_channel(
    propagator,
    *,
    phonon_dim: int,
    n_bar: float,
    convention: str = "undo_before_actual",
) -> tuple[qp.Qobj, qp.Qobj, dict[str, float | int | str]]:
    """Return the repository model's thermally reduced actual/error channels."""

    actual, metadata = thermal_qubit_channel_from_own_propagator(
        propagator,
        phonon_dim=phonon_dim,
        n_bar=n_bar,
    )
    ideal_inverse = qp.to_super(mg.ideal_ms_gate().dag())
    if convention == "undo_before_actual":
        error = actual * ideal_inverse
    elif convention == "undo_after_actual":
        error = ideal_inverse * actual
    else:
        raise ValueError(
            "convention must be 'undo_before_actual' or 'undo_after_actual'"
        )
    return actual, error, {
        **metadata,
        "error_channel_convention": str(convention),
    }
