"""Drive-amplitude calibration helpers for the Mølmer--Sørensen analysis.

The project uses ``U_MS(phi) = exp(+i phi XX)`` and defines the post-gate
error channel as ``S_actual * S_ideal^{-1}``.  With the error-generator basis
``-i h_XX [XX, rho]``, a positive ``h_XX`` therefore means that the physical
gate under-rotates by ``h_XX``.

The Kirchhoff--Wilhelm--Motzoi formulas implemented below are Eqs. (32), (35)
and (41) of PRX Quantum 6, 010328 (2025).  Their ``Omega`` is the carrier
drive, whereas this repository's scalar ``A`` is the first-sideband coupling.
For fixed Lamb--Dicke parameter, only the dimensionless ratios are compared,
so the common conversion factor cancels.
"""

from __future__ import annotations

import math


def amplitude_update_from_hxx(
    current_amplitude: float,
    h_xx_rad_per_gate: float,
    target_xx_angle_rad: float = math.pi / 4.0,
) -> float:
    """Return the next scalar drive amplitude inferred from ``h_XX``.

    The current physical entangling angle is estimated as

    ``phi_actual = phi_target - h_XX``.

    For a closed phase-space loop the geometric XX angle scales as the square
    of the drive amplitude.  Rescaling the measured angle to the target gives

    ``A_next = A_current * sqrt(phi_target / phi_actual)``.

    This is an estimator, not a replacement for re-simulation.  Higher-order
    terms make the actual response nonquadratic, which is why the notebook
    feeds the residual ``h_XX`` from a new full-Hamiltonian QPT back into this
    formula when additional calibration iterations are requested.
    """

    amplitude = float(current_amplitude)
    h_xx = float(h_xx_rad_per_gate)
    target = float(target_xx_angle_rad)
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("current_amplitude must be finite and positive")
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_xx_angle_rad must be finite and positive")
    if not math.isfinite(h_xx):
        raise ValueError("h_xx_rad_per_gate must be finite")

    actual_angle = target - h_xx
    if actual_angle <= 0.0:
        raise ValueError(
            "The inferred physical XX angle is non-positive; the quadratic "
            "amplitude update is outside its valid branch."
        )
    return amplitude * math.sqrt(target / actual_angle)


def kirchhoff_omega_ld(K: float, L: float, gate_duration: float, eta: float) -> float:
    """Lamb--Dicke optimum ``Omega_LD`` from Eq. (32)."""

    K, L, gate_duration, eta = _validate_kirchhoff_inputs(K, L, gate_duration, eta)
    return (
        math.pi
        / (gate_duration * eta)
        * math.sqrt((K * K - L * L) / (2.0 * K))
    )


def kirchhoff_omega_second_order(
    K: float,
    L: float,
    gate_duration: float,
    eta: float,
) -> float:
    """Second-Magnus-order corrected optimum ``Omega_2`` from Eq. (35)."""

    K, L, gate_duration, eta = _validate_kirchhoff_inputs(K, L, gate_duration, eta)
    numerator = (K * K - L * L) * (4.0 * K * K - L * L)
    denominator = 2.0 * K * (
        eta * eta * (2.0 * L * L - 5.0 * K * K)
        + (4.0 * K * K - L * L)
    )
    if denominator <= 0.0 or numerator <= 0.0:
        raise ValueError("Kirchhoff Eq. (35) is outside its real-valued domain")
    return math.pi / (gate_duration * eta) * math.sqrt(numerator / denominator)


def kirchhoff_omega_fourth_order(
    K: float,
    L: float,
    gate_duration: float,
    eta: float,
) -> float:
    """Fourth-Magnus-order corrected optimum ``Omega_4`` from Eq. (41).

    The paper writes a closed expression for ``Omega_4**2``.  The smaller
    quadratic root is used, as in Eq. (41).  The formula is valid only when
    ``s**2 - (K**2-L**2) >= 0``.
    """

    K, L, gate_duration, eta = _validate_kirchhoff_inputs(K, L, gate_duration, eta)
    difference = K * K - L * L
    # In the typeset equation the square-root bar covers 2K only:
    # s = sqrt(2 K) L eta (1 - eta^2).
    s_value = math.sqrt(2.0 * K) * L * eta * (1.0 - eta * eta)
    radicand = s_value * s_value - difference
    scale = max(s_value * s_value, difference, 1.0)
    if radicand < -1e-14 * scale:
        raise ValueError("Kirchhoff Eq. (41) is outside its real-valued domain")
    radicand = max(radicand, 0.0)
    omega_squared = (
        math.sqrt(2.0)
        * math.pi**2
        * L
        * (s_value - math.sqrt(radicand))
        / (math.sqrt(K) * gate_duration**2 * eta)
    )
    if omega_squared <= 0.0:
        raise ValueError("Kirchhoff Eq. (41) returned a non-positive Omega^2")
    return math.sqrt(omega_squared)


def kirchhoff_renormalization_ratios(
    K: float,
    L: float,
    eta: float,
) -> dict[str, float]:
    """Return Eq. (35)/(32) and Eq. (41)/(32) amplitude ratios.

    A unit gate duration is sufficient because it cancels from the ratios.
    """

    omega_ld = kirchhoff_omega_ld(K, L, 1.0, eta)
    omega_2 = kirchhoff_omega_second_order(K, L, 1.0, eta)
    omega_4 = kirchhoff_omega_fourth_order(K, L, 1.0, eta)
    return {
        "omega_2_over_omega_ld": omega_2 / omega_ld,
        "omega_4_over_omega_ld": omega_4 / omega_ld,
    }


def _validate_kirchhoff_inputs(
    K: float,
    L: float,
    gate_duration: float,
    eta: float,
) -> tuple[float, float, float, float]:
    K = float(K)
    L = float(L)
    gate_duration = float(gate_duration)
    eta = float(eta)
    if not all(math.isfinite(value) for value in (K, L, gate_duration, eta)):
        raise ValueError("Kirchhoff parameters must be finite")
    if K <= 0.0 or L <= 0.0 or K <= L:
        raise ValueError("Kirchhoff parameters require K > L > 0")
    if gate_duration <= 0.0:
        raise ValueError("gate_duration must be positive")
    if eta <= 0.0 or eta >= 1.0:
        raise ValueError("eta must satisfy 0 < eta < 1")
    return K, L, gate_duration, eta

