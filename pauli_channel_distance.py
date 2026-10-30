"""Distances from a qubit channel to its Pauli twirl.

Canonical convention: E(rho) = sum chi[m,n] P[m] rho P[n]^dagger,
Tr(P[m]^dagger P[n]) = d delta[m,n], and Tr(chi) = 1 for a TP map.
Choi order is input tensor output, with column-major vectorization.
Every reported 'distance' based on trace/diamond norm includes the factor 1/2.
No physical channel is reprojected, renormalized, or refitted here.
"""

from itertools import product

import numpy as np


def hermitian(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    return (matrix + matrix.conj().T) / 2


def trace_norm(matrix):
    """Schatten 1 norm, not an elementwise absolute sum."""
    return float(np.linalg.svd(matrix, compute_uv=False).sum())


def pauli_basis(n_qubits=2):
    single = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.diag([1, -1]).astype(complex),
    }
    labels, operators = [], []
    for factors in product(single, repeat=n_qubits):
        operator = np.ones((1, 1), complex)
        for name in factors:
            operator = np.kron(operator, single[name])
        labels.append("".join(factors))
        operators.append(operator)
    return labels, np.asarray(operators)


def qutip_chi_to_canonical(chi, labels):
    """Convert trace-one QuTiP chi (vec(P*) basis) to the standard P basis.

    QuTiP 5.0.4's _superpauli_basis stacks vec(P)^T then takes the adjoint.
    Hence its columns are vec(P*) = (-1)**number_of_Y * vec(P).
    The notebook independently checks this conversion against qp.to_choi.
    Raw qp.to_chi output has trace d**2 and must FIRST be divided by d**2.
    """
    chi = np.asarray(chi, dtype=complex)
    if chi.shape != (len(labels), len(labels)):
        raise ValueError("chi shape and Pauli labels disagree")
    signs = np.asarray([(-1) ** label.count("Y") for label in labels])
    return signs[:, None] * chi * signs[None, :]


def _basis_vectors(paulis):
    return np.column_stack([p.reshape(-1, order="F") for p in paulis])


def chi_to_choi(chi, paulis):
    """Unnormalized J = sum_ij |i><j| tensor E(|i><j|); Tr(J)=d for TP."""
    vectors = _basis_vectors(paulis)
    return vectors @ np.asarray(chi) @ vectors.conj().T


def choi_to_chi(choi, paulis):
    vectors = _basis_vectors(paulis)
    d = paulis[0].shape[0]
    return vectors.conj().T @ choi @ vectors / d**2


def partial_trace_output(choi, d):
    return np.einsum("iaja->ij", np.asarray(choi).reshape(d, d, d, d))


def apply_chi(chi, rho, paulis):
    return np.einsum(
        "mn,mab,bc,ndc->ad", chi, paulis, rho, paulis.conj(), optimize=True
    )


def chi_to_ptm(chi, paulis):
    """R[a,b] = Tr(P[a]^dagger E(P[b])) / d."""
    d = paulis[0].shape[0]
    return np.column_stack([
        np.einsum("aij,ij->a", paulis.conj(), apply_chi(chi, p, paulis)) / d
        for p in paulis
    ])


def pauli_twirl_chi(chi):
    return np.diag(np.diag(np.asarray(chi)))


def channel_diagnostics(chi, paulis, cp_tolerance=1e-9, tp_tolerance=1e-8):
    choi = chi_to_choi(chi, paulis)
    d = paulis[0].shape[0]
    hp_error = float(np.linalg.norm(choi - choi.conj().T))
    tp_error = float(np.linalg.norm(partial_trace_output(choi, d) - np.eye(d)))
    minimum = float(np.linalg.eigvalsh(hermitian(choi)).min())
    return {
        "chi_trace_error": float(abs(np.trace(chi) - 1)),
        "choi_hermiticity_error": hp_error,
        "choi_min_eigenvalue": minimum,
        "choi_tp_frobenius_error": tp_error,
        "cptp_within_tolerance": bool(
            minimum >= -cp_tolerance and hp_error <= tp_tolerance and tp_error <= tp_tolerance
        ),
    }


def diamond_bounds(choi_delta, d):
    """Analytic bounds on HALF the diamond norm of a Hermiticity-preserving map.

    L = ||J||_1/(2d), U_simple = ||J||_1/2,
    U_abs = ||Tr_out |J|||_infinity/2 <= U_simple.
    |J| >= +/-J is a feasible dual witness. Do not cap U at one: the
    routine also supports maps that are not differences of CPTP channels.
    """
    j = np.asarray(choi_delta, dtype=complex)
    if j.shape != (d*d, d*d):
        raise ValueError("Choi dimensions disagree with d")
    if np.linalg.norm(j - j.conj().T) > 1e-9 * max(1.0, np.linalg.norm(j)):
        raise ValueError("diamond_bounds requires a Hermiticity-preserving map")
    values, vectors = np.linalg.eigh(hermitian(j))
    abs_j = (vectors * np.abs(values)) @ vectors.conj().T
    norm1 = float(np.abs(values).sum())
    return {
        "choi_trace_distance": norm1 / (2*d),
        "diamond_lower_choi": norm1 / (2*d),
        "diamond_upper_simple": norm1 / 2,
        "diamond_upper_abs_choi": float(
            np.linalg.eigvalsh(hermitian(partial_trace_output(abs_j, d))).max() / 2
        ),
    }


def distance_metrics(chi, paulis):
    """Compare a canonical chi with its twirl, preserving the input matrices."""
    d = paulis[0].shape[0]
    twirled = pauli_twirl_chi(chi)
    delta = chi - twirled
    identity_chi = np.zeros_like(chi)
    identity_chi[0, 0] = 1
    error_norm = float(d * np.linalg.norm(chi - identity_chi))
    fro = float(d * np.linalg.norm(delta))
    ptm = chi_to_ptm(chi, paulis)
    ptm_twirl = chi_to_ptm(twirled, paulis)
    return {
        "ptm_frobenius_distance": fro,
        "error_ptm_frobenius_norm": error_norm,
        # 0/0 is deliberately undefined for the ideal identity channel.
        "relative_error_frobenius_distance": fro / error_norm if error_norm > 1e-14 else np.nan,
        "average_infidelity": float(d / (d+1) * (1 - chi[0, 0].real)),
        "twirl_infidelity_difference": float(d / (d+1) * (chi[0, 0] - twirled[0, 0]).real),
        "ptm_chi_frobenius_identity_error": float(abs(np.linalg.norm(ptm - ptm_twirl) - fro)),
        "ptm_twirl_diagonalization_error": float(np.linalg.norm(ptm_twirl - np.diag(np.diag(ptm)))),
        **diamond_bounds(chi_to_choi(delta, paulis), d),
    }


def diamond_distance_sdp(choi_delta, d, solver="SCS", solver_tolerance=1e-8,
                         gap_atol=1e-8, gap_rtol=1e-5):
    """Solve HP diamond primal and dual, returning a checked numerical bracket.

    Primal: max Tr(J W), rho>=0, Tr(rho)=1, -rho(x)I <= W <= rho(x)I.
    Dual: min t, Z>=+J, Z>=-J, Tr_out Z <= t I.
    Both objectives give the full norm; output distances are divided by 2.

    For reliability, project ONLY the optimizing rho onto density matrices
    and recompute a feasible-input lower bound. Shift ONLY the dual witness
    Z to satisfy Z>=+/-J and recompute an upper bound. The channel is never
    projected. Bounds are mathematical in exact arithmetic, numerical here:
    this is not an interval-arithmetic certificate or an ODE error bound.
    """
    import cvxpy as cp

    bounds = diamond_bounds(choi_delta, d)
    j_raw = hermitian(choi_delta)
    scale = float(np.linalg.norm(j_raw, 2))
    if scale == 0:
        return {
            "diamond_sdp_lower": 0.0, "diamond_sdp_upper": 0.0,
            "diamond_distance_estimate": 0.0, "diamond_sdp_gap": 0.0,
            "diamond_sdp_relative_gap": 0.0, "sdp_accepted": True,
            "sdp_primal_status": "zero_map", "sdp_dual_status": "zero_map",
            "sdp_solver": solver,
        }
    if solver not in cp.installed_solvers():
        raise RuntimeError(f"{solver} is unavailable; installed: {cp.installed_solvers()}")
    j = j_raw / scale
    rho = cp.Variable((d, d), hermitian=True)
    witness = cp.Variable((d*d, d*d), hermitian=True)
    sandwich = cp.kron(rho, np.eye(d))
    primal = cp.Problem(cp.Maximize(cp.real(cp.trace(j @ witness))), [
        rho >> 0, cp.trace(rho) == 1,
        sandwich - witness >> 0, sandwich + witness >> 0,
    ])
    z = cp.Variable((d*d, d*d), hermitian=True)
    t = cp.Variable()
    trace_z = cp.bmat([
        [cp.trace(z[i*d:(i+1)*d, k*d:(k+1)*d]) for k in range(d)]
        for i in range(d)
    ])
    dual = cp.Problem(cp.Minimize(t), [
        z - j >> 0, z + j >> 0, t*np.eye(d) - trace_z >> 0,
    ])
    options = {"solver": solver, "verbose": False}
    if solver == "SCS":
        options.update(eps=solver_tolerance, max_iters=100000)
    elif solver == "CLARABEL":
        options.update(tol_gap_abs=solver_tolerance, tol_gap_rel=solver_tolerance,
                       tol_feas=solver_tolerance, max_iter=300)
    primal.solve(**options)
    dual.solve(**options)
    valid_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if (primal.status not in valid_statuses or dual.status not in valid_statuses
            or rho.value is None or z.value is None or witness.value is None):
        raise RuntimeError(f"SDP failed: primal={primal.status}, dual={dual.status}")

    rho_raw = hermitian(rho.value)
    eig, vectors = np.linalg.eigh(rho_raw)
    clipped = np.maximum(eig, 0)
    if clipped.sum() <= 0:
        raise RuntimeError("SDP returned an invalid input density matrix")
    sqrt_rho = (vectors * np.sqrt(clipped / clipped.sum())) @ vectors.conj().T
    k = np.kron(sqrt_rho, np.eye(d))
    lower_input = scale * trace_norm(k @ j @ k) / 2

    z_raw = hermitian(z.value)
    min_dual = min(np.linalg.eigvalsh(z_raw - j).min(), np.linalg.eigvalsh(z_raw + j).min())
    safety = 1e-10 * max(1.0, np.linalg.norm(z_raw, 2))
    shift = max(0.0, -float(min_dual)) + safety
    z_feasible = z_raw + shift*np.eye(d*d)
    upper_dual = scale * float(np.linalg.eigvalsh(hermitian(partial_trace_output(z_feasible, d))).max()) / 2
    lower = max(lower_input, bounds["diamond_lower_choi"])
    upper = min(upper_dual, bounds["diamond_upper_abs_choi"])
    gap = upper - lower
    accepted = bool(-1e-12 <= gap <= gap_atol + gap_rtol * max(upper, 1e-15))
    # If numerical crossing is beyond roundoff, do not publish an estimate.
    sandwich_raw = np.kron(rho_raw, np.eye(d))
    primal_min = min(
        eig.min(),
        np.linalg.eigvalsh(hermitian(sandwich_raw - witness.value)).min(),
        np.linalg.eigvalsh(hermitian(sandwich_raw + witness.value)).min(),
    )
    return {
        "diamond_sdp_lower": float(lower), "diamond_sdp_upper": float(upper),
        "diamond_distance_estimate": float((lower + upper)/2) if accepted else np.nan,
        "diamond_sdp_gap": float(gap),
        "diamond_sdp_relative_gap": float(gap / max(upper, 1e-15)),
        "sdp_accepted": accepted,
        "sdp_primal_objective_distance": float(primal.value * scale / 2),
        "sdp_dual_objective_distance": float(dual.value * scale / 2),
        "sdp_primal_min_constraint_eigenvalue_scaled": float(primal_min),
        "sdp_primal_trace_error": float(abs(np.trace(rho_raw) - 1)),
        "sdp_dual_min_constraint_eigenvalue_scaled": float(min_dual),
        "sdp_dual_witness_shift_scaled": float(shift),
        "sdp_input_lower": float(lower_input),
        "sdp_repaired_dual_upper": float(upper_dual),
        "sdp_primal_status": primal.status, "sdp_dual_status": dual.status,
        "sdp_solver": solver,
    }
