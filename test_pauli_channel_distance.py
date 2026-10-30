"""Independent analytic and representation checks for channel distances."""

import numpy as np
import pytest
from scipy.linalg import expm

import pauli_channel_distance as distance


def unitary_chi(unitary, paulis):
    d = unitary.shape[0]
    coeff = np.einsum("aij,ij->a", paulis.conj(), unitary) / d
    return np.outer(coeff, coeff.conj())


def require_solver():
    cp = pytest.importorskip("cvxpy")
    if "SCS" not in cp.installed_solvers():
        pytest.skip("SCS is not installed")


def test_pauli_channel_is_its_own_twirl_and_identity_ratio_is_undefined():
    _, paulis = distance.pauli_basis()
    chi = np.diag([0.91] + [0.006]*15).astype(complex)
    values = distance.distance_metrics(chi, paulis)
    assert values["ptm_frobenius_distance"] == 0
    assert values["diamond_upper_abs_choi"] == 0
    assert values["relative_error_frobenius_distance"] == 0
    identity = np.zeros((16, 16), complex)
    identity[0, 0] = 1
    assert np.isnan(distance.distance_metrics(identity, paulis)["relative_error_frobenius_distance"])


def test_arbitrary_channel_representations_and_explicit_group_twirl():
    _, paulis = distance.pauli_basis()
    rng = np.random.default_rng(481)
    a = rng.normal(size=(12, 4)) + 1j*rng.normal(size=(12, 4))
    q, _ = np.linalg.qr(a)
    kraus = q.reshape(3, 4, 4)
    j = sum(np.outer(k.reshape(-1, order="F"), k.reshape(-1, order="F").conj()) for k in kraus)
    chi = distance.choi_to_chi(j, paulis)
    assert np.allclose(distance.chi_to_choi(chi, paulis), j, atol=1e-12)
    assert distance.channel_diagnostics(chi, paulis)["cptp_within_tolerance"]
    b = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
    rho = b @ b.conj().T
    rho /= np.trace(rho)
    exact = sum(k @ rho @ k.conj().T for k in kraus)
    assert np.allclose(distance.apply_chi(chi, rho, paulis), exact, atol=1e-12)
    group = sum(p @ distance.apply_chi(chi, p @ rho @ p, paulis) @ p for p in paulis) / 16
    diagonal = distance.apply_chi(distance.pauli_twirl_chi(chi), rho, paulis)
    assert np.allclose(group, diagonal, atol=1e-12)
    metrics = distance.distance_metrics(chi, paulis)
    assert metrics["ptm_chi_frobenius_identity_error"] < 1e-12
    assert metrics["ptm_twirl_diagonalization_error"] < 1e-12
    assert metrics["twirl_infidelity_difference"] == 0


def test_qutip_y_convention_matches_independent_choi_conversion():
    qp = pytest.importorskip("qutip")
    labels, paulis = distance.pauli_basis()
    # Odd-Y terms make a missing sign conversion detectable at order one.
    unitary = expm(-1j*(0.2*paulis[labels.index("YI")] + 0.13*paulis[labels.index("XZ")]))
    q = qp.Qobj(unitary, dims=[[2, 2], [2, 2]])
    stored = qp.to_chi(qp.to_super(q)).full() / 16
    canonical = distance.qutip_chi_to_canonical(stored, labels)
    assert np.allclose(canonical, unitary_chi(unitary, paulis), atol=1e-12)
    assert np.allclose(distance.chi_to_choi(canonical, paulis), qp.to_choi(q).full(), atol=1e-12)


@pytest.mark.parametrize("angle", [1e-4, 0.03, 0.2])
def test_coherent_xx_analytic_distance_and_infidelity(angle):
    labels, paulis = distance.pauli_basis()
    chi = unitary_chi(expm(-1j*angle*paulis[labels.index("XX")]), paulis)
    m = distance.distance_metrics(chi, paulis)
    expected = abs(np.sin(2*angle)) / 2
    assert np.isclose(m["diamond_lower_choi"], expected, atol=1e-12)
    assert np.isclose(m["diamond_upper_abs_choi"], expected, atol=1e-12)
    assert np.isclose(m["average_infidelity"], 4/5*np.sin(angle)**2, atol=1e-12)


def test_collective_dissipation_without_hamiltonian_is_non_pauli():
    labels, paulis = distance.pauli_basis()
    jump = paulis[labels.index("IX")] + paulis[labels.index("XI")]
    k = np.kron(jump.conj(), jump) - 0.5*np.kron(np.eye(4), jump@jump) - 0.5*np.kron((jump@jump).T, np.eye(4))
    superop = expm(0.01*k)
    # Build Choi by propagating matrix units, independently of chi conversion.
    j = np.zeros((16, 16), complex)
    for i in range(4):
        for l in range(4):
            matrix_unit = np.zeros((4, 4), complex)
            matrix_unit[i, l] = 1
            output = (superop @ matrix_unit.reshape(-1, order="F")).reshape(4, 4, order="F")
            j[i*4:(i+1)*4, l*4:(l+1)*4] = output
    chi = distance.choi_to_chi(j, paulis)
    assert distance.channel_diagnostics(chi, paulis)["cptp_within_tolerance"]
    assert distance.distance_metrics(chi, paulis)["diamond_lower_choi"] > 1e-3


def test_sdp_matches_pauli_channel_difference_total_variation():
    require_solver()
    _, paulis = distance.pauli_basis()
    p = np.array([0.85, 0.08, 0.07] + [0.0]*13)
    q = np.array([0.91, 0.02, 0.0, 0.07] + [0.0]*12)
    exact = np.abs(p-q).sum()/2
    result = distance.diamond_distance_sdp(distance.chi_to_choi(np.diag(p-q), paulis), 4)
    assert result["sdp_accepted"]
    assert result["diamond_sdp_lower"] <= exact + 1e-9
    assert result["diamond_sdp_upper"] >= exact - 1e-9
    assert abs(result["diamond_distance_estimate"] - exact) < 1e-7


def test_sdp_nonunital_amplitude_damping_vs_identity():
    require_solver()
    _, paulis = distance.pauli_basis(1)
    gamma = 0.13
    kraus = [np.diag([1, np.sqrt(1-gamma)]), np.array([[0, np.sqrt(gamma)], [0, 0]])]
    j = sum(np.outer(k.reshape(-1, order="F"), k.reshape(-1, order="F").conj()) for k in kraus)
    vec_i = np.eye(2).reshape(-1, order="F")
    delta = j - np.outer(vec_i, vec_i)
    result = distance.diamond_distance_sdp(delta, 2)
    assert result["sdp_accepted"]
    # For amplitude damping vs identity the optimum input is the excited state.
    assert abs(result["diamond_distance_estimate"] - gamma) < 1e-7


def test_sdp_coherent_xx_and_zero_map():
    require_solver()
    labels, paulis = distance.pauli_basis()
    angle = 0.07
    chi = unitary_chi(expm(-1j*angle*paulis[labels.index("XX")]), paulis)
    result = distance.diamond_distance_sdp(distance.chi_to_choi(chi-distance.pauli_twirl_chi(chi), paulis), 4)
    assert result["sdp_accepted"]
    assert abs(result["diamond_distance_estimate"] - abs(np.sin(2*angle))/2) < 1e-7
    assert distance.diamond_distance_sdp(np.zeros((16, 16)), 4)["diamond_distance_estimate"] == 0
