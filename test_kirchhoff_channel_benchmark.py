"""Physical convention tests for the unified-fidelity benchmark."""
from dataclasses import replace
import numpy as np
from scipy.linalg import expm

import kirchhoff_channel_benchmark as benchmark
import kirchhoff_paper_infidelity as paper


def test_environment_only_phase_is_not_a_qubit_gate_error():
    model = paper.build_paper_model(phonon_dim=3)
    environment = np.diag(np.exp(1j*np.array([0.0, 1.3, 2.4])))
    unitary = np.kron(environment, model.target_qubit_unitary)
    chi, row = benchmark.channel_from_propagator(unitary, model, 0.7)
    assert abs(row['standard_average_infidelity']) < 1e-12
    assert row['paper_average_infidelity'] > 0.1
    assert abs(chi[0, 0]-1) < 1e-12
    assert row['pauli_ptm_frobenius'] < 1e-12


def test_factorized_coherent_error_has_expected_fidelity_and_generator():
    model = paper.build_paper_model(phonon_dim=3)
    theta = 0.037
    yy = benchmark.PAULIS[benchmark.LABELS.index('YY')]
    actual = expm(-1j*theta*yy) @ model.target_qubit_unitary
    unitary = np.kron(np.eye(3), actual)
    _, row = benchmark.channel_from_propagator(unitary, model, 0.2)
    assert np.isclose(row['standard_average_infidelity'], 4/5*np.sin(theta)**2, atol=1e-12)
    assert np.isclose(row['paper_average_overlap_fidelity'], np.cos(theta), atol=1e-12)
    assert np.isclose(row['h_XX_rad_per_gate'], theta, atol=1e-12)


def test_entangled_environment_channel_matches_direct_pure_state_average():
    model = paper.build_paper_model(phonon_dim=3)
    rng = np.random.default_rng(914)
    a = rng.normal(size=(12, 12))+1j*rng.normal(size=(12, 12))
    unitary = expm(-0.12j*(a+a.conj().T))
    _, row = benchmark.channel_from_propagator(unitary, model, 0.4)
    weights, _ = paper.thermal_weights(3, 0.4)
    states = rng.normal(size=(4, 16000))+1j*rng.normal(size=(4, 16000))
    states /= np.linalg.norm(states, axis=0)
    target = model.target_qubit_unitary @ states
    fidelity = np.zeros(states.shape[1])
    for n, weight in enumerate(weights):
        initial = np.zeros((12, states.shape[1]), complex)
        initial[4*n:4*(n+1)] = states
        final = (unitary@initial).reshape(3, 4, -1)
        fidelity += weight*np.sum(abs(np.einsum('is,mis->ms',target.conj(),final))**2, axis=0)
    assert abs(np.mean(fidelity)-row['standard_average_fidelity']) < 0.004
    assert row['fidelity_crosscheck_error'] < 1e-12


def test_legacy_cache_key_contains_physical_parameters():
    settings = benchmark.Settings()
    omega = settings.landmarks()['Omega_4']
    path = benchmark.legacy_path(settings, omega)
    assert path.name == 'omega_mhz_1p089192470__afb5f707a84049.npz'
    assert benchmark.legacy_path(replace(settings, eta=0.19), omega) != path
    assert benchmark.legacy_path(replace(settings, phonon_dim=12), omega) != path
    assert benchmark.legacy_path(replace(settings, maximum_step_divisor=40), omega) is None
