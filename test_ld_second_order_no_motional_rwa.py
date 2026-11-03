"""Independent checks of the model-C phases, normalization, and reduction."""

import numpy as np
import pytest
import qutip as qp

import ld_second_order_no_motional_rwa as model_c
import ms_gate_functions as mg


def test_frequency_units_and_first_sideband_rabi_normalization():
    resolved = model_c.resolve_parameters({})
    assert resolved["nu_sim"] == pytest.approx(50)
    assert resolved["mu_sim"] == pytest.approx(49.5)
    assert resolved["omega0_sim"] == pytest.approx(2.5)
    assert resolved["omega0_over_2pi_hz"] == pytest.approx(50_000)
    assert resolved["delta_over_2pi_hz"] == pytest.approx(10_000)
    assert resolved["bichromatic_offset_hz"] == pytest.approx(990_000)
    assert resolved["solver_options"]["max_step"] <= 2 * np.pi / (24 * 149.5)


def test_direct_bichromatic_taylor_expression_and_hermiticity():
    # Build sigma_+ * exp(i*eta*Q) * (red + blue) + h.c.
    # independently, in a larger oscillator basis before truncation.
    nv, large = 5, 8
    resolved = model_c.resolve_parameters({"rho0": 0.31, "eta": 0.14})
    p = resolved["parameters"]
    h = model_c.build_hamiltonian(nv, resolved)
    a = qp.destroy(large).full()
    sigma_plus = (qp.tensor(qp.sigmap(), qp.qeye(2))
                  + qp.tensor(qp.qeye(2), qp.sigmap())).full()
    phi = -np.pi / 2 - p["rho0"]
    for t in (0.0, 0.017, 0.231):
        q = a * np.exp(-1j * resolved["nu_sim"] * t)
        q += a.conj().T * np.exp(1j * resolved["nu_sim"] * t)
        taylor = np.eye(large) + 1j * p["eta"] * q - p["eta"]**2 / 2 * (q @ q)
        tones = np.exp(1j * resolved["mu_sim"] * t + 1j * phi)
        tones += np.exp(-1j * resolved["mu_sim"] * t + 1j * phi)
        positive = resolved["omega0_sim"] / 2 * tones * np.kron(sigma_plus, taylor[:nv, :nv])
        expected = positive + positive.conj().T
        assert np.allclose(h(t).full(), expected, atol=2e-13)
        assert np.allclose(h(t).full(), h(t).full().conj().T, atol=2e-13)


def test_fourier_selection_recovers_existing_model_A_and_keeps_fast_terms():
    nv = 4
    resolved = model_c.resolve_parameters({"rho0": 0.27})
    h = model_c.build_hamiltonian(nv, resolved)
    delta = resolved["parameters"]["delta"]
    duration = resolved["parameters"]["t_gate_sim"]
    # nu/delta=100: every retained Fourier frequency is an integer gate harmonic.
    times = np.arange(1024) * duration / 1024
    matrices = np.stack([h(t).full() for t in times])
    minus = np.einsum("t,tij->ij", np.exp(1j * delta * times), matrices) / len(times)
    plus = np.einsum("t,tij->ij", np.exp(-1j * delta * times), matrices) / len(times)
    ops = mg._ms_gate_static_operators(nv, 0.1, False)
    target = 0.125 * (np.cos(0.27) * ops["Hx_minus"] + np.sin(0.27) * ops["Hy_minus"])
    assert np.allclose(minus, target.full(), atol=1e-12)
    assert np.allclose(plus, target.dag().full(), atol=1e-12)
    reconstructed_rwa = (np.exp(-1j * delta * times)[:, None, None] * minus
                         + np.exp(1j * delta * times)[:, None, None] * plus)
    assert np.linalg.norm(matrices - reconstructed_rwa) > 1  # C really retains fast terms


def test_unitary_thermal_reduction_matches_direct_partial_trace():
    rng = np.random.default_rng(731)
    nv = 3
    z = rng.normal(size=(4 * nv, 4 * nv)) + 1j * rng.normal(size=(4 * nv, 4 * nv))
    unitary = qp.Qobj(np.linalg.qr(z)[0], dims=[[2, 2, nv], [2, 2, nv]])
    channel = model_c.channel_from_unitary(unitary, 0.37, nv)
    for rho in mg.build_input_states()[::3]:
        joint = qp.tensor(rho, qp.thermal_dm(nv, 0.37))
        expected = (unitary * joint * unitary.dag()).ptrace([0, 1])
        actual = qp.vector_to_operator(channel * qp.operator_to_vector(rho))
        assert np.allclose(actual.full(), expected.full(), atol=2e-14)
    metrics = mg.choi_physicality_metrics(channel)
    assert metrics["tp_frobenius_error"] < 1e-12
    assert metrics["min_choi_eigenvalue"] > -1e-12


@pytest.mark.parametrize("parameters", [
    {"eta": 0}, {"laser_intensity_fluctuation": 0.01},
    {"A": [0.1, 0.2]}, {"unknown_rate": 1.0},
])
def test_unsupported_settings_are_not_silently_ignored(parameters):
    with pytest.raises(ValueError):
        model_c.resolve_parameters(parameters)
