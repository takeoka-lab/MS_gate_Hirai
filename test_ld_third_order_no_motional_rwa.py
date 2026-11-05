"""Hamiltonian-only checks for D; no channel propagation."""

import numpy as np
import qutip as qp

import ld_second_order_no_motional_rwa as model_c
import ld_third_order_no_motional_rwa as model_d


def test_cubic_term_is_projected_after_cubing_and_resolves_fastest_tone():
    nv, phi = 5, 0.31
    resolved = model_d.resolve_parameters({"rho0": phi, "eta": 0.14})
    nu, mu = resolved["nu_sim"], resolved["mu_sim"]
    fastest = 3 * nu + mu
    assert resolved["fastest_angular_frequency_sim"] == fastest
    assert resolved["solver_options"]["max_step"] <= 2*np.pi / (24 * fastest)
    hc = model_c.build_hamiltonian(nv, resolved)
    hd = model_d.build_hamiltonian(nv, resolved)
    a = qp.destroy(nv + 3).full()
    local = np.cos(phi)*qp.sigmax().full() + np.sin(phi)*qp.sigmay().full()
    spin = np.kron(local, np.eye(2)) + np.kron(np.eye(2), local)
    for t in (0.0, 0.013, 0.109):
        lowering = a * np.exp(-1j*nu*t)
        q = lowering + lowering.conj().T
        projected_cube = (q @ q @ q)[:nv, :nv]
        coefficient = -resolved["omega0_sim"] * np.cos(mu*t) * 0.14**3 / 6
        expected = hc(t).full() + coefficient * np.kron(spin, projected_cube)
        np.testing.assert_allclose(hd(t).full(), expected, rtol=0, atol=1e-13)
