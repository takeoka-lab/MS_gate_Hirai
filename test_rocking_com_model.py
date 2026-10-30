"""Radial-model regressions, supplied but NOT executed for this request.

These tests check algebra, units, pulse closure and QPT conventions. They do
not integrate a master equation. Run explicitly when desired:
    python -m pytest test_rocking_com_model.py
"""

import numpy as np
import pytest
import qutip as qp

import ms_gate_functions as mg
import rocking_com_model as rc


def _ket(q1, q2, nr, nc, dr=5, dc=4):
    return qp.tensor(qp.basis(2, q1), qp.basis(2, q2), qp.basis(dr, nr), qp.basis(dc, nc))


def _element(op, bra, ket):
    return complex(op.matrix_element(bra.dag(), ket))


def test_radial_frequencies_and_per_ion_ld_factors():
    modes = rc.radial_mode_parameters()
    assert modes["rocking_frequency_hz"] == pytest.approx(4_664_761.51587624)
    assert modes["com_frequency_hz"] == 4.9e6
    assert modes["eta_rocking"] == pytest.approx(0.08236699897407851)
    assert modes["eta_com"] == pytest.approx(0.08036555080148682)
    assert modes["eta_com"] < modes["eta_rocking"]
    with pytest.raises(ValueError):
        rc.radial_mode_parameters(axial_frequency_hz=5e6)
    with pytest.raises(ValueError):
        rc.radial_mode_parameters(rocking_frequency_hz=5e6)
    with pytest.raises(ValueError):
        rc.radial_mode_parameters(radial_projection=1.01)


def test_two_pulses_close_the_main_mode_and_produce_the_negative_xx_target():
    pulse = rc.two_loop_ms_pulse()
    scale = pulse["t_gate_phys"] / pulse["t_gate_sim"]
    delta, amplitude = pulse["delta"] / scale, pulse["A"] / scale
    assert delta / (2 * np.pi) == pytest.approx(25e3)
    np.testing.assert_allclose(pulse["drive_intervals_phys"], [[0, 40e-6], [60e-6, 100e-6]])
    # Analytic phase-space displacements from the continuous-time Hamiltonian.
    displacements = [amplitude / delta * (np.exp(1j * delta * a) - np.exp(1j * delta * b))
                     for a, b in pulse["drive_intervals_phys"]]
    np.testing.assert_allclose(displacements, 0, atol=1e-14)
    # Closed individual loops have no cross-loop displacement phase.
    xx_phase = -2 * (amplitude / delta)**2 * sum(
        delta * (b - a) - np.sin(delta * (b - a)) for a, b in pulse["drive_intervals_phys"]
    )
    assert xx_phase == pytest.approx(-np.pi / 4)
    # A single uninterrupted 100 us pulse would leave a nonzero displacement.
    assert abs(amplitude / delta * (1 - np.exp(1j * delta * pulse["t_gate_phys"]))) > 0.1


def test_plan_resolves_com_detuning_and_preserves_exact_pulse_boundaries():
    plan = rc.prepare_simulation(
        **rc.two_loop_ms_pulse(), n_bar_list=[0.01], phonon_dim_override=16,
        com_phonon_dim_override=8, time_points=5,
    )
    scale = plan["rates"]["scale_factor"]
    np.testing.assert_allclose(plan["modes"]["delta_com_sim"] / (2 * np.pi * scale), 260238.4841237599)
    np.testing.assert_allclose(
        [[s["start"] * scale, s["stop"] * scale] for s in plan["segments"]],
        [[0, 40e-6], [40e-6, 60e-6], [60e-6, 100e-6]],
    )
    assert [s["drive_on"] for s in plan["segments"]] == [True, False, True]
    assert plan["conditions"][0]["hilbert_dimension"] == 512
    assert plan["solver_max_step"] * scale <= 1 / (40 * 260238.4841237599) * (1 + 1e-12)


def test_rocking_is_antisymmetric_and_com_is_symmetric():
    ops = rc.rocking_com_operators(5, 0.08, phonon_dim_com=4, eta_com=0.07, use_full_order=False)
    for mode, excited in (("rocking", _ket(0, 0, 1, 0)), ("com", _ket(0, 0, 0, 1))):
        lower = ops[mode]["Hx_minus"]
        assert _element(lower, _ket(1, 0, 0, 0), excited) == pytest.approx(1)
        assert _element(lower, _ket(0, 1, 0, 0), excited) == pytest.approx(
            -1 if mode == "rocking" else 1
        )
        assert lower.dag() == ops[mode]["Hx_plus"]


def test_cross_ld_correction_keeps_the_physical_mode_occupations():
    er, ec, nr, nc = 0.08, 0.07, 2, 3
    ops = rc.rocking_com_operators(5, er, phonon_dim_com=4, eta_com=ec, use_full_order=True)
    initial = _ket(0, 0, nr, nc)
    rocking = _element(ops["rocking"]["Hx_minus"], _ket(0, 1, nr - 1, nc), initial)
    com = _element(ops["com"]["Hx_minus"], _ket(0, 1, nr, nc - 1), initial)
    assert rocking == pytest.approx(-np.sqrt(nr) * (1 - er**2 * nr / 2 - ec**2 * (nc + 0.5)))
    assert com == pytest.approx(np.sqrt(nc) * (1 - ec**2 * nc / 2 - er**2 * (nr + 0.5)))


def test_one_mode_uses_rocking_and_decoupled_com_recovers_it():
    one = rc.rocking_com_operators(5, 0.08, use_full_order=True)
    two = rc.rocking_com_operators(5, 0.08, phonon_dim_com=4, eta_com=0.0, use_full_order=True)
    z2 = qp.tensor(qp.qeye(2), qp.sigmaz(), qp.qeye(5))
    old = mg._ms_gate_static_operators(5, 0.08, True)
    assert (one["rocking"]["Hx_minus"] - z2 * old["Hx_minus"] * z2).norm() < 1e-12
    times = np.array([0.0, 1.0, 2.0])
    h1 = rc.build_hamiltonian(one, times, 0.5, 0.3, 0.125)
    h2 = rc.build_hamiltonian(two, times, 0.5, 0.3, 0.125,
                              mode_gap_sim=107, com_amplitude_ratio=0.0)
    for t in [0.123, 0.456, 1.789]:
        assert (h2(t) - qp.tensor(h1(t), qp.qeye(4))).norm() < 1e-12


def test_coarse_grid_preserves_both_mode_phases_and_hermiticity():
    ops = rc.rocking_com_operators(5, 0.08, phonon_dim_com=4, eta_com=0.07, use_full_order=False)
    times = np.array([0.0, 1.0, 2.0])
    ratio, gap, delta, amplitude = 0.7, 107.0, 0.5, 0.125
    h = rc.build_hamiltonian(ops, times, delta, 0.0, amplitude,
                             mode_gap_sim=gap, com_amplitude_ratio=ratio)
    for t in [0.123, 0.456, 1.789]:
        target = _ket(1, 0, 0, 0)
        r = _element(h(t), target, _ket(0, 0, 1, 0))
        c = _element(h(t), target, _ket(0, 0, 0, 1))
        assert r == pytest.approx(amplitude * np.exp(-1j * delta * t))
        assert c == pytest.approx(ratio * amplitude * np.exp(-1j * (delta + gap) * t))
        assert h(t).isherm


def test_single_mode_ignores_com_occupation_cutoff_and_noise():
    plan = rc.prepare_simulation(
        use_two_modes=False, **rc.two_loop_ms_pulse(), n_bar_list=[0.01],
        phonon_dim_override=16, n_bar_com=-1, com_phonon_dim_override=-1,
        com_heating_rate_phys=-1, com_dephasing_rate_phys=-1,
    )
    assert plan["conditions"][0]["Nv_com"] is None
    assert plan["conditions"][0]["hilbert_dimension"] == 64
    assert plan["rates"]["com_heating"] == 0


def test_spin_noise_is_added_once_and_idle_disables_the_force(monkeypatch):
    captured = []
    monkeypatch.setattr(qp, "MESolver", lambda h, c_ops, options: captured.append((h, c_ops)))
    plan = rc.prepare_simulation(
        **rc.two_loop_ms_pulse(), n_bar_list=[0.01], phonon_dim_override=3,
        com_phonon_dim_override=3, heating_rate_phys=1, com_heating_rate_phys=2,
        dephasing_rate_phys=3, com_dephasing_rate_phys=4,
        T2_star=0.3, rayleigh_rate_phys=5, raman_rate_phys=6,
    )
    laser = {"detuning": plan["detuning"], "int_strn": plan["amplitude"], "rho": plan["rho"]}
    rc.build_solver(plan, plan["conditions"][0], laser, drive_on=True)
    rc.build_solver(plan, plan["conditions"][0], laser, drive_on=False)
    # Six motion jumps and eight spin jumps; no second copy of the latter.
    assert len(captured[0][1]) == 14
    assert captured[1][0](0.37).norm() == pytest.approx(0.0)
    # The six optical-scattering jumps are zero in the OFF interval; motional
    # and magnetic terms (2*3 + 2 = 8) remain nonzero.
    idle_ops = captured[1][1]
    assert sum((op(0.37) if isinstance(op, qp.QobjEvo) else op).norm() > 0 for op in idle_ops) == 8


def test_negative_target_is_used_for_fidelity_and_error_chi(monkeypatch):
    ideal = mg.ideal_ms_gate(phi=-np.pi / 4)
    inputs = mg.build_input_states()
    simulation = {
        "parameters": {"n_bar_list": [0.01]}, "input_states_list": inputs,
        "results_list": [{"n_bar": 0.01, "outputs": [ideal * rho * ideal.dag() for rho in inputs]}],
    }
    monkeypatch.setattr(rc, "run_ms_gate_simulation", lambda **kwargs: simulation)
    result = rc.run_analysis(execute=True)
    assert result["f_avg_list"][0] == pytest.approx(1.0)
    np.testing.assert_allclose(result["error_ptm_list"][0], np.eye(16), atol=1e-12)
    assert result["chi_matrix_list"][0].shape == (16, 16)


def test_execute_false_never_prepares_or_simulates(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("No model preparation or evolution was authorized.")

    monkeypatch.setattr(rc, "prepare_simulation", forbidden)
    monkeypatch.setattr(rc, "run_ms_gate_simulation", forbidden)
    assert rc.run_analysis(execute=False) is None
