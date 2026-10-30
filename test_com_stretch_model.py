"""Checks for the COM/stretch extension. Added without executing simulations.

Run explicitly when wanted: python -m pytest test_com_stretch_model.py
Only the final test integrates master equations; other tests check operators,
units, noise multiplicity, interpolation, or the unchanged one-mode dispatch.
"""

import numpy as np
import pytest
import qutip as qp

import com_stretch_model as cs
import ms_gate_functions as mg


def _ket(q1, q2, nc, ns, dc=5, ds=5):
    return qp.tensor(qp.basis(2, q1), qp.basis(2, q2),
                     qp.basis(dc, nc), qp.basis(ds, ns))


def _element(operator, bra, ket):
    return complex(operator.matrix_element(bra.dag(), ket))


def test_single_mode_dispatch_is_the_original_function(monkeypatch):
    sentinel = object()
    calls = []

    def original(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(mg, "run_ms_gate_simulation", original)
    result = cs.run_ms_gate_simulation(
        use_two_modes=False, A=0.125, n_bar_list=[0.01],
        stretch_frequency_hz=-1, n_bar_stretch=-3,  # ignored in one-mode mode
    )
    assert result is sentinel
    assert calls == [{"A": 0.125, "n_bar_list": [0.01]}]


def test_execute_false_does_not_prepare_or_run(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("The simulation must not be called.")

    monkeypatch.setattr(cs, "run_ms_gate_simulation", forbidden)
    assert cs.run_analysis(execute=False, use_two_modes=True) is None


def test_axial_frequencies_and_lamb_dicke_ratio():
    modes = cs.axial_mode_parameters(1e6, 0.1)
    assert modes["stretch_frequency_hz"] == pytest.approx(np.sqrt(3) * 1e6)
    assert modes["eta_stretch"] == pytest.approx(0.1 / 3**0.25)
    assert modes["stretch_amplitude_ratio"] == pytest.approx(3**(-0.25))


def test_mode_gap_is_scaled_into_the_same_units_as_the_existing_detuning():
    plan = cs.prepare_simulation(
        use_two_modes=True, com_frequency_hz=1e6, stretch_frequency_hz=1.7e6,
        t_gate_phys=100e-6, delta=0.5, n_bar_list=[0.01], time_points=5,
    )
    expected = 2 * np.pi * 0.7e6 * 100e-6 / (2 * np.pi / 0.5)
    assert plan["modes"]["mode_gap_sim"] == pytest.approx(expected)
    np.testing.assert_allclose(plan["modes"]["delta_stretch_sim"], 0.5 + expected)
    assert plan["solver_max_step"] <= 2 * np.pi / (40 * (0.5 + expected))


def test_com_and_stretch_spin_couplings_have_correct_relative_signs():
    ops = cs.com_stretch_operators(5, 5, 0.1, 0.08, use_full_order=False)
    ground = _ket(0, 0, 0, 0)
    for mode, excited in (("com", _ket(0, 0, 1, 0)), ("stretch", _ket(0, 0, 0, 1))):
        lowering = ops[mode]["Hx_minus"]
        assert _element(lowering, _ket(1, 0, 0, 0), excited) == pytest.approx(1)
        assert _element(lowering, _ket(0, 1, 0, 0), excited) == pytest.approx(
            1 if mode == "com" else -1
        )
        assert lowering.dag() == ops[mode]["Hx_plus"]
    assert (ops["com"]["Hx_minus"] * ground).norm() == 0


def test_finite_ld_correction_includes_spectator_occupation_and_zero_point():
    ec, es, nc, ns = 0.10, 0.08, 2, 3
    ops = cs.com_stretch_operators(5, 5, ec, es, use_full_order=True)
    initial = _ket(0, 0, nc, ns)
    com = _element(ops["com"]["Hx_minus"], _ket(1, 0, nc - 1, ns), initial)
    stretch = _element(ops["stretch"]["Hx_minus"], _ket(1, 0, nc, ns - 1), initial)
    assert com == pytest.approx(np.sqrt(nc) * (1 - ec**2 * nc / 2 - es**2 * (ns + 0.5)))
    assert stretch == pytest.approx(np.sqrt(ns) * (1 - es**2 * ns / 2 - ec**2 * (nc + 0.5)))


def test_decoupled_stretch_reduces_to_existing_hamiltonian():
    times = np.linspace(0, 2, 11)
    ops = cs.com_stretch_operators(5, 3, 0.1, 0.0, use_full_order=True)
    two = cs.build_two_mode_hamiltonian(
        ops, times, 0.5, 0.3, 0.125, mode_gap_sim=9, stretch_amplitude_ratio=0,
    )
    old = mg._build_ms_hamiltonian(
        mg._ms_gate_static_operators(5, 0.1, True), times, 0.5, 0.3, 0.125
    )
    for time in (0, 0.73, 2):
        assert (two(time) - qp.tensor(old(time), qp.qeye(3))).norm() < 1e-12


def test_fast_spectator_phase_is_not_aliased_by_a_coarse_output_grid():
    times = np.linspace(0, 2, 3)
    gap, ratio = 107.0, 0.7
    ops = cs.com_stretch_operators(5, 5, 0.1, 0.08, use_full_order=False)
    h = cs.build_two_mode_hamiltonian(
        ops, times, 0.5, 0.0, 0.125, mode_gap_sim=gap, stretch_amplitude_ratio=ratio,
    )
    for t in (0.123, 0.456, 1.789):
        assert h(t).isherm
        target = _ket(1, 0, 0, 0)
        com = _element(h(t), target, _ket(0, 0, 1, 0))
        stretch = _element(h(t), target, _ket(0, 0, 0, 1))
        assert stretch / com == pytest.approx(ratio * np.exp(-1j * gap * t))


def test_spin_noise_is_not_duplicated_when_second_mode_is_added(monkeypatch):
    captured = {}

    def capture(hamiltonian, c_ops, options):
        captured["c_ops"] = c_ops
        return captured

    plan = cs.prepare_simulation(
        use_two_modes=True, n_bar_list=[0], phonon_dim_override=3,
        stretch_phonon_dim_override=3, heating_rate_phys=1, dephasing_rate_phys=2,
        stretch_heating_rate_phys=3, stretch_dephasing_rate_phys=4,
        T2_star=0.3, rayleigh_rate_phys=3, raman_rate_phys=1,
    )
    monkeypatch.setattr(qp, "MESolver", capture)
    cs.build_two_mode_solver(plan, plan["conditions"][0], {
        "detuning": plan["detuning"], "int_strn": plan["amplitude"], "rho": plan["rho"],
    })
    # Each mode: a, a.dag(), N (6 total); spin Z (2), Rayleigh Z (2), Raman +/- (4).
    assert len(captured["c_ops"]) == 14
    assert all(op.dims == [[2, 2, 3, 3], [2, 2, 3, 3]] for op in captured["c_ops"])


def test_independent_thermal_occupations_and_invalid_inputs():
    plan = cs.prepare_simulation(
        use_two_modes=True, n_bar_list=[0, 1], n_bar_stretch=[0.01, 0.2],
        phonon_dim_override=8, stretch_phonon_dim_override=4,
    )
    assert [r["n_bar_stretch"] for r in plan["conditions"]] == [0.01, 0.2]
    assert plan["conditions"][1]["initial_com_tail"] == pytest.approx(0.5**8)
    with pytest.raises(ValueError):
        cs.prepare_simulation(use_two_modes=True, n_bar_list=[0, 1], n_bar_stretch=[0.1])
    with pytest.raises(ValueError):
        cs.axial_mode_parameters(1e6, 0.1, stretch_frequency_hz=0.5e6)
    with pytest.raises(TypeError):
        cs.run_ms_gate_simulation(use_two_modes="False")


def test_zero_stretch_coupling_recovers_single_mode_qpt():
    """Small integration regression, to run only when simulation is authorized."""
    params = dict(
        A=0.125, delta=0.5, t_gate_sim=0.3, time_points=11,
        n_bar_list=[0.01], phonon_dim_override=3, eta=0.1, use_full_order=True,
        heating_rate_phys=0, dephasing_rate_phys=0, T2_star=np.inf,
        rayleigh_rate_phys=0, raman_rate_phys=0, show_progress=False, parallel_workers=1,
        solver_max_step=0.003,
    )
    one = cs.run_ms_gate_simulation(use_two_modes=False, **params)
    two = cs.run_ms_gate_simulation(
        use_two_modes=True, com_frequency_hz=1e3, eta_stretch=0,
        stretch_phonon_dim_override=3, n_bar_stretch=0.1, **params,
    )
    for a, b in zip(one["results_list"][0]["outputs"], two["results_list"][0]["outputs"]):
        assert (a - b).norm() < 1e-7
