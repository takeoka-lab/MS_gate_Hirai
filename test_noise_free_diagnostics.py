import numpy as np
import pandas as pd

import noise_free_diagnostics as diagnostics


def _small_parameters():
    return {
        "A": 0.125,
        "delta": 0.5,
        "rho0": 0.0,
        "n_bar_list": [0.01],
        "time_points": 31,
        "t_gate_phys": 100e-6,
        "heating_rate_phys": 10.0,
        "dephasing_rate_phys": 18.0,
        "T2_star": 0.3,
        "rayleigh_rate_phys": 3.0,
        "raman_rate_phys": 1.0,
        "eta": 0.1,
        "laser_intensity_fluctuation": 0.0,
        "laser_detuning_fluctuation": 0.0,
        "laser_rotation_angle_fluctuation": 0.0,
        "laser_noise_samples": 1,
        "laser_noise_seed": 1234,
        "use_full_order": True,
        "show_progress": False,
        "parallel_workers": 1,
    }


def test_noise_free_parameters_disable_all_four_noise_sources():
    source = _small_parameters()
    result = diagnostics.noise_free_parameters(source, nbar_values=[0.01, 4.0])

    assert source["heating_rate_phys"] == 10.0
    assert result["heating_rate_phys"] == 0.0
    assert result["dephasing_rate_phys"] == 0.0
    assert np.isinf(result["T2_star"])
    assert result["rayleigh_rate_phys"] == 0.0
    assert result["raman_rate_phys"] == 0.0
    assert result["n_bar_list"] == [0.01, 4.0]


def test_full_vs_ld_smoke_run_creates_comparison(tmp_path):
    full = pd.DataFrame(
        {"nbar": [0.01], "F_avg": [0.999], "infidelity": [0.001]}
    )
    result = diagnostics.run_full_vs_ld_comparison(
        output_dir=tmp_path / "full_vs_ld",
        base_parameters=_small_parameters(),
        nbar_values=[0.01],
        full_order_summary=full,
        execute=True,
    )

    assert len(result["comparison"]) == 1
    assert np.isfinite(result["comparison"].iloc[0]["infidelity_lamb_dicke"])
    assert result["figure_path"].exists()


def test_fock_and_closure_smoke_run(tmp_path):
    parameters = _small_parameters()
    fock = diagnostics.run_fock_resolved_analysis(
        output_dir=tmp_path / "fock",
        base_parameters=parameters,
        nbar_values=[0.01],
        execute=True,
        max_fock_n=2,
        phonon_buffer=8,
    )
    closure = diagnostics.build_phase_space_closure_analysis(
        output_dir=tmp_path / "closure",
        base_parameters=parameters,
        fock_curve=fock["curve"],
        fock_thermal_summary=fock["thermal_summary"],
    )

    assert fock["curve"]["fock_n"].tolist() == [0, 1, 2]
    assert len(closure["summary"]) == 1
    assert np.isfinite(closure["magnus"]["displacement_abs"])
    assert np.isclose(
        closure["trajectory"]["alpha_abs_per_spin_eigenvalue"].iloc[-1],
        closure["magnus"]["displacement_abs"],
    )
    assert closure["figure_path"].exists()


def test_convergence_builds_and_analyzes_requested_grid(tmp_path, monkeypatch):
    def fake_infidelity(parameters, n_bar):
        return (
            0.01
            + 1.0 / float(parameters["phonon_dim_override"])
            + 1.0 / float(parameters["time_points"])
            + 1e-4 * float(n_bar)
        )

    monkeypatch.setattr(diagnostics, "_single_qpt_infidelity", fake_infidelity)
    result = diagnostics.run_numerical_convergence(
        output_dir=tmp_path / "convergence",
        base_parameters=_small_parameters(),
        nbar_values=[0.01],
        execute=True,
        phonon_dim_factors=(1.0, 1.5),
        time_point_factors=(1.0, 2.0),
    )

    assert len(result["summary"]) == 4
    assert set(result["summary"]["sweep"]) == {"phonon_cutoff", "time_grid"}
    assert np.all(result["summary"]["abs_delta_to_reference"] >= 0.0)
    assert result["figure_path"].exists()
