import numpy as np

import drive_calibration_qpt_analysis as qpt_analysis
import error_generator_rate_nbar as analysis


def _base_parameters():
    return {
        "A": 0.125,
        "delta": 0.5,
        "rho0": 0.0,
        "n_bar_list": [0.01, 1.0, 2.0, 3.0, 4.0],
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


def test_rate_plan_has_twenty_requests_and_seventeen_unique_conditions():
    plan = analysis.build_single_axis_rate_plan(_base_parameters())

    assert len(plan["requests"]) == 20
    assert len(plan["catalog"]) == 17
    nominal = plan["requests"][np.isclose(plan["requests"]["multiplier"], 1.0)]
    assert nominal["condition_id"].nunique() == 1
    for source in analysis.NOISE_SOURCES:
        source_rows = plan["requests"][plan["requests"]["noise_source"].eq(source)]
        assert len(source_rows) == 5
        other_sources = set(analysis.NOISE_SOURCES) - {source}
        for other in other_sources:
            np.testing.assert_allclose(
                source_rows[analysis.RATE_COLUMNS[other]],
                plan["nominal_strengths"][other],
            )


def test_dry_run_reports_pointwise_qpt_budget(tmp_path):
    result = analysis.run_or_load_qpt_sweep(
        output_dir=tmp_path,
        base_parameters=_base_parameters(),
        nbar_values=[0.01, 1.0],
        execute=False,
    )

    assert not result["complete"]
    assert result["total_point_count"] == 34
    assert result["pending_nbar_count"] == 34
    assert result["pending_master_equation_evolutions"] == 34 * 16
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "condition_catalog.csv").exists()
    assert (tmp_path / "sweep_requests.csv").exists()


def test_identity_cache_extracts_four_nominal_views(tmp_path):
    base = _base_parameters()
    initial = analysis.run_or_load_qpt_sweep(
        output_dir=tmp_path,
        base_parameters=base,
        nbar_values=[0.01],
        rate_multipliers=[1.0],
        execute=False,
    )
    condition_id = initial["plan"]["catalog"].iloc[0]["condition_id"]
    chi = np.zeros((16, 16), dtype=complex)
    chi[0, 0] = 1.0
    qpt_analysis.save_qpt_point(
        analysis.qpt_cache_path(tmp_path, condition_id, 0.01),
        0.01,
        condition_id,
        chi,
        {"test": True},
    )

    complete = analysis.run_or_load_qpt_sweep(
        output_dir=tmp_path,
        base_parameters=base,
        nbar_values=[0.01],
        rate_multipliers=[1.0],
        execute=False,
    )
    generators = analysis.extract_cached_generators(complete)

    assert complete["complete"]
    assert len(generators["condition_summary"]) == 1
    assert len(generators["summary"]) == 4
    assert len(generators["coefficients"]) == 4 * 15
    assert set(generators["summary"]["noise_source"]) == set(
        analysis.NOISE_SOURCES
    )
    np.testing.assert_allclose(
        generators["summary"]["average_infidelity"], 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        generators["coefficients"]["gamma_per_gate"], 0.0, atol=1e-12
    )
