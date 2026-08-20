from itertools import combinations

import numpy as np
import pandas as pd

import pairwise_noise_correlation as correlation


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


def test_plan_has_six_bidirectional_pairs_and_59_unique_conditions():
    plan = correlation.build_pairwise_sweep_plan(_base_parameters())

    assert len(plan["pairs"]) == 6
    assert len(plan["pair_requests"]) == 6 * 2 * 5
    assert len(plan["single_requests"]) == 4 * 5
    assert len(plan["catalog"]) == 59
    assert plan["catalog"]["is_all_noise_zero"].sum() == 1
    assert plan["pair_requests"]["condition_id"].nunique() == 46


def test_rate_vector_parameters_zero_unselected_sources_and_split_scattering():
    condition = {
        "motional_heating_s^-1": 0.0,
        "motional_dephasing_s^-1": 36.0,
        "spin_dephasing_s^-1": 0.0,
        "photon_scattering_s^-1": 8.0,
    }
    params = correlation.parameters_for_rate_vector(
        _base_parameters(), condition, nbar_values=[0.01, 4.0]
    )

    assert params["heating_rate_phys"] == 0.0
    assert params["dephasing_rate_phys"] == 36.0
    assert np.isinf(params["T2_star"])
    assert params["rayleigh_rate_phys"] == 6.0
    assert params["raman_rate_phys"] == 2.0
    assert params["n_bar_list"] == [0.01, 4.0]


def test_interaction_formula_recovers_synthetic_pair_coefficients():
    plan = correlation.build_pairwise_sweep_plan(_base_parameters())
    nominal = plan["nominal_strengths"]
    main = {
        source: (index + 1) * 1e-4
        for index, source in enumerate(correlation.NOISE_SOURCES)
    }
    pair_coefficients = {
        tuple(sorted(pair)): (index + 1) * 1e-5
        for index, pair in enumerate(combinations(correlation.NOISE_SOURCES, 2))
    }
    rate_columns = {
        "motional_heating": "motional_heating_s^-1",
        "motional_dephasing": "motional_dephasing_s^-1",
        "spin_dephasing": "spin_dephasing_s^-1",
        "photon_scattering": "photon_scattering_s^-1",
    }
    rows = []
    for _, condition in plan["catalog"].iterrows():
        scaled = {
            source: float(condition[rate_columns[source]]) / nominal[source]
            for source in correlation.NOISE_SOURCES
        }
        infidelity = 0.002 + sum(
            main[source] * scaled[source]
            for source in correlation.NOISE_SOURCES
        )
        for pair, coefficient in pair_coefficients.items():
            infidelity += coefficient * scaled[pair[0]] * scaled[pair[1]]
        rows.append(
            {
                "condition_id": condition["condition_id"],
                "nbar": 0.01,
                "infidelity": infidelity,
            }
        )
    interactions = correlation.calculate_pairwise_interactions(
        plan, pd.DataFrame(rows)
    )

    assert len(interactions) == 60
    for _, row in interactions.iterrows():
        pair = tuple(sorted((row["source_i"], row["source_j"])))
        expected = pair_coefficients[pair] * row["varied_multiplier"]
        assert np.isclose(row["interaction_infidelity"], expected, atol=1e-15)


def test_dry_run_reuses_all_noise_zero_and_reports_4640_pending_evolutions(
    tmp_path,
):
    nbar_values = [0.01, 1.0, 2.0, 3.0, 4.0]
    zero = pd.DataFrame(
        {
            "condition": "all_four_noises_off",
            "nbar": nbar_values,
            "F_avg": np.full(5, 0.999),
            "infidelity": np.full(5, 0.001),
        }
    )
    result = correlation.run_pairwise_noise_correlation_sweep(
        output_dir=tmp_path,
        base_parameters=_base_parameters(),
        nbar_values=nbar_values,
        all_noise_zero_summary=zero,
        execute=False,
        resume=True,
    )

    assert len(result["summary"]) == 5
    assert result["pending_unique_conditions"] == 58
    assert result["pending_master_equation_evolutions"] == 4640
    assert not result["complete"]
    assert (tmp_path / "condition_catalog.csv").exists()
    assert (tmp_path / "pairwise_plot_requests.csv").exists()


def test_heating_dephasing_grid_has_25_points_and_reuses_16_pairwise_points():
    base = _base_parameters()
    pairwise = correlation.build_pairwise_sweep_plan(base)
    grid = correlation.build_two_noise_grid_plan(base)

    pairwise_ids = set(pairwise["catalog"]["condition_id"])
    grid_ids = set(grid["catalog"]["condition_id"])
    assert len(grid["catalog"]) == 25
    assert len(grid["grid_requests"]) == 25
    assert len(pairwise_ids & grid_ids) == 16
    assert len(grid_ids - pairwise_ids) == 9


def test_complete_grid_recovers_bilinear_synthetic_interaction():
    base = _base_parameters()
    plan = correlation.build_two_noise_grid_plan(base)
    nominal = plan["nominal_strengths"]
    rows = []
    for _, condition in plan["catalog"].iterrows():
        heating_multiplier = (
            condition["motional_heating_s^-1"]
            / nominal["motional_heating"]
        )
        dephasing_multiplier = (
            condition["motional_dephasing_s^-1"]
            / nominal["motional_dephasing"]
        )
        infidelity = (
            0.001
            + 2e-4 * heating_multiplier
            + 4e-4 * dephasing_multiplier
            - 3e-6 * heating_multiplier * dephasing_multiplier
        )
        rows.append(
            {
                "condition_id": condition["condition_id"],
                "nbar": 0.01,
                "infidelity": infidelity,
            }
        )
    grid = correlation.calculate_two_noise_grid_interactions(
        plan, pd.DataFrame(rows)
    )
    expected = -3e-6 * grid["multiplier_x"] * grid["multiplier_y"]

    np.testing.assert_allclose(
        grid["interaction_infidelity"], expected, atol=1e-15, rtol=0.0
    )
    np.testing.assert_allclose(
        grid["bilinear_residual"], 0.0, atol=1e-15, rtol=0.0
    )


def test_grid_dry_run_reuses_16_conditions_and_reports_720_evolutions(tmp_path):
    base = _base_parameters()
    nbar_values = [0.01, 1.0, 2.0, 3.0, 4.0]
    pairwise = correlation.build_pairwise_sweep_plan(base)
    grid_plan = correlation.build_two_noise_grid_plan(base)
    reusable_ids = set(pairwise["catalog"]["condition_id"]) & set(
        grid_plan["catalog"]["condition_id"]
    )
    catalog = grid_plan["catalog"].set_index("condition_id")
    rows = []
    for condition_id in reusable_ids:
        condition = catalog.loc[condition_id]
        for n_bar in nbar_values:
            rows.append(
                {
                    "condition_id": condition_id,
                    "nbar": n_bar,
                    "F_avg": 0.999,
                    "infidelity": 0.001,
                    "motional_heating_s^-1": condition[
                        "motional_heating_s^-1"
                    ],
                    "motional_dephasing_s^-1": condition[
                        "motional_dephasing_s^-1"
                    ],
                    "spin_dephasing_s^-1": 0.0,
                    "photon_scattering_s^-1": 0.0,
                    "is_all_noise_zero": condition["is_all_noise_zero"],
                }
            )
    result = correlation.run_two_noise_rate_grid(
        output_dir=tmp_path,
        base_parameters=base,
        nbar_values=nbar_values,
        reusable_summary=pd.DataFrame(rows),
        execute=False,
        resume=True,
    )

    assert len(result["summary"]) == 16 * 5
    assert result["pending_unique_conditions"] == 9
    assert result["pending_master_equation_evolutions"] == 720
    assert not result["complete"]
