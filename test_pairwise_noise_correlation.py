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
