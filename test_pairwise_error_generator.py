from pathlib import Path

import numpy as np
import pandas as pd

import pairwise_error_generator as analysis


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


def test_existing_infidelity_interactions_rank_all_six_as_subadditive():
    path = Path(
        "results/noise_rate_nbar_sweep/all_pairwise_5x5_grids/"
        "all_pairwise_grid_interactions.csv"
    )
    ranking = analysis.rank_pairwise_infidelity_interactions(
        pd.read_csv(path)
    )

    assert ranking["pair_id"].tolist() == [
        "motional_heating__motional_dephasing",
        "motional_dephasing__photon_scattering",
        "motional_dephasing__spin_dephasing",
        "motional_heating__photon_scattering",
        "spin_dephasing__photon_scattering",
        "motional_heating__spin_dephasing",
    ]
    assert not ranking["has_positive_enhancement"].any()
    assert (ranking["signed_C_at_max_abs"] < 0.0).all()
    assert np.isclose(
        ranking.iloc[0]["signed_C_at_max_abs"],
        -4.380776244e-5,
        rtol=1e-9,
    )


def test_top_three_pair_grids_deduplicate_to_sixty_five_conditions():
    pair_ids = [
        "motional_heating__motional_dephasing",
        "motional_dephasing__photon_scattering",
        "motional_dephasing__spin_dephasing",
    ]
    plan = analysis.build_selected_pairwise_plan(
        _base_parameters(), pair_ids
    )

    assert len(plan["requests"]) == 3 * 25
    assert len(plan["catalog"]) == 65
    assert plan["catalog"]["is_all_noise_zero"].sum() == 1


def test_pairwise_qpt_dry_run_reports_point_and_evolution_budget(tmp_path):
    pair_ids = [
        "motional_heating__motional_dephasing",
        "motional_dephasing__photon_scattering",
        "motional_dephasing__spin_dephasing",
    ]
    result = analysis.run_or_load_selected_pairwise_qpt(
        output_dir=tmp_path,
        base_parameters=_base_parameters(),
        nbar_values=[0.01, 1.0],
        pair_ids=pair_ids,
        execute=False,
    )

    assert not result["complete"]
    assert result["total_point_count"] == 65 * 2
    assert result["pending_nbar_count"] == 65 * 2
    assert result["pending_master_equation_evolutions"] == 65 * 2 * 16
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "condition_catalog.csv").exists()
    assert (tmp_path / "pairwise_grid_requests.csv").exists()


def test_component_interaction_recovers_synthetic_mixed_term():
    pair_id = "motional_heating__motional_dephasing"
    plan = analysis.build_selected_pairwise_plan(
        _base_parameters(), [pair_id], rate_multipliers=[0.0, 1.0]
    )
    coordinates = plan["requests"].set_index("condition_id")[[
        "multiplier_x", "multiplier_y"
    ]]
    rows = []
    for condition_id in plan["catalog"]["condition_id"]:
        x, y = coordinates.loc[condition_id]
        rows.append({
            "condition_id": condition_id,
            "nbar": 0.01,
            "pauli": "XX",
            "pauli_weight": 2,
            "mode_class": "correlated",
            "gamma_per_gate": 1.0 + x + 2.0 * y + 3.0 * x * y,
            "h_rad_per_gate": 4.0 + 2.0 * x + 3.0 * y - 5.0 * x * y,
        })

    interactions = analysis.calculate_pauli_generator_interactions(
        plan, pd.DataFrame(rows)
    )
    joint = interactions[
        np.isclose(interactions["multiplier_x"], 1.0)
        & np.isclose(interactions["multiplier_y"], 1.0)
    ].iloc[0]

    assert len(interactions) == 4
    assert np.isclose(joint["interaction_gamma_per_gate"], 3.0)
    assert np.isclose(joint["interaction_h_rad_per_gate"], -5.0)
    np.testing.assert_allclose(
        interactions.loc[
            interactions["multiplier_x"].eq(0.0)
            | interactions["multiplier_y"].eq(0.0),
            "interaction_gamma_per_gate",
        ],
        0.0,
        atol=1e-15,
    )
