from itertools import combinations

import numpy as np
import pandas as pd

import four_noise_rate_model as rate_model


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


def test_validation_plan_is_deterministic_unique_and_off_grid():
    plan_a = rate_model.build_validation_rate_plan(_base_parameters())
    plan_b = rate_model.build_validation_rate_plan(_base_parameters())

    pd.testing.assert_frame_equal(plan_a["catalog"], plan_b["catalog"])
    assert len(plan_a["catalog"]) == 12
    assert plan_a["catalog"]["condition_id"].nunique() == 12
    multiplier_columns = list(rate_model.MULTIPLIER_COLUMNS.values())
    values = plan_a["catalog"][multiplier_columns].to_numpy(float)
    assert np.all((values >= 0.1) & (values <= 3.9))
    training_grid = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    assert not np.any(
        np.isclose(values[..., None], training_grid).any(axis=-1).all(axis=1)
    )


def test_validation_dry_run_reports_960_evolutions(tmp_path):
    base = _base_parameters()
    plan = rate_model.build_validation_rate_plan(base)
    result = rate_model.run_validation_qpt(
        output_dir=tmp_path,
        base_parameters=base,
        nbar_values=base["n_bar_list"],
        plan=plan,
        execute=False,
        resume=True,
    )

    assert result["pending_condition_count"] == 12
    assert result["pending_master_equation_evolutions"] == 960
    assert not result["complete"]
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "validation_rate_catalog.csv").exists()


def _synthetic_training_and_validation():
    base = _base_parameters()
    sources = rate_model.NOISE_SOURCES
    nominal = {
        "motional_heating": 10.0,
        "motional_dephasing": 18.0,
        "spin_dephasing": 1.0 / 0.3,
        "photon_scattering": 4.0,
    }
    nbar_values = [0.01, 4.0]
    multipliers = [0.0, 0.5, 1.0, 2.0, 4.0]
    single_coefficients = {
        source: (index + 1) * 1e-4
        for index, source in enumerate(sources)
    }
    pair_coefficients = {
        tuple(pair): (index + 1) * 1e-5
        for index, pair in enumerate(combinations(sources, 2))
    }

    def zero(nbar):
        return 0.001 + 2e-5 * nbar

    def single(source, multiplier, nbar):
        return single_coefficients[source] * multiplier * (1.0 + 0.1 * nbar)

    def pair(source_i, source_j, multiplier_i, multiplier_j, nbar):
        coefficient = pair_coefficients[(source_i, source_j)]
        return coefficient * multiplier_i * multiplier_j * (1.0 + 0.05 * nbar)

    zero_summary = pd.DataFrame({
        "nbar": nbar_values,
        "infidelity": [zero(nbar) for nbar in nbar_values],
    })
    summary_rows = []
    interaction_rows = []
    for source_i, source_j in combinations(sources, 2):
        pair_id = f"{source_i}__{source_j}"
        for multiplier_i in multipliers:
            for multiplier_j in multipliers:
                rates = {source: 0.0 for source in sources}
                rates[source_i] = multiplier_i * nominal[source_i]
                rates[source_j] = multiplier_j * nominal[source_j]
                for nbar in nbar_values:
                    interaction = pair(
                        source_i, source_j, multiplier_i, multiplier_j, nbar
                    )
                    infidelity = (
                        zero(nbar)
                        + single(source_i, multiplier_i, nbar)
                        + single(source_j, multiplier_j, nbar)
                        + interaction
                    )
                    summary_rows.append({
                        "condition_id": (
                            f"{pair_id}_{multiplier_i}_{multiplier_j}_{nbar}"
                        ),
                        "nbar": nbar,
                        "infidelity": infidelity,
                        **{
                            rate_model.RATE_COLUMNS[source]: rates[source]
                            for source in sources
                        },
                    })
                    interaction_rows.append({
                        "pair_id": pair_id,
                        "source_x": source_i,
                        "source_y": source_j,
                        "nbar": nbar,
                        "multiplier_x": multiplier_i,
                        "multiplier_y": multiplier_j,
                        "interaction_infidelity": interaction,
                    })

    validation_rows = []
    validation_points = [
        (0.25, 0.75, 1.5, 3.25),
        (3.6, 1.25, 0.3, 2.4),
    ]
    for validation_index, values in enumerate(validation_points):
        multiplier_map = dict(zip(sources, values))
        for nbar in nbar_values:
            actual = zero(nbar)
            actual += sum(
                single(source, multiplier_map[source], nbar)
                for source in sources
            )
            actual += sum(
                pair(
                    source_i,
                    source_j,
                    multiplier_map[source_i],
                    multiplier_map[source_j],
                    nbar,
                )
                for source_i, source_j in combinations(sources, 2)
            )
            validation_rows.append({
                "validation_id": f"test_{validation_index}",
                "condition_id": f"condition_{validation_index}",
                "nbar": nbar,
                "infidelity": actual,
                **{
                    rate_model.MULTIPLIER_COLUMNS[source]: multiplier_map[source]
                    for source in sources
                },
            })
    return (
        base,
        pd.DataFrame(summary_rows),
        pd.DataFrame(interaction_rows),
        zero_summary,
        pd.DataFrame(validation_rows),
    )


def test_surface_and_reduced_pairwise_models_recover_bilinear_data(tmp_path):
    base, summary, interactions, zero, validation = (
        _synthetic_training_and_validation()
    )
    result = rate_model.build_model_comparison(
        base_parameters=base,
        pair_grid_summary=summary,
        pair_grid_interactions=interactions,
        all_noise_zero_summary=zero,
        validation_summary=validation,
    )

    predictions = result["predictions"]
    np.testing.assert_allclose(
        predictions["strict_pairwise_prediction"],
        predictions["actual_infidelity"],
        atol=1e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        predictions["bilinear_pairwise_prediction"],
        predictions["actual_infidelity"],
        atol=1e-15,
        rtol=0.0,
    )
    assert np.all(
        predictions["single_additive_prediction"]
        < predictions["actual_infidelity"]
    )
    paths = rate_model.save_model_comparison(result, tmp_path)
    assert all(path.exists() for path in paths.values())
