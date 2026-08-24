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


def test_changed_nbar_grid_reuses_common_cached_points(tmp_path):
    base = _base_parameters()
    plan = rate_model.build_zero_rate_plan(base)
    rate_model.run_validation_qpt(
        output_dir=tmp_path,
        base_parameters=base,
        nbar_values=[5.0, 6.0],
        plan=plan,
        execute=False,
        resume=True,
    )
    condition = plan["catalog"].iloc[0]
    cached = pd.DataFrame({
        "validation_id": "all_four_noises_off",
        "condition_id": condition["condition_id"],
        "nbar": [5.0, 6.0],
        "F_avg": [0.99, 0.98],
        "infidelity": [0.01, 0.02],
    })
    for source in rate_model.NOISE_SOURCES:
        cached[rate_model.MULTIPLIER_COLUMNS[source]] = 0.0
        cached[rate_model.RATE_COLUMNS[source]] = 0.0
    cached.to_csv(tmp_path / "validation_qpt_summary.csv", index=False)

    subset = rate_model.run_validation_qpt(
        output_dir=tmp_path,
        base_parameters=base,
        nbar_values=[6.0],
        plan=plan,
        execute=False,
        resume=True,
    )
    assert subset["complete"]
    assert len(subset["summary"]) == 1
    assert subset["summary"].iloc[0]["nbar"] == 6.0

    superset = rate_model.run_validation_qpt(
        output_dir=tmp_path,
        base_parameters=base,
        nbar_values=[5.0, 6.0, 7.0],
        plan=plan,
        execute=False,
        resume=True,
    )
    assert not superset["complete"]
    assert superset["pending_condition_count"] == 1
    assert superset["pending_nbar_count"] == 1
    assert superset["pending_master_equation_evolutions"] == 16
    assert len(superset["summary"]) == 2


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


def test_high_nbar_dry_run_budget_includes_zero_reference(tmp_path):
    base = _base_parameters()
    high_nbars = tuple(float(value) for value in range(5, 21))
    validation = rate_model.run_validation_qpt(
        output_dir=tmp_path / "validation",
        base_parameters=base,
        nbar_values=high_nbars,
        plan=rate_model.build_validation_rate_plan(base),
        execute=False,
        resume=True,
    )
    zero = rate_model.run_validation_qpt(
        output_dir=tmp_path / "zero",
        base_parameters=base,
        nbar_values=high_nbars,
        plan=rate_model.build_zero_rate_plan(base),
        execute=False,
        resume=True,
    )

    assert validation["pending_master_equation_evolutions"] == 3072
    assert zero["pending_master_equation_evolutions"] == 256
    assert zero["total_conditions"] == 1


def test_linear_tail_high_nbar_model_and_ncrit_recover_linear_data(tmp_path):
    base, summary, interactions, zero, validation = (
        _synthetic_training_and_validation()
    )
    in_domain = rate_model.build_model_comparison(
        base_parameters=base,
        pair_grid_summary=summary,
        pair_grid_interactions=interactions,
        all_noise_zero_summary=zero,
        validation_summary=validation,
    )
    high_nbars = [5.0, 10.0]

    def extrapolate_group(group, nbar):
        group = group.sort_values("nbar")
        x = group["nbar"].to_numpy(float)
        y = group["infidelity"].to_numpy(float)
        return y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (nbar - x[-1])

    high_zero = pd.DataFrame({
        "nbar": high_nbars,
        "infidelity": [
            extrapolate_group(zero, nbar) for nbar in high_nbars
        ],
    })
    high_rows = []
    for validation_id, group in validation.groupby("validation_id"):
        template = group.iloc[0]
        for nbar in high_nbars:
            high_rows.append({
                "validation_id": validation_id,
                "condition_id": template["condition_id"],
                "nbar": nbar,
                "infidelity": extrapolate_group(group, nbar),
                **{
                    rate_model.MULTIPLIER_COLUMNS[source]: template[
                        rate_model.MULTIPLIER_COLUMNS[source]
                    ]
                    for source in rate_model.NOISE_SOURCES
                },
            })
    extrapolation = rate_model.build_high_nbar_extrapolation(
        base_parameters=base,
        pair_grid_summary=summary,
        pair_grid_interactions=interactions,
        training_zero_summary=zero,
        high_nbar_zero_summary=high_zero,
        validation_summary=pd.DataFrame(high_rows),
    )
    predictions = extrapolation["predictions"]
    np.testing.assert_allclose(
        predictions["fixed_extrapolated_prediction"],
        predictions["actual_infidelity"],
        atol=2e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        predictions["zero_anchored_prediction"],
        predictions["actual_infidelity"],
        atol=2e-15,
        rtol=0.0,
    )

    crossing_metrics = extrapolation["metrics"].copy()
    crossing_mask = (
        crossing_metrics["scope"].eq("nbar")
        & crossing_metrics["model"].eq("fixed_nbar_extrapolation")
        & crossing_metrics["nbar"].eq(10.0)
    )
    crossing_metrics.loc[
        crossing_mask, "rmse_relative_to_noise_penalty"
    ] = 0.02
    criterion = rate_model.build_ncrit_summary(
        in_domain_metrics=in_domain["metrics"],
        extrapolation_metrics=crossing_metrics,
        relative_rmse_threshold=0.01,
    )
    row = criterion["summary"].iloc[0]
    assert row["status"] == "bracketed"
    assert row["discrete_nbar_crit"] == 5.0
    assert row["first_failed_tested_nbar"] == 10.0

    result = {"extrapolation": extrapolation, "criterion": criterion}
    paths = rate_model.save_ncrit_search(result, tmp_path / "saved")
    assert all(path.exists() for path in paths.values())
