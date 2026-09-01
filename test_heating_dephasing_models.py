from pathlib import Path

import numpy as np
import pandas as pd

import heating_dephasing_models as models


PAIR_GRID_PATH = Path(
    "results/noise_rate_nbar_sweep/all_pairwise_5x5_grids/"
    "all_pairwise_grid_interactions.csv"
)
GENERATOR_PATH = Path(
    "results/error_generator_rate_nbar/generator_pauli_coefficients.csv"
)


def _existing_grid():
    return models.prepare_pair_grid(pd.read_csv(PAIR_GRID_PATH))


def test_independent_overlap_decomposition_at_largest_observed_point():
    grid = _existing_grid()
    point = grid[
        np.isclose(grid["nbar"], 4.0)
        & np.isclose(grid["multiplier_x"], 4.0)
        & np.isclose(grid["multiplier_y"], 4.0)
    ].iloc[0]

    assert np.isclose(
        point["interaction_infidelity"], -4.380776244e-5, rtol=1e-9
    )
    assert np.isclose(
        point["independent_overlap"], -2.14749793e-5, rtol=1e-8
    )
    assert np.isclose(
        point["extra_interaction"], -2.23327832e-5, rtol=1e-8
    )


def test_rate_product_model_recovers_saved_nbar_dependence():
    _, fit, metrics = models.fit_rate_product_model(_existing_grid())

    expected_kappa = np.array([
        6.27694611e-8,
        4.74903663e-7,
        8.33110712e-7,
        1.13465564e-6,
        1.40473307e-6,
    ])
    np.testing.assert_allclose(
        fit["kappa_extra_per_multiplier_product"],
        expected_kappa,
        rtol=2e-7,
    )
    overall = metrics[metrics["scope"].eq("all")].iloc[0]
    assert overall["r_squared"] > 0.999


def test_pauli_overlap_is_dominated_by_ix_and_xi():
    grid = _existing_grid()
    _, contributions, _, metrics = models.fit_pauli_mode_overlap_model(
        grid, pd.read_csv(GENERATOR_PATH)
    )
    endpoint = contributions[
        np.isclose(contributions["nbar"], 4.0)
        & np.isclose(contributions["multiplier_heating"], 4.0)
        & np.isclose(
            contributions["multiplier_motional_dephasing"], 4.0
        )
    ]
    ix_xi_fraction = endpoint.loc[
        endpoint["pauli"].isin(["IX", "XI"]), "gamma_overlap_fraction"
    ].sum()

    assert ix_xi_fraction > 0.999
    overall = metrics[metrics["scope"].eq("all")].iloc[0]
    assert overall["r_squared"] > 0.99


def test_shared_motion_model_is_compact_and_accurate():
    _, fit, metrics = models.fit_shared_motion_model(_existing_grid())
    row = fit.iloc[0]

    assert np.isclose(row["alpha_zero_point"], 1.11392701e-7, rtol=2e-7)
    assert np.isclose(row["beta_per_nbar"], 3.34985918e-7, rtol=2e-7)
    overall = metrics[metrics["scope"].eq("all")].iloc[0]
    assert overall["r_squared"] > 0.99


def test_end_to_end_run_saves_tables_and_figures(tmp_path):
    result = models.run_existing_data_models(
        pair_interactions_path=PAIR_GRID_PATH,
        generator_coefficients_path=GENERATOR_PATH,
        output_dir=tmp_path,
    )

    assert len(result["grid"]) == 125
    assert set(result["metrics"]["model"]) == {
        "per_nbar_rate_product",
        "global_pauli_gamma_overlap",
        "shared_motion_linear_nbar",
    }
    assert all(path.exists() for path in result["paths"].values())


def test_mode_resolved_bilinear_model_recovers_synthetic_coefficients():
    expected_by_nbar = {
        0.01: (1.2e-7, 0.4e-7, 1.1e-7),
        4.0: (3.4e-7, 2.0e-7, 0.55e-7),
    }
    multipliers = [0.0, 0.5, 1.0, 2.0, 4.0]
    rows = []
    for nbar, (a_x, a_xx, b_xx) in expected_by_nbar.items():
        for multiplier_h in multipliers:
            for multiplier_d in multipliers:
                product = multiplier_h * multiplier_d
                values = {
                    "IX": (a_x * product, 0.0),
                    "XI": (a_x * product, 0.0),
                    "XX": (-a_xx * product, b_xx * product),
                }
                for pauli, (gamma, coherent) in values.items():
                    rows.append({
                        "pair_id": models.PAIR_ID,
                        "source_x": models.HEATING,
                        "source_y": models.MOTIONAL_DEPHASING,
                        "nbar": nbar,
                        "multiplier_x": multiplier_h,
                        "multiplier_y": multiplier_d,
                        "pauli": pauli,
                        "interaction_gamma_per_gate": gamma,
                        "interaction_h_rad_per_gate": coherent,
                    })

    grid = models.prepare_mode_resolved_pair_grid(pd.DataFrame(rows))
    modeled, coefficients = models.fit_mode_resolved_bilinear_model(grid)

    for nbar, expected in expected_by_nbar.items():
        fitted = coefficients[np.isclose(coefficients["nbar"], nbar)].iloc[0]
        np.testing.assert_allclose(
            [
                fitted["a_X_per_gate"],
                fitted["a_XX_per_gate"],
                fitted["b_XX_rad_per_gate"],
            ],
            expected,
            rtol=1e-12,
            atol=1e-20,
        )

    residual_columns = [
        "residual_C_gamma_X_mean_per_gate",
        "residual_C_gamma_XX_per_gate",
        "residual_C_h_XX_rad_per_gate",
    ]
    np.testing.assert_allclose(
        modeled[residual_columns].to_numpy(), 0.0, atol=1e-20
    )

    point_residuals, summary = (
        models.evaluate_mode_resolved_bilinear_residuals(modeled)
    )
    assert not point_residuals.empty
    assert set(summary["assessment"]) == {"strongly_supported"}
    np.testing.assert_allclose(
        summary[[
            "fit_relative_l2_error", "fit_normalized_max_abs_error",
            "loocv_relative_l2_error", "loocv_normalized_max_abs_error",
        ]].to_numpy(),
        0.0,
        atol=1e-14,
    )
