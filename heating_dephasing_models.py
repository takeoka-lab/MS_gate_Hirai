"""Existing-data models for heating--motional-dephasing nonadditivity.

No master-equation evolution is performed here.  The module uses the saved
two-noise infidelity grids and the saved single-axis Pauli generators to
separate three levels of explanation:

1. independent-channel overlap plus a per-nbar rate-product correction,
2. overlap of the Pauli dissipator modes produced by each noise source,
3. a two-parameter shared-motion model whose thermal response is linear in
   nbar over the already simulated range,
4. a mode-resolved bilinear model for local dissipative, correlated
   dissipative, and coherent XX interaction components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIR_ID = "motional_heating__motional_dephasing"
HEATING = "motional_heating"
MOTIONAL_DEPHASING = "motional_dephasing"
TWO_QUBIT_DIMENSION = 4


def _atomic_save_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def prepare_pair_grid(interactions: pd.DataFrame) -> pd.DataFrame:
    """Separate observed C into an independent overlap and an excess term."""

    required = {
        "pair_id", "source_x", "source_y", "nbar", "multiplier_x",
        "multiplier_y", "grid_infidelity", "x_only_infidelity",
        "y_only_infidelity", "zero_infidelity", "interaction_infidelity",
    }
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"Pair-grid table is missing columns: {missing}")
    grid = interactions[interactions["pair_id"].eq(PAIR_ID)].copy()
    if grid.empty:
        raise ValueError(f"Pair {PAIR_ID} is absent from the interaction table")
    source_pairs = set(zip(grid["source_x"], grid["source_y"]))
    if source_pairs != {(HEATING, MOTIONAL_DEPHASING)}:
        raise ValueError(
            "The selected grid does not use heating as x and motional "
            "dephasing as y"
        )

    grid["delta_heating_infidelity"] = (
        grid["x_only_infidelity"] - grid["zero_infidelity"]
    )
    grid["delta_motional_dephasing_infidelity"] = (
        grid["y_only_infidelity"] - grid["zero_infidelity"]
    )
    # For a d-dimensional depolarizing channel, independent average
    # infidelities compose as r12 = r1 + r2 - d/(d-1) r1 r2.
    grid["independent_overlap"] = -(
        TWO_QUBIT_DIMENSION / (TWO_QUBIT_DIMENSION - 1)
    ) * (
        grid["delta_heating_infidelity"]
        * grid["delta_motional_dephasing_infidelity"]
    )
    grid["extra_interaction"] = (
        grid["interaction_infidelity"] - grid["independent_overlap"]
    )
    grid["multiplier_product"] = (
        grid["multiplier_x"] * grid["multiplier_y"]
    )
    return grid.sort_values(
        ["nbar", "multiplier_y", "multiplier_x"]
    ).reset_index(drop=True)


def _prediction_metrics(
    name: str,
    frame: pd.DataFrame,
    prediction_column: str,
) -> pd.DataFrame:
    rows = []
    active = frame[
        frame["multiplier_x"].gt(0.0)
        & frame["multiplier_y"].gt(0.0)
    ]
    groups = [("all", np.nan, active)]
    groups.extend(
        ("nbar", float(nbar), group)
        for nbar, group in active.groupby("nbar", sort=True)
    )
    for scope, nbar, group in groups:
        actual = group["extra_interaction"].to_numpy(float)
        predicted = group[prediction_column].to_numpy(float)
        residual = actual - predicted
        centered = actual - actual.mean()
        denominator = float(centered @ centered)
        rows.append({
            "model": name,
            "scope": scope,
            "nbar": nbar,
            "count": len(group),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "max_abs_residual": float(np.max(np.abs(residual))),
            "nrmse_to_max_abs_extra": float(
                np.sqrt(np.mean(np.square(residual)))
                / max(np.max(np.abs(actual)), 1e-18)
            ),
            "r_squared": float(
                1.0 - (residual @ residual) / denominator
                if denominator > 0.0 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def fit_rate_product_model(
    grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit C_extra = -kappa(nbar) * m_h * m_d independently at each nbar."""

    active = grid[
        grid["multiplier_x"].gt(0.0)
        & grid["multiplier_y"].gt(0.0)
    ]
    rows = []
    for nbar, group in active.groupby("nbar", sort=True):
        product = group["multiplier_product"].to_numpy(float)
        extra = group["extra_interaction"].to_numpy(float)
        kappa = float(-(product @ extra) / (product @ product))
        prediction = -kappa * product
        residual = extra - prediction
        rows.append({
            "nbar": float(nbar),
            "kappa_extra_per_multiplier_product": kappa,
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "max_abs_residual": float(np.max(np.abs(residual))),
            "nrmse_to_max_abs_extra": float(
                np.sqrt(np.mean(np.square(residual)))
                / max(np.max(np.abs(extra)), 1e-18)
            ),
            "correlation": float(np.corrcoef(extra, prediction)[0, 1]),
        })
    fit = pd.DataFrame(rows)
    modeled = grid.merge(fit[[
        "nbar", "kappa_extra_per_multiplier_product"
    ]], on="nbar", how="left")
    modeled["model1_extra_prediction"] = -(
        modeled["kappa_extra_per_multiplier_product"]
        * modeled["multiplier_product"]
    )
    modeled["model1_total_C_prediction"] = (
        modeled["independent_overlap"]
        + modeled["model1_extra_prediction"]
    )
    metrics = _prediction_metrics(
        "per_nbar_rate_product", modeled, "model1_extra_prediction"
    )
    return modeled, fit, metrics


def _source_generator_deltas(
    coefficients: pd.DataFrame, source: str, prefix: str
) -> pd.DataFrame:
    selected = coefficients[coefficients["noise_source"].eq(source)].copy()
    required = {
        "noise_source", "nbar", "multiplier", "pauli",
        "h_rad_per_gate", "gamma_per_gate",
    }
    missing = required - set(coefficients.columns)
    if missing:
        raise ValueError(f"Generator table is missing columns: {missing}")
    baseline = selected[np.isclose(selected["multiplier"], 0.0)][[
        "nbar", "pauli", "h_rad_per_gate", "gamma_per_gate"
    ]].rename(columns={
        "h_rad_per_gate": "h_zero",
        "gamma_per_gate": "gamma_zero",
    })
    selected = selected.merge(baseline, on=["nbar", "pauli"], how="inner")
    selected[f"delta_h_{prefix}"] = (
        selected["h_rad_per_gate"] - selected["h_zero"]
    )
    selected[f"delta_gamma_{prefix}"] = (
        selected["gamma_per_gate"] - selected["gamma_zero"]
    )
    return selected[[
        "nbar", "multiplier", "pauli",
        f"delta_h_{prefix}", f"delta_gamma_{prefix}",
    ]].rename(columns={"multiplier": f"multiplier_{prefix}"})


def fit_pauli_mode_overlap_model(
    grid: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit C_extra from overlap of saved single-axis Pauli gamma changes."""

    heating = _source_generator_deltas(coefficients, HEATING, "heating")
    dephasing = _source_generator_deltas(
        coefficients, MOTIONAL_DEPHASING, "motional_dephasing"
    )
    contributions = heating.merge(
        dephasing, on=["nbar", "pauli"], how="inner"
    )
    contributions["gamma_mode_product"] = (
        contributions["delta_gamma_heating"]
        * contributions["delta_gamma_motional_dephasing"]
    )
    contributions["h_mode_product"] = (
        contributions["delta_h_heating"]
        * contributions["delta_h_motional_dephasing"]
    )
    scores = contributions.groupby([
        "nbar", "multiplier_heating", "multiplier_motional_dephasing"
    ], as_index=False).agg(
        gamma_overlap_score=("gamma_mode_product", "sum"),
        h_overlap_score=("h_mode_product", "sum"),
    )
    modeled = grid.merge(
        scores,
        left_on=["nbar", "multiplier_x", "multiplier_y"],
        right_on=[
            "nbar", "multiplier_heating", "multiplier_motional_dephasing"
        ],
        how="left",
    )
    active = modeled[
        modeled["multiplier_x"].gt(0.0)
        & modeled["multiplier_y"].gt(0.0)
    ]
    score = active["gamma_overlap_score"].to_numpy(float)
    extra = active["extra_interaction"].to_numpy(float)
    scale = float((score @ extra) / (score @ score))
    modeled["model2_extra_prediction"] = (
        scale * modeled["gamma_overlap_score"]
    )
    modeled["model2_total_C_prediction"] = (
        modeled["independent_overlap"]
        + modeled["model2_extra_prediction"]
    )

    scale_rows = [{
        "scope": "all",
        "nbar": np.nan,
        "gamma_overlap_scale": scale,
    }]
    for nbar, group in active.groupby("nbar", sort=True):
        score_n = group["gamma_overlap_score"].to_numpy(float)
        extra_n = group["extra_interaction"].to_numpy(float)
        scale_rows.append({
            "scope": "nbar",
            "nbar": float(nbar),
            "gamma_overlap_scale": float(
                (score_n @ extra_n) / (score_n @ score_n)
            ),
        })
    scales = pd.DataFrame(scale_rows)
    metrics = _prediction_metrics(
        "global_pauli_gamma_overlap", modeled, "model2_extra_prediction"
    )

    total_by_point = contributions.groupby([
        "nbar", "multiplier_heating", "multiplier_motional_dephasing"
    ])["gamma_mode_product"].transform("sum")
    contributions["gamma_overlap_fraction"] = np.where(
        np.abs(total_by_point) > 1e-30,
        contributions["gamma_mode_product"] / total_by_point,
        np.nan,
    )
    return modeled, contributions, scales, metrics


def fit_shared_motion_model(
    grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit C_extra = -(alpha + beta*nbar) * m_h * m_d."""

    active = grid[
        grid["multiplier_x"].gt(0.0)
        & grid["multiplier_y"].gt(0.0)
    ]
    product = active["multiplier_product"].to_numpy(float)
    nbar = active["nbar"].to_numpy(float)
    design = np.column_stack([product, nbar * product])
    extra = active["extra_interaction"].to_numpy(float)
    raw_coefficients = np.linalg.lstsq(design, extra, rcond=None)[0]
    alpha, beta = (-float(raw_coefficients[0]), -float(raw_coefficients[1]))
    modeled = grid.copy()
    modeled["model3_extra_prediction"] = -(
        alpha + beta * modeled["nbar"]
    ) * modeled["multiplier_product"]
    modeled["model3_total_C_prediction"] = (
        modeled["independent_overlap"]
        + modeled["model3_extra_prediction"]
    )
    fit = pd.DataFrame([{
        "alpha_zero_point": alpha,
        "beta_per_nbar": beta,
        "formula": "-(alpha + beta*nbar)*m_heating*m_motional_dephasing",
    }])
    metrics = _prediction_metrics(
        "shared_motion_linear_nbar", modeled, "model3_extra_prediction"
    )
    return modeled, fit, metrics


def _plot_c_decomposition(grid: pd.DataFrame, path: Path) -> Path:
    endpoint = grid[
        np.isclose(grid["multiplier_x"], 4.0)
        & np.isclose(grid["multiplier_y"], 4.0)
    ].sort_values("nbar")
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(
        endpoint["nbar"], endpoint["interaction_infidelity"], "o-",
        label=r"observed $C_{hd}$",
    )
    axis.plot(
        endpoint["nbar"], endpoint["independent_overlap"], "s--",
        label="independent-channel overlap",
    )
    axis.plot(
        endpoint["nbar"], endpoint["extra_interaction"], "^--",
        label=r"remaining $C_{hd}^{extra}$",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel(r"thermal occupation $\bar n$")
    axis.set_ylabel("infidelity correction")
    axis.set_title(r"Heating $\times$ motional dephasing at $m_h=m_d=4$")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_model_comparison(grid: pd.DataFrame, path: Path) -> Path:
    active = grid[
        grid["multiplier_x"].gt(0.0)
        & grid["multiplier_y"].gt(0.0)
    ]
    models = [
        ("model1_extra_prediction", "Model 1: per-nbar rate product"),
        ("model2_extra_prediction", "Model 2: Pauli-mode overlap"),
        ("model3_extra_prediction", "Model 3: shared-motion thermal law"),
    ]
    actual = active["extra_interaction"].to_numpy(float)
    lower = min(float(actual.min()), *(float(active[c].min()) for c, _ in models))
    upper = max(float(actual.max()), *(float(active[c].max()) for c, _ in models))
    margin = 0.05 * max(upper - lower, 1e-16)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), squeeze=False)
    for axis, (column, title) in zip(axes[0], models):
        scatter = axis.scatter(
            actual, active[column], c=active["nbar"], cmap="viridis",
            s=28, alpha=0.85,
        )
        axis.plot(
            [lower - margin, upper + margin],
            [lower - margin, upper + margin],
            color="black", linewidth=1.0, linestyle="--",
        )
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(r"observed $C_{hd}^{extra}$")
        axis.set_ylabel("model prediction")
        axis.grid(alpha=0.2)
    figure.colorbar(scatter, ax=axes.ravel().tolist(), label=r"$\bar n$")
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_mode_contributions(contributions: pd.DataFrame, path: Path) -> Path:
    endpoint = contributions[
        np.isclose(contributions["multiplier_heating"], 4.0)
        & np.isclose(contributions["multiplier_motional_dephasing"], 4.0)
    ].copy()
    endpoint["display_mode"] = np.where(
        endpoint["pauli"].isin(["IX", "XI", "XX"]),
        endpoint["pauli"], "other",
    )
    table = endpoint.groupby(
        ["nbar", "display_mode"], as_index=False
    )["gamma_mode_product"].sum()
    pivot = table.pivot(
        index="nbar", columns="display_mode", values="gamma_mode_product"
    ).fillna(0.0)
    for label in ["IX", "XI", "XX", "other"]:
        if label not in pivot:
            pivot[label] = 0.0
    pivot = pivot[["IX", "XI", "XX", "other"]]
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    bottom = np.zeros(len(pivot))
    for label in pivot.columns:
        values = pivot[label].to_numpy(float)
        axis.bar(pivot.index, values, bottom=bottom, label=label)
        bottom += values
    axis.set_xlabel(r"thermal occupation $\bar n$")
    axis.set_ylabel(r"$\Delta\gamma_P^{(h)}\Delta\gamma_P^{(d)}$")
    axis.set_title(r"Pauli-mode overlap at $m_h=m_d=4$")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def run_existing_data_models(
    *,
    pair_interactions_path: str | Path,
    generator_coefficients_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run and save all three models using existing CSV data only."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions = pd.read_csv(pair_interactions_path)
    coefficients = pd.read_csv(generator_coefficients_path)
    grid = prepare_pair_grid(interactions)
    model1_grid, model1_fit, metrics1 = fit_rate_product_model(grid)
    model2_grid, contributions, model2_scales, metrics2 = (
        fit_pauli_mode_overlap_model(model1_grid, coefficients)
    )
    final_grid, model3_fit, metrics3 = fit_shared_motion_model(model2_grid)
    metrics = pd.concat([metrics1, metrics2, metrics3], ignore_index=True)

    paths = {
        "grid": _atomic_save_csv(
            final_grid, output_dir / "heating_dephasing_model_grid.csv"
        ),
        "model1_fit": _atomic_save_csv(
            model1_fit, output_dir / "model1_rate_product_by_nbar.csv"
        ),
        "model2_contributions": _atomic_save_csv(
            contributions,
            output_dir / "model2_pauli_mode_contributions.csv",
        ),
        "model2_scales": _atomic_save_csv(
            model2_scales, output_dir / "model2_pauli_overlap_scales.csv"
        ),
        "model3_fit": _atomic_save_csv(
            model3_fit, output_dir / "model3_shared_motion_fit.csv"
        ),
        "metrics": _atomic_save_csv(
            metrics, output_dir / "model_comparison_metrics.csv"
        ),
    }
    paths["decomposition_figure"] = _plot_c_decomposition(
        final_grid, output_dir / "heating_dephasing_C_decomposition.png"
    )
    paths["comparison_figure"] = _plot_model_comparison(
        final_grid, output_dir / "heating_dephasing_model_comparison.png"
    )
    paths["mode_figure"] = _plot_mode_contributions(
        contributions, output_dir / "heating_dephasing_mode_overlap.png"
    )
    return {
        "grid": final_grid,
        "model1_fit": model1_fit,
        "model2_contributions": contributions,
        "model2_scales": model2_scales,
        "model3_fit": model3_fit,
        "metrics": metrics,
        "paths": paths,
    }


def prepare_mode_resolved_pair_grid(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Build the three-component grid used in the effective model.

    The sign convention is

        C_gamma_X  = (C_gamma_IX + C_gamma_XI) / 2,
        C_gamma_XX = -a_XX * m_h * m_d,
        C_h_XX     = +b_XX * m_h * m_d.

    Thus ``a_XX`` is reported as a positive magnitude even though the fitted
    interaction correction to ``gamma_XX`` is negative.
    """

    required = {
        "pair_id", "source_x", "source_y", "nbar", "multiplier_x",
        "multiplier_y", "pauli", "interaction_gamma_per_gate",
        "interaction_h_rad_per_gate",
    }
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(
            f"Pairwise-generator table is missing columns: {missing}"
        )
    selected = interactions[interactions["pair_id"].eq(PAIR_ID)].copy()
    if selected.empty:
        raise ValueError(f"Pair {PAIR_ID} is absent from the interaction table")
    source_pairs = set(zip(selected["source_x"], selected["source_y"]))
    if source_pairs != {(HEATING, MOTIONAL_DEPHASING)}:
        raise ValueError(
            "The selected grid does not use heating as x and motional "
            "dephasing as y"
        )

    coordinates = [
        "pair_id", "source_x", "source_y", "nbar",
        "multiplier_x", "multiplier_y",
    ]
    key = selected[selected["pauli"].isin(["IX", "XI", "XX"])]
    gamma = key.pivot(
        index=coordinates,
        columns="pauli",
        values="interaction_gamma_per_gate",
    ).reset_index()
    needed_modes = {"IX", "XI", "XX"}
    missing_modes = needed_modes - set(gamma.columns)
    if missing_modes:
        raise ValueError(f"Missing Pauli interaction modes: {missing_modes}")
    gamma = gamma.rename(columns={
        "IX": "C_gamma_IX_per_gate",
        "XI": "C_gamma_XI_per_gate",
        "XX": "C_gamma_XX_per_gate",
    })
    h_xx = key[key["pauli"].eq("XX")][
        coordinates + ["interaction_h_rad_per_gate"]
    ].rename(columns={
        "interaction_h_rad_per_gate": "C_h_XX_rad_per_gate"
    })
    grid = gamma.merge(h_xx, on=coordinates, how="inner", validate="one_to_one")
    grid["C_gamma_X_mean_per_gate"] = 0.5 * (
        grid["C_gamma_IX_per_gate"] + grid["C_gamma_XI_per_gate"]
    )
    grid["C_gamma_X_ion_asymmetry_per_gate"] = 0.5 * (
        grid["C_gamma_IX_per_gate"] - grid["C_gamma_XI_per_gate"]
    )
    grid["multiplier_product"] = (
        grid["multiplier_x"] * grid["multiplier_y"]
    )
    return grid.sort_values(
        ["nbar", "multiplier_y", "multiplier_x"]
    ).reset_index(drop=True)


def _origin_fit_metrics(
    multiplier_product: np.ndarray,
    response: np.ndarray,
) -> dict[str, float]:
    denominator = float(multiplier_product @ multiplier_product)
    if denominator <= 0.0:
        raise ValueError("A positive multiplier product is required for fitting")
    coefficient = float(
        (multiplier_product @ response) / denominator
    )
    prediction = coefficient * multiplier_product
    residual = response - prediction
    centered = response - response.mean()
    centered_norm = float(centered @ centered)
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    return {
        "coefficient": coefficient,
        "rmse": rmse,
        "max_abs_residual": float(np.max(np.abs(residual))),
        "nrmse_to_max_abs_response": float(
            rmse / max(np.max(np.abs(response)), 1e-30)
        ),
        "r_squared": float(
            1.0 - (residual @ residual) / centered_norm
            if centered_norm > 0.0 else np.nan
        ),
    }


def fit_mode_resolved_bilinear_model(
    grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the mode-resolved ``m_h*m_d`` model independently at each nbar."""

    active = grid[grid["multiplier_product"].gt(0.0)]
    if active.empty:
        raise ValueError("The mode-resolved grid has no active pair points")
    targets = {
        "a_X_per_gate": "C_gamma_X_mean_per_gate",
        # The minus sign makes a_XX the positive magnitude in -a_XX D_XX.
        "a_XX_per_gate": "negative_C_gamma_XX_per_gate",
        "b_XX_rad_per_gate": "C_h_XX_rad_per_gate",
    }
    active = active.copy()
    active["negative_C_gamma_XX_per_gate"] = -(
        active["C_gamma_XX_per_gate"]
    )
    rows = []
    for nbar, group in active.groupby("nbar", sort=True):
        product = group["multiplier_product"].to_numpy(float)
        row: dict[str, float] = {"nbar": float(nbar)}
        for coefficient_name, response_name in targets.items():
            metrics = _origin_fit_metrics(
                product, group[response_name].to_numpy(float)
            )
            row[coefficient_name] = metrics.pop("coefficient")
            metric_prefix = coefficient_name.removesuffix("_per_gate")
            for metric_name, value in metrics.items():
                row[f"{metric_prefix}_{metric_name}"] = value
        rows.append(row)
    coefficients = pd.DataFrame(rows)

    modeled = grid.merge(coefficients, on="nbar", how="left")
    modeled["model_C_gamma_X_mean_per_gate"] = (
        modeled["a_X_per_gate"] * modeled["multiplier_product"]
    )
    modeled["model_C_gamma_XX_per_gate"] = -(
        modeled["a_XX_per_gate"] * modeled["multiplier_product"]
    )
    modeled["model_C_h_XX_rad_per_gate"] = (
        modeled["b_XX_rad_per_gate"] * modeled["multiplier_product"]
    )
    for observed, predicted, residual in [
        (
            "C_gamma_X_mean_per_gate",
            "model_C_gamma_X_mean_per_gate",
            "residual_C_gamma_X_mean_per_gate",
        ),
        (
            "C_gamma_XX_per_gate",
            "model_C_gamma_XX_per_gate",
            "residual_C_gamma_XX_per_gate",
        ),
        (
            "C_h_XX_rad_per_gate",
            "model_C_h_XX_rad_per_gate",
            "residual_C_h_XX_rad_per_gate",
        ),
    ]:
        modeled[residual] = modeled[observed] - modeled[predicted]
    return modeled, coefficients


def _relative_residual_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = observed - predicted
    observed_l2 = float(np.linalg.norm(observed))
    peak = float(np.max(np.abs(observed)))
    centered = observed - observed.mean()
    centered_norm_squared = float(centered @ centered)
    residual_norm_squared = float(residual @ residual)
    return {
        "relative_l2_error": float(
            np.linalg.norm(residual) / observed_l2
            if observed_l2 > 0.0 else np.nan
        ),
        "normalized_max_abs_error": float(
            np.max(np.abs(residual)) / peak if peak > 0.0 else np.nan
        ),
        "normalized_rmse_to_peak": float(
            np.sqrt(np.mean(np.square(residual))) / peak
            if peak > 0.0 else np.nan
        ),
        "r_squared": float(
            1.0 - residual_norm_squared / centered_norm_squared
            if centered_norm_squared > 0.0 else np.nan
        ),
    }


def evaluate_mode_resolved_bilinear_residuals(
    modeled: pd.DataFrame,
    *,
    strong_relative_l2: float = 0.05,
    strong_normalized_max: float = 0.10,
    acceptable_relative_l2: float = 0.10,
    acceptable_normalized_max: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate in-sample and leave-one-out residuals of the bilinear model.

    The adequacy labels are operational model-reduction criteria, not
    statistical confidence intervals.  A label is assigned only when both
    the fitted residual and the leave-one-out prediction meet the threshold.
    """

    thresholds = [
        strong_relative_l2,
        strong_normalized_max,
        acceptable_relative_l2,
        acceptable_normalized_max,
    ]
    if not all(np.isfinite(value) and value > 0.0 for value in thresholds):
        raise ValueError("Residual thresholds must be finite and positive")
    if strong_relative_l2 > acceptable_relative_l2:
        raise ValueError("Strong L2 threshold must not exceed acceptable")
    if strong_normalized_max > acceptable_normalized_max:
        raise ValueError("Strong max threshold must not exceed acceptable")

    components = {
        "local_gamma_X": (
            "C_gamma_X_mean_per_gate",
            "model_C_gamma_X_mean_per_gate",
        ),
        "correlated_gamma_XX": (
            "C_gamma_XX_per_gate",
            "model_C_gamma_XX_per_gate",
        ),
        "coherent_h_XX": (
            "C_h_XX_rad_per_gate",
            "model_C_h_XX_rad_per_gate",
        ),
    }
    required = {
        "nbar", "multiplier_x", "multiplier_y", "multiplier_product",
        *{
            column
            for observed, predicted in components.values()
            for column in (observed, predicted)
        },
    }
    missing = required - set(modeled.columns)
    if missing:
        raise ValueError(f"Mode-resolved model is missing columns: {missing}")

    active = modeled[modeled["multiplier_product"].gt(0.0)].copy()
    if active.empty:
        raise ValueError("No active two-noise points are available")
    point_frames = []
    summary_rows = []
    for nbar, nbar_group in active.groupby("nbar", sort=True):
        product = nbar_group["multiplier_product"].to_numpy(float)
        product_norm_squared = float(product @ product)
        if product_norm_squared <= 0.0:
            continue
        for component, (observed_column, predicted_column) in components.items():
            observed = nbar_group[observed_column].to_numpy(float)
            predicted = nbar_group[predicted_column].to_numpy(float)
            fit_metrics = _relative_residual_metrics(observed, predicted)

            total_product_response = float(product @ observed)
            loocv_prediction = np.full_like(observed, np.nan, dtype=float)
            for index in range(len(product)):
                denominator = product_norm_squared - product[index] ** 2
                if denominator <= 0.0:
                    continue
                slope_without_point = (
                    total_product_response - product[index] * observed[index]
                ) / denominator
                loocv_prediction[index] = slope_without_point * product[index]
            finite_loocv = np.isfinite(loocv_prediction)
            loocv_metrics = _relative_residual_metrics(
                observed[finite_loocv], loocv_prediction[finite_loocv]
            )

            fit_passes_strong = (
                fit_metrics["relative_l2_error"] <= strong_relative_l2
                and fit_metrics["normalized_max_abs_error"]
                <= strong_normalized_max
            )
            loocv_passes_strong = (
                loocv_metrics["relative_l2_error"] <= strong_relative_l2
                and loocv_metrics["normalized_max_abs_error"]
                <= strong_normalized_max
            )
            fit_passes_acceptable = (
                fit_metrics["relative_l2_error"] <= acceptable_relative_l2
                and fit_metrics["normalized_max_abs_error"]
                <= acceptable_normalized_max
            )
            loocv_passes_acceptable = (
                loocv_metrics["relative_l2_error"] <= acceptable_relative_l2
                and loocv_metrics["normalized_max_abs_error"]
                <= acceptable_normalized_max
            )
            if fit_passes_strong and loocv_passes_strong:
                assessment = "strongly_supported"
            elif fit_passes_acceptable and loocv_passes_acceptable:
                assessment = "acceptable_approximation"
            else:
                assessment = "insufficient"

            summary_rows.append({
                "nbar": float(nbar),
                "component": component,
                "n_active_points": int(len(observed)),
                **{f"fit_{key}": value for key, value in fit_metrics.items()},
                **{
                    f"loocv_{key}": value
                    for key, value in loocv_metrics.items()
                },
                "assessment": assessment,
                "strong_relative_l2_threshold": strong_relative_l2,
                "strong_normalized_max_threshold": strong_normalized_max,
                "acceptable_relative_l2_threshold": acceptable_relative_l2,
                "acceptable_normalized_max_threshold": (
                    acceptable_normalized_max
                ),
            })

            points = nbar_group[[
                "nbar", "multiplier_x", "multiplier_y",
                "multiplier_product",
            ]].copy()
            points.insert(1, "component", component)
            points["observed"] = observed
            points["fitted_prediction"] = predicted
            points["fitted_residual"] = observed - predicted
            points["loocv_prediction"] = loocv_prediction
            points["loocv_residual"] = observed - loocv_prediction
            peak = float(np.max(np.abs(observed)))
            points["fitted_residual_over_peak"] = (
                points["fitted_residual"] / peak if peak > 0.0 else np.nan
            )
            points["loocv_residual_over_peak"] = (
                points["loocv_residual"] / peak if peak > 0.0 else np.nan
            )
            point_frames.append(points)

    point_residuals = pd.concat(point_frames, ignore_index=True).sort_values([
        "nbar", "component", "multiplier_y", "multiplier_x"
    ]).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values([
        "nbar", "component"
    ]).reset_index(drop=True)
    return point_residuals, summary


def _plot_mode_coefficients_vs_nbar(
    coefficients: pd.DataFrame,
    path: Path,
) -> Path:
    scale = 1e7
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(
        coefficients["nbar"], scale * coefficients["a_X_per_gate"],
        "o-", label=r"$a_X$: local $IX/XI$",
    )
    axes[0].plot(
        coefficients["nbar"], scale * coefficients["a_XX_per_gate"],
        "s-", label=r"$a_{XX}$: magnitude of $-C_\gamma^{XX}$",
    )
    axes[0].set_ylabel(r"dissipative coefficient [$10^{-7}$/gate]")
    axes[0].legend()

    axes[1].plot(
        coefficients["nbar"], scale * coefficients["b_XX_rad_per_gate"],
        "^-", color="tab:green", label=r"$b_{XX}$: coherent $XX$",
    )
    axes[1].set_ylabel(r"coherent coefficient [$10^{-7}$ rad/gate]")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel(r"thermal occupation $\bar n$")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(alpha=0.25)
    figure.suptitle(
        r"Coefficients of $K_{\rm int}\propto m_hm_d$"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def _select_nbar(frame: pd.DataFrame, requested: float) -> float:
    available = np.sort(frame["nbar"].unique().astype(float))
    matches = available[np.isclose(available, float(requested))]
    if not len(matches):
        raise ValueError(
            f"nbar={requested:g} is unavailable; choose from {available.tolist()}"
        )
    return float(matches[0])


def _plot_mode_rate_scaling(
    modeled: pd.DataFrame,
    coefficients: pd.DataFrame,
    path: Path,
    *,
    selected_nbar: float,
) -> Path:
    nbar = _select_nbar(modeled, selected_nbar)
    diagonal = modeled[
        np.isclose(modeled["nbar"], nbar)
        & np.isclose(modeled["multiplier_x"], modeled["multiplier_y"])
    ].sort_values("multiplier_x")
    coefficient = coefficients[np.isclose(coefficients["nbar"], nbar)].iloc[0]
    m_curve = np.linspace(0.0, float(diagonal["multiplier_x"].max()), 200)
    panels = [
        (
            "C_gamma_X_mean_per_gate", coefficient["a_X_per_gate"], 1.0,
            r"$C_{hd}^{(\gamma,X)}$", "[/gate]",
        ),
        (
            "C_gamma_XX_per_gate", coefficient["a_XX_per_gate"], -1.0,
            r"$C_{hd}^{(\gamma,XX)}$", "[/gate]",
        ),
        (
            "C_h_XX_rad_per_gate", coefficient["b_XX_rad_per_gate"], 1.0,
            r"$C_{hd}^{(h,XX)}$", "[rad/gate]",
        ),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for axis, (observed, fit_value, sign, title, unit) in zip(axes, panels):
        axis.scatter(
            diagonal["multiplier_x"], diagonal[observed],
            color="black", s=38, zorder=3, label="QPT interaction",
        )
        axis.plot(
            m_curve, sign * fit_value * np.square(m_curve),
            color="tab:red", label=r"bilinear model ($m_h=m_d=m$)",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel(r"common multiplier $m$")
        axis.set_ylabel(f"interaction {unit}")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(fr"Rate scaling at $\bar n={nbar:g}$")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_mode_model_heatmaps(
    modeled: pd.DataFrame,
    path: Path,
    *,
    selected_nbar: float,
) -> Path:
    nbar = _select_nbar(modeled, selected_nbar)
    selected = modeled[np.isclose(modeled["nbar"], nbar)]
    multipliers_x = np.sort(selected["multiplier_x"].unique())
    multipliers_y = np.sort(selected["multiplier_y"].unique())
    rows = [
        (
            "C_gamma_X_mean_per_gate",
            "model_C_gamma_X_mean_per_gate",
            "residual_C_gamma_X_mean_per_gate",
            r"local $C_{hd}^{(\gamma,X)}$", "[/gate]",
        ),
        (
            "C_gamma_XX_per_gate",
            "model_C_gamma_XX_per_gate",
            "residual_C_gamma_XX_per_gate",
            r"correlated $C_{hd}^{(\gamma,XX)}$", "[/gate]",
        ),
        (
            "C_h_XX_rad_per_gate",
            "model_C_h_XX_rad_per_gate",
            "residual_C_h_XX_rad_per_gate",
            r"coherent $C_{hd}^{(h,XX)}$", "[rad/gate]",
        ),
    ]
    figure, axes = plt.subplots(
        3, 3, figsize=(11.8, 9.2), constrained_layout=True
    )
    column_titles = ["QPT interaction", "bilinear model", "residual"]
    for row_index, (observed, predicted, residual, row_title, unit) in enumerate(rows):
        matrices = []
        for column in (observed, predicted, residual):
            matrix = selected.pivot(
                index="multiplier_y", columns="multiplier_x", values=column
            ).reindex(index=multipliers_y, columns=multipliers_x)
            matrices.append(matrix.to_numpy(float))
        vmax = max(
            float(np.max(np.abs(matrices[0]))),
            float(np.max(np.abs(matrices[1]))),
            1e-18,
        )
        images = []
        for column_index, (matrix, title) in enumerate(
            zip(matrices, column_titles)
        ):
            image = axes[row_index, column_index].imshow(
                matrix, origin="lower", aspect="auto", cmap="coolwarm",
                vmin=-vmax, vmax=vmax,
            )
            images.append(image)
            axes[row_index, column_index].set_title(title)
            axes[row_index, column_index].set_xticks(range(len(multipliers_x)))
            axes[row_index, column_index].set_xticklabels(
                [f"{value:g}" for value in multipliers_x]
            )
            axes[row_index, column_index].set_yticks(range(len(multipliers_y)))
            axes[row_index, column_index].set_yticklabels(
                [f"{value:g}" for value in multipliers_y]
            )
            axes[row_index, column_index].set_xlabel(r"$m_h$")
            axes[row_index, column_index].set_ylabel(r"$m_d$")
        axes[row_index, 0].text(
            -0.42, 0.5, row_title, rotation=90, va="center", ha="center",
            transform=axes[row_index, 0].transAxes,
        )
        figure.colorbar(
            images[-1], ax=axes[row_index, :].tolist(), shrink=0.80,
            label=f"interaction {unit}",
        )
    figure.suptitle(
        fr"Mode-resolved interaction model at $\bar n={nbar:g}$"
    )
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def run_mode_resolved_interaction_model(
    *,
    pairwise_generator_interactions_path: str | Path,
    output_dir: str | Path,
    selected_nbar: float = 4.0,
) -> dict[str, Any]:
    """Fit and plot the mode-resolved model without any ME/QPT evolution."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions = pd.read_csv(pairwise_generator_interactions_path)
    grid = prepare_mode_resolved_pair_grid(interactions)
    modeled, coefficients = fit_mode_resolved_bilinear_model(grid)
    nbar_key = f"{float(selected_nbar):g}".replace(".", "p")
    paths = {
        "grid": _atomic_save_csv(
            modeled, output_dir / "mode_resolved_bilinear_grid.csv"
        ),
        "coefficients": _atomic_save_csv(
            coefficients, output_dir / "mode_resolved_coefficients_by_nbar.csv"
        ),
    }
    paths["coefficient_figure"] = _plot_mode_coefficients_vs_nbar(
        coefficients, output_dir / "mode_coefficients_vs_nbar.png"
    )
    paths["rate_scaling_figure"] = _plot_mode_rate_scaling(
        modeled,
        coefficients,
        output_dir / f"mode_rate_scaling_nbar_{nbar_key}.png",
        selected_nbar=selected_nbar,
    )
    paths["heatmap_figure"] = _plot_mode_model_heatmaps(
        modeled,
        output_dir / f"mode_model_heatmaps_nbar_{nbar_key}.png",
        selected_nbar=selected_nbar,
    )
    return {
        "grid": modeled,
        "coefficients": coefficients,
        "paths": paths,
    }


__all__ = [
    "evaluate_mode_resolved_bilinear_residuals",
    "fit_mode_resolved_bilinear_model",
    "fit_pauli_mode_overlap_model",
    "fit_rate_product_model",
    "fit_shared_motion_model",
    "prepare_pair_grid",
    "prepare_mode_resolved_pair_grid",
    "run_existing_data_models",
    "run_mode_resolved_interaction_model",
]
