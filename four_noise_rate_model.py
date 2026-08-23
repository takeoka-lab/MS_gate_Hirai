"""Validation of arbitrary-rate four-noise infidelity models.

The training data are the single-noise axes and all six complete two-noise
5x5 grids.  Held-out four-noise rate vectors are evaluated with full QPT and
compared with three nested models:

1. single-noise additive,
2. pairwise correction from bilinear interpolation of each measured surface,
3. a reduced pairwise correction kappa_eff * m_i * m_j.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ms_gate_functions as mg
import pairwise_noise_correlation as pnc


NOISE_SOURCES = pnc.NOISE_SOURCES
RATE_COLUMNS = {
    "motional_heating": "motional_heating_s^-1",
    "motional_dephasing": "motional_dephasing_s^-1",
    "spin_dephasing": "spin_dephasing_s^-1",
    "photon_scattering": "photon_scattering_s^-1",
}
MULTIPLIER_COLUMNS = {
    source: f"{source}_multiplier" for source in NOISE_SOURCES
}
MODEL_COLUMNS = {
    "single_only_additive": "single_additive_prediction",
    "strict_pairwise_surface": "strict_pairwise_prediction",
    "reduced_bilinear_pairwise": "bilinear_pairwise_prediction",
}
RUNTIME_ONLY_PARAMETER_KEYS = {"parallel_workers", "show_progress"}


def _json_safe(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _scientific_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    selected = dict(parameters)
    for key in RUNTIME_ONLY_PARAMETER_KEYS:
        selected.pop(key, None)
    selected.pop("n_bar_list", None)
    return _json_safe(selected)


def _atomic_save_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def _condition_id(rates: Mapping[str, float]) -> str:
    values = [float(rates[source]) for source in NOISE_SOURCES]
    serialized = json.dumps(values, separators=(",", ":"))
    return "rates_" + sha256(serialized.encode("utf-8")).hexdigest()[:16]


def build_validation_rate_plan(
    base_parameters: Mapping[str, Any],
    *,
    n_points: int = 12,
    multiplier_bounds: tuple[float, float] = (0.1, 3.9),
    seed: int = 20260823,
) -> dict[str, Any]:
    """Build deterministic off-grid four-dimensional validation points."""

    n_points = int(n_points)
    if n_points < 4:
        raise ValueError("n_points must be at least 4")
    lower, upper = (float(value) for value in multiplier_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0 or upper <= lower:
        raise ValueError("multiplier_bounds must be finite with 0 <= lower < upper")
    nominal = {
        source: float(value)
        for source, value in mg.nominal_noise_source_strengths(
            base_parameters
        ).items()
        if source in NOISE_SOURCES
    }
    if set(nominal) != set(NOISE_SOURCES):
        raise ValueError("Could not determine all nominal noise rates")

    rng = np.random.default_rng(int(seed))
    unit_points = np.empty((n_points, len(NOISE_SOURCES)), dtype=float)
    for column in range(len(NOISE_SOURCES)):
        strata = rng.permutation(n_points)
        unit_points[:, column] = (strata + rng.random(n_points)) / n_points
    multipliers = lower + (upper - lower) * unit_points

    rows = []
    for index, point in enumerate(multipliers):
        multiplier_map = {
            source: float(point[source_index])
            for source_index, source in enumerate(NOISE_SOURCES)
        }
        rates = {
            source: multiplier_map[source] * nominal[source]
            for source in NOISE_SOURCES
        }
        rows.append({
            "validation_id": f"lhs_{index + 1:02d}",
            "condition_id": _condition_id(rates),
            **{
                MULTIPLIER_COLUMNS[source]: multiplier_map[source]
                for source in NOISE_SOURCES
            },
            **{
                RATE_COLUMNS[source]: rates[source]
                for source in NOISE_SOURCES
            },
        })
    return {
        "catalog": pd.DataFrame(rows),
        "nominal_strengths": nominal,
        "n_points": n_points,
        "multiplier_bounds": (lower, upper),
        "seed": int(seed),
    }


def _condition_complete(
    summary: pd.DataFrame,
    condition_id: str,
    nbar_values: Iterable[float],
) -> bool:
    if summary.empty or "condition_id" not in summary.columns:
        return False
    observed = np.sort(
        summary.loc[
            summary["condition_id"].eq(condition_id), "nbar"
        ].astype(float).unique()
    )
    expected = np.sort(np.asarray(tuple(nbar_values), dtype=float))
    return len(observed) == len(expected) and np.allclose(
        observed, expected, rtol=0.0, atol=1e-12
    )


def run_validation_qpt(
    *,
    output_dir: str | Path,
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float],
    plan: Mapping[str, Any],
    execute: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Run/load full four-noise QPT for held-out rate vectors."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nbar_values = tuple(float(value) for value in nbar_values)
    manifest = {
        "analysis": "held_out_four_noise_rate_validation",
        "version": 1,
        "base_parameters": _scientific_parameters(base_parameters),
        "nbar_values": list(nbar_values),
        "n_points": int(plan["n_points"]),
        "multiplier_bounds": list(plan["multiplier_bounds"]),
        "seed": int(plan["seed"]),
        "catalog": _json_safe(plan["catalog"].to_dict(orient="records")),
    }
    manifest_path = output_dir / "config.json"
    if resume and manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != manifest:
            raise RuntimeError(
                "Saved validation configuration differs from the current settings. "
                "Use a new output directory or set resume=False."
            )
    if execute or not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    catalog_path = _atomic_save_csv(
        plan["catalog"], output_dir / "validation_rate_catalog.csv"
    )
    summary_path = output_dir / "validation_qpt_summary.csv"
    if resume and summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.DataFrame()

    for condition_index, condition in plan["catalog"].iterrows():
        condition_id = str(condition["condition_id"])
        if resume and _condition_complete(summary, condition_id, nbar_values):
            continue
        if not execute:
            continue
        print(
            f"Run four-noise validation {condition_index + 1}/"
            f"{len(plan['catalog'])}: {condition['validation_id']}"
        )
        parameters = pnc.parameters_for_rate_vector(
            base_parameters, condition, nbar_values=nbar_values
        )
        simulation = mg.run_infidelity_analysis(show_plot=False, **parameters)
        rows = pd.DataFrame({
            "validation_id": str(condition["validation_id"]),
            "condition_id": condition_id,
            "nbar": np.asarray(
                simulation["parameters"]["n_bar_list"], dtype=float
            ),
            "F_avg": np.asarray(simulation["f_avg_list"], dtype=float),
            "infidelity": np.asarray(simulation["infidelity_list"], dtype=float),
        })
        for source in NOISE_SOURCES:
            rows[MULTIPLIER_COLUMNS[source]] = float(
                condition[MULTIPLIER_COLUMNS[source]]
            )
            rows[RATE_COLUMNS[source]] = float(condition[RATE_COLUMNS[source]])
        if not summary.empty:
            summary = summary.loc[
                ~summary["condition_id"].eq(condition_id)
            ].copy()
        summary = pd.concat([summary, rows], ignore_index=True)
        summary = summary.sort_values(
            ["validation_id", "nbar"]
        ).reset_index(drop=True)
        _atomic_save_csv(summary, summary_path)

    pending = plan["catalog"][
        ~plan["catalog"]["condition_id"].map(
            lambda condition_id: _condition_complete(
                summary, condition_id, nbar_values
            )
        )
    ].reset_index(drop=True)
    evolutions_per_condition = (
        len(nbar_values)
        * int(base_parameters.get("laser_noise_samples", 1)) * 16
    )
    return {
        "plan": plan,
        "summary": summary,
        "pending_conditions": pending,
        "complete": len(pending) == 0,
        "total_conditions": len(plan["catalog"]),
        "completed_conditions": len(plan["catalog"]) - len(pending),
        "pending_condition_count": len(pending),
        "pending_master_equation_evolutions": (
            len(pending) * evolutions_per_condition
        ),
        "catalog_path": catalog_path,
        "summary_path": summary_path,
        "output_dir": output_dir,
    }


def _aligned_series(
    table: pd.DataFrame,
    nbar_values: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    grouped = table.groupby("nbar", sort=True)["infidelity"].mean()
    observed = grouped.index.to_numpy(float)
    if len(observed) != len(nbar_values) or not np.allclose(
        observed, nbar_values, rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"{label} has an incomplete nbar grid")
    return grouped.to_numpy(float)


def _bilinear_interpolate(
    x_values: np.ndarray,
    y_values: np.ndarray,
    matrix: np.ndarray,
    x: float,
    y: float,
) -> float:
    if x < x_values[0] or x > x_values[-1]:
        raise ValueError(f"x={x:g} is outside the trained multiplier range")
    if y < y_values[0] or y > y_values[-1]:
        raise ValueError(f"y={y:g} is outside the trained multiplier range")
    x_high = int(np.searchsorted(x_values, x, side="right"))
    y_high = int(np.searchsorted(y_values, y, side="right"))
    x_high = min(max(x_high, 1), len(x_values) - 1)
    y_high = min(max(y_high, 1), len(y_values) - 1)
    x_low = x_high - 1
    y_low = y_high - 1
    x_fraction = (
        (x - x_values[x_low]) / (x_values[x_high] - x_values[x_low])
    )
    y_fraction = (
        (y - y_values[y_low]) / (y_values[y_high] - y_values[y_low])
    )
    lower = (
        (1.0 - x_fraction) * matrix[y_low, x_low]
        + x_fraction * matrix[y_low, x_high]
    )
    upper = (
        (1.0 - x_fraction) * matrix[y_high, x_low]
        + x_fraction * matrix[y_high, x_high]
    )
    return float((1.0 - y_fraction) * lower + y_fraction * upper)


def build_model_comparison(
    *,
    base_parameters: Mapping[str, Any],
    pair_grid_summary: pd.DataFrame,
    pair_grid_interactions: pd.DataFrame,
    all_noise_zero_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Predict held-out full-noise QPT with all three error-budget models."""

    if validation_summary.empty:
        return {
            "predictions": pd.DataFrame(),
            "metrics": pd.DataFrame(),
            "kappa_effective": pd.DataFrame(),
        }
    nominal = {
        source: float(value)
        for source, value in mg.nominal_noise_source_strengths(
            base_parameters
        ).items()
        if source in NOISE_SOURCES
    }
    nbar_values = np.sort(all_noise_zero_summary["nbar"].astype(float).unique())
    zero_infidelity = _aligned_series(
        all_noise_zero_summary, nbar_values, label="all-noise-zero reference"
    )
    zero_by_nbar = dict(zip(nbar_values, zero_infidelity))
    trained_multipliers = np.sort(
        pair_grid_interactions["multiplier_x"].astype(float).unique()
    )

    single_components: dict[tuple[str, float], np.ndarray] = {}
    for source in NOISE_SOURCES:
        for multiplier in trained_multipliers:
            mask = np.ones(len(pair_grid_summary), dtype=bool)
            for candidate in NOISE_SOURCES:
                expected = (
                    multiplier * nominal[source] if candidate == source else 0.0
                )
                mask &= np.isclose(
                    pair_grid_summary[RATE_COLUMNS[candidate]], expected
                )
            values = _aligned_series(
                pair_grid_summary[mask],
                nbar_values,
                label=f"single {source} at multiplier {multiplier:g}",
            )
            single_components[(source, float(multiplier))] = (
                values - zero_infidelity
            )

    pair_surfaces: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    kappa_rows = []
    for (pair_id, nbar), group in pair_grid_interactions.groupby(
        ["pair_id", "nbar"]
    ):
        x_values = np.sort(group["multiplier_x"].astype(float).unique())
        y_values = np.sort(group["multiplier_y"].astype(float).unique())
        matrix = (
            group.pivot(
                index="multiplier_y",
                columns="multiplier_x",
                values="interaction_infidelity",
            )
            .reindex(index=y_values, columns=x_values)
            .to_numpy(float)
        )
        pair_surfaces[(str(pair_id), float(nbar))] = (
            x_values, y_values, matrix
        )
        positive = group[
            (group["multiplier_x"] > 0.0)
            & (group["multiplier_y"] > 0.0)
        ]
        product = (
            positive["multiplier_x"] * positive["multiplier_y"]
        ).to_numpy(float)
        interaction = positive["interaction_infidelity"].to_numpy(float)
        kappa = float(product @ interaction / (product @ product))
        source_i = str(group.iloc[0]["source_x"])
        source_j = str(group.iloc[0]["source_y"])
        residual = interaction - kappa * product
        kappa_rows.append({
            "pair_id": str(pair_id),
            "source_i": source_i,
            "source_j": source_j,
            "nbar": float(nbar),
            "kappa_effective_multiplier": kappa,
            "kappa_effective_physical_s2": (
                kappa / (nominal[source_i] * nominal[source_j])
            ),
            "fit_rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "fit_nrmse_to_max_interaction": float(
                np.sqrt(np.mean(np.square(residual)))
                / max(np.max(np.abs(interaction)), 1e-18)
            ),
        })
    kappa_effective = pd.DataFrame(kappa_rows).sort_values(
        ["pair_id", "nbar"]
    ).reset_index(drop=True)
    kappa_lookup = {
        (row.pair_id, float(row.nbar)): float(row.kappa_effective_multiplier)
        for row in kappa_effective.itertuples(index=False)
    }

    rows = []
    for validation in validation_summary.itertuples(index=False):
        nbar = float(validation.nbar)
        if nbar not in zero_by_nbar:
            raise ValueError(f"Validation nbar={nbar:g} is outside training data")
        multipliers = {
            source: float(getattr(validation, MULTIPLIER_COLUMNS[source]))
            for source in NOISE_SOURCES
        }
        single_sum = 0.0
        for source in NOISE_SOURCES:
            curve = np.asarray([
                single_components[(source, float(multiplier))][
                    int(np.flatnonzero(np.isclose(nbar_values, nbar))[0])
                ]
                for multiplier in trained_multipliers
            ])
            single_sum += float(np.interp(
                multipliers[source], trained_multipliers, curve
            ))

        strict_pair_sum = 0.0
        bilinear_pair_sum = 0.0
        for pair_id, pair_group in pair_grid_interactions.groupby("pair_id"):
            source_i = str(pair_group.iloc[0]["source_x"])
            source_j = str(pair_group.iloc[0]["source_y"])
            x_values, y_values, matrix = pair_surfaces[(str(pair_id), nbar)]
            strict_pair_sum += _bilinear_interpolate(
                x_values, y_values, matrix,
                multipliers[source_i], multipliers[source_j],
            )
            bilinear_pair_sum += (
                kappa_lookup[(str(pair_id), nbar)]
                * multipliers[source_i] * multipliers[source_j]
            )
        zero_value = zero_by_nbar[nbar]
        actual = float(validation.infidelity)
        single_prediction = zero_value + single_sum
        strict_prediction = single_prediction + strict_pair_sum
        bilinear_prediction = single_prediction + bilinear_pair_sum
        full_noise_penalty = actual - zero_value
        denominator = max(abs(full_noise_penalty), 1e-18)
        row = {
            "validation_id": str(validation.validation_id),
            "condition_id": str(validation.condition_id),
            "nbar": nbar,
            **{
                MULTIPLIER_COLUMNS[source]: multipliers[source]
                for source in NOISE_SOURCES
            },
            "zero_infidelity": zero_value,
            "actual_infidelity": actual,
            "full_noise_penalty": full_noise_penalty,
            "single_noise_sum": single_sum,
            "strict_pairwise_correction": strict_pair_sum,
            "bilinear_pairwise_correction": bilinear_pair_sum,
            "single_additive_prediction": single_prediction,
            "strict_pairwise_prediction": strict_prediction,
            "bilinear_pairwise_prediction": bilinear_prediction,
        }
        for model_name, column in MODEL_COLUMNS.items():
            residual = actual - row[column]
            row[f"{model_name}_residual"] = residual
            row[f"{model_name}_relative_residual"] = residual / denominator
        rows.append(row)
    predictions = pd.DataFrame(rows).sort_values(
        ["validation_id", "nbar"]
    ).reset_index(drop=True)
    metrics = summarize_model_metrics(predictions)
    return {
        "predictions": predictions,
        "metrics": metrics,
        "kappa_effective": kappa_effective,
    }


def summarize_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize absolute and noise-penalty-normalized validation errors."""

    rows = []
    scopes = [("all", np.nan, predictions)]
    scopes.extend(
        ("nbar", float(nbar), group)
        for nbar, group in predictions.groupby("nbar")
    )
    for scope, nbar, group in scopes:
        for model_name, prediction_column in MODEL_COLUMNS.items():
            residual = (
                group["actual_infidelity"] - group[prediction_column]
            ).to_numpy(float)
            relative = residual / np.maximum(
                np.abs(group["full_noise_penalty"].to_numpy(float)), 1e-18
            )
            rows.append({
                "scope": scope,
                "nbar": nbar,
                "model": model_name,
                "count": len(group),
                "bias": float(np.mean(residual)),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "max_abs_residual": float(np.max(np.abs(residual))),
                "mean_abs_relative_to_noise_penalty": float(
                    np.mean(np.abs(relative))
                ),
                "rmse_relative_to_noise_penalty": float(
                    np.sqrt(np.mean(np.square(relative)))
                ),
                "max_abs_relative_to_noise_penalty": float(
                    np.max(np.abs(relative))
                ),
            })
    return pd.DataFrame(rows)


def plot_model_comparison(
    result: Mapping[str, pd.DataFrame],
    output_path: str | Path,
) -> Path:
    """Plot held-out prediction accuracy for the three candidate models."""

    predictions = result["predictions"]
    metrics = result["metrics"]
    if predictions.empty:
        raise ValueError("Model-comparison predictions are empty")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "single_only_additive": "tab:blue",
        "strict_pairwise_surface": "tab:green",
        "reduced_bilinear_pairwise": "tab:orange",
    }
    labels = {
        "single_only_additive": "Single-only additive",
        "strict_pairwise_surface": "Interpolated pair surfaces",
        "reduced_bilinear_pairwise": r"Reduced $\kappa_{ij}^{eff}m_im_j$",
    }
    figure, axes = plt.subplots(1, 3, figsize=(16.4, 4.8))
    actual = predictions["actual_infidelity"].to_numpy(float)
    limits = [
        min(actual.min(), *(predictions[col].min() for col in MODEL_COLUMNS.values())),
        max(actual.max(), *(predictions[col].max() for col in MODEL_COLUMNS.values())),
    ]
    padding = 0.04 * (limits[1] - limits[0])
    limits = [limits[0] - padding, limits[1] + padding]
    for model_name, prediction_column in MODEL_COLUMNS.items():
        axes[0].scatter(
            actual, predictions[prediction_column], s=24, alpha=0.72,
            color=colors[model_name], label=labels[model_name],
        )
    axes[0].plot(limits, limits, "k--", linewidth=1.0)
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_xlabel("Actual full-QPT infidelity")
    axes[0].set_ylabel("Predicted infidelity")
    axes[0].set_title("Held-out parity plot")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    box_values = []
    box_labels = []
    for model_name in MODEL_COLUMNS:
        box_values.append(
            100.0 * np.abs(
                predictions[f"{model_name}_relative_residual"].to_numpy(float)
            )
        )
        box_labels.append(labels[model_name])
    axes[1].boxplot(box_values, labels=box_labels, showfliers=True)
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=18, labelsize=7)
    axes[1].set_ylabel("|Residual| / full noise penalty (%)")
    axes[1].set_title("Error distribution over held-out points")
    axes[1].grid(alpha=0.25, which="both", axis="y")

    by_nbar = metrics[metrics["scope"].eq("nbar")]
    for model_name in MODEL_COLUMNS:
        subset = by_nbar[by_nbar["model"].eq(model_name)]
        axes[2].semilogy(
            subset["nbar"],
            100.0 * subset["rmse_relative_to_noise_penalty"],
            "o-", linewidth=1.8, color=colors[model_name],
            label=labels[model_name],
        )
    axes[2].set_xlabel(r"Mean phonon number $\bar n$")
    axes[2].set_ylabel("Relative RMSE (%)")
    axes[2].set_title("Generalization versus thermal occupation")
    axes[2].grid(alpha=0.25, which="both")
    axes[2].legend(fontsize=7)

    figure.suptitle(
        "Arbitrary-rate four-noise model validation",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_model_comparison(
    result: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save prediction, fit-coefficient, metric, and figure artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions_csv": _atomic_save_csv(
            result["predictions"], output_dir / "model_predictions.csv"
        ),
        "metrics_csv": _atomic_save_csv(
            result["metrics"], output_dir / "model_comparison_metrics.csv"
        ),
        "kappa_effective_csv": _atomic_save_csv(
            result["kappa_effective"], output_dir / "kappa_effective.csv"
        ),
    }
    paths["figure"] = plot_model_comparison(
        result, output_dir / "arbitrary_rate_model_comparison.png"
    )
    return paths


__all__ = [
    "NOISE_SOURCES",
    "RATE_COLUMNS",
    "MULTIPLIER_COLUMNS",
    "MODEL_COLUMNS",
    "build_validation_rate_plan",
    "run_validation_qpt",
    "build_model_comparison",
    "summarize_model_metrics",
    "plot_model_comparison",
    "save_model_comparison",
]
