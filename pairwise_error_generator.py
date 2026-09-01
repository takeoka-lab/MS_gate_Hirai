"""Pairwise error-channel QPT and Pauli-generator interaction analysis.

The scalar pairwise grids already provide

    C_ij = I_ij - I_i - I_j + I_0.

This module ranks those interactions, reruns selected pairwise grids while
retaining the full error channel, and applies the same effective-generator
decomposition used by :mod:`error_generator_rate_nbar`.  Shared zero-noise
and single-axis conditions are deduplicated across selected pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

import drive_calibration_qpt_analysis as qpt_analysis
import error_generator_rate_nbar as egrn
import pairwise_noise_correlation as pnc


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
    selected.pop("n_bar_list", None)
    selected.pop("parallel_workers", None)
    selected.pop("show_progress", None)
    return _json_safe(selected)


def _atomic_save_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def rank_pairwise_infidelity_interactions(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Rank pairs by their largest active-grid ``abs(C_ij)``.

    The sign is retained.  A positive value is super-additive error; a
    negative value is sub-additive and must not be described as enhancement.
    Axis points, for which one multiplier is zero, are excluded.
    """

    required = {
        "pair_id", "source_x", "source_y", "multiplier_x",
        "multiplier_y", "nbar", "interaction_infidelity",
        "grid_infidelity", "additive_prediction",
        "relative_interaction_to_pair_noise",
    }
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"Interaction table is missing columns: {missing}")
    active = interactions[
        interactions["multiplier_x"].gt(0.0)
        & interactions["multiplier_y"].gt(0.0)
    ].copy()
    if active.empty:
        return pd.DataFrame()

    rows = []
    for pair_id, group in active.groupby("pair_id", sort=False):
        worst = group.loc[group["interaction_infidelity"].abs().idxmax()]
        most_positive = float(group["interaction_infidelity"].max())
        rows.append({
            "pair_id": str(pair_id),
            "source_x": str(worst["source_x"]),
            "source_y": str(worst["source_y"]),
            "signed_C_at_max_abs": float(worst["interaction_infidelity"]),
            "max_abs_C": float(abs(worst["interaction_infidelity"])),
            "most_positive_C": most_positive,
            "has_positive_enhancement": bool(most_positive > 0.0),
            "nbar_at_max_abs": float(worst["nbar"]),
            "multiplier_x_at_max_abs": float(worst["multiplier_x"]),
            "multiplier_y_at_max_abs": float(worst["multiplier_y"]),
            "grid_infidelity_at_max_abs": float(worst["grid_infidelity"]),
            "additive_prediction_at_max_abs": float(
                worst["additive_prediction"]
            ),
            "relative_C_at_max_abs": float(
                worst["relative_interaction_to_pair_noise"]
            ),
        })
    ranking = pd.DataFrame(rows).sort_values(
        "max_abs_C", ascending=False
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def build_selected_pairwise_plan(
    base_parameters: Mapping[str, Any],
    pair_ids: Iterable[str],
    rate_multipliers: Iterable[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
) -> dict[str, Any]:
    """Build deduplicated conditions for selected complete pair grids."""

    pair_ids = tuple(str(value) for value in pair_ids)
    if not pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair_ids must be non-empty and unique")
    catalogs = []
    requests = []
    pair_plans = {}
    for pair_id in pair_ids:
        parts = pair_id.split("__")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair_id: {pair_id}")
        source_x, source_y = parts
        plan = pnc.build_two_noise_grid_plan(
            base_parameters,
            source_x=source_x,
            source_y=source_y,
            rate_multipliers=rate_multipliers,
        )
        pair_plans[pair_id] = plan
        catalogs.append(plan["catalog"])
        request = plan["grid_requests"].copy()
        request.insert(0, "pair_id", pair_id)
        requests.append(request)

    catalog = (
        pd.concat(catalogs, ignore_index=True)
        .drop_duplicates("condition_id")
        .sort_values(["is_all_noise_zero", "condition_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    request_table = pd.concat(requests, ignore_index=True).sort_values(
        ["pair_id", "multiplier_y", "multiplier_x"]
    ).reset_index(drop=True)
    return {
        "pair_ids": pair_ids,
        "rate_multipliers": tuple(float(value) for value in rate_multipliers),
        "catalog": catalog,
        "requests": request_table,
        "pair_plans": pair_plans,
    }


def _manifest_payload(
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float],
    plan: Mapping[str, Any],
    convention: str,
) -> dict[str, Any]:
    return {
        "analysis": "selected_pairwise_error_generator",
        "version": 1,
        "generator_model": "diagonal Pauli dissipators after CPTP projection",
        "error_channel_convention": str(convention),
        "base_parameters": _scientific_parameters(base_parameters),
        "nbar_values": [float(value) for value in nbar_values],
        "rate_multipliers": list(plan["rate_multipliers"]),
        "pair_ids": list(plan["pair_ids"]),
        "pairwise_reference": (
            "Both selected rates vary; all other dissipative rates are zero."
        ),
    }


def _ensure_manifest(
    output_dir: Path, payload: Mapping[str, Any], *, resume: bool
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    current = _json_safe(dict(payload))
    if resume and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != current:
            saved_without_nbar = dict(saved)
            current_without_nbar = dict(current)
            saved_without_nbar.pop("nbar_values", None)
            current_without_nbar.pop("nbar_values", None)
            if saved_without_nbar != current_without_nbar:
                raise RuntimeError(
                    "Saved pairwise generator configuration differs from "
                    "the current physical settings or selected pairs."
                )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(current, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _point_is_cached(
    output_dir: Path, condition_id: str, nbar: float
) -> bool:
    path = egrn.qpt_cache_path(output_dir, condition_id, nbar)
    if not path.exists():
        return False
    try:
        point = egrn.load_qpt_point(path)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    return (
        point["condition_id"] == condition_id
        and np.isclose(point["nbar"], nbar, rtol=0.0, atol=1e-12)
    )


def run_or_load_selected_pairwise_qpt(
    *,
    output_dir: str | Path,
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float],
    pair_ids: Iterable[str],
    rate_multipliers: Iterable[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
    convention: str = "undo_before_actual",
    execute: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Run/load full error channels for selected pairwise grids."""

    output_dir = Path(output_dir)
    nbar_values = tuple(float(value) for value in nbar_values)
    plan = build_selected_pairwise_plan(
        base_parameters, pair_ids, rate_multipliers
    )
    manifest_path = _ensure_manifest(
        output_dir,
        _manifest_payload(base_parameters, nbar_values, plan, convention),
        resume=resume,
    )
    catalog_path = _atomic_save_csv(
        plan["catalog"], output_dir / "condition_catalog.csv"
    )
    request_path = _atomic_save_csv(
        plan["requests"], output_dir / "pairwise_grid_requests.csv"
    )
    (output_dir / "channel_cache").mkdir(parents=True, exist_ok=True)

    total_conditions = len(plan["catalog"])
    for condition_index, condition in plan["catalog"].iterrows():
        condition_id = str(condition["condition_id"])
        missing = tuple(
            nbar for nbar in nbar_values
            if not _point_is_cached(output_dir, condition_id, nbar)
        )
        if not missing or not execute:
            continue
        print(
            f"Run pairwise error-channel QPT {condition_index + 1}/"
            f"{total_conditions}: {condition_id}, nbar={missing}"
        )
        parameters = pnc.parameters_for_rate_vector(
            base_parameters, condition, nbar_values=missing
        )
        points = qpt_analysis.calculate_error_channel_batch(
            missing, parameters, convention=convention
        )
        for point in points:
            metadata = {
                **point["metadata"],
                "condition_id": condition_id,
                **{
                    column: float(condition[column])
                    for column in egrn.RATE_COLUMNS.values()
                },
            }
            qpt_analysis.save_qpt_point(
                egrn.qpt_cache_path(
                    output_dir, condition_id, point["n_bar"]
                ),
                point["n_bar"], condition_id, point["chi"], metadata,
            )

    point_rows = []
    pending_rows = []
    for _, condition in plan["catalog"].iterrows():
        condition_id = str(condition["condition_id"])
        missing = []
        for nbar in nbar_values:
            if _point_is_cached(output_dir, condition_id, nbar):
                point_rows.append({
                    "condition_id": condition_id,
                    "nbar": nbar,
                    "cache_path": str(
                        egrn.qpt_cache_path(output_dir, condition_id, nbar)
                    ),
                })
            else:
                missing.append(nbar)
        if missing:
            pending_rows.append({
                "condition_id": condition_id,
                "missing_nbars": ",".join(f"{value:g}" for value in missing),
                "missing_nbar_count": len(missing),
            })
    point_index = pd.DataFrame(point_rows)
    pending = pd.DataFrame(pending_rows)
    if not point_index.empty:
        point_index = point_index.sort_values(
            ["condition_id", "nbar"]
        ).reset_index(drop=True)
        _atomic_save_csv(point_index, output_dir / "channel_cache_index.csv")
    pending_nbar_count = (
        int(pending["missing_nbar_count"].sum()) if not pending.empty else 0
    )
    total_point_count = len(plan["catalog"]) * len(nbar_values)
    return {
        "plan": plan,
        "manifest_path": manifest_path,
        "catalog_path": catalog_path,
        "request_path": request_path,
        "point_index": point_index,
        "pending": pending,
        "complete": pending_nbar_count == 0,
        "completed_point_count": len(point_index),
        "total_point_count": total_point_count,
        "pending_nbar_count": pending_nbar_count,
        "pending_master_equation_evolutions": (
            pending_nbar_count
            * int(base_parameters.get("laser_noise_samples", 1))
            * 16
        ),
        "output_dir": output_dir,
    }


def extract_pairwise_generators(
    sweep_result: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """Extract all 15 Hamiltonian and dissipative Pauli coefficients."""

    if not bool(sweep_result["complete"]):
        raise RuntimeError("Pairwise error-channel QPT cache is incomplete")
    point_index = sweep_result["point_index"]
    plan = sweep_result["plan"]
    output_dir = Path(sweep_result["output_dir"])
    condition_rates = (
        plan["catalog"].set_index("condition_id").to_dict("index")
    )
    summary_rows = []
    coefficient_rows = []
    for point_number, point in enumerate(
        point_index.itertuples(index=False), start=1
    ):
        print(
            f"Extract pairwise generator {point_number}/{len(point_index)}: "
            f"{point.condition_id}, nbar={point.nbar:g}"
        )
        cached = egrn.load_qpt_point(point.cache_path)
        generator = qpt_analysis.extract_pauli_generator_observables(
            cached["chi"]
        )
        hamiltonian = generator.pop(
            "hamiltonian_coefficients_rad_per_gate"
        )
        dissipator = generator.pop("pauli_dissipator_rates_per_gate")
        generator.pop("projected_chi")
        rates = condition_rates[point.condition_id]
        summary_rows.append({
            "condition_id": point.condition_id,
            "nbar": float(point.nbar),
            **{
                column: float(rates[column])
                for column in egrn.RATE_COLUMNS.values()
            },
            **{key: float(value) for key, value in generator.items()},
        })
        for label in egrn.NONIDENTITY_PAULIS:
            coefficient_rows.append({
                "condition_id": point.condition_id,
                "nbar": float(point.nbar),
                "pauli": label,
                "pauli_weight": int(egrn.PAULI_WEIGHTS[label]),
                "mode_class": (
                    "local" if egrn.PAULI_WEIGHTS[label] == 1
                    else "correlated"
                ),
                "h_rad_per_gate": float(hamiltonian[label]),
                "gamma_per_gate": float(dissipator[label]),
            })
    condition_summary = pd.DataFrame(summary_rows).sort_values(
        ["condition_id", "nbar"]
    ).reset_index(drop=True)
    condition_coefficients = pd.DataFrame(coefficient_rows).sort_values(
        ["condition_id", "nbar", "pauli"]
    ).reset_index(drop=True)
    _atomic_save_csv(
        condition_summary, output_dir / "pairwise_generator_summary.csv"
    )
    _atomic_save_csv(
        condition_coefficients,
        output_dir / "pairwise_generator_pauli_coefficients.csv",
    )
    return {
        "summary": condition_summary,
        "coefficients": condition_coefficients,
    }


def calculate_pauli_generator_interactions(
    plan: Mapping[str, Any],
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate component-resolved ``C_ij^(P)`` on every selected grid."""

    requests = plan["requests"].copy()
    values = coefficients[[
        "condition_id", "nbar", "pauli", "pauli_weight", "mode_class",
        "h_rad_per_gate", "gamma_per_gate",
    ]].copy()
    grid = requests.merge(values, on="condition_id", how="inner")
    grid = grid.rename(columns={
        "h_rad_per_gate": "grid_h_rad_per_gate",
        "gamma_per_gate": "grid_gamma_per_gate",
    })

    def merge_reference(
        frame: pd.DataFrame, request_column: str, prefix: str
    ) -> pd.DataFrame:
        reference = values[[
            "condition_id", "nbar", "pauli",
            "h_rad_per_gate", "gamma_per_gate",
        ]].rename(columns={
            "condition_id": request_column,
            "h_rad_per_gate": f"{prefix}_h_rad_per_gate",
            "gamma_per_gate": f"{prefix}_gamma_per_gate",
        })
        return frame.merge(
            reference, on=[request_column, "nbar", "pauli"], how="inner"
        )

    grid = merge_reference(grid, "x_only_condition_id", "x_only")
    grid = merge_reference(grid, "y_only_condition_id", "y_only")
    grid = merge_reference(grid, "zero_condition_id", "zero")
    grid["interaction_gamma_per_gate"] = (
        grid["grid_gamma_per_gate"]
        - grid["x_only_gamma_per_gate"]
        - grid["y_only_gamma_per_gate"]
        + grid["zero_gamma_per_gate"]
    )
    grid["interaction_h_rad_per_gate"] = (
        grid["grid_h_rad_per_gate"]
        - grid["x_only_h_rad_per_gate"]
        - grid["y_only_h_rad_per_gate"]
        + grid["zero_h_rad_per_gate"]
    )
    return grid.sort_values([
        "pair_id", "nbar", "multiplier_y", "multiplier_x", "pauli"
    ]).reset_index(drop=True)


def rank_pauli_generator_interactions(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Rank Pauli modes within each pair by maximum active-grid interaction."""

    active = interactions[
        interactions["multiplier_x"].gt(0.0)
        & interactions["multiplier_y"].gt(0.0)
    ]
    rows = []
    for (pair_id, pauli), group in active.groupby(
        ["pair_id", "pauli"], sort=False
    ):
        worst = group.loc[
            group["interaction_gamma_per_gate"].abs().idxmax()
        ]
        rows.append({
            "pair_id": pair_id,
            "pauli": pauli,
            "pauli_weight": int(worst["pauli_weight"]),
            "mode_class": worst["mode_class"],
            "signed_interaction_gamma_at_max_abs": float(
                worst["interaction_gamma_per_gate"]
            ),
            "max_abs_interaction_gamma": float(
                abs(worst["interaction_gamma_per_gate"])
            ),
            "nbar_at_max_abs": float(worst["nbar"]),
            "multiplier_x_at_max_abs": float(worst["multiplier_x"]),
            "multiplier_y_at_max_abs": float(worst["multiplier_y"]),
        })
    ranked = pd.DataFrame(rows).sort_values(
        ["pair_id", "max_abs_interaction_gamma"],
        ascending=[True, False],
    ).reset_index(drop=True)
    ranked["rank_within_pair"] = (
        ranked.groupby("pair_id").cumcount() + 1
    )
    return ranked


def plot_top_pauli_interaction_maps(
    interactions: pd.DataFrame,
    output_dir: str | Path,
    *,
    top_n: int = 3,
) -> dict[str, Path]:
    """Plot top component-resolved interaction surfaces for each pair."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking = rank_pauli_generator_interactions(interactions)
    paths = {}
    for pair_id, pair_table in interactions.groupby("pair_id", sort=False):
        top_labels = ranking[
            ranking["pair_id"].eq(pair_id)
        ].head(top_n)["pauli"].tolist()
        nbars = np.sort(pair_table["nbar"].unique())
        multipliers_x = np.sort(pair_table["multiplier_x"].unique())
        multipliers_y = np.sort(pair_table["multiplier_y"].unique())
        selected = pair_table[pair_table["pauli"].isin(top_labels)]
        vmax = max(
            float(selected["interaction_gamma_per_gate"].abs().max()),
            1e-16,
        )
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        figure, axes = plt.subplots(
            len(top_labels), len(nbars),
            figsize=(3.15 * len(nbars), 2.85 * len(top_labels)),
            squeeze=False,
        )
        image = None
        for row, label in enumerate(top_labels):
            for column, nbar in enumerate(nbars):
                axis = axes[row, column]
                table = (
                    selected[
                        selected["pauli"].eq(label)
                        & np.isclose(selected["nbar"], nbar)
                    ]
                    .pivot(
                        index="multiplier_y", columns="multiplier_x",
                        values="interaction_gamma_per_gate",
                    )
                    .reindex(index=multipliers_y, columns=multipliers_x)
                )
                image = axis.imshow(
                    table.to_numpy(float), origin="lower", aspect="auto",
                    cmap="coolwarm", norm=norm,
                )
                axis.set_title(fr"{label}, $\bar n={nbar:g}$")
                axis.set_xticks(range(len(multipliers_x)))
                axis.set_xticklabels(
                    [f"{value:g}" for value in multipliers_x], fontsize=7
                )
                axis.set_yticks(range(len(multipliers_y)))
                axis.set_yticklabels(
                    [f"{value:g}" for value in multipliers_y], fontsize=7
                )
                axis.set_xlabel("x / nominal", fontsize=8)
                axis.set_ylabel("y / nominal", fontsize=8)
        figure.suptitle(
            pair_id.replace("__", " × ").replace("_", " "), fontsize=13
        )
        figure.subplots_adjust(
            left=0.06, right=0.90, bottom=0.08, top=0.90,
            wspace=0.35, hspace=0.42,
        )
        colorbar_axis = figure.add_axes([0.92, 0.14, 0.015, 0.68])
        figure.colorbar(
            image, cax=colorbar_axis,
            label=r"$C_{ij}^{(P)}$ for $\gamma_P$ per gate",
        )
        path = output_dir / f"top_pauli_interactions__{pair_id}.png"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        paths[pair_id] = path
    return paths


def save_pairwise_generator_outputs(
    sweep_result: Mapping[str, Any],
    generator_result: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Calculate and save Pauli-resolved interactions, ranking, and figures."""

    output_dir = Path(sweep_result["output_dir"])
    interactions = calculate_pauli_generator_interactions(
        sweep_result["plan"], generator_result["coefficients"]
    )
    ranking = rank_pauli_generator_interactions(interactions)
    interaction_path = _atomic_save_csv(
        interactions, output_dir / "pairwise_generator_interactions.csv"
    )
    ranking_path = _atomic_save_csv(
        ranking, output_dir / "pairwise_generator_interaction_ranking.csv"
    )
    figures = plot_top_pauli_interaction_maps(
        interactions, output_dir / "interaction_mode_maps"
    )
    return {
        "interactions": interactions,
        "ranking": ranking,
        "interaction_path": interaction_path,
        "ranking_path": ranking_path,
        "figures": figures,
    }


__all__ = [
    "build_selected_pairwise_plan",
    "calculate_pauli_generator_interactions",
    "extract_pairwise_generators",
    "plot_top_pauli_interaction_maps",
    "rank_pairwise_infidelity_interactions",
    "rank_pauli_generator_interactions",
    "run_or_load_selected_pairwise_qpt",
    "save_pairwise_generator_outputs",
]
