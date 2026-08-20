"""Pairwise dissipative-noise cross sections for the MS-gate notebook.

For every unordered pair ``(i, j)``, this module keeps all other dissipative
noise sources off and evaluates two cross sections:

* sweep ``i`` while ``j`` is fixed at its nominal rate, and
* sweep ``j`` while ``i`` is fixed at its nominal rate.

Single-source-only references are calculated on the same rate grid so the
non-additive interaction

    C_ij = I_ij - I_i - I_j + I_0

can be evaluated without mixing background-noise conventions.  Conditions
shared by multiple plots are simulated only once and cached by their complete
four-rate vector.
"""

from __future__ import annotations

from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

import ms_gate_functions as mg


NOISE_SOURCES = (
    "motional_heating",
    "motional_dephasing",
    "spin_dephasing",
    "photon_scattering",
)
SOURCE_TITLES = {
    "motional_heating": "Motional heating",
    "motional_dephasing": "Motional dephasing",
    "spin_dephasing": "Spin dephasing",
    "photon_scattering": "Photon scattering",
}
SOURCE_SYMBOLS = {
    "motional_heating": r"$\dot{\bar n}$",
    "motional_dephasing": r"$\gamma_m$",
    "spin_dephasing": r"$\gamma_s$",
    "photon_scattering": r"$\gamma_{sc}$",
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


def _condition_id(rate_vector: Mapping[str, float]) -> str:
    values = [float(rate_vector[source]) for source in NOISE_SOURCES]
    serialized = json.dumps(values, separators=(",", ":"))
    return "rates_" + sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _multiplier_key(value: float) -> str:
    return f"{float(value):.12g}"


def _rate_columns(rate_vector: Mapping[str, float]) -> dict[str, float]:
    return {
        "motional_heating_s^-1": float(rate_vector["motional_heating"]),
        "motional_dephasing_s^-1": float(
            rate_vector["motional_dephasing"]
        ),
        "spin_dephasing_s^-1": float(rate_vector["spin_dephasing"]),
        "photon_scattering_s^-1": float(
            rate_vector["photon_scattering"]
        ),
    }


def _zero_rate_vector() -> dict[str, float]:
    return {source: 0.0 for source in NOISE_SOURCES}


def build_pairwise_sweep_plan(
    base_parameters: Mapping[str, Any],
    rate_multipliers: Iterable[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
    fixed_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Build deduplicated conditions and the six pairwise plot requests."""

    multipliers = tuple(float(value) for value in rate_multipliers)
    if len(multipliers) != len(set(multipliers)):
        raise ValueError("rate_multipliers must not contain duplicates")
    if any(not np.isfinite(value) or value < 0.0 for value in multipliers):
        raise ValueError("rate_multipliers must be finite and non-negative")
    if not any(np.isclose(value, 0.0) for value in multipliers):
        raise ValueError("rate_multipliers must contain 0.0")
    fixed_multiplier = float(fixed_multiplier)
    if not any(np.isclose(value, fixed_multiplier) for value in multipliers):
        raise ValueError("fixed_multiplier must be present in rate_multipliers")

    nominal = {
        source: float(value)
        for source, value in mg.nominal_noise_source_strengths(
            base_parameters
        ).items()
        if source in NOISE_SOURCES
    }
    if set(nominal) != set(NOISE_SOURCES):
        raise ValueError("Could not determine all four nominal noise rates")

    conditions: dict[str, dict[str, Any]] = {}

    def register(rate_vector: Mapping[str, float]) -> str:
        vector = {
            source: float(rate_vector.get(source, 0.0))
            for source in NOISE_SOURCES
        }
        condition_id = _condition_id(vector)
        row = {
            "condition_id": condition_id,
            **_rate_columns(vector),
            "is_all_noise_zero": bool(
                all(np.isclose(value, 0.0) for value in vector.values())
            ),
        }
        if condition_id in conditions and conditions[condition_id] != row:
            raise RuntimeError("Condition hash collision")
        conditions[condition_id] = row
        return condition_id

    zero_vector = _zero_rate_vector()
    zero_condition_id = register(zero_vector)

    single_rows = []
    single_lookup: dict[tuple[str, str], str] = {}
    for source in NOISE_SOURCES:
        for multiplier in multipliers:
            rates = _zero_rate_vector()
            rates[source] = multiplier * nominal[source]
            condition_id = register(rates)
            key = (source, _multiplier_key(multiplier))
            single_lookup[key] = condition_id
            single_rows.append(
                {
                    "noise_source": source,
                    "multiplier": multiplier,
                    "strength_s^-1": rates[source],
                    "condition_id": condition_id,
                }
            )

    pair_request_rows = []
    pairs = tuple(combinations(NOISE_SOURCES, 2))
    fixed_key = _multiplier_key(fixed_multiplier)
    for source_i, source_j in pairs:
        pair_id = f"{source_i}__{source_j}"
        for varied_source, fixed_source in (
            (source_i, source_j),
            (source_j, source_i),
        ):
            for multiplier in multipliers:
                rates = _zero_rate_vector()
                rates[varied_source] = multiplier * nominal[varied_source]
                rates[fixed_source] = fixed_multiplier * nominal[fixed_source]
                condition_id = register(rates)
                pair_request_rows.append(
                    {
                        "pair_id": pair_id,
                        "source_i": source_i,
                        "source_j": source_j,
                        "varied_source": varied_source,
                        "fixed_source": fixed_source,
                        "varied_multiplier": multiplier,
                        "varied_strength_s^-1": rates[varied_source],
                        "fixed_multiplier": fixed_multiplier,
                        "fixed_strength_s^-1": rates[fixed_source],
                        "condition_id": condition_id,
                        "varied_single_condition_id": single_lookup[
                            (varied_source, _multiplier_key(multiplier))
                        ],
                        "fixed_single_condition_id": single_lookup[
                            (fixed_source, fixed_key)
                        ],
                        "zero_condition_id": zero_condition_id,
                    }
                )

    catalog = pd.DataFrame(conditions.values()).sort_values(
        ["is_all_noise_zero", "condition_id"], ascending=[False, True]
    ).reset_index(drop=True)
    pair_requests = pd.DataFrame(pair_request_rows)
    single_requests = pd.DataFrame(single_rows)
    return {
        "rate_multipliers": multipliers,
        "fixed_multiplier": fixed_multiplier,
        "nominal_strengths": nominal,
        "pairs": pairs,
        "catalog": catalog,
        "pair_requests": pair_requests,
        "single_requests": single_requests,
        "zero_condition_id": zero_condition_id,
    }


def parameters_for_rate_vector(
    base_parameters: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    nbar_values: Iterable[float],
) -> dict[str, Any]:
    """Return simulation parameters for one complete four-rate vector."""

    parameters = dict(base_parameters)
    parameters["n_bar_list"] = [float(value) for value in nbar_values]
    heating = float(condition["motional_heating_s^-1"])
    motional_dephasing = float(condition["motional_dephasing_s^-1"])
    spin_dephasing = float(condition["spin_dephasing_s^-1"])
    scattering = float(condition["photon_scattering_s^-1"])
    parameters["heating_rate_phys"] = heating
    parameters["dephasing_rate_phys"] = motional_dephasing
    parameters["T2_star"] = (
        np.inf if np.isclose(spin_dephasing, 0.0) else 1.0 / spin_dephasing
    )
    rayleigh_nominal = float(base_parameters.get("rayleigh_rate_phys", 0.0))
    raman_nominal = float(base_parameters.get("raman_rate_phys", 0.0))
    scattering_nominal = rayleigh_nominal + raman_nominal
    rayleigh_fraction = (
        0.75
        if np.isclose(scattering_nominal, 0.0)
        else rayleigh_nominal / scattering_nominal
    )
    parameters["rayleigh_rate_phys"] = scattering * rayleigh_fraction
    parameters["raman_rate_phys"] = scattering * (1.0 - rayleigh_fraction)
    return parameters


def _manifest_payload(
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "analysis": "pairwise_dissipative_noise_cross_sections",
        "version": 1,
        "base_parameters": _scientific_parameters(base_parameters),
        "nbar_values": [float(value) for value in nbar_values],
        "rate_multipliers": list(plan["rate_multipliers"]),
        "fixed_multiplier": float(plan["fixed_multiplier"]),
        "noise_sources": list(NOISE_SOURCES),
        "nominal_strengths": _json_safe(plan["nominal_strengths"]),
        "sweep_definition": (
            "For every unordered pair, fix one source at fixed_multiplier "
            "times nominal and "
            "sweep the other source; all remaining sources are zero."
        ),
    }


def _ensure_manifest(
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    execute: bool,
    resume: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    current = _json_safe(dict(payload))
    if resume and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != current:
            raise RuntimeError(
                "Saved pairwise-noise configuration differs from the current "
                f"physical settings. Use a new output directory or set "
                f"resume=False: {path}"
            )
    if execute or not path.exists():
        path.write_text(
            json.dumps(current, indent=2, sort_keys=True), encoding="utf-8"
        )
    return path


def _condition_complete(
    summary: pd.DataFrame,
    condition_id: str,
    nbar_values: Iterable[float],
) -> bool:
    if summary.empty or "condition_id" not in summary:
        return False
    values = np.sort(
        summary.loc[
            summary["condition_id"].eq(condition_id), "nbar"
        ].astype(float).unique()
    )
    requested = np.sort(np.asarray(tuple(nbar_values), dtype=float))
    return len(values) == len(requested) and np.allclose(values, requested)


def _all_conditions_complete(
    summary: pd.DataFrame,
    catalog: pd.DataFrame,
    nbar_values: Iterable[float],
) -> bool:
    return all(
        _condition_complete(summary, condition_id, nbar_values)
        for condition_id in catalog["condition_id"]
    )


def _seed_zero_reference(
    summary: pd.DataFrame,
    all_noise_zero_summary: pd.DataFrame | None,
    plan: Mapping[str, Any],
    nbar_values: Iterable[float],
) -> pd.DataFrame:
    zero_condition_id = str(plan["zero_condition_id"])
    if _condition_complete(summary, zero_condition_id, nbar_values):
        return summary
    if all_noise_zero_summary is None or all_noise_zero_summary.empty:
        return summary
    requested = np.sort(np.asarray(tuple(nbar_values), dtype=float))
    available = np.sort(all_noise_zero_summary["nbar"].astype(float).unique())
    if len(requested) != len(available) or not np.allclose(requested, available):
        return summary
    zero_condition = plan["catalog"].loc[
        plan["catalog"]["condition_id"].eq(zero_condition_id)
    ].iloc[0]
    rows = all_noise_zero_summary[["nbar", "F_avg", "infidelity"]].copy()
    rows.insert(0, "condition_id", zero_condition_id)
    for column in (
        "motional_heating_s^-1",
        "motional_dephasing_s^-1",
        "spin_dephasing_s^-1",
        "photon_scattering_s^-1",
    ):
        rows[column] = float(zero_condition[column])
    rows["is_all_noise_zero"] = True
    if summary.empty:
        return rows.reindex(columns=summary.columns)
    summary = summary.loc[~summary["condition_id"].eq(zero_condition_id)].copy()
    return pd.concat([summary, rows], ignore_index=True)


def calculate_pairwise_interactions(
    plan: Mapping[str, Any],
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate C_ij for every completed pair request and nbar value."""

    if summary.empty:
        return pd.DataFrame()
    values = summary[["condition_id", "nbar", "infidelity"]].copy()
    pair = plan["pair_requests"].merge(
        values.rename(columns={"infidelity": "pair_infidelity"}),
        on="condition_id",
        how="inner",
    )

    def merge_reference(
        frame: pd.DataFrame,
        request_column: str,
        output_column: str,
    ) -> pd.DataFrame:
        reference = values.rename(
            columns={
                "condition_id": request_column,
                "infidelity": output_column,
            }
        )
        return frame.merge(reference, on=[request_column, "nbar"], how="inner")

    pair = merge_reference(
        pair, "varied_single_condition_id", "varied_only_infidelity"
    )
    pair = merge_reference(
        pair, "fixed_single_condition_id", "fixed_only_infidelity"
    )
    pair = merge_reference(pair, "zero_condition_id", "zero_infidelity")
    pair["additive_prediction"] = (
        pair["varied_only_infidelity"]
        + pair["fixed_only_infidelity"]
        - pair["zero_infidelity"]
    )
    pair["interaction_infidelity"] = (
        pair["pair_infidelity"] - pair["additive_prediction"]
    )
    pair["pair_noise_penalty"] = (
        pair["pair_infidelity"] - pair["zero_infidelity"]
    )
    denominator = np.maximum(np.abs(pair["pair_noise_penalty"]), 1e-18)
    pair["relative_interaction_to_pair_noise"] = (
        pair["interaction_infidelity"] / denominator
    )
    return pair.sort_values(
        ["pair_id", "varied_source", "varied_multiplier", "nbar"]
    ).reset_index(drop=True)


def _legend_label(source: str, strength: float) -> str:
    return (
        f"{SOURCE_SYMBOLS[source]}={float(strength):g} "
        + r"$\mathrm{s}^{-1}$"
    )


def plot_pairwise_cross_sections(
    plan: Mapping[str, Any],
    interactions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path | None]:
    """Plot 6x2 raw cross sections and their non-additive interactions."""

    output_dir = Path(output_dir)
    if interactions.empty:
        return {"infidelity_figure": None, "interaction_figure": None}
    pairs = plan["pairs"]
    multipliers = plan["rate_multipliers"]
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(multipliers)))

    raw_figure, raw_axes = plt.subplots(
        len(pairs), 2, figsize=(15.5, 4.0 * len(pairs)), squeeze=False
    )
    interaction_figure, interaction_axes = plt.subplots(
        len(pairs), 2, figsize=(15.5, 4.0 * len(pairs)), squeeze=False
    )
    for row_index, (source_i, source_j) in enumerate(pairs):
        pair_id = f"{source_i}__{source_j}"
        for column_index, (varied_source, fixed_source) in enumerate(
            ((source_i, source_j), (source_j, source_i))
        ):
            raw_axis = raw_axes[row_index, column_index]
            interaction_axis = interaction_axes[row_index, column_index]
            subset = interactions[
                interactions["pair_id"].eq(pair_id)
                & interactions["varied_source"].eq(varied_source)
                & interactions["fixed_source"].eq(fixed_source)
            ]
            for color, multiplier in zip(colors, multipliers):
                curve = subset[
                    np.isclose(subset["varied_multiplier"], multiplier)
                ].sort_values("nbar")
                if curve.empty:
                    continue
                label = _legend_label(
                    varied_source, curve["varied_strength_s^-1"].iloc[0]
                )
                raw_axis.semilogy(
                    curve["nbar"], curve["pair_infidelity"], "o-",
                    color=color, linewidth=1.7, markersize=4.5, label=label,
                )
                interaction_axis.plot(
                    curve["nbar"], curve["interaction_infidelity"], "o-",
                    color=color, linewidth=1.7, markersize=4.5, label=label,
                )
            zero_curve = (
                subset[["nbar", "zero_infidelity"]]
                .drop_duplicates("nbar")
                .sort_values("nbar")
            )
            raw_axis.semilogy(
                zero_curve["nbar"], zero_curve["zero_infidelity"],
                color="black", linestyle=":", linewidth=2.4,
                label="All four noises off",
            )
            fixed_strength = subset["fixed_strength_s^-1"].iloc[0]
            title = (
                f"Sweep {SOURCE_TITLES[varied_source]}\n"
                f"{SOURCE_TITLES[fixed_source]} fixed at "
                f"{fixed_strength:g} s$^{{-1}}$"
            )
            raw_axis.set_title(title)
            interaction_axis.set_title(title)
            raw_axis.set_xlabel(r"Mean phonon number $\bar n$")
            raw_axis.set_ylabel(r"Average infidelity $1-F_{avg}$")
            interaction_axis.set_xlabel(r"Mean phonon number $\bar n$")
            interaction_axis.set_ylabel(r"$C_{ij}$")
            interaction_axis.axhline(0.0, color="black", linewidth=1.0)
            raw_axis.grid(True, which="both", alpha=0.28)
            interaction_axis.grid(True, alpha=0.28)
            raw_axis.legend(fontsize=7.5)
            interaction_axis.legend(fontsize=7.5)

    raw_figure.suptitle(
        "Pairwise dissipative-noise cross sections (other noises off)",
        fontsize=16,
    )
    raw_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    raw_path = output_dir / "pairwise_infidelity_cross_sections.png"
    raw_figure.savefig(raw_path, dpi=220, bbox_inches="tight")
    plt.close(raw_figure)

    interaction_figure.suptitle(
        r"Pairwise non-additivity $C_{ij}=I_{ij}-I_i-I_j+I_0$",
        fontsize=16,
    )
    interaction_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    interaction_path = output_dir / "pairwise_interaction_cross_sections.png"
    interaction_figure.savefig(interaction_path, dpi=220, bbox_inches="tight")
    plt.close(interaction_figure)
    return {
        "infidelity_figure": raw_path,
        "interaction_figure": interaction_path,
    }


def run_pairwise_noise_correlation_sweep(
    *,
    output_dir: str | Path,
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float],
    rate_multipliers: Iterable[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
    fixed_multiplier: float = 1.0,
    all_noise_zero_summary: pd.DataFrame | None = None,
    execute: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Run/load all deduplicated pair and single-source QPT conditions."""

    output_dir = Path(output_dir)
    nbar_values = tuple(float(value) for value in nbar_values)
    plan = build_pairwise_sweep_plan(
        base_parameters, rate_multipliers, fixed_multiplier=fixed_multiplier
    )
    payload = _manifest_payload(base_parameters, nbar_values, plan)
    _ensure_manifest(output_dir, payload, execute=execute, resume=resume)
    summary_path = output_dir / "pairwise_qpt_summary.csv"
    interaction_path = output_dir / "pairwise_interaction_summary.csv"
    request_path = output_dir / "pairwise_plot_requests.csv"
    catalog_path = output_dir / "condition_catalog.csv"
    _atomic_save_csv(plan["catalog"], catalog_path)
    _atomic_save_csv(plan["pair_requests"], request_path)

    if resume and summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.DataFrame(columns=[
            "condition_id", "nbar", "F_avg", "infidelity",
            "motional_heating_s^-1", "motional_dephasing_s^-1",
            "spin_dephasing_s^-1", "photon_scattering_s^-1",
            "is_all_noise_zero",
        ])
    summary = _seed_zero_reference(
        summary, all_noise_zero_summary, plan, nbar_values
    )
    if not summary.empty:
        _atomic_save_csv(summary, summary_path)

    total_conditions = len(plan["catalog"])
    samples = int(base_parameters.get("laser_noise_samples", 1))
    for condition_index, condition in plan["catalog"].iterrows():
        condition_id = str(condition["condition_id"])
        if resume and _condition_complete(summary, condition_id, nbar_values):
            continue
        if not execute:
            continue
        print(
            f"Run pairwise condition {condition_index + 1}/{total_conditions}: "
            f"{condition_id}"
        )
        parameters = parameters_for_rate_vector(
            base_parameters, condition, nbar_values=nbar_values
        )
        result = mg.run_infidelity_analysis(show_plot=False, **parameters)
        rows = pd.DataFrame(
            {
                "condition_id": condition_id,
                "nbar": np.asarray(
                    result["parameters"]["n_bar_list"], dtype=float
                ),
                "F_avg": np.asarray(result["f_avg_list"], dtype=float),
                "infidelity": np.asarray(
                    result["infidelity_list"], dtype=float
                ),
            }
        )
        for column in (
            "motional_heating_s^-1",
            "motional_dephasing_s^-1",
            "spin_dephasing_s^-1",
            "photon_scattering_s^-1",
        ):
            rows[column] = float(condition[column])
        rows["is_all_noise_zero"] = bool(condition["is_all_noise_zero"])
        summary = summary.loc[~summary["condition_id"].eq(condition_id)].copy()
        summary = pd.concat([summary, rows], ignore_index=True)
        summary = summary.sort_values(
            ["condition_id", "nbar"]
        ).reset_index(drop=True)
        _atomic_save_csv(summary, summary_path)

    interactions = calculate_pairwise_interactions(plan, summary)
    if not interactions.empty:
        _atomic_save_csv(interactions, interaction_path)

    complete = _all_conditions_complete(summary, plan["catalog"], nbar_values)
    if complete:
        figures = plot_pairwise_cross_sections(
            plan, interactions, output_dir
        )
    else:
        figures = {"infidelity_figure": None, "interaction_figure": None}
    pending = plan["catalog"][
        ~plan["catalog"]["condition_id"].map(
            lambda condition_id: _condition_complete(
                summary, condition_id, nbar_values
            )
        )
    ].reset_index(drop=True)
    evolutions_per_condition = len(nbar_values) * samples * 16
    return {
        "plan": plan,
        "summary": summary,
        "interactions": interactions,
        "pending_conditions": pending,
        "complete": complete,
        "figures": figures,
        "total_unique_conditions": total_conditions,
        "pending_unique_conditions": len(pending),
        "evolutions_per_condition": evolutions_per_condition,
        "pending_master_equation_evolutions": (
            len(pending) * evolutions_per_condition
        ),
        "output_dir": output_dir,
    }


def build_two_noise_grid_plan(
    base_parameters: Mapping[str, Any],
    *,
    source_x: str = "motional_heating",
    source_y: str = "motional_dephasing",
    rate_multipliers: Iterable[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
) -> dict[str, Any]:
    """Build a complete rate grid for two sources with all others off."""

    if source_x not in NOISE_SOURCES or source_y not in NOISE_SOURCES:
        raise ValueError("source_x and source_y must be dissipative noise sources")
    if source_x == source_y:
        raise ValueError("source_x and source_y must be different")
    validated = build_pairwise_sweep_plan(
        base_parameters, rate_multipliers, fixed_multiplier=1.0
    )
    multipliers = validated["rate_multipliers"]
    nominal = validated["nominal_strengths"]
    conditions: dict[str, dict[str, Any]] = {}
    coordinate_lookup: dict[tuple[str, str], str] = {}

    for multiplier_y in multipliers:
        for multiplier_x in multipliers:
            rates = _zero_rate_vector()
            rates[source_x] = multiplier_x * nominal[source_x]
            rates[source_y] = multiplier_y * nominal[source_y]
            condition_id = _condition_id(rates)
            row = {
                "condition_id": condition_id,
                **_rate_columns(rates),
                "is_all_noise_zero": bool(
                    np.isclose(multiplier_x, 0.0)
                    and np.isclose(multiplier_y, 0.0)
                ),
            }
            conditions[condition_id] = row
            coordinate_lookup[
                (_multiplier_key(multiplier_x), _multiplier_key(multiplier_y))
            ] = condition_id

    zero_key = _multiplier_key(0.0)
    zero_condition_id = coordinate_lookup[(zero_key, zero_key)]
    request_rows = []
    for multiplier_y in multipliers:
        for multiplier_x in multipliers:
            x_key = _multiplier_key(multiplier_x)
            y_key = _multiplier_key(multiplier_y)
            request_rows.append(
                {
                    "source_x": source_x,
                    "source_y": source_y,
                    "multiplier_x": multiplier_x,
                    "multiplier_y": multiplier_y,
                    "strength_x_s^-1": multiplier_x * nominal[source_x],
                    "strength_y_s^-1": multiplier_y * nominal[source_y],
                    "condition_id": coordinate_lookup[(x_key, y_key)],
                    "x_only_condition_id": coordinate_lookup[(x_key, zero_key)],
                    "y_only_condition_id": coordinate_lookup[(zero_key, y_key)],
                    "zero_condition_id": zero_condition_id,
                }
            )
    catalog = pd.DataFrame(conditions.values()).sort_values(
        ["is_all_noise_zero", "condition_id"], ascending=[False, True]
    ).reset_index(drop=True)
    return {
        "source_x": source_x,
        "source_y": source_y,
        "rate_multipliers": multipliers,
        "nominal_strengths": nominal,
        "catalog": catalog,
        "grid_requests": pd.DataFrame(request_rows),
        "zero_condition_id": zero_condition_id,
    }


def _merge_reusable_conditions(
    summary: pd.DataFrame,
    reusable_summary: pd.DataFrame | None,
    catalog: pd.DataFrame,
    nbar_values: Iterable[float],
) -> pd.DataFrame:
    if reusable_summary is None or reusable_summary.empty:
        return summary
    allowed = set(catalog["condition_id"].astype(str))
    for condition_id in allowed:
        if _condition_complete(summary, condition_id, nbar_values):
            continue
        if not _condition_complete(reusable_summary, condition_id, nbar_values):
            continue
        rows = reusable_summary[
            reusable_summary["condition_id"].eq(condition_id)
        ].copy()
        summary = summary.loc[~summary["condition_id"].eq(condition_id)].copy()
        if summary.empty:
            summary = rows.reset_index(drop=True)
        else:
            summary = pd.concat([summary, rows], ignore_index=True)
    return summary


def calculate_two_noise_grid_interactions(
    plan: Mapping[str, Any],
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the interaction surface and its bilinear approximation."""

    if summary.empty:
        return pd.DataFrame()
    values = summary[["condition_id", "nbar", "infidelity"]].copy()
    grid = plan["grid_requests"].merge(
        values.rename(columns={"infidelity": "grid_infidelity"}),
        on="condition_id",
        how="inner",
    )

    def merge_reference(
        frame: pd.DataFrame,
        request_column: str,
        output_column: str,
    ) -> pd.DataFrame:
        reference = values.rename(
            columns={
                "condition_id": request_column,
                "infidelity": output_column,
            }
        )
        return frame.merge(reference, on=[request_column, "nbar"], how="inner")

    grid = merge_reference(grid, "x_only_condition_id", "x_only_infidelity")
    grid = merge_reference(grid, "y_only_condition_id", "y_only_infidelity")
    grid = merge_reference(grid, "zero_condition_id", "zero_infidelity")
    grid["additive_prediction"] = (
        grid["x_only_infidelity"]
        + grid["y_only_infidelity"]
        - grid["zero_infidelity"]
    )
    grid["interaction_infidelity"] = (
        grid["grid_infidelity"] - grid["additive_prediction"]
    )
    grid["pair_noise_penalty"] = (
        grid["grid_infidelity"] - grid["zero_infidelity"]
    )
    grid["relative_interaction_to_pair_noise"] = (
        grid["interaction_infidelity"]
        / np.maximum(np.abs(grid["pair_noise_penalty"]), 1e-18)
    )
    product = grid["multiplier_x"] * grid["multiplier_y"]
    grid["interaction_per_multiplier_product"] = np.where(
        product > 0.0,
        grid["interaction_infidelity"] / product,
        np.nan,
    )
    nominal = grid[
        np.isclose(grid["multiplier_x"], 1.0)
        & np.isclose(grid["multiplier_y"], 1.0)
    ][["nbar", "interaction_infidelity"]].rename(
        columns={"interaction_infidelity": "nominal_bilinear_coefficient"}
    )
    grid = grid.merge(nominal, on="nbar", how="left")
    grid["bilinear_prediction"] = (
        grid["nominal_bilinear_coefficient"] * product
    )
    grid["bilinear_residual"] = (
        grid["interaction_infidelity"] - grid["bilinear_prediction"]
    )
    grid["relative_bilinear_residual"] = np.where(
        product > 0.0,
        grid["bilinear_residual"]
        / np.maximum(np.abs(grid["interaction_infidelity"]), 1e-18),
        np.nan,
    )
    return grid.sort_values(
        ["nbar", "multiplier_y", "multiplier_x"]
    ).reset_index(drop=True)


def _annotated_grid_figure(
    grid: pd.DataFrame,
    *,
    value_column: str,
    multipliers: Iterable[float],
    title: str,
    colorbar_label: str,
    logarithmic: bool = False,
) -> plt.Figure:
    multipliers = tuple(float(value) for value in multipliers)
    nbar_values = tuple(sorted(grid["nbar"].astype(float).unique()))
    figure, axes = plt.subplots(
        1, len(nbar_values),
        figsize=(4.15 * len(nbar_values), 4.2),
        squeeze=False,
    )
    axes = axes[0]
    values = grid[value_column].to_numpy(float)
    if logarithmic:
        positive = values[values > 0.0]
        norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
        cmap = "viridis"
    else:
        scale = max(float(np.max(np.abs(values))), 1e-18)
        norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
        cmap = "RdBu_r"
    image = None
    for axis, n_bar in zip(axes, nbar_values):
        subset = grid[np.isclose(grid["nbar"], n_bar)]
        matrix = (
            subset.pivot(
                index="multiplier_y", columns="multiplier_x", values=value_column
            )
            .reindex(index=multipliers, columns=multipliers)
            .to_numpy(float)
        )
        image = axis.imshow(matrix, origin="lower", cmap=cmap, norm=norm)
        axis.set_xticks(range(len(multipliers)), [f"{value:g}" for value in multipliers])
        axis.set_yticks(range(len(multipliers)), [f"{value:g}" for value in multipliers])
        axis.set_xlabel(r"Heating multiplier $m_H$")
        axis.set_ylabel(r"Motional-dephasing multiplier $m_M$")
        axis.set_title(rf"$\bar n={n_bar:g}$")
        for row_index in range(len(multipliers)):
            for column_index in range(len(multipliers)):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index, row_index, f"{value:.1e}",
                    ha="center", va="center", fontsize=6.6,
                    color="white" if abs(value) > 0.55 * np.max(np.abs(matrix)) else "black",
                )
    if image is not None:
        figure.colorbar(image, ax=axes.tolist(), shrink=0.82, label=colorbar_label)
    figure.suptitle(title, fontsize=15)
    figure.subplots_adjust(left=0.05, right=0.93, bottom=0.13, top=0.84, wspace=0.32)
    return figure


def plot_two_noise_grid(
    plan: Mapping[str, Any],
    grid: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path | None]:
    """Plot infidelity, interaction, and bilinear-residual heatmaps."""

    output_dir = Path(output_dir)
    if grid.empty:
        return {
            "infidelity_figure": None,
            "interaction_figure": None,
            "bilinear_residual_figure": None,
        }
    multipliers = plan["rate_multipliers"]
    stem = f"{plan['source_x']}__{plan['source_y']}"
    specifications = (
        (
            "grid_infidelity",
            "Full 5x5 two-noise infidelity grid (other noises off)",
            r"Average infidelity $1-F_{avg}$",
            True,
            "infidelity_figure",
            output_dir / f"{stem}_infidelity_grid.png",
        ),
        (
            "interaction_infidelity",
            r"Pairwise interaction $C_{HM}$ over the full 5x5 grid",
            r"$C_{HM}$",
            False,
            "interaction_figure",
            output_dir / f"{stem}_interaction_grid.png",
        ),
        (
            "bilinear_residual",
            r"Residual from $C_{HM}(m_H,m_M)=C_{HM}(1,1)m_Hm_M$",
            "Bilinear residual",
            False,
            "bilinear_residual_figure",
            output_dir / f"{stem}_bilinear_residual_grid.png",
        ),
    )
    paths: dict[str, Path | None] = {}
    for column, title, colorbar, logarithmic, key, path in specifications:
        figure = _annotated_grid_figure(
            grid,
            value_column=column,
            multipliers=multipliers,
            title=title,
            colorbar_label=colorbar,
            logarithmic=logarithmic,
        )
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        paths[key] = path
    return paths


def run_two_noise_rate_grid(
    *,
    output_dir: str | Path,
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float],
    source_x: str = "motional_heating",
    source_y: str = "motional_dephasing",
    rate_multipliers: Iterable[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
    reusable_summary: pd.DataFrame | None = None,
    all_noise_zero_summary: pd.DataFrame | None = None,
    execute: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Run/load a complete 5x5 two-noise grid with cache reuse."""

    output_dir = Path(output_dir)
    nbar_values = tuple(float(value) for value in nbar_values)
    plan = build_two_noise_grid_plan(
        base_parameters,
        source_x=source_x,
        source_y=source_y,
        rate_multipliers=rate_multipliers,
    )
    payload = {
        "analysis": "complete_two_noise_rate_grid",
        "version": 1,
        "base_parameters": _scientific_parameters(base_parameters),
        "nbar_values": list(nbar_values),
        "source_x": source_x,
        "source_y": source_y,
        "rate_multipliers": list(plan["rate_multipliers"]),
        "nominal_strengths": _json_safe(plan["nominal_strengths"]),
    }
    _ensure_manifest(output_dir, payload, execute=execute, resume=resume)
    summary_path = output_dir / "two_noise_grid_qpt_summary.csv"
    interaction_path = output_dir / "two_noise_grid_interaction_summary.csv"
    request_path = output_dir / "two_noise_grid_requests.csv"
    _atomic_save_csv(plan["grid_requests"], request_path)
    if resume and summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.DataFrame(columns=[
            "condition_id", "nbar", "F_avg", "infidelity",
            "motional_heating_s^-1", "motional_dephasing_s^-1",
            "spin_dephasing_s^-1", "photon_scattering_s^-1",
            "is_all_noise_zero",
        ])
    summary = _merge_reusable_conditions(
        summary, reusable_summary, plan["catalog"], nbar_values
    )
    summary = _seed_zero_reference(
        summary, all_noise_zero_summary, plan, nbar_values
    )
    if not summary.empty:
        summary = summary.sort_values(["condition_id", "nbar"]).reset_index(drop=True)
        _atomic_save_csv(summary, summary_path)

    total_conditions = len(plan["catalog"])
    for condition_index, condition in plan["catalog"].iterrows():
        condition_id = str(condition["condition_id"])
        if resume and _condition_complete(summary, condition_id, nbar_values):
            continue
        if not execute:
            continue
        print(
            f"Run 5x5 grid condition {condition_index + 1}/{total_conditions}: "
            f"{condition_id}"
        )
        parameters = parameters_for_rate_vector(
            base_parameters, condition, nbar_values=nbar_values
        )
        result = mg.run_infidelity_analysis(show_plot=False, **parameters)
        rows = pd.DataFrame({
            "condition_id": condition_id,
            "nbar": np.asarray(result["parameters"]["n_bar_list"], dtype=float),
            "F_avg": np.asarray(result["f_avg_list"], dtype=float),
            "infidelity": np.asarray(result["infidelity_list"], dtype=float),
        })
        for column in (
            "motional_heating_s^-1",
            "motional_dephasing_s^-1",
            "spin_dephasing_s^-1",
            "photon_scattering_s^-1",
        ):
            rows[column] = float(condition[column])
        rows["is_all_noise_zero"] = bool(condition["is_all_noise_zero"])
        summary = summary.loc[~summary["condition_id"].eq(condition_id)].copy()
        summary = pd.concat([summary, rows], ignore_index=True)
        summary = summary.sort_values(["condition_id", "nbar"]).reset_index(drop=True)
        _atomic_save_csv(summary, summary_path)

    grid = calculate_two_noise_grid_interactions(plan, summary)
    if not grid.empty:
        _atomic_save_csv(grid, interaction_path)
    complete = _all_conditions_complete(summary, plan["catalog"], nbar_values)
    figures = (
        plot_two_noise_grid(plan, grid, output_dir)
        if complete
        else {
            "infidelity_figure": None,
            "interaction_figure": None,
            "bilinear_residual_figure": None,
        }
    )
    pending = plan["catalog"][
        ~plan["catalog"]["condition_id"].map(
            lambda condition_id: _condition_complete(summary, condition_id, nbar_values)
        )
    ].reset_index(drop=True)
    evolutions_per_condition = (
        len(nbar_values)
        * int(base_parameters.get("laser_noise_samples", 1))
        * 16
    )
    return {
        "plan": plan,
        "summary": summary,
        "grid": grid,
        "pending_conditions": pending,
        "complete": complete,
        "figures": figures,
        "total_unique_conditions": total_conditions,
        "reused_unique_conditions": total_conditions - len(pending),
        "pending_unique_conditions": len(pending),
        "evolutions_per_condition": evolutions_per_condition,
        "pending_master_equation_evolutions": len(pending) * evolutions_per_condition,
        "output_dir": output_dir,
    }


__all__ = [
    "NOISE_SOURCES",
    "SOURCE_TITLES",
    "build_pairwise_sweep_plan",
    "parameters_for_rate_vector",
    "calculate_pairwise_interactions",
    "plot_pairwise_cross_sections",
    "run_pairwise_noise_correlation_sweep",
    "build_two_noise_grid_plan",
    "calculate_two_noise_grid_interactions",
    "plot_two_noise_grid",
    "run_two_noise_rate_grid",
]
