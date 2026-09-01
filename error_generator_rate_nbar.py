"""Error-generator analysis over thermal occupation and noise rate.

This module completes the single-noise-axis part of the MS-gate analysis:

1. vary one of the four dissipative rates while keeping the other three at
   their nominal values,
2. reconstruct and cache the full QPT error channel for every ``nbar``, and
3. extract the diagonal-Pauli effective error generator from each channel.

The QPT cache is point-wise and resumable.  The nominal point shared by all
four sweep axes is simulated only once.
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

import drive_calibration_qpt_analysis as qpt_analysis
import ms_gate_functions as mg
import pairwise_noise_correlation as pnc


NOISE_SOURCES = pnc.NOISE_SOURCES
SOURCE_TITLES = pnc.SOURCE_TITLES
RATE_COLUMNS = {
    "motional_heating": "motional_heating_s^-1",
    "motional_dephasing": "motional_dephasing_s^-1",
    "spin_dephasing": "spin_dephasing_s^-1",
    "photon_scattering": "photon_scattering_s^-1",
}
PAULI_LABELS = tuple(label for label, _ in mg.pauli_labels_and_weights())
PAULI_WEIGHTS = dict(mg.pauli_labels_and_weights())
NONIDENTITY_PAULIS = PAULI_LABELS[1:]
LOCAL_PAULIS = tuple(
    label for label in NONIDENTITY_PAULIS if PAULI_WEIGHTS[label] == 1
)
CORRELATED_PAULIS = tuple(
    label for label in NONIDENTITY_PAULIS if PAULI_WEIGHTS[label] == 2
)
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
    selected.pop("n_bar_list", None)
    for key in RUNTIME_ONLY_PARAMETER_KEYS:
        selected.pop(key, None)
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


def _rate_columns(rate_vector: Mapping[str, float]) -> dict[str, float]:
    return {
        RATE_COLUMNS[source]: float(rate_vector[source])
        for source in NOISE_SOURCES
    }


def _nbar_key(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def build_single_axis_rate_plan(
    base_parameters: Mapping[str, Any],
    rate_multipliers: Iterable[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
) -> dict[str, Any]:
    """Build 20 plotted requests backed by 17 unique physical conditions."""

    multipliers = tuple(float(value) for value in rate_multipliers)
    if len(multipliers) != len(set(multipliers)):
        raise ValueError("rate_multipliers must not contain duplicates")
    if any(not np.isfinite(value) or value < 0.0 for value in multipliers):
        raise ValueError("rate_multipliers must be finite and non-negative")
    if not any(np.isclose(value, 1.0) for value in multipliers):
        raise ValueError("rate_multipliers must contain the nominal point 1.0")

    nominal_all = mg.nominal_noise_source_strengths(base_parameters)
    nominal = {
        source: float(nominal_all[source]) for source in NOISE_SOURCES
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in nominal.values()):
        raise ValueError("All four nominal dissipative rates must be positive")

    conditions: dict[str, dict[str, Any]] = {}
    requests = []
    for source in NOISE_SOURCES:
        for multiplier in multipliers:
            rates = dict(nominal)
            rates[source] = float(multiplier * nominal[source])
            condition_id = _condition_id(rates)
            conditions.setdefault(
                condition_id,
                {
                    "condition_id": condition_id,
                    **_rate_columns(rates),
                },
            )
            requests.append({
                "noise_source": source,
                "noise_title": SOURCE_TITLES[source],
                "multiplier": float(multiplier),
                "varied_strength_s^-1": float(rates[source]),
                "condition_id": condition_id,
                **_rate_columns(rates),
            })

    catalog = pd.DataFrame(conditions.values()).sort_values(
        "condition_id"
    ).reset_index(drop=True)
    request_table = pd.DataFrame(requests).sort_values(
        ["noise_source", "multiplier"]
    ).reset_index(drop=True)
    return {
        "catalog": catalog,
        "requests": request_table,
        "nominal_strengths": nominal,
        "rate_multipliers": multipliers,
    }


def _manifest_payload(
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float],
    plan: Mapping[str, Any],
    convention: str,
) -> dict[str, Any]:
    return {
        "analysis": "single_axis_error_generator_rate_nbar",
        "version": 1,
        "generator_model": "diagonal Pauli dissipators after CPTP projection",
        "error_channel_convention": str(convention),
        "base_parameters": _scientific_parameters(base_parameters),
        "nbar_values": [float(value) for value in nbar_values],
        "rate_multipliers": list(plan["rate_multipliers"]),
        "noise_sources": list(NOISE_SOURCES),
        "nominal_strengths": _json_safe(plan["nominal_strengths"]),
        "sweep_definition": (
            "Vary one dissipative rate; keep the other three nominal."
        ),
    }


def _ensure_manifest(
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    current = _json_safe(dict(payload))
    if resume and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != current:
            saved_without_nbar = dict(saved)
            current_without_nbar = dict(current)
            saved_nbar = saved_without_nbar.pop("nbar_values", None)
            current_nbar = current_without_nbar.pop("nbar_values", None)
            if saved_without_nbar != current_without_nbar:
                raise RuntimeError(
                    "Saved physical configuration differs from the current "
                    "settings. Use a new output directory or resume=False."
                )
            print(
                "Requested nbar grid changed; common point caches remain "
                f"reusable ({saved_nbar} -> {current_nbar})."
            )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(current, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def qpt_cache_path(
    output_dir: str | Path, condition_id: str, nbar: float
) -> Path:
    return (
        Path(output_dir)
        / "channel_cache"
        / f"{condition_id}__nbar_{_nbar_key(nbar)}.npz"
    )


def load_qpt_point(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate one cached trace-normalized chi matrix."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        chi = np.asarray(payload["chi_trace_normalized"], dtype=complex)
        nbar = float(np.asarray(payload["n_bar"]).item())
        condition_id = str(np.asarray(payload["condition"]).item())
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if chi.shape != (16, 16):
        raise ValueError(f"Unexpected chi shape in {path}: {chi.shape}")
    if not np.all(np.isfinite(chi)):
        raise ValueError(f"Non-finite chi entry in {path}")
    if not np.isclose(np.trace(chi), 1.0, atol=1e-8):
        raise ValueError(f"Chi is not trace normalized in {path}")
    return {
        "path": path,
        "nbar": nbar,
        "condition_id": condition_id,
        "chi": chi,
        "metadata": metadata,
    }


def _point_is_cached(
    output_dir: Path, condition_id: str, nbar: float
) -> bool:
    path = qpt_cache_path(output_dir, condition_id, nbar)
    if not path.exists():
        return False
    try:
        point = load_qpt_point(path)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    return (
        point["condition_id"] == condition_id
        and np.isclose(point["nbar"], nbar, rtol=0.0, atol=1e-12)
    )


def _missing_nbars(
    output_dir: Path,
    condition_id: str,
    nbar_values: Iterable[float],
) -> tuple[float, ...]:
    return tuple(
        float(nbar)
        for nbar in nbar_values
        if not _point_is_cached(output_dir, condition_id, float(nbar))
    )


def run_or_load_qpt_sweep(
    *,
    output_dir: str | Path,
    base_parameters: Mapping[str, Any],
    nbar_values: Iterable[float],
    rate_multipliers: Iterable[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
    convention: str = "undo_before_actual",
    execute: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Run or load the point-wise QPT channel sweep."""

    output_dir = Path(output_dir)
    nbar_values = tuple(float(value) for value in nbar_values)
    plan = build_single_axis_rate_plan(base_parameters, rate_multipliers)
    manifest = _manifest_payload(
        base_parameters, nbar_values, plan, convention
    )
    manifest_path = _ensure_manifest(output_dir, manifest, resume=resume)
    catalog_path = _atomic_save_csv(
        plan["catalog"], output_dir / "condition_catalog.csv"
    )
    request_path = _atomic_save_csv(
        plan["requests"], output_dir / "sweep_requests.csv"
    )
    cache_dir = output_dir / "channel_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_conditions = len(plan["catalog"])
    for condition_index, condition in plan["catalog"].iterrows():
        condition_id = str(condition["condition_id"])
        missing = _missing_nbars(output_dir, condition_id, nbar_values)
        if not missing or not execute:
            continue
        print(
            f"Run error-channel QPT {condition_index + 1}/{total_conditions}: "
            f"{condition_id}, nbar={missing}"
        )
        parameters = pnc.parameters_for_rate_vector(
            base_parameters, condition, nbar_values=missing
        )
        points = qpt_analysis.calculate_error_channel_batch(
            missing,
            parameters,
            convention=convention,
        )
        for point in points:
            metadata = {
                **point["metadata"],
                "condition_id": condition_id,
                **{
                    column: float(condition[column])
                    for column in RATE_COLUMNS.values()
                },
            }
            qpt_analysis.save_qpt_point(
                qpt_cache_path(
                    output_dir, condition_id, point["n_bar"]
                ),
                point["n_bar"],
                condition_id,
                point["chi"],
                metadata,
            )

    pending_rows = []
    point_rows = []
    for _, condition in plan["catalog"].iterrows():
        condition_id = str(condition["condition_id"])
        missing = _missing_nbars(output_dir, condition_id, nbar_values)
        if missing:
            pending_rows.append({
                "condition_id": condition_id,
                "missing_nbars": ",".join(f"{value:g}" for value in missing),
                "missing_nbar_count": len(missing),
            })
        for nbar in nbar_values:
            path = qpt_cache_path(output_dir, condition_id, nbar)
            if path.exists() and _point_is_cached(
                output_dir, condition_id, nbar
            ):
                point_rows.append({
                    "condition_id": condition_id,
                    "nbar": nbar,
                    "cache_path": str(path),
                })
    pending = pd.DataFrame(pending_rows)
    point_index = pd.DataFrame(point_rows)
    if not point_index.empty:
        point_index = point_index.sort_values(
            ["condition_id", "nbar"]
        ).reset_index(drop=True)
        _atomic_save_csv(point_index, output_dir / "channel_cache_index.csv")
    pending_nbar_count = int(
        pending["missing_nbar_count"].sum()
    ) if not pending.empty else 0
    evolutions_per_nbar = int(
        base_parameters.get("laser_noise_samples", 1)
    ) * 16
    return {
        "plan": plan,
        "manifest_path": manifest_path,
        "catalog_path": catalog_path,
        "request_path": request_path,
        "point_index": point_index,
        "pending": pending,
        "complete": pending_nbar_count == 0,
        "completed_point_count": len(point_index),
        "total_point_count": len(plan["catalog"]) * len(nbar_values),
        "pending_nbar_count": pending_nbar_count,
        "pending_master_equation_evolutions": (
            pending_nbar_count * evolutions_per_nbar
        ),
        "output_dir": output_dir,
    }


def _dominant_label(
    rates: Mapping[str, float], labels: Iterable[str]
) -> tuple[str, float]:
    labels = tuple(labels)
    label = max(labels, key=lambda item: float(rates[item]))
    return label, float(rates[label])


def extract_cached_generators(
    sweep_result: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, pd.DataFrame]:
    """Extract full Pauli generator coefficients from the cached channels."""

    if require_complete and not bool(sweep_result["complete"]):
        raise RuntimeError(
            "QPT cache is incomplete. Run the sweep with execute=True first."
        )
    plan = sweep_result["plan"]
    output_dir = Path(sweep_result["output_dir"])
    point_index = sweep_result["point_index"]
    catalog = plan["catalog"]
    if point_index.empty:
        return {
            "condition_summary": pd.DataFrame(),
            "condition_coefficients": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "coefficients": pd.DataFrame(),
        }

    condition_lookup = catalog.set_index("condition_id").to_dict("index")
    summary_rows = []
    coefficient_rows = []
    for point_number, point in enumerate(
        point_index.itertuples(index=False), start=1
    ):
        print(
            f"Extract generator {point_number}/{len(point_index)}: "
            f"{point.condition_id}, nbar={point.nbar:g}"
        )
        cached = load_qpt_point(point.cache_path)
        generator = qpt_analysis.extract_pauli_generator_observables(
            cached["chi"]
        )
        hamiltonian = generator.pop(
            "hamiltonian_coefficients_rad_per_gate"
        )
        dissipator = generator.pop("pauli_dissipator_rates_per_gate")
        generator.pop("projected_chi")
        dominant_local_label, dominant_local_rate = _dominant_label(
            dissipator, LOCAL_PAULIS
        )
        dominant_correlated_label, dominant_correlated_rate = (
            _dominant_label(dissipator, CORRELATED_PAULIS)
        )
        condition_rates = condition_lookup[point.condition_id]
        summary_rows.append({
            "condition_id": point.condition_id,
            "nbar": float(point.nbar),
            **{
                column: float(condition_rates[column])
                for column in RATE_COLUMNS.values()
            },
            "h_XX_rad_per_gate": float(hamiltonian["XX"]),
            "gamma_XX_per_gate": float(dissipator["XX"]),
            "dominant_local_pauli": dominant_local_label,
            "dominant_local_gamma_per_gate": dominant_local_rate,
            "dominant_correlated_pauli": dominant_correlated_label,
            "dominant_correlated_gamma_per_gate": dominant_correlated_rate,
            **{key: float(value) for key, value in generator.items()},
        })
        for label in NONIDENTITY_PAULIS:
            coefficient_rows.append({
                "condition_id": point.condition_id,
                "nbar": float(point.nbar),
                "pauli": label,
                "pauli_weight": int(PAULI_WEIGHTS[label]),
                "mode_class": (
                    "local" if PAULI_WEIGHTS[label] == 1 else "correlated"
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

    request_columns = [
        "noise_source", "noise_title", "multiplier",
        "varied_strength_s^-1", "condition_id",
    ]
    requests = plan["requests"][request_columns]
    summary = requests.merge(
        condition_summary, on="condition_id", how="inner"
    ).sort_values(
        ["noise_source", "multiplier", "nbar"]
    ).reset_index(drop=True)
    coefficients = requests.merge(
        condition_coefficients, on="condition_id", how="inner"
    ).sort_values(
        ["noise_source", "multiplier", "nbar", "pauli"]
    ).reset_index(drop=True)

    _atomic_save_csv(
        condition_summary,
        output_dir / "condition_generator_summary.csv",
    )
    _atomic_save_csv(
        condition_coefficients,
        output_dir / "condition_generator_pauli_coefficients.csv",
    )
    _atomic_save_csv(summary, output_dir / "generator_summary.csv")
    _atomic_save_csv(
        coefficients, output_dir / "generator_pauli_coefficients.csv"
    )
    return {
        "condition_summary": condition_summary,
        "condition_coefficients": condition_coefficients,
        "summary": summary,
        "coefficients": coefficients,
    }


def rank_dominant_modes(
    coefficients: pd.DataFrame,
    *,
    top_n: int = 3,
) -> pd.DataFrame:
    """Rank local and correlated dissipative modes at every sweep point."""

    if coefficients.empty:
        return pd.DataFrame()
    ranked = coefficients.copy()
    ranked["rank_within_class"] = (
        ranked.groupby(
            ["noise_source", "multiplier", "nbar", "mode_class"]
        )["gamma_per_gate"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return ranked[
        ranked["rank_within_class"].le(int(top_n))
    ].sort_values([
        "noise_source", "multiplier", "nbar", "mode_class",
        "rank_within_class",
    ]).reset_index(drop=True)


def plot_xx_rate_dependence(
    summary: pd.DataFrame, output_path: str | Path
) -> Path:
    """Plot the correlated XX dissipator versus nbar for every rate axis."""

    if summary.empty:
        raise ValueError("generator summary is empty")
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    for axis, source in zip(axes.flat, NOISE_SOURCES):
        subset = summary[summary["noise_source"].eq(source)]
        for multiplier, curve in subset.groupby("multiplier", sort=True):
            curve = curve.sort_values("nbar")
            axis.plot(
                curve["nbar"], curve["gamma_XX_per_gate"], "o-",
                linewidth=1.6, markersize=4, label=f"{multiplier:g}x",
            )
        axis.set_title(SOURCE_TITLES[source])
        axis.set_xlabel(r"Mean phonon number $\bar n$")
        axis.set_ylabel(r"$\gamma_{XX}$ per gate")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    figure.suptitle(
        r"Correlated $XX$ dissipator: $\bar n$ and rate dependence",
        fontsize=14,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_correlated_mode_maps(
    coefficients: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save one 3x3 correlated-Pauli heatmap panel for each noise axis."""

    if coefficients.empty:
        raise ValueError("generator coefficient table is empty")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for source in NOISE_SOURCES:
        source_table = coefficients[
            coefficients["noise_source"].eq(source)
            & coefficients["mode_class"].eq("correlated")
        ]
        multipliers = np.sort(source_table["multiplier"].unique())
        nbars = np.sort(source_table["nbar"].unique())
        values = source_table["gamma_per_gate"].to_numpy(float)
        vmax = max(float(np.nanmax(values)), 1e-16)
        figure, axes = plt.subplots(3, 3, figsize=(11.4, 9.0))
        image = None
        for axis, label in zip(axes.flat, CORRELATED_PAULIS):
            table = (
                source_table[source_table["pauli"].eq(label)]
                .pivot(index="nbar", columns="multiplier", values="gamma_per_gate")
                .reindex(index=nbars, columns=multipliers)
            )
            image = axis.imshow(
                table.to_numpy(float), origin="lower", aspect="auto",
                vmin=0.0, vmax=vmax, cmap="magma",
            )
            axis.set_title(label)
            axis.set_xticks(range(len(multipliers)))
            axis.set_xticklabels([f"{value:g}" for value in multipliers])
            axis.set_yticks(range(len(nbars)))
            axis.set_yticklabels([f"{value:g}" for value in nbars])
            axis.set_xlabel("rate / nominal")
            axis.set_ylabel(r"$\bar n$")
        figure.suptitle(
            f"Correlated Pauli dissipators — {SOURCE_TITLES[source]}",
            fontsize=14,
        )
        figure.subplots_adjust(
            left=0.07, right=0.87, bottom=0.07, top=0.90,
            wspace=0.34, hspace=0.42,
        )
        colorbar_axis = figure.add_axes([0.90, 0.13, 0.018, 0.70])
        figure.colorbar(
            image, cax=colorbar_axis, label=r"$\gamma_P$ per gate"
        )
        path = output_dir / f"correlated_modes__{source}.png"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        paths[source] = path
    return paths


def save_analysis_outputs(
    generator_result: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Save ranked tables and the two main generator figure families."""

    output_dir = Path(output_dir)
    summary = generator_result["summary"]
    coefficients = generator_result["coefficients"]
    ranked = rank_dominant_modes(coefficients, top_n=3)
    ranked_path = _atomic_save_csv(
        ranked, output_dir / "dominant_pauli_modes_top3.csv"
    )
    xx_path = plot_xx_rate_dependence(
        summary, output_dir / "gamma_xx_vs_nbar_by_noise_rate.png"
    )
    mode_paths = plot_correlated_mode_maps(
        coefficients, output_dir / "correlated_mode_maps"
    )
    return {
        "ranked": ranked,
        "ranked_path": ranked_path,
        "xx_figure": xx_path,
        "correlated_mode_figures": mode_paths,
    }


__all__ = [
    "CORRELATED_PAULIS",
    "LOCAL_PAULIS",
    "NOISE_SOURCES",
    "RATE_COLUMNS",
    "SOURCE_TITLES",
    "build_single_axis_rate_plan",
    "extract_cached_generators",
    "load_qpt_point",
    "plot_correlated_mode_maps",
    "plot_xx_rate_dependence",
    "qpt_cache_path",
    "rank_dominant_modes",
    "run_or_load_qpt_sweep",
    "save_analysis_outputs",
]
