"""Polynomial-rate validation for the isolated motional-dephasing response.

Training uses the existing multipliers 0, 0.5, 1, 2 and 4.  Independent QPT
validation points at 0.25, 0.75, 1.5 and 3 times nominal rate are cached in a
separate output directory.  Degree-2 through degree-4 expansions are compared
on both the complete generator and every H/S/C/A coefficient.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drive_calibration_qpt_analysis as qpt_analysis
import error_generator_rate_nbar as egrn
import noise_error_structure as structure
import pairwise_noise_correlation as pnc


SOURCE = "motional_dephasing"
DEFAULT_NOMINAL_RATE = 18.0
DEFAULT_TRAIN_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0, 4.0)
DEFAULT_VALIDATION_MULTIPLIERS = (0.25, 0.75, 1.5, 3.0)
DEFAULT_NBAR_VALUES = (0.01, 1.0, 2.0, 3.0, 4.0)
DEFAULT_DEGREES = (2, 3, 4)
DEFAULT_KEY_MODES = (
    ("H", "XX"),
    ("S", "IX"),
    ("C", "IX,XI"),
    ("S", "XX"),
)


def _base_parameters(
    training_cache: Path,
    *,
    parallel_workers: int,
    show_progress: bool,
) -> dict:
    configuration = json.loads(
        (training_cache / "config.json").read_text(encoding="utf-8")
    )
    parameters = dict(configuration["base_parameters"])
    parameters["show_progress"] = bool(show_progress)
    parameters["parallel_workers"] = int(parallel_workers)
    return parameters


def _rate_vector(
    multiplier: float,
    nominal_rate: float,
) -> dict[str, float]:
    return {
        "motional_heating": 0.0,
        "motional_dephasing": float(multiplier * nominal_rate),
        "spin_dephasing": 0.0,
        "photon_scattering": 0.0,
    }


def _condition(
    multiplier: float,
    nominal_rate: float,
) -> dict[str, float | str]:
    rates = _rate_vector(multiplier, nominal_rate)
    return {
        "condition_id": egrn._condition_id(rates),
        **{
            egrn.RATE_COLUMNS[source]: value
            for source, value in rates.items()
        },
    }


def _ensure_validation_qpt(
    *,
    training_cache: Path,
    validation_cache: Path,
    execute: bool,
    nominal_rate: float,
    training_multipliers: np.ndarray,
    validation_multipliers: np.ndarray,
    nbar_values: np.ndarray,
    degrees: tuple[int, ...],
    parallel_workers: int,
    show_progress: bool,
    qpt_batch_size: int,
    analysis_name: str = "motional_dephasing_polynomial_validation",
    run_label: str = "independent validation QPT",
) -> pd.DataFrame:
    validation_cache.mkdir(parents=True, exist_ok=True)
    (validation_cache / "channel_cache").mkdir(parents=True, exist_ok=True)
    base_parameters = _base_parameters(
        training_cache,
        parallel_workers=parallel_workers,
        show_progress=show_progress,
    )
    rows = []
    for multiplier in validation_multipliers:
        condition = _condition(multiplier, nominal_rate)
        condition_id = str(condition["condition_id"])
        missing = [
            float(nbar) for nbar in nbar_values
            if not egrn._point_is_cached(
                validation_cache, condition_id, float(nbar)
            )
        ]
        if missing and execute:
            for start in range(0, len(missing), qpt_batch_size):
                batch = missing[start:start + qpt_batch_size]
                print(
                    f"Run {run_label}: "
                    f"motional dephasing m={multiplier:g}, nbar={batch}",
                    flush=True,
                )
                parameters = pnc.parameters_for_rate_vector(
                    base_parameters, condition, nbar_values=batch
                )
                points = qpt_analysis.calculate_error_channel_batch(
                    batch, parameters, convention="undo_before_actual"
                )
                for point in points:
                    metadata = {
                        **point["metadata"],
                        "condition_id": condition_id,
                        "rate_multiplier": float(multiplier),
                        **{
                            column: float(condition[column])
                            for column in egrn.RATE_COLUMNS.values()
                        },
                    }
                    qpt_analysis.save_qpt_point(
                        egrn.qpt_cache_path(
                            validation_cache, condition_id, point["n_bar"]
                        ),
                        point["n_bar"], condition_id, point["chi"], metadata,
                    )
        for nbar in nbar_values:
            cached = egrn._point_is_cached(
                validation_cache, condition_id, float(nbar)
            )
            rows.append({
                "condition_id": condition_id,
                "rate_multiplier": float(multiplier),
                "rate_s^-1": float(multiplier * nominal_rate),
                "nbar": float(nbar),
                "cached": bool(cached),
                "cache_path": str(egrn.qpt_cache_path(
                    validation_cache, condition_id, float(nbar)
                )),
            })
    index = pd.DataFrame(rows)
    index.to_csv(validation_cache / "validation_cache_index.csv", index=False)
    missing_count = int((~index["cached"]).sum())
    if missing_count:
        raise RuntimeError(
            f"{missing_count} independent validation points are missing. "
            "Run with --execute."
        )
    manifest = {
        "analysis": analysis_name,
        "training_multipliers": training_multipliers.tolist(),
        "validation_multipliers": validation_multipliers.tolist(),
        "nbar_values": nbar_values.tolist(),
        "nominal_rate_s^-1": nominal_rate,
        "polynomial_degrees": list(degrees),
        "qpt_batch_size": qpt_batch_size,
        "parallel_workers": parallel_workers,
        "show_progress": show_progress,
        "other_explicit_rates_s^-1": 0.0,
        "base_parameters": {
            key: value for key, value in base_parameters.items()
            if key not in {"parallel_workers", "show_progress"}
        },
    }
    (validation_cache / "config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return index


def _training_condition_id(
    catalog: pd.DataFrame,
    multiplier: float,
    nominal_rate: float,
) -> str:
    source_column = structure.RATE_COLUMNS[SOURCE]
    mask = np.isclose(catalog[source_column], multiplier * nominal_rate)
    for source, column in structure.RATE_COLUMNS.items():
        if source != SOURCE:
            mask &= np.isclose(catalog[column], 0.0)
    selected = catalog[mask]
    if len(selected) != 1:
        raise ValueError(f"Missing training condition at m={multiplier:g}")
    return str(selected.iloc[0]["condition_id"])


def _fit_polynomial(x: np.ndarray, values: np.ndarray, degree: int):
    design = np.column_stack([x**power for power in range(1, degree + 1)])
    flat = values.reshape(len(x), -1)
    coefficients = np.linalg.lstsq(design, flat, rcond=None)[0]
    return coefficients.reshape((degree,) + values.shape[1:])


def _predict_polynomial(
    x: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    prediction = sum(
        np.multiply.outer(x**power, coefficient).reshape(
            (len(x),) + coefficient.shape
        )
        for power, coefficient in enumerate(coefficients, start=1)
    )
    return prediction


def _relative_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    residual = actual - prediction
    aggregate = float(np.linalg.norm(residual) / np.linalg.norm(actual))
    point_errors = np.array([
        np.linalg.norm(error) / np.linalg.norm(value)
        if np.linalg.norm(value) > 0.0 else np.nan
        for error, value in zip(residual, actual)
    ])
    return {
        "aggregate_relative_residual": aggregate,
        "max_point_relative_residual": float(np.nanmax(point_errors)),
        "mean_point_relative_residual": float(np.nanmean(point_errors)),
    }


def analyze_polynomial_orders(
    *,
    training_cache: str | Path = (
        "results/error_generator_rate_nbar/top3_pairwise_generator"
    ),
    output_dir: str | Path = "results/motional_dephasing_polynomial",
    execute: bool = False,
    nominal_rate: float = DEFAULT_NOMINAL_RATE,
    training_multipliers=DEFAULT_TRAIN_MULTIPLIERS,
    validation_multipliers=DEFAULT_VALIDATION_MULTIPLIERS,
    nbar_values=DEFAULT_NBAR_VALUES,
    degrees=DEFAULT_DEGREES,
    key_modes=DEFAULT_KEY_MODES,
    parallel_workers: int | None = None,
    show_progress: bool = True,
    qpt_batch_size: int = 1,
):
    """Fit and independently validate polynomial rate-response models.

    ``execute=True`` calculates only missing independent-QPT validation points.
    Training points must already exist in ``training_cache``.  The polynomial
    constant is fixed to zero because the fitted object is K(rate)-K(0).
    """

    training_cache = Path(training_cache)
    output_dir = Path(output_dir)
    nominal_rate = float(nominal_rate)
    training_multipliers = np.asarray(training_multipliers, dtype=float)
    validation_multipliers = np.asarray(validation_multipliers, dtype=float)
    nbar_values = np.asarray(nbar_values, dtype=float)
    degrees = tuple(int(degree) for degree in degrees)
    key_modes = tuple(tuple(mode) for mode in key_modes)
    if parallel_workers is None:
        parallel_workers = min(32, os.cpu_count() or 1)
    parallel_workers = int(parallel_workers)
    qpt_batch_size = int(qpt_batch_size)
    if nominal_rate <= 0.0:
        raise ValueError("nominal_rate must be positive")
    if parallel_workers < 1:
        raise ValueError("parallel_workers must be at least 1")
    if qpt_batch_size < 1:
        raise ValueError("qpt_batch_size must be at least 1")
    if not degrees or min(degrees) < 1:
        raise ValueError("degrees must contain positive integers")
    if not np.any(np.isclose(training_multipliers, 0.0)):
        raise ValueError("training_multipliers must include zero")
    positive_training = training_multipliers[training_multipliers > 0.0]
    if len(positive_training) < max(degrees):
        raise ValueError(
            "A zero-intercept degree-d fit needs at least d positive "
            "training multipliers"
        )
    if len(np.unique(training_multipliers)) != len(training_multipliers):
        raise ValueError("training_multipliers must be unique")
    if len(np.unique(validation_multipliers)) != len(validation_multipliers):
        raise ValueError("validation_multipliers must be unique")
    if np.any(validation_multipliers <= 0.0):
        raise ValueError("validation_multipliers must be positive")
    if np.any(nbar_values < 0.0):
        raise ValueError("nbar_values must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_cache = output_dir / "independent_validation_qpt"
    validation_index = _ensure_validation_qpt(
        training_cache=training_cache,
        validation_cache=validation_cache,
        execute=execute,
        nominal_rate=nominal_rate,
        training_multipliers=training_multipliers,
        validation_multipliers=validation_multipliers,
        nbar_values=nbar_values,
        degrees=degrees,
        parallel_workers=parallel_workers,
        show_progress=show_progress,
        qpt_batch_size=qpt_batch_size,
    )
    catalog = pd.read_csv(training_cache / "condition_catalog.csv")
    zero_id = _training_condition_id(catalog, 0.0, nominal_rate)

    generator_rows = []
    taxonomy_rows = []
    response_data = {}
    taxonomy_data = {}
    for nbar in nbar_values:
        zero_generator = structure._load_projected_generator(
            training_cache, zero_id, float(nbar)
        )[0]
        for dataset, multipliers in (
            ("training", training_multipliers),
            ("validation", validation_multipliers),
        ):
            for multiplier in multipliers:
                if dataset == "training":
                    condition_id = _training_condition_id(
                        catalog, float(multiplier), nominal_rate
                    )
                    cache = training_cache
                else:
                    condition_id = str(validation_index[
                        np.isclose(validation_index["rate_multiplier"], multiplier)
                    ].iloc[0]["condition_id"])
                    cache = validation_cache
                generator = structure._load_projected_generator(
                    cache, condition_id, float(nbar)
                )[0]
                response = generator - zero_generator
                taxonomy = structure._decompose_generator_taxonomy(response)
                response_data[(dataset, float(nbar), float(multiplier))] = response
                taxonomy_data[(dataset, float(nbar), float(multiplier))] = taxonomy
                generator_rows.append({
                    "dataset": dataset,
                    "nbar": float(nbar),
                    "rate_multiplier": float(multiplier),
                    "rate_s^-1": float(multiplier * nominal_rate),
                    "response_frobenius_norm": float(np.linalg.norm(response)),
                })
                for (sector, mode), value in zip(
                    taxonomy["metadata"], taxonomy["coefficients"]
                ):
                    taxonomy_rows.append({
                        "dataset": dataset,
                        "nbar": float(nbar),
                        "rate_multiplier": float(multiplier),
                        "rate_s^-1": float(multiplier * nominal_rate),
                        "sector": sector,
                        "mode": mode,
                        "coefficient_per_gate": float(value),
                    })

    generator_table = pd.DataFrame(generator_rows)
    taxonomy_table = pd.DataFrame(taxonomy_rows)
    metric_rows = []
    polynomial_rows = []
    coefficient_metric_rows = []
    selected_prediction_rows = []

    train_x = positive_training
    validation_x = validation_multipliers
    maximum_multiplier = float(max(
        np.max(training_multipliers), np.max(validation_multipliers)
    ))
    dense_x = np.linspace(0.0, maximum_multiplier, 401)
    for nbar in nbar_values:
        train_generators = np.stack([
            response_data[("training", float(nbar), float(multiplier))]
            for multiplier in train_x
        ])
        validation_generators = np.stack([
            response_data[("validation", float(nbar), float(multiplier))]
            for multiplier in validation_x
        ])
        train_taxonomies = [
            taxonomy_data[("training", float(nbar), float(multiplier))]
            for multiplier in train_x
        ]
        validation_taxonomies = [
            taxonomy_data[("validation", float(nbar), float(multiplier))]
            for multiplier in validation_x
        ]
        metadata = train_taxonomies[0]["metadata"]
        train_coefficients = np.stack([
            item["coefficients"] for item in train_taxonomies
        ])
        validation_coefficients = np.stack([
            item["coefficients"] for item in validation_taxonomies
        ])

        for degree in degrees:
            generator_polynomial = _fit_polynomial(
                train_x, train_generators, degree
            )
            train_prediction = _predict_polynomial(
                train_x, generator_polynomial
            )
            validation_prediction = _predict_polynomial(
                validation_x, generator_polynomial
            )
            for dataset, actual, prediction in (
                ("training", train_generators, train_prediction),
                ("independent_validation", validation_generators,
                 validation_prediction),
            ):
                metric_rows.append({
                    "nbar": float(nbar),
                    "polynomial_degree": degree,
                    "matrix_family": "full_generator",
                    "dataset": dataset,
                    **_relative_metrics(actual, prediction),
                })

            coefficient_polynomial = _fit_polynomial(
                train_x, train_coefficients, degree
            )
            train_coefficient_prediction = _predict_polynomial(
                train_x, coefficient_polynomial
            )
            validation_coefficient_prediction = _predict_polynomial(
                validation_x, coefficient_polynomial
            )
            for coefficient_index, (sector, mode) in enumerate(metadata):
                for power in range(1, degree + 1):
                    polynomial_rows.append({
                        "nbar": float(nbar),
                        "polynomial_degree": degree,
                        "sector": sector,
                        "mode": mode,
                        "rate_power": power,
                        "coefficient_per_nominal_multiplier_power": float(
                            coefficient_polynomial[power - 1, coefficient_index]
                        ),
                        "coefficient_per_rate_s_power": float(
                            coefficient_polynomial[power - 1, coefficient_index]
                            / nominal_rate**power
                        ),
                    })
                for dataset, actual, prediction in (
                    ("training", train_coefficients[:, coefficient_index],
                     train_coefficient_prediction[:, coefficient_index]),
                    ("independent_validation",
                     validation_coefficients[:, coefficient_index],
                     validation_coefficient_prediction[:, coefficient_index]),
                ):
                    actual_2d = actual[:, None]
                    prediction_2d = prediction[:, None]
                    coefficient_metric_rows.append({
                        "nbar": float(nbar),
                        "polynomial_degree": degree,
                        "sector": sector,
                        "mode": mode,
                        "dataset": dataset,
                        **_relative_metrics(actual_2d, prediction_2d),
                        "max_abs_actual_per_gate": float(
                            np.max(np.abs(actual))
                        ),
                    })

            for sector, mode in key_modes:
                coefficient_index = metadata.index((sector, mode))
                dense_prediction = _predict_polynomial(
                    dense_x, coefficient_polynomial[:, coefficient_index]
                )
                for multiplier, prediction in zip(dense_x, dense_prediction):
                    selected_prediction_rows.append({
                        "nbar": float(nbar),
                        "polynomial_degree": degree,
                        "sector": sector,
                        "mode": mode,
                        "rate_multiplier": float(multiplier),
                        "predicted_coefficient_per_gate": float(prediction),
                    })

    metrics = pd.DataFrame(metric_rows)
    polynomial_coefficients = pd.DataFrame(polynomial_rows)
    coefficient_metrics = pd.DataFrame(coefficient_metric_rows)
    selected_predictions = pd.DataFrame(selected_prediction_rows)
    generator_table.to_csv(output_dir / "rate_points.csv", index=False)
    taxonomy_table.to_csv(
        output_dir / "rate_hsca_coefficients.csv", index=False
    )
    metrics.to_csv(
        output_dir / "polynomial_matrix_validation_metrics.csv", index=False
    )
    coefficient_metrics.to_csv(
        output_dir / "polynomial_coefficient_validation_metrics.csv", index=False
    )
    polynomial_coefficients.to_csv(
        output_dir / "polynomial_hsca_coefficients.csv", index=False
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), sharex=True)
    for degree in degrees:
        selected = metrics[
            metrics["polynomial_degree"].eq(degree)
            & metrics["dataset"].eq("independent_validation")
        ].sort_values("nbar")
        axes[0].semilogy(
            selected["nbar"], selected["aggregate_relative_residual"],
            marker="o", label=f"degree {degree}",
        )
        axes[1].semilogy(
            selected["nbar"], selected["max_point_relative_residual"],
            marker="o", label=f"degree {degree}",
        )
    axes[0].set_title("Aggregate independent-validation error")
    axes[1].set_title("Worst validation-rate error")
    for axis in axes:
        axis.set_xlabel("nbar")
        axis.set_ylabel("relative full-generator error")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Motional-dephasing polynomial-rate model validation")
    figure.tight_layout()
    degree_figure = output_dir / "polynomial_degree_validation.png"
    figure.savefig(degree_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)

    ncols = min(2, max(1, len(key_modes)))
    nrows = int(np.ceil(len(key_modes) / ncols))
    curve_figure, curve_axes = plt.subplots(
        nrows, ncols,
        figsize=(6.0 * ncols, 4.25 * nrows),
        squeeze=False,
    )
    nbar = float(np.max(nbar_values))
    for axis, (sector, mode) in zip(curve_axes.flat, key_modes):
        actual = taxonomy_table[
            np.isclose(taxonomy_table["nbar"], nbar)
            & taxonomy_table["sector"].eq(sector)
            & taxonomy_table["mode"].eq(mode)
        ]
        training = actual[actual["dataset"].eq("training")]
        validation = actual[actual["dataset"].eq("validation")]
        axis.scatter(
            training["rate_multiplier"], training["coefficient_per_gate"],
            color="black", marker="o", label="training QPT", zorder=5,
        )
        axis.scatter(
            validation["rate_multiplier"], validation["coefficient_per_gate"],
            color="red", marker="x", s=65, label="independent QPT", zorder=6,
        )
        for degree in degrees:
            curve = selected_predictions[
                np.isclose(selected_predictions["nbar"], nbar)
                & selected_predictions["polynomial_degree"].eq(degree)
                & selected_predictions["sector"].eq(sector)
                & selected_predictions["mode"].eq(mode)
            ]
            axis.plot(
                curve["rate_multiplier"],
                curve["predicted_coefficient_per_gate"],
                label=f"degree {degree}",
            )
        axis.set_title(f"{sector}:{mode}, nbar={nbar:g}")
        axis.set_xlabel("rate / nominal rate")
        axis.set_ylabel("coefficient per gate")
        axis.grid(alpha=0.25)
    for axis in curve_axes.flat[len(key_modes):]:
        axis.set_visible(False)
    curve_axes[0, 0].legend(fontsize=8)
    curve_figure.suptitle(
        "Training and independent validation of selected coefficients"
    )
    curve_figure.tight_layout()
    curve_path = output_dir / "selected_coefficient_polynomial_fits.png"
    curve_figure.savefig(curve_path, dpi=220, bbox_inches="tight")
    plt.close(curve_figure)

    return {
        "validation_index": validation_index,
        "metrics": metrics,
        "coefficient_metrics": coefficient_metrics,
        "polynomial_coefficients": polynomial_coefficients,
        "degree_figure": degree_figure,
        "curve_figure": curve_path,
        "configuration": {
            "training_cache": training_cache,
            "output_dir": output_dir,
            "nominal_rate": nominal_rate,
            "training_multipliers": training_multipliers,
            "validation_multipliers": validation_multipliers,
            "nbar_values": nbar_values,
            "degrees": degrees,
            "key_modes": key_modes,
            "parallel_workers": parallel_workers,
            "show_progress": show_progress,
            "qpt_batch_size": qpt_batch_size,
        },
    }


def analyze_dyson_frechet_taylor_convergence(
    *,
    training_cache: str | Path = (
        "results/error_generator_rate_nbar/top3_pairwise_generator"
    ),
    output_dir: str | Path = (
        "results/motional_dephasing_polynomial/dyson_frechet_taylor"
    ),
    execute: bool = False,
    nominal_rate: float = DEFAULT_NOMINAL_RATE,
    step_sizes=(0.2, 0.1, 0.05),
    stencil_multiples=(1.0, 2.0, 3.0, 4.0),
    nbar_values=DEFAULT_NBAR_VALUES,
    truncation_degrees=DEFAULT_DEGREES,
    key_modes=DEFAULT_KEY_MODES,
    parallel_workers: int | None = None,
    show_progress: bool = True,
    qpt_batch_size: int = 1,
    coefficient_rtol: float = 0.05,
):
    """Test zero-rate convergence of one-sided Taylor coefficients.

    For every step size ``h``, a zero-intercept local polynomial is fitted to
    the one-sided stencil ``m = h * stencil_multiples``.  The fitted matrices
    are coefficients of powers of the dimensionless multiplier
    ``m = Gamma / nominal_rate``.  Division by ``nominal_rate**p`` converts the
    p-th coefficient to a Taylor coefficient per physical-rate power.

    Negative Lindblad rates are never used.  With a degree-d local polynomial,
    the expected leading discretization error of coefficient p is
    O(h**(d + 1 - p)).  Comparing geometrically refined stencils therefore
    tests whether the numerical coefficients approach the derivatives at zero.
    """

    training_cache = Path(training_cache)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nominal_rate = float(nominal_rate)
    step_sizes = np.asarray(step_sizes, dtype=float)
    stencil_multiples = np.asarray(stencil_multiples, dtype=float)
    nbar_values = np.asarray(nbar_values, dtype=float)
    truncation_degrees = tuple(int(value) for value in truncation_degrees)
    key_modes = tuple(tuple(value) for value in key_modes)
    coefficient_rtol = float(coefficient_rtol)
    if parallel_workers is None:
        parallel_workers = min(32, os.cpu_count() or 1)
    parallel_workers = int(parallel_workers)
    qpt_batch_size = int(qpt_batch_size)

    if nominal_rate <= 0.0:
        raise ValueError("nominal_rate must be positive")
    if np.any(step_sizes <= 0.0) or len(np.unique(step_sizes)) != len(step_sizes):
        raise ValueError("step_sizes must be unique and positive")
    if (
        np.any(stencil_multiples <= 0.0)
        or len(np.unique(stencil_multiples)) != len(stencil_multiples)
    ):
        raise ValueError("stencil_multiples must be unique and positive")
    if not truncation_degrees or min(truncation_degrees) < 1:
        raise ValueError("truncation_degrees must contain positive integers")
    fit_degree = max(truncation_degrees)
    if len(stencil_multiples) < fit_degree:
        raise ValueError(
            "The stencil needs at least max(truncation_degrees) positive points"
        )
    if len(step_sizes) < 3:
        raise ValueError(
            "At least three step sizes are needed to estimate convergence order"
        )
    if np.any(nbar_values < 0.0):
        raise ValueError("nbar_values must be non-negative")
    if parallel_workers < 1 or qpt_batch_size < 1:
        raise ValueError("parallel_workers and qpt_batch_size must be positive")
    if coefficient_rtol <= 0.0:
        raise ValueError("coefficient_rtol must be positive")

    step_sizes = np.sort(step_sizes)[::-1]
    stencil_multiples = np.sort(stencil_multiples)
    refinement_ratios = step_sizes[:-1] / step_sizes[1:]
    if not np.allclose(
        refinement_ratios, refinement_ratios[0], rtol=1e-10, atol=1e-12
    ):
        raise ValueError(
            "step_sizes must form a geometric refinement sequence, such as "
            "(0.2, 0.1, 0.05)"
        )
    refinement_ratio = float(refinement_ratios[0])
    if refinement_ratio <= 1.0:
        raise ValueError("step_sizes must run from coarse to fine")

    near_zero_multipliers = np.unique(np.concatenate([
        step * stencil_multiples for step in step_sizes
    ]))
    qpt_cache = output_dir / "near_zero_qpt"
    qpt_index = _ensure_validation_qpt(
        training_cache=training_cache,
        validation_cache=qpt_cache,
        execute=execute,
        nominal_rate=nominal_rate,
        training_multipliers=np.array([0.0]),
        validation_multipliers=near_zero_multipliers,
        nbar_values=nbar_values,
        degrees=truncation_degrees,
        parallel_workers=parallel_workers,
        show_progress=show_progress,
        qpt_batch_size=qpt_batch_size,
        analysis_name="motional_dephasing_dyson_frechet_taylor_convergence",
        run_label="near-zero Taylor QPT",
    )

    catalog = pd.read_csv(training_cache / "condition_catalog.csv")
    zero_id = _training_condition_id(catalog, 0.0, nominal_rate)
    rate_rows = []
    generator_coefficient_rows = []
    taxonomy_coefficient_rows = []
    generator_convergence_rows = []
    taxonomy_convergence_rows = []
    observed_order_rows = []
    truncation_point_rows = []
    truncation_summary_rows = []
    coefficient_matrices = {}

    for nbar in nbar_values:
        nbar = float(nbar)
        zero_generator = structure._load_projected_generator(
            training_cache, zero_id, nbar
        )[0]
        responses = {}
        for multiplier in near_zero_multipliers:
            selected = qpt_index[
                np.isclose(qpt_index["rate_multiplier"], multiplier)
            ]
            condition_id = str(selected.iloc[0]["condition_id"])
            generator = structure._load_projected_generator(
                qpt_cache, condition_id, nbar
            )[0]
            response = generator - zero_generator
            responses[float(multiplier)] = response
            rate_rows.append({
                "nbar": nbar,
                "rate_multiplier": float(multiplier),
                "rate_s^-1": float(multiplier * nominal_rate),
                "response_frobenius_norm": float(np.linalg.norm(response)),
            })

        fits = {}
        taxonomies = {}
        for step in step_sizes:
            stencil_x = step * stencil_multiples
            stencil_values = np.stack([
                responses[float(near_zero_multipliers[
                    np.argmin(np.abs(near_zero_multipliers - multiplier))
                ])]
                for multiplier in stencil_x
            ])
            fitted = _fit_polynomial(stencil_x, stencil_values, fit_degree)
            fits[float(step)] = fitted
            coefficient_matrices[
                f"nbar_{nbar:g}__h_{float(step):g}"
            ] = fitted
            step_taxonomies = []
            for power, coefficient_matrix in enumerate(fitted, start=1):
                taxonomy = structure._decompose_generator_taxonomy(
                    coefficient_matrix
                )
                step_taxonomies.append(taxonomy)
                generator_coefficient_rows.append({
                    "nbar": nbar,
                    "step_size": float(step),
                    "maximum_stencil_multiplier": float(
                        step * np.max(stencil_multiples)
                    ),
                    "rate_power": power,
                    "coefficient_frobenius_per_multiplier_power": float(
                        np.linalg.norm(coefficient_matrix)
                    ),
                    "coefficient_frobenius_per_rate_s_power": float(
                        np.linalg.norm(coefficient_matrix)
                        / nominal_rate**power
                    ),
                })
                for (sector, mode), value in zip(
                    taxonomy["metadata"], taxonomy["coefficients"]
                ):
                    taxonomy_coefficient_rows.append({
                        "nbar": nbar,
                        "step_size": float(step),
                        "maximum_stencil_multiplier": float(
                            step * np.max(stencil_multiples)
                        ),
                        "rate_power": power,
                        "sector": sector,
                        "mode": mode,
                        "coefficient_per_multiplier_power": float(value),
                        "coefficient_per_rate_s_power": float(
                            value / nominal_rate**power
                        ),
                    })
            taxonomies[float(step)] = step_taxonomies

        for coarse_step, fine_step in zip(step_sizes[:-1], step_sizes[1:]):
            coarse = fits[float(coarse_step)]
            fine = fits[float(fine_step)]
            for power in range(1, fit_degree + 1):
                difference = coarse[power - 1] - fine[power - 1]
                difference_norm = float(np.linalg.norm(difference))
                fine_norm = float(np.linalg.norm(fine[power - 1]))
                relative_change = (
                    difference_norm / fine_norm if fine_norm > 0.0 else np.nan
                )
                generator_convergence_rows.append({
                    "nbar": nbar,
                    "rate_power": power,
                    "coarse_step_size": float(coarse_step),
                    "fine_step_size": float(fine_step),
                    "fine_maximum_stencil_multiplier": float(
                        fine_step * np.max(stencil_multiples)
                    ),
                    "coefficient_difference_frobenius": difference_norm,
                    "fine_coefficient_frobenius": fine_norm,
                    "relative_coefficient_change": relative_change,
                    "relative_converged": bool(
                        np.isfinite(relative_change)
                        and relative_change <= coefficient_rtol
                    ),
                    "coefficient_rtol": coefficient_rtol,
                })

                coarse_taxonomy = taxonomies[float(coarse_step)][power - 1]
                fine_taxonomy = taxonomies[float(fine_step)][power - 1]
                for (sector, mode), coarse_value, fine_value in zip(
                    fine_taxonomy["metadata"],
                    coarse_taxonomy["coefficients"],
                    fine_taxonomy["coefficients"],
                ):
                    absolute_change = float(abs(coarse_value - fine_value))
                    fine_abs = float(abs(fine_value))
                    component_relative_change = (
                        absolute_change / fine_abs if fine_abs > 0.0 else np.nan
                    )
                    taxonomy_convergence_rows.append({
                        "nbar": nbar,
                        "rate_power": power,
                        "sector": sector,
                        "mode": mode,
                        "coarse_step_size": float(coarse_step),
                        "fine_step_size": float(fine_step),
                        "fine_maximum_stencil_multiplier": float(
                            fine_step * np.max(stencil_multiples)
                        ),
                        "absolute_coefficient_change": absolute_change,
                        "fine_absolute_coefficient": fine_abs,
                        "relative_coefficient_change": component_relative_change,
                        "relative_converged": bool(
                            np.isfinite(component_relative_change)
                            and component_relative_change <= coefficient_rtol
                        ),
                        "coefficient_rtol": coefficient_rtol,
                    })

        for index in range(len(step_sizes) - 2):
            coarse_step = float(step_sizes[index])
            middle_step = float(step_sizes[index + 1])
            fine_step = float(step_sizes[index + 2])
            for power in range(1, fit_degree + 1):
                coarse_middle_difference = float(np.linalg.norm(
                    fits[coarse_step][power - 1]
                    - fits[middle_step][power - 1]
                ))
                middle_fine_difference = float(np.linalg.norm(
                    fits[middle_step][power - 1]
                    - fits[fine_step][power - 1]
                ))
                if (
                    coarse_middle_difference > 0.0
                    and middle_fine_difference > 0.0
                ):
                    observed_order = float(
                        np.log(coarse_middle_difference / middle_fine_difference)
                        / np.log(refinement_ratio)
                    )
                else:
                    observed_order = np.nan
                expected_order = fit_degree + 1 - power
                observed_order_rows.append({
                    "nbar": nbar,
                    "rate_power": power,
                    "coarse_step_size": coarse_step,
                    "middle_step_size": middle_step,
                    "fine_step_size": fine_step,
                    "coarse_middle_difference_frobenius": (
                        coarse_middle_difference
                    ),
                    "middle_fine_difference_frobenius": (
                        middle_fine_difference
                    ),
                    "observed_convergence_order": observed_order,
                    "expected_asymptotic_order": expected_order,
                    "order_difference": float(
                        observed_order - expected_order
                    ) if np.isfinite(observed_order) else np.nan,
                })

        finest_step = float(step_sizes[-1])
        finest_coefficients = fits[finest_step]
        finest_stencil = finest_step * stencil_multiples
        actual_all = np.stack([
            responses[float(multiplier)]
            for multiplier in near_zero_multipliers
        ])
        for degree in truncation_degrees:
            predicted_all = _predict_polynomial(
                near_zero_multipliers, finest_coefficients[:degree]
            )
            calibration_mask = np.array([
                np.any(np.isclose(multiplier, finest_stencil))
                for multiplier in near_zero_multipliers
            ])
            for multiplier, actual, predicted, in_stencil in zip(
                near_zero_multipliers,
                actual_all,
                predicted_all,
                calibration_mask,
            ):
                residual_norm = float(np.linalg.norm(actual - predicted))
                actual_norm = float(np.linalg.norm(actual))
                truncation_point_rows.append({
                    "nbar": nbar,
                    "polynomial_degree": degree,
                    "rate_multiplier": float(multiplier),
                    "rate_s^-1": float(multiplier * nominal_rate),
                    "finest_step_size": finest_step,
                    "in_finest_fit_stencil": bool(in_stencil),
                    "actual_response_frobenius": actual_norm,
                    "residual_frobenius": residual_norm,
                    "relative_residual": (
                        residual_norm / actual_norm
                        if actual_norm > 0.0 else np.nan
                    ),
                })
            for subset_name, subset_mask in (
                ("all_near_zero_points", np.ones_like(calibration_mask, dtype=bool)),
                ("finest_fit_stencil", calibration_mask),
                ("holdout_beyond_finest_stencil", ~calibration_mask),
            ):
                if not np.any(subset_mask):
                    continue
                truncation_summary_rows.append({
                    "nbar": nbar,
                    "polynomial_degree": degree,
                    "subset": subset_name,
                    "finest_step_size": finest_step,
                    "finest_maximum_stencil_multiplier": float(
                        finest_step * np.max(stencil_multiples)
                    ),
                    **_relative_metrics(
                        actual_all[subset_mask], predicted_all[subset_mask]
                    ),
                })

    rate_table = pd.DataFrame(rate_rows)
    generator_coefficients = pd.DataFrame(generator_coefficient_rows)
    taxonomy_coefficients = pd.DataFrame(taxonomy_coefficient_rows)
    generator_convergence = pd.DataFrame(generator_convergence_rows)
    taxonomy_convergence = pd.DataFrame(taxonomy_convergence_rows)
    observed_orders = pd.DataFrame(observed_order_rows)
    truncation_points = pd.DataFrame(truncation_point_rows)
    truncation_summary = pd.DataFrame(truncation_summary_rows)

    rate_table.to_csv(output_dir / "taylor_rate_points.csv", index=False)
    generator_coefficients.to_csv(
        output_dir / "taylor_generator_coefficients.csv", index=False
    )
    generator_convergence.to_csv(
        output_dir / "taylor_generator_coefficient_convergence.csv", index=False
    )
    observed_orders.to_csv(
        output_dir / "taylor_generator_observed_orders.csv", index=False
    )
    taxonomy_coefficients.to_csv(
        output_dir / "taylor_hsca_coefficients.csv", index=False
    )
    taxonomy_convergence.to_csv(
        output_dir / "taylor_hsca_coefficient_convergence.csv", index=False
    )
    truncation_points.to_csv(
        output_dir / "taylor_truncation_point_errors.csv", index=False
    )
    truncation_summary.to_csv(
        output_dir / "taylor_truncation_summary.csv", index=False
    )
    np.savez_compressed(
        output_dir / "taylor_generator_coefficient_matrices.npz",
        **coefficient_matrices,
    )

    configuration = {
        "analysis": "motional_dephasing_dyson_frechet_taylor_convergence",
        "training_cache": str(training_cache),
        "output_dir": str(output_dir),
        "nominal_rate_s^-1": nominal_rate,
        "step_sizes": step_sizes.tolist(),
        "stencil_multiples": stencil_multiples.tolist(),
        "near_zero_multipliers": near_zero_multipliers.tolist(),
        "nbar_values": nbar_values.tolist(),
        "fit_degree": fit_degree,
        "truncation_degrees": list(truncation_degrees),
        "key_modes": [list(value) for value in key_modes],
        "refinement_ratio": refinement_ratio,
        "coefficient_rtol": coefficient_rtol,
        "parallel_workers": parallel_workers,
        "show_progress": bool(show_progress),
        "qpt_batch_size": qpt_batch_size,
    }
    (output_dir / "taylor_analysis_config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), squeeze=False)
    for axis, power in zip(axes.flat, range(1, fit_degree + 1)):
        selected = generator_convergence[
            generator_convergence["rate_power"].eq(power)
        ]
        for nbar in nbar_values:
            curve = selected[np.isclose(selected["nbar"], nbar)].sort_values(
                "fine_maximum_stencil_multiplier"
            )
            axis.loglog(
                curve["fine_maximum_stencil_multiplier"],
                curve["relative_coefficient_change"],
                marker="o",
                label=f"nbar={float(nbar):g}",
            )
        axis.axhline(
            coefficient_rtol, color="black", linestyle="--", linewidth=1.0
        )
        axis.set_title(f"Taylor coefficient p={power}")
        axis.set_xlabel("maximum multiplier in finer stencil")
        axis.set_ylabel("relative change from previous stencil")
        axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Zero-rate convergence of generator Taylor coefficients")
    figure.tight_layout()
    coefficient_figure = output_dir / "taylor_coefficient_convergence.png"
    figure.savefig(coefficient_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), sharex=True)
    holdout = truncation_summary[
        truncation_summary["subset"].eq("holdout_beyond_finest_stencil")
    ]
    for degree in truncation_degrees:
        selected = holdout[
            holdout["polynomial_degree"].eq(degree)
        ].sort_values("nbar")
        axes[0].semilogy(
            selected["nbar"], selected["aggregate_relative_residual"],
            marker="o", label=f"degree {degree}",
        )
        axes[1].semilogy(
            selected["nbar"], selected["max_point_relative_residual"],
            marker="o", label=f"degree {degree}",
        )
    axes[0].set_title("Aggregate near-zero holdout error")
    axes[1].set_title("Worst near-zero holdout error")
    for axis in axes:
        axis.set_xlabel("nbar")
        axis.set_ylabel("relative full-generator error")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Taylor truncation from the finest zero-rate stencil")
    figure.tight_layout()
    truncation_figure = output_dir / "taylor_truncation_validation.png"
    figure.savefig(truncation_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)

    ncols = min(2, max(1, len(key_modes)))
    nrows = int(np.ceil(len(key_modes) / ncols))
    figure, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.0 * ncols, 4.25 * nrows),
        squeeze=False,
    )
    maximum_nbar = float(np.max(nbar_values))
    for axis, (sector, mode) in zip(axes.flat, key_modes):
        selected = taxonomy_convergence[
            np.isclose(taxonomy_convergence["nbar"], maximum_nbar)
            & taxonomy_convergence["sector"].eq(sector)
            & taxonomy_convergence["mode"].eq(mode)
        ]
        for power in range(1, fit_degree + 1):
            curve = selected[selected["rate_power"].eq(power)].sort_values(
                "fine_maximum_stencil_multiplier"
            )
            axis.loglog(
                curve["fine_maximum_stencil_multiplier"],
                curve["relative_coefficient_change"],
                marker="o", label=f"p={power}",
            )
        axis.axhline(
            coefficient_rtol, color="black", linestyle="--", linewidth=1.0
        )
        axis.set_title(f"{sector}:{mode}, nbar={maximum_nbar:g}")
        axis.set_xlabel("maximum multiplier in finer stencil")
        axis.set_ylabel("relative coefficient change")
        axis.grid(alpha=0.25, which="both")
    for axis in axes.flat[len(key_modes):]:
        axis.set_visible(False)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Zero-rate convergence of selected H/S/C coefficients")
    figure.tight_layout()
    taxonomy_figure = output_dir / "taylor_selected_hsca_convergence.png"
    figure.savefig(taxonomy_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)

    return {
        "qpt_index": qpt_index,
        "rate_points": rate_table,
        "generator_coefficients": generator_coefficients,
        "generator_convergence": generator_convergence,
        "observed_orders": observed_orders,
        "taxonomy_coefficients": taxonomy_coefficients,
        "taxonomy_convergence": taxonomy_convergence,
        "truncation_points": truncation_points,
        "truncation_summary": truncation_summary,
        "coefficient_figure": coefficient_figure,
        "truncation_figure": truncation_figure,
        "taxonomy_figure": taxonomy_figure,
        "configuration": configuration,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    result = analyze_polynomial_orders(execute=arguments.execute)
    print(result["metrics"].to_string(index=False))
    print(f"Saved {result['degree_figure']}")
    print(f"Saved {result['curve_figure']}")
