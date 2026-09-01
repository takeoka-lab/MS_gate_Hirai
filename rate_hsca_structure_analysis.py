"""Rate-dependence audit of isolated-noise H/S/C/A generator structure.

The saved pairwise-generator cache contains all-zero and isolated-source axes
at multipliers 0, 0.5, 1, 2 and 4.  This script decomposes every axis point in
the complete H/S/C/A basis and tests separately whether

1. the normalized generator/stochastic structure is preserved with rate, and
2. each matrix or coefficient is linear in the varied rate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drive_calibration_qpt_analysis as qpt_analysis
import noise_error_structure as structure


NOMINAL_RATES = {
    "motional_heating": 10.0,
    "motional_dephasing": 18.0,
    "spin_dephasing": 10.0 / 3.0,
    "photon_scattering": 4.0,
}


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return np.nan
    return float(np.dot(left, right) / denominator)


def _fit_through_origin(x: np.ndarray, values: np.ndarray):
    """Fit values ~= x * slope for scalar or matrix-valued values."""
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    slope = np.tensordot(x, values, axes=(0, 0)) / float(np.dot(x, x))
    prediction = np.multiply.outer(x, slope).reshape(values.shape)
    residual = values - prediction
    relative_residual = float(
        np.linalg.norm(residual) / np.linalg.norm(values)
    ) if np.linalg.norm(values) > 0.0 else np.nan
    return slope, prediction, residual, relative_residual


def _stochastic_matrix(taxonomy: dict) -> np.ndarray:
    return structure._stochastic_eigenmodes(taxonomy)[1]


def _condition_for_axis(
    catalog: pd.DataFrame,
    source: str,
    multiplier: float,
) -> str:
    rate_columns = list(structure.RATE_COLUMNS.values())
    source_column = structure.RATE_COLUMNS[source]
    target_rate = NOMINAL_RATES[source] * float(multiplier)
    mask = np.isclose(catalog[source_column], target_rate)
    for column in rate_columns:
        if column != source_column:
            mask &= np.isclose(catalog[column], 0.0)
    selected = catalog[mask]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one isolated condition for {source}, m={multiplier}"
        )
    return str(selected.iloc[0]["condition_id"])


def analyze_rate_structure(
    *,
    cache_dir: str | Path = (
        "results/error_generator_rate_nbar/top3_pairwise_generator"
    ),
    output_dir: str | Path = "results/rate_hsca_structure",
):
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(cache_dir / "condition_catalog.csv")
    summary = pd.read_csv(cache_dir / "pairwise_generator_summary.csv")
    nbar_values = np.sort(summary["nbar"].unique().astype(float))
    multipliers = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    paulis = list(qpt_analysis._generator_design_data()[0][1:])
    collective_axis = np.zeros(len(paulis), dtype=float)
    collective_axis[paulis.index("IX")] = 1.0 / np.sqrt(2.0)
    collective_axis[paulis.index("XI")] = 1.0 / np.sqrt(2.0)

    coefficient_rows = []
    structure_rows = []
    matrix_fit_rows = []
    coefficient_fit_rows = []
    decompositions = {}
    generators = {}

    zero_condition = _condition_for_axis(
        catalog, "motional_heating", 0.0
    )
    zero_generators = {
        nbar: structure._load_projected_generator(
            cache_dir, zero_condition, nbar
        )[0]
        for nbar in nbar_values
    }

    for source in structure.RATE_COLUMNS:
        for multiplier in multipliers:
            condition_id = _condition_for_axis(catalog, source, multiplier)
            for nbar in nbar_values:
                generator = structure._load_projected_generator(
                    cache_dir, condition_id, nbar
                )[0]
                response = generator - zero_generators[nbar]
                taxonomy = structure._decompose_generator_taxonomy(response)
                generators[(source, nbar, multiplier)] = response
                decompositions[(source, nbar, multiplier)] = taxonomy
                for (sector, mode), coefficient in zip(
                    taxonomy["metadata"], taxonomy["coefficients"]
                ):
                    coefficient_rows.append({
                        "noise_source": source,
                        "noise_title": structure.SOURCE_TITLES[source],
                        "nbar": nbar,
                        "rate_multiplier": multiplier,
                        "rate_s^-1": multiplier * NOMINAL_RATES[source],
                        "sector": sector,
                        "mode": mode,
                        "coefficient_per_gate": float(coefficient),
                    })

    coefficients = pd.DataFrame(coefficient_rows)

    for source in structure.RATE_COLUMNS:
        for nbar in nbar_values:
            nonzero = multipliers[multipliers > 0.0]
            nominal_taxonomy = decompositions[(source, nbar, 1.0)]
            nominal_generator = generators[(source, nbar, 1.0)]
            nominal_stochastic = _stochastic_matrix(nominal_taxonomy)
            nominal_sector_fits = nominal_taxonomy["sector_fits"]

            matrix_families = {
                "full_generator": np.stack([
                    generators[(source, nbar, multiplier)]
                    for multiplier in nonzero
                ]),
                "stochastic_SC_matrix": np.stack([
                    _stochastic_matrix(
                        decompositions[(source, nbar, multiplier)]
                    )
                    for multiplier in nonzero
                ]),
            }
            for sector in ("H", "S", "C", "A"):
                matrix_families[f"{sector}_sector"] = np.stack([
                    decompositions[(source, nbar, multiplier)][
                        "sector_fits"
                    ][sector]
                    for multiplier in nonzero
                ])

            for family, values in matrix_families.items():
                slope, prediction, residual, relative_residual = (
                    _fit_through_origin(nonzero, values)
                )
                matrix_fit_rows.append({
                    "noise_source": source,
                    "noise_title": structure.SOURCE_TITLES[source],
                    "nbar": nbar,
                    "matrix_family": family,
                    "through_origin_relative_fit_residual": relative_residual,
                    "slope_frobenius_norm_per_nominal_multiplier": float(
                        np.linalg.norm(slope)
                    ),
                    "max_point_relative_deviation": float(max(
                        np.linalg.norm(error) / np.linalg.norm(value)
                        if np.linalg.norm(value) > 0.0 else np.nan
                        for error, value in zip(residual, values)
                    )),
                })

            selected = coefficients[
                coefficients["noise_source"].eq(source)
                & np.isclose(coefficients["nbar"], nbar)
            ]
            for (sector, mode), group in selected.groupby(
                ["sector", "mode"], sort=False
            ):
                group = group.sort_values("rate_multiplier")
                active = group[group["rate_multiplier"].gt(0.0)]
                x = active["rate_s^-1"].to_numpy(float)
                y = active["coefficient_per_gate"].to_numpy(float)
                slope, prediction, residual, relative_residual = (
                    _fit_through_origin(x, y)
                )
                max_abs = float(np.max(np.abs(y)))
                centered_sst = float(np.sum((y - np.mean(y)) ** 2))
                centered_r2 = (
                    1.0 - float(np.sum(residual**2)) / centered_sst
                    if centered_sst > 0.0 else np.nan
                )
                coefficient_fit_rows.append({
                    "noise_source": source,
                    "noise_title": structure.SOURCE_TITLES[source],
                    "nbar": nbar,
                    "sector": sector,
                    "mode": mode,
                    "slope_per_rate_s": float(slope),
                    "through_origin_relative_fit_residual": relative_residual,
                    "centered_R2_for_origin_fit": centered_r2,
                    "max_abs_coefficient_per_gate": max_abs,
                    "max_abs_deviation_normalized_to_max_response": (
                        float(np.max(np.abs(residual))) / max_abs
                        if max_abs > 0.0 else np.nan
                    ),
                    "resolved_above_1e-7_per_gate": max_abs >= 1e-7,
                })

            for multiplier in multipliers:
                taxonomy = decompositions[(source, nbar, multiplier)]
                response = generators[(source, nbar, multiplier)]
                stochastic = _stochastic_matrix(taxonomy)
                eigenvalues, eigenvectors = np.linalg.eigh(stochastic)
                leading_index = int(np.argmax(eigenvalues))
                leading_vector = eigenvectors[:, leading_index]
                collective_overlap = float(
                    abs(np.dot(leading_vector, collective_axis))
                )
                trace_positive = float(np.sum(eigenvalues[eigenvalues > 0]))
                collective_strength = float(
                    collective_axis @ stochastic @ collective_axis
                )
                values = {
                    (sector, mode): float(value)
                    for (sector, mode), value in zip(
                        taxonomy["metadata"], taxonomy["coefficients"]
                    )
                }
                local_mean = 0.5 * (
                    values[("S", "IX")] + values[("S", "XI")]
                )
                iz_xy_mean = 0.5 * (
                    values[("S", "IZ")] + values[("S", "XY")]
                )
                iy_xz_mean = 0.5 * (
                    values[("S", "IY")] + values[("S", "XZ")]
                )
                structure_rows.append({
                    "noise_source": source,
                    "noise_title": structure.SOURCE_TITLES[source],
                    "nbar": nbar,
                    "rate_multiplier": multiplier,
                    "rate_s^-1": multiplier * NOMINAL_RATES[source],
                    "response_frobenius_norm": float(np.linalg.norm(response)),
                    "full_generator_cosine_to_nominal": (
                        _cosine(response, nominal_generator)
                        if multiplier > 0.0 else np.nan
                    ),
                    "stochastic_shape_cosine_to_nominal": (
                        _cosine(stochastic, nominal_stochastic)
                        if multiplier > 0.0 else np.nan
                    ),
                    **{
                        f"{sector}_shape_cosine_to_nominal": (
                            _cosine(
                                taxonomy["sector_fits"][sector],
                                nominal_sector_fits[sector],
                            ) if multiplier > 0.0 else np.nan
                        )
                        for sector in ("H", "S", "C", "A")
                    },
                    "leading_stochastic_eigenvalue_per_gate": float(
                        eigenvalues[leading_index]
                    ),
                    "leading_axis_overlap_with_collective_X": (
                        collective_overlap
                    ),
                    "collective_X_rayleigh_coefficient_per_gate": (
                        collective_strength
                    ),
                    "collective_X_fraction_of_positive_stochastic_trace": (
                        collective_strength / trace_positive
                        if trace_positive > 0.0 else np.nan
                    ),
                    "S_IX_per_gate": values[("S", "IX")],
                    "S_XI_per_gate": values[("S", "XI")],
                    "C_IX_XI_per_gate": values[("C", "IX,XI")],
                    "C_IX_XI_over_local_X_mean": (
                        values[("C", "IX,XI")] / local_mean
                        if abs(local_mean) > 0.0 else np.nan
                    ),
                    "C_IZ_XY_over_IZ_XY_S_mean": (
                        values[("C", "IZ,XY")] / iz_xy_mean
                        if abs(iz_xy_mean) > 0.0 else np.nan
                    ),
                    "C_IY_XZ_over_IY_XZ_S_mean": (
                        values[("C", "IY,XZ")] / iy_xz_mean
                        if abs(iy_xz_mean) > 0.0 else np.nan
                    ),
                    "S_IX_over_S_XI": (
                        values[("S", "IX")] / values[("S", "XI")]
                        if abs(values[("S", "XI")]) > 0.0 else np.nan
                    ),
                    "S_XX_per_gate": values[("S", "XX")],
                    "H_XX_rad_per_gate": values[("H", "XX")],
                })

    structure_metrics = pd.DataFrame(structure_rows)
    matrix_linearity = pd.DataFrame(matrix_fit_rows)
    coefficient_linearity = pd.DataFrame(coefficient_fit_rows)
    coefficients.to_csv(
        output_dir / "rate_hsca_coefficients.csv", index=False
    )
    structure_metrics.to_csv(
        output_dir / "rate_structure_metrics.csv", index=False
    )
    matrix_linearity.to_csv(
        output_dir / "rate_matrix_linearity.csv", index=False
    )
    coefficient_linearity.to_csv(
        output_dir / "rate_coefficient_linearity.csv", index=False
    )

    figure, axes = plt.subplots(4, 3, figsize=(15.5, 15.0), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(nbar_values)))
    for row, source in enumerate(structure.RATE_COLUMNS):
        selected = structure_metrics[
            structure_metrics["noise_source"].eq(source)
        ]
        for color, nbar in zip(colors, nbar_values):
            group = selected[np.isclose(selected["nbar"], nbar)].sort_values(
                "rate_multiplier"
            )
            active = group[group["rate_multiplier"].gt(0.0)]
            axes[row, 0].semilogy(
                active["rate_multiplier"],
                np.maximum(
                    1.0 - active["stochastic_shape_cosine_to_nominal"],
                    1e-12,
                ),
                marker="o", color=color, label=f"nbar={nbar:g}",
            )
            axes[row, 1].plot(
                group["rate_multiplier"],
                group["leading_stochastic_eigenvalue_per_gate"],
                marker="o", color=color,
            )
            ratio_column = (
                "C_IX_XI_over_local_X_mean"
                if source in {"motional_heating", "motional_dephasing"}
                else "C_IZ_XY_over_IZ_XY_S_mean"
            )
            axes[row, 2].plot(
                active["rate_multiplier"],
                active[ratio_column],
                marker="o", color=color,
            )
        axes[row, 0].set_ylabel(structure.SOURCE_TITLES[source])
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].grid(alpha=0.25)
        axes[row, 2].grid(alpha=0.25)
        if row == 0:
            axes[row, 0].set_title("S/C shape deviation: 1 - cosine")
            axes[row, 1].set_title("Leading stochastic eigenvalue [/gate]")
            axes[row, 2].set_title("Key C / paired-S mean")
        if row == 3:
            for axis in axes[row]:
                axis.set_xlabel("rate / nominal rate")
    axes[0, 0].legend(fontsize=8, ncol=2)
    figure.suptitle("Rate preservation of isolated-noise H/S/C/A structure")
    figure.tight_layout()
    structure_figure = output_dir / "rate_structure_preservation.png"
    figure.savefig(structure_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)

    linearity_figure, linearity_axes = plt.subplots(
        2, 2, figsize=(11.5, 8.5), sharex=True
    )
    for axis, source in zip(linearity_axes.flat, structure.RATE_COLUMNS):
        selected = matrix_linearity[
            matrix_linearity["noise_source"].eq(source)
            & matrix_linearity["matrix_family"].eq("full_generator")
        ].sort_values("nbar")
        axis.semilogy(
            selected["nbar"],
            selected["through_origin_relative_fit_residual"],
            marker="o", label="full generator",
        )
        selected_sc = matrix_linearity[
            matrix_linearity["noise_source"].eq(source)
            & matrix_linearity["matrix_family"].eq("stochastic_SC_matrix")
        ].sort_values("nbar")
        axis.semilogy(
            selected_sc["nbar"],
            selected_sc["through_origin_relative_fit_residual"],
            marker="s", label="S/C matrix",
        )
        axis.set_title(structure.SOURCE_TITLES[source])
        axis.set_xlabel("nbar")
        axis.set_ylabel("relative residual of rate-linear fit")
        axis.grid(alpha=0.25)
        axis.legend()
    linearity_figure.suptitle(
        "Linearity of Delta K(rate) over 0.5--4 times nominal rate"
    )
    linearity_figure.tight_layout()
    linearity_path = output_dir / "rate_linearity_residuals.png"
    linearity_figure.savefig(linearity_path, dpi=220, bbox_inches="tight")
    plt.close(linearity_figure)

    selected_modes = {
        "motional_heating": [
            ("S", "IX"), ("C", "IX,XI"), ("H", "XX"), ("S", "XX"),
        ],
        "motional_dephasing": [
            ("S", "IX"), ("C", "IX,XI"), ("H", "XX"), ("S", "XX"),
        ],
        "spin_dephasing": [
            ("S", "IZ"), ("C", "IZ,XY"), ("S", "IY"),
            ("C", "IY,XZ"),
        ],
        "photon_scattering": [
            ("S", "IZ"), ("C", "IZ,XY"), ("S", "IY"),
            ("C", "IY,XZ"),
        ],
    }
    coefficient_figure, coefficient_axes = plt.subplots(
        4, 4, figsize=(17.0, 14.5), sharex=True, squeeze=False
    )
    for row, source in enumerate(structure.RATE_COLUMNS):
        for column, (sector, mode) in enumerate(selected_modes[source]):
            axis = coefficient_axes[row, column]
            selected = coefficients[
                coefficients["noise_source"].eq(source)
                & coefficients["sector"].eq(sector)
                & coefficients["mode"].eq(mode)
                & coefficients["rate_multiplier"].gt(0.0)
            ]
            for color, nbar in zip(colors, nbar_values):
                group = selected[np.isclose(selected["nbar"], nbar)].sort_values(
                    "rate_multiplier"
                )
                nominal = float(group[
                    np.isclose(group["rate_multiplier"], 1.0)
                ].iloc[0]["coefficient_per_gate"])
                ratio = group["coefficient_per_gate"] / (
                    group["rate_multiplier"] * nominal
                )
                axis.plot(
                    group["rate_multiplier"], ratio,
                    marker="o", color=color, label=f"nbar={nbar:g}",
                )
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
            axis.set_title(f"{sector}:{mode}")
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel(
                    f"{structure.SOURCE_TITLES[source]}\n"
                    r"$k(m)/(m k(1))$"
                )
            if row == 3:
                axis.set_xlabel("rate / nominal rate")
    coefficient_axes[0, 0].legend(fontsize=8, ncol=2)
    coefficient_figure.suptitle(
        "Selected H/S/C/A coefficient linearity; unity means exact rate linearity"
    )
    coefficient_figure.tight_layout()
    coefficient_linearity_path = (
        output_dir / "rate_selected_coefficient_linearity.png"
    )
    coefficient_figure.savefig(
        coefficient_linearity_path, dpi=220, bbox_inches="tight"
    )
    plt.close(coefficient_figure)

    return {
        "coefficients": coefficients,
        "structure_metrics": structure_metrics,
        "matrix_linearity": matrix_linearity,
        "coefficient_linearity": coefficient_linearity,
        "structure_figure": structure_figure,
        "linearity_figure": linearity_path,
        "coefficient_linearity_figure": coefficient_linearity_path,
    }


if __name__ == "__main__":
    result = analyze_rate_structure()
    print(result["matrix_linearity"].to_string(index=False))
    print(f"Saved {result['structure_figure']}")
    print(f"Saved {result['linearity_figure']}")
    print(f"Saved {result['coefficient_linearity_figure']}")
