"""Data-driven generator response of each isolated MS-gate noise source.

The reference channel contains no explicit dissipators.  For each source at
its nominal rate and each saved thermal occupation, this script forms the full
matrix-log response

    Delta K_i = log(R_i) - log(R_0),

before decomposing that response in two complementary ways:

1. Hamiltonian Pauli generators plus signed diagonal-Pauli dissipators.
2. The complete H/S/C/A trace-preserving generator basis of
   Blume-Kohout et al., ``A Taxonomy of Small Markovian Errors``.

The signed coefficients are deliberate: Delta K_i is a response around a
nontrivial reduced channel and is not itself required to be a completely
positive Lindblad generator.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import scipy.linalg

import drive_calibration_qpt_analysis as qpt_analysis
import error_generator_rate_nbar as rate_analysis
import ms_gate_functions as mg


RATE_COLUMNS = {
    "motional_heating": "motional_heating_s^-1",
    "motional_dephasing": "motional_dephasing_s^-1",
    "spin_dephasing": "spin_dephasing_s^-1",
    "photon_scattering": "photon_scattering_s^-1",
}

SOURCE_TITLES = {
    "motional_heating": "Motional heating/diffusion",
    "motional_dephasing": "Motional dephasing",
    "spin_dephasing": "Spin dephasing",
    "photon_scattering": "Photon scattering",
}


def _load_projected_generator(cache_dir: Path, condition_id: str, nbar: float):
    point = rate_analysis.load_qpt_point(
        rate_analysis.qpt_cache_path(cache_dir, condition_id, nbar)
    )
    superoperator, projected_chi, _ = (
        qpt_analysis.project_trace_normalized_chi_to_cptp(point["chi"])
    )
    ptm = np.real(
        np.asarray(mg.superoperator_to_ptm(superoperator), dtype=complex)
    )
    generator_complex = scipy.linalg.logm(ptm)
    return np.real(generator_complex), projected_chi


def _project_generator_response(delta_generator: np.ndarray):
    labels, hamiltonian_bases, hamiltonian_design, dissipator_design = (
        qpt_analysis._generator_design_data()
    )
    paulis = labels[1:]
    skew = 0.5 * (delta_generator - delta_generator.T)
    h_coefficients = np.linalg.lstsq(
        hamiltonian_design, skew.reshape(-1), rcond=None
    )[0]
    h_fit = sum(
        coefficient * hamiltonian_bases[label]
        for label, coefficient in zip(paulis, h_coefficients)
    )
    remaining = delta_generator - h_fit
    symmetric_remaining = 0.5 * (remaining + remaining.T)
    gamma_coefficients = np.linalg.lstsq(
        dissipator_design, symmetric_remaining.reshape(-1), rcond=None
    )[0]
    dissipator_fit = (dissipator_design @ gamma_coefficients).reshape(
        delta_generator.shape
    )
    residual = delta_generator - h_fit - dissipator_fit
    return {
        "paulis": paulis,
        "h_coefficients": h_coefficients,
        "gamma_coefficients": gamma_coefficients,
        "h_fit": h_fit,
        "dissipator_fit": dissipator_fit,
        "residual": residual,
    }


@lru_cache(maxsize=1)
def _taxonomy_design_data():
    """Return a complete two-qubit H/S/C/A generator design matrix.

    For nonidentity Paulis P and Q, the convention is

        H_P(rho) = -i [P, rho]
        S_P(rho) = P rho P - rho
        C_PQ(rho) = P rho Q + Q rho P
                    - 1/2 {{P Q + Q P, rho}}
        A_PQ(rho) = i (P rho Q - Q rho P
                    + 1/2 {{[P, Q], rho}}).

    The 15 H, 15 S, 105 C and 105 A elements span all 240 real
    trace-preserving, Hermiticity-preserving two-qubit generators.
    """
    labels = tuple(label for label, _ in mg.pauli_labels_and_weights())
    pauli_qobjs = tuple(mg.two_qubit_pauli_basis())
    nonidentity = tuple(zip(labels[1:], pauli_qobjs[1:]))
    columns = []
    metadata = []

    def append(sector, label, action):
        columns.append(
            qpt_analysis._action_to_ptm(action, pauli_qobjs).reshape(-1)
        )
        metadata.append((sector, label))

    for label, pauli in nonidentity:
        append(
            "H", label,
            lambda rho, p=pauli: -1j * (p * rho - rho * p),
        )
    for label, pauli in nonidentity:
        append(
            "S", label,
            lambda rho, p=pauli: p * rho * p - rho,
        )
    for (label_p, pauli_p), (label_q, pauli_q) in combinations(
        nonidentity, 2
    ):
        anticommutator_operator = pauli_p * pauli_q + pauli_q * pauli_p
        append(
            "C", f"{label_p},{label_q}",
            lambda rho, p=pauli_p, q=pauli_q,
            a=anticommutator_operator: (
                p * rho * q + q * rho * p
                - 0.5 * (a * rho + rho * a)
            ),
        )
    for (label_p, pauli_p), (label_q, pauli_q) in combinations(
        nonidentity, 2
    ):
        commutator_operator = pauli_p * pauli_q - pauli_q * pauli_p
        append(
            "A", f"{label_p},{label_q}",
            lambda rho, p=pauli_p, q=pauli_q,
            c=commutator_operator: 1j * (
                p * rho * q - q * rho * p
                + 0.5 * (c * rho + rho * c)
            ),
        )

    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) != 240:
        raise RuntimeError("H/S/C/A generator design is not full rank")
    return tuple(metadata), design, np.linalg.pinv(design)


def _decompose_generator_taxonomy(delta_generator: np.ndarray):
    metadata, design, pseudoinverse = _taxonomy_design_data()
    coefficients = pseudoinverse @ delta_generator.reshape(-1)
    fitted = (design @ coefficients).reshape(delta_generator.shape)
    sector_fits = {}
    for sector in ("H", "S", "C", "A"):
        mask = np.array([item[0] == sector for item in metadata], dtype=bool)
        sector_fits[sector] = (
            design[:, mask] @ coefficients[mask]
        ).reshape(delta_generator.shape)
    return {
        "metadata": metadata,
        "coefficients": coefficients,
        "fitted": fitted,
        "residual": delta_generator - fitted,
        "sector_fits": sector_fits,
    }


def _taxonomy_sector_statistics(taxonomy: dict):
    """Return coefficient and reconstructed-matrix statistics by sector."""
    coefficients = np.asarray(taxonomy["coefficients"], dtype=float)
    metadata = taxonomy["metadata"]
    rows = []
    for sector in ("H", "S", "C", "A"):
        indices = [
            index for index, item in enumerate(metadata) if item[0] == sector
        ]
        values = coefficients[indices]
        dominant_local_index = int(np.argmax(np.abs(values)))
        dominant_index = indices[dominant_local_index]
        rows.append({
            "sector": sector,
            "basis_element_count": len(indices),
            "coefficient_l1_norm": float(np.sum(np.abs(values))),
            "coefficient_l2_norm": float(np.linalg.norm(values)),
            "coefficient_max_abs": float(np.max(np.abs(values))),
            "dominant_mode": metadata[dominant_index][1],
            "dominant_coefficient_per_gate": float(
                coefficients[dominant_index]
            ),
            "sector_matrix_frobenius_norm": float(
                np.linalg.norm(taxonomy["sector_fits"][sector])
            ),
        })
    return rows


def _stochastic_eigenmodes(taxonomy: dict):
    """Diagonalize the real S/C Kossakowski block into principal jump axes."""
    paulis = list(qpt_analysis._generator_design_data()[0][1:])
    index = {label: position for position, label in enumerate(paulis)}
    matrix = np.zeros((len(paulis), len(paulis)), dtype=float)
    for (sector, mode), coefficient in zip(
        taxonomy["metadata"], taxonomy["coefficients"]
    ):
        coefficient = float(coefficient)
        if sector == "S":
            matrix[index[mode], index[mode]] = coefficient
        elif sector == "C":
            left, right = mode.split(",")
            matrix[index[left], index[right]] = coefficient
            matrix[index[right], index[left]] = coefficient

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    rows = []
    for rank, eigen_index in enumerate(order, start=1):
        eigenvalue = float(eigenvalues[eigen_index])
        vector = np.asarray(eigenvectors[:, eigen_index], dtype=float)
        largest_index = int(np.argmax(np.abs(vector)))
        if vector[largest_index] < 0.0:
            vector = -vector
        visible = np.abs(vector) >= 0.1 * np.max(np.abs(vector))
        expression = " ".join(
            f"{value:+.6f}*{label}"
            for label, value, keep in zip(paulis, vector, visible)
            if keep
        )
        row = {
            "eigenmode_rank_by_abs_eigenvalue": rank,
            "eigenvalue_per_gate": eigenvalue,
            "jump_axis_expression": expression,
        }
        row.update({f"axis_{label}": float(value) for label, value in zip(
            paulis, vector
        )})
        rows.append(row)
    return rows, matrix


def analyze_isolated_noise_structure(
    *,
    cache_dir: str | Path = (
        "results/error_generator_rate_nbar/top3_pairwise_generator"
    ),
    output_dir: str | Path = "results/noise_error_structure",
    fock_summary_path: str | Path = (
        "results/noise_rate_nbar_sweep/noise_free_diagnostics/"
        "fock_resolved/fock_thermal_summary.csv"
    ),
    nominal_rates: dict[str, float] | None = None,
):
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if nominal_rates is None:
        nominal_rates = {
            "motional_heating": 10.0,
            "motional_dephasing": 18.0,
            "spin_dephasing": 10.0 / 3.0,
            "photon_scattering": 4.0,
        }

    catalog = pd.read_csv(cache_dir / "condition_catalog.csv")
    summary = pd.read_csv(cache_dir / "pairwise_generator_summary.csv")
    rate_columns = list(RATE_COLUMNS.values())
    zero_rows = catalog[np.isclose(catalog[rate_columns].sum(axis=1), 0.0)]
    if len(zero_rows) != 1:
        raise ValueError("Expected exactly one all-noise-zero condition")
    zero_id = str(zero_rows.iloc[0]["condition_id"])
    nbar_values = np.sort(summary["nbar"].unique().astype(float))

    zero_generators = {}
    zero_chi = {}
    baseline_taxonomy_rows = []
    baseline_taxonomy_summary_rows = []
    sector_statistics_rows = []
    stochastic_eigenmode_rows = []
    for nbar in nbar_values:
        zero_generators[nbar], zero_chi[nbar] = _load_projected_generator(
            cache_dir, zero_id, nbar
        )
        baseline_taxonomy = _decompose_generator_taxonomy(
            zero_generators[nbar]
        )
        for (sector, label), coefficient in zip(
            baseline_taxonomy["metadata"],
            baseline_taxonomy["coefficients"],
        ):
            baseline_taxonomy_rows.append({
                "nbar": nbar,
                "sector": sector,
                "mode": label,
                "coefficient_per_gate": float(coefficient),
            })
        baseline_norm = float(np.linalg.norm(zero_generators[nbar]))
        baseline_coefficients = np.asarray(
            baseline_taxonomy["coefficients"], dtype=float
        )
        baseline_dominant_index = int(
            np.argmax(np.abs(baseline_coefficients))
        )
        baseline_stochastic_modes, baseline_stochastic_matrix = (
            _stochastic_eigenmodes(baseline_taxonomy)
        )
        baseline_eigenvalues = np.linalg.eigvalsh(
            baseline_stochastic_matrix
        )
        baseline_ii = float(np.real(zero_chi[nbar][0, 0]))
        baseline_taxonomy_summary_rows.append({
            "nbar": nbar,
            "average_infidelity": 4.0 / 5.0 * (1.0 - baseline_ii),
            "generator_frobenius_norm": baseline_norm,
            **{
                f"{sector}_sector_frobenius_norm": float(
                    np.linalg.norm(
                        baseline_taxonomy["sector_fits"][sector]
                    )
                )
                for sector in ("H", "S", "C", "A")
            },
            "taxonomy_reconstruction_residual_norm": float(
                np.linalg.norm(baseline_taxonomy["residual"])
            ),
            "taxonomy_reconstruction_residual_fraction": float(
                np.linalg.norm(baseline_taxonomy["residual"])
                / baseline_norm
            ),
            "dominant_taxonomy_sector": baseline_taxonomy["metadata"][
                baseline_dominant_index
            ][0],
            "dominant_taxonomy_mode": baseline_taxonomy["metadata"][
                baseline_dominant_index
            ][1],
            "dominant_taxonomy_coefficient_per_gate": float(
                baseline_coefficients[baseline_dominant_index]
            ),
            "stochastic_matrix_min_eigenvalue_per_gate": float(
                np.min(baseline_eigenvalues)
            ),
            "stochastic_matrix_max_eigenvalue_per_gate": float(
                np.max(baseline_eigenvalues)
            ),
            "stochastic_matrix_negative_eigenvalue_abs_sum_per_gate": float(
                np.sum(np.abs(baseline_eigenvalues[baseline_eigenvalues < 0]))
            ),
        })
        for row in _taxonomy_sector_statistics(baseline_taxonomy):
            sector_statistics_rows.append({
                "context": "noise_free_channel",
                "noise_source": "none",
                "noise_title": "All explicit noise zero",
                "nominal_rate_s^-1": 0.0,
                "nbar": nbar,
                **row,
            })
        for row in baseline_stochastic_modes:
            stochastic_eigenmode_rows.append({
                "context": "noise_free_channel",
                "noise_source": "none",
                "noise_title": "All explicit noise zero",
                "nominal_rate_s^-1": 0.0,
                "nbar": nbar,
                **row,
            })

    baseline_taxonomy_coefficients = pd.DataFrame(
        baseline_taxonomy_rows
    ).sort_values(["nbar", "sector", "mode"]).reset_index(drop=True)
    baseline_taxonomy_coefficients.to_csv(
        output_dir / "noise_free_taxonomy_coefficients.csv", index=False
    )
    baseline_taxonomy_summary = pd.DataFrame(
        baseline_taxonomy_summary_rows
    ).sort_values("nbar").reset_index(drop=True)
    baseline_taxonomy_summary.to_csv(
        output_dir / "noise_free_taxonomy_summary.csv", index=False
    )

    fock_comparison = pd.DataFrame()
    fock_summary_path = Path(fock_summary_path)
    if fock_summary_path.exists():
        fock_summary = pd.read_csv(fock_summary_path)
        comparison_rows = []
        for nbar in nbar_values:
            fock_candidates = fock_summary[
                np.isclose(fock_summary["thermal_n_bar"], nbar)
            ]
            if len(fock_candidates) != 1:
                continue
            fock_row = fock_candidates.iloc[0]
            baseline = baseline_taxonomy_coefficients[
                np.isclose(baseline_taxonomy_coefficients["nbar"], nbar)
            ]

            def coefficient(sector, mode):
                selected = baseline[
                    baseline["sector"].eq(sector)
                    & baseline["mode"].eq(mode)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"Missing baseline taxonomy coefficient {sector}:{mode}"
                    )
                return float(selected.iloc[0]["coefficient_per_gate"])

            qpt_h_xx = coefficient("H", "XX")
            qpt_s_xx = coefficient("S", "XX")
            qpt_s_ix = coefficient("S", "IX")
            qpt_s_xi = coefficient("S", "XI")
            qpt_c_ix_xi = coefficient("C", "IX,XI")
            qpt_local_x_mean = 0.5 * (qpt_s_ix + qpt_s_xi)
            fock_h_xx = float(fock_row["predicted_h_XX_rad_per_gate"])
            fock_phase_gamma = float(
                fock_row["gamma_XX_phase_dispersion_per_gate"]
            )
            fock_residual_gamma = float(
                fock_row["residual_motion_gamma_per_gate"]
            )

            def relative_error(prediction, reference):
                return abs(prediction - reference) / abs(reference)

            comparison_rows.append({
                "nbar": nbar,
                "fock_h_XX_rad_per_gate": fock_h_xx,
                "qpt_H_XX_rad_per_gate": qpt_h_xx,
                "h_XX_relative_error": relative_error(
                    fock_h_xx, qpt_h_xx
                ),
                "fock_phase_dispersion_gamma_per_gate": fock_phase_gamma,
                "qpt_S_XX_per_gate": qpt_s_xx,
                "gamma_XX_relative_error": relative_error(
                    fock_phase_gamma, qpt_s_xx
                ),
                "fock_residual_motion_gamma_per_gate": (
                    fock_residual_gamma
                ),
                "qpt_S_IX_per_gate": qpt_s_ix,
                "qpt_S_XI_per_gate": qpt_s_xi,
                "qpt_local_X_mean_per_gate": qpt_local_x_mean,
                "qpt_C_IX_XI_per_gate": qpt_c_ix_xi,
                "residual_vs_local_X_mean_relative_error": relative_error(
                    fock_residual_gamma, qpt_local_x_mean
                ),
                "residual_vs_C_IX_XI_relative_error": relative_error(
                    fock_residual_gamma, qpt_c_ix_xi
                ),
                "collective_X_coefficient_spread": max(
                    qpt_s_ix, qpt_s_xi, qpt_c_ix_xi
                ) - min(qpt_s_ix, qpt_s_xi, qpt_c_ix_xi),
            })
        fock_comparison = pd.DataFrame(comparison_rows)
        fock_comparison.to_csv(
            output_dir / "noise_free_fock_generator_comparison.csv",
            index=False,
        )

    coefficient_rows = []
    taxonomy_coefficient_rows = []
    taxonomy_summary_rows = []
    summary_rows = []
    for source, rate_column in RATE_COLUMNS.items():
        mask = np.isclose(catalog[rate_column], nominal_rates[source])
        for other_source, other_column in RATE_COLUMNS.items():
            if other_source != source:
                mask &= np.isclose(catalog[other_column], 0.0)
        candidates = catalog[mask]
        if len(candidates) != 1:
            raise ValueError(f"Could not identify isolated condition for {source}")
        condition_id = str(candidates.iloc[0]["condition_id"])

        for nbar in nbar_values:
            source_generator, source_chi = _load_projected_generator(
                cache_dir, condition_id, nbar
            )
            delta_generator = source_generator - zero_generators[nbar]
            projection = _project_generator_response(delta_generator)
            taxonomy = _decompose_generator_taxonomy(delta_generator)
            h_coefficients = projection["h_coefficients"]
            gamma_coefficients = projection["gamma_coefficients"]
            paulis = projection["paulis"]
            weights = dict(mg.pauli_labels_and_weights())

            for pauli, h_value, gamma_value in zip(
                paulis, h_coefficients, gamma_coefficients
            ):
                coefficient_rows.append({
                    "noise_source": source,
                    "noise_title": SOURCE_TITLES[source],
                    "nominal_rate_s^-1": nominal_rates[source],
                    "nbar": nbar,
                    "pauli": pauli,
                    "pauli_weight": weights[pauli],
                    "mode_class": "local" if weights[pauli] == 1 else "weight2",
                    "delta_h_rad_per_gate": float(h_value),
                    "delta_gamma_per_gate": float(gamma_value),
                    "dh_drate_s": float(h_value / nominal_rates[source]),
                    "dgamma_drate_s": float(gamma_value / nominal_rates[source]),
                })

            local_mask = np.array(
                [weights[pauli] == 1 for pauli in paulis], dtype=bool
            )
            dominant_h_index = int(np.argmax(np.abs(h_coefficients)))
            dominant_gamma_index = int(np.argmax(np.abs(gamma_coefficients)))
            source_ii = float(np.real(source_chi[0, 0]))
            zero_ii = float(np.real(zero_chi[nbar][0, 0]))
            source_infidelity = 4.0 / 5.0 * (1.0 - source_ii)
            zero_infidelity = 4.0 / 5.0 * (1.0 - zero_ii)
            response_norm = float(np.linalg.norm(delta_generator))
            residual_norm = float(np.linalg.norm(projection["residual"]))
            summary_rows.append({
                "noise_source": source,
                "noise_title": SOURCE_TITLES[source],
                "nominal_rate_s^-1": nominal_rates[source],
                "nbar": nbar,
                "infidelity_increment": source_infidelity - zero_infidelity,
                "delta_generator_frobenius_norm": response_norm,
                "hamiltonian_response_frobenius_norm": float(
                    np.linalg.norm(projection["h_fit"])
                ),
                "diagonal_pauli_response_frobenius_norm": float(
                    np.linalg.norm(projection["dissipator_fit"])
                ),
                "unmodeled_response_frobenius_norm": residual_norm,
                "unmodeled_response_fraction": (
                    residual_norm / response_norm if response_norm > 0.0 else np.nan
                ),
                "signed_local_gamma_sum_per_gate": float(
                    np.sum(gamma_coefficients[local_mask])
                ),
                "signed_weight2_gamma_sum_per_gate": float(
                    np.sum(gamma_coefficients[~local_mask])
                ),
                "dominant_h_pauli": paulis[dominant_h_index],
                "dominant_h_rad_per_gate": float(
                    h_coefficients[dominant_h_index]
                ),
                "dominant_gamma_pauli": paulis[dominant_gamma_index],
                "dominant_gamma_per_gate": float(
                    gamma_coefficients[dominant_gamma_index]
                ),
            })

            for (sector, label), coefficient in zip(
                taxonomy["metadata"], taxonomy["coefficients"]
            ):
                taxonomy_coefficient_rows.append({
                    "noise_source": source,
                    "noise_title": SOURCE_TITLES[source],
                    "nominal_rate_s^-1": nominal_rates[source],
                    "nbar": nbar,
                    "sector": sector,
                    "mode": label,
                    "coefficient_per_gate": float(coefficient),
                    "coefficient_per_rate_s": float(
                        coefficient / nominal_rates[source]
                    ),
                })

            taxonomy_coefficients = np.asarray(
                taxonomy["coefficients"], dtype=float
            )
            taxonomy_metadata = taxonomy["metadata"]
            stochastic_modes, stochastic_matrix = _stochastic_eigenmodes(
                taxonomy
            )
            stochastic_eigenvalues = np.linalg.eigvalsh(stochastic_matrix)
            dominant_index = int(np.argmax(np.abs(taxonomy_coefficients)))
            taxonomy_summary_rows.append({
                "noise_source": source,
                "noise_title": SOURCE_TITLES[source],
                "nominal_rate_s^-1": nominal_rates[source],
                "nbar": nbar,
                **{
                    f"{sector}_sector_frobenius_norm": float(
                        np.linalg.norm(taxonomy["sector_fits"][sector])
                    )
                    for sector in ("H", "S", "C", "A")
                },
                "taxonomy_reconstruction_residual_norm": float(
                    np.linalg.norm(taxonomy["residual"])
                ),
                "taxonomy_reconstruction_residual_fraction": float(
                    np.linalg.norm(taxonomy["residual"]) / response_norm
                    if response_norm > 0.0 else np.nan
                ),
                "dominant_taxonomy_sector": taxonomy_metadata[
                    dominant_index
                ][0],
                "dominant_taxonomy_mode": taxonomy_metadata[
                    dominant_index
                ][1],
                "dominant_taxonomy_coefficient_per_gate": float(
                    taxonomy_coefficients[dominant_index]
                ),
                "stochastic_matrix_min_eigenvalue_per_gate": float(
                    np.min(stochastic_eigenvalues)
                ),
                "stochastic_matrix_max_eigenvalue_per_gate": float(
                    np.max(stochastic_eigenvalues)
                ),
                "stochastic_matrix_negative_eigenvalue_abs_sum_per_gate": (
                    float(np.sum(np.abs(
                        stochastic_eigenvalues[stochastic_eigenvalues < 0]
                    )))
                ),
            })
            for row in _taxonomy_sector_statistics(taxonomy):
                sector_statistics_rows.append({
                    "context": "isolated_noise_response",
                    "noise_source": source,
                    "noise_title": SOURCE_TITLES[source],
                    "nominal_rate_s^-1": nominal_rates[source],
                    "nbar": nbar,
                    **row,
                })
            for row in stochastic_modes:
                stochastic_eigenmode_rows.append({
                    "context": "isolated_noise_response",
                    "noise_source": source,
                    "noise_title": SOURCE_TITLES[source],
                    "nominal_rate_s^-1": nominal_rates[source],
                    "nbar": nbar,
                    **row,
                })

    coefficients = pd.DataFrame(coefficient_rows).sort_values(
        ["noise_source", "nbar", "pauli"]
    ).reset_index(drop=True)
    response_summary = pd.DataFrame(summary_rows).sort_values(
        ["noise_source", "nbar"]
    ).reset_index(drop=True)
    coefficients.to_csv(
        output_dir / "isolated_noise_generator_coefficients.csv", index=False
    )
    response_summary.to_csv(
        output_dir / "isolated_noise_generator_summary.csv", index=False
    )
    taxonomy_coefficients = pd.DataFrame(taxonomy_coefficient_rows).sort_values(
        ["noise_source", "nbar", "sector", "mode"]
    ).reset_index(drop=True)
    taxonomy_summary = pd.DataFrame(taxonomy_summary_rows).sort_values(
        ["noise_source", "nbar"]
    ).reset_index(drop=True)
    taxonomy_coefficients.to_csv(
        output_dir / "isolated_noise_taxonomy_coefficients.csv", index=False
    )
    taxonomy_summary.to_csv(
        output_dir / "isolated_noise_taxonomy_summary.csv", index=False
    )
    full_taxonomy_coefficients = pd.concat([
        baseline_taxonomy_coefficients.assign(
            context="noise_free_channel",
            noise_source="none",
            noise_title="All explicit noise zero",
            **{"nominal_rate_s^-1": 0.0},
        )[[
            "context", "noise_source", "noise_title", "nominal_rate_s^-1",
            "nbar", "sector", "mode", "coefficient_per_gate",
        ]],
        taxonomy_coefficients.assign(
            context="isolated_noise_response"
        )[[
            "context", "noise_source", "noise_title", "nominal_rate_s^-1",
            "nbar", "sector", "mode", "coefficient_per_gate",
        ]],
    ], ignore_index=True).sort_values(
        ["context", "noise_source", "nbar", "sector", "mode"]
    ).reset_index(drop=True)
    full_taxonomy_coefficients.to_csv(
        output_dir / "full_taxonomy_coefficients.csv", index=False
    )

    overall_rows = []
    for row in baseline_taxonomy_summary.to_dict("records"):
        overall_rows.append({
            "context": "noise_free_channel",
            "noise_source": "none",
            "noise_title": "All explicit noise zero",
            "nominal_rate_s^-1": 0.0,
            "nbar": row["nbar"],
            "infidelity_value": row["average_infidelity"],
            "infidelity_value_meaning": "average_infidelity",
            "generator_frobenius_norm": row["generator_frobenius_norm"],
            **{
                key: value for key, value in row.items()
                if key not in {
                    "nbar", "average_infidelity",
                    "generator_frobenius_norm",
                }
            },
        })
    response_lookup = response_summary.set_index(
        ["noise_source", "nbar"]
    )
    for row in taxonomy_summary.to_dict("records"):
        response_row = response_lookup.loc[(
            row["noise_source"], row["nbar"]
        )]
        overall_rows.append({
            "context": "isolated_noise_response",
            "noise_source": row["noise_source"],
            "noise_title": row["noise_title"],
            "nominal_rate_s^-1": row["nominal_rate_s^-1"],
            "nbar": row["nbar"],
            "infidelity_value": response_row["infidelity_increment"],
            "infidelity_value_meaning": "infidelity_increment_vs_noise_free",
            "generator_frobenius_norm": response_row[
                "delta_generator_frobenius_norm"
            ],
            **{
                key: value for key, value in row.items()
                if key not in {
                    "noise_source", "noise_title", "nominal_rate_s^-1",
                    "nbar",
                }
            },
        })
    full_taxonomy_overall_summary = pd.DataFrame(overall_rows).sort_values(
        ["context", "noise_source", "nbar"]
    ).reset_index(drop=True)
    full_taxonomy_overall_summary.to_csv(
        output_dir / "full_taxonomy_overall_summary.csv", index=False
    )
    sector_statistics = pd.DataFrame(sector_statistics_rows).sort_values(
        ["context", "noise_source", "nbar", "sector"]
    ).reset_index(drop=True)
    sector_statistics.to_csv(
        output_dir / "full_taxonomy_sector_statistics.csv", index=False
    )
    stochastic_eigenmodes = pd.DataFrame(
        stochastic_eigenmode_rows
    ).sort_values([
        "context", "noise_source", "nbar",
        "eigenmode_rank_by_abs_eigenvalue",
    ]).reset_index(drop=True)
    stochastic_eigenmodes.to_csv(
        output_dir / "full_taxonomy_stochastic_eigenmodes.csv", index=False
    )

    figure, axes = plt.subplots(4, 2, figsize=(17.0, 14.5), squeeze=False)
    pauli_order = list(qpt_analysis._generator_design_data()[0][1:])
    for row_index, source in enumerate(RATE_COLUMNS):
        selected = coefficients[coefficients["noise_source"].eq(source)]
        for column_index, (value_column, title, unit) in enumerate([
            ("delta_gamma_per_gate", "signed diagonal-Pauli response", "/gate"),
            ("delta_h_rad_per_gate", "Hamiltonian response", "rad/gate"),
        ]):
            matrix = selected.pivot(
                index="nbar", columns="pauli", values=value_column
            ).reindex(index=nbar_values, columns=pauli_order)
            values = matrix.to_numpy(float)
            scale = float(np.max(np.abs(values)))
            image = axes[row_index, column_index].imshow(
                values,
                aspect="auto",
                origin="lower",
                cmap="coolwarm",
                norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale),
            )
            axes[row_index, column_index].set_xticks(range(len(pauli_order)))
            axes[row_index, column_index].set_xticklabels(
                pauli_order, rotation=45, ha="right"
            )
            axes[row_index, column_index].set_yticks(range(len(nbar_values)))
            axes[row_index, column_index].set_yticklabels(
                [f"{value:g}" for value in nbar_values]
            )
            axes[row_index, column_index].set_ylabel(r"$\bar n$")
            axes[row_index, column_index].set_title(
                f"{SOURCE_TITLES[source]}: {title} [{unit}]"
            )
            figure.colorbar(image, ax=axes[row_index, column_index], shrink=0.82)
    figure.suptitle(
        r"Isolated nominal-noise response $\Delta K_i=K_i-K_0$",
        y=1.002,
    )
    figure.tight_layout()
    figure_path = output_dir / "isolated_noise_generator_heatmaps.png"
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    sector_figure, sector_axes = plt.subplots(
        2, 2, figsize=(11.5, 8.0), sharex=True
    )
    for axis, source in zip(sector_axes.flat, RATE_COLUMNS):
        selected = taxonomy_summary[
            taxonomy_summary["noise_source"].eq(source)
        ]
        for sector in ("H", "S", "C", "A"):
            axis.plot(
                selected["nbar"],
                selected[f"{sector}_sector_frobenius_norm"],
                marker="o",
                label=sector,
            )
        axis.set_title(SOURCE_TITLES[source])
        axis.set_xlabel(r"$\bar n$")
        axis.set_ylabel(r"sector matrix norm $\|\Delta K\|_F$")
        axis.grid(alpha=0.25)
        axis.legend()
    sector_figure.suptitle(
        r"Complete H/S/C/A decomposition of $\Delta K_i=K_i-K_0$"
    )
    sector_figure.tight_layout()
    sector_figure_path = output_dir / "isolated_noise_taxonomy_sector_norms.png"
    sector_figure.savefig(sector_figure_path, dpi=220, bbox_inches="tight")
    plt.close(sector_figure)
    return {
        "coefficients": coefficients,
        "summary": response_summary,
        "figure_path": figure_path,
        "taxonomy_coefficients": taxonomy_coefficients,
        "taxonomy_summary": taxonomy_summary,
        "taxonomy_figure_path": sector_figure_path,
        "baseline_taxonomy_coefficients": baseline_taxonomy_coefficients,
        "baseline_taxonomy_summary": baseline_taxonomy_summary,
        "fock_comparison": fock_comparison,
        "sector_statistics": sector_statistics,
        "stochastic_eigenmodes": stochastic_eigenmodes,
        "full_taxonomy_coefficients": full_taxonomy_coefficients,
        "full_taxonomy_overall_summary": full_taxonomy_overall_summary,
    }


if __name__ == "__main__":
    result = analyze_isolated_noise_structure()
    print(result["summary"].to_string(index=False))
    print(f"Saved {result['figure_path']}")
    print(result["taxonomy_summary"].to_string(index=False))
    print(f"Saved {result['taxonomy_figure_path']}")
