"""Self-contained QPT analysis used by the hXX drive-calibration notebook cells.

The notebook previously reused functions defined many cells earlier.  Keeping
the physical re-QPT and its CPTP/error-generator post-processing here makes the
calibration section safe to run after a kernel restart.
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import qutip as qp
import scipy.linalg
import scipy.optimize

import ms_gate_functions as mg


TWO_QUBIT_SUPER_DIMS = [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]


def _chi_raw_to_superoperator(chi_raw):
    chi_qobj = qp.Qobj(
        np.asarray(chi_raw, dtype=complex),
        dims=TWO_QUBIT_SUPER_DIMS,
        superrep="chi",
    )
    return qp.to_super(chi_qobj)


def _choi_partial_trace_over_output(choi_matrix, dimension=4):
    tensor = np.asarray(choi_matrix, dtype=complex).reshape(
        dimension, dimension, dimension, dimension
    )
    return np.einsum("iaja->ij", tensor)


def _project_to_tp_affine(choi_matrix, dimension=4):
    hermitian = 0.5 * (choi_matrix + choi_matrix.conj().T)
    residual = (
        _choi_partial_trace_over_output(hermitian, dimension)
        - np.eye(dimension)
    )
    return hermitian - np.kron(residual, np.eye(dimension) / dimension)


def _project_to_psd_cone(matrix):
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    return (
        eigenvectors * np.maximum(eigenvalues, 0.0)
    ) @ eigenvectors.conj().T


def nearest_cptp_choi_dykstra(
    choi_matrix,
    dimension=4,
    tolerance=1e-11,
    max_iterations=5000,
):
    """Project a Choi matrix onto the intersection of CP and TP constraints."""

    current = 0.5 * (choi_matrix + choi_matrix.conj().T)
    psd_correction = np.zeros_like(current)
    tp_correction = np.zeros_like(current)

    for iteration in range(1, int(max_iterations) + 1):
        psd_input = current + psd_correction
        psd_projected = _project_to_psd_cone(psd_input)
        psd_correction = psd_input - psd_projected

        tp_input = psd_projected + tp_correction
        updated = _project_to_tp_affine(tp_input, dimension)
        tp_correction = tp_input - updated

        updated_hermitian = 0.5 * (updated + updated.conj().T)
        min_eigenvalue = float(np.linalg.eigvalsh(updated_hermitian).min())
        tp_error = float(
            np.linalg.norm(
                _choi_partial_trace_over_output(updated_hermitian, dimension)
                - np.eye(dimension)
            )
        )
        relative_step = float(
            np.linalg.norm(updated - current) / max(np.linalg.norm(current), 1.0)
        )
        current = updated
        if (
            min_eigenvalue >= -tolerance
            and tp_error <= tolerance
            and relative_step <= tolerance
        ):
            break
    else:
        warnings.warn("CPTP Dykstra projection reached the iteration limit")

    return current, {
        "iterations": iteration,
        "min_choi_eigenvalue": min_eigenvalue,
        "tp_frobenius_error": tp_error,
        "relative_final_step": relative_step,
    }


def project_trace_normalized_chi_to_cptp(
    chi,
    tolerance=1e-11,
    max_iterations=5000,
):
    """CPTP-project a trace-normalized two-qubit Pauli-basis chi matrix."""

    # QuTiP's two-qubit chi convention has trace d**2=16.
    source_super = _chi_raw_to_superoperator(np.asarray(chi, dtype=complex) * 16.0)
    source_choi = qp.to_choi(source_super).full()
    projected_choi, status = nearest_cptp_choi_dykstra(
        source_choi,
        dimension=4,
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
    )
    projected_choi_qobj = qp.Qobj(
        projected_choi,
        dims=TWO_QUBIT_SUPER_DIMS,
        superrep="choi",
    )
    projected_super = qp.to_super(projected_choi_qobj)
    projected_chi_raw = qp.to_chi(projected_super).full()
    projected_chi = projected_chi_raw / np.trace(projected_chi_raw)
    return projected_super, projected_chi, status


def _action_to_ptm(action, pauli_qobjs):
    dimension = pauli_qobjs[0].shape[0]
    matrix = np.zeros((len(pauli_qobjs), len(pauli_qobjs)), dtype=complex)
    for column, input_pauli in enumerate(pauli_qobjs):
        output = action(input_pauli)
        for row, measurement_pauli in enumerate(pauli_qobjs):
            matrix[row, column] = (
                measurement_pauli.dag() * output
            ).tr() / dimension
    return np.real_if_close(matrix).astype(float)


@lru_cache(maxsize=1)
def _generator_design_data():
    labels = tuple(label for label, _ in mg.pauli_labels_and_weights())
    pauli_qobjs = tuple(mg.two_qubit_pauli_basis())
    hamiltonian_bases = {
        label: _action_to_ptm(
            lambda rho, p=pauli: -1j * (p * rho - rho * p),
            pauli_qobjs,
        )
        for label, pauli in zip(labels[1:], pauli_qobjs[1:])
    }
    dissipator_bases = {
        label: _action_to_ptm(
            lambda rho, p=pauli: p * rho * p - rho,
            pauli_qobjs,
        )
        for label, pauli in zip(labels[1:], pauli_qobjs[1:])
    }
    hamiltonian_design = np.column_stack(
        [hamiltonian_bases[label].reshape(-1) for label in labels[1:]]
    )
    dissipator_design = np.column_stack(
        [dissipator_bases[label].reshape(-1) for label in labels[1:]]
    )
    return (
        labels,
        hamiltonian_bases,
        hamiltonian_design,
        dissipator_design,
    )


def extract_xx_generator_observables(
    chi,
    cptp_tolerance=1e-11,
    cptp_max_iterations=5000,
):
    """Return CPTP-projected hXX, gammaXX and infidelity for one QPT chi."""

    projected_super, projected_chi, projection_status = (
        project_trace_normalized_chi_to_cptp(
            chi,
            tolerance=cptp_tolerance,
            max_iterations=cptp_max_iterations,
        )
    )
    labels, hamiltonian_bases, hamiltonian_design, dissipator_design = (
        _generator_design_data()
    )
    ptm = np.asarray(mg.superoperator_to_ptm(projected_super), dtype=complex)
    generator_complex = scipy.linalg.logm(ptm)
    generator = np.real(generator_complex)
    skew_generator = 0.5 * (generator - generator.T)
    h_coefficients = np.linalg.lstsq(
        hamiltonian_design,
        skew_generator.reshape(-1),
        rcond=None,
    )[0]
    h_fit = sum(
        coefficient * hamiltonian_bases[label]
        for label, coefficient in zip(labels[1:], h_coefficients)
    )
    remaining = generator - h_fit
    symmetric_remaining = 0.5 * (remaining + remaining.T)
    gamma_coefficients, gamma_residual = scipy.optimize.nnls(
        dissipator_design,
        symmetric_remaining.reshape(-1),
    )

    ii_index = labels.index("II")
    xx_index = labels.index("XX")
    h_xx = float(h_coefficients[labels[1:].index("XX")])
    gamma_xx = float(gamma_coefficients[labels[1:].index("XX")])
    average_infidelity = 4.0 / 5.0 * (
        1.0 - float(np.real(projected_chi[ii_index, ii_index]))
    )
    return {
        "h_XX_rad_per_gate": h_xx,
        "gamma_XX_per_gate": gamma_xx,
        "average_infidelity": average_infidelity,
        "abs_chi_II_XX": float(abs(projected_chi[ii_index, xx_index])),
        "chi_XX_XX": float(np.real(projected_chi[xx_index, xx_index])),
        "generator_imaginary_frobenius_norm": float(
            np.linalg.norm(np.imag(generator_complex))
        ),
        "gamma_nnls_residual": float(gamma_residual),
        "cptp_projection_iterations": int(projection_status["iterations"]),
        "projected_chi": projected_chi,
    }


def calculate_error_channel_batch(
    n_bar_values,
    simulation_params,
    parameter_overrides=None,
    convention="undo_before_actual",
):
    """Run the project's full-order master-equation QPT for selected nbar."""

    params = dict(simulation_params)
    if parameter_overrides:
        params.update(parameter_overrides)
    params["n_bar_list"] = [float(value) for value in n_bar_values]
    error_result = mg.generate_error_channel_matrices(
        convention=convention,
        **params,
    )
    composition = mg.validate_error_channel_composition(
        error_result,
        desired_convention=convention,
    )
    batch_results = []
    for index, n_bar in enumerate(params["n_bar_list"]):
        chi_raw = np.asarray(
            error_result["error_chi_matrix_list"][index],
            dtype=complex,
        )
        raw_trace = np.trace(chi_raw)
        chi = chi_raw / raw_trace
        physicality = mg.choi_physicality_metrics(
            error_result["S_error_qobj_list"][index]
        )
        metadata = {
            "n_bar": float(n_bar),
            "phonon_dim": int(error_result["results_list"][index]["Nv"]),
            "raw_trace_real": float(np.real(raw_trace)),
            "raw_trace_imag": float(np.imag(raw_trace)),
            "trace_normalized_hermiticity_fro": float(
                np.linalg.norm(chi - chi.conj().T)
            ),
            "convention_error_fro": float(
                composition["max_desired_convention_error"]
            ),
            **physicality,
        }
        batch_results.append(
            {"n_bar": float(n_bar), "chi": chi, "metadata": metadata}
        )
    return batch_results


def save_qpt_point(path, n_bar, condition_name, chi, metadata):
    """Atomically save one trace-normalized QPT result."""

    path = Path(path)
    temporary_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        n_bar=float(n_bar),
        condition=condition_name,
        chi_trace_normalized=np.asarray(chi, dtype=complex),
        metadata_json=json.dumps(metadata, default=str),
    )
    temporary_path.replace(path)
