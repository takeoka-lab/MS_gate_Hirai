
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np


PAULI_LABELS_2Q = (
    "II",
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
)


@dataclass(frozen=True)
class PauliNoiseModel:
    """Container for a two-qubit Pauli noise model."""

    n_bar: float | None
    labels: tuple[str, ...]
    probabilities: tuple[float, ...]

    @property
    def probability_by_label(self) -> dict[str, float]:
        return dict(zip(self.labels, self.probabilities))

    @property
    def stim_pauli_channel_2_probabilities(self) -> tuple[float, ...]:
        """Return Stim PAULI_CHANNEL_2 probabilities, excluding ``II``."""
        return self.probabilities[1:]

    def as_dict(self) -> dict[str, object]:
        return {
            "n_bar": self.n_bar,
            "pauli_labels": list(self.labels),
            "probabilities": list(self.probabilities),
            "probability_by_label": self.probability_by_label,
            "stim_pauli_channel_2_probabilities": list(
                self.stim_pauli_channel_2_probabilities
            ),
        }


def _as_array(value, dtype=complex):
    if hasattr(value, "full"):
        value = value.full()
    return np.asarray(value, dtype=dtype)


def _normalize_probabilities(probabilities, atol=1e-12):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (16,):
        raise ValueError(f"probabilities must have shape (16,); got {probabilities.shape}.")

    if np.any(probabilities < -atol):
        min_probability = float(np.min(probabilities))
        raise ValueError(
            "Pauli probabilities contain a negative value larger than numerical "
            f"tolerance: {min_probability}."
        )

    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if total <= atol:
        raise ValueError("Pauli probabilities sum to zero.")

    return probabilities / total


def pauli_probabilities_from_chi(chi, trace_normalize=True, atol=1e-12):
    """Return 16 Pauli-twirled probabilities from a two-qubit chi matrix.

    The probabilities are ``real(diag(chi))`` after optional trace normalization.
    The label order is ``II, IX, IY, IZ, XI, ..., ZZ``.
    """
    chi = _as_array(chi, dtype=complex)
    if chi.shape != (16, 16):
        raise ValueError(f"chi must have shape (16, 16); got {chi.shape}.")

    if trace_normalize:
        trace = np.trace(chi)
        if abs(trace) <= atol:
            raise ValueError("chi trace is too close to zero for trace normalization.")
        chi = chi / trace

    diagonal = np.real(np.diag(chi))
    return _normalize_probabilities(diagonal, atol=atol)


def build_pauli_noise_model(n_bar, probabilities, labels=PAULI_LABELS_2Q, atol=1e-12):
    """Build a serializable two-qubit Pauli noise model."""
    if tuple(labels) != PAULI_LABELS_2Q:
        labels = tuple(labels)
        if len(labels) != 16:
            raise ValueError(f"labels must contain 16 entries; got {len(labels)}.")
    else:
        labels = PAULI_LABELS_2Q

    probabilities = tuple(float(p) for p in _normalize_probabilities(probabilities, atol=atol))
    return PauliNoiseModel(
        n_bar=None if n_bar is None else float(n_bar),
        labels=tuple(labels),
        probabilities=probabilities,
    )


def pauli_noise_from_chi(chi, n_bar=None, trace_normalize=True, atol=1e-12):
    """Convert a two-qubit chi matrix into a PauliNoiseModel."""
    probabilities = pauli_probabilities_from_chi(
        chi,
        trace_normalize=trace_normalize,
        atol=atol,
    )
    return build_pauli_noise_model(n_bar=n_bar, probabilities=probabilities, atol=atol)


def _lookup_mapping_by_float_key(mapping, target, atol=1e-12):
    for key, value in mapping.items():
        if np.isclose(float(key), float(target), atol=atol, rtol=0.0):
            return value
    raise KeyError(f"n_bar={target} was not found.")


def _ordered_chi_points(error_result):
    if not isinstance(error_result, Mapping):
        raise TypeError("error_result must be a mapping.")

    preferred_nbars = error_result.get("parameters", {}).get("n_bar_list")

    if "error_chi_by_n_bar" in error_result:
        chi_by_nbar = error_result["error_chi_by_n_bar"]
        if preferred_nbars is None:
            n_bars = sorted(float(key) for key in chi_by_nbar.keys())
        else:
            n_bars = [float(n_bar) for n_bar in preferred_nbars]
        return [(n_bar, _lookup_mapping_by_float_key(chi_by_nbar, n_bar)) for n_bar in n_bars]

    if "error_chi_matrix_list" in error_result:
        chi_values = error_result["error_chi_matrix_list"]
    elif "error_chi_qobj_list" in error_result:
        chi_values = error_result["error_chi_qobj_list"]
    else:
        raise KeyError(
            "error_result must contain error_chi_by_n_bar, error_chi_matrix_list, "
            "or error_chi_qobj_list."
        )

    if preferred_nbars is not None:
        n_bars = [float(n_bar) for n_bar in preferred_nbars]
    elif "results_list" in error_result:
        n_bars = [float(entry["n_bar"]) for entry in error_result["results_list"]]
    else:
        raise KeyError(
            "error_result needs parameters['n_bar_list'] or results_list to identify n_bar."
        )

    if len(n_bars) != len(chi_values):
        raise ValueError("The number of n_bar values does not match the number of chi matrices.")

    return sorted(zip(n_bars, chi_values), key=lambda item: item[0])


def save_error_result_chi_csv(error_result, csv_path):
    """Save chi matrices from an ``error_result`` dictionary to a CSV file.

    The CSV format is long-form:
    ``n_bar,row,col,real,imag``.
    This keeps complex 16x16 chi matrices easy to inspect and reconstruct.
    """
    points = _ordered_chi_points(error_result)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("n_bar", "row", "col", "real", "imag"))
        writer.writeheader()
        for n_bar, chi in points:
            chi = _as_array(chi, dtype=complex)
            if chi.shape != (16, 16):
                raise ValueError(f"chi must have shape (16, 16); got {chi.shape}.")
            for row in range(16):
                for col in range(16):
                    value = chi[row, col]
                    writer.writerow(
                        {
                            "n_bar": f"{float(n_bar):.17g}",
                            "row": row,
                            "col": col,
                            "real": f"{float(np.real(value)):.17g}",
                            "imag": f"{float(np.imag(value)):.17g}",
                        }
                    )

    return csv_path


def save_pauli_error_csv(error_result, csv_path, trace_normalize=True, atol=1e-12):
    """Save Pauli-twirled probabilities from ``error_result`` to CSV.

    The CSV format is ``n_bar,pauli,probability`` with 16 rows per ``n_bar``.
    Probabilities are obtained from ``real(diag(chi))`` after optional trace
    normalization.
    """
    points = _ordered_chi_points(error_result)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("n_bar", "pauli", "probability"))
        writer.writeheader()
        for n_bar, chi in points:
            probabilities = pauli_probabilities_from_chi(
                chi,
                trace_normalize=trace_normalize,
                atol=atol,
            )
            for label, probability in zip(PAULI_LABELS_2Q, probabilities):
                writer.writerow(
                    {
                        "n_bar": f"{float(n_bar):.17g}",
                        "pauli": label,
                        "probability": f"{float(probability):.17g}",
                    }
                )

    return csv_path


def load_error_result_chi_csv(csv_path):
    """Load a CSV made by ``save_error_result_chi_csv`` as an ``error_result``."""
    csv_path = Path(csv_path)
    matrices = {}

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"n_bar", "row", "col", "real", "imag"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must contain the columns: n_bar, row, col, real, imag."
            )

        for entry in reader:
            n_bar = float(entry["n_bar"])
            row = int(entry["row"])
            col = int(entry["col"])
            if not (0 <= row < 16 and 0 <= col < 16):
                raise ValueError(f"row and col must be in [0, 15]; got row={row}, col={col}.")
            if n_bar not in matrices:
                matrices[n_bar] = np.zeros((16, 16), dtype=complex)
            matrices[n_bar][row, col] = complex(float(entry["real"]), float(entry["imag"]))

    n_bars = sorted(matrices)
    return {
        "parameters": {"n_bar_list": n_bars},
        "error_chi_matrix_list": [matrices[n_bar] for n_bar in n_bars],
        "error_chi_by_n_bar": {n_bar: matrices[n_bar] for n_bar in n_bars},
    }


def load_pauli_error_csv(csv_path):
    """Load a ``n_bar,pauli,probability`` CSV as ordered Pauli probabilities."""
    csv_path = Path(csv_path)
    by_n_bar = {}

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"n_bar", "pauli", "probability"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("CSV must contain the columns: n_bar, pauli, probability.")

        for entry in reader:
            n_bar = float(entry["n_bar"])
            pauli = entry["pauli"]
            probability = float(entry["probability"])
            if pauli not in PAULI_LABELS_2Q:
                raise ValueError(f"Unknown Pauli label: {pauli}.")
            by_n_bar.setdefault(n_bar, {})[pauli] = probability

    loaded = {}
    for n_bar, probability_by_label in by_n_bar.items():
        missing = [label for label in PAULI_LABELS_2Q if label not in probability_by_label]
        if missing:
            raise ValueError(f"n_bar={n_bar} is missing Pauli probabilities: {missing}.")
        loaded[n_bar] = _normalize_probabilities(
            [probability_by_label[label] for label in PAULI_LABELS_2Q]
        )

    return loaded


def pauli_noise_from_pauli_error_csv(csv_path, n_bar, selection="exact", atol=1e-12):
    """Return a Pauli noise model from a saved Pauli error CSV."""
    target_n_bar = float(n_bar)
    probability_by_n_bar = load_pauli_error_csv(csv_path)
    points = sorted(probability_by_n_bar.items(), key=lambda item: item[0])

    if selection == "exact":
        probabilities = probability_by_n_bar.get(target_n_bar)
        if probabilities is None:
            probabilities = _lookup_mapping_by_float_key(
                probability_by_n_bar,
                target_n_bar,
                atol=atol,
            )
        chosen_n_bar = target_n_bar
    elif selection == "nearest":
        chosen_n_bar, probabilities = min(
            points,
            key=lambda point: abs(point[0] - target_n_bar),
        )
    elif selection == "interpolate":
        n_bars = np.asarray([point[0] for point in points], dtype=float)
        if len(n_bars) < 2:
            raise ValueError("At least two n_bar points are required for interpolation.")
        if target_n_bar < n_bars[0] or target_n_bar > n_bars[-1]:
            raise ValueError(
                f"n_bar={target_n_bar} is outside the interpolation range "
                f"[{n_bars[0]}, {n_bars[-1]}]."
            )
        probability_table = np.vstack([point[1] for point in points])
        probabilities = np.asarray(
            [
                np.interp(target_n_bar, n_bars, probability_table[:, index])
                for index in range(16)
            ]
        )
        chosen_n_bar = target_n_bar
    else:
        raise ValueError("selection must be 'exact', 'nearest', or 'interpolate'.")

    return build_pauli_noise_model(
        n_bar=chosen_n_bar,
        probabilities=probabilities,
        atol=atol,
    ).as_dict()


def _interpolate_probabilities(points, target_n_bar, trace_normalize=True, atol=1e-12):
    n_bars = np.asarray([point[0] for point in points], dtype=float)
    probability_table = np.vstack(
        [
            pauli_probabilities_from_chi(
                chi,
                trace_normalize=trace_normalize,
                atol=atol,
            )
            for _, chi in points
        ]
    )

    if len(n_bars) < 2:
        raise ValueError("At least two n_bar points are required for interpolation.")
    if target_n_bar < n_bars[0] or target_n_bar > n_bars[-1]:
        raise ValueError(
            f"n_bar={target_n_bar} is outside the interpolation range "
            f"[{n_bars[0]}, {n_bars[-1]}]."
        )

    interpolated = np.asarray(
        [np.interp(target_n_bar, n_bars, probability_table[:, index]) for index in range(16)]
    )
    return _normalize_probabilities(interpolated, atol=atol)


def pauli_noise_from_error_result(
    error_result,
    n_bar,
    *,
    selection="exact",
    trace_normalize=True,
    atol=1e-12,
):
    target_n_bar = float(n_bar)
    points = _ordered_chi_points(error_result)

    if selection == "exact":
        chi = dict(points).get(target_n_bar)
        if chi is None:
            chi = _lookup_mapping_by_float_key(dict(points), target_n_bar, atol=atol)
        model = pauli_noise_from_chi(
            chi,
            n_bar=target_n_bar,
            trace_normalize=trace_normalize,
            atol=atol,
        )
    elif selection == "nearest":
        chosen_n_bar, chi = min(points, key=lambda point: abs(point[0] - target_n_bar))
        model = pauli_noise_from_chi(
            chi,
            n_bar=chosen_n_bar,
            trace_normalize=trace_normalize,
            atol=atol,
        )
    elif selection == "interpolate":
        probabilities = _interpolate_probabilities(
            points,
            target_n_bar,
            trace_normalize=trace_normalize,
            atol=atol,
        )
        model = build_pauli_noise_model(
            n_bar=target_n_bar,
            probabilities=probabilities,
            atol=atol,
        )
    else:
        raise ValueError("selection must be 'exact', 'nearest', or 'interpolate'.")

    return model.as_dict()


def stim_pauli_channel_2_text(probabilities, targets=("q0", "q1"), atol=1e-12):
    """Format probabilities as a Stim PAULI_CHANNEL_2 instruction string."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape == (16,):
        probabilities = probabilities[1:]
    if probabilities.shape != (15,):
        raise ValueError(
            "Stim PAULI_CHANNEL_2 probabilities must contain 15 values "
            f"(or 16 values including II); got {probabilities.shape}."
        )
    if np.any(probabilities < -atol):
        raise ValueError("Stim probabilities must be non-negative.")
    if np.sum(probabilities) > 1.0 + atol:
        raise ValueError("Stim non-identity probabilities must sum to at most 1.")

    prob_text = ", ".join(f"{float(p):.17g}" for p in probabilities)
    target_text = " ".join(str(target) for target in targets)
    return f"PAULI_CHANNEL_2({prob_text}) {target_text}"


def pauli_noise_table(noise_model):
    """Return a compact list of ``{'pauli': label, 'probability': p}`` rows."""
    labels = noise_model["pauli_labels"]
    probabilities = noise_model["probabilities"]
    return [
        {"pauli": label, "probability": float(probability)}
        for label, probability in zip(labels, probabilities)
    ]
