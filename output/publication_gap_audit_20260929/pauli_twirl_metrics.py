"""Evaluate saved channels; no new dynamical simulation or SDP is performed.

The input chi uses trace-one normalization and unnormalized Pauli operators.
For d=4, ||R(E)-R(T(E))||_F = d*||chi-diag(chi)||_F.
The relative Frobenius denominator is ||R(E)-I||_F, not ||R(E)||_F.
In this convention chi is unitarily equivalent to the normalized Choi state.
Consequently, for Delta=E-T(E),
  ||chi_off||_1 / 2 <= ||Delta||_diamond / 2 <= d*||chi_off||_1 / 2.
These are matrix-norm bounds, not SDP estimates or solver error certificates.
"""

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/calibration_channel_comparison/projected_chi_before_after.npz"


def metrics(chi):
    chi = np.asarray(chi, dtype=complex)
    assert chi.shape == (16, 16)
    assert abs(np.trace(chi) - 1) < 1e-8
    assert np.linalg.norm(chi - chi.conj().T) < 1e-8
    # The saved CPTP projection has finite numerical tolerance.
    assert np.linalg.eigvalsh((chi + chi.conj().T) / 2).min() > -1e-8
    off = chi - np.diag(np.diag(chi))
    error = chi.copy()
    error[0, 0] -= 1
    trace_norm = float(np.linalg.svd(off, compute_uv=False).sum())
    return {
        "ptm_frobenius_twirl_difference": float(4 * np.linalg.norm(off)),
        "relative_error_frobenius_twirl_difference": float(
            np.linalg.norm(off) / np.linalg.norm(error)
        ),
        "diamond_distance_lower_bound": trace_norm / 2,
        "diamond_distance_upper_bound": 4 * trace_norm / 2,
        "average_infidelity": float(4 / 5 * (1 - chi[0, 0].real)),
        "min_chi_eigenvalue": float(np.linalg.eigvalsh(chi).min()),
    }


def main():
    rows = []
    with np.load(SOURCE, allow_pickle=False) as data:
        for i, nbar in enumerate(data["n_bar"]):
            for condition in ("before", "after"):
                rows.append({
                    "n_bar": float(nbar),
                    "condition": condition,
                    **metrics(data[f"chi_{condition}"][i]),
                })
    destination = Path(__file__).with_name("pauli_twirl_metrics.csv")
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(destination)
    for before, after in zip(rows[::2], rows[1::2]):
        factor = (before["ptm_frobenius_twirl_difference"]
                  / after["ptm_frobenius_twirl_difference"])
        separated = (after["diamond_distance_upper_bound"]
                     < before["diamond_distance_lower_bound"])
        print(f"nbar={before['n_bar']:g}: Frobenius reduction={factor:.3f}; "
              f"diamond upper_after < lower_before: {separated}")


if __name__ == "__main__":
    main()
