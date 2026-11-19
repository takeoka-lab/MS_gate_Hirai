"""Compare each stored 100 µs post-gate channel with its own Pauli twirl."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from drive_calibration_qpt_analysis import nearest_cptp_choi_dykstra
from pauli_channel_distance import (
    channel_diagnostics, chi_to_choi, choi_to_chi, diamond_distance_sdp,
    pauli_basis, pauli_twirl_chi, qutip_chi_to_canonical,
)

DATA = Path(__file__).resolve().parent / "results/matched_pulse_calibration_sweep/100us"


def compare(tolerance):
    labels, paulis = pauli_basis()
    rows = []
    for point in pd.read_csv(DATA / "summary.csv").itertuples():
        with np.load(DATA / f"{point.condition}_{point.n_bar:g}.npz") as cache:
            if not np.isclose(float(cache["gate_time_s"]), 100e-6, atol=1e-12, rtol=0):
                raise ValueError("Expected a 100 µs gate")
            raw = qutip_chi_to_canonical(cache["chi"], labels)
        choi, _ = nearest_cptp_choi_dykstra(chi_to_choi(raw, paulis))
        chi = choi_to_chi(choi, paulis)
        if not channel_diagnostics(chi, paulis)["cptp_within_tolerance"]:
            raise ValueError(f"CPTP projection failed for {point.condition}, {point.n_bar}")
        delta = chi_to_choi(chi - pauli_twirl_chi(chi), paulis)
        result = diamond_distance_sdp(delta, 4)
        low, high = result["diamond_sdp_lower"], result["diamond_sdp_upper"]
        status = "unresolved"
        if result["sdp_accepted"]:
            if high <= tolerance:
                status = "within"
            elif low > tolerance:
                status = "outside"
        rows.append(dict(condition=point.condition, n_bar=point.n_bar,
                         tolerance=tolerance, classification=status, **result))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, required=True)
    args = parser.parse_args()
    if not 0 <= args.tolerance <= 1:
        parser.error("tolerance must be between zero and one")
    table = compare(args.tolerance)
    table.to_csv(DATA / "diamond_distances.csv", index=False)
    print(table[["condition", "n_bar", "diamond_distance_estimate", "classification"]].to_string(index=False))
