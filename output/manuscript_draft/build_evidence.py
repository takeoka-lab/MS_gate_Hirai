"""Re-evaluate saved channels for the manuscript; no dynamical propagation."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pauli_channel_distance import (
    apply_chi, channel_diagnostics, pauli_basis, pauli_twirl_chi,
    qutip_chi_to_canonical,
)

SOURCE = ROOT / "results/pauli_twirl_channel_distance"
DEST = ROOT / "results/manuscript_draft"
DEST.mkdir(parents=True, exist_ok=True)
paired_path = SOURCE / "channel_distances_paired.csv"
matrix_path = SOURCE / "inputs/projected_chi_before_after.npz"
paired = pd.read_csv(paired_path)
matrices = np.load(matrix_path)
labels, paulis = pauli_basis()
ii, xx, ix, xi = [labels.index(s) for s in ("II", "XX", "IX", "XI")]
assert np.allclose(paired["n_bar"], matrices["n_bar"])
summary, states, diagnostics = [], [], []
bell_states = {
    "Phi+": np.array([1, 0, 0, 1]) / np.sqrt(2),
    "Psi-": np.array([0, 1, -1, 0]) / np.sqrt(2),
}
for k, row in paired.iterrows():
    chi = qutip_chi_to_canonical(matrices["chi_after"][k], labels)
    twirl = pauli_twirl_chi(chi)
    diag = channel_diagnostics(chi, paulis)
    assert diag["cptp_within_tolerance"]
    assert np.isclose(4 / 5 * (1 - chi[ii, ii].real),
                      row["average_infidelity_after"], atol=1e-12, rtol=0)
    # Channel-level projection on the real collective cross-term pattern.
    # The pattern has four entries: +w at IX,XI and -w at II,XX,
    # together with their Hermitian conjugates.
    off = chi - twirl
    w = (chi[ix, xi].real - chi[ii, xx].real) / 2
    pattern = np.zeros_like(chi)
    pattern[ix, xi] = pattern[xi, ix] = w
    pattern[ii, xx] = pattern[xx, ii] = -w
    norm2 = np.linalg.norm(off) ** 2
    share = np.linalg.norm(pattern) ** 2 / norm2
    residual = np.linalg.norm(off - pattern) / np.linalg.norm(off)
    assert np.isclose(share + residual ** 2, 1, atol=1e-12)
    record = {name: row[name] for name in (
        "n_bar", "A_factor_after", "h_XX_rad_per_gate_before",
        "h_XX_rad_per_gate_after", "average_infidelity_before",
        "average_infidelity_after", "diamond_distance_estimate_before",
        "diamond_distance_estimate_after", "relative_error_frobenius_distance_after",
        "diamond_sdp_gap_before", "diamond_sdp_gap_after",
        "taxonomy_C_IX_XI_after",
    )}
    record.update(
        collective_cross_weight=w,
        real_collective_pattern_offdiag_squared_fraction=share,
        offdiag_relative_residual_after_pattern=residual,
        chi_II_XX_real=chi[ii, xx].real,
        chi_II_XX_imag=chi[ii, xx].imag,
        chi_IX_XI_real=chi[ix, xi].real,
        collective_pair_mismatch=abs(chi[ix, xi].real + chi[ii, xx].real),
    )
    summary.append(record)
    diagnostics.append(dict(n_bar=row["n_bar"], **diag))
    for name, vector in bell_states.items():
        rho = np.outer(vector, vector.conj())
        r_full = 1 - np.trace(rho @ apply_chi(chi, rho, paulis)).real
        r_pauli = 1 - np.trace(rho @ apply_chi(twirl, rho, paulis)).real
        predicted_delta = -np.trace(rho @ apply_chi(pattern, rho, paulis)).real
        states.append(dict(
            n_bar=row["n_bar"], state=name, state_infidelity_full=r_full,
            state_infidelity_pauli=r_pauli, difference=r_full-r_pauli,
            collective_pattern_predicted_difference=predicted_delta,
            residual_difference=r_full-r_pauli-predicted_delta,
        ))
pd.DataFrame(summary).to_csv(DEST / "channel_summary.csv", index=False)
pd.DataFrame(states).to_csv(DEST / "bell_state_comparison.csv", index=False)
pd.DataFrame(diagnostics).to_csv(DEST / "channel_diagnostics.csv", index=False)
manifest = {
    "date": "2026-10-05",
    "scope": "Post-processing of saved CPTP-projected matrices; no new dynamics, convergence study, QEC, or matched-prior-model benchmark.",
    "convention": "canonical trace-one chi; Pauli order II,IX,...,ZZ; distances are half diamond norms",
    "pattern_fraction": "Squared Frobenius projection weight in chi off-diagonal space, NOT a fraction of diamond distance or gate infidelity.",
    "source_sha256": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (paired_path, matrix_path, Path(__file__), ROOT / "pauli_channel_distance.py")
    },
}
(DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(pd.DataFrame(summary)[[
    "n_bar", "collective_cross_weight",
    "real_collective_pattern_offdiag_squared_fraction",
    "offdiag_relative_residual_after_pattern",
]].to_string(index=False))
print("Maximum Bell-difference residual:",
      max(abs(s["residual_difference"]) for s in states))
print("Evidence saved to", DEST)
