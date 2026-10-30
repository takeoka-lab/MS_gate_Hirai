"""Postprocess saved calibrated MS channels; no master equation is rerun.

Run with Python + NumPy. Outputs go beside this script. Bell infidelity is
1 - <psi|N(|psi><psi|)|psi>, not a logical error rate or average infidelity.
These Bell states are XX eigenstates, so the ideal MS gate leaves their
density matrices invariant. The same comparison therefore applies to the
full gate for these inputs.
"""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/calibration_channel_comparison"
raw = np.load(SOURCE / "post_gate_noise_chi_before_after.npz")
projected = np.load(SOURCE / "projected_chi_before_after.npz")
single = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}
paulis = np.array([np.kron(single[p[0]], single[p[1]]) for p in raw["pauli_labels"]])
bells = {"Phi+": [1, 0, 0, 1], "Phi-": [1, 0, 0, -1],
         "Psi+": [0, 1, 1, 0], "Psi-": [0, 1, -1, 0]}


def apply_channel(chi, rho):
    return np.einsum("mn,mab,bc,ncd->ad", chi, paulis, rho, paulis, optimize=True)


rows, diagnostics = [], []
for i, nbar in enumerate(raw["n_bar"]):
    if nbar > 4:
        continue
    chi = projected["chi_after"][i]
    twirl = np.diag(np.diag(chi))
    tp = sum((chi[m, n] * paulis[n] @ paulis[m]
              for m in range(16) for n in range(16)), np.zeros((4, 4), complex))
    diagnostics.append({
        "nbar": float(nbar),
        "chi_trace_real": float(np.trace(chi).real),
        "chi_hermiticity_fro": float(np.linalg.norm(chi - chi.conj().T)),
        "min_chi_eigenvalue": float(np.linalg.eigvalsh(chi).min()),
        "direct_pauli_tp_fro": float(np.linalg.norm(tp - np.eye(4))),
        "raw_to_projected_chi_fro": float(np.linalg.norm(chi - raw["chi_after"][i])),
    })
    for name, vector in bells.items():
        v = np.array(vector, dtype=complex) / np.sqrt(2)
        rho = np.outer(v, v.conj())
        output = apply_channel(chi, rho)
        output_pt = apply_channel(twirl, rho)
        # Independently check that diagonalizing chi agrees with group twirling.
        group_pt = sum((p @ apply_channel(chi, p @ rho @ p) @ p
                        for p in paulis), np.zeros((4, 4), complex)) / 16
        assert np.linalg.norm(group_pt - output_pt) < 1e-12
        assert abs(np.trace(output) - 1) < 1e-6
        assert np.linalg.norm(output - output.conj().T) < 1e-12
        delta = output - output_pt
        error = float(1 - np.trace(rho @ output).real)
        error_pt = float(1 - np.trace(rho @ output_pt).real)
        raw_error = float(1 - np.trace(rho @ apply_channel(raw["chi_after"][i], rho)).real)
        rows.append({
            "nbar": float(nbar), "state": name,
            "state_infidelity_original": error,
            "state_infidelity_pauli_twirl": error_pt,
            "pauli_minus_original_state_infidelity": error_pt - error,
            "pauli_to_original_state_infidelity_ratio": error_pt / error,
            "half_output_difference_trace_norm": float(np.linalg.svd(delta, compute_uv=False).sum() / 2),
            "raw_vs_projected_state_infidelity_difference": abs(raw_error - error),
            "average_gate_infidelity_both": float(4 / 5 * (1 - chi[0, 0].real)),
        })

with (OUT / "bell_pauli_comparison.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT / "bell_pauli_diagnostics.json").write_text(json.dumps({
    "source_raw": str(SOURCE / "post_gate_noise_chi_before_after.npz"),
    "source_projected": str(SOURCE / "projected_chi_before_after.npz"),
    "convention": str(raw["convention"]),
    "interpretation": "Saved-channel postprocessing only; not QEC or a new converged dynamical simulation.",
    "diagnostics": diagnostics,
}, ensure_ascii=False, indent=2) + "\n")
for row in rows:
    if row["nbar"] == 4 and row["state"] in ("Phi+", "Phi-"):
        print(json.dumps(row, ensure_ascii=False))
