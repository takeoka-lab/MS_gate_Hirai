"""Compare repeated rectangular/Hughes gates before and after Pauli twirling.

Uses the saved Results.ipynb channels at n_bar=4, gate time=100 us, all noise
on. Calibration means drive-amplitude calibration at fixed pulse shape.
No QPT or calibration is launched, including when a cache is missing.

Run manually:
    python repeated_gate_pauli_twirl_comparison.py
    python repeated_gate_pauli_twirl_comparison.py --hughes-reference 21700

Default Hughes shape: delta_min_ref=22245.459873243974 Hz BEFORE compression
to 100 us (the optimized shape in Results.ipynb). The 21700 option selects
the earlier shape for BOTH nominal and calibrated channels.

Two distinct comparisons are saved:
  error_only: r(E**n, I)
  physical_gate: r((E U)**n, U**n), U=exp(+i pi XX/4)
E is the post-gate error map, E=G U^{-1}. Twirling is applied to E once
BEFORE composing n independent copies, never to the final n-gate map.
Both comparisons assume a stationary memoryless reduced spin channel:
motion is reset to the same thermal state between gates. They do not model
continuous spin-motion evolution or noise correlated across gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drive_calibration_qpt_analysis import project_trace_normalized_chi_to_cptp
from pauli_channel_distance import (
    channel_diagnostics,
    chi_to_ptm,
    pauli_basis,
    pauli_twirl_chi,
    qutip_chi_to_canonical,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results/matched_pulse_calibration_sweep/100us"
DEFAULT_OUTPUT = ROOT / "results/repeated_gate_pauli_twirl"
REPETITIONS = (1, 3, 5, 10)


def input_conditions(hughes_reference):
    nominal, calibrated = {
        "22245": ("nominal_hughes_22245_n4", "smooth_detuning_4"),
        "21700": ("nominal_hughes_21700_n4", "detuning_check_21700_n4"),
    }[hughes_reference]
    return {
        "Nominal rectangular": "standard_nominal_4",
        "Calibrated rectangular": "standard_4",
        "Nominal Hughes": nominal,
        "Calibrated Hughes": calibrated,
    }


def unitary_ptm(unitary, paulis):
    d = unitary.shape[0]
    return np.column_stack([
        np.einsum("aij,ij->a", paulis.conj(), unitary @ p @ unitary.conj().T) / d
        for p in paulis
    ])


def average_infidelity(actual, ideal, dimension=4):
    """Average infidelity to a unitary ideal channel in orthonormal PTM basis."""
    overlap = np.trace(ideal.conj().T @ actual)
    if abs(overlap.imag) > 1e-8:
        raise ValueError(f"Unexpected complex fidelity overlap: {overlap}")
    # No clipping: preserve small numerical deviations for inspection.
    return float(1 - (overlap.real + dimension) / (dimension * (dimension + 1)))


def compare(input_dir, hughes_reference="22245", repetitions=REPETITIONS):
    repetitions = tuple(repetitions)
    if not repetitions or any(int(n) != n or n < 1 for n in repetitions):
        raise ValueError("Repetition counts must be positive integers")
    labels, paulis = pauli_basis(2)
    xx = paulis[labels.index("XX")]
    unitary = (np.eye(4) + 1j * xx) / np.sqrt(2)
    ideal = unitary_ptm(unitary, paulis)
    identity = np.eye(16)
    records, diagnostics, matrices = [], [], {}
    conditions = input_conditions(hughes_reference)
    missing = [input_dir / f"{stem}.npz" for stem in conditions.values()
               if not (input_dir / f"{stem}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"Saved channels required; no QPT fallback: {missing}")

    for condition, stem in conditions.items():
        path = input_dir / f"{stem}.npz"
        with np.load(path, allow_pickle=False) as data:
            raw = np.asarray(data["chi"], dtype=complex)
            scale = float(data["scale"])
            if "gate_time_s" in data and not np.isclose(
                float(data["gate_time_s"]), 100e-6, rtol=1e-8, atol=1e-12
            ):
                raise ValueError(f"Expected a 100 us channel: {path}")
        if raw.shape != (16, 16) or not np.all(np.isfinite(raw)):
            raise ValueError(f"Invalid chi matrix: {path}")
        if not np.isclose(np.trace(raw), 1, rtol=0, atol=1e-8):
            raise ValueError(f"Expected trace-one QuTiP chi: {path}")
        # Match the CPTP projection used in the Results.ipynb twirling analysis.
        # Apply exactly the same projection BEFORE branching into full/twirled.
        _, projected, status = project_trace_normalized_chi_to_cptp(raw)
        chi = qutip_chi_to_canonical(projected, labels)
        physicality = channel_diagnostics(chi, paulis)
        if not physicality["cptp_within_tolerance"]:
            raise ValueError(f"CPTP projection failed for {path}: {physicality}")
        diagnostics.append({
            "condition": condition, "source": str(path), "amplitude_scale": scale,
            "projection_chi_frobenius_change": float(np.linalg.norm(projected - raw)),
            "projection_status": status, **physicality,
        })
        full = chi_to_ptm(chi, paulis)
        twirl = chi_to_ptm(pauli_twirl_chi(chi), paulis)
        if not np.isclose(average_infidelity(full, identity),
                          average_infidelity(twirl, identity), rtol=0, atol=1e-10):
            raise ValueError("Single-gate infidelity must be invariant under twirling")
        matrices[f"{stem}_full_error_ptm"] = full
        matrices[f"{stem}_twirled_error_ptm"] = twirl
        for mode in ("error_only", "physical_gate"):
            for model, error in (("full", full), ("twirled", twirl)):
                gate = error if mode == "error_only" else error @ ideal
                target = identity if mode == "error_only" else ideal
                r1 = average_infidelity(gate, target)
                for n in repetitions:
                    rn = average_infidelity(np.linalg.matrix_power(gate, n),
                                             np.linalg.matrix_power(target, n))
                    records.append({
                        "condition": condition, "mode": mode, "model": model,
                        "n": n, "n_bar": 4.0, "gate_time_s": 100e-6,
                        "amplitude_scale": scale, "average_infidelity": rn,
                        "average_infidelity_percent": 100 * rn,
                        "n_times_single_gate_infidelity": n * r1,
                        "source": str(path),
                    })
    matrices["ideal_gate_ptm"] = ideal
    return pd.DataFrame(records), diagnostics, matrices


def save_comparison(table, diagnostics, matrices, output_dir, hughes_reference):
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "repeated_infidelity.csv", index=False)
    wide = table.pivot(index=["condition", "mode", "n"], columns="model",
                       values="average_infidelity").reset_index()
    wide["full_minus_twirled"] = wide["full"] - wide["twirled"]
    wide.to_csv(output_dir / "twirling_difference.csv", index=False)
    np.savez_compressed(output_dir / "single_gate_ptms.npz", **matrices)
    (output_dir / "metadata.json").write_text(json.dumps({
        "n_bar": 4.0, "gate_time_s": 100e-6,
        "hughes_delta_min_before_compression_hz": (
            22245.459873243974 if hughes_reference == "22245" else 21700.0),
        "assumption": "Stationary reduced spin channel; thermal motion reset each gate",
        "convention": "E = G U^-1; U = exp(+i pi XX/4); per-gate twirling of E",
        "diagnostics": diagnostics,
    }, indent=2, default=str) + "\n", encoding="utf-8")

    conditions = list(table.condition.unique())
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, layout="constrained")
    for row, mode in enumerate(("error_only", "physical_gate")):
        for col, condition in enumerate(conditions):
            ax = axes[row, col]
            subset = table[(table.condition == condition) & (table["mode"] == mode)]
            for model, style in (("full", "o-"), ("twirled", "s--")):
                data = subset[subset.model == model].sort_values("n")
                ax.plot(data.n, data.average_infidelity_percent, style, label=model)
            ax.set_title(condition)
            ax.set_xticks(sorted(table.n.unique()))
            ax.set_xlabel("Repetitions n")
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel(f"{mode}\nAverage infidelity (%)")
            ax.legend()
    fig.suptitle("n_bar = 4; 100 us/gate; independent channel composition")
    fig.savefig(output_dir / "repeated_infidelity.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hughes-reference", choices=("22245", "21700"), default="22245")
    parser.add_argument("--repetitions", type=int, nargs="+", default=list(REPETITIONS))
    args = parser.parse_args()
    table, diagnostics, matrices = compare(
        args.input_dir, args.hughes_reference, args.repetitions)
    destination = args.output_dir / f"hughes_{args.hughes_reference}"
    save_comparison(table, diagnostics, matrices, destination, args.hughes_reference)
    print(table[["condition", "mode", "model", "n", "average_infidelity"]].to_string(index=False))
    print(f"Saved results: {destination}")


if __name__ == "__main__":
    main()
