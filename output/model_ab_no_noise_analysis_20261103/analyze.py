"""Independent, noise-free checks for the rectangular-pulse A/B comparison.

At T=2*pi/delta, H(t)=R(t) Hc R(t)^dagger with R=exp(i*delta*N*t),
so U(T)=exp[-i*T*(delta*N+g*Sx*(B+Bdag))].  This spectral calculation
has no time-grid interpolation or master-equation integration error.
It solves the same truncated first-sideband Hamiltonians, not the full
optical interaction.  Saved simulation outputs are read, never altered.
"""

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/ld_second_order_rwa_vs_corrected_sideband/b9551654dce042c7"
OUTPUT = Path(__file__).resolve().parent
CONFIG = json.loads((SOURCE / "config.json").read_text())
PARAM = CONFIG["base_parameters"]
DELTA = PARAM["delta"]
ETA = PARAM["eta"]
G = PARAM["A"]
T = 2 * np.pi / DELTA
TARGET = np.pi / 4
XX_EIGEN = np.array([1., -1., -1., 1.])


def conditional_unitaries(model, dim, amplitude):
    n = np.arange(dim)
    lower = np.sqrt(np.arange(1, dim, dtype=float))
    if model == "B":
        lower *= 1 - ETA**2 * np.arange(1, dim) / 2
    eigen, vectors = eigh_tridiagonal(DELTA * n, 2 * amplitude * lower)
    positive = (vectors * np.exp(-1j * T * eigen)) @ vectors.T
    parity = (-1.)**n
    negative = parity[:, None] * positive * parity[None, :]
    unit = np.eye(dim, dtype=complex)
    assert np.linalg.norm(positive.conj().T @ positive - unit) < 1e-10
    return [positive, unit, unit, negative]


def schur_channel(unitaries, nbar):
    n = np.arange(unitaries[0].shape[0])
    q = nbar / (nbar + 1)
    probabilities = (1 - q) * q**n
    probabilities /= probabilities.sum()
    flattened = np.stack([(u * np.sqrt(probabilities)[None, :]).ravel() for u in unitaries])
    coherence = flattened @ flattened.conj().T
    assert np.max(np.abs(np.diag(coherence) - 1)) < 1e-12
    assert np.linalg.eigvalsh(coherence).min() > -1e-12
    return coherence, probabilities


def average_fidelity(coherence, angle=TARGET):
    ideal = np.exp(1j * angle * XX_EIGEN)
    process = np.sum(coherence * ideal.conj()[:, None] * ideal[None, :]).real / 16
    return (4 * process + 1) / 5


def point(model, nbar, dim, amplitude=G):
    unitaries = conditional_unitaries(model, dim, amplitude)
    coherence, probabilities = schur_channel(unitaries, nbar)
    forced_overlap = np.diag(unitaries[0])
    average_overlap = np.dot(probabilities, forced_overlap)
    fitted_angle = .5 * np.angle(average_overlap)
    fock_angles = .5 * np.unwrap(np.angle(forced_overlap))
    mean_angle = np.dot(probabilities, fock_angles)
    phase_only_overlap = np.dot(probabilities, np.exp(2j * fock_angles))
    uncorrected = 1 - average_fidelity(coherence)
    unitary_corrected = 1 - average_fidelity(coherence, fitted_angle)
    return {
        "model": model, "nbar": nbar, "phonon_dim": dim, "drive_amplitude": amplitude,
        "infidelity": uncorrected,
        "fitted_xx_angle_rad": fitted_angle,
        "fitted_angle_error_rad": fitted_angle - TARGET,
        "infidelity_after_best_xx_unitary": unitary_corrected,
        "infidelity_removed_by_best_xx_unitary": uncorrected - unitary_corrected,
        "fock_angle_mean_rad": mean_angle,
        "fock_angle_std_rad": np.sqrt(np.dot(probabilities, (fock_angles-mean_angle)**2)),
        "forced_null_overlap_magnitude": abs(average_overlap),
        "phase_only_overlap_magnitude": abs(phase_only_overlap),
        "mean_fock_return_probability": np.dot(probabilities, abs(forced_overlap)**2),
        "plus_minus_overlap_real": coherence[0, 3].real,
        "omitted_initial_thermal_weight": (nbar / (nbar+1))**dim,
    }, forced_overlap


summary = pd.read_csv(SOURCE / "summary.csv")
saved = summary[summary["case"] == "no_noise"]
checks = []
rows = []
fock_rows = []

for nbar in CONFIG["nbar_values"]:
    for model in ("A", "B"):
        saved_row = saved[(saved.model == model) & (saved.nbar == nbar)].iloc[0]
        saved_dim = int(saved_row.phonon_dim)
        small, _ = point(model, nbar, saved_dim)
        large100, _ = point(model, nbar, 100)
        large, overlaps = point(model, nbar, 140)
        checks.append({
            "model": model, "nbar": nbar, "saved_phonon_dim": saved_dim,
            "saved_mesolve_infidelity": saved_row.infidelity,
            "spectral_same_cutoff_infidelity": small["infidelity"],
            "spectral_dim100_infidelity": large100["infidelity"],
            "spectral_dim140_infidelity": large["infidelity"],
            "dim100_vs_dim140_absolute_difference": abs(large100["infidelity"] - large["infidelity"]),
        })
        rows.append(large)
        if nbar == CONFIG["nbar_values"][0]:
            for n in range(31):
                fock_rows.append({"model": model, "fock_n": n,
                                  "xx_angle_rad": .5*np.angle(overlaps[n]),
                                  "return_probability": abs(overlaps[n])**2})

calibrations = []
for nbar in CONFIG["nbar_values"]:
    def objective(amplitude):
        c, _ = schur_channel(conditional_unitaries("B", 140, amplitude), nbar)
        return 1 - average_fidelity(c)
    opt = minimize_scalar(objective, bounds=(.9 * G, 1.15 * G), method="bounded",
                          options={"xatol": 1e-11})
    assert opt.success
    calibrated, _ = point("B", nbar, 140, opt.x)
    check100, _ = point("B", nbar, 100, opt.x)
    calibrations.append({**calibrated, "amplitude_ratio": opt.x/G,
                         "dim100_vs_dim140_absolute_difference": abs(check100["infidelity"]-calibrated["infidelity"]),
                         "calibration_criterion": "minimize noise-free average gate infidelity at each nbar"})

pd.DataFrame(checks).to_csv(OUTPUT / "cutoff_and_solver_check.csv", index=False)
pd.DataFrame(rows).to_csv(OUTPUT / "no_noise_decomposition.csv", index=False)
pd.DataFrame(calibrations).to_csv(OUTPUT / "amplitude_calibration.csv", index=False)
pd.DataFrame(fock_rows).to_csv(OUTPUT / "fock_angles.csv", index=False)
report = {
    "source_results": str(SOURCE.relative_to(ROOT)),
    "source_config_sha256": hashlib.sha256((SOURCE / "config.json").read_bytes()).hexdigest(),
    "method": "exact conditional rotating-frame spectral propagation for rectangular constant-phase driving",
    "base_parameters": PARAM,
    "cutoff_comparison": [100, 140],
    "unitary_correction": "best fitted XX rotation of the fixed-amplitude channel; not drive recalibration",
    "amplitude_calibration": "separate per-temperature optimization of the scalar drive; no dissipators",
    "max_cutoff_difference": max(r["dim100_vs_dim140_absolute_difference"] for r in checks),
    "full_default_master_equation_sweep_rerun": False,
}
(OUTPUT / "analysis_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print('CUTOFF CHECK')
print(pd.DataFrame(checks).to_string(index=False))
print('DECOMPOSITION')
print(pd.DataFrame(rows).query("model == 'B'").to_string(index=False))
print('CALIBRATED')
print(pd.DataFrame(calibrations)[['nbar','drive_amplitude','amplitude_ratio','infidelity','fitted_angle_error_rad','fock_angle_std_rad']].to_string(index=False))
