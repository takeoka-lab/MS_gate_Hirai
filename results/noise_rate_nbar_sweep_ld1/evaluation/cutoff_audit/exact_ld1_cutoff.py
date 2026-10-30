"""Independent finite-cutoff LD1 check using a constant rotating-frame Hamiltonian.

No time integration, QPT reconstruction, or matrix-square-root fidelity is used.
For a one-loop constant drive, R(T)=exp(i*delta*T*N)=I, so in a sector
Sx=s the oscillator propagator is exp[-i*T*(delta*N+A*s*(a+a.dag()))].
The infinite-dimensional ideal spin phases are exp[i*2*pi*(A/delta)^2*s^2].
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
LD1 = OUT.parent.parent
config = json.loads((LD1 / "config.json").read_text())
p = config["base_parameters"]
assert p["use_full_order"] is False
assert p["rho0"] == 0 and p["delta"] > 0
A, delta = p["A"], p["delta"]
T = 2 * np.pi / delta
saved = pd.read_csv(LD1 / "all_noise_zero_summary.csv").set_index("nbar")


def current_cutoff(nbar, alpha):
    # The current get_optimal_nv_general rule, independently evaluated.
    mean = nbar + alpha**2
    variance = (2*nbar+1)*alpha**2 + nbar*(nbar+1)
    estimate = int(np.ceil(mean + (10+2*np.log1p(nbar))*np.sqrt(variance)))
    return max(estimate, int(5*nbar)+20, 2)


def exact_finite_cutoff_infidelity(nbar, dimension):
    a = np.diag(np.sqrt(np.arange(1, dimension)), 1)
    number = np.diag(np.arange(dimension))
    weights = (nbar / (nbar+1))**np.arange(dimension)
    weights /= weights.sum()
    target_weighted_sum = np.zeros((dimension, dimension), complex)
    for s in (2, 0, 0, -2):
        eigenvalues, vectors = np.linalg.eigh(delta*number + A*s*(a+a.T))
        unitary = (vectors*np.exp(-1j*T*eigenvalues)) @ vectors.conj().T
        phase = np.exp(1j*2*np.pi*(A/delta)**2*s*s)
        target_weighted_sum += np.conj(phase)*unitary
    # Entanglement fidelity = sum_n p_n ||sum_s u_s^* U_s |n>||^2 / d^2.
    fe = np.real(np.sum(np.abs(target_weighted_sum)**2 * weights[None, :])) / 16
    return float(1 - (4*fe+1)/5)


rows = []
for nbar in config["nbar_values"]:
    base_dimension = current_cutoff(nbar, 2*A/delta)
    for factor in (1.0, 1.5, 2.0):
        dimension = int(np.ceil(factor*base_dimension))
        rows.append({
            "nbar": nbar, "cutoff_factor": factor, "phonon_dim": dimension,
            "saved_infidelity": saved.loc[nbar, "infidelity"],
            "saved_raw_infidelity": 1-saved.loc[nbar, "F_avg"],
            "exact_truncated_infidelity": exact_finite_cutoff_infidelity(nbar, dimension),
        })
table = pd.DataFrame(rows)
table.to_csv(OUT / "exact_cutoff_comparison.csv", index=False)
print(table.to_string(index=False))
