"""Evaluate saved LD1 sweeps without rerunning QPT or changing source data."""
from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ms_gate_ld1_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FULL = HERE.parent / "noise_rate_nbar_sweep"
OUT = HERE / "evaluation"
OUT.mkdir(exist_ok=True)
config = json.loads((HERE / "config.json").read_text())
full_config = json.loads((FULL / "config.json").read_text())
assert config["base_parameters"]["use_full_order"] is False
assert full_config["base_parameters"]["use_full_order"] is True
for cfg in (config, full_config):
    for key in ("parallel_workers", "show_progress", "use_full_order"):
        cfg["base_parameters"].pop(key, None)
assert config == full_config, "Other scientific settings differ."

ld = pd.read_csv(HERE / "summary.csv")
full = pd.read_csv(FULL / "summary.csv")
zero = pd.read_csv(HERE / "all_noise_zero_summary.csv").set_index("nbar")
sources = config["noise_sources"]
nbars = np.array(config["nbar_values"])
expected = {(s, float(m), float(n)) for s in sources
            for m in config["rate_multipliers"] for n in nbars}
for table in (ld, full):
    assert not table.duplicated(["noise_source", "multiplier", "nbar"]).any()
    assert set(zip(table.noise_source, table.multiplier, table.nbar)) == expected
    assert np.isfinite(table[["F_avg", "infidelity"]].to_numpy()).all()
    assert table.infidelity.between(0, 1).all()
    assert (table[table.multiplier == 1].groupby("nbar").infidelity
            .agg(lambda x: x.max() - x.min()).max() < 1e-12)
assert np.array_equal(np.sort(zero.index), nbars)

def curve(table, source, multiplier):
    return (table[(table.noise_source == source) & (table.multiplier == multiplier)]
            .set_index("nbar").infidelity.reindex(nbars))

nominal = curve(ld, sources[0], 1)
full_nominal = curve(full, sources[0], 1)
comparison = pd.DataFrame({"full_nominal": full_nominal, "ld1_nominal": nominal,
                           "ld1_no_noise": zero.infidelity})
comparison["reduction_percent"] = 100 * (1 - nominal / full_nominal)
comparison.to_csv(OUT / "nominal_comparison.csv")

increments = pd.DataFrame({s: nominal - curve(ld, s, 0) for s in sources})
increments.to_csv(OUT / "leave_one_out_increments.csv")
summary_rows = []
for s in sources:
    y = increments[s]
    slope, intercept = np.polyfit(nbars, y, 1)
    summary_rows.append(dict(source=s, low_nbar_increment=y.iloc[0],
                             high_nbar_increment=y.iloc[-1],
                             endpoint_change_percent=100 * (y.iloc[-1] / y.iloc[0] - 1),
                             slope_per_nbar=slope, intercept=intercept,
                             max_linear_fit_residual=max(abs(y - (slope * nbars + intercept)))))
source_summary = pd.DataFrame(summary_rows)
source_summary.to_csv(OUT / "source_summary.csv", index=False)

rate_rows = []
for s in sources:
    for m in (0.5, 1, 2, 4):
        observed = curve(ld, s, m) - curve(ld, s, 0)
        deviation = 100 * (observed / (m * increments[s]) - 1)
        rate_rows.append(dict(source=s, multiplier=m,
                             max_absolute_deviation_from_nominal_scaling_percent=max(abs(deviation))))
pd.DataFrame(rate_rows).to_csv(OUT / "rate_linearity.csv", index=False)

no_md = curve(ld, "motional_dephasing", 0)
no_md_corrected = no_md - zero.infidelity
params = config["base_parameters"]
tg = params["t_gate_phys"]
gamma_m = params["dephasing_rate_phys"]
# First order in dissipative rates, one closed loop, A/delta = 1/4,
# Sx = XI + IX, and the repository's Lindblad normalization.
assert np.isclose(params["A"] / params["delta"], 0.25)
md_theory = gamma_m * tg * (0.4 * nbars + 0.275)
other_theory = tg * (0.4 * params["heating_rate_phys"]
                    + 0.8 / params["T2_star"]
                    + 1.6 * params["rayleigh_rate_phys"]
                    + 0.8 * params["raman_rate_phys"])
theory = pd.DataFrame({"nbar": nbars, "md_first_order": md_theory,
                      "md_leave_one_out": increments.motional_dephasing.to_numpy(),
                      "all_noise_first_order": other_theory + md_theory,
                      "all_noise_qpt": nominal.to_numpy()})
theory.to_csv(OUT / "first_order_theory_comparison.csv", index=False)
metrics = {
    "complete_sweep_rows_per_model": len(ld),
    "ld1_no_noise_max_infidelity": float(zero.infidelity.max()),
    "ld1_no_noise_min_raw_infidelity": float((1 - zero.F_avg).min()),
    "no_md_minus_no_noise_range": float(np.ptp(no_md_corrected)),
    "thermal_growth_removed_by_switching_off_md_fraction": float(
        (increments.motional_dephasing.iloc[-1] - increments.motional_dephasing.iloc[0])
        / (nominal.iloc[-1] - nominal.iloc[0])),
    "md_first_order_theory_max_relative_difference": float(
        max(abs(increments.motional_dephasing.to_numpy() / md_theory - 1))),
    "all_noise_first_order_theory_max_relative_difference": float(
        max(abs(nominal.to_numpy() / (other_theory + md_theory) - 1))),
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
ax = axes[0]
ax.plot(nbars, full_nominal * 1e3, "o-", color="#d97706", label=r"With relative $O(\eta^2)$ correction")
ax.plot(nbars, nominal * 1e3, "o-", color="#2563eb", label="LD1: all nominal noises")
ax.plot(nbars, no_md * 1e3, "s--", color="#059669", label="LD1: motional dephasing off")
ax.plot(nbars, zero.infidelity * 1e3, "k:", label="LD1: all noises off")
ax.set(title="A   Total gate error", ylabel=r"Average infidelity $1-F_{\rm avg}$ ($\times 10^{-3}$)")
ax.legend(fontsize=9, loc="upper left")
labels = {"motional_heating": "Motional heating", "motional_dephasing": "Motional dephasing",
          "spin_dephasing": "Spin dephasing", "photon_scattering": "Photon scattering"}
colors = ["#2563eb", "#d97706", "#7c3aed", "#059669"]
ax = axes[1]
for s, color in zip(sources, colors):
    ax.plot(nbars, increments[s] * 1e3, "o-", label=labels[s], color=color)
ax.set(title="B   LD1: effect of removing each source",
       ylabel=r"$I_{\rm all}-I_{\rm without\ i}$ ($\times 10^{-3}$)")
ax.legend(fontsize=9, loc="upper left")
for ax in axes:
    ax.set_xlabel(r"Initial mean phonon number $\bar n$")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
fig.suptitle("First-order Lamb-Dicke result: the remaining thermal trend comes from motional dephasing", fontsize=12)
fig.text(0.5, 0.025, "Panel B keeps the other three rates nominal; these are not isolated-source infidelities.",
         ha="center", fontsize=9, color="#475569")
fig.tight_layout(rect=(0, 0.065, 1, 0.94))
fig.savefig(OUT / "ld1_evaluation.png", dpi=200)
plt.close(fig)
print(json.dumps(metrics, indent=2))
print(source_summary.to_string(index=False))
print("Saved evaluation to", OUT)
