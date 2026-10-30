"""Plot saved Fock/QPT coefficients; verify predictions from Fock overlaps."""

from pathlib import Path
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ms_gate_fock_qpt_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / "results/noise_error_structure/noise_free_fock_generator_comparison.csv"
CURVE = ROOT / "results/noise_rate_nbar_sweep/noise_free_diagnostics/fock_resolved/fock_resolved_curve.csv"

with SOURCE.open() as handle:
    rows = [{key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)]
with CURVE.open() as handle:
    curve = [{key: float(value) for key, value in row.items()}
             for row in csv.DictReader(handle)]

n = np.array([row["fock_n"] for row in curve])
c = np.array([row["coherence_real"] + 1j * row["coherence_imag"] for row in curve])
theta = np.array([row["theta_xx_rad"] for row in curve])
for row in rows:
    mu = row["nbar"]
    q = mu / (1 + mu)
    weights = (1 - q) * q**n
    weights /= weights.sum()
    thermal_c = np.dot(weights, c)
    thermal_z = np.dot(weights, np.exp(2j * theta))
    predicted = [np.pi / 4 - np.angle(thermal_c) / 2,
                 -np.log(abs(thermal_z)) / 2,
                 -np.log(abs(thermal_c)) / 2 + np.log(abs(thermal_z)) / 2]
    saved = [row["fock_h_XX_rad_per_gate"],
             row["fock_phase_dispersion_gamma_per_gate"],
             row["fock_residual_motion_gamma_per_gate"]]
    np.testing.assert_allclose(predicted, saved, rtol=1e-9, atol=1e-14)

panels = [
    (r"Coherent $XX$ offset", "fock_h_XX_rad_per_gate", "qpt_H_XX_rad_per_gate",
     1e-2, r"$h_{XX}$ [$10^{-2}$ rad / gate]"),
    (r"Stochastic $XX$", "fock_phase_dispersion_gamma_per_gate", "qpt_S_XX_per_gate",
     1e-3, r"$\gamma_{XX}$ [$10^{-3}$ / gate]"),
    (r"Collective $X$", "fock_residual_motion_gamma_per_gate", "qpt_local_X_mean_per_gate",
     1e-4, r"$\gamma_{\mathrm{col}}$ [$10^{-4}$ / gate]"),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.labelcolor": "#26364a",
                     "text.color": "#26364a", "xtick.color": "#46586d",
                     "ytick.color": "#46586d", "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.7),
                         gridspec_kw={"height_ratios": [1.35, 1]}, sharex=True)
fig.subplots_adjust(top=.78, bottom=.15, left=.075, right=.98, hspace=.38, wspace=.30)
fig.suptitle("Fock-based prediction and QPT generator coefficients", x=.075,
             ha="left", y=.976, fontsize=17, fontweight="bold")
fig.text(.075, .923, "Same Hamiltonian and gate settings; all explicit dissipative rates set to zero.", fontsize=10.5)
x = np.array([row["nbar"] for row in rows])
long_rows = []
for k, (title, fkey, qkey, scale, ylabel) in enumerate(panels):
    f = np.array([row[fkey] for row in rows])
    ref = np.array([row[qkey] for row in rows])
    percent = 100 * np.abs(f - ref) / np.abs(ref)
    ax, err = axes[:, k]
    ax.plot(x, f / scale, "o-", color="#117a79", lw=1.8, ms=5,
            label="Fock prediction")
    ax.plot(x, ref / scale, "s--", color="#cf681c", lw=1.3, ms=8,
            markerfacecolor="none", markeredgewidth=1.3, label="QPT extraction")
    ax.set(title=title, ylabel=ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#dce3eb", lw=.65)
    err.plot(x, percent, "o-", color="#4b5d98", lw=1.5, ms=4)
    err.set_ylim(0, percent.max() * 1.32)
    err.set_ylabel("Relative difference [%]")
    err.set_xlabel(r"Initial mean phonon number $\bar n$")
    err.set_xticks(x, ["0.01", "1", "2", "3", "4"])
    err.set_xlim(-.18, 4.18)
    err.grid(axis="y", color="#dce3eb", lw=.65)
    err.text(.04, .87, f"Maximum: {percent.max():.5g}%", transform=err.transAxes,
             fontweight="bold", color="#4b5d98")
    for mu, fv, qv, ev in zip(x, f, ref, percent):
        long_rows.append({"nbar": mu, "coefficient": fkey,
                          "fock_prediction": fv, "qpt_reference": qv,
                          "absolute_relative_difference_percent": ev})

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.069, .896),
           ncol=2, frameon=False)
fig.text(.075, .063,
         r"Relative difference = $100\,|x_{\rm Fock}-x_{\rm QPT}|/|x_{\rm QPT}|$. Lines connect the five saved points.",
         fontsize=9)
fig.text(.075, .025,
         r"Collective-X reference: $(\gamma_{IX}+\gamma_{XI})/2\simeq c_{IX,XI}$, for the jump operator $IX+XI$.",
         fontsize=9)
for extension in ("png", "pdf"):
    fig.savefig(OUT / f"fock_qpt_comparison.{extension}", dpi=190, facecolor="white")
plt.close(fig)
with (OUT / "comparison_values.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
    writer.writeheader()
    writer.writerows(long_rows)
print("Verified all 15 Fock predictions against saved Fock overlaps.")
for row in long_rows:
    print(f"nbar={row['nbar']:g}  {row['coefficient']}: "
          f"Fock={row['fock_prediction']:.9g}, QPT={row['qpt_reference']:.9g}, "
          f"difference={row['absolute_relative_difference_percent']:.7g}%")
print(OUT / "fock_qpt_comparison.png")
