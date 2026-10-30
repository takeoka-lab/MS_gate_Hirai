"""Make the two-panel QI2026 abstract figure from reported manuscript values.

Source: main.pdf, pp. 4-5, Table III and Eqs. (40)-(41), supplied by the author.
Values are rounded as in the manuscript; no new simulation is performed.
Run with a Python environment containing matplotlib (3.8 or newer).
"""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "output" / "figures"
PDF_DIR = ROOT / "output" / "pdf"
STEM = "qi2026_calibration_nbar4"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

H_BEFORE, H_AFTER = 7.27e-2, 4.34e-4
CHI = [4.27e-4, -2.76e-3, 2.76e-3]
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#6B7280"

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.75,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

# Exact physical size: 170 x 68 mm, suitable for the template's 170-mm text width.
fig, (left, right) = plt.subplots(
    1, 2, figsize=(170 / 25.4, 68 / 25.4),
    gridspec_kw={"width_ratios": [1, 1.24]},
)
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.205, top=0.82, wspace=0.63)
fig.text(0.985, 0.967, r"$\bar{n}=4$", ha="right", va="top", fontsize=10)

for ax in (left, right):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=8)

left.set_title("(a)  Rotation-error calibration", loc="left", pad=12)
left.set_yscale("log")
left.set_ylim(1e-4, 2.1e-1)
left.set_xlim(-0.32, 1.40)
left.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
left.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=20))
left.yaxis.set_minor_formatter(NullFormatter())
left.grid(axis="y", which="major", linewidth=0.45, color="#DDE1E5")
left.set_ylabel(r"$|h_{XX}|$ (rad/gate)", labelpad=5)
left.set_xticks([0, 1], ["Before", "After"])
left.set_xlabel("Amplitude calibration", labelpad=7)

# Points avoid the arbitrary baseline of a bar chart on a logarithmic axis.
left.plot([0, 1], [H_BEFORE, H_AFTER], color="#9AA1A9", lw=1.1, ls=(0, (3, 3)), zorder=2)
left.scatter([0], [H_BEFORE], s=44, color=GRAY, edgecolor="white", linewidth=0.65, zorder=4)
left.scatter([1], [H_AFTER], s=44, color=BLUE, edgecolor="white", linewidth=0.65, zorder=4)
left.annotate(r"$7.27\times10^{-2}$", (0, H_BEFORE), xytext=(3, 8),
              textcoords="offset points", ha="left", va="bottom", fontsize=9)
left.annotate(r"$4.34\times10^{-4}$", (1, H_AFTER), xytext=(0, -11),
              textcoords="offset points", ha="center", va="top", fontsize=9, color=BLUE)
left.text(0.50, 1.8e-2, r"$\approx 167$-fold" + "\nreduction", ha="left", va="top",
          fontsize=9, linespacing=1.1, color="#252A31")
left.annotate("", xy=(1.23, H_AFTER), xytext=(1.23, H_BEFORE),
              arrowprops={"arrowstyle": "->", "color": "#555D66", "lw": 0.9,
                          "shrinkA": 0, "shrinkB": 0})

right.set_title("(b)  Residual non-Pauli components", loc="left", pad=12)
right.set_xlim(-0.58, 2.58)
right.set_ylim(-3.75, 3.75)
right.yaxis.set_major_locator(MultipleLocator(1))
right.grid(axis="y", linewidth=0.45, color="#E4E7EB")
right.axhline(0, color="#4B515A", lw=0.85, zorder=3)
right.set_ylabel(r"Process-matrix element ($\times10^{-3}$)", labelpad=5)
right.set_xticks([0, 1, 2], [r"$\mathrm{Im}\,\chi_{II,XX}$",
                            r"$\mathrm{Re}\,\chi_{II,XX}$",
                            r"$\mathrm{Re}\,\chi_{IX,XI}$"])
right.set_xlabel("After amplitude calibration", labelpad=7)
bars = right.bar([0, 1, 2], [x * 1e3 for x in CHI], width=0.53,
                 color=[BLUE, ORANGE, ORANGE], edgecolor=[BLUE, ORANGE, ORANGE],
                 linewidth=0.65, zorder=4)
for bar, label, value, color in zip(bars, ["+0.427", "-2.76", "+2.76"],
                                   [x * 1e3 for x in CHI], [BLUE, ORANGE, ORANGE]):
    right.annotate(label, (bar.get_x() + bar.get_width() / 2, value),
                   xytext=(0, 5 if value > 0 else -5), textcoords="offset points",
                   ha="center", va="bottom" if value > 0 else "top", fontsize=9, color=color)

fig.savefig(FIG_DIR / f"{STEM}.png", dpi=600)
fig.savefig(PDF_DIR / f"{STEM}.pdf", metadata={
    "Title": "MS gate amplitude calibration and residual non-Pauli components",
    "Subject": "Mean phonon number 4; reported numerical values from the supplied manuscript",
    "Keywords": "QI2026, Molmer-Sorensen, amplitude calibration, non-Pauli errors",
})

caption = (
    "FIG 1. Effect of drive-amplitude calibration at a mean phonon number of four. "
    "(a) Magnitude of the XX rotation-error coefficient before and after calibration "
    "(logarithmic vertical axis). (b) Selected off-diagonal elements of the calibrated "
    "process matrix. The rotation error is strongly suppressed, while real off-diagonal "
    "components consistent with collective X-type dissipation remain."
)
(FIG_DIR / f"{STEM}_caption.txt").write_text(caption + "\n")
with (FIG_DIR / f"{STEM}_data.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["panel", "quantity", "condition", "nbar", "value", "unit"])
    writer.writerows([
        ["a", "abs(h_XX)", "before", 4, H_BEFORE, "rad/gate"],
        ["a", "abs(h_XX)", "after", 4, H_AFTER, "rad/gate"],
        ["b", "Im(chi_II_XX)", "after", 4, CHI[0], "dimensionless"],
        ["b", "Re(chi_II_XX)", "after", 4, CHI[1], "dimensionless"],
        ["b", "Re(chi_IX_XI)", "after", 4, CHI[2], "dimensionless"],
    ])
print(json.dumps({
    "png": str(FIG_DIR / f"{STEM}.png"),
    "pdf": str(PDF_DIR / f"{STEM}.pdf"),
    "figure_mm": [170, 68],
    "ratio_from_rounded_values": H_BEFORE / H_AFTER,
    "ratio_annotation": "approximately 167, following the manuscript",
}, indent=2))
