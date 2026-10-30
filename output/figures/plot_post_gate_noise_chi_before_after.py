"""Reproduce the notebook's stacked before/after chi heatmaps from saved QPT.

Run from the project root:
    MPLCONFIGDIR=/tmp/ms_gate_mpl_cache python3 \
        output/figures/plot_post_gate_noise_chi_before_after.py --nbar 0.01

The source archive is the one used by angle_calibration_channel_comparison.ipynb.
Only the displayed II,II entry is zeroed; the stored process matrices are intact.
"""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ms_gate_functions as mg

OUTPUT_DIR = ROOT / "results" / "calibration_channel_comparison"
SOURCE = OUTPUT_DIR / "post_gate_noise_chi_before_after.npz"


def add_row(combined, source, source_axes, bottom_inches):
    """Keep the original helper's individual panel and colorbar placement."""
    source.canvas.draw()
    row_height = source.get_figheight()
    total_height = combined.get_figheight()

    def position(axis):
        x, y, width, height = axis.get_position().bounds
        return [x, (bottom_inches + y * row_height) / total_height,
                width, height * row_height / total_height]

    for original in source_axes:
        axis = combined.add_axes(position(original))
        original_image = original.images[0]
        im = axis.imshow(
            original_image.get_array(), origin=original_image.origin,
            cmap=original_image.get_cmap(),
            vmin=original_image.get_clim()[0], vmax=original_image.get_clim()[1],
        )
        axis.set_title(original.get_title(), fontsize=12)
        axis.set_xlabel(original.get_xlabel())
        axis.set_ylabel(original.get_ylabel())
        axis.set_xticks(original.get_xticks())
        axis.set_yticks(original.get_yticks())
        axis.set_xticklabels(
            [label.get_text() for label in original.get_xticklabels()],
            rotation=90, fontsize=9,
        )
        axis.set_yticklabels(
            [label.get_text() for label in original.get_yticklabels()], fontsize=9,
        )
        colorbar_axis = combined.add_axes(position(original_image.colorbar.ax))
        colorbar = combined.colorbar(im, cax=colorbar_axis)
        colorbar.ax.tick_params(labelsize=9)

    title = source._suptitle
    x, y = title.get_position()
    combined.text(
        x, (bottom_inches + y * row_height) / total_height, title.get_text(),
        ha=title.get_ha(), va=title.get_va(), fontsize=title.get_fontsize(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbar", type=float, default=0.01)
    args = parser.parse_args()

    with np.load(SOURCE, allow_pickle=False) as payload:
        matches = np.flatnonzero(np.isclose(payload["n_bar"], args.nbar, rtol=0, atol=1e-12))
        if len(matches) != 1:
            raise ValueError(f"No unique saved nbar={args.nbar}; available: {payload['n_bar']}")
        if str(payload["convention"].item()) != "N_post = S_noisy @ inverse(S_ideal)":
            raise ValueError("Unexpected channel convention")
        labels = payload["pauli_labels"].tolist()
        if labels != [p + q for p in "IXYZ" for q in "IXYZ"]:
            raise ValueError("Unexpected Pauli basis order")
        pairs = {condition: payload[f"chi_{condition}"][matches[0]].copy()
                 for condition in ("before", "after")}

    stem = str(float(args.nbar)).replace(".", "p")
    combined = plt.figure(figsize=(15.0, 10.5))
    for condition, label, bottom in [
        ("before", "Before angle calibration", 5.5),
        ("after", "After angle calibration", 0.0),
    ]:
        chi = pairs[condition]
        if chi.shape != (16, 16) or not np.isfinite(chi).all():
            raise ValueError("Expected a finite 16 x 16 process matrix")
        if not np.isclose(np.trace(chi), 1.0, rtol=0, atol=1e-10):
            raise ValueError("Expected trace-normalized chi")
        if not np.allclose(chi, chi.conj().T, rtol=0, atol=1e-10):
            raise ValueError("Expected Hermitian chi")
        ii = labels.index("II")
        identity_value = chi[ii, ii]
        shown = chi.copy()
        shown[ii, ii] = 0.0
        title = (
            rf"{label}: post-gate noise channel at $\bar n={args.nbar:g}$"
            + "\n"
            + r"$\mathcal{N}_{post}=\mathcal{S}_{noisy}\mathcal{S}_{ideal}^{-1}$, "
            + rf"display-only $\chi_{{II,II}}=0$ (original {identity_value.real:.8f})"
        )
        figure, axes = mg.plot_chi_matrix(
            shown, title=title, components=("real", "imag", "abs"),
            figsize=(15.0, 5.0), show_title=True, show_component_titles=True,
            show_axis_labels=True, tick_labelsize=9, colorbar_tick_labelsize=9,
        )
        add_row(combined, figure, axes, bottom)
        for suffix in ("png", "pdf"):
            path = OUTPUT_DIR / f"post_gate_noise_chi_{condition}_nbar_{stem}.{suffix}"
            figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        print(f"nbar={args.nbar:g} {condition}: chi[II,II]={identity_value.real:.8f}")

    for suffix in ("png", "pdf"):
        path = OUTPUT_DIR / f"post_gate_noise_chi_before_after_nbar_{stem}.{suffix}"
        combined.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(combined)


if __name__ == "__main__":
    main()
