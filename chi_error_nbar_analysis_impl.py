#!/usr/bin/env python
# coding: utf-8

# # χ行列の主要エラー要素 vs 平均フォノン数
# 
# 理想MSゲートを取り除いた post-gate error channel
# $\mathcal{N}_{\mathrm{post}}=\mathcal{E}_{\mathrm{actual}}\circ\mathcal{U}_{\mathrm{ideal}}^\dagger$ のχ行列を $\mathrm{Tr}(\chi)=1$ に規格化し、主要成分
# $\chi_{XX,XX}$ と $|\chi_{II,XX}|$ を $\bar n=0.01$ から20まで解析します。
# 
# - Pauli順序: `II, IX, IY, IZ, XI, XX, ... , ZZ`
# - `II,XX` は複素成分なので、比較用の主量には絶対値を使います。厳密再計算時は実部・虚部も保存します。
# - フィットは $\bar n=0.01$ のデータ点を必ず通る多項式です。既定は4次で、1〜5次の指標も比較します。
# - 式は **0.01〜20の範囲内の経験式** です。範囲外への外挿には使わないでください。
# 
# 初期状態では既存の81点QPT結果をキャッシュから読み、すぐに図と式を再生成します。`RECOMPUTE = True` にすると、各点を逐次再計算してCSVを安全に更新します。

# In[1]:


from pathlib import Path
import importlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

import ms_gate_functions as mg
mg = importlib.reload(mg)

_WORKFLOW_CONFIG = dict(globals().get("WORKFLOW_CONFIG") or {})


def _config(name, default):
    """Return a notebook-supplied override or the implementation default."""

    return _WORKFLOW_CONFIG.get(name, default)


OUTPUT_DIR = Path(_config("OUTPUT_DIR", "results/chi_error_element_fit"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUTPUT_DIR / "chi_error_element_sweep.csv"

NBAR_GRID = np.asarray(
    _config("NBAR_GRID", np.r_[0.01, np.arange(0.25, 20.0001, 0.25)]),
    dtype=float,
)
RECOMPUTE = bool(_config("RECOMPUTE", False))
PARALLEL_WORKERS = int(_config("PARALLEL_WORKERS", 4))
FIT_DEGREE = int(_config("FIT_DEGREE", 4))
ERROR_CHANNEL_CONVENTION = str(
    _config("ERROR_CHANNEL_CONVENTION", "undo_before_actual")
)

print(f"Sweep points: {len(NBAR_GRID)} ({NBAR_GRID[0]} ... {NBAR_GRID[-1]})")
print(f"Cache: {CACHE_PATH}")
print(f"RECOMPUTE = {RECOMPUTE}")


# ## シミュレーション設定
# 
# `main.ipynb` の現在の基準条件を明示的に固定しています。再計算は高い $\bar n$ ほどフォノン空間が大きくなり時間がかかるため、1点ごとにCSVへ保存します。中断後は同じセルを再実行すれば、`direct_simulation` になっている点を飛ばして続行できます。

# In[2]:


SIMULATION_PARAMS = {
    "A": 0.125,
    "delta": 0.5,
    "rho0": 0.0,
    "time_points": 500,
    "t_gate_phys": 100e-6,
    "heating_rate_phys": 10.0,
    "dephasing_rate_phys": 18.0,
    "T2_star": 0.3,
    "rayleigh_rate_phys": 3.0,
    "raman_rate_phys": 1.0,
    "eta": 0.1,
    "laser_intensity_fluctuation": 0.0,
    "laser_detuning_fluctuation": 0.0,
    "laser_rotation_angle_fluctuation": 0.0,
    "laser_noise_samples": 1,
    "laser_noise_seed": 1234,
    "use_full_order": True,
    "show_progress": True,
    "parallel_workers": PARALLEL_WORKERS,
}
SIMULATION_PARAMS.update(dict(_config("SIMULATION_PARAMS", {})))
pd.Series(SIMULATION_PARAMS, name="value").to_frame()


# In[3]:


PAULI_LABELS = [label for label, _ in mg.pauli_labels_and_weights()]
II_INDEX = PAULI_LABELS.index("II")
XX_INDEX = PAULI_LABELS.index("XX")

def _trace_normalized_error_chi(n_bar):
    point_params = dict(SIMULATION_PARAMS)
    point_params["n_bar_list"] = [float(n_bar)]
    channel_result = mg.generate_chi_matrices(**point_params)
    error_result = mg.generate_error_channel_matrices(
        channel_result=channel_result,
        convention=ERROR_CHANNEL_CONVENTION,
    )
    chi = np.asarray(error_result["error_chi_matrix_list"][0], dtype=complex)
    trace = np.trace(chi)
    if abs(trace) < 1e-15:
        raise ValueError(f"n_bar={n_bar}: chi trace is too close to zero")
    return chi / trace

def simulate_target_elements(n_bar):
    chi = _trace_normalized_error_chi(n_bar)
    xx_xx = chi[XX_INDEX, XX_INDEX]
    ii_xx = chi[II_INDEX, XX_INDEX]
    if abs(np.imag(xx_xx)) > 1e-10:
        warnings.warn(f"n_bar={n_bar}: chi[XX,XX] has Im={np.imag(xx_xx):.3e}")
    return {
        "n_bar": float(n_bar),
        "chi_XX_XX": float(np.real(xx_xx)),
        "chi_II_XX_real": float(np.real(ii_xx)),
        "chi_II_XX_imag": float(np.imag(ii_xx)),
        "chi_II_XX_abs": float(abs(ii_xx)),
        "data_source": "direct_simulation",
    }

def _atomic_save_csv(frame, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.sort_values("n_bar").to_csv(temporary_path, index=False)
    temporary_path.replace(path)

def load_or_recompute_sweep():
    if CACHE_PATH.exists():
        frame = pd.read_csv(CACHE_PATH)
    else:
        frame = pd.DataFrame(columns=[
            "n_bar", "chi_XX_XX", "chi_II_XX_real",
            "chi_II_XX_imag", "chi_II_XX_abs", "data_source",
        ])

    if RECOMPUTE:
        for point_index, n_bar in enumerate(NBAR_GRID, start=1):
            exact_mask = (
                np.isclose(frame.get("n_bar", pd.Series(dtype=float)), n_bar)
                & frame.get("data_source", pd.Series(dtype=str)).eq("direct_simulation")
            )
            if exact_mask.any():
                print(f"[{point_index:02d}/{len(NBAR_GRID)}] n_bar={n_bar:g}: cached exact result")
                continue
            print(f"[{point_index:02d}/{len(NBAR_GRID)}] n_bar={n_bar:g}: simulating")
            row = simulate_target_elements(n_bar)
            frame = frame.loc[~np.isclose(frame["n_bar"].astype(float), n_bar)]
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
            _atomic_save_csv(frame, CACHE_PATH)

    if frame.empty:
        raise FileNotFoundError(
            f"No cached data at {CACHE_PATH}. Set RECOMPUTE=True and rerun this cell."
        )

    frame = frame.sort_values("n_bar").reset_index(drop=True)
    missing = [
        float(n_bar) for n_bar in NBAR_GRID
        if not np.isclose(frame["n_bar"].astype(float), n_bar).any()
    ]
    if missing:
        raise ValueError(f"Sweep cache is missing n_bar points: {missing}")
    return frame

sweep_df = load_or_recompute_sweep()
print(sweep_df["data_source"].value_counts().to_string())
display(sweep_df.head())
display(sweep_df.tail())


# ### キャッシュ精度について
# 
# `historical_qpt_and_heatmap_recovery` のうち $\chi_{XX,XX}$ は既存QPTの数値CSV、$|\chi_{II,XX}|$ は同じQPTから生成済みのχヒートマップのカラーバーから復元した値です。そのため後者は図の解像度相当（概ね $10^{-4}$ 以下）の復元誤差を含みます。論文値など高精度が必要な場合は `RECOMPUTE=True` で直接値に置き換えてください。

# In[4]:


def anchored_polynomial_fit(x, y, degree):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    x0, y0 = float(x[0]), float(y[0])
    u = x - x0
    design = np.column_stack([u ** power for power in range(1, degree + 1)])
    coefficients = np.linalg.lstsq(design, y - y0, rcond=None)[0]

    def predict(x_new):
        shifted = np.asarray(x_new, dtype=float) - x0
        result = np.full_like(shifted, y0, dtype=float)
        for power, coefficient in enumerate(coefficients, start=1):
            result += coefficient * shifted ** power
        return result

    fitted = predict(x)
    residual = y - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    max_abs_residual = float(np.max(np.abs(residual)))

    terms = [f"{y0:.8e}"]
    for power, coefficient in enumerate(coefficients, start=1):
        sign = "+" if coefficient >= 0 else "-"
        terms.append(f" {sign} {abs(coefficient):.8e}(n-0.01)^{power}")
    equation = "y(n) = " + "".join(terms)
    return {
        "degree": int(degree), "x0": x0, "y0": y0,
        "coefficients": coefficients, "predict": predict,
        "fitted": fitted, "residual": residual,
        "r_squared": r_squared, "rmse": rmse,
        "max_abs_residual": max_abs_residual, "equation": equation,
    }

x_data = sweep_df["n_bar"].to_numpy(float)
FIT_TARGETS = {
    r"$\chi_{XX,XX}$": "chi_XX_XX",
    r"$|\chi_{II,XX}|$": "chi_II_XX_abs",
}

diagnostic_rows = []
for label, column in FIT_TARGETS.items():
    for degree in range(1, 6):
        fit = anchored_polynomial_fit(x_data, sweep_df[column], degree)
        diagnostic_rows.append({
            "element": label.replace("$", ""),
            "degree": degree,
            "R_squared": fit["r_squared"],
            "RMSE": fit["rmse"],
            "max_abs_residual": fit["max_abs_residual"],
        })
fit_diagnostics_df = pd.DataFrame(diagnostic_rows)
display(fit_diagnostics_df)


# In[5]:


selected_fits = {
    column: anchored_polynomial_fit(x_data, sweep_df[column], FIT_DEGREE)
    for column in FIT_TARGETS.values()
}

summary_rows = []
for label, column in FIT_TARGETS.items():
    fit = selected_fits[column]
    row = {
        "element": label.replace("$", ""),
        "degree": fit["degree"],
        "n_bar_anchor": fit["x0"],
        "value_at_anchor": fit["y0"],
        "R_squared": fit["r_squared"],
        "RMSE": fit["rmse"],
        "max_abs_residual": fit["max_abs_residual"],
        "equation": fit["equation"],
    }
    for power, coefficient in enumerate(fit["coefficients"], start=1):
        row[f"a{power}"] = coefficient
    summary_rows.append(row)

fit_summary_df = pd.DataFrame(summary_rows)
fit_summary_path = OUTPUT_DIR / "chi_error_element_fit_summary.csv"
fit_summary_df.to_csv(fit_summary_path, index=False)
display(fit_summary_df)

for _, row in fit_summary_df.iterrows():
    display(Markdown(
        f"**{row['element']}**: `{row['equation']}`<br>"
        f"$R^2={row['R_squared']:.8f}$, RMSE = {row['RMSE']:.3e}"
    ))
print(f"Saved: {fit_summary_path}")


# In[6]:


x_dense = np.linspace(NBAR_GRID.min(), NBAR_GRID.max(), 1000)
colors = {"chi_XX_XX": "#0072B2", "chi_II_XX_abs": "#D55E00"}
titles = {
    "chi_XX_XX": r"Pauli $XX$ error weight $\chi_{XX,XX}$",
    "chi_II_XX_abs": r"Coherent $XX$ error indicator $|\chi_{II,XX}|$",
}
y_labels = {
    "chi_XX_XX": r"$XX$ error weight $\chi_{XX,XX}$",
    "chi_II_XX_abs": r"Coherent indicator $|\chi_{II,XX}|$",
}

def equation_text(fit):
    lines = [r"$x=\bar n-0.01$", rf"$y={fit['y0']:.4e}$"]
    for power, coefficient in enumerate(fit["coefficients"], start=1):
        sign = "+" if coefficient >= 0 else "-"
        exponent = "" if power == 1 else f"^{{{power}}}"
        lines.append(rf"${sign}\ {abs(coefficient):.4e}x{exponent}$")
    lines.append(rf"$R^2={fit['r_squared']:.8f}$")
    lines.append(rf"$\mathrm{{RMSE}}={fit['rmse']:.2e}$")
    return chr(10).join(lines)

fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))
for column_index, column in enumerate(FIT_TARGETS.values()):
    fit = selected_fits[column]
    color = colors[column]
    y_data = sweep_df[column].to_numpy(float)
    ax = axes[column_index]
    ax.scatter(
        x_data, y_data, s=22, color=color, alpha=0.75, label="QPT sweep", zorder=3
    )
    ax.plot(
        x_dense, fit["predict"](x_dense), color="black", linewidth=2.2,
        label=f"anchored degree-{FIT_DEGREE} fit",
    )
    ax.set_title(titles[column], fontsize=14)
    ax.set_xlabel(r"Mean phonon number $\bar n$")
    ax.set_ylabel(y_labels[column])
    ax.legend(frameon=True, loc="lower right")
    ax.text(
        0.03, 0.97, equation_text(fit),
        transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.92},
    )

for ax in axes.ravel():
    ax.grid(True, alpha=0.28)
    ax.set_xlim(NBAR_GRID.min() - 0.2, NBAR_GRID.max() + 0.2)

fig.suptitle(
    r"Major post-gate error χ elements vs $\bar n$ ($\mathrm{Tr}(\chi)=1$)",
    fontsize=16,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
figure_png = OUTPUT_DIR / "chi_error_elements_vs_nbar_fit.png"
figure_pdf = OUTPUT_DIR / "chi_error_elements_vs_nbar_fit.pdf"
fig.savefig(figure_png, dpi=300, bbox_inches="tight")
fig.savefig(figure_pdf, bbox_inches="tight")
plt.show()

prediction_df = pd.DataFrame({"n_bar": x_dense})
for column, fit in selected_fits.items():
    prediction_df[f"{column}_fit"] = fit["predict"](x_dense)
prediction_path = OUTPUT_DIR / "chi_error_element_fit_curves.csv"
prediction_df.to_csv(prediction_path, index=False)
print(f"Saved: {figure_png}")
print(f"Saved: {figure_pdf}")
print(f"Saved: {prediction_path}")


# In[7]:


# RECOMPUTE=True で直接計算した場合のみ、II,XX の複素成分も確認する。
complex_columns = ["chi_II_XX_real", "chi_II_XX_imag"]
if sweep_df[complex_columns].notna().all().all():
    fig_complex, ax_complex = plt.subplots(figsize=(7.5, 4.7))
    ax_complex.plot(sweep_df["n_bar"], sweep_df["chi_II_XX_real"], label=r"$\Re\chi_{II,XX}$")
    ax_complex.plot(sweep_df["n_bar"], sweep_df["chi_II_XX_imag"], label=r"$\Im\chi_{II,XX}$")
    ax_complex.plot(sweep_df["n_bar"], sweep_df["chi_II_XX_abs"], label=r"$|\chi_{II,XX}|$", linewidth=2.2)
    ax_complex.set_xlabel(r"Mean phonon number $\bar n$")
    ax_complex.set_ylabel("Trace-normalized χ element")
    ax_complex.grid(True, alpha=0.3)
    ax_complex.legend()
    fig_complex.tight_layout()
    complex_path = OUTPUT_DIR / "chi_II_XX_complex_components.png"
    fig_complex.savefig(complex_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {complex_path}")
else:
    print("Complex II,XX components are not in the recovered cache. Set RECOMPUTE=True for direct values.")


# ## その他の主要なχ対角ノイズ
# 
# 既存81点QPTの全Pauli対角成分を順位付けします。`II`（無誤差）と上で解析済みの `XX` を除き、区間内最大値が $\chi_{XX,XX}$ の最大値の1%以上となる成分を「その他の主要ノイズ」とします。選ばれた各成分に、上と同じ端点固定4次最小二乗フィットを適用します。

# In[8]:


PAULI_DIAGONAL_CACHE_PATH = Path("pauli_twirled_error_probabilities.zip")
MAJOR_OTHER_RELATIVE_THRESHOLD = 0.01

pauli_long_df = pd.read_csv(PAULI_DIAGONAL_CACHE_PATH, compression="zip")
required_pauli_columns = {"n_bar", "pauli", "probability"}
if not required_pauli_columns.issubset(pauli_long_df.columns):
    raise ValueError(f"Pauli cache must contain {sorted(required_pauli_columns)}")
pauli_wide_df = (
    pauli_long_df.pivot(index="n_bar", columns="pauli", values="probability")
    .sort_index()
)
missing_pauli_points = [
    float(n_bar) for n_bar in NBAR_GRID
    if not np.isclose(pauli_wide_df.index.to_numpy(float), n_bar).any()
]
if missing_pauli_points:
    raise ValueError(f"Pauli cache is missing n_bar points: {missing_pauli_points}")

xx_peak = float(pauli_wide_df["XX"].max())
ranking_rows = []
for pauli in PAULI_LABELS:
    values = pauli_wide_df[pauli].to_numpy(float)
    ranking_rows.append({
        "pauli": pauli,
        "max_over_sweep": float(np.max(values)),
        "value_at_nbar_0p01": float(values[0]),
        "value_at_nbar_20": float(values[-1]),
        "relative_to_XX_peak": float(np.max(values) / xx_peak),
    })
pauli_ranking_df = (
    pd.DataFrame(ranking_rows)
    .sort_values("max_over_sweep", ascending=False)
    .reset_index(drop=True)
)
other_ranking_df = pauli_ranking_df[~pauli_ranking_df["pauli"].isin(["II", "XX"])].copy()
OTHER_MAJOR_PAULIS = other_ranking_df.loc[
    other_ranking_df["relative_to_XX_peak"] >= MAJOR_OTHER_RELATIVE_THRESHOLD,
    "pauli",
].tolist()
if not OTHER_MAJOR_PAULIS:
    OTHER_MAJOR_PAULIS = other_ranking_df.head(2)["pauli"].tolist()

ranking_path = OUTPUT_DIR / "chi_pauli_diagonal_noise_ranking.csv"
pauli_ranking_df.to_csv(ranking_path, index=False)
print("Selected other major Pauli noises:", OTHER_MAJOR_PAULIS)
print(f"Selection threshold: {100 * MAJOR_OTHER_RELATIVE_THRESHOLD:.1f}% of max chi[XX,XX]")
display(other_ranking_df.head(12))
print(f"Saved: {ranking_path}")


# In[9]:


other_x_data = pauli_wide_df.index.to_numpy(float)
other_noise_fits = {
    pauli: anchored_polynomial_fit(other_x_data, pauli_wide_df[pauli], FIT_DEGREE)
    for pauli in OTHER_MAJOR_PAULIS
}

other_summary_rows = []
for pauli, fit in other_noise_fits.items():
    row = {
        "element": f"chi_{{{pauli},{pauli}}}",
        "pauli": pauli,
        "degree": fit["degree"],
        "n_bar_anchor": fit["x0"],
        "value_at_anchor": fit["y0"],
        "R_squared": fit["r_squared"],
        "RMSE": fit["rmse"],
        "max_abs_residual": fit["max_abs_residual"],
        "equation": fit["equation"],
    }
    for power, coefficient in enumerate(fit["coefficients"], start=1):
        row[f"a{power}"] = coefficient
    other_summary_rows.append(row)
other_fit_summary_df = pd.DataFrame(other_summary_rows)
other_summary_path = OUTPUT_DIR / "chi_other_major_noise_fit_summary.csv"
other_fit_summary_df.to_csv(other_summary_path, index=False)
display(other_fit_summary_df)

n_other = len(OTHER_MAJOR_PAULIS)
fig_other, axes_other = plt.subplots(1, n_other, figsize=(7.0 * n_other, 5.8), squeeze=False)
axes_other = axes_other.ravel()
other_colors = plt.get_cmap("tab10").colors
for panel_index, pauli in enumerate(OTHER_MAJOR_PAULIS):
    ax = axes_other[panel_index]
    fit = other_noise_fits[pauli]
    y_values = pauli_wide_df[pauli].to_numpy(float)
    color = other_colors[panel_index % len(other_colors)]
    ax.scatter(other_x_data, y_values, s=22, color=color, alpha=0.75, label="QPT sweep", zorder=3)
    ax.plot(x_dense, fit["predict"](x_dense), color="black", linewidth=2.2, label=f"anchored degree-{FIT_DEGREE} fit")
    ax.set_title(rf"Pauli error weight $\chi_{{{pauli},{pauli}}}$", fontsize=14)
    ax.set_xlabel(r"Mean phonon number $\bar n$")
    ax.set_ylabel(rf"${pauli}$ error weight $\chi_{{{pauli},{pauli}}}$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.set_xlim(NBAR_GRID.min() - 0.2, NBAR_GRID.max() + 0.2)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True, loc="lower right")
    ax.text(
        0.03, 0.97, equation_text(fit),
        transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.92},
    )

fig_other.suptitle(
    r"Other major post-gate Pauli error weights vs $\bar n$ ($\mathrm{Tr}(\chi)=1$)",
    fontsize=16,
)
fig_other.tight_layout(rect=(0, 0, 1, 0.94))
other_figure_png = OUTPUT_DIR / "chi_other_major_noise_vs_nbar_fit.png"
other_figure_pdf = OUTPUT_DIR / "chi_other_major_noise_vs_nbar_fit.pdf"
fig_other.savefig(other_figure_png, dpi=300, bbox_inches="tight")
fig_other.savefig(other_figure_pdf, bbox_inches="tight")
plt.show()

other_prediction_df = pd.DataFrame({"n_bar": x_dense})
for pauli, fit in other_noise_fits.items():
    other_prediction_df[f"chi_{pauli}_{pauli}_fit"] = fit["predict"](x_dense)
other_prediction_path = OUTPUT_DIR / "chi_other_major_noise_fit_curves.csv"
other_prediction_df.to_csv(other_prediction_path, index=False)
print(f"Saved: {other_summary_path}")
print(f"Saved: {other_figure_png}")
print(f"Saved: {other_figure_pdf}")
print(f"Saved: {other_prediction_path}")


# ### その他の主要ノイズの読み方
# 
# この基準では `IX` と `XI` が選ばれます。$\chi_{IX,IX}$ は第2イオン、$\chi_{XI,XI}$ は第1イオンに作用する単一量子ビット $X$ エラーの重みです。2本がほぼ一致することは、2イオンにほぼ対称なノイズが作用していることを示します。次点以下の対角成分は、区間内最大でも $\chi_{XX,XX}$ 最大値の約0.16%以下です。

# ## 論文化に向けた最低限の検証コード
# 
# 以下は、(1) 生の複素χ行列の直接保存、(2) 数値収束試験、(3) ノイズ源アブレーション、(4) 全256成分の順位付け、(5) モデル比較・交差検証・信頼区間、(6) 再現性情報の保存を行います。重いQPT計算は誤実行を防ぐため既定で無効です。スイッチを1つずつ `True` にし、上から順に実行してください。

# In[10]:


import json
import platform
import subprocess
from contextlib import contextmanager

import matplotlib
import qutip as qp
import scipy

RUN_EXACT_FULL_CHI_SWEEP = bool(_config("RUN_EXACT_FULL_CHI_SWEEP", False))
RUN_NUMERICAL_CONVERGENCE = bool(_config("RUN_NUMERICAL_CONVERGENCE", False))
RUN_NOISE_SOURCE_ABLATION = bool(_config("RUN_NOISE_SOURCE_ABLATION", False))
FORCE_RECOMPUTE_PUBLICATION_CACHE = bool(
    _config("FORCE_RECOMPUTE_PUBLICATION_CACHE", False)
)

EXACT_FULL_CHI_DIR = OUTPUT_DIR / "exact_full_chi"
CONVERGENCE_DIR = OUTPUT_DIR / "numerical_convergence"
ABLATION_DIR = OUTPUT_DIR / "noise_source_ablation"
for directory in (EXACT_FULL_CHI_DIR, CONVERGENCE_DIR, ABLATION_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONVERGENCE_NBARS = list(_config("CONVERGENCE_NBARS", [0.01, 4.0, 20.0]))
ABLATION_NBARS = list(
    _config("ABLATION_NBARS", [0.01, 1.0, 4.0, 10.0, 20.0])
)
NOISE_SOURCES = [
    "motional_heating", "motional_dephasing",
    "spin_dephasing", "photon_scattering",
]
FULL_CHI_TOP_K = int(_config("FULL_CHI_TOP_K", 12))
BOOTSTRAP_SAMPLES = int(_config("BOOTSTRAP_SAMPLES", 1000))
BOOTSTRAP_SEED = int(_config("BOOTSTRAP_SEED", 20260805))

print("Heavy publication checks:")
print(" exact full chi sweep =", RUN_EXACT_FULL_CHI_SWEEP)
print(" numerical convergence =", RUN_NUMERICAL_CONVERGENCE)
print(" noise-source ablation =", RUN_NOISE_SOURCE_ABLATION)


# ### 1. 81点の生の複素χ行列を直接保存
# 
# `RUN_EXACT_FULL_CHI_SWEEP=True` で実行します。各 $\bar n$ の16×16複素χ行列を画像ではなくNPZへ直接保存し、同時にCP・TP、Hermiticity、規約、フォノン打ち切り次元を記録します。既に保存済みの点はスキップされます。

# In[11]:


def _safe_nbar_stem(n_bar):
    return str(float(n_bar)).replace("-", "m").replace(".", "p")

def calculate_error_channel_point(n_bar, parameter_overrides=None):
    params = dict(SIMULATION_PARAMS)
    if parameter_overrides:
        params.update(parameter_overrides)
    params["n_bar_list"] = [float(n_bar)]
    error_result = mg.generate_error_channel_matrices(
        convention=ERROR_CHANNEL_CONVENTION,
        **params,
    )
    chi_raw = np.asarray(error_result["error_chi_matrix_list"][0], dtype=complex)
    raw_trace = np.trace(chi_raw)
    if abs(raw_trace) < 1e-15:
        raise ValueError(f"n_bar={n_bar}: raw chi trace is too close to zero")
    chi = chi_raw / raw_trace
    physicality = mg.choi_physicality_metrics(error_result["S_error_qobj_list"][0])
    composition = mg.validate_error_channel_composition(
        error_result, desired_convention=ERROR_CHANNEL_CONVENTION
    )
    metadata = {
        "n_bar": float(n_bar),
        "phonon_dim": int(error_result["results_list"][0]["Nv"]),
        "raw_trace_real": float(np.real(raw_trace)),
        "raw_trace_imag": float(np.imag(raw_trace)),
        "trace_normalized_hermiticity_fro": float(np.linalg.norm(chi - chi.conj().T)),
        "convention_error_fro": float(composition["max_desired_convention_error"]),
        **physicality,
    }
    return chi_raw, chi, metadata

def save_exact_full_chi_point(n_bar, chi_raw, chi, metadata):
    path = EXACT_FULL_CHI_DIR / f"error_chi_nbar_{_safe_nbar_stem(n_bar)}.npz"
    temporary_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        n_bar=float(n_bar),
        chi_raw=chi_raw,
        chi_trace_normalized=chi,
        pauli_labels=np.asarray(PAULI_LABELS),
        metadata_json=json.dumps(metadata, default=str),
    )
    temporary_path.replace(path)
    return path

def update_target_cache_from_exact_chi(n_bar, chi):
    frame = pd.read_csv(CACHE_PATH) if CACHE_PATH.exists() else pd.DataFrame()
    ii_xx = chi[II_INDEX, XX_INDEX]
    row = {
        "n_bar": float(n_bar),
        "chi_XX_XX": float(np.real(chi[XX_INDEX, XX_INDEX])),
        "chi_II_XX_real": float(np.real(ii_xx)),
        "chi_II_XX_imag": float(np.imag(ii_xx)),
        "chi_II_XX_abs": float(abs(ii_xx)),
        "data_source": "direct_simulation",
    }
    if not frame.empty:
        frame = frame.loc[~np.isclose(frame["n_bar"].astype(float), float(n_bar))]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    _atomic_save_csv(frame, CACHE_PATH)

def load_exact_full_chi_points():
    points = []
    for path in EXACT_FULL_CHI_DIR.glob("error_chi_nbar_*.npz"):
        with np.load(path, allow_pickle=False) as data:
            points.append({
                "n_bar": float(data["n_bar"]),
                "chi": np.asarray(data["chi_trace_normalized"], dtype=complex),
                "path": path,
                "metadata": json.loads(str(data["metadata_json"])),
            })
    return sorted(points, key=lambda point: point["n_bar"])

exact_index_rows = []
if RUN_EXACT_FULL_CHI_SWEEP:
    for point_index, n_bar in enumerate(NBAR_GRID, start=1):
        output_path = EXACT_FULL_CHI_DIR / f"error_chi_nbar_{_safe_nbar_stem(n_bar)}.npz"
        if output_path.exists() and not FORCE_RECOMPUTE_PUBLICATION_CACHE:
            print(f"[{point_index:02d}/{len(NBAR_GRID)}] n_bar={n_bar:g}: cached")
            continue
        print(f"[{point_index:02d}/{len(NBAR_GRID)}] n_bar={n_bar:g}: exact QPT")
        chi_raw, chi, metadata = calculate_error_channel_point(n_bar)
        saved_path = save_exact_full_chi_point(n_bar, chi_raw, chi, metadata)
        update_target_cache_from_exact_chi(n_bar, chi)
        exact_index_rows.append({**metadata, "path": str(saved_path)})
        pd.DataFrame(exact_index_rows).to_csv(EXACT_FULL_CHI_DIR / "latest_run_index.csv", index=False)

exact_full_chi_points = load_exact_full_chi_points()
print(f"Exact full-chi cache: {len(exact_full_chi_points)}/{len(NBAR_GRID)} points")
if exact_full_chi_points:
    exact_metadata_df = pd.DataFrame([point["metadata"] for point in exact_full_chi_points])
    exact_metadata_df.to_csv(EXACT_FULL_CHI_DIR / "exact_full_chi_validation.csv", index=False)
    display(exact_metadata_df)


# ### 2. 数値収束性（phonon cutoff・時間刻み・solver tolerance）
# 
# 論文では、観測したχ成分が数値設定を厳しくしても変わらないことを示します。代表点 $\bar n=0.01,4,20$ で、基準設定に対するχ行列全体のFrobenius差と主要4成分の差を比較します。
# 
# `RUN_NUMERICAL_CONVERGENCE=True` にして実行すると、各条件を独立したNPZへ保存します。途中で停止しても、保存済み条件は再利用されます。

# In[12]:


CONVERGENCE_CASES = {
    "baseline": {"phonon_scale": 1.0, "time_points": 500, "atol": 1e-12, "rtol": 1e-9},
    "phonon_cutoff_1p2x": {"phonon_scale": 1.2, "time_points": 500, "atol": 1e-12, "rtol": 1e-9},
    "time_grid_750": {"phonon_scale": 1.0, "time_points": 750, "atol": 1e-12, "rtol": 1e-9},
    "strict_solver_tol": {"phonon_scale": 1.0, "time_points": 500, "atol": 1e-13, "rtol": 1e-10},
}

@contextmanager
def temporary_numerical_controls(forced_phonon_dim, atol, rtol):
    original_estimate_phonon_dim = mg.estimate_phonon_dim
    original_solver_options = mg._solver_options

    def fixed_phonon_dim(n_bar, alpha_max):
        return int(forced_phonon_dim)

    def controlled_solver_options(store_states=True, max_step=None):
        options = {"progress_bar": None, "atol": float(atol), "rtol": float(rtol)}
        if max_step is not None:
            if max_step <= 0:
                raise ValueError("max_step must be positive when provided")
            options["max_step"] = float(max_step)
        if not store_states:
            options.update({"store_final_state": True, "store_states": False})
        return options

    mg.estimate_phonon_dim = fixed_phonon_dim
    mg._solver_options = controlled_solver_options
    try:
        yield
    finally:
        mg.estimate_phonon_dim = original_estimate_phonon_dim
        mg._solver_options = original_solver_options

def baseline_phonon_dim(n_bar):
    alpha_max = mg._estimate_alpha_max(
        SIMULATION_PARAMS["A"],
        SIMULATION_PARAMS["delta"],
        SIMULATION_PARAMS["laser_intensity_fluctuation"],
        SIMULATION_PARAMS["laser_detuning_fluctuation"],
    )
    return int(mg.estimate_phonon_dim(float(n_bar), alpha_max))

def convergence_cache_path(case_name, n_bar):
    return CONVERGENCE_DIR / f"{case_name}__nbar_{_safe_nbar_stem(n_bar)}.npz"

def run_convergence_point(case_name, case, n_bar):
    base_dim = baseline_phonon_dim(n_bar)
    forced_dim = max(2, int(np.ceil(base_dim * case["phonon_scale"])))
    overrides = {
        "time_points": int(case["time_points"]),
        "parallel_workers": 1,
        "show_progress": False,
    }
    with temporary_numerical_controls(forced_dim, case["atol"], case["rtol"]):
        chi_raw, chi, metadata = calculate_error_channel_point(n_bar, overrides)
    metadata.update({
        "case": case_name,
        "baseline_phonon_dim": base_dim,
        "forced_phonon_dim": forced_dim,
        "time_points": int(case["time_points"]),
        "solver_atol": float(case["atol"]),
        "solver_rtol": float(case["rtol"]),
    })
    path = convergence_cache_path(case_name, n_bar)
    temporary_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        n_bar=float(n_bar),
        case=case_name,
        chi_trace_normalized=chi,
        metadata_json=json.dumps(metadata, default=str),
    )
    temporary_path.replace(path)
    return path

if RUN_NUMERICAL_CONVERGENCE:
    total_jobs = len(CONVERGENCE_NBARS) * len(CONVERGENCE_CASES)
    job_index = 0
    for n_bar in CONVERGENCE_NBARS:
        for case_name, case in CONVERGENCE_CASES.items():
            job_index += 1
            path = convergence_cache_path(case_name, n_bar)
            if path.exists() and not FORCE_RECOMPUTE_PUBLICATION_CACHE:
                print(f"[{job_index:02d}/{total_jobs}] {case_name}, n_bar={n_bar:g}: cached")
                continue
            print(f"[{job_index:02d}/{total_jobs}] {case_name}, n_bar={n_bar:g}: simulating")
            run_convergence_point(case_name, case, n_bar)

convergence_records = []
for n_bar in CONVERGENCE_NBARS:
    loaded = {}
    for case_name in CONVERGENCE_CASES:
        path = convergence_cache_path(case_name, n_bar)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                loaded[case_name] = {
                    "chi": np.asarray(data["chi_trace_normalized"], dtype=complex),
                    "metadata": json.loads(str(data["metadata_json"])),
                }
    if "baseline" not in loaded:
        continue
    baseline_chi = loaded["baseline"]["chi"]
    for case_name, result in loaded.items():
        chi = result["chi"]
        metadata = result["metadata"]
        convergence_records.append({
            "n_bar": float(n_bar),
            "case": case_name,
            "phonon_dim": metadata["forced_phonon_dim"],
            "time_points": metadata["time_points"],
            "solver_atol": metadata["solver_atol"],
            "solver_rtol": metadata["solver_rtol"],
            "chi_frobenius_difference_from_baseline": float(np.linalg.norm(chi - baseline_chi)),
            "delta_chi_XX_XX": float(np.real(chi[XX_INDEX, XX_INDEX] - baseline_chi[XX_INDEX, XX_INDEX])),
            "delta_abs_chi_II_XX": float(abs(chi[II_INDEX, XX_INDEX]) - abs(baseline_chi[II_INDEX, XX_INDEX])),
            "delta_chi_IX_IX": float(np.real(chi[PAULI_LABELS.index("IX"), PAULI_LABELS.index("IX")] - baseline_chi[PAULI_LABELS.index("IX"), PAULI_LABELS.index("IX")])),
            "delta_chi_XI_XI": float(np.real(chi[PAULI_LABELS.index("XI"), PAULI_LABELS.index("XI")] - baseline_chi[PAULI_LABELS.index("XI"), PAULI_LABELS.index("XI")])),
        })

convergence_df = pd.DataFrame(convergence_records)
if convergence_df.empty:
    print("Numerical-convergence cache is empty. Set RUN_NUMERICAL_CONVERGENCE=True to calculate it.")
else:
    convergence_csv = CONVERGENCE_DIR / "numerical_convergence_summary.csv"
    convergence_df.to_csv(convergence_csv, index=False)
    display(convergence_df)

    plot_df = convergence_df[convergence_df["case"] != "baseline"]
    fig_conv, ax_conv = plt.subplots(figsize=(8.2, 5.2))
    for case_name, group in plot_df.groupby("case"):
        ax_conv.semilogy(
            group["n_bar"],
            np.maximum(group["chi_frobenius_difference_from_baseline"], 1e-18),
            marker="o",
            linewidth=2,
            label=case_name,
        )
    ax_conv.set_xlabel(r"Mean phonon number $\bar n$")
    ax_conv.set_ylabel(r"$\|\chi_{\rm test}-\chi_{\rm baseline}\|_F$")
    ax_conv.set_title("Numerical convergence of the full error χ matrix")
    ax_conv.grid(True, which="both", alpha=0.28)
    ax_conv.legend()
    fig_conv.tight_layout()
    conv_png = CONVERGENCE_DIR / "numerical_convergence.png"
    conv_pdf = CONVERGENCE_DIR / "numerical_convergence.pdf"
    fig_conv.savefig(conv_png, dpi=300, bbox_inches="tight")
    fig_conv.savefig(conv_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved: {convergence_csv}")


# ### 3. ノイズ源ablation（原因の切り分け）
# 
# `all`（全ノイズ）、`none`（独立ノイズなし）、`only`（そのノイズだけ）、`without`（そのノイズだけ除去）を比較します。非線形な相互作用があるため、寄与を単純に足し算とは解釈せず、`all - without` と `only - none` の両方を示します。
# 
# `RUN_NOISE_SOURCE_ABLATION=True` で代表5点を計算し、各条件の複素χ行列をNPZへ保存します。

# In[13]:


def parameters_with_all_independent_noise_zero(base_parameters):
    params = dict(base_parameters)
    params.update({
        "heating_rate_phys": 0.0,
        "dephasing_rate_phys": 0.0,
        "T2_star": np.inf,
        "rayleigh_rate_phys": 0.0,
        "raman_rate_phys": 0.0,
        "laser_intensity_fluctuation": 0.0,
        "laser_detuning_fluctuation": 0.0,
        "laser_rotation_angle_fluctuation": 0.0,
    })
    return params

def parameters_without_noise_source(base_parameters, noise_source):
    params = dict(base_parameters)
    if noise_source == "motional_heating":
        params["heating_rate_phys"] = 0.0
    elif noise_source == "motional_dephasing":
        params["dephasing_rate_phys"] = 0.0
    elif noise_source == "spin_dephasing":
        params["T2_star"] = np.inf
    elif noise_source == "photon_scattering":
        params["rayleigh_rate_phys"] = 0.0
        params["raman_rate_phys"] = 0.0
    else:
        raise ValueError(f"Unsupported ablation source: {noise_source}")
    return params

def build_ablation_conditions():
    nominal_strengths = mg.nominal_noise_source_strengths(SIMULATION_PARAMS)
    conditions = {
        "all": dict(SIMULATION_PARAMS),
        "none": parameters_with_all_independent_noise_zero(SIMULATION_PARAMS),
    }
    for source in NOISE_SOURCES:
        conditions[f"only__{source}"] = mg.simulation_parameters_with_single_noise_source(
            base_parameters=SIMULATION_PARAMS,
            noise_source=source,
            strength=nominal_strengths[source],
        )
        conditions[f"without__{source}"] = parameters_without_noise_source(
            SIMULATION_PARAMS, source
        )
    return conditions

def ablation_cache_path(condition_name, n_bar):
    return ABLATION_DIR / f"{condition_name}__nbar_{_safe_nbar_stem(n_bar)}.npz"

ABLATION_CONDITIONS = build_ablation_conditions()
if RUN_NOISE_SOURCE_ABLATION:
    total_jobs = len(ABLATION_NBARS) * len(ABLATION_CONDITIONS)
    job_index = 0
    for n_bar in ABLATION_NBARS:
        for condition_name, condition_params in ABLATION_CONDITIONS.items():
            job_index += 1
            path = ablation_cache_path(condition_name, n_bar)
            if path.exists() and not FORCE_RECOMPUTE_PUBLICATION_CACHE:
                print(f"[{job_index:02d}/{total_jobs}] {condition_name}, n_bar={n_bar:g}: cached")
                continue
            print(f"[{job_index:02d}/{total_jobs}] {condition_name}, n_bar={n_bar:g}: simulating")
            overrides = dict(condition_params)
            overrides.update({"parallel_workers": 1, "show_progress": False})
            _, chi, metadata = calculate_error_channel_point(n_bar, overrides)
            metadata.update({"condition": condition_name})
            temporary_path = path.with_name(path.stem + ".tmp.npz")
            np.savez_compressed(
                temporary_path,
                n_bar=float(n_bar),
                condition=condition_name,
                chi_trace_normalized=chi,
                metadata_json=json.dumps(metadata, default=str),
            )
            temporary_path.replace(path)

def ablation_observables(chi):
    ix = PAULI_LABELS.index("IX")
    xi = PAULI_LABELS.index("XI")
    return {
        "chi_XX_XX": float(np.real(chi[XX_INDEX, XX_INDEX])),
        "abs_chi_II_XX": float(abs(chi[II_INDEX, XX_INDEX])),
        "chi_IX_IX": float(np.real(chi[ix, ix])),
        "chi_XI_XI": float(np.real(chi[xi, xi])),
    }

ablation_raw_rows = []
ablation_values = {}
for n_bar in ABLATION_NBARS:
    for condition_name in ABLATION_CONDITIONS:
        path = ablation_cache_path(condition_name, n_bar)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            chi = np.asarray(data["chi_trace_normalized"], dtype=complex)
        values = ablation_observables(chi)
        ablation_values[(float(n_bar), condition_name)] = values
        ablation_raw_rows.append({
            "n_bar": float(n_bar), "condition": condition_name, **values
        })

ablation_raw_df = pd.DataFrame(ablation_raw_rows)
ablation_attribution_rows = []
for n_bar in ABLATION_NBARS:
    all_values = ablation_values.get((float(n_bar), "all"))
    none_values = ablation_values.get((float(n_bar), "none"))
    if all_values is None or none_values is None:
        continue
    for source in NOISE_SOURCES:
        only_values = ablation_values.get((float(n_bar), f"only__{source}"))
        without_values = ablation_values.get((float(n_bar), f"without__{source}"))
        if only_values is None or without_values is None:
            continue
        for observable in all_values:
            ablation_attribution_rows.append({
                "n_bar": float(n_bar),
                "noise_source": source,
                "observable": observable,
                "leave_one_out_all_minus_without": all_values[observable] - without_values[observable],
                "isolated_only_minus_none": only_values[observable] - none_values[observable],
            })

ablation_attribution_df = pd.DataFrame(ablation_attribution_rows)
if ablation_attribution_df.empty:
    print("Noise-ablation cache is empty. Set RUN_NOISE_SOURCE_ABLATION=True to calculate it.")
else:
    raw_path = ABLATION_DIR / "noise_ablation_raw_observables.csv"
    attribution_path = ABLATION_DIR / "noise_ablation_attribution.csv"
    ablation_raw_df.to_csv(raw_path, index=False)
    ablation_attribution_df.to_csv(attribution_path, index=False)
    display(ablation_attribution_df)

    observable_order = ["chi_XX_XX", "abs_chi_II_XX", "chi_IX_IX", "chi_XI_XI"]
    fig_ablation, axes_ablation = plt.subplots(2, 2, figsize=(13.0, 9.0), sharex=True)
    for ax, observable in zip(axes_ablation.ravel(), observable_order):
        panel = ablation_attribution_df[ablation_attribution_df["observable"] == observable]
        for source, group in panel.groupby("noise_source"):
            ax.plot(
                group["n_bar"], group["leave_one_out_all_minus_without"],
                marker="o", linewidth=2, label=source,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(observable)
        ax.set_xlabel(r"Mean phonon number $\bar n$")
        ax.set_ylabel("all − without source")
        ax.grid(True, alpha=0.28)
    axes_ablation[0, 0].legend(fontsize=9)
    fig_ablation.suptitle("Noise-source leave-one-out attribution (non-additive diagnostic)")
    fig_ablation.tight_layout(rect=(0, 0, 1, 0.96))
    ablation_png = ABLATION_DIR / "noise_ablation_leave_one_out.png"
    ablation_pdf = ABLATION_DIR / "noise_ablation_leave_one_out.pdf"
    fig_ablation.savefig(ablation_png, dpi=300, bbox_inches="tight")
    fig_ablation.savefig(ablation_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved: {attribution_path}")


# ### 4. 全256成分のランキング（見落とし確認）
# 
# 生の複素χキャッシュから、Hermitian対称な重複を除く上三角136成分（対角16、オフ対角120）を全て比較します。各成分の $\max_{\bar n}|\chi_{P,Q}|$ で順位付けし、上位成分を可視化します。
# 
# このセルは「`XI,IX` 以外は小さいのか」を、対角成分だけでなくオフ対角成分も含めて確認するためのものです。

# In[14]:


full_chi_long_rows = []
for point in exact_full_chi_points:
    chi = point["chi"]
    for row_index, row_label in enumerate(PAULI_LABELS):
        for col_index in range(row_index, len(PAULI_LABELS)):
            col_label = PAULI_LABELS[col_index]
            value = chi[row_index, col_index]
            full_chi_long_rows.append({
                "n_bar": point["n_bar"],
                "row_pauli": row_label,
                "column_pauli": col_label,
                "component": f"{row_label},{col_label}",
                "is_diagonal": row_index == col_index,
                "real": float(np.real(value)),
                "imag": float(np.imag(value)),
                "abs": float(abs(value)),
            })

full_chi_long_df = pd.DataFrame(full_chi_long_rows)
if full_chi_long_df.empty:
    full_chi_ranking_df = pd.DataFrame()
    print("Exact full-chi cache is empty. Run section 1 first to rank all 256 matrix entries.")
else:
    full_chi_long_path = EXACT_FULL_CHI_DIR / "full_chi_components_long.csv"
    full_chi_long_df.to_csv(full_chi_long_path, index=False)
    full_chi_ranking_df = (
        full_chi_long_df.groupby(
            ["component", "row_pauli", "column_pauli", "is_diagonal"], as_index=False
        )
        .agg(max_abs_over_sweep=("abs", "max"), mean_abs_over_sweep=("abs", "mean"))
        .sort_values("max_abs_over_sweep", ascending=False)
        .reset_index(drop=True)
    )
    full_chi_ranking_df.insert(0, "rank", np.arange(1, len(full_chi_ranking_df) + 1))
    full_chi_ranking_path = EXACT_FULL_CHI_DIR / "full_chi_component_ranking.csv"
    full_chi_ranking_df.to_csv(full_chi_ranking_path, index=False)
    display(full_chi_ranking_df.head(25))

    top_components = full_chi_ranking_df.head(FULL_CHI_TOP_K)["component"].tolist()
    fig_rank, ax_rank = plt.subplots(figsize=(10.0, 6.2))
    for component in top_components:
        group = full_chi_long_df[full_chi_long_df["component"] == component].sort_values("n_bar")
        ax_rank.semilogy(
            group["n_bar"], np.maximum(group["abs"], 1e-18),
            marker="o", markersize=3, linewidth=1.7, label=component,
        )
    ax_rank.set_xlabel(r"Mean phonon number $\bar n$")
    ax_rank.set_ylabel(r"$|\chi_{P,Q}|$")
    ax_rank.set_title(f"Top {FULL_CHI_TOP_K} full-χ components over the exact sweep")
    ax_rank.grid(True, which="both", alpha=0.25)
    ax_rank.legend(ncol=2, fontsize=9)
    fig_rank.tight_layout()
    rank_png = EXACT_FULL_CHI_DIR / "full_chi_top_components.png"
    rank_pdf = EXACT_FULL_CHI_DIR / "full_chi_top_components.pdf"
    fig_rank.savefig(rank_png, dpi=300, bbox_inches="tight")
    fig_rank.savefig(rank_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved: {full_chi_ranking_path}")


# ### 5. fitモデル比較・交差検証・bootstrap信頼区間
# 
# 1〜5次のアンカー付き多項式について、$R^2$ だけでなく AICc、BIC、5-fold CV RMSE を比較します。そのうえで、本文で使う4次式について残差bootstrapによる95%区間を計算します。
# 
# 注意：この区間は**回帰近似の不確かさ**です。実験誤差や数値打切り誤差ではありません。後者は上の収束性検証で別に評価します。

# In[15]:


MODEL_VALIDATION_TARGETS = {
    "chi_XX_XX": (x_data, sweep_df["chi_XX_XX"].to_numpy(float)),
    "abs_chi_II_XX": (x_data, sweep_df["chi_II_XX_abs"].to_numpy(float)),
    "chi_IX_IX": (other_x_data, pauli_wide_df["IX"].to_numpy(float)),
    "chi_XI_XI": (other_x_data, pauli_wide_df["XI"].to_numpy(float)),
}

def anchored_kfold_rmse(x, y, degree, n_splits=5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    non_anchor_indices = np.arange(1, len(x))
    folds = np.array_split(non_anchor_indices, n_splits)
    squared_errors = []
    for test_indices in folds:
        train_mask = np.ones(len(x), dtype=bool)
        train_mask[test_indices] = False
        train_mask[0] = True
        fit = anchored_polynomial_fit(x[train_mask], y[train_mask], degree)
        predictions = fit["predict"](x[test_indices])
        squared_errors.extend((y[test_indices] - predictions) ** 2)
    return float(np.sqrt(np.mean(squared_errors)))

model_comparison_rows = []
for target, (target_x, target_y) in MODEL_VALIDATION_TARGETS.items():
    for degree in range(1, 6):
        fit = anchored_polynomial_fit(target_x, target_y, degree)
        n_observations = len(target_x)
        parameter_count = degree
        rss = max(float(np.sum(fit["residual"] ** 2)), np.finfo(float).tiny)
        aic = n_observations * np.log(rss / n_observations) + 2 * parameter_count
        aicc = aic + (
            2 * parameter_count * (parameter_count + 1)
            / (n_observations - parameter_count - 1)
        )
        bic = n_observations * np.log(rss / n_observations) + parameter_count * np.log(n_observations)
        model_comparison_rows.append({
            "target": target,
            "degree": degree,
            "R_squared": fit["r_squared"],
            "RMSE": fit["rmse"],
            "max_abs_residual": fit["max_abs_residual"],
            "AICc": float(aicc),
            "BIC": float(bic),
            "CV_RMSE_5fold": anchored_kfold_rmse(target_x, target_y, degree),
            "equation": fit["equation"],
        })

model_comparison_df = pd.DataFrame(model_comparison_rows)
model_comparison_df["best_AICc"] = model_comparison_df.groupby("target")["AICc"].transform("min").eq(model_comparison_df["AICc"])
model_comparison_df["best_CV"] = model_comparison_df.groupby("target")["CV_RMSE_5fold"].transform("min").eq(model_comparison_df["CV_RMSE_5fold"])
model_comparison_path = OUTPUT_DIR / "fit_model_comparison_aicc_bic_cv.csv"
model_comparison_df.to_csv(model_comparison_path, index=False)
display(model_comparison_df)

rng = np.random.default_rng(BOOTSTRAP_SEED)
bootstrap_curve_df = pd.DataFrame({"n_bar": x_dense})
bootstrap_coefficient_rows = []
bootstrap_selected_rows = []
for target, (target_x, target_y) in MODEL_VALIDATION_TARGETS.items():
    base_fit = anchored_polynomial_fit(target_x, target_y, FIT_DEGREE)
    centered_residuals = base_fit["residual"] - np.mean(base_fit["residual"])
    prediction_samples = np.empty((BOOTSTRAP_SAMPLES, len(x_dense)))
    coefficient_samples = np.empty((BOOTSTRAP_SAMPLES, FIT_DEGREE))
    for sample_index in range(BOOTSTRAP_SAMPLES):
        synthetic_y = base_fit["fitted"] + rng.choice(
            centered_residuals, size=len(target_y), replace=True
        )
        synthetic_y[0] = target_y[0]
        sample_fit = anchored_polynomial_fit(target_x, synthetic_y, FIT_DEGREE)
        prediction_samples[sample_index] = sample_fit["predict"](x_dense)
        coefficient_samples[sample_index] = sample_fit["coefficients"]

    lower, upper = np.percentile(prediction_samples, [2.5, 97.5], axis=0)
    bootstrap_curve_df[f"{target}_fit"] = base_fit["predict"](x_dense)
    bootstrap_curve_df[f"{target}_ci95_lower"] = lower
    bootstrap_curve_df[f"{target}_ci95_upper"] = upper
    bootstrap_coefficient_rows.append({
        "target": target,
        "coefficient": "a0_anchor",
        "estimate": base_fit["y0"],
        "ci95_lower": base_fit["y0"],
        "ci95_upper": base_fit["y0"],
    })
    for power, estimate in enumerate(base_fit["coefficients"], start=1):
        coefficient_lower, coefficient_upper = np.percentile(
            coefficient_samples[:, power - 1], [2.5, 97.5]
        )
        bootstrap_coefficient_rows.append({
            "target": target,
            "coefficient": f"a{power}",
            "estimate": float(estimate),
            "ci95_lower": float(coefficient_lower),
            "ci95_upper": float(coefficient_upper),
        })
    chosen_model = model_comparison_df[
        (model_comparison_df["target"] == target)
        & (model_comparison_df["degree"] == FIT_DEGREE)
    ].iloc[0]
    bootstrap_selected_rows.append({
        "target": target,
        "chosen_degree": FIT_DEGREE,
        "equation": base_fit["equation"],
        "R_squared": base_fit["r_squared"],
        "RMSE": base_fit["rmse"],
        "AICc": chosen_model["AICc"],
        "BIC": chosen_model["BIC"],
        "CV_RMSE_5fold": chosen_model["CV_RMSE_5fold"],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    })

bootstrap_coefficient_df = pd.DataFrame(bootstrap_coefficient_rows)
bootstrap_selected_df = pd.DataFrame(bootstrap_selected_rows)
bootstrap_curve_path = OUTPUT_DIR / "fit_bootstrap_95ci_curves.csv"
bootstrap_coefficient_path = OUTPUT_DIR / "fit_bootstrap_coefficient_95ci.csv"
bootstrap_selected_path = OUTPUT_DIR / "fit_selected_models_publication.csv"
bootstrap_curve_df.to_csv(bootstrap_curve_path, index=False)
bootstrap_coefficient_df.to_csv(bootstrap_coefficient_path, index=False)
bootstrap_selected_df.to_csv(bootstrap_selected_path, index=False)
display(bootstrap_selected_df)
display(bootstrap_coefficient_df)

fig_ci, axes_ci = plt.subplots(2, 2, figsize=(13.0, 9.0))
for ax, (target, (target_x, target_y)) in zip(axes_ci.ravel(), MODEL_VALIDATION_TARGETS.items()):
    ax.scatter(target_x, target_y, s=18, alpha=0.65, label="sweep data", zorder=3)
    ax.plot(x_dense, bootstrap_curve_df[f"{target}_fit"], color="black", linewidth=2, label=f"degree-{FIT_DEGREE} fit")
    ax.fill_between(
        x_dense,
        bootstrap_curve_df[f"{target}_ci95_lower"],
        bootstrap_curve_df[f"{target}_ci95_upper"],
        color="#56B4E9", alpha=0.30, label="residual bootstrap 95% interval",
    )
    ax.set_title(target)
    ax.set_xlabel(r"Mean phonon number $\bar n$")
    ax.set_ylabel("χ component")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
fig_ci.suptitle("Anchored polynomial fits with residual-bootstrap intervals")
fig_ci.tight_layout(rect=(0, 0, 1, 0.96))
bootstrap_png = OUTPUT_DIR / "fit_bootstrap_95ci.png"
bootstrap_pdf = OUTPUT_DIR / "fit_bootstrap_95ci.pdf"
fig_ci.savefig(bootstrap_png, dpi=300, bbox_inches="tight")
fig_ci.savefig(bootstrap_pdf, bbox_inches="tight")
plt.show()
print(f"Saved: {model_comparison_path}")
print(f"Saved: {bootstrap_selected_path}")


# ### 6. 再現性manifest
# 
# 最後に、使用した物理パラメータ、χ規格化、ソフトウェア版、Git状態、キャッシュ充足数をJSONへ固定します。論文用の図を更新した際は、このmanifestも同時に保存してください。

# In[16]:


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value

def git_output(arguments):
    try:
        return subprocess.check_output(
            ["git", *arguments], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        return f"unavailable: {error}"

manifest = {
    "generated_at_timezone": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
    "notebook": str(Path("chi_error_element_nbar_fit.ipynb").resolve()),
    "simulation_parameters": json_safe(SIMULATION_PARAMS),
    "n_bar_grid": json_safe(NBAR_GRID),
    "error_channel_convention": ERROR_CHANNEL_CONVENTION,
    "chi_normalization": "chi / trace(chi)",
    "pauli_basis_order": PAULI_LABELS,
    "fit": {
        "family": "polynomial anchored at n_bar=0.01",
        "reported_degree": FIT_DEGREE,
        "candidate_degrees": [1, 2, 3, 4, 5],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    },
    "publication_run_flags": {
        "exact_full_chi_sweep": RUN_EXACT_FULL_CHI_SWEEP,
        "numerical_convergence": RUN_NUMERICAL_CONVERGENCE,
        "noise_source_ablation": RUN_NOISE_SOURCE_ABLATION,
    },
    "cache_status": {
        "target_element_points": int(len(sweep_df)),
        "pauli_diagonal_points": int(len(pauli_wide_df)),
        "exact_full_chi_points": int(len(exact_full_chi_points)),
        "expected_sweep_points": int(len(NBAR_GRID)),
        "numerical_convergence_rows": int(len(convergence_df)),
        "noise_ablation_attribution_rows": int(len(ablation_attribution_df)),
    },
    "software_versions": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "qutip": qp.__version__,
    },
    "git": {
        "commit": git_output(["rev-parse", "HEAD"]),
        "status_short": git_output(["status", "--short"]),
    },
}
manifest_path = OUTPUT_DIR / "publication_reproducibility_manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
    encoding="utf-8",
)
display(pd.Series(manifest["cache_status"], name="count").to_frame())
print(f"Saved: {manifest_path}")


# ### 論文用の推奨実行順
# 
# 1. まず通常実行し、既存データ、fit比較、bootstrap図が再現できることを確認する。
# 2. `RUN_EXACT_FULL_CHI_SWEEP=True` にして81点の生χを確定する。
# 3. `RUN_NUMERICAL_CONVERGENCE=True` と `RUN_NOISE_SOURCE_ABLATION=True` を順に実行する。
# 4. 重い計算が完了したら各フラグを `False` に戻し、ノート全体を再実行して全ランキング・図・manifestを固定する。
# 
# キャッシュがあるため、重い計算は途中停止後も続きから再開できます。

# ## 論文化のための拡張検証
# 
# ここでは、前節で残った5点を検証します。
# 
# 1. Dykstra法による最小Frobenius距離CPTP射影と、射影前後の結論比較
# 2. error generator $\log\mathcal S_{\rm error}$ によるHamiltonian型 `XX` とPauli散逸型 `XX` の分離
# 3. channel-levelの最適 `XX` 角補正と、$h_{XX}$から算出したdrive amplitudeによる全Hamiltonian再QPT
# 4. $\eta$、$A/\delta$、gate time、motional dephasing rateに対する頑健性
# 5. $(2\bar n+1)$ と $\bar n(\bar n+1)$ を用いた物理モデルfit
# 
# CPTP射影・generator分解・channel-level補正・物理fitは、保存済み81点を使うため高速です。再QPTを伴う部分だけ明示的なスイッチで有効化します。

# In[17]:


import hashlib
import math
import os
import time

ADVANCED_DIR = OUTPUT_DIR / "advanced_publication_validation"
CPTP_DIR = ADVANCED_DIR / "cptp_projection"
GENERATOR_DIR = ADVANCED_DIR / "error_generator"
CORRECTION_DIR = ADVANCED_DIR / "xx_angle_correction"
CONTROL_DIR = ADVANCED_DIR / "physical_control_validation"
ROBUSTNESS_DIR = ADVANCED_DIR / "parameter_robustness"
PHYSICAL_FIT_DIR = ADVANCED_DIR / "physical_model_fit"
for directory in (
    ADVANCED_DIR, CPTP_DIR, GENERATOR_DIR, CORRECTION_DIR,
    CONTROL_DIR, ROBUSTNESS_DIR, PHYSICAL_FIT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

try:
    import psutil
    PHYSICAL_CORES = psutil.cpu_count(logical=False) or (os.cpu_count() or 1)
    MEMORY_GIB = psutil.virtual_memory().total / 2**30
except Exception:
    PHYSICAL_CORES = os.cpu_count() or 1
    MEMORY_GIB = 16.0

# 高n_barでは1 workerあたり数GBを見込む。大容量機でも過剰並列を避けるため既定上限は8。
AUTO_PROCESS_WORKERS = max(
    1,
    min(int(PHYSICAL_CORES) - 1 if PHYSICAL_CORES > 1 else 1, int(MEMORY_GIB // 3.5), 8),
)
FAST_PROCESS_WORKERS = int(
    _config(
        "FAST_PROCESS_WORKERS",
        os.environ.get("MS_GATE_WORKERS", AUTO_PROCESS_WORKERS),
    )
)
for thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")

RUN_PHYSICAL_CONTROL_QPT = bool(_config("RUN_PHYSICAL_CONTROL_QPT", False))
RUN_PARAMETER_ROBUSTNESS_QPT = bool(
    _config("RUN_PARAMETER_ROBUSTNESS_QPT", False)
)
FORCE_RECOMPUTE_ADVANCED_QPT = bool(
    _config("FORCE_RECOMPUTE_ADVANCED_QPT", False)
)

CONTROL_VALIDATION_NBARS = list(
    _config("CONTROL_VALIDATION_NBARS", [0.01, 4.0, 20.0])
)
ROBUSTNESS_NBARS = list(_config("ROBUSTNESS_NBARS", [0.01, 4.0, 20.0]))
CPTP_TOLERANCE = float(_config("CPTP_TOLERANCE", 1e-11))
CPTP_MAX_ITERATIONS = int(_config("CPTP_MAX_ITERATIONS", 5000))

print(f"Machine: {PHYSICAL_CORES} physical cores, {MEMORY_GIB:.1f} GiB RAM")
print(f"QPT process workers: {FAST_PROCESS_WORKERS}")
print("RUN_PHYSICAL_CONTROL_QPT =", RUN_PHYSICAL_CONTROL_QPT)
print("RUN_PARAMETER_ROBUSTNESS_QPT =", RUN_PARAMETER_ROBUSTNESS_QPT)


# ### 7. CPTP射影と結論の安定性
# 
# Choi行列に対し、正半定値集合（CP）と部分トレースが恒等行列になるアフィン集合（TP）へのDykstra交互射影を行います。これはFrobeniusノルムで元のChoi行列に近いCPTP行列を求める反復法です。
# 
# 射影後に主要χ成分、全成分ランキング、温度クロスオーバーが維持されるかを定量化します。

# In[18]:


TWO_QUBIT_SUPER_DIMS = [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]

def chi_raw_to_superoperator(chi_raw):
    chi_qobj = qp.Qobj(
        np.asarray(chi_raw, dtype=complex),
        dims=TWO_QUBIT_SUPER_DIMS,
        superrep="chi",
    )
    return qp.to_super(chi_qobj)

def choi_partial_trace_over_output(choi_matrix, dimension=4):
    tensor = np.asarray(choi_matrix, dtype=complex).reshape(
        dimension, dimension, dimension, dimension
    )
    return np.einsum("iaja->ij", tensor)

def project_to_tp_affine(choi_matrix, dimension=4):
    hermitian = 0.5 * (choi_matrix + choi_matrix.conj().T)
    residual = choi_partial_trace_over_output(hermitian, dimension) - np.eye(dimension)
    return hermitian - np.kron(residual, np.eye(dimension) / dimension)

def project_to_psd_cone(matrix):
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.conj().T

def nearest_cptp_choi_dykstra(
    choi_matrix,
    dimension=4,
    tolerance=CPTP_TOLERANCE,
    max_iterations=CPTP_MAX_ITERATIONS,
):
    current = 0.5 * (choi_matrix + choi_matrix.conj().T)
    psd_correction = np.zeros_like(current)
    tp_correction = np.zeros_like(current)

    for iteration in range(1, max_iterations + 1):
        psd_input = current + psd_correction
        psd_projected = project_to_psd_cone(psd_input)
        psd_correction = psd_input - psd_projected

        tp_input = psd_projected + tp_correction
        updated = project_to_tp_affine(tp_input, dimension)
        tp_correction = tp_input - updated

        updated_hermitian = 0.5 * (updated + updated.conj().T)
        min_eigenvalue = float(np.linalg.eigvalsh(updated_hermitian).min())
        tp_error = float(np.linalg.norm(
            choi_partial_trace_over_output(updated_hermitian, dimension) - np.eye(dimension)
        ))
        relative_step = float(
            np.linalg.norm(updated - current) / max(np.linalg.norm(current), 1.0)
        )
        current = updated
        if (
            min_eigenvalue >= -tolerance
            and tp_error <= tolerance
            and relative_step <= tolerance
        ):
            break
    else:
        warnings.warn("CPTP Dykstra projection reached the iteration limit")

    return current, {
        "iterations": iteration,
        "min_choi_eigenvalue": min_eigenvalue,
        "tp_frobenius_error": tp_error,
        "relative_final_step": relative_step,
    }

def project_chi_point_to_cptp(point):
    source_super = chi_raw_to_superoperator(point["chi"] * 16.0)
    source_choi = qp.to_choi(source_super).full()
    projected_choi, projection_status = nearest_cptp_choi_dykstra(source_choi)
    projected_choi_qobj = qp.Qobj(
        projected_choi, dims=TWO_QUBIT_SUPER_DIMS, superrep="choi"
    )
    projected_super = qp.to_super(projected_choi_qobj)
    projected_chi_raw = qp.to_chi(projected_super).full()
    projected_chi = projected_chi_raw / np.trace(projected_chi_raw)
    return projected_super, projected_chi_raw, projected_chi, projection_status

cptp_points = []
cptp_comparison_rows = []
for point in exact_full_chi_points:
    output_path = CPTP_DIR / f"cptp_chi_nbar_{_safe_nbar_stem(point['n_bar'])}.npz"
    if output_path.exists() and not FORCE_RECOMPUTE_ADVANCED_QPT:
        with np.load(output_path, allow_pickle=False) as cached:
            projected_chi_raw = np.asarray(cached["chi_raw"], dtype=complex)
            projected_chi = np.asarray(
                cached["chi_trace_normalized"], dtype=complex
            )
            status = json.loads(str(cached["projection_status_json"].item()))
        projected_super = chi_raw_to_superoperator(projected_chi_raw)
    else:
        projected_super, projected_chi_raw, projected_chi, status = (
            project_chi_point_to_cptp(point)
        )
        temporary_path = output_path.with_name(output_path.stem + ".tmp.npz")
        np.savez_compressed(
            temporary_path,
            n_bar=float(point["n_bar"]),
            chi_raw=projected_chi_raw,
            chi_trace_normalized=projected_chi,
            projection_status_json=json.dumps(status),
        )
        temporary_path.replace(output_path)
    physicality = mg.choi_physicality_metrics(
        projected_super, tp_tol=CPTP_TOLERANCE, cp_tol=CPTP_TOLERANCE
    )
    raw_chi = point["chi"]
    cptp_points.append({
        "n_bar": float(point["n_bar"]),
        "chi": projected_chi,
        "chi_raw": projected_chi_raw,
        "super": projected_super,
        "status": status,
        "path": output_path,
    })
    cptp_comparison_rows.append({
        "n_bar": float(point["n_bar"]),
        "raw_min_choi_eigenvalue": float(point["metadata"]["min_choi_eigenvalue"]),
        "projected_min_choi_eigenvalue": physicality["min_choi_eigenvalue"],
        "projected_tp_frobenius_error": physicality["tp_frobenius_error"],
        "projection_iterations": status["iterations"],
        "chi_frobenius_shift": float(np.linalg.norm(projected_chi - raw_chi)),
        "delta_chi_XX_XX": float(np.real(projected_chi[XX_INDEX, XX_INDEX] - raw_chi[XX_INDEX, XX_INDEX])),
        "delta_abs_chi_II_XX": float(abs(projected_chi[II_INDEX, XX_INDEX]) - abs(raw_chi[II_INDEX, XX_INDEX])),
        "delta_chi_IX_IX": float(np.real(projected_chi[PAULI_LABELS.index("IX"), PAULI_LABELS.index("IX")] - raw_chi[PAULI_LABELS.index("IX"), PAULI_LABELS.index("IX")])),
        "delta_chi_XI_XI": float(np.real(projected_chi[PAULI_LABELS.index("XI"), PAULI_LABELS.index("XI")] - raw_chi[PAULI_LABELS.index("XI"), PAULI_LABELS.index("XI")])),
        "projected_cp_pass": physicality["cp_pass"],
        "projected_tp_pass": physicality["tp_pass"],
    })

cptp_comparison_df = pd.DataFrame(cptp_comparison_rows)
if cptp_comparison_df.empty:
    print("Exact full-chi data are required before CPTP projection.")
else:
    cptp_summary_path = CPTP_DIR / "cptp_projection_comparison.csv"
    cptp_comparison_df.to_csv(cptp_summary_path, index=False)

    projected_component_rows = []
    for point in cptp_points:
        for row_index, row_label in enumerate(PAULI_LABELS):
            for column_index in range(row_index, len(PAULI_LABELS)):
                value = point["chi"][row_index, column_index]
                projected_component_rows.append({
                    "n_bar": point["n_bar"],
                    "component": f"{row_label},{PAULI_LABELS[column_index]}",
                    "abs": float(abs(value)),
                })
    projected_ranking_df = (
        pd.DataFrame(projected_component_rows)
        .groupby("component", as_index=False)
        .agg(max_abs_over_sweep=("abs", "max"))
        .sort_values("max_abs_over_sweep", ascending=False)
        .reset_index(drop=True)
    )
    projected_ranking_df.insert(0, "rank", np.arange(1, len(projected_ranking_df) + 1))
    projected_ranking_df.to_csv(CPTP_DIR / "cptp_full_chi_ranking.csv", index=False)

    raw_top = full_chi_ranking_df.head(FULL_CHI_TOP_K)["component"].tolist()
    projected_top = projected_ranking_df.head(FULL_CHI_TOP_K)["component"].tolist()
    top_overlap = len(set(raw_top) & set(projected_top))
    print(f"CPTP pass after projection: {cptp_comparison_df['projected_cp_pass'].sum()}/{len(cptp_comparison_df)}")
    print(f"TP pass after projection: {cptp_comparison_df['projected_tp_pass'].sum()}/{len(cptp_comparison_df)}")
    print(f"Top-{FULL_CHI_TOP_K} ranking overlap: {top_overlap}/{FULL_CHI_TOP_K}")
    print("Maximum full-chi projection shift:", cptp_comparison_df["chi_frobenius_shift"].max())
    display(cptp_comparison_df.describe().T)
    display(projected_ranking_df.head(20))

    fig_cptp, axes_cptp = plt.subplots(1, 2, figsize=(13.0, 4.8))
    axes_cptp[0].plot(
        cptp_comparison_df["n_bar"],
        cptp_comparison_df["raw_min_choi_eigenvalue"],
        label="raw", linewidth=2,
    )
    axes_cptp[0].plot(
        cptp_comparison_df["n_bar"],
        cptp_comparison_df["projected_min_choi_eigenvalue"],
        label="CPTP projected", linewidth=2,
    )
    axes_cptp[0].axhline(0.0, color="black", linewidth=0.8)
    axes_cptp[0].set_xlabel(r"Mean phonon number $\bar n$")
    axes_cptp[0].set_ylabel("Minimum Choi eigenvalue")
    axes_cptp[0].set_title("Complete positivity before/after projection")
    axes_cptp[0].legend()
    axes_cptp[0].grid(True, alpha=0.28)

    axes_cptp[1].semilogy(
        cptp_comparison_df["n_bar"],
        np.maximum(cptp_comparison_df["chi_frobenius_shift"], 1e-18),
        color="#D55E00", linewidth=2,
    )
    axes_cptp[1].set_xlabel(r"Mean phonon number $\bar n$")
    axes_cptp[1].set_ylabel(r"$\|\chi_{\rm CPTP}-\chi_{\rm raw}\|_F$")
    axes_cptp[1].set_title("Size of the CPTP correction")
    axes_cptp[1].grid(True, which="both", alpha=0.28)
    fig_cptp.tight_layout()
    fig_cptp.savefig(CPTP_DIR / "cptp_projection_stability.png", dpi=300, bbox_inches="tight")
    fig_cptp.savefig(CPTP_DIR / "cptp_projection_stability.pdf", bbox_inches="tight")
    plt.show()


# ### 8. error generatorによるコヒーレント／確率的誤差の分離
# 
# CPTP射影後のPauli transfer matrixを $R$ とし、一ゲートあたりのgenerator
# 
# \[
# K=\log R
# \]
# 
# を計算します。反対称部分を $-i[H,\rho]$ のPauli Hamiltonian基底へ最小二乗射影し、対称残差を $\mathcal D[P](\rho)=P\rho P-\rho$ のPauli散逸基底へ非負最小二乗射影します。
# 
# `h_XX` はHamiltonian型XX角（rad/gate）、`gamma_XX` はPauli散逸型XX率（1/gate）です。残差も保存し、この簡略モデルがgenerator全体をどこまで説明するか確認します。

# In[19]:


PAULI_QOBJS = mg.two_qubit_pauli_basis()

def action_to_ptm(action):
    dimension = PAULI_QOBJS[0].shape[0]
    matrix = np.zeros((len(PAULI_QOBJS), len(PAULI_QOBJS)), dtype=complex)
    for column, input_pauli in enumerate(PAULI_QOBJS):
        output = action(input_pauli)
        for row, measurement_pauli in enumerate(PAULI_QOBJS):
            matrix[row, column] = (measurement_pauli.dag() * output).tr() / dimension
    return np.real_if_close(matrix).astype(float)

HAMILTONIAN_GENERATOR_BASES = {
    label: action_to_ptm(lambda rho, p=pauli: -1j * (p * rho - rho * p))
    for label, pauli in zip(PAULI_LABELS[1:], PAULI_QOBJS[1:])
}
PAULI_DISSIPATOR_BASES = {
    label: action_to_ptm(lambda rho, p=pauli: p * rho * p - rho)
    for label, pauli in zip(PAULI_LABELS[1:], PAULI_QOBJS[1:])
}

hamiltonian_design = np.column_stack([
    HAMILTONIAN_GENERATOR_BASES[label].reshape(-1)
    for label in PAULI_LABELS[1:]
])
dissipator_design = np.column_stack([
    PAULI_DISSIPATOR_BASES[label].reshape(-1)
    for label in PAULI_LABELS[1:]
])

generator_rows = []
generator_coefficient_rows = []
for point in cptp_points:
    ptm = np.asarray(mg.superoperator_to_ptm(point["super"]), dtype=complex)
    generator_complex = scipy.linalg.logm(ptm)
    imaginary_norm = float(np.linalg.norm(np.imag(generator_complex)))
    generator = np.real(generator_complex)
    skew_generator = 0.5 * (generator - generator.T)

    h_coefficients = np.linalg.lstsq(
        hamiltonian_design, skew_generator.reshape(-1), rcond=None
    )[0]
    h_fit = sum(
        coefficient * HAMILTONIAN_GENERATOR_BASES[label]
        for label, coefficient in zip(PAULI_LABELS[1:], h_coefficients)
    )
    remaining = generator - h_fit
    symmetric_remaining = 0.5 * (remaining + remaining.T)
    gamma_coefficients, nnls_residual = scipy.optimize.nnls(
        dissipator_design, symmetric_remaining.reshape(-1)
    )
    dissipator_fit = sum(
        coefficient * PAULI_DISSIPATOR_BASES[label]
        for label, coefficient in zip(PAULI_LABELS[1:], gamma_coefficients)
    )
    fitted_generator = h_fit + dissipator_fit

    chi = point["chi"]
    coherent_angle_from_chi = 0.5 * np.arctan2(
        2.0 * np.imag(chi[II_INDEX, XX_INDEX]),
        np.real(chi[II_INDEX, II_INDEX] - chi[XX_INDEX, XX_INDEX]),
    )
    generator_rows.append({
        "n_bar": point["n_bar"],
        "h_XX_rad_per_gate": float(h_coefficients[PAULI_LABELS[1:].index("XX")]),
        "gamma_XX_per_gate": float(gamma_coefficients[PAULI_LABELS[1:].index("XX")]),
        "chi_coherent_XX_angle_rad": float(coherent_angle_from_chi),
        "generator_frobenius_norm": float(np.linalg.norm(generator)),
        "hamiltonian_fit_frobenius_norm": float(np.linalg.norm(h_fit)),
        "pauli_dissipator_fit_frobenius_norm": float(np.linalg.norm(dissipator_fit)),
        "unmodeled_generator_frobenius_norm": float(np.linalg.norm(generator - fitted_generator)),
        "generator_reconstruction_error": float(np.linalg.norm(scipy.linalg.expm(generator) - np.real(ptm))),
        "logm_imaginary_frobenius_norm": imaginary_norm,
        "nnls_residual": float(nnls_residual),
    })
    for label, h_coefficient, gamma_coefficient in zip(
        PAULI_LABELS[1:], h_coefficients, gamma_coefficients
    ):
        generator_coefficient_rows.append({
            "n_bar": point["n_bar"],
            "pauli": label,
            "hamiltonian_coefficient_rad_per_gate": float(h_coefficient),
            "pauli_dissipator_rate_per_gate": float(gamma_coefficient),
        })

generator_df = pd.DataFrame(generator_rows).sort_values("n_bar").reset_index(drop=True)
generator_coefficients_df = pd.DataFrame(generator_coefficient_rows)
if generator_df.empty:
    print("CPTP-projected points are required for error-generator analysis.")
else:
    generator_df.to_csv(GENERATOR_DIR / "error_generator_summary.csv", index=False)
    generator_coefficients_df.to_csv(
        GENERATOR_DIR / "error_generator_pauli_coefficients.csv", index=False
    )
    display(generator_df)

    fig_generator, axes_generator = plt.subplots(1, 2, figsize=(13.0, 4.8))
    axes_generator[0].plot(
        generator_df["n_bar"], generator_df["h_XX_rad_per_gate"],
        linewidth=2.2, label=r"generator $h_{XX}$",
    )
    axes_generator[0].plot(
        generator_df["n_bar"], generator_df["chi_coherent_XX_angle_rad"],
        linewidth=2.0, linestyle="--", label=r"angle inferred from $\chi$",
    )
    axes_generator[0].set_xlabel(r"Mean phonon number $\bar n$")
    axes_generator[0].set_ylabel("Coherent XX angle (rad/gate)")
    axes_generator[0].set_title("Hamiltonian-type correlated XX error")
    axes_generator[0].grid(True, alpha=0.28)
    axes_generator[0].legend()

    axes_generator[1].semilogy(
        generator_df["n_bar"],
        np.maximum(generator_df["gamma_XX_per_gate"], 1e-18),
        linewidth=2.2, color="#D55E00",
    )
    axes_generator[1].set_xlabel(r"Mean phonon number $\bar n$")
    axes_generator[1].set_ylabel(r"Pauli dissipator rate $\gamma_{XX}$ (1/gate)")
    axes_generator[1].set_title("Stochastic Pauli-XX generator component")
    axes_generator[1].grid(True, which="both", alpha=0.28)
    fig_generator.tight_layout()
    fig_generator.savefig(GENERATOR_DIR / "xx_generator_decomposition.png", dpi=300, bbox_inches="tight")
    fig_generator.savefig(GENERATOR_DIR / "xx_generator_decomposition.pdf", bbox_inches="tight")
    plt.show()


# ### 9. 最適XX角補正と実パルス検証
# 
# まず、各CPTP error channelの後段に $U_c=\exp(i\theta_c XX)$ を合成し、average infidelityを最小化する $\theta_c$ を求めます。これは再QPT不要で、温度依存のエンタングリング角校正が到達できる上限を示します。
# 
# 続く重いセルでは、drive amplitude、gate time、detuning、RMS規格化したsmooth pulseを実際のHamiltonianへ入れて再QPTします。`RUN_PHYSICAL_CONTROL_QPT=True` の場合のみ実行され、条件・温度ごとのNPZキャッシュから再開できます。

# In[20]:


def trace_normalized_chi_from_super(superoperator):
    chi_raw = qp.to_chi(superoperator).full()
    return chi_raw / np.trace(chi_raw)

def average_infidelity_from_trace_normalized_chi(chi, dimension=4):
    entanglement_fidelity = float(np.real(chi[II_INDEX, II_INDEX]))
    return dimension / (dimension + 1.0) * (1.0 - entanglement_fidelity)

def compose_xx_correction(error_superoperator, correction_angle):
    correction_superoperator = qp.to_super(mg.ideal_ms_gate(phi=float(correction_angle)))
    return correction_superoperator * error_superoperator

channel_correction_rows = []
channel_corrected_points = []
for point in cptp_points:
    def correction_objective(angle):
        corrected_super = compose_xx_correction(point["super"], angle)
        corrected_chi = trace_normalized_chi_from_super(corrected_super)
        return average_infidelity_from_trace_normalized_chi(corrected_chi)

    optimum = scipy.optimize.minimize_scalar(
        correction_objective,
        bounds=(-0.5, 0.5),
        method="bounded",
        options={"xatol": 1e-11},
    )
    before_chi = point["chi"]
    corrected_super = compose_xx_correction(point["super"], optimum.x)
    corrected_chi = trace_normalized_chi_from_super(corrected_super)
    output_path = CORRECTION_DIR / f"corrected_chi_nbar_{_safe_nbar_stem(point['n_bar'])}.npz"
    np.savez_compressed(
        output_path,
        n_bar=float(point["n_bar"]),
        optimal_xx_correction_angle_rad=float(optimum.x),
        chi_trace_normalized=corrected_chi,
    )
    channel_corrected_points.append({
        "n_bar": point["n_bar"],
        "chi": corrected_chi,
        "optimal_angle": float(optimum.x),
    })
    channel_correction_rows.append({
        "n_bar": point["n_bar"],
        "optimal_xx_correction_angle_rad": float(optimum.x),
        "average_infidelity_before": average_infidelity_from_trace_normalized_chi(before_chi),
        "average_infidelity_after": average_infidelity_from_trace_normalized_chi(corrected_chi),
        "abs_chi_II_XX_before": float(abs(before_chi[II_INDEX, XX_INDEX])),
        "abs_chi_II_XX_after": float(abs(corrected_chi[II_INDEX, XX_INDEX])),
        "chi_XX_XX_before": float(np.real(before_chi[XX_INDEX, XX_INDEX])),
        "chi_XX_XX_after": float(np.real(corrected_chi[XX_INDEX, XX_INDEX])),
        "chi_IX_IX_before": float(np.real(before_chi[PAULI_LABELS.index("IX"), PAULI_LABELS.index("IX")])),
        "chi_IX_IX_after": float(np.real(corrected_chi[PAULI_LABELS.index("IX"), PAULI_LABELS.index("IX")])),
        "optimizer_success": bool(optimum.success),
    })

channel_correction_df = pd.DataFrame(channel_correction_rows).sort_values("n_bar")
if not channel_correction_df.empty:
    channel_correction_df["infidelity_reduction_factor"] = (
        channel_correction_df["average_infidelity_before"]
        / channel_correction_df["average_infidelity_after"]
    )
    channel_correction_df.to_csv(
        CORRECTION_DIR / "xx_angle_correction_summary.csv", index=False
    )
    display(channel_correction_df)

    fig_correction, axes_correction = plt.subplots(1, 3, figsize=(16.0, 4.8))
    axes_correction[0].plot(
        channel_correction_df["n_bar"],
        channel_correction_df["optimal_xx_correction_angle_rad"], linewidth=2.2,
    )
    axes_correction[0].set_ylabel(r"Optimal $XX$ correction angle (rad)")
    axes_correction[0].set_title("Temperature-adaptive calibration")

    for ax, prefix, title in [
        (axes_correction[1], "abs_chi_II_XX", r"$|\chi_{II,XX}|$"),
        (axes_correction[2], "chi_XX_XX", r"$\chi_{XX,XX}$"),
    ]:
        ax.plot(channel_correction_df["n_bar"], channel_correction_df[f"{prefix}_before"], label="before", linewidth=2)
        ax.plot(channel_correction_df["n_bar"], channel_correction_df[f"{prefix}_after"], label="after", linewidth=2)
        ax.set_title(title)
        ax.legend()
    for ax in axes_correction:
        ax.set_xlabel(r"Mean phonon number $\bar n$")
        ax.grid(True, alpha=0.28)
    fig_correction.tight_layout()
    fig_correction.savefig(CORRECTION_DIR / "xx_angle_correction.png", dpi=300, bbox_inches="tight")
    fig_correction.savefig(CORRECTION_DIR / "xx_angle_correction.pdf", bbox_inches="tight")
    plt.show()


# In[21]:


BASE_T_GATE_SIM = 2.0 * np.pi / abs(float(SIMULATION_PARAMS["delta"]))

CONTROL_CANDIDATES = [
    {"name": "baseline", "kind": "baseline", "factor": 1.0},
    *[
        {"name": f"amplitude_{factor:.3f}", "kind": "amplitude", "factor": factor}
        for factor in [0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
        if not np.isclose(factor, 1.0)
    ],
    {"name": "gate_time_0.970", "kind": "gate_time", "factor": 0.97},
    {"name": "gate_time_1.030", "kind": "gate_time", "factor": 1.03},
    {"name": "detuning_0.970", "kind": "detuning", "factor": 0.97},
    {"name": "detuning_1.030", "kind": "detuning", "factor": 1.03},
    {"name": "pulse_sin2_rms", "kind": "pulse", "shape": "sin2", "factor": 1.0},
    {"name": "pulse_blackman_rms", "kind": "pulse", "shape": "blackman", "factor": 1.0},
]

def normalized_control_envelope(shape_name, number_of_points):
    phase = np.linspace(0.0, 1.0, int(number_of_points))
    if shape_name == "sin2":
        envelope = np.sin(np.pi * phase) ** 2
    elif shape_name == "blackman":
        envelope = np.blackman(int(number_of_points))
    else:
        raise ValueError(f"Unknown pulse shape: {shape_name}")
    rms = float(np.sqrt(np.mean(envelope ** 2)))
    if rms <= 0:
        raise ValueError("Pulse RMS must be positive")
    return envelope / rms

def control_candidate_overrides(candidate):
    overrides = {
        "parallel_workers": FAST_PROCESS_WORKERS,
        "show_progress": False,
    }
    kind = candidate["kind"]
    factor = float(candidate.get("factor", 1.0))
    if kind == "baseline":
        return overrides
    if kind == "amplitude":
        overrides["A"] = float(SIMULATION_PARAMS["A"]) * factor
    elif kind == "gate_time":
        overrides["t_gate_sim"] = BASE_T_GATE_SIM * factor
        overrides["t_gate_phys"] = float(SIMULATION_PARAMS["t_gate_phys"]) * factor
    elif kind == "detuning":
        overrides["delta"] = float(SIMULATION_PARAMS["delta"]) * factor
        overrides["t_gate_sim"] = BASE_T_GATE_SIM / factor
        overrides["t_gate_phys"] = float(SIMULATION_PARAMS["t_gate_phys"]) / factor
    elif kind == "pulse":
        envelope = normalized_control_envelope(
            candidate["shape"], SIMULATION_PARAMS["time_points"]
        )
        overrides["A"] = float(SIMULATION_PARAMS["A"]) * factor * envelope
        overrides["t_gate_sim"] = BASE_T_GATE_SIM
    else:
        raise ValueError(f"Unknown control kind: {kind}")
    return overrides

def advanced_qpt_cache_path(directory, condition_name, n_bar):
    safe_condition = condition_name.replace(".", "p").replace("-", "m")
    return directory / f"{safe_condition}__nbar_{_safe_nbar_stem(n_bar)}.npz"

def save_advanced_qpt_point(path, n_bar, condition_name, chi, metadata):
    temporary_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        n_bar=float(n_bar),
        condition=condition_name,
        chi_trace_normalized=chi,
        metadata_json=json.dumps(metadata, default=str),
    )
    temporary_path.replace(path)

def chi_publication_observables(chi):
    ii = float(max(np.real(chi[II_INDEX, II_INDEX]), 1e-15))
    coherent_weight = float(abs(chi[II_INDEX, XX_INDEX]) ** 2 / ii)
    return {
        "chi_II_II": ii,
        "chi_XX_XX": float(np.real(chi[XX_INDEX, XX_INDEX])),
        "abs_chi_II_XX": float(abs(chi[II_INDEX, XX_INDEX])),
        "coherent_XX_equivalent_weight": coherent_weight,
        "chi_IX_IX": float(np.real(chi[PAULI_LABELS.index("IX"), PAULI_LABELS.index("IX")])),
        "chi_XI_XI": float(np.real(chi[PAULI_LABELS.index("XI"), PAULI_LABELS.index("XI")])),
        "average_infidelity": average_infidelity_from_trace_normalized_chi(chi),
        "control_score": float(
            np.real(chi[XX_INDEX, XX_INDEX])
            + coherent_weight
            + np.real(chi[PAULI_LABELS.index("IX"), PAULI_LABELS.index("IX")])
            + np.real(chi[PAULI_LABELS.index("XI"), PAULI_LABELS.index("XI")])
        ),
    }

def calculate_error_channel_batch(n_bar_values, parameter_overrides=None):
    params = dict(SIMULATION_PARAMS)
    if parameter_overrides:
        params.update(parameter_overrides)
    params["n_bar_list"] = [float(value) for value in n_bar_values]
    error_result = mg.generate_error_channel_matrices(
        convention=ERROR_CHANNEL_CONVENTION,
        **params,
    )
    composition = mg.validate_error_channel_composition(
        error_result, desired_convention=ERROR_CHANNEL_CONVENTION
    )
    batch_results = []
    for index, n_bar in enumerate(params["n_bar_list"]):
        chi_raw = np.asarray(error_result["error_chi_matrix_list"][index], dtype=complex)
        raw_trace = np.trace(chi_raw)
        chi = chi_raw / raw_trace
        physicality = mg.choi_physicality_metrics(error_result["S_error_qobj_list"][index])
        metadata = {
            "n_bar": float(n_bar),
            "phonon_dim": int(error_result["results_list"][index]["Nv"]),
            "raw_trace_real": float(np.real(raw_trace)),
            "raw_trace_imag": float(np.imag(raw_trace)),
            "trace_normalized_hermiticity_fro": float(np.linalg.norm(chi - chi.conj().T)),
            "convention_error_fro": float(composition["max_desired_convention_error"]),
            **physicality,
        }
        batch_results.append({"n_bar": float(n_bar), "chi": chi, "metadata": metadata})
    return batch_results

if RUN_PHYSICAL_CONTROL_QPT:
    active_candidates = [
        candidate for candidate in CONTROL_CANDIDATES
        if candidate["kind"] != "baseline"
    ]
    for candidate_index, candidate in enumerate(active_candidates, start=1):
        missing_nbars = [
            n_bar for n_bar in CONTROL_VALIDATION_NBARS
            if FORCE_RECOMPUTE_ADVANCED_QPT
            or not advanced_qpt_cache_path(CONTROL_DIR, candidate["name"], n_bar).exists()
        ]
        if not missing_nbars:
            print(f"[{candidate_index:02d}/{len(active_candidates)}] {candidate['name']}: all cached")
            continue
        print(
            f"[{candidate_index:02d}/{len(active_candidates)}] {candidate['name']}: "
            f"batched QPT for n_bar={missing_nbars} with {FAST_PROCESS_WORKERS} workers"
        )
        started_at = time.perf_counter()
        overrides = control_candidate_overrides(candidate)
        for result in calculate_error_channel_batch(missing_nbars, overrides):
            result["metadata"].update({"candidate": candidate})
            output_path = advanced_qpt_cache_path(
                CONTROL_DIR, candidate["name"], result["n_bar"]
            )
            save_advanced_qpt_point(
                output_path, result["n_bar"], candidate["name"],
                result["chi"], result["metadata"],
            )
        print(f"  completed in {time.perf_counter() - started_at:.1f} s")

control_rows = []
baseline_cptp_by_nbar = {point["n_bar"]: point["chi"] for point in cptp_points}
for n_bar in CONTROL_VALIDATION_NBARS:
    if float(n_bar) in baseline_cptp_by_nbar:
        control_rows.append({
            "n_bar": float(n_bar), "candidate": "baseline", "kind": "baseline",
            "factor": 1.0, **chi_publication_observables(baseline_cptp_by_nbar[float(n_bar)]),
        })
    for candidate in CONTROL_CANDIDATES:
        if candidate["kind"] == "baseline":
            continue
        path = advanced_qpt_cache_path(CONTROL_DIR, candidate["name"], n_bar)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            chi = np.asarray(data["chi_trace_normalized"], dtype=complex)
        control_rows.append({
            "n_bar": float(n_bar), "candidate": candidate["name"],
            "kind": candidate["kind"], "factor": float(candidate.get("factor", 1.0)),
            **chi_publication_observables(chi),
        })

physical_control_df = pd.DataFrame(control_rows)
if len(physical_control_df) <= len(CONTROL_VALIDATION_NBARS):
    print(
        f"Physical-control QPT cache is not populated. Set RUN_PHYSICAL_CONTROL_QPT=True "
        f"to run {len(CONTROL_VALIDATION_NBARS) * (len(CONTROL_CANDIDATES) - 1)} resumable QPT points."
    )
else:
    physical_control_df.to_csv(CONTROL_DIR / "physical_control_qpt_summary.csv", index=False)
    best_control_df = (
        physical_control_df.sort_values("control_score")
        .groupby("n_bar", as_index=False)
        .first()
    )
    display(best_control_df)
    fig_control, ax_control = plt.subplots(figsize=(9.0, 5.4))
    for candidate, group in physical_control_df.groupby("candidate"):
        ax_control.semilogy(group["n_bar"], group["control_score"], marker="o", label=candidate)
    ax_control.set_xlabel(r"Mean phonon number $\bar n$")
    ax_control.set_ylabel("χ-based control score")
    ax_control.set_title("Physical control re-simulation: before/after candidates")
    ax_control.grid(True, which="both", alpha=0.28)
    ax_control.legend(ncol=2, fontsize=8)
    fig_control.tight_layout()
    fig_control.savefig(CONTROL_DIR / "physical_control_qpt_comparison.png", dpi=300, bbox_inches="tight")
    fig_control.savefig(CONTROL_DIR / "physical_control_qpt_comparison.pdf", bbox_inches="tight")
    plt.show()


# ### 10. $h_{XX}(\bar n)$からのdrive amplitude校正と全Hamiltonian再QPT
# 
# error-channel規約は $\mathcal S_{\rm error}=\mathcal S_{\rm actual}\mathcal S_{\rm ideal}^{-1}$、理想ゲートは
# $U_{\rm MS}=\exp(+i\phi_*XX)$、$\phi_*=\pi/4$ です。generatorを
# $K_H\rho=-ih_{XX}[XX,\rho]$ と書くため、$h_{XX}>0$ は実ゲート角
# $\phi_{\rm actual}\simeq\phi_*-h_{XX}$ のunder-rotationを表します。
# 
# 閉じた位相空間軌道で $\phi\propto A^2$ と近似すると、最初の補正振幅は
# 
# \[
# A_{\rm next}=A_{\rm current}
# \sqrt{\frac{\phi_*}{\phi_*-h_{XX}}}
# \]
# 
# です。この式は初期推定にのみ使い、補正振幅を実際にプロジェクトのmaster-equation Hamiltonianへ戻して16入力状態のQPTを再実行します。再QPT後の残留 $h_{XX}$ を同じ式へ戻せるため、`HXX_MAX_FEEDBACK_ITERATIONS` 回まで閉ループ校正できます。各QPTは独立NPZへ保存され、中断後に再開できます。
# 
# Kirchhoff–Wilhelm–Motzoi, PRX Quantum **6**, 010328 (2025) の Eqs. (32), (35), (41) も実装し、$\Omega_2/\Omega_{\rm LD}$、$\Omega_4/\Omega_{\rm LD}$ と比較します。同論文の $\Omega$ はcarrier drive、本コードの $A$ はfirst-sideband couplingですが、$\eta$ 固定なら振幅比では換算係数が消えます。ただし同論文の $K=\nu T/(2\pi)$ と $L=\delta_{\rm bich}T/(2\pi)$ は現在の有効Hamiltonian入力に含まれません。したがって、既定では一周条件 $K-L=1$ の有効域を帯で表示し、実験値がある場合だけ `KIRCHHOFF_REFERENCE_K` を指定して一点比較します。

# In[ ]:


from pathlib import Path
import importlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

import drive_amplitude_calibration as dac
dac = importlib.reload(dac)

# この節だけを再実行しても保存先と軽量な入力表を復元できるようにする。
OUTPUT_DIR = Path(globals().get("OUTPUT_DIR", "results/chi_error_element_fit"))
ADVANCED_DIR = Path(globals().get(
    "ADVANCED_DIR", OUTPUT_DIR / "advanced_publication_validation"
))
ADVANCED_DIR.mkdir(parents=True, exist_ok=True)

if "SIMULATION_PARAMS" not in globals():
    manifest_path = OUTPUT_DIR / "publication_reproducibility_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "SIMULATION_PARAMS is unavailable. Run Cells 2-4 first, or generate "
            "publication_reproducibility_manifest.json by running the notebook above."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    SIMULATION_PARAMS = manifest["simulation_parameters"]
    print(f"Loaded SIMULATION_PARAMS from {manifest_path}")

if "generator_df" not in globals():
    generator_summary_path = (
        ADVANCED_DIR / "error_generator" / "error_generator_summary.csv"
    )
    if not generator_summary_path.exists():
        raise RuntimeError(
            "generator_df is unavailable. Run Cell 35 first, or generate "
            "error_generator_summary.csv by running the notebook above."
        )
    generator_df = pd.read_csv(generator_summary_path)
    print(f"Loaded generator_df from {generator_summary_path}")

DRIVE_CALIBRATION_DIR = ADVANCED_DIR / "hxx_drive_amplitude_calibration"
DRIVE_CALIBRATION_QPT_DIR = DRIVE_CALIBRATION_DIR / "qpt_cache"
DRIVE_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_CALIBRATION_QPT_DIR.mkdir(parents=True, exist_ok=True)

# Falseでも81点のhXX由来補正量とKirchhoff比較は直ちに生成される。
# Trueの場合のみ、下の代表温度でfull master-equation QPTを実行する。
RUN_HXX_DRIVE_CALIBRATION_QPT = bool(
    _config("RUN_HXX_DRIVE_CALIBRATION_QPT", False)
)
FORCE_RECOMPUTE_HXX_DRIVE_QPT = bool(
    _config("FORCE_RECOMPUTE_HXX_DRIVE_QPT", False)
)
HXX_DRIVE_CALIBRATION_NBARS = list(_config(
    "HXX_DRIVE_CALIBRATION_NBARS",
    [0.01, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0],
))
HXX_MAX_FEEDBACK_ITERATIONS = int(
    _config("HXX_MAX_FEEDBACK_ITERATIONS", 2)
)
HXX_CONVERGENCE_TOL_RAD = float(
    _config("HXX_CONVERGENCE_TOL_RAD", 2e-3)
)
HXX_MAX_AMPLITUDE_FACTOR = float(
    _config("HXX_MAX_AMPLITUDE_FACTOR", 1.6)
)
TARGET_XX_ANGLE_RAD = float(_config("TARGET_XX_ANGLE_RAD", np.pi / 4.0))

# Kirchhoff et al. のK,L。実験のmotional-mode周波数が分かる場合、
# K = f_mode[Hz] * t_gate[s] を設定する。一周条件では L = K - 1。
KIRCHHOFF_REFERENCE_K = _config("KIRCHHOFF_REFERENCE_K", None)
KIRCHHOFF_LOOP_NUMBER = float(_config("KIRCHHOFF_LOOP_NUMBER", 1.0))
KIRCHHOFF_SCAN_K_MAX = float(_config("KIRCHHOFF_SCAN_K_MAX", 250.0))

if generator_df.empty:
    drive_amplitude_prediction_df = pd.DataFrame()
    print("error-generator result is required before drive calibration.")
else:
    baseline_amplitude_array = np.asarray(SIMULATION_PARAMS["A"], dtype=float)
    if baseline_amplitude_array.ndim != 0:
        raise ValueError("hXX amplitude feedback requires a scalar baseline A")
    BASE_DRIVE_AMPLITUDE = float(baseline_amplitude_array)

    prediction_rows = []
    for row in generator_df.sort_values("n_bar").itertuples():
        corrected_amplitude = dac.amplitude_update_from_hxx(
            BASE_DRIVE_AMPLITUDE,
            row.h_XX_rad_per_gate,
            TARGET_XX_ANGLE_RAD,
        )
        prediction_rows.append({
            "n_bar": float(row.n_bar),
            "h_XX_baseline_rad_per_gate": float(row.h_XX_rad_per_gate),
            "inferred_actual_xx_angle_rad": float(
                TARGET_XX_ANGLE_RAD - row.h_XX_rad_per_gate
            ),
            "A_baseline": BASE_DRIVE_AMPLITUDE,
            "A_hxx_first_update": corrected_amplitude,
            "A_hxx_factor": corrected_amplitude / BASE_DRIVE_AMPLITUDE,
        })
    drive_amplitude_prediction_df = pd.DataFrame(prediction_rows)
    drive_amplitude_prediction_df.to_csv(
        DRIVE_CALIBRATION_DIR / "hxx_drive_amplitude_prediction.csv", index=False
    )

eta_for_kirchhoff = float(SIMULATION_PARAMS["eta"])
minimum_scan_k = max(
    KIRCHHOFF_LOOP_NUMBER + 1e-6,
    np.sqrt(KIRCHHOFF_LOOP_NUMBER) / eta_for_kirchhoff,
)
kirchhoff_rows = []
for K_value in np.geomspace(minimum_scan_k, KIRCHHOFF_SCAN_K_MAX, 400):
    L_value = K_value - KIRCHHOFF_LOOP_NUMBER
    try:
        ratios = dac.kirchhoff_renormalization_ratios(
            K_value, L_value, eta_for_kirchhoff
        )
    except ValueError:
        continue
    kirchhoff_rows.append({
        "K": float(K_value),
        "L": float(L_value),
        "K_minus_L": float(KIRCHHOFF_LOOP_NUMBER),
        "eta": eta_for_kirchhoff,
        **ratios,
    })
kirchhoff_scan_df = pd.DataFrame(kirchhoff_rows)
kirchhoff_scan_df.to_csv(
    DRIVE_CALIBRATION_DIR / "kirchhoff_renormalization_scan.csv", index=False
)

kirchhoff_reference = None
if KIRCHHOFF_REFERENCE_K is not None:
    reference_K = float(KIRCHHOFF_REFERENCE_K)
    reference_L = reference_K - KIRCHHOFF_LOOP_NUMBER
    kirchhoff_reference = {
        "K": reference_K,
        "L": reference_L,
        **dac.kirchhoff_renormalization_ratios(
            reference_K, reference_L, eta_for_kirchhoff
        ),
    }
    pd.DataFrame([kirchhoff_reference]).to_csv(
        DRIVE_CALIBRATION_DIR / "kirchhoff_reference_point.csv", index=False
    )

if not drive_amplitude_prediction_df.empty:
    kirchhoff_comparison_df = drive_amplitude_prediction_df.copy()
    if not kirchhoff_scan_df.empty:
        for column in ["omega_2_over_omega_ld", "omega_4_over_omega_ld"]:
            kirchhoff_comparison_df[f"kirchhoff_{column}_scan_min"] = float(
                kirchhoff_scan_df[column].min()
            )
            kirchhoff_comparison_df[f"kirchhoff_{column}_scan_max"] = float(
                kirchhoff_scan_df[column].max()
            )
    if kirchhoff_reference is None:
        kirchhoff_comparison_df["kirchhoff_reference_status"] = (
            "scan_only__set_KIRCHHOFF_REFERENCE_K_for_direct_comparison"
        )
        kirchhoff_comparison_df["kirchhoff_omega_4_reference_ratio"] = np.nan
    else:
        kirchhoff_comparison_df["kirchhoff_reference_status"] = "configured"
        kirchhoff_comparison_df["kirchhoff_omega_4_reference_ratio"] = (
            kirchhoff_reference["omega_4_over_omega_ld"]
        )
    kirchhoff_comparison_df.to_csv(
        DRIVE_CALIBRATION_DIR / "hxx_vs_kirchhoff_amplitude_comparison.csv",
        index=False,
    )

    fig_prediction, ax_prediction = plt.subplots(figsize=(9.2, 5.4))
    ax_prediction.plot(
        drive_amplitude_prediction_df["n_bar"],
        drive_amplitude_prediction_df["A_hxx_factor"],
        linewidth=2.4,
        color="#0072B2",
        label=r"$h_{XX}$ feedback: $A_{\rm next}/A_0$",
    )
    if not kirchhoff_scan_df.empty:
        lower = float(kirchhoff_scan_df["omega_4_over_omega_ld"].min())
        upper = float(kirchhoff_scan_df["omega_4_over_omega_ld"].max())
        ax_prediction.axhspan(
            lower, upper, alpha=0.20, color="#D55E00",
            label=(
                r"Kirchhoff Eq. (41), $K-L=1$ valid scan "
                f"({kirchhoff_scan_df['K'].min():.1f}<=K<={kirchhoff_scan_df['K'].max():.0f})"
            ),
        )
    if kirchhoff_reference is not None:
        ax_prediction.axhline(
            kirchhoff_reference["omega_4_over_omega_ld"],
            color="#D55E00", linestyle="--", linewidth=2,
            label=(
                r"Kirchhoff Eq. (41) reference: "
                f"K={kirchhoff_reference['K']:.3g}, L={kirchhoff_reference['L']:.3g}"
            ),
        )
    ax_prediction.axhline(1.0, color="black", linewidth=0.8)
    ax_prediction.set_xlabel(r"Mean phonon number $\bar n$")
    ax_prediction.set_ylabel("Drive-amplitude renormalization factor")
    ax_prediction.set_title(r"Temperature-adaptive $h_{XX}$ calibration vs Kirchhoff formula")
    ax_prediction.grid(True, alpha=0.28)
    ax_prediction.legend(fontsize=8)
    fig_prediction.tight_layout()
    fig_prediction.savefig(
        DRIVE_CALIBRATION_DIR / "hxx_vs_kirchhoff_amplitude_factor.png",
        dpi=300, bbox_inches="tight",
    )
    fig_prediction.savefig(
        DRIVE_CALIBRATION_DIR / "hxx_vs_kirchhoff_amplitude_factor.pdf",
        bbox_inches="tight",
    )
    plt.show()
    display(drive_amplitude_prediction_df.iloc[[0, 1, 4, 16, 40, -1]])


# In[ ]:


import hashlib
import importlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize

import drive_calibration_qpt_analysis as dcqa
dcqa = importlib.reload(dcqa)

# Cell 31を飛ばしてもキャッシュ生成と既定並列数が壊れないようにする。
ERROR_CHANNEL_CONVENTION = globals().get(
    "ERROR_CHANNEL_CONVENTION", "undo_before_actual"
)
CPTP_TOLERANCE = float(globals().get("CPTP_TOLERANCE", 1e-11))
CPTP_MAX_ITERATIONS = int(globals().get("CPTP_MAX_ITERATIONS", 5000))
if "FAST_PROCESS_WORKERS" not in globals():
    available_cores = os.cpu_count() or 1
    FAST_PROCESS_WORKERS = int(os.environ.get(
        "MS_GATE_WORKERS", max(1, min(available_cores - 1, 4))
    ))


def _drive_calibration_json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _drive_calibration_nbar_stem(n_bar):
    return f"{float(n_bar):.12g}".replace("-", "m").replace(".", "p")


def generator_observables_from_trace_normalized_chi(chi):
    # Cell 33/35に依存せず、同一規約でCPTP射影とgenerator分解を行う。
    return dcqa.extract_xx_generator_observables(
        chi,
        cptp_tolerance=CPTP_TOLERANCE,
        cptp_max_iterations=CPTP_MAX_ITERATIONS,
    )


def drive_feedback_cache_path(n_bar, iteration, amplitude):
    parameter_signature = {
        key: SIMULATION_PARAMS[key]
        for key in [
            "delta", "rho0", "time_points", "t_gate_phys",
            "heating_rate_phys", "dephasing_rate_phys", "T2_star",
            "rayleigh_rate_phys", "raman_rate_phys", "eta", "use_full_order",
        ]
    }
    payload = {
        "n_bar": float(n_bar),
        "iteration": int(iteration),
        "amplitude": float(amplitude),
        "parameters": parameter_signature,
        "convention": ERROR_CHANNEL_CONVENTION,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            default=_drive_calibration_json_safe,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return DRIVE_CALIBRATION_QPT_DIR / (
        f"hxx_feedback_i{int(iteration):02d}__nbar_"
        f"{_drive_calibration_nbar_stem(n_bar)}__{digest}.npz"
    )


def resolve_drive_feedback_cache_path(n_bar, iteration, amplitude):
    """Resolve historical caches despite harmless CSV float round-tripping."""

    exact_path = drive_feedback_cache_path(n_bar, iteration, amplitude)
    if exact_path.exists():
        return exact_path
    pattern = (
        f"hxx_feedback_i{int(iteration):02d}__nbar_"
        f"{_drive_calibration_nbar_stem(n_bar)}__*.npz"
    )
    for candidate in sorted(DRIVE_CALIBRATION_QPT_DIR.glob(pattern)):
        try:
            with np.load(candidate, allow_pickle=False) as data:
                cached_n_bar = float(np.asarray(data["n_bar"]).item())
                metadata = json.loads(
                    str(np.asarray(data["metadata_json"]).item())
                )
            cached_amplitude = float(metadata["A_calibrated"])
            cached_iteration = int(metadata.get("iteration", iteration))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            np.isclose(cached_n_bar, n_bar, rtol=0.0, atol=1e-12)
            and cached_iteration == int(iteration)
            and np.isclose(
                cached_amplitude, amplitude, rtol=1e-12, atol=1e-14
            )
        ):
            return candidate
    return exact_path


drive_feedback_rows = []
if not drive_amplitude_prediction_df.empty:
    for n_bar in HXX_DRIVE_CALIBRATION_NBARS:
        baseline_match = generator_df.loc[
            np.isclose(generator_df["n_bar"].astype(float), float(n_bar))
        ]
        if baseline_match.empty:
            warnings.warn(f"No baseline generator point for n_bar={n_bar:g}")
            continue
        current_amplitude = BASE_DRIVE_AMPLITUDE
        current_h_xx = float(baseline_match.iloc[0]["h_XX_rad_per_gate"])

        for iteration in range(1, HXX_MAX_FEEDBACK_ITERATIONS + 1):
            next_amplitude = dac.amplitude_update_from_hxx(
                current_amplitude, current_h_xx, TARGET_XX_ANGLE_RAD
            )
            amplitude_factor = next_amplitude / BASE_DRIVE_AMPLITUDE
            if amplitude_factor > HXX_MAX_AMPLITUDE_FACTOR:
                raise ValueError(
                    f"n_bar={n_bar:g}: requested A/A0={amplitude_factor:.3f} "
                    f"exceeds HXX_MAX_AMPLITUDE_FACTOR={HXX_MAX_AMPLITUDE_FACTOR:.3f}"
                )

            cache_path = resolve_drive_feedback_cache_path(
                n_bar, iteration, next_amplitude
            )
            if FORCE_RECOMPUTE_HXX_DRIVE_QPT or not cache_path.exists():
                if not RUN_HXX_DRIVE_CALIBRATION_QPT:
                    print(
                        f"pending n_bar={n_bar:g}, iteration={iteration}: "
                        f"A={next_amplitude:.8f} (A/A0={amplitude_factor:.4f})"
                    )
                    break
                print(
                    f"hXX feedback QPT: n_bar={n_bar:g}, iteration={iteration}, "
                    f"A={next_amplitude:.8f}, workers={FAST_PROCESS_WORKERS}"
                )
                started_at = time.perf_counter()
                qpt_result = dcqa.calculate_error_channel_batch(
                    [n_bar],
                    SIMULATION_PARAMS,
                    {
                        "A": next_amplitude,
                        "parallel_workers": FAST_PROCESS_WORKERS,
                        "show_progress": False,
                    },
                    convention=ERROR_CHANNEL_CONVENTION,
                )[0]
                qpt_result["metadata"].update({
                    "calibration_method": "hxx_quadratic_feedback",
                    "iteration": iteration,
                    "input_h_XX_rad_per_gate": current_h_xx,
                    "A_baseline": BASE_DRIVE_AMPLITUDE,
                    "A_calibrated": next_amplitude,
                    "A_factor": amplitude_factor,
                })
                dcqa.save_qpt_point(
                    cache_path,
                    n_bar,
                    f"hxx_feedback_iteration_{iteration}",
                    qpt_result["chi"],
                    qpt_result["metadata"],
                )
                print(f"  completed in {time.perf_counter() - started_at:.1f} s")

            with np.load(cache_path, allow_pickle=False) as data:
                calibrated_chi = np.asarray(
                    data["chi_trace_normalized"], dtype=complex
                )
            observables = generator_observables_from_trace_normalized_chi(
                calibrated_chi
            )
            drive_feedback_rows.append({
                "n_bar": float(n_bar),
                "iteration": int(iteration),
                "input_h_XX_rad_per_gate": current_h_xx,
                "A_calibrated": next_amplitude,
                "A_factor": amplitude_factor,
                "cache_path": str(cache_path),
                **{key: value for key, value in observables.items() if key != "projected_chi"},
            })
            current_amplitude = next_amplitude
            current_h_xx = observables["h_XX_rad_per_gate"]
            if abs(current_h_xx) <= HXX_CONVERGENCE_TOL_RAD:
                break

drive_feedback_qpt_df = pd.DataFrame(drive_feedback_rows)
if not drive_feedback_qpt_df.empty:
    drive_feedback_qpt_df = drive_feedback_qpt_df.sort_values(
        ["n_bar", "iteration"]
    ).reset_index(drop=True)
    drive_feedback_qpt_df.to_csv(
        DRIVE_CALIBRATION_DIR / "hxx_drive_feedback_qpt_iterations.csv",
        index=False,
    )
    display(drive_feedback_qpt_df)
else:
    print(
        "hXX physical-feedback QPT cache is empty. Set "
        "RUN_HXX_DRIVE_CALIBRATION_QPT=True to run the resumable calibration."
    )


# In[ ]:


from pathlib import Path
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drive_calibration_qpt_analysis as dcqa
dcqa = importlib.reload(dcqa)

OUTPUT_DIR = Path(globals().get("OUTPUT_DIR", "results/chi_error_element_fit"))
ADVANCED_DIR = Path(globals().get(
    "ADVANCED_DIR", OUTPUT_DIR / "advanced_publication_validation"
))
DRIVE_CALIBRATION_DIR = Path(globals().get(
    "DRIVE_CALIBRATION_DIR",
    ADVANCED_DIR / "hxx_drive_amplitude_calibration",
))
CPTP_DIR = Path(globals().get(
    "CPTP_DIR", ADVANCED_DIR / "cptp_projection"
))
HXX_DRIVE_CALIBRATION_NBARS = globals().get(
    "HXX_DRIVE_CALIBRATION_NBARS",
    [0.01, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0],
)
HXX_CONVERGENCE_TOL_RAD = float(globals().get(
    "HXX_CONVERGENCE_TOL_RAD", 2e-3
))
CPTP_TOLERANCE = float(globals().get("CPTP_TOLERANCE", 1e-11))
CPTP_MAX_ITERATIONS = int(globals().get("CPTP_MAX_ITERATIONS", 5000))

if "generator_df" not in globals():
    generator_df = pd.read_csv(
        ADVANCED_DIR / "error_generator" / "error_generator_summary.csv"
    )
if "drive_feedback_qpt_df" not in globals():
    feedback_csv = (
        DRIVE_CALIBRATION_DIR / "hxx_drive_feedback_qpt_iterations.csv"
    )
    drive_feedback_qpt_df = (
        pd.read_csv(feedback_csv) if feedback_csv.exists() else pd.DataFrame()
    )


def _result_nbar_stem(n_bar):
    return str(float(n_bar)).replace("-", "m").replace(".", "p")


baseline_calibration_rows = []
for n_bar in HXX_DRIVE_CALIBRATION_NBARS:
    generator_match = generator_df.loc[
        np.isclose(generator_df["n_bar"].astype(float), float(n_bar))
    ]
    cptp_path = CPTP_DIR / (
        f"cptp_chi_nbar_{_result_nbar_stem(n_bar)}.npz"
    )
    if generator_match.empty or not cptp_path.exists():
        continue
    generator_row = generator_match.iloc[0]
    with np.load(cptp_path, allow_pickle=False) as data:
        baseline_chi = np.asarray(data["chi_trace_normalized"], dtype=complex)
    baseline_observables = dcqa.extract_xx_generator_observables(
        baseline_chi,
        cptp_tolerance=CPTP_TOLERANCE,
        cptp_max_iterations=CPTP_MAX_ITERATIONS,
    )
    baseline_calibration_rows.append({
        "n_bar": float(n_bar),
        "h_XX_before_rad_per_gate": float(generator_row["h_XX_rad_per_gate"]),
        "gamma_XX_before_per_gate": float(generator_row["gamma_XX_per_gate"]),
        "average_infidelity_before": baseline_observables["average_infidelity"],
        "abs_chi_II_XX_before": baseline_observables["abs_chi_II_XX"],
        "chi_XX_XX_before": baseline_observables["chi_XX_XX"],
    })
baseline_calibration_df = pd.DataFrame(baseline_calibration_rows)

if drive_feedback_qpt_df.empty:
    final_drive_calibration_df = pd.DataFrame()
else:
    final_drive_calibration_df = (
        drive_feedback_qpt_df.sort_values(["n_bar", "iteration"])
        .groupby("n_bar", as_index=False)
        .tail(1)
        .rename(columns={
            "h_XX_rad_per_gate": "h_XX_after_rad_per_gate",
            "gamma_XX_per_gate": "gamma_XX_after_per_gate",
            "average_infidelity": "average_infidelity_after",
            "abs_chi_II_XX": "abs_chi_II_XX_after",
            "chi_XX_XX": "chi_XX_XX_after",
        })
        .merge(baseline_calibration_df, on="n_bar", how="left")
        .sort_values("n_bar")
        .reset_index(drop=True)
    )
    final_drive_calibration_df["abs_h_XX_reduction_factor"] = (
        np.abs(final_drive_calibration_df["h_XX_before_rad_per_gate"])
        / np.maximum(
            np.abs(final_drive_calibration_df["h_XX_after_rad_per_gate"]), 1e-15
        )
    )
    final_drive_calibration_df["infidelity_reduction_factor"] = (
        final_drive_calibration_df["average_infidelity_before"]
        / np.maximum(final_drive_calibration_df["average_infidelity_after"], 1e-15)
    )
    final_drive_calibration_df["h_XX_converged"] = (
        np.abs(final_drive_calibration_df["h_XX_after_rad_per_gate"])
        <= HXX_CONVERGENCE_TOL_RAD
    )
    final_drive_calibration_df.to_csv(
        DRIVE_CALIBRATION_DIR / "hxx_drive_calibration_final_summary.csv",
        index=False,
    )
    display(final_drive_calibration_df)

    fig_feedback, axes_feedback = plt.subplots(2, 2, figsize=(13.0, 9.0))
    for ax, before, after, title, log_scale in [
        (
            axes_feedback[0, 0], "h_XX_before_rad_per_gate",
            "h_XX_after_rad_per_gate", r"Hamiltonian $h_{XX}$", False,
        ),
        (
            axes_feedback[0, 1], "gamma_XX_before_per_gate",
            "gamma_XX_after_per_gate", r"Stochastic $\gamma_{XX}$", True,
        ),
        (
            axes_feedback[1, 0], "average_infidelity_before",
            "average_infidelity_after", "Average infidelity", True,
        ),
    ]:
        ax.plot(final_drive_calibration_df["n_bar"], final_drive_calibration_df[before], "o-", label="before")
        ax.plot(final_drive_calibration_df["n_bar"], final_drive_calibration_df[after], "o-", label="after")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.28)
    axes_feedback[1, 1].plot(
        final_drive_calibration_df["n_bar"],
        final_drive_calibration_df["A_factor"],
        "o-", color="#009E73",
    )
    axes_feedback[1, 1].set_title(r"Validated drive factor $A/A_0$")
    axes_feedback[1, 1].grid(True, alpha=0.28)
    for ax in axes_feedback.flat:
        ax.set_xlabel(r"Mean phonon number $\bar n$")
    fig_feedback.suptitle("Full-Hamiltonian QPT after hXX drive calibration", y=1.01)
    fig_feedback.tight_layout()
    fig_feedback.savefig(
        DRIVE_CALIBRATION_DIR / "hxx_drive_calibration_before_after.png",
        dpi=300, bbox_inches="tight",
    )
    fig_feedback.savefig(
        DRIVE_CALIBRATION_DIR / "hxx_drive_calibration_before_after.pdf",
        bbox_inches="tight",
    )
    plt.show()

hxx_drive_completed_nbars = (
    int(final_drive_calibration_df["n_bar"].nunique())
    if not final_drive_calibration_df.empty else 0
)
hxx_drive_expected_nbars = len(HXX_DRIVE_CALIBRATION_NBARS)
hxx_drive_converged_nbars = (
    int(final_drive_calibration_df["h_XX_converged"].sum())
    if not final_drive_calibration_df.empty else 0
)
print(
    f"hXX physical calibration: {hxx_drive_completed_nbars}/{hxx_drive_expected_nbars} "
    f"temperatures re-QPT, {hxx_drive_converged_nbars} converged"
)


# ### 11. パラメータ頑健性
# 
# 代表3温度について、$eta$、$A/\delta$、gate time、motional dephasing rateを個別に変化させます。各点は独立NPZへ保存され、再開可能です。
# 
# 外側で複数QPTを同時起動せず、1つのQPT内の16入力状態をプロセス並列化することで、メモリ過剰使用とBLASのnested parallelismを避けています。

# In[22]:


ROBUSTNESS_CONDITIONS = [
    {"name": "baseline", "parameter": "baseline", "factor": 1.0, "overrides": {}},
]
for factor in [0.8, 1.2]:
    ROBUSTNESS_CONDITIONS.append({
        "name": f"eta_{factor:.2f}", "parameter": "eta", "factor": factor,
        "overrides": {"eta": float(SIMULATION_PARAMS["eta"]) * factor},
    })
for factor in [0.9, 1.1]:
    ROBUSTNESS_CONDITIONS.append({
        "name": f"A_over_delta_{factor:.2f}", "parameter": "A_over_delta", "factor": factor,
        "overrides": {"A": float(SIMULATION_PARAMS["A"]) * factor},
    })
for factor in [0.95, 1.05]:
    ROBUSTNESS_CONDITIONS.append({
        "name": f"gate_time_{factor:.2f}", "parameter": "gate_time", "factor": factor,
        "overrides": {
            "t_gate_sim": BASE_T_GATE_SIM * factor,
            "t_gate_phys": float(SIMULATION_PARAMS["t_gate_phys"]) * factor,
        },
    })
for factor in [0.0, 0.5, 2.0]:
    ROBUSTNESS_CONDITIONS.append({
        "name": f"motional_dephasing_{factor:.2f}",
        "parameter": "motional_dephasing_rate", "factor": factor,
        "overrides": {
            "dephasing_rate_phys": float(SIMULATION_PARAMS["dephasing_rate_phys"]) * factor
        },
    })

if RUN_PARAMETER_ROBUSTNESS_QPT:
    active_conditions = [
        condition for condition in ROBUSTNESS_CONDITIONS
        if condition["parameter"] != "baseline"
    ]
    for condition_index, condition in enumerate(active_conditions, start=1):
        missing_nbars = [
            n_bar for n_bar in ROBUSTNESS_NBARS
            if FORCE_RECOMPUTE_ADVANCED_QPT
            or not advanced_qpt_cache_path(ROBUSTNESS_DIR, condition["name"], n_bar).exists()
        ]
        if not missing_nbars:
            print(f"[{condition_index:02d}/{len(active_conditions)}] {condition['name']}: all cached")
            continue
        print(
            f"[{condition_index:02d}/{len(active_conditions)}] {condition['name']}: "
            f"batched QPT for n_bar={missing_nbars} with {FAST_PROCESS_WORKERS} workers"
        )
        started_at = time.perf_counter()
        overrides = dict(condition["overrides"])
        overrides.update({"parallel_workers": FAST_PROCESS_WORKERS, "show_progress": False})
        for result in calculate_error_channel_batch(missing_nbars, overrides):
            result["metadata"].update({
                "condition": condition["name"],
                "parameter": condition["parameter"],
                "factor": condition["factor"],
            })
            output_path = advanced_qpt_cache_path(
                ROBUSTNESS_DIR, condition["name"], result["n_bar"]
            )
            save_advanced_qpt_point(
                output_path, result["n_bar"], condition["name"],
                result["chi"], result["metadata"],
            )
        print(f"  completed in {time.perf_counter() - started_at:.1f} s")

robustness_rows = []
for n_bar in ROBUSTNESS_NBARS:
    if float(n_bar) in baseline_cptp_by_nbar:
        robustness_rows.append({
            "n_bar": float(n_bar), "condition": "baseline",
            "parameter": "baseline", "factor": 1.0,
            **chi_publication_observables(baseline_cptp_by_nbar[float(n_bar)]),
        })
    for condition in ROBUSTNESS_CONDITIONS:
        if condition["parameter"] == "baseline":
            continue
        path = advanced_qpt_cache_path(ROBUSTNESS_DIR, condition["name"], n_bar)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            chi = np.asarray(data["chi_trace_normalized"], dtype=complex)
        robustness_rows.append({
            "n_bar": float(n_bar), "condition": condition["name"],
            "parameter": condition["parameter"], "factor": condition["factor"],
            **chi_publication_observables(chi),
        })

parameter_robustness_df = pd.DataFrame(robustness_rows)
if len(parameter_robustness_df) <= len(ROBUSTNESS_NBARS):
    expected_jobs = len(ROBUSTNESS_NBARS) * (len(ROBUSTNESS_CONDITIONS) - 1)
    print(
        f"Parameter-robustness cache is not populated. Set RUN_PARAMETER_ROBUSTNESS_QPT=True "
        f"to run {expected_jobs} resumable QPT points."
    )
else:
    parameter_robustness_df.to_csv(
        ROBUSTNESS_DIR / "parameter_robustness_summary.csv", index=False
    )
    baseline_scores = (
        parameter_robustness_df[parameter_robustness_df["parameter"] == "baseline"]
        .set_index("n_bar")["control_score"]
    )
    parameter_robustness_df["score_relative_to_baseline"] = parameter_robustness_df.apply(
        lambda row: row["control_score"] / baseline_scores.loc[row["n_bar"]], axis=1
    )
    display(parameter_robustness_df)

    parameters_to_plot = ["eta", "A_over_delta", "gate_time", "motional_dephasing_rate"]
    fig_robust, axes_robust = plt.subplots(2, 2, figsize=(13.0, 9.0))
    for ax, parameter in zip(axes_robust.ravel(), parameters_to_plot):
        panel = parameter_robustness_df[parameter_robustness_df["parameter"] == parameter]
        for n_bar, group in panel.groupby("n_bar"):
            ax.plot(group["factor"], group["score_relative_to_baseline"], marker="o", label=rf"$\bar n={n_bar:g}$")
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_title(parameter)
        ax.set_xlabel("factor relative to baseline")
        ax.set_ylabel("χ score / baseline")
        ax.grid(True, alpha=0.28)
        ax.legend()
    fig_robust.tight_layout()
    fig_robust.savefig(ROBUSTNESS_DIR / "parameter_robustness.png", dpi=300, bbox_inches="tight")
    fig_robust.savefig(ROBUSTNESS_DIR / "parameter_robustness.pdf", bbox_inches="tight")
    plt.show()


# ### 12. $(2\bar n+1)$に基づく物理モデルfit
# 
# 任意の4次多項式を主張の根拠にせず、熱占有に自然な
# 
# \[
# x=2\bar n+1,\qquad q=\bar n(\bar n+1)
# \]
# 
# を用います。`thermal_linear`、`thermal_bosonic_2`、`thermal_bosonic_3`を物理候補とし、従来の`generic_quartic`は参照モデルとしてAICc・BIC・5-fold CVで比較します。

# In[23]:


def physical_design_matrix(n_bar, model_name):
    n_bar = np.asarray(n_bar, dtype=float)
    thermal_x = 2.0 * n_bar + 1.0
    thermal_pair = n_bar * (n_bar + 1.0)
    if model_name == "thermal_linear":
        return np.column_stack([np.ones_like(n_bar), thermal_x]), ["1", "(2n+1)"]
    if model_name == "thermal_bosonic_2":
        return np.column_stack([np.ones_like(n_bar), thermal_x, thermal_pair]), ["1", "(2n+1)", "n(n+1)"]
    if model_name == "thermal_bosonic_3":
        return np.column_stack([
            np.ones_like(n_bar), thermal_x, thermal_pair, thermal_x * thermal_pair
        ]), ["1", "(2n+1)", "n(n+1)", "(2n+1)n(n+1)"]
    if model_name == "generic_quartic":
        return np.column_stack([n_bar ** power for power in range(5)]), [
            "1", "n", "n^2", "n^3", "n^4"
        ]
    raise ValueError(f"Unknown physical model: {model_name}")

def fit_named_linear_model(n_bar, values, model_name):
    design, term_names = physical_design_matrix(n_bar, model_name)
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = design @ coefficients
    residual = np.asarray(values) - fitted
    rss = max(float(np.sum(residual ** 2)), np.finfo(float).tiny)
    n_observations = len(values)
    parameter_count = design.shape[1]
    aic = n_observations * np.log(rss / n_observations) + 2 * parameter_count
    aicc = aic + 2 * parameter_count * (parameter_count + 1) / (
        n_observations - parameter_count - 1
    )
    bic = n_observations * np.log(rss / n_observations) + parameter_count * np.log(n_observations)
    equation_terms = []
    for coefficient, term_name in zip(coefficients, term_names):
        equation_terms.append(f"({coefficient:.8e})*{term_name}")
    return {
        "coefficients": coefficients,
        "term_names": term_names,
        "fitted": fitted,
        "residual": residual,
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "AICc": float(aicc),
        "BIC": float(bic),
        "equation": "y(n) = " + " + ".join(equation_terms),
    }

def physical_model_cv_rmse(n_bar, values, model_name, n_splits=5):
    n_bar = np.asarray(n_bar, dtype=float)
    values = np.asarray(values, dtype=float)
    folds = np.array_split(np.arange(len(n_bar)), n_splits)
    errors = []
    for test_indices in folds:
        train_mask = np.ones(len(n_bar), dtype=bool)
        train_mask[test_indices] = False
        train_design, _ = physical_design_matrix(n_bar[train_mask], model_name)
        coefficients = np.linalg.lstsq(train_design, values[train_mask], rcond=None)[0]
        test_design, _ = physical_design_matrix(n_bar[test_indices], model_name)
        errors.extend((values[test_indices] - test_design @ coefficients) ** 2)
    return float(np.sqrt(np.mean(errors)))

cptp_by_nbar = {point["n_bar"]: point["chi"] for point in cptp_points}
physical_nbar = np.asarray(sorted(cptp_by_nbar), dtype=float)
PHYSICAL_FIT_TARGETS = {}
if len(physical_nbar):
    PHYSICAL_FIT_TARGETS.update({
        "chi_XX_XX": np.asarray([np.real(cptp_by_nbar[n][XX_INDEX, XX_INDEX]) for n in physical_nbar]),
        "abs_chi_II_XX": np.asarray([abs(cptp_by_nbar[n][II_INDEX, XX_INDEX]) for n in physical_nbar]),
        "chi_IX_IX": np.asarray([np.real(cptp_by_nbar[n][PAULI_LABELS.index("IX"), PAULI_LABELS.index("IX")]) for n in physical_nbar]),
        "chi_XI_XI": np.asarray([np.real(cptp_by_nbar[n][PAULI_LABELS.index("XI"), PAULI_LABELS.index("XI")]) for n in physical_nbar]),
    })
if not generator_df.empty:
    generator_indexed = generator_df.set_index("n_bar").reindex(physical_nbar)
    PHYSICAL_FIT_TARGETS["generator_h_XX"] = generator_indexed["h_XX_rad_per_gate"].to_numpy(float)
    PHYSICAL_FIT_TARGETS["generator_gamma_XX"] = generator_indexed["gamma_XX_per_gate"].to_numpy(float)

PHYSICAL_MODELS = [
    "thermal_linear", "thermal_bosonic_2", "thermal_bosonic_3", "generic_quartic"
]
physical_fit_rows = []
physical_fit_objects = {}
for target, values in PHYSICAL_FIT_TARGETS.items():
    for model_name in PHYSICAL_MODELS:
        fit = fit_named_linear_model(physical_nbar, values, model_name)
        physical_fit_objects[(target, model_name)] = fit
        physical_fit_rows.append({
            "target": target,
            "model": model_name,
            "model_family": "generic_reference" if model_name == "generic_quartic" else "thermal_physics",
            "parameter_count": len(fit["coefficients"]),
            "RMSE": fit["RMSE"],
            "AICc": fit["AICc"],
            "BIC": fit["BIC"],
            "CV_RMSE_5fold": physical_model_cv_rmse(physical_nbar, values, model_name),
            "equation": fit["equation"],
        })

physical_model_comparison_df = pd.DataFrame(physical_fit_rows)
if physical_model_comparison_df.empty:
    print("CPTP and generator results are required for physical-model fitting.")
else:
    physical_model_comparison_df["best_AICc_overall"] = (
        physical_model_comparison_df.groupby("target")["AICc"].transform("min")
        == physical_model_comparison_df["AICc"]
    )
    physical_only = physical_model_comparison_df[
        physical_model_comparison_df["model_family"] == "thermal_physics"
    ].copy()
    best_physical_indices = physical_only.groupby("target")["AICc"].idxmin()
    physical_model_comparison_df["best_AICc_physical"] = False
    physical_model_comparison_df.loc[best_physical_indices, "best_AICc_physical"] = True
    physical_model_comparison_df.to_csv(
        PHYSICAL_FIT_DIR / "physical_model_comparison.csv", index=False
    )
    best_physical_df = physical_model_comparison_df[
        physical_model_comparison_df["best_AICc_physical"]
    ].copy()
    best_physical_df.to_csv(PHYSICAL_FIT_DIR / "selected_physical_models.csv", index=False)
    display(physical_model_comparison_df)
    display(best_physical_df)

    figure_targets = list(PHYSICAL_FIT_TARGETS)[:6]
    fig_physical, axes_physical = plt.subplots(2, 3, figsize=(16.0, 9.0))
    dense_nbar = np.linspace(physical_nbar.min(), physical_nbar.max(), 1000)
    for ax, target in zip(axes_physical.ravel(), figure_targets):
        values = PHYSICAL_FIT_TARGETS[target]
        selected_model = best_physical_df[best_physical_df["target"] == target].iloc[0]["model"]
        physical_fit = physical_fit_objects[(target, selected_model)]
        generic_fit = physical_fit_objects[(target, "generic_quartic")]
        physical_dense_design, _ = physical_design_matrix(dense_nbar, selected_model)
        generic_dense_design, _ = physical_design_matrix(dense_nbar, "generic_quartic")
        ax.scatter(physical_nbar, values, s=18, alpha=0.65, label="CPTP data")
        ax.plot(
            dense_nbar, physical_dense_design @ physical_fit["coefficients"],
            linewidth=2.2, label=selected_model,
        )
        ax.plot(
            dense_nbar, generic_dense_design @ generic_fit["coefficients"],
            linewidth=1.6, linestyle="--", label="generic quartic",
        )
        ax.set_title(target)
        ax.set_xlabel(r"Mean phonon number $\bar n$")
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=8)
    fig_physical.suptitle("Thermal-physics models versus descriptive quartic fits")
    fig_physical.tight_layout(rect=(0, 0, 1, 0.96))
    fig_physical.savefig(PHYSICAL_FIT_DIR / "physical_model_fits.png", dpi=300, bbox_inches="tight")
    fig_physical.savefig(PHYSICAL_FIT_DIR / "physical_model_fits.pdf", bbox_inches="tight")
    plt.show()


# ### 13. 論文化判定サマリー
# 
# 主要な検証値と未完了の重いQPTを一表にまとめます。`status` がすべて `complete` になった時点で、最終的な物理結論を固定します。

# In[24]:


summary_checks = []

if not cptp_comparison_df.empty:
    summary_checks.extend([
        {
            "check": "CPTP projection",
            "status": "complete" if cptp_comparison_df["projected_cp_pass"].all() and cptp_comparison_df["projected_tp_pass"].all() else "review",
            "result": f"CP={int(cptp_comparison_df['projected_cp_pass'].sum())}/{len(cptp_comparison_df)}, TP={int(cptp_comparison_df['projected_tp_pass'].sum())}/{len(cptp_comparison_df)}",
        },
        {
            "check": "CPTP conclusion stability",
            "status": "complete" if top_overlap == FULL_CHI_TOP_K else "review",
            "result": f"Top-{FULL_CHI_TOP_K} overlap={top_overlap}/{FULL_CHI_TOP_K}, max ||delta chi||_F={cptp_comparison_df['chi_frobenius_shift'].max():.3e}",
        },
    ])

if not generator_df.empty:
    low_generator = generator_df.iloc[0]
    high_generator = generator_df.iloc[-1]
    summary_checks.append({
        "check": "XX error-generator separation",
        "status": "complete",
        "result": (
            f"h_XX: {low_generator['h_XX_rad_per_gate']:.4e} -> {high_generator['h_XX_rad_per_gate']:.4e} rad/gate; "
            f"gamma_XX: {low_generator['gamma_XX_per_gate']:.4e} -> {high_generator['gamma_XX_per_gate']:.4e} /gate"
        ),
    })

if not channel_correction_df.empty:
    high_correction = channel_correction_df.iloc[-1]
    summary_checks.append({
        "check": "Channel-level XX compensation",
        "status": "complete",
        "result": (
            f"n_bar={high_correction['n_bar']:g}: avg infidelity "
            f"{high_correction['average_infidelity_before']:.4e} -> {high_correction['average_infidelity_after']:.4e} "
            f"({high_correction['infidelity_reduction_factor']:.2f}x reduction)"
        ),
    })

summary_checks.append({
    "check": "hXX-derived drive calibration re-QPT",
    "status": (
        "complete" if hxx_drive_completed_nbars >= hxx_drive_expected_nbars
        else "pending"
    ),
    "result": (
        f"{hxx_drive_completed_nbars}/{hxx_drive_expected_nbars} temperatures re-QPT; "
        f"{hxx_drive_converged_nbars} with |h_XX| <= {HXX_CONVERGENCE_TOL_RAD:.1e} rad/gate"
    ),
})

completed_control_points = max(0, len(physical_control_df) - len(CONTROL_VALIDATION_NBARS))
expected_control_points = len(CONTROL_VALIDATION_NBARS) * (len(CONTROL_CANDIDATES) - 1)
summary_checks.append({
    "check": "Physical-control re-QPT",
    "status": "complete" if completed_control_points >= expected_control_points else "pending",
    "result": f"{completed_control_points}/{expected_control_points} non-baseline QPT points",
})

completed_robustness_points = max(0, len(parameter_robustness_df) - len(ROBUSTNESS_NBARS))
expected_robustness_points = len(ROBUSTNESS_NBARS) * (len(ROBUSTNESS_CONDITIONS) - 1)
summary_checks.append({
    "check": "Parameter robustness re-QPT",
    "status": "complete" if completed_robustness_points >= expected_robustness_points else "pending",
    "result": f"{completed_robustness_points}/{expected_robustness_points} non-baseline QPT points",
})

if not physical_model_comparison_df.empty:
    overall_generic_count = int(
        (
            physical_model_comparison_df["best_AICc_overall"]
            & (physical_model_comparison_df["model_family"] == "generic_reference")
        ).sum()
    )
    summary_checks.append({
        "check": "Thermal physical-model comparison",
        "status": "complete",
        "result": (
            f"best physical model saved for {physical_model_comparison_df['target'].nunique()} targets; "
            f"generic quartic remains AICc-best for {overall_generic_count} targets"
        ),
    })

advanced_publication_summary_df = pd.DataFrame(summary_checks)
advanced_publication_summary_df.to_csv(
    ADVANCED_DIR / "advanced_publication_checklist.csv", index=False
)
display(advanced_publication_summary_df)

advanced_manifest = {
    "generated_at_timezone": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
    "machine": {
        "physical_cores": int(PHYSICAL_CORES),
        "memory_gib": float(MEMORY_GIB),
        "qpt_process_workers": int(FAST_PROCESS_WORKERS),
    },
    "flags": {
        "run_physical_control_qpt": RUN_PHYSICAL_CONTROL_QPT,
        "run_hxx_drive_calibration_qpt": RUN_HXX_DRIVE_CALIBRATION_QPT,
        "run_parameter_robustness_qpt": RUN_PARAMETER_ROBUSTNESS_QPT,
        "force_recompute_advanced_qpt": FORCE_RECOMPUTE_ADVANCED_QPT,
    },
    "cptp": {
        "tolerance": CPTP_TOLERANCE,
        "max_chi_frobenius_shift": float(cptp_comparison_df["chi_frobenius_shift"].max()),
        "top_k_overlap": int(top_overlap),
        "top_k": int(FULL_CHI_TOP_K),
    },
    "qpt_completion": {
        "physical_control_points": int(completed_control_points),
        "physical_control_expected": int(expected_control_points),
        "hxx_drive_calibration_temperatures": int(hxx_drive_completed_nbars),
        "hxx_drive_calibration_expected": int(hxx_drive_expected_nbars),
        "hxx_drive_calibration_converged": int(hxx_drive_converged_nbars),
        "robustness_points": int(completed_robustness_points),
        "robustness_expected": int(expected_robustness_points),
    },
    "checklist": advanced_publication_summary_df.to_dict(orient="records"),
}
(ADVANCED_DIR / "advanced_publication_manifest.json").write_text(
    json.dumps(advanced_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)

pending_checks = advanced_publication_summary_df[
    advanced_publication_summary_df["status"] != "complete"
]
if pending_checks.empty:
    display(Markdown("**拡張検証はすべて完了しています。最終的な論文化判断へ進めます。**"))
else:
    display(Markdown(
        "**現時点の結論:** CPTP安定性・generator分解・channel-level補正は完了しました。"
        "実Hamiltonianでの制御再QPTとパラメータ頑健性QPTは未完了です。"
    ))


# ### 拡張検証の実行順と計算資源
# 
# 1. ノート全体を通常実行する。CPTP射影、generator、channel-level XX補正、物理fitは81点キャッシュから生成される。
# 2. `RUN_HXX_DRIVE_CALIBRATION_QPT=True` として、$h_{XX}$由来の温度別drive amplitudeを全Hamiltonianで再QPTする。完了後は `False` に戻す。
# 3. 必要なら `KIRCHHOFF_REFERENCE_K` に実機の $K=f_{mode}t_{gate}$ を入れ、Kirchhoff Eq. (41)との一点比較を固定する。
# 4. `RUN_PHYSICAL_CONTROL_QPT=True` としてその他の実パルス候補を再QPTする。完了後は `False` に戻す。
# 5. `RUN_PARAMETER_ROBUSTNESS_QPT=True` として4パラメータを再QPTする。完了後は `False` に戻す。
# 6. 最後に全QPTフラグを `False` にして全セルを再実行し、CSV・図を固定する。
# 
# 現在のマシンでは自動的に4 worker程度が選ばれます。RAMが多いデスクトップでは最大8 workerまで増えます。手動指定する場合は、Jupyter起動前に `MS_GATE_WORKERS=6` のように設定してください。外側の候補ループは逐次、各QPT内の16入力状態を並列に処理するため、nested multiprocessingは発生しません。

# ## 読み方
# 
# - $\chi_{XX,XX}$ はPauli twirl後にも残る確率的な `XX` エラー成分です。
# - $\chi_{II,XX}$ は `II` と `XX` のコヒーレント結合で、Pauli twirlでは捨てられるオフ対角成分です。
# - 二つを並べることで、温度上昇に伴う確率的エラーとコヒーレントエラーの増加を同時に比較できます。
# - その他の主要対角成分として `IX` と `XI` が選ばれ、両イオンの単一量子ビット $X$ エラーがほぼ対称に増えることが分かります。
# - フィット次数を変更する場合は、上の1〜5次比較表と物理的解釈の両方を確認してください。
