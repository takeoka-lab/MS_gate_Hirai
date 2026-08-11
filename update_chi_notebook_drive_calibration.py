"""Insert the hXX-driven physical amplitude calibration section into the notebook.

This script is intentionally idempotent so the generated notebook cells can be
updated without duplicating the section.
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_name("chi_error_element_nbar_fit.ipynb")
SECTION_ID = "hxx-drive-calibration-intro"


def source_lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def markdown_cell(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source_lines(text),
    }


def code_cell(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


INTRO = r"""
### 10. $h_{XX}(\bar n)$からのdrive amplitude校正と全Hamiltonian再QPT

error-channel規約は $\mathcal S_{\rm error}=\mathcal S_{\rm actual}\mathcal S_{\rm ideal}^{-1}$、理想ゲートは
$U_{\rm MS}=\exp(+i\phi_*XX)$、$\phi_*=\pi/4$ です。generatorを
$K_H\rho=-ih_{XX}[XX,\rho]$ と書くため、$h_{XX}>0$ は実ゲート角
$\phi_{\rm actual}\simeq\phi_*-h_{XX}$ のunder-rotationを表します。

閉じた位相空間軌道で $\phi\propto A^2$ と近似すると、最初の補正振幅は

\[
A_{\rm next}=A_{\rm current}
\sqrt{\frac{\phi_*}{\phi_*-h_{XX}}}
\]

です。この式は初期推定にのみ使い、補正振幅を実際にプロジェクトのmaster-equation Hamiltonianへ戻して16入力状態のQPTを再実行します。再QPT後の残留 $h_{XX}$ を同じ式へ戻せるため、`HXX_MAX_FEEDBACK_ITERATIONS` 回まで閉ループ校正できます。各QPTは独立NPZへ保存され、中断後に再開できます。

Kirchhoff–Wilhelm–Motzoi, PRX Quantum **6**, 010328 (2025) の Eqs. (32), (35), (41) も実装し、$\Omega_2/\Omega_{\rm LD}$、$\Omega_4/\Omega_{\rm LD}$ と比較します。同論文の $\Omega$ はcarrier drive、本コードの $A$ はfirst-sideband couplingですが、$\eta$ 固定なら振幅比では換算係数が消えます。ただし同論文の $K=\nu T/(2\pi)$ と $L=\delta_{\rm bich}T/(2\pi)$ は現在の有効Hamiltonian入力に含まれません。したがって、既定では一周条件 $K-L=1$ の有効域を帯で表示し、実験値がある場合だけ `KIRCHHOFF_REFERENCE_K` を指定して一点比較します。
"""


PREDICTION_CODE = r"""
import drive_amplitude_calibration as dac
dac = importlib.reload(dac)

DRIVE_CALIBRATION_DIR = ADVANCED_DIR / "hxx_drive_amplitude_calibration"
DRIVE_CALIBRATION_QPT_DIR = DRIVE_CALIBRATION_DIR / "qpt_cache"
DRIVE_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_CALIBRATION_QPT_DIR.mkdir(parents=True, exist_ok=True)

# Falseでも81点のhXX由来補正量とKirchhoff比較は直ちに生成される。
# Trueの場合のみ、下の代表温度でfull master-equation QPTを実行する。
RUN_HXX_DRIVE_CALIBRATION_QPT = False
FORCE_RECOMPUTE_HXX_DRIVE_QPT = False
HXX_DRIVE_CALIBRATION_NBARS = [0.01, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
HXX_MAX_FEEDBACK_ITERATIONS = 2
HXX_CONVERGENCE_TOL_RAD = 2e-3
HXX_MAX_AMPLITUDE_FACTOR = 1.6
TARGET_XX_ANGLE_RAD = np.pi / 4.0

# Kirchhoff et al. のK,L。実験のmotional-mode周波数が分かる場合、
# K = f_mode[Hz] * t_gate[s] を設定する。一周条件では L = K - 1。
KIRCHHOFF_REFERENCE_K = None
KIRCHHOFF_LOOP_NUMBER = 1.0
KIRCHHOFF_SCAN_K_MAX = 250.0

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
"""


QPT_CODE = r"""
def generator_observables_from_trace_normalized_chi(chi):
    # CPTP-project one QPT result and extract hXX/gammaXX consistently.
    projected_super, _, projected_chi, projection_status = (
        project_chi_point_to_cptp({"chi": np.asarray(chi, dtype=complex)})
    )
    ptm = np.asarray(mg.superoperator_to_ptm(projected_super), dtype=complex)
    generator_complex = scipy.linalg.logm(ptm)
    generator = np.real(generator_complex)
    skew_generator = 0.5 * (generator - generator.T)
    h_coefficients = np.linalg.lstsq(
        hamiltonian_design, skew_generator.reshape(-1), rcond=None
    )[0]
    h_fit = sum(
        coefficient * HAMILTONIAN_GENERATOR_BASES[label]
        for label, coefficient in zip(PAULI_LABELS[1:], h_coefficients)
    )
    symmetric_remaining = 0.5 * (
        (generator - h_fit) + (generator - h_fit).T
    )
    gamma_coefficients, gamma_residual = scipy.optimize.nnls(
        dissipator_design, symmetric_remaining.reshape(-1)
    )
    h_xx = float(h_coefficients[PAULI_LABELS[1:].index("XX")])
    gamma_xx = float(gamma_coefficients[PAULI_LABELS[1:].index("XX")])
    return {
        "h_XX_rad_per_gate": h_xx,
        "gamma_XX_per_gate": gamma_xx,
        "average_infidelity": average_infidelity_from_trace_normalized_chi(
            projected_chi
        ),
        "abs_chi_II_XX": float(abs(projected_chi[II_INDEX, XX_INDEX])),
        "chi_XX_XX": float(np.real(projected_chi[XX_INDEX, XX_INDEX])),
        "generator_imaginary_frobenius_norm": float(
            np.linalg.norm(np.imag(generator_complex))
        ),
        "gamma_nnls_residual": float(gamma_residual),
        "cptp_projection_iterations": int(projection_status["iterations"]),
        "projected_chi": projected_chi,
    }


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
        json.dumps(payload, sort_keys=True, default=json_safe).encode("utf-8")
    ).hexdigest()[:12]
    return DRIVE_CALIBRATION_QPT_DIR / (
        f"hxx_feedback_i{int(iteration):02d}__nbar_{_safe_nbar_stem(n_bar)}__{digest}.npz"
    )


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

            cache_path = drive_feedback_cache_path(
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
                qpt_result = calculate_error_channel_batch(
                    [n_bar],
                    {
                        "A": next_amplitude,
                        "parallel_workers": FAST_PROCESS_WORKERS,
                        "show_progress": False,
                    },
                )[0]
                qpt_result["metadata"].update({
                    "calibration_method": "hxx_quadratic_feedback",
                    "iteration": iteration,
                    "input_h_XX_rad_per_gate": current_h_xx,
                    "A_baseline": BASE_DRIVE_AMPLITUDE,
                    "A_calibrated": next_amplitude,
                    "A_factor": amplitude_factor,
                })
                save_advanced_qpt_point(
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
"""


RESULT_CODE = r"""
baseline_calibration_rows = []
for n_bar in HXX_DRIVE_CALIBRATION_NBARS:
    generator_match = generator_df.loc[
        np.isclose(generator_df["n_bar"].astype(float), float(n_bar))
    ]
    cptp_match = [
        point for point in cptp_points
        if np.isclose(float(point["n_bar"]), float(n_bar))
    ]
    if generator_match.empty or not cptp_match:
        continue
    generator_row = generator_match.iloc[0]
    baseline_chi = cptp_match[0]["chi"]
    baseline_calibration_rows.append({
        "n_bar": float(n_bar),
        "h_XX_before_rad_per_gate": float(generator_row["h_XX_rad_per_gate"]),
        "gamma_XX_before_per_gate": float(generator_row["gamma_XX_per_gate"]),
        "average_infidelity_before": average_infidelity_from_trace_normalized_chi(
            baseline_chi
        ),
        "abs_chi_II_XX_before": float(abs(baseline_chi[II_INDEX, XX_INDEX])),
        "chi_XX_XX_before": float(np.real(baseline_chi[XX_INDEX, XX_INDEX])),
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
"""


def replace_in_cell(cells: list[dict], cell_id: str, old: str, new: str) -> None:
    cell = next(cell for cell in cells if cell.get("id") == cell_id)
    text = "".join(cell["source"])
    if old not in text:
        if new in text:
            return
        raise RuntimeError(f"Expected text not found in cell {cell_id}: {old!r}")
    cell["source"] = source_lines(text.replace(old, new))


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cells[:] = [cell for cell in cells if cell.get("id") not in {
        SECTION_ID,
        "hxx-drive-calibration-prediction",
        "hxx-drive-calibration-qpt",
        "hxx-drive-calibration-results",
    }]

    insertion_index = next(
        index for index, cell in enumerate(cells)
        if cell.get("id") == "parameter-robustness-intro"
    )
    cells[insertion_index:insertion_index] = [
        markdown_cell(SECTION_ID, INTRO),
        code_cell("hxx-drive-calibration-prediction", PREDICTION_CODE),
        code_cell("hxx-drive-calibration-qpt", QPT_CODE),
        code_cell("hxx-drive-calibration-results", RESULT_CODE),
    ]

    replace_in_cell(cells, "parameter-robustness-intro", "### 10.", "### 11.")
    replace_in_cell(cells, "physical-model-fit-intro", "### 11.", "### 12.")
    replace_in_cell(cells, "advanced-publication-summary-intro", "### 12.", "### 13.")

    replace_in_cell(
        cells,
        "advanced-publication-intro",
        "3. channel-levelの最適 `XX` 角補正、および実パルスパラメータによる再QPT検証\n",
        "3. channel-levelの最適 `XX` 角補正と、$h_{XX}$から算出したdrive amplitudeによる全Hamiltonian再QPT\n",
    )

    summary_cell = next(
        cell for cell in cells if cell.get("id") == "advanced-publication-summary-code"
    )
    summary_text = "".join(summary_cell["source"])
    marker = "completed_control_points = max(0, len(physical_control_df) - len(CONTROL_VALIDATION_NBARS))\n"
    hxx_summary = '''summary_checks.append({
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

'''
    if '"check": "hXX-derived drive calibration re-QPT"' not in summary_text:
        if marker not in summary_text:
            raise RuntimeError("Could not locate physical-control summary marker")
        summary_text = summary_text.replace(marker, hxx_summary + marker)
    if '"run_hxx_drive_calibration_qpt"' not in summary_text:
        summary_text = summary_text.replace(
            '"run_physical_control_qpt": RUN_PHYSICAL_CONTROL_QPT,\n',
            '"run_physical_control_qpt": RUN_PHYSICAL_CONTROL_QPT,\n'
            '        "run_hxx_drive_calibration_qpt": RUN_HXX_DRIVE_CALIBRATION_QPT,\n',
        )
    if '"hxx_drive_calibration_temperatures"' not in summary_text:
        summary_text = summary_text.replace(
            '"physical_control_expected": int(expected_control_points),\n',
            '"physical_control_expected": int(expected_control_points),\n'
            '        "hxx_drive_calibration_temperatures": int(hxx_drive_completed_nbars),\n'
            '        "hxx_drive_calibration_expected": int(hxx_drive_expected_nbars),\n'
            '        "hxx_drive_calibration_converged": int(hxx_drive_converged_nbars),\n',
        )
    summary_cell["source"] = source_lines(summary_text)

    replace_in_cell(
        cells,
        "advanced-publication-run-order",
        "2. `RUN_PHYSICAL_CONTROL_QPT=True` として実パルス候補を再QPTする。完了後は `False` に戻す。\n3. `RUN_PARAMETER_ROBUSTNESS_QPT=True` として4パラメータを再QPTする。完了後は `False` に戻す。\n4. 最後に両フラグを `False` にして全セルを再実行し、CSV・図を固定する。",
        "2. `RUN_HXX_DRIVE_CALIBRATION_QPT=True` として、$h_{XX}$由来の温度別drive amplitudeを全Hamiltonianで再QPTする。完了後は `False` に戻す。\n3. 必要なら `KIRCHHOFF_REFERENCE_K` に実機の $K=f_{mode}t_{gate}$ を入れ、Kirchhoff Eq. (41)との一点比較を固定する。\n4. `RUN_PHYSICAL_CONTROL_QPT=True` としてその他の実パルス候補を再QPTする。完了後は `False` に戻す。\n5. `RUN_PARAMETER_ROBUSTNESS_QPT=True` として4パラメータを再QPTする。完了後は `False` に戻す。\n6. 最後に全QPTフラグを `False` にして全セルを再実行し、CSV・図を固定する。",
    )

    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
