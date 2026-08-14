"""Rebuild chi_error_element_nbar_fit.ipynb as a configuration front end."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_name("chi_error_element_nbar_fit.ipynb")


def _lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def _markdown(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": _lines(text),
    }


def _code(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


CELLS = [
    _markdown(
        "workflow-intro",
        r"""
# χ行列の主要エラー要素とdrive校正

このノートは**設定と実行だけ**を担当します。関数、QPT、CPTP射影、fit、描画、キャッシュ処理はPython側へ移しました。

- 設定API: `chi_error_nbar_workflow.py`
- 解析実装: `chi_error_nbar_analysis_impl.py`
- 独立検証ステージ: `chi_error_nbar_stages.py`
- drive補正式: `drive_amplitude_calibration.py`
- drive再QPT解析: `drive_calibration_qpt_analysis.py`

通常は下の設定セルだけ変更し、最後の`workflow.run(CONFIG)`を実行してください。重いフラグは既定で`False`です。
""",
    ),
    _code(
        "workflow-imports",
        r"""
import importlib

import numpy as np
import pandas as pd
from IPython.display import display

import chi_error_nbar_workflow as workflow
import chi_error_nbar_stages as stages
workflow = importlib.reload(workflow)
stages = importlib.reload(stages)
""",
    ),
    _markdown(
        "workflow-common-config-intro",
        "## 1. 共通設定\n\n通常変更するのはこのセルです。",
    ),
    _code(
        "workflow-common-config",
        r"""
CONFIG = workflow.default_config()

CONFIG["OUTPUT_DIR"] = "results/chi_error_element_fit"
CONFIG["NBAR_GRID"] = np.r_[0.01, np.arange(0.25, 20.0001, 0.25)]
CONFIG["FIT_DEGREE"] = 4
CONFIG["ERROR_CHANNEL_CONVENTION"] = "undo_before_actual"

# 並列数。高n_barではメモリ使用量も増えるため、まず4を推奨。
CONFIG["PARALLEL_WORKERS"] = 4
CONFIG["FAST_PROCESS_WORKERS"] = 4

display(pd.Series({
    key: value for key, value in CONFIG.items()
    if key not in {"SIMULATION_PARAMS", "NBAR_GRID"}
}, name="value").to_frame())
""",
    ),
    _markdown(
        "workflow-simulation-config-intro",
        "## 2. Hamiltonian・ノイズ設定\n\n物理条件を変更する場合はこのセルだけ編集します。",
    ),
    _code(
        "workflow-simulation-config",
        r"""
SIMULATION_PARAMS = CONFIG["SIMULATION_PARAMS"]
SIMULATION_PARAMS.update({
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
    "parallel_workers": CONFIG["PARALLEL_WORKERS"],
})
display(pd.Series(SIMULATION_PARAMS, name="value").to_frame())
""",
    ),
    _markdown(
        "workflow-publication-config-intro",
        "## 3. 論文用検証フラグ\n\n`False`では保存済みキャッシュを読みます。`True`は不足点を計算します。`FORCE_*`は既存キャッシュも再計算します。",
    ),
    _code(
        "workflow-publication-config",
        r"""
CONFIG.update({
    "RECOMPUTE": False,
    "RUN_EXACT_FULL_CHI_SWEEP": False,
    "RUN_NUMERICAL_CONVERGENCE": False,
    "RUN_NOISE_SOURCE_ABLATION": False,
    "FORCE_RECOMPUTE_PUBLICATION_CACHE": False,
    "CONVERGENCE_NBARS": [0.01, 4.0, 20.0],
    "ABLATION_NBARS": [0.01, 1.0, 4.0, 10.0, 20.0],
    "BOOTSTRAP_SAMPLES": 1000,
    "BOOTSTRAP_SEED": 20260805,
    "CPTP_TOLERANCE": 1e-11,
    "CPTP_MAX_ITERATIONS": 5000,
    "FORCE_RECOMPUTE_ADVANCED_QPT": False,
    "RUN_PHYSICAL_CONTROL_QPT": False,
    "RUN_PARAMETER_ROBUSTNESS_QPT": False,
    "CONTROL_VALIDATION_NBARS": [0.01, 1.0, 2.0, 3.0, 4.0],
    "ROBUSTNESS_NBARS": [0.01, 1.0, 2.0, 3.0, 4.0],
})
""",
    ),
    _markdown(
        "workflow-drive-config-intro",
        r"""
## 4. $h_{XX}$ drive-amplitude校正

`RUN_HXX_DRIVE_CALIBRATION_QPT=True`にすると、補正振幅でfull-order master-equation QPTを実行します。各温度点は独立キャッシュなので中断・再開できます。
""",
    ),
    _code(
        "workflow-drive-config",
        r"""
CONFIG.update({
    "RUN_HXX_DRIVE_CALIBRATION_QPT": False,
    "FORCE_RECOMPUTE_HXX_DRIVE_QPT": False,
    "HXX_DRIVE_CALIBRATION_NBARS": [
        0.01, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0,
    ],
    "HXX_MAX_FEEDBACK_ITERATIONS": 2,
    "HXX_CONVERGENCE_TOL_RAD": 2e-3,
    "HXX_MAX_AMPLITUDE_FACTOR": 1.6,
    "TARGET_XX_ANGLE_RAD": np.pi / 4.0,

    # 実機値がある場合: K = f_mode[Hz] * t_gate[s]
    "KIRCHHOFF_REFERENCE_K": None,
    "KIRCHHOFF_LOOP_NUMBER": 1.0,
    "KIRCHHOFF_SCAN_K_MAX": 250.0,
})
""",
    ),
    _markdown(
        "workflow-run-intro",
        "## 5. 基本解析と保存済み検証の読込\n\n設定を検証した後、解析本体を一度だけ呼びます。重いフラグが`False`なら既存キャッシュを読みます。以下の独立検証セルより先に一度実行してください。",
    ),
    _code(
        "workflow-run",
        r"""
workflow.validate_config(CONFIG)
RESULTS = workflow.run(CONFIG)
print("analysis completed")
""",
    ),
    _markdown(
        "independent-drive-qpt-intro",
        r"""
## 6. 独立実行：drive補正後のfull-Hamiltonian再QPT

10温度点を集計します。`RUN_DRIVE_RE_QPT=True`で不足点だけを計算し、完了済み点はキャッシュから自動的に読みます。実行後は論文用チェックリストも即座に更新されます。
""",
    ),
    _code(
        "independent-drive-qpt",
        r"""
DRIVE_RE_QPT_NBARS = list(CONFIG["HXX_DRIVE_CALIBRATION_NBARS"])
RUN_DRIVE_RE_QPT = False
FORCE_DRIVE_RE_QPT = False

DRIVE_RE_QPT_RESULT = stages.run_drive_feedback_stage(
    CONFIG,
    DRIVE_RE_QPT_NBARS,
    run_qpt=RUN_DRIVE_RE_QPT,
    force_recompute=FORCE_DRIVE_RE_QPT,
    max_feedback_iterations=CONFIG["HXX_MAX_FEEDBACK_ITERATIONS"],
)
print(DRIVE_RE_QPT_RESULT["status"])
if not DRIVE_RE_QPT_RESULT["summary"].empty:
    display(DRIVE_RE_QPT_RESULT["summary"])
""",
    ),
    _markdown(
        "independent-fock-xx-angle-intro",
        r"""
## 7. フォノン数ごとのXX角

熱平均の $\bar n$ ではなく、各Fock状態 $|n\rangle$ から出発したときのXX角
$\theta_{XX}^{(n)}=\tfrac12\arg\langle n|U_{S_x=+2}(T)|n\rangle$ を出します。
$h_{XX}^{(n)}=\pi/4-\theta_{XX}^{(n)}$ です。この計算はLindbladノイズを外し、full-order Hamiltonianが生む熱的な角ずれと残留spin-motion entanglementを切り分けます。

既定は代表的な低温・中間・高温の3校正振幅とbaselineの計4曲線です。`FOCK_ANGLE_REFERENCE_NBARS`を変えれば比較点を追加できます。
""",
    ),
    _code(
        "independent-fock-xx-angle",
        r"""
import importlib
from IPython.display import Image

stages = importlib.reload(stages)

FOCK_ANGLE_REFERENCE_NBARS = [0.01, 4.0, 20.0]
FOCK_THERMAL_TAIL_TOLERANCE = 1e-5
FOCK_MAX_N = None  # None: 最大bar-nとtail toleranceから自動決定
FOCK_PLOT_MAX_N = 80
FOCK_PHONON_BUFFER = 24
FORCE_FOCK_ANGLE_RECOMPUTE = False
SHOW_FOCK_ANGLE_PROGRESS = True

FOCK_XX_RESULT = stages.run_fock_xx_angle_stage(
    CONFIG,
    FOCK_ANGLE_REFERENCE_NBARS,
    thermal_tail_tolerance=FOCK_THERMAL_TAIL_TOLERANCE,
    max_fock_n=FOCK_MAX_N,
    plot_max_fock_n=FOCK_PLOT_MAX_N,
    phonon_buffer=FOCK_PHONON_BUFFER,
    force_recompute=FORCE_FOCK_ANGLE_RECOMPUTE,
    show_progress=SHOW_FOCK_ANGLE_PROGRESS,
)
print(FOCK_XX_RESULT["status"])
display(FOCK_XX_RESULT["matched_summary"][[
    "condition", "thermal_n_bar", "amplitude",
    "effective_theta_xx_rad", "predicted_h_XX_rad_per_gate",
    "gamma_XX_phase_dispersion_per_gate",
    "gamma_XX_with_residual_motion_per_gate",
    "qpt_h_XX_rad_per_gate", "qpt_gamma_XX_per_gate",
    "thermal_tail_mass",
]])
display(Image(filename=str(FOCK_XX_RESULT["figure_path"])))
""",
    ),
    _markdown(
        "independent-kirchhoff-intro",
        r"""
## 8. 独立実行：Kirchhoff解析式との直接比較

`K=f_mode[Hz] * t_gate[s]`です。実機のmode周波数を設定するか、既知の`K`を直接設定してください。両方が`None`の場合は計算せず待機します。
""",
    ),
    _code(
        "independent-kirchhoff",
        r"""
KIRCHHOFF_MODE_FREQUENCY_HZ = None
KIRCHHOFF_REFERENCE_K = None
KIRCHHOFF_LOOP_NUMBER = 1.0

KIRCHHOFF_DIRECT_RESULT = None
if KIRCHHOFF_MODE_FREQUENCY_HZ is None and KIRCHHOFF_REFERENCE_K is None:
    print("Set KIRCHHOFF_MODE_FREQUENCY_HZ or KIRCHHOFF_REFERENCE_K.")
else:
    KIRCHHOFF_DIRECT_RESULT = stages.run_kirchhoff_direct_comparison_stage(
        CONFIG,
        mode_frequency_hz=KIRCHHOFF_MODE_FREQUENCY_HZ,
        reference_k=KIRCHHOFF_REFERENCE_K,
        loop_number=KIRCHHOFF_LOOP_NUMBER,
    )
    print(KIRCHHOFF_DIRECT_RESULT["reference"])
    display(KIRCHHOFF_DIRECT_RESULT["comparison"])
""",
    ),
    _markdown(
        "independent-control-intro",
        r"""
## 9. 独立実行：他の物理制御との比較

固定drive scan、gate time、detuning、$\sin^2$ pulse、Blackman pulseを同じfull-Hamiltonian QPTとCPTP指標で比較します。既定条件は11候補×5温度=55点です。候補全体と各候補内のQPT evolutionを2段のprogress barで表示します。
""",
    ),
    _code(
        "independent-control",
        r"""
CONTROL_QPT_NBARS = [0.01, 1.0, 2.0, 3.0, 4.0]
RUN_CONTROL_QPT = False
FORCE_CONTROL_QPT = False
SHOW_CONTROL_PROGRESS = True

DRIVE_AMPLITUDE_FACTORS = [0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
GATE_TIME_FACTORS = [0.97, 1.03]
DETUNING_FACTORS = [0.97, 1.03]
PULSE_SHAPES = ["sin2", "blackman"]

PHYSICAL_CONTROL_RESULT = stages.run_physical_control_stage(
    CONFIG,
    CONTROL_QPT_NBARS,
    run_qpt=RUN_CONTROL_QPT,
    force_recompute=FORCE_CONTROL_QPT,
    show_progress=SHOW_CONTROL_PROGRESS,
    amplitude_factors=DRIVE_AMPLITUDE_FACTORS,
    gate_time_factors=GATE_TIME_FACTORS,
    detuning_factors=DETUNING_FACTORS,
    pulse_shapes=PULSE_SHAPES,
)
print(PHYSICAL_CONTROL_RESULT["status"])
display(PHYSICAL_CONTROL_RESULT["candidates"])
display(PHYSICAL_CONTROL_RESULT["best"])
""",
    ),
    _markdown(
        "independent-control-screening-report-intro",
        r"""
### 9.1 screening結果の集計とグラフ

保存済みQPTだけを読み、$|h_{XX}|$、$\gamma_{XX}$、physical average infidelity、baseline比を1枚にまとめます。このセルはQPTを再実行しません。
""",
    ),
    _code(
        "independent-control-screening-report",
        r"""
import importlib
from IPython.display import Image

stages = importlib.reload(stages)
SCREENING_REPORT_NBARS = [0.01, 2.0, 4.0, 10.0, 20.0]

CONTROL_SCREENING_REPORT = stages.run_physical_control_screening_report(
    CONFIG,
    SCREENING_REPORT_NBARS,
)
print(CONTROL_SCREENING_REPORT["status"])
display(CONTROL_SCREENING_REPORT["winners"][[
    "n_bar",
    "best_abs_h_XX_candidate", "h_XX_reduction_factor",
    "best_infidelity_candidate", "infidelity_improvement_factor",
    "best_infidelity_abs_h_XX_ratio",
    "best_infidelity_gamma_XX_ratio",
    "best_control_score_candidate",
]])
display(Image(filename=str(CONTROL_SCREENING_REPORT["figure_path"])))
""",
    ),
    _markdown(
        "independent-robustness-intro",
        r"""
## 10. 独立実行：パラメータ頑健性

$\eta$、$A/\delta$、gate time、motional dephasing rateを個別に変更し、$h_{XX}$、$\gamma_{XX}$、average infidelityを再評価します。既定条件は9条件×5温度=45点です。条件全体と各条件内のQPT evolutionを2段のprogress barで表示します。
""",
    ),
    _code(
        "independent-robustness",
        r"""
ROBUSTNESS_QPT_NBARS = [0.01, 1.0, 2.0, 3.0, 4.0]
RUN_ROBUSTNESS_QPT = False
FORCE_ROBUSTNESS_QPT = False
SHOW_ROBUSTNESS_PROGRESS = True

ETA_FACTORS = [0.8, 1.2]
A_OVER_DELTA_FACTORS = [0.9, 1.1]
ROBUSTNESS_GATE_TIME_FACTORS = [0.95, 1.05]
MOTIONAL_DEPHASING_FACTORS = [0.0, 0.5, 2.0]

ROBUSTNESS_RESULT = stages.run_parameter_robustness_stage(
    CONFIG,
    ROBUSTNESS_QPT_NBARS,
    run_qpt=RUN_ROBUSTNESS_QPT,
    force_recompute=FORCE_ROBUSTNESS_QPT,
    show_progress=SHOW_ROBUSTNESS_PROGRESS,
    eta_factors=ETA_FACTORS,
    a_over_delta_factors=A_OVER_DELTA_FACTORS,
    gate_time_factors=ROBUSTNESS_GATE_TIME_FACTORS,
    motional_dephasing_factors=MOTIONAL_DEPHASING_FACTORS,
)
print(ROBUSTNESS_RESULT["status"])
display(ROBUSTNESS_RESULT["conditions"])
display(ROBUSTNESS_RESULT["summary"])
""",
    ),
    _markdown(
        "workflow-results-intro",
        "## 11. 主要結果\n\n詳細なCSV・PNG・PDFは`CONFIG[\"OUTPUT_DIR\"]`以下へ保存されます。",
    ),
    _code(
        "workflow-results",
        r"""
RESULT_TABLES = [
    "fit_summary_df",
    "advanced_publication_summary_df",
    "final_drive_calibration_df",
]
for table_name in RESULT_TABLES:
    table = RESULTS.get(table_name)
    if isinstance(table, pd.DataFrame) and not table.empty:
        print(table_name)
        display(table)
""",
    ),
]


def main() -> None:
    if NOTEBOOK_PATH.exists():
        original = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        metadata = original.get("metadata", {})
        nbformat = original.get("nbformat", 4)
        nbformat_minor = original.get("nbformat_minor", 5)
    else:
        metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        }
        nbformat = 4
        nbformat_minor = 5
    notebook = {
        "cells": CELLS,
        "metadata": metadata,
        "nbformat": nbformat,
        "nbformat_minor": nbformat_minor,
    }
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
