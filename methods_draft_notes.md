# Methods草案の根拠と編集メモ

作成日：2026-10-04。本文は `methods_draft.tex`。既存の原稿・解析コード・データは変更していない。

## 採用した範囲

直前のIntroductionに合わせ、単一運動モードの有効MSハミルトニアン、熱的初期状態と動作中の散逸、数値QPT、post-gateチャネル、生成子分解、駆動振幅較正、Fock状態別解析を記載した。現時点で保存済みのPauli近似との距離解析も含めた。本文の数値比較範囲は、較正前後とdiamond距離の入力がそろっている nbar = 0.01, 1, 2, 3, 4 とした。

Kirchhoffモデルの別実装との比較、波形最適化、ノイズ率の混合微分、高温側の拡張は本草案の対象に含めていない。これらを最終論文のResultsに採用するときは、対応するMethodsを追加する。

## コードとの対応

| 本文 | 確認した根拠 |
|---|---|
| ハミルトニアンと B 演算子 | `ms_gate_functions.py`: `_ms_gate_static_operators`, `_build_ms_hamiltonian` |
| 熱状態、入力状態、master equation | `ms_gate_functions.py`: `_build_c_ops`, `build_input_states`, `run_ms_gate_simulation` |
| 名目値・較正の停止条件 | `chi_error_nbar_workflow.py`: `default_simulation_params`, `default_config` |
| post-gate合成順序 | `ms_gate_functions.py`: `remove_ideal_gate_from_channel` の `undo_before_actual`。既定引数は逆順なので、本文は保存解析で明示指定されている順序を採用 |
| CPTP射影とHamiltonian射影 | `drive_calibration_qpt_analysis.py`: `nearest_cptp_choi_dykstra`, `_generator_observables_from_superoperator` |
| 完全なH/S/C/A分解 | `noise_error_structure.py`: `_taxonomy_design_data`, `_decompose_generator_taxonomy` |
| 三成分近似 | `angle_calibration_channel_comparison.ipynb`: `analyze_channel`。完全基底係数とは区別 |
| 振幅更新と再QPT | `drive_amplitude_calibration.py`, `chi_error_nbar_stages.py`: `run_drive_feedback_stage` |
| Fock解析 | `phonon_xx_angle_analysis.py`: `calculate_fock_resolved_xx_angles`, `summarize_thermal_xx_angle` |
| Fock解析のtailとbuffer | `chi_error_nbar_stages.py`: `run_fock_resolved_xx_angle_stage` の該当引数（thermal tail 1e-5、buffer 24） |
| 較正前後の5点 | `results/pauli_twirl_channel_distance/inputs/preparation_manifest.json`, `channel_comparison_long.csv` |
| 距離の定義・規約・SDP | `pauli_channel_distance.py`, `pauli_twirl_channel_distance.ipynb`, `results/pauli_twirl_channel_distance/manifest.json` |

## 投稿前に確定する点

1. **数値収束。** 草案は solver tolerance、係数の500点線形補間、cutoff選択規則を記載したが、最終データのcutoff・補間格子・ODE精度への収束が検証済みとは書いていない。採用する図ごとの収束結果と誤差規模を後から追加する。
2. **温度範囲。** 0.01–20 の別のスキャンを掲載する場合は、対応する計算条件と近似の妥当性を追記する。現在の5点を論文全体の最終範囲として固定する意図はない。
3. **CPTP射影前後の表示。** `drive_amplitude_chi_before_after.ipynb` の主ヒートマップは保存された未射影行列、生成子と距離は射影後の行列。最終図のcaptionにどちらかを明記する。草案は主解析を射影後とし、未射影行列も保持すると記述した。
4. **χの規約。** 草案のχは標準Pauli演算子の作用で定義したcanonical χ。現在のQuTiP 5.0.4由来のtrace-one保存行列には奇数個のYに対応する符号変換 `chi = D @ chi_saved @ D`, `D_aa = (-1)**number_of_Y` が必要で、距離コードが実装・照合済み。保存χを本文の作用式へ直接代入しない。
5. **全次数ではない。** `use_full_order=True` というコード名でも、保持しているのは第一側帯波演算子の相対 O(eta^2) 補正。草案はこの範囲を明記した。
6. **較正の物理条件。** 各nbarで独立に較正し、散乱率を含む散逸係数を固定する名目比較。レーザー強度変更に伴う散乱率変更を含める別解析とは区別する。現在の5点の保存結果は全点1回の更新で停止目標を満たしている。
7. **Fock解析からの同定。** forced–null overlapだけで全チャネルを決定したとは書いていない。位相分散の診断量、三成分フィット、完全H/S/C/A係数は別の量。collective散逸の対応は他の成分とフィット残差も含めてResultsで検証する。

独立コンパイルのためにプリアンブルと参考文献を付けている。論文に移す場合は `BEGIN METHODS`–`END METHODS` の範囲と必要なマクロを移し、引用キー `QPT`, `Markovian_Errors`, `WatrousDiamond` を既存の文献管理に合わせる。数値実装の詳細を短くする場合は、最後の小節とcutoff規則をSupplementary Methodsに移せる。
