# Pauli近似の論文化に向けた既存チャネルの再評価

2026-09-29。保存済みCPTP射影後のχ行列の後処理のみ。新しいmaster-equation計算、数値収束試験、SDPによるdiamond normの厳密値計算は行っていない。

## 定義

誤差チャネルを E、Pauli twirlを T とする。Pauli演算子は非正規化、χはtrace-one、系の次元は d=4。

- `ptm_frobenius_twirl_difference` = ||R(E)-R(T(E))||_F。
- `relative_error_frobenius_twirl_difference` = ||R(E)-R(T(E))||_F / ||R(E)-I||_F。
- `diamond_distance_*_bound` は epsilon = (1/2)||E-T(E)||_diamond の上下界。
- `average_infidelity` = (4/5)(1-chi[II,II])。

chi_off = chi-diag(chi) は E-T(E) の正規化Choi行列とユニタリ同値である。したがって

    ||chi_off||_1 / 2 <= epsilon <= d ||chi_off||_1 / 2.

下界は最大エンタングル入力で得られる。上界は非正規化Choi行列Jについて ||Delta||_diamond <= ||J(Delta)||_1 を用いる。これらは緩い一般界であり、SDP値ではない。保存行列と浮動小数点計算への評価であり、物理モデルやODE誤差を保証するものではない。

## 今回確認したこと

全7点（nbar=0.01,1,2,4,6,8,10）で、較正後のdiamond距離上界が較正前の下界より小さい。したがって保存行列については、Frobenius距離だけでなくdiamond距離も減少していると判定できる（数値許容誤差の範囲）。

nbar=4では：

| 指標 | 較正前 | 較正後 |
|---|---:|---:|
| PTM Frobenius距離 | 0.4044234 | 0.0222227 |
| 残留誤差に対する相対Frobenius距離 | 0.983893 | 0.383272 |
| diamond距離下界 | 0.0741110 | 0.00574433 |
| diamond距離上界 | 0.296444 | 0.0229773 |
| average infidelity | 0.0122808 | 0.00936560 |

Frobenius距離は約18.2分の1。ただし相対差38.3%は残り、上記の相対Frobenius距離は誤り確率ではない。既存のBell状態の後処理 `output/research_review_20260916/bell_pauli_comparison.csv` も、較正後の元チャネルとPauli twirlで状態依存の予測差が残ることを示している。

## 論文化で優先する検証

1. 上記指標を正式に定義し、較正前後とinfidelity最適化の既存対照群を同じ条件で比較する。diamond距離の上下界が用途に対して緩すぎる場合はSDPを実行する。
2. 最終指標そのもののcutoff、時間補間、ODE許容誤差、CPTP射影への安定性を確認する。Hamiltonianの有限LD打切り誤差はこれと別に比較する。
3. 温度ごとに独立較正する理想条件に加え、固定較正点からの温度・振幅・detuning・noise-rateのずれで改善が残る領域を示す。
4. HだけでなくC/A成分を含めて残差を説明する。特にD[IX+XI]は非Pauli成分を持つ。一方、Pauli確率p_XXは二量子ビット相関を持っていてもPauliモデル内で扱える。
5. 回路/QECの予測精度まで主張する場合は、元チャネルとtwirlを短い回路または小規模符号で比較する。運動モードを各ゲートで再初期化しない場合、単一ゲート縮約チャネルの単純合成が妥当かも検証する。

## 証明と用語

評価範囲を限定した数値論文なら、任意パラメータに対する近似保証定理は必須ではない。定義、数値収束、誤差の定量評価、適用範囲が必要。

Pauli twirlingを実装してノイズを変えることと、元チャネルをそのtwirlで近似することは異なる。理想的なtwirlでは元チャネルが非Pauliでも平均チャネルはPauliになる。今回の距離は後者の近似誤差を評価する。

Pauli twirlは平均ゲート忠実度を保存するため、その一致だけでは近似精度を評価できない。また、一般に T(exp(K)) と exp(T(K)) は一致しないので、チャネルのtwirlを生成元の対角化で代用しない。

再現コマンド：プロジェクトルートから `python3 output/publication_gap_audit_20260929/pauli_twirl_metrics.py`。
