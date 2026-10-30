# 一次 Lamb–Dicke 計算の評価

追記: 後続の独立計算と代表点 QPT により、無散逸残差の主因が Fock 基底の打ち切りであることを確認した。基底拡大後には元の QPT パイプラインの微小な非物理性と下限処理も顕在化する。[追加検証](cutoff_audit/findings.md) を参照。以下は最初の保存済み CSV に対する評価である。

評価日: 2026-10-21。保存済み CSV の解析であり、追加の QPT 計算は行っていない。

今回の結果は、LD 高次補正を除くと無散逸の熱依存誤差がほぼ消え、残る主要な温度依存性が motional dephasing に由来することを支持する。

入力は `results/noise_rate_nbar_sweep_ld1/` の `config.json`、`summary.csv`、`all_noise_zero_summary.csv`。比較対象は `results/noise_rate_nbar_sweep/`。物理・数値設定の差は `use_full_order: True -> False` のみで、実行設定として並列数が 7 から 6 に変わっている。各モデルとも 4 散逸源 × 5 レート × 5 温度の 100 行が揃い、重複キー、非有限値、不正な infidelity はない。4 パネルの nominal 条件も完全に一致する。無散逸参照も 5 温度すべて揃っている。

全ノイズを基準値に固定した infidelity は、nbar = 0.01 で補正あり 0.001946502、LD1 0.001725932、nbar = 4 で補正あり 0.012280765、LD1 0.004556110。高温点ではモデル間の差が約 62.9% に達する。この比較はモデルから高次補正を除いた効果であり、実際の装置の性能改善を意味しない。また、差には高次補正と散逸の相互作用も含まれるため、差の全量を無散逸誤差と同一視しない。

各パネルでは他の 3 種類の散逸レートが基準値で残っている。したがって、今回計算できる寄与は

\[
\Delta I_i^{\rm remove}(\bar n)=I_{\rm all}(\bar n)-I_{\rm without\ i}(\bar n)
\]

であり、単独散逸の `I_only - I_none` とは異なる。これは有限レートでの非加算効果も含む差分で、各差分を足して厳密な誤差分解と解釈しない。

| 除去する散逸源 | nbar = 0.01 の差分 | nbar = 4 の差分 | 端点間変化 |
|---|---:|---:|---:|
| Motional heating | 3.98984e-4 | 3.95430e-4 | -0.891% |
| Motional dephasing | 5.00488e-4 | 3.32876e-3 | 約 6.65 倍 |
| Spin dephasing | 2.66123e-4 | 2.65131e-4 | -0.373% |
| Photon scattering | 5.58914e-4 | 5.56835e-4 | -0.372% |

Motional dephasing をゼロにしたときの全 infidelity は 0.001225445〜0.001233569 に留まる。同じ温度の無散逸参照を引くと、その全温度範囲での変動幅は 9.74e-8。全ノイズの端点間増加の 99.93% は motional dephasing を除去すると消える。これは保存値による比率で、統計的な信頼区間や数値誤差の上限ではない。

この実装の規格化に合わせた一次摂動とも整合する。理想的な一周軌道、A/delta = 1/4、Sx = XI + IX、散逸項 gamma_m D[N] を用いると、変位振幅の時間平均は <|alpha|^2> = 1/8、<|alpha|^4> = 3/128 となる。二量子ビットの平均ゲート infidelity に対する D[Sx] と D[Sx^2] の係数は、それぞれ 8/5 と 16/5。このため、散逸レートの一次までで

\[
\Delta I_m^{(1)}=\gamma_m t_g\left[\frac{2\bar n+1}{5}+\frac{3}{40}\right]
=\gamma_m t_g(0.4\bar n+0.275).
\]

ここで gamma_m = 18 s^-1、t_g = 100 us なので、予測は 4.95e-4 + 7.20e-4 nbar。観測差分の直線フィットは 4.968e-4 + 7.088e-4 nbar。全 5 点で上式との相対差は最大 1.37% である。有限レートでの非線形性と、他の散逸との混合効果を含むデータとの比較である。

同じ規格化では heating の一次寄与は 0.4 * heating_rate * t_g、spin dephasing は 0.8 * gamma_s * t_g、scattering は (1.6 * gamma_Rayleigh + 0.8 * gamma_Raman) * t_g となる。全ノイズの一次予測は 0.001721667 + 0.000720000 nbar で、全 5 点の QPT 結果との差は 1% 未満。

一般的な温度依存性については Sutherland et al., "One- and two-qubit gate infidelities due to motional errors in trapped ions and electrons", Table 2 / Eq. (65), (72) と整合する: https://arxiv.org/html/2111.01913 。上記の具体的係数はこのリポジトリの Lindblad と平均ゲート fidelity の規格化で算出したもので、文献の係数をそのまま移したものではない。

レートに対する線形性は、各レートをゼロにした曲線との差を取った上で、nominal 差分の倍率倍と比較した。0.5〜4 倍の範囲で、最大のずれは heating 0.194%、spin dephasing 0.073%、scattering 0.127%。Motional dephasing では 2 倍で最大 1.12%、4 倍で最大 3.25% の sublinear なずれがある。基準レート近傍の一次摂動は有用だが、高レートほど有限レート効果が見える。

無散逸では理想的な一次 LD の閉じた軌道からゼロ誤差を期待するが、保存結果には最大 8.15e-6 の残差がある。nbar = 0.01 では F_avg = 1.000000010093404 で、raw の 1-F_avg は -1.01e-8。図の 1e-16 は `ms_gate_functions.py` の `max(1.0-f, 1e-16)` による表示用の下限であり、達成精度ではない。残差を物理的な温度依存性として解釈しない。今回の解析は cutoff・時間刻み・solver 許容誤差の新たな収束試験を含まず、数値残差の原因を特定したものでもない。

元の対数軸では下限処理された 1e-16 が縦軸を広げ、ノイズありの曲線差が潰れて見える。`ld1_evaluation.png` は線形軸でモデル間比較と各散逸源の除去差分を示す。元画像、元 CSV、ノートブックは変更していない。

再生成: プロジェクトルートから `python3 results/noise_rate_nbar_sweep_ld1/evaluate_results.py`。集計値は同じ evaluation フォルダの CSV と `metrics.json` に保存される。
