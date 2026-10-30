# 論文草稿の方針と根拠

2026-10-05。今後の本文は [manuscript.tex](manuscript.tex) を編集する。
既存の methods_draft.tex は今回統合したMethodsの元原稿として保持した。
旧ノート・旧パラグラフ案は、それぞれの時点の検討記録である。

## 採用した中心主張

> 有限Lamb–Dicke補正と物理雑音を含むMSゲートについて、
> 振幅較正後のPauli近似の誤差を定量化し、
> どの成分がその誤差を支配するかを明らかにする。

三成分分解、既知の熱的機構、QPTそれ自体を新規性の中心には置かない。
新しい予測モデルやQECシミュレーションの開発は、現原稿の完成条件には含めない。
「どの成分が支配するか」は評価指標を明示する。
現段階で示せるのは、保存チャネルの非対角Frobenius二乗ノルムとBell状態の差に対する
collective-X交差項の寄与であり、微視的雑音源別の寄与率は別の検証である。

## 本文の流れ

1. Introduction：既知の機構・チャネル研究を認めたうえで、較正後の近似誤差評価を目的に置く。
2. Methods：有効Hamiltonian、散逸、数値QPT、post-gate規約、較正、生成子、距離を定義する。
3. Results 3.1：較正によりhXXとinfidelityがどう変わるか。
4. Results 3.2：同じ較正後チャネルとそのPauli twirlとの距離。
5. Results 3.3：捨てられる非対角成分がcollective交差項のパターンに集中すること。
6. Results 3.4：そのパターンがBell状態生存確率の過小評価・過大評価を説明すること。
7. Discussion：先行研究との関係、近似としてのtwirlと物理的twirlの違い、対象範囲。

図1は保存5点の距離と相対Frobenius誤差を示す。
表1は物理パラメータ、表2は較正、表3は交差項の寄与、表4はBell比較。
図の座標と参考文献はTeXに埋め込み、単独でコンパイルできるようにした。

## 今回確認した数値と出典

主入力は以下の保存データで、新しいmaster equation計算は実行していない。

- results/pauli_twirl_channel_distance/inputs/projected_chi_before_after.npz
- results/pauli_twirl_channel_distance/channel_distances_paired.csv
- results/calibration_channel_comparison/three_component_model_diagnostics.csv

今回の再評価は次のコマンドで再現できる。

    python3 output/manuscript_draft/build_evidence.py

出力は results/manuscript_draft/ に保存した。

| 主張 | 根拠 | 留意点 |
|---|---|---|
| 較正後にdiamond距離が低下する | 保存5点のprimal/dual SDPと距離CSV | 全点の最終ODE/cutoff収束を意味しない |
| 実のcollective交差パターンが二乗ノルムの96.6–98.8% | channel_summary.csv | χ非対角部分への直交射影。diamond距離・infidelityの寄与率ではない |
| Bell状態のinfidelity差はほぼ±2w | bell_state_comparison.csv | 保存行列上の関係。最大残差約2.18e-8は物理計算の誤差評価ではない |
| nbar=4のBell測定差はdiamond距離の約96.1%に達する | 同CSVと距離CSV | 特定入力・測定による下界。最適入力と断定しない |

交差パターンは

    Δ_col(ρ) = w [IX ρ XI + XI ρ IX − XX ρ − ρ XX]
    w = (Re χ[IX,XI] − Re χ[II,XX]) / 2

で、χの4つの実エントリに対応する。
これはチャネル差を構成する線形写像であり、単独のCPTPチャネルではない。
有限チャネルの係数wとlog Rから得る生成子係数cを混同しない。
チャネルのPauli twirlと、生成子を対角化して指数を取る操作も混同しない。

再評価では、入力温度の対応、canonical χ規約、ChoiのCPTP許容誤差、
保存infidelityとの一致、直交射影のノルム分解を確認した。
ソースファイルのSHA256は results/manuscript_draft/manifest.json に保存した。

## 先行研究との位置づけ

- [Sørensen–Mølmer (2000)](https://arxiv.org/abs/quant-ph/0002024), Sec. III.B, Eqs. (17), (25), (26)：
  平均結合の補正とフォノン分散に由来する残差は既知。
- [Kirchhoff–Wilhelm–Motzoi (2025)](https://arxiv.org/abs/2404.17478), Table I, Sec. IV–V：
  高次誤差演算子、振幅補正、波形変更を既に解析している。
- [Tinkey et al. (2021)](https://arxiv.org/abs/2101.04648), Sec. 4：
  較正状態の異なるMSゲートの誤差χ構造を実験的に評価している。
- [Ma et al. (2025)](https://arxiv.org/abs/2411.00751), Eq. (1)：
  運動dephasingのKrausチャネルにcollective-spin構造を導出済み。
  今回最優先で同じ条件へ合わせる比較対象。
- [Blume-Kohout et al. (2022)](https://arxiv.org/abs/2103.01928)：
  H/S/C/A分類と縮約モデル構成は既存手法。

これらの機構を初めて発見した、先行研究はfidelityしか見ていない、
今回の設定の組合せが文献にないので直ちに新規である、といった主張は置いていない。

## 原稿を完成させるために残る検証

1. **最終観測量の数値収束**：
   cutoff・係数補間格子・ODE許容誤差を変え、較正後の距離・w・Bell差を検査する。
2. **Maらのモデルとの比較**：
   LD・運動dephasing単独・同じ軸/結合規約で照合し、
   有限LD補正・較正後という今回の評価が何を追加するかを確認する。
3. **物理雑音源の切り分け**：
   較正後の制御を固定したablationで、熱的縮約と外部散逸が交差項に寄与する大きさを調べる。
   各ablationで再較正する比較とは問いが異なる。
4. **射影と残留Hamiltonianの影響**：
   raw/CPTP射影の影響と、未補正のHamiltonian成分が主要結論を変えないか確認する。

これらは本文の内部用付録にも明記した。
主張の方向は固定し、検証結果に応じて数値・適用範囲を更新する。
