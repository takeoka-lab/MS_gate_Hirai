# Methods原稿のレビュー（2026-10-06）

対象：[貼付原稿](</Users/hirainoa/.codex/attachments/f58d2d4f-99b8-4559-87d5-cb463bb8695f/Pasted text.txt>)。原稿・実装は変更していない。主張を「較正後のPauli近似の誤差構造とBell状態への影響」とした場合のレビュー。

## 全体評価

Hamiltonian、理想ゲートの符号・係数、post-gate規約、H/S/C/A基底、平均不忠実度、twirl、距離の規格化は、確認した実装・定義と整合している。数式を全面的に改める必要はない。

優先する修正は、(1) 先行研究との規格化・周波数の対応を明記する、(2) 中心主張に使うチャネル上の交差成分・寄与率・Bell観測量を定義する、(3) 補足資料へ移す実装詳細を整理する、の三点。式の整合性の確認は、近似モデルの物理的妥当性や数値収束の検証を代替しない。

## 1. 先行研究との記号・規約の対応

| 項目 | 確認結果と推奨 |
|---|---|
| 集団スピン | 本文の $S_x=XI+IX$ はMaら・Kirchhoffらの $J_x=(XI+IX)/2$ の2倍。現在の定義を維持して $S_x=2J_x$ を一文で示す。したがって $\mathcal D[S_x]=4\mathcal D[J_x]$。係数を直接比較しない。 |
| 駆動振幅 $A$ | 有効側帯波結合を表す記号として適切。先行研究のcarrier Rabi frequency $\Omega$ と同一ではない。Maらの補足Eq. (4)の線形力を通常の $x,p$ 定義で書き直すと、軸・光位相を合わせた結合の大きさは $\lvert A\rvert=\eta\lvert\Omega\rvert/2$。光学モデル全体の同一性を意味しない。 |
| 離調 $\delta$ | 本文では運動側帯波からの小さい離調。Maら・Kirchhoffらの $\delta$ は二色光のcarrierからの離調で、本文の量はその規約の $\nu-\delta$ に対応する。この違いは比較時に必ず示す。本文だけなら「residual detuning from the motional sideband」と明記すればよい。 |
| ゲート角・符号 | $U_\star=e^{+i\phi_\star XX}$、$\phi_\star=\pi/4$ は内部整合。$R_{XX}(\vartheta)=e^{-i\vartheta XX/2}$ を使う規約なら $\vartheta=-2\phi$。記号だけを変更して係数を据え置かない。 |
| H/S/C/A | Blume-KohoutらのEqs. (10), (14)–(16)と一致。$K=\log R$ の $K$ を同論文の $L$ に合わせる必要はない。 |
| post-gate規約 | $\mathcal G=\mathcal E\circ\mathcal U_\star$ はBlume-Kohoutらの $G=\mathcal E\bar G$ と一致。現記載を維持する。 |
| χ・PTM・Choi | 標準Pauli基底、$\operatorname{Tr}\chi=1$、PTMの係数1/4、非規格化Choiの出力部分トレース $I_4$ は整合。規格化を先行研究に合わせて無断で変えない。 |

出典：[Ma et al. (2025), 本文Eq. (1)](https://arxiv.org/html/2411.00751v2)、[同補足Sec. II, Eq. (4)](https://arxiv.org/pdf/2411.00751)、[Kirchhoff et al. (2025), Eqs. (12)–(14)](https://arxiv.org/html/2404.17478v3)、[Blume-Kohout et al. (2022), Secs. II–III](https://arxiv.org/html/2103.01928)。

**本文に追加する最小の英文例：**

> We use $S_x=X\otimes I+I\otimes X=2J_x$, where $J_x$ is the conventionally normalized collective-spin operator. The parameter $A(t)$ denotes an effective sideband coupling, and $\delta(t)$ is the residual detuning from the motional sideband.

Maらとの比較箇所または補足資料に、軸・振幅・離調・散逸演算子の変換表を置く。本文全体を各論文の記号へ統一する必要はない。

## 2. 主張に対応する定義を追加する

貼付Methodsには、チャネルの交差パターンへの射影とBell状態の比較が未記載。既存manuscript.texではResults中に定義されているため、Methodsへ移す場合は重複を避ける。Methodsで定義し、Resultsでは測定した数値と解釈を述べる構成を推奨する。

### 2.1 チャネル上の集団的交差パターン

$$
\Delta_{\rm col}(\rho)=w\bigl(IX\rho XI+XI\rho IX-XX\rho-\rho XX\bigr),
\qquad
w=\frac{\operatorname{Re}\chi_{IX,XI}-\operatorname{Re}\chi_{II,XX}}{2}.
$$

$\chi_{\rm col}$ を $\Delta_{\rm col}$ のχ行列とする。非ゼロ成分は $(IX,XI),(XI,IX)$ の $+w$、$(II,XX),(XX,II)$ の $-w$ の4成分。

$$
q_{\rm col}=
\frac{\|\chi_{\rm col}\|_F^2}
{\|\chi-\operatorname{diag}(\chi)\|_F^2}.
$$

この定義が既存結果の96.6–98.8%に対応する。$f_F$ はPTM誤差のノルム比であり、別の量。

**英文例：**

> To quantify the contribution of the collective cross pattern, we project the off-diagonal part of $\chi$ onto the real pattern defined above. We report its squared Frobenius weight $q_{\rm col}$. This weight quantifies a matrix decomposition and is not a fraction of the diamond distance or gate infidelity. The map $\Delta_{\rm col}$ is a component of the channel difference, not a standalone CPTP channel.

有限チャネルの $w$、完全生成子の $c_{IX,XI}$、三成分フィットの $\gamma_{\rm col}$ を区別する。現在の分解結果だけで交差項の微視的雑音源まで一意に同定したとは書かない。

### 2.2 Bell状態の生存不忠実度

$$
|\Phi^+\rangle=(|00\rangle+|11\rangle)/\sqrt2,
\qquad
|\Psi^-\rangle=(|01\rangle-|10\rangle)/\sqrt2,
$$
$$
r_\psi(\mathcal E)=1-\langle\psi|\mathcal E(|\psi\rangle\langle\psi|)|\psi\rangle.
$$

両状態はXXの固有状態なので、理想ゲートはそれらの密度行列を変えない。このため、ここでの生存不忠実度は理想ゲート出力との比較に一致する。一般の入力状態へ無条件に同じ説明を広げない。

$$
\delta r_\psi=r_\psi(\mathcal E)-r_\psi(\mathcal E_{\rm P}),
\qquad
\delta r_{\Phi^+}^{\rm col}=2w,
\quad
\delta r_{\Psi^-}^{\rm col}=-2w.
$$

最後の2式は交差パターン単独の寄与に対して厳密。元のチャネル差では他成分の残差を加える。保存行列における近似関係の精度はResultsで報告する。

## 3. 意味が明確になる表現修正

### 3.1 Fock解析の「Hamiltonian contribution」

現文：

> To isolate the Hamiltonian contribution of thermal motion, ...

推奨：

> To characterize intrinsic thermal errors arising from unitary spin–motion dynamics, we set all explicit dissipative rates to zero and resolve the evolution by the initial motional Fock state.

理由：全体系の発展がunitaryでも、運動をトレースしたスピンチャネルには非unitary成分が生じる。現在の言葉だと、H/S/C/AのH成分だけを調べるようにも読める。

### 3.2 三成分モデルのHamiltonian係数抽出

現文の「obtained from the antisymmetric part」だけでは、反対称部分全体をHamiltonianと同一視したように読める。次のcalibration節では射影が明記されており、実装も射影している。

推奨：

> We extract $h_{XX}$ by least-squares projection of the antisymmetric part of $K$ onto the PTM representation of $\mathcal H_{XX}$, and fit the two dissipative coefficients to the symmetric part using nonnegative least squares.

完全H/S/C/A分解と三成分フィットの係数を区別している現記載は維持する。

### 3.3 Spin dephasingの $T_2^*$

$(\Gamma_s/2)\mathcal D[Z]$ は、単独では $\rho_{01}(t)=e^{-\Gamma_s t}\rho_{01}(0)$ を与える。この因子は実装と整合している。ただし、本稿は時間一定のMarkovian dephasingモデルなので、$T_{2,s}$ の方が解釈しやすい。

推奨：

> We model independent Markovian spin dephasing with $\Gamma_s=1/T_{2,s}$, where $T_{2,s}$ is the exponential coherence-decay time in the absence of the other noise terms.

もし $T_2^*$ を実験値との対応のため残すなら、「その値を有効な指数減衰時間として使用する」というモデル化を明記する。記号変更は必須ではなく、実験の準静的揺らぎを再現したという誤解を防ぐための提案。

### 3.4 散乱・運動dephasingの係数

現在の式は実装と一致する。係数は演算子の規格化とセットで読む。

- $\Gamma_{\rm Ray}\mathcal D[Z]$ 単独によるスピンcoherence減衰率は $2\Gamma_{\rm Ray}$。
- 対称Raman項のcoherence減衰率は $\Gamma_{\rm Ram}/2$、population differenceの減衰率は $\Gamma_{\rm Ram}$。
- $\Gamma_m\mathcal D[N]$ による $\rho_{nm}$ の減衰率は $\Gamma_m(n-m)^2/2$。

パラメータ表の「coefficient」は適切。測定された散乱光子数率と無条件に同一視しない。必要なら補足資料で上記の換算を示す。未定義の $\sigma_j^\pm=(X_j\pm iY_j)/2$ は一度定義する。

### 3.5 較正の実行回数と目的

更新式・符号・停止目標は実装と整合。現在の5点は全点1回の更新で終了しているため、AUTHOR CHECKを次の本文へ置き換えられる。

> We allow at most two amplitude updates and stop when $ |h_{XX}|<2\times10^{-3}$. All five baseline conditions satisfy this criterion after one update; the achieved residuals are reported in the Results.

この手順は $h_{XX}$ を減らす較正であり、平均不忠実度を最小化する最適化ではない。「optimal calibration」などとは呼ばない。散逸係数、特に散乱係数を固定した比較であるという記載は残す。

### 3.6 冒頭の短縮例

> We reconstruct the reduced two-qubit channel of an MS gate from master-equation simulations and factor out the target unitary. We then adjust the drive amplitude to suppress the coherent XX error and quantify the remaining discrepancy from the channel’s Pauli twirl, including its collective cross component and its effect on Bell-state survival probabilities.

Fock解析を補足資料へ移す場合、冒頭で独立した主解析として紹介する必要はない。

## 4. 本文から補足資料へ移す候補

| 内容 | 推奨する扱い |
|---|---|
| Hamiltonian、B演算子、相対 $O(\eta^2)$ の近似範囲 | 本文に残す。full optical modelではない点も短く残す。近似の出典を付す。 |
| 熱初期状態・master equation・物理パラメータ表 | 本文に残す。初期温度と動的雑音を別のパラメータとして扱う点は一文でよい。 |
| 16入力の列挙・$M_{\rm out}M_{\rm in}^{-1}$ | 本文では16個の線形独立な積状態とだけ記し、具体的な入力集合・再構成式は補足へ移せる。 |
| χ・PTM・post-gate規約・CPTP処理 | 本文に短く残す。Dykstraの詳細や許容誤差は補足へ。 |
| 完全H/S/C/Aの4式・240次元・NNLS詳細 | 主結果で全分類を使うなら本文に残してもよい。現在の中心主張なら、本文は $h_{XX}$ と集団的交差項の説明に絞り、完全分解・三成分フィットは補足へ移せる。 |
| Fock解析全節 | 物理機構の裏付けとしてSupplementary Methodsへ。本文にFock解析の結果を大きく掲載する場合のみ、最小限の定義を残す。 |
| cutoffの経験式・無次元実装単位・solver設定 | 補足へ。本文は実際に採用した条件と収束確認の要約を示す。 |
| $D_J$ | 本文の結果・議論で使わなければ補足へ。$\epsilon_P$、$q_{\rm col}$、Bell差を優先する。 |
| QEC未計算・ゲート間の運動memory | Discussionの適用範囲へまとめる。 |

削除・統合しやすい文：

- “The $\bar n=0$ case is understood as the motional ground state.”：現在の5点には0がなく、熱状態の定義からも分かる。
- “Trace normalization of $\chi$ alone is not used as a physicality test.”：直前にChoiのCP/TP条件を示しているので省略可能。
- “Logical error rates of a particular QEC circuit are not computed ...”：Discussionへ。
- “current”, “baseline implementation”, AUTHOR CHECK：研究途中の記録としての言葉は最終原稿から除き、採用した条件・実施済みの手順を書く。

残すべき区別：channel twirlとgeneratorの対角化、物理的にランダム化したゲートとモデル上のPauli近似、固定した散乱率、第一側帯波の有限次数近似、$w$ と生成子係数、収束検証とSDP gap。

## 5. 再現性・数値設定で残る確認

1. cutoff・係数補間格子・ODE tolerancesを変えたときの最終観測量（$\epsilon_P,w,q_{\rm col},\delta r_\psi$）の変化を示す。設定値の列挙だけでは収束を示せない。未実施の結果を本文に書き足さない。
2. QuTiP由来の保存χから標準Pauli規約への変換を補足資料に示す。使用中の実装では、raw to_chiの16による規格化と、奇数個のYに対応する符号変換が必要。距離・新しいBell比較のコードでは変換済み。
3. 未射影とCPTP射影後のデータを図・表で区別する。数値補正が交差項やBell差を支配していないことを示す。
4. QuTiP・SciPy・CVXPY・SCSの実際に使用したバージョンをコード公開情報または補足資料に記す。今回、新しい時間発展・収束計算は実施していない。
5. \\ii, \\Diss, \\Tr, \\ket, \\bra, booktabs等は貼付部分の外に定義が必要。既存manuscript.texには該当するマクロがある。今回はLaTeXファイルを編集・コンパイルしていない。

## 6. 推奨するMethodsの構成

1. Physical model and dissipative dynamics
2. Channel reconstruction and conventions
3. Error generator and amplitude calibration
4. Pauli approximation, collective cross component, and Bell-state diagnostics
5. Numerical implementation（短い概要、詳細と収束検証は補足資料）

冷却・波形変更の比較は、追加結果を本論文に採用する時点で、その資源制約・散乱率の扱い・較正目的を追記する。現段階の主Methodsに未実施のプロトコルを含めない。

## 確認した実装

- `ms_gate_functions.py`：Hamiltonian、散逸、16入力、時間・周波数の単位、理想ゲート、cutoff選択。
- `drive_calibration_qpt_analysis.py`：CPTP射影、Hamiltonian射影。
- `drive_amplitude_calibration.py`：更新式とunderrotationの符号。
- `noise_error_structure.py`：H/S/C/Aの完全基底。
- `angle_calibration_channel_comparison.ipynb` の `analyze_channel`：三成分フィット。
- `phonon_xx_angle_analysis.py`：Fock overlap、熱分布、tailとbuffer。
- `pauli_channel_distance.py`：canonical χ、twirl、距離の1/2、SDP。
- `output/manuscript_draft/build_evidence.py`：交差成分の射影とBell状態。

追加の小さな代数確認では、$\mathcal D[S_x]=\mathcal D[IX]+\mathcal D[XI]+\mathcal C_{IX,XI}$、Bell差の係数 $\pm2w$、物理単位での $A_0/(2\pi)=2.5\,\mathrm{kHz}$ を確認した。
