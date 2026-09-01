# Fock 状態からの MS ゲート誤差3成分の解析的導出

## 0. 結論

この計算で用いている有限 Lamb--Dicke (LD) Hamiltonian では、熱運動に由来する主要な誤差生成子は

\[
\mathcal K_{\rm err}
\simeq
h_{XX}\,\mathcal H[XX]
+\gamma_{XX}\,\mathcal D[XX]
+\gamma_{\rm col}\,\mathcal D[IX+XI]
\]

と書ける。ここで

\[
\mathcal H[P](\rho)=-i[P,\rho],\qquad
\mathcal D[L](\rho)=L\rho L^\dagger-\frac12\{L^\dagger L,\rho\}.
\]

3係数の物理的起源は次の通りである。

1. \(h_{XX}\): Fock 数に依存する \(XX\) 回転角の熱平均（coherent mean shift）
2. \(\gamma_{XX}\): Fock 数ごとの回転角のばらつき（stochastic phase dispersion）
3. \(\gamma_{\rm col}\): 位相空間軌道が完全に閉じないことによる残留スピン運動絡み合い（collective-\(X\) dephasing）

以下では、実装中の Hamiltonian からこれらを順に導く。なお `use_full_order=True` の現実装は光学相互作用の厳密な全次数式ではなく、第一側帯波演算子を \(O(\eta^2)\) まで残した Hamiltonian を厳密時間発展している。このノートの解析式も、その実装モデルに対応する。

## 1. 有限 LD 演算子と Fock 依存力

矩形パルスに対する Hamiltonian を

\[
H(t)=A S_x\left(B e^{-i\delta t}+B^\dagger e^{i\delta t}\right),
\qquad S_x=IX+XI
\]

とする。コード中の運動演算子は

\[
B=a-\frac{\eta^2}{2}(N+1)a=a f(N),
\qquad f(N)=1-\frac{\eta^2}{2}N
\]

である。したがって

\[
B|n\rangle=\sqrt n\,f_n|n-1\rangle,
\qquad f_n=1-\frac{\eta^2}{2}n .
\]

通常の LD 近似では \(f_n=1\) であり、すべての Fock 状態が同じ幾何学的位相を得る。有限 LD 補正を入れると力の強さが \(n\) に依存し、これが最初の二成分の起源になる。

## 2. 第二 Magnus 項：Fock 依存 \(XX\) 位相

\(S_x\) 固有値を \(s\in\{2,0,0,-2\}\) とすると、各スピン枝の運動 Hamiltonian は

\[
H_s(t)=g_s\left(B e^{-i\delta t}+B^\dagger e^{i\delta t}\right),
\qquad g_s=sA
\]

である。1ループ \(T=2\pi/\delta\) では第一 Magnus 項は消え、第二項は

\[
\Omega_{2,s}
=i\,2\pi\left(\frac{g_s}{\delta}\right)^2[B,B^\dagger]
\]

となる。交換子は Fock 基底で対角であり、

\[
[B,B^\dagger]|n\rangle=\Delta_n|n\rangle,
\]

\[
\boxed{
\Delta_n=(n+1)f_{n+1}^2-nf_n^2
=1-\eta^2(2n+1)
+\frac{\eta^4}{4}(3n^2+3n+1)
}
\]

を得る。\(s=2\) の forced branch と \(s=0\) の null branch の相対位相を \(2\theta_n\) と定義すると、

\[
\boxed{
\theta_n^{(2)}=\theta_{\rm LD}\Delta_n,
\qquad
\theta_{\rm LD}=4\pi\left(\frac{A}{\delta}\right)^2
}
\]

である。現在のパラメータ \(A=0.125,\ \delta=0.5,\ \eta=0.1\) では \(\theta_{\rm LD}=\pi/4\) であり、

\[
\theta_n^{(2)}
=0.777564
-0.0156490\,n
+5.89049\times10^{-5}n^2 .
\]

数値的に得た \(n=0,\ldots,51\) の位相は

\[
\theta_n^{\rm num}
\simeq0.771842
-0.0154539\,n
+5.75093\times10^{-5}n^2
\]

で、最大残差は \(2.66\times10^{-5}\,\mathrm{rad}\) である。第二 Magnus 項は \(n\) 依存性をよく説明する一方、\(n\) にほぼ依存しない約 \(-5.75\times10^{-3}\,\mathrm{rad}\) のオフセットを残す。これは強駆動・高次 Magnus 項による基準角のずれであり、以下では \(n=0\) の一点で較正して

\[
\theta_n^{\rm ana}=\theta_0^{\rm num}
+\theta_{\rm LD}(\Delta_n-\Delta_0)
\]

と置く。この一点較正後は、温度依存部分を自由パラメータなしで予言できる。

## 3. 熱平均から \(h_{XX}\) と \(\gamma_{XX}\) を得る

平均フォノン数を \(\bar n=\mu\) とすると、熱分布は

\[
p_n=(1-q)q^n,\qquad q=\frac{\mu}{1+\mu}.
\]

Fock 状態ごとの forced--null overlap を

\[
c_n=\langle n|U_{s=2}(T)|n\rangle
=r_n e^{2i\theta_n}
\]

と定義する。位相だけを熱平均した量と、リークも含む完全な overlap はそれぞれ

\[
Z=\sum_{n=0}^{\infty}p_n e^{2i\theta_n},
\qquad
C=\sum_{n=0}^{\infty}p_n c_n
\]

である。目標角を \(\theta_\star=\pi/4\) とすると、三係数は

\[
\boxed{
h_{XX}=\theta_\star-\frac12\arg C
}
\]

\[
\boxed{
\gamma_{XX}=-\frac12\log|Z|
}
\]

\[
\boxed{
\gamma_{\rm col}=-\frac12\log|C|-\gamma_{XX}
}
\]

と抽出できる。\(r_n\) と \(\theta_n\) の相関が弱い極限では、第一式の \(C\) を \(Z\) に置き換えても同じになる。この3式は forced--null coherence を厳密に再現し、後述する弱リーク展開によって各項の物理的意味が分離される。

実際、理想ゲートを除いた forced--null coherence に三成分の生成子を作用させると

\[
e^{-2ih_{XX}}e^{-2\gamma_{XX}}e^{-2\gamma_{\rm col}}

\]

を得る。これは

\[
e^{-2i\theta_\star}C

\]

と一致する。

### 3.1 線形近似での閉形式

\(\theta_n=\theta_0-\kappa n\) と近似すれば、幾何級数から

\[
Z=e^{2i\theta_0}\frac{1-q}{1-qe^{-2i\kappa}}
\]

となる。したがって

\[
\theta_{\rm eff}
=\frac12\arg Z
=\theta_0-\frac12
\operatorname{atan2}
\left(q\sin2\kappa,\,1-q\cos2\kappa\right),
\]

\[
\gamma_{XX}
=\frac14\log
\frac{1-2q\cos2\kappa+q^2}{(1-q)^2}.
\]

特に \(|\kappa|\ll1\) なら

\[
\boxed{
h_{XX}\simeq(\theta_\star-\theta_0)+\kappa\mu,
\qquad
\gamma_{XX}\simeq\kappa^2\mu(1+\mu)
}
\]

である。したがって平均ずれはほぼ \(\bar n\) に比例し、位相分散は熱分布の分散 \(\bar n(1+\bar n)\) に比例する。

### 3.2 二次の Fock 依存性

\(\theta_n=\theta_0+bn+cn^2\) を残す場合、最初の cumulant 近似は

\[
\theta_{\rm eff}\simeq\mathbb E[\theta_n],
\qquad
\gamma_{XX}\simeq\operatorname{Var}(\theta_n)
\]

であり、熱分布のモーメントを用いて

\[
\mathbb E[\theta_n]
=\theta_0+b\mu+c(2\mu^2+\mu),
\]

\[
\operatorname{Var}(\theta_n)
=b^2\mu(1+\mu)
+2bc\,\mu(1+\mu)(4\mu+1)
+c^2\mu(1+\mu)(20\mu^2+12\mu+1).
\]

より高精度には、解析的な \(\theta_n^{\rm ana}\) を上の \(Z\) の幾何重み付き級数へ直接代入すればよい。

## 4. 第三 Magnus 項：残留運動リークと \(\gamma_{\rm col}\)

有限 LD 演算子では \([B,B^\dagger]\) が定数でないため、高次交換子が消えない。1ループの第三 Magnus 項は

\[
\Omega_{3,s}
=i\,2\pi\left(\frac{g_s}{\delta}\right)^3 Q,
\]

\[
Q=[B,[B,B^\dagger]]+[B^\dagger,[B^\dagger,B]].
\]

\(\Delta_n\) を用いると

\[
Q|n\rangle=q_n|n-1\rangle+q_{n+1}|n+1\rangle,
\]

\[
\boxed{
q_n=\sqrt n\,f_n(\Delta_n-\Delta_{n-1}),
\qquad q_0=0
}
\]

となる。したがって forced branch \((s=2,\ g_s=2A)\) における弱リーク確率は

\[
\boxed{
\ell_n^{(3)}
\simeq
\left[2\pi\left(\frac{2A}{\delta}\right)^3\right]^2
\left(q_n^2+q_{n+1}^2\right)
}
\]

である。\(r_n=\sqrt{1-\ell_n}\simeq1-\ell_n/2\) を用いると、位相分散を除いた残留減衰は

\[
\boxed{
\gamma_{\rm col}
\simeq\frac14\sum_n p_n\ell_n^{(3)}
}
\]

となる。これはスピン枝 \(s\) ごとに残留変位が異なるため生じる dephasing であり、結合演算子は \(S_x=IX+XI\) である。特に

\[
\mathcal D[IX+XI]
=\mathcal D[IX]+\mathcal D[XI]
+\mathcal C_{IX,XI},
\]

\[
\mathcal C_{P,Q}(\rho)
=P\rho Q+Q\rho P-\frac12\{PQ+QP,\rho\},
\]

なので、数値的な generator で \(S_{IX}\), \(S_{XI}\), \(C_{IX,XI}\) の係数がほぼ等しかった事実も、この Fock 枝の解析から説明できる。

先頭の \(O(\eta^2)\) だけを残すと \(q_n\simeq-2\eta^2\sqrt n\) であり、

\[
\gamma_{\rm col}
\propto \eta^4\left(\frac{2A}{\delta}\right)^6(2\bar n+1)
\]

となる。定量比較には、上の正確な \(f_n,\Delta_n,q_n\) を用いた熱和の方がよい。

## 5. 既存数値結果との照合

### 5.1 平均角ずれと位相分散

\(n=0\) の位相だけを較正し、第二 Magnus 項の \(\Delta_n\) から温度依存性を予測した。

| \(\bar n\) | \(h_{XX}^{\rm num}\) | \(h_{XX}^{\rm ana}\) | \(\gamma_{XX}^{\rm num}\) | \(\gamma_{XX}^{\rm ana}\) |
|---:|---:|---:|---:|---:|
| 0.01 | 0.0137369 | 0.0137390 | 2.3913e-6 | 2.4541e-6 |
| 1 | 0.0288390 | 0.0290554 | 4.5936e-4 | 4.7158e-4 |
| 2 | 0.0438260 | 0.0442921 | 1.3352e-3 | 1.3720e-3 |
| 3 | 0.0585297 | 0.0592932 | 2.5850e-3 | 2.6602e-3 |
| 4 | 0.0729375 | 0.0740587 | 4.1656e-3 | 4.2966e-3 |

\(\bar n=4\) でも、平均角ずれは約 1.5%、位相分散は約 3.1% の差に収まる。

### 5.2 集団 \(X\) 成分

第三 Magnus 項の \(q_n\) を熱平均した予測は次の通りである。

| \(\bar n\) | \(\gamma_{\rm col}^{\rm num}\) | \(\gamma_{\rm col}^{\Omega_3}\) | 相対差 |
|---:|---:|---:|---:|
| 0.01 | 5.9979e-5 | 6.1330e-5 | +2.25% |
| 1 | 1.6774e-4 | 1.7172e-4 | +2.37% |
| 2 | 2.6556e-4 | 2.7241e-4 | +2.58% |
| 3 | 3.5315e-4 | 3.6312e-4 | +2.82% |
| 4 | 4.3146e-4 | 4.4470e-4 | +3.07% |

また、数値的な Fock leakage \(\ell_n=1-|c_n|^2\) をそのまま

\[
\gamma_{\rm col}\simeq\frac14\sum_n p_n\ell_n
\]

へ代入すると全範囲で 0.84% 以内に一致する。したがって、約3%の残差は主に第三 Magnus 近似の打ち切りであり、\(\gamma_{\rm col}\) を Fock leakage の熱平均と解釈すること自体は非常に精密である。

## 6. 論文で主張できることと限界

この導出から、数値的に見つかった3成分は単なる fit ではなく、同じ有限 LD Hamiltonian の異なる統計量として統一的に説明できる。

\[
\boxed{
\begin{aligned}
h_{XX}&\leftarrow \text{Fock 依存位相の平均},\\
\gamma_{XX}&\leftarrow \text{Fock 依存位相の分散},\\
\gamma_{\rm col}&\leftarrow \text{Fock 依存の残留軌道リーク}.
\end{aligned}
}
\]

特に、単一の角度較正で除けるのは第一成分だけである。第二、第三成分は温度分布による不可逆な混合として残るため、「平均回転角を較正しても誤差床が残る」という数値結果に解析的な根拠を与える。

現段階で解析的に完全でないのは、\(n\) にほぼ依存しない強駆動オフセットと、第三 Magnus 項より上の小さなリーク補正である。したがって論文では「全次数の厳密解」とはせず、

- \(n=0\) の一点較正を施した第二 Magnus 予測
- 第三 Magnus による残留リーク予測
- full numerical evolution / QPT による検証

という構成にするのが正確である。

## 参照データ・実装

- `ms_gate_functions.py`: 有限 LD Hamiltonian
- `phonon_xx_angle_analysis.py`: \(c_n,Z,C\) と三係数の数値抽出
- `noise_error_structure.py`: QPT generator の基底規約
- `results/noise_rate_nbar_sweep/noise_free_diagnostics/fock_resolved/fock_resolved_curve.csv`: Fock 分解データ
- `results/noise_rate_nbar_sweep/noise_free_diagnostics/fock_resolved/fock_thermal_summary.csv`: 熱平均結果
