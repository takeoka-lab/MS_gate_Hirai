# MS-gate current-results physics audit

Date: 2026-08-29

## Bottom line

The strongest physical result currently supported by the saved simulations is
not a large correlation between dissipators. It is a thermally sampled,
finite-Lamb--Dicke correction to the spin-dependent force. In the present
model,

\[
a_{\rm eff}=a-\frac{\eta^2}{2}(n+1)a,
\]

so different Fock sectors undergo different force amplitudes and entangling
angles. The saved Fock-resolved calculation quantitatively separates the
resulting error into

1. a thermal mean-angle shift, observed as coherent \(h_{XX}\);
2. thermal angle dispersion, observed as correlated \(\gamma_{XX}\); and
3. residual spin--motion distinguishability, observed as a collective
   stochastic-X mode with
   \(S_{IX}\simeq S_{XI}\simeq C_{IX,XI}\).

The pairwise-rate data instead show that the four dissipative sources are very
nearly additive under the nominal conditions. Their mixed response is real in
the simulation at large rates, but it is a small correction and should not be
described as evidence of statistically correlated baths.

## Scope and model assumptions

- Gate: rectangular single-mode MS gate, \(t_g=100\,\mu\mathrm{s}\),
  \(A=0.125\), \(\delta=0.5\), \(\eta=0.1\).
- Nominal physical rates: heating/diffusion \(10\,\mathrm{s}^{-1}\), motional
  dephasing \(18\,\mathrm{s}^{-1}\), spin dephasing
  \(3.33\,\mathrm{s}^{-1}\), photon scattering \(4\,\mathrm{s}^{-1}\).
- The switch called `use_full_order` is not an exact all-orders optical
  interaction. It retains the displayed \(O(\eta^2)\) correction to the first
  sideband operator.
- The saved generator coefficients are extracted after a CPTP projection and
  have now been decomposed both into diagonal Pauli dissipators and into the
  complete signed \(H/S/C/A\) trace-preserving generator basis.  The latter
  reconstructs each isolated-source response to relative residual below
  \(1.5\times10^{-12}\).

## 1. Baseline error budget

| \(\bar n\) | nominal infidelity | all-noise-zero infidelity | intrinsic fraction | \(h_{XX}\) [rad/gate] | \(\gamma_{XX}\) [/gate] |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 1.947e-3 | 2.488e-4 | 12.8% | 1.374e-2 | 1.639e-4 |
| 1 | 3.644e-3 | 1.302e-3 | 35.7% | 2.881e-2 | 6.159e-4 |
| 2 | 5.981e-3 | 3.026e-3 | 50.6% | 4.374e-2 | 1.485e-3 |
| 4 | 1.228e-2 | 8.221e-3 | 66.9% | 7.271e-2 | 4.293e-3 |

Thus the low-temperature error is mostly explained by the explicit
dissipators, whereas at \(\bar n=4\) the finite-Lamb--Dicke, noise-free floor is
already about two thirds of the total infidelity.

The dominant fitted Pauli dissipator changes from local \(XI/IX\) at low
temperature to correlated \(XX\) between \(\bar n=2\) and 3. The fitted
\(XX\) fraction of the total diagonal-Pauli rate rises from 7.3% at
\(\bar n=0.01\) to 42.0% at \(\bar n=4\).

## 2. Direct evidence for the finite-Lamb--Dicke thermal mechanism

The saved full-versus-Lamb--Dicke comparison gives a nearly vanishing
noise-free infidelity in the Lamb--Dicke calculation. The difference accounts
for 99.7--100.0% of the noise-free full-model infidelity over the saved
\(\bar n=0.01\)--4 grid.

The first-order phase-space trajectory is closed to numerical precision
(first-order displacement about \(10^{-16}\)). Therefore, the increasing error
is not an ordinary first-order failure to close the ideal MS trajectory.

More importantly, the Fock-resolved model predicts the noise-free QPT
generator without fitting the QPT coefficients:

| mapped quantity | QPT coefficient | largest relative mismatch on saved grid |
|---|---|---:|
| mean Fock-dependent XX angle | \(h_{XX}\) | 0.0052% |
| thermal variance of XX angle | \(\gamma_{XX}\) | 0.36% |
| residual-motion loss | common collective-X coefficient \(S_{IX}\simeq S_{XI}\simeq C_{IX,XI}\) | 1.02% |

This is the clearest present evidence for a physical mechanism. It explains
both the coherent component and two distinct irreversible components.

The comparison in this table is specifically between the Fock-resolved,
noise-free prediction and the all-explicit-noise-zero QPT generator at the
same \(\bar n\).  Point by point,

\[
h_{XX}^{\rm Fock}=\frac{\pi}{4}-\frac12\arg\!\left(\sum_n p_n c_n\right)
\quad\hbox{is compared with}\quad H_{XX}^{\rm QPT},
\]

\[
\gamma_{\rm phase}^{\rm Fock}
=-\frac12\log\left|\sum_n p_n e^{2i\theta_n}\right|
\quad\hbox{is compared with}\quad S_{XX}^{\rm QPT},
\]

and

\[
\gamma_{\rm residual}^{\rm Fock}
=-\frac12\log\left|\sum_n p_n c_n\right|
-\gamma_{\rm phase}^{\rm Fock}
\]

is compared with
\((S_{IX}^{\rm QPT}+S_{XI}^{\rm QPT})/2\).  The quoted number is the maximum
of \(|x_{\rm Fock}-x_{\rm QPT}|/|x_{\rm QPT}|\) over
\(\bar n=0.01,1,2,3,4\).  The newly retained correlation coefficient
\(C_{IX,XI}^{\rm QPT}\) equals the same local-X mean to within
\(1.2\times10^{-8}\) per gate across this grid, and comparison of the Fock
residual with it gives the same 1.02% maximum mismatch.

This equality identifies the reduced stochastic generator as

\[
a\left(S_{IX}+S_{XI}+C_{IX,XI}\right)
=a\,\mathcal D[IX+XI],
\]

up to the stated generator convention.  A forced/null coherence connects
collective-X eigenvalues 2 and 0, so this term multiplies the coherence by
\(e^{-2a}\). Therefore \(-\tfrac12\log|c|=a\), which is why the scalar
residual-motion loss agrees with the mean of the two displayed diagonal
coefficients.  The mean alone did not establish independent local noise; the
off-diagonal \(C_{IX,XI}\) term is essential.

## 3. What each explicit noise source does

The following are isolated nominal-rate infidelity increments above the
all-noise-zero channel:

| source | \(\bar n=0.01\) | \(\bar n=1\) | \(\bar n=4\) | trend |
|---|---:|---:|---:|---|
| motional heating/diffusion | 3.900e-4 | 3.822e-4 | 3.585e-4 | nearly constant |
| motional dephasing | 4.831e-4 | 1.138e-3 | 2.889e-3 | strong thermal growth |
| spin dephasing | 2.665e-4 | 2.662e-4 | 2.639e-4 | nearly constant |
| photon scattering | 5.596e-4 | 5.589e-4 | 5.540e-4 | nearly constant |

The same ordering is obtained from leave-one-out calculations with the other
three rates kept nominal. Motional dephasing is therefore the source of most
of the *additional rate-induced temperature dependence*.

The generator-resolved isolated-source data support the following empirical
mapping.  These statements are based on the directly calculated
\(\Delta K_i=K_i-K_0\) at the nominal rate, not on a fitted rate/temperature
model:

- Heating/diffusion adds almost purely the collective stochastic mode
  \(a\mathcal D[IX+XI]\):
  \(S_{IX}\simeq S_{XI}\simeq C_{IX,XI}=a\).  The coefficient changes only
  from \(2.438\times10^{-4}\) at \(\bar n=0.01\) to
  \(2.256\times10^{-4}\) at \(\bar n=4\).  Its coherent response is a much
  smaller positive \(H_{XX}\), \(7.74\times10^{-6}\) to
  \(9.90\times10^{-6}\) rad/gate.
- Motional dephasing adds the same collective-X mode, but its coefficient grows
  from \(2.217\times10^{-4}\) to \(1.789\times10^{-3}\), an 8.1-fold increase.
  It also adds a separate \(S_{XX}\) component that is roughly constant/slightly
  decreasing, \(1.613\times10^{-4}\) to \(1.278\times10^{-4}\), and a
  negative coherent \(H_{XX}\) response whose magnitude grows from
  \(8.22\times10^{-6}\) to \(2.40\times10^{-4}\) rad/gate.  Thus the thermal
  growth is specifically in the collective-X part, not in its XX stochastic
  part.
- Spin dephasing and photon scattering mainly populate pairs such as
  \((IZ,XY)\), \((ZI,YX)\), \((IY,XZ)\), and \((YI,ZX)\), including nonzero
  C-sector coefficients between the members of each pair.  Their total
  generator norms are nearly temperature independent.  Their Hamiltonian
  response is negligible at the displayed scale.

For all four sources the A-sector (active/nonunital) norm is below 0.1% of the
full response.  The large residual left by a diagonal-Pauli-only fit for the
motional sources was not numerical noise: it is almost entirely the C-sector
coefficient \(C_{IX,XI}\).  This is a correlation *inside the effective error
generator*, caused by the shared gate dynamics; it is not evidence that the
underlying environmental baths are statistically correlated.

## 4. Pairwise-rate nonadditivity

All saved pairwise infidelity interactions are negative. At the largest saved
point, \(m_i=m_j=4\), the largest three are

| pair | largest \(|C_{ij}^{(I)}|\) | location |
|---|---:|---|
| heating x motional dephasing | 4.381e-5 | \(\bar n=4\) |
| motional dephasing x photon scattering | 3.252e-5 | \(\bar n=4\) |
| motional dephasing x spin dephasing | 1.498e-5 | \(\bar n=4\) |

At nominal rates the corresponding values at \(\bar n=4\) are only
\(-2.84\times10^{-6}\), \(-2.15\times10^{-6}\), and
\(-0.984\times10^{-6}\).

For heating x motional dephasing, the full matrix-log interaction norm is only
0.016% of the additive generator change at nominal rates and 0.062% at
\(m_h=m_d=4,\bar n=4\). Across the three generator-resolved pairs and the
checked endpoints, the ratio remains about 0.005--0.10%.

Within the fitted diagonal-Pauli subspace, the large-rate heating x motional
dephasing interaction has

\[
C_{hd}^{(\gamma,IX/XI)}>0,\qquad
C_{hd}^{(\gamma,XX)}<0,
\]

and is approximately bilinear in the two multipliers. The local \(IX/XI\)
bilinear model is strongly supported at every saved temperature. The
\(\gamma_{XX}\) model is not resolved at \(\bar n=0.01\), where that response
is tiny, and the coherent \(h_{XX}\) bilinear model becomes inadequate at the
highest temperatures.

Two qualifications are essential:

1. About 69% of the full \(C_K\) Frobenius norm at
   \((\bar n,m_h,m_d)=(4,4,4)\) is outside the current Hamiltonian plus
   diagonal-Pauli interaction fit. Therefore the reported `78:22` local/XX
   split is a split *inside the fitted diagonal-Pauli part*, not of the entire
   pairwise interaction.
2. The nominal pairwise signals are comparable to the scale that has not yet
   been directly converged for the mixed difference. A convergence calculation
   of \(C_K\), not only of each infidelity, is required before interpreting the
   nominal mixed coefficients.

The safe conclusion is near-additivity with a small dynamical cross-response,
not correlated baths.

## 5. Generator feedback and the irreversible floor

The saved all-noise amplitude feedback reduces \(|h_{XX}|\) by more than two
orders of magnitude. Nevertheless, the infidelity reduction is limited:

| \(\bar n\) | before | after | fraction removed | \(\gamma_{XX}\) factor after/before |
|---:|---:|---:|---:|---:|
| 0.01 | 1.947e-3 | 1.819e-3 | 6.6% | 1.04 |
| 1 | 3.644e-3 | 3.104e-3 | 14.8% | 1.08 |
| 2 | 5.981e-3 | 4.796e-3 | 19.8% | 1.12 |
| 4 | 1.228e-2 | 9.366e-3 | 23.7% | 1.21 |

This supports the physical distinction between the correctable thermal mean
angle and an irreversible floor produced by thermal angle dispersion,
spin--motion distinguishability, and explicit dissipators. The increase in
\(\gamma_{XX}\) after retuning also shows that cancelling the mean coherent
angle does not cancel the Fock-sector spread.

## 6. Pauli approximation audit

Before feedback, a Pauli-only approximation misses about 99% of the error-PTM
norm because coherent \(h_{XX}\) dominates. Adding the fitted Hamiltonian
generator reduces the relative channel error to about 4.7--5.4%.

After feedback, the Pauli-only channel still misses 38--46% of the residual
error-PTM norm over \(\bar n=0.01\)--4. The unmodeled generator is almost
entirely a symmetric off-diagonal block in the traceless Pauli sector, rather
than a nonunital affine term. Consequently, the existing data do **not** yet
support the statement that generator feedback makes the residual channel
Pauli-diagonal.

The previously quoted 4.3% median unmodeled/full-generator ratio is misleading
for this question because its denominator is dominated by the large coherent
\(h_{XX}\). Relative to the non-Hamiltonian generator, the omitted fraction is
about 39--46%.

## 7. Control results that support the mechanism

- A smooth-detuning pulse reduces the noise-free floor and motional-noise
  penalties. With all noise present it is slightly worse at \(\bar n=0.01\)
  because its duration is 225.8 microseconds, but it improves infidelity by
  factors 1.23--1.31 for \(\bar n=1\)--4. The saved ablation attributes the
  improvement mainly to heating and motional-dephasing suppression, opposed by
  a nearly temperature-independent spin/optical penalty from the longer gate.
- The separate amplitude-pulse optimization produces only a small improvement
  at high temperature and worsens the \(\bar n=0.01\) result. Its robustness
  ensemble is explicitly a placeholder, so it is not presently a physics
  conclusion.

## 8. Numerical and completeness audit

- The five targeted analysis test files pass: 23 tests passed.
- The baseline CPTP projection is generally negligible; at isolated grid
  points it corrects small solver-induced non-CP eigenvalues.
- The saved cutoff/time-grid test changes the noise-free infidelity by about
  \(10^{-6}\) at \(\bar n=4\). This is negligible for the main thermal floor,
  but not automatically negligible for nominal pairwise differences of a few
  \(10^{-6}\).
- Several broader physical-control and parameter-robustness grids are marked
  pending. They should not be used for final claims.

## Claims currently supported

1. The finite-Lamb--Dicke correction makes the MS force and geometric phase
   Fock-state dependent.
2. Thermal averaging converts this into a correctable mean \(XX\) underrotation,
   an irreversible \(XX\) phase-dispersion term, and a collective-X stochastic
   term from residual spin--motion distinguishability.
3. Motional dephasing is the explicit noise source responsible for most of the
   rate-induced temperature dependence over the chosen nominal model.
4. The four explicit dissipators are almost additive; heating x motional
   dephasing gives the largest mixed correction, but the correction is small.
5. Cancelling \(h_{XX}\) exposes a sizable irreversible, non-Pauli-diagonal
   generator floor.

## Claims not yet supported

- Statistical correlation between environmental baths.
- A large or practically dominant pairwise dissipator interaction.
- Validity of a diagonal Pauli approximation after feedback.
- An exact all-orders finite-Lamb--Dicke model.
- Interpretation of tiny individual interaction coefficients below a direct
  mixed-difference convergence floor.

## Minimum next analyses

1. Extend the completed full \(H/S/C/A\) decomposition from isolated-source
   responses to the full mixed derivative, rather than subtracting
   independently fitted NNLS Pauli rates.
2. Perform centered, small-step rate derivatives around the nominal point and
   converge the mixed quantity \(C_K\) itself in phonon cutoff and time step.
3. Promote the Fock-resolved mapping to an analytic derivation from
   \(a_{\rm eff}\), including explicit formulas for the mean angle, angle
   variance, and residual-motion coherence.
4. Compare exact and Pauli-approximated residual channels with an operational
   metric or a short QEC simulation after feedback.
