# Rate preservation and linearity of isolated-noise H/S/C/A structure

Date: 2026-08-29

## Scope

Each physical rate is varied alone with the other three explicit rates set to
zero.  The saved multipliers are

\[
m=0,0.5,1,2,4,
\qquad \Gamma=m\Gamma_{\rm nominal},
\]

at \(\bar n=0.01,1,2,3,4\).  The response is

\[
\Delta K_i(\Gamma,\bar n)=K_i(\Gamma,\bar n)-K_0(\bar n).
\]

Thus all statements below are established only over 0.5--4 times the nominal
rate, not for mathematically arbitrary or asymptotically large rates.

## Collective X

With \(X_1=XI\) and \(X_2=IX\), collective X means that the same stochastic
rotation acts on both ions:

\[
L_X=X_1+X_2=XI+IX.
\]

For a single realization, a common random angle gives

\[
U_\epsilon=e^{-i\epsilon(X_1+X_2)}
=e^{-i\epsilon X_1}\otimes e^{-i\epsilon X_2}.
\]

Averaging over the common fluctuation produces

\[
\mathcal D[X_1+X_2]
=S_{IX}+S_{XI}+C_{IX,XI}.
\]

It is therefore neither two independent local-X channels nor the single Pauli
channel \(\mathcal D[XX]\).  In the X basis it dephases states with different
total-X eigenvalues \(+2,0,0,-2\).

## Structure-preservation tests

The stochastic S/C matrix shape is compared with its nominal-rate value using

\[
\cos\theta_B(m)=
\frac{\langle B(m),B(1)\rangle_F}
{\|B(m)\|_F\|B(1)\|_F}.
\]

Unity means the normalized stochastic structure is identical, independent of
its overall strength.

| source | minimum stochastic-shape cosine | maximum 1-cosine | conclusion |
|---|---:|---:|---|
| motional heating | 0.999999800 | 2.00e-7 | structure preserved |
| motional dephasing | 0.999963698 | 3.63e-5 | structure very nearly preserved, with small high-temperature curvature |
| spin dephasing | 0.999997665 | 2.33e-6 | stochastic subspace preserved |
| photon scattering | 0.999999567 | 4.33e-7 | stochastic subspace preserved |

For heating and motional dephasing the leading stochastic eigenvector has
overlap at least 0.9999999977 with
\((IX+XI)/\sqrt2\).  The collective relations are preserved across the rate
grid:

| source | range of C(IX,XI)/mean[S(IX),S(XI)] | range of S(IX)/S(XI) |
|---|---:|---:|
| heating | 0.9999388--1.0000063 | 0.9998854--1.0002307 |
| motional dephasing | 0.9999866--1.0000028 | 0.9998983--1.0000935 |

For spin and photon scattering, the paired blocks
\((IZ,XY)\), \((ZI,YX)\), \((IY,XZ)\), and \((YI,ZX)\) are also stable at
fixed temperature.  Across the rate grid the selected C/paired-S ratios vary
by at most 0.35% for spin dephasing and 0.21% for photon scattering.  Their
absolute ratios do change with temperature, so rate invariance should not be
confused with temperature invariance.

## Matrix-level rate linearity

For each temperature, the complete response is fitted through the origin:

\[
\Delta K(m)\simeq mD,
\qquad
D=\frac{\sum_m m\Delta K(m)}{\sum_m m^2}.
\]

The quoted residual is

\[
\epsilon_K=
\frac{\sqrt{\sum_m\|\Delta K(m)-mD\|_F^2}}
{\sqrt{\sum_m\|\Delta K(m)\|_F^2}}.
\]

| source | full-generator residual range over nbar | S/C-matrix residual range | assessment |
|---|---:|---:|---|
| heating | 2.58e-5--9.14e-5 | 2.42e-5--8.78e-5 | linear |
| motional dephasing | 1.21e-3--6.65e-3 | 1.20e-3--7.88e-3 | near-linear but resolved curvature |
| spin dephasing | 9.30e-5--2.81e-4 | 1.68e-4--4.98e-4 | linear |
| photon scattering | 4.69e-5--1.12e-4 | 6.96e-5--2.00e-4 | linear |

The H and A sectors of spin/photon are too small for a useful relative
linearity test: their large relative residuals come from dividing by a
near-zero signal.  The physically resolved S/C coefficients remain linear.

## Coefficient-level linearity

The most transparent point test is

\[
q_k(4)=\frac{k(4\Gamma_{\rm nominal})}
{4k(\Gamma_{\rm nominal})}.
\]

Exact rate linearity gives \(q_k(4)=1\).

### Motional heating

- \(S_{IX}\): maximum deviation from unity 5.5e-5.
- \(C_{IX,XI}\): maximum deviation 7.0e-5.
- \(H_{XX}\): maximum deviation 0.36%.
- \(S_{XX}\): up to 4.7% deviation, but this coefficient is tiny compared
  with the collective-X coefficient and does not affect the full-matrix
  conclusion.

Thus the physically dominant heating response is rate-linear to much better
than 0.01% over the saved interval.

### Motional dephasing

At \(\bar n=4\),

\[
q_{S_{IX}}(4)=0.9780,
\qquad
q_{C_{IX,XI}}(4)=0.9780,
\]

\[
q_{H_{XX}}(4)=0.9732,
\qquad
q_{S_{XX}}(4)=1.2118.
\]

The collective-X coefficients are therefore about 2.2% sublinear at four
times nominal rate.  The coherent XX response is also sublinear.  The separate
stochastic XX coefficient is superlinear and is 21.2% above the nominal-rate
linear extrapolation at \(\bar n=4\).  The ratio of the XX stochastic mode to
the collective-X mode consequently increases by about 24%.

At lower temperatures the curvature is smaller.  Hence motional dephasing
retains its dominant Pauli axis but not all internal mode ratios exactly.

### Spin dephasing

For the resolved coefficients \(S_{IZ},S_{XY},C_{IZ,XY},S_{IY},S_{XZ}\), and
\(C_{IY,XZ}\), the largest four-times-nominal deviation from linear scaling is
0.153%.  The corresponding origin-fit \(R^2\) values are all above
0.9999991.

### Photon scattering

For the same resolved paired-block coefficients, the largest
four-times-nominal deviation is 0.086%.  The origin-fit \(R^2\) values are all
above 0.9999997 for the displayed main coefficients.

## Bottom line

1. The dominant Pauli axes and stochastic subspaces are preserved throughout
   the saved 0.5--4 times nominal-rate grid.
2. Heating, spin dephasing and photon scattering are effectively rate-linear
   at the full-generator and resolved-coefficient levels.
3. Motional dephasing is only approximately linear.  Its collective-X axis is
   preserved, but at high temperature the collective-X coefficient becomes
   sublinear while the separate stochastic-XX coefficient becomes
   superlinear.
4. Therefore a first-order model \(K=K_0+\Gamma K^{(1)}\) is excellent for
   heating/spin/photon over this grid, whereas motional dephasing needs a
   quadratic rate term if percent-level accuracy is required at high rate and
   high temperature.
