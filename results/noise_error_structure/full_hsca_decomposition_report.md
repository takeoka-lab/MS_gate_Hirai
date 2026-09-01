# Complete H/S/C/A decomposition

Date: 2026-08-29

## 1. Scope

This report contains the complete two-qubit trace-preserving generator
decomposition for the saved temperatures

\[
\bar n=0.01,1,2,3,4.
\]

Two distinct objects are decomposed:

1. the all-explicit-noise-zero channel itself,
   \(K_0(\bar n)=\log R_0(\bar n)\);
2. each isolated nominal-noise response,
   \(\Delta K_i(\bar n)=\log R_i(\bar n)-\log R_0(\bar n)\).

The nominal rates used for the second object are heating
\(10\,\mathrm{s}^{-1}\), motional dephasing \(18\,\mathrm{s}^{-1}\), spin
dephasing \(10/3\,\mathrm{s}^{-1}\), and photon scattering
\(4\,\mathrm{s}^{-1}\).

All channels are CPTP-projected before the Pauli-transfer-matrix logarithm is
taken.  The isolated response is a signed difference of two generators; it is
not assumed to be a completely positive generator by itself.

## 2. Basis definition and coefficient counts

For nonidentity two-qubit Paulis \(P,Q\), the convention is

\[
H_P(\rho)=-i[P,\rho],
\]

\[
S_P(\rho)=P\rho P-\rho,
\]

\[
C_{P,Q}(\rho)=P\rho Q+Q\rho P
-\frac12\{PQ+QP,\rho\},
\]

\[
A_{P,Q}(\rho)=i\left(P\rho Q-Q\rho P
+\frac12\{[P,Q],\rho\}\right).
\]

The full basis contains

| sector | number of coefficients | meaning |
|---|---:|---|
| H | 15 | Hamiltonian/coherent |
| S | 15 | diagonal stochastic Pauli |
| C | 105 | real symmetric correlations between Pauli axes |
| A | 105 | active/nonunital sector |
| total | 240 | complete real TP/HP two-qubit generator |

There are 25 decomposed objects, hence the complete coefficient table contains
\(25\times240=6000\) rows.  No coefficient is thresholded or omitted in the
combined CSV.

## 3. Overall noise-free channel

The following norms are Frobenius norms of the reconstructed PTM-generator
matrix contributed by each sector.  They should not be interpreted as
probabilities.

| nbar | infidelity | norm K | norm H | norm S | norm C | norm A | dominant coefficient | residual fraction |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0.01 | 2.4881e-4 | 7.7711e-2 | 7.7707e-2 | 5.9561e-4 | 4.7979e-4 | 1.4303e-7 | H:XX = 1.373679e-2 | 2.14e-14 |
| 1 | 1.3019e-3 | 1.6319e-1 | 1.6314e-1 | 3.8043e-3 | 1.3496e-3 | 1.5588e-7 | H:XX = 2.883915e-2 | 8.01e-15 |
| 2 | 3.0260e-3 | 2.4809e-1 | 2.4791e-1 | 9.3229e-3 | 2.1460e-3 | 4.1046e-7 | H:XX = 4.382438e-2 | 2.89e-15 |
| 3 | 5.3505e-3 | 3.3153e-1 | 3.3109e-1 | 1.6870e-2 | 2.8494e-3 | 3.1070e-7 | H:XX = 5.852880e-2 | 3.55e-15 |
| 4 | 8.2215e-3 | 4.1347e-1 | 4.1262e-1 | 2.6257e-2 | 3.4872e-3 | 2.5235e-7 | H:XX = 7.294131e-2 | 7.08e-15 |

The dominant coefficients within each sector are:

| nbar | H | S | C | A |
|---:|---|---|---|---|
| 0.01 | XX = 1.373679e-2 | IX = 5.997393e-5 | IX,XI = 5.997229e-5 | IX,XI = -8.004719e-9 |
| 1 | XX = 2.883915e-2 | XX = 4.600387e-4 | IX,XI = 1.687048e-4 | IX,XX = 1.425871e-8 |
| 2 | XX = 4.382438e-2 | XX = 1.335518e-3 | IX,XI = 2.682467e-4 | XI,XX = 4.481936e-8 |
| 3 | XX = 5.852880e-2 | XX = 2.583122e-3 | IX,XI = 3.561707e-4 | XX,XZ = -2.484008e-8 |
| 4 | XX = 7.294131e-2 | XX = 4.164479e-3 | IX,XI = 4.359038e-4 | XX,YX = -1.781616e-8 |

The stochastic S/C matrix has two physically resolved leading axes:

| nbar | eigenvalue 1 and axis | eigenvalue 2 and axis |
|---:|---|---|
| 0.01 | 1.199451e-4, (IX+XI)/sqrt(2) | 2.382887e-6, XX |
| 1 | 4.600387e-4, XX | 3.374104e-4, (IX+XI)/sqrt(2) |
| 2 | 1.335518e-3, XX | 5.364938e-4, (IX+XI)/sqrt(2) |
| 3 | 2.583122e-3, XX | 7.123427e-4, (IX+XI)/sqrt(2) |
| 4 | 4.164479e-3, XX | 8.718078e-4, (IX+XI)/sqrt(2) |

The collective-axis eigenvalue is twice the displayed common coefficient
\(S_{IX}\simeq S_{XI}\simeq C_{IX,XI}\), because the normalized eigenvector
is \((IX+XI)/\sqrt2\).

## 4. Isolated nominal-noise responses: overall information

### Motional heating/diffusion

| nbar | Delta infidelity | norm Delta K | norm H | norm S | norm C | norm A | leading stochastic eigenvalue/axis |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.01 | 3.8998e-4 | 3.0848e-3 | 4.3787e-5 | 2.3895e-3 | 1.9506e-3 | 1.9616e-7 | 4.876436e-4, (IX+XI)/sqrt(2) |
| 1 | 3.8217e-4 | 3.0258e-3 | 4.5049e-5 | 2.3439e-3 | 1.9129e-3 | 1.0786e-7 | 4.782279e-4, (IX+XI)/sqrt(2) |
| 2 | 3.7428e-4 | 2.9679e-3 | 4.7535e-5 | 2.2994e-3 | 1.8759e-3 | 2.2563e-7 | 4.689713e-4, (IX+XI)/sqrt(2) |
| 3 | 3.6634e-4 | 2.9113e-3 | 5.1563e-5 | 2.2558e-3 | 1.8397e-3 | 4.4416e-7 | 4.599351e-4, (IX+XI)/sqrt(2) |
| 4 | 3.5847e-4 | 2.8564e-3 | 5.6010e-5 | 2.2135e-3 | 1.8046e-3 | 3.0205e-7 | 4.511549e-4, (IX+XI)/sqrt(2) |

At every temperature the dominant coefficients are
\(S_{IX}\simeq S_{XI}\simeq C_{IX,XI}\).  The dominant H coefficient is
positive XX, from 7.7405e-6 to 9.9012e-6 rad/gate.  All displayed A
coefficients are at the 1e-8 level and their sector matrix norm is at most
4.45e-7.

### Motional dephasing

| nbar | Delta infidelity | norm Delta K | norm H | norm S | norm C | norm A | leading stochastic eigenvalue/axis | second stochastic mode |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.01 | 4.8309e-4 | 3.3149e-3 | 4.6471e-5 | 2.8001e-3 | 1.7736e-3 | 1.8444e-7 | 4.434060e-4, (IX+XI)/sqrt(2) | 1.613121e-4, XX |
| 1 | 1.1382e-3 | 8.4965e-3 | 2.3186e-4 | 6.7919e-3 | 5.0997e-3 | 1.3244e-7 | 1.274925e-3, (IX+XI)/sqrt(2) | 1.555251e-4, XX |
| 2 | 1.7615e-3 | 1.3547e-2 | 5.2670e-4 | 1.0686e-2 | 8.3102e-3 | 3.0113e-7 | 2.077554e-3, (IX+XI)/sqrt(2) | 1.486681e-4, XX |
| 3 | 2.3454e-3 | 1.8383e-2 | 9.0871e-4 | 1.4408e-2 | 1.1380e-2 | 2.2284e-7 | 2.845050e-3, (IX+XI)/sqrt(2) | 1.395449e-4, XX |
| 4 | 2.8895e-3 | 2.3006e-2 | 1.3602e-3 | 1.7959e-2 | 1.4315e-2 | 5.2404e-7 | 3.578703e-3, (IX+XI)/sqrt(2) | 1.277686e-4, XX |

The dominant H mode is negative XX and changes from -8.2151e-6 to
\(-2.4044\times10^{-4}\) rad/gate.  The dominant C mode is IX,XI at every temperature.  The
dominant S mode is IX or XI.  The collective-X stochastic eigenvalue increases
by a factor 8.07 while the separate XX stochastic eigenvalue decreases by
about 21%.

### Spin dephasing

| nbar | Delta infidelity | norm Delta K | norm H | norm S | norm C | norm A | largest stochastic eigenvalue |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 2.6651e-4 | 1.4446e-3 | 1.1697e-7 | 1.4312e-3 | 1.9674e-4 | 2.7600e-7 | 8.191851e-5 |
| 1 | 2.6617e-4 | 1.4332e-3 | 1.1516e-7 | 1.4190e-3 | 2.0128e-4 | 3.3678e-7 | 7.053163e-5 |
| 2 | 2.6558e-4 | 1.4324e-3 | 1.0488e-7 | 1.4169e-3 | 2.1036e-4 | 5.1308e-7 | 6.865134e-5 |
| 3 | 2.6480e-4 | 1.4330e-3 | 2.1127e-7 | 1.4160e-3 | 2.2003e-4 | 5.8247e-7 | 6.828731e-5 |
| 4 | 2.6382e-4 | 1.4339e-3 | 4.1389e-7 | 1.4154e-3 | 2.2952e-4 | 1.3477e-6 | 6.835755e-5 |

The resolved stochastic subspace is spanned mainly by
\(IZ,XY,ZI,YX\) and \(IY,XZ,YI,ZX\).  Because several eigenvalues are nearly
degenerate, the individual eigenvectors may rotate inside their degenerate
subspace; the subspace and its eigenvalues are more robust than a particular
displayed vector.  All 15 eigenvectors and all Pauli components are retained in
the eigenmode CSV.

### Photon scattering

| nbar | Delta infidelity | norm Delta K | norm H | norm S | norm C | norm A | largest stochastic eigenvalue |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 5.5958e-4 | 2.9969e-3 | 1.1024e-7 | 2.9726e-3 | 3.8089e-4 | 2.9079e-7 | 1.527215e-4 |
| 1 | 5.5884e-4 | 2.9826e-3 | 9.3173e-8 | 2.9567e-3 | 3.9255e-4 | 2.3217e-7 | 1.342195e-4 |
| 2 | 5.5762e-4 | 2.9823e-3 | 2.7183e-7 | 2.9539e-3 | 4.1059e-4 | 4.7085e-7 | 1.313897e-4 |
| 3 | 5.5600e-4 | 2.9839e-3 | 3.4172e-7 | 2.9529e-3 | 4.2912e-4 | 5.1724e-7 | 1.310174e-4 |
| 4 | 5.5398e-4 | 2.9860e-3 | 2.2450e-7 | 2.9523e-3 | 4.4740e-4 | 5.0781e-7 | 1.313642e-4 |

The stochastic subspace has the same principal Pauli blocks as spin
dephasing, with additional small IX and XI diagonal terms.  H and A remain
negligible on the scale of S.

## 5. Signed stochastic eigenvalues

The minimum eigenvalue of the S/C stochastic matrix is between about
-2e-12 and -1e-10 per gate for the noise-free channels and between about
-6e-9 and -4.3e-8 per gate for isolated-noise differences.  These values are
many orders of magnitude smaller than the resolved positive eigenvalues.  The
isolated \(\Delta K_i\) is not required to have a positive-semidefinite
Kossakowski matrix, so the signed values are retained rather than clipped.

## 6. Reconstruction completeness

- Noise-free maximum relative H/S/C/A reconstruction residual:
  \(2.14\times10^{-14}\).
- Isolated-noise maximum relative H/S/C/A reconstruction residual:
  \(1.46\times10^{-12}\).
- Largest A-sector norm divided by the full generator-response norm:
  \(9.40\times10^{-4}\), attained for spin dephasing at \(\bar n=4\).

Thus the previous diagonal-Pauli residual is resolved almost entirely into the
C sector, while the A sector is below 0.1% of every isolated response.

## 7. Complete machine-readable outputs

| file | rows | contents |
|---|---:|---|
| `full_taxonomy_coefficients.csv` | 6000 | every H/S/C/A coefficient for every context and temperature; no thresholding |
| `full_taxonomy_overall_summary.csv` | 25 | infidelity, total/sector matrix norms, reconstruction residual, stochastic eigenvalue extrema |
| `full_taxonomy_sector_statistics.csv` | 100 | coefficient L1/L2/max norms, dominant mode and matrix norm for each sector |
| `full_taxonomy_stochastic_eigenmodes.csv` | 375 | all 15 S/C eigenvalues and all 15 Pauli-axis components for every object |
| `noise_free_fock_generator_comparison.csv` | 5 | Fock-resolved prediction versus noise-free QPT H/S/C coefficients |
| `isolated_noise_generator_heatmaps.png` | -- | signed H and diagonal-S coefficients |
| `isolated_noise_taxonomy_sector_norms.png` | -- | H/S/C/A sector norms versus temperature |

The stochastic eigenvector sign is fixed only for readability by making its
largest-magnitude component positive.  Inside an exactly or nearly degenerate
eigenspace, individual eigenvectors are basis-dependent; the eigenspace is the
invariant information.
