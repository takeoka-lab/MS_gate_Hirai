# Smooth-detuning comparison report

## Conventions
- Pulse generation uses SI units and angular frequencies in rad/s.
- The paper detuning sign is flipped to match the repository standard-gate and +iXX convention.
- The paper Omega_g is mapped to the simulator coefficient A=Omega_g/2.

## Calibration
```json
{
  "delta_min_hz": 22245.459873243974,
  "F_avg_noise_free_analytic": 0.9999999673844867,
  "xx_angle_rad": 0.7854201386086438,
  "entangling_angle_error_rad": 2.1975211195512934e-05,
  "alpha_final_abs": 0.0001405285612222253,
  "gate_time_s": 0.0002258,
  "integrated_omega2": 312029.51724793983
}
```

## Model limitations
- The current `use_full_order` model is a finite Lamb-Dicke correction, not an exact all-order optical interaction.
- Laser phase noise, beam-pointing noise, differential AC Stark shifts, multi-mode spectator coupling, and a measured laser-noise PSD are not added by assumption.
- Single-noise infidelities need not add to the all-noise infidelity.
- Numerical conclusions require the completed QPT sweep and cutoff/time-step convergence checks.

## Numerical results
Completed rows: 4.
- At nbar=0.01, all-noise F_avg=0.99792800, G_I=0.9394.
- At the largest completed nbar=0.01, G_I=0.9394.
- Weight-2 Pauli probability at low nbar is 0.0011884.
- Chi off-diagonal ratio is 0.706997; exact/Pauli distance is 0.0122179.
- Noise-free QPT at nbar=0.01: F_avg=0.99993867, G_I=4.056.

Quick mode is a smoke comparison; use full mode and convergence checks for physics claims.