# MS gate laser-pulse optimization

This workflow optimizes the optical sideband-coupling envelope `A(t)`, rather
than copying the detuning ramp of the laser-free smooth-gate experiment.

The search is performed in two stages:

1. Search a symmetric amplitude waveform with the fast first-order Magnus MS
   model, averaged over temperature, detuning offsets, and laser-intensity
   errors.
2. Run the selected waveform through the existing full spin-motion process
   tomography, and compare the exact and post-gate-error chi matrices before
   and after optimization.

## Notebook example

```python
import importlib
import numpy as np

import laser_pulse_optimization as lpo
import ms_gate_functions as mg

mg = importlib.reload(mg)
lpo = importlib.reload(lpo)


# Current constant-pulse gate: A=0.125, delta=0.5, t_gate=2*pi/delta.
config = lpo.LaserPulseSearchConfig(
    duration=2 * np.pi / 0.5,
    detuning=0.5,
    initial_amplitude=0.125,
    amplitude_max=0.30,
    control_points=9,
    time_points=501,
    n_bar_values=(0.01, 1.0, 4.0),

    # Replace these trial ensembles with measured laser/mode fluctuations.
    detuning_offsets=(-0.025, 0.0, 0.025),
    intensity_scales=(0.985, 1.0, 1.015),

    nominal_weight=200.0,
    worst_case_weight=0.5,
    smoothness_weight=1e-6,
    maxiter=80,
    popsize=10,
    seed=1234,
)

optimization = lpo.optimize_laser_amplitude_pulse(config)
display(optimization["summary"])
display(optimization["robustness"])


# Full QPT verification and chi-matrix comparison.
simulation_parameters = {
    "n_bar_list": [0.01, 1.0, 2.0, 3.0, 4.0],
    "t_gate_phys": 100e-6,
    "heating_rate_phys": 10.0,
    "dephasing_rate_phys": 18.0,
    "T2_star": 0.3,
    "rayleigh_rate_phys": 3.0,
    "raman_rate_phys": 1.0,
    "eta": 0.1,
    "use_full_order": True,
    "laser_intensity_fluctuation": 0.0,
    "laser_detuning_fluctuation": 0.0,
    "laser_rotation_angle_fluctuation": 0.0,
    "laser_noise_samples": 1,
    "laser_noise_seed": 1234,
    "show_progress": True,
    "parallel_workers": 8,

    # Optical Raman/Rayleigh scattering is proportional to laser intensity.
    # The supplied rates are interpreted at A=0.125.
    "laser_scattering_scales_with_intensity": True,
    "scattering_reference_amplitude": 0.125,
}

verification = lpo.verify_optimized_pulse_with_full_qpt(
    optimization,
    simulation_parameters,
    output_dir="laser_pulse_optimization_outputs",
    top_k=20,
)

display(verification["summary"])
display(verification["top_chi_changes"])
```

The output directory contains:

- `laser_pulse_waveforms.csv`: baseline and optimized `A(t)`.
- `analytic_fidelity_summary.csv`: fast-search metrics.
- `analytic_robustness_sweep.csv`: all temperature/noise scenarios.
- `full_qpt_fidelity_chi_summary.csv`: full-simulation fidelity and chi norms.
- `top_chi_matrix_changes.csv`: largest changed Pauli-basis chi elements.
- `laser_pulse_and_fidelity_comparison.png/pdf`: pulse and fidelity comparison.
- `chi_impact_nbar_*.png/pdf`: baseline, optimized, and delta chi heatmaps.

## Important interpretation

- Negative control amplitudes are not used by the default optimizer. If optical
  phase flips are available, phase modulation should be introduced explicitly
  rather than interpreting negative laser power literally.
- `zero_endpoints=False` keeps the existing square pulse inside the feasible
  search set. Set it to `True` only when the simulated gate window includes the
  complete optical turn-on/off ramps.
- The search model is exact only for the first-order, single-mode MS Hamiltonian.
  Claims about the final fidelity must use the full-QPT verification results.
- The default detuning offsets and intensity scales above are placeholders. For
  physically meaningful optimization, replace them with measured distributions
  or a measured noise power spectral density.
