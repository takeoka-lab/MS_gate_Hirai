import numpy as np
import pandas as pd

import chi_error_nbar_workflow as workflow
import phonon_xx_angle_analysis as analysis


def test_required_fock_cutoff_respects_thermal_tail_tolerance():
    n_bar = 20.0
    tolerance = 1e-5
    max_fock_n = analysis.required_fock_cutoff(n_bar, tolerance)
    ratio = n_bar / (1.0 + n_bar)
    assert ratio ** (max_fock_n + 1) <= tolerance
    assert ratio ** max_fock_n > tolerance


def test_constant_fock_angle_has_no_thermal_phase_dispersion():
    theta = np.pi / 4.0 - 0.02
    coherence = np.exp(2j * theta)
    curve = pd.DataFrame({
        "fock_n": np.arange(30),
        "coherence_real": np.full(30, coherence.real),
        "coherence_imag": np.full(30, coherence.imag),
        "coherence_abs": np.ones(30),
        "theta_xx_rad": np.full(30, theta),
        "theta_error_rad": np.full(30, 0.02),
    })
    summary = analysis.summarize_thermal_xx_angle(curve, 4.0)
    assert np.isclose(summary["predicted_h_XX_rad_per_gate"], 0.02)
    assert summary["gamma_XX_phase_dispersion_per_gate"] < 1e-14
    assert summary["gamma_XX_with_residual_motion_per_gate"] < 1e-14


def test_small_full_order_curve_returns_one_angle_per_fock_state():
    params = workflow.default_simulation_params()
    params["time_points"] = 100
    curve = analysis.calculate_fock_resolved_xx_angles(
        params,
        max_fock_n=2,
        phonon_buffer=8,
    )
    assert curve["fock_n"].tolist() == [0, 1, 2]
    assert np.all(np.isfinite(curve["theta_xx_rad"]))
    assert np.all((curve["coherence_abs"] >= 0.0))
    assert np.all((curve["coherence_abs"] <= 1.0 + 1e-10))
    assert curve["theta_xx_rad"].is_monotonic_decreasing
