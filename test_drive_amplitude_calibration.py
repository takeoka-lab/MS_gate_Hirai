import math

import numpy as np
import pytest

import drive_amplitude_calibration as dac


def test_hxx_amplitude_update_has_expected_sign_and_quadratic_angle():
    amplitude = 0.125
    target = math.pi / 4.0
    h_xx = 0.1

    corrected = dac.amplitude_update_from_hxx(amplitude, h_xx, target)

    assert corrected > amplitude
    inferred_before = target - h_xx
    inferred_after = inferred_before * (corrected / amplitude) ** 2
    assert np.isclose(inferred_after, target)


def test_hxx_amplitude_update_rejects_wrong_logarithm_branch():
    with pytest.raises(ValueError, match="non-positive"):
        dac.amplitude_update_from_hxx(0.125, math.pi / 4.0)


def test_kirchhoff_fourth_order_formula_solves_equation_40():
    K = 20.0
    L = 19.0
    eta = 0.1
    duration = 3.2
    omega = dac.kirchhoff_omega_fourth_order(K, L, duration, eta)
    x_value = omega**2 * duration**2
    left = (
        -K
        * x_value
        * eta**2
        * (4.0 * math.pi**2 * L**2 * eta**2 - 4.0 * math.pi**2 * L**2 + x_value)
        / (4.0 * math.pi**3 * L**2 * (K**2 - L**2))
    )
    assert np.isclose(left, math.pi / 2.0, rtol=1e-10, atol=1e-12)


def test_kirchhoff_ratios_are_duration_independent_and_near_unity():
    ratios = dac.kirchhoff_renormalization_ratios(K=100.0, L=99.0, eta=0.1)

    assert 0.9 < ratios["omega_2_over_omega_ld"] < 1.2
    assert 0.9 < ratios["omega_4_over_omega_ld"] < 1.2


def test_kirchhoff_fourth_order_domain_is_checked():
    with pytest.raises(ValueError, match="real-valued domain"):
        dac.kirchhoff_omega_fourth_order(K=2.0, L=1.0, gate_duration=1.0, eta=0.01)
