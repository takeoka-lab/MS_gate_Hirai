import numpy as np

import drive_calibration_qpt_analysis as analysis


def test_identity_chi_has_zero_xx_generator_and_infidelity():
    chi = np.zeros((16, 16), dtype=complex)
    chi[0, 0] = 1.0

    result = analysis.extract_xx_generator_observables(chi)

    assert abs(result["h_XX_rad_per_gate"]) < 1e-12
    assert abs(result["gamma_XX_per_gate"]) < 1e-12
    assert abs(result["average_infidelity"]) < 1e-12
    assert result["cptp_projection_iterations"] >= 1
