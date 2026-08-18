import numpy as np
import qutip as qp

import drive_calibration_qpt_analysis as analysis
import ms_gate_functions as mg


def test_identity_chi_has_zero_xx_generator_and_infidelity():
    chi = np.zeros((16, 16), dtype=complex)
    chi[0, 0] = 1.0

    result = analysis.extract_xx_generator_observables(chi)

    assert abs(result["h_XX_rad_per_gate"]) < 1e-12
    assert abs(result["gamma_XX_per_gate"]) < 1e-12
    assert abs(result["average_infidelity"]) < 1e-12
    assert result["cptp_projection_iterations"] >= 1


def test_direct_superoperator_generator_matches_cptp_chi_route():
    actual = qp.to_super(mg.ideal_ms_gate(phi=np.pi / 4.0 - 0.03))
    error = actual * qp.to_super(mg.ideal_ms_gate().dag())
    direct = analysis.extract_pauli_generator_from_superoperator(error)
    chi_raw = qp.to_chi(error).full()
    projected = analysis.extract_pauli_generator_observables(
        chi_raw / np.trace(chi_raw)
    )

    assert np.isclose(
        direct["hamiltonian_coefficients_rad_per_gate"]["XX"],
        projected["hamiltonian_coefficients_rad_per_gate"]["XX"],
        atol=1e-12,
    )
    assert np.isclose(
        direct["pauli_dissipator_rates_per_gate"]["XX"],
        projected["pauli_dissipator_rates_per_gate"]["XX"],
        atol=1e-12,
    )
    assert np.isclose(
        direct["average_infidelity"],
        projected["average_infidelity"],
        atol=1e-12,
    )
