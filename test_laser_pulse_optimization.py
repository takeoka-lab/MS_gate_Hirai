import unittest

import numpy as np

import laser_pulse_optimization as lpo
import ms_gate_functions as mg


class TimeDependentControlTests(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(0.0, 2.0, 101)
        self.operators = mg._ms_gate_static_operators(6, 0.1, False)

    def test_scalar_controls_match_constant_waveforms(self):
        scalar_hamiltonian = mg._build_ms_hamiltonian(
            self.operators,
            self.time,
            detuning=0.5,
            rho=0.2,
            effective_amplitude=0.125,
        )
        waveform_hamiltonian = mg._build_ms_hamiltonian(
            self.operators,
            self.time,
            detuning=np.full_like(self.time, 0.5),
            rho=np.full_like(self.time, 0.2),
            effective_amplitude=np.full_like(self.time, 0.125),
        )

        for sample_time in self.time[::10]:
            difference = scalar_hamiltonian(sample_time) - waveform_hamiltonian(
                sample_time
            )
            self.assertLess(difference.norm(), 1e-12)

    def test_time_dependent_detuning_uses_integrated_phase(self):
        detuning = 0.3 + 0.2 * self.time
        phase = mg._integrated_control_phase(detuning, self.time)
        expected_phase = 0.3 * self.time + 0.1 * self.time**2
        np.testing.assert_allclose(phase, expected_phase, atol=1e-12, rtol=0.0)

    def test_intensity_scaled_scattering_turns_off_with_laser(self):
        intensity_scale = np.linspace(0.0, 1.0, len(self.time))
        collapse_operators = mg._build_c_ops(
            self.operators,
            rayleigh_scattering_rate=0.1,
            raman_scattering_rate=0.2,
            time_grid=self.time,
            scattering_intensity_scale=intensity_scale,
        )
        self.assertEqual(len(collapse_operators), 6)
        for collapse_operator in collapse_operators:
            self.assertLess(collapse_operator(self.time[0]).norm(), 1e-12)
            self.assertGreater(collapse_operator(self.time[-1]).norm(), 0.0)


class AnalyticLaserPulseTests(unittest.TestCase):
    def test_constant_ms_gate_has_target_phase_and_unit_fidelity(self):
        detuning = 0.5
        amplitude = 0.125
        duration = 2 * np.pi / detuning
        time = np.linspace(0.0, duration, 2001)
        waveform = np.full_like(time, amplitude)

        metrics = lpo.ms_magnus_metrics(time, waveform, detuning)
        self.assertLess(metrics["displacement_abs"], 1e-10)
        self.assertAlmostEqual(metrics["geometric_phase"], np.pi / 8, places=5)
        self.assertAlmostEqual(metrics["xx_angle"], np.pi / 4, places=5)

        fidelity = lpo.analytic_ms_average_gate_fidelity(
            time,
            waveform,
            detuning,
            n_bar=10.0,
        )
        self.assertGreater(fidelity, 1.0 - 1e-10)

    def test_symmetric_control_points_are_mirrored(self):
        time = np.linspace(0.0, 1.0, 101)
        amplitude, _, nodes = lpo.build_symmetric_laser_amplitude(
            [0.10, 0.15, 0.20],
            time,
            control_points=5,
            zero_endpoints=False,
        )
        np.testing.assert_allclose(nodes, nodes[::-1])
        np.testing.assert_allclose(amplitude, amplitude[::-1])

        ramped_amplitude, _, ramped_nodes = lpo.build_symmetric_laser_amplitude(
            [0.15, 0.20],
            time,
            control_points=5,
            zero_endpoints=True,
        )
        self.assertEqual(ramped_nodes[0], 0.0)
        self.assertEqual(ramped_nodes[-1], 0.0)
        np.testing.assert_allclose(ramped_amplitude, ramped_amplitude[::-1])

    def test_detuning_error_reduces_fidelity(self):
        detuning = 0.5
        duration = 2 * np.pi / detuning
        time = np.linspace(0.0, duration, 1001)
        amplitude = np.full_like(time, 0.125)
        nominal = lpo.analytic_ms_average_gate_fidelity(
            time,
            amplitude,
            detuning,
            n_bar=4.0,
        )
        offset = lpo.analytic_ms_average_gate_fidelity(
            time,
            amplitude,
            detuning + 0.025,
            n_bar=4.0,
        )
        self.assertGreater(nominal, offset)


if __name__ == "__main__":
    unittest.main()
