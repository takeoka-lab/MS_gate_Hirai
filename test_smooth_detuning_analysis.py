import numpy as np
import pandas as pd

import smooth_detuning_analysis as sda


def test_default_sweep_uses_requested_nbar_values():
    config = sda.NbarSweepConfig()
    assert config.nbar_values == (0.01, 1.0, 2.0, 3.0, 4.0)


def test_smooth_waveform_boundaries_and_integrated_phase():
    config = sda.SmoothDetuningConfigSI()
    result = sda.validate_smooth_waveform(config, time_points=8001)
    pulse = sda.build_smooth_pulse_si(config, time_points=8001)
    phase = sda.integrated_phase(pulse["time_s"], pulse["delta_rad_s"])
    recovered = np.gradient(phase, pulse["time_s"])

    assert result["relative_phase_derivative_error"] < 5e-3
    assert np.max(
        np.abs(recovered[2:-2] - pulse["delta_rad_s"][2:-2])
    ) / np.max(np.abs(pulse["delta_rad_s"])) < 5e-3


def test_constant_detuning_limit_and_sign_validation():
    config = sda.SmoothDetuningConfigSI(
        delta_max_hz=40e3,
        delta_min_hz=40e3,
    )
    pulse = sda.build_smooth_pulse_si(config, time_points=1001)
    assert np.allclose(pulse["delta_rad_s"], 2 * np.pi * 40e3)

    opposite_sign = sda.SmoothDetuningConfigSI(
        delta_max_hz=40e3,
        delta_min_hz=-20e3,
    )
    try:
        sda.build_smooth_pulse_si(opposite_sign, time_points=1001)
    except ValueError:
        pass
    else:
        raise AssertionError("Opposite-sign detunings must be rejected.")


def test_calibration_and_analytic_preview_cover_all_temperatures():
    calibrated, calibration = sda.calibrate_delta_min()
    preview = sda.analytic_nbar_preview(
        sda.DEFAULT_NBAR_VALUES,
        smooth_config=calibrated,
    )

    assert calibration["F_avg_noise_free_analytic"] > 0.9999
    assert set(preview["nbar"]) == set(sda.DEFAULT_NBAR_VALUES)
    assert set(preview["pulse_type"]) == {"standard", "smooth_detuning"}
    assert len(preview) == 2 * len(sda.DEFAULT_NBAR_VALUES)


def test_plot_infidelity_comparison(tmp_path):
    preview = sda.analytic_nbar_preview()
    output = tmp_path / "infidelity_vs_nbar.png"
    sda.plot_infidelity_comparison(preview, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_residual_spin_motion_component_is_explicit():
    calibrated, _ = sda.calibrate_delta_min()
    residual = sda.residual_spin_motion_infidelity(
        nbar_values=(0.01, 1.0),
        smooth_config=calibrated,
    )
    assert len(residual) == 4
    assert set(residual["pulse_type"]) == {"standard", "smooth_detuning"}
    assert (residual["residual_spin_motion_infidelity"] >= 0).all()


def test_noise_attribution_sums_to_all_noise_advantage():
    values = {
        "none": {"standard": 1.0, "smooth_detuning": 0.8},
        "heating_only": {"standard": 1.5, "smooth_detuning": 1.1},
        "motional_dephasing_only": {"standard": 1.7, "smooth_detuning": 1.2},
        "all_motional": {"standard": 2.4, "smooth_detuning": 1.6},
        "all_noise": {"standard": 3.0, "smooth_detuning": 2.1},
    }
    rows = [
        {
            "nbar": 1.0,
            "noise_case": noise_case,
            "pulse_type": pulse_type,
            "infidelity": infidelity,
        }
        for noise_case, pulse_values in values.items()
        for pulse_type, infidelity in pulse_values.items()
    ]
    attribution = sda.build_noise_attribution(pd.DataFrame(rows))
    components = attribution[
        attribution.mechanism != "total_all_noise_advantage"
    ].smooth_advantage.sum()
    total = attribution[
        attribution.mechanism == "total_all_noise_advantage"
    ].smooth_advantage.iloc[0]
    assert np.isclose(components, total)


def test_photon_scattering_only_combines_existing_rayleigh_and_raman_rates():
    parameters = sda.noise_parameters("photon_scattering_only")

    assert parameters["rayleigh_rate_phys"] == sda.BASE_NOISE["rayleigh_rate_phys"]
    assert parameters["raman_rate_phys"] == sda.BASE_NOISE["raman_rate_phys"]
    assert parameters["heating_rate_phys"] == 0.0
    assert parameters["dephasing_rate_phys"] == 0.0
    assert parameters["T2_star"] == 1e99


def test_spin_optical_individual_errors_subtract_matching_qpt_floor():
    baseline = pd.DataFrame(
        [
            {
                "pulse_type": pulse_type,
                "noise_case": "none",
                "nbar": nbar,
                "infidelity": floor,
            }
            for nbar in (0.01, 1.0)
            for pulse_type, floor in (
                ("standard", 0.001 + 0.0001 * nbar),
                ("smooth_detuning", 0.0012 + 0.0001 * nbar),
            )
        ]
    )
    increments = {
        "spin_dephasing_only": {"standard": 0.002, "smooth_detuning": 0.003},
        "photon_scattering_only": {
            "standard": 0.0004,
            "smooth_detuning": 0.0008,
        },
    }
    rows = []
    for noise_case, pulse_increments in increments.items():
        for baseline_row in baseline.itertuples():
            rows.append(
                {
                    "pulse_type": baseline_row.pulse_type,
                    "noise_case": noise_case,
                    "nbar": baseline_row.nbar,
                    "infidelity": (
                        baseline_row.infidelity
                        + pulse_increments[baseline_row.pulse_type]
                    ),
                }
            )

    errors = sda.build_spin_optical_individual_errors(
        pd.DataFrame(rows),
        baseline,
    )

    assert len(errors) == 4
    spin = errors[errors.noise_case == "spin_dephasing_only"]
    scattering = errors[errors.noise_case == "photon_scattering_only"]
    assert np.allclose(spin.standard_individual_error, 0.002)
    assert np.allclose(spin.smooth_detuning_individual_error, 0.003)
    assert np.allclose(spin.smooth_error_reduction, -0.001)
    assert np.allclose(scattering.standard_individual_error, 0.0004)
    assert np.allclose(scattering.smooth_detuning_individual_error, 0.0008)


def test_plot_spin_optical_noise_only_infidelity(tmp_path):
    rows = [
        {
            "noise_case": noise_case,
            "pulse_type": pulse_type,
            "nbar": nbar,
            "infidelity": infidelity,
        }
        for noise_case in sda.SPIN_OPTICAL_INDIVIDUAL_CASES
        for pulse_type, infidelity in (
            ("standard", 5e-4),
            ("smooth_detuning", 1e-3),
        )
        for nbar in (0.01, 1.0)
    ]
    output = tmp_path / "spin_optical_noise_only_infidelity.png"

    sda.plot_spin_optical_noise_only_infidelity(pd.DataFrame(rows), output)

    assert output.exists()
    assert output.stat().st_size > 0
