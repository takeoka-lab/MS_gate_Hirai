import pytest

import chi_error_nbar_stages as stages
import chi_error_nbar_workflow as workflow


def test_physical_control_stage_defines_eleven_nonbaseline_candidates():
    params = workflow.default_simulation_params()
    candidates = stages._control_candidates(
        params,
        amplitude_factors=[0.95, 1.00, 1.05, 1.10, 1.15, 1.20],
        gate_time_factors=[0.97, 1.03],
        detuning_factors=[0.97, 1.03],
        pulse_shapes=["sin2", "blackman"],
    )
    assert len(candidates) == 11
    assert {candidate["kind"] for candidate in candidates} == {
        "amplitude",
        "gate_time",
        "detuning",
        "pulse",
    }


def test_robustness_stage_defines_nine_nonbaseline_conditions():
    conditions = stages._robustness_conditions(
        workflow.default_simulation_params(),
        eta_factors=[0.8, 1.2],
        a_over_delta_factors=[0.9, 1.1],
        gate_time_factors=[0.95, 1.05],
        motional_dephasing_factors=[0.0, 0.5, 2.0],
    )
    assert len(conditions) == 9
    assert {condition["parameter"] for condition in conditions} == {
        "eta",
        "A_over_delta",
        "gate_time",
        "motional_dephasing_rate",
    }


def test_kirchhoff_direct_comparison_requires_a_physical_reference():
    with pytest.raises(ValueError, match="mode_frequency_hz or reference_k"):
        stages.run_kirchhoff_direct_comparison_stage(workflow.default_config())


def test_default_control_and_robustness_nbar_ranges_are_low_temperature():
    config = workflow.default_config()
    expected = [0.01, 1.0, 2.0, 3.0, 4.0]
    assert config["CONTROL_VALIDATION_NBARS"] == expected
    assert config["ROBUSTNESS_NBARS"] == expected
