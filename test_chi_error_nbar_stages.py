import pytest
import json

import numpy as np
import pandas as pd

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


def test_drive_completion_refreshes_checklist_and_manifest(tmp_path):
    advanced = tmp_path / "advanced_publication_validation"
    advanced.mkdir()
    checklist_path = advanced / "advanced_publication_checklist.csv"
    pd.DataFrame([{
        "check": "hXX-derived drive calibration re-QPT",
        "status": "pending",
        "result": "8/10 temperatures re-QPT",
    }]).to_csv(checklist_path, index=False)
    manifest_path = advanced / "advanced_publication_manifest.json"
    manifest_path.write_text(
        json.dumps({"qpt_completion": {}, "checklist": []}),
        encoding="utf-8",
    )
    nbar_values = [0.01, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
    summary = pd.DataFrame({
        "n_bar": nbar_values,
        "h_XX_converged": [True] * 10,
    })
    config = workflow.default_config()

    result = stages._refresh_drive_completion_artifacts(
        {"advanced": advanced}, config, summary
    )

    refreshed = pd.read_csv(checklist_path).iloc[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["completed_count"] == 10
    assert refreshed["status"] == "complete"
    assert refreshed["result"].startswith("10/10")
    assert manifest["qpt_completion"][
        "hxx_drive_calibration_temperatures"
    ] == 10


def test_drive_cache_resolver_tolerates_csv_float_round_trip(tmp_path):
    config = workflow.default_config()
    paths = stages._paths({**config, "OUTPUT_DIR": tmp_path})
    n_bar = 4.0
    iteration = 1
    stored_amplitude = 0.1312216323146464
    requested_amplitude = np.nextafter(stored_amplitude, np.inf)
    candidate = paths["drive_qpt"] / (
        "hxx_feedback_i01__nbar_4__historicalhash.npz"
    )
    np.savez_compressed(
        candidate,
        n_bar=np.asarray(n_bar),
        metadata_json=np.asarray(json.dumps({
            "iteration": iteration,
            "A_calibrated": stored_amplitude,
        })),
    )

    resolved = stages._resolve_drive_cache_path(
        paths, config, n_bar, iteration, requested_amplitude
    )

    assert resolved == candidate
