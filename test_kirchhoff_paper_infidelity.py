import numpy as np
import pandas as pd
import qutip as qp

import chi_error_nbar_stages as stages
import coherent_limit_comparison as coherent_bridge
import drive_calibration_qpt_analysis as qpt_analysis
import kirchhoff_paper_infidelity as paper
import ms_gate_functions as mg
import model_specific_hxx_calibration as hxx_calibration


def test_paper_drive_landmarks_match_reference_parameters():
    landmarks = paper.drive_landmarks(
        K=28.0,
        L=25.0,
        gate_duration_seconds=28e-6,
        eta=0.18,
    )

    assert np.isclose(landmarks["omega_ld"] / 1e6, 1.0503254405)
    assert np.isclose(landmarks["omega_2"] / 1e6, 1.0688994745)
    assert np.isclose(landmarks["omega_4"] / 1e6, 1.0891924704)


def test_sideband_hamiltonian_is_hermitian_and_uses_paper_step_count():
    model = paper.build_paper_model(
        K=28.0,
        L=25.0,
        eta=0.18,
        phonon_dim=8,
        sideband_cutoff=3,
    )
    hamiltonian = paper.dimensionless_hamiltonian(
        model,
        0.371,
        omega_times_gate=1.0,
    )

    assert np.allclose(hamiltonian, hamiltonian.conj().T, atol=2e-12)
    assert paper.automatic_trotter_steps(model) == 6849


def test_thermal_weights_report_the_omitted_tail():
    weights, tail = paper.thermal_weights(8, 1.0)

    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(tail, 1.0 / 256.0)


def test_thermal_channel_and_yy_to_xx_mapping_remove_the_ideal_target():
    model = paper.build_paper_model(
        K=28.0,
        L=25.0,
        eta=0.18,
        phonon_dim=3,
        sideband_cutoff=1,
    )
    ideal_full_propagator = np.kron(
        np.eye(model.phonon_dim), model.target_qubit_unitary
    )
    _, error_channel, metadata = paper.kirchhoff_xx_error_channel(
        ideal_full_propagator,
        model,
        n_bar=0.7,
        convention="undo_before_actual",
    )
    chi_raw = qp.to_chi(error_channel).full()
    observables = qpt_analysis.extract_pauli_generator_observables(
        chi_raw / np.trace(chi_raw)
    )

    assert metadata["kraus_count"] == model.phonon_dim**2
    assert metadata["kraus_completeness_frobenius_error"] < 1e-12
    assert metadata["yy_to_xx_target_superoperator_frobenius_error"] < 1e-12
    assert abs(observables["average_infidelity"]) < 1e-12
    assert abs(
        observables["hamiltonian_coefficients_rad_per_gate"]["XX"]
    ) < 1e-12


def test_coherent_bridge_mapping_and_own_thermal_channel():
    amplitude = coherent_bridge.carrier_to_effective_sideband_amplitude(
        1.0e6,
        eta=0.18,
        gate_duration_seconds=28e-6,
    )
    assert np.isclose(amplitude, 2.52)
    assert np.isclose(
        coherent_bridge.effective_detuning(28.0, 25.0),
        6.0 * np.pi,
    )

    phonon_dim = 3
    ideal_full = np.kron(mg.ideal_ms_gate().full(), np.eye(phonon_dim))
    _, error, metadata = coherent_bridge.own_xx_error_channel(
        ideal_full,
        phonon_dim=phonon_dim,
        n_bar=0.4,
    )
    observables = qpt_analysis.extract_pauli_generator_from_superoperator(error)
    assert metadata["kraus_completeness_frobenius_error"] < 1e-12
    assert abs(observables["average_infidelity"]) < 1e-12
    assert abs(
        observables["hamiltonian_coefficients_rad_per_gate"]["XX"]
    ) < 1e-12
    assert abs(
        observables["pauli_dissipator_rates_per_gate"]["XX"]
    ) < 1e-12


def test_thermal_qpt_stage_consumes_a_cached_propagator_without_evolution(
    tmp_path,
):
    model = paper.build_paper_model(
        K=28.0,
        L=25.0,
        eta=0.18,
        phonon_dim=2,
        sideband_cutoff=1,
    )
    source_path = tmp_path / "paper_propagator.npz"
    np.savez_compressed(
        source_path,
        propagator=np.kron(
            np.eye(model.phonon_dim), model.target_qubit_unitary
        ),
        metadata_json=np.asarray("{}"),
    )
    paper_result = {
        "propagator_metadata": pd.DataFrame([{
            "omega_per_second": 1.0e6,
            "omega_mhz": 1.0,
            "landmark": "Omega_4",
            "cache_path": str(source_path),
        }]),
        "summary": pd.DataFrame({"n_bar": [0.2]}),
        "status": {
            "K": 28.0,
            "L": 25.0,
            "eta": 0.18,
            "phonon_dim": 2,
            "sideband_cutoff": 1,
        },
    }

    result = stages.run_kirchhoff_thermal_qubit_qpt_stage(
        {"OUTPUT_DIR": tmp_path / "results"},
        paper_result,
        nbar_values=[0.2],
        show_progress=False,
    )

    assert result["status"]["completed_channel_points"] == 1
    assert result["status"]["pending_channel_points"] == 0
    assert result["status"]["raw_cp_pass"] == 1
    assert abs(result["summary"].iloc[0]["average_infidelity"]) < 1e-12
    assert result["summary_path"].exists()
    assert result["coefficient_path"].exists()
    assert result["chi_component_path"].exists()


def test_stage_can_prepare_a_resumable_paper_sweep_without_propagating(tmp_path):
    result = stages.run_kirchhoff_paper_infidelity_stage(
        {"OUTPUT_DIR": tmp_path},
        omega_mhz_values=[1.07],
        nbar_values=[0.02, 0.1],
        run_simulation=False,
        show_progress=False,
    )

    # The requested point plus the three exact analytic landmarks.
    assert result["status"]["expected_drive_points"] == 4
    assert result["status"]["completed_drive_points"] == 0
    assert result["status"]["pending_drive_points"] == 4
    assert result["summary"].empty
    assert result["figure_path"] is None


def test_coherent_limit_stage_runs_one_direct_comparison(tmp_path):
    model = paper.build_paper_model(
        K=28.0,
        L=25.0,
        eta=0.18,
        phonon_dim=2,
        sideband_cutoff=1,
    )
    source_path = tmp_path / "paper_omega4.npz"
    np.savez_compressed(
        source_path,
        propagator=np.kron(
            np.eye(model.phonon_dim), model.target_qubit_unitary
        ),
        metadata_json=np.asarray("{}"),
    )
    paper_result = {
        "propagator_metadata": pd.DataFrame([{
            "omega_per_second": 1.0e6,
            "omega_mhz": 1.0,
            "landmark": "Omega_4",
            "cache_path": str(source_path),
        }]),
        "status": {
            "K": 28.0,
            "L": 25.0,
            "eta": 0.18,
            "phonon_dim": 2,
            "sideband_cutoff": 1,
            "gate_duration_seconds": 28e-6,
        },
    }

    result = stages.run_kirchhoff_coherent_limit_comparison_stage(
        {"OUTPUT_DIR": tmp_path / "results"},
        paper_result,
        nbar_values=[0.02],
        landmarks=["Omega_4"],
        model_variants={"own_lamb_dicke": False},
        time_points=51,
        run_own_propagation=True,
        show_progress=False,
    )

    assert result["status"]["completed_own_propagators"] == 1
    assert result["status"]["completed_comparison_points"] == 1
    assert result["status"]["pending_own_propagators"] == 0
    assert result["status"]["chi_role"].startswith("diagnostic")
    assert result["comparison_path"].exists()
    assert result["coefficient_difference_path"].exists()


def test_safeguarded_hxx_root_finds_a_linear_zero():
    result = hxx_calibration.find_hxx_zero(
        lambda omega: {
            "omega_per_second": float(omega),
            "h_XX_rad_per_gate": (float(omega) - 2.0) / 10.0,
        },
        [1.0, 3.0],
        lower_bound=0.5,
        upper_bound=3.5,
        tolerance_rad=1e-12,
        max_iterations=2,
    )

    assert result["converged"]
    assert np.isclose(result["best"]["omega_per_second"], 2.0)


def test_model_specific_calibration_can_reuse_exact_cached_roots(tmp_path):
    phonon_dim = 2
    paper_model = paper.build_paper_model(
        K=28.0,
        L=25.0,
        eta=0.18,
        phonon_dim=phonon_dim,
        sideband_cutoff=1,
    )
    omega = 1.09e6
    paper_path = tmp_path / "paper_exact.npz"
    np.savez_compressed(
        paper_path,
        propagator=np.kron(
            np.eye(phonon_dim), paper_model.target_qubit_unitary
        ),
        metadata_json=np.asarray("{}"),
    )
    own_path = tmp_path / "own_exact.npz"
    np.savez_compressed(
        own_path,
        propagator=np.kron(mg.ideal_ms_gate().full(), np.eye(phonon_dim)),
        metadata_json=np.asarray("{}"),
    )
    paper_result = {
        "propagator_metadata": pd.DataFrame([{
            "omega_per_second": omega,
            "omega_mhz": omega / 1e6,
            "landmark": "Omega_4",
            "cache_path": str(paper_path),
        }]),
        "status": {
            "K": 28.0,
            "L": 25.0,
            "eta": 0.18,
            "phonon_dim": phonon_dim,
            "sideband_cutoff": 1,
            "gate_duration_seconds": 28e-6,
        },
    }
    coherent_rows = []
    for model_name in ("Kirchhoff_full_carrier", "own_lamb_dicke", "own_eta2_corrected"):
        coherent_rows.append({
            "model": model_name,
            "landmark": "Omega_4",
            "omega_per_second": omega,
            "n_bar": 0.02,
            "source_cache": str(
                paper_path if model_name == "Kirchhoff_full_carrier" else own_path
            ),
            "h_XX_rad_per_gate": 0.0,
            "gamma_XX_per_gate": 0.0,
            "average_infidelity": 0.0,
        })
    coherent_result = {
        "model_summary": pd.DataFrame(coherent_rows),
        "status": {
            "model_variants": {
                "own_lamb_dicke": False,
                "own_eta2_corrected": True,
            }
        },
    }

    result = stages.run_model_specific_hxx_zero_calibration_stage(
        {"OUTPUT_DIR": tmp_path / "results"},
        paper_result,
        coherent_result,
        nbar_values=[0.02],
        omega_bounds_mhz=(1.04, 1.18),
        run_calibration_propagation=False,
        show_progress=False,
    )

    assert result["status"]["converged_roots"] == 3
    assert result["status"]["pending_roots"] == 0
    assert result["status"]["newly_computed_propagators"] == 0
    assert len(result["comparison"]) == 2
