"""Configuration API for ``chi_error_element_nbar_fit.ipynb``.

The notebook is intentionally a thin front end.  All numerical work, helper
functions, plotting, caching and publication checks live in
``chi_error_nbar_analysis_impl.py`` and are executed with the configuration
dictionary supplied here.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, Mapping

import numpy as np


IMPLEMENTATION_PATH = Path(__file__).with_name("chi_error_nbar_analysis_impl.py")


def default_simulation_params() -> dict[str, Any]:
    """Return an independent copy of the baseline full-order MS parameters."""

    return {
        "A": 0.125,
        "delta": 0.5,
        "rho0": 0.0,
        "time_points": 500,
        "t_gate_phys": 100e-6,
        "heating_rate_phys": 10.0,
        "dephasing_rate_phys": 18.0,
        "T2_star": 0.3,
        "rayleigh_rate_phys": 3.0,
        "raman_rate_phys": 1.0,
        "eta": 0.1,
        "laser_intensity_fluctuation": 0.0,
        "laser_detuning_fluctuation": 0.0,
        "laser_rotation_angle_fluctuation": 0.0,
        "laser_noise_samples": 1,
        "laser_noise_seed": 1234,
        "use_full_order": True,
        "show_progress": True,
        "parallel_workers": 4,
    }


def default_config() -> dict[str, Any]:
    """Return notebook defaults with every expensive recomputation disabled."""

    return {
        "OUTPUT_DIR": "results/chi_error_element_fit",
        "NBAR_GRID": np.r_[0.01, np.arange(0.25, 20.0001, 0.25)],
        "RECOMPUTE": False,
        "PARALLEL_WORKERS": 4,
        "FAST_PROCESS_WORKERS": 4,
        "FIT_DEGREE": 4,
        "ERROR_CHANNEL_CONVENTION": "undo_before_actual",
        "SIMULATION_PARAMS": default_simulation_params(),
        "RUN_EXACT_FULL_CHI_SWEEP": False,
        "RUN_NUMERICAL_CONVERGENCE": False,
        "RUN_NOISE_SOURCE_ABLATION": False,
        "FORCE_RECOMPUTE_PUBLICATION_CACHE": False,
        "CONVERGENCE_NBARS": [0.01, 4.0, 20.0],
        "ABLATION_NBARS": [0.01, 1.0, 4.0, 10.0, 20.0],
        "FULL_CHI_TOP_K": 12,
        "BOOTSTRAP_SAMPLES": 1000,
        "BOOTSTRAP_SEED": 20260805,
        "RUN_PHYSICAL_CONTROL_QPT": False,
        "RUN_PARAMETER_ROBUSTNESS_QPT": False,
        "FORCE_RECOMPUTE_ADVANCED_QPT": False,
        "CONTROL_VALIDATION_NBARS": [0.01, 1.0, 2.0, 3.0, 4.0],
        "ROBUSTNESS_NBARS": [0.01, 1.0, 2.0, 3.0, 4.0],
        "CPTP_TOLERANCE": 1e-11,
        "CPTP_MAX_ITERATIONS": 5000,
        "RUN_HXX_DRIVE_CALIBRATION_QPT": False,
        "FORCE_RECOMPUTE_HXX_DRIVE_QPT": False,
        "HXX_DRIVE_CALIBRATION_NBARS": [
            0.01,
            1.0,
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            12.0,
            16.0,
            20.0,
        ],
        "HXX_MAX_FEEDBACK_ITERATIONS": 2,
        "HXX_CONVERGENCE_TOL_RAD": 2e-3,
        "HXX_MAX_AMPLITUDE_FACTOR": 1.6,
        "TARGET_XX_ANGLE_RAD": np.pi / 4.0,
        "KIRCHHOFF_REFERENCE_K": None,
        "KIRCHHOFF_LOOP_NUMBER": 1.0,
        "KIRCHHOFF_SCAN_K_MAX": 250.0,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject common configuration errors before the long workflow starts."""

    required = {"OUTPUT_DIR", "NBAR_GRID", "SIMULATION_PARAMS"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing workflow configuration keys: {missing}")

    nbar_grid = np.asarray(config["NBAR_GRID"], dtype=float)
    if nbar_grid.ndim != 1 or len(nbar_grid) == 0:
        raise ValueError("NBAR_GRID must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(nbar_grid)) or np.any(nbar_grid < 0.0):
        raise ValueError("NBAR_GRID must contain finite non-negative values")

    simulation_params = dict(config["SIMULATION_PARAMS"])
    for key in ("A", "delta", "eta", "time_points", "t_gate_phys"):
        if key not in simulation_params:
            raise ValueError(f"SIMULATION_PARAMS is missing {key!r}")
    if int(config.get("FIT_DEGREE", 4)) < 1:
        raise ValueError("FIT_DEGREE must be positive")
    for key in ("PARALLEL_WORKERS", "FAST_PROCESS_WORKERS"):
        if int(config.get(key, 1)) < 1:
            raise ValueError(f"{key} must be positive")


def run(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute the complete analysis and return its result namespace.

    Passing ``None`` uses the fast, cache-reading defaults.  The returned
    dictionary exposes the data frames created by the implementation, such as
    ``fit_summary_df``, ``generator_df`` and ``final_drive_calibration_df``.
    """

    resolved = default_config()
    if config is not None:
        supplied = dict(config)
        supplied_simulation = supplied.pop("SIMULATION_PARAMS", None)
        resolved.update(supplied)
        if supplied_simulation is not None:
            merged_simulation = default_simulation_params()
            merged_simulation.update(dict(supplied_simulation))
            resolved["SIMULATION_PARAMS"] = merged_simulation
    validate_config(resolved)
    return runpy.run_path(
        str(IMPLEMENTATION_PATH),
        init_globals={"WORKFLOW_CONFIG": resolved},
    )
