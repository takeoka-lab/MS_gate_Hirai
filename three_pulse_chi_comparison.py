"""Controlled post-gate chi comparison for three MS-gate pulse choices.

The comparison keeps the repository Hamiltonian, noise rates, Lamb--Dicke
parameter, QPT convention, and thermal state fixed.  Kirchhoff et al.'s
fourth-order result is transferred only as the dimensionless amplitude ratio
Omega_4/Omega_LD.  It is therefore an evaluation of their calibration rule in
the repository model, not a reproduction of their full carrier-sideband model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drive_amplitude_calibration as drive_formula
import drive_calibration_qpt_analysis as qpt_analysis
import smooth_detuning_analysis as smooth_analysis


CONDITION_ORDER = (
    "standard_rectangular",
    "smooth_detuning",
    "kirchhoff_eq41",
)
CONDITION_LABELS = {
    "standard_rectangular": "Standard rectangular",
    "smooth_detuning": "Smooth detuning",
    "kirchhoff_eq41": "Kirchhoff Eq. (41)",
}


@dataclass(frozen=True)
class ThreePulseComparisonConfig:
    """User-editable physical and numerical settings for one heatmap set."""

    n_bar: float = 4.0
    workers: int = 4
    execute_qpt: bool = True
    force_recompute: bool = False
    mode_frequency_hz: float = 1.0e6
    reference_k: float | None = None
    loop_number: float = 1.0
    eta: float = 0.1
    noise_case: str = "all_noise"
    scattering_model: str = "constant_rate"
    standard_time_points: int = 501
    smooth_time_points: int = 1501
    standard_solver_max_step_s: float = 0.2e-6
    smooth_solver_max_step_s: float = 0.05e-6
    random_seed: int = 1234
    show_progress: bool = True
    use_full_order: bool = True

    def __post_init__(self) -> None:
        if self.n_bar < 0.0:
            raise ValueError("n_bar must be non-negative")
        if self.workers < 1:
            raise ValueError("workers must be at least one")
        if not (0.0 < self.eta < 1.0):
            raise ValueError("eta must satisfy 0 < eta < 1")
        if self.reference_k is None and self.mode_frequency_hz <= 0.0:
            raise ValueError("mode_frequency_hz must be positive")
        if self.loop_number <= 0.0:
            raise ValueError("loop_number must be positive")
        if self.standard_time_points < 2 or self.smooth_time_points < 2:
            raise ValueError("time-point counts must be at least two")
        if (
            self.standard_solver_max_step_s <= 0.0
            or self.smooth_solver_max_step_s <= 0.0
        ):
            raise ValueError("solver max steps must be positive")


def _safe_nbar(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def _load_npz_chi(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        chi = np.asarray(payload[key], dtype=complex)
    if chi.shape != (16, 16):
        raise ValueError(f"Expected a 16x16 chi matrix in {path}; got {chi.shape}")
    if not np.isclose(np.trace(chi), 1.0, atol=1e-9):
        raise ValueError(f"Expected trace-normalized chi in {path}")
    return chi


def _numeric_mapping_matches(actual: dict, expected: dict) -> bool:
    if set(expected) - set(actual):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, (int, float)):
            if not np.isclose(float(actual_value), float(expected_value), rtol=1e-10):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _previous_smooth_cache(
    project_root: Path,
    condition: str,
    config: ThreePulseComparisonConfig,
    standard_config: smooth_analysis.StandardGateConfigSI,
    smooth_config: smooth_analysis.SmoothDetuningConfigSI,
) -> Path | None:
    """Return an older compatible standard/smooth cache, if it exists."""

    if condition not in {"standard_rectangular", "smooth_detuning"}:
        return None
    previous_dir = project_root / "results" / "smooth_detuning_nbar_sweep"
    previous_config_path = previous_dir / "config_used.yaml"
    if not previous_config_path.exists():
        return None
    try:
        previous = json.loads(previous_config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    sweep = previous.get("sweep", {})
    compatible = (
        _numeric_mapping_matches(previous.get("standard", {}), asdict(standard_config))
        and _numeric_mapping_matches(
            previous.get("smooth_detuning", {}), asdict(smooth_config)
        )
        and sweep.get("noise_case") == config.noise_case
        and sweep.get("scattering_model") == config.scattering_model
        and int(sweep.get("standard_time_points", -1)) == config.standard_time_points
        and int(sweep.get("smooth_time_points", -1)) == config.smooth_time_points
        and np.isclose(
            float(sweep.get("standard_solver_max_step_s", np.nan)),
            config.standard_solver_max_step_s,
        )
        and np.isclose(
            float(sweep.get("smooth_solver_max_step_s", np.nan)),
            config.smooth_solver_max_step_s,
        )
        and bool(sweep.get("use_full_order")) == config.use_full_order
        and any(
            np.isclose(float(value), config.n_bar)
            for value in sweep.get("nbar_values", [])
        )
        and np.isclose(config.eta, 0.1)
    )
    if not compatible:
        return None

    legacy_name = "standard" if condition == "standard_rectangular" else condition
    signature = str(previous.get("run_signature", ""))[:12]
    candidate = (
        previous_dir
        / "channels"
        / f"{signature}__{legacy_name}__nbar_{_safe_nbar(config.n_bar)}.npz"
    )
    return candidate if candidate.exists() else None


def prepare_comparison_pulses(config: ThreePulseComparisonConfig):
    """Build the three waveforms and return their shared-model metadata."""

    standard_config = smooth_analysis.StandardGateConfigSI()
    calibrated_smooth, smooth_calibration = smooth_analysis.calibrate_delta_min()
    standard = smooth_analysis.build_standard_pulse_si(
        standard_config,
        time_points=config.standard_time_points,
    )
    smooth = smooth_analysis.build_smooth_pulse_si(
        calibrated_smooth,
        time_points=config.smooth_time_points,
    )

    reference_k = (
        float(config.reference_k)
        if config.reference_k is not None
        else float(config.mode_frequency_hz) * standard_config.gate_time_s
    )
    reference_l = reference_k - float(config.loop_number)
    ratios = drive_formula.kirchhoff_renormalization_ratios(
        reference_k,
        reference_l,
        config.eta,
    )
    fourth_order_ratio = float(ratios["omega_4_over_omega_ld"])
    kirchhoff = {
        key: np.asarray(value).copy() for key, value in standard.items()
    }
    kirchhoff["A_rad_s"] *= fourth_order_ratio
    kirchhoff["omega_rad_s"] *= fourth_order_ratio

    pulses = {
        "standard_rectangular": standard,
        "smooth_detuning": smooth,
        "kirchhoff_eq41": kirchhoff,
    }
    metadata = {
        "standard_config": asdict(standard_config),
        "smooth_config": asdict(calibrated_smooth),
        "smooth_calibration": smooth_calibration,
        "kirchhoff": {
            "K": reference_k,
            "L": reference_l,
            "K_minus_L": float(config.loop_number),
            "eta": float(config.eta),
            **{key: float(value) for key, value in ratios.items()},
            "transfer_rule": "A_repository = A_standard * (Omega_4/Omega_LD)",
        },
    }
    return pulses, metadata, standard_config, calibrated_smooth


def _comparison_signature(config: ThreePulseComparisonConfig, metadata: dict) -> str:
    physical_config = asdict(config)
    # Worker count and resume/display switches do not change the channel.
    for operational_key in (
        "workers",
        "execute_qpt",
        "force_recompute",
        "show_progress",
    ):
        physical_config.pop(operational_key)
    payload = {
        "physical_config": physical_config,
        "standard_config": metadata["standard_config"],
        "smooth_config": metadata["smooth_config"],
        "kirchhoff": metadata["kirchhoff"],
        "error_channel_convention": "undo_before_actual",
        "model": "repository_full_order_first_sideband",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _qpt_sweep_config(config: ThreePulseComparisonConfig):
    return smooth_analysis.NbarSweepConfig(
        nbar_values=(float(config.n_bar),),
        noise_case=config.noise_case,
        scattering_model=config.scattering_model,
        standard_time_points=config.standard_time_points,
        smooth_time_points=config.smooth_time_points,
        standard_solver_max_step_s=config.standard_solver_max_step_s,
        smooth_solver_max_step_s=config.smooth_solver_max_step_s,
        parallel_workers=config.workers,
        random_seed=config.random_seed,
        show_progress=config.show_progress,
        use_full_order=config.use_full_order,
    )


def _obtain_channels(
    project_root: Path,
    output_dir: Path,
    config: ThreePulseComparisonConfig,
    pulses: dict,
    metadata: dict,
    standard_config: smooth_analysis.StandardGateConfigSI,
    smooth_config: smooth_analysis.SmoothDetuningConfigSI,
):
    signature = _comparison_signature(config, metadata)
    cache_dir = output_dir / "qpt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sweep_config = _qpt_sweep_config(config)
    channels: dict[str, np.ndarray] = {}
    cache_records: list[dict] = []

    for condition in CONDITION_ORDER:
        cache_path = (
            cache_dir
            / f"{signature[:12]}__{condition}__nbar_{_safe_nbar(config.n_bar)}.npz"
        )
        source_path: Path | None = None
        source_kind: str | None = None
        if cache_path.exists() and not config.force_recompute:
            chi = _load_npz_chi(cache_path, "chi_trace_normalized")
            source_path, source_kind = cache_path, "comparison_cache"
        elif not config.force_recompute:
            previous_path = _previous_smooth_cache(
                project_root,
                condition,
                config,
                standard_config,
                smooth_config,
            )
            if previous_path is not None:
                chi = _load_npz_chi(previous_path, "error_chi")
                source_path, source_kind = previous_path, "reused_smooth_study_cache"
                qpt_analysis.save_qpt_point(
                    cache_path,
                    config.n_bar,
                    condition,
                    chi,
                    {
                        "comparison_signature": signature,
                        "source_path": str(previous_path),
                        "source_kind": source_kind,
                    },
                )
        if source_path is None:
            if not config.execute_qpt:
                raise FileNotFoundError(
                    f"No compatible cache for {condition}. "
                    "Set execute_qpt=True to calculate the missing channel."
                )
            max_step = (
                config.smooth_solver_max_step_s
                if condition == "smooth_detuning"
                else config.standard_solver_max_step_s
            )
            params = smooth_analysis._qpt_parameters(
                pulses[condition],
                max_step,
                sweep_config,
                standard_config,
            )
            params["eta"] = float(config.eta)
            point = qpt_analysis.calculate_error_channel_batch(
                (config.n_bar,),
                params,
                convention="undo_before_actual",
            )[0]
            chi = np.asarray(point["chi"], dtype=complex)
            source_path, source_kind = cache_path, "calculated"
            qpt_analysis.save_qpt_point(
                cache_path,
                config.n_bar,
                condition,
                chi,
                {
                    **point["metadata"],
                    "comparison_signature": signature,
                    "source_kind": source_kind,
                },
            )
        channels[condition] = chi
        cache_records.append({
            "condition": condition,
            "source_kind": source_kind,
            "source_path": str(source_path),
        })
    return channels, pd.DataFrame(cache_records), signature


def _summarize_channels(
    channels: dict[str, np.ndarray],
    pulses: dict,
    config: ThreePulseComparisonConfig,
    metadata: dict,
    analyze_channel: Callable[[np.ndarray], dict],
) -> pd.DataFrame:
    labels = list(qpt_analysis._generator_design_data()[0])
    ii_index = labels.index("II")
    standard_peak = float(np.max(np.abs(pulses["standard_rectangular"]["A_rad_s"])))
    rows = []
    for condition in CONDITION_ORDER:
        chi = np.asarray(channels[condition], dtype=complex)
        chi_error_part = chi.copy()
        identity_weight = float(np.real(chi_error_part[ii_index, ii_index]))
        chi_error_part[ii_index, ii_index] = 0.0
        offdiagonal = chi_error_part - np.diag(np.diag(chi_error_part))
        diagonal = np.diag(np.diag(chi_error_part))
        error_norm = max(float(np.linalg.norm(chi_error_part)), 1e-15)
        analyzed = analyze_channel(chi)
        diagnostics = smooth_analysis.pulse_diagnostics(condition, pulses[condition])
        rows.append({
            "condition": condition,
            "label": CONDITION_LABELS[condition],
            "n_bar": float(config.n_bar),
            "gate_time_us": 1e6 * float(pulses[condition]["time_s"][-1]),
            "peak_A_over_2pi_kHz": (
                float(np.max(np.abs(pulses[condition]["A_rad_s"]))) / (2 * np.pi * 1e3)
            ),
            "A_scale_vs_standard": (
                float(np.max(np.abs(pulses[condition]["A_rad_s"])))
                / max(standard_peak, 1e-15)
            ),
            "chi_II_II_raw": identity_weight,
            "chi_error_frobenius_norm": error_norm,
            "chi_diagonal_fraction": float(np.linalg.norm(diagonal) / error_norm),
            "chi_offdiagonal_fraction": float(np.linalg.norm(offdiagonal) / error_norm),
            "alpha_final_abs_first_order": diagnostics["alpha_final_abs"],
            "xx_angle_first_order_rad": diagnostics["xx_angle_rad"],
            "kirchhoff_K": metadata["kirchhoff"]["K"],
            "kirchhoff_L": metadata["kirchhoff"]["L"],
            "kirchhoff_omega4_ratio": metadata["kirchhoff"][
                "omega_4_over_omega_ld"
            ],
            **analyzed["scalars"],
        })
    return pd.DataFrame(rows)


def plot_shared_scale_chi_heatmaps(
    channels: dict[str, np.ndarray],
    n_bar: float,
    output_dir: Path,
):
    """Plot Re, Im and abs(chi), sharing each component scale across rows."""

    labels = list(qpt_analysis._generator_design_data()[0])
    ii_index = labels.index("II")
    display_chi = {}
    identity_weights = {}
    for condition in CONDITION_ORDER:
        matrix = np.asarray(channels[condition], dtype=complex).copy()
        identity_weights[condition] = float(np.real(matrix[ii_index, ii_index]))
        matrix[ii_index, ii_index] = 0.0
        display_chi[condition] = matrix

    real_limit = max(
        max(float(np.max(np.abs(np.real(matrix)))), 1e-12)
        for matrix in display_chi.values()
    )
    imag_limit = max(
        max(float(np.max(np.abs(np.imag(matrix)))), 1e-12)
        for matrix in display_chi.values()
    )
    abs_limit = max(
        max(float(np.max(np.abs(matrix))), 1e-12)
        for matrix in display_chi.values()
    )
    component_specs = (
        ("Re chi", np.real, "RdBu_r", -real_limit, real_limit),
        ("Im chi", np.imag, "RdBu_r", -imag_limit, imag_limit),
        ("abs(chi)", np.abs, "viridis", 0.0, abs_limit),
    )

    figure, axes = plt.subplots(
        len(CONDITION_ORDER),
        len(component_specs),
        figsize=(17.0, 14.0),
        constrained_layout=True,
    )
    column_images = []
    ticks = np.arange(len(labels))
    for row, condition in enumerate(CONDITION_ORDER):
        matrix = display_chi[condition]
        for column, (title, transform, cmap, vmin, vmax) in enumerate(component_specs):
            axis = axes[row, column]
            image = axis.imshow(
                transform(matrix),
                origin="upper",
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            if row == 0:
                axis.set_title(title, fontsize=12)
                column_images.append(image)
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)
            axis.set_xticklabels(labels, rotation=90, fontsize=7)
            axis.set_yticklabels(labels, fontsize=7)
            axis.set_xlabel("Pauli column")
            axis.set_ylabel("Pauli row")
            if column == 0:
                axis.text(
                    -0.42,
                    0.5,
                    (
                        f"{CONDITION_LABELS[condition]}\n"
                        f"original chi[II,II]={identity_weights[condition]:.7f}"
                    ),
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    for column, image in enumerate(column_images):
        figure.colorbar(
            image,
            ax=list(axes[:, column]),
            shrink=0.78,
            pad=0.02,
            label=component_specs[column][0],
        )
    figure.suptitle(
        (
            f"Post-gate noise chi comparison at nbar={n_bar:g}\n"
            "same model/noise; shared scale within each column; "
            "display-only chi[II,II]=0"
        ),
        fontsize=15,
    )
    figure_path_png = output_dir / f"post_gate_chi_three_pulse_nbar_{_safe_nbar(n_bar)}.png"
    figure_path_pdf = output_dir / f"post_gate_chi_three_pulse_nbar_{_safe_nbar(n_bar)}.pdf"
    figure.savefig(figure_path_png, dpi=300, bbox_inches="tight")
    figure.savefig(figure_path_pdf, bbox_inches="tight")
    return figure, (figure_path_png, figure_path_pdf)


def run_three_pulse_chi_comparison(
    project_root,
    output_dir,
    config: ThreePulseComparisonConfig,
    analyze_channel: Callable[[np.ndarray], dict],
):
    """Load/resume QPT, write comparison products, and return the heatmap."""

    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pulses, metadata, standard_config, smooth_config = prepare_comparison_pulses(config)
    channels, cache_table, signature = _obtain_channels(
        project_root,
        output_dir,
        config,
        pulses,
        metadata,
        standard_config,
        smooth_config,
    )
    summary = _summarize_channels(
        channels,
        pulses,
        config,
        metadata,
        analyze_channel,
    )

    summary_path = output_dir / "three_pulse_channel_summary.csv"
    cache_table_path = output_dir / "three_pulse_cache_sources.csv"
    stack_path = output_dir / "post_gate_chi_three_pulse.npz"
    summary.to_csv(summary_path, index=False)
    cache_table.to_csv(cache_table_path, index=False)
    np.savez_compressed(
        stack_path,
        condition=np.asarray(CONDITION_ORDER),
        pauli_labels=np.asarray(qpt_analysis._generator_design_data()[0]),
        n_bar=float(config.n_bar),
        chi=np.stack([channels[name] for name in CONDITION_ORDER]),
    )

    figure, figure_paths = plot_shared_scale_chi_heatmaps(
        channels,
        config.n_bar,
        output_dir,
    )
    run_metadata = {
        "comparison_signature": signature,
        "config": asdict(config),
        **metadata,
        "error_channel_convention": "N_post = S_noisy @ inverse(S_ideal)",
        "noise_and_model_control": (
            "All three channels use the same repository full-order Hamiltonian, "
            "noise rates, eta, nbar, and post-gate convention."
        ),
        "kirchhoff_scope_caveat": (
            "Eq. (41) is transferred as Omega_4/Omega_LD to the rectangular "
            "repository pulse; this is not the paper's full carrier-sideband model."
        ),
        "duration_caveat": (
            "The calibrated smooth-detuning protocol has a longer gate duration "
            "than the rectangular protocols; the heatmap compares complete protocols."
        ),
        "cache_sources": cache_table.to_dict(orient="records"),
    }
    metadata_path = output_dir / "three_pulse_comparison_metadata.json"
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "channels": channels,
        "cache_sources": cache_table,
        "metadata": run_metadata,
        "figure": figure,
        "paths": {
            "summary": summary_path,
            "cache_sources": cache_table_path,
            "chi_stack": stack_path,
            "metadata": metadata_path,
            "figure_png": figure_paths[0],
            "figure_pdf": figure_paths[1],
        },
    }
