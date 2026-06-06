"""
Noise-source comparison for post-gate MS-gate error channels.

Implementation outline:
- Accept either an in-memory dictionary of noise-source results or files under
  data/<noise_source>/*.npy.
- For each source, compare the nominal post-gate channel against a source-specific
  baseline if available: ||R_post(lambda_k) - R_post(0)||_F / ||I||_F.
- Use chi diagonal entries as Pauli-twirled probabilities when explicit
  pauli_probs are not provided.
- Save three paper/slide-oriented figures and one summary CSV.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAULI_LABELS_2Q = [
    "II",
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
]


def frobenius_norm(A):
    """Return ||A||_F."""
    return float(np.linalg.norm(np.asarray(A, dtype=complex), ord="fro"))


def compute_pauli_weight(label):
    """Pauli weight = number of non-identity letters in a Pauli string."""
    return sum(char != "I" for char in str(label))


def _safe_ratio(numerator, denominator):
    if denominator is None or abs(denominator) < 1e-15:
        return np.nan
    return float(numerator / denominator)


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "full"):
        value = value.full()
    return np.asarray(value, dtype=complex)


def _validate_matrix_shape(name, matrix, shape=(16, 16)):
    matrix = _to_numpy(matrix)
    if matrix is None:
        return None
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {matrix.shape}.")
    return matrix


def _validate_pauli_probs(pauli_probs):
    pauli_probs = np.asarray(pauli_probs, dtype=float)
    if pauli_probs.shape != (16,):
        raise ValueError(f"pauli_probs must have shape (16,); got {pauli_probs.shape}.")
    return pauli_probs


def _trace_normalized_chi(chi):
    chi = _validate_matrix_shape("chi", chi)
    trace_chi = np.trace(chi)
    if abs(trace_chi) > 1e-15:
        chi = chi / trace_chi
    return chi


def pauli_probs_from_chi(chi):
    """Use p_P = real(diag(chi)) after trace normalization."""
    chi = _trace_normalized_chi(chi)
    pauli_probs = np.real(np.diag(chi)).astype(float)
    prob_sum = float(np.sum(pauli_probs))
    if abs(prob_sum) > 1e-15:
        pauli_probs = pauli_probs / prob_sum
    return _validate_pauli_probs(pauli_probs)


def compute_weighted_pauli_summary(pauli_labels, pauli_probs):
    """Compute p_err, p_w1, and p_w2 from Pauli-twirled probabilities."""
    pauli_probs = _validate_pauli_probs(pauli_probs)
    if len(pauli_labels) != 16:
        raise ValueError(f"pauli_labels must contain 16 labels; got {len(pauli_labels)}.")

    weights = np.asarray([compute_pauli_weight(label) for label in pauli_labels])
    p_ii = float(pauli_probs[0])
    p_w1 = float(np.sum(pauli_probs[weights == 1]))
    p_w2 = float(np.sum(pauli_probs[weights == 2]))
    p_err = float(1.0 - p_ii)

    return {
        "p_II": p_ii,
        "p_err": p_err,
        "p_w1": p_w1,
        "p_w2": p_w2,
        "p_w1_fraction_given_error": _safe_ratio(p_w1, p_err),
        "p_w2_fraction_given_error": _safe_ratio(p_w2, p_err),
        "p_w2_over_w1": _safe_ratio(p_w2, p_w1),
        "pauli_prob_sum": float(np.sum(pauli_probs)),
    }


def compute_channel_deformation(entry):
    """
    Preferred metric:
        ||R_post(lambda_k) - R_post(0)||_F / ||I||_F

    Fallback metric:
        ||lambda_k * dR_post/dlambda_k||_F / ||I||_F
    """
    R_post = _validate_matrix_shape("R_post", entry.get("R_post"))
    R_baseline = _validate_matrix_shape("R_baseline", entry.get("R_baseline"))
    dR_dlambda = _validate_matrix_shape("dR_dlambda", entry.get("dR_dlambda"))
    identity_norm = frobenius_norm(np.eye(16, dtype=complex))

    if R_post is not None and R_baseline is not None:
        return _safe_ratio(frobenius_norm(R_post - R_baseline), identity_norm)

    if dR_dlambda is not None:
        strength = float(entry.get("strength", 1.0))
        return _safe_ratio(frobenius_norm(strength * dR_dlambda), identity_norm)

    warnings.warn(
        "Neither R_baseline nor dR_dlambda was provided; "
        "channel deformation is set to NaN.",
        RuntimeWarning,
    )
    return np.nan


def compute_non_pauli_metrics(chi):
    """
    Compute non-Pauli/coherent-like metrics from a Pauli-basis chi matrix.

    C_off = ||chi - diag(diag(chi))||_F / ||chi||_F
    C_IE  = sqrt(||chi[0,1:]||_2^2 + ||chi[1:,0]||_2^2) / ||chi||_F
    C_EE  = ||chi[1:,1:] - diag(diag(chi[1:,1:]))||_F / ||chi||_F
    """
    if chi is None:
        warnings.warn("chi was not provided; non-Pauli metrics are set to NaN.")
        return {"C_off": np.nan, "C_IE": np.nan, "C_EE": np.nan, "chi_norm": np.nan}

    chi = _trace_normalized_chi(chi)
    chi_norm = frobenius_norm(chi)
    if chi_norm < 1e-15:
        return {"C_off": np.nan, "C_IE": np.nan, "C_EE": np.nan, "chi_norm": chi_norm}

    offdiag = chi - np.diag(np.diag(chi))
    identity_error_coupling = np.sqrt(
        np.sum(np.abs(chi[0, 1:]) ** 2) + np.sum(np.abs(chi[1:, 0]) ** 2)
    )
    error_block = chi[1:, 1:]
    error_block_offdiag = error_block - np.diag(np.diag(error_block))

    return {
        "C_off": _safe_ratio(frobenius_norm(offdiag), chi_norm),
        "C_IE": _safe_ratio(float(identity_error_coupling), chi_norm),
        "C_EE": _safe_ratio(frobenius_norm(error_block_offdiag), chi_norm),
        "chi_norm": chi_norm,
    }


def _select_point(points, target_s=1.0):
    if not points:
        raise ValueError("points must not be empty.")
    return min(points, key=lambda point: abs(float(point.get("s", 0.0)) - float(target_s)))


def _normalize_point_entry(source, source_data, nominal_s=1.0, baseline_s=0.0):
    """Convert direct entries or sweep-point lists into the common entry format."""
    if isinstance(source_data, dict) and "points" in source_data:
        source_data = source_data["points"]

    if isinstance(source_data, list):
        points = [dict(point) for point in source_data]
        nominal = dict(_select_point(points, target_s=nominal_s))
        baseline = _select_point(points, target_s=baseline_s)
        entry = {
            "source": source,
            "strength": nominal.get("strength", nominal.get("s", np.nan)),
            "R_post": nominal.get("R_post", nominal.get("R", nominal.get("error_ptm"))),
            "R_baseline": baseline.get(
                "R_post", baseline.get("R", baseline.get("error_ptm"))
            ),
            "dR_dlambda": nominal.get("dR_dlambda", nominal.get("ptm_derivative")),
            "chi": nominal.get("chi", nominal.get("error_chi")),
            "pauli_probs": nominal.get("pauli_probs"),
            "nominal_s": nominal.get("s", nominal_s),
            "baseline_s": baseline.get("s", baseline_s),
            "nbar": nominal.get("nbar", np.nan),
        }
        if entry["pauli_probs"] is None and entry["chi"] is not None:
            entry["pauli_probs"] = pauli_probs_from_chi(entry["chi"])
        return entry

    if isinstance(source_data, dict):
        entry = dict(source_data)
        entry.setdefault("source", source)
        entry["R_post"] = entry.get("R_post", entry.get("R", entry.get("error_ptm")))
        entry["chi"] = entry.get("chi", entry.get("error_chi"))
        if entry.get("pauli_probs") is None and entry.get("chi") is not None:
            entry["pauli_probs"] = pauli_probs_from_chi(entry["chi"])
        return entry

    raise TypeError(f"Unsupported data type for source {source}: {type(source_data)}")


def normalize_results(results, nominal_s=1.0, baseline_s=0.0):
    if not isinstance(results, dict):
        raise TypeError("results must be a dictionary keyed by noise source.")
    return {
        source: _normalize_point_entry(
            source,
            source_data,
            nominal_s=nominal_s,
            baseline_s=baseline_s,
        )
        for source, source_data in results.items()
    }


def build_summary_dataframe(results):
    """Build one summary row per noise source."""
    normalized = normalize_results(results)
    rows = []

    for source, entry in normalized.items():
        channel_deformation = compute_channel_deformation(entry)

        pauli_probs = entry.get("pauli_probs")
        if pauli_probs is None and entry.get("chi") is not None:
            pauli_probs = pauli_probs_from_chi(entry["chi"])
        if pauli_probs is None:
            warnings.warn(
                f"{source}: pauli_probs and chi are missing; Pauli summary is NaN.",
                RuntimeWarning,
            )
            pauli_summary = {
                "p_II": np.nan,
                "p_err": np.nan,
                "p_w1": np.nan,
                "p_w2": np.nan,
                "p_w1_fraction_given_error": np.nan,
                "p_w2_fraction_given_error": np.nan,
                "p_w2_over_w1": np.nan,
                "pauli_prob_sum": np.nan,
            }
        else:
            pauli_summary = compute_weighted_pauli_summary(PAULI_LABELS_2Q, pauli_probs)

        non_pauli_metrics = compute_non_pauli_metrics(entry.get("chi"))

        rows.append(
            {
                "noise_source": source,
                "strength": entry.get("strength", np.nan),
                "nominal_s": entry.get("nominal_s", np.nan),
                "baseline_s": entry.get("baseline_s", np.nan),
                "nbar": entry.get("nbar", np.nan),
                "channel_deformation": channel_deformation,
                **pauli_summary,
                **non_pauli_metrics,
            }
        )

    df = pd.DataFrame(rows)
    if "channel_deformation" in df:
        df = df.sort_values(
            "channel_deformation", ascending=False, na_position="last"
        ).reset_index(drop=True)
    return df


def _save_figure(fig, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    base = save_path.with_suffix("") if save_path.suffix else save_path
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


def _ordered_plot_dataframe(df):
    return df.sort_values(
        "channel_deformation", ascending=False, na_position="last"
    ).reset_index(drop=True)


def plot_channel_deformation(df, save_path):
    plot_df = _ordered_plot_dataframe(df)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(plot_df["noise_source"], plot_df["channel_deformation"], color="tab:blue")
    ax.set_ylabel(r"Normalized channel deformation")
    ax.set_xlabel("Noise source")
    ax.set_title("Noise-source comparison of channel deformation")
    ax.grid(True, axis="y", alpha=0.35)
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def plot_pauli_composition(df, save_path):
    plot_df = _ordered_plot_dataframe(df)
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(x, plot_df["p_w1"], label="weight-1", color="tab:blue")
    ax.bar(x, plot_df["p_w2"], bottom=plot_df["p_w1"], label="weight-2", color="tab:orange")
    ax.plot(x, plot_df["p_err"], "ko", label=r"$p_{err}$")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["noise_source"], rotation=35, ha="right")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Noise source")
    ax.set_title("Pauli error composition by noise source")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def plot_non_pauli_metrics(df, save_path):
    plot_df = _ordered_plot_dataframe(df)
    x = np.arange(len(plot_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width, plot_df["C_off"], width=width, label=r"$C_{off}$")
    ax.bar(x, plot_df["C_IE"], width=width, label=r"$C_{IE}$")
    ax.bar(x + width, plot_df["C_EE"], width=width, label=r"$C_{EE}$")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["noise_source"], rotation=35, ha="right")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Noise source")
    ax.set_title("Non-Pauli / coherent-like structure by noise source")
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def load_results_from_data_dir(data_dir="data"):
    """
    Load optional file-based results.

    TODO: place files like the following when notebook variables are unavailable:
        data/<noise_source>/R_post.npy
        data/<noise_source>/R_baseline.npy
        data/<noise_source>/dR_dlambda.npy
        data/<noise_source>/chi.npy
        data/<noise_source>/pauli_probs.npy
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return {}

    results = {}
    for source_dir in sorted(path for path in data_path.iterdir() if path.is_dir()):
        entry = {}
        for key, filename in {
            "R_post": "R_post.npy",
            "R_baseline": "R_baseline.npy",
            "dR_dlambda": "dR_dlambda.npy",
            "chi": "chi.npy",
            "pauli_probs": "pauli_probs.npy",
        }.items():
            file_path = source_dir / filename
            if file_path.exists():
                entry[key] = np.load(file_path)
        if entry:
            results[source_dir.name] = entry

    return results


def _lookup_mapping_by_float_key(mapping, target_key):
    for key, value in mapping.items():
        if np.isclose(float(key), float(target_key)):
            return value
    raise KeyError(f"Key {target_key} was not found.")


def _ordered_nbars(error_result):
    if "results_list" in error_result:
        return [float(data["n_bar"]) for data in error_result["results_list"]]
    if "error_ptm_by_n_bar" in error_result:
        return sorted(float(key) for key in error_result["error_ptm_by_n_bar"].keys())
    raise KeyError("error_result does not contain n_bar information.")


def entry_from_temperature_sweep(
    error_result,
    source_name="initial_temperature",
    baseline_nbar=0.01,
    target_nbar=None,
):
    """Create an initial-temperature comparison entry from an error_result dictionary."""
    nbars = _ordered_nbars(error_result)
    if target_nbar is None:
        target_nbar = nbars[-1]
    baseline_nbar = nbars[int(np.argmin(np.abs(np.asarray(nbars) - float(baseline_nbar))))]
    target_nbar = nbars[int(np.argmin(np.abs(np.asarray(nbars) - float(target_nbar))))]

    if "error_ptm_by_n_bar" in error_result:
        R_post = _lookup_mapping_by_float_key(error_result["error_ptm_by_n_bar"], target_nbar)
        R_baseline = _lookup_mapping_by_float_key(
            error_result["error_ptm_by_n_bar"], baseline_nbar
        )
    else:
        index_target = nbars.index(target_nbar)
        index_baseline = nbars.index(baseline_nbar)
        R_post = error_result["error_ptm_list"][index_target]
        R_baseline = error_result["error_ptm_list"][index_baseline]

    if "error_chi_by_n_bar" in error_result:
        chi = _lookup_mapping_by_float_key(error_result["error_chi_by_n_bar"], target_nbar)
    else:
        index_target = nbars.index(target_nbar)
        chi = error_result["error_chi_matrix_list"][index_target]

    return {
        "source": source_name,
        "strength": target_nbar,
        "nbar": target_nbar,
        "R_post": R_post,
        "R_baseline": R_baseline,
        "chi": chi,
        "pauli_probs": pauli_probs_from_chi(chi),
    }


def collect_results_from_notebook(
    namespace,
    include_initial_temperature=True,
    baseline_nbar=0.01,
    target_nbar=None,
):
    """Collect existing notebook variables without rerunning any simulation."""
    results = {}

    if "noise_source_sweep_results" in namespace:
        results.update(namespace["noise_source_sweep_results"])
    elif "post_gate_channel_figure_results" in namespace:
        figure_results = namespace["post_gate_channel_figure_results"]
        if isinstance(figure_results, dict) and "source_metrics" in figure_results:
            results.update(figure_results["source_metrics"])

    if include_initial_temperature and "error_result" in namespace:
        try:
            results.setdefault(
                "initial_temperature",
                entry_from_temperature_sweep(
                    namespace["error_result"],
                    baseline_nbar=baseline_nbar,
                    target_nbar=target_nbar,
                ),
            )
        except Exception as exc:
            warnings.warn(f"initial_temperature entry could not be created: {exc}")

    return results


def main(
    results=None,
    output_dir="outputs",
    data_dir="data",
    namespace=None,
    include_initial_temperature=True,
):
    """
    Run the full comparison workflow.

    If results is None:
    - use notebook globals when namespace is provided;
    - otherwise try file-based loading from data_dir.
    """
    if results is None:
        if namespace is not None:
            results = collect_results_from_notebook(
                namespace,
                include_initial_temperature=include_initial_temperature,
            )
        else:
            results = {}

    if not results:
        results = load_results_from_data_dir(data_dir=data_dir)

    if not results:
        raise RuntimeError(
            "No noise-source results were found. TODO: provide an in-memory results "
            "dictionary or place .npy files under data/<noise_source>/."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_df = build_summary_dataframe(results)
    summary_csv = output_path / "noise_source_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    figures = {}
    figures["channel_deformation"] = plot_channel_deformation(
        summary_df, output_path / "noise_source_channel_deformation"
    )[0]
    figures["pauli_composition"] = plot_pauli_composition(
        summary_df, output_path / "noise_source_pauli_composition"
    )[0]
    figures["nonpauli_metrics"] = plot_non_pauli_metrics(
        summary_df, output_path / "noise_source_nonpauli_metrics"
    )[0]

    print(f"Saved: {summary_csv}")
    print(f"Saved figures to: {output_path}")

    return {
        "summary_df": summary_df,
        "figures": figures,
        "output_dir": output_path,
        "summary_csv": summary_csv,
    }


if __name__ == "__main__":
    main()
