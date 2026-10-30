"""COM-only / axial COM + stretch MS gates, using the existing QPT pipeline.

The tensor order is (ion 1, ion 2, COM[, stretch]). Equal-mass ions in one
harmonic well and equal optical spin phases at the ions are assumed. COM
couples to X1+X2 (or Y1+Y2), stretch to X1-X2 (or Y1-Y2).

Conventions match ms_gate_functions.py: H/hbar = A S (B exp(-i Phi) + h.c.),
and A already contains the COM Lamb-Dicke factor. We define delta_m =
omega_m - mu, hence delta_stretch = delta_COM + (omega_stretch - omega_COM).
Physical angular frequencies are multiplied by t_gate_phys / t_gate_sim.

use_full_order retains only the relative O(eta**2) first-sideband correction,
including spectator Debye-Waller terms. It is NOT an all-orders optical model.
Mixed/higher sidebands, radial modes, micromotion, and correlated motional
noise are outside this model. Initial thermal states and motional dissipators
are independent between modes; laser fluctuations are shared by both modes.

Nothing runs on import. use_two_modes=False delegates the simulation directly
to ms_gate_functions.run_ms_gate_simulation without changing that implementation.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import inspect
import json
from pathlib import Path

import numpy as np
import qutip as qp

import ms_gate_functions as mg


def _number(name, value, *, positive=False):
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return value


def _integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def axial_mode_parameters(com_frequency_hz, eta_com, *,
                          stretch_frequency_hz=None, eta_stretch=None):
    """Resolve frequencies and per-ion Lamb-Dicke factors (including 1/sqrt(2)).

    None selects omega_s = sqrt(3)*omega_c and eta_s/eta_c = sqrt(omega_c/omega_s).
    At nonzero eta_com, eta_stretch=0 disables its drive and Debye-Waller effect.
    eta_com=eta_stretch=0 is allowed as a formal LD limit at fixed effective A.
    """
    fc = _number("com_frequency_hz", com_frequency_hz, positive=True)
    fs = np.sqrt(3.0) * fc if stretch_frequency_hz is None else _number(
        "stretch_frequency_hz", stretch_frequency_hz, positive=True
    )
    if fs <= fc:
        raise ValueError("Axial stretch frequency must exceed the COM frequency.")
    ec = _number("eta_com", eta_com)
    ratio = np.sqrt(fc / fs)
    es = ec * ratio if eta_stretch is None else _number("eta_stretch", eta_stretch)
    if ec > 0:
        ratio = es / ec
    elif es > 0:
        raise ValueError("A nonzero eta_stretch requires nonzero eta_com.")
    return {
        "com_frequency_hz": fc,
        "stretch_frequency_hz": fs,
        "eta_com": ec,
        "eta_stretch": es,
        "stretch_amplitude_ratio": ratio,
        "detuning_convention": "delta_m = omega_m - mu",
    }


def com_stretch_operators(phonon_dim_com, phonon_dim_stretch, eta_com,
                         eta_stretch, *, use_full_order=True):
    """Lift the existing operators, change the stretch spin sign, add DW terms.

    With N_m=a_m.dag()*a_m, the COM lowering operator is
      B_c = a_c - eta_c**2/2 * (N_c+1)*a_c
                  - eta_s**2 * (N_s+1/2)*a_c,
    and B_s follows by exchanging c and s. Products of these corrections,
    which would be of relative order eta**4, are deliberately not introduced.
    """
    nc = _integer("phonon_dim_com", phonon_dim_com, 2)
    ns = _integer("phonon_dim_stretch", phonon_dim_stretch, 2)
    ec = _number("eta_com", eta_com)
    es = _number("eta_stretch", eta_stretch)
    ic, is_ = qp.qeye(nc), qp.qeye(ns)
    com = {
        key: qp.tensor(op, is_)
        for key, op in mg._ms_gate_static_operators(nc, ec, use_full_order).items()
    }
    stretch = {
        key: qp.tensor(op, ic).permute([0, 1, 3, 2])
        for key, op in mg._ms_gate_static_operators(ns, es, use_full_order).items()
    }
    z2 = qp.tensor(qp.qeye(2), qp.sigmaz(), ic, is_)
    for key in ("Hx_minus", "Hx_plus", "Hy_minus", "Hy_plus"):
        stretch[key] = z2 * stretch[key] * z2

    if use_full_order:
        ac, as_ = qp.destroy(nc), qp.destroy(ns)
        n_com, n_stretch = ac.dag() * ac, as_.dag() * as_
        for axis, pauli in (("x", qp.sigmax()), ("y", qp.sigmay())):
            p1, p2 = qp.tensor(pauli, qp.qeye(2)), qp.tensor(qp.qeye(2), pauli)
            key = f"H{axis}_minus"
            com[key] -= es**2 * qp.tensor(p1 + p2, ac, n_stretch + 0.5 * is_)
            stretch[key] -= ec**2 * qp.tensor(p1 - p2, n_com + 0.5 * ic, as_)
            com[f"H{axis}_plus"] = com[key].dag()
            stretch[f"H{axis}_plus"] = stretch[key].dag()
    return {"com": com, "stretch": stretch}


def build_two_mode_hamiltonian(operators, time_grid, detuning_com, rho,
                               amplitude_com, *, mode_gap_sim,
                               stretch_amplitude_ratio):
    """Reuse COM coefficients; apply the fast stretch phase analytically.

    Interpolating exp(-i delta_stretch*t) on the old 500-point COM grid can
    alias the spectator motion. Instead, interpolate the same slow envelope
    as the existing model, and multiply it by exp(-i mode_gap_sim*t) exactly.
    H is in simulation angular-frequency units (H/hbar), as in the old solver.
    """
    time_grid = mg._as_time_grid(time_grid)
    delta = mg._as_control_values("detuning_com", detuning_com, time_grid)
    phase = mg._integrated_control_phase(delta, time_grid)
    rho = mg._as_control_values("rho", rho, time_grid)
    amplitude = mg._as_control_values("amplitude_com", amplitude_com, time_grid)
    gap = float(mode_gap_sim)
    if not np.isfinite(gap):
        raise ValueError("mode_gap_sim must be finite.")
    ratio = _number("stretch_amplitude_ratio", stretch_amplitude_ratio)
    h_com = mg._build_ms_hamiltonian(
        operators["com"], time_grid, delta, rho, amplitude
    )
    phase_minus = qp.coefficient(lambda t: np.exp(-1j * gap * t))
    phase_plus = qp.coefficient(lambda t: np.exp(1j * gap * t))
    terms = []
    for axis, projection in (("x", np.cos(rho)), ("y", np.sin(rho))):
        envelope = ratio * amplitude * projection * np.exp(-1j * phase)
        minus = qp.coefficient(envelope, tlist=time_grid, order=1) * phase_minus
        plus = qp.coefficient(envelope.conj(), tlist=time_grid, order=1) * phase_plus
        terms.extend([
            [operators["stretch"][f"H{axis}_minus"], minus],
            [operators["stretch"][f"H{axis}_plus"], plus],
        ])
    return h_com + qp.QobjEvo(terms)


def _resolved_parameters(parameters):
    bound = inspect.signature(mg.run_ms_gate_simulation).bind(**parameters)
    bound.apply_defaults()
    p = dict(bound.arguments)
    if p["n_bar_list"] is None:
        p["n_bar_list"] = [0.01, 1, 2, 3, 4, 5]
    p["n_bar_list"] = [_number("n_bar", n) for n in p["n_bar_list"]]
    if not p["n_bar_list"] or len(set(p["n_bar_list"])) != len(p["n_bar_list"]):
        raise ValueError("n_bar_list must be nonempty and contain distinct values.")
    p["time_points"] = _integer("time_points", p["time_points"], 2)
    p["laser_noise_samples"] = _integer("laser_noise_samples", p["laser_noise_samples"])
    p["parallel_workers"] = _integer(
        "parallel_workers", 30 if p["parallel_workers"] is None else p["parallel_workers"]
    )
    for key in ("heating_rate_phys", "dephasing_rate_phys", "rayleigh_rate_phys",
                "raman_rate_phys", "eta", "laser_intensity_fluctuation",
                "laser_detuning_fluctuation", "laser_rotation_angle_fluctuation"):
        p[key] = _number(key, p[key])
    t2 = float(p["T2_star"])
    if np.isnan(t2) or t2 <= 0:
        raise ValueError("T2_star must be positive or np.inf.")
    p["T2_star"] = t2
    p["t_gate_phys"] = _number("t_gate_phys", p["t_gate_phys"], positive=True)
    if p["t_gate_sim"] is None:
        delta = np.asarray(p["delta"], dtype=float)
        if delta.ndim != 0 or not np.isfinite(delta) or delta == 0:
            raise ValueError("A finite nonzero scalar delta or explicit t_gate_sim is required.")
        p["t_gate_sim"] = 2 * np.pi / abs(float(delta))
    p["t_gate_sim"] = _number("t_gate_sim", p["t_gate_sim"], positive=True)
    if p["phonon_dim_override"] is not None:
        p["phonon_dim_override"] = _integer("phonon_dim_override", p["phonon_dim_override"], 2)
    if p["solver_max_step"] is not None:
        p["solver_max_step"] = _number("solver_max_step", p["solver_max_step"], positive=True)
    return p


def _alpha_bound(amplitude, detuning, times, intensity_noise, detuning_noise):
    # Use the existing off-resonant estimate when possible. Crossing a sideband
    # is allowed with explicit gate time, so use the finite-time force bound there.
    lower = float(np.min(np.abs(detuning))) - 3 * detuning_noise
    crosses_zero = float(np.min(detuning)) <= 0 <= float(np.max(detuning))
    if lower > 0 and not crosses_zero:
        return mg._estimate_alpha_max(amplitude, detuning, intensity_noise, detuning_noise)
    return float(np.max(np.abs(amplitude))) * (1 + 3 * intensity_noise) * (times[-1] - times[0])


def prepare_simulation(*, use_two_modes=False, com_frequency_hz=1.0e6,
                       stretch_frequency_hz=None, eta_stretch=None,
                       n_bar_stretch=0.01, stretch_heating_rate_phys=0.0,
                       stretch_dephasing_rate_phys=0.0,
                       stretch_phonon_dim_override=None, **simulation_parameters):
    """Return settings/cutoff estimates only; never build a solver or evolve a state.

    n_bar_list sweeps COM temperature. n_bar_stretch is a fixed mean occupation
    or a list paired with n_bar_list, not a Cartesian grid or a common temperature.
    Frequencies in Hz are illustrative until replaced by the measured LIZ values.
    """
    if not isinstance(use_two_modes, (bool, np.bool_)):
        raise TypeError("use_two_modes must be True or False.")
    p = _resolved_parameters(simulation_parameters)
    times = np.linspace(0.0, p["t_gate_sim"], p["time_points"])
    amplitude = mg._as_control_values("A", p["A"], times)
    delta = mg._as_control_values("delta", p["delta"], times)
    rho = mg._as_control_values("rho0", p["rho0"], times)
    if not all(np.all(np.isfinite(v)) for v in (amplitude, delta, rho)):
        raise ValueError("Control waveforms must be finite.")
    rates = mg._scaled_noise_rates(
        p["t_gate_phys"], p["t_gate_sim"], p["heating_rate_phys"],
        p["dephasing_rate_phys"], p["T2_star"], p["rayleigh_rate_phys"],
        p["raman_rate_phys"],
    )
    ac = _alpha_bound(amplitude, delta, times, p["laser_intensity_fluctuation"],
                      p["laser_detuning_fluctuation"])
    modes = None
    ns_values = [None] * len(p["n_bar_list"])
    a_s = 0.0
    max_step = p["solver_max_step"]
    if use_two_modes:
        modes = axial_mode_parameters(
            com_frequency_hz, p["eta"], stretch_frequency_hz=stretch_frequency_hz,
            eta_stretch=eta_stretch,
        )
        modes["mode_gap_sim"] = (
            2 * np.pi * (modes["stretch_frequency_hz"] - modes["com_frequency_hz"])
            * rates["scale_factor"]
        )
        modes["delta_stretch_sim"] = delta + modes["mode_gap_sim"]
        ns_values = np.asarray(n_bar_stretch, dtype=float)
        if ns_values.ndim == 0:
            ns_values = np.full(len(p["n_bar_list"]), float(ns_values))
        if ns_values.shape != (len(p["n_bar_list"]),):
            raise ValueError("n_bar_stretch must be scalar or match n_bar_list length.")
        ns_values = [_number("n_bar_stretch", value) for value in ns_values]
        if stretch_phonon_dim_override is not None:
            stretch_phonon_dim_override = _integer(
                "stretch_phonon_dim_override", stretch_phonon_dim_override, 2
            )
        modes["stretch_heating_rate_phys"] = _number(
            "stretch_heating_rate_phys", stretch_heating_rate_phys
        )
        modes["stretch_dephasing_rate_phys"] = _number(
            "stretch_dephasing_rate_phys", stretch_dephasing_rate_phys
        )
        rates["stretch_heating"] = modes["stretch_heating_rate_phys"] * rates["scale_factor"]
        rates["stretch_dephasing"] = modes["stretch_dephasing_rate_phys"] * rates["scale_factor"]
        a_s = _alpha_bound(
            amplitude * modes["stretch_amplitude_ratio"], modes["delta_stretch_sim"],
            times, p["laser_intensity_fluctuation"], p["laser_detuning_fluctuation"],
        )
        # Resolve the fastest phase even if the output grid is coarse.
        fastest = max(float(np.max(np.abs(delta))),
                      float(np.max(np.abs(modes["delta_stretch_sim"]))))
        fastest += 3 * p["laser_detuning_fluctuation"]
        safe_step = float(np.min(np.diff(times)))
        if fastest > 0:
            safe_step = min(safe_step, 2 * np.pi / (40 * fastest))
        max_step = safe_step if max_step is None else min(max_step, safe_step)
    rows = []
    for n_c, n_s in zip(p["n_bar_list"], ns_values):
        nc = p["phonon_dim_override"] or mg.estimate_phonon_dim(
            n_c, (2 if use_two_modes else 1) * ac
        )
        ns = None if not use_two_modes else (
            stretch_phonon_dim_override or mg.estimate_phonon_dim(n_s, 2 * a_s)
        )
        rows.append({
            "n_bar": n_c, "n_bar_stretch": n_s, "Nv": nc, "Nv_stretch": ns,
            "hilbert_dimension": 4 * nc * (ns or 1),
            "initial_com_tail": (n_c / (1 + n_c))**nc,
            "initial_stretch_tail": None if ns is None else (n_s / (1 + n_s))**ns,
        })
    return {
        "use_two_modes": bool(use_two_modes), "parameters": p, "modes": modes,
        "tlist": times, "amplitude": amplitude, "detuning": delta, "rho": rho,
        "rates": rates, "conditions": rows,
        "alpha_max_est": ac, "alpha_max_est_stretch": a_s,
        "solver_max_step": max_step,
    }


def build_two_mode_solver(plan, condition, laser_parameters):
    """Construct a MESolver without executing it; spin noise is included once."""
    if not plan["use_two_modes"]:
        raise ValueError("build_two_mode_solver needs use_two_modes=True.")
    p, modes, rates = plan["parameters"], plan["modes"], plan["rates"]
    ops = com_stretch_operators(
        condition["Nv"], condition["Nv_stretch"], p["eta"], modes["eta_stretch"],
        use_full_order=p["use_full_order"],
    )
    hamiltonian = build_two_mode_hamiltonian(
        ops, plan["tlist"], laser_parameters["detuning"], laser_parameters["rho"],
        laser_parameters["int_strn"], mode_gap_sim=modes["mode_gap_sim"],
        stretch_amplitude_ratio=modes["stretch_amplitude_ratio"],
    )
    intensity_scale = None
    if p["laser_scattering_scales_with_intensity"]:
        reference = p["scattering_reference_amplitude"]
        if reference is None:
            reference = float(np.max(np.abs(plan["amplitude"])))
        reference = _number("scattering_reference_amplitude", reference, positive=True)
        intensity_scale = (np.abs(laser_parameters["int_strn"]) / reference)**2
    c_ops = mg._build_c_ops(
        ops["com"], heating_rate=rates["heating"], dephasing_rate=rates["dephasing"],
        spin_dephasing_rate=rates["spin"], rayleigh_scattering_rate=rates["rayleigh"],
        raman_scattering_rate=rates["raman"], time_grid=plan["tlist"],
        scattering_intensity_scale=intensity_scale,
    )
    c_ops += mg._build_c_ops(
        ops["stretch"], heating_rate=rates["stretch_heating"],
        dephasing_rate=rates["stretch_dephasing"],
    )
    # If a sampled detuning exceeds the nominal 3-sigma estimate, resolve it too.
    sampled_delta = np.asarray(laser_parameters["detuning"], dtype=float)
    fastest = max(float(np.max(np.abs(sampled_delta))),
                  float(np.max(np.abs(sampled_delta + modes["mode_gap_sim"]))))
    max_step = plan["solver_max_step"]
    if fastest > 0:
        max_step = min(max_step, 2 * np.pi / (40 * fastest))
    options = mg._solver_options(store_states=False, max_step=max_step)
    options["nsteps"] = max(10000, int(20 * np.max(np.diff(plan["tlist"])) / max_step))
    return qp.MESolver(hamiltonian, c_ops, options=options)


def _run_condition(task):
    plan, condition, samples = task
    inputs = mg.build_input_states()
    thermal = qp.tensor(qp.thermal_dm(condition["Nv"], condition["n_bar"]),
                        qp.thermal_dm(condition["Nv_stretch"], condition["n_bar_stretch"]))
    outputs = [None] * len(inputs)
    edge_com = edge_stretch = 0.0
    for laser in samples:
        solver = build_two_mode_solver(plan, condition, laser)
        for index, state in enumerate(inputs):
            result = solver.run(qp.tensor(state, thermal), plan["tlist"], e_ops=[])
            final = mg._final_state_from_result(result)
            reduced = final.ptrace([0, 1]) / len(samples)
            outputs[index] = reduced if outputs[index] is None else outputs[index] + reduced
            edge_com = max(edge_com, float(np.real(final.ptrace(2).diag()[-2:].sum())))
            edge_stretch = max(edge_stretch, float(np.real(final.ptrace(3).diag()[-2:].sum())))
    return {
        **condition, "outputs": outputs, "sampled_laser_params": samples,
        "max_final_com_edge_population": edge_com,
        "max_final_stretch_edge_population": edge_stretch,
    }


def run_ms_gate_simulation(*, use_two_modes=False, com_frequency_hz=1.0e6,
                          stretch_frequency_hz=None, eta_stretch=None,
                          n_bar_stretch=0.01, stretch_heating_rate_phys=0.0,
                          stretch_dephasing_rate_phys=0.0,
                          stretch_phonon_dim_override=None, **simulation_parameters):
    """Drop-in QPT simulation result; False uses the original simulation verbatim.

    Two-mode parallelism is over COM occupation values, reusing one solver for
    all 16 input states within each laser sample. The notebook defaults to one
    worker because density-matrix memory grows as (4*Nc*Ns)**2.
    """
    if not isinstance(use_two_modes, (bool, np.bool_)):
        raise TypeError("use_two_modes must be True or False.")
    if not use_two_modes:
        return mg.run_ms_gate_simulation(**simulation_parameters)
    plan = prepare_simulation(
        use_two_modes=True, com_frequency_hz=com_frequency_hz,
        stretch_frequency_hz=stretch_frequency_hz, eta_stretch=eta_stretch,
        n_bar_stretch=n_bar_stretch, stretch_heating_rate_phys=stretch_heating_rate_phys,
        stretch_dephasing_rate_phys=stretch_dephasing_rate_phys,
        stretch_phonon_dim_override=stretch_phonon_dim_override, **simulation_parameters,
    )
    p = plan["parameters"]
    has_laser_noise = any(p[key] > 0 for key in (
        "laser_intensity_fluctuation", "laser_detuning_fluctuation",
        "laser_rotation_angle_fluctuation",
    ))
    sample_count = p["laser_noise_samples"] if has_laser_noise else 1
    rng = np.random.default_rng(p["laser_noise_seed"])
    tasks = []
    for row in plan["conditions"]:
        samples = [mg.sample_laser_parameters(
            plan["detuning"], plan["amplitude"], plan["rho"],
            intensity_fluctuation=p["laser_intensity_fluctuation"],
            detuning_fluctuation=p["laser_detuning_fluctuation"],
            rotation_angle_fluctuation=p["laser_rotation_angle_fluctuation"], rng=rng,
        ) for _ in range(sample_count)]
        tasks.append((plan, row, samples))
    # Resolve callables to arrays before sending jobs to another process.
    p.update(A=plan["amplitude"], delta=plan["detuning"], rho0=plan["rho"])
    results = []
    workers = min(p["parallel_workers"], len(tasks))
    executor = None
    try:
        if workers > 1:
            mg._configure_parallel_thread_env()
            executor = ProcessPoolExecutor(max_workers=workers, **mg._process_pool_executor_kwargs())
            iterator = executor.map(_run_condition, tasks)
        else:
            iterator = map(_run_condition, tasks)
        for index, result in enumerate(iterator):
            results.append(result)
            if p["show_progress"]:
                print(f"COM+stretch: {index + 1}/{len(tasks)}; "
                      f"n_COM={result['n_bar']}, n_stretch={result['n_bar_stretch']}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return {
        "parameters": {
            **p, "use_two_modes": True, "n_bar_stretch": [r["n_bar_stretch"] for r in results],
            "effective_laser_samples": sample_count,
            "stretch_phonon_dim_override": stretch_phonon_dim_override,
            "solver_max_step": plan["solver_max_step"],
        },
        "mode_parameters": plan["modes"], "rates": plan["rates"],
        "tlist": plan["tlist"], "input_states_list": mg.build_input_states(),
        "results_list": results, "alpha_max_est": plan["alpha_max_est"],
        "alpha_max_est_stretch": plan["alpha_max_est_stretch"],
    }


def run_analysis(*, execute=False, use_two_modes=False, show_plot=False,
                 error_channel_convention="undo_before_actual", **parameters):
    """Run the new simulation followed by the unchanged fidelity/chi/PTM helpers.

    execute=False returns None immediately: no solver, files, or cached results.
    The original exp(+i*pi*XX/4) target is kept; no two-mode recalibration is hidden.
    """
    if not execute:
        return None
    simulation = run_ms_gate_simulation(use_two_modes=use_two_modes, **parameters)
    result = mg.run_infidelity_analysis(show_plot=show_plot, simulation_result=simulation)
    return mg.generate_error_channel_matrices(
        channel_result=result, convention=error_channel_convention
    )


def summary_table(result):
    """Tabulate existing fidelity results plus explicit mode occupations/cutoffs."""
    import pandas as pd

    if result is None:
        return pd.DataFrame()
    rows = []
    for data, fidelity in zip(result["results_list"], result["f_avg_list"]):
        rows.append({
            "model": "COM+stretch" if result["parameters"].get("use_two_modes") else "COM",
            "n_bar_com": data["n_bar"], "n_bar_stretch": data.get("n_bar_stretch"),
            "Nv_com": data["Nv"], "Nv_stretch": data.get("Nv_stretch"),
            "F_avg": fidelity, "infidelity": 1.0 - fidelity,
            "max_final_com_edge_population": data.get("max_final_com_edge_population", np.nan),
            "max_final_stretch_edge_population": data.get("max_final_stretch_edge_population", np.nan),
        })
    return pd.DataFrame(rows)


def plot_results(result, *, index=0):
    """Plot infidelity and use the existing post-gate chi/PTM plotting functions."""
    import matplotlib.pyplot as plt

    if result is None:
        return None
    table = summary_table(result)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(table["n_bar_com"], np.maximum(table["infidelity"], 1e-16), "o-")
    ax.set(xlabel=r"COM mean occupation $\bar n_c$", ylabel=r"$1-F_{avg}$",
           title=str(table["model"].iloc[0]))
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    chi_fig, _ = mg.plot_chi_matrix(result["error_chi_qobj_list"][index])
    ptm_fig, _ = mg.plot_ptm(result["error_ptm_list"][index])
    return {"infidelity": fig, "error_chi": chi_fig, "error_ptm": ptm_fig}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if callable(value):
        raise TypeError("Save evaluated waveform arrays rather than a callable.")
    return value


def save_results(result, output_dir):
    """Save only explicitly computed results; refuse to overwrite a previous run."""
    if result is None:
        return None
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    config = {
        "parameters": result["parameters"], "modes": result.get("mode_parameters"),
        "rates": result["rates"], "error_channel_convention": result["error_channel_convention"],
        "chi_convention": "Unmodified QuTiP chi from ms_gate_functions; not trace-normalized.",
    }
    (directory / "config.json").write_text(
        json.dumps(_jsonable(config), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    summary_table(result).to_csv(directory / "summary.csv", index=False)
    np.savez_compressed(
        directory / "channels.npz", tlist=result["tlist"],
        n_bar_com=result["parameters"]["n_bar_list"],
        n_bar_stretch=np.array([d.get("n_bar_stretch", np.nan) for d in result["results_list"]]),
        superoperators=np.stack([s.full() for s in result["S_qobj_list"]]),
        error_superoperators=np.stack([s.full() for s in result["S_error_qobj_list"]]),
        chi=np.stack(result["chi_matrix_list"]),
        error_chi=np.stack(result["error_chi_matrix_list"]),
        error_ptm=np.stack(result["error_ptm_list"]),
    )
    return directory
