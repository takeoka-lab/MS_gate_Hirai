"""Radial rocking MS gate with an optional radial COM spectator.

Kaufmann et al., arXiv:1707.03695, p. 2 supplies trap frequencies and a
100 us / 25 kHz reference. Their experiment uses a COM geometric phase gate,
not this rocking MS gate. The pulse, mode selection and LD geometry below
are explicit model choices, not a reconstruction of their pulse sequence.

Tensor order: (ion 1, ion 2, rocking[, radial COM]). Equal optical spin phases
give S_r = sigma_1 - sigma_2 and S_c = sigma_1 + sigma_2. We retain the existing
sideband RWA and relative O(eta**2) finite-LD correction. One- and two-mode
calculations use the same rocking drive and ideal gate. Nothing runs on import.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import qutip as qp

import com_stretch_model as cs
import ms_gate_functions as mg


PAPER_PARAMETERS = {
    "source": "https://arxiv.org/pdf/1707.03695#page=2",
    "axial_frequency_hz": 1.5e6,
    "radial_low_com_frequency_hz": 4.1e6,
    "radial_high_com_frequency_hz": 4.9e6,
    "reference_detuning_magnitude_hz": 25e3,
    "reference_gate_duration_s": 100e-6,
    "ramsey_coherence_time_s": 0.3,
    "radial_heating_range_quanta_per_s": [3.0, 20.0],
    "raman_wavelength_m_approx": 397e-9,
    "experimental_gate": "geometric phase gate on the higher-frequency radial COM",
}


def radial_mode_parameters(axial_frequency_hz=1.5e6, com_frequency_hz=4.9e6,
                           *, rocking_frequency_hz=None, eta_rocking=None,
                           eta_com=None, radial_projection=1.0):
    """Equal-mass two-ion harmonic well; Hz inputs, per-ion unsigned LD factors.

The default LD estimate assumes 40 u, 397 nm orthogonal Raman beams and their
effective wavevector aligned with the selected radial axis. Projection=1 is
an assumption. Paper p. 2 establishes perpendicularity to the *axial* axis,
but not this alignment with one of the two radial axes. Explicit eta values
override that estimate. The mode participation 1/sqrt(2) is included once.
"""
    fa = cs._number("axial_frequency_hz", axial_frequency_hz, positive=True)
    fc = cs._number("com_frequency_hz", com_frequency_hz, positive=True)
    if fc <= fa:
        raise ValueError("A stable linear two-ion crystal requires f_COM > f_axial.")
    fr = np.sqrt(fc**2 - fa**2) if rocking_frequency_hz is None else cs._number(
        "rocking_frequency_hz", rocking_frequency_hz, positive=True
    )
    if fr >= fc:
        raise ValueError("The radial rocking frequency must be below radial COM.")
    projection = cs._number("radial_projection", radial_projection)
    if projection > 1:
        raise ValueError("radial_projection must lie between zero and one.")
    hbar, mass = 1.054571817e-34, 40.0 * 1.66053906660e-27
    k_eff = np.sqrt(2.0) * 2 * np.pi / 397e-9 * projection
    er = (k_eff * np.sqrt(hbar / (4 * mass * 2 * np.pi * fr))
          if eta_rocking is None else cs._number("eta_rocking", eta_rocking))
    ratio = np.sqrt(fr / fc)
    ec = er * ratio if eta_com is None else cs._number("eta_com", eta_com)
    if er > 0:
        ratio = ec / er
    elif ec > 0:
        raise ValueError("Nonzero eta_com requires nonzero eta_rocking.")
    return {
        "axial_frequency_hz": fa, "rocking_frequency_hz": fr,
        "com_frequency_hz": fc, "eta_rocking": er, "eta_com": ec,
        "com_amplitude_ratio": ratio, "radial_projection": projection,
        "frequency_gap_hz": fc - fr,
        "gate_mode": "radial rocking", "spectator_mode": "radial COM",
        "detuning_convention": "delta_m = omega_m - mu",
        "eta_source": "397 nm, 90 degree beams, assumed radial projection"
                      if eta_rocking is None else "explicit eta_rocking",
    }


def two_loop_ms_pulse(detuning_hz=25e3, gate_duration_s=100e-6, delta_sim=0.5):
    """Two rectangular one-loop pulses separated by an idle interval.

Default timing is 40 us ON, 20 us OFF, 40 us ON. This is our MS adaptation,
not the paper's gate sequence. Positive delta = omega_rock - mu and S_r=X1-X2
give U = exp(-i*pi*XX/4), up to global phase, in the noiseless single-mode LD
limit. A/delta = 1/(4*sqrt(2)); finite-LD and spectator shifts are not refitted.
"""
    detuning = cs._number("detuning_hz", detuning_hz, positive=True)
    duration = cs._number("gate_duration_s", gate_duration_s, positive=True)
    delta = cs._number("delta_sim", delta_sim, positive=True)
    loop_time = 1.0 / detuning
    if duration < 2 * loop_time:
        raise ValueError("gate_duration_s must accommodate two complete loops.")
    return {
        "A": delta / (4 * np.sqrt(2.0)), "delta": delta,
        "t_gate_sim": duration * 2 * np.pi * detuning / delta,
        "t_gate_phys": duration,
        "drive_intervals_phys": [(0.0, loop_time), (duration - loop_time, duration)],
    }


def _pulse_segments(intervals, duration, scale):
    if intervals is None:
        intervals = [(0.0, duration)]
    segments, end = [], 0.0
    for start, stop in intervals:
        start, stop = float(start), float(stop)
        if not np.isfinite(start + stop) or not (end <= start < stop <= duration):
            raise ValueError("Drive intervals must be ordered, disjoint and inside the gate.")
        if start > end:
            segments.append({"start": end / scale, "stop": start / scale, "drive_on": False})
        segments.append({"start": start / scale, "stop": stop / scale, "drive_on": True})
        end = stop
    if end < duration:
        segments.append({"start": end / scale, "stop": duration / scale, "drive_on": False})
    return segments


def prepare_simulation(*, use_two_modes=True, axial_frequency_hz=1.5e6,
                       com_frequency_hz=4.9e6, rocking_frequency_hz=None,
                       eta_com=None, radial_projection=1.0, n_bar_com=0.01,
                       com_heating_rate_phys=0.0, com_dephasing_rate_phys=0.0,
                       com_phonon_dim_override=None, drive_intervals_phys=None,
                       **simulation_parameters):
    """Resolve settings and cutoff estimates without constructing a solver.

The old arguments A, delta, eta, n_bar_list, heating/dephasing_rate_phys and
phonon_dim_override now describe ROCKING. COM-only arguments are ignored in
one-mode dynamics. Neither an initial thermal tail nor this size estimate
guarantees convergence during the driven gate.
"""
    if not isinstance(use_two_modes, (bool, np.bool_)):
        raise TypeError("use_two_modes must be True or False.")
    modes = radial_mode_parameters(
        axial_frequency_hz, com_frequency_hz, rocking_frequency_hz=rocking_frequency_hz,
        eta_rocking=simulation_parameters.get("eta"),
        eta_com=eta_com if use_two_modes else None, radial_projection=radial_projection,
    )
    simulation_parameters["eta"] = modes["eta_rocking"]
    p = cs._resolved_parameters(simulation_parameters)
    times = np.linspace(0.0, p["t_gate_sim"], p["time_points"])
    amplitude = mg._as_control_values("A", p["A"], times)
    delta = mg._as_control_values("delta", p["delta"], times)
    rho = mg._as_control_values("rho0", p["rho0"], times)
    if not all(np.all(np.isfinite(v)) for v in (amplitude, delta, rho)):
        raise ValueError("Control waveforms must be finite.")
    rates = mg._scaled_noise_rates(
        p["t_gate_phys"], p["t_gate_sim"], p["heating_rate_phys"],
        p["dephasing_rate_phys"], p["T2_star"], p["rayleigh_rate_phys"], p["raman_rate_phys"],
    )
    scale = rates["scale_factor"]
    segments = _pulse_segments(drive_intervals_phys, p["t_gate_phys"], scale)
    p["drive_intervals_phys"] = [
        (s["start"] * scale, s["stop"] * scale) for s in segments if s["drive_on"]
    ]
    modes["mode_gap_sim"] = 2 * np.pi * modes["frequency_gap_hz"] * scale
    modes["delta_com_sim"] = delta + modes["mode_gap_sim"]
    modes["drive_frequency_hz"] = modes["rocking_frequency_hz"] - delta / (2 * np.pi * scale)
    com_values = [None] * len(p["n_bar_list"])
    if use_two_modes:
        com_values = np.asarray(n_bar_com, dtype=float)
        if com_values.ndim == 0:
            com_values = np.full(len(p["n_bar_list"]), float(com_values))
        if com_values.shape != (len(p["n_bar_list"]),):
            raise ValueError("n_bar_com must be scalar or match n_bar_list length.")
        com_values = [cs._number("n_bar_com", value) for value in com_values]
        if com_phonon_dim_override is not None:
            com_phonon_dim_override = cs._integer("com_phonon_dim_override", com_phonon_dim_override, 2)
        for label, value in (("heating", com_heating_rate_phys), ("dephasing", com_dephasing_rate_phys)):
            rates[f"com_{label}"] = cs._number(f"com_{label}_rate_phys", value) * scale
    else:
        rates.update(com_heating=0.0, com_dephasing=0.0)

    # Sum per-pulse bounds so interrupted/nonclosing pulses cannot silently use
    # the continuous-drive estimate. Include the spin eigenvalue bound |S|<=2.
    active_count = sum(s["drive_on"] for s in segments)
    ar = active_count * cs._alpha_bound(
        amplitude, delta, times, p["laser_intensity_fluctuation"], p["laser_detuning_fluctuation"]
    )
    ac = 0.0 if not use_two_modes else active_count * cs._alpha_bound(
        amplitude * modes["com_amplitude_ratio"], modes["delta_com_sim"], times,
        p["laser_intensity_fluctuation"], p["laser_detuning_fluctuation"],
    )
    fastest = float(np.max(np.abs(delta)))
    if use_two_modes:
        fastest = max(fastest, float(np.max(np.abs(modes["delta_com_sim"]))))
    fastest += 3 * p["laser_detuning_fluctuation"]
    max_step = float(np.min(np.diff(times)))
    if fastest > 0:
        max_step = min(max_step, 2 * np.pi / (40 * fastest))
    if p["solver_max_step"] is not None:
        max_step = min(max_step, p["solver_max_step"])
    rows = []
    for nr_mean, nc_mean in zip(p["n_bar_list"], com_values):
        nr = p["phonon_dim_override"] or mg.estimate_phonon_dim(nr_mean, 2 * ar)
        nc = None if not use_two_modes else (
            com_phonon_dim_override or mg.estimate_phonon_dim(nc_mean, 2 * ac)
        )
        rows.append({
            "n_bar": nr_mean, "Nv": nr,  # legacy QPT keys refer to the gate mode
            "n_bar_rocking": nr_mean, "n_bar_com": nc_mean,
            "Nv_rocking": nr, "Nv_com": nc,
            "hilbert_dimension": 4 * nr * (nc or 1),
            "initial_rocking_tail": (nr_mean / (1 + nr_mean))**nr,
            "initial_com_tail": None if nc is None else (nc_mean / (1 + nc_mean))**nc,
        })
    return {
        "use_two_modes": bool(use_two_modes), "parameters": p, "modes": modes,
        "rates": rates, "tlist": times, "amplitude": amplitude, "detuning": delta,
        "rho": rho, "segments": segments, "conditions": rows,
        "alpha_max_est": ar, "alpha_max_est_com": ac, "solver_max_step": max_step,
    }


def rocking_com_operators(phonon_dim_rocking, eta_rocking, *,
                          phonon_dim_com=None, eta_com=0.0, use_full_order=True):
    """Reuse the original sideband and cross-Debye-Waller operators.

Conjugate drive operators by Z on ion 2: the old first-mode sum becomes the
rocking difference, and the old second-mode difference becomes the COM sum.
The physical mode order is rocking, COM. Collapse operators are not duplicated.
"""
    nr = cs._integer("phonon_dim_rocking", phonon_dim_rocking, 2)
    if phonon_dim_com is None:
        operators = {"rocking": dict(mg._ms_gate_static_operators(nr, eta_rocking, use_full_order))}
        z2 = qp.tensor(qp.qeye(2), qp.sigmaz(), qp.qeye(nr))
    else:
        nc = cs._integer("phonon_dim_com", phonon_dim_com, 2)
        old = cs.com_stretch_operators(nr, nc, eta_rocking, eta_com, use_full_order=use_full_order)
        operators = {"rocking": old["com"], "com": old["stretch"]}
        z2 = qp.tensor(qp.qeye(2), qp.sigmaz(), qp.qeye(nr), qp.qeye(nc))
    for mode in operators.values():
        for key in ("Hx_minus", "Hx_plus", "Hy_minus", "Hy_plus"):
            mode[key] = z2 * mode[key] * z2
    return operators


def build_hamiltonian(operators, time_grid, detuning, rho, amplitude, *,
                      mode_gap_sim=0.0, com_amplitude_ratio=0.0):
    """Interpolate controls and integrated phase, not a rapidly rotating exponential.

This is exact for constant controls even on a coarse output grid; the COM
frequency gap is evaluated analytically. Pulse boundaries are handled by
separate solver intervals, without interpolation across ON/OFF steps.
"""
    times = mg._as_time_grid(time_grid)
    delta = mg._as_control_values("detuning", detuning, times)
    rho = mg._as_control_values("rho", rho, times)
    amplitude = mg._as_control_values("amplitude", amplitude, times)
    phase = qp.coefficient(mg._integrated_control_phase(delta, times), tlist=times, order=1)
    terms = []
    for name, ops in operators.items():
        gap = mode_gap_sim if name == "com" else 0.0
        ratio = com_amplitude_ratio if name == "com" else 1.0
        # Bind gap as a default so COM and rocking keep distinct frequencies.
        rotating = qp.coefficient(
            lambda t, gap=gap: np.exp(-1j * (phase(t) + gap * t)),
            function_style="pythonic",
        )
        for axis, projection in (("x", np.cos(rho)), ("y", np.sin(rho))):
            envelope = qp.coefficient(ratio * amplitude * projection, tlist=times, order=1)
            minus = envelope * rotating
            terms.extend([[ops[f"H{axis}_minus"], minus], [ops[f"H{axis}_plus"], minus.conj()]])
    return qp.QobjEvo(terms)


def build_solver(plan, condition, laser_parameters, *, drive_on=True):
    p, modes, rates = plan["parameters"], plan["modes"], plan["rates"]
    ops = rocking_com_operators(
        condition["Nv_rocking"], p["eta"], phonon_dim_com=condition["Nv_com"],
        eta_com=modes["eta_com"], use_full_order=p["use_full_order"],
    )
    amplitude = laser_parameters["int_strn"] if drive_on else np.zeros_like(plan["amplitude"])
    hamiltonian = build_hamiltonian(
        ops, plan["tlist"], laser_parameters["detuning"], laser_parameters["rho"], amplitude,
        mode_gap_sim=modes["mode_gap_sim"], com_amplitude_ratio=modes["com_amplitude_ratio"],
    )
    intensity_scale = None if drive_on else np.zeros_like(plan["amplitude"])
    if p["laser_scattering_scales_with_intensity"]:
        reference = p["scattering_reference_amplitude"]
        if reference is None:
            reference = float(np.max(np.abs(plan["amplitude"])))
        reference = cs._number("scattering_reference_amplitude", reference, positive=True)
        intensity_scale = (np.abs(amplitude) / reference)**2
    c_ops = mg._build_c_ops(
        ops["rocking"], heating_rate=rates["heating"], dephasing_rate=rates["dephasing"],
        spin_dephasing_rate=rates["spin"], rayleigh_scattering_rate=rates["rayleigh"],
        raman_scattering_rate=rates["raman"], time_grid=plan["tlist"],
        scattering_intensity_scale=intensity_scale,
    )
    if plan["use_two_modes"]:
        c_ops += mg._build_c_ops(ops["com"], heating_rate=rates["com_heating"],
                                 dephasing_rate=rates["com_dephasing"])
    sampled_delta = np.asarray(laser_parameters["detuning"], dtype=float)
    fastest = float(np.max(np.abs(sampled_delta)))
    if plan["use_two_modes"]:
        fastest = max(fastest, float(np.max(np.abs(sampled_delta + modes["mode_gap_sim"]))))
    max_step = plan["solver_max_step"]
    if fastest > 0:
        max_step = min(max_step, 2 * np.pi / (40 * fastest))
    options = mg._solver_options(store_states=False, max_step=max_step)
    longest = max(s["stop"] - s["start"] for s in plan["segments"])
    options["nsteps"] = max(10000, int(20 * longest / max_step))
    return qp.MESolver(hamiltonian, c_ops, options=options)


def _run_condition(task):
    plan, condition, samples = task
    inputs = mg.build_input_states()
    thermal = qp.thermal_dm(condition["Nv_rocking"], condition["n_bar_rocking"])
    if plan["use_two_modes"]:
        thermal = qp.tensor(thermal, qp.thermal_dm(condition["Nv_com"], condition["n_bar_com"]))
    outputs = [None] * len(inputs)
    edge_rocking = edge_com = 0.0
    for laser in samples:
        solvers = {on: build_solver(plan, condition, laser, drive_on=on)
                   for on in {s["drive_on"] for s in plan["segments"]}}
        for index, state in enumerate(inputs):
            final = qp.tensor(state, thermal)
            for segment in plan["segments"]:
                evolution = solvers[segment["drive_on"]].run(
                    final, [segment["start"], segment["stop"]], e_ops=[]
                )
                final = mg._final_state_from_result(evolution)
            reduced = final.ptrace([0, 1]) / len(samples)
            outputs[index] = reduced if outputs[index] is None else outputs[index] + reduced
            edge_rocking = max(edge_rocking, float(np.real(final.ptrace(2).diag()[-2:].sum())))
            if plan["use_two_modes"]:
                edge_com = max(edge_com, float(np.real(final.ptrace(3).diag()[-2:].sum())))
    return {
        **condition, "outputs": outputs, "sampled_laser_params": samples,
        "max_final_rocking_edge_population": edge_rocking,
        "max_final_com_edge_population": edge_com if plan["use_two_modes"] else None,
    }


def run_ms_gate_simulation(*, use_two_modes=True, **parameters):
    """Same rocking gate in both branches; True adds a radial COM spectator."""
    plan = prepare_simulation(use_two_modes=use_two_modes, **parameters)
    p = plan["parameters"]
    has_noise = any(p[key] > 0 for key in (
        "laser_intensity_fluctuation", "laser_detuning_fluctuation", "laser_rotation_angle_fluctuation"
    ))
    count = p["laser_noise_samples"] if has_noise else 1
    rng = np.random.default_rng(p["laser_noise_seed"])
    tasks = []
    for condition in plan["conditions"]:
        samples = [mg.sample_laser_parameters(
            plan["detuning"], plan["amplitude"], plan["rho"],
            intensity_fluctuation=p["laser_intensity_fluctuation"],
            detuning_fluctuation=p["laser_detuning_fluctuation"],
            rotation_angle_fluctuation=p["laser_rotation_angle_fluctuation"], rng=rng,
        ) for _ in range(count)]
        tasks.append((plan, condition, samples))
    p.update(A=plan["amplitude"], delta=plan["detuning"], rho0=plan["rho"])
    workers = min(p["parallel_workers"], len(tasks))
    executor, results = None, []
    try:
        if workers > 1:
            mg._configure_parallel_thread_env()
            executor = ProcessPoolExecutor(max_workers=workers, **mg._process_pool_executor_kwargs())
            iterator = executor.map(_run_condition, tasks)
        else:
            iterator = map(_run_condition, tasks)
        for row in iterator:
            results.append(row)
            if p["show_progress"]:
                print(f"Rocking{' + radial COM' if use_two_modes else ''}: "
                      f"{len(results)}/{len(tasks)}; n_rocking={row['n_bar_rocking']}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return {
        "parameters": {**p, "use_two_modes": bool(use_two_modes),
                       "effective_laser_samples": count, "solver_max_step": plan["solver_max_step"]},
        "mode_parameters": plan["modes"], "reference_parameters": dict(PAPER_PARAMETERS),
        "segments": plan["segments"], "rates": plan["rates"], "tlist": plan["tlist"],
        "input_states_list": mg.build_input_states(), "results_list": results,
        "alpha_max_est": plan["alpha_max_est"], "alpha_max_est_com": plan["alpha_max_est_com"],
    }


def run_analysis(*, execute=False, use_two_modes=True, ideal_phi=-np.pi / 4,
                 show_plot=False, error_channel_convention="undo_before_actual", **parameters):
    """Reuse QPT/chi/PTM routines with an explicit rocking-gate ideal target."""
    if not execute:
        return None
    simulation = run_ms_gate_simulation(use_two_modes=use_two_modes, **parameters)
    result = mg.generate_chi_matrices(simulation_result=simulation)
    ideal = mg.ideal_ms_gate(phi=ideal_phi)
    chi_ideal = qp.to_chi(qp.to_super(ideal))
    fidelities = [(4 * qp.process_fidelity(chi, chi_ideal) + 1) / 5
                  for chi in result["chi_qobj_list"]]
    result.update(ideal_unitary=ideal, ideal_phi=float(ideal_phi), chi_ideal=chi_ideal,
                  f_avg_list=fidelities, infidelity_list=[1.0 - f for f in fidelities])
    result = mg.generate_error_channel_matrices(
        channel_result=result, ideal_unitary=ideal, convention=error_channel_convention
    )
    if show_plot:
        plot_results(result)
    return result


def summary_table(result):
    import pandas as pd

    if result is None:
        return pd.DataFrame()
    rows = []
    for data, fidelity in zip(result["results_list"], result["f_avg_list"]):
        rows.append({
            "model": "rocking + radial COM" if result["parameters"]["use_two_modes"] else "rocking",
            **{key: data[key] for key in (
                "n_bar_rocking", "n_bar_com", "Nv_rocking", "Nv_com", "hilbert_dimension",
                "max_final_rocking_edge_population", "max_final_com_edge_population",
            )},
            "F_avg": fidelity, "infidelity": 1.0 - fidelity,
        })
    return pd.DataFrame(rows)


def plot_results(result, *, index=0):
    import matplotlib.pyplot as plt

    if result is None:
        return None
    table = summary_table(result)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(table["n_bar_rocking"], np.maximum(table["infidelity"], 1e-16), "o-")
    ax.set(xlabel=r"Rocking mean occupation $\bar n_r$", ylabel=r"$1-F_{avg}$",
           title=str(table["model"].iloc[0]))
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    chi_fig, _ = mg.plot_chi_matrix(result["chi_qobj_list"][index], title="Actual gate chi")
    error_fig, _ = mg.plot_chi_matrix(result["error_chi_qobj_list"][index], title="Error channel chi")
    ptm_fig, _ = mg.plot_ptm(result["error_ptm_list"][index], title="Error channel PTM")
    return {"infidelity": fig, "chi": chi_fig, "error_chi": error_fig, "error_ptm": ptm_fig}


def save_results(result, output_dir):
    if result is None:
        return None
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    config = {key: result[key] for key in (
        "parameters", "mode_parameters", "reference_parameters", "segments", "rates",
        "ideal_phi", "error_channel_convention",
    )}
    config["chi_convention"] = "Unmodified QuTiP chi; not trace-normalized."
    (directory / "config.json").write_text(
        json.dumps(cs._jsonable(config), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    summary_table(result).to_csv(directory / "summary.csv", index=False)
    np.savez_compressed(
        directory / "channels.npz", tlist=result["tlist"],
        n_bar_rocking=result["parameters"]["n_bar_list"],
        n_bar_com=np.array([d["n_bar_com"] if d["n_bar_com"] is not None else np.nan
                            for d in result["results_list"]]),
        ideal_unitary=result["ideal_unitary"].full(),
        superoperators=np.stack([s.full() for s in result["S_qobj_list"]]),
        error_superoperators=np.stack([s.full() for s in result["S_error_qobj_list"]]),
        chi=np.stack(result["chi_matrix_list"]), error_chi=np.stack(result["error_chi_matrix_list"]),
        error_ptm=np.stack(result["error_ptm_list"]),
    )
    return directory
