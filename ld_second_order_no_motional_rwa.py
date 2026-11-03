"""Model C: optical RWA, Taylor expansion through eta**2, no motional RWA.

Uses the existing qubits x motion ordering, +XX gate convention, and collapse
operators. A is the existing effective first-sideband amplitude, not the
single-tone carrier Rabi frequency. All Hamiltonians returned are H / hbar.
Only a constant, balanced bichromatic drive is implemented here.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import qutip as qp

import ms_gate_functions as mg


def resolve_parameters(parameters, *, mode_frequency_hz=1e6,
                       fastest_period_steps=24, solver_atol=1e-12,
                       solver_rtol=1e-9, solver_method="vern9"):
    """Convert physical mode frequency to the existing simulation time units."""
    defaults = {
        "A": 0.125, "delta": 0.5, "rho0": 0.0, "eta": 0.1,
        "time_points": 500, "t_gate_phys": 100e-6, "t_gate_sim": None,
        "heating_rate_phys": 10.0, "dephasing_rate_phys": 18.0,
        "T2_star": 0.3, "rayleigh_rate_phys": 3.0, "raman_rate_phys": 1.0,
        "laser_intensity_fluctuation": 0.0, "laser_detuning_fluctuation": 0.0,
        "laser_rotation_angle_fluctuation": 0.0,
        "laser_noise_samples": 1, "laser_noise_seed": 1234,
        "solver_max_step": None,
    }
    unknown = set(parameters) - set(defaults)
    if unknown:
        raise ValueError(f"Unsupported model-C parameters: {sorted(unknown)}")
    p = {**defaults, **parameters}
    for name in ("A", "delta", "rho0", "eta", "t_gate_phys"):
        if np.ndim(p[name]) != 0 or not np.isfinite(p[name]):
            raise ValueError(f"{name} must be a finite scalar (rectangular drive).")
        p[name] = float(p[name])
    if p["delta"] <= 0 or p["eta"] <= 0 or p["t_gate_phys"] <= 0:
        raise ValueError("delta, eta, and t_gate_phys must be positive.")
    for name in ("laser_intensity_fluctuation", "laser_detuning_fluctuation",
                 "laser_rotation_angle_fluctuation"):
        if p[name] != 0:
            raise ValueError(f"Model C currently requires {name}=0.")
    for name in ("heating_rate_phys", "dephasing_rate_phys",
                 "rayleigh_rate_phys", "raman_rate_phys"):
        if not np.isfinite(p[name]) or p[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if not p["T2_star"] > 0:
        raise ValueError("T2_star must be positive, or infinity for no dephasing.")
    if int(p["time_points"]) != p["time_points"] or p["time_points"] < 2:
        raise ValueError("time_points must be an integer >= 2.")
    p["time_points"] = int(p["time_points"])
    if p["t_gate_sim"] is None:
        p["t_gate_sim"] = 2 * np.pi / p["delta"]
    if not np.isfinite(p["t_gate_sim"]) or p["t_gate_sim"] <= 0:
        raise ValueError("t_gate_sim must be positive and finite.")
    mode_frequency_hz = float(mode_frequency_hz)
    fastest_period_steps = float(fastest_period_steps)
    if not np.isfinite(mode_frequency_hz) or mode_frequency_hz <= 0:
        raise ValueError("mode_frequency_hz must be positive and finite.")
    if not np.isfinite(fastest_period_steps) or fastest_period_steps < 8:
        raise ValueError("fastest_period_steps must be at least 8.")
    for name, value in (("solver_atol", solver_atol), ("solver_rtol", solver_rtol)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")

    scale = p["t_gate_phys"] / p["t_gate_sim"]
    nu = 2 * np.pi * mode_frequency_hz * scale
    # The repository uses a*exp(-i*delta*t), so mu = nu - delta.
    # Using nu + delta would reverse the entangling-phase convention.
    mu = nu - p["delta"]
    if mu <= 0:
        raise ValueError("The motional frequency must exceed the sideband detuning.")
    omega0 = 2 * p["A"] / p["eta"]
    fastest = 2 * nu + mu
    step = min(2 * np.pi / (fastest_period_steps * fastest),
               p["t_gate_sim"] / (p["time_points"] - 1))
    if p["solver_max_step"] is not None:
        if not np.isfinite(p["solver_max_step"]) or p["solver_max_step"] <= 0:
            raise ValueError("solver_max_step must be positive and finite.")
        step = min(step, p["solver_max_step"])
    rates = mg._scaled_noise_rates(
        p["t_gate_phys"], p["t_gate_sim"], p["heating_rate_phys"],
        p["dephasing_rate_phys"], p["T2_star"], p["rayleigh_rate_phys"],
        p["raman_rate_phys"],
    )
    return {
        "parameters": p, "rates": rates,
        "mode_frequency_hz": mode_frequency_hz,
        "seconds_per_sim_unit": scale, "nu_sim": nu, "mu_sim": mu,
        "omega0_sim": omega0, "omega0_over_2pi_hz": omega0 / scale / (2 * np.pi),
        "delta_over_2pi_hz": p["delta"] / scale / (2 * np.pi),
        "bichromatic_offset_hz": mu / scale / (2 * np.pi),
        "fastest_angular_frequency_sim": fastest,
        "fastest_period_steps": fastest_period_steps,
        "solver_options": {
            "method": str(solver_method), "atol": float(solver_atol),
            "rtol": float(solver_rtol), "max_step": float(step),
            "nsteps": 1_000_000, "progress_bar": None, "normalize_output": False,
        },
    }


def build_hamiltonian(phonon_dim, resolved):
    """H_C/hbar = Omega0*cos(mu*t)*(S_perp*(1-eta^2*Q^2/2)+eta*S_rho*Q).

    Q = a*exp(-i*nu*t) + a.dag()*exp(i*nu*t).
    Q^2 is normal ordered BEFORE the Fock truncation, avoiding a spurious
    upper-boundary commutator term from squaring a finite oscillator matrix.
    Coefficients are analytic functions, never samples of the fast carrier.
    """
    if int(phonon_dim) != phonon_dim or phonon_dim < 2:
        raise ValueError("phonon_dim must be an integer >= 2.")
    nv = int(phonon_dim)
    p = resolved["parameters"]
    eta, rho = p["eta"], p["rho0"]
    nu, mu, omega = resolved["nu_sim"], resolved["mu_sim"], resolved["omega0_sim"]
    a = qp.destroy(nv)
    number = a.dag() * a
    identity = qp.qeye(nv)
    sx = qp.tensor(qp.sigmax(), qp.qeye(2)) + qp.tensor(qp.qeye(2), qp.sigmax())
    sy = qp.tensor(qp.sigmay(), qp.qeye(2)) + qp.tensor(qp.qeye(2), qp.sigmay())
    spin = np.cos(rho) * sx + np.sin(rho) * sy
    perpendicular = -np.sin(rho) * sx + np.cos(rho) * sy
    carrier = qp.tensor(perpendicular, identity - eta**2 * (number + 0.5 * identity))
    first = qp.tensor(spin, a)
    second = qp.tensor(perpendicular, a * a)

    def c0(t, args):
        return omega * np.cos(mu * t)

    def c1(t, args):
        return omega * eta * np.cos(mu * t) * np.exp(-1j * nu * t)

    def c1_dag(t, args):
        return np.conj(c1(t, args))

    def c2(t, args):
        return -0.5 * omega * eta**2 * np.cos(mu * t) * np.exp(-2j * nu * t)

    def c2_dag(t, args):
        return np.conj(c2(t, args))

    return qp.QobjEvo([
        [carrier, c0], [first, c1], [first.dag(), c1_dag],
        [second, c2], [second.dag(), c2_dag],
    ])


def _collapse_operators(phonon_dim, resolved):
    operators = mg._ms_gate_static_operators(phonon_dim, resolved["parameters"]["eta"], False)
    rates = resolved["rates"]
    return mg._build_c_ops(
        operators, heating_rate=rates["heating"], dephasing_rate=rates["dephasing"],
        spin_dephasing_rate=rates["spin"], rayleigh_scattering_rate=rates["rayleigh"],
        raman_scattering_rate=rates["raman"],
    )


def channel_from_unitary(unitary, nbar, phonon_dim):
    """Exactly trace the thermal motion out of a qubits x motion propagator."""
    nv = int(phonon_dim)
    tensor_u = unitary.full().reshape(4, nv, 4, nv)
    weights = np.real(qp.thermal_dm(nv, nbar).diag())
    mapped_units = np.einsum("omin,pmjn,n->opij", tensor_u, tensor_u.conj(), weights, optimize=True)
    dims = [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]
    return qp.Qobj(mapped_units.reshape(16, 16, order="F"), dims=dims, superrep="super")


def _evolve_input(task):
    resolved, nv, nbar, index = task
    hamiltonian = build_hamiltonian(nv, resolved)
    collapse = _collapse_operators(nv, resolved)
    solver = qp.MESolver(
        hamiltonian, c_ops=collapse,
        options={**resolved["solver_options"], "store_states": False, "store_final_state": True},
    )
    initial = qp.tensor(mg.build_input_states()[index], qp.thermal_dm(nv, nbar))
    # max_step resolves the optical-RWA carrier/sidebands even with just two
    # requested output times. nsteps applies to this whole interval.
    result = solver.run(initial, [0.0, resolved["parameters"]["t_gate_sim"]])
    return result.final_state.ptrace([0, 1])


def simulate_channel(parameters, *, nbar, phonon_dim, mode_frequency_hz=1e6,
                     fastest_period_steps=24, solver_atol=1e-12,
                     solver_rtol=1e-9, solver_method="vern9", parallel_workers=1):
    """Return one qubit channel; unitary partial trace or 16-input Lindblad QPT."""
    if not np.isfinite(nbar) or nbar < 0:
        raise ValueError("nbar must be finite and nonnegative.")
    if int(phonon_dim) != phonon_dim or phonon_dim < 2:
        raise ValueError("phonon_dim must be an integer >= 2.")
    if int(parallel_workers) != parallel_workers or parallel_workers < 1:
        raise ValueError("parallel_workers must be a positive integer.")
    nv = int(phonon_dim)
    resolved = resolve_parameters(
        parameters, mode_frequency_hz=mode_frequency_hz,
        fastest_period_steps=fastest_period_steps, solver_atol=solver_atol,
        solver_rtol=solver_rtol, solver_method=solver_method,
    )
    collapse = _collapse_operators(nv, resolved)
    if not collapse:
        # No dissipation: one propagator suffices for all tomography inputs.
        unitary = qp.propagator(
            build_hamiltonian(nv, resolved), resolved["parameters"]["t_gate_sim"],
            options=resolved["solver_options"],
        )
        channel = channel_from_unitary(unitary, nbar, nv)
        residual = unitary.dag() * unitary - qp.qeye(unitary.dims[0])
        diagnostics = {"unitarity_frobenius_error": float(np.linalg.norm(residual.full()))}
        method = "unitary_propagation_and_thermal_partial_trace"
    else:
        tasks = [(resolved, nv, float(nbar), i) for i in range(16)]
        if parallel_workers > 1:
            mg._configure_parallel_thread_env()
            with ProcessPoolExecutor(
                max_workers=min(int(parallel_workers), 16),
                **mg._process_pool_executor_kwargs(),
            ) as pool:
                outputs = list(pool.map(_evolve_input, tasks))
        else:
            outputs = [_evolve_input(task) for task in tasks]
        channel = mg.process_superoperator_from_states(mg.build_input_states(), outputs)
        diagnostics = {}
        method = "lindblad_16_input_qpt"
    diagnostics.update(mg.choi_physicality_metrics(channel))
    return {"channel": channel, "resolved": resolved, "method": method,
            "diagnostics": diagnostics, "phonon_dim": nv, "nbar": float(nbar)}
