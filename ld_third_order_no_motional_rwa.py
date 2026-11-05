"""Model D: balanced bichromatic MS drive through absolute LD order three.

Use the same rotating frame, Rabi convention, carrier, LD² terms, collapse
operators and target as model C. Frequencies in H/hbar use simulation time.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import qutip as qp

import ld_second_order_no_motional_rwa as model_c
import ms_gate_functions as mg


def resolve_parameters(parameters, *, mode_frequency_hz=4.9e6, **numerics):
    resolved = model_c.resolve_parameters(
        parameters, mode_frequency_hz=mode_frequency_hz, **numerics,
    )
    fastest = 3 * resolved["nu_sim"] + resolved["mu_sim"]
    step = 2 * np.pi / (resolved["fastest_period_steps"] * fastest)
    options = resolved["solver_options"]
    options["max_step"] = min(options["max_step"], step)
    resolved["fastest_angular_frequency_sim"] = fastest
    return resolved


def build_hamiltonian(phonon_dim, resolved):
    """H_D/hbar = H_C/hbar - Omega0*cos(mu*t)*eta³*S_phi*Q(t)³/6."""
    lower_orders = model_c.build_hamiltonian(phonon_dim, resolved)
    nv = int(phonon_dim)
    eta, phi = (resolved["parameters"][key] for key in ("eta", "rho0"))
    nu, mu, omega = (resolved[key] for key in ("nu_sim", "mu_sim", "omega0_sim"))
    local_spin = np.cos(phi) * qp.sigmax() + np.sin(phi) * qp.sigmay()
    spin = qp.tensor(local_spin, qp.qeye(2)) + qp.tensor(qp.qeye(2), local_spin)
    a = qp.destroy(nv)
    number = a.dag() * a
    # Normal order in the infinite oscillator, THEN project to the finite basis.
    # Q³ = [3*a*n*exp(-i*nu*t) + a³*exp(-3i*nu*t)] + H.c.
    one = qp.tensor(spin, a * number)
    three = qp.tensor(spin, a * a * a)

    def c_one(t, args):
        return -omega * eta**3 / 2 * np.cos(mu*t) * np.exp(-1j*nu*t)

    def c_three(t, args):
        return -omega * eta**3 / 6 * np.cos(mu*t) * np.exp(-3j*nu*t)

    lowering = qp.QobjEvo([[one, c_one], [three, c_three]])
    return lower_orders + lowering + lowering.dag()


def _evolve_input(task):
    resolved, nv, nbar, index = task
    solver = qp.MESolver(
        build_hamiltonian(nv, resolved),
        c_ops=model_c._collapse_operators(nv, resolved),
        options={**resolved["solver_options"], "store_states": False, "store_final_state": True},
    )
    initial = qp.tensor(mg.build_input_states()[index], qp.thermal_dm(nv, nbar))
    result = solver.run(initial, [0.0, resolved["parameters"]["t_gate_sim"]])
    return result.final_state.ptrace([0, 1])


def simulate_channel(parameters, *, nbar, phonon_dim, parallel_workers=1, **numerics):
    """Compute the actual two-qubit channel with the same reduction as model C."""
    if not np.isfinite(nbar) or nbar < 0:
        raise ValueError("nbar must be finite and nonnegative.")
    if int(phonon_dim) != phonon_dim or phonon_dim < 2:
        raise ValueError("phonon_dim must be an integer >= 2.")
    if int(parallel_workers) != parallel_workers or parallel_workers < 1:
        raise ValueError("parallel_workers must be a positive integer.")
    nv = int(phonon_dim)
    resolved = resolve_parameters(parameters, **numerics)
    if not model_c._collapse_operators(nv, resolved):
        unitary = qp.propagator(
            build_hamiltonian(nv, resolved), resolved["parameters"]["t_gate_sim"],
            options=resolved["solver_options"],
        )
        return model_c.channel_from_unitary(unitary, nbar, nv)

    inputs = mg.build_input_states()
    tasks = [(resolved, nv, float(nbar), i) for i in range(len(inputs))]
    if parallel_workers == 1:
        outputs = [_evolve_input(task) for task in tasks]
    else:
        mg._configure_parallel_thread_env()
        with ProcessPoolExecutor(
            max_workers=min(int(parallel_workers), len(inputs)),
            **mg._process_pool_executor_kwargs(),
        ) as pool:
            outputs = list(pool.map(_evolve_input, tasks))
    return mg.process_superoperator_from_states(inputs, outputs)
