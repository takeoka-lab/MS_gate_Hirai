import numpy as np
import qutip as qp

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache


_PARALLEL_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _configure_parallel_thread_env():
    for env_var in _PARALLEL_THREAD_ENV_VARS:
        os.environ.setdefault(env_var, "1")


def _process_pool_executor_kwargs():
    if "fork" not in mp.get_all_start_methods():
        return {}
    return {"mp_context": mp.get_context("fork")}


def _validate_nonnegative(name, value):
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _as_time_grid(time):
    time_grid = np.asarray(time, dtype=float)

    if time_grid.ndim != 1:
        raise ValueError("time must be a 1D array.")

    if len(time_grid) < 2:
        raise ValueError("time must contain at least two points.")

    if not np.all(np.diff(time_grid) > 0):
        raise ValueError("time must be strictly increasing.")

    return time_grid


def _as_control_values(name, values, time_grid):
    """Return a real control waveform sampled on ``time_grid``.

    Controls may be scalars, one-dimensional arrays with the same length as the
    time grid, or callables evaluated as ``values(time_grid)``.  Keeping this
    conversion in one place lets the original scalar MS-gate API remain fully
    backward compatible while supporting shaped laser pulses.
    """
    if callable(values):
        values = values(time_grid)

    control = np.asarray(values, dtype=float)
    if control.ndim == 0:
        return np.full(time_grid.shape, float(control), dtype=float)
    if control.shape != time_grid.shape:
        raise ValueError(
            f"{name} must be a scalar, callable, or an array with shape "
            f"{time_grid.shape}; got {control.shape}."
        )
    if not np.all(np.isfinite(control)):
        raise ValueError(f"{name} must contain only finite values.")
    return control


def _integrated_control_phase(detuning_values, time_grid):
    """Return Phi(t) = integral detuning(s) ds using trapezoidal integration."""
    phase = np.zeros_like(time_grid, dtype=float)
    dt = np.diff(time_grid)
    phase[1:] = np.cumsum(
        0.5 * (detuning_values[1:] + detuning_values[:-1]) * dt
    )
    return phase


def _solver_options(store_states=True, max_step=None):
    options = {"progress_bar": None, "atol": 10**-12, "rtol": 10**-9}
    if max_step is not None:
        if max_step <= 0:
            raise ValueError("max_step must be positive when provided.")
        options["max_step"] = float(max_step)
    if not store_states:
        options.update({"store_final_state": True, "store_states": False})
    return options


def _final_state_from_result(result):
    if hasattr(result, "final_state") and result.final_state is not None:
        return result.final_state
    return result.states[-1]


@lru_cache(maxsize=32)
def _ms_gate_static_operators(phonon_dim, eta, use_full_order):
    a = qp.destroy(phonon_dim)
    ad = a.dag()
    n_op = ad * a

    sx_collective = (
        qp.tensor(qp.sigmax(), qp.identity(2))
        + qp.tensor(qp.identity(2), qp.sigmax())
    )
    sy_collective = (
        qp.tensor(qp.sigmay(), qp.identity(2))
        + qp.tensor(qp.identity(2), qp.sigmay())
    )

    if use_full_order:
        op_minus = a - (eta**2 / 2.0) * (n_op + qp.identity(phonon_dim)) * a
        op_plus = op_minus.dag()
    else:
        op_minus = a
        op_plus = ad

    id_qubits = qp.tensor(qp.identity(2), qp.identity(2))
    id_phonon = qp.identity(phonon_dim)

    return {
        "Hx_minus": qp.tensor(sx_collective, op_minus),
        "Hy_minus": qp.tensor(sy_collective, op_minus),
        "Hx_plus": qp.tensor(sx_collective, op_plus),
        "Hy_plus": qp.tensor(sy_collective, op_plus),
        "heating_create": qp.tensor(id_qubits, ad),
        "heating_destroy": qp.tensor(id_qubits, a),
        "motional_dephasing": qp.tensor(id_qubits, n_op),
        "spin_z_1": qp.tensor(qp.sigmaz(), qp.identity(2), id_phonon),
        "spin_z_2": qp.tensor(qp.identity(2), qp.sigmaz(), id_phonon),
        "spin_plus_1": qp.tensor(qp.sigmap(), qp.identity(2), id_phonon),
        "spin_minus_1": qp.tensor(qp.sigmam(), qp.identity(2), id_phonon),
        "spin_plus_2": qp.tensor(qp.identity(2), qp.sigmap(), id_phonon),
        "spin_minus_2": qp.tensor(qp.identity(2), qp.sigmam(), id_phonon),
    }


def _build_ms_hamiltonian(operators, time_grid, detuning, rho, effective_amplitude):
    time_grid = _as_time_grid(time_grid)
    detuning_t = _as_control_values("detuning", detuning, time_grid)
    rho_t = _as_control_values("rho", rho, time_grid)
    amplitude_t = _as_control_values(
        "effective_amplitude", effective_amplitude, time_grid
    )
    phase_t = _integrated_control_phase(detuning_t, time_grid)
    x_amp = amplitude_t * np.cos(rho_t)
    y_amp = amplitude_t * np.sin(rho_t)

    coef_x_minus = qp.coefficient(
        x_amp * np.exp(-1j * phase_t),
        tlist=time_grid,
        order=1,
    )
    coef_y_minus = qp.coefficient(
        y_amp * np.exp(-1j * phase_t),
        tlist=time_grid,
        order=1,
    )
    coef_x_plus = qp.coefficient(
        x_amp * np.exp(1j * phase_t),
        tlist=time_grid,
        order=1,
    )
    coef_y_plus = qp.coefficient(
        y_amp * np.exp(1j * phase_t),
        tlist=time_grid,
        order=1,
    )

    return (
        operators["Hx_minus"] * coef_x_minus
        + operators["Hy_minus"] * coef_y_minus
        + operators["Hx_plus"] * coef_x_plus
        + operators["Hy_plus"] * coef_y_plus
    )


def _build_c_ops(
    operators,
    heating_rate=0.0,
    dephasing_rate=0.0,
    spin_dephasing_rate=0.0,
    rayleigh_scattering_rate=0.0,
    raman_scattering_rate=0.0,
    time_grid=None,
    scattering_intensity_scale=None,
):
    c_ops = []

    if heating_rate > 0:
        c_ops.append(np.sqrt(heating_rate) * operators["heating_create"])
        c_ops.append(np.sqrt(heating_rate) * operators["heating_destroy"])

    if dephasing_rate > 0:
        c_ops.append(np.sqrt(dephasing_rate) * operators["motional_dephasing"])

    if spin_dephasing_rate > 0:
        c_ops.append(np.sqrt(spin_dephasing_rate / 2) * operators["spin_z_1"])
        c_ops.append(np.sqrt(spin_dephasing_rate / 2) * operators["spin_z_2"])

    scattering_coefficient = None
    if scattering_intensity_scale is not None:
        if time_grid is None:
            raise ValueError(
                "time_grid is required when scattering_intensity_scale is provided."
            )
        time_grid = _as_time_grid(time_grid)
        intensity_scale_t = _as_control_values(
            "scattering_intensity_scale",
            scattering_intensity_scale,
            time_grid,
        )
        if np.any(intensity_scale_t < 0):
            raise ValueError("scattering_intensity_scale must be non-negative.")
        scattering_coefficient = qp.coefficient(
            np.sqrt(intensity_scale_t),
            tlist=time_grid,
            order=1,
        )

    def scaled_scattering_operator(operator, base_rate):
        operator = np.sqrt(base_rate) * operator
        if scattering_coefficient is None:
            return operator
        return operator * scattering_coefficient

    if rayleigh_scattering_rate > 0:
        c_ops.append(
            scaled_scattering_operator(
                operators["spin_z_1"], rayleigh_scattering_rate
            )
        )
        c_ops.append(
            scaled_scattering_operator(
                operators["spin_z_2"], rayleigh_scattering_rate
            )
        )

    if raman_scattering_rate > 0:
        raman_rate_per_jump = raman_scattering_rate / 2
        c_ops.append(
            scaled_scattering_operator(
                operators["spin_plus_1"], raman_rate_per_jump
            )
        )
        c_ops.append(
            scaled_scattering_operator(
                operators["spin_minus_1"], raman_rate_per_jump
            )
        )
        c_ops.append(
            scaled_scattering_operator(
                operators["spin_plus_2"], raman_rate_per_jump
            )
        )
        c_ops.append(
            scaled_scattering_operator(
                operators["spin_minus_2"], raman_rate_per_jump
            )
        )

    return c_ops


def _prepare_ms_solver(
    phonon_dim,
    eta,
    use_full_order,
    time_grid,
    detuning,
    rho,
    effective_amplitude,
    heating_rate=0.0,
    dephasing_rate=0.0,
    spin_dephasing_rate=0.0,
    rayleigh_scattering_rate=0.0,
    raman_scattering_rate=0.0,
    scattering_intensity_scale=None,
    solver_max_step=None,
    store_states=True,
):
    operators = _ms_gate_static_operators(phonon_dim, float(eta), bool(use_full_order))
    H = _build_ms_hamiltonian(
        operators,
        time_grid=time_grid,
        detuning=detuning,
        rho=rho,
        effective_amplitude=effective_amplitude,
    )
    c_ops = _build_c_ops(
        operators,
        heating_rate=heating_rate,
        dephasing_rate=dephasing_rate,
        spin_dephasing_rate=spin_dephasing_rate,
        rayleigh_scattering_rate=rayleigh_scattering_rate,
        raman_scattering_rate=raman_scattering_rate,
        time_grid=time_grid,
        scattering_intensity_scale=scattering_intensity_scale,
    )
    return qp.MESolver(
        H,
        c_ops=c_ops,
        options=_solver_options(
            store_states=store_states,
            max_step=solver_max_step,
        ),
    )


def coherent_ms_propagator(
    A,
    delta,
    *,
    rho0=0.0,
    phonon_dim=8,
    eta=0.1,
    use_full_order=True,
    time_points=501,
    t_gate_sim=None,
    solver_method="vern9",
    solver_max_step=None,
    solver_atol=1e-11,
    solver_rtol=1e-9,
):
    """Propagate the repository's coherent MS Hamiltonian once.

    This uses exactly the Hamiltonian builder employed by ``MSGate`` and
    ``run_ms_gate_simulation``, but omits collapse operators and returns the
    qubits⊗motion propagator.  Thermal input states can therefore be applied
    afterwards without repeating the time evolution for every ``n_bar`` or
    every tomography input state.
    """

    phonon_dim = int(phonon_dim)
    time_points = int(time_points)
    if phonon_dim < 2:
        raise ValueError("phonon_dim must be at least 2")
    if time_points < 2:
        raise ValueError("time_points must be at least 2")
    if t_gate_sim is None:
        detuning_array = np.asarray(delta, dtype=float)
        if detuning_array.ndim != 0 or float(detuning_array) == 0.0:
            raise ValueError(
                "scalar nonzero delta or explicit t_gate_sim is required"
            )
        t_gate_sim = 2.0 * np.pi / abs(float(detuning_array))
    t_gate_sim = float(t_gate_sim)
    if t_gate_sim <= 0.0:
        raise ValueError("t_gate_sim must be positive")

    time_grid = np.linspace(0.0, t_gate_sim, time_points)
    operators = _ms_gate_static_operators(
        phonon_dim, float(eta), bool(use_full_order)
    )
    hamiltonian = _build_ms_hamiltonian(
        operators,
        time_grid=time_grid,
        detuning=delta,
        rho=rho0,
        effective_amplitude=A,
    )
    if solver_max_step is None:
        solver_max_step = t_gate_sim / (time_points - 1)
    options = {
        "progress_bar": None,
        "method": str(solver_method),
        "nsteps": 100000,
        "max_step": float(solver_max_step),
        "atol": float(solver_atol),
        "rtol": float(solver_rtol),
    }
    propagator = qp.propagator(
        hamiltonian,
        [0.0, t_gate_sim],
        options=options,
    )[-1]
    identity = qp.qeye(propagator.dims[0])
    return propagator, {
        "A": float(np.asarray(A)) if np.asarray(A).ndim == 0 else "waveform",
        "delta": (
            float(np.asarray(delta))
            if np.asarray(delta).ndim == 0
            else "waveform"
        ),
        "rho0": (
            float(np.asarray(rho0))
            if np.asarray(rho0).ndim == 0
            else "waveform"
        ),
        "phonon_dim": phonon_dim,
        "eta": float(eta),
        "use_full_order": bool(use_full_order),
        "time_points": time_points,
        "t_gate_sim": t_gate_sim,
        "solver_method": str(solver_method),
        "solver_max_step": float(solver_max_step),
        "solver_atol": float(solver_atol),
        "solver_rtol": float(solver_rtol),
        "unitarity_frobenius_error": float(
            np.linalg.norm((propagator.dag() * propagator - identity).full())
        ),
    }


def sample_laser_parameters(
    detuning,
    int_strn,
    rho,
    intensity_fluctuation=0.0,
    detuning_fluctuation=0.0,
    rotation_angle_fluctuation=0.0,
    rng=None,
):
    _validate_nonnegative("intensity_fluctuation", intensity_fluctuation)
    _validate_nonnegative("detuning_fluctuation", detuning_fluctuation)
    _validate_nonnegative("rotation_angle_fluctuation", rotation_angle_fluctuation)

    if rng is None:
        rng = np.random.default_rng()

    intensity_shift = rng.normal(0.0, intensity_fluctuation)
    detuning_shift = rng.normal(0.0, detuning_fluctuation)
    rotation_angle_shift = rng.normal(0.0, rotation_angle_fluctuation)

    return {
        "detuning": detuning + detuning_shift,
        "int_strn": int_strn * (1.0 + intensity_shift),
        "rho": rho + rotation_angle_shift,
        "intensity_shift": intensity_shift,
        "detuning_shift": detuning_shift,
        "rotation_angle_shift": rotation_angle_shift,
    }


def MSGate(
    Atom0,
    phonon0,
    detuning,
    int_strn,
    time,
    rho,
    heating_rate=0.0,
    dephasing_rate=0.0,
    spin_dephasing_rate=0.0,
    rayleigh_scattering_rate=0.0,
    raman_scattering_rate=0.0,
    eta=0.1,
    use_full_order=True,
    intensity_fluctuation=0.0,
    detuning_fluctuation=0.0,
    rotation_angle_fluctuation=0.0,
    int_strn_is_sideband_coupling=True,
    enable_quasi_static_noise=True,
    store_states=True,
    rng=None,
    laser_scattering_scales_with_intensity=False,
    scattering_reference_amplitude=None,
    solver_max_step=None,
):
    if rng is None:
        rng = np.random.default_rng()

    if enable_quasi_static_noise:
        laser_params = sample_laser_parameters(
            detuning,
            int_strn,
            rho,
            intensity_fluctuation=intensity_fluctuation,
            detuning_fluctuation=detuning_fluctuation,
            rotation_angle_fluctuation=rotation_angle_fluctuation,
            rng=rng,
        )
        detuning = laser_params["detuning"]
        int_strn = laser_params["int_strn"]
        rho = laser_params["rho"]

    if int_strn_is_sideband_coupling:
        effective_amplitude = int_strn
    else:
        effective_amplitude = eta * int_strn

    scattering_intensity_scale = None
    if laser_scattering_scales_with_intensity:
        amplitude_t = _as_control_values("int_strn", effective_amplitude, time)
        if scattering_reference_amplitude is None:
            scattering_reference_amplitude = float(np.max(np.abs(amplitude_t)))
        if scattering_reference_amplitude <= 0:
            raise ValueError("scattering_reference_amplitude must be positive.")
        scattering_intensity_scale = (
            np.abs(amplitude_t) / float(scattering_reference_amplitude)
        ) ** 2

    phonon_dim = phonon0.dims[0][0]
    solver = _prepare_ms_solver(
        phonon_dim=phonon_dim,
        eta=eta,
        use_full_order=use_full_order,
        time_grid=time,
        detuning=detuning,
        rho=rho,
        effective_amplitude=effective_amplitude,
        heating_rate=heating_rate,
        dephasing_rate=dephasing_rate,
        spin_dephasing_rate=spin_dephasing_rate,
        rayleigh_scattering_rate=rayleigh_scattering_rate,
        raman_scattering_rate=raman_scattering_rate,
        scattering_intensity_scale=scattering_intensity_scale,
        solver_max_step=solver_max_step,
        store_states=store_states,
    )
    result = solver.run(qp.tensor(Atom0, phonon0), time, e_ops=[])

    return result


def get_optimal_nv_general(n_bar, alpha_max, threshold=1e-7):
    if n_bar == 0:
        mean_coh = abs(alpha_max) ** 2
        cutoff = int(np.ceil(mean_coh + 8 * np.sqrt(mean_coh) + 10))
        return cutoff

    mean_val = n_bar + abs(alpha_max) ** 2
    var_val = (2 * n_bar + 1) * abs(alpha_max) ** 2 + n_bar * (n_bar + 1)
    std_dev = np.sqrt(var_val)

    k_factor = 10 + 2 * np.log(1 + n_bar)

    cutoff_candidate = int(np.ceil(mean_val + k_factor * std_dev))
    cutoff_safe = max(cutoff_candidate, int(5 * n_bar) + 20)

    return cutoff_safe


def process_superoperator_from_states(input_states, output_states):
    if len(input_states) != len(output_states):
        raise ValueError("input_states and output_states must have the same length.")

    dim = input_states[0].shape[0]
    d2 = dim**2
    A_mat = np.zeros((d2, d2), dtype=complex)
    B_mat = np.zeros((d2, d2), dtype=complex)

    for k, (input_state, output_state) in enumerate(zip(input_states, output_states)):
        vec_in = qp.operator_to_vector(input_state).full()
        vec_out = qp.operator_to_vector(output_state).full()
        A_mat[:, k] = vec_in.flatten()
        B_mat[:, k] = vec_out.flatten()

    S_matrix = B_mat @ np.linalg.inv(A_mat)
    operator_dims = input_states[0].dims
    return qp.Qobj(S_matrix, dims=[operator_dims, operator_dims])

def build_input_states():
    g = qp.basis(2, 0)
    e = qp.basis(2, 1)
    plus = (g + e).unit()
    plusi = (g + 1j * e).unit()
    basis_1q = [qp.ket2dm(g), qp.ket2dm(e), qp.ket2dm(plus), qp.ket2dm(plusi)]
    return [qp.tensor(rho1, rho2) for rho1 in basis_1q for rho2 in basis_1q]


def _scaled_noise_rates(
    t_gate_phys,
    t_gate_sim,
    heating_rate_phys,
    dephasing_rate_phys,
    T2_star,
    rayleigh_rate_phys,
    raman_rate_phys,
):
    scale_factor = t_gate_phys / t_gate_sim
    rate_magnetic = 1.0 / T2_star
    return {
        "scale_factor": scale_factor,
        "heating": heating_rate_phys * scale_factor,
        "dephasing": dephasing_rate_phys * scale_factor,
        "spin": rate_magnetic * scale_factor,
        "rayleigh": rayleigh_rate_phys * scale_factor,
        "raman": raman_rate_phys * scale_factor,
    }


def _estimate_alpha_max(A, delta, intensity_fluctuation=0.0, detuning_fluctuation=0.0):
    amplitude_max = float(np.max(np.abs(np.asarray(A, dtype=float))))
    detuning_abs = np.abs(np.asarray(delta, dtype=float))
    detuning_min = float(np.min(detuning_abs))
    detuning_floor = max(
        detuning_min - 3.0 * detuning_fluctuation,
        0.1 * detuning_min,
    )
    if detuning_floor <= 0:
        raise ValueError("detuning magnitude must be positive.")
    return 2 * amplitude_max * (1.0 + 3.0 * intensity_fluctuation) / detuning_floor


def estimate_phonon_dim(n_bar, alpha_max):
    return max(get_optimal_nv_general(n_bar, alpha_max), 2)


def _run_ms_gate_evolution_task(task):
    (
        n_bar_index,
        sample_idx,
        state_idx,
        n_bar,
        Nv,
        laser_params,
        simulation_params,
    ) = task

    tlist = simulation_params["tlist"]
    rates = simulation_params["rates"]
    input_state = build_input_states()[state_idx]
    th = qp.thermal_dm(Nv, n_bar)
    scattering_intensity_scale = None
    if simulation_params["laser_scattering_scales_with_intensity"]:
        scattering_intensity_scale = (
            np.abs(laser_params["int_strn"])
            / simulation_params["scattering_reference_amplitude"]
        ) ** 2
    solver = _prepare_ms_solver(
        phonon_dim=Nv,
        eta=simulation_params["eta"],
        use_full_order=simulation_params["use_full_order"],
        time_grid=tlist,
        detuning=laser_params["detuning"],
        rho=laser_params["rho"],
        effective_amplitude=laser_params["int_strn"],
        heating_rate=rates["heating"],
        dephasing_rate=rates["dephasing"],
        spin_dephasing_rate=rates["spin"],
        rayleigh_scattering_rate=rates["rayleigh"],
        raman_scattering_rate=rates["raman"],
        scattering_intensity_scale=scattering_intensity_scale,
        solver_max_step=simulation_params["solver_max_step"],
        store_states=False,
    )
    result = solver.run(qp.tensor(input_state, th), tlist, e_ops=[])
    final_state_total = _final_state_from_result(result)
    final_state_ion = qp.ptrace(final_state_total, (0, 1))

    return n_bar_index, sample_idx, state_idx, final_state_ion


def run_ms_gate_simulation(
    A=0.125,
    delta=0.5,
    rho0=0.0,
    n_bar_list=None,
    time_points=500,
    t_gate_phys=100e-6,
    heating_rate_phys=5.0,
    dephasing_rate_phys=18.0,
    T2_star=0.3,
    rayleigh_rate_phys=3.0,
    raman_rate_phys=1.0,
    eta=0.1,
    laser_intensity_fluctuation=0.0,
    laser_detuning_fluctuation=0.0,
    laser_rotation_angle_fluctuation=0.0,
    laser_noise_samples=1,
    laser_noise_seed=1234,
    use_full_order=True,
    show_progress=True,
    parallel_workers=30,
    t_gate_sim=None,
    laser_scattering_scales_with_intensity=False,
    scattering_reference_amplitude=None,
    solver_max_step=None,
    phonon_dim_override=None,
):
    if n_bar_list is None:
        n_bar_list = [0.01, 1, 2, 3, 4, 5]
    if laser_noise_samples < 1:
        raise ValueError("laser_noise_samples must be at least 1.")
    if parallel_workers is None:
        parallel_workers = 30
    parallel_workers = int(parallel_workers)
    if parallel_workers < 1:
        raise ValueError("parallel_workers must be at least 1.")
    if phonon_dim_override is not None:
        phonon_dim_override = int(phonon_dim_override)
        if phonon_dim_override < 2:
            raise ValueError("phonon_dim_override must be at least 2.")

    if t_gate_sim is None:
        detuning_array = np.asarray(delta, dtype=float)
        if detuning_array.ndim != 0:
            raise ValueError(
                "t_gate_sim is required when delta is a time-dependent waveform."
            )
        if float(detuning_array) == 0:
            raise ValueError("delta must be non-zero when t_gate_sim is omitted.")
        t_gate_sim = 2 * np.pi / abs(float(detuning_array))
    if t_gate_sim <= 0:
        raise ValueError("t_gate_sim must be positive.")

    tlist = np.linspace(0, float(t_gate_sim), time_points)
    amplitude_t = _as_control_values("A", A, tlist)
    detuning_t = _as_control_values("delta", delta, tlist)
    rho_t = _as_control_values("rho0", rho0, tlist)

    if laser_scattering_scales_with_intensity:
        if scattering_reference_amplitude is None:
            scattering_reference_amplitude = float(np.max(np.abs(amplitude_t)))
        scattering_reference_amplitude = float(scattering_reference_amplitude)
        if scattering_reference_amplitude <= 0:
            raise ValueError("scattering_reference_amplitude must be positive.")
    input_states_list = build_input_states()
    has_static_laser_noise = any(
        value > 0
        for value in (
            laser_intensity_fluctuation,
            laser_detuning_fluctuation,
            laser_rotation_angle_fluctuation,
        )
    )
    effective_laser_samples = laser_noise_samples if has_static_laser_noise else 1
    rng = np.random.default_rng(laser_noise_seed)

    total_evolutions = len(n_bar_list) * effective_laser_samples * len(input_states_list)
    evolutions_per_n_bar = effective_laser_samples * len(input_states_list)
    progress_bar = None
    progress_fallback = False
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress_bar = tqdm(
                total=total_evolutions,
                desc=f"MS gate simulation ({evolutions_per_n_bar} evolutions / n_bar)",
                unit="evolution",
                leave=True,
            )
        except Exception:
            progress_fallback = True
            print(
                "MS gate simulation progress: "
                f"{len(n_bar_list)} n_bar values x {effective_laser_samples} laser samples "
                f"x {len(input_states_list)} input states = {total_evolutions} evolutions "
                f"({evolutions_per_n_bar} evolutions / n_bar)"
            )

    rates = _scaled_noise_rates(
        t_gate_phys=t_gate_phys,
        t_gate_sim=tlist[-1],
        heating_rate_phys=heating_rate_phys,
        dephasing_rate_phys=dephasing_rate_phys,
        T2_star=T2_star,
        rayleigh_rate_phys=rayleigh_rate_phys,
        raman_rate_phys=raman_rate_phys,
    )
    alpha_max_est = _estimate_alpha_max(
        amplitude_t,
        detuning_t,
        intensity_fluctuation=laser_intensity_fluctuation,
        detuning_fluctuation=laser_detuning_fluctuation,
    )

    results_list = [
        {
            "n_bar": n_bar,
            "Nv": (
                phonon_dim_override
                if phonon_dim_override is not None
                else estimate_phonon_dim(n_bar, alpha_max_est)
            ),
            "outputs": [None] * len(input_states_list),
            "sampled_laser_params": [
                sample_laser_parameters(
                    detuning_t,
                    amplitude_t,
                    rho_t,
                    intensity_fluctuation=laser_intensity_fluctuation,
                    detuning_fluctuation=laser_detuning_fluctuation,
                    rotation_angle_fluctuation=laser_rotation_angle_fluctuation,
                    rng=rng,
                )
                for _ in range(effective_laser_samples)
            ],
        }
        for n_bar in n_bar_list
    ]

    try:
        if parallel_workers > 1 and total_evolutions > 1:
            _configure_parallel_thread_env()
            simulation_params = {
                "tlist": tlist,
                "rates": rates,
                "eta": eta,
                "use_full_order": use_full_order,
                "laser_scattering_scales_with_intensity": (
                    laser_scattering_scales_with_intensity
                ),
                "scattering_reference_amplitude": scattering_reference_amplitude,
                "solver_max_step": solver_max_step,
            }
            tasks = [
                (
                    n_bar_index,
                    sample_idx,
                    state_idx,
                    data["n_bar"],
                    data["Nv"],
                    laser_params,
                    simulation_params,
                )
                for n_bar_index, data in enumerate(results_list)
                for sample_idx, laser_params in enumerate(data["sampled_laser_params"])
                for state_idx in range(len(input_states_list))
            ]
            completed_by_n_bar = [0] * len(results_list)
            tasks_per_n_bar = effective_laser_samples * len(input_states_list)
            max_workers = min(parallel_workers, len(tasks))

            if progress_bar is not None:
                progress_bar.set_postfix(workers=max_workers)

            def store_task_result(task_result, executor_label):
                n_bar_index, sample_idx, state_idx, final_state_ion = task_result
                data = results_list[n_bar_index]
                weighted_output = final_state_ion / effective_laser_samples

                if data["outputs"][state_idx] is None:
                    data["outputs"][state_idx] = weighted_output
                else:
                    data["outputs"][state_idx] += weighted_output

                completed_by_n_bar[n_bar_index] += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        workers=max_workers,
                        executor=executor_label,
                        n_bar=data["n_bar"],
                        phonon_dim=data["Nv"],
                        sample=f"{sample_idx + 1}/{effective_laser_samples}",
                    )
                elif (
                    progress_fallback
                    and completed_by_n_bar[n_bar_index] == tasks_per_n_bar
                ):
                    print(
                        f"n_bar = {data['n_bar']} Simulation Finished "
                        f"(Dim: {data['Nv']})"
                    )

            def collect_parallel_results(executor, executor_label):
                if progress_bar is not None:
                    progress_bar.set_postfix(workers=max_workers, executor=executor_label)
                futures = [executor.submit(_run_ms_gate_evolution_task, task) for task in tasks]
                for future in as_completed(futures):
                    store_task_result(future.result(), executor_label)

            try:
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    **_process_pool_executor_kwargs(),
                ) as executor:
                    collect_parallel_results(executor, "process")
            except PermissionError:
                if progress_fallback:
                    print(
                        "Process-based parallelism is unavailable in this environment; "
                        "falling back to serial execution."
                    )
                for task in tasks:
                    store_task_result(_run_ms_gate_evolution_task(task), "serial")
        else:
            for data in results_list:
                n_bar = data["n_bar"]
                Nv = data["Nv"]
                th = qp.thermal_dm(Nv, n_bar)

                if progress_bar is not None:
                    progress_bar.set_postfix(
                        n_bar=n_bar,
                        phonon_dim=Nv,
                        sample=f"0/{effective_laser_samples}",
                    )

                for sample_idx, laser_params in enumerate(data["sampled_laser_params"]):
                    if progress_bar is not None:
                        progress_bar.set_postfix(
                            n_bar=n_bar,
                            phonon_dim=Nv,
                            sample=f"{sample_idx + 1}/{effective_laser_samples}",
                        )

                    scattering_intensity_scale = None
                    if laser_scattering_scales_with_intensity:
                        scattering_intensity_scale = (
                            np.abs(laser_params["int_strn"])
                            / scattering_reference_amplitude
                        ) ** 2

                    solver = _prepare_ms_solver(
                        phonon_dim=Nv,
                        eta=eta,
                        use_full_order=use_full_order,
                        time_grid=tlist,
                        detuning=laser_params["detuning"],
                        rho=laser_params["rho"],
                        effective_amplitude=laser_params["int_strn"],
                        heating_rate=rates["heating"],
                        dephasing_rate=rates["dephasing"],
                        spin_dephasing_rate=rates["spin"],
                        rayleigh_scattering_rate=rates["rayleigh"],
                        raman_scattering_rate=rates["raman"],
                        scattering_intensity_scale=scattering_intensity_scale,
                        solver_max_step=solver_max_step,
                        store_states=False,
                    )

                    for state_idx, input_state in enumerate(input_states_list):
                        result = solver.run(qp.tensor(input_state, th), tlist, e_ops=[])
                        final_state_total = _final_state_from_result(result)
                        final_state_ion = qp.ptrace(final_state_total, (0, 1))
                        weighted_output = final_state_ion / effective_laser_samples

                        if data["outputs"][state_idx] is None:
                            data["outputs"][state_idx] = weighted_output
                        else:
                            data["outputs"][state_idx] += weighted_output

                        if progress_bar is not None:
                            progress_bar.update(1)

                if progress_bar is not None:
                    progress_bar.set_postfix(
                        n_bar=n_bar,
                        phonon_dim=Nv,
                        sample=f"{effective_laser_samples}/{effective_laser_samples}",
                    )
                elif progress_fallback:
                    print(f"n_bar = {n_bar} Simulation Finished (Dim: {Nv})")
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return {
        "parameters": {
            "A": A,
            "delta": delta,
            "rho0": rho0,
            "n_bar_list": list(n_bar_list),
            "time_points": time_points,
            "t_gate_phys": t_gate_phys,
            "heating_rate_phys": heating_rate_phys,
            "dephasing_rate_phys": dephasing_rate_phys,
            "T2_star": T2_star,
            "rayleigh_rate_phys": rayleigh_rate_phys,
            "raman_rate_phys": raman_rate_phys,
            "eta": eta,
            "laser_intensity_fluctuation": laser_intensity_fluctuation,
            "laser_detuning_fluctuation": laser_detuning_fluctuation,
            "laser_rotation_angle_fluctuation": laser_rotation_angle_fluctuation,
            "laser_noise_samples": laser_noise_samples,
            "effective_laser_samples": effective_laser_samples,
            "laser_noise_seed": laser_noise_seed,
            "use_full_order": use_full_order,
            "parallel_workers": parallel_workers,
            "t_gate_sim": float(t_gate_sim),
            "laser_scattering_scales_with_intensity": (
                laser_scattering_scales_with_intensity
            ),
            "scattering_reference_amplitude": scattering_reference_amplitude,
            "solver_max_step": solver_max_step,
            "phonon_dim_override": phonon_dim_override,
        },
        "rates": rates,
        "tlist": tlist,
        "input_states_list": input_states_list,
        "results_list": results_list,
        "alpha_max_est": alpha_max_est,
    }


def generate_process_channels(simulation_result=None, **simulation_parameters):
    if simulation_result is None:
        simulation_result = run_ms_gate_simulation(**simulation_parameters)

    input_states_list = simulation_result["input_states_list"]
    S_qobj_list = [
        process_superoperator_from_states(input_states_list, data["outputs"])
        for data in simulation_result["results_list"]
    ]

    return {
        **simulation_result,
        "S_qobj_list": S_qobj_list,
    }


def generate_chi_matrices(simulation_result=None, show_summary=False, **simulation_parameters):
    if simulation_result is None or "S_qobj_list" not in simulation_result:
        simulation_result = generate_process_channels(
            simulation_result=simulation_result,
            **simulation_parameters,
        )

    chi_qobj_list = [qp.to_chi(S_qobj) for S_qobj in simulation_result["S_qobj_list"]]
    chi_matrix_list = [chi.full() for chi in chi_qobj_list]
    chi_by_n_bar = {
        data["n_bar"]: chi
        for data, chi in zip(simulation_result["results_list"], chi_qobj_list)
    }

    if show_summary:
        print("--- Chi Matrix Summary ---")
        for data, chi in zip(simulation_result["results_list"], chi_qobj_list):
            trace_val = np.real_if_close(chi.tr())
            print(f"n_bar = {data['n_bar']}: shape = {chi.shape}, trace = {trace_val}")

    return {
        **simulation_result,
        "chi_qobj_list": chi_qobj_list,
        "chi_matrix_list": chi_matrix_list,
        "chi_by_n_bar": chi_by_n_bar,
    }


def ideal_ms_gate(phi=np.pi / 4):
    XX = qp.tensor(qp.sigmax(), qp.sigmax())
    return (1j * phi * XX).expm()


def remove_ideal_gate_from_channel(
    S_qobj,
    ideal_unitary=None,
    phi=np.pi / 4,
    convention="undo_after_actual",
):
    if ideal_unitary is None:
        ideal_unitary = ideal_ms_gate(phi=phi)

    ideal_inverse_super = qp.to_super(ideal_unitary.dag())

    if convention == "undo_after_actual":
        return ideal_inverse_super * S_qobj

    if convention == "undo_before_actual":
        return S_qobj * ideal_inverse_super

    raise ValueError("convention must be 'undo_after_actual' or 'undo_before_actual'.")


def two_qubit_pauli_basis():
    one_qubit_basis = [qp.identity(2), qp.sigmax(), qp.sigmay(), qp.sigmaz()]
    return [qp.tensor(a, b) for a in one_qubit_basis for b in one_qubit_basis]


def superoperator_to_ptm(S_qobj, pauli_basis=None):
    if pauli_basis is None:
        pauli_basis = two_qubit_pauli_basis()

    dim = pauli_basis[0].shape[0]
    n_basis = len(pauli_basis)
    ptm = np.zeros((n_basis, n_basis), dtype=complex)

    for col, basis_op in enumerate(pauli_basis):
        out_vec = S_qobj * qp.operator_to_vector(basis_op)
        out_op = qp.vector_to_operator(out_vec)
        for row, measure_op in enumerate(pauli_basis):
            ptm[row, col] = (measure_op.dag() * out_op).tr() / dim

    return np.real_if_close(ptm)


def generate_error_channel_matrices(
    channel_result=None,
    ideal_unitary=None,
    phi=np.pi / 4,
    convention="undo_after_actual",
    show_summary=False,
    **simulation_parameters,
):
    if channel_result is None or "S_qobj_list" not in channel_result:
        channel_result = generate_process_channels(
            simulation_result=channel_result,
            **simulation_parameters,
        )

    S_error_qobj_list = [
        remove_ideal_gate_from_channel(
            S_qobj,
            ideal_unitary=ideal_unitary,
            phi=phi,
            convention=convention,
        )
        for S_qobj in channel_result["S_qobj_list"]
    ]
    error_chi_qobj_list = [qp.to_chi(S_error) for S_error in S_error_qobj_list]
    error_chi_matrix_list = [chi.full() for chi in error_chi_qobj_list]
    error_ptm_list = [superoperator_to_ptm(S_error) for S_error in S_error_qobj_list]

    error_chi_by_n_bar = {
        data["n_bar"]: chi
        for data, chi in zip(channel_result["results_list"], error_chi_qobj_list)
    }
    error_ptm_by_n_bar = {
        data["n_bar"]: ptm
        for data, ptm in zip(channel_result["results_list"], error_ptm_list)
    }

    if show_summary:
        print("--- Error Channel Summary ---")
        print(f"Convention: {convention}")
        for data, chi, ptm in zip(
            channel_result["results_list"],
            error_chi_qobj_list,
            error_ptm_list,
        ):
            trace_val = np.real_if_close(chi.tr())
            ptm_identity_error = np.linalg.norm(ptm - np.eye(ptm.shape[0]))
            print(
                f"n_bar = {data['n_bar']}: "
                f"error chi shape = {chi.shape}, "
                f"trace = {trace_val}, "
                f"||PTM - I|| = {ptm_identity_error:.4e}"
            )

    return {
        **channel_result,
        "S_error_qobj_list": S_error_qobj_list,
        "error_chi_qobj_list": error_chi_qobj_list,
        "error_chi_matrix_list": error_chi_matrix_list,
        "error_ptm_list": error_ptm_list,
        "error_chi_by_n_bar": error_chi_by_n_bar,
        "error_ptm_by_n_bar": error_ptm_by_n_bar,
        "error_channel_convention": convention,
    }


def pauli_labels_and_weights():
    labels = _two_qubit_pauli_labels()
    return [(label, sum(ch != "I" for ch in label)) for label in labels]


def validate_pauli_label_order(expected_prefix=None):
    import pandas as pd

    if expected_prefix is None:
        expected_prefix = ["II", "IX", "IY", "IZ", "XI"]

    labels_and_weights = pauli_labels_and_weights()
    rows = [
        {"index": index, "label": label, "weight": weight}
        for index, (label, weight) in enumerate(labels_and_weights)
    ]
    df = pd.DataFrame(rows)
    prefix_matches = df["label"].iloc[: len(expected_prefix)].tolist() == expected_prefix

    return {
        "prefix_matches": prefix_matches,
        "expected_prefix": expected_prefix,
        "actual_prefix": df["label"].iloc[: len(expected_prefix)].tolist(),
        "pauli_order_df": df,
    }


def _ideal_inverse_superoperator(phi=np.pi / 4, ideal_unitary=None):
    if ideal_unitary is None:
        ideal_unitary = ideal_ms_gate(phi=phi)
    return qp.to_super(ideal_unitary.dag())


def _superoperator_difference_norms(A, B):
    diff = (A - B).full()
    return {
        "frobenius": float(np.linalg.norm(diff)),
        "max_abs": float(np.max(np.abs(diff))),
    }


def validate_error_channel_composition(
    error_result,
    desired_convention="undo_before_actual",
    ideal_unitary=None,
    phi=np.pi / 4,
):
    import pandas as pd

    if "S_qobj_list" not in error_result or "S_error_qobj_list" not in error_result:
        raise ValueError("error_result must contain S_qobj_list and S_error_qobj_list.")

    ideal_inverse_super = _ideal_inverse_superoperator(phi=phi, ideal_unitary=ideal_unitary)
    rows = []
    for data, S_actual, S_error in zip(
        error_result["results_list"],
        error_result["S_qobj_list"],
        error_result["S_error_qobj_list"],
    ):
        expected_post = S_actual * ideal_inverse_super
        expected_pre = ideal_inverse_super * S_actual

        post_norms = _superoperator_difference_norms(S_error, expected_post)
        pre_norms = _superoperator_difference_norms(S_error, expected_pre)

        rows.append(
            {
                "n_bar": data["n_bar"],
                "stored_convention": error_result.get("error_channel_convention", None),
                "desired_convention": desired_convention,
                "post_expected_frobenius_error": post_norms["frobenius"],
                "post_expected_max_abs_error": post_norms["max_abs"],
                "pre_expected_frobenius_error": pre_norms["frobenius"],
                "pre_expected_max_abs_error": pre_norms["max_abs"],
            }
        )

    df = pd.DataFrame(rows)

    if desired_convention == "undo_before_actual":
        match_error = float(df["post_expected_frobenius_error"].max())
    elif desired_convention == "undo_after_actual":
        match_error = float(df["pre_expected_frobenius_error"].max())
    else:
        raise ValueError("desired_convention must be 'undo_before_actual' or 'undo_after_actual'.")

    return {
        "desired_convention_matches": match_error < 1e-10,
        "max_desired_convention_error": match_error,
        "composition_df": df,
    }


def _choi_trace_over_output_qutip_order(choi_matrix):
    matrix = np.asarray(choi_matrix, dtype=complex)
    d2 = matrix.shape[0]
    d = int(round(np.sqrt(d2)))
    if matrix.shape != (d2, d2) or d * d != d2:
        raise ValueError("Choi matrix dimension must be d^2 x d^2.")

    traced = np.zeros((d, d), dtype=complex)
    for in_row in range(d):
        for in_col in range(d):
            for out_index in range(d):
                traced[in_row, in_col] += matrix[
                    in_row * d + out_index,
                    in_col * d + out_index,
                ]
    return traced


def choi_physicality_metrics(S_qobj, tp_tol=1e-8, cp_tol=1e-10):
    choi = qp.to_choi(S_qobj)
    choi_matrix = choi.full()
    choi_matrix_herm = 0.5 * (choi_matrix + choi_matrix.conj().T)
    eigvals = np.linalg.eigvalsh(choi_matrix_herm)

    d = int(round(np.sqrt(choi_matrix.shape[0])))
    choi_trace = np.trace(choi_matrix)
    expected_trace = d

    tp_matrix = _choi_trace_over_output_qutip_order(choi_matrix)
    tp_residual = tp_matrix - np.eye(d)
    tp_frobenius_error = np.linalg.norm(tp_residual)
    tp_max_abs_error = np.max(np.abs(tp_residual))

    return {
        "dimension": d,
        "choi_trace": np.real_if_close(choi_trace).item(),
        "expected_choi_trace_for_tp": expected_trace,
        "choi_trace_error": float(abs(choi_trace - expected_trace)),
        "tp_frobenius_error": float(tp_frobenius_error),
        "tp_max_abs_error": float(tp_max_abs_error),
        "min_choi_eigenvalue": float(np.min(eigvals)),
        "negative_choi_eigenvalue_count": int(np.sum(eigvals < -cp_tol)),
        "tp_pass": bool(tp_frobenius_error <= tp_tol),
        "cp_pass": bool(np.min(eigvals) >= -cp_tol),
    }


def validate_channel_physicality(
    channel_result,
    channel_key="S_error_qobj_list",
    tp_tol=1e-8,
    cp_tol=1e-10,
):
    import pandas as pd

    if channel_key not in channel_result:
        raise ValueError(f"channel_result does not contain '{channel_key}'.")

    rows = []
    for data, S_qobj in zip(channel_result["results_list"], channel_result[channel_key]):
        metrics = choi_physicality_metrics(S_qobj, tp_tol=tp_tol, cp_tol=cp_tol)
        rows.append({"n_bar": data["n_bar"], **metrics})

    return pd.DataFrame(rows)


def validate_error_channel_reliability(
    error_result,
    desired_convention="undo_before_actual",
    tp_tol=1e-8,
    cp_tol=1e-10,
    show_summary=False,
):
    pauli_check = validate_pauli_label_order()
    composition_check = validate_error_channel_composition(
        error_result,
        desired_convention=desired_convention,
    )
    physicality_df = validate_channel_physicality(
        error_result,
        channel_key="S_error_qobj_list",
        tp_tol=tp_tol,
        cp_tol=cp_tol,
    )

    if show_summary:
        print("--- Reliability Checks ---")
        print(f"Pauli label prefix matches: {pauli_check['prefix_matches']}")
        print(f"Expected prefix: {pauli_check['expected_prefix']}")
        print(f"Actual prefix:   {pauli_check['actual_prefix']}")
        print(
            f"Desired convention '{desired_convention}' matches: "
            f"{composition_check['desired_convention_matches']} "
            f"(max error = {composition_check['max_desired_convention_error']:.4e})"
        )
        print("--- Choi Physicality ---")
        print(physicality_df)

    return {
        **pauli_check,
        **composition_check,
        "physicality_df": physicality_df,
    }


_NOISE_SOURCE_ALIASES = {
    "motional_heating": "motional_heating",
    "heating": "motional_heating",
    "motional heating": "motional_heating",
    "motional_dephasing": "motional_dephasing",
    "motional dephasing": "motional_dephasing",
    "spin_dephasing": "spin_dephasing",
    "spin dephasing": "spin_dephasing",
    "photon_scattering": "photon_scattering",
    "photon scattering": "photon_scattering",
    "scattering": "photon_scattering",
    "amplitude_fluctuation": "amplitude_fluctuation",
    "amplitude fluctuation": "amplitude_fluctuation",
    "amp": "amplitude_fluctuation",
    "detuning_fluctuation": "detuning_fluctuation",
    "detuning fluctuation": "detuning_fluctuation",
    "detuning": "detuning_fluctuation",
    "rotation_angle_fluctuation": "rotation_angle_fluctuation",
    "rotation angle fluctuation": "rotation_angle_fluctuation",
    "phase_fluctuation": "rotation_angle_fluctuation",
    "phase fluctuation": "rotation_angle_fluctuation",
    "rho": "rotation_angle_fluctuation",
}


_DEFAULT_PTM_DERIVATIVE_STEPS = {
    "motional_heating": 1.0,
    "motional_dephasing": 1.0,
    "spin_dephasing": 1.0,
    "photon_scattering": 1.0,
    "amplitude_fluctuation": 1e-3,
    "detuning_fluctuation": 1e-4,
    "rotation_angle_fluctuation": 1e-3,
}


def _normalize_noise_source(noise_source):
    key = str(noise_source).strip().lower()
    if key not in _NOISE_SOURCE_ALIASES:
        valid = ", ".join(sorted(set(_NOISE_SOURCE_ALIASES.values())))
        raise ValueError(f"Unknown noise_source '{noise_source}'. Valid sources: {valid}")
    return _NOISE_SOURCE_ALIASES[key]


def _zero_independent_noise_parameters(params):
    allowed_parameter_keys = {
        "A",
        "delta",
        "rho0",
        "n_bar_list",
        "time_points",
        "t_gate_phys",
        "heating_rate_phys",
        "dephasing_rate_phys",
        "T2_star",
        "rayleigh_rate_phys",
        "raman_rate_phys",
        "eta",
        "laser_intensity_fluctuation",
        "laser_detuning_fluctuation",
        "laser_rotation_angle_fluctuation",
        "laser_noise_samples",
        "laser_noise_seed",
        "use_full_order",
        "show_progress",
        "parallel_workers",
    }
    params = {
        key: value
        for key, value in dict(params or {}).items()
        if key in allowed_parameter_keys
    }
    params.update(
        {
            "heating_rate_phys": 0.0,
            "dephasing_rate_phys": 0.0,
            "T2_star": np.inf,
            "rayleigh_rate_phys": 0.0,
            "raman_rate_phys": 0.0,
            "laser_intensity_fluctuation": 0.0,
            "laser_detuning_fluctuation": 0.0,
            "laser_rotation_angle_fluctuation": 0.0,
        }
    )
    return params


def simulation_parameters_with_single_noise_source(
    base_parameters=None,
    noise_source="motional_heating",
    strength=0.0,
    photon_scattering_rayleigh_fraction=None,
):
    _validate_nonnegative("strength", strength)
    source = _normalize_noise_source(noise_source)
    params = _zero_independent_noise_parameters(base_parameters)

    if source == "motional_heating":
        params["heating_rate_phys"] = strength
    elif source == "motional_dephasing":
        params["dephasing_rate_phys"] = strength
    elif source == "spin_dephasing":
        params["T2_star"] = np.inf if strength == 0 else 1.0 / strength
    elif source == "photon_scattering":
        if photon_scattering_rayleigh_fraction is None:
            rayleigh_ref = float((base_parameters or {}).get("rayleigh_rate_phys", 0.0))
            raman_ref = float((base_parameters or {}).get("raman_rate_phys", 0.0))
            total_ref = rayleigh_ref + raman_ref
            if total_ref > 0:
                photon_scattering_rayleigh_fraction = rayleigh_ref / total_ref
            else:
                photon_scattering_rayleigh_fraction = 0.75
        params["rayleigh_rate_phys"] = strength * photon_scattering_rayleigh_fraction
        params["raman_rate_phys"] = strength * (1.0 - photon_scattering_rayleigh_fraction)
    elif source == "amplitude_fluctuation":
        params["laser_intensity_fluctuation"] = strength
    elif source == "detuning_fluctuation":
        params["laser_detuning_fluctuation"] = strength
    elif source == "rotation_angle_fluctuation":
        params["laser_rotation_angle_fluctuation"] = strength

    return params


def _ptm_derivative_from_results(result_plus, result_minus, denominator):
    ptm_derivative_list = [
        (ptm_plus - ptm_minus) / denominator
        for ptm_plus, ptm_minus in zip(
            result_plus["error_ptm_list"],
            result_minus["error_ptm_list"],
        )
    ]
    ptm_derivative_by_n_bar = {
        data["n_bar"]: ptm
        for data, ptm in zip(result_plus["results_list"], ptm_derivative_list)
    }
    derivative_norms = {
        data["n_bar"]: {
            "frobenius": float(np.linalg.norm(ptm)),
            "max_abs": float(np.max(np.abs(ptm))),
        }
        for data, ptm in zip(result_plus["results_list"], ptm_derivative_list)
    }
    return ptm_derivative_list, ptm_derivative_by_n_bar, derivative_norms


def differentiate_error_ptm_by_noise_source(
    noise_source,
    base_parameters=None,
    strength=0.0,
    step=None,
    method="auto",
    photon_scattering_rayleigh_fraction=None,
    convention="undo_after_actual",
    show_summary=False,
):
    source = _normalize_noise_source(noise_source)
    _validate_nonnegative("strength", strength)
    if step is None:
        step = _DEFAULT_PTM_DERIVATIVE_STEPS[source]
    _validate_nonnegative("step", step)
    if step == 0:
        raise ValueError("step must be positive.")

    use_central = method == "central" or (method == "auto" and strength >= step)

    if use_central:
        minus_strength = strength - step
        plus_strength = strength + step
        denominator = 2.0 * step
    elif method in {"auto", "forward"}:
        minus_strength = strength
        plus_strength = strength + step
        denominator = step
    else:
        raise ValueError("method must be 'auto', 'forward', or 'central'.")

    common_kwargs = {
        "photon_scattering_rayleigh_fraction": photon_scattering_rayleigh_fraction,
    }
    minus_params = simulation_parameters_with_single_noise_source(
        base_parameters=base_parameters,
        noise_source=source,
        strength=minus_strength,
        **common_kwargs,
    )
    plus_params = simulation_parameters_with_single_noise_source(
        base_parameters=base_parameters,
        noise_source=source,
        strength=plus_strength,
        **common_kwargs,
    )

    result_minus = generate_error_channel_matrices(
        convention=convention,
        **minus_params,
    )
    result_plus = generate_error_channel_matrices(
        convention=convention,
        **plus_params,
    )
    ptm_derivative_list, ptm_derivative_by_n_bar, derivative_norms = _ptm_derivative_from_results(
        result_plus,
        result_minus,
        denominator,
    )

    if show_summary:
        diff_method = "central" if use_central else "forward"
        print(f"--- d(error PTM)/d({source}) ---")
        print(f"method = {diff_method}, strength = {strength}, step = {step}")
        for n_bar, norms in derivative_norms.items():
            print(
                f"n_bar = {n_bar}: "
                f"||dPTM||_F = {norms['frobenius']:.4e}, "
                f"max|dPTM| = {norms['max_abs']:.4e}"
            )

    return {
        "noise_source": source,
        "strength": strength,
        "step": step,
        "method": "central" if use_central else "forward",
        "result_minus": result_minus,
        "result_plus": result_plus,
        "ptm_derivative_list": ptm_derivative_list,
        "ptm_derivative_by_n_bar": ptm_derivative_by_n_bar,
        "derivative_norms": derivative_norms,
    }


def differentiate_error_ptm_noise_sources(
    noise_sources=None,
    base_parameters=None,
    strength=0.0,
    steps=None,
    method="auto",
    photon_scattering_rayleigh_fraction=None,
    convention="undo_after_actual",
    show_summary=False,
):
    if noise_sources is None:
        noise_sources = [
            "motional_heating",
            "motional_dephasing",
            "spin_dephasing",
            "photon_scattering",
            "amplitude_fluctuation",
            "detuning_fluctuation",
            "rotation_angle_fluctuation",
        ]
    steps = dict(steps or {})
    normalized_sources = [_normalize_noise_source(noise_source) for noise_source in noise_sources]

    common_kwargs = {
        "photon_scattering_rayleigh_fraction": photon_scattering_rayleigh_fraction,
    }

    can_share_zero_reference = strength == 0 and method in {"auto", "forward"}
    shared_zero_result = None
    if can_share_zero_reference and normalized_sources:
        zero_params = simulation_parameters_with_single_noise_source(
            base_parameters=base_parameters,
            noise_source=normalized_sources[0],
            strength=0.0,
            **common_kwargs,
        )
        shared_zero_result = generate_error_channel_matrices(
            convention=convention,
            **zero_params,
        )

    results = {}
    for source in normalized_sources:
        step = steps.get(source, _DEFAULT_PTM_DERIVATIVE_STEPS[source])

        if shared_zero_result is not None:
            plus_params = simulation_parameters_with_single_noise_source(
                base_parameters=base_parameters,
                noise_source=source,
                strength=step,
                **common_kwargs,
            )
            result_plus = generate_error_channel_matrices(
                convention=convention,
                **plus_params,
            )
            ptm_derivative_list, ptm_derivative_by_n_bar, derivative_norms = (
                _ptm_derivative_from_results(result_plus, shared_zero_result, step)
            )
            result = {
                "noise_source": source,
                "strength": strength,
                "step": step,
                "method": "forward",
                "result_minus": shared_zero_result,
                "result_plus": result_plus,
                "ptm_derivative_list": ptm_derivative_list,
                "ptm_derivative_by_n_bar": ptm_derivative_by_n_bar,
                "derivative_norms": derivative_norms,
            }
            if show_summary:
                print(f"--- d(error PTM)/d({source}) ---")
                print(f"method = forward, strength = 0.0, step = {step}")
                for n_bar, norms in derivative_norms.items():
                    print(
                        f"n_bar = {n_bar}: "
                        f"||dPTM||_F = {norms['frobenius']:.4e}, "
                        f"max|dPTM| = {norms['max_abs']:.4e}"
                    )
            results[source] = result
        else:
            results[source] = differentiate_error_ptm_by_noise_source(
                source,
                base_parameters=base_parameters,
                strength=strength,
                step=step,
                method=method,
                photon_scattering_rayleigh_fraction=photon_scattering_rayleigh_fraction,
                convention=convention,
                show_summary=show_summary,
            )

    return results


def _two_qubit_pauli_labels():
    one_qubit_labels = ["I", "X", "Y", "Z"]
    return [a + b for a in one_qubit_labels for b in one_qubit_labels]


def plot_chi_matrix(
    chi,
    title=None,
    components=("real", "imag", "abs"),
    figsize=None,
    cmap_real="RdBu_r",
    cmap_abs="viridis",
    show_title=True,
    show_component_titles=True,
    show_axis_labels=True,
    tick_labelsize=8,
    component_title_fontsize=12,
    colorbar_tick_labelsize=9,
):
    import matplotlib.pyplot as plt

    chi_matrix = chi.full() if hasattr(chi, "full") else np.asarray(chi)
    labels = _two_qubit_pauli_labels() if chi_matrix.shape == (16, 16) else None

    if figsize is None:
        figsize = (5 * len(components), 4.8)

    fig, axes = plt.subplots(1, len(components), figsize=figsize, squeeze=False)
    axes = axes[0]

    real_imag_scale = max(
        np.max(np.abs(np.real(chi_matrix))),
        np.max(np.abs(np.imag(chi_matrix))),
        1e-16,
    )

    component_map = {
        "real": (np.real(chi_matrix), "Re(chi)", cmap_real, -real_imag_scale, real_imag_scale),
        "imag": (np.imag(chi_matrix), "Im(chi)", cmap_real, -real_imag_scale, real_imag_scale),
        "abs": (np.abs(chi_matrix), "|chi|", cmap_abs, 0.0, None),
    }

    for ax, component in zip(axes, components):
        data, component_title, cmap, vmin, vmax = component_map[component]
        im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        if show_component_titles:
            ax.set_title(component_title, fontsize=component_title_fontsize)
        if show_axis_labels:
            ax.set_xlabel("Input Pauli basis")
            ax.set_ylabel("Output Pauli basis")

        if labels is not None:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=tick_labelsize)
            ax.set_yticklabels(labels, fontsize=tick_labelsize)

        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=colorbar_tick_labelsize)

    if show_title and title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes


def plot_ptm(
    ptm,
    title=None,
    figsize=(6, 5),
    cmap="RdBu_r",
    show_title=True,
    show_axis_labels=True,
    tick_labelsize=8,
    title_fontsize=12,
    colorbar_tick_labelsize=9,
):
    import matplotlib.pyplot as plt

    ptm_matrix = np.asarray(ptm)
    labels = _two_qubit_pauli_labels() if ptm_matrix.shape == (16, 16) else None
    scale = max(np.max(np.abs(np.real(ptm_matrix))), 1e-16)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(np.real(ptm_matrix), origin="lower", cmap=cmap, vmin=-scale, vmax=scale)
    if show_title:
        ax.set_title("PTM" if title is None else title, fontsize=title_fontsize)
    if show_axis_labels:
        ax.set_xlabel("Input Pauli basis")
        ax.set_ylabel("Output Pauli basis")

    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=tick_labelsize)
        ax.set_yticklabels(labels, fontsize=tick_labelsize)

    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=colorbar_tick_labelsize)
    fig.tight_layout()
    return fig, ax


def ptm_nbar_derivative(R_list, nbar_values, nbar_index, method="central"):
    nbar_values = np.asarray(nbar_values, dtype=float)
    if len(R_list) != len(nbar_values):
        raise ValueError("len(R_list) must match len(nbar_values).")
    if len(R_list) < 2:
        raise ValueError("At least two PTMs are required.")
    if not (0 <= nbar_index < len(R_list)):
        raise ValueError("nbar_index is out of range.")

    if method == "central":
        if nbar_index == 0 or nbar_index == len(R_list) - 1:
            raise ValueError("central difference requires 0 < nbar_index < len(R_list)-1.")
        return (
            np.asarray(R_list[nbar_index + 1]) - np.asarray(R_list[nbar_index - 1])
        ) / (nbar_values[nbar_index + 1] - nbar_values[nbar_index - 1])

    if method == "auto":
        if nbar_index == 0:
            return (
                np.asarray(R_list[1]) - np.asarray(R_list[0])
            ) / (nbar_values[1] - nbar_values[0])
        if nbar_index == len(R_list) - 1:
            return (
                np.asarray(R_list[-1]) - np.asarray(R_list[-2])
            ) / (nbar_values[-1] - nbar_values[-2])
        return (
            np.asarray(R_list[nbar_index + 1]) - np.asarray(R_list[nbar_index - 1])
        ) / (nbar_values[nbar_index + 1] - nbar_values[nbar_index - 1])

    raise ValueError("method must be 'central' or 'auto'.")


def top_ptm_derivative_components(
    D,
    pauli_labels=None,
    top_k=10,
    exclude_identity=True,
):
    import pandas as pd

    if pauli_labels is None:
        pauli_labels = _two_qubit_pauli_labels()

    D = np.asarray(D)
    rows = []
    for a, out_label in enumerate(pauli_labels):
        for b, in_label in enumerate(pauli_labels):
            if exclude_identity and out_label == "II" and in_label == "II":
                continue
            value = D[a, b]
            rows.append(
                {
                    "output": out_label,
                    "input": in_label,
                    "value": value,
                    "real_value": float(np.real(value)),
                    "imag_value": float(np.imag(value)),
                    "abs_value": float(abs(value)),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values("abs_value", ascending=False).head(top_k).reset_index(drop=True)


def analyze_top_ptm_nbar_derivative_components(
    error_result,
    nbar_index=None,
    top_k=10,
    exclude_identity=True,
    method="central",
    output_dir=None,
):
    nbar_values = np.asarray(error_result["parameters"]["n_bar_list"], dtype=float)
    R_list = error_result["error_ptm_list"]

    if nbar_index is None:
        if len(R_list) < 3:
            nbar_index = 0
            method = "auto"
        else:
            nbar_index = len(R_list) // 2

    D = ptm_nbar_derivative(R_list, nbar_values, nbar_index=nbar_index, method=method)
    top_df = top_ptm_derivative_components(
        D,
        pauli_labels=_two_qubit_pauli_labels(),
        top_k=top_k,
        exclude_identity=exclude_identity,
    )
    top_df.insert(0, "nbar_index", nbar_index)
    top_df.insert(1, "nbar", nbar_values[nbar_index])
    top_df.insert(2, "difference_method", method)

    if output_dir is not None:
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        top_df.to_csv(output_path / "top_ptm_nbar_derivative_components.csv", index=False)

    return {
        "nbar_values": nbar_values,
        "nbar_index": nbar_index,
        "nbar": nbar_values[nbar_index],
        "D": D,
        "top_df": top_df,
    }


CHI_PAULI_LABELS_2Q = _two_qubit_pauli_labels()


def _as_complex_array(value):
    return np.asarray(value.full() if hasattr(value, "full") else value, dtype=complex)


def _safe_ratio(numerator, denominator, eps=1e-15):
    if abs(denominator) < eps:
        return np.nan
    return numerator / denominator


def _trace_normalize_chi(chi, eps=1e-15):
    chi = _as_complex_array(chi)
    trace = np.trace(chi)
    if abs(trace) < eps:
        raise ValueError("chi trace is too close to zero for trace normalization.")
    return chi / trace


def _offdiag_part(matrix):
    return matrix - np.diag(np.diag(matrix))


def _pauli_weight(label):
    return sum(char != "I" for char in label)


def summarize_chi_nbar_dependence(
    nbar_values,
    chi_list,
    output_dir="chi_scalar_summary",
    pauli_labels=None,
):
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    if pauli_labels is None:
        pauli_labels = CHI_PAULI_LABELS_2Q

    nbar_values = np.asarray(nbar_values, dtype=float)
    if len(nbar_values) != len(chi_list):
        raise ValueError("len(nbar_values) must match len(chi_list).")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chi_normed_list = [_trace_normalize_chi(chi) for chi in chi_list]
    baseline_chi = chi_normed_list[0]
    baseline_norm = np.linalg.norm(baseline_chi, ord="fro")

    rows = []
    for nbar, chi in zip(nbar_values, chi_normed_list):
        chi_trace = np.real_if_close(np.trace(chi)).item()
        chi_norm = np.linalg.norm(chi, ord="fro")
        offdiag_ratio = _safe_ratio(np.linalg.norm(_offdiag_part(chi), ord="fro"), chi_norm)

        error_block = chi[1:, 1:]
        error_block_norm = np.linalg.norm(error_block, ord="fro")
        error_offdiag_ratio = _safe_ratio(
            np.linalg.norm(_offdiag_part(error_block), ord="fro"),
            error_block_norm,
        )

        identity_coupling_norm = np.sqrt(
            np.linalg.norm(chi[0, 1:]) ** 2 + np.linalg.norm(chi[1:, 0]) ** 2
        )
        identity_error_coupling_ratio = _safe_ratio(identity_coupling_norm, chi_norm)

        diag_real = np.real(np.diag(chi))
        diag_sum = np.sum(diag_real)
        if abs(diag_sum) < 1e-15:
            diag_prob = np.full_like(diag_real, np.nan, dtype=float)
        else:
            diag_prob = diag_real / diag_sum

        p_ii = diag_prob[0]
        pauli_error_probability = 1.0 - p_ii
        pauli_weight_1_probability = float(
            np.nansum(
                [
                    p
                    for p, label in zip(diag_prob, pauli_labels)
                    if _pauli_weight(label) == 1
                ]
            )
        )
        pauli_weight_2_probability = float(
            np.nansum(
                [
                    p
                    for p, label in zip(diag_prob, pauli_labels)
                    if _pauli_weight(label) == 2
                ]
            )
        )

        chi_distance_from_baseline = _safe_ratio(
            np.linalg.norm(chi - baseline_chi, ord="fro"),
            baseline_norm,
        )
        min_diag_probability = float(np.nanmin(diag_prob))

        rows.append(
            {
                "nbar": nbar,
                "chi_trace": chi_trace,
                "chi_norm": chi_norm,
                "offdiag_ratio": offdiag_ratio,
                "error_offdiag_ratio": error_offdiag_ratio,
                "identity_error_coupling_ratio": identity_error_coupling_ratio,
                "pauli_error_probability": pauli_error_probability,
                "pauli_weight_1_probability": pauli_weight_1_probability,
                "pauli_weight_2_probability": pauli_weight_2_probability,
                "chi_distance_from_baseline": chi_distance_from_baseline,
                "min_diag_probability": min_diag_probability,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path / "chi_scalar_summary.csv", index=False)

    figures = {}
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    coherence_labels = {
        "offdiag_ratio": r"$C_{\chi}$",
        "error_offdiag_ratio": r"$C_{\tilde{\chi}}$",
        "identity_error_coupling_ratio": r"$C_{\chi - \tilde{\chi}}$",
    }
    for column in ["offdiag_ratio", "error_offdiag_ratio", "identity_error_coupling_ratio"]:
        ax1.plot(summary_df["nbar"], summary_df[column], marker="o", label=coherence_labels[column])
    ax1.set_xlabel(r"Mean phonon number $\bar{n}_0$")
    ax1.set_ylabel("ratio")
    ax1.grid(True, alpha=0.4)
    ax1.legend()
    fig1.tight_layout()
    figures["chi_coherence_ratios"] = fig1

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for column in [
        "pauli_error_probability",
        "pauli_weight_1_probability",
        "pauli_weight_2_probability",
    ]:
        ax2.semilogy(summary_df["nbar"], summary_df[column], marker="o", label=column)
    ax2.set_xlabel(r"Mean phonon number $\bar{n}_0$")
    ax2.set_ylabel("probability")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.4)
    ax2.legend()
    fig2.tight_layout()
    figures["pauli_error_probabilities"] = fig2

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    ax3.plot(summary_df["nbar"], summary_df["chi_distance_from_baseline"], marker="o")
    ax3.set_xlabel(r"Mean phonon number $\bar{n}_0$")
    ax3.set_ylabel("chi distance from baseline")
    ax3.grid(True, alpha=0.4)
    fig3.tight_layout()
    figures["chi_distance_from_baseline"] = fig3

    for name, fig in figures.items():
        fig.savefig(output_path / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_path / f"{name}.pdf", bbox_inches="tight")

    return summary_df, figures


def top_chi_nbar_derivative_components(
    nbar_values,
    chi_list,
    nbar_index,
    top_k=20,
    pauli_labels=None,
):
    import pandas as pd

    if pauli_labels is None:
        pauli_labels = CHI_PAULI_LABELS_2Q

    nbar_values = np.asarray(nbar_values, dtype=float)
    chi_normed_list = [_trace_normalize_chi(chi) for chi in chi_list]
    n_chi = len(chi_normed_list)

    if n_chi != len(nbar_values):
        raise ValueError("len(nbar_values) must match len(chi_list).")
    if n_chi < 2:
        raise ValueError("At least two chi matrices are required for finite difference.")
    if not (0 <= nbar_index < n_chi):
        raise ValueError("nbar_index is out of range.")

    if nbar_index == 0:
        dchi = (chi_normed_list[1] - chi_normed_list[0]) / (
            nbar_values[1] - nbar_values[0]
        )
    elif nbar_index == n_chi - 1:
        dchi = (chi_normed_list[-1] - chi_normed_list[-2]) / (
            nbar_values[-1] - nbar_values[-2]
        )
    else:
        dchi = (chi_normed_list[nbar_index + 1] - chi_normed_list[nbar_index - 1]) / (
            nbar_values[nbar_index + 1] - nbar_values[nbar_index - 1]
        )

    flat_order = np.argsort(np.abs(dchi).ravel())[::-1][:top_k]
    rows = []
    for flat_idx in flat_order:
        row_index, col_index = np.unravel_index(flat_idx, dchi.shape)
        value = dchi[row_index, col_index]
        rows.append(
            {
                "m": int(row_index),
                "n": int(col_index),
                "row_label": pauli_labels[row_index]
                if row_index < len(pauli_labels)
                else str(row_index),
                "col_label": pauli_labels[col_index]
                if col_index < len(pauli_labels)
                else str(col_index),
                "abs_derivative": float(abs(value)),
                "real_derivative": float(np.real(value)),
                "imag_derivative": float(np.imag(value)),
            }
        )

    return pd.DataFrame(rows)


def run_error_chi_scalar_summary(
    error_result,
    output_dir="chi_scalar_summary",
    top_derivative_index=0,
    top_k=20,
):
    import pandas as pd
    from pathlib import Path

    nbar_values = np.asarray(error_result["parameters"]["n_bar_list"], dtype=float)
    chi_list = error_result["error_chi_matrix_list"]
    summary_df, figures = summarize_chi_nbar_dependence(
        nbar_values=nbar_values,
        chi_list=chi_list,
        output_dir=output_dir,
    )
    if len(nbar_values) >= 2:
        derivative_top_df = top_chi_nbar_derivative_components(
            nbar_values=nbar_values,
            chi_list=chi_list,
            nbar_index=top_derivative_index,
            top_k=top_k,
        )
    else:
        derivative_top_df = pd.DataFrame(
            columns=[
                "m",
                "n",
                "row_label",
                "col_label",
                "abs_derivative",
                "real_derivative",
                "imag_derivative",
            ]
        )
    derivative_top_df.to_csv(
        Path(output_dir) / "top_chi_nbar_derivative_components.csv",
        index=False,
    )
    return summary_df, figures, derivative_top_df


def plot_chi_offdiagonal_presentation_summary(
    error_result,
    output_dir="presentation_chi_offdiag",
    trace_normalize=True,
    make_backup_figures=True,
):
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nbar_values = np.asarray(error_result["parameters"]["n_bar_list"], dtype=float)
    chi_list = error_result["error_chi_matrix_list"]

    rows = []
    for nbar, chi in zip(nbar_values, chi_list):
        chi = _as_complex_array(chi)
        raw_trace = np.trace(chi)
        if trace_normalize:
            chi = _trace_normalize_chi(chi)

        offdiag = _offdiag_part(chi)
        identity_to_error = chi[0, 1:].ravel()
        error_to_identity = chi[1:, 0].ravel()
        identity_error = np.concatenate([identity_to_error, error_to_identity])
        error_error_offdiag = _offdiag_part(chi[1:, 1:])

        D_off_chi = float(np.linalg.norm(offdiag, ord="fro"))
        D_II_P_chi = float(np.sqrt(np.sum(np.abs(identity_to_error) ** 2)))
        D_P_II_chi = float(np.sqrt(np.sum(np.abs(error_to_identity) ** 2)))
        D_IE_chi = float(np.sqrt(np.sum(np.abs(identity_error) ** 2)))
        D_EE_chi = float(np.linalg.norm(error_error_offdiag, ord="fro"))

        D_off_sq = D_off_chi**2
        rows.append(
            {
                "nbar": nbar,
                "trace_normalized": trace_normalize,
                "raw_chi_trace": np.real_if_close(raw_trace).item(),
                "D_off_chi": D_off_chi,
                "D_II_P_chi": D_II_P_chi,
                "D_P_II_chi": D_P_II_chi,
                "D_IE_chi": D_IE_chi,
                "D_EE_chi": D_EE_chi,
                "IE_fraction": _safe_ratio(D_IE_chi**2, D_off_sq),
                "EE_fraction": _safe_ratio(D_EE_chi**2, D_off_sq),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path / "chi_offdiagonal_presentation_summary.csv", index=False)

    figures = {}

    fig_main, ax_main = plt.subplots(figsize=(7, 4.5))
    ax_main.plot(
        summary_df["nbar"],
        summary_df["D_IE_chi"],
        marker="o",
        linestyle="-",
        color="black",
        linewidth=2.6,
        label=r"$D_{\mathrm{IE}}=\sqrt{\|\chi_{\mathrm{II},P}\|_2^2+\|\chi_{P,\mathrm{II}}\|_2^2}$",
    )
    ax_main.set_xlabel(r"Mean phonon number $\bar{n}_0$")
    ax_main.set_ylabel("Frobenius norm")
    ax_main.grid(True, alpha=0.4)
    ax_main.legend(fontsize=11)
    fig_main.tight_layout()
    fig_main.savefig(output_path / "fig_chi_offdiag_main.png", dpi=300, bbox_inches="tight")
    fig_main.savefig(output_path / "fig_chi_offdiag_main.pdf", bbox_inches="tight")
    fig_main.savefig(output_path / "fig_chi_identity_error_blocks_main.png", dpi=300, bbox_inches="tight")
    fig_main.savefig(output_path / "fig_chi_identity_error_blocks_main.pdf", bbox_inches="tight")
    figures["main"] = fig_main

    if not make_backup_figures:
        return summary_df, figures

    fig_backup, ax_backup = plt.subplots(figsize=(7, 4.5))
    ax_backup.plot(
        summary_df["nbar"],
        summary_df["D_EE_chi"],
        marker="o",
        color="tab:green",
        label=r"$D_{\mathrm{EE}}(\chi)$",
    )
    ax_backup.set_xlabel(r"Mean phonon number $\bar{n}_0$")
    ax_backup.set_ylabel("Frobenius norm")
    if trace_normalize:
        ax_backup.set_title(r"Trace-normalized error $\chi$")
    ax_backup.grid(True, alpha=0.4)
    ax_backup.legend()
    fig_backup.tight_layout()
    fig_backup.savefig(output_path / "fig_chi_offdiag_EE_backup.png", dpi=300, bbox_inches="tight")
    fig_backup.savefig(output_path / "fig_chi_offdiag_EE_backup.pdf", bbox_inches="tight")
    figures["EE_backup"] = fig_backup

    fig_fraction, ax_fraction = plt.subplots(figsize=(7, 4.5))
    ax_fraction.plot(
        summary_df["nbar"],
        summary_df["IE_fraction"],
        marker="o",
        label=r"$D_{\mathrm{IE}}^2 / D_{\mathrm{off}}^2$",
    )
    ax_fraction.plot(
        summary_df["nbar"],
        summary_df["EE_fraction"],
        marker="o",
        label=r"$D_{\mathrm{EE}}^2 / D_{\mathrm{off}}^2$",
    )
    ax_fraction.set_xlabel(r"Mean phonon number $\bar{n}_0$")
    ax_fraction.set_ylabel("squared norm fraction")
    if trace_normalize:
        ax_fraction.set_title(r"Trace-normalized error $\chi$")
    ax_fraction.set_ylim(-0.02, 1.02)
    ax_fraction.grid(True, alpha=0.4)
    ax_fraction.legend()
    fig_fraction.tight_layout()
    fig_fraction.savefig(output_path / "fig_chi_offdiag_fraction.png", dpi=300, bbox_inches="tight")
    fig_fraction.savefig(output_path / "fig_chi_offdiag_fraction.pdf", bbox_inches="tight")
    figures["fraction"] = fig_fraction

    return summary_df, figures


def load_temperature_results(error_result, pauli_labels=None):
    if pauli_labels is None:
        pauli_labels = CHI_PAULI_LABELS_2Q

    if "error_ptm_by_n_bar" in error_result:
        preferred_nbars = error_result.get("parameters", {}).get("n_bar_list", None)
        R_items = _ordered_mapping_items(error_result["error_ptm_by_n_bar"], preferred_nbars)
    else:
        R_items = [
            (data["n_bar"], R_post)
            for data, R_post in zip(error_result["results_list"], error_result["error_ptm_list"])
        ]

    chi_by_nbar = dict(error_result.get("error_chi_by_n_bar", {}))
    if not chi_by_nbar:
        chi_by_nbar = {
            data["n_bar"]: chi
            for data, chi in zip(error_result["results_list"], error_result["error_chi_matrix_list"])
        }

    points = []
    for nbar, R_post in R_items:
        chi = _lookup_mapping_by_float_key(chi_by_nbar, nbar)
        if chi is None:
            raise ValueError(f"chi for nbar={nbar} was not found.")
        points.append(
            {
                "nbar": float(nbar),
                "R_post": _as_complex_array(R_post),
                "chi": _as_complex_array(chi),
                "pauli_labels": pauli_labels,
            }
        )
    return sorted(points, key=lambda item: item["nbar"])


def _extract_channel_point_from_result(channel_result, nbar=None):
    if "error_ptm_by_n_bar" in channel_result:
        R_mapping = channel_result["error_ptm_by_n_bar"]
        chi_mapping = channel_result["error_chi_by_n_bar"]
        if nbar is None:
            nbar = next(iter(R_mapping.keys()))
        R_post = _lookup_mapping_by_float_key(R_mapping, nbar)
        chi = _lookup_mapping_by_float_key(chi_mapping, nbar)
    else:
        nbars = [data["n_bar"] for data in channel_result["results_list"]]
        if nbar is None:
            index = 0
        else:
            index = int(np.argmin(np.abs(np.asarray(nbars, dtype=float) - float(nbar))))
        R_post = channel_result["error_ptm_list"][index]
        chi = channel_result["error_chi_matrix_list"][index]
        nbar = nbars[index]

    if R_post is None or chi is None:
        raise ValueError(f"R_post/chi for nbar={nbar} was not found.")
    return float(nbar), _as_complex_array(R_post), _as_complex_array(chi)


def _normalize_custom_noise_source_points(source_data, selected_nbar=None):
    if isinstance(source_data, dict) and "points" in source_data:
        source_data = source_data["points"]

    if isinstance(source_data, list):
        points = []
        for index, point in enumerate(source_data):
            if selected_nbar is not None and "nbar" in point:
                if not np.isclose(float(point["nbar"]), float(selected_nbar)):
                    continue
            R_post = point.get("R_post", point.get("R", point.get("error_ptm")))
            chi = point.get("chi", point.get("error_chi"))
            if R_post is None or chi is None:
                raise ValueError("Each noise source point must contain R_post and chi.")
            points.append(
                {
                    "s": float(point.get("s", index)),
                    "strength": point.get("strength", np.nan),
                    "nbar": float(point.get("nbar", selected_nbar if selected_nbar is not None else np.nan)),
                    "R_post": _as_complex_array(R_post),
                    "chi": _as_complex_array(chi),
                }
            )
        return sorted(points, key=lambda item: item["s"])

    if isinstance(source_data, dict):
        s_values = source_data.get("s_values", source_data.get("s_list", None))
        R_list = source_data.get("R_post_list", source_data.get("error_ptm_list", None))
        chi_list = source_data.get("chi_list", source_data.get("error_chi_matrix_list", None))
        if s_values is None or R_list is None or chi_list is None:
            raise ValueError(
                "TODO: update load_noise_source_results() for this noise_source_results format."
            )
        return [
            {
                "s": float(s),
                "strength": np.nan,
                "nbar": float(selected_nbar) if selected_nbar is not None else np.nan,
                "R_post": _as_complex_array(R_post),
                "chi": _as_complex_array(chi),
            }
            for s, R_post, chi in sorted(zip(s_values, R_list, chi_list), key=lambda item: item[0])
        ]

    raise ValueError("TODO: update load_noise_source_results() for this data format.")


def load_noise_source_results(
    noise_source_results=None,
    ptm_derivative_results=None,
    selected_nbar=None,
):
    if noise_source_results is not None:
        return {
            source: _normalize_custom_noise_source_points(source_data, selected_nbar=selected_nbar)
            for source, source_data in noise_source_results.items()
        }

    if ptm_derivative_results is not None:
        loaded = {}
        for source, result in ptm_derivative_results.items():
            minus_nbar, R_minus, chi_minus = _extract_channel_point_from_result(
                result["result_minus"],
                nbar=selected_nbar,
            )
            plus_nbar, R_plus, chi_plus = _extract_channel_point_from_result(
                result["result_plus"],
                nbar=selected_nbar,
            )
            loaded[source] = [
                {
                    "s": 0.0,
                    "strength": result.get("strength", 0.0),
                    "nbar": minus_nbar,
                    "R_post": R_minus,
                    "chi": chi_minus,
                },
                {
                    "s": 1.0,
                    "strength": result.get("step", np.nan),
                    "nbar": plus_nbar,
                    "R_post": R_plus,
                    "chi": chi_plus,
                },
            ]
        return loaded

    raise ValueError(
        "Noise-source sweep data were not found. TODO: provide noise_source_sweep_results "
        "as {source: [{'s': ..., 'R_post': ..., 'chi': ...}, ...]} or run the existing "
        "ptm_derivative_results cell and pass it as a two-point sweep."
    )


def nominal_noise_source_strengths(base_parameters):
    params = dict(base_parameters or {})

    T2_star = float(params.get("T2_star", np.inf))
    spin_dephasing = 0.0 if not np.isfinite(T2_star) or T2_star <= 0 else 1.0 / T2_star

    return {
        "motional_heating": float(params.get("heating_rate_phys", 0.0)),
        "motional_dephasing": float(params.get("dephasing_rate_phys", 0.0)),
        "spin_dephasing": spin_dephasing,
        "photon_scattering": float(params.get("rayleigh_rate_phys", 0.0))
        + float(params.get("raman_rate_phys", 0.0)),
        "amplitude_fluctuation": float(params.get("laser_intensity_fluctuation", 0.0)),
        "detuning_fluctuation": float(params.get("laser_detuning_fluctuation", 0.0)),
        "rotation_angle_fluctuation": float(
            params.get("laser_rotation_angle_fluctuation", 0.0)
        ),
    }


def active_noise_sources_from_nominal_strengths(
    base_parameters,
    noise_sources=None,
    zero_tol=0.0,
):
    nominal_strengths = nominal_noise_source_strengths(base_parameters)
    if noise_sources is None:
        noise_sources = list(nominal_strengths.keys())

    active_sources = []
    skipped_sources = {}
    for source in noise_sources:
        normalized_source = _normalize_noise_source(source)
        nominal_strength = nominal_strengths.get(normalized_source, 0.0)
        if np.isfinite(nominal_strength) and abs(nominal_strength) > zero_tol:
            active_sources.append(normalized_source)
        else:
            skipped_sources[normalized_source] = nominal_strength

    return active_sources, skipped_sources, nominal_strengths


def run_noise_source_strength_sweep(
    base_parameters,
    noise_sources=None,
    s_values=(0.0, 0.5, 1.0, 2.0, 4.0),
    selected_nbar=None,
    zero_tol=0.0,
    convention="undo_before_actual",
    show_summary=True,
):
    base_parameters = dict(base_parameters or {})
    active_sources, skipped_sources, nominal_strengths = active_noise_sources_from_nominal_strengths(
        base_parameters,
        noise_sources=noise_sources,
        zero_tol=zero_tol,
    )

    if selected_nbar is None:
        selected_nbar = list(base_parameters.get("n_bar_list", [0.01]))[0]

    if show_summary:
        print("--- Noise Source Strength Sweep ---")
        print(f"s values: {list(s_values)}")
        print(f"selected nbar: {selected_nbar}")
        print(f"active sources: {active_sources}")
        if skipped_sources:
            print(f"skipped zero-strength sources: {skipped_sources}")

    sweep_results = {}
    for source in active_sources:
        nominal_strength = nominal_strengths[source]
        points = []
        if show_summary:
            print(f"--- source = {source}, nominal strength = {nominal_strength} ---")

        for s_value in s_values:
            strength = float(s_value) * nominal_strength
            params = simulation_parameters_with_single_noise_source(
                base_parameters=base_parameters,
                noise_source=source,
                strength=strength,
            )
            params["n_bar_list"] = [selected_nbar]

            channel_result = generate_error_channel_matrices(
                convention=convention,
                **params,
            )
            nbar_used, R_post, chi = _extract_channel_point_from_result(
                channel_result,
                nbar=selected_nbar,
            )
            points.append(
                {
                    "s": float(s_value),
                    "strength": strength,
                    "nominal_strength": nominal_strength,
                    "nbar": nbar_used,
                    "R_post": R_post,
                    "chi": chi,
                    "parameters": channel_result["parameters"],
                }
            )
            if show_summary:
                print(f"s = {s_value}: strength = {strength}")

        sweep_results[source] = {
            "nominal_strength": nominal_strength,
            "points": points,
        }

    return sweep_results


def _channel_summary_metrics(R_post, chi, pauli_labels=None):
    if pauli_labels is None:
        pauli_labels = CHI_PAULI_LABELS_2Q

    R_post = _as_complex_array(R_post)
    chi = _as_complex_array(chi)
    chi_normed = _trace_normalize_chi(chi)

    pauli_probs = np.real(np.diag(chi_normed))
    p_err = float(1.0 - pauli_probs[0])
    p_w1 = float(
        np.sum([p for p, label in zip(pauli_probs, pauli_labels) if _pauli_weight(label) == 1])
    )
    p_w2 = float(
        np.sum([p for p, label in zip(pauli_probs, pauli_labels) if _pauli_weight(label) == 2])
    )

    chi_norm = np.linalg.norm(chi_normed, ord="fro")
    offdiag = _offdiag_part(chi_normed)
    identity_error = np.concatenate([chi_normed[0, 1:].ravel(), chi_normed[1:, 0].ravel()])
    C_off = _safe_ratio(np.linalg.norm(offdiag, ord="fro"), chi_norm)
    C_IE = _safe_ratio(np.sqrt(np.sum(np.abs(identity_error) ** 2)), chi_norm)

    identity_ptm = np.eye(R_post.shape[0], dtype=complex)
    D_R = _safe_ratio(
        np.linalg.norm(R_post - identity_ptm, ord="fro"),
        np.linalg.norm(identity_ptm, ord="fro"),
    )

    return {
        "pauli_probs": pauli_probs,
        "pauli_prob_sum": float(np.sum(pauli_probs)),
        "p_err": p_err,
        "p_w1": p_w1,
        "p_w2": p_w2,
        "weight2_fraction": _safe_ratio(p_w2, p_err),
        "C_off": C_off,
        "C_IE": C_IE,
        "D_R": D_R,
        "max_imag_R": float(np.max(np.abs(np.imag(R_post)))),
        "max_imag_chi": float(np.max(np.abs(np.imag(chi)))),
    }


def _print_channel_sanity(label, R_post, chi, pauli_labels=None):
    metrics = _channel_summary_metrics(R_post, chi, pauli_labels=pauli_labels)
    print(f"--- Sanity check: {label} ---")
    print(f"R_post shape: {np.shape(R_post)}")
    print(f"chi shape: {np.shape(chi)}")
    print(f"sum of Pauli probabilities: {metrics['pauli_prob_sum']:.8e}")
    print(f"p_err: {metrics['p_err']:.8e}")
    print(f"p_w1: {metrics['p_w1']:.8e}")
    print(f"p_w2: {metrics['p_w2']:.8e}")
    print(f"C_off: {metrics['C_off']:.8e}")
    print(f"C_IE: {metrics['C_IE']:.8e}")
    print(f"||R_post - I||_F / ||I||_F: {metrics['D_R']:.8e}")
    print(f"max imaginary part of R_post: {metrics['max_imag_R']:.8e}")
    print(f"max imaginary part of chi: {metrics['max_imag_chi']:.8e}")
    return metrics


def _format_pauli_heatmap_axis(ax, pauli_labels, show_axis_labels=True, tick_labelsize=7):
    ax.set_xticks(range(len(pauli_labels)))
    ax.set_yticks(range(len(pauli_labels)))
    ax.set_xticklabels(pauli_labels, rotation=90, fontsize=tick_labelsize)
    ax.set_yticklabels(pauli_labels, fontsize=tick_labelsize)
    if show_axis_labels:
        ax.set_xlabel("Input Pauli")
        ax.set_ylabel("Output Pauli")


def make_post_gate_noise_channel_figures(
    temperature_result,
    noise_source_results=None,
    ptm_derivative_results=None,
    selected_nbar=None,
    baseline_nbar=0.01,
    output_dir="post_gate_noise_channel_figures",
    pauli_labels=None,
):
    import matplotlib.pyplot as plt
    from pathlib import Path

    if pauli_labels is None:
        pauli_labels = CHI_PAULI_LABELS_2Q

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    temperature_points = load_temperature_results(temperature_result, pauli_labels=pauli_labels)
    for point in temperature_points:
        _print_channel_sanity(
            f"temperature nbar={point['nbar']}",
            point["R_post"],
            point["chi"],
            pauli_labels=pauli_labels,
        )

    baseline_index = int(
        np.argmin(np.abs(np.asarray([point["nbar"] for point in temperature_points]) - baseline_nbar))
    )
    baseline_point = temperature_points[baseline_index]
    baseline_R = baseline_point["R_post"]
    delta_points = [
        point for index, point in enumerate(temperature_points) if index != baseline_index
    ]

    figures = {}
    if delta_points:
        delta_R_list = [point["R_post"] - baseline_R for point in delta_points]
        vmax = max(np.max(np.abs(np.real(delta_R))) for delta_R in delta_R_list)
        vmax = max(vmax, 1e-16)

        fig_A, axes_A = plt.subplots(
            1,
            len(delta_points),
            figsize=(4.0 * len(delta_points), 4.2),
            squeeze=False,
            constrained_layout=True,
        )
        axes_A = axes_A[0]
        for ax, point, delta_R in zip(axes_A, delta_points, delta_R_list):
            image = ax.imshow(
                np.real(delta_R),
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            _format_pauli_heatmap_axis(
                ax,
                pauli_labels,
                show_axis_labels=False,
                tick_labelsize=10,
            )
        fig_A.colorbar(image, ax=axes_A, fraction=0.025, pad=0.02)
        fig_A.savefig(output_path / "figure_A_temperature_delta_R_heatmap.png", dpi=300, bbox_inches="tight")
        fig_A.savefig(output_path / "figure_A_temperature_delta_R_heatmap.pdf", bbox_inches="tight")
        figures["A"] = fig_A
    else:
        print("Figure A was skipped because only the baseline temperature point was found.")

    try:
        source_points = load_noise_source_results(
            noise_source_results=noise_source_results,
            ptm_derivative_results=ptm_derivative_results,
            selected_nbar=selected_nbar,
        )
    except ValueError as exc:
        print(str(exc))
        return {"temperature_points": temperature_points, "noise_source_points": None, "figures": figures}

    source_metrics = {}
    for source, points in source_points.items():
        source_metrics[source] = []
        for point in points:
            metrics = _print_channel_sanity(
                f"{source}, s={point['s']}, nbar={point['nbar']}",
                point["R_post"],
                point["chi"],
                pauli_labels=pauli_labels,
            )
            source_metrics[source].append({**point, **metrics})

    fig_D, ax_D = plt.subplots(figsize=(7, 4.5))
    fig_E1, ax_E1 = plt.subplots(figsize=(7, 4.5))
    fig_E2, ax_E2 = plt.subplots(figsize=(7, 4.5))
    fig_F1, ax_F1 = plt.subplots(figsize=(7, 4.5))
    fig_F2, ax_F2 = plt.subplots(figsize=(7, 4.5))
    fig_F3, ax_F3 = plt.subplots(figsize=(7, 4.5))

    nominal_sources = []
    nominal_w1 = []
    nominal_w2 = []
    nominal_delta_R = []

    for source, points in source_metrics.items():
        points = sorted(points, key=lambda item: item["s"])
        s_values = np.asarray([point["s"] for point in points], dtype=float)
        source_baseline = min(points, key=lambda point: abs(point["s"]))
        source_baseline_R = np.asarray(source_baseline["R_post"])
        identity_norm = np.linalg.norm(np.eye(source_baseline_R.shape[0]), ord="fro")
        source_delta_distances = [
            _safe_ratio(
                np.linalg.norm(np.asarray(point["R_post"]) - source_baseline_R, ord="fro"),
                identity_norm,
            )
            for point in points
        ]
        for point, distance in zip(points, source_delta_distances):
            point["D_R_source"] = float(distance)

        ax_D.plot(s_values, source_delta_distances, marker="o", label=source)
        ax_E1.plot(s_values, [point["p_err"] for point in points], marker="o", label=source)
        ax_E2.plot(
            s_values,
            [point["weight2_fraction"] for point in points],
            marker="o",
            label=source,
        )
        ax_F1.plot(s_values, [point["C_off"] for point in points], marker="o", label=source)
        ax_F2.plot(s_values, [point["C_IE"] for point in points], marker="o", label=source)
        ax_F3.plot(
            [point["p_err"] for point in points],
            [point["C_off"] for point in points],
            marker="o",
            label=source,
        )
        for point in points:
            ax_F3.annotate(
                f"{point['s']:.2g}",
                (point["p_err"], point["C_off"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

        nominal = min(points, key=lambda point: abs(point["s"] - 1.0))
        nominal_sources.append(source)
        nominal_w1.append(nominal["p_w1"])
        nominal_w2.append(nominal["p_w2"])
        nominal_delta_R.append(nominal["R_post"] - np.eye(nominal["R_post"].shape[0]))

    ax_D.set_xlabel("normalized noise strength $s$")
    ax_D.set_ylabel(r"$||R_{post}(s)-R_{post}(0)||_F / ||I||_F$")
    ax_D.grid(True, alpha=0.4)
    ax_D.legend()
    fig_D.tight_layout()
    fig_D.savefig(output_path / "figure_D_channel_deformation_by_noise_source.png", dpi=300, bbox_inches="tight")
    fig_D.savefig(output_path / "figure_D_channel_deformation_by_noise_source.pdf", bbox_inches="tight")
    figures["D"] = fig_D

    ax_E1.set_xlabel("normalized noise strength $s$")
    ax_E1.set_ylabel(r"$p_{err}=1-p_{II}$")
    ax_E1.grid(True, alpha=0.4)
    ax_E1.legend()
    fig_E1.tight_layout()
    fig_E1.savefig(output_path / "figure_E1_pauli_error_probability.png", dpi=300, bbox_inches="tight")
    fig_E1.savefig(output_path / "figure_E1_pauli_error_probability.pdf", bbox_inches="tight")
    figures["E1"] = fig_E1

    ax_E2.set_xlabel("normalized noise strength $s$")
    ax_E2.set_ylabel(r"$p_{w2}/p_{err}$")
    ax_E2.grid(True, alpha=0.4)
    ax_E2.legend()
    fig_E2.tight_layout()
    fig_E2.savefig(output_path / "figure_E2_weight2_fraction.png", dpi=300, bbox_inches="tight")
    fig_E2.savefig(output_path / "figure_E2_weight2_fraction.pdf", bbox_inches="tight")
    figures["E2"] = fig_E2

    fig_E3, ax_E3 = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(nominal_sources))
    ax_E3.bar(x, nominal_w1, label="weight-1")
    ax_E3.bar(x, nominal_w2, bottom=nominal_w1, label="weight-2")
    ax_E3.set_xticks(x)
    ax_E3.set_xticklabels(nominal_sources, rotation=30, ha="right")
    ax_E3.set_ylabel("Pauli error probability")
    ax_E3.grid(True, axis="y", alpha=0.4)
    ax_E3.legend()
    fig_E3.tight_layout()
    fig_E3.savefig(output_path / "figure_E3_weight_decomposition_at_nominal_strength.png", dpi=300, bbox_inches="tight")
    fig_E3.savefig(output_path / "figure_E3_weight_decomposition_at_nominal_strength.pdf", bbox_inches="tight")
    figures["E3"] = fig_E3

    ax_F1.set_xlabel("normalized noise strength $s$")
    ax_F1.set_ylabel(r"$C_{off}$")
    ax_F1.grid(True, alpha=0.4)
    ax_F1.legend()
    fig_F1.tight_layout()
    fig_F1.savefig(output_path / "figure_F1_chi_offdiag_ratio.png", dpi=300, bbox_inches="tight")
    fig_F1.savefig(output_path / "figure_F1_chi_offdiag_ratio.pdf", bbox_inches="tight")
    figures["F1"] = fig_F1

    ax_F2.set_xlabel("normalized noise strength $s$")
    ax_F2.set_ylabel(r"$C_{IE}$")
    ax_F2.grid(True, alpha=0.4)
    ax_F2.legend()
    fig_F2.tight_layout()
    fig_F2.savefig(output_path / "figure_F2_identity_error_coupling_ratio.png", dpi=300, bbox_inches="tight")
    fig_F2.savefig(output_path / "figure_F2_identity_error_coupling_ratio.pdf", bbox_inches="tight")
    figures["F2"] = fig_F2

    ax_F3.set_xlabel(r"$p_{err}$")
    ax_F3.set_ylabel(r"$C_{off}$")
    ax_F3.grid(True, alpha=0.4)
    ax_F3.legend()
    fig_F3.tight_layout()
    fig_F3.savefig(output_path / "figure_F3_pauli_error_vs_nonpauli_ratio.png", dpi=300, bbox_inches="tight")
    fig_F3.savefig(output_path / "figure_F3_pauli_error_vs_nonpauli_ratio.pdf", bbox_inches="tight")
    figures["F3"] = fig_F3

    if nominal_delta_R:
        n_sources = len(nominal_sources)
        ncols = n_sources if n_sources <= 4 else int(np.ceil(n_sources / 2))
        nrows = int(np.ceil(n_sources / ncols))
        vmax = max(np.max(np.abs(np.real(delta_R))) for delta_R in nominal_delta_R)
        vmax = max(vmax, 1e-16)
        fig_G, axes_G = plt.subplots(
            nrows,
            ncols,
            figsize=(4.0 * ncols, 4.0 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        flat_axes = axes_G.ravel()
        for ax, source, delta_R in zip(flat_axes, nominal_sources, nominal_delta_R):
            image = ax.imshow(
                np.real(delta_R),
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            _format_pauli_heatmap_axis(
                ax,
                pauli_labels,
                show_axis_labels=False,
                tick_labelsize=10,
            )
        for ax in flat_axes[len(nominal_sources):]:
            ax.axis("off")
        fig_G.colorbar(image, ax=flat_axes[: len(nominal_sources)], fraction=0.025, pad=0.02)
        fig_G.savefig(output_path / "figure_G_source_fingerprint_heatmap.png", dpi=300, bbox_inches="tight")
        fig_G.savefig(output_path / "figure_G_source_fingerprint_heatmap.pdf", bbox_inches="tight")
        figures["G"] = fig_G

    return {
        "temperature_points": temperature_points,
        "noise_source_points": source_points,
        "source_metrics": source_metrics,
        "figures": figures,
        "output_dir": output_path,
    }


def save_noise_source_error_ptm_top_components(
    ptm_derivative_results,
    output_dir="chi_scalar_summary",
    n_bar_values=None,
    top_k_per_noise_source=1,
    pauli_labels=None,
):
    import pandas as pd
    from pathlib import Path

    if pauli_labels is None:
        pauli_labels = CHI_PAULI_LABELS_2Q

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def select_nbars(requested_nbars, available_nbars):
        if requested_nbars is None:
            requested_values = [available_nbars[0]]
        elif np.isscalar(requested_nbars):
            requested_values = [requested_nbars]
        else:
            requested_values = list(requested_nbars)

        selected = []
        for requested in requested_values:
            for available in available_nbars:
                if np.isclose(float(available), float(requested)):
                    selected.append(available)
                    break
            else:
                raise KeyError(
                    f"Requested n_bar={requested} is not available. "
                    f"Available n_bar values: {available_nbars}"
                )
        return selected

    summary_rows = []
    for noise_source, result in ptm_derivative_results.items():
        available_nbars = list(result["ptm_derivative_by_n_bar"].keys())
        selected_nbars = select_nbars(n_bar_values, available_nbars)

        for nbar in selected_nbars:
            derivative = np.asarray(result["ptm_derivative_by_n_bar"][nbar])
            flat_order = np.argsort(np.abs(derivative).ravel())[::-1]

            kept = 0
            for flat_idx in flat_order:
                output_index, input_index = np.unravel_index(flat_idx, derivative.shape)
                output_label = pauli_labels[output_index]
                input_label = pauli_labels[input_index]

                if output_label == "II" and input_label == "II":
                    continue

                value = derivative[output_index, input_index]
                norms = result["derivative_norms"][nbar]
                summary_rows.append(
                    {
                        "noise_source": noise_source,
                        "n_bar": nbar,
                        "rank": kept + 1,
                        "output": output_label,
                        "input": input_label,
                        "output_index": output_index,
                        "input_index": input_index,
                        "value": value,
                        "real_value": float(np.real(value)),
                        "imag_value": float(np.imag(value)),
                        "abs_value": float(abs(value)),
                        "frobenius_norm": norms["frobenius"],
                        "max_abs_norm": norms["max_abs"],
                        "finite_difference_method": result["method"],
                        "strength": result["strength"],
                        "step": result["step"],
                    }
                )
                kept += 1
                if kept >= top_k_per_noise_source:
                    break

    top_df = pd.DataFrame(summary_rows)
    output_file = output_path / "noise_source_error_ptm_top_components.csv"
    top_df.to_csv(output_file, index=False)
    return top_df


def plot_noise_source_error_ptm_derivatives(
    ptm_derivative_results,
    n_bar_values=None,
    noise_sources=None,
):
    if noise_sources is None:
        noise_sources = list(ptm_derivative_results.keys())

    first_source = noise_sources[0]
    available_nbars = list(ptm_derivative_results[first_source]["ptm_derivative_by_n_bar"].keys())

    if n_bar_values is None:
        requested_nbars = [available_nbars[0]]
    elif np.isscalar(n_bar_values):
        requested_nbars = [n_bar_values]
    else:
        requested_nbars = list(n_bar_values)

    selected_nbars = []
    for requested in requested_nbars:
        for available in available_nbars:
            if np.isclose(float(available), float(requested)):
                selected_nbars.append(available)
                break
        else:
            raise KeyError(
                f"Requested n_bar={requested} is not available. "
                f"Available n_bar values: {available_nbars}"
            )

    figures = {}
    axes = {}
    for nbar in selected_nbars:
        figures[nbar] = {}
        axes[nbar] = {}
        for noise_source in noise_sources:
            derivative_ptm = ptm_derivative_results[noise_source]["ptm_derivative_by_n_bar"][nbar]
            fig, ax = plot_ptm(
                derivative_ptm,
                title=f"d Error PTM / d {noise_source}, n_bar = {nbar}",
            )
            figures[nbar][noise_source] = fig
            axes[nbar][noise_source] = ax

    return figures, axes, selected_nbars


def _single_pauli_commutes(pauli_a, pauli_b):
    return pauli_a == "I" or pauli_b == "I" or pauli_a == pauli_b


def _commutation_sign(pauli_a, pauli_b):
    anticommutes = sum(
        not _single_pauli_commutes(char_a, char_b)
        for char_a, char_b in zip(pauli_a, pauli_b)
    )
    return 1 if anticommutes % 2 == 0 else -1


def commutation_sign_matrix(pauli_labels=None):
    if pauli_labels is None:
        pauli_labels = CHI_PAULI_LABELS_2Q
    return np.array(
        [
            [
                _commutation_sign(row_label, col_label)
                for col_label in pauli_labels
            ]
            for row_label in pauli_labels
        ],
        dtype=float,
    )


def pauli_probs_from_ptm_diagonal(ptm_diagonal, pauli_labels=None):
    if pauli_labels is None:
        pauli_labels = CHI_PAULI_LABELS_2Q
    lambdas = np.real(np.asarray(ptm_diagonal, dtype=complex))
    signs = commutation_sign_matrix(pauli_labels)
    return (signs @ lambdas) / len(pauli_labels)


def depolarizing_from_infidelity(r, pauli_labels=None, dimension=4):
    if pauli_labels is None:
        pauli_labels = CHI_PAULI_LABELS_2Q
    p_dep_total = ((dimension + 1) / dimension) * r
    p_each = p_dep_total / (len(pauli_labels) - 1)
    probs = np.full(len(pauli_labels), p_each, dtype=float)
    probs[pauli_labels.index("II")] = 1.0 - p_dep_total
    alpha = 1.0 - (dimension**2 / (dimension**2 - 1)) * p_dep_total
    R_dep = np.diag([1.0] + [alpha] * (len(pauli_labels) - 1))
    return p_dep_total, p_each, alpha, probs, R_dep


def compare_error_channel_to_depolarizing(
    error_result,
    output_dir="depolarizing_comparison",
    pauli_labels=None,
    pauli_probs_by_nbar=None,
    show_summary=True,
):
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    if pauli_labels is None:
        pauli_labels = CHI_PAULI_LABELS_2Q

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dimension = 4
    n_pauli = len(pauli_labels)
    ii_index = pauli_labels.index("II")
    xx_index = pauli_labels.index("XX") if "XX" in pauli_labels else None
    weight1_indices = [i for i, label in enumerate(pauli_labels) if _pauli_weight(label) == 1]
    weight2_indices = [i for i, label in enumerate(pauli_labels) if _pauli_weight(label) == 2]
    I_ptm = np.eye(n_pauli, dtype=complex)

    if "error_ptm_by_n_bar" in error_result:
        preferred_nbars = error_result.get("parameters", {}).get("n_bar_list", None)
        R_items = _ordered_mapping_items(error_result["error_ptm_by_n_bar"], preferred_nbars)
    else:
        nbars = [data["n_bar"] for data in error_result["results_list"]]
        R_items = list(zip(nbars, error_result["error_ptm_list"]))

    chi_by_nbar = dict(error_result.get("error_chi_by_n_bar", {}))
    if not chi_by_nbar and "error_chi_matrix_list" in error_result:
        nbars = [data["n_bar"] for data in error_result["results_list"]]
        chi_by_nbar = dict(zip(nbars, error_result["error_chi_matrix_list"]))
    pauli_probs_by_nbar = dict(pauli_probs_by_nbar or {})

    param_rows = []
    distance_rows = []
    pauli_rows = []
    top_rows = []
    sanity_rows = []

    for raw_nbar, R_raw in R_items:
        nbar = float(raw_nbar)
        R_exact = _as_complex_array(R_raw)
        if R_exact.shape != (n_pauli, n_pauli):
            raise ValueError(
                f"R_exact for nbar={nbar} has shape {R_exact.shape}, "
                f"expected {(n_pauli, n_pauli)}"
            )

        R_trace = float(np.real(np.trace(R_exact)))
        F_avg_exact = (R_trace + dimension) / (dimension * (dimension + 1))
        infidelity_r = 1.0 - F_avg_exact
        p_dep_total, p_dep_each, alpha_dep, p_dep_probs, R_dep = depolarizing_from_infidelity(
            infidelity_r,
            pauli_labels=pauli_labels,
            dimension=dimension,
        )

        R_PT = np.diag(np.diag(R_exact))
        p_given = _lookup_mapping_by_float_key(pauli_probs_by_nbar, nbar)
        if p_given is None:
            p_PT = pauli_probs_from_ptm_diagonal(np.diag(R_PT), pauli_labels)
        else:
            p_PT = np.real(np.asarray(p_given, dtype=complex))
            total = np.sum(p_PT)
            if abs(total) > 1e-15:
                p_PT = p_PT / total

        denominator = np.linalg.norm(R_exact - I_ptm, ord="fro")
        fro_exact_PT = float(np.linalg.norm(R_exact - R_PT, ord="fro"))
        fro_PT_dep = float(np.linalg.norm(R_PT - R_dep, ord="fro"))
        fro_exact_dep = float(np.linalg.norm(R_exact - R_dep, ord="fro"))

        p_err_PT = float(1.0 - p_PT[ii_index])
        p_err_dep = float(1.0 - p_dep_probs[ii_index])
        p_XX_PT = float(p_PT[xx_index]) if xx_index is not None else np.nan
        p_XX_dep = float(p_dep_probs[xx_index]) if xx_index is not None else np.nan
        TVD_PT_dep = float(0.5 * np.sum(np.abs(p_PT - p_dep_probs)))
        weight1_PT = float(np.sum(p_PT[weight1_indices]))
        weight2_PT = float(np.sum(p_PT[weight2_indices]))
        weight1_dep = float(np.sum(p_dep_probs[weight1_indices]))
        weight2_dep = float(np.sum(p_dep_probs[weight2_indices]))

        chi_exact = _lookup_mapping_by_float_key(chi_by_nbar, nbar)
        C_chi, identity_error_coupling_ratio = _chi_coherence_metrics(
            chi_exact,
            ii_index=ii_index,
        )

        param_rows.append(
            {
                "nbar": nbar,
                "F_avg_exact": F_avg_exact,
                "infidelity_r": infidelity_r,
                "p_dep_total": p_dep_total,
                "p_dep_each_nonidentity": p_dep_each,
                "alpha_dep": alpha_dep,
            }
        )
        distance_rows.append(
            {
                "nbar": nbar,
                "fro_exact_PT": fro_exact_PT,
                "fro_PT_dep": fro_PT_dep,
                "fro_exact_dep": fro_exact_dep,
                "rel_exact_PT": _safe_ratio(fro_exact_PT, denominator),
                "rel_PT_dep": _safe_ratio(fro_PT_dep, denominator),
                "rel_exact_dep": _safe_ratio(fro_exact_dep, denominator),
                "C_chi": C_chi,
                "identity_error_coupling_ratio": identity_error_coupling_ratio,
            }
        )
        pauli_rows.append(
            {
                "nbar": nbar,
                "p_err_PT": p_err_PT,
                "p_err_dep": p_err_dep,
                "weight1_PT": weight1_PT,
                "weight1_dep": weight1_dep,
                "weight2_PT": weight2_PT,
                "weight2_dep": weight2_dep,
                "p_XX_PT": p_XX_PT,
                "p_XX_dep": p_XX_dep,
                "ratio_XX_PT_over_dep": _safe_ratio(p_XX_PT, p_XX_dep),
                "TVD_PT_dep": TVD_PT_dep,
            }
        )

        non_identity_order = [idx for idx in np.argsort(p_PT)[::-1] if idx != ii_index]
        for rank, idx in enumerate(non_identity_order[:10], start=1):
            top_rows.append(
                {
                    "nbar": nbar,
                    "rank": rank,
                    "pauli": pauli_labels[idx],
                    "p_PT": float(p_PT[idx]),
                    "p_dep_uniform": float(p_dep_probs[idx]),
                    "ratio_PT_over_dep": _safe_ratio(float(p_PT[idx]), float(p_dep_probs[idx])),
                    "excess_PT_minus_dep": float(p_PT[idx] - p_dep_probs[idx]),
                }
            )

        sanity_rows.append(
            {
                "nbar": nbar,
                "F_avg_in_0_1": bool(-1e-12 <= F_avg_exact <= 1.0 + 1e-12),
                "p_dep_in_0_1": bool(-1e-12 <= p_dep_total <= 1.0 + 1e-12),
                "sum_p_PT": float(np.sum(p_PT)),
                "sum_p_dep_probs": float(np.sum(p_dep_probs)),
                "R_dep_shape_ok": R_dep.shape == (n_pauli, n_pauli),
                "R_PT_offdiag_norm": float(
                    np.linalg.norm(R_PT - np.diag(np.diag(R_PT)), ord="fro")
                ),
            }
        )

    parameter_df = pd.DataFrame(param_rows).sort_values("nbar")
    distance_df = pd.DataFrame(distance_rows).sort_values("nbar")
    pauli_df = pd.DataFrame(pauli_rows).sort_values("nbar")
    top_df = pd.DataFrame(top_rows).sort_values(["nbar", "rank"])
    sanity_df = pd.DataFrame(sanity_rows).sort_values("nbar")

    csv_tex_pairs = [
        (parameter_df, "depolarizing_parameter_summary"),
        (distance_df, "channel_approximation_distance_summary"),
        (pauli_df, "pauli_distribution_vs_depolarizing_summary"),
        (top_df, "top_pauli_vs_depolarizing"),
    ]
    for df, stem in csv_tex_pairs:
        df.to_csv(output_path / f"{stem}.csv", index=False)
        tex = df.to_latex(index=False, escape=False, float_format=lambda value: f"{value:.3e}")
        (output_path / f"{stem}.tex").write_text(tex)

    sanity_df.to_csv(output_path / "sanity_check_summary.csv", index=False)

    figures = {}
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(distance_df["nbar"], distance_df["fro_exact_PT"], marker="o", label=r"$||R_{exact}-R_{PT}||_F$")
    ax1.plot(distance_df["nbar"], distance_df["fro_PT_dep"], marker="o", label=r"$||R_{PT}-R_{dep}||_F$")
    ax1.plot(distance_df["nbar"], distance_df["fro_exact_dep"], marker="o", label=r"$||R_{exact}-R_{dep}||_F$")
    ax1.set_xlabel(r"Mean phonon number $\bar{n}_0$")
    ax1.set_ylabel("Frobenius norm")
    ax1.grid(True, alpha=0.35)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(output_path / "fig_channel_distance_vs_nbar.pdf", bbox_inches="tight")
    figures["channel_distance_vs_nbar"] = fig1

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(pauli_df["nbar"], pauli_df["p_XX_PT"], marker="o", label=r"$p_{XX}^{PT}$")
    ax2.plot(pauli_df["nbar"], pauli_df["p_XX_dep"], marker="o", label=r"$p_{XX}^{dep}$")
    ax2.set_xlabel(r"Mean phonon number $\bar{n}_0$")
    ax2.set_ylabel("probability")
    ax2.grid(True, alpha=0.35)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(output_path / "fig_xx_probability_vs_depolarizing.pdf", bbox_inches="tight")
    figures["xx_probability_vs_depolarizing"] = fig2

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    ax3.plot(pauli_df["nbar"], pauli_df["weight1_PT"], marker="o", label="weight1_PT")
    ax3.plot(pauli_df["nbar"], pauli_df["weight2_PT"], marker="o", label="weight2_PT")
    ax3.plot(pauli_df["nbar"], pauli_df["weight1_dep"], marker="o", label="weight1_dep")
    ax3.plot(pauli_df["nbar"], pauli_df["weight2_dep"], marker="o", label="weight2_dep")
    ax3.set_xlabel(r"Mean phonon number $\bar{n}_0$")
    ax3.set_ylabel("probability")
    ax3.grid(True, alpha=0.35)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(output_path / "fig_weight_distribution_vs_depolarizing.pdf", bbox_inches="tight")
    figures["weight_distribution_vs_depolarizing"] = fig3

    if show_summary:
        print("Saved outputs to:", output_path.resolve())
        print("Pauli labels:", pauli_labels)
        print("--- Sanity check ---")
        print(sanity_df.to_string(index=False))
        print("F_avg all in [0,1]:", bool(sanity_df["F_avg_in_0_1"].all()))
        print("p_dep all in [0,1]:", bool(sanity_df["p_dep_in_0_1"].all()))
        print("max |sum(p_PT)-1|:", float(np.max(np.abs(sanity_df["sum_p_PT"] - 1.0))))
        print(
            "max |sum(p_dep_probs)-1|:",
            float(np.max(np.abs(sanity_df["sum_p_dep_probs"] - 1.0))),
        )
        print("R_dep shape all OK:", bool(sanity_df["R_dep_shape_ok"].all()))
        print("max R_PT offdiag norm:", float(np.max(sanity_df["R_PT_offdiag_norm"])))

    return {
        "depolarizing_parameter_summary_df": parameter_df,
        "channel_approximation_distance_summary_df": distance_df,
        "pauli_distribution_vs_depolarizing_summary_df": pauli_df,
        "top_pauli_vs_depolarizing_df": top_df,
        "sanity_check_df": sanity_df,
        "figures": figures,
        "pauli_labels": pauli_labels,
        "output_dir": output_path,
    }


def _ordered_mapping_items(mapping, preferred_keys=None):
    if preferred_keys is None:
        return list(mapping.items())

    items = []
    used_keys = set()
    for requested in preferred_keys:
        matched = None
        for key in mapping:
            if key in used_keys:
                continue
            try:
                if np.isclose(float(key), float(requested)):
                    matched = key
                    break
            except Exception:
                if key == requested:
                    matched = key
                    break
        if matched is not None:
            items.append((matched, mapping[matched]))
            used_keys.add(matched)

    for key, value in mapping.items():
        if key not in used_keys:
            items.append((key, value))
    return items


def _lookup_mapping_by_float_key(mapping, requested_key):
    for key, value in mapping.items():
        try:
            if np.isclose(float(key), float(requested_key)):
                return value
        except Exception:
            if key == requested_key:
                return value
    return None


def _chi_coherence_metrics(chi, ii_index=0):
    if chi is None:
        return np.nan, np.nan
    chi = _as_complex_array(chi)
    chi_norm = np.linalg.norm(chi, ord="fro")
    if chi_norm < 1e-15:
        return np.nan, np.nan
    offdiag = chi - np.diag(np.diag(chi))
    C_chi = np.linalg.norm(offdiag, ord="fro") / chi_norm
    mask = np.ones(chi.shape[0], dtype=bool)
    mask[ii_index] = False
    coupling = np.sqrt(
        np.sum(np.abs(chi[ii_index, mask]) ** 2)
        + np.sum(np.abs(chi[mask, ii_index]) ** 2)
    )
    return C_chi, coupling / chi_norm


def run_infidelity_analysis(show_plot=True, **simulation_parameters):
    result = generate_chi_matrices(**simulation_parameters)

    XX = qp.tensor(qp.sigmax(), qp.sigmax())
    phi = np.pi / 4
    U_ideal = (1j * phi * XX).expm()
    chi_ideal = qp.to_chi(qp.to_super(U_ideal))

    dim = 4
    f_avg_list = []
    for chi_real in result["chi_qobj_list"]:
        f_pro = qp.process_fidelity(chi_real, chi_ideal)
        f_avg = (dim * f_pro + 1) / (dim + 1)
        f_avg_list.append(f_avg)

    infidelity_list = [max(1.0 - f, 1e-16) for f in f_avg_list]

    if show_plot:
        import matplotlib.pyplot as plt

        n_bar_list = result["parameters"]["n_bar_list"]
        plt.figure(figsize=(8, 5))
        plt.semilogy(
            n_bar_list,
            infidelity_list,
            "o-",
            linewidth=2,
            label="Infidelity (Error Rate)",
        )
        plt.axhline(
            y=0.01,
            color="r",
            linestyle="--",
            label="Target Error $10^{-2}$ (99%)",
        )
        plt.axhline(
            y=0.001,
            color="g",
            linestyle=":",
            label="Target Error $10^{-3}$ (99.9%)",
        )
        plt.xlabel(r"Mean phonon number $\bar{n}_0$")
        plt.ylabel("Infidelity ($1 - F_{avg}$)")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.show()

    print("--- Result Summary (Error Rate) ---")
    for n, f in zip(result["parameters"]["n_bar_list"], f_avg_list):
        err = 1.0 - f
        print(f"n_bar = {n}: Error = {err:.6e} (Fidelity = {f:.6f})")

    return {
        **result,
        "chi_ideal": chi_ideal,
        "f_avg_list": f_avg_list,
        "infidelity_list": infidelity_list,
    }


__all__ = [
    "MSGate",
    "coherent_ms_propagator",
    "sample_laser_parameters",
    "get_optimal_nv_general",
    "estimate_phonon_dim",
    "process_superoperator_from_states",
    "build_input_states",
    "run_ms_gate_simulation",
    "generate_process_channels",
    "generate_chi_matrices",
    "ideal_ms_gate",
    "remove_ideal_gate_from_channel",
    "two_qubit_pauli_basis",
    "superoperator_to_ptm",
    "generate_error_channel_matrices",
    "pauli_labels_and_weights",
    "validate_pauli_label_order",
    "validate_error_channel_composition",
    "choi_physicality_metrics",
    "validate_channel_physicality",
    "validate_error_channel_reliability",
    "simulation_parameters_with_single_noise_source",
    "differentiate_error_ptm_by_noise_source",
    "differentiate_error_ptm_noise_sources",
    "plot_chi_matrix",
    "plot_ptm",
    "ptm_nbar_derivative",
    "top_ptm_derivative_components",
    "analyze_top_ptm_nbar_derivative_components",
    "summarize_chi_nbar_dependence",
    "top_chi_nbar_derivative_components",
    "run_error_chi_scalar_summary",
    "plot_chi_offdiagonal_presentation_summary",
    "load_temperature_results",
    "load_noise_source_results",
    "nominal_noise_source_strengths",
    "active_noise_sources_from_nominal_strengths",
    "run_noise_source_strength_sweep",
    "make_post_gate_noise_channel_figures",
    "save_noise_source_error_ptm_top_components",
    "plot_noise_source_error_ptm_derivatives",
    "commutation_sign_matrix",
    "pauli_probs_from_ptm_diagonal",
    "depolarizing_from_infidelity",
    "compare_error_channel_to_depolarizing",
    "run_infidelity_analysis",
]
