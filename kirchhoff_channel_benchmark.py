"""Same-model MS calibration comparison, reusing the Kirchhoff implementation.

The publication's overlap is retained as a reproduction diagnostic. Standard
channel average fidelity is the common gate-quality metric. Legacy propagators
are read only when their original parameter-dependent filename matches exactly.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import qutip as qp
from scipy.optimize import brentq, minimize_scalar
from threadpoolctl import threadpool_limits

import kirchhoff_paper_infidelity as paper
import drive_calibration_qpt_analysis as generator_analysis
import pauli_channel_distance as distances


ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / 'results/chi_error_element_fit/advanced_publication_validation/hxx_drive_amplitude_calibration/kirchhoff_paper_infidelity'
LABELS, PAULIS = distances.pauli_basis(2)


@dataclass(frozen=True)
class Settings:
    K: float = 28.0
    L: float = 25.0
    eta: float = 0.18
    gate_duration_seconds: float = 28e-6
    phonon_dim: int = 8
    sideband_cutoff: int = 3
    solver: str = 'adaptive_dop853'
    relative_tolerance: float = 1e-9
    absolute_tolerance: float = 1e-11
    trotter_step_scale: float = 1.0
    maximum_step_divisor: float = 20.0

    def model(self):
        return paper.build_paper_model(K=self.K, L=self.L, eta=self.eta,
                                      phonon_dim=self.phonon_dim,
                                      sideband_cutoff=self.sideband_cutoff)

    def landmarks(self):
        values = paper.drive_landmarks(K=self.K, L=self.L, eta=self.eta,
                                      gate_duration_seconds=self.gate_duration_seconds)
        return dict(zip(('Omega_LD', 'Omega_2', 'Omega_4'),
                        (values['omega_ld'], values['omega_2'], values['omega_4'])))


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def legacy_path(settings, omega):
    """Reproduce the original cache key; never infer parameters from array size."""
    if settings.maximum_step_divisor != 20.0:
        return None
    payload = {
        'K': float(settings.K), 'L': float(settings.L),
        'mode_frequency_hz': settings.K / settings.gate_duration_seconds,
        'gate_duration_seconds': settings.gate_duration_seconds,
        'eta': settings.eta, 'phonon_dim': settings.phonon_dim,
        'sideband_cutoff': settings.sideband_cutoff, 'solver': settings.solver,
        'trotter_steps': None, 'trotter_step_scale': settings.trotter_step_scale,
        'relative_tolerance': settings.relative_tolerance,
        'absolute_tolerance': settings.absolute_tolerance,
        'omega_per_second': float(omega), 'implementation': 'kirchhoff_paper_v1',
    }
    digest = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:14]
    stem = f'{omega / 1e6:.9f}'.replace('.', 'p')
    return LEGACY / 'propagator_cache' / f'omega_mhz_{stem}__{digest}.npz'


def channel_from_propagator(propagator, model, nbar):
    """Compute unprojected channels and independently cross-check F_avg."""
    actual, error, metadata = paper.kirchhoff_xx_error_channel(
        propagator, model, n_bar=float(nbar), convention='undo_before_actual')
    # Using Choi directly avoids dependence on QuTiP's chi Pauli-Y convention.
    choi = qp.to_choi(error).full()
    chi = distances.choi_to_chi(choi, PAULIS)
    diagnostics = distances.channel_diagnostics(chi, PAULIS)
    if not diagnostics['cptp_within_tolerance']:
        raise ValueError(f'Raw channel failed physicality check: {diagnostics}')
    probabilities, tail = paper.thermal_weights(model.phonon_dim, nbar)
    blocks = propagator.reshape(model.phonon_dim, 4, model.phonon_dim, 4)
    # F_e = sum_mn p_n |Tr(U_target^dag U_mn)|^2 / d^2.
    traces = np.einsum('ij,mjni->mn', model.target_qubit_unitary.conj().T, blocks)
    entanglement_fidelity = float(np.sum(abs(traces)**2 * probabilities[None, :]) / 16)
    favg_kraus = (4 * entanglement_fidelity + 1) / 5
    favg_chi = (4 * chi[0, 0].real + 1) / 5
    favg_qutip = float(qp.average_gate_fidelity(error))
    crosscheck = max(abs(favg_kraus-favg_chi), abs(favg_qutip-favg_chi))
    if crosscheck > 1e-9:
        raise ValueError(f'Fidelity conventions disagree: {crosscheck}')
    obs = generator_analysis.extract_pauli_generator_from_superoperator(error)
    metric = distances.distance_metrics(chi, PAULIS)
    row = {
        'n_bar': float(nbar), 'standard_average_fidelity': favg_chi,
        'standard_average_infidelity': 1-favg_chi,
        **paper.paper_fidelities(propagator, model, n_bar=float(nbar)),
        'h_XX_rad_per_gate': obs['hamiltonian_coefficients_rad_per_gate']['XX'],
        'h_YY_rad_per_gate': obs['hamiltonian_coefficients_rad_per_gate']['YY'],
        'h_ZZ_rad_per_gate': obs['hamiltonian_coefficients_rad_per_gate']['ZZ'],
        'generator_imaginary_frobenius_norm': obs['generator_imaginary_frobenius_norm'],
        'pauli_ptm_frobenius': metric['ptm_frobenius_distance'],
        'ideal_ptm_frobenius': metric['error_ptm_frobenius_norm'],
        'pauli_choi_trace_distance': metric['choi_trace_distance'],
        'fidelity_crosscheck_error': crosscheck,
        'twirl_infidelity_difference': metric['twirl_infidelity_difference'],
        'thermal_tail_mass': tail,
        **metadata, **diagnostics,
    }
    return chi, row


class Benchmark:
    def __init__(self, output, settings=Settings(), nbar=0.02):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.nbar = float(nbar)
        self.model = settings.model()
        self.records = {}
        self.channels = {}
        self.provenance = {}

    def propagator(self, omega):
        omega = float(omega)
        payload = {**asdict(self.settings), 'omega_per_second': omega,
                   'paper_module_sha256': file_hash(ROOT/'kirchhoff_paper_infidelity.py')}
        key = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
        path = self.output / 'propagators' / f'{key}.npz'
        previous = legacy_path(self.settings, omega)
        source_kind = 'benchmark_cache'
        if not path.exists() and previous is not None and previous.exists():
            path, source_kind = previous, 'legacy_exact_parameter_hash'
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                unitary = archive['propagator'].copy()
                metadata = json.loads(str(archive['metadata_json']))
        else:
            print(f'Propagating Omega={omega/1e6:.9f}, N={self.settings.phonon_dim}, '
                  f'm={self.settings.sideband_cutoff}, {self.settings.solver}', flush=True)
            start = time.perf_counter()
            with threadpool_limits(limits=1):
                if self.settings.solver == 'adaptive_dop853':
                    unitary, metadata = paper.propagate_adaptive_ode(
                        self.model, omega_per_second=omega,
                        gate_duration_seconds=self.settings.gate_duration_seconds,
                        relative_tolerance=self.settings.relative_tolerance,
                        absolute_tolerance=self.settings.absolute_tolerance,
                        maximum_step=1/(self.settings.maximum_step_divisor *
                                        max(abs(self.model.frequencies))))
                else:
                    unitary, metadata = paper.propagate_paper_trotter(
                        self.model, omega_per_second=omega,
                        gate_duration_seconds=self.settings.gate_duration_seconds,
                        step_scale=self.settings.trotter_step_scale)
            metadata.update(wall_time_seconds=time.perf_counter()-start,
                            parameters=payload)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, propagator=unitary,
                                metadata_json=json.dumps(metadata, sort_keys=True))
            source_kind = 'new_simulation'
        if unitary.shape != (4*self.settings.phonon_dim,)*2:
            raise ValueError(f'Unexpected cached dimension: {path}')
        source_path = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        provenance = {'path': source_path, 'sha256': file_hash(path),
                      'source_kind': source_kind, 'parameters': payload,
                      'solver_metadata': metadata}
        self.provenance[str(omega)] = provenance
        return unitary, provenance

    def evaluate(self, omega_mhz):
        omega = float(omega_mhz)*1e6
        if omega not in self.records:
            unitary, source = self.propagator(omega)
            with threadpool_limits(limits=1):
                chi, row = channel_from_propagator(unitary, self.model, self.nbar)
            row.update(omega_per_second=omega, omega_mhz=omega/1e6,
                       source_path=source['path'], source_kind=source['source_kind'])
            self.records[omega] = row
            self.channels[omega] = chi
        return self.records[omega]

    def legacy_sweep(self):
        """Read metadata for candidate drives; exact cache-key matching gates reuse."""
        metadata = pd.read_csv(LEGACY/'kirchhoff_paper_propagator_metadata.csv',
                               float_precision='round_trip')
        for omega in metadata.omega_per_second:
            path = legacy_path(self.settings, float(omega))
            if path is not None and path.exists():
                self.evaluate(float(omega)/1e6)
        if len(self.records) < 3:
            center = self.settings.landmarks()['Omega_4']/1e6
            for point in np.linspace(center*0.96, center*1.04, 17):
                self.evaluate(point)
        return self.table()

    def table(self):
        return pd.DataFrame(list(self.records.values())).sort_values('omega_mhz')

    def calibrate(self):
        """Refine separate objectives on the same full model and temperature.

        Both scalar minima are bracketed by the saved amplitude scan. They are
        local minima in that scan, not claims of global optimal control.
        """
        scan = self.table().reset_index(drop=True)
        selected = {name: self.evaluate(omega/1e6)
                    for name, omega in self.settings.landmarks().items()}
        calibration = {}
        for label, column in [('paper_overlap_opt', 'paper_average_infidelity'),
                              ('standard_Favg_opt', 'standard_average_infidelity')]:
            idx = int(scan[column].idxmin())
            if idx == 0 or idx == len(scan)-1:
                raise ValueError(f'{label} optimum lies at the scan boundary')
            bracket = tuple(scan.iloc[[idx-1, idx, idx+1]].omega_mhz)
            result = minimize_scalar(lambda x: self.evaluate(x)[column],
                                     bracket=bracket, method='brent',
                                     options={'xtol': 2e-8, 'maxiter': 60})
            if not result.success:
                raise RuntimeError(f'{label}: {result.message}')
            selected[label] = self.evaluate(result.x)
            calibration[label] = {'omega_mhz': float(result.x), 'objective': column,
                                  'success': bool(result.success),
                                  'bracket_mhz': bracket, 'nfev': int(result.nfev)}
        scan = self.table().reset_index(drop=True)
        h = scan.h_XX_rad_per_gate.to_numpy()
        indices = np.flatnonzero(h[:-1]*h[1:] <= 0)
        if not len(indices):
            raise ValueError('No hXX sign change in the amplitude scan')
        target = self.settings.landmarks()['Omega_4']/1e6
        idx = min(indices, key=lambda i: abs(scan.iloc[i].omega_mhz-target))
        bracket = tuple(scan.iloc[[idx, idx+1]].omega_mhz)
        root, result = brentq(lambda x: self.evaluate(x)['h_XX_rad_per_gate'],
                             *bracket, xtol=1e-10, full_output=True)
        if not result.converged:
            raise RuntimeError('hXX root did not converge')
        selected['hXX_zero'] = self.evaluate(root)
        calibration['hXX_zero'] = {'omega_mhz': float(root), 'success': True,
                                   'bracket_mhz': bracket,
                                   'residual_hXX': selected['hXX_zero']['h_XX_rad_per_gate']}
        rows = [dict(condition=name, **row) for name, row in selected.items()]
        self.table().to_csv(self.output/'amplitude_sweep.csv', index=False)
        (self.output/'calibration.json').write_text(json.dumps(calibration, indent=2))
        return pd.DataFrame(rows)

    def selected_distances(self, selected, run_sdp=True):
        rows, matrices = [], {}
        identity = np.zeros((16, 16), complex)
        identity[0, 0] = 1
        for _, source in selected.iterrows():
            name = source['condition']
            chi = self.channels[float(source['omega_per_second'])]
            matrices[name] = chi
            row = source.to_dict()
            for target, reference in [('pauli', distances.pauli_twirl_chi(chi)),
                                      ('ideal', identity)]:
                choi = distances.chi_to_choi(chi-reference, PAULIS)
                result = distances.diamond_bounds(choi, 4)
                if run_sdp:
                    with threadpool_limits(limits=1):
                        result.update(distances.diamond_distance_sdp(choi, 4))
                    if not result['sdp_accepted']:
                        raise RuntimeError(f'Unresolved SDP bracket: {name}, {target}')
                row.update({f'{target}_{key}': value for key, value in result.items()})
            rows.append(row)
        np.savez_compressed(self.output/'selected_channels.npz', **matrices,
                            pauli_labels=np.asarray(LABELS))
        result = pd.DataFrame(rows)
        result.to_csv(self.output/'calibration_comparison.csv', index=False)
        self.save_provenance()
        return result

    def save_provenance(self):
        (self.output/'provenance.json').write_text(json.dumps({
            'settings': asdict(self.settings), 'nbar': self.nbar,
            'sources': self.provenance,
            'module_hashes': {name: file_hash(ROOT/name) for name in (
                'kirchhoff_channel_benchmark.py', 'kirchhoff_paper_infidelity.py',
                'drive_amplitude_calibration.py', 'drive_calibration_qpt_analysis.py',
                'pauli_channel_distance.py')},
        }, indent=2))


def convergence_checks(output, selected, settings=Settings(), nbar=0.02):
    """At fixed selected controls separate integration, cutoff and model changes."""
    variants = {
        'baseline_N8_m3': settings,
        'tighter_ODE': replace(settings, relative_tolerance=1e-11,
                               absolute_tolerance=1e-13, maximum_step_divisor=40.0),
        'Fock_N12': replace(settings, phonon_dim=12),
        'Fock_N16': replace(settings, phonon_dim=16),
        'sideband_m4_N16': replace(settings, phonon_dim=16, sideband_cutoff=4),
        'sideband_m5_N16': replace(settings, phonon_dim=16, sideband_cutoff=5),
        'paper_Trotter': replace(settings, solver='paper_trotter'),
        'paper_Trotter_half_dt': replace(settings, solver='paper_trotter', trotter_step_scale=2.0),
    }
    # The paper's published points and both main calibration choices.
    chosen = selected[selected.condition.isin(['Omega_2', 'Omega_4', 'standard_Favg_opt', 'hXX_zero'])]
    rows, baseline = [], {}
    for name, variant in variants.items():
        bench = Benchmark(Path(output)/'convergence', variant, nbar)
        for _, selected_row in chosen.iterrows():
            condition = selected_row['condition']
            if name in ('paper_Trotter', 'paper_Trotter_half_dt') and condition not in ('Omega_2', 'Omega_4'):
                continue
            row = bench.evaluate(float(selected_row.omega_mhz)).copy()
            chi = bench.channels[float(row['omega_per_second'])]
            if name == 'baseline_N8_m3':
                baseline[condition] = (row, chi)
            base_row, base_chi = baseline[condition]
            row.update(variant=name, condition=condition, **asdict(variant),
                       delta_standard_infidelity=row['standard_average_infidelity']-base_row['standard_average_infidelity'],
                       delta_paper_infidelity=row['paper_average_infidelity']-base_row['paper_average_infidelity'],
                       channel_ptm_change=4*np.linalg.norm(chi-base_chi))
            bound = distances.diamond_bounds(distances.chi_to_choi(chi-base_chi, PAULIS), 4)
            row['channel_diamond_change_upper'] = bound['diamond_upper_abs_choi']
            # Useful even if a cutoff moves the optimum: compare final fixed controls.
            for target, ref in [('pauli', distances.pauli_twirl_chi(chi)),
                                ('ideal', np.diag([1.0]+[0.0]*15))]:
                b = distances.diamond_bounds(distances.chi_to_choi(chi-ref, PAULIS), 4)
                row[f'{target}_diamond_lower'] = b['diamond_lower_choi']
                row[f'{target}_diamond_upper'] = b['diamond_upper_abs_choi']
            rows.append(row)
        # Variant-specific file, so that provenance for each setting is retained.
        bench.save_provenance()
        (Path(output)/'convergence'/f'provenance_{name}.json').write_text(
            (Path(output)/'convergence/provenance.json').read_text())
        pd.DataFrame(rows).to_csv(Path(output)/'convergence_checks.csv', index=False)
    return pd.DataFrame(rows)
