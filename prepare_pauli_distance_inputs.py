"""Prepare the requested nbar=0.01,1,2,3,4 distance sweep without interpolation.

Reuse the existing comparison bundle at four points. At nbar=3 reuse the
baseline QPT, perform the same hXX amplitude feedback with fresh QPT when
--run-qpt is supplied, and store an independent input bundle for the distance
notebook. Original comparison archives and calibration summaries are untouched.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import logm

import chi_error_nbar_workflow as workflow
import drive_amplitude_calibration as calibration
import drive_calibration_qpt_analysis as qpt
import ms_gate_functions as mg
import noise_error_structure as structure


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/calibration_channel_comparison'
DESTINATION = ROOT / 'results/pauli_twirl_channel_distance/inputs'
NBARS = [0.01, 1.0, 2.0, 3.0, 4.0]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(chi, nbar, condition, amplitude_factor, source_path):
    superoperator, projected, _ = qpt.project_trace_normalized_chi_to_cptp(chi)
    observables = qpt.extract_pauli_generator_from_superoperator(superoperator)
    taxonomy = structure._decompose_generator_taxonomy(
        np.real(logm(np.asarray(mg.superoperator_to_ptm(superoperator), complex))))
    coefficients = dict(zip(taxonomy['metadata'], taxonomy['coefficients']))
    row = {
        'n_bar': nbar, 'condition': condition, 'A_factor': amplitude_factor,
        'source_path': str(source_path.relative_to(ROOT)),
        'h_XX_rad_per_gate': observables['hamiltonian_coefficients_rad_per_gate']['XX'],
        'average_infidelity': observables['average_infidelity'],
        'taxonomy_S_XX': float(coefficients[('S', 'XX')]),
        'taxonomy_S_IX': float(coefficients[('S', 'IX')]),
        'taxonomy_S_XI': float(coefficients[('S', 'XI')]),
        'taxonomy_C_IX_XI': float(coefficients[('C', 'IX,XI')]),
    }
    return projected, row


def main(run_qpt=False):
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE/'projected_chi_before_after.npz', allow_pickle=False) as a:
        projected_old = {key: a[key].copy() for key in a.files}
    with np.load(SOURCE/'post_gate_noise_chi_before_after.npz', allow_pickle=False) as a:
        raw_old = {key: a[key].copy() for key in a.files}
    summary = pd.read_csv(SOURCE/'channel_comparison_long.csv')
    raw_channels, projected_channels, rows = {}, {}, []
    for nbar in NBARS:
        if nbar == 3:
            continue
        i = np.flatnonzero(np.isclose(projected_old['n_bar'], nbar)).item()
        for condition in ('before', 'after'):
            raw_channels[nbar, condition] = raw_old[f'chi_{condition}'][i]
            projected_channels[nbar, condition] = projected_old[f'chi_{condition}'][i]
            rows.append(summary.loc[np.isclose(summary.n_bar, nbar)
                                    & summary.condition.eq(condition)].iloc[0].to_dict())

    baseline_path = ROOT/'results/chi_error_element_fit/exact_full_chi/error_chi_nbar_3p0.npz'
    with np.load(baseline_path, allow_pickle=False) as a:
        baseline = a['chi_trace_normalized'].copy()
    projected_channels[3.0, 'before'], before = analyze(baseline, 3.0, 'before', 1.0, baseline_path)
    raw_channels[3.0, 'before'] = baseline
    rows.append(before)
    params = workflow.default_simulation_params()
    params.update(show_progress=False, parallel_workers=4)
    amplitude, hxx = params['A'], before['h_XX_rad_per_gate']
    tolerance, history = 2e-3, []
    for iteration in range(1, 3):
        amplitude = calibration.amplitude_update_from_hxx(amplitude, hxx)
        signature = {'simulation_params': {**params, 'A': amplitude}, 'iteration': iteration,
                     'n_bar': 3.0, 'convention': 'undo_before_actual',
                     'baseline_sha256': sha256(baseline_path)}
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
        cache_path = DESTINATION / f'hxx_feedback_nbar_3_i{iteration:02d}_{digest}.npz'
        if not cache_path.exists():
            if not run_qpt:
                raise FileNotFoundError('Missing nbar=3 QPT; run this script with --run-qpt')
            print(f'nbar=3, iteration={iteration}: A={amplitude:.12g}, input hXX={hxx:.9g}', flush=True)
            started = time.perf_counter()
            point = qpt.calculate_error_channel_batch([3.0], params, {'A': amplitude})[0]
            point['metadata'].update(signature)
            point['metadata'].update(input_h_XX_rad_per_gate=hxx,
                                      elapsed_seconds=time.perf_counter()-started)
            qpt.save_qpt_point(cache_path, 3.0, f'hxx_feedback_iteration_{iteration}',
                               point['chi'], point['metadata'])
        with np.load(cache_path, allow_pickle=False) as a:
            after_chi = a['chi_trace_normalized'].copy()
        projected_after, after = analyze(after_chi, 3.0, 'after', amplitude/params['A'], cache_path)
        hxx = after['h_XX_rad_per_gate']
        history.append({'iteration': iteration, 'A': amplitude, 'A_factor': amplitude/params['A'],
                        'h_XX_rad_per_gate': hxx, 'source_path': str(cache_path.relative_to(ROOT)),
                        'cache_sha256': sha256(cache_path)})
        print(f'Finished: hXX={hxx:.9g}, infidelity={after["average_infidelity"]:.9g}', flush=True)
        if abs(hxx) <= tolerance:
            break
    if abs(hxx) > tolerance:
        raise RuntimeError('nbar=3 feedback did not reach the existing hXX tolerance')
    raw_channels[3.0, 'after'] = after_chi
    projected_channels[3.0, 'after'] = projected_after
    after['feedback_iteration'] = iteration
    rows.append(after)

    np.savez_compressed(DESTINATION/'projected_chi_before_after.npz', n_bar=NBARS,
        **{f'chi_{c}': np.stack([projected_channels[n, c] for n in NBARS]) for c in ('before', 'after')})
    np.savez_compressed(DESTINATION/'post_gate_noise_chi_before_after.npz', n_bar=NBARS,
        pauli_labels=raw_old['pauli_labels'], convention=raw_old['convention'],
        **{f'chi_{c}': np.stack([raw_channels[n, c] for n in NBARS]) for c in ('before', 'after')})
    pd.DataFrame(rows).sort_values(['n_bar', 'condition']).to_csv(
        DESTINATION/'channel_comparison_long.csv', index=False)
    sources = [SOURCE/'projected_chi_before_after.npz', SOURCE/'post_gate_noise_chi_before_after.npz',
               SOURCE/'channel_comparison_long.csv', baseline_path, Path(__file__)]
    (DESTINATION/'preparation_manifest.json').write_text(json.dumps({
        'n_bar_values': NBARS, 'method': 'Exact saved channels plus new hXX-feedback QPT at nbar=3; no interpolation',
        'new_simulation_params': params, 'hxx_tolerance': tolerance, 'nbar3_history': history,
        'source_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
    }, indent=2)+'\n')
    print('Input bundle:', DESTINATION, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-qpt', action='store_true', help='Calculate missing nbar=3 feedback QPT')
    main(parser.parse_args().run_qpt)
