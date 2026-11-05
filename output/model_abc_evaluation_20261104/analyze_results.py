"""Read-only analysis of the completed 6c0f70983e00c9e5 ABC simulation.

The saved simulation and source files are never altered. Extra no-noise A/B
checks use exact rotating-frame spectral propagation, not a new ODE sweep.
The fitted XX correction is an output unitary, NOT drive calibration.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ms_abc_evaluation_mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/ms_abc_evaluation_cache')

from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pauli_channel_distance as pcd

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/ld_second_order_rwa_vs_corrected_sideband/with_model_c/6c0f70983e00c9e5'
cfg = json.loads((SOURCE / 'config.json').read_text())
params = cfg['ab_run_config']['base_parameters']
summary = pd.read_csv(SOURCE / 'summary.csv')
data = np.load(SOURCE / 'channels.npz')
labels, paulis = pcd.pauli_basis(2)
assert labels == data['pauli_labels'].tolist()
idx = {s: i for i, s in enumerate(labels)}
XX, identity = paulis[idx['XX']], paulis[0]
target = (identity + 1j * XX) / np.sqrt(2)
undo = np.kron(target.T, target.conj().T)


def super_to_choi(s):
    return s.reshape(4, 4, 4, 4, order='F').transpose(2, 0, 3, 1).reshape(16, 16)


def infidelity(chi):
    return .8 * (1 - chi[0, 0].real)


def distance(a, b):
    return pcd.trace_norm(pcd.chi_to_choi(a - b, paulis)) / 8


def best_xx(chi):
    p = idx['XX']
    alpha = .5 * np.arctan2(2 * chi[0, p].imag, (chi[0, 0] - chi[p, p]).real)
    rotation = np.cos(alpha) * identity + 1j * np.sin(alpha) * XX
    transform = np.einsum('aij,ik,bkj->ab', paulis.conj(), rotation, paulis) / 4
    corrected = transform @ chi @ transform.conj().T
    optimal = ((chi[0, 0] + chi[p, p]).real / 2
               + np.hypot((chi[0, 0] - chi[p, p]).real / 2, chi[0, p].imag))
    assert abs(corrected[0, 0].real - optimal) < 1e-12
    return alpha, corrected


def probability_gap(chi, state, observable):
    delta = chi - pcd.pauli_twirl_chi(chi)
    return .5 * np.trace(observable @ pcd.apply_chi(delta, state, paulis)).real


rho00 = np.diag([1., 0., 0., 0.])
plus_i = np.array([1, 1j]) / np.sqrt(2)
yy_state = np.kron(plus_i, plus_i)
rho_yy = np.outer(yy_state, yy_state.conj())
channels, extended, top_rows = {}, [], []
max_reconstruction_error = 0.

for k in range(len(data['model'])):
    model, case, nbar = str(data['model'][k]), str(data['case'][k]), float(data['nbar'][k])
    chi = data['chi_error'][k]
    channels[model, case, nbar] = chi
    reconstructed = pcd.choi_to_chi(super_to_choi(data['superoperators'][k] @ undo), paulis)
    max_reconstruction_error = max(max_reconstruction_error, np.linalg.norm(chi-reconstructed))
    row = summary[(summary.model == model) & (summary['case'] == case) & (summary.nbar == nbar)].iloc[0]
    assert abs(infidelity(chi) - row.infidelity) < 1e-12
    alpha, corrected = best_xx(chi)
    metrics = pcd.distance_metrics(chi, paulis)
    corrected_metrics = pcd.distance_metrics(corrected, paulis)
    extended.append({
        **row.to_dict(), 'xx_error_angle_rad': -alpha,
        'best_xx_output_correction_rad': alpha,
        'infidelity_after_best_xx_output_correction': infidelity(corrected),
        'twirl_distance_after_best_xx_output_correction': corrected_metrics['choi_trace_distance'],
        'twirl_half_diamond_lower': metrics['diamond_lower_choi'],
        'twirl_half_diamond_upper': metrics['diamond_upper_abs_choi'],
        'witness_00_YX_plus_probability_gap': probability_gap(chi, rho00, paulis[idx['YX']]),
        'witness_plus_i_plus_i_ZZ_plus_probability_gap': probability_gap(chi, rho_yy, paulis[idx['ZZ']]),
        'chi_II_XX_real': chi[0, idx['XX']].real,
        'chi_II_XX_imag': chi[0, idx['XX']].imag,
        'chi_IX_XI_real': chi[idx['IX'], idx['XI']].real,
        'chi_IX_XI_imag': chi[idx['IX'], idx['XI']].imag,
    })
    for kind, positions in (
            ('diagonal', [(i, i) for i in range(1, 16)]),
            ('offdiagonal', [(i, j) for i in range(16) for j in range(i+1, 16)])):
        positions.sort(key=lambda ij: abs(chi[ij]), reverse=True)
        for i, j in positions[:6]:
            top_rows.append(dict(model=model, case=case, nbar=nbar, kind=kind,
                                 row=labels[i], column=labels[j], real=chi[i, j].real,
                                 imag=chi[i, j].imag, magnitude=abs(chi[i, j])))

assert max_reconstruction_error < 1e-12
extended = pd.DataFrame(extended).sort_values(['case', 'nbar', 'model'])
extended.to_csv(OUT / 'channel_diagnostics.csv', index=False)
pd.DataFrame(top_rows).to_csv(OUT / 'dominant_chi_components.csv', index=False)

noise_rows = []
individual = ['motional_heating', 'motional_dephasing', 'spin_dephasing', 'photon_scattering']
for model in 'ABC':
    for nbar in cfg['nbar_values']:
        base = channels[model, 'no_noise', nbar]
        for case in individual + ['all_noise']:
            chi = channels[model, case, nbar]
            delta_chi = chi - base
            noise_rows.append(dict(model=model, case=case, nbar=nbar,
                                   delta_infidelity=infidelity(chi)-infidelity(base),
                                   channel_distance_from_no_noise=distance(chi, base),
                                   delta_chi_II_XX_real=delta_chi[0, idx['XX']].real,
                                   delta_chi_II_XX_imag=delta_chi[0, idx['XX']].imag,
                                   delta_chi_IX_XI_real=delta_chi[idx['IX'], idx['XI']].real))
noise = pd.DataFrame(noise_rows)
noise.to_csv(OUT / 'noise_increments.csv', index=False)

# Exact finite-basis A/B reference at the closed-loop time T=2*pi/delta.
# In the Sx eigenbasis, s={2,0,0,-2}; U_s=exp[-i*T*(delta*n+s*g*(b+bdag))].
# R(T)=exp(i*delta*n*T)=I, so no time integrator is required.
h = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
hx = np.kron(h, h)
sx_basis_transform = np.kron(hx, hx)


def exact_first_sideband(model, nbar, dim):
    n = np.arange(dim)
    off = np.sqrt(np.arange(1, dim, dtype=float))
    if model == 'B':
        off *= 1 - params['eta']**2 * np.arange(1, dim) / 2
    values, vectors = eigh_tridiagonal(params['delta'] * n, 2 * params['A'] * off)
    unitary = (vectors * np.exp(-1j * 2*np.pi/params['delta'] * values)) @ vectors.T
    parity = (-1.)**n
    blocks = [unitary, np.eye(dim), np.eye(dim), parity[:, None]*unitary*parity[None, :]]
    p = (nbar / (nbar+1))**n
    p /= p.sum()
    weighted = np.stack([(u*np.sqrt(p)[None, :]).ravel() for u in blocks])
    coherence = weighted @ weighted.conj().T
    assert np.max(abs(np.diag(coherence) - 1)) < 1e-12
    superop = (sx_basis_transform * coherence.reshape(-1, order='F')) @ sx_basis_transform.T
    chi = pcd.choi_to_chi(super_to_choi(superop @ undo), paulis)
    assert np.linalg.eigvalsh(pcd.chi_to_choi(chi, paulis)).min() > -1e-11
    return chi


checks = []
for model in 'AB':
    for nbar in cfg['nbar_values']:
        dim = cfg['ab_run_config']['phonon_dimensions'][str(nbar)]
        saved = channels[model, 'no_noise', nbar]
        small = exact_first_sideband(model, nbar, dim)
        large = exact_first_sideband(model, nbar, 140)
        middle = exact_first_sideband(model, nbar, 100)
        _, corrected = best_xx(large)
        checks.append(dict(model=model, nbar=nbar, saved_phonon_dim=dim,
                           saved_infidelity=infidelity(saved),
                           exact_same_cutoff_infidelity=infidelity(small),
                           exact_dim100_infidelity=infidelity(middle),
                           exact_dim140_infidelity=infidelity(large),
                           saved_vs_exact_same_cutoff_choi_distance=distance(saved, small),
                           exact_same_cutoff_vs_dim140_choi_distance=distance(small, large),
                           exact_dim100_vs_dim140_choi_distance=distance(middle, large),
                           exact_dim140_infidelity_after_best_xx=infidelity(corrected)))
checks = pd.DataFrame(checks)
checks.to_csv(OUT / 'independent_ab_cutoff_check.csv', index=False)

metadata = dict(source=str(SOURCE), config_sha256=hashlib.sha256((SOURCE/'config.json').read_bytes()).hexdigest(),
                saved_channels=90, amplitude_calibrated=False,
                independent_check='closed-loop A/B spectral propagation; original cutoff vs 100 vs 140',
                no_new_model_c_propagation=True, no_saved_results_modified=True,
                max_saved_chi_reconstruction_error=max_reconstruction_error,
                numpy_version=np.__version__,
                output_unitary_correction_is_not_amplitude_calibration=True)
(OUT/'analysis_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
colors = {'A': '#3467a9', 'B': '#cf6b27', 'C': '#148b7a'}
fig, axes = plt.subplots(2, 2, figsize=(11, 8.3), constrained_layout=True)
for model in 'ABC':
    no = extended.query("model == @model and case == 'no_noise'").sort_values('nbar')
    alln = extended.query("model == @model and case == 'all_noise'").sort_values('nbar')
    axes[0, 0].semilogy(no.nbar, no.infidelity, 'o-', color=colors[model], label=model)
    axes[0, 1].plot(alln.nbar, alln.twirl_choi_trace_distance, 'o-', color=colors[model], label=model)
    axes[1, 0].plot(no.nbar, no.infidelity, 'o-', color=colors[model], label=f'{model}: saved')
    axes[1, 0].plot(no.nbar, no.infidelity_after_best_xx_output_correction, '--', color=colors[model], label=f'{model}: best XX correction')
    mot = noise.query("model == @model and case == 'motional_dephasing'").sort_values('nbar')
    axes[1, 1].plot(mot.nbar, mot.delta_infidelity, 'o-', color=colors[model], label=model)
axes[0, 0].set(title='No dissipation: saved infidelity', ylabel='Average gate infidelity')
axes[0, 0].text(.03, .97, 'A/C: cutoff affects the small residuals', va='top', transform=axes[0, 0].transAxes, fontsize=9)
axes[0, 1].set(title='All noise: distance to Pauli twirl', ylabel='Normalized Choi trace distance')
axes[1, 0].set(title='No dissipation: XX output correction\nDiagnostic only; no drive recalibration', ylabel='Average gate infidelity')
axes[1, 1].set(title='Motional dephasing: added infidelity', ylabel='r(motional dephasing) - r(no noise)')
for ax in axes.flat:
    ax.set_xlabel('Mean phonon number')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='lower right' if ax is axes[0, 0] else 'best')
fig.savefig(OUT/'abc_evaluation.png', dpi=170)
fig.savefig(OUT/'abc_evaluation.pdf')
plt.close(fig)

print('NO NOISE AND ALL NOISE')
print(extended.query("case in ['no_noise','all_noise']")[['model','case','nbar','infidelity','xx_error_angle_rad','infidelity_after_best_xx_output_correction','twirl_choi_trace_distance','twirl_distance_after_best_xx_output_correction']].to_string(index=False))
print('INDEPENDENT CUTOFF CHECK')
print(checks.to_string(index=False))
print('NOISE INCREMENTS AT NBAR=4')
print(noise.query('nbar == 4').to_string(index=False))
print('WITNESSES AT NBAR=4')
print(extended.query("nbar == 4 and case in ['no_noise','motional_dephasing','all_noise']")[['model','case','twirl_half_diamond_lower','twirl_half_diamond_upper','witness_00_YX_plus_probability_gap','witness_plus_i_plus_i_ZZ_plus_probability_gap','chi_II_XX_real','chi_II_XX_imag','chi_IX_XI_real']].to_string(index=False))
