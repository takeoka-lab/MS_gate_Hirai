"""Recreate the two-panel manuscript figure from saved QPT channels."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import logm
from drive_calibration_qpt_analysis import project_trace_normalized_chi_to_cptp
from ms_gate_functions import superoperator_to_ptm
from noise_error_structure import _decompose_generator_taxonomy

root = Path(__file__).resolve().parent
cache = root / 'results/chi_error_element_fit/noise_source_ablation'
terms = ['H:XX', 'S:XX', 'S:IX', 'S:XI', 'C:IX,XI']
rows = []
for condition in ['all', 'none']:
    for path in sorted(cache.glob(f'{condition}__nbar_*.npz')):
        with np.load(path) as data:
            channel, chi, _ = project_trace_normalized_chi_to_cptp(data['chi_trace_normalized'])
            nbar = float(data['n_bar'].item())
        row = dict(condition=condition, nbar=nbar, infidelity=4/5*(1-chi[0,0].real))
        if condition == 'all':
            fit = _decompose_generator_taxonomy(logm(superoperator_to_ptm(channel)).real)
            coefficients = {f'{s}:{m}':float(v) for (s,m),v in zip(fit['metadata'],fit['coefficients'])}
            row.update({term:coefficients[term] for term in terms})
        rows.append(row)
df = pd.DataFrame(rows).sort_values(['condition','nbar'])
on = df[df.condition == 'all'].set_index('nbar')
off = df[df.condition == 'none'].set_index('nbar')
assert on.index.equals(off.index)
assert np.isfinite(on[terms].values).all() and (on[terms].values > 0).all()
# These fixed five terms are selected by their ranks at nbar=2,3,4,
# not by a claim that they remain the five largest at every temperature.
rank_data = pd.read_csv(root/'results/noise_error_structure/nominal_rectangular_hsca_coefficients.csv')
for nb in [2,3,4]:
    part = rank_data[rank_data.nbar == nb].copy()
    part['magnitude'] = part.coefficient_per_gate.abs()
    selected = part.nlargest(5,'magnitude')
    assert set(selected.sector+':'+selected['mode']) == set(terms)

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.labelsize':10,
    'legend.fontsize':8.5, 'xtick.labelsize':9, 'ytick.labelsize':9,
    'axes.linewidth':0.8, 'pdf.fonttype':42, 'ps.fonttype':42,
    'xtick.direction':'in', 'ytick.direction':'in'})
fig, axes = plt.subplots(2,1,figsize=(5.1,5.8),sharex=True,layout='constrained')
a,b=axes
for series,label,color,marker,ls in [(on,'All sources on','#0072B2','o','-'),
                                    (off,'All sources off','#555555','s','--')]:
    a.plot(series.index,100*series.infidelity,color=color,marker=marker,ls=ls,lw=1.5,ms=4.5,mfc='white',label=label)
a.set_ylabel(r'Average gate infidelity (%)')
a.set_ylim(-0.3,10.8)
a.legend(loc='upper left',bbox_to_anchor=(0.14,1),frameon=False)
a.text(0.025,0.94,'(a)',transform=a.transAxes,ha='left',va='top',fontweight='bold',fontsize=12)
styles = [('#0072B2','o','-'),('#D55E00','s','--'),('#009E73','^','-'),('#CC79A7','v',':'),('#333333','x','-.')]
for i,(term,(color,marker,ls)) in enumerate(zip(terms,styles)):
    sector,mode=term.split(':')
    b.plot(on.index,on[term],color=color,marker=marker,ls=ls,lw=1.4,
           ms=[4.5,4.5,7,4.5,5][i],mfc='none',mew=0.9,label=rf'${sector.lower()}_{{{mode}}}$')
b.set_yscale('log')
b.set_ylabel('Generator coefficient per gate')
b.set_xlabel(r'Initial mean phonon number $\bar n$')
b.text(0.025,0.94,'(b)',transform=b.transAxes,ha='left',va='top',fontweight='bold',fontsize=12)
b.legend(loc='lower right',frameon=False,ncol=2,columnspacing=1.2,handlelength=2.4)
b.set_xticks([0,4,8,12,16,20]);b.set_xlim(-0.5,20.7)
for ax in axes:
    ax.tick_params(top=True,right=True)
    ax.grid(axis='y',which='major',color='0.9',lw=0.6)
    ax.set_axisbelow(True)
folder=root/'figures';folder.mkdir(exist_ok=True)
fig.savefig(folder/'nominal_generator_vs_nbar.pdf',bbox_inches='tight',pad_inches=0.04)
fig.savefig(folder/'nominal_generator_vs_nbar.png',dpi=220,bbox_inches='tight',pad_inches=0.04)
df.to_csv(folder/'nominal_generator_vs_nbar.csv',index=False)
plt.close(fig)
print(df.to_string(index=False))
