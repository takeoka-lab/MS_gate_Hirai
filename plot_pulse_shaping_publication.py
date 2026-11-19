"""Generate manuscript figures from matched 100-us pulse channels."""
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
ROOT=Path(__file__).resolve().parent
src=ROOT/'results/matched_pulse_calibration_sweep/100us'
out=ROOT/'figures';out.mkdir(exist_ok=True)
summary=pd.read_csv(src/'summary.csv')
distance=pd.read_csv(src/'diamond_distances.csv')
conditions=['standard_nominal','standard','smooth_detuning']
labels=['Nominal rectangular','Calibrated rectangular','Calibrated shaped']
colors=['#68737B','#D55E00','#009E73'];markers=['o','s','^']
for key in conditions:
    for frame in (summary,distance):
        assert frame[frame.condition==key].sort_values('n_bar').n_bar.tolist()==[0.01,1.,2.,3.,4.]
assert distance.sdp_accepted.all()

def coefficients(condition):
    with np.load(src/f'{condition}_4.npz') as data:
        assert np.isclose(float(data['gate_time_s']),100e-6)
        channel,_,_=project_trace_normalized_chi_to_cptp(data['chi'])
    gen=logm(superoperator_to_ptm(channel))
    assert np.linalg.norm(gen.imag)<1e-8
    fit=_decompose_generator_taxonomy(gen.real)
    return pd.Series(fit['coefficients'],index=[f'{s}:{m}' for s,m in fit['metadata']])
rect=coefficients('standard');shaped=coefficients('smooth_detuning')
assert abs(rect['H:XX'])<=2e-3 and abs(shaped['H:XX'])<=2e-3
change=shaped-rect
changed=change.loc[change.abs().nlargest(5).index]
remaining=shaped.loc[shaped.abs().nlargest(5).index]
pd.DataFrame({'rectangular':rect,'shaped':shaped,'change':change}).to_csv(out/'pulse_shaping_residual_errors_100us.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
 'legend.fontsize':8,'axes.linewidth':0.8,'pdf.fonttype':42,'ps.fonttype':42,
 'xtick.direction':'in','ytick.direction':'in'})
def finish(ax):
    ax.tick_params(top=True,right=True)
    ax.set_axisbelow(True)
def tag(ax,text):
    ax.text(0.02,0.96,text,transform=ax.transAxes,va='top',fontweight='bold',fontsize=11)
def term_label(term):
    s,m=term.split(':');return rf'${s.lower()}_{{{m}}}$'
def save(fig,stem):
    for ext in ['pdf','png']:fig.savefig(out/f'{stem}.{ext}',dpi=220,bbox_inches='tight',pad_inches=0.05)
    plt.close(fig)
fig=plt.figure(figsize=(7.2,5.7),layout='constrained')
grid=fig.add_gridspec(2,2,height_ratios=[1,1.05])
a=fig.add_subplot(grid[0,:]);b=fig.add_subplot(grid[1,0]);c=fig.add_subplot(grid[1,1])
for key,label,color,marker in zip(conditions,labels,colors,markers):
    d=summary[summary.condition==key].sort_values('n_bar')
    a.plot(d.n_bar,d.infidelity_percent,label=label,color=color,marker=marker,ms=4.5,mfc='white',lw=1.5)
a.set(xlabel=r'Initial mean phonon number $\bar n$',ylabel='Average gate infidelity (%)',xticks=[0,1,2,3,4],ylim=(0,1.42))
a.legend(loc='upper left',bbox_to_anchor=(0.10,0.98),frameon=False)
a.grid(axis='y',color='0.9',lw=.6);tag(a,'(a)')
for ax,series,title,label in [(b,changed,r'Coefficient changes, $\bar n=4$',r'$10^3\,\Delta c_k$ per gate'),
                              (c,remaining,r'Shaped-pulse residuals, $\bar n=4$',r'$10^3\,c_k$ per gate')]:
    y=np.arange(5)
    ax.barh(y,series.values*1e3,color=colors[2],height=.58)
    ax.set_yticks(y,[term_label(t) for t in series.index]);ax.invert_yaxis()
    ax.axvline(0,color='0.4',lw=.7)
    ax.set_xlabel(label);ax.set_title(title,fontsize=9,pad=8)
    ax.grid(axis='x',color='0.9',lw=.6)
    ax.margins(x=.22)
    for i,v in enumerate(series.values*1e3):
        ax.annotate(f'{v:+.3f}',(v,i),xytext=((4 if v>=0 else -4),0),textcoords='offset points',
                    ha=('left' if v>=0 else 'right'),va='center',fontsize=7.5)
# Panel labels above the axes avoid overlap with horizontal bars.
b.text(-.16,1.08,'(b)',transform=b.transAxes,fontweight='bold',fontsize=11)
c.text(-.16,1.08,'(c)',transform=c.transAxes,fontweight='bold',fontsize=11)
for ax in (a,b,c):finish(ax)
save(fig,'pulse_shaping_residual_errors_100us')

fig,ax=plt.subplots(figsize=(5.2,3.8),layout='constrained')
for key,label,color,marker in zip(conditions,labels,colors,markers):
    d=distance[distance.condition==key].sort_values('n_bar')
    values=d.diamond_distance_estimate.to_numpy()
    errors=np.maximum(0,np.vstack([values-d.diamond_sdp_lower.to_numpy(),d.diamond_sdp_upper.to_numpy()-values]))
    ax.errorbar(d.n_bar,values,yerr=errors,label=label,color=color,marker=marker,ms=5,mfc='white',lw=1.5,capsize=2)
ax.axhline(1e-3,color='0.35',ls='--',lw=1)
ax.text(3.95,1.1e-3,r'$\tau=10^{-3}$',ha='right',va='bottom',fontsize=9)
ax.set(yscale='log',xlabel=r'Initial mean phonon number $\bar n$',ylabel=r'Pauli approximation error $\epsilon_{\mathrm{P}}$',
       xticks=[0,1,2,3,4],ylim=(2e-4,1.2e-1),xlim=(-.12,4.12))
ax.legend(frameon=False,loc='upper left',fontsize=8)
ax.grid(axis='y',which='major',color='0.9',lw=.6);finish(ax)
save(fig,'pauli_distance_three_conditions_100us')
print('Generated two vector PDFs and previews. Coefficient changes:')
print(changed.to_string());print('Residuals:');print(remaining.to_string())
