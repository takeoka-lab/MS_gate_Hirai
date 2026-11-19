"""Publication figure from saved, temperature-resolved amplitude calibration."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
folder = ROOT / 'results/calibration_channel_comparison'
pairs = pd.read_csv(ROOT / 'results/pauli_twirl_channel_distance/channel_distances_paired.csv').sort_values('n_bar')
pairs = pairs[pairs.n_bar.isin([0.01, 1., 2., 3., 4.])].copy()
top = pd.read_csv(folder / 'drive_calibration_hsca_top5.csv')
selected = top[(top.n_bar == 4) & (top.ranking == 'change')].sort_values('rank')
assert pairs.n_bar.tolist() == [0.01, 1., 2., 3., 4.]
assert len(selected) == 5
assert np.isfinite(selected[['before','after']]).all().all()
assert (selected[['before','after']] > 0).all().all()
for row in selected.itertuples():
    key = 'h_XX_rad_per_gate_' if row.sector == 'H' else f'taxonomy_{row.sector}_{row.pauli.replace(",", "_")}_'
    for stage in ['before','after']:
        assert np.isclose(getattr(row,stage), pairs.loc[pairs.n_bar == 4,key+stage].iloc[0])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
    'axes.labelsize':10,'legend.fontsize':8.5,'axes.linewidth':0.8,
    'pdf.fonttype':42,'ps.fonttype':42,'xtick.direction':'in','ytick.direction':'in'})
fig,(left,right)=plt.subplots(1,2,figsize=(7.2,3.25),layout='constrained')
styles=[('before','Nominal amplitude','#0072B2','o'),
        ('after','Calibrated amplitude','#D55E00','s')]
for stage,label,color,marker in styles:
    left.plot(pairs.n_bar,100*pairs['average_infidelity_'+stage],
              label=label,color=color,marker=marker,ms=4.5,lw=1.5,mfc='white')
left.set(xlabel=r'Initial mean phonon number $\bar n$',
         ylabel='Average gate infidelity (%)',xticks=[0,1,2,3,4],ylim=(0,1.42))
left.legend(loc='upper left',bbox_to_anchor=(0.01,0.90),frameon=False)
left.text(0.025,0.97,'(a)',transform=left.transAxes,va='top',fontweight='bold',fontsize=11)

x=np.arange(len(selected))
for i,row in enumerate(selected.itertuples()):
    right.plot([i,i],[row.before,row.after],color='0.7',lw=1.2,zorder=1)
for stage,label,color,marker in styles:
    right.plot(x,selected[stage],ls='none',marker=marker,color=color,
               ms=5.5,mfc='white',mew=1.3,label=label,zorder=3)
labels=[rf'${r.sector.lower()}_{{{r.pauli}}}$' for r in selected.itertuples()]
right.set_xticks(x,labels)
right.set_yscale('log')
right.set(ylabel='Generator coefficient per gate',ylim=(2.5e-4,0.18),xlim=(-0.5,4.5))
right.text(0.025,0.97,'(b)',transform=right.transAxes,va='top',fontweight='bold',fontsize=11)
right.text(0.97,0.97,r'$\bar n=4$',transform=right.transAxes,ha='right',va='top')
right.set_xlabel(r'Five largest changes, ranked by $|\Delta c_k|$')
for ax in [left,right]:
    ax.grid(axis='y',which='major',color='0.9',lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(top=True,right=True)
out=ROOT/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'drive_amplitude_calibration_100us.pdf',bbox_inches='tight',pad_inches=0.04)
fig.savefig(out/'drive_amplitude_calibration_100us.png',dpi=220,bbox_inches='tight',pad_inches=0.04)
plt.close(fig)
print('Saved PDF and PNG; all five coefficients cross-checked against channel summary.')
