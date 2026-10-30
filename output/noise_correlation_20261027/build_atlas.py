"""Plot the saved MS-gate noise data; never run or modify simulations.

Run with the project's Python (numpy, pandas, matplotlib). Then run
build_pdf.py with a Python containing reportlab. All numbers come from CSVs;
the manifest records file hashes, row counts and figure-level provenance.
"""
from pathlib import Path
import os
import json
import hashlib
import itertools
import html

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ms_noise_atlas_mpl")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
DATA = OUT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
R = "results/noise_rate_nbar_sweep/"
G = "results/noise_error_structure/"
E = "results/error_generator_rate_nbar/top3_pairwise_generator/"
N = [0.01, 1., 2., 3., 4.]
M = [0., .5, 1., 2., 4.]
S = ["motional_heating", "motional_dephasing", "spin_dephasing", "photon_scattering"]
EN = dict(zip(S, ["Heating", "Motional dephasing", "Spin dephasing", "Photon scattering"]))
JP = dict(zip(S, ["運動加熱", "運動位相緩和", "スピン位相緩和", "光散乱"]))
EN["none"] = "Initial thermal state / no explicit noise"
JP["none"] = "初期平均フォノン数（明示的な散逸なし）"
COL = dict(zip(S, ["#d97706", "#007f86", "#8154a2", "#2874a6"]))
COL["none"] = "#25334b"
NC = plt.cm.viridis(np.linspace(.08, .85, 5))
PAIRS = [f"{a}__{b}" for a, b in itertools.combinations(S, 2)]
PC = dict(zip(PAIRS, ["#bd4c33", "#cc9b20", "#637d39", "#007f86", "#3967b0", "#8154a2"]))
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
    "axes.titlesize":12, "axes.labelsize":11, "axes.spines.top":False,
    "axes.spines.right":False, "axes.grid":True, "grid.alpha":.18,
    "axes.axisbelow":True, "legend.frameon":False, "legend.fontsize":9,
    "figure.facecolor":"white", "savefig.facecolor":"white", "svg.fonttype":"none"})
MANIFEST = {}
PAGES = []
CHECKS = {}

def read(path):
    p = ROOT / path
    d = pd.read_csv(p)
    MANIFEST[path] = {"rows":len(d), "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    return d

def cfg(path):
    p = ROOT / path
    MANIFEST[path] = {"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    return json.loads(p.read_text())

def new(rows=1, cols=2, size=None):
    f, a = plt.subplots(rows, cols, figsize=size or (13.6, 7.4), squeeze=False)
    f.subplots_adjust(left=.085, right=.96, bottom=.14, top=.82, wspace=.32, hspace=.48)
    return f, a

def save(f, slug, title, subtitle, caption, paths, section):
    num = len(PAGES)+1
    f.text(.055,.95,f"{num:02d}  {title}",fontsize=19,weight="bold",color="#192b42",va="top")
    f.text(.055,.89,subtitle,fontsize=11,color="#526176",va="top")
    f.text(.055,.035,"MS GATE  |  SAVED-DATA ATLAS  |  2026-10-27",fontsize=8,color="#68788b")
    name = f"{num:02d}_{slug}"
    for ext in ["png", "svg"]:
        f.savefig(FIG/f"{name}.{ext}",dpi=180)
    plt.close(f)
    PAGES.append(dict(number=num, slug=slug, title=title, subtitle=subtitle,
                      caption=caption, sources=paths, section=section,
                      png=f"figures/{name}.png", svg=f"figures/{name}.svg"))

def line(ax,x,y,label=None,color=None,**kw):
    ax.plot(x,y,marker="o",ms=4,lw=1.8,label=label,color=color,**kw)

def nb(ax,y):
    ax.set_xlabel(r"Initial mean phonon number $\bar n$")
    ax.set_ylabel(y); ax.set_xticks(N,["0.01","1","2","3","4"])

def rate(ax,y):
    ax.set_xlabel(r"Rate multiplier $m=\Gamma/\Gamma_{\rm nom}$")
    ax.set_ylabel(y); ax.set_xticks(M)

def pairlabel(pair, short=False):
    names={"motional_heating":"H", "motional_dephasing":"M", "spin_dephasing":"S", "photon_scattering":"P"} if short else EN
    return " x ".join(names[s] for s in pair.split("__"))

def symmap(ax, arr, xs, ys, label, vmax=None, annotate=False):
    vmax = vmax or max(float(np.nanmax(np.abs(arr))),1e-15)
    im=ax.imshow(arr, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),aspect="auto")
    ax.set_xticks(range(len(xs)), xs); ax.set_yticks(range(len(ys)),ys);ax.grid(False)
    if annotate:
        for j in range(arr.shape[0]):
            for i in range(arr.shape[1]):
                v=arr[j,i]
                if np.isfinite(v): ax.text(i,j,f"{v:.2f}",ha="center",va="center",fontsize=8,color="white" if abs(v)>.65*vmax else "#263549")
    ax.figure.colorbar(im,ax=ax,shrink=.85,pad=.03,label=label)
    return im

base = cfg(R+"config.json")
for pair in PAIRS:
    pcfg=cfg(R+f"all_pairwise_5x5_grids/{pair}/config.json")
    for key in ["A","delta","eta","t_gate_phys","use_full_order","time_points"]:
        assert pcfg["base_parameters"][key] == base["base_parameters"][key], (pair,key)

paths=[R+"all_noise_zero_vs_nominal.csv", R+"four_noise_reconstruction/four_noise_single_components.csv",R+"four_noise_reconstruction/four_noise_reconstruction.csv"]
b, singles, recon = [read(p) for p in paths]
gridpath=R+"all_pairwise_5x5_grids/all_pairwise_grid_interactions.csv"
grid=read(gridpath)
assert len(grid)==750 and not grid.duplicated(["pair_id","nbar","multiplier_x","multiplier_y"]).any()
assert set(grid.pair_id)==set(PAIRS)
recomputed=grid.grid_infidelity-grid.x_only_infidelity-grid.y_only_infidelity+grid.zero_infidelity
CHECKS["max_Cij_formula_error"]=float(np.max(abs(recomputed-grid.interaction_infidelity)))
assert CHECKS["max_Cij_formula_error"]<1e-13
CHECKS["max_zero_axis_Cij"]=float(grid.loc[(grid.multiplier_x==0)|(grid.multiplier_y==0),"interaction_infidelity"].abs().max())
assert CHECKS["max_zero_axis_Cij"]<1e-13
active=grid[(grid.multiplier_x>0)&(grid.multiplier_y>0)]
CHECKS["positive_active_Cij_count"]=int((active.interaction_infidelity>0).sum())
for pair in PAIRS:
    assert len(grid[grid.pair_id==pair])==125
singlemat=singles.pivot(index="nbar",columns="noise_source",values="single_noise_contribution").loc[N,S]
thermal=singlemat.subtract(singlemat.loc[.01],axis="columns")
thermal.to_csv(DATA/"thermal_mixed_response.csv")

# 1: coherent overview, without mixing initial-state and rate conventions.
f,a=new(2,2)
line(a[0,0],b.nbar,b.nominal_infidelity*1e3,"All four nominal",COL["none"])
line(a[0,0],b.nbar,b.all_noise_zero_infidelity*1e3,"All four off","#9b9fa7")
nb(a[0,0],r"Infidelity $I$ ($10^{-3}$)");a[0,0].legend();a[0,0].set_title("Total error and thermal baseline")
for s in S: line(a[0,1],N,singlemat[s]*1e3,EN[s],COL[s])
nb(a[0,1],r"$\Delta I_i=I_i-I_0$ ($10^{-3}$)");a[0,1].legend(ncol=2);a[0,1].set_title("Each source alone, nominal rate")
for s in S: line(a[1,0],N,thermal[s]*1e3,EN[s],COL[s])
nb(a[1,0],r"$\Delta I_i(\bar n)-\Delta I_i(0.01)$ ($10^{-3}$)");a[1,0].set_title("Initial-temperature x source response")
line(a[1,1],N,recon.first_order_relative_residual_to_noise_penalty*100,"Single-source sum",COL["none"])
line(a[1,1],N,recon.pairwise_relative_residual_to_noise_penalty*100,"Including all six pairs",COL[S[1]])
nb(a[1,1],"(Actual - prediction) / noise penalty (%)");a[1,1].legend();a[1,1].axhline(0,color="gray",lw=.7);a[1,1].set_title("Small departure from additivity")
save(f,"overview","Five inputs: the current error budget","100 us gate | eta=0.1 | finite-LD corrected model | markers are saved calculations",
     "初期平均フォノン数は初期状態のパラメータで、他の4項は散逸レート。右上は他の散逸をゼロにした単独増分。左下はその増分から nbar=0.01 の値を引いた温度との混合応答。右下の分母は全ノイズによる増分 I_all-I_0。主要な温度依存は無散逸の基準誤差と運動位相緩和に現れる。線は保存点を結ぶ表示であり、追加計算や補間精度の保証ではない。",paths,"概要")

# 2: all available isolated rate-axis points, deduplicated from pair grids.
qpath=R+"all_pairwise_5x5_grids/all_pairwise_grid_qpt_summary.csv"
q=read(qpath);f,a=new(2,2)
zero_check=q[q.is_all_noise_zero].merge(b,on="nbar",validate="one_to_one")
CHECKS["max_grid_vs_baseline_zero_difference"]=float((zero_check.infidelity-zero_check.all_noise_zero_infidelity).abs().max())
assert CHECKS["max_grid_vs_baseline_zero_difference"]<1e-13
nominal=dict(zip(S,[10.,18.,10/3,4.]))
solo=[]
for ax,s in zip(a.flat,S):
    mask=np.ones(len(q),dtype=bool)
    for other in S:
        if other!=s: mask &= q[other+"_s^-1"].eq(0)
    d=q[mask].copy();d["multiplier"]=d[s+"_s^-1"]/nominal[s]
    d=d.merge(b[["nbar","all_noise_zero_infidelity"]],on="nbar",validate="many_to_one")
    d["increment"]=d.infidelity-d.all_noise_zero_infidelity
    d["noise_source"]=s;solo.append(d)
    for n,c in zip(N,NC):
        z=d[d.nbar==n].sort_values("multiplier")
        assert len(z)==5
        line(ax,z.multiplier,z.increment*1e3,f"{n:g}",c)
    ax.set_title(f"{EN[s]}  ({nominal[s]:g} /s at m=1)");rate(ax,r"$I_i-I_0$ ($10^{-3}$)");ax.legend(title=r"$\bar n$",ncol=3)
pd.concat(solo).to_csv(DATA/"single_source_rate_axes.csv",index=False)
save(f,"isolated_rate_axes","Isolated source response versus rate","Other three rates = 0 | 25 saved points per source", "全ペア格子に共通の単独軸から重複なく抽出した100点。運動位相緩和だけが強い温度依存を持つ。レートの物理単位は各パネルの基準値で換算する。",[qpath,paths[0]],"単独ノイズと温度")

spath=R+"summary.csv";sweep=read(spath);f,a=new(2,2)
for ax,s in zip(a.flat,S):
    for m,c in zip(M,NC):
        d=sweep[(sweep.noise_source==s)&(sweep.multiplier==m)].sort_values("nbar")
        line(ax,d.nbar,d.infidelity*1e3,f"m={m:g}",c)
    ax.set_title(EN[s]);nb(ax,r"Total infidelity ($10^{-3}$)");ax.legend(ncol=3)
save(f,"nominal_background_sweep","Rate sweeps with the other noises present","Other three rates = nominal | this is a different background from the isolated-source plots", "既存ノートブックの100点。各パネルで指定レートだけを変え、他の3レートは基準値に固定。m=0 でも他の散逸と有限LD補正が残るので、単独寄与や無散逸とは異なる。",[spath],"単独ノイズと温度")

f,a=new(2,2)
for j,m in enumerate([1.,4.]):
    d=grid[(grid.multiplier_x==m)&(grid.multiplier_y==m)]
    for p in PAIRS:
        z=d[d.pair_id==p].sort_values("nbar")
        line(a[0,j],z.nbar,z.interaction_infidelity*1e6,pairlabel(p,True),PC[p])
        line(a[1,j],z.nbar,z.relative_interaction_to_pair_noise*100,pairlabel(p,True),PC[p])
    a[0,j].set_title(f"Both rates = {m:g} x nominal");nb(a[0,j],r"Signed $C^{(I)}_{ij}$ ($10^{-6}$)");a[0,j].legend(ncol=3)
    nb(a[1,j],r"$100 C^{(I)}_{ij}/(I_{ij}-I_0)$ (%)")
save(f,"pairwise_summary","All six pairs: signed nonadditivity","C = I_ij - I_i - I_j + I_0 | H: heating, M: motional dephasing, S: spin, P: scattering",
     "他の2散逸をゼロにしたペア比較。負のCは単独増分の和より小さいことを表す。統計的に相関した環境浴の証拠やPearson相関係数ではない。下段はペアによる誤差増分で割った値。微小な混合差分の確定桁数には直接の収束確認が必要。",[gridpath],"全6ペア")

# 5-10: complete 750-point grid; common scale within each row, exact cell labels.
for p in PAIRS:
    d=grid[grid.pair_id==p]; sx,sy=p.split("__")
    f,a=new(2,5,size=(17.5,8.2));f.subplots_adjust(left=.055,right=.91,wspace=.2,hspace=.5,bottom=.16,top=.8)
    vI=d.grid_infidelity.max()*1e3;vC=d.interaction_infidelity.abs().max()*1e6
    for k,n in enumerate(N):
        z=d[d.nbar==n]
        for row,col,scale,vmax,cmap in [(0,"grid_infidelity",1e3,vI,"YlGnBu"),(1,"interaction_infidelity",1e6,vC,"RdBu_r")]:
            mat=z.pivot(index="multiplier_y",columns="multiplier_x",values=col).reindex(index=M,columns=M).values*scale
            ax=a[row,k];norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax) if row else None
            im=ax.imshow(mat,origin="lower",cmap=cmap,aspect="equal",norm=norm,**({} if row else {"vmin":0,"vmax":vmax}))
            ax.set_xticks(range(5),["0","0.5","1","2","4"]);ax.set_yticks(range(5),["0","0.5","1","2","4"]);ax.grid(False)
            ax.set_xlabel("Heating m" if sx==S[0] else "Spin m" if sx==S[2] else "Motional deph. m")
            if k==0: ax.set_ylabel(EN[sy]+" m")
            if row==0: ax.set_title(rf"$\bar n={n:g}$")
            for jj in range(5):
                for ii in range(5):
                    v=mat[jj,ii]; ax.text(ii,jj,f"{v:.2f}",ha="center",va="center",fontsize=7.2,color="white" if (abs(v)>.65*vmax) else "#243246")
            if k==4:
                ca=f.add_axes([.93,.53 if row==0 else .19,.012,.24])
                f.colorbar(im,cax=ca,label=(r"$I_{ij}$ ($10^{-3}$)" if row==0 else r"$C^{(I)}_{ij}$ ($10^{-6}$)"))
    save(f,p,"Pair grid: "+pairlabel(p),"Top: total infidelity | Bottom: signed interaction | all 125 saved points | equal cell spacing denotes sampled levels",
         f"{JP[sx]}×{JP[sy]}の全125点。上段は総infidelity、下段は混合差分C。温度間の比較ができるよう、各段で同じ色範囲を使用。レート倍率0、0.5、1、2、4は不等間隔の数値だが、表形式としてセルを等間隔に表示。値は表示桁で丸め、元の精度はCSVに保持。",[gridpath],"全6ペア")

f,a=new(1,3,size=(15,6.4))
for col,label,c in [("full_noise_infidelity","Saved full QPT",COL["none"]),("first_order_prediction","Single-source sum",COL[S[0]]),("pairwise_prediction","Plus six pairs",COL[S[1]])]:
    line(a[0,0],recon.nbar,recon[col]*1e3,label,c)
nb(a[0,0],r"Infidelity ($10^{-3}$)");a[0,0].legend()
for col,label,c in [("first_order_residual","Single-source sum",COL[S[0]]),("pairwise_residual","Plus six pairs",COL[S[1]])]:
    line(a[0,1],recon.nbar,recon[col]*1e6,label,c)
nb(a[0,1],r"Actual - prediction ($10^{-6}$)");a[0,1].axhline(0,color="gray",lw=.7)
comp=read(R+"four_noise_reconstruction/four_noise_pair_components.csv")
for p in PAIRS:
    d=comp[comp.pair_id==p].sort_values("nbar");line(a[0,2],d.nbar,d.interaction_infidelity*1e6,pairlabel(p,True),PC[p])
nb(a[0,2],r"Each nominal pair $C^{(I)}_{ij}$ ($10^{-6}$)");a[0,2].legend(ncol=2)
save(f,"four_noise_reconstruction","Reconstructing the four-noise calculation","I_pred = I_0 + sum(Delta I_i) + sum(C_ij) | all rates nominal",
     "単独寄与の和と6ペア補正から、4散逸を同時に入れた計算を再構成。中央は大きく拡大した差であり、ペア補正後の約1e-7の一致を数値精度の証明とは扱わない。",[paths[2],R+"four_noise_reconstruction/four_noise_pair_components.csv"],"4ノイズ同時作用")

vp=R+"four_noise_rate_validation/model_predictions.csv";mp=R+"four_noise_rate_validation/model_comparison_metrics.csv"
val=read(vp);metrics=read(mp);f,a=new(1,3,size=(15,6.4))
models=[("single_only_additive","Single sum",COL[S[0]]),("strict_pairwise_surface","Pair surfaces",COL[S[1]]),("reduced_bilinear_pairwise","Bilinear pairs",COL[S[3]])]
for mod,label,c in models:
    d=metrics[(metrics.scope=="nbar")&(metrics.model==mod)].sort_values("nbar")
    line(a[0,0],d.nbar,d.rmse*1e6,label,c)
    line(a[0,1],d.nbar,d.rmse_relative_to_noise_penalty*100,label,c)
    a[0,2].scatter(val.actual_infidelity*1e3,val[mod+"_residual"]*1e6,label=label,color=c,s=18,alpha=.75)
nb(a[0,0],r"RMSE ($10^{-6}$)");nb(a[0,1],"RMSE / noise penalty (%)");a[0,0].legend()
a[0,2].set_xlabel(r"Actual total infidelity ($10^{-3}$)");a[0,2].set_ylabel(r"Actual - model ($10^{-6}$)");a[0,2].axhline(0,color="gray",lw=.7)
save(f,"four_rate_validation","Four-rate validation away from the nominal point","60 saved validations = 12 rate vectors x 5 temperatures | residual = actual - prediction",
     "4レート同時変更の60検証点。単独加算、ペア曲面、ペア双線形モデルを比較。基準点の再構成と異なり、任意レート側ではペア補正後にも残差がある。ここでの相対RMSEはノイズ増分を基準とする。",[vp,mp],"4ノイズ同時作用")

hp=R+"four_noise_nbar_critical_search/high_nbar_model_predictions.csv";cp=R+"four_noise_nbar_critical_search/nbar_critical_curve.csv"
high=read(hp);crit=read(cp);f,a=new(1,2)
zpath=R+"four_noise_nbar_critical_search/all_noise_zero/validation_qpt_summary.csv";zh=read(zpath)
line(a[0,0],b.nbar,b.all_noise_zero_infidelity,"Training range",COL["none"]);line(a[0,0],zh.nbar,zh.infidelity,"High-temperature saved QPT",COL[S[0]])
pred=high.groupby("nbar",as_index=False).first().sort_values("nbar")
line(a[0,0],pred.nbar,pred.predicted_zero_infidelity,"Low-n extrapolation",COL[S[3]],ls="--")
a[0,0].set_xlabel(r"$\bar n$");a[0,0].set_ylabel("No-noise infidelity");a[0,0].legend()
for phase,d in crit.groupby("phase"):
    d=d.sort_values("nbar");line(a[0,1],d.nbar,d.rmse_relative_to_noise_penalty*100,str(phase))
a[0,1].axhline(1,color="#bd4c33",ls="--",label="1% criterion");a[0,1].set_yscale("log");a[0,1].set_xlabel(r"$\bar n$");a[0,1].set_ylabel("Model RMSE / noise penalty (%)");a[0,1].legend()
save(f,"high_nbar","High-temperature extension and extrapolation limits","Saved validation extends to nbar=20; complete six-pair grids extend only to nbar=4",
     "高温域にも無散逸16点と4ノイズ同時作用84点がある。低温データからの外挿は、保存された1%判定でnbar=4が最後の合格点、5が最初の不合格点。これはそのモデル・離散検証条件の適用限界であり、物理的な相転移や普遍的な臨界温度ではない。",[hp,cp,zpath],"4ノイズ同時作用")

fp=G+"noise_free_fock_generator_comparison.csv";fo=read(fp);f,a=new(1,3,size=(15,6.4))
for ax,cols,ylab in zip(a.flat,[("qpt_H_XX_rad_per_gate","fock_h_XX_rad_per_gate"),("qpt_S_XX_per_gate","fock_phase_dispersion_gamma_per_gate"),("qpt_C_IX_XI_per_gate","fock_residual_motion_gamma_per_gate")],[r"$H_{XX}$ (rad/gate)",r"$S_{XX}$ (/gate)",r"$C_{IX,XI}$ (/gate)"]):
    line(ax,fo.nbar,fo[cols[0]],"QPT",COL["none"]);line(ax,fo.nbar,fo[cols[1]],"Fock-resolved prediction",COL[S[0]],ls="--");nb(ax,ylab);ax.legend()
a[0,0].set_title("Thermal mean angle");a[0,1].set_title("XX angle dispersion");a[0,2].set_title("Residual motion / collective X")
save(f,"thermal_mechanism","Initial phonon number: three resolved error components","All four explicit dissipators off | effective generator K_0 = log(R_0)",
     "初期熱分布だけの効果を、平均XX角のずれ、XX角のばらつき、残留運動による集団X成分に分ける。Fock分解予測とQPTの直接比較。無散逸でも有限LD補正により誤差が残る。この図のS_XXと集団XのC_IX,XIは別成分。",[fp],"誤差生成子の相関")

tp=G+"full_taxonomy_coefficients.csv";op=G+"full_taxonomy_overall_summary.csv"
tax=read(tp);overall=read(op);f,a=new(2,3,size=(15,8))
for ax,s in zip(a.flat,["none"]+S):
    d=overall[overall.noise_source==s].sort_values("nbar")
    for sec,c in zip("HSCA",["#25334b","#007f86","#c36b2a","#8b73a8"]):
        line(ax,d.nbar,d[f"{sec}_sector_frobenius_norm"],sec,c)
    nb(ax,"Sector Frobenius norm");ax.set_yscale("log");ax.set_title(EN[s]);ax.legend(ncol=4)
a[1,2].axis("off");a[1,2].text(0,.9,"H: coherent Hamiltonian\nS: diagonal Pauli stochastic terms\nC: symmetric cross-Pauli terms\nA: imaginary cross-Pauli sector\n\nNorms are not probabilities.\nNo-noise panel: K_0\nSource panels: K_i - K_0",va="top",linespacing=1.6,fontsize=12)
save(f,"hsca_sectors","Generator structure for all five inputs","Complete 240-coefficient H/S/C/A decomposition | source responses are signed generator differences",
     "各散逸の基準レートでの応答をΔK_i=K_i-K_0として分解。初期熱状態のパネルのみK_0そのもの。縦軸は各セクターが作る行列のFrobeniusノルムで、確率や加算的誤差予算ではない。H/S/C/Aの係数全6000行を元CSVから参照できる。",[tp,op],"誤差生成子の相関")

# Per-input correlation atlas: retain signed coefficients and all 15x15 S/C entries.
paulis=[x+y for x in "IXYZ" for y in "IXYZ" if x+y!="II"]
for s in ["none"]+S:
    d=tax[tax.noise_source==s];f,a=new(2,2,size=(14,8))
    for ax,sec,k in [(a[0,0],"S",6),(a[0,1],"C",6)]:
        z=d[d.sector==sec];ranking=z.groupby("mode").coefficient_per_gate.apply(lambda v:v.abs().max()).sort_values(ascending=False)
        chosen=ranking.head(k).index
        for mode in chosen:
            v=z[z["mode"]==mode].sort_values("nbar");line(ax,v.nbar,v.coefficient_per_gate*1e3,mode)
        nb(ax,f"{sec} coefficient ($10^{{-3}}$/gate)");ax.legend(ncol=3);ax.set_title(f"Top {k} {sec} modes (ranked over all temperatures)");ax.axhline(0,color="gray",lw=.7)
    z=d[(d.sector=="H")&(d["mode"]=="XX")].sort_values("nbar")
    line(a[1,0],z.nbar,z.coefficient_per_gate*1e6,r"$H_{XX}$",COL[s]);nb(a[1,0],r"$H_{XX}$ ($10^{-6}$ rad/gate)");a[1,0].axhline(0,color="gray",lw=.7)
    mat=np.zeros((15,15));z=d[d.nbar==4]
    for row in z.itertuples():
        if row.sector=="S": mat[paulis.index(row.mode),paulis.index(row.mode)]=row.coefficient_per_gate
        if row.sector=="C":
            p1,p2=row.mode.split(",");i,j=paulis.index(p1),paulis.index(p2);mat[i,j]=mat[j,i]=row.coefficient_per_gate
    symmap(a[1,1],mat*1e3,paulis,paulis,r"$10^{-3}$/gate");a[1,1].tick_params(labelsize=7);a[1,1].tick_params(axis="x",rotation=90);a[1,1].set_title(r"Full S/C coefficient matrix at $\bar n=4$")
    tiny_note=" スピン位相緩和・光散乱のH_XXは約1e-8 rad/gateの微小量で、この図だけで非ゼロの物理応答を確定しない。" if s in S[2:] else ""
    save(f,"modes_"+s,"Correlation modes: "+EN[s],"No explicit noise: K_0 | isolated source: Delta K_i at its nominal rate | coefficient signs are retained",
         f"{JP[s]}の主要成分。上段は温度全域の最大絶対値で選んだS/C各6モード、左下はH_XX、右下はnbar=4における15×15全S/C行列。右下の色範囲はこの入力内で設定しているため、他ページとの強度比較にはカラーバーを使う。ΔKの負係数は符号付き差分であり、単独の物理的崩壊率とは限らない。"+tiny_note,[tp],"誤差生成子の相関")

rp="results/rate_hsca_structure/";rs=read(rp+"rate_structure_metrics.csv");rl=read(rp+"rate_matrix_linearity.csv");f,a=new(2,2)
for s in S:
    d=rs[(rs.noise_source==s)&(rs.nbar==4)&(rs.rate_multiplier>0)].sort_values("rate_multiplier")
    line(a[0,0],d.rate_multiplier,1-d.stochastic_shape_cosine_to_nominal,EN[s],COL[s])
    z=rl[(rl.noise_source==s)&(rl.matrix_family=="full_generator")].sort_values("nbar")
    line(a[0,1],z.nbar,z.through_origin_relative_fit_residual*100,EN[s],COL[s])
for s in S[:2]:
    d=rs[(rs.noise_source==s)&(rs.nbar==4)&(rs.rate_multiplier>0)].sort_values("rate_multiplier")
    line(a[1,0],d.rate_multiplier,d.C_IX_XI_over_local_X_mean,EN[s],COL[s])
d=rs[(rs.noise_source==S[1])&(rs.nbar==4)&(rs.rate_multiplier>0)].sort_values("rate_multiplier")
for col,label in [("S_IX_per_gate","S:IX"),("C_IX_XI_per_gate","C:IX,XI"),("S_XX_per_gate","S:XX"),("H_XX_rad_per_gate","H:XX")]:
    v=float(d.loc[d.rate_multiplier==1,col].iloc[0]);line(a[1,1],d.rate_multiplier,d[col]/(d.rate_multiplier*v),label)
a[0,0].set_yscale("symlog",linthresh=1e-9);rate(a[0,0],"1 - stochastic shape cosine");a[0,0].legend(ncol=2);a[0,0].set_title(r"Shape change at $\bar n=4$")
nb(a[0,1],"Full-generator linear-fit residual (%)");a[0,1].set_yscale("log")
rate(a[1,0],r"$C_{IX,XI}/[(S_{IX}+S_{XI})/2]$");a[1,0].legend();a[1,0].ticklabel_format(useOffset=False,axis="y")
rate(a[1,1],r"$k(m)/[m k(1)]$");a[1,1].legend(ncol=2);a[1,1].set_title(r"Motional dephasing at $\bar n=4$");a[1,1].axhline(1,color="gray",lw=.7)
save(f,"rate_structure","Does the correlation structure survive rate changes?","Saved range: 0.5-4 x nominal | shape stability and strength linearity are separate questions",
     "主要な確率的モードの向きはレートを変えてもほぼ保存される。運動ノイズではC_IX,XIと局所X係数の平均の比がほぼ1で、集団X構造を保つ。一方、運動位相緩和の高温・高レートではS_XXが線形外挿から増え、集団X係数はやや減る。左上はゼロを表示できるsymlog軸。",[rp+"rate_structure_metrics.csv",rp+"rate_matrix_linearity.csv"],"レート依存と検証")

cfg(E+"config.json")
egpath=E+"pairwise_generator_interactions.csv";eg=read(egpath);f,a=new(3,2,size=(14,10))
for row,p in enumerate([PAIRS[0],PAIRS[4],PAIRS[3]]):
    d=eg[eg.pair_id==p];diag=d[(d.nbar==4)&(d.multiplier_x==d.multiplier_y)]
    rank=diag.groupby("pauli").interaction_gamma_per_gate.apply(lambda x:x.abs().max()).sort_values(ascending=False).head(4).index
    for mode in rank:
        z=diag[diag.pauli==mode].sort_values("multiplier_x");line(a[row,0],z.multiplier_x,z.interaction_gamma_per_gate*1e6,mode)
        z=d[(d.pauli==mode)&(d.multiplier_x==4)&(d.multiplier_y==4)].sort_values("nbar");line(a[row,1],z.nbar,z.interaction_gamma_per_gate*1e6,mode)
    a[row,0].set_title(pairlabel(p));rate(a[row,0],r"Mixed $\gamma_P$ ($10^{-6}$/gate)");a[row,0].legend(ncol=4);a[row,0].axhline(0,color="gray",lw=.7)
    nb(a[row,1],r"Mixed $\gamma_P$ ($10^{-6}$/gate)");a[row,1].axhline(0,color="gray",lw=.7)
save(f,"pair_generator","Pairwise response inside the diagonal-Pauli fit","Left: nbar=4, m_x=m_y | Right: m_x=m_y=4 | top four modes selected at nbar=4",
     "生成子を保存した上位3ペアについて、対角Pauli散逸フィット係数の混合差分を表示。これは完全H/S/C/A混合生成子の全成分ではなく、対角Pauli部分へのフィット内の結果。全6ペアの完全な生成子解析が完了したことを意味しない。",[egpath,E+"config.json"],"レート依存と検証")

kp=R+"pairwise_perturbative_kappa/perturbative_kappa_raw.csv";ke=R+"pairwise_perturbative_kappa/perturbative_kappa_extrapolated.csv";kr=read(kp);kx=read(ke);f,a=new(2,3,size=(15,8))
for ax,p in zip(a.flat,PAIRS):
    for n,c in zip(N,NC):
        d=kr[(kr.pair_id==p)&(kr.nbar==n)].sort_values("epsilon_multiplier")
        line(ax,d.epsilon_multiplier,d.kappa_multiplier*1e6,f"{n:g}",c)
    ax.set_title(pairlabel(p,True));ax.set_xlabel(r"Small rate multiplier $\epsilon$");ax.set_ylabel(r"$C(\epsilon,\epsilon)/\epsilon^2$ ($10^{-6}$)");ax.axhline(0,color="gray",lw=.7)
a[0,0].legend(title=r"$\bar n$",ncol=3)
save(f,"small_rate_stability","Small-rate mixed derivatives: numerical stability check","H: heating, M: motional dephasing, S: spin, P: scattering | flat curves would support a stable limit",
     "ゼロレート近傍の混合係数を差分幅ごとに比較。差分をepsilonの2乗で割るため、元の数値誤差が増幅される。曲線が水平に収束していない条件があり、外挿値の符号や桁数を確定した物理量として扱わない。",[kp,ke],"レート依存と検証")

ldpath=R+"noise_free_diagnostics/full_vs_lamb_dicke/full_vs_lamb_dicke_summary.csv";ld=read(ldpath)
cutpath="results/noise_rate_nbar_sweep_ld1/evaluation/cutoff_audit/exact_cutoff_comparison.csv";cut=read(cutpath)
newpath="results/single_mode_ms_gate_infidelity/single_noise/ld2/summary.csv";fresh=read(newpath)
f,a=new(1,3,size=(15,6.4))
line(a[0,0],ld.nbar,ld.infidelity_full_order*1e3,"Finite-LD corrected",COL["none"])
line(a[0,0],ld.nbar,ld.infidelity_lamb_dicke*1e3,"LD1 saved residual",COL[S[3]])
nb(a[0,0],r"No-noise infidelity ($10^{-3}$)");a[0,0].legend()
for factor,d in cut[cut.nbar>0.01].groupby("cutoff_factor"):
    d=d.sort_values("nbar");line(a[0,1],d.nbar,d.exact_truncated_infidelity,f"Cutoff x{factor:g}")
a[0,1].set_yscale("log");nb(a[0,1],"LD1 exact truncated infidelity");a[0,1].legend();a[0,1].set_title("Independent cutoff audit; nbar >= 1")
check=fresh.merge(singles,on=["noise_source","nbar"],validate="one_to_one")
check["difference"]=check.infidelity-check.single_only_infidelity
CHECKS["max_new_vs_core_single_infidelity_difference"]=float(check.difference.abs().max())
for s in S:
    d=check[check.noise_source==s].sort_values("nbar");line(a[0,2],d.nbar,d.difference*1e6,EN[s],COL[s])
nb(a[0,2],r"Separate run - core data ($10^{-6}$)");a[0,2].legend(ncol=1);a[0,2].axhline(0,color="gray",lw=.7)
check.to_csv(DATA/"separate_run_consistency.csv",index=False)
save(f,"numerical_checks","Model order and numerical checks","Finite-LD corrected is not exact all-orders | LD1 cutoff residual is not a physical thermal floor",
     "LD1の無散逸残差は基底次元を増やすと減少する。nbar=0.01の保存値1e-16は下限処理によるため、中央の対数比較には含めない。右は別保存系列の単独ノイズ計算との実測差で、元系列を混合して上書きしない。主たる有限LDの温度依存はこの差より大きいが、微小なペア差分の収束を保証する図ではない。",[ldpath,cutpath,newpath,paths[1]],"レート依存と検証")

pp="results/motional_dephasing_polynomial/polynomial_matrix_validation_metrics.csv";tt="results/motional_dephasing_polynomial/dyson_frechet_taylor/taylor_truncation_summary.csv"
poly=read(pp);taylor=read(tt);f,a=new(1,2)
for degree in [2,3,4]:
    d=poly[(poly.polynomial_degree==degree)&(poly.dataset=="independent_validation")].sort_values("nbar")
    line(a[0,0],d.nbar,d.aggregate_relative_residual*100,f"Degree {degree}")
    d=taylor[(taylor.polynomial_degree==degree)&(taylor.subset=="holdout_beyond_finest_stencil")].sort_values("nbar")
    line(a[0,1],d.nbar,d.aggregate_relative_residual*100,f"Degree {degree}")
for ax in a.flat: nb(ax,"Relative matrix residual (%)");ax.set_yscale("log");ax.legend()
a[0,0].set_title("Finite-range polynomial: independent validation");a[0,1].set_title("Near-zero Taylor: outside finest stencil")
save(f,"polynomial_validation","Motional-dephasing nonlinear response: model checks","Saved polynomial fits and near-zero derivative estimates answer different questions",
     "運動位相緩和の高次レートモデルの保存検証。左は有限範囲の多項式フィットの独立検証、右はゼロ近傍Taylor係数を推定した最小ステンシルの外側での検証。良好な有限範囲フィットだけで高階微分の収束を主張しない。",[pp,tt],"レート依存と検証")

ld1path="results/single_mode_ms_gate_infidelity/single_noise/ld1/summary.csv"
ld1solo=read(ld1path)
for model in ["ld1","ld2"]:
    cfg(f"results/single_mode_ms_gate_infidelity/single_noise/{model}/config.json")
f,a=new(2,2)
for ax,s in zip(a.flat,S):
    for data,label,color,ls in [(ld1solo,"LD1",COL[S[3]],"--"),(fresh,"Finite-LD corrected",COL["none"],"-")]:
        d=data[data.noise_source==s].sort_values("nbar")
        line(ax,d.nbar,d.infidelity*1e3,label,color,ls=ls)
    ax.set_title(EN[s]);nb(ax,r"Single-source total infidelity ($10^{-3}$)");ax.legend()
save(f,"single_source_model_order","Single-source calculations: comparison of model order","Separate saved single-noise runs | one nominal dissipator at a time | total error includes each model's intrinsic baseline",
     "別保存系列にあるLD1と有限LD補正モデルの単独ノイズ計算、計40点。表示は単独ノイズを入れた総誤差I_iで、無散逸基準を引いた増分ではない。したがって有限LDモデルの全パネルに共通する温度増加を、その散逸だけの効果と解釈しない。単独増分の比較は図1・2を参照。",[ld1path,newpath],"レート依存と検証")

# Data excerpts and provenance; all raw values are retained in source CSVs.
summary=singlemat.copy();summary.insert(0,"I_0",b.set_index("nbar").all_noise_zero_infidelity);summary.insert(1,"I_all",b.set_index("nbar").nominal_infidelity)
summary.to_csv(DATA/"nominal_error_budget.csv")
nompairs=grid[(grid.multiplier_x==1)&(grid.multiplier_y==1)].copy()
nompairs.to_csv(DATA/"nominal_pair_interactions.csv",index=False)
rank=active.loc[active.groupby("pair_id").interaction_infidelity.apply(lambda x:x.abs().idxmax())].copy().sort_values("interaction_infidelity")
rank.to_csv(DATA/"pair_extrema.csv",index=False)
kx.to_csv(DATA/"small_rate_extrapolation_diagnostics.csv",index=False)
CHECKS["max_full_taxonomy_reconstruction_fraction"]=float(overall.taxonomy_reconstruction_residual_fraction.max())
CHECKS["max_pairwise_reconstruction_abs_residual"]=float(recon.pairwise_residual.abs().max())
for key in ["nominal_infidelity","all_noise_zero_infidelity"]: assert np.isfinite(b[key]).all()
assert all((ROOT/p).exists() for page in PAGES for p in page["sources"])
manifest={"date":"2026-10-27","method":"saved data only; no QPT rerun; no source data overwritten",
          "nbar_values":N,"rate_multipliers":M,"nominal_rates_s^-1":nominal,
          "sources":MANIFEST,"checks":CHECKS,"figure_count":len(PAGES)}
(OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(OUT/"figures.json").write_text(json.dumps(PAGES,ensure_ascii=False,indent=2))
inventory=pd.DataFrame([{"path":p,**v} for p,v in MANIFEST.items()]);inventory.to_csv(DATA/"source_inventory.csv",index=False)

intro="""# ノイズ相関・既存データ図集（2026-10-27）

初期平均フォノン数 nbar、運動位相緩和、運動加熱、スピン位相緩和、光散乱を、保存済みデータから整理した図集。新たなマスター方程式計算は実行していません。

- 基準：矩形単一モードMSゲート、100 µs、eta=0.1、A=0.125、delta=0.5。`use_full_order=True` は有限LDの O(eta²) 補正を含むモデルで、厳密な全次数ではありません。
- 基準レート：運動加熱10 /s、運動位相緩和18 /s、スピン位相緩和10/3 /s、光散乱4 /s（Rayleigh 3 + Raman 1）。
- 主系列：nbar = 0.01, 1, 2, 3, 4、レート倍率 = 0, 0.5, 1, 2, 4。
- 全6ペアの5×5格子が全温度で揃い、750点あります。元QPTは共通軸を重複除去すると565点です。
- 「相関」は、(1) 温度と散逸の混合応答、(2) infidelityの非加算性、(3) 誤差生成子内部のPauli間相関を区別します。浴の統計相関を測定したものではありません。
- C_ij = I_ij - I_i - I_j + I_0。負値は単独寄与の和より誤差が小さいことを表します。
- K_0 = log R_0、ΔK_i = log R_i - log R_0。生成子はCPTP射影後の保存解析、infidelityは各CSVの保存値です。生成子係数は確率ではありません。
- S_P(ρ)=PρP-ρ、C_PQ(ρ)=PρQ+QρP−{PQ+QP,ρ}/2、H_P(ρ)=−i[P,ρ]。
- S_IX ≃ S_XI ≃ C_IX,XI は集団的D[IX+XI]を表し、D[XX]とは別です。

## 主な読み取り

1. 無散逸の有限LD誤差と、運動位相緩和の単独増分が温度に対して増加します。他の3散逸の増分は比較的平坦です。
2. 全6ペアの保存された非ゼロ格子でC_ijは負。最大絶対値は運動加熱×運動位相緩和、nbar=4、両レート4倍です。
3. 基準レートの4ノイズ同時作用はほぼ加算的です。ペア補正で再構成残差はさらに小さくなります。
4. 運動ノイズには集団Xの相関成分が現れます。スピン位相緩和・光散乱にはIZ/XY等のPauli対の成分が現れます。
5. ゼロレート近傍の微小差分は数値的に不安定な条件があります。全6ペアの完全H/S/C/A混合生成子や、その直接収束確認は本図集のデータでは揃っていません。

## ファイル

- `figures/`: 全図のPNGと編集可能なSVG。
- `data/`: 基準誤差予算、全6ペアの基準値と極値、温度混合応答、別計算との比較、出典一覧。
- `manifest.json`: 入力ファイルのSHA-256、行数、検算結果。
- `noise_correlation_atlas.ipynb`: 図を順番に閲覧するノートブック。
- `build_atlas.py` と `build_pdf.py`: 再生成スクリプト。
- PDF: `../pdf/noise_correlation_atlas_20261027.pdf`。

再生成: プロジェクトのPythonで `python3 output/noise_correlation_20261027/build_atlas.py` を実行後、reportlabを含むPythonで同じディレクトリの `build_pdf.py` を実行。

## 図の一覧
"""
budget_table="| nbar | 全ノイズ I_all | 無散逸 I_0 | 運動加熱 ΔI | 運動位相緩和 ΔI | スピン位相緩和 ΔI | 光散乱 ΔI |\n|---:|---:|---:|---:|---:|---:|---:|\n"
for n,row in summary.iterrows():
    budget_table+="| "+f"{n:g}"+" | "+" | ".join(f"{row[col]:.6e}" for col in ["I_all","I_0"]+S)+" |\n"
intro=intro.replace("## 図の一覧", "## 基準レートでの数値\n\n"+budget_table+"\n## 図の一覧")
toc="\n".join(f"- {p['number']:02d} [{p['title']}](#{p['slug']})" for p in PAGES)
body=[]
for p in PAGES:
    body.append(f"\n<a id=\"{p['slug']}\"></a>\n\n## {p['number']:02d} {p['title']}\n\n![{p['title']}]({p['png']})\n\n{p['caption']}\n\n出典: "+", ".join(f"`{s}`" for s in p['sources']))
(OUT/"README.md").write_text(intro+toc+"\n"+"\n".join(body),encoding="utf-8")
cells=[{"cell_type":"markdown","metadata":{},"source":(intro+toc).splitlines(True)}]
for p in PAGES:
    text=f"## {p['number']:02d} {p['title']}\n\n{p['caption']}\n\n![{p['title']}]({p['png']})\n\n出典: "+", ".join(f"`{s}`" for s in p['sources'])
    cells.append({"cell_type":"markdown","metadata":{},"source":text.splitlines(True)})
notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
for i,c in enumerate(cells): c["id"]=f"atlas-{i:03d}"
(OUT/"noise_correlation_atlas.ipynb").write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
# A local document gallery, no server or network dependencies.
nav="".join(f'<a href="#{p["slug"]}">{p["number"]:02d} {html.escape(p["title"])}</a>' for p in PAGES)
articles="".join(f'<article id="{p["slug"]}" data-section="{p["section"]}"><p class="tag">{p["section"]}</p><h2>{p["number"]:02d} {html.escape(p["title"])}</h2><a href="{p["svg"]}"><img src="{p["png"]}" loading="lazy" alt="{html.escape(p["title"])}"></a><p>{p["caption"]}</p><details><summary>出典</summary>'+"".join(f'<p><code>{html.escape(s)}</code></p>' for s in p["sources"])+"</details></article>" for p in PAGES)
doc='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>ノイズ相関・既存データ図集</title><style>body{margin:0;background:#edf1f5;color:#1b2d43;font:16px/1.8 system-ui}header,main{max-width:1250px;margin:auto;padding:32px}header{background:#fff}h1{font-size:32px}nav{columns:2}nav a{display:block;font-size:13px;color:#156078;text-decoration:none}article{background:white;padding:25px;margin:24px 0;border-radius:10px;scroll-margin:15px}img{width:100%;height:auto}.tag{color:#007f86;font-size:13px}code{font-size:12px;overflow-wrap:anywhere}details{color:#68788b}a{color:#007f86}@media(max-width:750px){nav{columns:1}main,header{padding:12px}}</style><header><p class="tag">2026-10-27 / SAVED-DATA ATLAS</p><h1>ノイズ相関・既存データ図集</h1><p>初期平均フォノン数・運動位相緩和・運動加熱・スピン位相緩和・光散乱。全6ペア750点と、生成子の相関成分、数値検証を整理。</p><p>図をクリックするとSVGを拡大できます。詳細な定義・注意点はREADMEおよびPDFを参照。</p><nav>'''+nav+'</nav></header><main>'+articles+'</main></html>'
(OUT/"index.html").write_text(doc)
print(json.dumps({"figures":len(PAGES),"sources":len(MANIFEST),"checks":CHECKS},ensure_ascii=False,indent=2))
