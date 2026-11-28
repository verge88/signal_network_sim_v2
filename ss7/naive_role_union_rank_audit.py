"""RANK-only cross-world audit for role-aware naive compromise evidence.

Uses only observable KIE-derived drift features. No CAL/TEST rows, labels or
thresholds are consumed. One RANK world calibrates a benign score threshold;
the other evaluates it, then roles are swapped. The audit chooses a stable
aggregation form for a later frozen detector revision.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from ss7 import cu_compact_channel_development as C
from ss7.incident_calibration_study import elapsed, threshold

RANK=(22101,22102)
BASELINE_END=int(__import__("os").getenv("SS7_DEPLOY_BASELINE",str(C.WARMUP)))
ROLE_MSG={
    "SP":("ATI","PSI","SRI_SM","MT_FSM","SAI","UL","ISD","CL"),
    "SCP":("ATI","PSI","SRI","PRN","SRI_SM","MT_FSM","SAI","UL","ISD","CL"),
    "IGW":("SRI","PRN","SRI_SM","MT_FSM","ATI","PSI"),
    "STP":("ATI","PSI","SRI","PRN","SRI_SM","MT_FSM","SAI","UL","ISD","CL"),
}
VARIANTS=("max","top2","l2","max_total","stable_max","stable_top2","stable_l2","stable_max_total","count_max","count_top2","count_l2","count_max_total","composition_max","composition_top2","composition_l2")
SMOOTH=(1,3,5,10)
ROLE_RAW_DAY=4.0

def stable_shift(d,col):
    """Causal robust recent-vs-past shift without score saturation."""
    out=np.zeros(len(d),float)
    ratio=("ratio_" in col) or col.endswith(("_share","_entropy"))
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        x=pd.Series(g[col].to_numpy(float),index=ix)
        recent=x.rolling(10,min_periods=4).mean()
        past=x.shift(10)
        center=past.rolling(60,min_periods=15).median()
        q25=past.rolling(60,min_periods=15).quantile(.25)
        q75=past.rolling(60,min_periods=15).quantile(.75)
        robust=(q75-q25)/1.349
        c=center.to_numpy(float)
        if ratio:
            floor=np.full(len(ix),.02)
        else:
            floor=np.sqrt(np.maximum(np.abs(c),1.0))
        scale=np.maximum(np.nan_to_num(robust.to_numpy(float),nan=0.0),floor)
        z=(recent.to_numpy(float)-c)/scale
        out[ix]=np.arcsinh(np.nan_to_num(z,nan=0.0,posinf=0.0,neginf=0.0))
    return out

def reconstructed_count(d,m):
    # Compact KIE already carries total mtype counts. The stored ratio form is
    # algebraically inverted here without adding any wire field.
    return (np.nan_to_num(d["total_messages"].to_numpy(float),nan=0.0)
            *np.nan_to_num(d["map_share"].to_numpy(float),nan=0.0)
            *np.nan_to_num(d[f"obs_ratio_{m}"].to_numpy(float),nan=0.0))

def stable_array_shift(d,x,ratio=False):
    out=np.zeros(len(d),float)
    x=np.asarray(x,float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        v=pd.Series(x[ix],index=ix)
        recent=v.rolling(10,min_periods=4).mean()
        past=v.shift(10)
        center=past.rolling(60,min_periods=15).median()
        q25=past.rolling(60,min_periods=15).quantile(.25)
        q75=past.rolling(60,min_periods=15).quantile(.75)
        robust=(q75-q25)/1.349
        c=center.to_numpy(float)
        floor=(np.full(len(ix),.02) if ratio else np.sqrt(np.maximum(np.abs(c),1.0)))
        scale=np.maximum(np.nan_to_num(robust.to_numpy(float),nan=0.0),floor)
        z=(recent.to_numpy(float)-c)/scale
        out[ix]=np.arcsinh(np.nan_to_num(z,nan=0.0,posinf=0.0,neginf=0.0))
    return out

def composition_residuals(d,role):
    """Node-local MAP composition residuals using only attack-free warm-up."""
    msgs=[m for m in ROLE_MSG[role] if f"obs_ratio_{m}" in d.columns]
    out=np.zeros((len(d),len(msgs)),float)
    if not msgs:return out
    total=(np.nan_to_num(d["total_messages"].to_numpy(float),nan=0.0)
           *np.nan_to_num(d["map_share"].to_numpy(float),nan=0.0))
    counts=np.column_stack([reconstructed_count(d,m) for m in msgs])
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        warm=(g.t.to_numpy()<BASELINE_END)
        if not warm.any():continue
        den=float(total[ix][warm].sum())
        if den<=0:continue
        p=np.clip(counts[ix][warm].sum(axis=0)/den,1e-6,1-1e-6)
        exp=total[ix,None]*p[None,:]
        # Multinomial variance plus 10% relative over-dispersion protects
        # against AR traffic and measurement noise absent from ideal Poisson.
        var=exp*(1-p[None,:])+(0.10*exp)**2+1.0
        z=(counts[ix]-exp)/np.sqrt(var)
        out[ix]=np.arcsinh(np.maximum(z,0.0))
    return out

def raw_score(d,role,variant):
    stable=variant.startswith("stable_")
    count=variant.startswith("count_")
    composition=variant.startswith("composition_")
    base_variant=variant.split("_",1)[1] if (stable or count or composition) else variant
    if composition:
        z=composition_residuals(d,role)
        if z.shape[1]==0:return np.zeros(len(d),float)
    elif count:
        msgs=[m for m in ROLE_MSG[role] if f"obs_ratio_{m}" in d.columns]
        if not msgs:return np.zeros(len(d),float)
        z=np.column_stack([np.maximum(stable_array_shift(d,reconstructed_count(d,m)),0.0) for m in msgs])
    elif stable:
        cols=[f"obs_ratio_{m}" for m in ROLE_MSG[role] if f"obs_ratio_{m}" in d.columns]
        if not cols:return np.zeros(len(d),float)
        z=np.column_stack([np.maximum(stable_shift(d,c),0.0) for c in cols])
    else:
        cols=[f"drift_obs_ratio_{m}_z" for m in ROLE_MSG[role] if f"drift_obs_ratio_{m}_z" in d.columns]
        if not cols:return np.zeros(len(d),float)
        z=np.maximum(np.nan_to_num(d[cols].to_numpy(float),nan=0.0),0.0)
    if base_variant=="max":
        score=z.max(axis=1)
    elif base_variant=="top2":
        zz=np.sort(z,axis=1)
        score=zz[:,-min(2,zz.shape[1]):].mean(axis=1)
    elif base_variant=="l2":
        score=np.sqrt((z*z).mean(axis=1))
    elif base_variant=="max_total":
        if composition:
            total=np.zeros(len(d),float)
        elif count:
            total=np.maximum(stable_array_shift(d,d["total_messages"].to_numpy(float)),0.0)
        elif stable:
            total=np.maximum(stable_shift(d,"total_messages"),0.0)
        else:
            total=np.maximum(np.nan_to_num(d["drift_total_messages_z"].to_numpy(float),nan=0.0),0.0)
        score=np.maximum(z.max(axis=1),.6*total)
    else:raise ValueError(variant)
    return score

def smooth(d,s,w):
    if w==1:return np.asarray(s,float)
    out=np.zeros(len(d),float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        out[ix]=pd.Series(s[ix],index=ix).rolling(w,min_periods=max(1,w//2)).mean().fillna(0.0).to_numpy()
    return out

def load(root,seed):
    return C.load(root,"compact",seed)[0]

def evaluate_holdout(cal_items,ev,se,variant,w,role):
    cuts=[]
    for cal,sc in cal_items:
        cal_role=cal.ntype.astype(str).to_numpy()==role
        cal_neg=cal_role&(cal.label.to_numpy()==0)
        cuts.append(threshold(sc[cal_neg],ROLE_RAW_DAY,elapsed(cal)))
    cut=float(max(cuts))

    ev_role=ev.ntype.astype(str).to_numpy()==role
    y=((ev.attack=="compromise")&(ev.sophistication=="naive")&ev_role).to_numpy(int)
    eligible=ev_role&((ev.label.to_numpy()==0)|y.astype(bool))
    ye=y[eligible];score=se[eligible]
    neg=eligible&(ev.label.to_numpy()==0)
    pos=y.astype(bool)
    return dict(
        cut=float(cut),
        auc=float(roc_auc_score(ye,score)) if ye.sum() and (~ye.astype(bool)).sum() else np.nan,
        ap=float(average_precision_score(ye,score)) if ye.sum() else np.nan,
        positives=int(pos.sum()),
        detected=int((se[pos]>cut).sum()),
        row_tpr=float((se[pos]>cut).mean()) if pos.sum() else np.nan,
        false_rows_day=float((se[neg]>cut).sum()/elapsed(ev)),
    )

def main(root,out,seeds=RANK):
    global RANK
    RANK=tuple(int(x) for x in seeds)
    if len(RANK)<2:raise ValueError("audit requires at least two RANK worlds")
    frames={s:load(root,s) for s in RANK}
    rows=[]
    cache={}
    for variant in VARIANTS:
        for role in ROLE_MSG:
            base={seed:raw_score(frames[seed],role,variant) for seed in RANK}
            for w in SMOOTH:
                for seed in RANK:
                    cache[(seed,variant,w,role)]=smooth(frames[seed],base[seed],w)
                for b in RANK:
                    cals=[a for a in RANK if a!=b]
                    r=evaluate_holdout(
                        [(frames[a],cache[(a,variant,w,role)]) for a in cals],
                        frames[b],cache[(b,variant,w,role)],variant,w,role)
                    r.update(variant=variant,smooth=w,role=role,
                        cal_seed=";".join(map(str,cals)),eval_seed=b)
                    rows.append(r)
    q=pd.DataFrame(rows)
    q.to_csv(out/"cross_world.csv",index=False)
    agg=q.groupby(["variant","smooth"]).agg(
        mean_auc=("auc","mean"),min_auc=("auc","min"),
        mean_ap=("ap","mean"),min_ap=("ap","min"),
        mean_row_tpr=("row_tpr","mean"),min_row_tpr=("row_tpr","min"),
        worst_false_rows_day=("false_rows_day","max")
    ).reset_index()
    # Predeclared selection: maximize worst-role/world row TPR, then min AP,
    # subject to a modest cross-world raw false-row envelope.
    ok=agg[agg.worst_false_rows_day<=8.0].copy()
    cand=ok if len(ok) else agg
    cand=cand.sort_values(["min_row_tpr","min_ap","mean_ap"],ascending=False)
    winner=cand.iloc[0].to_dict()
    agg.to_csv(out/"summary.csv",index=False)
    (out/"winner.json").write_text(json.dumps(winner,indent=2))
    print("SUMMARY\n",agg.sort_values(["min_row_tpr","min_ap"],ascending=False).to_string(index=False),flush=True)
    print("WINNER",json.dumps(winner,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",default=list(RANK))
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);main(a.root,a.output,a.seeds)
