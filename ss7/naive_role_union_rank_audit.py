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
ROLE_MSG={
    "SP":("ATI","PSI","SRI_SM","MT_FSM","SAI","UL","ISD","CL"),
    "SCP":("ATI","PSI","SRI","PRN","SRI_SM","MT_FSM","SAI","UL","ISD","CL"),
    "IGW":("SRI","PRN","SRI_SM","MT_FSM","ATI","PSI"),
    "STP":("ATI","PSI","SRI","PRN","SRI_SM","MT_FSM","SAI","UL","ISD","CL"),
}
VARIANTS=("max","top2","l2","max_total")
SMOOTH=(1,3,5,10)
ROLE_RAW_DAY=4.0

def raw_score(d,role,variant):
    cols=[f"drift_obs_ratio_{m}_z" for m in ROLE_MSG[role] if f"drift_obs_ratio_{m}_z" in d.columns]
    if not cols:return np.zeros(len(d),float)
    z=np.maximum(np.nan_to_num(d[cols].to_numpy(float),nan=0.0),0.0)
    if variant=="max":
        s=z.max(axis=1)
    elif variant=="top2":
        z=np.sort(z,axis=1)
        s=z[:,-min(2,z.shape[1]):].mean(axis=1)
    elif variant=="l2":
        s=np.sqrt((z*z).mean(axis=1))
    elif variant=="max_total":
        total=np.maximum(np.nan_to_num(d["drift_total_messages_z"].to_numpy(float),nan=0.0),0.0)
        s=np.maximum(z.max(axis=1),.6*total)
    else:raise ValueError(variant)
    return s

def smooth(d,s,w):
    if w==1:return np.asarray(s,float)
    out=np.zeros(len(d),float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        out[ix]=pd.Series(s[ix],index=ix).rolling(w,min_periods=max(1,w//2)).mean().fillna(0.0).to_numpy()
    return out

def load(root,seed):
    return C.load(root,"compact",seed)[0]

def evaluate_pair(cal,ev,variant,w,role):
    sc=smooth(cal,raw_score(cal,role,variant),w)
    se=smooth(ev,raw_score(ev,role,variant),w)
    cal_role=cal.ntype.astype(str).to_numpy()==role
    ev_role=ev.ntype.astype(str).to_numpy()==role
    cal_neg=cal_role&(cal.label.to_numpy()==0)
    cut=threshold(sc[cal_neg],ROLE_RAW_DAY,elapsed(cal))

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

def main(root,out):
    frames={s:load(root,s) for s in RANK}
    rows=[]
    for variant in VARIANTS:
        for w in SMOOTH:
            for role in ROLE_MSG:
                for a,b in ((RANK[0],RANK[1]),(RANK[1],RANK[0])):
                    r=evaluate_pair(frames[a],frames[b],variant,w,role)
                    r.update(variant=variant,smooth=w,role=role,cal_seed=a,eval_seed=b)
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
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);main(a.root,a.output)
