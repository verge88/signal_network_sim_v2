"""Fast robustness audit for a preregistered subset of NAIVE SS7 signals.

Feature families were nominated from earlier RANK diagnostics. This script
does not inspect CAL or TEST and reports per-world and worst-world ranking.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from ss7.cu_compact_channel_development import _causal_robust_z

SEEDS=(21001,21002,21003,21101,21102)
SPEC={
 "IGW":(
   ("obs_ratio_MT_FSM",60,"pos"),("obs_ratio_SRI_SM",60,"pos"),
   ("total_messages",30,"abs"),("mtype_entropy",30,"abs")),
 "SCP":(
   ("obs_ratio_ATI",30,"abs"),("obs_ratio_PSI",60,"abs"),
   ("obs_ratio_ATI",60,"abs"),("mtype_entropy",30,"abs"),
   ("mtype_entropy",60,"abs")),
 "SP":(
   ("obs_ratio_ATI",60,"pos"),("obs_ratio_PSI",60,"pos"),
   ("obs_ratio_ATI",30,"pos"),("obs_ratio_PSI",30,"pos"),
   ("obs_ratio_ATI",30,"abs"),("obs_ratio_PSI",30,"abs"),
   ("obs_ratio_SAI",60,"abs")),
 "STP":(
   ("obs_ratio_ISD",60,"pos"),("obs_ratio_ATI",30,"abs"),
   ("obs_ratio_PSI",30,"abs"),("obs_ratio_ATI",60,"abs"),
   ("obs_ratio_PSI",60,"abs"),("obs_ratio_SRI_SM",60,"abs"),
   ("map_share",60,"abs"),("isup_share",60,"abs")),
}

def roll(d,x,win):
    out=np.zeros(len(d),float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        out[ix]=pd.Series(x[ix],index=ix).rolling(
            win,min_periods=max(10,win//3)).mean().fillna(0.).to_numpy()
    return out

def score(d,feature,win,direction):
    z=_causal_robust_z(d,d[feature].to_numpy(float))
    z=np.abs(z) if direction=="abs" else np.maximum(z,0.)
    return roll(d,z,win)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in SEEDS:
        d,_,_=joblib.load(a.root/f"world-compact-{seed}"/"world.joblib")
        role=d.ntype.astype(str).to_numpy()
        naive=(d.attack.to_numpy()=="compromise")&(d.sophistication.astype(str).to_numpy()=="naive")
        neg=d.label.to_numpy()==0
        for r,specs in SPEC.items():
            use=(role==r)&(naive|neg)
            y=naive[use].astype(int)
            if len(np.unique(y))<2:continue
            for feature,win,direction in specs:
                if feature not in d.columns:continue
                q=score(d,feature,win,direction)[use]
                rows.append(dict(seed=seed,role=r,feature=feature,window=win,direction=direction,
                    auc=roc_auc_score(y,q),ap=average_precision_score(y,q),
                    positives=int(y.sum()),negatives=int((y==0).sum())))
    df=pd.DataFrame(rows)
    df["candidate"]=df.feature+":"+df.window.astype(str)+":"+df.direction
    s=df.groupby(["role","candidate"]).agg(
        worlds=("seed","nunique"),auc_mean=("auc","mean"),auc_min=("auc","min"),
        auc_sd=("auc","std"),ap_mean=("ap","mean"),ap_min=("ap","min")
    ).reset_index()
    s=s.sort_values(["role","auc_min","ap_min"],ascending=[True,False,False])
    df.to_csv(a.output/"scores.csv",index=False);s.to_csv(a.output/"summary.csv",index=False)
    print("ROBUST SUBSET\n",s.to_string(index=False),flush=True)

if __name__=="__main__":main()
