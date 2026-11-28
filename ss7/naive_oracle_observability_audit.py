"""Diagnostic upper bound for NAIVE-compromise observability.

This is NOT a deployable detector. Ground-truth labels are used only to report
the best achievable single-feature episode recall after each feature receives
an oracle same-world normal threshold. If the oracle itself fails, further
threshold/model tuning of the same observables is not scientifically useful.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from ss7 import fix as F
from ss7 import cu_compact_channel_development as C
from ss7.incident_calibration_study import threshold

BASELINE_END=180
RAW_DAY=4.0
SMOOTH=5

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def smooth(d,x):
    out=np.full(len(d),np.nan)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        out[ix]=pd.Series(x[ix],index=ix).rolling(
            SMOOTH,min_periods=2).mean().to_numpy()
    return out

def candidates(d):
    keep=[]
    base=set(C.BASE)|set(C.KIE_GROUP_DERIVED)|set(getattr(C,"SUB_RESID_NAMES",()))
    for c in d.columns:
        if c in base or c.startswith("drift_"):
            if pd.api.types.is_numeric_dtype(d[c]):keep.append(c)
    return sorted(set(keep))

def evaluate_feature(d,eligible,col,direction):
    x=d[col].to_numpy(float)*direction
    s=smooth(d,x)
    normal=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)&np.isfinite(s)
    if normal.sum()<100:return None
    cut=float(threshold(s[normal],RAW_DAY,monitor_days(d)))
    z3=C.streak_score(d,np.nan_to_num(s,nan=-np.inf),3)
    eps=[]
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy();e=eligible.get(int(eid))
        eps.append(dict(episode=int(eid),role=str(g.ntype.iloc[0]),
            hidden_rate=float(e.hidden_rate) if e is not None else np.nan,
            hit1=int(np.any(s[ix]>cut)),hit3=int(np.any(z3[ix]>cut))))
    ep=pd.DataFrame(eps)
    return dict(feature=col,direction=int(direction),threshold=cut,
        recall1=float(ep.hit1.mean()),recall3=float(ep.hit3.mean()),
        low_recall3=float(ep.loc[ep.hidden_rate<.5,"hit3"].mean()),
        high_recall3=float(ep.loc[ep.hidden_rate>=.5,"hit3"].mean()),
        false_rows_day=float((s[normal]>cut).sum()/monitor_days(d)))

def main(root,out,seeds):
    rows=[]
    for seed in map(int,seeds):
        d,eligible,_=C.load(root,"compact",seed)
        for col in candidates(d):
            for direction in (1,-1):
                m=evaluate_feature(d,eligible,col,direction)
                if m is not None:m["seed"]=seed;rows.append(m)
    df=pd.DataFrame(rows)
    df.to_csv(out/"all_features.csv",index=False)
    best=[]
    for seed,g in df.groupby("seed"):
        q=g.sort_values(["recall3","recall1","high_recall3","low_recall3"],
                        ascending=False).iloc[0].to_dict()
        best.append(q)
    b=pd.DataFrame(best);b.to_csv(out/"oracle_best_by_world.csv",index=False)
    # Also report features that are consistently useful across all worlds.
    agg=df.groupby(["feature","direction"]).agg(
        mean_recall3=("recall3","mean"),min_recall3=("recall3","min"),
        mean_high_recall3=("high_recall3","mean"),
        min_high_recall3=("high_recall3","min"),
        mean_low_recall3=("low_recall3","mean"),
        min_low_recall3=("low_recall3","min"),
        worst_false_rows_day=("false_rows_day","max")).reset_index()
    agg=agg.sort_values(["min_recall3","mean_recall3","min_high_recall3","mean_high_recall3"],
                        ascending=False)
    agg.to_csv(out/"consistent_features.csv",index=False)
    summary=dict(
        oracle_mean_recall3=float(b.recall3.mean()),
        oracle_min_recall3=float(b.recall3.min()),
        oracle_mean_high_recall3=float(b.high_recall3.mean()),
        oracle_mean_low_recall3=float(b.low_recall3.mean()),
        best_consistent=agg.iloc[0].to_dict())
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("ORACLE_BEST\n",b.to_string(index=False),flush=True)
    print("CONSISTENT_TOP\n",agg.head(20).to_string(index=False),flush=True)
    print("SUMMARY",json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,a.seeds)
