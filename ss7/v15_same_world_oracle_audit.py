"""Diagnostic upper bound for v15 rolling subscriber-KIE.

This is NOT a deployable calibration: each RANK world's own normal rows choose
its threshold. It answers only whether any threshold at a stated false-window
budget could recover NAIVE compromise episodes. CAL/TEST are never read.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from ss7 import fix as F
from ss7 import cu_compact_channel_development as C
from ss7.incident_calibration_study import threshold

BASELINE_END=int(os.getenv("SS7_DEPLOY_BASELINE","180"))
BUDGETS=(4.0,20.0)
FEATURE=C.SUB_PEER_SERVICE_ROLL30_NAME

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def eval_budget(seed,d,eligible,budget):
    s=d[FEATURE].to_numpy(float);tt=d.t.to_numpy()
    normal=(d.label.to_numpy()==0)&(tt>=BASELINE_END)&np.isfinite(s)
    cut=float(threshold(s[normal],budget,monitor_days(d)))
    z3=C.streak_score(d,np.nan_to_num(s,nan=-np.inf),3)
    eps=[]
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy();e=eligible.get(int(eid))
        eps.append(dict(seed=seed,budget=budget,episode=int(eid),
            role=str(g.ntype.iloc[0]),
            hidden_rate=float(e.hidden_rate) if e is not None else np.nan,
            hit1=int(np.any(s[ix]>cut)),
            hit3=int(np.any(z3[ix]>cut)),
            peak=float(np.nanmax(s[ix])),threshold=cut))
    ep=pd.DataFrame(eps)
    low=ep.hidden_rate<.5
    return ep,dict(seed=seed,budget=budget,threshold=cut,
        false_rows_day=float((s[normal]>cut).sum()/monitor_days(d)),
        recall1=float(ep.hit1.mean()),recall3=float(ep.hit3.mean()),
        low_recall3=float(ep.loc[low,"hit3"].mean()) if low.any() else np.nan,
        high_recall3=float(ep.loc[~low,"hit3"].mean()) if (~low).any() else np.nan,
        max_attack_peak=float(ep.peak.max()))

def main(root,out,seeds):
    rows=[];episodes=[]
    for seed in map(int,seeds):
        d,eligible,_=C.load(root,"compact",seed)
        for budget in BUDGETS:
            ep,m=eval_budget(seed,d,eligible,budget)
            rows.append(m);episodes.append(ep)
    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    summary=df.groupby("budget").agg(
        mean_recall3=("recall3","mean"),min_recall3=("recall3","min"),
        mean_low_recall3=("low_recall3","mean"),
        mean_high_recall3=("high_recall3","mean"),
        worst_false_rows_day=("false_rows_day","max"),
        min_threshold=("threshold","min"),max_threshold=("threshold","max"),
        max_attack_peak=("max_attack_peak","max")).reset_index()
    df.to_csv(out/"worlds.csv",index=False);ed.to_csv(out/"episodes.csv",index=False)
    summary.to_csv(out/"summary.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary.to_dict("records"),indent=2))
    print("WORLDS\n",df.to_string(index=False),flush=True)
    print("SUMMARY\n",summary.to_string(index=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,a.seeds)
