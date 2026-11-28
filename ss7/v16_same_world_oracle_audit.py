"""Diagnostic same-world oracle for frozen v16 RANK worlds.

This is NOT a deployable calibration method. It answers one question only:
does either v16 identity-stable score contain enough amplitude separation to
detect NAIVE compromise under a 20 false-window/day budget when calibration
shift is removed completely? CAL/TEST are never loaded.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from ss7 import fix as F
from ss7 import cu_compact_channel_development as C
from ss7.incident_calibration_study import threshold, notify

BASELINE_END=int(os.getenv("SS7_DEPLOY_BASELINE","180"))
RAW_DAY=20.0
FEATURES=(C.SUB_PEER_SERVICE_IDMEAN5_NAME,C.SUB_PEER_SERVICE_IDMIN3_NAME)

def days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def one(d,s,feature,seed,eligible):
    normal=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)&np.isfinite(s)
    cut=float(threshold(s[normal],RAW_DAY,days(d)))
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    eps=[]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy();e=eligible.get(int(eid))
        fire=s[ix]>cut
        eps.append(dict(seed=seed,feature=feature,episode=int(eid),
            role=str(g.ntype.iloc[0]),
            hidden_rate=float(e.hidden_rate) if e is not None else np.nan,
            hit=int(np.any(fire)),
            peak=float(np.nanmax(s[ix])) if np.isfinite(s[ix]).any() else np.nan,
            threshold=cut))
    ep=pd.DataFrame(eps)
    fwin=float((s[normal]>cut).sum()/days(d))
    return ep,dict(seed=seed,feature=feature,threshold=cut,
                   normal_max=float(np.nanmax(s[normal])),
                   normal_q999=float(np.nanquantile(s[normal],.999)),
                   attack_max=float(ep.peak.max()),
                   attack_mean_peak=float(ep.peak.mean()),
                   recall=float(ep.hit.mean()),
                   false_windows_day=fwin)

def main(root,out,seeds):
    rows=[];episodes=[]
    for seed in map(int,seeds):
        d,eligible,_=C.load(root,"compact",seed)
        for feature in FEATURES:
            s=d[feature].to_numpy(float)
            ep,m=one(d,s,feature,seed,eligible)
            rows.append(m);episodes.append(ep)
    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    summary=df.groupby("feature").agg(
        mean_recall=("recall","mean"),min_recall=("recall","min"),
        mean_threshold=("threshold","mean"),
        min_attack_max=("attack_max","min"),
        max_normal_q999=("normal_q999","max"),
        worst_false_windows_day=("false_windows_day","max")).reset_index()
    byrole=ed.groupby(["feature","role"]).agg(
        episodes=("episode","count"),recall=("hit","mean"),
        mean_peak=("peak","mean"),max_peak=("peak","max")).reset_index()
    df.to_csv(out/"worlds.csv",index=False);ed.to_csv(out/"episodes.csv",index=False)
    summary.to_csv(out/"summary.csv",index=False);byrole.to_csv(out/"by_role.csv",index=False)
    print("WORLDS\n",df.to_string(index=False),flush=True)
    print("SUMMARY\n",summary.to_string(index=False),flush=True)
    print("BY_ROLE\n",byrole.to_string(index=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,a.seeds)
