"""Diagnostic upper bound for neighborhood observability of NAIVE compromise.

NOT a deployable detector: same-world normal thresholds are used only as an
oracle upper bound. An episode is detected if the compromised node or one of
its directly adjacent signaling nodes produces a peer-KIE alert during the
episode. No CAL/TEST data are loaded.
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
RAW_DAY=4.0
SMOOTH=5
FEATURE="kie_peer_sub_max_resid"

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def score(d):
    x=d[FEATURE].to_numpy(float)
    out=np.full(len(d),np.nan)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        out[ix]=pd.Series(x[ix],index=ix).rolling(
            SMOOTH,min_periods=2).mean().to_numpy()
    return out

def topology(seed):
    sim=C.IntegratedSim(F.SimConfig(seed=int(seed),days=C.DAYS),
                        F.RngBook(int(seed)),"compact")
    return sim.topo

def main(root,out,seeds):
    rows=[];episodes=[]
    for seed in map(int,seeds):
        d,eligible,_=C.load(root,"compact",seed)
        s=score(d);tt=d.t.to_numpy();nn=d.nid.to_numpy()
        normal=(d.label.to_numpy()==0)&(tt>=BASELINE_END)&np.isfinite(s)
        cut=float(threshold(s[normal],RAW_DAY,monitor_days(d)))
        z3=C.streak_score(d,np.nan_to_num(s,nan=-np.inf),3)
        topo=topology(seed)
        eps=[]
        for eid,e in eligible.items():
            if e.kind!="compromise" or e.sophistication.value!="naive":continue
            local={int(e.nid)}
            neigh=local|{int(topo.peer(k,e.nid)) for k in topo.links_of(e.nid)}
            time=(tt>=e.t0)&(tt<=e.t1)
            lm=time&np.isin(nn,list(local))
            nm=time&np.isin(nn,list(neigh))
            eps.append(dict(seed=seed,episode=int(eid),role=topo.nodes[e.nid].ntype.value,
                local_hit1=int(np.any(s[lm]>cut)),local_hit3=int(np.any(z3[lm]>cut)),
                neighbor_hit1=int(np.any(s[nm]>cut)),neighbor_hit3=int(np.any(z3[nm]>cut)),
                local_peak=float(np.nanmax(s[lm])) if lm.any() else np.nan,
                neighbor_peak=float(np.nanmax(s[nm])) if nm.any() else np.nan))
        ep=pd.DataFrame(eps);episodes.append(ep)
        rows.append(dict(seed=seed,threshold=cut,episodes=len(ep),
            local_recall3=float(ep.local_hit3.mean()),
            neighbor_recall3=float(ep.neighbor_hit3.mean()),
            neighbor_recall1=float(ep.neighbor_hit1.mean()),
            false_rows_day=float((s[normal]>cut).sum()/monitor_days(d))))
    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    byrole=ed.groupby("role").agg(
        episodes=("episode","count"),
        local_recall3=("local_hit3","mean"),
        neighbor_recall3=("neighbor_hit3","mean"),
        neighbor_recall1=("neighbor_hit1","mean")).reset_index()
    summary=dict(
        oracle_local_mean_recall3=float(df.local_recall3.mean()),
        oracle_neighbor_mean_recall3=float(df.neighbor_recall3.mean()),
        oracle_neighbor_min_recall3=float(df.neighbor_recall3.min()),
        oracle_neighbor_mean_recall1=float(df.neighbor_recall1.mean()),
        worst_false_rows_day=float(df.false_rows_day.max()))
    df.to_csv(out/"worlds.csv",index=False);ed.to_csv(out/"episodes.csv",index=False)
    byrole.to_csv(out/"by_role.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("WORLDS\n",df.to_string(index=False),flush=True)
    print("BY_ROLE\n",byrole.to_string(index=False),flush=True)
    print("SUMMARY",json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,a.seeds)
