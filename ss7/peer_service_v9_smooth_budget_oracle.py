"""Diagnostic v9 smooth-5 oracle under actual 20/2 budgets.

The transformation is frozen from v9 RANK: peer-service max residual followed
by a causal 5-window mean. Same-world normal thresholds make this an upper
bound, not a deployable detector. CAL/TEST are never read.
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
INCIDENT_DAY=2.0
FEATURE="kie_peer_service_sub_max_resid"
SMOOTH=5
STREAKS=(3,5,7,9,12)

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

def main(root,out,seeds):
    rows=[];episodes=[]
    for seed in map(int,seeds):
        d,eligible,_=C.load(root,"compact",seed)
        s=score(d);tt=d.t.to_numpy();days=monitor_days(d)
        normal=(d.label.to_numpy()==0)&(tt>=BASELINE_END)&np.isfinite(s)
        cut=float(threshold(s[normal],RAW_DAY,days))
        chosen=STREAKS[-1];notif=np.nan
        for L in STREAKS:
            z0=C.streak_score(d,np.where(normal,s,-np.inf),L)
            nr=float(len(notify(d,z0,cut))/days)
            if nr<=INCIDENT_DAY:
                chosen=L;notif=nr;break
        z=C.streak_score(d,np.nan_to_num(s,nan=-np.inf),chosen)
        eps=[]
        q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
        for eid,g in q.groupby("episode_id",sort=False):
            ix=g.index.to_numpy();e=eligible.get(int(eid))
            eps.append(dict(seed=seed,episode=int(eid),role=str(g.ntype.iloc[0]),
                hidden_rate=float(e.hidden_rate) if e is not None else np.nan,
                detected=int(np.any(z[ix]>cut)),
                peak=float(np.nanmax(s[ix]))))
        ep=pd.DataFrame(eps);episodes.append(ep)
        rows.append(dict(seed=seed,threshold=cut,streak=chosen,
            false_windows_day=float((s[normal]>cut).sum()/days),
            false_notifications_day=float(notif),
            recall=float(ep.detected.mean()),
            low_recall=float(ep.loc[ep.hidden_rate<.5,"detected"].mean()),
            high_recall=float(ep.loc[ep.hidden_rate>=.5,"detected"].mean())))
    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    byrole=ed.groupby("role").agg(episodes=("episode","count"),
        recall=("detected","mean"),mean_peak=("peak","mean")).reset_index()
    summary=dict(mean_recall=float(df.recall.mean()),min_recall=float(df.recall.min()),
        mean_low_recall=float(df.low_recall.mean()),
        mean_high_recall=float(df.high_recall.mean()),
        worst_false_windows_day=float(df.false_windows_day.max()),
        worst_false_notifications_day=float(df.false_notifications_day.max()))
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
