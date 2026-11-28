"""Diagnostic v9 oracle under the actual detector budgets (20 raw / 2 incidents per day).

NOT deployable: thresholds are estimated from same-world normal monitoring rows
to measure an upper bound. The v9 feature is frozen; only persistence length is
chosen from the predeclared detector set to meet the incident budget.
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
STREAKS=(3,5,7,9,12)

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def main(root,out,seeds):
    rows=[];episodes=[]
    for seed in map(int,seeds):
        d,eligible,_=C.load(root,"compact",seed)
        s=d[FEATURE].to_numpy(float);tt=d.t.to_numpy()
        normal=(d.label.to_numpy()==0)&(tt>=BASELINE_END)&np.isfinite(s)
        days=monitor_days(d)
        cut=float(threshold(s[normal],RAW_DAY,days))
        chosen=STREAKS[-1];notif=np.nan
        for L in STREAKS:
            z=C.streak_score(d,np.where(normal,s,-np.inf),L)
            nr=float(len(notify(d,z,cut))/days)
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
