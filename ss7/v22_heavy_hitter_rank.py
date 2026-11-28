"""Fresh RANK-only audit for compact heavy-hitter subscriber KIE v22.

Six causal features derived from top-4 keyed subscriber identities are tested
without fitting a classifier. Thresholds transfer leave-one-world-out across
RANK worlds. CAL/TEST are never loaded.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from ss7 import fix as F
from ss7 import cu_compact_channel_development as C
from ss7.incident_calibration_study import threshold, notify

BASELINE_END=int(os.getenv("SS7_DEPLOY_BASELINE","180"))
RAW_DAY=20.0
NOTICE_DAY=2.0
FEATURES=C.SUB_HH_NAMES

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def world_cut(d,s):
    normal=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)&np.isfinite(s)
    days=monitor_days(d)
    raw=float(threshold(s[normal],RAW_DAY,days))
    b=C.bucket_peaks(d,np.where(normal,s,-np.inf))
    b=b[np.isfinite(b)]
    inc=float(threshold(b,NOTICE_DAY,days)) if len(b) else raw
    return max(raw,inc)

def evaluate(d,s,cut,eligible):
    tt=d.t.to_numpy()
    y=((d.attack=="compromise")&(d.sophistication=="naive")).to_numpy(int)
    valid=((d.label.to_numpy()==0)|y.astype(bool))&(tt>=BASELINE_END)&np.isfinite(s)
    yy=y[valid];ss=s[valid]
    auc=float(roc_auc_score(yy,ss)) if yy.sum() and (~yy.astype(bool)).sum() else np.nan
    ap=float(average_precision_score(yy,ss)) if yy.sum() else np.nan

    normal=(d.label.to_numpy()==0)&(tt>=BASELINE_END)&np.isfinite(s)
    fire=np.isfinite(s)&(s>cut)
    nidx=notify(d,np.where(np.isfinite(s),s,-np.inf),cut)
    fnot=float(sum(bool(normal[j]) for j in nidx)/monitor_days(d))

    eps=[]
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy();e=eligible.get(int(eid));hit=np.flatnonzero(fire[ix])
        eps.append(dict(
            episode=int(eid),role=str(g.ntype.iloc[0]),
            hidden_rate=float(e.hidden_rate) if e is not None else np.nan,
            hit=int(len(hit)>0),delay_seconds=float(hit[0]*30) if len(hit) else np.nan,
            peak=float(np.nanmax(s[ix])) if np.isfinite(s[ix]).any() else np.nan))
    ep=pd.DataFrame(eps);low=ep.hidden_rate<.5
    return ep,dict(
        auc=auc,ap=ap,episode_recall=float(ep.hit.mean()),
        low_recall=float(ep.loc[low,"hit"].mean()) if low.any() else np.nan,
        high_recall=float(ep.loc[~low,"hit"].mean()) if (~low).any() else np.nan,
        false_windows_day=float(fire[normal].sum()/monitor_days(d)),
        false_notifications_day=fnot)

def main(root,out,seeds):
    seeds=tuple(map(int,seeds))
    loaded={s:C.load(root,"compact",s) for s in seeds}
    worlds={s:x[0] for s,x in loaded.items()}
    eligible={s:x[1] for s,x in loaded.items()}
    rows=[];episodes=[]
    for feature in FEATURES:
        scores={s:worlds[s][feature].to_numpy(float) for s in seeds}
        for held in seeds:
            cals=[x for x in seeds if x!=held]
            cut=float(max(world_cut(worlds[x],scores[x]) for x in cals))
            ep,m=evaluate(worlds[held],scores[held],cut,eligible[held])
            m.update(feature=feature,eval_seed=held,
                     cal_seed=";".join(map(str,cals)),threshold=cut)
            rows.append(m);ep["feature"]=feature;ep["eval_seed"]=held;episodes.append(ep)

    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    agg=df.groupby("feature").agg(
        mean_auc=("auc","mean"),min_auc=("auc","min"),
        mean_ap=("ap","mean"),min_ap=("ap","min"),
        mean_recall=("episode_recall","mean"),min_recall=("episode_recall","min"),
        mean_low_recall=("low_recall","mean"),mean_high_recall=("high_recall","mean"),
        worst_false_windows_day=("false_windows_day","max"),
        worst_false_notifications_day=("false_notifications_day","max")).reset_index()
    agg["budget_ok"]=(agg.worst_false_windows_day<=RAW_DAY)&(
        agg.worst_false_notifications_day<=NOTICE_DAY)
    ok=agg[agg.budget_ok].copy();cand=ok if len(ok) else agg
    cand=cand.sort_values(["min_recall","mean_recall","min_ap","mean_ap"],
                          ascending=False)
    winner=cand.iloc[0].to_dict()
    byrole=ed[ed.feature==winner["feature"]].groupby("role").agg(
        episodes=("episode","count"),recall=("hit","mean"),
        mean_peak=("peak","mean")).reset_index()
    df.to_csv(out/"cross_world.csv",index=False)
    agg.to_csv(out/"summary.csv",index=False)
    ed.to_csv(out/"episodes.csv",index=False)
    byrole.to_csv(out/"by_role.csv",index=False)
    (out/"winner.json").write_text(json.dumps(winner,indent=2))
    print("SUMMARY\n",agg.sort_values(
        ["min_recall","mean_recall","min_ap"],ascending=False).to_string(index=False),flush=True)
    print("BY_ROLE\n",byrole.to_string(index=False),flush=True)
    print("WINNER",json.dumps(winner,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,a.seeds)
