"""Fresh RANK-only audit for independent authorization/capability KIE v23.

The detector is a fixed security invariant, not a trained classifier. A high-risk
MAP operation is counted by an independent verifier; operations without a valid
capability context contribute to a 5-minute rolling missing-capability count.
Only aggregate counters are visible to the coordinator. CAL/TEST are not loaded.

Two predeclared operating points are evaluated without RANK threshold tuning:
  - count3: >=3 missing capabilities in 5 minutes
  - count5: >=5 missing capabilities in 5 minutes
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

from ss7 import fix as F
from ss7 import cu_compact_channel_development as C

BASELINE_END=int(os.getenv("SS7_DEPLOY_BASELINE","180"))
RULES=(("count3",3.0),("count5",5.0))
RAW_DAY=20.0
NOTICE_DAY=2.0
REFRACTORY=60  # 30 minutes at 30 s/window
COMP_SOPH=("naive","statistical","adaptive")
EXTERNAL=("sms_interception","location_tracking","signalling_dos","irsf")

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def notifications(d,fire):
    """Node-local rising-edge notifications with a 30-minute refractory period."""
    out=[]
    t=d.t.to_numpy();nid=d.nid.to_numpy()
    for node in np.unique(nid):
        ix=np.flatnonzero(nid==node)
        last=-10**9;prev=False
        for j in ix:
            cur=bool(fire[j])
            if cur and (not prev) and int(t[j])-last>=REFRACTORY:
                out.append(int(j));last=int(t[j])
            prev=cur
    return out

def episode_table(d,fire,eligible):
    rows=[]
    attacks=d[d.episode_id>=0]
    for eid,g in attacks.groupby("episode_id",sort=False):
        ix=g.index.to_numpy()
        e=eligible.get(int(eid))
        hit=np.flatnonzero(fire[ix])
        rows.append(dict(
            episode=int(eid),
            attack=str(g.attack.iloc[0]),
            sophistication=str(g.sophistication.iloc[0]),
            role=str(g.ntype.iloc[0]),
            hidden_rate=float(e.hidden_rate) if e is not None and
                        getattr(e,"kind","")=="compromise" else np.nan,
            hit=int(len(hit)>0),
            delay_seconds=float(hit[0]*30) if len(hit) else np.nan,
            peak_missing_5m=float(np.nanmax(
                g.kie_cap_missing_5m.to_numpy(float))) if
                np.isfinite(g.kie_cap_missing_5m.to_numpy(float)).any() else np.nan))
    return pd.DataFrame(rows)

def evaluate(d,eligible,threshold):
    score=d.kie_cap_missing_5m.to_numpy(float)
    eligible_time=d.t.to_numpy()>=BASELINE_END
    fire=eligible_time&np.isfinite(score)&(score>=threshold)
    normal=(d.label.to_numpy()==0)&eligible_time
    notes=notifications(d,fire)
    ep=episode_table(d,fire,eligible)

    comp=ep.attack=="compromise"
    naive=comp&(ep.sophistication=="naive")
    stat=comp&(ep.sophistication=="statistical")
    adap=comp&(ep.sophistication=="adaptive")
    ext=ep.attack.isin(EXTERNAL)
    days=monitor_days(d)
    m=dict(
        compromise_recall=float(ep.loc[comp,"hit"].mean()),
        naive_recall=float(ep.loc[naive,"hit"].mean()),
        statistical_recall=float(ep.loc[stat,"hit"].mean()),
        adaptive_recall=float(ep.loc[adap,"hit"].mean()),
        external_episode_activation=float(ep.loc[ext,"hit"].mean()),
        false_windows_day=float((fire&normal).sum()/days),
        false_notifications_day=float(sum(bool(normal[j]) for j in notes)/days),
        notifications_day=float(len(notes)/days))
    return ep,m

def main(root,out,seeds):
    seeds=tuple(map(int,seeds))
    rows=[];episodes=[]
    overhead=[]
    for seed in seeds:
        d,eligible,cu=C.load(root,"compact",seed)
        overhead.append(dict(seed=seed,**cu))
        required=set(C.CAP_KIE_NAMES+C.CAP_DERIVED_NAMES)
        miss=sorted(required-set(d.columns))
        if miss:raise KeyError(f"missing v23 fields: {miss}")
        for rule,thr in RULES:
            ep,m=evaluate(d,eligible,thr)
            m.update(seed=seed,rule=rule,threshold=thr)
            rows.append(m)
            ep["seed"]=seed;ep["rule"]=rule;ep["threshold"]=thr
            episodes.append(ep)

    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    agg=df.groupby("rule").agg(
        mean_compromise_recall=("compromise_recall","mean"),
        min_compromise_recall=("compromise_recall","min"),
        mean_naive_recall=("naive_recall","mean"),
        min_naive_recall=("naive_recall","min"),
        mean_statistical_recall=("statistical_recall","mean"),
        min_statistical_recall=("statistical_recall","min"),
        mean_adaptive_recall=("adaptive_recall","mean"),
        min_adaptive_recall=("adaptive_recall","min"),
        worst_external_episode_activation=("external_episode_activation","max"),
        worst_false_windows_day=("false_windows_day","max"),
        worst_false_notifications_day=("false_notifications_day","max"),
        worst_notifications_day=("notifications_day","max")).reset_index()
    agg["threshold"]=agg["rule"].map(dict(RULES)).astype(float)
    agg["budget_ok"]=(agg.worst_false_windows_day<=RAW_DAY)&(
        agg.worst_false_notifications_day<=NOTICE_DAY)
    agg["accepted"]=(agg.budget_ok&
        (agg.min_naive_recall>=.90)&
        (agg.min_statistical_recall>=.90)&
        (agg.min_adaptive_recall>=.90)&
        (agg.worst_external_episode_activation<=.05))

    # Prefer the stricter accepted rule; otherwise report best worst-case recall.
    accepted=agg[agg.accepted].copy()
    cand=accepted if len(accepted) else agg.copy()
    cand=cand.sort_values(
        ["min_naive_recall","min_statistical_recall","min_adaptive_recall",
         "worst_false_notifications_day","worst_false_windows_day","threshold"],
        ascending=[False,False,False,True,True,False])
    winner=cand.iloc[0].to_dict()

    wrule=winner["rule"]
    sub=ed[ed.rule==wrule]
    byrole=sub[sub.attack=="compromise"].groupby("role").agg(
        episodes=("episode","count"),recall=("hit","mean"),
        mean_delay_seconds=("delay_seconds","mean"),
        mean_peak_missing_5m=("peak_missing_5m","mean")).reset_index()
    bysoph=sub[sub.attack=="compromise"].groupby("sophistication").agg(
        episodes=("episode","count"),recall=("hit","mean"),
        mean_delay_seconds=("delay_seconds","mean"),
        mean_peak_missing_5m=("peak_missing_5m","mean")).reset_index()
    byattack=sub.groupby("attack").agg(
        episodes=("episode","count"),activation=("hit","mean")).reset_index()

    df.to_csv(out/"cross_world.csv",index=False)
    agg.to_csv(out/"summary.csv",index=False)
    ed.to_csv(out/"episodes.csv",index=False)
    byrole.to_csv(out/"by_role.csv",index=False)
    bysoph.to_csv(out/"by_sophistication.csv",index=False)
    byattack.to_csv(out/"by_attack.csv",index=False)
    pd.DataFrame(overhead).to_csv(out/"cu_overhead.csv",index=False)
    (out/"winner.json").write_text(json.dumps(winner,indent=2))

    print("CROSS_WORLD\n",df.to_string(index=False),flush=True)
    print("SUMMARY\n",agg.to_string(index=False),flush=True)
    print("BY_ROLE\n",byrole.to_string(index=False),flush=True)
    print("BY_SOPH\n",bysoph.to_string(index=False),flush=True)
    print("BY_ATTACK\n",byattack.to_string(index=False),flush=True)
    print("WINNER",json.dumps(winner,indent=2),flush=True)
    print("CU_OVERHEAD\n",pd.DataFrame(overhead).to_string(index=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,a.seeds)
