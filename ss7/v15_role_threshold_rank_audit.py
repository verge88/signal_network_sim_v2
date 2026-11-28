"""RANK-only role-conditioned threshold audit for the frozen v15 score.

The four node roles have different benign tail distributions. A single global
cut lets one heavy-tailed role dominate calibration. This audit allocates the
fixed global raw-window budget across roles in proportion to their normal-row
exposure on calibration worlds, then applies one conservative threshold per
observable node role. Total budget remains 20 windows/day; TEST is not loaded.
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
FEATURE=C.SUB_PEER_SERVICE_ROLL30_NAME
CONFIRMS=(1,3)
ROLES=tuple(x.value for x in F.NodeType)

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def allocation(frames):
    counts={r:0 for r in ROLES}
    for d in frames:
        normal=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)
        rr=d.ntype.astype(str).to_numpy()
        for r in ROLES:counts[r]+=int(np.sum(normal&(rr==r)))
    total=max(sum(counts.values()),1)
    return {r:RAW_DAY*counts[r]/total for r in ROLES}

def role_thresholds(frames,scores):
    alloc=allocation(frames);cuts={}
    for r in ROLES:
        vals=[]
        for d,s in zip(frames,scores):
            m=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)&(
                d.ntype.astype(str).to_numpy()==r)&np.isfinite(s)
            if m.any():
                # threshold() budget is per WORLD-day, so role budget is the
                # allocated share of the global 20/day limit.
                vals.append(threshold(s[m],alloc[r],monitor_days(d)))
        cuts[r]=float(max(vals)) if vals else np.inf
    return cuts,alloc

def margin(d,s,cuts):
    rr=d.ntype.astype(str).to_numpy()
    th=np.array([cuts.get(str(r),np.inf) for r in rr],float)
    return np.asarray(s,float)-th

def evaluate(d,s,cuts,confirm,eligible):
    m=margin(d,s,cuts);tt=d.t.to_numpy();finite=np.isfinite(m)
    y=((d.attack=="compromise")&(d.sophistication=="naive")).to_numpy(int)
    valid=((d.label.to_numpy()==0)|y.astype(bool))&(tt>=BASELINE_END)&finite
    yy=y[valid];mm=m[valid]
    auc=float(roc_auc_score(yy,mm)) if yy.sum() and (~yy.astype(bool)).sum() else np.nan
    ap=float(average_precision_score(yy,mm)) if yy.sum() else np.nan

    neg=(d.label.to_numpy()==0)&(tt>=BASELINE_END)&finite
    fwin=float((m[neg]>0).sum()/monitor_days(d))
    z=m if confirm==1 else C.streak_score(d,np.nan_to_num(m,nan=-np.inf),confirm)
    nidx=notify(d,np.nan_to_num(z,nan=-np.inf),0.0)
    labels=d.label.to_numpy()
    fnot=sum(bool((labels[max(0,j-2):j+1]==0).all()) for j in nidx)/monitor_days(d)

    eps=[]
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy();e=eligible.get(int(eid))
        fire=z[ix]>0;first=np.flatnonzero(fire)
        eps.append(dict(episode=int(eid),role=str(g.ntype.iloc[0]),
            hidden_rate=float(e.hidden_rate) if e is not None else np.nan,
            hit=int(np.any(fire)),
            delay_seconds=float(first[0]*30) if len(first) else np.nan,
            peak_margin=float(np.nanmax(z[ix])) if np.isfinite(z[ix]).any() else np.nan))
    ep=pd.DataFrame(eps);low=ep.hidden_rate<.5
    return ep,dict(auc=auc,ap=ap,episodes=len(ep),
        episode_recall=float(ep.hit.mean()),
        low_recall=float(ep.loc[low,"hit"].mean()) if low.any() else np.nan,
        high_recall=float(ep.loc[~low,"hit"].mean()) if (~low).any() else np.nan,
        false_windows_day=fwin,false_notifications_day=float(fnot))

def main(root,out,seeds):
    seeds=tuple(int(x) for x in seeds)
    loaded={seed:C.load(root,"compact",seed) for seed in seeds}
    worlds={seed:x[0] for seed,x in loaded.items()}
    eligible={seed:x[1] for seed,x in loaded.items()}
    scores={seed:worlds[seed][FEATURE].to_numpy(float) for seed in seeds}
    rows=[];episodes=[];cut_rows=[]
    for confirm in CONFIRMS:
        for held in seeds:
            cals=[x for x in seeds if x!=held]
            frames=[worlds[x] for x in cals];sc=[scores[x] for x in cals]
            cuts,alloc=role_thresholds(frames,sc)
            ep,m=evaluate(worlds[held],scores[held],cuts,confirm,eligible[held])
            m.update(confirm=confirm,eval_seed=held,cal_seed=";".join(map(str,cals)))
            rows.append(m);ep["confirm"]=confirm;ep["eval_seed"]=held;episodes.append(ep)
            for r in ROLES:
                cut_rows.append(dict(confirm=confirm,eval_seed=held,role=r,
                                     threshold=cuts[r],budget_share=alloc[r]))
    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True);cd=pd.DataFrame(cut_rows)
    agg=df.groupby("confirm").agg(
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
                          ascending=[False,False,False,False])
    winner=cand.iloc[0].to_dict();confirm=int(winner["confirm"])
    byrole=ed[ed.confirm==confirm].groupby("role").agg(
        episodes=("episode","count"),recall=("hit","mean"),
        mean_peak_margin=("peak_margin","mean")).reset_index()
    df.to_csv(out/"cross_world.csv",index=False);agg.to_csv(out/"summary.csv",index=False)
    cd.to_csv(out/"thresholds.csv",index=False);ed.to_csv(out/"episodes.csv",index=False)
    byrole.to_csv(out/"by_role.csv",index=False)
    (out/"winner.json").write_text(json.dumps(winner,indent=2))
    print("SUMMARY\n",agg.to_string(index=False),flush=True)
    print("BY_ROLE\n",byrole.to_string(index=False),flush=True)
    print("WINNER",json.dumps(winner,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,a.seeds)
