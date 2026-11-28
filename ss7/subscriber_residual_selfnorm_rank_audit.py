"""Minimal RANK-only v7.1 audit: node-self-normalized subscriber residual.

Method choice is intentionally frozen: max_resid -> node commissioning robust
normalization -> 5-window causal mean -> one global cross-world threshold.
No CAL/TEST data and no variant search are used here.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from ss7 import fix as F
from ss7 import cu_compact_channel_development as C
from ss7.incident_calibration_study import threshold

BASELINE_END=int(os.getenv("SS7_DEPLOY_BASELINE","180"))
RAW_DAY=4.0
SMOOTH=5
MAP_GROUPS=(1,2,3)

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def raw_score(d):
    cols=[f"kie_sub_g{g}_max_resid" for g in MAP_GROUPS]
    x=np.nan_to_num(d[cols].to_numpy(float),nan=0.0,posinf=0.0,neginf=0.0)
    return np.max(x,axis=1)

def selfnorm(d):
    raw=raw_score(d);out=np.zeros(len(d),float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy();tt=g.t.to_numpy();x=raw[ix]
        warm=(tt<BASELINE_END)&np.isfinite(x)
        a=x[warm]
        if len(a)<20:continue
        center=float(np.median(a));q25,q75=np.quantile(a,[.25,.75])
        scale=max(float((q75-q25)/1.349),0.10)
        z=np.maximum((x-center)/scale,0.0)
        out[ix]=np.arcsinh(z)
    return out

def smooth(d,s,w=SMOOTH):
    out=np.zeros(len(d),float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        out[ix]=pd.Series(s[ix],index=ix).rolling(
            w,min_periods=max(2,w//2)).mean().fillna(0.0).to_numpy()
    return out

def world_cut(d,s):
    neg=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)
    return float(threshold(s[neg],RAW_DAY,monitor_days(d)))

def episode_metrics(d,s,cut):
    t=d.t.to_numpy();normal=(d.label.to_numpy()==0)&(t>=BASELINE_END)
    z3=C.streak_score(d,s,3)
    eps=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    rows=[]
    for eid,g in eps.groupby("episode_id",sort=False):
        ix=g.index.to_numpy();role=str(g.ntype.iloc[0])
        rows.append(dict(episode=int(eid),role=role,
            detected_1=int(np.any(s[ix]>cut)),
            detected_3=int(np.any(z3[ix]>cut)),
            peak=float(np.max(s[ix]))))
    ep=pd.DataFrame(rows)
    y=((d.attack=="compromise")&(d.sophistication=="naive")).to_numpy(int)
    valid=((d.label.to_numpy()==0)|y.astype(bool))&(t>=BASELINE_END)
    yy=y[valid];ss=s[valid]
    return ep,dict(
        auc=float(roc_auc_score(yy,ss)),
        ap=float(average_precision_score(yy,ss)),
        episodes=len(ep),
        episode_recall_1=float(ep.detected_1.mean()),
        episode_recall_3=float(ep.detected_3.mean()),
        false_rows_day=float((s[normal]>cut).sum()/monitor_days(d))
    )

def main(root,out,seeds):
    seeds=tuple(int(x) for x in seeds)
    worlds={seed:C.load(root,"compact",seed)[0] for seed in seeds}
    scores={seed:smooth(d,selfnorm(d)) for seed,d in worlds.items()}
    rows=[];episodes=[]
    for held in seeds:
        cals=[x for x in seeds if x!=held]
        cut=float(max(world_cut(worlds[x],scores[x]) for x in cals))
        ep,m=episode_metrics(worlds[held],scores[held],cut)
        m.update(eval_seed=held,cal_seed=";".join(map(str,cals)),threshold=cut)
        rows.append(m)
        ep["eval_seed"]=held;ep["threshold"]=cut;episodes.append(ep)
    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    byrole=ed.groupby("role").agg(
        episodes=("episode","count"),
        recall_1=("detected_1","mean"),
        recall_3=("detected_3","mean"),
        mean_peak=("peak","mean")).reset_index()
    summary=dict(
        mean_auc=float(df.auc.mean()),min_auc=float(df.auc.min()),
        mean_ap=float(df.ap.mean()),min_ap=float(df.ap.min()),
        mean_episode_recall_1=float(df.episode_recall_1.mean()),
        min_episode_recall_1=float(df.episode_recall_1.min()),
        mean_episode_recall_3=float(df.episode_recall_3.mean()),
        min_episode_recall_3=float(df.episode_recall_3.min()),
        worst_global_false_rows_day=float(df.false_rows_day.max()),
        budget_ok=bool(df.false_rows_day.max()<=RAW_DAY))
    df.to_csv(out/"cross_world.csv",index=False)
    ed.to_csv(out/"episodes.csv",index=False)
    byrole.to_csv(out/"by_role.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("CROSS_WORLD\n",df.to_string(index=False),flush=True)
    print("BY_ROLE\n",byrole.to_string(index=False),flush=True)
    print("SUMMARY",json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,a.seeds)
