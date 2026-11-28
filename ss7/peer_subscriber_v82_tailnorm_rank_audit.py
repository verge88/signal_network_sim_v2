"""RANK-only v8.2: node commissioning tail-normalization.

Frozen method: peer max residual -> per-node q95/q99 tail scale -> positive
asinh transform -> causal 5-window mean -> global cross-world threshold.
No CAL/TEST data and no parameter grid search.
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
FEATURE="kie_peer_sub_max_resid"
SMOOTH=5

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def tailnorm(d):
    x=d[FEATURE].to_numpy(float)
    out=np.full(len(d),np.nan)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy();tt=g.t.to_numpy();v=x[ix]
        warm=(tt<BASELINE_END)&np.isfinite(v)
        a=v[warm]
        if len(a)<40:continue
        q95,q99=np.quantile(a,[.95,.99])
        scale=max(float(q99-q95),0.10)
        z=np.maximum((v-q95)/scale,0.0)
        z=np.arcsinh(z)
        out[ix]=pd.Series(z,index=ix).rolling(
            SMOOTH,min_periods=2).mean().to_numpy()
    return out

def world_cut(d,s):
    neg=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)&np.isfinite(s)
    return float(threshold(s[neg],RAW_DAY,monitor_days(d)))

def evaluate(d,s,cut):
    tt=d.t.to_numpy();finite=np.isfinite(s)
    y=((d.attack=="compromise")&(d.sophistication=="naive")).to_numpy(int)
    valid=((d.label.to_numpy()==0)|y.astype(bool))&(tt>=BASELINE_END)&finite
    yy=y[valid];ss=s[valid]
    auc=float(roc_auc_score(yy,ss)) if yy.sum() and (~yy.astype(bool)).sum() else np.nan
    ap=float(average_precision_score(yy,ss)) if yy.sum() else np.nan
    neg=(d.label.to_numpy()==0)&(tt>=BASELINE_END)&finite
    false_day=float((s[neg]>cut).sum()/monitor_days(d))
    z3=C.streak_score(d,np.nan_to_num(s,nan=-np.inf),3)
    eps=[]
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy()
        eps.append(dict(episode=int(eid),role=str(g.ntype.iloc[0]),
            detected_1=int(np.any(s[ix]>cut)),
            detected_3=int(np.any(z3[ix]>cut)),
            peak=float(np.nanmax(s[ix]))))
    ep=pd.DataFrame(eps)
    return ep,dict(auc=auc,ap=ap,episodes=len(ep),
        episode_recall_1=float(ep.detected_1.mean()),
        episode_recall_3=float(ep.detected_3.mean()),
        false_rows_day=false_day)

def main(root,out,seeds):
    seeds=tuple(int(x) for x in seeds)
    worlds={seed:C.load(root,"compact",seed)[0] for seed in seeds}
    scores={seed:tailnorm(d) for seed,d in worlds.items()}
    rows=[];episodes=[]
    for held in seeds:
        cals=[x for x in seeds if x!=held]
        cut=float(max(world_cut(worlds[x],scores[x]) for x in cals))
        ep,m=evaluate(worlds[held],scores[held],cut)
        m.update(eval_seed=held,cal_seed=";".join(map(str,cals)),threshold=cut)
        rows.append(m);ep["eval_seed"]=held;episodes.append(ep)
    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    byrole=ed.groupby("role").agg(episodes=("episode","count"),
        recall_1=("detected_1","mean"),recall_3=("detected_3","mean"),
        mean_peak=("peak","mean")).reset_index()
    summary=dict(mean_auc=float(df.auc.mean()),min_auc=float(df.auc.min()),
        mean_ap=float(df.ap.mean()),min_ap=float(df.ap.min()),
        mean_episode_recall_1=float(df.episode_recall_1.mean()),
        min_episode_recall_1=float(df.episode_recall_1.min()),
        mean_episode_recall_3=float(df.episode_recall_3.mean()),
        min_episode_recall_3=float(df.episode_recall_3.min()),
        worst_global_false_rows_day=float(df.false_rows_day.max()),
        budget_ok=bool(df.false_rows_day.max()<=RAW_DAY))
    df.to_csv(out/"cross_world.csv",index=False);ed.to_csv(out/"episodes.csv",index=False)
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
