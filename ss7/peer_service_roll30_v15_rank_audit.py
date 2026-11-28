"""RANK-only validation for rolling-before-residual peer-service KIE v15.

The feature is frozen before RANK inspection:
32 keyed bins, uint16 wire counters, service-stable benign hash probabilities, peer x exact MAP service cells, causal
30-interval (15 min) raw-count accumulation, commissioning-normalized max
positive residual. No CAL or TEST data are loaded.
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
FEATURE=C.SUB_PEER_SERVICE_ROLL30_NAME

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def world_cut(d,s):
    m=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)&np.isfinite(s)
    return float(threshold(s[m],RAW_DAY,monitor_days(d)))

def evaluate(d,s,cut,eligible):
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
        ix=g.index.to_numpy();e=eligible.get(int(eid))
        hit1=bool(np.any(s[ix]>cut));hit3=bool(np.any(z3[ix]>cut))
        first=np.flatnonzero(z3[ix]>cut)
        delay=float(first[0]*30) if len(first) else np.nan
        eps.append(dict(episode=int(eid),role=str(g.ntype.iloc[0]),
            hidden_rate=float(e.hidden_rate) if e is not None else np.nan,
            hit1=int(hit1),hit3=int(hit3),delay_seconds=delay,
            peak=float(np.nanmax(s[ix]))))
    ep=pd.DataFrame(eps)
    return ep,dict(auc=auc,ap=ap,episodes=len(ep),
        episode_recall_1=float(ep.hit1.mean()),
        episode_recall_3=float(ep.hit3.mean()),
        low_recall_3=float(ep.loc[ep.hidden_rate<.5,"hit3"].mean()),
        high_recall_3=float(ep.loc[ep.hidden_rate>=.5,"hit3"].mean()),
        false_rows_day=false_day)

def main(root,out,seeds):
    seeds=tuple(int(x) for x in seeds)
    loaded={seed:C.load(root,"compact",seed) for seed in seeds}
    worlds={seed:x[0] for seed,x in loaded.items()}
    eligible={seed:x[1] for seed,x in loaded.items()}
    scores={seed:worlds[seed][FEATURE].to_numpy(float) for seed in seeds}

    rows=[];episodes=[]
    for held in seeds:
        cals=[x for x in seeds if x!=held]
        cut=float(max(world_cut(worlds[x],scores[x]) for x in cals))
        ep,m=evaluate(worlds[held],scores[held],cut,eligible[held])
        m.update(eval_seed=held,cal_seed=";".join(map(str,cals)),threshold=cut)
        rows.append(m);ep["eval_seed"]=held;episodes.append(ep)

    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    byrole=ed.groupby("role").agg(
        episodes=("episode","count"),recall1=("hit1","mean"),
        recall3=("hit3","mean"),mean_peak=("peak","mean")).reset_index()
    summary=dict(
        mean_auc=float(df.auc.mean()),min_auc=float(df.auc.min()),
        mean_ap=float(df.ap.mean()),min_ap=float(df.ap.min()),
        mean_episode_recall_1=float(df.episode_recall_1.mean()),
        min_episode_recall_1=float(df.episode_recall_1.min()),
        mean_episode_recall_3=float(df.episode_recall_3.mean()),
        min_episode_recall_3=float(df.episode_recall_3.min()),
        mean_low_recall_3=float(df.low_recall_3.mean()),
        mean_high_recall_3=float(df.high_recall_3.mean()),
        worst_false_rows_day=float(df.false_rows_day.max()),
        accepted=bool((df.false_rows_day.max()<=RAW_DAY) and
                      (df.episode_recall_3.min()>0) and
                      (df.episode_recall_3.mean()>=.25)))
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
