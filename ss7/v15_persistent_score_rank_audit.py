"""RANK-only persistent-score audit for v15 rolling subscriber KIE.

Instead of thresholding a single-window score and asking for persistence later,
the causal rolling minimum itself is calibrated as the detector score. This
tests whether long targeted episodes separate from short benign extremes.
No CAL/TEST data and no new simulator worlds are used.
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
LENGTHS=(5,10,20)  # 2.5, 5, 10 minutes
FEATURE=C.SUB_PEER_SERVICE_ROLL30_NAME

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def persistent(d,s,length):
    return C.streak_score(d,np.nan_to_num(s,nan=-np.inf),length)

def world_cut(d,z):
    m=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)&np.isfinite(z)
    return float(threshold(z[m],RAW_DAY,monitor_days(d)))

def evaluate(d,z,cut,eligible):
    tt=d.t.to_numpy();finite=np.isfinite(z)
    y=((d.attack=="compromise")&(d.sophistication=="naive")).to_numpy(int)
    valid=((d.label.to_numpy()==0)|y.astype(bool))&(tt>=BASELINE_END)&finite
    yy=y[valid];zz=z[valid]
    auc=float(roc_auc_score(yy,zz)) if yy.sum() and (~yy.astype(bool)).sum() else np.nan
    ap=float(average_precision_score(yy,zz)) if yy.sum() else np.nan
    neg=(d.label.to_numpy()==0)&(tt>=BASELINE_END)&finite
    false_day=float((z[neg]>cut).sum()/monitor_days(d))
    eps=[]
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy();e=eligible.get(int(eid))
        hit=bool(np.any(z[ix]>cut))
        first=np.flatnonzero(z[ix]>cut)
        eps.append(dict(episode=int(eid),role=str(g.ntype.iloc[0]),
            hidden_rate=float(e.hidden_rate) if e is not None else np.nan,
            hit=int(hit),delay_seconds=float(first[0]*30) if len(first) else np.nan,
            peak=float(np.max(z[ix]))))
    ep=pd.DataFrame(eps);low=ep.hidden_rate<.5
    return ep,dict(auc=auc,ap=ap,episodes=len(ep),
        episode_recall=float(ep.hit.mean()),
        low_recall=float(ep.loc[low,"hit"].mean()) if low.any() else np.nan,
        high_recall=float(ep.loc[~low,"hit"].mean()) if (~low).any() else np.nan,
        false_rows_day=false_day)

def main(root,out,seeds):
    seeds=tuple(int(x) for x in seeds)
    loaded={seed:C.load(root,"compact",seed) for seed in seeds}
    worlds={seed:x[0] for seed,x in loaded.items()}
    eligible={seed:x[1] for seed,x in loaded.items()}
    base={seed:worlds[seed][FEATURE].to_numpy(float) for seed in seeds}
    rows=[];episodes=[]
    for length in LENGTHS:
        score={seed:persistent(worlds[seed],base[seed],length) for seed in seeds}
        for held in seeds:
            cals=[x for x in seeds if x!=held]
            cut=float(max(world_cut(worlds[x],score[x]) for x in cals))
            ep,m=evaluate(worlds[held],score[held],cut,eligible[held])
            m.update(length=length,eval_seed=held,
                     cal_seed=";".join(map(str,cals)),threshold=cut)
            rows.append(m);ep["length"]=length;ep["eval_seed"]=held;episodes.append(ep)
    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    agg=df.groupby("length").agg(
        mean_auc=("auc","mean"),min_auc=("auc","min"),
        mean_ap=("ap","mean"),min_ap=("ap","min"),
        mean_recall=("episode_recall","mean"),min_recall=("episode_recall","min"),
        mean_low_recall=("low_recall","mean"),
        mean_high_recall=("high_recall","mean"),
        worst_false_rows_day=("false_rows_day","max")).reset_index()
    ok=agg[agg.worst_false_rows_day<=RAW_DAY].copy()
    cand=ok if len(ok) else agg
    cand=cand.sort_values(["min_recall","mean_recall","min_ap","mean_ap"],
                          ascending=[False,False,False,False])
    winner=cand.iloc[0].to_dict()
    chosen=int(winner["length"])
    byrole=ed[ed.length==chosen].groupby("role").agg(
        episodes=("episode","count"),recall=("hit","mean"),
        mean_peak=("peak","mean")).reset_index()
    df.to_csv(out/"cross_world.csv",index=False);agg.to_csv(out/"summary.csv",index=False)
    ed.to_csv(out/"episodes.csv",index=False);byrole.to_csv(out/"by_role.csv",index=False)
    (out/"winner.json").write_text(json.dumps(winner,indent=2))
    print("CROSS_WORLD\n",df.to_string(index=False),flush=True)
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
