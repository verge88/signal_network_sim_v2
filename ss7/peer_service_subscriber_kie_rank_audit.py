"""RANK-only validation for peer x exact-MAP-service subscriber KIE v9.

Inputs are coordinator-visible residuals derived from keyed subscriber hash
counters conditioned on adjacent SS7 peer and exact MAP service. Raw subscriber
identifiers are never exported. No CAL or TEST world is loaded.
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
VARIANTS=("kie_peer_service_sub_max_resid","kie_peer_service_sub_rms_resid",
          "kie_peer_service_sub_chi2")
SMOOTH=(1,3,5)

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def smooth(d,s,w):
    if w==1:return np.asarray(s,float)
    out=np.full(len(d),np.nan)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        out[ix]=pd.Series(s[ix],index=ix).rolling(
            w,min_periods=max(1,w//2)).mean().to_numpy()
    return out

def world_cut(d,s):
    neg=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)&np.isfinite(s)
    return float(threshold(s[neg],RAW_DAY,monitor_days(d)))

def evaluate(d,s,cut):
    tt=d.t.to_numpy()
    y=((d.attack=="compromise")&(d.sophistication=="naive")).to_numpy(int)
    valid=((d.label.to_numpy()==0)|y.astype(bool))&(tt>=BASELINE_END)&np.isfinite(s)
    yy=y[valid];ss=s[valid]
    auc=float(roc_auc_score(yy,ss)) if yy.sum() and (~yy.astype(bool)).sum() else np.nan
    ap=float(average_precision_score(yy,ss)) if yy.sum() else np.nan
    neg=(d.label.to_numpy()==0)&(tt>=BASELINE_END)&np.isfinite(s)
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
    rows=[];episodes=[]
    for variant in VARIANTS:
        base={seed:worlds[seed][variant].to_numpy(float) for seed in seeds}
        for w in SMOOTH:
            score={seed:smooth(worlds[seed],base[seed],w) for seed in seeds}
            for held in seeds:
                cals=[x for x in seeds if x!=held]
                cut=float(max(world_cut(worlds[x],score[x]) for x in cals))
                ep,m=evaluate(worlds[held],score[held],cut)
                m.update(variant=variant,smooth=w,eval_seed=held,
                    cal_seed=";".join(map(str,cals)),threshold=cut)
                rows.append(m)
                ep["variant"]=variant;ep["smooth"]=w;ep["eval_seed"]=held
                episodes.append(ep)
    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    agg=df.groupby(["variant","smooth"]).agg(
        mean_auc=("auc","mean"),min_auc=("auc","min"),
        mean_ap=("ap","mean"),min_ap=("ap","min"),
        mean_episode_recall_1=("episode_recall_1","mean"),
        min_episode_recall_1=("episode_recall_1","min"),
        mean_episode_recall_3=("episode_recall_3","mean"),
        min_episode_recall_3=("episode_recall_3","min"),
        worst_false_rows_day=("false_rows_day","max")).reset_index()
    accepted=agg[(agg.worst_false_rows_day<=RAW_DAY)&
                 (agg.min_episode_recall_3>0)].copy()
    ranking=accepted if len(accepted) else agg
    ranking=ranking.sort_values(
        ["min_episode_recall_3","mean_episode_recall_3",
         "min_episode_recall_1","mean_episode_recall_1","min_ap","mean_ap"],
        ascending=[False,False,False,False,False,False])
    winner=ranking.iloc[0].to_dict()
    winner["accepted"]=bool(len(accepted))
    byrole=ed[(ed.variant==winner["variant"])&
              (ed.smooth==int(winner["smooth"]))].groupby("role").agg(
        episodes=("episode","count"),recall_1=("detected_1","mean"),
        recall_3=("detected_3","mean"),mean_peak=("peak","mean")).reset_index()
    df.to_csv(out/"cross_world.csv",index=False)
    agg.to_csv(out/"summary.csv",index=False)
    ed.to_csv(out/"episodes.csv",index=False)
    byrole.to_csv(out/"by_role.csv",index=False)
    (out/"winner.json").write_text(json.dumps(winner,indent=2))
    print("SUMMARY\n",agg.sort_values(
        ["min_episode_recall_3","mean_episode_recall_3","min_ap"],
        ascending=False).to_string(index=False),flush=True)
    print("BY_ROLE\n",byrole.to_string(index=False),flush=True)
    print("WINNER",json.dumps(winner,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,a.seeds)
