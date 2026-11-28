"""RANK-only v8.1: role-stratified calibration of peer subscriber KIE.

Feature and smoothing are frozen from v8 RANK diagnosis:
kie_peer_sub_max_resid, causal 5-window mean. The global 4 false-row/day
budget is split equally across the four observable SS7 node roles.
No CAL or TEST data are read.
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
ROLES=tuple(x.value for x in F.NodeType)
ROLE_DAY=RAW_DAY/len(ROLES)
FEATURE="kie_peer_sub_max_resid"

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

def role_cut(d,s,role):
    mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)&         (d.t.to_numpy()>=BASELINE_END)&np.isfinite(s)
    return float(threshold(s[mask],ROLE_DAY,monitor_days(d)))

def threshold_vector(d,cuts):
    role=d.ntype.astype(str).to_numpy()
    return np.array([cuts.get(x,np.inf) for x in role],float)

def evaluate(d,s,cuts):
    cut=threshold_vector(d,cuts)
    margin=s-cut
    finite=np.isfinite(margin)
    tt=d.t.to_numpy()
    y=((d.attack=="compromise")&(d.sophistication=="naive")).to_numpy(int)
    valid=((d.label.to_numpy()==0)|y.astype(bool))&(tt>=BASELINE_END)&finite
    yy=y[valid];mm=margin[valid]
    auc=float(roc_auc_score(yy,mm)) if yy.sum() and (~yy.astype(bool)).sum() else np.nan
    ap=float(average_precision_score(yy,mm)) if yy.sum() else np.nan
    neg=(d.label.to_numpy()==0)&(tt>=BASELINE_END)&finite
    false_day=float((margin[neg]>0).sum()/monitor_days(d))
    z3=C.streak_score(d,np.nan_to_num(margin,nan=-np.inf),3)

    eps=[]
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy()
        eps.append(dict(episode=int(eid),role=str(g.ntype.iloc[0]),
            detected_1=int(np.any(margin[ix]>0)),
            detected_3=int(np.any(z3[ix]>0)),
            peak_margin=float(np.nanmax(margin[ix]))))
    ep=pd.DataFrame(eps)
    return ep,dict(auc=auc,ap=ap,episodes=len(ep),
        episode_recall_1=float(ep.detected_1.mean()),
        episode_recall_3=float(ep.detected_3.mean()),
        false_rows_day=false_day)

def main(root,out,seeds):
    seeds=tuple(int(x) for x in seeds)
    worlds={seed:C.load(root,"compact",seed)[0] for seed in seeds}
    scores={seed:score(d) for seed,d in worlds.items()}
    rows=[];episodes=[]
    for held in seeds:
        cals=[x for x in seeds if x!=held]
        cuts={}
        for role in ROLES:
            cuts[role]=float(max(role_cut(worlds[x],scores[x],role) for x in cals))
        ep,m=evaluate(worlds[held],scores[held],cuts)
        m.update(eval_seed=held,cal_seed=";".join(map(str,cals)))
        rows.append(m);ep["eval_seed"]=held;episodes.append(ep)
    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    byrole=ed.groupby("role").agg(episodes=("episode","count"),
        recall_1=("detected_1","mean"),recall_3=("detected_3","mean"),
        mean_peak_margin=("peak_margin","mean")).reset_index()
    summary=dict(mean_auc=float(df.auc.mean()),min_auc=float(df.auc.min()),
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
