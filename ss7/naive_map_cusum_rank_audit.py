"""RANK-only sequential detector for persistent NAIVE SS7 compromise.

Observable input only: compact-KIE message totals and MAP operation ratios.
One fixed detector is tested: recent 5 intervals vs preceding 60-interval
baseline (5-interval guard), followed by a one-sided leaky CUSUM.
No CAL/TEST data and no hyperparameter grid search are used.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from ss7 import fix as F
from ss7 import cu_compact_channel_development as C
from ss7.incident_calibration_study import threshold

BASELINE_END=180
RAW_DAY=4.0
RECENT=5
BASE=60
GAP=5
LEAK=.98
K=.50
MAP_GROUPS=(
    ("SRI_SM","MT_FSM","MO_FSM"),
    ("ATI","PSI","UL","ISD","CL","PURGE"),
    ("SRI","PRN","SAI"),
)

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def group_counts(d):
    total=d.total_messages.to_numpy(float)
    mshare=d.map_share.to_numpy(float)
    map_total=np.maximum(total*mshare,0.0)
    out=[]
    for names in MAP_GROUPS:
        ratio=np.zeros(len(d),float)
        valid=np.ones(len(d),bool)
        for m in names:
            v=d[f"obs_ratio_{m}"].to_numpy(float)
            valid&=np.isfinite(v)
            ratio+=np.nan_to_num(v,nan=0.0)
        x=map_total*ratio
        # A stale report must not be replayed as a new count observation.
        valid&=(d.cu_delivered.to_numpy(float)>0)&np.isfinite(map_total)
        x[~valid]=np.nan
        out.append(x)
    return out

def innovation(d,x):
    out=np.full(len(d),np.nan)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy();a=x[ix]
        s=pd.Series(a,index=ix)
        recent=s.rolling(RECENT,min_periods=3).mean()
        past=s.shift(GAP)
        mu=past.rolling(BASE,min_periods=30).mean()
        sd=past.rolling(BASE,min_periods=30).std()
        # Poisson floor + empirical variation makes low-volume MAP groups stable.
        scale=np.sqrt(np.maximum(mu,1.0)/RECENT + np.maximum(sd,0.0)**2/RECENT)
        z=(recent-mu)/(scale+1e-6)
        out[ix]=np.clip(z.to_numpy(float),-6.0,6.0)
    return out

def score(d):
    parts=[innovation(d,x) for x in group_counts(d)]
    z=np.nanmax(np.column_stack(parts),axis=1)
    z=np.where(np.isfinite(z),z,np.nan)
    out=np.zeros(len(d),float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy();c=0.0
        for j in ix:
            v=z[j]
            if not np.isfinite(v):
                c=max(0.0,LEAK*c-.10)
            else:
                c=max(0.0,LEAK*c+max(v,0.0)-K)
            out[j]=c
    return np.arcsinh(out)

def world_cut(d,s):
    neg=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)
    return float(threshold(s[neg],RAW_DAY,monitor_days(d)))

def evaluate(d,s,cut):
    tt=d.t.to_numpy();normal=(d.label.to_numpy()==0)&(tt>=BASELINE_END)
    y=((d.attack=="compromise")&(d.sophistication=="naive")).to_numpy(int)
    valid=((d.label.to_numpy()==0)|y.astype(bool))&(tt>=BASELINE_END)
    yy=y[valid];ss=s[valid]
    z3=C.streak_score(d,s,3)
    eps=[]
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy()
        eps.append(dict(episode=int(eid),role=str(g.ntype.iloc[0]),
            hidden_rate=float(g.hidden_rate.iloc[0]) if "hidden_rate" in g else np.nan,
            detected_1=int(np.any(s[ix]>cut)),
            detected_3=int(np.any(z3[ix]>cut)),
            peak=float(np.max(s[ix]))))
    ep=pd.DataFrame(eps)
    return ep,dict(
        auc=float(roc_auc_score(yy,ss)),ap=float(average_precision_score(yy,ss)),
        episodes=len(ep),episode_recall_1=float(ep.detected_1.mean()),
        episode_recall_3=float(ep.detected_3.mean()),
        false_rows_day=float((s[normal]>cut).sum()/monitor_days(d)))

def main(root,out,seeds):
    seeds=tuple(int(x) for x in seeds)
    worlds={seed:C.load(root,"compact",seed)[0] for seed in seeds}
    scores={seed:score(d) for seed,d in worlds.items()}
    rows=[];episodes=[]
    for held in seeds:
        cals=[x for x in seeds if x!=held]
        cut=float(max(world_cut(worlds[x],scores[x]) for x in cals))
        ep,m=evaluate(worlds[held],scores[held],cut)
        m.update(eval_seed=held,cal_seed=";".join(map(str,cals)),threshold=cut)
        rows.append(m);ep["eval_seed"]=held;ep["threshold"]=cut;episodes.append(ep)
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
