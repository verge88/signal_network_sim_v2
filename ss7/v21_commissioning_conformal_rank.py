"""Deployment-conformal RANK audit for the frozen v19 detector.

The v19 logistic model is frozen. Each deployment world calibrates itself only
from its attack-free commissioning interval: per-node robust normalization,
global empirical tail ranking, 5-minute decision cadence, and two consecutive
decision confirmations. No attack label is used for calibration and CAL/TEST
are never loaded.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from ss7 import fix as F
from ss7 import cu_compact_channel_development as C
from ss7 import v19_block_logit_rank as V19

BASELINE_END=int(os.getenv("SS7_DEPLOY_BASELINE","180"))
RAW_DAY=20.0
NOTICE_DAY=2.0
HORIZON=10          # 5-minute causal average
CADENCE=10          # one decision every 5 minutes
CONFIRM_DECISIONS=2
EPS=1e-6

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return np.log(p)-np.log1p(-p)

def normalized_rolling(d,p):
    x=logit(p);z=np.full(len(d),np.nan);r=np.full(len(d),np.nan)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy();tt=g.t.to_numpy();a=x[ix]
        warm=(tt<BASELINE_END)&np.isfinite(a)
        if warm.sum()<30:continue
        med=float(np.median(a[warm]))
        mad=float(np.median(np.abs(a[warm]-med)))
        scale=max(1.4826*mad,.15)
        zz=np.clip((a-med)/scale,-8.0,8.0);z[ix]=zz
        rr=pd.Series(zz,index=ix).rolling(HORIZON,min_periods=7).mean()
        r[ix]=rr.to_numpy(float)
    return r

def commissioning_tail(d,s):
    tt=d.t.to_numpy()
    base=(tt<BASELINE_END)&np.isfinite(s)
    vals=np.sort(s[base])
    if len(vals)<200:raise ValueError("insufficient commissioning tail support")
    return vals

def empirical_survival(vals,x):
    x=np.asarray(x,float)
    # Conformal-style finite-sample upper-tail probability.
    rank=len(vals)-np.searchsorted(vals,x,side="left")
    return (rank+1.0)/(len(vals)+1.0)

def decision_stream(d,s):
    tt=d.t.to_numpy()
    decision=(tt>=BASELINE_END)&(tt%CADENCE==0)&np.isfinite(s)
    return decision

def evaluate(d,s,eligible):
    vals=commissioning_tail(d,s)
    decision=decision_stream(d,s)
    ndec_day=float(decision.sum()/monitor_days(d))
    # Fixed deployment budget: expected raw decision alarms <= RAW_DAY/day.
    alpha=min(.25,RAW_DAY/max(ndec_day,1.0))
    p=np.ones(len(d),float)
    p[decision]=empirical_survival(vals,s[decision])
    raw=decision&(p<=alpha)

    # Two consecutive 5-minute decisions on the same node are needed.
    confirmed=np.zeros(len(d),bool)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy();di=ix[decision[ix]]
        if len(di)<CONFIRM_DECISIONS:continue
        for j in range(CONFIRM_DECISIONS-1,len(di)):
            seg=di[j-CONFIRM_DECISIONS+1:j+1]
            if np.all(raw[seg]) and np.all(np.diff(d.t.to_numpy()[seg])==CADENCE):
                confirmed[di[j]]=True

    # Notifications have a 30-minute refractory period per node.
    notices=[]
    refractory=60
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy();last=-10**9
        for j in ix[confirmed[ix]]:
            tj=int(d.t.iloc[j])
            if tj-last>=refractory:
                notices.append(j);last=tj

    y=V19.target(d)
    valid=((d.label.to_numpy()==0)|y.astype(bool))&(d.t.to_numpy()>=BASELINE_END)&np.isfinite(s)
    yy=y[valid];ss=-np.log10(np.clip(p[valid],1e-12,1.0))
    auc=float(roc_auc_score(yy,ss)) if yy.sum() and (~yy.astype(bool)).sum() else np.nan
    ap=float(average_precision_score(yy,ss)) if yy.sum() else np.nan

    normal=d.label.to_numpy()==0
    false_decisions_day=float((confirmed&normal).sum()/monitor_days(d))
    false_notices_day=float(sum(bool(normal[j]) for j in notices)/monitor_days(d))

    eps=[]
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy();e=eligible.get(int(eid))
        hit=np.flatnonzero(confirmed[ix])
        eps.append(dict(
            episode=int(eid),role=str(g.ntype.iloc[0]),
            hidden_rate=float(e.hidden_rate) if e is not None else np.nan,
            hit=int(len(hit)>0),delay_seconds=float(hit[0]*30) if len(hit) else np.nan,
            min_p=float(np.min(p[ix])) if len(ix) else 1.0))
    ep=pd.DataFrame(eps);low=ep.hidden_rate<.5
    return ep,dict(
        auc=auc,ap=ap,alpha=float(alpha),decision_rows_day=ndec_day,
        episode_recall=float(ep.hit.mean()),
        low_recall=float(ep.loc[low,"hit"].mean()) if low.any() else np.nan,
        high_recall=float(ep.loc[~low,"hit"].mean()) if (~low).any() else np.nan,
        false_windows_day=false_decisions_day,
        false_notifications_day=false_notices_day,
        commissioning_n=len(vals))

def main(root,model_path,out,seeds):
    seeds=tuple(map(int,seeds))
    loaded={s:C.load(root,"compact",s) for s in seeds}
    model=joblib.load(model_path)
    rows=[];episodes=[]
    for seed in seeds:
        d,eligible,_=loaded[seed]
        p=V19.score(d,model)
        s=normalized_rolling(d,p)
        ep,m=evaluate(d,s,eligible)
        m["seed"]=seed;rows.append(m)
        ep["seed"]=seed;episodes.append(ep)

    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    summary=dict(
        mean_auc=float(df.auc.mean()),min_auc=float(df.auc.min()),
        mean_ap=float(df.ap.mean()),min_ap=float(df.ap.min()),
        mean_recall=float(df.episode_recall.mean()),
        min_recall=float(df.episode_recall.min()),
        mean_low_recall=float(df.low_recall.mean()),
        mean_high_recall=float(df.high_recall.mean()),
        worst_false_windows_day=float(df.false_windows_day.max()),
        worst_false_notifications_day=float(df.false_notifications_day.max()))
    summary["budget_ok"]=bool(summary["worst_false_windows_day"]<=RAW_DAY and
                              summary["worst_false_notifications_day"]<=NOTICE_DAY)
    byrole=ed.groupby("role").agg(
        episodes=("episode","count"),recall=("hit","mean"),
        mean_min_p=("min_p","mean")).reset_index()

    df.to_csv(out/"worlds.csv",index=False)
    ed.to_csv(out/"episodes.csv",index=False)
    byrole.to_csv(out/"by_role.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("WORLDS\n",df.to_string(index=False),flush=True)
    print("BY_ROLE\n",byrole.to_string(index=False),flush=True)
    print("SUMMARY",json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.model,a.output,a.seeds)
