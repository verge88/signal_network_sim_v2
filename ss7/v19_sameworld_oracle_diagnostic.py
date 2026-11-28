"""Diagnostic upper bound for the frozen v19 score.

Uses true normal labels ONLY to answer whether any same-world threshold meeting
the operational budgets can detect NAIVE compromise. This is not a deployable
detector and never touches CAL/TEST.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from ss7 import fix as F
from ss7 import cu_compact_channel_development as C
from ss7 import v19_block_logit_rank as V19
from ss7.incident_calibration_study import threshold, notify

BASELINE_END=int(os.getenv("SS7_DEPLOY_BASELINE","180"))
BUDGETS=(4.0,20.0)
NOTICE_DAY=2.0
CONFIRMS=(1,3)

def days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,1/F.INTERVALS_PER_DAY)

def cut_for(d,s,raw):
    normal=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)
    rawcut=float(threshold(s[normal],raw,days(d)))
    return rawcut

def eval_one(d,s,cut,confirm):
    normal=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)
    z=C.streak_score(d,s,confirm) if confirm>1 else s
    notices=notify(d,z,cut)
    false_notices=sum(bool(normal[j]) for j in notices)/days(d)
    false_rows=float(((z>cut)&normal).sum()/days(d))
    rows=[]
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy()
        rows.append(dict(episode=int(eid),role=str(g.ntype.iloc[0]),
                         hit=int(np.any(z[ix]>cut)),
                         peak=float(np.max(z[ix]))))
    ep=pd.DataFrame(rows)
    return dict(recall=float(ep.hit.mean()),false_rows_day=false_rows,
                false_notifications_day=float(false_notices),
                min_attack_peak=float(ep.peak.min()),max_attack_peak=float(ep.peak.max()))

def main(root,model_path,out,seeds):
    model=joblib.load(model_path)
    rows=[]
    for seed in map(int,seeds):
        d,_,_=C.load(root,"compact",seed)
        s=V19.score(d,model)
        for raw in BUDGETS:
            cut=cut_for(d,s,raw)
            for confirm in CONFIRMS:
                m=eval_one(d,s,cut,confirm)
                rows.append(dict(seed=seed,raw_budget=raw,confirm=confirm,threshold=cut,**m))
    df=pd.DataFrame(rows)
    df["budget_ok"]=(df.false_rows_day<=df.raw_budget)&(df.false_notifications_day<=NOTICE_DAY)
    df.to_csv(out/"oracle.csv",index=False)
    summary=df.groupby(["raw_budget","confirm"]).agg(
        mean_recall=("recall","mean"),min_recall=("recall","min"),
        worst_false_rows_day=("false_rows_day","max"),
        worst_false_notifications_day=("false_notifications_day","max"),
        min_threshold=("threshold","min"),max_threshold=("threshold","max"),
        max_attack_peak=("max_attack_peak","max")).reset_index()
    summary.to_csv(out/"summary.csv",index=False)
    print("ORACLE\n",df.to_string(index=False),flush=True)
    print("SUMMARY\n",summary.to_string(index=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.model,a.output,a.seeds)
