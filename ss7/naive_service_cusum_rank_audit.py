"""RANK-only signed CUSUM on observable SS7 MAP transaction templates.

Each channel approximates a legitimate MAP transaction from operation counts:
SMS_MT, SMS_MO, LOCATION, ROAM_SRI, AUTH, MOBILITY. A signed CUSUM is kept
per channel; negative innovations reset evidence before channels are fused.
The method is fixed a priori and reads no CAL/TEST data.
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
OPS=("SRI_SM","MT_FSM","MO_FSM","SRI","PRN","UL","ISD","ATI","PSI","SAI","CL","PURGE")

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def operation_counts(d):
    total=d.total_messages.to_numpy(float)
    map_total=np.maximum(total*d.map_share.to_numpy(float),0.0)
    delivered=d.cu_delivered.to_numpy(float)>0
    out={}
    for m in OPS:
        r=d[f"obs_ratio_{m}"].to_numpy(float)
        x=map_total*r
        x[~(delivered&np.isfinite(x)&np.isfinite(r))]=np.nan
        out[m]=x
    return out

def _min_valid(*xs):
    a=np.column_stack(xs)
    valid=np.isfinite(a).all(axis=1)
    out=np.full(len(a),np.nan)
    out[valid]=np.min(a[valid],axis=1)
    return out

def service_counts(d):
    o=operation_counts(d)
    return {
        "sms_mt":_min_valid(o["SRI_SM"],o["MT_FSM"]),
        "sms_mo":o["MO_FSM"].copy(),
        "location":_min_valid(o["ATI"],o["PSI"]),
        "roam_sri":_min_valid(o["SRI"],o["PRN"]),
        "auth":o["SAI"].copy(),
        "mobility":_min_valid(o["UL"],o["ISD"]/2.0,o["CL"]),
    }

def innovation(d,x):
    out=np.full(len(d),np.nan)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy();a=x[ix]
        s=pd.Series(a,index=ix)
        recent=s.rolling(RECENT,min_periods=3).mean()
        past=s.shift(GAP)
        mu=past.rolling(BASE,min_periods=30).mean()
        sd=past.rolling(BASE,min_periods=30).std()
        scale=np.sqrt(np.maximum(mu,1.0)/RECENT + np.maximum(sd,0.0)**2/RECENT)
        out[ix]=np.clip(((recent-mu)/(scale+1e-6)).to_numpy(float),-6.0,6.0)
    return out

def channel_cusum(d,z):
    out=np.zeros(len(d),float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy();c=0.0
        for j in ix:
            v=z[j]
            if not np.isfinite(v):
                c=max(0.0,LEAK*c-.5)
            else:
                # Signed innovation: negative evidence resets the accumulator.
                c=max(0.0,LEAK*c+v-K)
            out[j]=c
    return out

def score(d):
    ch=[]
    for _,x in service_counts(d).items():
        ch.append(channel_cusum(d,innovation(d,x)))
    return np.arcsinh(np.max(np.column_stack(ch),axis=1))

def world_cut(d,s):
    neg=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)
    return float(threshold(s[neg],RAW_DAY,monitor_days(d)))

def evaluate(d,s,cut):
    tt=d.t.to_numpy();normal=(d.label.to_numpy()==0)&(tt>=BASELINE_END)
    y=((d.attack=="compromise")&(d.sophistication=="naive")).to_numpy(int)
    valid=((d.label.to_numpy()==0)|y.astype(bool))&(tt>=BASELINE_END)
    yy=y[valid];ss=s[valid];z3=C.streak_score(d,s,3)
    eps=[]
    for eid,g in d[(d.attack=="compromise")&(d.sophistication=="naive")].groupby("episode_id",sort=False):
        ix=g.index.to_numpy()
        eps.append(dict(episode=int(eid),role=str(g.ntype.iloc[0]),
            detected_1=int(np.any(s[ix]>cut)),detected_3=int(np.any(z3[ix]>cut)),
            peak=float(np.max(s[ix]))))
    ep=pd.DataFrame(eps)
    return ep,dict(auc=float(roc_auc_score(yy,ss)),ap=float(average_precision_score(yy,ss)),
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
