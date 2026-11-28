"""RANK-only commissioning CUSUM audit for NAIVE SS7 compromise.

Each world builds a node-local benign baseline from the predeclared
commissioning interval. Cross-world thresholds are calibrated on one RANK
world and evaluated on the other, then reversed. CAL and TEST are never read.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from ss7 import fix as F
from ss7 import cu_compact_channel_development as C
from ss7 import naive_role_union_rank_audit as U
from ss7.incident_calibration_study import threshold, bucket_peaks

BASELINE_END=int(os.getenv("SS7_DEPLOY_BASELINE","180"))
RAW_DAY=4.0
NOTICE_DAY=1.0
VARIANTS=("composition_max","composition_top2","composition_l2")
QUANTILES=(.80,.90,.95)
HORIZONS=(10,20,40)

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,1/F.INTERVALS_PER_DAY)

def fused_base(d,variant):
    out=np.zeros(len(d),float)
    roles=d.ntype.astype(str).to_numpy()
    for role in U.ROLE_MSG:
        mask=roles==role
        if mask.any():
            q=U.raw_score(d,role,variant)
            out[mask]=q[mask]
    return out

def local_margin(d,base,q):
    out=np.zeros(len(d),float)
    tt=d.t.to_numpy()
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        warm=ix[tt[ix]<BASELINE_END]
        if not len(warm):
            continue
        anchor=float(np.quantile(base[warm],q))
        out[ix]=base[ix]-anchor
    out[tt<BASELINE_END]=0.0
    return out

def finite_cusum(d,margin,horizon):
    """Maximum positive suffix sum within the last horizon causal windows."""
    out=np.zeros(len(d),float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        v=margin[ix]
        t=g.t.to_numpy()
        # Split any time gaps, though generated worlds normally have none.
        starts=np.r_[0,np.flatnonzero(np.diff(t)!=1)+1]
        ends=np.r_[starts[1:],len(ix)]
        for a,b in zip(starts,ends):
            x=v[a:b]
            pref=np.r_[0.0,np.cumsum(x)]
            for j in range(len(x)):
                lo=max(0,j-horizon+1)
                out[ix[a+j]]=max(0.0,pref[j+1]-np.min(pref[lo:j+1]))
    return out/np.sqrt(float(horizon))

def score(d,variant,q,horizon):
    return finite_cusum(d,local_margin(d,fused_base(d,variant),q),horizon)

def false_bucket_peaks(d,s):
    valid=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)
    z=np.where(valid,s,-np.inf)
    p=bucket_peaks(d,z)
    return p[np.isfinite(p)]

def calibrate(d,s):
    days=monitor_days(d)
    valid=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)
    raw=threshold(s[valid],RAW_DAY,days)
    peaks=false_bucket_peaks(d,s)
    notice=threshold(peaks,NOTICE_DAY,days) if len(peaks) else -np.inf
    return float(max(raw,notice))

def evaluate(d,eligible,s,thr):
    days=monitor_days(d)
    valid=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)
    false_raw=float((s[valid]>thr).sum()/days)
    peaks=false_bucket_peaks(d,s)
    false_notice=float((peaks>thr).sum()/days) if len(peaks) else 0.0
    eps=[(i,e) for i,e in eligible.items()
         if e.kind=="compromise" and e.sophistication.value=="naive"]
    detected=0;by_role={};by_rate={}
    t=d.t.to_numpy();nid=d.nid.to_numpy()
    for i,e in eps:
        hit=bool(np.any((nid==e.nid)&(t>=e.t0)&(t<=e.t1)&(s>thr)))
        detected+=int(hit)
        r=C.F.NodeType(C.F.NodeType(e.sophistication).value).value if False else None
        role=str(d.loc[(d.episode_id==i),"ntype"].iloc[0])
        by_role.setdefault(role,[0,0]);by_role[role][0]+=int(hit);by_role[role][1]+=1
        key=str(e.hidden_rate);by_rate.setdefault(key,[0,0]);by_rate[key][0]+=int(hit);by_rate[key][1]+=1
    return dict(episodes=len(eps),detected=detected,recall=detected/max(len(eps),1),
                false_windows_day=false_raw,false_notifications_day=false_notice,
                by_role=by_role,by_rate=by_rate)

def main(root,out,seeds):
    seeds=tuple(int(x) for x in seeds)
    if len(seeds)<2:raise ValueError("need at least two RANK worlds")
    worlds={s:C.load(root,"compact",s) for s in seeds}
    rows=[]
    for variant in VARIANTS:
        for q in QUANTILES:
            prepared={}
            for seed,(d,eligible,_) in worlds.items():
                base=fused_base(d,variant)
                margin=local_margin(d,base,q)
                for h in HORIZONS:
                    prepared[(seed,h)]=finite_cusum(d,margin,h)
            for h in HORIZONS:
                for b in seeds:
                    cals=[a for a in seeds if a!=b]
                    thresholds=[calibrate(worlds[a][0],prepared[(a,h)]) for a in cals]
                    thr=float(max(thresholds))
                    db,eb,_=worlds[b]
                    m=evaluate(db,eb,prepared[(b,h)],thr)
                    rows.append(dict(variant=variant,q=q,horizon=h,
                        cal_seed=";".join(map(str,cals)),eval_seed=b,
                        threshold=thr,episodes=m["episodes"],detected=m["detected"],recall=m["recall"],
                        false_windows_day=m["false_windows_day"],
                        false_notifications_day=m["false_notifications_day"],
                        by_role=json.dumps(m["by_role"],sort_keys=True),
                        by_rate=json.dumps(m["by_rate"],sort_keys=True)))
    df=pd.DataFrame(rows);df.to_csv(out/"cross_world.csv",index=False)
    agg=df.groupby(["variant","q","horizon"]).agg(
        mean_recall=("recall","mean"),min_recall=("recall","min"),
        worst_raw=("false_windows_day","max"),
        worst_notice=("false_notifications_day","max")).reset_index()
    ok=agg[(agg.worst_raw<=RAW_DAY)&(agg.worst_notice<=NOTICE_DAY)].copy()
    cand=ok if len(ok) else agg
    cand=cand.sort_values(["min_recall","mean_recall","worst_notice","worst_raw"],
                          ascending=[False,False,True,True])
    winner=cand.iloc[0].to_dict()
    agg.to_csv(out/"summary.csv",index=False)
    (out/"winner.json").write_text(json.dumps(winner,indent=2))
    print("SUMMARY\n",agg.sort_values(["min_recall","mean_recall"],ascending=False).to_string(index=False),flush=True)
    print("WINNER",json.dumps(winner,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,tuple(a.seeds))
