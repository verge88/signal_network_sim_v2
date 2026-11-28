"""RANK-only adaptive occupancy scan for v15 peer-service KIE.

Why: v15 has high row-level AUC but rare benign maxima exceed every attack peak,
so amplitude thresholding (including same-world oracle) yields zero episode
recall. This detector converts the v15 score into a causal within-node tail
indicator relative to recent history, then thresholds the fraction of recent
windows in that tail. Isolated benign extremes contribute at most one vote.

No CAL/TEST data, labels, or new simulator features are used by the detector.
RANK labels are used only after scoring to compare predeclared candidates.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from ss7 import fix as F
from ss7 import cu_compact_channel_development as C
from ss7.incident_calibration_study import threshold, notify, bucket_peaks

BASELINE_END=int(os.getenv("SS7_DEPLOY_BASELINE","180"))
RAW_DAY=20.0
NOTICE_DAY=2.0
FEATURE=C.SUB_PEER_SERVICE_ROLL30_NAME

# Small predeclared development grid. History is 60 minutes (120 x 30 s);
# a 2.5-minute guard prevents the current onset from immediately contaminating
# its own reference distribution.
TAIL_Q=(0.90,0.95,0.975)
SCAN=(10,20,40)  # 5, 10, 20 minutes
HISTORY=120
GAP=5
MIN_HISTORY=60

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def adaptive_occupancy(d,raw,q,window):
    raw=np.asarray(raw,float)
    out=np.full(len(d),np.nan,float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        x=pd.Series(raw[ix],index=ix,dtype=float)
        # Entirely causal: threshold at t is estimated from values ending at
        # t-GAP. No current/future score and no label enters the baseline.
        cut=x.shift(GAP).rolling(HISTORY,min_periods=MIN_HISTORY).quantile(q)
        tail=(x>cut).where(cut.notna())
        occ=tail.rolling(window,min_periods=max(3,window//2)).mean()
        out[ix]=occ.to_numpy(float)
    return out

def world_raw_cut(d,s):
    m=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)&np.isfinite(s)
    return float(threshold(s[m],RAW_DAY,monitor_days(d)))

def world_notice_cut(d,s):
    m=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)
    z=np.where(m,s,-np.inf)
    peaks=bucket_peaks(d,z)
    peaks=peaks[np.isfinite(peaks)]
    return float(threshold(peaks,NOTICE_DAY,monitor_days(d)))

def evaluate(d,s,cut,eligible):
    tt=d.t.to_numpy();finite=np.isfinite(s)
    y=((d.attack=="compromise")&(d.sophistication=="naive")).to_numpy(int)
    valid=((d.label.to_numpy()==0)|y.astype(bool))&(tt>=BASELINE_END)&finite
    yy=y[valid];ss=s[valid]
    auc=float(roc_auc_score(yy,ss)) if yy.sum() and (~yy.astype(bool)).sum() else np.nan
    ap=float(average_precision_score(yy,ss)) if yy.sum() else np.nan

    neg=(d.label.to_numpy()==0)&(tt>=BASELINE_END)&finite
    false_windows=float((s[neg]>cut).sum()/monitor_days(d))
    # notify() is label-blind and de-duplicates by node/time bucket.
    nidx=notify(d,np.nan_to_num(s,nan=-np.inf),cut)
    false_notif=sum(
        bool((d.label.to_numpy()[max(0,j-2):j+1]==0).all()) for j in nidx
    )/monitor_days(d)

    eps=[]
    qd=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    for eid,g in qd.groupby("episode_id",sort=False):
        ix=g.index.to_numpy();e=eligible.get(int(eid))
        hit=bool(np.any(s[ix]>cut))
        first=np.flatnonzero(s[ix]>cut)
        eps.append(dict(
            episode=int(eid),role=str(g.ntype.iloc[0]),
            hidden_rate=float(e.hidden_rate) if e is not None else np.nan,
            hit=int(hit),
            delay_seconds=float(first[0]*30) if len(first) else np.nan,
            peak=float(np.nanmax(s[ix])) if np.isfinite(s[ix]).any() else np.nan))
    ep=pd.DataFrame(eps);low=ep.hidden_rate<.5
    return ep,dict(
        auc=auc,ap=ap,episodes=len(ep),
        episode_recall=float(ep.hit.mean()),
        low_recall=float(ep.loc[low,"hit"].mean()) if low.any() else np.nan,
        high_recall=float(ep.loc[~low,"hit"].mean()) if (~low).any() else np.nan,
        false_windows_day=false_windows,
        false_notifications_day=float(false_notif))

def main(root,out,seeds):
    seeds=tuple(int(x) for x in seeds)
    loaded={seed:C.load(root,"compact",seed) for seed in seeds}
    worlds={seed:x[0] for seed,x in loaded.items()}
    eligible={seed:x[1] for seed,x in loaded.items()}
    raw={seed:worlds[seed][FEATURE].to_numpy(float) for seed in seeds}

    rows=[];episodes=[]
    for q in TAIL_Q:
        for window in SCAN:
            scores={seed:adaptive_occupancy(worlds[seed],raw[seed],q,window)
                    for seed in seeds}
            for held in seeds:
                cals=[x for x in seeds if x!=held]
                # Score amplitude and notification budgets are both learned
                # only on the other RANK worlds. Taking max is conservative.
                rawcut=max(world_raw_cut(worlds[x],scores[x]) for x in cals)
                noticecut=max(world_notice_cut(worlds[x],scores[x]) for x in cals)
                cut=float(max(rawcut,noticecut))
                ep,m=evaluate(worlds[held],scores[held],cut,eligible[held])
                m.update(q=q,window=window,eval_seed=held,
                         cal_seed=";".join(map(str,cals)),threshold=cut,
                         raw_cut=float(rawcut),notice_cut=float(noticecut))
                rows.append(m)
                ep["q"]=q;ep["window"]=window;ep["eval_seed"]=held;episodes.append(ep)

    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    agg=df.groupby(["q","window"]).agg(
        mean_auc=("auc","mean"),min_auc=("auc","min"),
        mean_ap=("ap","mean"),min_ap=("ap","min"),
        mean_recall=("episode_recall","mean"),min_recall=("episode_recall","min"),
        mean_low_recall=("low_recall","mean"),
        mean_high_recall=("high_recall","mean"),
        worst_false_windows_day=("false_windows_day","max"),
        worst_false_notifications_day=("false_notifications_day","max")
    ).reset_index()
    agg["budget_ok"]=(agg.worst_false_windows_day<=RAW_DAY)&(
        agg.worst_false_notifications_day<=NOTICE_DAY)
    ok=agg[agg.budget_ok].copy()
    cand=ok if len(ok) else agg
    cand=cand.sort_values(
        ["min_recall","mean_recall","min_ap","mean_ap",
         "worst_false_notifications_day","worst_false_windows_day"],
        ascending=[False,False,False,False,True,True])
    winner=cand.iloc[0].to_dict()
    q=float(winner["q"]);window=int(winner["window"])
    byrole=ed[(ed.q==q)&(ed.window==window)].groupby("role").agg(
        episodes=("episode","count"),recall=("hit","mean"),
        mean_peak=("peak","mean")).reset_index()

    df.to_csv(out/"cross_world.csv",index=False)
    agg.to_csv(out/"summary.csv",index=False)
    ed.to_csv(out/"episodes.csv",index=False)
    byrole.to_csv(out/"by_role.csv",index=False)
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
