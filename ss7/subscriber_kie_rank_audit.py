"""RANK-only observability audit for privacy-preserving subscriber KIE v6.

No subscriber identifiers are exported. The simulator exposes only keyed hash
bin summaries already carried by compact KIE. Each held-out RANK world is
evaluated with thresholds calibrated on the other RANK worlds.
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
VARIANTS=("top1","entropy","occupied","max","top1_entropy")
SMOOTH=(1,3,5,10)
MAP_GROUPS=(1,2,3)

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def commissioning_shift(d,col,direction):
    out=np.zeros(len(d),float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        v=g[col].to_numpy(float)
        warm=(g.t.to_numpy()<BASELINE_END)&np.isfinite(v)
        a=v[warm]
        if len(a)<20:continue
        center=float(np.median(a));q25,q75=np.quantile(a,[.25,.75])
        scale=max(float((q75-q25)/1.349),.02)
        z=direction*(v-center)/scale
        z=np.where(np.isfinite(v),z,0.0)
        out[ix]=np.arcsinh(np.maximum(z,0.0))
    return out

def raw_score(d,variant):
    parts={}
    for g in MAP_GROUPS:
        parts[(g,"top1")]=commissioning_shift(d,f"kie_sub_g{g}_top1",1.0)
        parts[(g,"entropy")]=commissioning_shift(d,f"kie_sub_g{g}_entropy",-1.0)
        parts[(g,"occupied")]=commissioning_shift(d,f"kie_sub_g{g}_occupied",-1.0)
    if variant in ("top1","entropy","occupied"):
        return np.maximum.reduce([parts[(g,variant)] for g in MAP_GROUPS])
    if variant=="top1_entropy":
        return np.maximum.reduce([parts[(g,k)] for g in MAP_GROUPS for k in ("top1","entropy")])
    if variant=="max":
        return np.maximum.reduce(list(parts.values()))
    raise ValueError(variant)

def smooth(d,s,w):
    if w==1:return np.asarray(s,float)
    out=np.zeros(len(d),float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        out[ix]=pd.Series(s[ix],index=ix).rolling(
            w,min_periods=max(1,w//2)).mean().fillna(0.0).to_numpy()
    return out

def role_cut(cal_items,role):
    cuts=[]
    for d,s in cal_items:
        mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)&(d.t.to_numpy()>=BASELINE_END)
        cuts.append(threshold(s[mask],RAW_DAY,monitor_days(d)))
    return float(max(cuts))

def evaluate(d,s,role,cut):
    role_mask=d.ntype.astype(str).to_numpy()==role
    y=((d.attack=="compromise")&(d.sophistication=="naive")&role_mask).to_numpy(int)
    valid=role_mask&((d.label.to_numpy()==0)|y.astype(bool))&(d.t.to_numpy()>=BASELINE_END)
    yy=y[valid];ss=s[valid]
    auc=float(roc_auc_score(yy,ss)) if yy.sum() and (~yy.astype(bool)).sum() else np.nan
    ap=float(average_precision_score(yy,ss)) if yy.sum() else np.nan
    neg=(d.label.to_numpy()==0)&role_mask&(d.t.to_numpy()>=BASELINE_END)
    false_day=float((s[neg]>cut).sum()/monitor_days(d))

    z3=C.streak_score(d,s,3)
    eps=[]
    for eid,g in d[(d.attack=="compromise")&(d.sophistication=="naive")&
                   (d.ntype.astype(str)==role)].groupby("episode_id",sort=False):
        ix=g.index.to_numpy()
        eps.append(ix)
    ep1=sum(bool(np.any(s[ix]>cut)) for ix in eps)
    ep3=sum(bool(np.any(z3[ix]>cut)) for ix in eps)
    return dict(auc=auc,ap=ap,episodes=len(eps),
                episode_recall_1=ep1/max(len(eps),1),
                episode_recall_3=ep3/max(len(eps),1),
                false_rows_day=false_day)

def main(root,out,seeds):
    seeds=tuple(int(x) for x in seeds)
    worlds={seed:C.load(root,"compact",seed)[0] for seed in seeds}
    roles=tuple(x.value for x in F.NodeType)
    rows=[]
    for variant in VARIANTS:
        base={seed:raw_score(d,variant) for seed,d in worlds.items()}
        for w in SMOOTH:
            score={seed:smooth(worlds[seed],base[seed],w) for seed in seeds}
            for held in seeds:
                cals=[x for x in seeds if x!=held]
                for role in roles:
                    cut=role_cut([(worlds[x],score[x]) for x in cals],role)
                    m=evaluate(worlds[held],score[held],role,cut)
                    rows.append(dict(variant=variant,smooth=w,role=role,
                        cal_seed=";".join(map(str,cals)),eval_seed=held,threshold=cut,**m))
    df=pd.DataFrame(rows);df.to_csv(out/"cross_world.csv",index=False)
    agg=df.groupby(["variant","smooth"]).agg(
        mean_auc=("auc","mean"),min_auc=("auc","min"),
        mean_ap=("ap","mean"),min_ap=("ap","min"),
        mean_episode_recall_1=("episode_recall_1","mean"),
        min_episode_recall_1=("episode_recall_1","min"),
        mean_episode_recall_3=("episode_recall_3","mean"),
        min_episode_recall_3=("episode_recall_3","min"),
        worst_false_rows_day=("false_rows_day","max")).reset_index()
    ok=agg[agg.worst_false_rows_day<=RAW_DAY].copy()
    cand=ok if len(ok) else agg
    cand=cand.sort_values(["min_episode_recall_3","mean_episode_recall_3",
                           "min_ap","mean_ap"],
                          ascending=[False,False,False,False])
    winner=cand.iloc[0].to_dict()
    agg.to_csv(out/"summary.csv",index=False)
    (out/"winner.json").write_text(json.dumps(winner,indent=2))
    print("SUMMARY\n",agg.sort_values(
        ["min_episode_recall_3","mean_episode_recall_3","min_ap"],
        ascending=False).to_string(index=False),flush=True)
    print("WINNER",json.dumps(winner,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,a.seeds)
