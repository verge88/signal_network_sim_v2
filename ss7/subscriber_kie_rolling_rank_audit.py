"""RANK-only rolling subscriber-KIE audit (v6b).

Uses only keyed subscriber-hash bin counts already transported by compact KIE.
Raw subscriber identifiers never leave the monitored node. Candidate rules are
selected by leave-one-world-out RANK validation; TEST is never loaded here.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from ss7 import fix as F
from ss7 import cu_compact_channel_development as C
from ss7.incident_calibration_study import threshold

BASELINE_END=int(os.getenv("SS7_DEPLOY_BASELINE","180"))
RAW_DAY=4.0
WINDOWS=(10,20,40,60)  # 5, 10, 20, 30 min at 30 s intervals
VARIANTS=("max_delta","js","max_z")
MAP_GROUPS=(1,2,3)
EPS=1e-12

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def _bins(d,g):
    cols=[f"kie_sub_g{g}_b{b}" for b in range(C.SUB_HASH_BINS)]
    miss=[c for c in cols if c not in d.columns]
    if miss: raise KeyError(miss[:3])
    return d[cols].fillna(0.0).to_numpy(float)

def _baseline_profile(gdf,X):
    warm=(gdf.t.to_numpy()<BASELINE_END)
    q=X[warm].sum(axis=0)
    if q.sum()<=0:
        return np.full(X.shape[1],1/X.shape[1],float)
    q=np.maximum(q,0.0)
    return q/q.sum()

def node_group_score(gdf,X,window,variant):
    q=_baseline_profile(gdf,X)
    roll=pd.DataFrame(X,index=gdf.index).rolling(
        window,min_periods=max(4,window//3)).sum().to_numpy(float)
    tot=np.nansum(roll,axis=1)
    P=np.divide(roll,tot[:,None],out=np.zeros_like(roll),
                where=tot[:,None]>0)

    if variant=="max_delta":
        score=np.max(P-q[None,:],axis=1)
    elif variant=="js":
        M=.5*(P+q[None,:])
        klp=np.sum(np.where(P>0,P*np.log((P+EPS)/(M+EPS)),0.0),axis=1)
        klq=np.sum(q[None,:]*np.log((q[None,:]+EPS)/(M+EPS)),axis=1)
        score=.5*(klp+klq)
    elif variant=="max_z":
        # Multinomial sampling scale plus a 2%-share overdispersion floor.
        var=q*(1-q)/np.maximum(tot[:,None],1.0)+0.02**2
        score=np.max((P-q[None,:])/np.sqrt(var),axis=1)
    else:
        raise ValueError(variant)
    score=np.where((tot>0)&np.isfinite(score),score,0.0)
    return np.maximum(score,0.0)

def raw_score(d,window,variant):
    out=np.zeros(len(d),float)
    # Node-wise baseline is deployment-realistic and prevents topology/world
    # volume differences from becoming attack evidence.
    for _,gdf in d.groupby("nid",sort=False):
        ix=gdf.index.to_numpy()
        parts=[]
        for sg in MAP_GROUPS:
            parts.append(node_group_score(gdf,_bins(gdf,sg),window,variant))
        out[ix]=np.maximum.reduce(parts)
    return out

def role_cut(items,role):
    cuts=[]
    for d,s in items:
        mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)&(d.t.to_numpy()>=BASELINE_END)
        cuts.append(threshold(s[mask],RAW_DAY,monitor_days(d)))
    return float(max(cuts))

def evaluate(d,s,role,cut):
    r=(d.ntype.astype(str).to_numpy()==role)
    y=((d.attack=="compromise")&(d.sophistication=="naive")&r).to_numpy(int)
    valid=r&((d.label.to_numpy()==0)|y.astype(bool))&(d.t.to_numpy()>=BASELINE_END)
    yy=y[valid];ss=s[valid]
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc=float(roc_auc_score(yy,ss)) if yy.sum() and (~yy.astype(bool)).sum() else np.nan
    ap=float(average_precision_score(yy,ss)) if yy.sum() else np.nan

    neg=(d.label.to_numpy()==0)&r&(d.t.to_numpy()>=BASELINE_END)
    false_day=float((s[neg]>cut).sum()/monitor_days(d))
    z3=C.streak_score(d,s,3)

    eps=[]
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")&
        (d.ntype.astype(str)==role)]
    for _,g in q.groupby("episode_id",sort=False):
        eps.append(g.index.to_numpy())
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
        for window in WINDOWS:
            scores={seed:raw_score(d,window,variant) for seed,d in worlds.items()}
            for held in seeds:
                cals=[x for x in seeds if x!=held]
                for role in roles:
                    cut=role_cut([(worlds[x],scores[x]) for x in cals],role)
                    m=evaluate(worlds[held],scores[held],role,cut)
                    rows.append(dict(variant=variant,window=window,role=role,
                        cal_seed=";".join(map(str,cals)),eval_seed=held,
                        threshold=cut,**m))
    df=pd.DataFrame(rows);df.to_csv(out/"cross_world.csv",index=False)
    agg=df.groupby(["variant","window"]).agg(
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
                           "min_episode_recall_1","mean_episode_recall_1",
                           "min_ap","mean_ap"],
                          ascending=[False,False,False,False,False,False])
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
