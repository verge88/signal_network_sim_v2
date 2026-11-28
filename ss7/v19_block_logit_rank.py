"""Low-capacity block-thinned NAIVE-compromise detector for KIE v19.

The detector is deliberately simple: one strongly L2-regularized logistic
regression over eight causal identity-summary features plus observable node-role
indicators. Training uses only every tenth 30-s interval (5-minute cadence),
with equal total weight per positive episode and per normal node/world. This
prevents long autocorrelated traces from dominating the fit.

TRAIN worlds fit the model. RANK worlds are never used for fitting; they are
used only for leave-one-world-out threshold transfer and evaluation. CAL/TEST
are not loaded.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ss7 import fix as F
from ss7 import cu_compact_channel_development as C
from ss7.incident_calibration_study import threshold, notify

BASELINE_END=int(os.getenv("SS7_DEPLOY_BASELINE","180"))
RAW_DAY=20.0
NOTICE_DAY=2.0
TRAIN_CADENCE=10       # 5 min, intentionally non-overlapping training cadence
CONFIRM=3
C_REG=.05
FEATURES=tuple(C.SUB_MULTIVIEW_NAMES)
ROLES=tuple(x.value for x in F.NodeType)
MODEL_COLS=FEATURES+tuple("role_"+r for r in ROLES)

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def design(d):
    X=pd.DataFrame(index=d.index)
    for c in FEATURES:X[c]=pd.to_numeric(d[c],errors="coerce")
    rr=d.ntype.astype(str)
    for r in ROLES:X["role_"+r]=(rr==r).astype(float)
    return X[list(MODEL_COLS)]

def target(d):
    return ((d.attack=="compromise")&(d.sophistication=="naive")).to_numpy(int)

def train_rows(d):
    y=target(d)
    normal=(d.label.to_numpy()==0)
    usable=normal|y.astype(bool)
    cadence=(d.t.to_numpy()>=BASELINE_END)&(d.t.to_numpy()%TRAIN_CADENCE==0)
    return usable&cadence

def balanced_weights(d,mask):
    """Equal class/world mass; positive episodes and normal nodes share mass."""
    idx=np.flatnonzero(mask);q=d.iloc[idx].copy();y=target(q)
    w=np.zeros(len(q),float)

    # Every TRAIN world contributes one unit of positive and one unit of normal
    # evidence regardless of topology size or episode duration.
    for seed,gseed in q.groupby("world_seed",sort=False):
        loc=gseed.index.to_numpy()
        # Convert original q index to positional indices.
        posmap={v:i for i,v in enumerate(q.index.to_numpy())}
        pp=np.array([posmap[v] for v in loc],dtype=int)
        yp=y[pp]

        pos=pp[yp==1]
        if len(pos):
            eids=q.iloc[pos].episode_id.to_numpy()
            ue=np.unique(eids)
            for eid in ue:
                z=pos[eids==eid]
                w[z]=1.0/(len(ue)*len(z))

        neg=pp[yp==0]
        if len(neg):
            nodes=q.iloc[neg].nid.to_numpy()
            un=np.unique(nodes)
            for nid in un:
                z=neg[nodes==nid]
                w[z]=1.0/(len(un)*len(z))

    # Equal total class mass even if a world happened to contribute no sample.
    ps=w[y==1].sum();ns=w[y==0].sum()
    if ps>0:w[y==1]*=.5/ps
    if ns>0:w[y==0]*=.5/ns
    return idx,w

def fit(train_worlds):
    frames=[]
    for seed,d in train_worlds.items():
        q=d.copy();q["world_seed"]=seed;frames.append(q)
    all_d=pd.concat(frames,ignore_index=True)
    mask=train_rows(all_d);idx,w=balanced_weights(all_d,mask)
    q=all_d.iloc[idx];y=target(q)

    model=make_pipeline(
        SimpleImputer(strategy="median",keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=C_REG,penalty="l2",solver="lbfgs",
                           max_iter=1000,random_state=42))
    model.fit(design(q),y,logisticregression__sample_weight=w)
    return model,all_d,q,y,w

def score(d,model):
    return model.predict_proba(design(d))[:,1]

def world_cut(d,s):
    m=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)&np.isfinite(s)
    return float(threshold(s[m],RAW_DAY,monitor_days(d)))

def row_metrics(d,s):
    y=target(d)
    valid=((d.label.to_numpy()==0)|y.astype(bool))&(d.t.to_numpy()>=BASELINE_END)
    yy=y[valid];ss=s[valid]
    return dict(
        auc=float(roc_auc_score(yy,ss)) if yy.sum() and (~yy.astype(bool)).sum() else np.nan,
        ap=float(average_precision_score(yy,ss)) if yy.sum() else np.nan)

def evaluate(d,s,cut,eligible):
    base=row_metrics(d,s)
    tt=d.t.to_numpy();normal=(d.label.to_numpy()==0)&(tt>=BASELINE_END)
    raw_fire=s>cut
    z=C.streak_score(d,s,CONFIRM)
    nidx=notify(d,z,cut)
    labels=d.label.to_numpy()
    fnot=sum(bool((labels[max(0,j-2):j+1]==0).all()) for j in nidx)/monitor_days(d)

    eps=[]
    q=d[(d.attack=="compromise")&(d.sophistication=="naive")]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy();e=eligible.get(int(eid))
        fire=z[ix]>cut;first=np.flatnonzero(fire)
        eps.append(dict(
            episode=int(eid),role=str(g.ntype.iloc[0]),
            hidden_rate=float(e.hidden_rate) if e is not None else np.nan,
            hit=int(np.any(fire)),
            delay_seconds=float(first[0]*30) if len(first) else np.nan,
            peak=float(np.max(z[ix]))))
    ep=pd.DataFrame(eps);low=ep.hidden_rate<.5
    base.update(
        episodes=len(ep),episode_recall=float(ep.hit.mean()),
        low_recall=float(ep.loc[low,"hit"].mean()) if low.any() else np.nan,
        high_recall=float(ep.loc[~low,"hit"].mean()) if (~low).any() else np.nan,
        false_windows_day=float(raw_fire[normal].sum()/monitor_days(d)),
        false_notifications_day=float(fnot))
    return ep,base

def main(root,out,train_seeds,rank_seeds):
    train_seeds=tuple(map(int,train_seeds));rank_seeds=tuple(map(int,rank_seeds))
    train={s:C.load(root,"compact",s)[0] for s in train_seeds}
    rank_loaded={s:C.load(root,"compact",s) for s in rank_seeds}
    rank={s:x[0] for s,x in rank_loaded.items()}
    eligible={s:x[1] for s,x in rank_loaded.items()}

    model,all_train,q,y,w=fit(train)
    joblib.dump(model,out/"model.joblib")

    # Diagnostic only: quantify train->RANK transfer before thresholding.
    diagnostics=[]
    for split,worlds in (("train",train),("rank",rank)):
        for seed,d in worlds.items():
            s=score(d,model);m=row_metrics(d,s)
            diagnostics.append(dict(split=split,seed=seed,**m))
    diag=pd.DataFrame(diagnostics);diag.to_csv(out/"generalization.csv",index=False)

    rs={seed:score(d,model) for seed,d in rank.items()}
    rows=[];episodes=[]
    for held in rank_seeds:
        cals=[x for x in rank_seeds if x!=held]
        cut=float(max(world_cut(rank[x],rs[x]) for x in cals))
        ep,m=evaluate(rank[held],rs[held],cut,eligible[held])
        m.update(eval_seed=held,cal_seed=";".join(map(str,cals)),threshold=cut,
                 confirm=CONFIRM)
        rows.append(m);ep["eval_seed"]=held;ep["threshold"]=cut;episodes.append(ep)
    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    summary=dict(
        mean_auc=float(df.auc.mean()),min_auc=float(df.auc.min()),
        mean_ap=float(df.ap.mean()),min_ap=float(df.ap.min()),
        mean_episode_recall=float(df.episode_recall.mean()),
        min_episode_recall=float(df.episode_recall.min()),
        mean_low_recall=float(df.low_recall.mean()),
        mean_high_recall=float(df.high_recall.mean()),
        worst_false_windows_day=float(df.false_windows_day.max()),
        worst_false_notifications_day=float(df.false_notifications_day.max()))
    summary["budget_ok"]=bool(summary["worst_false_windows_day"]<=RAW_DAY and
                              summary["worst_false_notifications_day"]<=NOTICE_DAY)

    # Standardized coefficients make the low-capacity model auditable.
    lr=model.named_steps["logisticregression"]
    coef=pd.DataFrame({"feature":MODEL_COLS,"coefficient":lr.coef_[0]})
    coef["abs_coefficient"]=coef.coefficient.abs()
    coef=coef.sort_values("abs_coefficient",ascending=False)
    byrole=ed.groupby("role").agg(episodes=("episode","count"),
        recall=("hit","mean"),mean_peak=("peak","mean")).reset_index()

    df.to_csv(out/"cross_world.csv",index=False);ed.to_csv(out/"episodes.csv",index=False)
    coef.to_csv(out/"coefficients.csv",index=False);byrole.to_csv(out/"by_role.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("TRAIN_ROWS",len(q),"POS",int(y.sum()),"WEIGHT_POS",float(w[y==1].sum()),
          "WEIGHT_NEG",float(w[y==0].sum()),flush=True)
    print("GENERALIZATION\n",diag.to_string(index=False),flush=True)
    print("CROSS_WORLD\n",df.to_string(index=False),flush=True)
    print("BY_ROLE\n",byrole.to_string(index=False),flush=True)
    print("COEFFICIENTS\n",coef.to_string(index=False),flush=True)
    print("SUMMARY",json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--train-seeds",type=int,nargs="+",required=True)
    ap.add_argument("--rank-seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,a.train_seeds,a.rank_seeds)
