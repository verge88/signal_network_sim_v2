"""Development-only observability audit for unmasked (NAIVE) SS7 compromise.

Uses CAL worlds only. Scores are causal functions of observable telemetry and do
not use labels in their construction. Labels are used only to measure ranking
quality and effect size. Do not select a final detector on TEST from this file.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from ss7 import fix as F
from ss7.cu_compact_channel_development import (
    MAP, _causal_robust_z, _profile_js_rows, naive_self_score
)

CAL=(21201,21202)
ROLL=(10,30,60)

def rolling_mean(d, x, win):
    x=np.asarray(x,float);out=np.full(len(d),np.nan)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        out[ix]=pd.Series(x[ix],index=ix).rolling(win,min_periods=max(3,win//3)).mean().to_numpy()
    return np.nan_to_num(out,nan=0.0,posinf=0.0,neginf=0.0)

def candidates(d):
    out={}
    raw=["local_"+c for c in F.OWN if "local_"+c in d.columns]
    raw += [c for c in d.columns if c.startswith("local_obs_ratio_")]
    for c in raw:
        x=d[c].to_numpy(float)
        z=_causal_robust_z(d,x)
        out["posz:"+c]=np.maximum(z,0.0)
        out["absz:"+c]=np.abs(z)
        for w in ROLL:
            out[f"mean{w}:posz:{c}"]=rolling_mean(d,np.maximum(z,0.0),w)
            out[f"mean{w}:absz:{c}"]=rolling_mean(d,np.abs(z),w)
    js=_profile_js_rows(d)
    selfscore=naive_self_score(d)
    out["profile_js"]=js
    out["selfnorm_legacy"]=selfscore
    for w in ROLL:
        out[f"mean{w}:profile_js"]=rolling_mean(d,js,w)
        out[f"mean{w}:selfnorm_legacy"]=rolling_mean(d,selfscore,w)
    # Existing causal history/zone/CU features are already observable.
    for c in d.columns:
        if c.startswith("hist_"):
            x=np.nan_to_num(d[c].to_numpy(float),nan=0.0,posinf=0.0,neginf=0.0)
            out["abs:"+c]=np.abs(x)
        elif c.startswith("zone_dev_") or c in ("cu_staleness","cu_response_time_jitter","cu_consistency_error"):
            x=np.nan_to_num(d[c].to_numpy(float),nan=0.0,posinf=0.0,neginf=0.0)
            out["abs:"+c]=np.abs(x)
    return out

def evaluate(seed, d):
    naive=(d.attack.to_numpy()=="compromise")&(d.sophistication.astype(str).to_numpy()=="naive")
    neg=d.label.to_numpy()==0
    use=naive|neg
    y=naive[use].astype(int)
    if y.sum()==0 or (~y.astype(bool)).sum()==0:raise ValueError("missing classes")
    rows=[]
    C=candidates(d)
    roles=d.ntype.astype(str).to_numpy()
    for name,s in C.items():
        full=np.asarray(s,float)
        z=full[use]
        if not np.isfinite(z).all():continue
        auc=float(roc_auc_score(y,z));ap=float(average_precision_score(y,z))
        rows.append(dict(seed=seed,role="ALL",score=name,auc=auc,ap=ap,
                         positive=int(y.sum()),negative=int((y==0).sum())))
        for role in sorted(np.unique(roles)):
            ru=use&(roles==role)
            yy=naive[ru].astype(int)
            zz=full[ru]
            if len(np.unique(yy))<2 or not np.isfinite(zz).all():continue
            rows.append(dict(seed=seed,role=role,score=name,
                auc=float(roc_auc_score(yy,zz)),ap=float(average_precision_score(yy,zz)),
                positive=int(yy.sum()),negative=int((yy==0).sum())))
    # Effect sizes for the most interpretable raw observables.
    effects=[]
    for c in ["local_total_messages","local_map_share","local_isup_share","local_mtype_entropy",
              "local_dest_entropy","local_n_unique_dest","local_dest_top1_ratio","local_in_out_ratio"]:
        if c not in d:continue
        x=d[c].to_numpy(float)
        for role in sorted(d.ntype.astype(str).unique()):
            nmask=neg&(d.ntype.astype(str).to_numpy()==role)
            pmask=naive&(d.ntype.astype(str).to_numpy()==role)
            if nmask.sum()<20 or pmask.sum()<5:continue
            mu0=np.nanmean(x[nmask]);sd0=np.nanstd(x[nmask])
            mu1=np.nanmean(x[pmask])
            effects.append(dict(seed=seed,role=role,feature=c,normal_mean=mu0,naive_mean=mu1,
                                effect=(mu1-mu0)/(sd0+1e-9),normal_n=int(nmask.sum()),naive_n=int(pmask.sum())))
    # Rate-stratified naive summaries for a small fixed score family.
    rates=[]
    fixed={k:v for k,v in C.items() if k in (
        "posz:local_total_messages","absz:local_total_messages",
        "mean30:posz:local_total_messages","profile_js","mean30:profile_js","selfnorm_legacy")}
    for rate in sorted(d.loc[naive,"rate"].unique()) if "rate" in d.columns else []:
        pass
    return pd.DataFrame(rows),pd.DataFrame(effects)

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tabs=[];eff=[]
    for seed in CAL:
        d,e,cu=joblib.load(a.root/f"world-compact-{seed}"/"world.joblib")
        r,x=evaluate(seed,d);tabs.append(r);eff.append(x)
    scores=pd.concat(tabs,ignore_index=True)
    effects=pd.concat(eff,ignore_index=True)
    summary=scores.groupby(["role","score"]).agg(auc_mean=("auc","mean"),auc_min=("auc","min"),
        ap_mean=("ap","mean"),ap_min=("ap","min"),worlds=("seed","nunique")).reset_index()
    summary=summary.sort_values(["role","auc_min","ap_mean"],ascending=[True,False,False])
    scores.to_csv(a.output/"scores.csv",index=False)
    summary.to_csv(a.output/"summary.csv",index=False)
    effects.to_csv(a.output/"effects.csv",index=False)
    (a.output/"protocol.json").write_text(json.dumps(dict(cal=CAL,roll=ROLL,
        scope="development CAL only; labels used only for diagnostics; TEST untouched by this audit"),indent=2))
    for role in ("ALL","IGW","SCP","SP","STP"):
        q=summary[summary.role==role].head(15)
        if len(q):print("TOP CAUSAL SCORES",role,"\n",q.to_string(index=False),flush=True)
    print("EFFECTS\n",effects.to_string(index=False),flush=True)

if __name__=="__main__":main()
