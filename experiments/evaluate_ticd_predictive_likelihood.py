#!/usr/bin/env python3
"""Literature-guided predictive likelihood architecture; frozen comparators."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from evaluate_ticd_adaptive_loss_fusion import KINDS,CAL_NORMAL,NATIVE,WEAK,POLICY,group,select
from evaluate_ticd_full_count_evidence import pair_metrics
from evaluate_ticd_separate_robust_fusion import quick
from evaluate_ticd_observability_factorial import short
from ticd_measured_loss import labels
from ticd_sequential_physical import sequence_features
from ticd_correlated_block_evidence import block_features
from ticd_joint_channel_calibration import warning_bound
from ticd_predictive_likelihood import features,PredictiveFusion,SOURCES,TIMES,predictive_logprob,posterior,increments
from scipy.stats import binom

CAL=tuple(dict.fromkeys(CAL_NORMAL+("probe_delay_clean","observer_outage")))
SCALE={k:[480001+100*i+j for j in range(4)] for i,k in enumerate(CAL)}
VAL={k:[485001+100*i+j for j in range(6)] for i,k in enumerate(CAL)}
DEV={k:[490001+100*i+j for j in range(4)] for i,k in enumerate(KINDS)}
TEST={k:[500001+100*i+j for j in range(10)] for i,k in enumerate(KINDS)}
def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"joint_channel_models.pkl").open("rb") as f:pkg=pickle.load(f)
    refs=pkg["references"];base=pkg["models"]["proposed"];jr=pkg["joint_reference"]
    ref=refs["adaptive_fast"];cal=pkg["calibration"];causal=ref.core.causal
    seeds=[s for part in (SCALE,VAL,DEV,TEST) for ss in part.values() for s in ss]
    assert len(seeds)==len(set(seeds)) and min(seeds)>470001
    # Exact censoring likelihood must sum to 1 over observed ACK values.
    for m in (1,3,10,100):
        for n in (0,m//2,m):
            for p0 in (.001,.03,.2):
                for delta in (0.,.005,.04):
                    A=np.arange(n+1);M=np.full(n+1,m);N=np.full(n+1,n)
                    v=np.exp(predictive_logprob(M,N,A,np.full((n+1,1),p0),delta)).sum()
                    assert abs(v-1)<1e-10,(m,n,p0,delta,v)
    def prep(raw,c):
        x=ref.prepare(raw,c);x=sequence_features(x,c,ref);x=block_features(x,c,ref)
        x=features(jr.prepare(x));x["pred_audit"]=ref.audit.score(x)/ref.audit.threshold
        return x
    def clean(part):
        res=[]
        for k,ss in part.items():
            raw,_,_,c=group(ss,cal,k,causal);x=labels(prep(raw,c),raw);assert not x.is_anomaly.any()
            cols=list(dict.fromkeys(["simulation_seed","is_anomaly","primary_observer_valid","split_eligible","warn_0",
                *base.columns(),*[k for k in x if k.startswith("pred_")]]))
            res.extend(g[cols].copy() for _,g in x.groupby("simulation_seed"))
        return res
    scale=clean(SCALE)
    scales={f"pred_{s}_{t}":max(.25,float(np.median([np.quantile(g[f"pred_{s}_{t}"],.995,method="higher") for g in scale]))) for s in SOURCES for t in TIMES}
    del scale
    runs=clean(VAL);wt=warning_bound(runs,.001)
    candidates={}
    for audit in (False,True):
        family="audit" if audit else "primary"
        for b in (.003,.005,.007):
            name=f"{family}__control__{b}"
            candidates[name]=PredictiveFusion("none","none",0.,b,base,wt,audit).calibrate(runs,scales)
            for s in SOURCES:
                for t in TIMES:
                    for w in (.5,1.,1.5):
                        name=f"{family}__{s}__{t}__{w}__{b}"
                        candidates[name]=PredictiveFusion(s,t,w,b,base,wt,audit).calibrate(runs,scales)
    del runs
    print("PRED_REFERENCE",json.dumps({"scales":scales,"candidates":len(candidates)}),flush=True)
    dev={};dp={}
    for k,ss in DEV.items():
        raw,_,_,c=group(ss,cal,k,causal);x=labels(prep(raw,c),raw);dev[k]={};dp[k]={}
        for n,m in candidates.items():dev[k][n],dp[k][n]=quick(m,x,m.hybrid)
        print("PRED_DEV",json.dumps({"cohort":k,"candidates":len(dev[k])}),flush=True)
    selection={n:select(n,dev,dp,m.hybrid) for n,m in candidates.items()}
    best=lambda ns:max(ns,key=lambda n:(selection[n]["feasible"],selection[n]["utility"]))
    specs={}
    for family in ("primary","audit"):
        keys=[n for n in candidates if n.startswith(family+"__")]
        specs[family+"_selected"]=best(keys)
        specs[family+"_augmented"]=best(n for n in keys if candidates[n].weight>0)
        aug=candidates[specs[family+"_augmented"]]
        specs[family+"_matched_control"]=f"{family}__control__{aug.budget}"
    models={n:candidates[v] for n,v in specs.items()}
    for n in ("frozen_adaptive_fast","frozen_sequential","frozen_block","frozen_oracle"):models[n]=pkg["models"][n]
    models["frozen_joint"]=base
    print("PRED_SELECTED",json.dumps({"selected":specs,"selection":selection,"feasible_count":sum(v["feasible"] for v in selection.values())}),flush=True)
    (a.out/"development_selection.json").write_text(json.dumps({"selected":specs,"selection":selection,"development_only":True},indent=2)+"\n")
    raw,_,_,c=group([510001],cal,"clean_loss10",causal)
    latent,_,_,lc=group([510001],cal,"latent_identical",causal)
    x=prep(raw,c);l=prep(latent,lc);pd.testing.assert_frame_equal(x,l)
    bad=["cu_combined_loss_prob","master_ack_shortfall_adjusted","dev_master_ack_shortfall_vs_zone",
         "ack_shortfall_ewma_6","ack_shortfall_ewma_16","ack_shortfall_ewma_32","zone_causal_median","zone_observed_loss_median"]
    bad += [col for col in raw if col.startswith("gt_") or col in ("is_anomaly","attack_type","attacker_sophistication","normal_event","is_compromised","eval_active_deficit")]
    pd.testing.assert_frame_equal(x,prep(raw.drop(columns=bad,errors="ignore"),c.drop(columns=bad,errors="ignore")))
    prefix=prep(raw.loc[raw.interval<180],c)
    for col in [k for k in x if k.startswith("pred_") or k.startswith("joint_")]:
        left=prefix[col];right=x.loc[prefix.index,col]
        if col=="pred_valid":
            assert not left.isna().any() and not right.isna().any()
            assert left.isin([True,False]).all() and right.isin([True,False]).all()
            np.testing.assert_array_equal(left.to_numpy(dtype=bool),right.to_numpy(dtype=bool),err_msg=col)
        else:
            u=left.to_numpy(dtype=float);v=right.to_numpy(dtype=float)
            assert np.isfinite(u).all() and np.isfinite(v).all(),col
            np.testing.assert_allclose(u,v,atol=1e-9,rtol=0,err_msg=col)
    print("PRED_PREFIX_VERIFIED",flush=True)
    # ACK cannot influence inferred transport distribution.
    perturbed=x.copy();perturbed["cu_n_ack_transit"]=0
    for u,v in zip(posterior(x)[3:],posterior(perturbed)[3:]):np.testing.assert_array_equal(u,v)
    for n in specs:
        m=models[n];z=m.predict_levels(x);assert np.array_equal(z,m.predict_levels(l))
        assert np.array_equal(z[prefix.index.to_numpy()],m.predict_levels(prefix))
        sub=x.sample(n=200,random_state=42);assert np.array_equal(z[sub.index],m.predict_levels(sub))
        missing=x.copy();missing["primary_observer_valid"]=False;missing["pred_audit"]=0.
        assert not m.predict_levels(missing).any()
    with (a.out/"predictive_models.pkl").open("wb") as f:
        pickle.dump({"models":models,"frozen_package":pkg,"scales":scales,"selected":specs},f,pickle.HIGHEST_PROTOCOL)
    with (a.out/"predictive_models.pkl").open("rb") as f:reload=pickle.load(f)
    for n in specs:assert np.array_equal(models[n].predict_levels(x),reload["models"][n].predict_levels(x))
    held={};per={};coverage={}
    for k,ss in TEST.items():
        raw,_,_,c=group(ss,cal,k,causal);x=labels(prep(raw,c),raw)
        held[k]={};per[k]={}
        y=x.is_anomaly.to_numpy(dtype=bool);v=x.pred_valid.to_numpy(dtype=bool)
        coverage[k]={"valid_rows":int(v.sum()),"valid_attack_rows":int((v&y).sum()),"attack_rows":int(y.sum())}
        for n,m in models.items():
            xx=labels(refs["oracle"].prepare(raw,c),raw) if n=="frozen_oracle" else x
            held[k][n],per[k][n]=pair_metrics(m,xx,m.hybrid)
        print("PRED_HELDOUT",json.dumps({"cohort":k,**{n:short(v) for n,v in held[k].items()},"likelihood_coverage":coverage[k]}),flush=True)
    summary={}
    for n,m in models.items():
        rows=[r for k in KINDS for r in per[k][n]]
        delays=[d for k in NATIVE for d in held[k][n]["node_delay"]]
        summary[n]={"mean_FPR":float(np.mean([r["confirmed_alert_FPR"] for r in rows])),
            "max_FPR":max(r["confirmed_alert_FPR"] for r in rows),"max_FWR":max(r["false_warning_rate"] for r in rows),
            "FPR_pass":sum(r["confirmed_alert_FPR"]<=.01 for r in rows),"FWR_pass":sum(r["false_warning_rate"]<=.005 for r in rows),
            "native_recall":float(np.mean([held[k][n]["confirmed_recall"] for k in NATIVE])),
            "weak_recall":float(np.mean([held[k][n]["confirmed_recall"] for k in WEAK])),
            "policy_recall":float(np.mean([held[k][n]["confirmed_recall"] for k in POLICY])),
            "affected_node_detection_fraction":float(np.mean([d["detected"] for d in delays])),
            "node_delay_censored":delays}
    paired={}
    comparisons=[("primary_selected",n) for n in ("frozen_adaptive_fast","frozen_joint","frozen_oracle")]
    comparisons += [("primary_augmented","primary_matched_control"),("audit_augmented","audit_matched_control"),("audit_selected","primary_selected")]
    for left,right in comparisons:
        for label,ks,metric in (("native",NATIVE,"confirmed_recall"),("weak",WEAK,"confirmed_recall"),("policy",POLICY,"confirmed_recall"),("FPR",KINDS,"confirmed_alert_FPR")):
            ds=[np.array([u[metric]-v[metric] for u,v in zip(per[k][left],per[k][right])]) for k in ks]
            d=np.concatenate(ds);rng=np.random.default_rng(61319)
            boot=np.mean([g[rng.integers(0,len(g),(4000,len(g)))].mean(1) for g in ds],axis=0)
            paired[left+"_vs_"+right+"_"+label]={"delta":float(d.mean()),"n":len(d),"CI95":np.quantile(boot,[.025,.975]).tolist(),
                "p_unadjusted":1. if np.all(d==0) else float(wilcoxon(d).pvalue)}
    prev=0.
    for i,v in enumerate(sorted(paired.values(),key=lambda v:v["p_unadjusted"])):
        prev=max(prev,min(1.,(len(paired)-i)*v["p_unadjusted"]));v["p_holm"]=prev
    gates={n:summary[n]["FPR_pass"]==400 and summary[n]["FWR_pass"]==400 for n in ("primary_selected","audit_selected")}
    report={"source_run":45626804800,"scale_seeds":SCALE,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
        "selected":specs,"selection":selection,"summary":summary,"heldout":held,"per_seed":per,"paired":paired,"coverage":coverage,
        "censor_likelihood_normalization_verified":True,"ack_not_used_in_transport_posterior":True,
        "prefix_causality":True,"oracle_removal_verified":True,"observational_equivalence_verified":True,"reload_verified":True,
        "test_used_for_selection":False,"simulator_specific_likelihood":True,"synthetic_policy_audit":True,
        "real_field_validation":False,"operational_gates":gates}
    (a.out/"predictive_report.json").write_text(json.dumps(report,indent=2)+"\n")
    compact={n:{k:v for k,v in m.items() if k!="node_delay_censored"} for n,m in summary.items()}
    print("PRED_FINAL",json.dumps({"selected":specs,"summary":compact,"paired":paired,"operational_gates":gates}),flush=True)
if __name__=="__main__":main()
