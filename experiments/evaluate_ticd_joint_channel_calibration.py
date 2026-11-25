#!/usr/bin/env python3
"""Matched joint threshold study on fresh disjoint seeds, no test tuning."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from evaluate_ticd_adaptive_loss_fusion import KINDS,CAL_NORMAL,NATIVE,WEAK,group,select
from evaluate_ticd_full_count_evidence import pair_metrics
from evaluate_ticd_separate_robust_fusion import quick
from evaluate_ticd_observability_factorial import short
from ticd_measured_loss import labels
from ticd_sequential_physical import sequence_features
from ticd_correlated_block_evidence import block_features
from ticd_joint_channel_calibration import JointReference,JointFusion,PROFILES,BLOCKS,RULES,OPERATING,warning_bound

CAL=tuple(dict.fromkeys(CAL_NORMAL+("probe_delay_clean","observer_outage")))
SCALE={k:[380001+100*i+j for j in range(3)] for i,k in enumerate(CAL)}
VAL={k:[385001+100*i+j for j in range(4)] for i,k in enumerate(CAL)}
DEV={k:[390001+100*i+j for j in range(5)] for i,k in enumerate(KINDS)}
TEST={k:[400001+100*i+j for j in range(10)] for i,k in enumerate(KINDS)}
MODES=("adaptive_fast","adaptive_stable","hybrid","active","passive","oracle")
WARN_BUDGET=.001

def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"correlated_block_models.pkl").open("rb") as f:pkg=pickle.load(f)
    refs=pkg["references"];cal=pkg["calibration"];causal=refs["adaptive_fast"].core.causal
    frozen={m:pkg["models"]["frozen_"+m] for m in MODES}
    base=frozen["adaptive_fast"];old=pkg["models"]["frozen_sequential"]
    priorblock=pkg["models"]["proposed"]
    seeds=[s for part in (SCALE,VAL,DEV,TEST) for ss in part.values() for s in ss]
    assert len(seeds)==len(set(seeds)) and min(seeds)>370001
    def features(raw,c,mode):
        x=refs[mode].prepare(raw,c)
        if mode=="adaptive_fast":
            x=sequence_features(x,c,refs[mode]);x=block_features(x,c,refs[mode])
        return x
    def clean(part,reference=None):
        result={}
        for k,ss in part.items():
            raw,_,_,c=group(ss,cal,k,causal)
            x=features(raw,c,"adaptive_fast")
            if reference is not None:x=reference.prepare(x)
            x=labels(x,raw)
            assert not x.is_anomaly.any()
            # Keep only required calibration data, not full feature frames.
            cols=["simulation_seed","is_anomaly","primary_observer_valid","split_eligible","warn_0"]
            cols += [c for c in x if c.startswith("joint_") or (reference is None and c.startswith("block_"))]
            result[k]=x[cols].copy()
        return result
    scale=clean(SCALE)
    jr=JointReference(base,old).fit(scale)
    del scale
    val=clean(VAL,jr)
    runs=[g for x in val.values() for _,g in x.groupby("simulation_seed")]
    wt=warning_bound(runs,WARN_BUDGET)
    candidates={}
    for profile in PROFILES:
        for block in BLOCKS:
            for rule in RULES:
                for j,(budget,q) in enumerate(OPERATING):
                    n=f"{profile}__{block}__{rule}__{j}"
                    candidates[n]=JointFusion(profile,block,rule,budget,q,wt).calibrate(runs)
    del val,runs
    print("JOINT_CANDIDATES",json.dumps({"candidates":len(candidates),"reference":jr.manifest(),"warning_threshold":wt}),flush=True)
    dev={};dp={}
    for k,ss in DEV.items():
        raw,_,_,c=group(ss,cal,k,causal)
        x=labels(jr.prepare(features(raw,c,"adaptive_fast")),raw)
        dev[k]={};dp[k]={}
        for n,m in candidates.items():dev[k][n],dp[k][n]=quick(m,x,False)
        print("JOINT_DEV",json.dumps({"cohort":k,"candidates":len(candidates)}),flush=True)
    selection={n:select(n,dev,dp,False) for n in candidates}
    for n,v in selection.items():
        v["default_recall"]=dev["default"][n]["confirmed_recall"]
        v["utility"]+=.15*v["default_recall"]+.30*v["weak_recall"]
    def best(keys):
        return max(keys,key=lambda n:(selection[n]["feasible"],selection[n]["utility"]))
    # All combinations, including absence of a block, compete without
    # preferential selection of a new component. Record genuine matched
    # controls from the best augmented and best support configurations.
    chosen=best(candidates)
    augmented=best(n for n,m in candidates.items() if m.block!="none")
    support=best(n for n,m in candidates.items() if m.rule=="support")
    no_block=augmented.split("__");no_block[1]="none";no_block="__".join(no_block)
    max_control=support.split("__");max_control[2]="max";max_control="__".join(max_control)
    chosen_names={"proposed":chosen,"best_augmented":augmented,
                  "matched_no_block":no_block,"best_support":support,"matched_max":max_control}
    models={**{"frozen_"+k:v for k,v in frozen.items()},
            "frozen_sequential":old,"frozen_block":priorblock,
            **{k:candidates[v] for k,v in chosen_names.items()}}
    print("JOINT_SELECTED",json.dumps({"selected":chosen_names,"selection":selection,
         "feasible_count":sum(v["feasible"] for v in selection.values())}),flush=True)
    # Independent audit, prior benign profiles only. Complete observed
    # prefixes and arbitrary prediction batches must agree.
    raw,_,_,c=group([410001],cal,"clean_loss10",causal)
    latent,_,_,lc=group([410001],cal,"latent_identical",causal)
    def prep(r,c):return jr.prepare(features(r,c,"adaptive_fast"))
    x=prep(raw,c);l=prep(latent,lc)
    pd.testing.assert_frame_equal(x,l)
    prefix=prep(raw.loc[raw.interval<180],c)
    for col in [k for k in x if k.startswith("joint_")]:
        np.testing.assert_allclose(prefix[col],x.loc[prefix.index,col],atol=1e-10,rtol=0)
    bad=["cu_combined_loss_prob","master_ack_shortfall_adjusted","dev_master_ack_shortfall_vs_zone",
         "ack_shortfall_ewma_6","ack_shortfall_ewma_16","ack_shortfall_ewma_32","zone_causal_median","zone_observed_loss_median"]
    bad += [col for col in raw if col.startswith("gt_") or col in ("is_anomaly","attack_type","attacker_sophistication","normal_event","is_compromised","eval_active_deficit")]
    pd.testing.assert_frame_equal(x,prep(raw.drop(columns=bad,errors="ignore"),c.drop(columns=bad,errors="ignore")))
    for name in chosen_names:
        m=models[name];z=m.predict_levels(x);sample=x.sample(n=200,random_state=42)
        assert np.array_equal(z,m.predict_levels(l))
        assert np.array_equal(z[sample.index.to_numpy()],m.predict_levels(sample))
        assert np.array_equal(z[prefix.index.to_numpy()],m.predict_levels(prefix))
        missing=x.copy();missing["primary_observer_valid"]=False
        assert not m.predict_levels(missing).any()
    with (a.out/"joint_channel_models.pkl").open("wb") as f:
        pickle.dump({"models":models,"references":refs,"joint_reference":jr,"calibration":cal,
                     "selected":chosen_names},f,pickle.HIGHEST_PROTOCOL)
    with (a.out/"joint_channel_models.pkl").open("rb") as f:loaded=pickle.load(f)
    assert np.array_equal(models["proposed"].predict_levels(x),loaded["models"]["proposed"].predict_levels(x))
    held={};per={}
    for k,ss in TEST.items():
        raw,_,_,c=group(ss,cal,k,causal)
        held[k]={};per[k]={}
        for mode in MODES:
            xx=features(raw,c,mode)
            names=["frozen_"+mode]
            if mode=="adaptive_fast":
                xx=jr.prepare(xx);names+=["frozen_sequential","frozen_block",*chosen_names]
            xx=labels(xx,raw)
            for name in names:held[k][name],per[k][name]=pair_metrics(models[name],xx,False)
        print("JOINT_HELDOUT",json.dumps({"cohort":k,**{n:short(v) for n,v in held[k].items()}}),flush=True)
    summary={};dispersion={}
    for name in models:
        rows=[r for k in KINDS for r in per[k][name]]
        summary[name]={"mean_FPR":float(np.mean([r["confirmed_alert_FPR"] for r in rows])),
          "max_FPR":max(r["confirmed_alert_FPR"] for r in rows),"max_FWR":max(r["false_warning_rate"] for r in rows),
          "FPR_pass":sum(r["confirmed_alert_FPR"]<=.01 for r in rows),
          "FWR_pass":sum(r["false_warning_rate"]<=.005 for r in rows),
          "weak_recall":float(np.mean([held[k][name]["confirmed_recall"] for k in WEAK])),
          "native_recall":float(np.mean([held[k][name]["confirmed_recall"] for k in NATIVE]))}
        dispersion[name]={}
        for k in KINDS:
            dispersion[name][k]={}
            for metric in ("confirmed_recall","confirmed_alert_FPR","false_warning_rate"):
                v=[r[metric] for r in per[k][name] if r[metric] is not None]
                dispersion[name][k][metric]={"mean":float(np.mean(v)),"sd":float(np.std(v,ddof=1)),"n":len(v)} if len(v)>1 else None
        print("JOINT_SUMMARY",json.dumps({"model":name,**summary[name]}),flush=True)
    paired={}
    comparisons=[("proposed",r) for r in ("frozen_adaptive_fast","frozen_sequential","frozen_block","frozen_oracle","frozen_hybrid")]
    comparisons += [("best_augmented","matched_no_block"),("best_support","matched_max")]
    for left,right in comparisons:
        for label,ks,metric in (("native",NATIVE,"confirmed_recall"),("weak",WEAK,"confirmed_recall"),("FPR",KINDS,"confirmed_alert_FPR")):
            ds=[]
            for k in ks:
                a0=per[k][left];b0=per[k][right]
                assert [r["seed"] for r in a0]==[r["seed"] for r in b0]
                ds.append(np.array([u[metric]-v[metric] for u,v in zip(a0,b0)]))
            d=np.concatenate(ds);rng=np.random.default_rng(61319)
            boot=np.mean([g[rng.integers(0,len(g),(4000,len(g)))].mean(1) for g in ds],axis=0)
            paired[left+"_vs_"+right+"_"+label]={"delta":float(d.mean()),"n":len(d),"CI95":np.quantile(boot,[.025,.975]).tolist(),
                "p_unadjusted":1. if np.all(d==0) else float(wilcoxon(d).pvalue)}
    prev=0.
    for i,v in enumerate(sorted(paired.values(),key=lambda v:v["p_unadjusted"])):
        prev=max(prev,min(1.,(len(paired)-i)*v["p_unadjusted"]));v["p_holm"]=prev
    report={"source_run":45612416004,"frozen_feature_references":True,"real_field_validation":False,
        "test_used_for_selection":False,"scale_seeds":SCALE,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
        "selected":chosen_names,"selection":selection,"joint_reference":jr.manifest(),
        "manifests":{n:m.manifest() for n,m in models.items()},
        "development":dev,"development_by_seed":dp,"heldout":held,"per_seed":per,"summary":summary,
        "per_cohort_mean_sd":dispersion,"paired":paired,"prefix_causality":True,"label_and_batch_invariance":True,
        "oracle_removal_verified":True,"reload_verified":True,"all_original_invariants_preserved":True,
        "observational_equivalence_verified":True,"missing_primary_no_alert_verified":True,
        "joint_calibration_no_independence_assumption":True,"all_configurations_eligible":True,"primary_only":True,
        "scenario_units":{"weak_0025":.0025,"weak_005":.005,"weak_01":.01,"role_loss_suffix":"baseline link loss multiplier"},
        "operational_gate_pass":summary["proposed"]["FPR_pass"]==len(KINDS)*10 and summary["proposed"]["FWR_pass"]==len(KINDS)*10}
    (a.out/"joint_channel_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("JOINT_FINAL",json.dumps({"selected":chosen_names,"summary":summary,"paired":paired,"operational_gate_pass":report["operational_gate_pass"]}),flush=True)
if __name__=="__main__":main()
