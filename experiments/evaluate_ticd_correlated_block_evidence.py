#!/usr/bin/env python3
"""Independent development and test of correlation-aware block evidence."""
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
from ticd_measured_loss import labels,MeasuredFusion
from ticd_sequential_physical import sequence_features
from ticd_correlated_block_evidence import block_features,BlockAugmentedFusion,WINDOWS,SCORE_COLS

CAL=tuple(dict.fromkeys(CAL_NORMAL+("probe_delay_clean","observer_outage")))
VAL={k:[340001+100*i+j for j in range(4)] for i,k in enumerate(CAL)}
DEV={k:[350001+100*i+j for j in range(3)] for i,k in enumerate(KINDS)}
TEST={k:[360001+100*i+j for j in range(5)] for i,k in enumerate(KINDS)}
MULTS=(1.,1.15,1.35)
BLOCK_BUDGETS=(.001,.002)
WARNING_BUDGET=.0005
MODES=("adaptive_fast","adaptive_stable","hybrid","active","passive","oracle")

def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"sequential_physical_models.pkl").open("rb") as f:pkg=pickle.load(f)
    refs=pkg["references"];cal=pkg["calibration"];causal=refs["adaptive_fast"].core.causal
    seeds=[s for part in (VAL,DEV,TEST) for ss in part.values() for s in ss]
    assert len(seeds)==len(set(seeds)) and min(seeds)>330001
    def prepare(raw,c,mode,extra=True):
        x=refs[mode].prepare(raw,c)
        if extra and mode=="adaptive_fast":
            x=sequence_features(x,c,refs[mode])
            x=block_features(x,c,refs[mode])
        return x
    val={}
    for k,ss in VAL.items():
        raw,_,_,c=group(ss,cal,k,causal)
        val[k]=labels(prepare(raw,c,"adaptive_fast"),raw)
        assert not val[k].is_anomaly.any()
    frozen={m:pkg["models"]["frozen_"+m] for m in MODES}
    assert frozen["adaptive_fast"].observation_mode=="adaptive_fast"
    base=frozen["adaptive_fast"]
    candidates={}
    old=pkg["models"]["proposed"]
    for i,mult in enumerate(MULTS):
        candidates[f"control_{i}"]=BlockAugmentedFusion(base,old,None,mult,0.,WARNING_BUDGET).calibrate(val)
        for j,budget in enumerate(BLOCK_BUDGETS):
            for col in SCORE_COLS:
                name=f"{col}_m{i}_b{j}"
                candidates[name]=BlockAugmentedFusion(base,old,col,mult,budget,WARNING_BUDGET).calibrate(val)
    del val
    dev={};dp={}
    for k,ss in DEV.items():
        raw,_,_,c=group(ss,cal,k,causal)
        x=labels(prepare(raw,c,"adaptive_fast"),raw)
        dev[k]={};dp[k]={}
        for n,m in candidates.items():dev[k][n],dp[k][n]=quick(m,x,False)
        print("BLOCK_DEV",json.dumps({"cohort":k,"candidates":len(candidates)}),flush=True)
    selection={n:select(n,dev,dp,False) for n in candidates}
    # Selection uses development only. Penalize regression in standard
    # and weak scenarios; explicitly prefer satisfying both operating limits.
    for n,v in selection.items():
        v["default_recall"]=dev["default"][n]["confirmed_recall"]
        v["utility"]+=.15*v["default_recall"]+.25*v["weak_recall"]
    chosen=max((n for n in candidates if n.startswith("block_supported_") or n.startswith("block_context_")),
        key=lambda n:(selection[n]["feasible"],selection[n]["utility"]))
    mtag=chosen.split("_")[-2][1:]
    ablation=chosen.replace("block_supported_","block_plain_supported_").replace("block_context_","block_plain_")
    no_support=chosen.replace("block_supported_","block_context_")
    control="control_"+mtag
    models={**{"frozen_"+k:v for k,v in frozen.items()},
            "frozen_sequential":old,"recalibrated":candidates[control],"proposed":candidates[chosen],
            "no_context_correction":candidates[ablation],"no_block_confirmation":candidates[no_support]}
    print("BLOCK_SELECTED",json.dumps({"chosen":chosen,"ablation":ablation,"control":control,"development":selection}),flush=True)
    # Causal feature and score checks: truncate full operational prefix,
    # remove oracle / derived statistics / labels, compare equivalent worlds.
    raw,_,_,c=group([370001],cal,"clean_loss10",causal)
    latent,_,_,lc=group([370001],cal,"latent_identical",causal)
    x=prepare(raw,c,"adaptive_fast")
    l=prepare(latent,lc,"adaptive_fast")
    pd.testing.assert_frame_equal(x,l)
    prefix=prepare(raw.loc[raw.interval<180],c,"adaptive_fast")
    seqcols=[k for k in x if k.startswith(("phys_seq_","block_"))]
    for col in seqcols:np.testing.assert_allclose(prefix[col],x.loc[prefix.index,col],atol=1e-10,rtol=0)
    bad=["cu_combined_loss_prob","master_ack_shortfall_adjusted","dev_master_ack_shortfall_vs_zone",
         "ack_shortfall_ewma_6","ack_shortfall_ewma_16","ack_shortfall_ewma_32","zone_causal_median","zone_observed_loss_median"]
    bad += [col for col in raw if col.startswith("gt_") or col in ("is_anomaly","attack_type","attacker_sophistication","normal_event","is_compromised","eval_active_deficit")]
    stripped=prepare(raw.drop(columns=bad,errors="ignore"),c.drop(columns=bad,errors="ignore"),"adaptive_fast")
    pd.testing.assert_frame_equal(x,stripped)
    for name in ("proposed","no_context_correction","no_block_confirmation","recalibrated"):
        m=models[name];z=m.predict_levels(x);sample=x.sample(n=200,random_state=42)
        assert np.array_equal(z,m.predict_levels(l))
        assert np.array_equal(z[sample.index.to_numpy()],m.predict_levels(sample))
        assert np.array_equal(z[prefix.index.to_numpy()],m.predict_levels(prefix))
    with (a.out/"correlated_block_models.pkl").open("wb") as f:
        pickle.dump({"models":models,"references":refs,"calibration":cal,"selected":chosen},f,pickle.HIGHEST_PROTOCOL)
    with (a.out/"correlated_block_models.pkl").open("rb") as f:loaded=pickle.load(f)
    assert np.array_equal(models["proposed"].predict_levels(x),loaded["models"]["proposed"].predict_levels(x))
    held={};per={}
    for k,ss in TEST.items():
        raw,_,_,c=group(ss,cal,k,causal)
        held[k]={};per[k]={}
        for mode in MODES:
            xx=labels(prepare(raw,c,mode),raw)
            names=["frozen_"+mode]
            if mode=="adaptive_fast":names+=["frozen_sequential","recalibrated","proposed","no_context_correction","no_block_confirmation"]
            for name in names:
                held[k][name],per[k][name]=pair_metrics(models[name],xx,False)
        print("BLOCK_HELDOUT",json.dumps({"cohort":k,**{n:short(v) for n,v in held[k].items()}}),flush=True)
    summary={}
    for name in models:
        rows=[r for k in KINDS for r in per[k][name]]
        summary[name]={"mean_FPR":float(np.mean([r["confirmed_alert_FPR"] for r in rows])),
          "max_FPR":max(r["confirmed_alert_FPR"] for r in rows),"max_FWR":max(r["false_warning_rate"] for r in rows),
          "FPR_pass":sum(r["confirmed_alert_FPR"]<=.01 for r in rows),
          "FWR_pass":sum(r["false_warning_rate"]<=.005 for r in rows),
          "weak_recall":float(np.mean([held[k][name]["confirmed_recall"] for k in WEAK])),
          "native_recall":float(np.mean([held[k][name]["confirmed_recall"] for k in NATIVE]))}
        print("BLOCK_SUMMARY",json.dumps({"model":name,**summary[name]}),flush=True)
    paired={}
    for other in ("frozen_adaptive_fast","frozen_sequential","recalibrated","no_context_correction","no_block_confirmation","frozen_oracle","frozen_hybrid"):
        for label,ks,metric in (("native",NATIVE,"confirmed_recall"),("weak",WEAK,"confirmed_recall"),("FPR",KINDS,"confirmed_alert_FPR")):
            ds=[np.array([u[metric]-v[metric] for u,v in zip(per[k]["proposed"],per[k][other])]) for k in ks]
            d=np.concatenate(ds);rng=np.random.default_rng(61319)
            boot=np.mean([g[rng.integers(0,len(g),(4000,len(g)))].mean(1) for g in ds],axis=0)
            paired[other+"_"+label]={"delta":float(d.mean()),"n":len(d),"CI95":np.quantile(boot,[.025,.975]).tolist(),
                "p_unadjusted":1. if np.all(d==0) else float(wilcoxon(d).pvalue)}
    prev=0.
    for i,v in enumerate(sorted(paired.values(),key=lambda v:v["p_unadjusted"])):
        prev=max(prev,min(1.,(len(paired)-i)*v["p_unadjusted"]));v["p_holm"]=prev
    report={"source_run":45598072369,"frozen_feature_references":True,"real_field_validation":False,
        "test_used_for_selection":False,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
        "selected":chosen,"ablation":ablation,"no_block_confirmation":no_support,"selection":selection,"manifests":{n:m.manifest() for n,m in models.items()},
        "development":dev,"development_by_seed":dp,"heldout":held,"per_seed":per,"summary":summary,"paired":paired,
        "prefix_causality":True,"label_and_batch_invariance":True,"oracle_removal_verified":True,"reload_verified":True,
        "scenario_units":{"weak_0025":0.0025,"weak_005":0.005,"weak_01":0.01,"role_loss_suffix":"multiplier of baseline link loss, not percent"},"primary_only":True,"all_original_invariants_preserved":True,"observational_equivalence_verified":True,
        "operational_gate_pass":summary["proposed"]["FPR_pass"]==len(KINDS)*5 and summary["proposed"]["FWR_pass"]==len(KINDS)*5}
    (a.out/"correlated_block_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("BLOCK_FINAL",json.dumps({"selected":chosen,"summary":summary,"paired":paired,"operational_gate_pass":report["operational_gate_pass"]}),flush=True)
if __name__=="__main__":main()
