#!/usr/bin/env python3
"""Development and independent confirmation of a supplementary TICD channel."""
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
from ticd_sequential_physical import sequence_features,SequentialPhysicalFusion,SPECS

CAL=tuple(dict.fromkeys(CAL_NORMAL+("probe_delay_clean","observer_outage")))
VAL={k:[300001+100*i+j for j in range(4)] for i,k in enumerate(CAL)}
DEV={k:[310001+100*i+j for j in range(3)] for i,k in enumerate(KINDS)}
TEST={k:[320001+100*i+j for j in range(5)] for i,k in enumerate(KINDS)}
BUDGETS=((.001,.001,.0015,.0025,0.,.002),(.00075,.00075,.001,.002,0.,.0035))
MODES=("adaptive_fast","adaptive_stable","hybrid","active","passive","oracle")

def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"adaptive_loss_fusion_models.pkl").open("rb") as f:pkg=pickle.load(f)
    refs=pkg["references"];cal=pkg["calibration"];causal=refs["adaptive_fast"].core.causal
    seeds=[s for part in (VAL,DEV,TEST) for ss in part.values() for s in ss]
    assert len(seeds)==len(set(seeds)) and min(seeds)>276901
    def prepare(raw,c,mode,extra=True):
        x=refs[mode].prepare(raw,c)
        if extra and mode=="adaptive_fast":x=sequence_features(x,c,refs[mode])
        return x
    val={}
    for k,ss in VAL.items():
        raw,_,_,c=group(ss,cal,k,causal)
        val[k]=labels(prepare(raw,c,"adaptive_fast"),raw)
        assert not val[k].is_anomaly.any()
    frozen={m:pkg["models"][m+"_primary"] for m in MODES}
    assert frozen["adaptive_fast"].observation_mode=="adaptive_fast"
    base=frozen["adaptive_fast"]
    candidates={}
    # Recalibrated original architecture is a matched control. Same normal
    # calibration data as the candidates, identical frozen feature references.
    candidates["recalibrated"]=MeasuredFusion("adaptive_fast",base.loss_window,base.budgets,"worst_run").calibrate(val)
    for bidx,b in enumerate(BUDGETS):
        for j,_ in enumerate(SPECS):
            for weight in (0,1):
                col=f"phys_seq_{j}_{weight}"
                name=f"seq_{j}_{weight}_budget{bidx}"
                candidates[name]=SequentialPhysicalFusion("adaptive_fast",base.loss_window,b,col).calibrate(val)
    del val
    dev={};dp={}
    for k,ss in DEV.items():
        raw,_,_,c=group(ss,cal,k,causal)
        x=labels(prepare(raw,c,"adaptive_fast"),raw)
        dev[k]={};dp[k]={}
        for n,m in candidates.items():dev[k][n],dp[k][n]=quick(m,x,False)
        print("SEQ_DEV",json.dumps({"cohort":k,"candidates":len(candidates)}),flush=True)
    selection={n:select(n,dev,dp,False) for n in candidates}
    # Fix the candidate using development only; evaluate the exact paired
    # reliability ablation (same budget and decay), not a retuned substitute.
    chosen=max((n for n in candidates if n.startswith("seq_") and n.split("_")[2]=="1"),
        key=lambda n:(selection[n]["feasible"],selection[n]["utility"]))
    ablation=chosen.replace("_1_budget","_0_budget")
    models={**{"frozen_"+k:v for k,v in frozen.items()},
            "recalibrated":candidates["recalibrated"],"proposed":candidates[chosen],
            "no_reliability_weight":candidates[ablation]}
    print("SEQ_SELECTED",json.dumps({"chosen":chosen,"ablation":ablation,"development":selection}),flush=True)
    # Causal feature and score checks: truncate full operational prefix,
    # remove oracle / derived statistics / labels, compare equivalent worlds.
    raw,_,_,c=group([330001],cal,"clean_loss10",causal)
    latent,_,_,lc=group([330001],cal,"latent_identical",causal)
    x=prepare(raw,c,"adaptive_fast")
    l=prepare(latent,lc,"adaptive_fast")
    pd.testing.assert_frame_equal(x,l)
    prefix=prepare(raw.loc[raw.interval<180],c,"adaptive_fast")
    seqcols=[k for k in x if k.startswith("phys_seq_")]
    for col in seqcols:np.testing.assert_allclose(prefix[col],x.loc[prefix.index,col],atol=1e-10,rtol=0)
    bad=["cu_combined_loss_prob","master_ack_shortfall_adjusted","dev_master_ack_shortfall_vs_zone",
         "ack_shortfall_ewma_6","ack_shortfall_ewma_16","ack_shortfall_ewma_32","zone_causal_median","zone_observed_loss_median"]
    bad += [col for col in raw if col.startswith("gt_") or col in ("is_anomaly","attack_type","attacker_sophistication","normal_event","is_compromised","eval_active_deficit")]
    stripped=prepare(raw.drop(columns=bad,errors="ignore"),c.drop(columns=bad,errors="ignore"),"adaptive_fast")
    pd.testing.assert_frame_equal(x,stripped)
    for name in ("proposed","no_reliability_weight","recalibrated"):
        m=models[name];z=m.predict_levels(x);sample=x.sample(n=200,random_state=42)
        assert np.array_equal(z,m.predict_levels(l))
        assert np.array_equal(z[sample.index.to_numpy()],m.predict_levels(sample))
        assert np.array_equal(z[prefix.index.to_numpy()],m.predict_levels(prefix))
    with (a.out/"sequential_physical_models.pkl").open("wb") as f:
        pickle.dump({"models":models,"references":refs,"calibration":cal,"selected":chosen},f,pickle.HIGHEST_PROTOCOL)
    with (a.out/"sequential_physical_models.pkl").open("rb") as f:loaded=pickle.load(f)
    assert np.array_equal(models["proposed"].predict_levels(x),loaded["models"]["proposed"].predict_levels(x))
    held={};per={}
    for k,ss in TEST.items():
        raw,_,_,c=group(ss,cal,k,causal)
        held[k]={};per[k]={}
        for mode in MODES:
            xx=labels(prepare(raw,c,mode),raw)
            names=["frozen_"+mode]
            if mode=="adaptive_fast":names+=["recalibrated","proposed","no_reliability_weight"]
            for name in names:
                held[k][name],per[k][name]=pair_metrics(models[name],xx,False)
        print("SEQ_HELDOUT",json.dumps({"cohort":k,**{n:short(v) for n,v in held[k].items()}}),flush=True)
    summary={}
    for name in models:
        rows=[r for k in KINDS for r in per[k][name]]
        summary[name]={"mean_FPR":float(np.mean([r["confirmed_alert_FPR"] for r in rows])),
          "max_FPR":max(r["confirmed_alert_FPR"] for r in rows),"max_FWR":max(r["false_warning_rate"] for r in rows),
          "FPR_pass":sum(r["confirmed_alert_FPR"]<=.01 for r in rows),
          "FWR_pass":sum(r["false_warning_rate"]<=.005 for r in rows),
          "weak_recall":float(np.mean([held[k][name]["confirmed_recall"] for k in WEAK])),
          "native_recall":float(np.mean([held[k][name]["confirmed_recall"] for k in NATIVE]))}
        print("SEQ_SUMMARY",json.dumps({"model":name,**summary[name]}),flush=True)
    paired={}
    for other in ("frozen_adaptive_fast","recalibrated","no_reliability_weight","frozen_oracle","frozen_hybrid"):
        for label,ks,metric in (("native",NATIVE,"confirmed_recall"),("weak",WEAK,"confirmed_recall"),("FPR",KINDS,"confirmed_alert_FPR")):
            ds=[np.array([u[metric]-v[metric] for u,v in zip(per[k]["proposed"],per[k][other])]) for k in ks]
            d=np.concatenate(ds);rng=np.random.default_rng(61319)
            boot=np.mean([g[rng.integers(0,len(g),(4000,len(g)))].mean(1) for g in ds],axis=0)
            paired[other+"_"+label]={"delta":float(d.mean()),"n":len(d),"CI95":np.quantile(boot,[.025,.975]).tolist(),
                "p_unadjusted":1. if np.all(d==0) else float(wilcoxon(d).pvalue)}
    prev=0.
    for i,v in enumerate(sorted(paired.values(),key=lambda v:v["p_unadjusted"])):
        prev=max(prev,min(1.,(len(paired)-i)*v["p_unadjusted"]));v["p_holm"]=prev
    report={"source_run":45589776522,"frozen_feature_references":True,"real_field_validation":False,
        "test_used_for_selection":False,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
        "selected":chosen,"ablation":ablation,"selection":selection,"manifests":{n:m.manifest() for n,m in models.items()},
        "development":dev,"development_by_seed":dp,"heldout":held,"per_seed":per,"summary":summary,"paired":paired,
        "prefix_causality":True,"label_and_batch_invariance":True,"oracle_removal_verified":True,"reload_verified":True,
        "primary_only":True,"all_original_invariants_preserved":True,"observational_equivalence_verified":True,
        "operational_gate_pass":summary["proposed"]["FPR_pass"]==len(KINDS)*5 and summary["proposed"]["FWR_pass"]==len(KINDS)*5}
    (a.out/"sequential_physical_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("SEQ_FINAL",json.dumps({"selected":chosen,"summary":summary,"paired":paired,"operational_gate_pass":report["operational_gate_pass"]}),flush=True)
if __name__=="__main__":main()
