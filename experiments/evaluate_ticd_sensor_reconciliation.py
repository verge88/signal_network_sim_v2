#!/usr/bin/env python3
"""Sensor reconciliation, independent variance calibration and full refit."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from evaluate_ticd_adaptive_loss_fusion import KINDS,CAL_NORMAL,NORMAL,NATIVE,WEAK,group,select
from evaluate_ticd_full_count_evidence import pair_metrics
from evaluate_ticd_separate_robust_fusion import quick
from evaluate_ticd_observability_factorial import short
from diagnose_ticd_root_causes import observed
from ticd_measured_loss import FullMeasuredReference,measured_frame,labels
from ticd_sensor_reconciliation import VarianceCalibration,regime,estimate_reconciled
from ticd_sequential_physical import sequence_features
from ticd_correlated_block_evidence import block_features
from ticd_joint_channel_calibration import JointReference,JointFusion,warning_bound
import ticd_measured_loss as measured

CAL=tuple(dict.fromkeys(CAL_NORMAL+("probe_delay_clean","observer_outage")))
VAR={k:[430001+100*i+j for j in range(4)] for i,k in enumerate(CAL)}
TRAIN={k:[435001+100*i+j for j in range(4)] for i,k in enumerate(NORMAL)}
SCALE={k:[440001+100*i+j for j in range(3)] for i,k in enumerate(CAL)}
VAL={k:[445001+100*i+j for j in range(4)] for i,k in enumerate(CAL)}
DEV={k:[450001+100*i+j for j in range(3)] for i,k in enumerate(KINDS)}
TEST={k:[460001+100*i+j for j in range(10)] for i,k in enumerate(KINDS)}
MODES=("adaptive_fast","reconciled_raw","reconciled_cal")
PROFILES=("balanced","legacy_fast","weak_physical")
BLOCKS=("none","plain","context","supported","plain_supported")
BUDGETS=(.003,.005,.007)
def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"joint_channel_models.pkl").open("rb") as f:pkg=pickle.load(f)
    oldrefs=pkg["references"];oldmodels=pkg["models"];source=oldrefs["adaptive_fast"].core
    cal=pkg["calibration"];causal=source.causal
    seeds=[s for part in (VAR,TRAIN,SCALE,VAL,DEV,TEST) for ss in part.values() for s in ss]
    assert len(seeds)==len(set(seeds)) and min(seeds)>420008
    vc=VarianceCalibration()
    items=[]
    for k,ss in VAR.items():
        raw,c=observed(ss,cal,k,causal)
        assert not raw.is_anomaly.any()
        x=measured_frame(raw,"reconciled_raw")
        items.append((x,raw._diagnostic_path_p.to_numpy(dtype=float)))
    vc.fit(items);del items
    print("RECON_VARIANCE_CAL",json.dumps(vc.manifest()),flush=True)
    tr=[group(ss,cal,k,causal) for k,ss in TRAIN.items()]
    train=pd.concat([v[0] for v in tr],ignore_index=True);commission=pd.concat([v[3] for v in tr],ignore_index=True)
    del tr
    scale_groups={k:group(ss,cal,k,causal) for k,ss in SCALE.items()}
    val_groups={k:group(ss,cal,k,causal) for k,ss in VAL.items()}
    refs={};jrefs={};candidates={}
    def feat(ref,raw,c):
        x=ref.prepare(raw,c)
        x=sequence_features(x,c,ref);return block_features(x,c,ref)
    def prepared(ref,jr,raw,c):return labels(jr.prepare(feat(ref,raw,c)),raw)
    base=oldmodels["frozen_adaptive_fast"];old=oldmodels["frozen_sequential"]
    for mode in MODES:
        ref=FullMeasuredReference(mode,source)
        if mode=="reconciled_cal":ref.variance_calibrator=vc
        ref.fit(train,commission,{k:v[0] for k,v in scale_groups.items()})
        refs[mode]=ref
        scale={k:labels(feat(ref,v[0],v[3]),v[0]) for k,v in scale_groups.items()}
        jr=JointReference(base,old).fit(scale);jrefs[mode]=jr;del scale
        val={k:prepared(ref,jr,v[0],v[3]) for k,v in val_groups.items()}
        runs=[g for x in val.values() for _,g in x.groupby("simulation_seed")]
        wt=warning_bound(runs,.001)
        for prof in PROFILES:
            for block in BLOCKS:
                for rule in ("max","support"):
                    for j,b in enumerate(BUDGETS):
                        name=f"{mode}__{prof}__{block}__{rule}__{j}"
                        candidates[name]=JointFusion(prof,block,rule,b,1.,wt).calibrate(runs)
        del val,runs
        print("RECON_REFERENCE",json.dumps({"mode":mode,"joint":jr.manifest(),"warning_threshold":wt}),flush=True)
    del train,commission,scale_groups,val_groups
    dev={};dp={}
    for k,ss in DEV.items():
        raw,_,_,c=group(ss,cal,k,causal);dev[k]={};dp[k]={}
        for mode in MODES:
            x=prepared(refs[mode],jrefs[mode],raw,c)
            for n,m in candidates.items():
                if n.startswith(mode+"__"):dev[k][n],dp[k][n]=quick(m,x,False)
        print("RECON_DEV",json.dumps({"cohort":k,"candidates":len(dev[k])}),flush=True)
    selection={n:select(n,dev,dp,False) for n in candidates}
    for n,v in selection.items():
        v["utility"]+=.30*v["weak_recall"]+.15*dev["default"][n]["confirmed_recall"]
    best=lambda ns:max(ns,key=lambda n:(selection[n]["feasible"],selection[n]["utility"]))
    chosen={mode:best(n for n in candidates if n.startswith(mode+"__")) for mode in MODES}
    # Select the operational candidate from both reconciliation modes;
    # matched refit baseline remains an explicit comparator.
    proposed=best([chosen["reconciled_raw"],chosen["reconciled_cal"]])
    # Exact same fusion configuration with and without variance calibration.
    ablation=proposed.replace("reconciled_cal__","reconciled_raw__") if proposed.startswith("reconciled_cal__") else proposed.replace("reconciled_raw__","reconciled_cal__")
    specs={"proposed":proposed,"matched_variance":ablation,"refit_old":chosen["adaptive_fast"],
           "refit_raw":chosen["reconciled_raw"],"refit_cal":chosen["reconciled_cal"]}
    models={n:candidates[v] for n,v in specs.items()}
    models.update({n:oldmodels[n] for n in ("frozen_adaptive_fast","frozen_sequential","frozen_block","frozen_oracle")})
    models["frozen_joint"]=oldmodels["proposed"]
    print("RECON_SELECTED",json.dumps({"selected":specs,"selection":selection,"feasible_count":sum(v["feasible"] for v in selection.values())}),flush=True)
    # Verify prefix causality and primitive observation boundary for all
    # new references; no simulated truth is passed to inference.
    raw,_,_,c=group([470001],cal,"clean_loss10",causal)
    latent,_,_,lc=group([470001],cal,"latent_identical",causal)
    bad=["cu_combined_loss_prob","master_ack_shortfall_adjusted","dev_master_ack_shortfall_vs_zone",
         "ack_shortfall_ewma_6","ack_shortfall_ewma_16","ack_shortfall_ewma_32","zone_causal_median","zone_observed_loss_median"]
    bad += [col for col in raw if col.startswith("gt_") or col in ("is_anomaly","attack_type","attacker_sophistication","normal_event","is_compromised","eval_active_deficit")]
    for mode in MODES:
        def prep(r,c):return jrefs[mode].prepare(feat(refs[mode],r,c))
        x=prep(raw,c);l=prep(latent,lc);pd.testing.assert_frame_equal(x,l)
        pd.testing.assert_frame_equal(x,prep(raw.drop(columns=bad,errors="ignore"),c.drop(columns=bad,errors="ignore")))
        prefix=prep(raw.loc[raw.interval<180],c)
        for col in [k for k in x if k.startswith("joint_") or k.startswith("measured_")]:
            np.testing.assert_allclose(prefix[col],x.loc[prefix.index,col],atol=1e-10,rtol=0)
        for n,v in specs.items():
            if not v.startswith(mode+"__"):continue
            z=models[n].predict_levels(x);assert np.array_equal(z,models[n].predict_levels(l))
            assert np.array_equal(z[prefix.index.to_numpy()],models[n].predict_levels(prefix))
            missing=x.copy();missing["primary_observer_valid"]=False;assert not models[n].predict_levels(missing).any()
    held={};per={};diagnostics={}
    for k,ss in TEST.items():
        raw,c=observed(ss,cal,k,causal)
        truth=raw._diagnostic_path_p.to_numpy(dtype=float)
        # Strip diagnostic ground truth before any reference preparation.
        drop=[col for col in raw if col.startswith("_diagnostic_")]
        raw=raw.drop(columns=drop);c=c.drop(columns=drop,errors="ignore")
        held[k]={};per[k]={};diagnostics[k]={}
        def stats(x):
            good=x.primary_observer_valid.to_numpy(dtype=bool)
            y=raw.is_anomaly.to_numpy(dtype=bool);err=x.measured_oneway_loss.to_numpy()-truth
            sd=np.sqrt(np.maximum(x.measured_loss_variance.to_numpy(),1e-9))
            out={}
            for label,mask in (("all",good),("normal",good&~y),("attack",good&y)):
                out[label]={"n":int(mask.sum()),"bias":float(err[mask].mean()),"rmse":float(np.sqrt(np.mean(err[mask]**2))),
                   "mean_sd":float(sd[mask].mean()),"coverage_95":float((np.abs(err[mask])<=1.96*sd[mask]).mean()),
                   "standardized_mse":float(np.mean((err[mask]/sd[mask])**2))} if mask.any() else None
            return out
        for mode in MODES:
            x=prepared(refs[mode],jrefs[mode],raw,c)
            diagnostics[k][mode]=stats(x)
            for n,v in specs.items():
                if v.startswith(mode+"__"):held[k][n],per[k][n]=pair_metrics(models[n],x,False)
        # Measurement-only ablations; do not re-tune detector or claim
        # their recall. The same physical traces are used for all variants.
        for name,rec,lag in (("no_reconciliation",False,True),("no_delay_correction",True,False)):
            def est(x,pc,vc):return estimate_reconciled(x,pc,vc,reconcile=rec,delay_correct=lag)
            with patch.object(measured,"estimate_reconciled",est):
                diagnostics[k][name]=stats(measured_frame(raw,"reconciled_raw"))
        x=labels(pkg["joint_reference"].prepare(feat(oldrefs["adaptive_fast"],raw,c)),raw)
        for n in ("frozen_adaptive_fast","frozen_sequential","frozen_block","frozen_joint"):
            held[k][n],per[k][n]=pair_metrics(models[n],x,False)
        x=labels(oldrefs["oracle"].prepare(raw,c),raw)
        held[k]["frozen_oracle"],per[k]["frozen_oracle"]=pair_metrics(models["frozen_oracle"],x,False)
        print("RECON_MEASUREMENT",json.dumps({"cohort":k,"diagnostics":diagnostics[k]}),flush=True)
        print("RECON_HELDOUT",json.dumps({"cohort":k,**{n:short(v) for n,v in held[k].items()}}),flush=True)
    summary={}
    for n in models:
        rows=[r for k in KINDS for r in per[k][n]]
        summary[n]={"mean_FPR":float(np.mean([r["confirmed_alert_FPR"] for r in rows])),
          "max_FPR":max(r["confirmed_alert_FPR"] for r in rows),"max_FWR":max(r["false_warning_rate"] for r in rows),
          "FPR_pass":sum(r["confirmed_alert_FPR"]<=.01 for r in rows),"FWR_pass":sum(r["false_warning_rate"]<=.005 for r in rows),
          "weak_recall":float(np.mean([held[k][n]["confirmed_recall"] for k in WEAK])),
          "native_recall":float(np.mean([held[k][n]["confirmed_recall"] for k in NATIVE]))}
    paired={}
    for other in ("frozen_joint","refit_old","matched_variance","frozen_adaptive_fast","frozen_oracle"):
        for label,ks,metric in (("native",NATIVE,"confirmed_recall"),("weak",WEAK,"confirmed_recall"),("FPR",KINDS,"confirmed_alert_FPR")):
            ds=[np.array([u[metric]-v[metric] for u,v in zip(per[k]["proposed"],per[k][other])]) for k in ks]
            d=np.concatenate(ds);rng=np.random.default_rng(61319)
            boot=np.mean([g[rng.integers(0,len(g),(4000,len(g)))].mean(1) for g in ds],axis=0)
            paired[other+"_"+label]={"delta":float(d.mean()),"n":len(d),"CI95":np.quantile(boot,[.025,.975]).tolist(),
              "p_unadjusted":1. if np.all(d==0) else float(wilcoxon(d).pvalue)}
    prev=0.
    for i,v in enumerate(sorted(paired.values(),key=lambda v:v["p_unadjusted"])):
        prev=max(prev,min(1.,(len(paired)-i)*v["p_unadjusted"]));v["p_holm"]=prev
    with (a.out/"sensor_reconciliation_models.pkl").open("wb") as f:
        pickle.dump({"models":models,"references":refs,"joint_references":jrefs,"variance_calibration":vc,
            "frozen_package":pkg,"calibration":cal,"selected":specs},f,pickle.HIGHEST_PROTOCOL)
    with (a.out/"sensor_reconciliation_models.pkl").open("rb") as f:reload=pickle.load(f)
    assert reload["selected"]==specs and reload["variance_calibration"].manifest()==vc.manifest()
    report={"source_run":45626804800,"diagnostic_run":45652567764,"variance_seeds":VAR,"training_seeds":TRAIN,
        "scale_seeds":SCALE,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,"selected":specs,
        "selection":selection,"summary":summary,"heldout":held,"per_seed":per,"paired":paired,"measurement_diagnostics":diagnostics,
        "variance_calibration":vc.manifest(),"all_original_invariants_preserved":True,"full_reference_refit":True,
        "test_used_for_selection":False,"oracle_truth_used_for_variance_calibration_only":True,
        "oracle_removal_verified":True,"reload_verified":True,"prefix_causality":True,"observational_equivalence_verified":True,
        "matched_200_population_assumed":True,"real_field_validation":False,
        "operational_gate_pass":summary["proposed"]["FPR_pass"]==400 and summary["proposed"]["FWR_pass"]==400}
    (a.out/"sensor_reconciliation_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("RECON_FINAL",json.dumps({"selected":specs,"summary":summary,"paired":paired,
        "operational_gate_pass":report["operational_gate_pass"]}),flush=True)
if __name__=="__main__":main()
