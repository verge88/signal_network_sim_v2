#!/usr/bin/env python3
"""Compare original fixed-scale TICD to multi-scale invariant TICD.

Calibration: exact externally anchored Di Mauro benign SIP method profile
from previous checksum-verified real-PCAP calibration artifact.
All training samples are clean synthetic simulations. Attack labels are used
ONLY for architecture choice on development synthetic seeds and final
evaluation on untouched, separately seeded synthetic test runs.

No parameter, threshold or candidate selection uses test labels or scores.
"""
from __future__ import annotations
import argparse,hashlib,json,pickle,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
sys.path.insert(0,str(ROOT/"experiments"))
from retrain_sip_ticd_calibrated import simulation,evaluate
from ticd_recalibrated import FrozenReferenceTICD
from ticd_temporal_invariants import MultiScaleInvariantTICD

PARTITIONS={
  "train":[1101,1102],
  "threshold_validation":[1201,1202],
  "architecture_development":[1301,1302],
  "untouched_test":[1401,1402,1403,1404,1405],
}
MODELS=("baseline",)+MultiScaleInvariantTICD.CANDIDATES

def threshold_at_fpr(scores,fpr=0.01):
    # Single threshold from CLEAN VALIDATION, strictly greater.
    return float(np.quantile(scores,1-fpr,method="higher"))

def mean_standard_deviation(values):
    valid=np.asarray([v for v in values if v is not None],dtype=float)
    return {"mean":float(np.mean(valid)) if len(valid) else None,
            "sd":float(np.std(valid,ddof=1)) if len(valid)>1 else None,
            "n":len(valid)}

def selected_utility(eval_out):
    """Development-only utility, not used for one-class weight fitting."""
    comp=eval_out["by_compromise_sophistication"]
    stat=comp["statistical"]["recall"] or 0.
    adap=comp["adaptive"]["recall"] or 0.
    total=eval_out["tpr_recall"] or 0.
    fpr=eval_out["fpr"] or 0.
    # Dev synthetic labelled evaluation is separate from threshold
    # validation. Penalize dev FPR beyond 2% to avoid brittle candidates.
    return (0.4*stat+0.4*adap+0.2*total)-2.0*max(0.,fpr-0.02)

def run(args):
    started=time.time();out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    cal=json.loads(args.calibration.read_text())
    if cal.get("calibration_kind")!="partial_benign_SIP_method_profile":
        raise RuntimeError("Expected real-PCAP derived calibration manifest")
    if cal["source_archive_sha256"]!="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7":
        raise RuntimeError("Calibration does not match independently verified LTE archive")
    assert len(set(sum(PARTITIONS.values(),[])))==sum(len(x) for x in PARTITIONS.values())
    data={};seeds_log=[]
    for partition,seeds in PARTITIONS.items():
        observed=[]
        for seed in seeds:
            sample,meta=simulation(seed,cal,args.hours,
                                   attacks_enabled=partition in ("architecture_development","untouched_test"))
            sample["experiment_partition"]=partition
            observed.append(sample)
            meta["partition"]=partition
            seeds_log.append(meta)
            print("MULTISCALE_SIM",json.dumps({"partition":partition,"seed":seed,
                "rows":len(sample),"attacks":int(sample.is_anomaly.sum())}),flush=True)
        df=pd.concat(observed,ignore_index=True)
        data[partition]=df
        df.to_csv(out/(partition+".csv.gz"),index=False,compression="gzip")
    for p in ("train","threshold_validation"):
        if (data[p]["is_anomaly"]!=0).any():
            raise RuntimeError(f"{p} unexpectedly includes attack-labelled data")
    if len(data["untouched_test"])<10000 or data["untouched_test"]["is_anomaly"].sum()<1000:
        raise RuntimeError("Insufficient held-out evaluation")
    # Fit baseline once on strictly normal and share immutable reference with
    # all multiscale candidates (fit never references attack label 1).
    base=FrozenReferenceTICD(seed=42).fit(data["train"])
    models={"baseline":base}
    for mode in MultiScaleInvariantTICD.CANDIDATES:
        models[mode]=MultiScaleInvariantTICD(base_model=base,mode=mode).fit(data["train"])
    val=data["threshold_validation"]
    validation_scores={name:mdl.score(val) for name,mdl in models.items()}
    for n,v in validation_scores.items():
        if not np.isfinite(v).all():raise RuntimeError(f"Nonfinite validation score {n}")
    thresholds={name:threshold_at_fpr(v) for name,v in validation_scores.items()}
    validation_actual_fpr={name:float((validation_scores[name]>thresholds[name]).mean())
                           for name in models}
    if any(v>0.011 for v in validation_actual_fpr.values()):
        raise RuntimeError("Validation budget violation")
    # Development labels are used only for architecture selection, not model
    # fit, normal-reference scales, nor threshold.
    dev=data["architecture_development"]
    developmental={name:evaluate(model,dev,thresholds[name])
                   for name,model in models.items()}
    utilities={name:float(selected_utility(developmental[name])) for name in models}
    # Baseline remains comparator; choose among 4 proposed upgraded branches.
    selected=max(MultiScaleInvariantTICD.CANDIDATES,key=lambda x:(utilities[x],-MODELS.index(x)))
    print("ARCHITECTURE_SELECTED_ON_DEV",json.dumps({
         "selected":selected,"utilities":utilities,
         "development_compromise":{k:v["by_compromise_sophistication"]
                                    for k,v in developmental.items()}},ensure_ascii=False),flush=True)

    # Exactly one final held-out evaluation pass across five new seeds.
    test=data["untouched_test"]
    baseline_results=evaluate(base,test,thresholds["baseline"])
    proposed_results=evaluate(models[selected],test,thresholds[selected])
    by_seed={}
    for name in ("baseline",selected):
        rows=[]
        for seed in PARTITIONS["untouched_test"]:
            s=test[test.simulation_seed==seed]
            m=evaluate(models[name],s,thresholds[name])
            m["simulation_seed"]=seed
            rows.append(m)
        by_seed[name]=rows
    diffs={m:mean_standard_deviation([n[m] for n in by_seed[selected]])
           for m in ("auc_roc","auc_pr","f1","fpr","tpr_recall","precision")}
    paired=[]
    for old,new in zip(by_seed["baseline"],by_seed[selected],strict=True):
        paired.append({"seed":old["simulation_seed"],
                       "recall_improvement":new["tpr_recall"]-old["tpr_recall"],
                       "fpr_change":new["fpr"]-old["fpr"],
                       "roc_auc_change":new["auc_roc"]-old["auc_roc"],
                       "naive_recall_change":new["by_compromise_sophistication"]["naive"]["recall"]
                                       -old["by_compromise_sophistication"]["naive"]["recall"],
                       "statistical_recall_change":new["by_compromise_sophistication"]["statistical"]["recall"]
                                       -old["by_compromise_sophistication"]["statistical"]["recall"],
                       "adaptive_recall_change":new["by_compromise_sophistication"]["adaptive"]["recall"]
                                       -old["by_compromise_sophistication"]["adaptive"]["recall"]})
    meta={
      "scope":"simulation-to-simulation only, real LTE PCAP partial method calibration",
      "real_calibration_archive_sha256":cal["source_archive_sha256"],
      "architecture_selection":"development attacks with penalized statistical/adaptive macro-recall, separate from final test",
      "candidate_modes":list(MultiScaleInvariantTICD.CANDIDATES),
      "selected_mode":selected,
      "normal_training_only":True,
      "test_data_never_used_for_fit_scale_threshold_or_selection":True,
      "train_val_dev_test_seeds":PARTITIONS,
      "same_attacks_times_for_all_seeds":True,
      "same_test_records_for_comparator_and_selected_model":True,
      "calibration_parameter_weights":cal["method_weights"],
      "candidate_normal_scales":models[selected].scales,
      "validation_thresholds":thresholds,
      "actual_validation_fpr":validation_actual_fpr,
      "development_selection_utilities":utilities,
      "development_metrics":{name:{
          "fpr":v["fpr"],"recall":v["tpr_recall"],"f1":v["f1"],
          "by_compromise_sophistication":v["by_compromise_sophistication"]
        } for name,v in developmental.items()},
      "comparison_on_unseen_simulated_test":{
        "baseline":baseline_results,"multiscale_ticd":proposed_results
      },
      "per_test_seed":by_seed,
      "paired_seed_differences":paired,
      "multiscale_mean_sd":diffs,
      "normal_train_rows":len(data["train"]),
      "normal_threshold_rows":len(val),
      "development_rows":len(dev),
      "test_rows":len(test),
      "synthetic_test_attack_count":int(test["is_anomaly"].sum()),
      "simulation_schedules":seeds_log,
      "runtime_seconds":time.time()-started,
      "external_ticd_recall":None,
      "external_ticd_fpr":None,
      "external_real_compromise_metrics":None,
      "limitations":"Held-out random seeds, not held-out attack mechanisms or independent real CU/zone compromise telemetry."
    }
    for name in ("baseline","multiscale_ticd"):
        v=meta["comparison_on_unseen_simulated_test"][name]
        print("FINAL_HELDOUT",json.dumps({"name":name,"TP":v["tp"],"TN":v["tn"],
           "FP":v["fp"],"FN":v["fn"],"ROC_AUC":v["auc_roc"],
           "PR_AUC":v["auc_pr"],"F1":v["f1"],"recall":v["tpr_recall"],
           "FPR":v["fpr"],"compromise":v["by_compromise_sophistication"]}),flush=True)
    (out/"comparison_report.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")
    with (out/"ticd_multiscale_model.pkl").open("wb") as f:
        pickle.dump({"model":models[selected],"decision_threshold":thresholds[selected],
                     "baseline_model":base,"baseline_threshold":thresholds["baseline"],
                     "calibration":cal,"selected_mode":selected,"partitions":PARTITIONS},f)
    (out/"architecture_manifest.json").write_text(json.dumps({
       "selected_mode":selected,"candidates":list(models),
       "thresholds_from_clean_validation":thresholds,
       "required_observables":list(MultiScaleInvariantTICD.REQUIRED),
       "features_from_gt":[],"train_seed_partition":PARTITIONS,
       "warning":"Pickle must be loaded only from trusted CI source. No real-world TICD validation."},
       indent=2,ensure_ascii=False)+"\n")
    # Verification independent of original batch composition/ordering;
    # the temporal input feature is causal and embedded in individual rows.
    reconstructed=pickle.load((out/"ticd_multiscale_model.pkl").open("rb"))
    model=reconstructed["model"]
    sample=test.sample(n=120,random_state=19)
    s1=model.score(sample);s2=model.score(test)[sample.index.to_numpy()]
    if not np.array_equal(s1,s2):raise RuntimeError("Score changes with test batch composition")
    assert reconstructed["decision_threshold"]==thresholds[selected]
    assert not any(x.startswith("gt_") for x in models[selected].REQUIRED)
    print("MULTISCALE_RETRAINING_COMPLETE",json.dumps({
       "selected":selected,"test_rows":len(test),
       "selected_tpr":proposed_results["tpr_recall"],
       "selected_fpr":proposed_results["fpr"],
       "baseline_tpr":baseline_results["tpr_recall"],
       "baseline_fpr":baseline_results["fpr"],
       "batch_invariant_scores":True},ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--calibration",required=True,type=Path)
    ap.add_argument("--out-dir",type=Path,default=Path("results_ticd_multiscale"))
    ap.add_argument("--hours",type=float,default=2.)
    run(ap.parse_args())
