#!/usr/bin/env python3
"""Loss-robust SIP TICD: normal-only conditional calibration, held-out tests.

The ORIGINAL physical TICD from an immutable earlier CI artifact is FROZEN.
Only a conditional benign-score reference is fitted; each candidate threshold
is calibrated on separate benign simulations with different network losses.

Architecture selection is based on labelled SYNTHETIC DEVELOPMENT scenarios.
Held-out seeds are not generated until AFTER selection and all thresholds have
been fixed. Four attacks/truth-report conditions and pure-benign loss shifts
are evaluated, with identical records supplied to every comparator.

This is simulation verification, NOT field validation of real compromised SIP.
"""
from __future__ import annotations
import argparse,json,pickle,sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
sys.path.insert(0,str(ROOT/"experiments"))
from ticd_loss_conditioned import LossConditionedTICD
from ticd_physical_closure import prepare_master_closure_features
from retrain_sip_ticd_calibrated import simulation,evaluate
from stress_sip_ticd_multiscale import simulate_stress

SOURCE_SHA="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
LOSS_MULTIPLIERS=(1.,3.,5.,10.)
TRAIN={1:[4101,4102],3:[4201,4202],5:[4301,4302],10:[4401,4402]}
NORMAL_VALIDATION={1:[5101,5102],3:[5201,5202],5:[5301,5302],10:[5401,5402]}
DEV={
  "default":[6101,6102],
  "weak_covert_loss5":[6201,6202],
  "truthful_compromised_reports":[6301,6302],
  "weak_covert_loss10":[6401,6402]
}
HELDOUT={
  "default":[7101,7102,7103],
  "weak_covert_loss5":[7201,7202,7203],
  "truthful_compromised_reports":[7301,7302,7303],
  "weak_covert_loss10":[7401,7402,7403],
  "clean_loss7":[7501,7502,7503],
  "clean_loss15":[7601,7602,7603],
}
COMPARATORS=("old_physical_1pc","physical_robust","old_temporal_1pc")
CANDIDATES=("loss","loss_node_shrink","loss_node")

def verify_seeds(old_package):
    all_new=[]
    for section in (TRAIN,NORMAL_VALIDATION,DEV,HELDOUT):
        all_new.extend(seed for arr in section.values() for seed in arr)
    if len(all_new)!=len(set(all_new)):
        raise ValueError("Non-disjoint model seed partitions")
    old=old_package["split_seeds"]
    old_all=set(old["train"]+old["validation"]+
        sum(old["development"].values(),[])+sum(old["heldout"].values(),[]))
    if old_all.intersection(all_new):
        raise ValueError("New seed accidentally reused source trained model seed")

def make_clean(seed,cal,multiplier):
    d,meta=simulation(seed,cal,2.,attacks_enabled=False,loss_multiplier=multiplier)
    d=prepare_master_closure_features(d)
    d["network_loss_multiplier"]=multiplier
    if d["is_anomaly"].sum()!=0:
        raise ValueError("Normal reference contaminated by synthetic attacks")
    return d

def make_attack(seed,cal,kind):
    if kind=="default":
        d,meta=simulation(seed,cal,2.,attacks_enabled=True)
        mult=1.
    elif kind=="weak_covert_loss5":
        d,meta=simulate_stress(seed,cal,"weak_covert_loss5",2.)
        mult=5.
    elif kind=="weak_covert_loss10":
        d,meta=simulate_stress(seed,cal,"weak_covert",2.,loss_multiplier=10.)
        mult=10.
    elif kind=="truthful_compromised_reports":
        d,meta=simulate_stress(seed,cal,"truthful_compromised_reports",2.)
        mult=1.
    else:
        raise ValueError("Unknown synthetic attack stress cohort")
    d=prepare_master_closure_features(d)
    d["network_loss_multiplier"]=mult
    if d["is_anomaly"].sum()==0:
        raise ValueError("No attacks generated in development/test scenario")
    return d

def make_cohort(seeds,cal,kind,loss_multiplier=None):
    frames=[]
    for seed in seeds:
        if kind=="clean":
            d=make_clean(seed,cal,loss_multiplier)
        else:
            d=make_attack(seed,cal,kind)
        d["stress_cohort"]=kind
        frames.append(d)
        print("LOSS_AWARE_SIM",json.dumps({
          "seed":seed,"cohort":kind,"loss_multiplier":float(d["network_loss_multiplier"].iloc[0]),
          "rows":len(d),"attacks":int(d.is_anomaly.sum())}),flush=True)
    return pd.concat(frames,ignore_index=True)

def metrics(model,df,threshold):
    x=evaluate(model,df,threshold)
    return {
      "rows":int(x["rows"]),"normal":int(x["normal"]),
      "attacks":int(x["attacks"]),
      "tp":int(x["tp"]),"fp":int(x["fp"]),"tn":int(x["tn"]),"fn":int(x["fn"]),
      "recall":x["tpr_recall"],"fpr":x["fpr"],"precision":x["precision"],
      "f1":x["f1"],"roc_auc":x["auc_roc"],"pr_auc":x["auc_pr"],
      "by_attack":x["by_attack"],
      "by_sophistication":x["by_compromise_sophistication"],
    }

def model_score_to_threshold(model,normal_by_loss):
    # Conservative P99 over every predeclared benign validation regime.
    # Threshold is frozen here; NO test scores or labels are consulted.
    q={}
    for multiplier,df in normal_by_loss.items():
        scores=model.score(df)
        if not np.isfinite(scores).all():
            raise RuntimeError("NaN or infinity in normal validation scores")
        q[str(multiplier)]=float(np.quantile(scores,.99,method="higher"))
    threshold=max(q.values())
    normal_val_fpr={str(mult):float((model.score(df)>threshold).mean())
                    for mult,df in normal_by_loss.items()}
    if max(normal_val_fpr.values())>0.01001:
        raise RuntimeError("A benign validation loss cohort exceeds budget")
    return threshold,q,normal_val_fpr

def per_seed_summary(model,df,threshold):
    return [dict(seed=int(seed),**metrics(model,g,threshold))
           for seed,g in df.groupby("simulation_seed",sort=True)]

def utility(per_cohort):
    # Declared PRIOR to seeing held-out test results:
    # prioritise recall to adaptive threat under truthful reporting AND
    # statistical threat when physical losses are high.
    low=per_cohort["weak_covert_loss5"]
    high=per_cohort["weak_covert_loss10"]
    truthful=per_cohort["truthful_compromised_reports"]
    default=per_cohort["default"]
    def soph(x,name):
        return float(x["by_sophistication"][name]["recall"] or 0.)
    utility=(.30*soph(low,"statistical")+
             .25*soph(high,"statistical")+
             .20*soph(truthful,"adaptive")+
             .15*soph(truthful,"statistical")+
             .10*float(default["recall"] or 0.))
    # Strict development FPR penalty, above 2% on any attacked condition.
    penalty=4.*sum(max(0.,x["fpr"]-.02) for x in per_cohort.values())
    return float(utility-penalty)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True,
                    help="directory with previously CI-trained physical detector pickle")
    ap.add_argument("--out",type=Path,default=Path("results_ticd_loss_adaptive"))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    with (args.source/"ticd_physical_closure_model.pkl").open("rb") as f:
        old=pickle.load(f)
    if old["normal_calibration"]["source_archive_sha256"]!=SOURCE_SHA:
        raise RuntimeError("Prior model isn't from pinned real SIP capture calibration")
    if old["model_name"]!="physical":
        raise RuntimeError("Unexpected prior TICD architecture")
    verify_seeds(old)
    old_physical=old["model"]
    old_temporal=old["temporal_model"]
    physical_1pc=float(old["threshold"])
    temporal_1pc=float(old["temporal_threshold"])
    calibration=old["normal_calibration"]

    # BENIGN-ONLY loss-shifted score distributions (not adaptation on attacks).
    benign_train={int(mult):make_cohort(TRAIN[int(mult)],calibration,"clean",mult)
                  for mult in LOSS_MULTIPLIERS}
    benign_val={int(mult):make_cohort(NORMAL_VALIDATION[int(mult)],calibration,"clean",mult)
                for mult in LOSS_MULTIPLIERS}
    train_all=pd.concat(list(benign_train.values()),ignore_index=True)
    assert train_all.is_anomaly.sum()==0
    assert not any(df.is_anomaly.sum() for df in benign_val.values())
    # Reuse physically meaningful master-side score without retraining its
    # temporal or invariant base components using any attack labels.
    models={
      "old_physical_1pc":old_physical,
      "physical_robust":old_physical,
      "old_temporal_1pc":old_temporal
    }
    for mode in CANDIDATES:
        models[mode]=LossConditionedTICD(old_physical,mode=mode,bins=8,min_group=120).fit(train_all)

    thresholds={
      "old_physical_1pc":physical_1pc,
      "old_temporal_1pc":temporal_1pc
    }
    candidate_thresholds={}
    normal_validation_fpr={}
    for name in ("physical_robust",)+CANDIDATES:
        threshold,qs,fprs=model_score_to_threshold(models[name],benign_val)
        thresholds[name]=threshold
        candidate_thresholds[name]={"per_loss_quantile":qs,"max_quantile":threshold}
        normal_validation_fpr[name]=fprs

    # Development labelled scenarios, independent of normal train/validation.
    dev={}
    for kind,seeds in DEV.items():
        df=make_cohort(seeds,calibration,kind)
        dev[kind]={name:metrics(model,df,thresholds[name])
                   for name,model in models.items()}
    criterion={}
    for name in ("physical_robust",)+CANDIDATES:
        criterion[name]=utility({kind:dev[kind][name] for kind in DEV})
    # Require demonstrable development utility above conservative original.
    selected=max(("physical_robust",)+CANDIDATES,key=lambda x:(criterion[x],-list(criterion).index(x)))
    print("LOSS_AWARE_ARCHITECTURE_SELECTION",json.dumps({
        "selected":selected,"criterion":criterion,
        "thresholds":thresholds,"normal_validation_fpr":normal_validation_fpr,
        "development_recall":{name:{cohort:dev[cohort][name]["recall"]
                             for cohort in DEV} for name in criterion},
    },ensure_ascii=False),flush=True)

    # Final fresh evaluation; no parameter fit or candidate selection after this.
    final={}
    per_seeds={}
    for kind,seeds in HELDOUT.items():
        if kind.startswith("clean_loss"):
            mult=int(kind.removeprefix("clean_loss"))
            df=make_cohort(seeds,calibration,"clean",float(mult))
        else:
            df=make_cohort(seeds,calibration,kind)
        final[kind]={}
        per_seeds[kind]={}
        for name in ("old_physical_1pc","physical_robust","old_temporal_1pc",selected):
            v=metrics(models[name],df,thresholds[name])
            final[kind][name]=v
            per_seeds[kind][name]=per_seed_summary(models[name],df,thresholds[name])
        print("LOSS_AWARE_HELDOUT",json.dumps({
          "cohort":kind,
          "test":{"rows":len(df),"attacks":int(df.is_anomaly.sum())},
          "comparison":{name:{"recall":v["recall"],"fpr":v["fpr"],
                              "statistical":v["by_sophistication"]["statistical"]["recall"],
                              "adaptive":v["by_sophistication"]["adaptive"]["recall"]}
                        for name,v in final[kind].items()}
        },ensure_ascii=False),flush=True)

    pkg={"selected_name":selected,"model":models[selected],"threshold":thresholds[selected],
         "original_model":old_physical,"original_threshold":physical_1pc,
         "robust_physical_threshold":thresholds["physical_robust"],
         "temporal_model":old_temporal,"temporal_threshold":temporal_1pc,
         "calibration_source_sha256":SOURCE_SHA,
         "normal_train_seeds":TRAIN,"normal_validation_seeds":NORMAL_VALIDATION,
         "development_seeds":DEV,"untouched_test_seeds":HELDOUT}
    with (args.out/"ticd_loss_adaptive_model.pkl").open("wb") as f:
        pickle.dump(pkg,f,pickle.HIGHEST_PROTOCOL)
    # Reproducible reload and batch compositional invariance
    with (args.out/"ticd_loss_adaptive_model.pkl").open("rb") as f:
        loaded=pickle.load(f)
    sample=make_cohort([7999],calibration,"weak_covert_loss10")
    all_scores=loaded["model"].score(sample)
    subset=sample.sample(150,random_state=23)
    diff=float(np.max(np.abs(loaded["model"].score(subset)-all_scores[subset.index])))
    if diff>1e-12:
        raise RuntimeError(f"Scoring depends on test batch composition: {diff}")
    audit={
      "experiment":"SIP TICD conditional normal-loss reference under independent network loss shift",
      "real_calibration_source_sha256":SOURCE_SHA,
      "real_calibration_is_only_benign_method_profile":True,
      "external_real_world_ticd_validation":False,
      "frozen_previous_detector":True,
      "new_parameters_fitted_only_on_normal_train":True,
      "thresholds_selected_only_on_normal_loss_aware_validation":True,
      "architecture_selected_on_labelled_synthetic_development_only":True,
      "untouched_test_not_used_to_fit_or_select":True,
      "normal_loss_multipliers_train":list(LOSS_MULTIPLIERS),
      "train_seeds":TRAIN,"normal_validation_seeds":NORMAL_VALIDATION,
      "development_seeds":DEV,"untouched_test_seeds":HELDOUT,
      "original_threshold":physical_1pc,
      "thresholds":thresholds,
      "reference_thresholds_by_loss":candidate_thresholds,
      "normal_validation_fpr":normal_validation_fpr,
      "model_reference":{m:models[m].reference_manifest() for m in CANDIDATES},
      "selection_utilities":criterion,"selected_model":selected,
      "development_metrics":dev,"heldout_metrics":final,
      "heldout_per_seed":per_seeds,
      "train_normal_rows":len(train_all),
      "normal_val_rows":sum(len(v) for v in benign_val.values()),
      "runtime_seconds":time.time()-started,
      "reload_success":True,"max_batch_score_difference":diff,
      "limits":"SIP simulator truth and CU polling loss covariance have not been externally validated; no real compromised proxy telemetry. Events within simulated seeds are dependent."
    }
    (args.out/"loss_aware_experiment_report.json").write_text(
        json.dumps(audit,indent=2,ensure_ascii=False)+"\n")
    print("LOSS_AWARE_FINAL",json.dumps({
      "model":selected,"reload":True,"train_rows":len(train_all),
      "validation_rows":audit["normal_val_rows"],
      "performance":{c:{
        "old_physical_fpr":v["old_physical_1pc"]["fpr"],
        "conditional_fpr":v[selected]["fpr"],
        "old_physical_recall":v["old_physical_1pc"]["recall"],
        "conditional_recall":v[selected]["recall"],
      } for c,v in final.items()},
      "batch_score_difference":diff,
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
