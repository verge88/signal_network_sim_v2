#!/usr/bin/env python3
"""Experiment: zone-contextual warning-only channel with frozen SIP TICD.

The level-2 alarm model is unchanged. All candidate warning transformations
observe same-time master-side causal score medians and robust reported zone
SIP load medians. Reported loads MAY be attacker controlled; no claim of
Byzantine robustness or operational field validation is made.

Separate simulator seed partitions:
  * train: benign-only estimate of nominal zone load
  * validation: benign-only threshold for both alert and warning
  * development: labelled SYNTHETIC attacks to choose warning context variant
  * test: completely new seeds, generated after decisions are frozen

Never refit/adapt scores or thresholds to test batches or test labels.
"""
from __future__ import annotations
import argparse,json,pickle,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
sys.path.insert(0,str(ROOT/"experiments"))
from ticd_two_tier import TwoTierSIPTICD
from ticd_zonal_warnings import ZoneContextTwoTierTICD,prepare_zonal_warning_context
from retrain_ticd_causal_ack import cohort
from evaluate_ticd_two_tier import metrics as tier_metrics

CAPTURE_SHA="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
TRAIN={1:[21101,21102],3:[21201,21202],5:[21301,21302],
       10:[21401,21402],15:[21501,21502]}
VALIDATION={1:[22101,22102],3:[22201,22202],5:[22301,22302],
            10:[22401,22402],15:[22501,22502]}
DEVELOPMENT={
  "default":[23101,23102],
  "weak_loss10":[23201,23202],
  "weak_loss15":[23301,23302],
  "weak_loss20":[23401,23402],
  "truthful":[23501,23502]
}
TEST={
  "default":[24101,24102,24103],
  "weak_loss5":[24201,24202,24203],
  "weak_loss10":[24301,24302,24303],
  "weak_loss15":[24401,24402,24403],
  "weak_loss20":[24501,24502,24503],
  "truthful":[24601,24602,24603],
  "clean_loss1":[24701,24702,24703],
  "clean_loss7":[24801,24802,24803],
  "clean_loss20":[24901,24902,24903],
}
SPECS=(
  ("baseline","baseline",0.0),
  ("zone_median_0p5","peer",0.5),
  ("zone_median_1p0","peer",1.0),
  ("zone_load_0p5","load",0.5),
  ("zone_load_1p0","load",1.0),
  ("zone_joint_0p5","joint",0.5),
  ("zone_joint_1p0","joint",1.0),
)
WARNING_FWR_BUDGET=0.005

def check_no_seed_reuse(previous):
    old={}
    for section in ("normal_validation_seeds","development_seeds","heldout_seeds"):
        old[section]=previous[section]
    old_set={s for d in old.values() for xs in d.values() for s in xs}
    new_groups=[TRAIN,VALIDATION,DEVELOPMENT,TEST]
    new=[s for group in new_groups for values in group.values() for s in values]
    if len(new)!=len(set(new)) or old_set.intersection(new):
        raise RuntimeError("Overlapping simulation seeds")
    return {"previous_new_overlap":False,"new_partitions_disjoint":True,
            "previous_seed_count":len(old_set),"fresh_seed_count":len(new)}

def load_model(path):
    with path.open("rb") as f:old=pickle.load(f)
    if "model" not in old or not isinstance(old["model"],TwoTierSIPTICD):
        raise TypeError("Expected previously frozen two-level TICD artifact")
    if old["calibration"]["source_archive_sha256"]!=CAPTURE_SHA:
        raise RuntimeError("Unverified real SIP method-profile calibration")
    if old["calibration"]["calibration_kind"]!="partial_benign_SIP_method_profile":
        raise RuntimeError("Expected partial benign-only SIP PCAP calibration")
    return old

def produce(seeds,cal,kind,causal):
    if kind.startswith("clean_loss"):
        df=cohort(seeds,cal,"clean",float(kind.removeprefix("clean_loss")))
    else:
        df=cohort(seeds,cal,kind)
    source=prepare_zonal_warning_context(df,causal)
    if len(source)!=len(df):
        raise RuntimeError("Zone reference changed the number of records")
    if source["zone_member_count"].max()<2:
        raise RuntimeError("Zone comparator has no comparable peers")
    if kind.startswith("clean_loss") and int(df.is_anomaly.sum()):
        raise RuntimeError("Contaminated clean simulation")
    print("ZONAL_WARNING_COHORT",json.dumps({
      "kind":kind,"seeds":list(seeds),"rows":len(source),
      "attacks":int(source.is_anomaly.sum()),
      "median_zone_members":float(source.zone_member_count.median())
    },ensure_ascii=False),flush=True)
    return source

def reference_threshold(scores,upper_quantile):
    if not len(scores) or not np.isfinite(scores).all():
        raise RuntimeError("Bad normal reference scores")
    return float(np.quantile(scores,upper_quantile,method="higher"))

def fit_normal_thresholds(models,validation):
    # Confirmation is the SAME score and SAME threshold for ALL modes.
    primary=models["baseline"]
    val_parts={m:primary.score_channels(df) for m,df in validation.items()}
    alert_by_loss={str(m):reference_threshold(parts["instant"],.99)
                   for m,parts in val_parts.items()}
    alarm_threshold=max(alert_by_loss.values())
    alarm_fpr={str(m):float(np.mean(p["instant"]>alarm_threshold))
               for m,p in val_parts.items()}
    if max(alarm_fpr.values())>.01001:
        raise RuntimeError("Alert reference FPR exceeds 1% budget")
    thresholds={};refs={}
    for name,model in models.items():
        per_loss={}
        for m,df in validation.items():
            s=model.score_channels(df)
            available=s["instant"]<=alarm_threshold
            if available.sum()<100:
                raise RuntimeError("Too few normal non-alert reference records")
            per_loss[str(m)]=reference_threshold(s["warning"][available],
                                                 1.-WARNING_FWR_BUDGET)
        threshold=max(per_loss.values())
        model.set_thresholds(alarm_threshold,threshold)
        fwr={str(m):tier_metrics(model,df)["false_warning_rate"]
             for m,df in validation.items()}
        if max(fwr.values())>WARNING_FWR_BUDGET+1e-4:
            raise RuntimeError("Warning reference budget not achieved: "+name)
        refs[name]={"threshold":threshold,
                    "validation_warning_fwr":fwr,"p995_by_loss":per_loss}
        thresholds[name]=threshold
    return alarm_threshold,alert_by_loss,alarm_fpr,thresholds,refs

def selection_utility(dev):
    """Fixed before running holdout; no tuning from test results.

    Incremental statistical coverage matters; penalize failed transfer of
    benign warning FWR to attack-containing normal records, especially the
    default scenario where the previous two-tier model had high FWR.
    """
    def incr(kind):
        return float(dev[kind]["by_sophistication"]["statistical"]["warning_only_incremental"] or 0.)
    val=(.25*incr("weak_loss10")+
         .25*incr("weak_loss15")+
         .20*incr("weak_loss20")+
         .10*float(dev["truthful"]["warning_only_incremental_attack_coverage"] or 0.)+
         .20*float(dev["default"]["warning_only_incremental_attack_coverage"] or 0.))
    fwr=[float(v["false_warning_rate"] or 0.) for v in dev.values()]
    default_fwr=float(dev["default"]["false_warning_rate"] or 0.)
    # Explicit penalty on the already disclosed FWR transport failure.
    penalty=18.*max(default_fwr-.005,0.)+            8.*sum(max(x-.01,0.) for x in fwr)+            3.*max(float(dev["default"]["combined_normal_signal_rate"])-.015,0.)
    return float(val-penalty)

def diagnostic_false_warning_context(df,mod):
    """EVALUATION ONLY: uses ground truth to stratify errors, not score."""
    levels=mod.predict_levels(df)
    normal=df["is_anomaly"].to_numpy(dtype=int)==0
    warned=(levels==1)&normal
    if not normal.any():return {}
    summary={}
    for event,g in df.loc[normal].groupby("normal_event",sort=False):
        idx=g.index
        # Dataframe produced by cohort has zero-to-N-1 RangeIndex.
        summary[str(event)]={
          "normal_rows":len(g),
          "false_warning_rows":int(warned[idx.to_numpy()].sum()),
        }
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prior",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results_ticd_zone_context"))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    old=load_model(a.prior/"ticd_two_tier_model.pkl")
    seeds=check_no_seed_reuse(old)
    cal=old["calibration"]
    causal=old["model"].causal_model
    assert causal.fitted

    # Fit only benign zone load reference; no new TICD weights fitted.
    train=pd.concat([produce(vals,cal,"clean_loss"+str(loss),causal)
                     for loss,vals in TRAIN.items()],ignore_index=True)
    if int(train["is_anomaly"].sum())!=0:raise RuntimeError("Non-normal zone reference")
    models={name:ZoneContextTwoTierTICD(causal,mode,strength).fit(train)
            for name,mode,strength in SPECS}
    print("ZONAL_WARNING_TRAIN_REFERENCE",json.dumps({
      "rows":len(train),
      "zone_reported_load_P50":models["baseline"].reference_load,
      "model_variants":list(models)
    },ensure_ascii=False),flush=True)

    normal={m:produce(vals,cal,"clean_loss"+str(m),causal)
            for m,vals in VALIDATION.items()}
    alert_t,alert_q,alert_fpr,warn_ts,warn_ref=fit_normal_thresholds(models,normal)
    print("ZONAL_WARNING_THRESHOLDS",json.dumps({
      "alert_threshold":alert_t,"validation_alert_fpr":alert_fpr,
      "warning_thresholds":warn_ts,
      "validation_warning_fwr":{k:v["validation_warning_fwr"] for k,v in warn_ref.items()}
    },ensure_ascii=False),flush=True)

    development={}
    for kind,vals in DEVELOPMENT.items():
        df=produce(vals,cal,kind,causal)
        development[kind]={name:tier_metrics(mod,df) for name,mod in models.items()}
    utility={name:selection_utility({k:development[k][name] for k in DEVELOPMENT})
             for name in models}
    # The baseline is among candidates; if context doesn't help on development,
    # honestly retain the original warning branch.
    chosen=max(models,key=lambda name:(utility[name],-list(models).index(name)))
    selected=models[chosen]
    print("ZONAL_WARNING_SELECTED_ON_DEV",json.dumps({
      "selected":chosen,"utilities":utility,
      "developer_fwr":{name:development["default"][name]["false_warning_rate"]
                       for name in models},
      "selected_manifest":selected.manifest()
    },ensure_ascii=False),flush=True)

    # FINAL TEST is generated AFTER candidate and both thresholds are frozen.
    baseline=models["baseline"]
    final={};per_seed={};diagnostics={}
    for kind,vals in TEST.items():
        df=produce(vals,cal,kind,causal)
        final[kind]={
          "baseline":tier_metrics(baseline,df),
          "zonal_selected":tier_metrics(selected,df)}
        per_seed[kind]={name:[
            {"seed":int(seed),**tier_metrics(mod,part)}
            for seed,part in df.groupby("simulation_seed",sort=True)]
            for name,mod in (("baseline",baseline),("zonal_selected",selected))}
        if kind=="default":
            # These labels are used ONLY for diagnostics after test.
            diagnostics["default_benign_warning_events"]={
              "baseline":diagnostic_false_warning_context(df,baseline),
              "selected":diagnostic_false_warning_context(df,selected)}
        print("ZONAL_WARNING_HELDOUT",json.dumps({
          "cohort":kind,
          "baseline":{
            "alarm_fpr":final[kind]["baseline"]["confirmed_alert_FPR"],
            "false_warning_rate":final[kind]["baseline"]["false_warning_rate"],
            "confirmed_recall":final[kind]["baseline"]["confirmed_recall"],
            "incremental_warning_coverage":final[kind]["baseline"]["warning_only_incremental_attack_coverage"],
            "statistical":final[kind]["baseline"]["by_sophistication"]["statistical"]},
          "selected":{
            "alarm_fpr":final[kind]["zonal_selected"]["confirmed_alert_FPR"],
            "false_warning_rate":final[kind]["zonal_selected"]["false_warning_rate"],
            "confirmed_recall":final[kind]["zonal_selected"]["confirmed_recall"],
            "incremental_warning_coverage":final[kind]["zonal_selected"]["warning_only_incremental_attack_coverage"],
            "statistical":final[kind]["zonal_selected"]["by_sophistication"]["statistical"]}},
           ensure_ascii=False),flush=True)

    # The *complete* simultaneous zone snapshot is prepared BEFORE
    # scoring; scoring a shuffled subset must never redo medians.
    audit=produce([25901],cal,"weak_loss10",causal)
    all_score=selected.score_channels(audit)
    sampled=audit.sample(240,random_state=23)
    part_score=selected.score_channels(sampled)
    for key in ("instant","warning"):
        delta=float(np.max(np.abs(part_score[key]-
               all_score[key][sampled.index.to_numpy()])))
        if delta>1e-12:raise RuntimeError("Batch-dependence "+key)
    if not np.array_equal(selected.predict_levels(sampled),
                          selected.predict_levels(audit)[sampled.index.to_numpy()]):
        raise RuntimeError("Subset changed alert/warning class")
    # Explicit adversarial oracle/permutation test; no access to any
    # synthetic true node state, attack flag or scenario label while scoring.
    oracle_cols=[c for c in audit.columns if c.startswith("gt_") or c in (
        "is_anomaly","attack_type","normal_event","is_compromised",
        "attacker_sophistication","network_loss_multiplier")]
    stripped=audit.drop(columns=oracle_cols)
    reconstructed=prepare_zonal_warning_context(stripped,causal)
    original=selected.predict_levels(audit)
    no_oracle=selected.predict_levels(reconstructed)
    if not np.array_equal(original,no_oracle):
        raise RuntimeError("Test/simulator oracle leakage into detector output")
    if len(set(oracle_cols))<5:
        raise RuntimeError("Expected synthetic ground-truth label columns absent")

    pkg={
      "model":selected,"baseline_model":baseline,"selected_variant":chosen,
      "normal_training_seeds":TRAIN,"normal_validation_seeds":VALIDATION,
      "development_seeds":DEVELOPMENT,"heldout_seeds":TEST,
      "capture_calibration":cal,
      "expected_scoring_preprocessor":"prepare_zonal_warning_context",
      "thresholds_selected_from_clean_normal_only":True
    }
    with (a.out/"ticd_zone_context_model.pkl").open("wb") as f:
        pickle.dump(pkg,f,pickle.HIGHEST_PROTOCOL)
    with (a.out/"ticd_zone_context_model.pkl").open("rb") as f:
        restored=pickle.load(f)
    if not np.array_equal(restored["model"].predict_levels(sampled),
                          selected.predict_levels(sampled)):
        raise RuntimeError("Persisted model reloading changed decisions")

    report={
      "type":"TICD zone-contextual early warnings, simulation-only evaluation",
      "real_capture_sha256":CAPTURE_SHA,
      "real_capture_calibration_scope":"benign INVITE/ACK/BYE conditional shares only",
      "full_real_attack_validation":False,
      "master_raw_ack_invariants_frozen":True,
      "zone_master_CU_context":True,
      "zone_load_is_median_of_potentially_untrusted_node_reports":True,
      "no_attack_oracle_access_in_detector":True,
      "all_thresholds_from_independent_benign_validation_only":True,
      "selection_from_separate_labelled_synthetic_development_only":True,
      "heldout_not_used_for_selection":True,
      "warning_not_equated_to_confirmed_alert":True,
      "seed_provenance":seeds,
      "train_seeds":TRAIN,"validation_seeds":VALIDATION,
      "development_seeds":DEVELOPMENT,"heldout_seeds":TEST,
      "benign_training_rows":len(train),
      "benign_validation_rows":sum(map(len,normal.values())),
      "alert_threshold":alert_t,"alert_reference_P99_by_loss":alert_q,
      "alert_reference_FPR_by_loss":alert_fpr,
      "warning_budget_reference":WARNING_FWR_BUDGET,
      "warning_thresholds":warn_ts,
      "warning_reference":warn_ref,
      "dev_selection_utility":utility,"selected_variant":chosen,
      "selected_manifest":selected.manifest(),
      "development_results":development,
      "heldout_results":final,"heldout_per_seed":per_seed,
      "posthoc_only_label_diagnostics":diagnostics,
      "model_reload_verified":True,
      "no_batch_composition_leak":True,
      "no_oracle_feature_leak":True,
      "elapsed_seconds":time.time()-started,
      "scientific_limits":"Zone reported SIP load may be compromised; a zone majority attack can poison robust medians. Real multi-point SIP traffic and trusted CU telemetry with attack ground truth are unavailable."
    }
    (a.out/"zone_context_ticd_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("ZONAL_WARNING_FINAL",json.dumps({
      "selected":chosen,"alert_threshold":alert_t,
      "warning_threshold":selected.warning_threshold,
      "results":{kind:{
        "base_FWR":r["baseline"]["false_warning_rate"],
        "new_FWR":r["zonal_selected"]["false_warning_rate"],
        "base_alert_FPR":r["baseline"]["confirmed_alert_FPR"],
        "new_alert_FPR":r["zonal_selected"]["confirmed_alert_FPR"],
        "base_stat_alert":r["baseline"]["by_sophistication"]["statistical"]["confirmed_recall"],
        "new_stat_alert":r["zonal_selected"]["by_sophistication"]["statistical"]["confirmed_recall"],
        "base_stat_warning":r["baseline"]["by_sophistication"]["statistical"]["warning_only_incremental"],
        "new_stat_warning":r["zonal_selected"]["by_sophistication"]["statistical"]["warning_only_incremental"]}
        for kind,r in final.items()},
      "model_reload":True,"oracle_test":True,"batch_invariance":True},
      ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
