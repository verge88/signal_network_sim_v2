#!/usr/bin/env python3
"""Independent test of two decision tiers for frozen SIP TICD models.

Tier 2 (high confidence): frozen loss-normalized physical TICD; its
threshold is the maximum normal P99 across five separate benign validation
loss regimes. Tier 1 (early warning): causal ACK persistence only, threshold
chosen from BENIGN validation while excluding high-confidence alerts.

Choose a warning false-positive budget using a separately LABELLED SYNTHETIC
development cohort; do not use held-out labels for this choice. Finally
evaluate on new seeds with exactly one complete test pass.

Report conditional *false alarm* and *false warning* rates SEPARATELY,
along with combined suspicious-activity coverage. Warning-only detections
must NEVER be reported as confirmed TICD recall.

These metrics are SIMULATION-TO-SIMULATION, not real-network validation.
"""
from __future__ import annotations
import argparse,hashlib,json,pickle,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
sys.path.insert(0,str(ROOT/"experiments"))
from ticd_causal_ack import CausalPersistentACKTICD
from ticd_two_tier import TwoTierSIPTICD
from retrain_ticd_causal_ack import cohort

SOURCE_SHA256="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
BENIGN_VALIDATION={
    1:[13101,13102],3:[13201,13202],5:[13301,13302],
    10:[13401,13402],15:[13501,13502]
}
DEVELOPMENT={
    "default":[13601,13602],
    "weak_loss10":[13701,13702],
    "weak_loss15":[13801,13802],
    "truthful":[13901,13902]
}
UNTOUCHED_TEST={
    "default":[14101,14102,14103],
    "weak_loss5":[14201,14202,14203],
    "weak_loss10":[14301,14302,14303],
    "weak_loss15":[14401,14402,14403],
    "weak_loss20":[14501,14502,14503],
    "truthful":[14601,14602,14603],
    "clean_loss1":[14701,14702,14703],
    "clean_loss7":[14801,14802,14803],
    "clean_loss20":[14901,14902,14903]
}
WARNING_BUDGETS=(0.005,0.01,0.02,0.03)
ALL_ATTACK_LABELS=("signaling_dos","register_hijack",
                   "register_bruteforce","spit","proxy_compromise")
SOPHISTICATION=("naive","statistical","adaptive")

def check_provenance(prior):
    if prior["selected_name"]!="causal_short_w0p7":
        raise ValueError("Unexpected frozen causal model")
    cal=prior["normal_calibration"]
    if cal.get("source_archive_sha256")!=SOURCE_SHA256 or cal.get("calibration_kind")!="partial_benign_SIP_method_profile":
        raise ValueError("Calibration is not from source-pinned real benign LTE SIP")
    if not isinstance(prior["model"],CausalPersistentACKTICD):
        raise TypeError("Wrong causal detector source model")
    historic=prior["seed_partitions"]
    past={s for part in historic.values() for ls in part.values() for s in ls}
    new_groups=[BENIGN_VALIDATION,DEVELOPMENT,UNTOUCHED_TEST]
    fresh=[s for group in new_groups for ls in group.values() for s in ls]
    if len(set(fresh))!=len(fresh) or past.intersection(fresh):
        raise RuntimeError("Data-leakage: simulator seed reused between experiments")
    return cal,past

def make_group(seeds,cal,kind):
    if kind.startswith("clean_loss"):
        return cohort(seeds,cal,"clean",float(kind.removeprefix("clean_loss")))
    return cohort(seeds,cal,kind)

def threshold_from_norm(scores,quantile):
    if len(scores)==0 or not np.isfinite(scores).all():
        raise ValueError("Invalid normal validation score distribution")
    return float(np.quantile(scores,quantile,method="higher"))

def calibrate_tiers(causal_model,normal_val):
    # Stage 1: upper-confidence alarms based SOLELY on normal validation.
    benign_scores={m:causal_model.score_components(df)
                   for m,df in normal_val.items()}
    alarm_q={str(m):threshold_from_norm(s["instant"],.99)
             for m,s in benign_scores.items()}
    alarm_threshold=max(alarm_q.values())
    alarm_fpr={str(m):float(np.mean(s["instant"]>alarm_threshold))
               for m,s in benign_scores.items()}
    if max(alarm_fpr.values())>0.01001:
        raise RuntimeError("Normal validation high-confidence FPR budget exceeded")

    # Stage 2: the warning-only channel is calibrated on remaining benign
    # records, excluding already confirmed alerts. This ensures a warning
    # has a separate explicit false-positive budget.
    warning_thresholds={}
    warning_reference={}
    for budget in WARNING_BUDGETS:
        w_q={}
        for m,s in benign_scores.items():
            nonalert=s["instant"]<=alarm_threshold
            w_q[str(m)]=threshold_from_norm(s["causal"][nonalert],1.-budget)
        warning_t=max(w_q.values())
        fwr={}
        combined={}
        for m,s in benign_scores.items():
            alarm=s["instant"]>alarm_threshold
            warn=(~alarm)&(s["causal"]>warning_t)
            fwr[str(m)]=float(np.mean(warn))
            combined[str(m)]=float(np.mean(alarm|warn))
        if max(fwr.values())>budget+1e-4:
            raise RuntimeError("Warning normal validation FWR exceeds budget")
        warning_thresholds[budget]=warning_t
        warning_reference[str(budget)]={
          "conditional_p99_by_loss":w_q,
          "warning_fpr_by_loss":fwr,
          "combined_fpr_by_loss":combined
        }
    return alarm_threshold,alarm_q,alarm_fpr,warning_thresholds,warning_reference

def frac(a,b):
    return float(a/b) if b else None

def episode_count(df,signal_mask):
    """A new episode starts after a gap, change of node, or prior no-signal.

    An episode consists only of falsely flagged NORMAL rows; attack rows
    reset false episodes even if the detector remains active.
    """
    idx=df[["simulation_seed","node_id","interval"]].copy()
    idx["signal"]=np.asarray(signal_mask,dtype=bool)
    idx=idx.sort_values(["simulation_seed","node_id","interval"],kind="stable")
    grouped=idx.groupby(["simulation_seed","node_id"],sort=False)
    prior=grouped["signal"].shift(1,fill_value=False)
    prior_time=grouped["interval"].shift(1)
    starts=idx["signal"] & ((~prior) | ((idx["interval"]-prior_time)!=1))
    return int(starts.sum())

def metrics(model,df):
    y=(df["is_anomaly"].to_numpy(dtype=int)>0)
    masks=model.predict_masks(df)
    a=masks["confirmed_alert"]
    w=masks["warning_only"]
    u=masks["any_signal"]
    if not np.all((a&w)==False):
        raise RuntimeError("Confirmed and warning event labels overlap")
    normal=~y
    na=int(normal.sum())
    pa=int(y.sum())
    confirmed_fp=int(np.sum(a&normal))
    warning_fp=int(np.sum(w&normal))
    result={
      "rows":len(df),"normal":na,"attack":pa,
      "confirmed_TP":int(np.sum(a&y)),
      "confirmed_FP":confirmed_fp,
      "warning_only_TP":int(np.sum(w&y)),
      "warning_only_FP":warning_fp,
      "covered_attack_rows":int(np.sum(u&y)),
      "confirmed_alert_FPR":frac(confirmed_fp,na),
      "false_warning_rate":frac(warning_fp,na),
      "combined_normal_signal_rate":frac(int(np.sum(u&normal)),na),
      "confirmed_recall":frac(int(np.sum(a&y)),pa),
      "warning_only_incremental_attack_coverage":frac(int(np.sum(w&y)),pa),
      "combined_attack_coverage":frac(int(np.sum(u&y)),pa),
      "confirmed_precision":frac(int(np.sum(a&y)),int(a.sum())),
      "warning_precision":frac(int(np.sum(w&y)),int(w.sum())),
      "false_alert_episodes":episode_count(df,a&normal),
      "false_warning_episodes":episode_count(df,w&normal),
      "by_attack":{},"by_sophistication":{}
    }
    def category(mask):
        n=int(mask.sum())
        return {
          "n":n,
          "confirmed_recall":frac(int(np.sum(a&mask)),n),
          "warning_only_incremental":frac(int(np.sum(w&mask)),n),
          "any_signal_coverage":frac(int(np.sum(u&mask)),n),
        }
    for label in ALL_ATTACK_LABELS:
        mask=y&(df["attack_type"].astype(str).to_numpy()==label)
        result["by_attack"][label]=category(mask)
    for label in SOPHISTICATION:
        mask=(y&(df["attack_type"].astype(str).to_numpy()=="proxy_compromise")&
              (df["attacker_sophistication"].astype(str).to_numpy()==label))
        result["by_sophistication"][label]=category(mask)
    return result

def utility(development,allowed_warning_budget):
    """Predeclared synthetic development selection, NOT heldout optimisation.

    Encourage incremental warning coverage for statistical compromise at
    high loss, penalise warnings on non-attacking rows. Hard 3%-FWR dev cap.
    """
    def inc(kind,level):
        return development[kind]["by_sophistication"][level]["warning_only_incremental"] or 0.
    value=(
       .40*inc("weak_loss10","statistical")+
       .35*inc("weak_loss15","statistical")+
       .15*inc("truthful","adaptive")+
       .10*(development["default"]["warning_only_incremental_attack_coverage"] or 0.))
    fwr=np.array([v["false_warning_rate"] or 0. for v in development.values()])
    return float(value-2.0*float(fwr.mean())-10.0*sum(max(0.,x-.03) for x in fwr)
                 -0.1*allowed_warning_budget)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--frozen",type=Path,required=True,
                    help="directory containing previously normal-trained causal TICD")
    ap.add_argument("--out",type=Path,default=Path("results_ticd_two_tier"))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    with (args.frozen/"ticd_causal_ack_model.pkl").open("rb") as f:
        prior=pickle.load(f)
    calibration,old_seeds=check_provenance(prior)
    causal=prior["model"]

    # No training of a previously observed detector. Only normal threshold
    # calibration on a fresh disjoint set of benign simulations.
    normals={loss:make_group(seeds,calibration,"clean_loss"+str(loss))
             for loss,seeds in BENIGN_VALIDATION.items()}
    for df in normals.values():
        if df["is_anomaly"].sum():
            raise RuntimeError("Contamination of normal validation")
    alert_t,alert_q,alert_fprs,warn_ts,warn_info=calibrate_tiers(causal,normals)
    candidates={str(b):TwoTierSIPTICD(causal,alert_t,warn_ts[b])
                for b in WARNING_BUDGETS}
    print("TWO_TIER_NORMAL_THRESHOLDS",json.dumps({
      "alert_threshold":alert_t,"alert_validation_FPR":alert_fprs,
      "warning_thresholds":warn_ts,
      "warning_validation_reference":warn_info
    },ensure_ascii=False),flush=True)

    development={}
    for label,seeds in DEVELOPMENT.items():
        x=make_group(seeds,calibration,label)
        development[label]={k:metrics(model,x) for k,model in candidates.items()}
        print("TWO_TIER_DEV",json.dumps({
          "cohort":label,"rows":len(x),
          "candidate_warning_FWR":{k:v["false_warning_rate"]
                                   for k,v in development[label].items()}
        },ensure_ascii=False),flush=True)
    scores={str(b):utility({k:development[k][str(b)] for k in DEVELOPMENT},b)
            for b in WARNING_BUDGETS}
    selected=max(scores,key=lambda k:(scores[k],-WARNING_BUDGETS.index(float(k))))
    chosen=candidates[selected]
    # Freeze all operating-point parameters BEFORE generating heldout test.
    print("TWO_TIER_SELECTED_ON_DEV",json.dumps({
      "selected_warning_budget":selected,
      "thresholds":chosen.manifest(),
      "development_utility":scores
    },ensure_ascii=False),flush=True)

    # Comparators: prior detector's original threshold and conservatively
    # recalibrated confirmation-only tier. Warnings are never counted as alert.
    initial_alarm_t=float(prior["baseline_threshold"])
    old_alarm_model=TwoTierSIPTICD(causal,initial_alarm_t,float("1e12"))
    conservative_alarm_model=TwoTierSIPTICD(causal,alert_t,float("1e12"))
    models={"prior_alarm":old_alarm_model,
            "conservative_alarm":conservative_alarm_model,
            "two_tier":chosen}
    final={};per_seed={}
    for label,seeds in UNTOUCHED_TEST.items():
        df=make_group(seeds,calibration,label)
        final[label]={n:metrics(mod,df) for n,mod in models.items()}
        per_seed[label]={
           name:[{"seed":int(seed),**metrics(mod,frame)}
                 for seed,frame in df.groupby("simulation_seed")]
           for name,mod in models.items()
        }
        print("TWO_TIER_HELDOUT",json.dumps({
          "cohort":label,"rows":len(df),
          "prior":{"alarm_FPR":final[label]["prior_alarm"]["confirmed_alert_FPR"],
                   "alarm_recall":final[label]["prior_alarm"]["confirmed_recall"]},
          "conservative":{"alarm_FPR":final[label]["conservative_alarm"]["confirmed_alert_FPR"],
                          "alarm_recall":final[label]["conservative_alarm"]["confirmed_recall"]},
          "two_tier":{
            "alarm_FPR":final[label]["two_tier"]["confirmed_alert_FPR"],
            "warning_FWR":final[label]["two_tier"]["false_warning_rate"],
            "combined_signal_FPR":final[label]["two_tier"]["combined_normal_signal_rate"],
            "alarm_recall":final[label]["two_tier"]["confirmed_recall"],
            "warning_incremental_coverage":final[label]["two_tier"]["warning_only_incremental_attack_coverage"],
            "combined_attack_coverage":final[label]["two_tier"]["combined_attack_coverage"],
            "statistical":final[label]["two_tier"]["by_sophistication"]["statistical"],
          }
        },ensure_ascii=False),flush=True)

    pack={
      "model":chosen,
      "prior_alarm_threshold":initial_alarm_t,
      "conservative_alarm_threshold":alert_t,
      "warning_threshold":chosen.warning_threshold,
      "warning_reference_budget":float(selected),
      "calibration":calibration,
      "normal_validation_seeds":BENIGN_VALIDATION,
      "development_seeds":DEVELOPMENT,
      "heldout_seeds":UNTOUCHED_TEST
    }
    with (args.out/"ticd_two_tier_model.pkl").open("wb") as f:
        pickle.dump(pack,f,pickle.HIGHEST_PROTOCOL)
    with (args.out/"ticd_two_tier_model.pkl").open("rb") as f:
        loaded=pickle.load(f)

    # Recorded complete observations include past-only per-node ACK history;
    # subsampling must NOT change scores and warning/alert decisions.
    independent=make_group([15901],calibration,"weak_loss10")
    full=loaded["model"].predict_levels(independent)
    excerpt=independent.sample(170,random_state=33)
    abbreviated=loaded["model"].predict_levels(excerpt)
    if not np.array_equal(abbreviated,full[excerpt.index.to_numpy()]):
        raise RuntimeError("Tier labels depend on test packet composition")
    if not np.array_equal(loaded["model"].predict_levels(independent),
                          loaded["model"].predict_levels(independent)):
        raise RuntimeError("Non-deterministic decisions")

    output={
      "scope":"Two-level SIP TICD: fixed confirmed alarms and separately budgeted causal-only warnings",
      "evidence":"Simulation only; real LTE SIP used for benign method-profile partial calibration",
      "real_benign_profile_SHA256":SOURCE_SHA256,
      "real_world_SIP_detector_validation":False,
      "frozen_prior_detector":True,
      "all_thresholds_from_new_normal_validation_only":True,
      "warning_budget_selected_only_on_labelled_synthetic_development":True,
      "heldout_not_used_for_any_selection":True,
      "historical_seeds_checked_no_overlap":True,
      "normal_validation_seeds":BENIGN_VALIDATION,
      "development_seeds":DEVELOPMENT,
      "heldout_seeds":UNTOUCHED_TEST,
      "normal_validation_rows":sum(len(x) for x in normals.values()),
      "candidate_warning_FPR_budgets":list(WARNING_BUDGETS),
      "original_frozen_alarm_threshold":initial_alarm_t,
      "new_confirmed_alarm_threshold":alert_t,
      "normal_alarm_FPR_by_loss":alert_fprs,
      "normal_alarm_P99_by_loss":alert_q,
      "normal_warning_reference":warn_info,
      "warning_thresholds":{str(k):v for k,v in warn_ts.items()},
      "selected_warning_budget":float(selected),
      "selected_model_manifest":chosen.manifest(),
      "development_utilities":scores,
      "development_metrics":development,
      "untouched_test_metrics":final,
      "per_test_seed_metrics":per_seed,
      "reloaded_model_verified":True,
      "test_batch_label_invariance":True,
      "runtime_seconds":time.time()-started,
      "no_real_compromise_ground_truth":True,
      "notes":"Warnings are not confirmed alarms. Combined signal coverage has a different and larger false-positive denominator; report alarm FPR, warning-only FWR, and combined-signal rate separately. Simulated seed is the analysis cluster."
    }
    (args.out/"two_tier_ticd_report.json").write_text(
        json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    (args.out/"two_tier_model_manifest.json").write_text(
        json.dumps({"model_class":"TwoTierSIPTICD",**chosen.manifest(),
                    "selected_warning_fpr_budget":float(selected),
                    "real_capture_sha256":SOURCE_SHA256,
                    "pickle_security":"Load only trusted GitHub Actions artifacts"},
                   ensure_ascii=False,indent=2)+"\n")
    print("TWO_TIER_FINAL",json.dumps({
       "selected_warning_budget":selected,
       "alarm_threshold":alert_t,
       "warning_threshold":chosen.warning_threshold,
       "test":{k:{
          "confirmed_recall":v["two_tier"]["confirmed_recall"],
          "warning_incremental":v["two_tier"]["warning_only_incremental_attack_coverage"],
          "combined_coverage":v["two_tier"]["combined_attack_coverage"],
          "confirmed_FPR":v["two_tier"]["confirmed_alert_FPR"],
          "warning_only_FWR":v["two_tier"]["false_warning_rate"],
          "total_FPR":v["two_tier"]["combined_normal_signal_rate"],
          "statistical":v["two_tier"]["by_sophistication"]["statistical"]
         } for k,v in final.items()},
       "model_reload":True,"batch_stability":True
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
