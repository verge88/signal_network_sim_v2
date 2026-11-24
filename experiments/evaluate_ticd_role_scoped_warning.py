#!/usr/bin/env python3
"""Ablation and untouched-seed test of role-scoped SIP TICD early warnings.

The role inventory is known to the operator. It is NOT an attack/compromise
label. High-confidence physical alerts remain identical for ALL SIP roles.

The warning-only channel is optionally scoped to PROXY, PROXY+REGISTRAR, or
all network functions except UA_GW, alongside the previous unrestricted
warning channel. The source model (trained PhysicalClosure/LossConditioned/
causal ACK + optional zone-load reference) is wholly frozen. Only the two
thresholds are independently calibrated using NEW normal validation seeds.
The role policy is selected only on NEW LABELLED SYNTHETIC development
scenarios. All final test seeds are unseen and disjoint.

Previously disclosed run 45345156303/241xx is used only to formulate the
hypothesis of role-related false warnings. It is NEVER used for selection
or for the final outcome metrics below.

There is no real operator SIP-compromise ground truth in this experiment.
"""
from __future__ import annotations
import argparse,json,pickle,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
sys.path.insert(0,str(ROOT/"experiments"))
from ticd_role_scoped_warning import RoleScopedWarningTICD
from evaluate_ticd_zonal_warnings import produce
from evaluate_ticd_two_tier import metrics as tier_metrics
from ticd_zonal_warnings import prepare_zonal_warning_context

SOURCE_SHA="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
NORMAL_VALIDATION={
  1:[31101,31102],3:[31201,31202],5:[31301,31302],
  10:[31401,31402],15:[31501,31502],
}
DEVELOPMENT={
  "default":[32101,32102],
  "weak_loss5":[32201,32202],
  "weak_loss10":[32301,32302],
  "weak_loss15":[32401,32402],
  "weak_loss20":[32501,32502],
  "truthful":[32601,32602],
}
UNTOUCHED_TEST={
  "default":[33101,33102,33103],
  "weak_loss5":[33201,33202,33203],
  "weak_loss10":[33301,33302,33303],
  "weak_loss15":[33401,33402,33403],
  "weak_loss20":[33501,33502,33503],
  "truthful":[33601,33602,33603],
  "clean_loss1":[33701,33702,33703],
  "clean_loss7":[33801,33802,33803],
  "clean_loss20":[33901,33902,33903],
}
SPECS={
  "all_raw":("baseline",None),
  "all_zone_load":("selected",None),
  "no_gateway_raw":("baseline",("PROXY","REDIRECT","REGISTRAR")),
  "no_gateway_zone_load":("selected",("PROXY","REDIRECT","REGISTRAR")),
  "proxy_only_raw":("baseline",("PROXY",)),
  "proxy_only_zone_load":("selected",("PROXY",)),
  "proxy_and_registrar_raw":("baseline",("PROXY","REGISTRAR")),
  "proxy_and_registrar_zone_load":("selected",("PROXY","REGISTRAR")),
}
REFERENCE_WARNING_FWR=0.005

def check_prior(prior):
    if prior["selected_variant"]!="zone_load_0p5":
        raise ValueError("This test requires pinned zone-load model")
    if prior["capture_calibration"]["source_archive_sha256"]!=SOURCE_SHA:
        raise ValueError("Incorrect real benign external SIP calibration")
    if prior["capture_calibration"]["calibration_kind"]!="partial_benign_SIP_method_profile":
        raise ValueError("Only expected benign INVITE/ACK/BYE external calibration")
    old_seeds={s for key in (
        "normal_training_seeds","normal_validation_seeds",
        "development_seeds","heldout_seeds") for ls in prior[key].values() for s in ls}
    all_new=[s for d in (NORMAL_VALIDATION,DEVELOPMENT,UNTOUCHED_TEST)
               for arr in d.values() for s in arr]
    if len(set(all_new))!=len(all_new) or set(all_new)&old_seeds:
        raise RuntimeError("Previous and new simulation partitions overlap")
    if any(s<=25901 for s in all_new):
        raise RuntimeError("Potential old historical seed reused")
    return len(old_seeds),len(all_new)

def fquantile(values,p=.995):
    if len(values)<50 or not np.isfinite(values).all():
        raise ValueError("Invalid normal calibration subgroup")
    return float(np.quantile(values,p,method="higher"))

def calibrate(models,val):
    # All roles share an identical alert threshold because strong-alert
    # evidence is unchanged for every candidate architecture.
    check=models["all_raw"]
    al_q={str(m):fquantile(check.score_channels(frame)["instant"],.99)
          for m,frame in val.items()}
    alert_threshold=max(al_q.values())
    alarm_fpr={str(m):float(np.mean(check.score_channels(frame)["instant"]>alert_threshold))
               for m,frame in val.items()}
    if max(alarm_fpr.values())>.01001:
        raise RuntimeError("Confirmed-alert reference FPR >1% in normal validation")
    details={}
    for name,model in models.items():
        qs={}
        for loss,frame in val.items():
            s=model.score_channels(frame)
            eligible=model.eligible_mask(frame)
            active=eligible&(s["instant"]<=alert_threshold)
            qs[str(loss)]=fquantile(s["warning"][active],1.-REFERENCE_WARNING_FWR)
        threshold=max(qs.values())
        model.freeze_thresholds(alert_threshold,threshold)
        fwr={str(loss):tier_metrics(model,frame)["false_warning_rate"]
             for loss,frame in val.items()}
        assert max(fwr.values())<=REFERENCE_WARNING_FWR+1e-4
        details[name]={"warning_threshold":threshold,
                       "normal_P995_by_loss":qs,"normal_warning_FWR_by_loss":fwr}
    return alert_threshold,alarm_fpr,details

def scalar(m,attack_scenario):
    val=m["by_sophistication"]["statistical"]["warning_only_incremental"]
    if val is None and attack_scenario:raise ValueError("Statistical attacks absent")
    return float(val or 0.0)

def utility(s):
    """PREDECLARED dev-only decision: maximise weak-proxy warning coverage.

    The reference target is <=0.5% false warnings on benign rows. This is
    penalised in development (including concurrent attacks), unlike the
    prior one-class normal-only calibration. It is NOT a test guarantee.
    """
    v=(.15*scalar(s["weak_loss5"],True)+
       .25*scalar(s["weak_loss10"],True)+
       .25*scalar(s["weak_loss15"],True)+
       .25*scalar(s["weak_loss20"],True)+
       .10*float(s["truthful"]["warning_only_incremental_attack_coverage"] or 0.0))
    fwr=[float(x["false_warning_rate"]) for x in s.values()]
    default_fwr=float(s["default"]["false_warning_rate"])
    cost=12.*max(0.,default_fwr-REFERENCE_WARNING_FWR)
    cost+=5.*sum(max(0.,x-REFERENCE_WARNING_FWR) for x in fwr)
    cost+=.5*float(np.mean(fwr))
    return float(v-cost)

def breakdown_by_role(mod,df):
    levels=mod.predict_levels(df)
    y=df["is_anomaly"].to_numpy(dtype=int)>0
    role=df["node_type"].astype(str).to_numpy()
    out={}
    for key in sorted(set(role)):
        ix=role==key
        benign=ix&(~y);attacked=ix&y
        out[key]={
          "normal_rows":int(benign.sum()),
          "false_warning_only":int(((levels==1)&benign).sum()),
          "false_confirmed_alarms":int(((levels==2)&benign).sum()),
          "normal_warning_FWR":float(((levels==1)&benign).sum()/benign.sum()) if benign.sum() else None,
          "attack_rows":int(attacked.sum()),
          "warning_attack_rows":int(((levels==1)&attacked).sum()),
          "alert_attack_rows":int(((levels==2)&attacked).sum()),
        }
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prior",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("results_ticd_role_warnings"))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    with (args.prior/"ticd_zone_context_model.pkl").open("rb") as f:prior=pickle.load(f)
    old_count,new_count=check_prior(prior)
    cal=prior["capture_calibration"]
    source_models={"baseline":prior["baseline_model"],"selected":prior["model"]}
    frozen=source_models["baseline"].causal_model
    for mod in source_models.values():
        if mod.causal_model is not frozen:
            # Two pickle branches can be equal but separate references:
            # both originate from the same trained causal detector artifact.
            assert mod.causal_model.history==frozen.history
    models={name:RoleScopedWarningTICD(source_models[source],roles)
            for name,(source,roles) in SPECS.items()}

    validation={
      m:produce(seeds,cal,"clean_loss"+str(m),frozen)
      for m,seeds in NORMAL_VALIDATION.items()
    }
    for data in validation.values():
        if data["is_anomaly"].sum()!=0:
            raise RuntimeError("Contaminated normal threshold validation")
    alarm_t,alarm_ref,thresholds=calibrate(models,validation)
    print("ROLE_WARNING_VALIDATION",json.dumps({
      "alert_threshold":alarm_t,
      "normal_alert_FPR":alarm_ref,
      "warning_thresholds":{name:t["warning_threshold"] for name,t in thresholds.items()},
      "normal_warning_FWR":{name:t["normal_warning_FWR_by_loss"] for name,t in thresholds.items()}
    },ensure_ascii=False),flush=True)

    dev={}
    for kind,seeds in DEVELOPMENT.items():
        d=produce(seeds,cal,kind,frozen)
        dev[kind]={name:tier_metrics(mod,d) for name,mod in models.items()}
    scores={name:utility({key:dev[key][name] for key in DEVELOPMENT})
            for name in models}
    chosen=max(models,key=lambda x:(scores[x],-list(models).index(x)))
    selected=models[chosen]
    print("ROLE_WARNING_SELECTED_ON_DEVELOPMENT",json.dumps({
      "selected":chosen,"utilities":scores,
      "reference_dev_FWR":{name:dev["default"][name]["false_warning_rate"]
                           for name in models},
      "selected_policy":selected.manifest()
    },ensure_ascii=False),flush=True)

    # Absolutely no model, threshold or role selection after this point.
    comparators={"old_all_raw":models["all_raw"],
                 "old_all_zone_load":models["all_zone_load"],
                 "role_scoped_selected":selected}
    heldout={};per_seed={};roles={}
    for kind,seeds in UNTOUCHED_TEST.items():
        data=produce(seeds,cal,kind,frozen)
        heldout[kind]={name:tier_metrics(mod,data)
                       for name,mod in comparators.items()}
        per_seed[kind]={name:[
            {"seed":int(s),**tier_metrics(mod,part)}
            for s,part in data.groupby("simulation_seed",sort=True)]
            for name,mod in comparators.items()}
        roles[kind]={name:breakdown_by_role(mod,data)
                     for name,mod in comparators.items()}
        a=heldout[kind]["old_all_raw"]
        b=heldout[kind]["role_scoped_selected"]
        if a["confirmed_FP"]!=b["confirmed_FP"] or a["confirmed_TP"]!=b["confirmed_TP"]:
            raise RuntimeError("Role scope unexpectedly changed strong alarms")
        print("ROLE_WARNING_HELDOUT",json.dumps({
            "cohort":kind,
            "old":{"warning_FWR":a["false_warning_rate"],
                   "alert_FPR":a["confirmed_alert_FPR"],
                   "statistical_warning":a["by_sophistication"]["statistical"]["warning_only_incremental"]},
            "selected":{"warning_FWR":b["false_warning_rate"],
                        "alert_FPR":b["confirmed_alert_FPR"],
                        "statistical_warning":b["by_sophistication"]["statistical"]["warning_only_incremental"],
                        "statistical_confirmed":b["by_sophistication"]["statistical"]["confirmed_recall"]},
            "warning_episodes":[x["false_warning_episodes"]
                 for x in per_seed[kind]["role_scoped_selected"]]},
          ensure_ascii=False),flush=True)

    # Final checks against potentially attacker-influenced known node label:
    # classifier must not read simulated is_compromised/attack_type/gt_*.
    audit=produce([34901],cal,"weak_loss10",frozen)
    original=selected.predict_levels(audit)
    excludes=[c for c in audit if c.startswith("gt_") or c in (
      "attack_type","is_anomaly","is_compromised","attacker_sophistication",
      "normal_event","network_loss_multiplier")]
    without=audit.drop(columns=excludes)
    if not np.array_equal(original,selected.predict_levels(without)):
        raise RuntimeError("Oracle leakage through warning role filter")
    # source zone fields were computed on complete snapshot; slicing is
    # safe for pure scoring when frozen fields are already attached.
    sub=audit.sample(140,random_state=11)
    if not np.array_equal(selected.predict_levels(sub),original[sub.index]):
        raise RuntimeError("Warning scores depend on test batch composition")
    # Explicit test that warning suppression cannot affect strong-alert flag.
    strong=models["all_raw"].predict_levels(audit)==2
    assert np.array_equal(strong,original==2)

    package={
      "selected_model":selected,
      "all_raw_comparator":models["all_raw"],
      "all_zone_load_comparator":models["all_zone_load"],
      "selected_name":chosen,
      "alert_threshold":alarm_t,
      "warning_threshold":selected.warning_threshold,
      "normal_calibration":cal,
      "normal_validation_seeds":NORMAL_VALIDATION,
      "development_seeds":DEVELOPMENT,
      "heldout_seeds":UNTOUCHED_TEST,
      "source_previous_run":45252940147
    }
    with (args.out/"role_scoped_ticd_model.pkl").open("wb") as f:
        pickle.dump(package,f,pickle.HIGHEST_PROTOCOL)
    with (args.out/"role_scoped_ticd_model.pkl").open("rb") as f:
        loaded=pickle.load(f)
    if not np.array_equal(loaded["selected_model"].predict_levels(sub),
                          selected.predict_levels(sub)):
        raise RuntimeError("Reloaded role-scoped detector changed decisions")

    result={
      "scope":"Role-scoped SIP two-level TICD specialised early warning for covert proxy compromise",
      "status":"SIMULATION ONLY",
      "external_benign_LTE_archive_sha256":SOURCE_SHA,
      "real_field_SIP_compromise_validation":False,
      "exploratory_origin":"Prior old test + posthoc per-role diagnostic #45345156303, never model selection",
      "source_frozen_model_run_id":45252940147,
      "source_frozen_prior_SIP_models_not_refitted":True,
      "prior_split_count":old_count,"new_seed_count":new_count,
      "warning_is_not_confirmed":True,
      "strong_alarm_unchanged_all_node_types":True,
      "operator_role_inventory_required":True,
      "excluded_roles_are_not_covered_by_early_warning":True,
      "thresholds_only_from_benign_validation":True,
      "selected_only_on_labelled_synthetic_development":True,
      "untouched_final_test":True,
      "warning_only_budget_normal_validation":REFERENCE_WARNING_FWR,
      "normal_validation_seeds":NORMAL_VALIDATION,
      "development_seeds":DEVELOPMENT,
      "untouched_test_seeds":UNTOUCHED_TEST,
      "candidate_role_scopes":{name:m.manifest()["eligible_roles"] for name,m in models.items()},
      "normal_validation_rows":sum(len(x) for x in validation.values()),
      "alert_threshold":alarm_t,"alarm_reference_FPR":alarm_ref,
      "warning_reference_thresholds":thresholds,
      "development_utility":scores,"selected_name":chosen,
      "selected_manifest":selected.manifest(),
      "dev_results":dev,
      "heldout":heldout,"heldout_by_seed":per_seed,
      "post_test_role_breakdown":roles,
      "label_oracle_invariance":True,
      "test_batch_invariance":True,
      "model_reload":True,
      "elapsed_s":time.time()-start,
      "limitations":"Warned roles chosen for a synthetic proxy-compromise model. Generalisation to arbitrary SIP function compromise not established. Source LTE-A captures calibrate benign method weights only. Loss, master transit ACK and CU are simulator dependent."
    }
    (args.out/"role_warning_report.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("ROLE_WARNING_FINAL",json.dumps({
      "selected":chosen,"policy":selected.manifest(),
      "heldout":{kind:{
         "old_FWR":models_["old_all_raw"]["false_warning_rate"],
         "scoped_FWR":models_["role_scoped_selected"]["false_warning_rate"],
         "old_stat_warning":models_["old_all_raw"]["by_sophistication"]["statistical"]["warning_only_incremental"],
         "scoped_stat_warning":models_["role_scoped_selected"]["by_sophistication"]["statistical"]["warning_only_incremental"],
         "confirmed_FPR":models_["role_scoped_selected"]["confirmed_alert_FPR"]
       } for kind,models_ in heldout.items()},
      "oracle_test":True,"reloaded":True
    },ensure_ascii=False),flush=True)

if __name__=="__main__":main()
