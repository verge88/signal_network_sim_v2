#!/usr/bin/env python3
"""Independent corroboration-gate study after a documented FPR regression.

Retains the prior frozen CausalPersistentACKTICD and LossConditionedTICD.
Only changes the rule joining instantaneous and persistent master ACK
evidence; all candidate thresholds fit separate normal validation sets.
Selects gate on new labelled simulated development seeds, then evaluates
once on entirely new unseen simulator runs (including loss x20).

This is not an independent real-network detector test.
"""
from __future__ import annotations
import argparse,json,pickle,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
sys.path.insert(0,str(ROOT/"experiments"))
from ticd_corroborated_causal import CorroboratedCausalTICD
from retrain_ticd_causal_ack import cohort,summary
from ticd_causal_ack import CausalPersistentACKTICD

VAL={1:[11201,11202],3:[11301,11302],5:[11401,11402],
     10:[11501,11502],15:[11601,11602]}
DEV={
 "default":[11701,11702],
 "weak_loss10":[11801,11802],
 "weak_loss15":[11901,11902],
 "truthful":[12001,12002],
}
TEST={
 "default":[12101,12102,12103],
 "weak_loss5":[12201,12202,12203],
 "weak_loss10":[12301,12302,12303],
 "weak_loss15":[12401,12402,12403],
 "truthful":[12501,12502,12503],
 "clean_loss7":[12601,12602,12603],
 "clean_loss20":[12701,12702,12703],
}
GATES=(.3,.4,.5,.6,.7,.8,.9)

def threshold(model,val):
    q={str(m):float(np.quantile(model.score(df),.99,method="higher"))
       for m,df in val.items()}
    t=max(q.values())
    fpr={str(m):float((model.score(df)>t).mean()) for m,df in val.items()}
    if not all(x<=.01001 for x in fpr.values()):
        raise RuntimeError("Normal validation 1% FPR budget violated")
    return t,fpr,q

def criterion(metrics):
    # Declared PRIOR to this experiment's final holdout: prefer long-term
    # statistical compromise detection while penalizing FPR >1.2% on any
    # simulated development cohort.
    a=metrics["weak_loss10"];b=metrics["weak_loss15"]
    truth=metrics["truthful"];ordinary=metrics["default"]
    def soph(x,n):return float(x["by_sophistication"][n]["recall"] or 0.)
    recall=(.4*soph(a,"statistical")+
            .35*soph(b,"statistical")+
            .15*soph(truth,"adaptive")+
            .10*float(ordinary["recall"] or 0.))
    fpr_penalty=8.0*sum(max(0.,m["fpr"]-.012) for m in metrics.values())
    return float(recall-fpr_penalty)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prior",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("results_ticd_causal_gated"))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    with (args.prior/"ticd_causal_ack_model.pkl").open("rb") as f:prior=pickle.load(f)
    if not isinstance(prior["model"],CausalPersistentACKTICD):
        raise RuntimeError("Expected frozen causal ACK model")
    if prior["selected_name"]!="causal_short_w0p7":
        raise RuntimeError("Pinned prior trained model unexpected")
    cal=prior["normal_calibration"]
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    prev=prior["seed_partitions"]
    previous={s for group in prev.values() for ls in group.values() for s in ls}
    this=[s for group in (VAL,DEV,TEST) for ls in group.values() for s in ls]
    if len(this)!=len(set(this)) or set(this).intersection(previous):
        raise RuntimeError("Overlapping train, validation, dev or test seed")
    models={"loss_only":prior["baseline_model"],
            "causal_ungated":prior["model"]}
    for gate in GATES:
        name="gate_"+str(gate).replace(".","p")
        models[name]=CorroboratedCausalTICD(prior["model"],gate)

    benign_val={m:cohort(seeds,cal,"clean",m) for m,seeds in VAL.items()}
    thresholds={};nominal_fprs={}
    for name,model in models.items():
        t,f,q=threshold(model,benign_val)
        thresholds[name]=t
        nominal_fprs[name]={"fpr":f,"p99":q}
    print("CORROBORATION_NORMAL_REFERENCE",json.dumps({
        "thresholds":thresholds,
        "fpr_by_loss":{k:v["fpr"] for k,v in nominal_fprs.items()}},ensure_ascii=False),flush=True)

    dev={}
    for kind,seeds in DEV.items():
        df=cohort(seeds,cal,kind)
        dev[kind]={name:summary(model,df,thresholds[name])
                   for name,model in models.items()}
    utilities={name:criterion({k:dev[k][name] for k in DEV})
               for name in models}
    selected=max(models,key=lambda k:(utilities[k],-list(models).index(k)))
    print("CORROBORATION_SELECTED",json.dumps({
        "selected":selected,"utilities":utilities,"threshold":thresholds[selected]},ensure_ascii=False),flush=True)

    final={};per_seed={}
    for kind,seeds in TEST.items():
        if kind.startswith("clean_loss"):
            df=cohort(seeds,cal,"clean",float(kind.removeprefix("clean_loss")))
        else:
            df=cohort(seeds,cal,kind)
        final[kind]={};per_seed[kind]={}
        for name in ("loss_only","causal_ungated",selected):
            # Deduplicate if selection picks existing comparator.
            if name in final[kind]:continue
            v=summary(models[name],df,thresholds[name])
            final[kind][name]=v
            per_seed[kind][name]=[
                {"seed":int(seed),**summary(models[name],frame,thresholds[name])}
                for seed,frame in df.groupby("simulation_seed")]
        print("CORROBORATION_HELDOUT",json.dumps({
           "cohort":kind,
           "evaluated":{k:{"recall":v["recall"],"fpr":v["fpr"],
                          "statistical":v["by_sophistication"]["statistical"]["recall"]}
                        for k,v in final[kind].items()}},ensure_ascii=False),flush=True)

    model_package={
      "model":models[selected],"selected":selected,"threshold":thresholds[selected],
      "baseline_model":models["loss_only"],"baseline_threshold":thresholds["loss_only"],
      "prior_causal_model":models["causal_ungated"],
      "prior_causal_threshold":thresholds["causal_ungated"],
      "loss_model_source_checksum":cal["source_archive_sha256"],
      "seed_partitions":{"old":prev,"validation":VAL,"development":DEV,"test":TEST},
    }
    with (args.out/"ticd_corroborated_model.pkl").open("wb") as f:
        pickle.dump(model_package,f,pickle.HIGHEST_PROTOCOL)
    with (args.out/"ticd_corroborated_model.pkl").open("rb") as f:
        reloaded=pickle.load(f)
    check=cohort([12901],cal,"weak_loss10")
    full=reloaded["model"].score(check)
    sample=check.sample(150,random_state=55)
    diff=float(np.max(np.abs(reloaded["model"].score(sample)-full[sample.index])))
    if diff>1e-12:raise RuntimeError("Scoring changes with batch composition")
    report={
     "scope":"secondary TICD architecture study after a previously observed FPR regression, simulation ONLY",
     "real_benign_lte_sha256":cal["source_archive_sha256"],
     "real_attack_ticd_validation":False,
     "prior_causal_detector_parameters_unchanged":True,
     "prior_loss_conditioned_detector_parameters_unchanged":True,
     "no_fitting_on_heldout_data":True,
     "normal_validation_seeds":VAL,"development_seeds":DEV,
     "untouched_test_seeds":TEST,
     "used_previous_test_only_to_motivate_new_gate":True,
     "new_test_independent_of_all_previous_seeds":True,
     "gate_candidates":list(GATES),
     "normal_validation_rows":sum(len(x) for x in benign_val.values()),
     "validation_thresholds":thresholds,
     "reference_val_fpr":nominal_fprs,
     "development_model_utility":utilities,
     "selected":selected,
     "development_metrics":dev,
     "heldout":final,
     "by_test_seed":per_seed,
     "model_reloaded":True,
     "batch_independence_max_diff":diff,
     "elapsed_seconds":time.time()-start,
     "limitations":"All attack, CU transit and loss observations are simulated; source LTE PCAP only calibrated benign SIP method shares. Repeated rows and seeds share attack mechanisms."
    }
    (args.out/"corroborated_causal_report.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("CORROBORATION_FINAL",json.dumps({
      "selected":selected,
      "summary":{k:{"loss_only_fpr":v["loss_only"]["fpr"],
        "causal_fpr":v["causal_ungated"]["fpr"],
        "selected_fpr":v[selected]["fpr"],
        "loss_only_statistical":v["loss_only"]["by_sophistication"]["statistical"]["recall"],
        "causal_statistical":v["causal_ungated"]["by_sophistication"]["statistical"]["recall"],
        "selected_statistical":v[selected]["by_sophistication"]["statistical"]["recall"]}
        for k,v in final.items()},
      "reloaded":True,"max_batch_diff":diff},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
