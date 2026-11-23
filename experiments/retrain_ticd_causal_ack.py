#!/usr/bin/env python3
"""Retrain a causal ACK persistence channel and compare on fresh disjoint SIP runs.

Previously trained physical + conditional-loss TICD is entirely frozen.
New EWMA evidence scale uses only simulated BENIGN training streams.
Candidate operating points: WORST benign P99 among disjoint clean
validation regimes. Labelled simulated development chooses EWMA branch.
A further set of untouched seeds is generated only AFTER selection.
All attack scenarios and CU loss paths are simulator-generated; the real
LTE-A SIP PCAP only calibrated benign INVITE/ACK/BYE proportions.
"""
from __future__ import annotations
import argparse,json,pickle,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
sys.path.insert(0,str(ROOT/"experiments"))
from ticd_causal_ack import CausalPersistentACKTICD,add_causal_ack_history
from ticd_physical_closure import prepare_master_closure_features
from retrain_sip_ticd_calibrated import simulation,evaluate
from stress_sip_ticd_multiscale import simulate_stress

CHECKSUM="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
TRAIN={1:[8101,8102],3:[8201,8202],5:[8301,8302],10:[8401,8402],15:[8501,8502]}
VAL={1:[8601,8602],3:[8701,8702],5:[8801,8802],10:[8901,8902],15:[9001,9002]}
DEV={
 "default":[9101,9102],
 "weak_loss10":[9201,9202],
 "truthful":[9301,9302],
 "weak_loss15":[9401,9402],
}
TEST={
 "default":[10101,10102,10103],
 "weak_loss5":[10201,10202,10203],
 "weak_loss10":[10301,10302,10303],
 "weak_loss15":[10401,10402,10403],
 "truthful":[10501,10502,10503],
 "clean_loss7":[10601,10602,10603],
 "clean_loss20":[10701,10702,10703],
}
CANDIDATES=(("short",.7),("medium",.7),("long",.7),
            ("medium",1.0),("long",1.0))
def build(seed,cal,kind,mult=None):
    if kind=="clean":
        d,_=simulation(seed,cal,2.,False,loss_multiplier=float(mult))
    elif kind=="default":
        d,_=simulation(seed,cal,2.,True)
    elif kind=="truthful":
        d,_=simulate_stress(seed,cal,"truthful_compromised_reports")
    else:
        multiplier=float(kind.removeprefix("weak_loss"))
        d,_=simulate_stress(seed,cal,"weak_covert",loss_multiplier=multiplier)
    d=prepare_master_closure_features(d)
    d=add_causal_ack_history(d)
    d["network_loss_multiplier"]=float(mult) if kind=="clean" else (
        1. if kind in ("default","truthful") else float(kind.removeprefix("weak_loss")))
    if kind=="clean" and d["is_anomaly"].sum():
        raise RuntimeError("Normal reference contaminated by attack labels")
    if kind!="clean" and d["is_anomaly"].sum()==0:
        raise RuntimeError("Missing simulated attack")
    return d

def cohort(seeds,cal,kind,mult=None):
    xs=[]
    for seed in seeds:
        df=build(seed,cal,kind,mult)
        xs.append(df)
        print("CAUSAL_ACK_RUN",json.dumps({"kind":kind,"loss":mult,
            "seed":seed,"rows":len(df),"attacks":int(df.is_anomaly.sum())}),flush=True)
    return pd.concat(xs,ignore_index=True)

def thresholds_from_clean_validation(model,val):
    qs={str(k):float(np.quantile(model.score(v),.99,method="higher"))
         for k,v in val.items()}
    threshold=max(qs.values())
    fpr={str(k):float((model.score(v)>threshold).mean()) for k,v in val.items()}
    if max(fpr.values())>.01001:
        raise RuntimeError("1% normal validation FPR target violated")
    return float(threshold),qs,fpr

def summary(model,df,threshold):
    x=evaluate(model,df,threshold)
    return {
      "normal":int(x["normal"]),"attacks":int(x["attacks"]),
      "tp":int(x["tp"]),"tn":int(x["tn"]),"fp":int(x["fp"]),"fn":int(x["fn"]),
      "recall":x["tpr_recall"],"fpr":x["fpr"],"f1":x["f1"],
      "precision":x["precision"],"roc_auc":x["auc_roc"],"pr_auc":x["auc_pr"],
      "by_sophistication":x["by_compromise_sophistication"],
      "by_attack":x["by_attack"],
    }

def objective(dev):
    def soph(g,name):
        return dev[g]["by_sophistication"][name]["recall"] or 0.0
    # Fixed *before seeing final test*. Encourage low covert high loss.
    score=(.35*soph("weak_loss10","statistical")+
           .30*soph("weak_loss15","statistical")+
           .15*soph("truthful","adaptive")+
           .10*soph("truthful","statistical")+
           .10*float(dev["default"]["recall"] or 0))
    penalty=4*sum(max(0.,v["fpr"]-.02) for v in dev.values())
    return float(score-penalty)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prior",required=True,type=Path)
    p.add_argument("--calibration",required=True,type=Path)
    p.add_argument("--output",type=Path,default=Path("results_ticd_causal_ack"))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    started=time.time()
    with (a.prior/"ticd_loss_adaptive_model.pkl").open("rb") as f:old=pickle.load(f)
    if old["calibration_source_sha256"]!=CHECKSUM or old["selected_name"]!="loss_node":
        raise RuntimeError("Wrong provenance of original loss-aware trained TICD")
    existing=old["model"]
    calibration=json.loads(a.calibration.read_text())
    if calibration.get("calibration_kind")!="partial_benign_SIP_method_profile":
        raise RuntimeError("Expected exactly the independent LTE-A SIP calibration manifest")
    if calibration["source_archive_sha256"]!=CHECKSUM:
        raise RuntimeError("External-calibration checksum mismatch")
    seeds_by_group={
        "train":TRAIN,"validation":VAL,"development":DEV,"heldout":TEST}
    all_seeds=[seed for part in seeds_by_group.values() for ls in part.values() for seed in ls]
    if len(set(all_seeds))!=len(all_seeds):
        raise RuntimeError("Overlapping seed partitions")
    old_seeds=old["normal_train_seeds"]
    previous=set(seed for ls in old_seeds.values() for seed in ls)
    for key in ("normal_validation_seeds","development_seeds","untouched_test_seeds"):
        previous.update(seed for ls in old[key].values() for seed in ls)
    if previous.intersection(all_seeds):
        raise RuntimeError("Reuse of prior experiment seeds")

    trains={m:cohort(seeds,calibration,"clean",m) for m,seeds in TRAIN.items()}
    val={m:cohort(seeds,calibration,"clean",m) for m,seeds in VAL.items()}
    train=pd.concat(list(trains.values()),ignore_index=True)
    assert int(train["is_anomaly"].sum())==0
    candidates={}
    for span,weight in CANDIDATES:
        name="causal_"+span+"_w"+str(weight).replace(".","p")
        candidates[name]=CausalPersistentACKTICD(existing,history=span,weight=weight).fit(train)
    models={"baseline":existing,**candidates}
    thresholds={};normal_val={}
    for name,model in models.items():
        theta,quantiles,fpr=thresholds_from_clean_validation(model,val)
        thresholds[name]=theta
        normal_val[name]={"quantiles":quantiles,"fpr":fpr}
    print("CAUSAL_ACK_REFERENCE",json.dumps({
        "thresholds":thresholds,
        "normal_validation_fpr":{k:v["fpr"] for k,v in normal_val.items()}},
        ensure_ascii=False),flush=True)

    development={}
    for label,seeds in DEV.items():
        d=cohort(seeds,calibration,label)
        development[label]={name:summary(model,d,thresholds[name])
                            for name,model in models.items()}
    scores={}
    for name in candidates:
        scores[name]=objective({label:development[label][name] for label in DEV})
    best=max(candidates,key=lambda n:(scores[n],-list(candidates).index(n)))
    print("CAUSAL_ACK_SELECTED_ON_DEV",json.dumps({
        "selected":best,"utilities":scores,
        "baseline_utility":objective({x:development[x]["baseline"] for x in DEV})},ensure_ascii=False),flush=True)

    # Never use final test data to refit threshold, choose weight or span.
    final={};by_seed={}
    for label,seeds in TEST.items():
        if label.startswith("clean_loss"):
            mult=float(label.removeprefix("clean_loss"))
            d=cohort(seeds,calibration,"clean",mult)
        else:
            d=cohort(seeds,calibration,label)
        final[label]={};by_seed[label]={}
        for name in ("baseline",best):
            v=summary(models[name],d,thresholds[name])
            final[label][name]=v
            by_seed[label][name]=[
                {"seed":int(seed),**summary(models[name],x,thresholds[name])}
                for seed,x in d.groupby("simulation_seed")]
        print("CAUSAL_ACK_HELDOUT",json.dumps({
          "cohort":label,"n":len(d),
          "base":{"recall":final[label]["baseline"]["recall"],
                  "fpr":final[label]["baseline"]["fpr"],
                  "statistical":final[label]["baseline"]["by_sophistication"]["statistical"]["recall"]},
          "selected":{"recall":final[label][best]["recall"],
                  "fpr":final[label][best]["fpr"],
                  "statistical":final[label][best]["by_sophistication"]["statistical"]["recall"]}},
           ensure_ascii=False),flush=True)

    package={
      "model":models[best],"threshold":thresholds[best],
      "baseline_model":existing,"baseline_threshold":thresholds["baseline"],
      "original_baseline_threshold":old["threshold"],
      "selected_name":best,"normal_calibration":calibration,
      "seed_partitions":seeds_by_group,
      "upstream_model_frozen":True}
    with (a.output/"ticd_causal_ack_model.pkl").open("wb") as f:
        pickle.dump(package,f,pickle.HIGHEST_PROTOCOL)
    with (a.output/"ticd_causal_ack_model.pkl").open("rb") as f:
        loaded=pickle.load(f)
    sample=cohort([10801],calibration,"weak_loss10")
    same=loaded["model"].score(sample)
    part=sample.sample(n=180,random_state=42)
    score_diff=float(np.max(np.abs(loaded["model"].score(part)-same[part.index])))
    if score_diff>1e-12:
        raise RuntimeError("Batch-dependent TICD evidence after freezing")
    # Causal history must NEVER be influenced by later observations:
    altered=sample.copy();mask=(altered.node_id==2)&(altered.interval>=200)
    altered.loc[mask,"master_ack_shortfall_adjusted"]=0.
    rec=add_causal_ack_history(altered)
    original=add_causal_ack_history(sample)
    check=(sample.node_id==2)&(sample.interval<200)
    for column in ("ack_shortfall_ewma_6","ack_shortfall_ewma_16","ack_shortfall_ewma_32"):
        if not np.allclose(rec.loc[check,column],original.loc[check,column],
                           atol=0,rtol=0):
            raise RuntimeError("Future value leaked into prior causal EWMA")
    report={
      "result_kind":"new causal ACK weak-signal accumulation, SIMULATION ONLY",
      "actual_research_capture_calibration":"benign SIP INVITE/ACK/BYE only",
      "calibration_source_sha256":CHECKSUM,
      "full_real_TICD_validation":False,
      "trained_prior_detector_frozen":True,
      "normal_only_training":True,
      "normal_only_validation":True,
      "architecture_selection_uses_synthetic_development_labels":True,
      "test_labels_used_only_for_final_evaluation":True,
      "training_seeds":TRAIN,"normal_validation_seeds":VAL,
      "development_seeds":DEV,"heldout_seeds":TEST,
      "training_normal_rows":len(train),
      "validation_normal_rows":sum(len(x) for x in val.values()),
      "candidate_details":{k:v.reference_manifest() for k,v in candidates.items()},
      "decision_thresholds":thresholds,
      "normal_validation_fpr":normal_val,
      "development_architecture_utilities":scores,
      "chosen":best,
      "development_metrics":development,
      "heldout_results":final,
      "per_test_seed":by_seed,
      "reloaded":True,"max_batch_score_difference":score_diff,
      "future_value_leak_test_passed":True,
      "elapsed_seconds":time.time()-started,
      "scientific_limits":"The simulator controls attack impact on ACK closure; all attacks and CU telemetry remain simulated. Loss estimates are CU path proxy; sample rows are dependent within each run."
    }
    (a.output/"causal_ack_retraining_report.json").write_text(
       json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("CAUSAL_ACK_FINAL",json.dumps({
      "selected":best,"model_reloaded":True,"causal_test":True,
      "normal_train_rows":len(train),"normal_val_rows":report["validation_normal_rows"],
      "summary":{name:{"old_fpr":s["baseline"]["fpr"],
                       "new_fpr":s[best]["fpr"],
                       "old_recall":s["baseline"]["recall"],
                       "new_recall":s[best]["recall"],
                       "old_statistical":s["baseline"]["by_sophistication"]["statistical"]["recall"],
                       "new_statistical":s[best]["by_sophistication"]["statistical"]["recall"]}
                 for name,s in final.items()}},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
