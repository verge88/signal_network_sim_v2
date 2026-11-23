#!/usr/bin/env python3
"""Frozen-normal TICD with independent physical ACK closure channel.

Stage B2: Candidate physical branches are selected using development
simulations with altered attacker reporting/loss. Selected model and prior
temporal TICD are scored ONCE on fresh disjoint seeded scenario realizations.
No external detector validation is claimed.
"""
from __future__ import annotations
import argparse,json,pickle,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
sys.path.insert(0,str(ROOT/"experiments"))
from retrain_sip_ticd_calibrated import simulation,evaluate
from stress_sip_ticd_multiscale import simulate_stress
from ticd_recalibrated import FrozenReferenceTICD
from ticd_temporal_invariants import MultiScaleInvariantTICD
from ticd_physical_closure import PhysicalClosureTICD,prepare_master_closure_features

TRAIN=[2101,2102]
VAL=[2201,2202]
DEV={
  "default":[2301,2302],
  "weak_covert":[2401,2402],
  "weak_covert_loss5":[2501,2502],
  "truthful_compromised_reports":[2601,2602],
}
HELDOUT={
  "default":[3101,3102,3103],
  "weak_covert":[3201,3202,3203],
  "weak_covert_loss5":[3301,3302,3303],
  "truthful_compromised_reports":[3401,3402,3403],
}
VARIANTS=("physical","physical_zone","physical_zone_context")
MODEL_NAMES=("baseline","temporal")+VARIANTS

def load_calibration(p):
    x=json.loads(p.read_text())
    if x["calibration_kind"]!="partial_benign_SIP_method_profile" or x["source_archive_sha256"]!="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7":
        raise ValueError("Untrusted/mismatched real benign SIP calibration")
    return x

def produce_group(seeds,cal,kind):
    frames=[]
    for seed in seeds:
        if kind=="default":
            df,meta=simulation(seed,cal,2.,attacks_enabled=True)
        elif kind=="clean":
            df,meta=simulation(seed,cal,2.,attacks_enabled=False)
        else:
            df,meta=simulate_stress(seed,cal,kind,2.)
        df=prepare_master_closure_features(df)
        if df["is_anomaly"].sum()==0 and kind!="clean":
            raise ValueError(f"No attack in {kind}")
        if df["is_anomaly"].sum()>0 and kind=="clean":
            raise ValueError(f"Unexpected attack in {kind}")
        df["experiment_cohort"]=kind
        frames.append(df)
        print("PHYSICAL_COHORT",json.dumps({"kind":kind,"seed":seed,"rows":len(df),
              "positive":int(df.is_anomaly.sum())}),flush=True)
    return pd.concat(frames,ignore_index=True)

def calibrate(models,val):
    thresholds={}
    reference_fpr={}
    for name,model in models.items():
        score=model.score(val)
        if not np.isfinite(score).all():
            raise ValueError(f"Nonfinite {name} normal validation scores")
        threshold=float(np.quantile(score,0.99,method="higher"))
        fpr=float((score>threshold).mean())
        assert fpr<=.011
        thresholds[name]=threshold
        reference_fpr[name]=fpr
    return thresholds,reference_fpr

def development_objective(per_group):
    """Predeclared selection utility on *development* labels only."""
    low=per_group["weak_covert"]
    hi=per_group["weak_covert_loss5"]
    truthful=per_group["truthful_compromised_reports"]
    default=per_group["default"]
    def soph(item,name):
        return item["by_compromise_sophistication"][name]["recall"] or 0.
    # Explicitly penalize the low-intensity and truthful report weaknesses.
    utility=(0.25*soph(low,"statistical")+
             0.20*soph(hi,"statistical")+
             0.30*soph(truthful,"adaptive")+
             0.15*soph(truthful,"statistical")+
             0.10*soph(default,"adaptive"))
    # Selection disfavours variants that are fragile to loss shift.
    penalty=2*sum(max(0.,r["fpr"]-0.025) for r in per_group.values())
    return float(utility-penalty)

def scalar_metrics(stats):
    return {"n":stats["rows"],"positive":stats["attacks"],
      "auc_roc":stats["auc_roc"],"auc_pr":stats["auc_pr"],
      "recall":stats["tpr_recall"],"fpr":stats["fpr"],
      "f1":stats["f1"],"precision":stats["precision"],
      "TP":stats["tp"],"FP":stats["fp"],"TN":stats["tn"],"FN":stats["fn"],
      "by_attack":stats["by_attack"],
      "by_sophistication":stats["by_compromise_sophistication"]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--calibration",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,default=Path("results_ticd_physical"))
    args=ap.parse_args();args.out_dir.mkdir(exist_ok=True,parents=True)
    began=time.time()
    calibration=load_calibration(args.calibration)
    all_ids=TRAIN+VAL+[x for ls in DEV.values() for x in ls]+[x for ls in HELDOUT.values() for x in ls]
    if len(set(all_ids))!=len(all_ids):
        raise ValueError("Leakage: overlapping simulation seeds")
    train=produce_group(TRAIN,calibration,"clean")
    val=produce_group(VAL,calibration,"clean")
    assert train["is_anomaly"].sum()==0 and val["is_anomaly"].sum()==0
    base=FrozenReferenceTICD(seed=42).fit(train)
    temporal=MultiScaleInvariantTICD(base_model=base,mode="temporal").fit(train)
    models={"baseline":base,"temporal":temporal}
    for variant in VARIANTS:
        models[variant]=PhysicalClosureTICD(base_model=base,mode=variant).fit(train)
    thresholds,reference_fpr=calibrate(models,val)
    print("NORMAL_REFERENCE_THRESHOLDS",json.dumps(
        {"thresholds":thresholds,"validation_fpr":reference_fpr}),flush=True)

    dev_results={}
    development_utility={}
    for kind,seeds in DEV.items():
        df=produce_group(seeds,calibration,kind)
        dev_results[kind]={name:scalar_metrics(evaluate(model,df,thresholds[name]))
                           for name,model in models.items()}
    for variant in VARIANTS:
        # Utility operates only on development synthetic labels.
        # Convert summaries back to a lightweight metrics schema.
        p={}
        for kind,mp in dev_results.items():
            m=mp[variant]
            p[kind]={"fpr":m["fpr"],
              "by_compromise_sophistication":m["by_sophistication"]}
        development_utility[variant]=development_objective(p)
    selected=max(VARIANTS,key=lambda x:(development_utility[x],-VARIANTS.index(x)))
    print("PHYSICAL_ARCHITECTURE_CHOSEN",json.dumps({
       "selected":selected,"dev_utility":development_utility,
       "dev_compromise_by_model":{k:{n:v["by_sophistication"]
            for n,v in dev_results[k].items()} for k in dev_results}},ensure_ascii=False),flush=True)

    # Only now generate previously unobserved heldout records.
    result={};seed_rows={}
    for kind,seeds in HELDOUT.items():
        d=produce_group(seeds,calibration,kind)
        result[kind]={}
        seed_rows[kind]={}
        for name in ("baseline","temporal",selected):
            evaluation=scalar_metrics(evaluate(models[name],d,thresholds[name]))
            result[kind][name]=evaluation
            per_seed=[]
            for seed in seeds:
                chunk=d[d.simulation_seed==seed]
                q=scalar_metrics(evaluate(models[name],chunk,thresholds[name]))
                q["seed"]=seed
                per_seed.append(q)
            seed_rows[kind][name]=per_seed
        print("PHYSICAL_HELDOUT",json.dumps({"cohort":kind,
          "models":{k:{"fpr":v["fpr"],"recall":v["recall"],
                       "sophistication":v["by_sophistication"]}
                    for k,v in result[kind].items()}},ensure_ascii=False),flush=True)

    manifest={
      "scope":"Simulation-based robustness of TICD physical closure architecture after partial real LTE method-profile calibration",
      "source_real_lte_sha256":calibration["source_archive_sha256"],
      "normal_train_only":True,
      "learned_reference_scales_only_normal_train":True,
      "thresholds_only_clean_validation":True,
      "development_attack_labels_used_only_for_architecture_choice":True,
      "heldout_labels_used_only_for_evaluation":True,
      "method":"physical ACK/200 closure deficit adjusted by master-measured link impairment, plus concurrent zone median and original signed/temporal I1/I2",
      "not_deterministic_invariant":"Physical ACK shortage is statistical, can arise naturally from call behavior and loss",
      "training_seeds":TRAIN,"normal_validation_seeds":VAL,
      "development_seeds":DEV,"heldout_seeds":HELDOUT,
      "selected":selected,"candidate_variants":list(VARIANTS),
      "thresholds":thresholds,"validation_fpr":reference_fpr,
      "development_utilities":development_utility,
      "development_summary":dev_results,
      "heldout_comparison":result,
      "per_heldout_seed":seed_rows,
      "normal_train_rows":len(train),"normal_val_rows":len(val),
      "normal_reference_scales":{
          k:dict(v.scales) for k,v in models.items() if isinstance(v,PhysicalClosureTICD)},
      "no_real_compromise_validation":True,
      "simulator_features_not_yet_validated_in_real_cu_telemetry":True,
      "elapsed_seconds":time.time()-began}
    (args.out_dir/"physical_closure_comparison.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    package={"model":models[selected],"model_name":selected,
       "threshold":thresholds[selected],
       "baseline_model":base,"baseline_threshold":thresholds["baseline"],
       "temporal_model":temporal,"temporal_threshold":thresholds["temporal"],
       "normal_calibration":calibration,"split_seeds":{
           "train":TRAIN,"validation":VAL,"development":DEV,"heldout":HELDOUT}}
    with (args.out_dir/"ticd_physical_closure_model.pkl").open("wb") as f:
        pickle.dump(package,f,pickle.HIGHEST_PROTOCOL)
    # Model must score on a random subset with exactly the same features.
    reloaded=pickle.load((args.out_dir/"ticd_physical_closure_model.pkl").open("rb"))
    sample=produce_group([3901],calibration,"weak_covert")
    full_score=reloaded["model"].score(sample)
    slice_rows=sample.sample(240,random_state=19)
    part_score=reloaded["model"].score(slice_rows)
    assert np.allclose(part_score,full_score[slice_rows.index.to_numpy()],atol=1e-12,rtol=0)
    print("PHYSICAL_TICD_FINAL",json.dumps({
        "selected":selected,
        "heldout":{k:{"selected_fpr":v[selected]["fpr"],
            "selected_recall":v[selected]["recall"],
            "truthful_adaptive_recall":v[selected]["by_sophistication"]["adaptive"]["recall"],
            "temporal_fpr":v["temporal"]["fpr"],
            "temporal_recall":v["temporal"]["recall"]}
            for k,v in result.items()},
        "model_reload":True,
        "test_batch_independence":True},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
