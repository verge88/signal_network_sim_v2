#!/usr/bin/env python3
"""Independent audit of archived TICD heldout results at simulation-seed level.

No model fitting, source trace parsing, retuning or fresh test-data inspection:
only archived summary metrics and provenance. Cluster = simulated run/seed,
NOT individual row, to avoid treating correlated 20-second observations as
independent samples.
"""
import argparse,hashlib,json,math,statistics
from pathlib import Path

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path("verified_ticd_loss_adaptive"))
    a=parser.parse_args()
    d=json.loads(a.source.read_text())
    out=a.out;out.mkdir(exist_ok=True,parents=True)
    selected=d["selected_model"]
    assert d["reload_success"] and d["max_batch_score_difference"]<=1e-12
    assert d["real_calibration_source_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    assert d["external_real_world_ticd_validation"] is False
    assert d["new_parameters_fitted_only_on_normal_train"]
    assert d["thresholds_selected_only_on_normal_loss_aware_validation"]
    all_seeds=[]
    for kind in ("train_seeds","normal_validation_seeds","development_seeds","untouched_test_seeds"):
        all_seeds.extend(x for arr in d[kind].values() for x in arr)
    assert len(all_seeds)==len(set(all_seeds))
    q=d["normal_validation_fpr"][selected]
    assert all(x<=0.01001 for x in q.values())

    summary={}
    all_signs=[]
    for cohort,models in d["heldout_per_seed"].items():
        old=models["old_physical_1pc"]
        new=models[selected]
        assert len(old)==len(new)==3
        per_seed=[]
        for b,n in zip(old,new):
            assert b["seed"]==n["seed"]
            assert b["normal"]==n["normal"] and b["attacks"]==n["attacks"]
            dfpr=n["fpr"]-b["fpr"]
            drec=None if b["recall"] is None else n["recall"]-b["recall"]
            per_seed.append({
              "seed":b["seed"],"FPR_old":b["fpr"],"FPR_new":n["fpr"],
              "FPR_change":dfpr,"recall_old":b["recall"],
              "recall_new":n["recall"],"recall_change":drec,
              "delta_FP":n["fp"]-b["fp"],"delta_TP":n["tp"]-b["tp"]})
        deltas=[x["FPR_change"] for x in per_seed]
        rec=[x["recall_change"] for x in per_seed if x["recall_change"] is not None]
        summary[cohort]={
           "seed_count":len(per_seed),
           "all_three_seeds_improve_FPR":all(x<0 for x in deltas),
           "mean_FPR_change_percentage_points":100*statistics.mean(deltas),
           "sd_FPR_change_percentage_points":100*statistics.stdev(deltas),
           "mean_recall_change_percentage_points":100*statistics.mean(rec) if rec else None,
           "sd_recall_change_percentage_points":100*statistics.stdev(rec) if len(rec)>1 else None,
           "per_seed":per_seed}
        pooled=d["heldout_metrics"][cohort][selected]
        assert sum(x["delta_FP"] for x in per_seed)==(
            pooled["fp"]-d["heldout_metrics"][cohort]["old_physical_1pc"]["fp"])
        print("SEED_CLUSTER_AUDIT",json.dumps({
          "cohort":cohort,"all_FPR_reduced":summary[cohort]["all_three_seeds_improve_FPR"],
          "mean_FPR_change_pp":summary[cohort]["mean_FPR_change_percentage_points"],
          "mean_recall_change_pp":summary[cohort]["mean_recall_change_percentage_points"]
        },ensure_ascii=False),flush=True)
    final={
      "status":"success",
      "original_report_sha256":hashlib.sha256(a.source.read_bytes()).hexdigest(),
      "model":selected,
      "independent_evaluation_unit":"simulation_seed, not 20-second row",
      "real_external_ticd_validation":False,
      "seed_overlap":False,
      "cohort_level_audit":summary,
      "caveat":"Three seeds per condition quantify modest within-simulator variation only; no real network confidence interval."
    }
    (out/"paired_seed_audit.json").write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n")
    print("PAIRWISE_SEED_CLUSTER_AUDIT_SUCCESS",json.dumps({
        x:v["mean_FPR_change_percentage_points"] for x,v in summary.items()
    },ensure_ascii=False),flush=True)

if __name__=="__main__":main()
