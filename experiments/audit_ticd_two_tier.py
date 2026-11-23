#!/usr/bin/env python3
"""Read-only verification of SIP two-tier alarm and warning experiment.

Audits only immutable GitHub Actions JSON; NEVER uses labels to adapt a
detector. Simulated run/seed is the independent analysis cluster, not an
individual event row. No frequentist real-world confidence interval claim.
"""
from __future__ import annotations
import argparse,hashlib,json,statistics
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--out",type=Path,default=Path("results_ticd_two_tier_audit"))
    a=parser.parse_args()
    d=json.loads(a.source.read_text())
    if d["real_world_SIP_detector_validation"]:
        raise ValueError("Incorrect claim of external full TICD validation")
    assert d["all_thresholds_from_new_normal_validation_only"]
    assert d["heldout_not_used_for_any_selection"]
    assert d["reloaded_model_verified"] and d["test_batch_label_invariance"]
    assert d["selected_warning_budget"] in d["candidate_warning_FPR_budgets"]
    source_digest=hashlib.sha256(a.source.read_bytes()).hexdigest()
    partitions=("normal_validation_seeds","development_seeds","heldout_seeds")
    ids=[seed for key in partitions for seq in d[key].values() for seed in seq]
    assert len(ids)==len(set(ids))
    selected=str(d["selected_warning_budget"])
    vref=d["normal_warning_reference"][selected]["warning_fpr_by_loss"]
    assert max(vref.values())<=d["selected_warning_budget"]+1e-4
    alert_ref=d["normal_alarm_FPR_by_loss"]
    assert max(alert_ref.values())<=0.01001

    comparison={}
    for cohort,results in d["untouched_test_metrics"].items():
        m=results["two_tier"]
        s=d["per_test_seed_metrics"][cohort]["two_tier"]
        if len(s)!=3:
            raise RuntimeError("Unexpected number of independent test seeds")
        for name in ("confirmed_TP","confirmed_FP","warning_only_TP",
                     "warning_only_FP","covered_attack_rows","normal","attack"):
            assert sum(row[name] for row in s)==m[name],(cohort,name)
        assert m["confirmed_TP"]+m["warning_only_TP"]==m["covered_attack_rows"]
        assert m["confirmed_FP"]+m["warning_only_FP"]<=m["normal"]
        assert m["combined_normal_signal_rate"]==(
            (m["confirmed_FP"]+m["warning_only_FP"])/m["normal"])
        if m["attack"]:
            assert m["combined_attack_coverage"]==(
                m["confirmed_TP"]+m["warning_only_TP"])/m["attack"]
        else:
            assert m["combined_attack_coverage"] is None
        fwr=[float(row["false_warning_rate"]) for row in s]
        alarm=[float(row["confirmed_alert_FPR"]) for row in s]
        warning_episodes=[int(row["false_warning_episodes"]) for row in s]
        alarm_episodes=[int(row["false_alert_episodes"]) for row in s]
        comparison[cohort]={
          "independent_seed_count":len(s),
          "seed_ids":[row["seed"] for row in s],
          "pooled_FPR_confirmed":m["confirmed_alert_FPR"],
          "pooled_FWR_warning":m["false_warning_rate"],
          "pooled_FPR_union":m["combined_normal_signal_rate"],
          "pooled_confirmed_attack_recall":m["confirmed_recall"],
          "pooled_extra_warning_attack_coverage":m["warning_only_incremental_attack_coverage"],
          "pooled_any_signal_attack_coverage":m["combined_attack_coverage"],
          "mean_seed_FWR":statistics.mean(fwr),
          "sd_seed_FWR":statistics.stdev(fwr),
          "min_seed_FWR":min(fwr),
          "max_seed_FWR":max(fwr),
          "mean_seed_confirmed_FPR":statistics.mean(alarm),
          "sd_seed_confirmed_FPR":statistics.stdev(alarm),
          "false_warning_episodes_total":sum(warning_episodes),
          "false_alert_episodes_total":sum(alarm_episodes),
          "false_warning_episodes_per_2h_run":warning_episodes,
          "false_alert_episodes_per_2h_run":alarm_episodes,
          "all_3_seeds_above_ref_warning_budget":all(
              x>d["selected_warning_budget"] for x in fwr),
        }
        print("TWO_TIER_PAIRED_SEED_AUDIT",json.dumps({
          "cohort":cohort,"fwr_mean":comparison[cohort]["mean_seed_FWR"],
          "fwr_range":[min(fwr),max(fwr)],
          "warning_episodes":warning_episodes,
          "alert_episodes":alarm_episodes
        },ensure_ascii=False),flush=True)

    a.out.mkdir(exist_ok=True,parents=True)
    result={
      "status":"verified",
      "source_report_sha256":source_digest,
      "source_run_id":45249660732,
      "all_alarm_warning_confusion_counts_reconciled":True,
      "all_seed_partitions_disjoint":True,
      "normal_validation_warning_budget_respected":True,
      "real_SIP_compromise_validation":False,
      "seed_level_inference_only":True,
      "normal_reference_budget":d["selected_warning_budget"],
      "test_default_warning_budget_exceeded":
          comparison["default"]["pooled_FWR_warning"]>d["selected_warning_budget"],
      "test_pure_normal_loss20_confirmed_FPR_exceeds_1pct":
          comparison["clean_loss20"]["pooled_FPR_confirmed"]>0.01,
      "cohorts":comparison,
      "scientific_limit":"Only three repeated independent simulator seeds per condition; simulates identical attack mechanisms and cannot establish live network FPR."
    }
    (a.out/"two_tier_seed_audit.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("TWO_TIER_AUDIT_COMPLETE",json.dumps({
      "verified":True,
      "default_3of3_above_budget":comparison["default"]["all_3_seeds_above_ref_warning_budget"],
      "default_pooled_false_warning_rate":comparison["default"]["pooled_FWR_warning"],
      "loss20_confirmed_fpr":comparison["clean_loss20"]["pooled_FPR_confirmed"]
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
