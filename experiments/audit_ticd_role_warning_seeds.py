#!/usr/bin/env python3
"""Read-only paired simulation-seed audit for role-scoped SIP TICD warning.

No training, adaptation, candidate selection or threshold updates.
Reports per-seed raw confusion counts and warning episodes. Simulated runs
are not evidence of real-world TICD recall/FPR.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,type=Path)
    ap.add_argument("--out",type=Path,default=Path("results_ticd_role_seed_audit"))
    args=ap.parse_args()
    d=json.loads(args.source.read_text())
    assert d["source_frozen_prior_SIP_models_not_refitted"]
    assert d["thresholds_only_from_benign_validation"]
    assert d["selected_only_on_labelled_synthetic_development"]
    assert d["untouched_final_test"] and not d["real_field_SIP_compromise_validation"]
    assert d["label_oracle_invariance"] and d["test_batch_invariance"] and d["model_reload"]
    roles=d["selected_manifest"]["eligible_roles"]
    assert isinstance(roles,list) and "UA_GW" not in roles
    assert all(k in ("PROXY","REDIRECT","REGISTRAR") for k in roles)

    new_seeds=[s for p in ("normal_validation_seeds","development_seeds","untouched_test_seeds")
               for ls in d[p].values() for s in ls]
    assert len(new_seeds)==len(set(new_seeds))
    assert all(s>25901 for s in new_seeds)
    assert max(d["alarm_reference_FPR"].values())<=0.01001
    for detail in d["warning_reference_thresholds"].values():
        assert max(detail["normal_warning_FWR_by_loss"].values())<=0.0051

    audit={}
    improves=ties=regresses=0
    for cohort,models in d["heldout"].items():
        old=models["old_all_raw"];new=models["role_scoped_selected"]
        per=d["heldout_by_seed"][cohort]
        prior=per["old_all_raw"];selected=per["role_scoped_selected"]
        assert len(prior)==len(selected)==3
        assert old["confirmed_FP"]==new["confirmed_FP"]
        assert old["confirmed_TP"]==new["confirmed_TP"]
        assert old["normal"]==new["normal"] and old["attack"]==new["attack"]
        for column in ("normal","attack","confirmed_TP","confirmed_FP",
                       "warning_only_TP","warning_only_FP",
                       "covered_attack_rows","false_warning_episodes"):
            assert sum(x[column] for x in prior)==old[column],(cohort,column,"old")
            assert sum(x[column] for x in selected)==new[column],(cohort,column,"new")
        assert old["confirmed_TP"]+old["warning_only_TP"]==old["covered_attack_rows"]
        assert new["confirmed_TP"]+new["warning_only_TP"]==new["covered_attack_rows"]
        assert new["warning_only_FP"]<=new["normal"]
        assert old["false_warning_rate"]==old["warning_only_FP"]/old["normal"]
        assert new["false_warning_rate"]==new["warning_only_FP"]/new["normal"]
        if new["attack"]:
            assert new["combined_attack_coverage"]==new["covered_attack_rows"]/new["attack"]
        else: assert new["combined_attack_coverage"] is None
        label_roles=d["post_test_role_breakdown"][cohort]["role_scoped_selected"]
        assert label_roles["UA_GW"]["false_warning_only"]==0
        assert label_roles["UA_GW"]["warning_attack_rows"]==0
        assert sum(v["false_warning_only"] for v in label_roles.values())==new["warning_only_FP"]
        assert sum(v["warning_attack_rows"] for v in label_roles.values())==new["warning_only_TP"]

        diffs=[]
        for a,b in zip(prior,selected):
            assert a["seed"]==b["seed"]
            assert a["normal"]==b["normal"] and a["attack"]==b["attack"]
            assert a["confirmed_FP"]==b["confirmed_FP"]
            assert a["confirmed_TP"]==b["confirmed_TP"]
            delta=b["false_warning_rate"]-a["false_warning_rate"]
            if delta< -1e-12:improves+=1
            elif delta>1e-12:regresses+=1
            else:ties+=1
            diffs.append({
              "seed":a["seed"],
              "normal_rows":a["normal"],
              "old_warning_FP":a["warning_only_FP"],
              "new_warning_FP":b["warning_only_FP"],
              "old_FWR":a["false_warning_rate"],
              "new_FWR":b["false_warning_rate"],
              "difference_pp":100.*delta,
              "old_warning_episodes":a["false_warning_episodes"],
              "new_warning_episodes":b["false_warning_episodes"],
              "old_confirmed_TP":a["confirmed_TP"],
              "new_confirmed_TP":b["confirmed_TP"],
            })
        audit[cohort]={
          "matched_simulation_seed_pairs":diffs,
          "old_pooled_warning_FP":old["warning_only_FP"],
          "new_pooled_warning_FP":new["warning_only_FP"],
          "old_pooled_warning_FWR":old["false_warning_rate"],
          "new_pooled_warning_FWR":new["false_warning_rate"],
          "delta_pooled_pp":100.*(new["false_warning_rate"]-old["false_warning_rate"]),
          "old_false_warning_episodes":old["false_warning_episodes"],
          "new_false_warning_episodes":new["false_warning_episodes"],
          "confirmed_FPR_unchanged":old["confirmed_alert_FPR"]==new["confirmed_alert_FPR"],
          "statistical_attack_warning_old":old["by_sophistication"]["statistical"]["warning_only_incremental"],
          "statistical_attack_warning_new":new["by_sophistication"]["statistical"]["warning_only_incremental"],
          "source_test_raw_roles":d["post_test_role_breakdown"][cohort]["old_all_raw"],
          "selected_test_roles":label_roles
        }
        print("ROLE_SEED_AUDIT_COHORT",json.dumps({
           "cohort":cohort,
           "delta_fwr_pp":audit[cohort]["delta_pooled_pp"],
           "false_warning_events_old_new":[old["warning_only_FP"],new["warning_only_FP"]],
           "false_warning_episodes_old_new":[old["false_warning_episodes"],new["false_warning_episodes"]],
           "per_seed_delta_pp":[x["difference_pp"] for x in diffs]},
           ensure_ascii=False),flush=True)
    assert improves+ties+regresses==27
    assert improves>regresses
    assert audit["default"]["new_pooled_warning_FWR"]<.005
    assert audit["default"]["old_pooled_warning_FWR"]>.005
    output={
      "scope":"Read-only pairwise audit of independent SIMULATED seed runs, no real SIP field data",
      "source_action_run":45345567415,
      "source_report_sha256":hashlib.sha256(args.source.read_bytes()).hexdigest(),
      "source_selected_role_policy":d["selected_name"],
      "test_seed_pairs":27,
      "fwr_improved_seeds":improves,
      "fwr_tied_seeds":ties,
      "fwr_regressed_seeds":regresses,
      "all_confusion_counts_reconciled":True,
      "alert_decisions_frozen_for_all_roles":True,
      "excluded_UA_gateway_early_warning":True,
      "new_test_unseen_simulation_seeds":True,
      "no_real_SIP_detector_validation":True,
      "new_still_violates_1pct_confirmed_FPR_at_clean_loss20":
          d["heldout"]["clean_loss20"]["role_scoped_selected"]["confirmed_alert_FPR"]>.01,
      "new_does_not_guarantee_warning_FWR_on_every_single_seed":True,
      "by_cohort":audit,
      "limits":"Each condition has only three simulator runs; reuse common generator and fixed threat model for compromised SIP proxies. UA_GW early-warning coverage intentionally removed."
    }
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/"role_warning_seed_audit.json").write_text(
        json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print("ROLE_SEED_AUDIT_FINAL",json.dumps({
      "paired_runs_improved":improves,"unchanged":ties,"worsened":regresses,
      "default_fwr_old":audit["default"]["old_pooled_warning_FWR"],
      "default_fwr_new":audit["default"]["new_pooled_warning_FWR"],
      "default_warning_episodes_old":audit["default"]["old_false_warning_episodes"],
      "default_warning_episodes_new":audit["default"]["new_false_warning_episodes"],
      "clean_loss20_high_confidence_FPR":d["heldout"]["clean_loss20"]["role_scoped_selected"]["confirmed_alert_FPR"]},
      ensure_ascii=False),flush=True)

if __name__=="__main__":main()
