#!/usr/bin/env python3
"""Frozen score, run-level normal calibration and fresh paired test.

Selection uses development data only. Calibration run IDs are never used
in inference. No row-IID confidence intervals or real-field claims.
"""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from ticd_run_robust_alert import RunRobustAlertTICD
from evaluate_ticd_zonal_warnings import produce
from evaluate_ticd_two_tier import metrics
from evaluate_ticd_high_loss_alert import compact
from evaluate_ticd_role_adaptive_causal import role_stats

LOSSES=(1,5,7,10,12,15,20,25)
KINDS=("default","weak_loss5","weak_loss10","weak_loss15","weak_loss20",
       "weak_loss25","truthful","clean_loss1","clean_loss7","clean_loss12",
       "clean_loss20","clean_loss25")
VAL={m:[71001+100*i+j for j in range(5)] for i,m in enumerate(LOSSES)}
DEV={k:[72001+100*i+j for j in range(4)] for i,k in enumerate(KINDS)}
TEST={k:[74001+100*i+j for j in range(5)] for i,k in enumerate(KINDS)}
MODES=("pooled","p90_run","worst_run")
BUDGETS=(.004,.006,.008)

def dev_stats(pooled,per_seed):
    runs=[x for part in per_seed.values() for x in part]
    fprs=np.array([x["confirmed_alert_FPR"] for x in runs])
    fwrs=np.array([x["false_warning_rate"] for x in runs])
    recall={k:pooled[k]["by_sophistication"]["statistical"]["confirmed_recall"]
            for k in ("weak_loss5","weak_loss10","weak_loss15","weak_loss20","weak_loss25")}
    minimum=min(recall["weak_loss20"],recall["weak_loss25"])
    deficit=max(0.,.85-recall["weak_loss20"])+max(0.,.85-recall["weak_loss25"])
    fwr_violations=sum(x>.005 for x in fwrs)
    return {"recall":recall,"recall_min_high_loss":minimum,"recall_shortfall":deficit,
            "fpr_pass":int((fprs<=.01).sum()),"fwr_pass":int((fwrs<=.005).sum()),
            "max_fpr":float(fprs.max()),"max_fwr":float(fwrs.max()),
            "mean_fpr":float(fprs.mean()),"mean_excess_fpr":float(np.maximum(fprs-.01,0).mean()),
            "fwr_violations":int(fwr_violations),
            "mean_weak_recall":float(np.mean(list(recall.values()))),
            "feasible_recall":bool(deficit==0.)}

def selection_key(s):
    # First preserve the prespecified high-loss recall floor. Among eligible
    # models prioritize passing runs, warning budget, worst FPR and recall.
    # If no candidate reaches the floor, choose smallest shortfall and
    # disclose that the target failed; never relax it on heldout data.
    return (int(s["feasible_recall"]),-s["recall_shortfall"],
            s["fpr_pass"],s["fwr_pass"],-s["max_fpr"],
            s["mean_weak_recall"],-s["mean_fpr"])

def audit_summary(heldout,by_seed,name):
    runs=[x for part in by_seed.values() for x in part[name]]
    f=np.array([x["confirmed_alert_FPR"] for x in runs])
    w=np.array([x["false_warning_rate"] for x in runs])
    return {"runs":len(runs),"FPR_pass_runs":int((f<=.01).sum()),"FWR_pass_runs":int((w<=.005).sum()),
            "FPR_pass_cohorts":sum(x[name]["confirmed_alert_FPR"]<=.01 for x in heldout.values()),
            "FWR_pass_cohorts":sum(x[name]["false_warning_rate"]<=.005 for x in heldout.values()),
            "max_FPR_run":float(f.max()),"max_FWR_run":float(w.max()),
            "mean_FPR_run":float(f.mean()),"p90_FPR_run":float(np.quantile(f,.9)),
            "false_alert_rows":sum(x["confirmed_FP"] for x in runs),
            "false_alert_episodes":sum(x["false_alert_episodes"] for x in runs),
            "false_warning_rows":sum(x["warning_only_FP"] for x in runs),
            "violations":[{"seed":x["seed"],"FPR":x["confirmed_alert_FPR"],"FWR":x["false_warning_rate"]}
                          for x in runs if x["confirmed_alert_FPR"]>.01 or x["false_warning_rate"]>.005]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"role_adaptive_causal_ticd.pkl").open("rb") as f:pkg=pickle.load(f)
    assert pkg["selected"]=="weighted_w0.5_g0.0_b0.008"
    prior=pkg["model"];cal=pkg["calibration"]
    assert prior.mode=="weighted" and prior.gateway_weight==.5
    assert cal["calibration_kind"]=="partial_benign_SIP_method_profile"
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    old=json.loads((a.prior/"role_adaptive_causal_report.json").read_text())
    old_seeds={s for k in ("normal_validation_seeds","development_seeds","test_seeds") for ls in old[k].values() for s in ls}
    new=[s for part in (VAL,DEV,TEST) for ls in part.values() for s in ls]
    assert len(new)==len(set(new))==148 and min(new)>65901 and not old_seeds.intersection(new)
    causal=prior.prior.prior.prior.source_model.causal_model
    normal={m:produce(ls,cal,"clean_loss"+str(m),causal) for m,ls in VAL.items()}
    models={};ref={}
    for mode in MODES:
        for b in BUDGETS:
            n=f"{mode}_{b}"
            mod=RunRobustAlertTICD(prior,mode,b).calibrate(normal)
            cohorts={str(k):float((mod.score(df)>mod.threshold).mean()) for k,df in normal.items()}
            runs={str(int(s)):float((mod.score(part)>mod.threshold).mean())
                  for df in normal.values() for s,part in df.groupby("simulation_seed")}
            assert max(cohorts.values())<=b+1e-12
            if mode=="worst_run":assert max(runs.values())<=b+1e-12
            models[n]=mod;ref[n]={"cohort_FPR":cohorts,"run_FPR":runs,"manifest":mod.manifest()}
            print("ROBUST_NORMAL",json.dumps({"candidate":n,"threshold":mod.threshold,
                  "max_cohort_FPR":max(cohorts.values()),"max_run_FPR":max(runs.values())}),flush=True)
    # Include the exact prior only as a comparison, not as a calibrated candidate.
    dev={};dev_seed={}
    for k,ls in DEV.items():
        df=produce(ls,cal,k,causal)
        dev[k]={n:metrics(m,df) for n,m in models.items()}
        dev_seed[k]={n:[{"seed":int(s),**metrics(m,part)} for s,part in df.groupby("simulation_seed")]
                     for n,m in models.items()}
    scores={n:dev_stats({k:x[n] for k,x in dev.items()},{k:x[n] for k,x in dev_seed.items()}) for n in models}
    selected=max(models,key=lambda n:(selection_key(scores[n]),-list(models).index(n)))
    control=max((n for n in models if n.startswith("pooled_")),key=lambda n:selection_key(scores[n]))
    robust=max((n for n in models if not n.startswith("pooled_")),key=lambda n:selection_key(scores[n]))
    print("ROBUST_SELECTED",json.dumps({"selected":selected,"control":control,"best_robust":robust,
          "development":scores,"selected_threshold":models[selected].threshold}),flush=True)
    comparison={"prior":prior,"pooled_control":models[control],"robust_control":models[robust],"selected":models[selected]}
    heldout={};by_seed={};roles={}
    for k,ls in TEST.items():
        df=produce(ls,cal,k,causal)
        heldout[k]={n:metrics(m,df) for n,m in comparison.items()}
        by_seed[k]={n:[{"seed":int(s),**metrics(m,x)} for s,x in df.groupby("simulation_seed")]
                    for n,m in comparison.items()}
        roles[k]={n:role_stats(m,df) for n,m in comparison.items()}
        for n,v in heldout[k].items():
            for c in ("normal","attack","confirmed_FP","confirmed_TP","warning_only_FP","warning_only_TP","covered_attack_rows"):
                assert sum(x[c] for x in by_seed[k][n])==v[c]
        print("ROBUST_HELDOUT",json.dumps({"cohort":k,**{n:compact(v) for n,v in heldout[k].items()}}),flush=True)
    mod=models[selected]
    audit=produce([75901],cal,"weak_loss25",causal)
    y=mod.predict_levels(audit)
    # Seed identity is only needed upstream for causal feature construction.
    # Already prepared rows must not depend on simulation metadata in scoring.
    drop=[c for c in audit if c.startswith("gt_") or c in ("is_anomaly","attack_type","is_compromised",
        "attacker_sophistication","normal_event","network_loss_multiplier","simulation_seed")]
    assert np.array_equal(y,mod.predict_levels(audit.drop(columns=drop)))
    sub=audit.sample(200,random_state=31)
    assert np.array_equal(y[sub.index.to_numpy()],mod.predict_levels(sub))
    assert np.array_equal(mod.score(audit),prior.score(audit))
    with (a.out/"run_robust_ticd.pkl").open("wb") as f:
        pickle.dump({"model":mod,"prior":prior,"pooled_control":models[control],
                     "robust_control":models[robust],"selected":selected,"calibration":cal},f,pickle.HIGHEST_PROTOCOL)
    with (a.out/"run_robust_ticd.pkl").open("rb") as f:restored=pickle.load(f)
    assert np.array_equal(y,restored["model"].predict_levels(audit))
    summary={n:audit_summary(heldout,by_seed,n) for n in comparison}
    out={"scope":"Simulation-only run-robust calibration","source_run":45377337545,"real_field_validation":False,
         "normal_validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
         "selected":selected,"pooled_control":control,"robust_control":robust,
         "manifest":mod.manifest(),"normal_reference":ref,"development_selection":scores,
         "development":dev,"development_by_seed":dev_seed,"heldout":heldout,"per_seed":by_seed,
         "roles":roles,"summary":summary,"heldout_used_for_selection":False,"score_frozen":True,
         "warning_reference_frozen":True,"seed_oracle_invariance":True,"batch_invariance":True,"model_reload":True}
    (a.out/"run_robust_report.json").write_text(json.dumps(out,indent=2)+"\n")
    for n,s in summary.items(): print("ROBUST_SUMMARY",json.dumps({"model":n,**s}),flush=True)
    print("ROBUST_FINAL",json.dumps({"selected":selected,"pooled_control":control,"robust_control":robust,"audits_passed":True}),flush=True)
if __name__=="__main__":main()
