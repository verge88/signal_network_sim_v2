"""Post-hoc error diagnosis of frozen physical/causal TICD.
Previously disclosed test seeds are reused ONLY for diagnosis. No fitting.
"""
from __future__ import annotations
import argparse,json,pickle,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from evaluate_ticd_zonal_warnings import produce
from evaluate_ticd_two_tier import metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prior",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    report_path=a.prior/"physical_causal_alert_report.json"
    r=json.loads(report_path.read_text())
    with (a.prior/"physical_causal_alert_ticd.pkl").open("rb") as f: pkg=pickle.load(f)
    model=pkg["model"];old=pkg["prior"];cal=pkg["calibration"]
    assert pkg["selected"]==r["selected"]=="max_short_0.008"
    paired=[]
    for k,parts in r["per_seed"].items():
        for x,y in zip(parts["prior"],parts["selected"]):
            assert x["seed"]==y["seed"] and x["normal"]==y["normal"] and x["attack"]==y["attack"]
            paired.append({"cohort":k,"seed":y["seed"],"fpr_old":x["confirmed_alert_FPR"],
                           "fpr_new":y["confirmed_alert_FPR"],"fwr_new":y["false_warning_rate"],
                           "stat_old":x["by_sophistication"]["statistical"]["confirmed_recall"],
                           "stat_new":y["by_sophistication"]["statistical"]["confirmed_recall"],
                           "fp_old":x["confirmed_FP"],"fp_new":y["confirmed_FP"],
                           "episodes_old":x["false_alert_episodes"],"episodes_new":y["false_alert_episodes"]})
    assert len(paired)==36
    summary={"fpr_improved":sum(x["fpr_new"]<x["fpr_old"] for x in paired),
             "fpr_pass":sum(x["fpr_new"]<=.01 for x in paired),
             "fwr_pass":sum(x["fwr_new"]<=.005 for x in paired),
             **{k:sum(x[k] for x in paired) for k in ("fp_old","fp_new","episodes_old","episodes_new")}}
    print("FUSION_PAIRED_AUDIT",json.dumps(summary),flush=True)
    causal=old.prior.source_model.causal_model
    cases={}
    for k in ("default","clean_loss7","weak_loss20"):
        df=produce(r["test_seeds"][k],cal,k,causal)
        lv=model.predict_levels(df)
        before=old.predict_levels(df)
        normal=df.is_anomaly.to_numpy()==0
        instant=old.normalized(df)/old.scale
        history=model.reference.score(df)*model.weight
        a_hit=instant>model.threshold
        h_hit=history>model.threshold
        assert np.array_equal(lv==2,a_hit|h_hit)
        fp=normal&(lv==2)
        expected=r["heldout"][k]["selected"]
        assert int(fp.sum())==expected["confirmed_FP"]
        assert int((normal&(lv==1)).sum())==expected["warning_only_FP"]
        z=df[["node_type","normal_event","simulation_seed"]].copy()
        z["normal"]=normal;z["fp"]=fp;z["old_fp"]=normal&(before==2)
        z["history_only_fp"]=fp&h_hit&~a_hit
        z["instant_only_fp"]=fp&a_hit&~h_hit
        z["both_fp"]=fp&a_hit&h_hit
        z["false_warning"]=normal&(lv==1)
        def table(cols):
            return z.groupby(cols,dropna=False)[["normal","fp","old_fp","history_only_fp","instant_only_fp","both_fp","false_warning"]].sum().reset_index().to_dict("records")
        cases[k]={"role":table(["node_type"]),"event":table(["normal_event"]),
                  "seed_role":table(["simulation_seed","node_type"])}
        print("FUSION_ERROR_SOURCE",json.dumps({"cohort":k,**cases[k]}),flush=True)
    result={"source_run":45366626708,"source_sha256":hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "scope":"Exploratory reuse of old synthetic heldout; no fitting; no independent field validation",
            "paired":paired,"summary":summary,"diagnosis":cases}
    (a.out/"fusion_error_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print("FUSION_ERROR_AUDIT_VERIFIED",flush=True)
if __name__=="__main__":main()
