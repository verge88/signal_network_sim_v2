#!/usr/bin/env python3
"""Read-only post hoc research audit of warning-only false positives.

Runs only OLD and already reported 241xx / 243xx / 247xx simulation seeds.
This is hypothesis discovery, NEVER the confirmation experiment, and no
threshold/model weights are updated by this analysis.

Uses is_anomaly, attack_type, normal_event and proximity to attack solely to
EXPLAIN errors AFTER a fully frozen TICD scoring decision.
"""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
sys.path.insert(0,str(ROOT/"experiments"))
from retrain_ticd_causal_ack import cohort
from ticd_zonal_warnings import prepare_zonal_warning_context

DIAGNOSTIC_OLD_TEST={
  "default":[24101,24102,24103],
  "weak_loss10":[24301,24302,24303],
  "clean_loss1":[24701,24702,24703]
}

def prepare(df,model):
    df=prepare_zonal_warning_context(df,model.causal_model)
    df=df.copy()
    df["warning_only"]=model.predict_levels(df)==1
    df["confirmed_alarm"]=model.predict_levels(df)==2
    df["benign"]=df["is_anomaly"].to_numpy(dtype=int)==0
    zone=df.groupby(["simulation_seed","interval","zone_id"],sort=False)
    df["any_attack_in_current_zone"]=zone["is_anomaly"].transform("max")>0
    df["any_attack_in_current_interval"]=df.groupby(
        ["simulation_seed","interval"],sort=False)["is_anomaly"].transform("max")>0
    return df

def grouping(df,cols):
    out=[]
    for key,frame in df.groupby(cols,dropna=False,sort=True):
        if not isinstance(key,tuple):key=(key,)
        normal=frame[frame.benign]
        flagged=normal[normal.warning_only]
        attack=frame[~frame.benign]
        out.append({
          **{name:str(v) for name,v in zip(cols,key)},
          "normal_rows":len(normal),
          "false_warning_rows":len(flagged),
          "FWR":float(len(flagged)/len(normal)) if len(normal) else None,
          "confirmed_false_alarms":int(normal.confirmed_alarm.sum()),
          "attack_rows":len(attack),
          "warning_on_attack_rows":int(attack.warning_only.sum()),
          "confirmed_attack_rows":int(attack.confirmed_alarm.sum()),
        })
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prior",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("results_warning_root_cause"))
    args=ap.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    with (args.prior/"ticd_zone_context_model.pkl").open("rb") as f:
        pkg=pickle.load(f)
    assert pkg["selected_variant"]=="zone_load_0p5"
    cal=pkg["capture_calibration"]
    original=pkg["baseline_model"]
    selected=pkg["model"]
    results={}
    for cohort_label,seeds in DIAGNOSTIC_OLD_TEST.items():
        if cohort_label.startswith("clean_loss"):
            kind="clean"
            mult=float(cohort_label.removeprefix("clean_loss"))
        else:
            kind=cohort_label
            mult=None
        xs=[cohort([seed],cal,kind,mult=mult) for seed in seeds]
        original_batch=pd.concat(xs,ignore_index=True)
        result={}
        for name,mod in (("baseline",original),("zone_selected",selected)):
            df=prepare(original_batch,mod)
            nf=df[df.benign]
            assert not (nf.warning_only & nf.confirmed_alarm).any()
            tables={
              "node_type":grouping(df,["node_type"]),
              "normal_event":grouping(df,["normal_event"]),
              "node_type_and_current_zone_attack":
                   grouping(df,["node_type","any_attack_in_current_zone"]),
              "node_type_and_current_global_attack":
                   grouping(df,["node_type","any_attack_in_current_interval"]),
              "zone_id":grouping(df,["zone_id"]),
              "node_id":grouping(df,["node_id","node_type"]),
            }
            result[name]={
              "rows":len(df),"normal_rows":len(nf),
              "false_warnings":int(nf.warning_only.sum()),
              "false_alarms":int(nf.confirmed_alarm.sum()),
              "attack_rows":int((~df.benign).sum()),
              "attack_warnings":int(df.loc[~df.benign,"warning_only"].sum()),
              "attack_alerts":int(df.loc[~df.benign,"confirmed_alarm"].sum()),
              "tables":tables
            }
            print("WARNING_SOURCE_SUMMARY",json.dumps({
               "cohort":cohort_label,"detector":name,
               "normal":len(nf),"false_warnings":int(nf.warning_only.sum()),
               "node_type":tables["node_type"],
               "normal_event":tables["normal_event"]
            },ensure_ascii=False),flush=True)
        results[cohort_label]=result
    output={
        "purpose":"post hoc exploratory FWR root-cause diagnostic only",
        "data_reuse":"Earlier heldout seeds deliberately reused for DIAGNOSTIC, not for model selection or final confirmation",
        "real_network_validation":False,
        "frozen_decisions":True,
        "source_action_id":45252940147,
        "seeds":DIAGNOSTIC_OLD_TEST,
        "results":results
    }
    (args.out/"warning_source_diagnosis.json").write_text(
        json.dumps(output,indent=2,ensure_ascii=False)+"\n")
    print("WARNING_SOURCE_DONE",json.dumps({
      kind:{name:{"false_warning_rows":stats["false_warnings"],
                  "normal_rows":stats["normal_rows"]}
            for name,stats in models.items()}
      for kind,models in results.items()},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
