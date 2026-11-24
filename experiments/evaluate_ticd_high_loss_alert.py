#!/usr/bin/env python3
"""Benign-only high-loss strong alert recalibration, untouched paired test.

The simulator explicitly supports loss multipliers up to 25, so do not
modify its validity limits to obtain artificial x30 data. Multipliers
are scenario parameters only, never detector inputs. All evidence remains
simulated except the upstream benign SIP method frequency calibration.
"""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
sys.path.insert(0,str(ROOT/"experiments"))
from ticd_high_loss_alert import HighLossAlertTICD
from evaluate_ticd_zonal_warnings import produce
from evaluate_ticd_two_tier import metrics

LOSSES=(1,5,10,15,20,25)
TRAIN={m:[41001+100*i,41002+100*i] for i,m in enumerate(LOSSES)}
VAL={m:[42001+100*i,42002+100*i] for i,m in enumerate(LOSSES)}
DEV={k:[43001+100*i,43002+100*i] for i,k in enumerate((
    "default","weak_loss10","weak_loss20","weak_loss25","truthful",
    "clean_loss20","clean_loss25"))}
TEST={k:[44001+100*i,44002+100*i,44003+100*i] for i,k in enumerate((
    "default","weak_loss5","weak_loss10","weak_loss15","weak_loss20",
    "weak_loss25","truthful","clean_loss1","clean_loss7","clean_loss20",
    "clean_loss25","clean_loss12"))}
SPECS=[(m,b,q) for m,b in (("fixed",1),("loss",6),("loss_role",6),("loss_role",3))
       for q in (.006,.008)]
def compact(x):
    return {"FPR":x["confirmed_alert_FPR"],"FWR":x["false_warning_rate"],
            "union_FPR":x["combined_normal_signal_rate"],"recall":x["confirmed_recall"],
            "coverage":x["combined_attack_coverage"],
            "statistical":x["by_sophistication"]["statistical"],
            "false_alert_episodes":x["false_alert_episodes"]}
def objective(groups):
    attacks=[v for v in groups.values() if v["attack"]]
    weak=[groups[k]["by_sophistication"]["statistical"]["confirmed_recall"]
          for k in ("weak_loss10","weak_loss20","weak_loss25")]
    benefit=.55*float(np.mean(weak))+.30*float(np.mean([v["confirmed_recall"] for v in attacks]))+.15*float(np.mean([v["combined_attack_coverage"] for v in attacks]))
    fprs=[v["confirmed_alert_FPR"] for v in groups.values()]
    fwrs=[v["false_warning_rate"] for v in groups.values()]
    # Prioritise transfer of low FPR beyond benign calibration.
    penalty=12.*sum(max(0.,v-.01) for v in fprs)+5.*sum(max(0.,v-.005) for v in fwrs)+.3*float(np.mean(fprs))
    return benefit-penalty
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prior",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"role_scoped_ticd_model.pkl").open("rb") as f: prev=pickle.load(f)
    assert prev["selected_name"]=="no_gateway_zone_load"
    prior=prev["selected_model"];cal=prev["normal_calibration"]
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    assert cal["calibration_kind"]=="partial_benign_SIP_method_profile"
    causal=prior.source_model.causal_model
    previous={s for p in ("normal_validation_seeds","development_seeds","heldout_seeds")
              for values in prev[p].values() for s in values}
    new=[s for part in (TRAIN,VAL,DEV,TEST) for values in part.values() for s in values]
    assert len(set(new))==len(new) and not previous.intersection(new) and min(new)>34901
    train=pd.concat([produce(seeds,cal,"clean_loss"+str(m),causal) for m,seeds in TRAIN.items()],ignore_index=True)
    validation={m:produce(seeds,cal,"clean_loss"+str(m),causal) for m,seeds in VAL.items()}
    assert train.is_anomaly.sum()==0 and all(df.is_anomaly.sum()==0 for df in validation.values())
    models={};refs={}
    for mode,bins,budget in SPECS:
        name=f"{mode}_{bins}_budget{budget}"
        mod=HighLossAlertTICD(prior,mode,bins,.993).fit(train).calibrate(validation,budget)
        models[name]=mod
        ref={str(m):compact(metrics(mod,df)) for m,df in validation.items()}
        for m,df in validation.items():
            actual=mod.predict_masks(df)["confirmed_alert"]
            expected=mod.normalized(df)>mod.scale
            assert np.array_equal(actual,expected), f"Calibration/inference decision mismatch: {name}, loss={m}"
        print("HIGHLOSS_CANDIDATE_REFERENCE",json.dumps({"candidate":name,"budget":budget,"scale":mod.scale,"fpr_by_loss":{k:v["FPR"] for k,v in ref.items()}}),flush=True)
        assert max(v["FPR"] for v in ref.values())<=budget+1e-12, f"Normal FPR exceeds target: {name}, {ref}"
        refs[name]=ref
    print("HIGHLOSS_CALIBRATION",json.dumps({"normal_train_rows":len(train),"normal_validation_rows":sum(map(len,validation.values())),
        "reference_max_fpr":{k:max(v["FPR"] for v in r.values()) for k,r in refs.items()} }),flush=True)
    dev={}
    for k,seeds in DEV.items():
        df=produce(seeds,cal,k,causal)
        dev[k]={name:metrics(mod,df) for name,mod in models.items()}
    utilities={name:objective({k:v[name] for k,v in dev.items()}) for name in models}
    chosen=max(models,key=lambda name:(utilities[name],-list(models).index(name)))
    selected=models[chosen]
    # Compare with actual frozen prior and equally calibrated best fixed
    # reference chosen on development; neither is chosen on holdout.
    fixed=max((name for name in models if name.startswith("fixed_")),key=lambda name:utilities[name])
    print("HIGHLOSS_SELECTED",json.dumps({"selected":chosen,"fixed":fixed,"utilities":utilities,
        "manifest":selected.manifest()}),flush=True)
    comparison={"prior":prior,"expanded_fixed":models[fixed],"selected":selected}
    result={};byseed={}
    for k,seeds in TEST.items():
        df=produce(seeds,cal,k,causal)
        result[k]={name:metrics(mod,df) for name,mod in comparison.items()}
        byseed[k]={name:[{"seed":int(s),**metrics(mod,part)} for s,part in df.groupby("simulation_seed")]
                    for name,mod in comparison.items()}
        print("HIGHLOSS_HELDOUT",json.dumps({"cohort":k,**{name:compact(v) for name,v in result[k].items()}}),flush=True)
    check=produce([45901],cal,"weak_loss25",causal)
    original=selected.predict_levels(check)
    drop=[c for c in check if c.startswith("gt_") or c in ("is_anomaly","attack_type","is_compromised","attacker_sophistication","normal_event","network_loss_multiplier")]
    assert np.array_equal(original,selected.predict_levels(check.drop(columns=drop)))
    sub=check.sample(200,random_state=21)
    assert np.array_equal(original[sub.index.to_numpy()],selected.predict_levels(sub))
    with (a.out/"high_loss_alert_ticd.pkl").open("wb") as f:
        pickle.dump({"model":selected,"selected":chosen,"prior":prior,"calibration":cal},f,pickle.HIGHEST_PROTOCOL)
    with (a.out/"high_loss_alert_ticd.pkl").open("rb") as f: restored=pickle.load(f)
    assert np.array_equal(original,restored["model"].predict_levels(check))
    report={"scope":"simulated SIP strong-alert calibration only","real_field_validation":False,
            "source_run":45345567415,"train_seeds":TRAIN,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
            "warning_channel_frozen":True,"selected":chosen,"fixed":fixed,"manifest":selected.manifest(),
            "reference_max_loss_multiplier":25,"validation":refs,"dev_utilities":utilities,"development":dev,
            "heldout":result,"per_seed":byseed,"oracle_invariance":True,"batch_invariance":True,"model_reload":True,
            "all_roles_strong_alert":True,"heldout_used_for_selection":False,
            "warning_only_is_not_confirmed":True}
    (a.out/"high_loss_alert_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("HIGHLOSS_FINAL",json.dumps({"selected":chosen,"heldout":{k:{n:compact(v) for n,v in pair.items()} for k,pair in result.items()},"oracle_invariance":True,"model_reload":True}),flush=True)
if __name__=="__main__": main()
