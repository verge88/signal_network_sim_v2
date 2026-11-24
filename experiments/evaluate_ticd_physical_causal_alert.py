#!/usr/bin/env python3
"""Benign-calibrated physical/causal fusion, independent simulator seed test."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from ticd_physical_causal_alert import CausalPhysicalReference,PhysicalCausalAlertTICD
from evaluate_ticd_zonal_warnings import produce
from evaluate_ticd_two_tier import metrics
from evaluate_ticd_high_loss_alert import compact

LOSSES=(1,5,7,10,12,15,20,25)
TRAIN={m:[51001+100*i,51002+100*i] for i,m in enumerate(LOSSES)}
VAL={m:[52001+100*i,52002+100*i] for i,m in enumerate(LOSSES)}
DEV={k:[53001+100*i,53002+100*i] for i,k in enumerate((
    "default","weak_loss5","weak_loss10","weak_loss15","weak_loss20","weak_loss25",
    "truthful","clean_loss7","clean_loss20","clean_loss25"))}
TEST={k:[54001+100*i+j for j in range(3)] for i,k in enumerate((
    "default","weak_loss5","weak_loss10","weak_loss15","weak_loss20","weak_loss25",
    "truthful","clean_loss1","clean_loss7","clean_loss12","clean_loss20","clean_loss25"))}
SPECS=[("instant",None,1.,b) for b in (.006,.008)]
SPECS += [(mode,h,w,b) for mode,h,w in (
    ("max","short",1.),("max","medium",1.),("max","long",1.),
    ("gated","short",1.),("gated","medium",1.),("blend","medium",1.))
    for b in (.006,.008)]

def utility(groups):
    atk=[v for v in groups.values() if v["attack"]]
    weak=[groups[k]["by_sophistication"]["statistical"]["confirmed_recall"]
          for k in ("weak_loss10","weak_loss15","weak_loss20","weak_loss25")]
    fprs=[v["confirmed_alert_FPR"] for v in groups.values()]
    fwrs=[v["false_warning_rate"] for v in groups.values()]
    benefit=.65*float(np.mean(weak))+.25*float(np.mean([v["confirmed_recall"] for v in atk]))+.10*float(np.mean([v["combined_attack_coverage"] for v in atk]))
    penalty=18.*sum(max(0.,x-.01) for x in fprs)+6.*sum(max(0.,x-.005) for x in fwrs)+.5*float(np.mean(fprs))
    return benefit-penalty

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"high_loss_alert_ticd.pkl").open("rb") as f:package=pickle.load(f)
    assert package["selected"]=="loss_role_3_budget0.008"
    prior=package["model"];cal=package["calibration"]
    assert cal["calibration_kind"]=="partial_benign_SIP_method_profile"
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    old=json.loads((a.prior/"high_loss_alert_report.json").read_text())
    old_seeds={s for key in ("train_seeds","validation_seeds","development_seeds","test_seeds") for ls in old[key].values() for s in ls}
    new=[s for part in (TRAIN,VAL,DEV,TEST) for ls in part.values() for s in ls]
    assert len(new)==len(set(new)) and min(new)>45901 and not old_seeds.intersection(new)
    causal=prior.prior.source_model.causal_model
    train=pd.concat([produce(seeds,cal,"clean_loss"+str(m),causal) for m,seeds in TRAIN.items()],ignore_index=True)
    validation={m:produce(seeds,cal,"clean_loss"+str(m),causal) for m,seeds in VAL.items()}
    refs={h:CausalPhysicalReference(prior,h).fit(train) for h in ("short","medium","long")}
    models={};reference_metrics={}
    for mode,h,w,b in SPECS:
        name=f"{mode}_{h or 'none'}_{b}"
        model=PhysicalCausalAlertTICD(prior,refs.get(h),mode,w).calibrate(validation,b)
        scores={str(m):compact(metrics(model,df)) for m,df in validation.items()}
        assert max(v["FPR"] for v in scores.values())<=b+1e-12,(name,scores)
        models[name]=model;reference_metrics[name]=scores
        print("FUSION_NORMAL",json.dumps({"name":name,"threshold":model.threshold,"budget":b,"max_fpr":max(v["FPR"] for v in scores.values())}),flush=True)
    dev={}
    for k,seeds in DEV.items():
        df=produce(seeds,cal,k,causal)
        dev[k]={name:metrics(m,df) for name,m in models.items()}
    utilities={n:utility({k:v[n] for k,v in dev.items()}) for n in models}
    chosen=max(models,key=lambda n:(utilities[n],-list(models).index(n)))
    control=max((n for n in models if n.startswith("instant_")),key=lambda n:utilities[n])
    print("FUSION_SELECTED",json.dumps({"selected":chosen,"control":control,"utilities":utilities,"manifest":models[chosen].manifest()}),flush=True)
    compare={"prior":prior,"instant_recalibrated":models[control],"selected":models[chosen]}
    heldout={};per_seed={}
    for k,seeds in TEST.items():
        df=produce(seeds,cal,k,causal)
        heldout[k]={n:metrics(m,df) for n,m in compare.items()}
        per_seed[k]={n:[{"seed":int(s),**metrics(m,x)} for s,x in df.groupby("simulation_seed")] for n,m in compare.items()}
        for n,v in heldout[k].items():
            for col in ("confirmed_TP","confirmed_FP","warning_only_TP","warning_only_FP","normal","attack","covered_attack_rows"):
                assert sum(x[col] for x in per_seed[k][n])==v[col]
        print("FUSION_HELDOUT",json.dumps({"cohort":k,**{n:compact(v) for n,v in heldout[k].items()}}),flush=True)
    selected=models[chosen]
    x=produce([55901],cal,"weak_loss25",causal)
    y=selected.predict_levels(x)
    drop=[c for c in x if c.startswith("gt_") or c in ("is_anomaly","attack_type","is_compromised","attacker_sophistication","normal_event","network_loss_multiplier")]
    assert np.array_equal(y,selected.predict_levels(x.drop(columns=drop)))
    sub=x.sample(200,random_state=19)
    assert np.array_equal(y[sub.index.to_numpy()],selected.predict_levels(sub))
    with (a.out/"physical_causal_alert_ticd.pkl").open("wb") as f:
        pickle.dump({"model":selected,"prior":prior,"control":models[control],"calibration":cal,"selected":chosen},f,pickle.HIGHEST_PROTOCOL)
    with (a.out/"physical_causal_alert_ticd.pkl").open("rb") as f:r=pickle.load(f)
    assert np.array_equal(y,r["model"].predict_levels(x))
    out={"scope":"Independent simulated SIP runs, not field validation","real_field_validation":False,
         "source_run":45359975383,"train_seeds":TRAIN,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
         "normal_train_rows":len(train),"normal_validation_rows":sum(map(len,validation.values())),
         "selected":chosen,"control":control,"manifest":selected.manifest(),"utilities":utilities,
         "normal_reference_metrics":reference_metrics,"development":dev,"heldout":heldout,"per_seed":per_seed,
         "warning_reference_frozen":True,"all_roles_strong_alert":True,"heldout_used_for_selection":False,
         "label_invariance":True,"batch_invariance":True,"model_reload":True}
    (a.out/"physical_causal_alert_report.json").write_text(json.dumps(out,indent=2)+"\n")
    print("FUSION_FINAL",json.dumps({"selected":chosen,"control":control,"heldout":{k:{n:compact(v) for n,v in pair.items()} for k,pair in heldout.items()},"audits_passed":True}),flush=True)
if __name__=="__main__":main()
