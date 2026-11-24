#!/usr/bin/env python3
"""Role-adaptive causal alert ablation, normal calibration and new heldout.
Earlier disclosed test data motivated the hypothesis, never select the
new policy on those observations. The old model and references stay frozen.
"""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from ticd_role_adaptive_causal_alert import RoleAdaptiveCausalAlertTICD
from evaluate_ticd_zonal_warnings import produce
from evaluate_ticd_two_tier import metrics
from evaluate_ticd_high_loss_alert import compact

LOSSES=(1,5,7,10,12,15,20,25)
VAL={m:[61001+100*i+j for j in range(3)] for i,m in enumerate(LOSSES)}
DEV={k:[62001+100*i+j for j in range(3)] for i,k in enumerate((
    "default","weak_loss5","weak_loss10","weak_loss15","weak_loss20",
    "weak_loss25","truthful","clean_loss1","clean_loss7","clean_loss20","clean_loss25"))}
TEST={k:[64001+100*i+j for j in range(4)] for i,k in enumerate((
    "default","weak_loss5","weak_loss10","weak_loss15","weak_loss20",
    "weak_loss25","truthful","clean_loss1","clean_loss7","clean_loss12",
    "clean_loss20","clean_loss25"))}
SPECS=[
    ("uniform",1.,0.),
    ("weighted",.85,0.),
    ("weighted",.70,0.),
    ("weighted",.50,0.),
    ("gated",1.,.25),
    ("gated",1.,.50),
    ("joint",.70,.25),
]
BUDGETS=(.006,.008)

def objective(pooled,paired):
    # Predeclared: reward weak compromise detection; penalize violations
    # both in pooled cohorts and across individual independent runs.
    attacks=[v for v in pooled.values() if v["attack"]]
    weak=[pooled[k]["by_sophistication"]["statistical"]["confirmed_recall"]
          for k in ("weak_loss10","weak_loss15","weak_loss20","weak_loss25")]
    benefit=.65*float(np.mean(weak))+.25*float(np.mean([x["confirmed_recall"] for x in attacks]))+.10*float(np.mean([x["combined_attack_coverage"] for x in attacks]))
    fprs=[x["confirmed_alert_FPR"] for x in pooled.values()]
    fwrs=[x["false_warning_rate"] for x in pooled.values()]
    seeds=[x for cohort in paired.values() for x in cohort]
    cost=18.*sum(max(0.,x-.01) for x in fprs)+6.*sum(max(0.,x-.005) for x in fwrs)
    cost+=10.*float(np.mean([max(0.,x["confirmed_alert_FPR"]-.01) for x in seeds]))
    cost+=4.*float(np.mean([max(0.,x["false_warning_rate"]-.005) for x in seeds]))
    return benefit-cost-.5*float(np.mean(fprs))

def role_stats(model,df):
    levels=model.predict_levels(df)
    y=df.is_anomaly.to_numpy(dtype=int)>0
    role=df.node_type.astype(str).to_numpy()
    out={}
    for k in sorted(set(role)):
        good=(role==k)&~y;bad=(role==k)&y
        out[k]={"normal":int(good.sum()),"attack":int(bad.sum()),
                "alert_FP":int((good&(levels==2)).sum()),"alert_TP":int((bad&(levels==2)).sum()),
                "warning_FP":int((good&(levels==1)).sum()),"warning_TP":int((bad&(levels==1)).sum())}
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prior",required=True,type=Path);p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"physical_causal_alert_ticd.pkl").open("rb") as f:pkg=pickle.load(f)
    assert pkg["selected"]=="max_short_0.008"
    prior=pkg["model"];cal=pkg["calibration"]
    assert cal["calibration_kind"]=="partial_benign_SIP_method_profile"
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    old=json.loads((a.prior/"physical_causal_alert_report.json").read_text())
    old_seeds={s for k in ("train_seeds","validation_seeds","development_seeds","test_seeds") for ls in old[k].values() for s in ls}
    seeds=[s for part in (VAL,DEV,TEST) for ls in part.values() for s in ls]
    assert len(seeds)==len(set(seeds))==105 and min(seeds)>55901 and not old_seeds.intersection(seeds)
    causal=prior.prior.prior.source_model.causal_model
    normal={m:produce(ls,cal,"clean_loss"+str(m),causal) for m,ls in VAL.items()}
    models={};normal_metrics={}
    for mode,w,g in SPECS:
        for b in BUDGETS:
            name=f"{mode}_w{w}_g{g}_b{b}"
            m=RoleAdaptiveCausalAlertTICD(prior,mode,w,g).calibrate(normal,b)
            ref={str(k):compact(metrics(m,df)) for k,df in normal.items()}
            assert max(x["FPR"] for x in ref.values())<=b+1e-12
            models[name]=m;normal_metrics[name]=ref
            print("ROLE_CAUSAL_NORMAL",json.dumps({"candidate":name,"threshold":m.threshold,"max_FPR":max(x["FPR"] for x in ref.values())}),flush=True)
    dev={};dev_seed={}
    for k,ls in DEV.items():
        df=produce(ls,cal,k,causal)
        dev[k]={n:metrics(m,df) for n,m in models.items()}
        dev_seed[k]={n:[{"seed":int(s),**metrics(m,part)} for s,part in df.groupby("simulation_seed")] for n,m in models.items()}
    u={n:objective({k:x[n] for k,x in dev.items()},{k:x[n] for k,x in dev_seed.items()}) for n in models}
    selected=max(models,key=lambda n:(u[n],-list(models).index(n)))
    control=max((n for n in models if n.startswith("uniform_")),key=lambda n:u[n])
    model=models[selected]
    print("ROLE_CAUSAL_SELECTED",json.dumps({"selected":selected,"control":control,"utilities":u,"manifest":model.manifest()}),flush=True)
    comparisons={"prior":prior,"uniform_recalibrated":models[control],"selected":model}
    heldout={};by_seed={};roles={}
    for k,ls in TEST.items():
        df=produce(ls,cal,k,causal)
        heldout[k]={n:metrics(m,df) for n,m in comparisons.items()}
        by_seed[k]={n:[{"seed":int(s),**metrics(m,part)} for s,part in df.groupby("simulation_seed")] for n,m in comparisons.items()}
        roles[k]={n:role_stats(m,df) for n,m in comparisons.items()}
        for n,v in heldout[k].items():
            for col in ("normal","attack","confirmed_FP","confirmed_TP","warning_only_FP","warning_only_TP","covered_attack_rows"):
                assert sum(x[col] for x in by_seed[k][n])==v[col]
        print("ROLE_CAUSAL_HELDOUT",json.dumps({"cohort":k,**{n:compact(v) for n,v in heldout[k].items()},"roles":roles[k]}),flush=True)
    audit=produce([65901],cal,"weak_loss25",causal)
    y=model.predict_levels(audit)
    drop=[c for c in audit if c.startswith("gt_") or c in ("is_anomaly","attack_type","is_compromised","attacker_sophistication","normal_event","network_loss_multiplier")]
    assert np.array_equal(y,model.predict_levels(audit.drop(columns=drop)))
    sub=audit.sample(200,random_state=23)
    assert np.array_equal(y[sub.index.to_numpy()],model.predict_levels(sub))
    # Role adaptation cannot suppress an instantaneous exceedance of
    # its own calibrated threshold; compare unchanged instant component.
    aa,hh=model.components(audit)
    assert np.all((aa<=model.threshold)|(y==2))
    gw=audit.node_type.astype(str).to_numpy()=="UA_GW"
    assert np.array_equal(hh[~gw],prior.reference.score(audit)[~gw]*prior.weight)
    with (a.out/"role_adaptive_causal_ticd.pkl").open("wb") as f:
        pickle.dump({"model":model,"prior":prior,"control":models[control],"selected":selected,"calibration":cal},f,pickle.HIGHEST_PROTOCOL)
    with (a.out/"role_adaptive_causal_ticd.pkl").open("rb") as f:restored=pickle.load(f)
    assert np.array_equal(y,restored["model"].predict_levels(audit))
    out={"scope":"Simulation-only role-adaptive strong alerts","source_run":45366626708,
         "diagnostic_origin":45373734770,"real_field_validation":False,
         "normal_validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
         "selected":selected,"control":control,"manifest":model.manifest(),
         "utilities":u,"normal_reference":normal_metrics,"development":dev,"development_by_seed":dev_seed,
         "heldout":heldout,"per_seed":by_seed,"roles":roles,"heldout_used_for_selection":False,
         "all_roles_instant_alert":True,"history_not_disabled":True,"warning_reference_frozen":True,
         "label_invariance":True,"batch_invariance":True,"model_reload":True}
    (a.out/"role_adaptive_causal_report.json").write_text(json.dumps(out,indent=2)+"\n")
    for n in comparisons:
        entries=[x for p in by_seed.values() for x in p[n]]
        print("ROLE_CAUSAL_SUMMARY",json.dumps({"model":n,"FPR_pass_seeds":sum(x["confirmed_alert_FPR"]<=.01 for x in entries),
          "FWR_pass_seeds":sum(x["false_warning_rate"]<=.005 for x in entries),"total_seeds":len(entries),
          "FPR_pass_cohorts":sum(p[n]["confirmed_alert_FPR"]<=.01 for p in heldout.values()),
          "FWR_pass_cohorts":sum(p[n]["false_warning_rate"]<=.005 for p in heldout.values()),
          "max_FPR_seed":max(x["confirmed_alert_FPR"] for x in entries)}),flush=True)
    print("ROLE_CAUSAL_FINAL",json.dumps({"selected":selected,"control":control,"audits_passed":True}),flush=True)
if __name__=="__main__":main()
