#!/usr/bin/env python3
"""Complete paired component ablation; all policies and seeds prespecified."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from ticd_invariant_ablation import InvariantAblationTICD,VARIANTS
from evaluate_ticd_zonal_warnings import produce
from evaluate_ticd_two_tier import metrics
from evaluate_ticd_high_loss_alert import compact

LOSSES=(1,5,7,10,12,15,20,25)
KINDS=("default","weak_loss5","weak_loss10","weak_loss15","weak_loss20","weak_loss25",
       "truthful","clean_loss1","clean_loss7","clean_loss12","clean_loss20","clean_loss25")
TRAIN={m:[81001+100*i+j for j in range(4)] for i,m in enumerate(LOSSES)}
VAL={m:[82001+100*i+j for j in range(4)] for i,m in enumerate(LOSSES)}
TEST={k:[84001+100*i+j for j in range(6)] for i,k in enumerate(KINDS)}
POLICIES=("frozen","refit")

class CachedDecision:
    def __init__(self,df,levels):
        assert df.index.is_unique
        self.levels=pd.Series(levels,index=df.index)
    def predict_masks(self,df):
        y=self.levels.loc[df.index].to_numpy()
        return {"normal":y==0,"warning_only":y==1,"confirmed_alert":y==2,"any_signal":y>0}

def evaluate(mod,df):
    y=mod.predict_levels(df)
    cache=CachedDecision(df,y)
    total=metrics(cache,df)
    runs=[{"seed":int(s),**metrics(cache,x)} for s,x in df.groupby("simulation_seed")]
    for key in ("normal","attack","confirmed_FP","confirmed_TP","warning_only_FP","warning_only_TP","covered_attack_rows"):
        assert sum(x[key] for x in runs)==total[key]
    return total,runs

def effect_report(per_seed,policy):
    # Pairwise effects by independent simulator run, not by node-time row.
    # Macro average across scenarios; stratified paired bootstrap for intervals.
    out={}
    rng=np.random.default_rng(9271)
    tests=[]
    for variant in VARIANTS:
        if variant=="full":continue
        key=policy+"__"+variant;base=policy+"__full"
        out[variant]={}
        for label,cohorts,get in [
            ("FPR",KINDS,lambda x:x["confirmed_alert_FPR"]),
            ("FWR",KINDS,lambda x:x["false_warning_rate"]),
            ("statistical_recall",tuple("weak_loss"+str(m) for m in (5,10,15,20,25)),
                lambda x:x["by_sophistication"]["statistical"]["confirmed_recall"]),
            ("high_loss_statistical_recall",("weak_loss20","weak_loss25"),
                lambda x:x["by_sophistication"]["statistical"]["confirmed_recall"]),
            ("truthful_compromise_recall",("truthful",),
                lambda x:x["by_attack"]["proxy_compromise"]["confirmed_recall"]),
            ("attack_coverage",KINDS[:7],lambda x:x["combined_attack_coverage"])]:
            ds=[]
            for k in cohorts:
                x=per_seed[k][base];y=per_seed[k][key]
                assert [r["seed"] for r in x]==[r["seed"] for r in y]
                ds.append(np.array([get(a)-get(b) for a,b in zip(x,y)],dtype=float))
            flat=np.concatenate(ds)
            assert np.isfinite(flat).all()
            boot=np.mean([d[rng.integers(0,len(d),size=(4000,len(d)))].mean(axis=1) for d in ds],axis=0)
            p=1. if np.all(flat==0) else float(wilcoxon(flat,zero_method="wilcox",alternative="two-sided").pvalue)
            obj={"difference_full_minus_ablated":float(np.mean([d.mean() for d in ds])),
                 "bootstrap_95":[float(z) for z in np.quantile(boot,[.025,.975])],
                 "n_runs":len(flat),"p_unadjusted":p,
                 "positive":int((flat>0).sum()),"negative":int((flat<0).sum()),"ties":int((flat==0).sum())}
            out[variant][label]=obj;tests.append(obj)
    # Holm adjustment within each policy across all declared paired tests.
    prev=0.
    for i,obj in enumerate(sorted(tests,key=lambda x:x["p_unadjusted"])):
        prev=max(prev,min(1.,(len(tests)-i)*obj["p_unadjusted"]))
        obj["p_holm"]=prev
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"role_adaptive_causal_ticd.pkl").open("rb") as f:pkg=pickle.load(f)
    assert pkg["selected"]=="weighted_w0.5_g0.0_b0.008"
    source=pkg["model"];cal=pkg["calibration"]
    assert cal["calibration_kind"]=="partial_benign_SIP_method_profile"
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    seeds=[s for part in (TRAIN,VAL,TEST) for ls in part.values() for s in ls]
    assert len(seeds)==len(set(seeds))==136 and min(seeds)>75901
    causal=source.prior.prior.prior.source_model.causal_model
    train=pd.concat([produce(ls,cal,"clean_loss"+str(m),causal) for m,ls in TRAIN.items()],ignore_index=True)
    validation={m:produce(ls,cal,"clean_loss"+str(m),causal) for m,ls in VAL.items()}
    models={};refs={}
    for policy in POLICIES:
        for variant in VARIANTS:
            name=policy+"__"+variant
            mod=InvariantAblationTICD(source,variant,refit=policy=="refit")
            if policy=="refit":mod.fit_normal(train,validation)
            models[name]=mod
            ref={str(m):compact(metrics(mod,df)) for m,df in validation.items()}
            refs[name]={"manifest":mod.manifest(),"validation":ref}
            print("ABLATION_REFERENCE",json.dumps({"name":name,"manifest":mod.manifest(),
                 "max_FPR":max(v["FPR"] for v in ref.values()),"max_FWR":max(v["FWR"] for v in ref.values())}),flush=True)
    # Independent audit, before final test. Exact equality to shipped source,
    # and total removal of all active evidence (both strong and warning).
    audit=produce([85901],cal,"weak_loss25",causal)
    assert np.array_equal(source.predict_levels(audit),models["frozen__full"].predict_levels(audit))
    for policy in POLICIES:
        null=models[policy+"__no_all_invariants"]
        assert not null.predict_levels(audit).any()
        assert not null.score(audit).any()
        assert not null.warning(audit)[0].any()
    for name,mod in models.items():
        y=mod.predict_levels(audit)
        drop=[c for c in audit if c.startswith("gt_") or c in ("is_anomaly","attack_type","is_compromised",
              "attacker_sophistication","normal_event","network_loss_multiplier")]
        assert np.array_equal(y,mod.predict_levels(audit.drop(columns=drop)))
        sub=audit.sample(200,random_state=61)
        assert np.array_equal(y[sub.index.to_numpy()],mod.predict_levels(sub))
    heldout={};by_seed={}
    for k,ls in TEST.items():
        df=produce(ls,cal,k,causal)
        heldout[k]={};by_seed[k]={}
        for name,mod in models.items():
            v,r=evaluate(mod,df);heldout[k][name]=v;by_seed[k][name]=r
        print("ABLATION_HELDOUT",json.dumps({"cohort":k,**{n:compact(v) for n,v in heldout[k].items()}}),flush=True)
    effects={p:effect_report(by_seed,p) for p in POLICIES}
    summary={}
    for name in models:
        rs=[x for d in by_seed.values() for x in d[name]]
        summary[name]={"FPR_mean_run":float(np.mean([x["confirmed_alert_FPR"] for x in rs])),
           "FWR_mean_run":float(np.mean([x["false_warning_rate"] for x in rs])),
           "FPR_max_run":max(x["confirmed_alert_FPR"] for x in rs),
           "FPR_pass":sum(x["confirmed_alert_FPR"]<=.01 for x in rs),
           "FWR_pass":sum(x["false_warning_rate"]<=.005 for x in rs),
           "statistical_recall":{k:heldout[k][name]["by_sophistication"]["statistical"]["confirmed_recall"] for k in KINDS if k.startswith("weak_loss")},
           "truthful_compromise_recall":heldout["truthful"][name]["by_attack"]["proxy_compromise"]["confirmed_recall"],
           "ordinary_attack_recall":heldout["default"][name]["confirmed_recall"],
           "false_alert_episodes":sum(x["false_alert_episodes"] for x in rs)}
        print("ABLATION_SUMMARY",json.dumps({"model":name,**summary[name]}),flush=True)
    with (a.out/"invariant_ablation_models.pkl").open("wb") as f:pickle.dump(models,f,pickle.HIGHEST_PROTOCOL)
    with (a.out/"invariant_ablation_models.pkl").open("rb") as f:reloaded=pickle.load(f)
    for n in models:assert np.array_equal(models[n].predict_levels(audit),reloaded[n].predict_levels(audit))
    report={"scope":"paired synthetic component ablation, not real attack validation","source_run":45377337545,
       "training_seeds":TRAIN,"validation_seeds":VAL,"test_seeds":TEST,"variants":VARIANTS,"policies":POLICIES,
       "calibration_budgets":{"FPR":.008,"FWR":.005},"reference":refs,"heldout":heldout,"per_seed":by_seed,
       "summary":summary,"paired_effects":effects,"selection_on_test":False,
       "real_field_validation":False,"exact_frozen_full":True,"null_control_verified":True,
       "label_invariance":True,"batch_invariance":True,"reload_verified":True}
    (a.out/"invariant_ablation_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("ABLATION_FINAL",json.dumps({"variants":len(VARIANTS),"policies":len(POLICIES),
          "test_runs":72,"audits_passed":True}),flush=True)
if __name__=="__main__":main()
