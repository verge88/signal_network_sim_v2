#!/usr/bin/env python3
"""Paired study of separate channel calibration and bounded count memory."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from ticd_separate_robust_fusion import BoundedCountReference,SeparateChannelTICD,MODES,prepare_channel_scores
from evaluate_ticd_observability_factorial import group,evaluation,short,master_metrics
from evaluate_ticd_full_count_evidence import NATIVE,POLICY,DEV_KINDS,LOSSES,pair_metrics
from simulate_ticd_extended_observation import KINDS

TRAIN={m:[121001+100*i+j for j in range(3)] for i,m in enumerate(LOSSES)}
VAL={m:[122001+100*i+j for j in range(4)] for i,m in enumerate(LOSSES)}
DEV={k:[124001+100*i+j for j in range(4)] for i,k in enumerate(DEV_KINDS)}
TEST={k:[127001+100*i+j for j in range(6)] for i,k in enumerate(KINDS)}
BUDGETS=((.004,.004,0.),(.003,.005,0.),(.003,.003,.002),(.002,.003,.003),(.002,.002,.004))
CAL_MODES=("pooled","p90run")

def quick(model,df,hybrid):
    valid=df.primary_observer_valid.to_numpy(dtype=bool)
    if hybrid:z=model.predict_levels(df)
    else:
        z=np.zeros(len(df),dtype=np.int8)
        if valid.any():z[valid]=model.predict_levels(df.loc[valid])
    y=df.is_anomaly.to_numpy(dtype=bool)
    def stat(ix):
        a=z[ix]==2;w=z[ix]==1;b=y[ix]
        return {"confirmed_recall":float(a[b].mean()) if b.any() else None,
                "confirmed_alert_FPR":float(a[~b].mean()),
                "false_warning_rate":float(w[~b].mean())}
    return stat(np.ones(len(df),dtype=bool)),[{"seed":int(s),**stat(df.simulation_seed.to_numpy()==s)} for s in df.simulation_seed.unique()]

def selection(name,dev,per,hybrid):
    ks=list(NATIVE)+["clean_loss7","clean_loss25"]+(list(POLICY) if hybrid else [])
    ref="prior_audit_count" if hybrid else "prior_primary_count"
    high=[k for k in NATIVE if k.startswith("role_") and k.endswith("loss25")]
    low=[k for k in NATIVE if k.startswith("weak_")]
    group_k=("group_majority","group_zone","variable_deficit")
    r=lambda k,n:dev[k][n]["confirmed_recall"]
    native=lambda n:float(.4*np.mean([r(k,n) for k in high])+.3*np.mean([r(k,n) for k in low])+.2*np.mean([r(k,n) for k in group_k])+.1*np.mean([r(k,n) for k in NATIVE]))
    n=native(name);nref=native(ref)
    pol=float(np.mean([r(k,name) for k in POLICY])) if hybrid else None
    pref=float(np.mean([r(k,"prior_audit_control") for k in POLICY])) if hybrid else None
    f=np.array([v["confirmed_alert_FPR"] for k in ks for v in per[k][name]])
    w=np.array([v["false_warning_rate"] for k in ks for v in per[k][name]])
    default_max=max(v["confirmed_alert_FPR"] for v in per["default"][name])
    # Prefer feasible development candidates. Explicitly expose failure
    # when no candidate jointly satisfies these targets.
    recall_short=max(0.,nref-.03-n)+(max(0.,pref-.03-pol) if hybrid else 0.)
    default_excess=max(0.,default_max-.01)
    feasible=recall_short<=1e-12 and default_excess<=1e-12
    value=(.75*n+.25*pol if hybrid else n)-20*default_excess-15*np.maximum(f-.01,0).mean()-8*np.maximum(w-.005,0).mean()-.5*f.mean()-2*recall_short
    return {"feasible":bool(feasible),"utility":float(value),"native_weighted_recall":n,"prior_native_weighted_recall":nref,
            "policy_recall":pol,"policy_target_reference":pref,"recall_shortfall":recall_short,
            "default_max_FPR":default_max,"FPR_pass":int((f<=.01).sum()),"FWR_pass":int((w<=.005).sum()),
            "max_FPR":float(f.max()),"max_FWR":float(w.max())}

def diagnostics(df,model,hybrid):
    # Label-based attribution for evaluation ONLY; no feature is returned to
    # the detector. Report alert persistence after a labelled episode ends.
    valid=df.primary_observer_valid.to_numpy(dtype=bool)
    if hybrid:z=model.predict_levels(df)
    else:
        z=np.zeros(len(df),dtype=np.int8)
        if valid.any():z[valid]=model.predict_levels(df.loc[valid])
    x=df[["simulation_seed","node_id","node_type","interval","is_anomaly","normal_event"]].copy()
    x["alert"]=z==2
    if isinstance(model,SeparateChannelTICD):
        for k,v in model.channels(df).items():x[k]=v
    else:
        a,b,c=model.components(df)
        x["core"]=a>model.threshold;x["count"]=b>model.threshold;x["audit"]=c>model.threshold
    x=x.sort_values(["simulation_seed","node_id","interval"],kind="stable")
    x["last_positive"]=x.interval.where(x.is_anomaly>0)
    x["last_positive"]=x.groupby(["simulation_seed","node_id"],sort=False).last_positive.ffill()
    x["recent_positive"]=(x.interval-x.last_positive).between(1,48)
    x=x[x.is_anomaly==0]
    def counts(g):
        return {"normal":len(g),"FP":int(g.alert.sum()),
                "core_FP":int((g.alert&g.core).sum()),"count_FP":int((g.alert&g["count"]).sum()),
                "audit_FP":int((g.alert&g.audit).sum()),
                "count_only_FP":int((g.alert&g["count"]&~g.core&~g.audit).sum())}
    return {"post_attack_48":counts(x[x.recent_positive]),"other":counts(x[~x.recent_positive]),
            "roles":{str(k):counts(g) for k,g in x.groupby("node_type")},
            "benign_events":{str(k):counts(g) for k,g in x.groupby("normal_event")}}

def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"full_count_ticd_models.pkl").open("rb") as f:pkg=pickle.load(f)
    old=pkg["models"];base=old["primary_selected"];old_hybrid=old["audit_selected"]
    assert tuple(base.windows)==tuple(old_hybrid.windows)==(48,) and base.core.variant=="full"
    assert old_hybrid.audit is not None
    cal=pkg["calibration"];core=base.core;audit=old_hybrid.audit;prior_ref=pkg["reference"]
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    causal=old["frozen_full"].causal
    seeds=[s for part in (TRAIN,VAL,DEV,TEST) for xs in part.values() for s in xs]
    assert len(seeds)==len(set(seeds))==290 and min(seeds)>110901
    train=pd.concat([group(ls,cal,"clean_loss"+str(m),causal)[0] for m,ls in TRAIN.items()],ignore_index=True)
    ref=BoundedCountReference(prior_ref).fit(train)
    prepare=lambda x:prepare_channel_scores(ref.prepare(x),core,audit)
    val={m:prepare(group(ls,cal,"clean_loss"+str(m),causal)[0]) for m,ls in VAL.items()}
    models={}
    for mode in MODES:
        for b in BUDGETS:
            for cm in CAL_MODES:
                name=("audit" if b[2]>0 else "primary")+"__"+mode+"__"+ "_".join(str(int(x*1000)) for x in b)+"__"+cm
                models[name]=SeparateChannelTICD(mode,b,cm).calibrate(val)
    controls={"prior_primary_count":base,"prior_audit_count":old_hybrid,"prior_audit_control":old["audit_control"]}
    print("SPLIT_REFERENCE",json.dumps({"candidates":len(models),"reference":ref.manifest(),
        "manifests":{n:m.manifest() for n,m in models.items()}}),flush=True)
    dev={};dev_per={};diagnostic={}
    for k,ls in DEV.items():
        x,_,_=group(ls,cal,k,causal);x=prepare(x)
        dev[k]={};dev_per[k]={}
        for n,m in {**controls,**models}.items():
            hybrid=n.startswith(("audit__","prior_audit"))
            v,r=quick(m,x,hybrid);dev[k][n]=v;dev_per[k][n]=r
        if k=="default":
            diagnostic["development_prior"]={n:diagnostics(x,m,n.startswith("prior_audit")) for n,m in controls.items()}
            print("SPLIT_DIAG",json.dumps(diagnostic["development_prior"]),flush=True)
        print("SPLIT_DEV",json.dumps({"cohort":k,"candidates":len(models),
            "prior":{n:dev[k][n] for n in controls}}),flush=True)
    select={n:selection(n,dev,dev_per,m.hybrid) for n,m in models.items()}
    def choose(h,raw=False):
        return max((n for n,m in models.items() if m.hybrid==h and (not raw or m.count_mode=="raw48")),
            key=lambda n:(select[n]["feasible"],select[n]["utility"],-list(models).index(n)))
    primary=choose(False);hybrid=choose(True);raw_primary=choose(False,True);raw_hybrid=choose(True,True)
    chosen={**controls,"split_raw_primary":models[raw_primary],"split_raw_audit":models[raw_hybrid],
            "selected_primary":models[primary],"selected_audit":models[hybrid]}
    print("SPLIT_SELECTED",json.dumps({"primary":primary,"hybrid":hybrid,"raw_primary":raw_primary,"raw_hybrid":raw_hybrid,
        "primary_selection":select[primary],"hybrid_selection":select[hybrid],
        "feasible_primary_count":sum(v["feasible"] for n,v in select.items() if n.startswith("primary__")),
        "feasible_audit_count":sum(v["feasible"] for n,v in select.items() if n.startswith("audit__"))}),flush=True)
    # Audit causal preparation and exactly indistinguishable telemetry.
    x,_,_=group([131901],cal,"clean_loss10",causal);l,_,_=group([131901],cal,"latent_identical",causal)
    labels=["is_anomaly","attack_type","attacker_sophistication"]
    pd.testing.assert_frame_equal(x.drop(columns=labels),l.drop(columns=labels))
    xx=prepare(x);ll=prepare(l);pref=prepare(x.loc[x.interval<180])
    for c in (*MODES,"split_core","split_audit","split_warning"):
        assert np.allclose(pref[c],xx.loc[pref.index,c],rtol=0,atol=1e-10)
    bad=[c for c in x if c.startswith("gt_") or c in labels+["is_compromised","normal_event"]]
    stripped=prepare(x.drop(columns=bad))
    for n,m in chosen.items():
        z=m.predict_levels(xx)
        assert np.array_equal(z,m.predict_levels(ll))
        assert np.array_equal(z,m.predict_levels(stripped))
        sub=xx.sample(n=200,random_state=71);assert np.array_equal(z[sub.index.to_numpy()],m.predict_levels(sub))
    heldout={};per_seed={};masters={};meta={}
    for k,ls in TEST.items():
        x,m,md=group(ls,cal,k,causal);x=prepare(x)
        meta[k]=md;heldout[k]={};per_seed[k]={}
        for n,mod in chosen.items():
            h=n in ("prior_audit_count","prior_audit_control","split_raw_audit","selected_audit")
            v,r=pair_metrics(mod,x,h);heldout[k][n]=v;per_seed[k][n]=r
        if k=="default":
            diagnostic["heldout"]={n:diagnostics(x,mod,n in ("prior_audit_count","prior_audit_control","split_raw_audit","selected_audit")) for n,mod in chosen.items()}
            print("SPLIT_DIAG_HELDOUT",json.dumps(diagnostic["heldout"]),flush=True)
        masters[k]=master_metrics(pkg["master_audit"],m)
        print("SPLIT_HELDOUT",json.dumps({"cohort":k,**{n:short(v) for n,v in heldout[k].items()}}),flush=True)
    paired={}
    for family,ks in (("primary",NATIVE),("audit",NATIVE+POLICY)):
        for metric in ("confirmed_recall","confirmed_alert_FPR"):
            ds=[]
            for k in ks:
                u=per_seed[k]["selected_"+family];v=per_seed[k]["prior_"+family+"_count"]
                ds.append(np.array([a[metric]-b[metric] for a,b in zip(u,v)]))
            d=np.concatenate(ds);rng=np.random.default_rng(7303)
            boot=np.mean([g[rng.integers(0,len(g),(4000,len(g)))].mean(1) for g in ds],axis=0)
            paired[family+"_"+metric]={"delta_selected_minus_prior":float(d.mean()),"n":len(d),
                "CI95":np.quantile(boot,[.025,.975]).tolist(),"p_unadjusted":1. if np.all(d==0) else float(wilcoxon(d).pvalue)}
    prev=0.
    for i,v in enumerate(sorted(paired.values(),key=lambda x:x["p_unadjusted"])):
        prev=max(prev,min(1.,(len(paired)-i)*v["p_unadjusted"]));v["p_holm"]=prev
    summary={}
    for n in chosen:
        rs=[r for k in TEST for r in per_seed[k][n]]
        summary[n]={"mean_FPR":float(np.mean([r["confirmed_alert_FPR"] for r in rs])),
          "max_FPR":max(r["confirmed_alert_FPR"] for r in rs),"max_FWR":max(r["false_warning_rate"] for r in rs),
          "FPR_pass":sum(r["confirmed_alert_FPR"]<=.01 for r in rs),"FWR_pass":sum(r["false_warning_rate"]<=.005 for r in rs)}
        print("SPLIT_SUMMARY",json.dumps({"model":n,**summary[n]}),flush=True)
    path=a.out/"separate_robust_models.pkl"
    with path.open("wb") as f:pickle.dump({"models":chosen,"reference":ref,"core":core,"audit":audit,
        "calibration":cal,"master_audit":pkg["master_audit"]},f,pickle.HIGHEST_PROTOCOL)
    with path.open("rb") as f:reload=pickle.load(f)
    for n,m in chosen.items():assert np.array_equal(m.predict_levels(xx),reload["models"][n].predict_levels(xx))
    report={"source_run":45438008085,"scope":"synthetic SIP, frozen full TICD and synthetic audit; robust count and split calibration",
      "real_field_validation":False,"all_original_invariants_preserved":True,"external_audit_is_synthetic":True,
      "test_used_for_selection":False,"training_seeds":TRAIN,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
      "reference":ref.manifest(),"candidates":{n:m.manifest() for n,m in models.items()},"selection":select,
      "selected":{"primary":primary,"audit":hybrid,"raw_primary":raw_primary,"raw_audit":raw_hybrid},
      "development":dev,"development_by_seed":dev_per,"heldout":heldout,"per_seed":per_seed,"summary":summary,
      "diagnostics":diagnostic,"paired":paired,"metadata":meta,"master_audit":masters,
      "prefix_causality":True,"observational_equivalence_verified":True,"label_and_batch_invariance":True,"reload_verified":True}
    (a.out/"separate_robust_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("SPLIT_FINAL",json.dumps({"main_runs":len(seeds),"test_runs":150,"candidates":len(models),
          "selected":report["selected"],"paired":paired,"audits_passed":True}),flush=True)
if __name__=="__main__":main()
