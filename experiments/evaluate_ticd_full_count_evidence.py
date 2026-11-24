#!/usr/bin/env python3
"""Full TICD + sample-size-aware evidence; separate train/val/dev/test."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from ticd_full_count_evidence import CountReference,FullCountTICD,WINDOWS
from ticd_invariant_ablation import InvariantAblationTICD
from ticd_observability_factorial import PolicyEvidence
from evaluate_ticd_observability_factorial import group,evaluation,short,master_metrics
from simulate_ticd_extended_observation import KINDS
from evaluate_ticd_two_tier import metrics

LOSSES=(1,5,7,10,12,15,20,25)
NATIVE=tuple(k for k in KINDS if k=="default" or k.startswith(("role_","group_","weak_")) or k=="variable_deficit")
POLICY=("policy_route","policy_binding","policy_group","observer_outage_policy")
DEV_KINDS=tuple(k for k in KINDS if k in NATIVE or k in POLICY or k in ("clean_loss7","clean_loss25"))
TRAIN={m:[101001+100*i+j for j in range(3)] for i,m in enumerate(LOSSES)}
VAL={m:[102001+100*i+j for j in range(3)] for i,m in enumerate(LOSSES)}
DEV={k:[104001+100*i+j for j in range(3)] for i,k in enumerate(DEV_KINDS)}
TEST={k:[107001+100*i+j for j in range(5)] for i,k in enumerate(KINDS)}
SPECS={"control":((),0.),"short16":((16,),1.),"mid48":((48,),1.),
       "long96":((96,),1.),"multi_half":(WINDOWS,.5),"multi_full":(WINDOWS,1.)}
def pair_metrics(model,d,hybrid):
    v,cache=evaluation(model,d,hybrid)
    rows=[{"seed":int(s),**metrics(cache,g)} for s,g in d.groupby("simulation_seed")]
    for c in ("normal","attack","confirmed_TP","confirmed_FP","warning_only_FP","warning_only_TP"):
        assert sum(r[c] for r in rows)==v[c]
    return v,rows
def utility(name,dev,per,hybrid):
    # Predeclared priorities: hard-to-observe roles and small/variable
    # effects, with costs for violations of FPR/FWR at run and cohort level.
    recall=lambda k:dev[k][name]["confirmed_recall"]
    high=[k for k in NATIVE if k.startswith("role_") and k.endswith("loss25")]
    low=[k for k in NATIVE if k.startswith("weak_")]
    r=.4*np.mean([recall(k) for k in high])+.3*np.mean([recall(k) for k in low])+.2*np.mean([recall(k) for k in ("group_majority","group_zone","variable_deficit")])+.1*np.mean([recall(k) for k in NATIVE])
    ks=list(NATIVE)+["clean_loss7","clean_loss25"]
    if hybrid:
        r=.75*r+.25*np.mean([recall(k) for k in POLICY]);ks+=list(POLICY)
    q=[dev[k][name] for k in ks];runs=[v for k in ks for v in per[k][name]]
    f=np.array([v["confirmed_alert_FPR"] for v in runs]);w=np.array([v["false_warning_rate"] for v in runs])
    excess=15*np.mean(np.maximum(f-.01,0))+8*np.mean(np.maximum(w-.005,0))
    excess+=15*np.mean([max(v["confirmed_alert_FPR"]-.01,0) for v in q])+8*np.mean([max(v["false_warning_rate"]-.005,0) for v in q])
    return float(r-excess-.5*f.mean())
def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"role_adaptive_causal_ticd.pkl").open("rb") as f:pkg=pickle.load(f)
    assert pkg["selected"]=="weighted_w0.5_g0.0_b0.008"
    source=pkg["model"];cal=pkg["calibration"]
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    causal=source.prior.prior.prior.source_model.causal_model
    seeds=[s for part in (TRAIN,VAL,DEV,TEST) for xs in part.values() for s in xs]
    assert len(seeds)==len(set(seeds))==236 and min(seeds)>97901
    tr_groups={m:group(ls,cal,"clean_loss"+str(m),causal) for m,ls in TRAIN.items()}
    train=pd.concat([x[0] for x in tr_groups.values()],ignore_index=True)
    master_train=pd.concat([x[1] for x in tr_groups.values()],ignore_index=True)
    val_groups={m:group(ls,cal,"clean_loss"+str(m),causal) for m,ls in VAL.items()}
    val={m:x[0] for m,x in val_groups.items()};master_val={m:x[1] for m,x in val_groups.items()}
    # Verify counter consistency with original measured ratio, not gt_*.
    def check_counts(x):
        n=x.cu_n_200_transit.to_numpy();b=x.cu_n_ack_transit.to_numpy()
        valid=(x.cu_delivered.to_numpy()>0)
        assert np.all(n>=b) and np.all(b>=0)
        assert np.all(n[~valid]==0) and np.all(b[~valid]==0)
        assert np.allclose(x.loc[valid,"cu_dcr_transit"],b[valid]/np.maximum(n[valid],1))
    check_counts(train)
    for x in val.values():check_counts(x)
    ref=CountReference().fit(train)
    pv={m:ref.prepare(x) for m,x in val.items()}
    core=InvariantAblationTICD(source,"full",refit=True).fit_normal(train,val)
    frozen=InvariantAblationTICD(source,"full",refit=False)
    audit=PolicyEvidence().fit(train,val);master_audit=PolicyEvidence().fit(master_train,master_val)
    candidates={}
    for name,(windows,weight) in SPECS.items():
        for h in (False,True):
            key=("audit__" if h else "primary__")+name
            candidates[key]=FullCountTICD(core,windows,weight,audit if h else None).calibrate(pv)
            print("COUNT_REFERENCE",json.dumps({"name":key,"manifest":candidates[key].manifest()}),flush=True)
    dev={};dev_per={}
    for k,ls in DEV.items():
        x,_,_=group(ls,cal,k,causal);check_counts(x);x=ref.prepare(x)
        dev[k]={};dev_per[k]={}
        for name,mod in candidates.items():
            v,r=pair_metrics(mod,x,name.startswith("audit__"));dev[k][name]=v;dev_per[k][name]=r
        print("COUNT_DEV",json.dumps({"cohort":k,**{n:short(v) for n,v in dev[k].items()}}),flush=True)
    utilities={n:utility(n,dev,dev_per,n.startswith("audit__")) for n in candidates}
    primary=max((n for n in candidates if n.startswith("primary__")),key=lambda n:(utilities[n],-list(candidates).index(n)))
    hybrid=max((n for n in candidates if n.startswith("audit__")),key=lambda n:(utilities[n],-list(candidates).index(n)))
    print("COUNT_SELECTED",json.dumps({"primary":primary,"hybrid":hybrid,"utilities":utilities}),flush=True)
    models={"frozen_full":frozen,"primary_control":candidates["primary__control"],
            "primary_selected":candidates[primary],"audit_control":candidates["audit__control"],
            "audit_selected":candidates[hybrid]}
    # Independent audit: prefix-causal preparation, exact frozen equality,
    # identical telemetry counterfactual, label removal and prepared-row batching.
    x,_,_=group([110901],cal,"clean_loss10",causal)
    l,_,_=group([110901],cal,"latent_identical",causal)
    labels=["is_anomaly","attack_type","attacker_sophistication"]
    pd.testing.assert_frame_equal(x.drop(columns=labels),l.drop(columns=labels))
    xx=ref.prepare(x);ll=ref.prepare(l)
    pref=ref.prepare(x.loc[x.interval<180])
    for w in WINDOWS:
        c="count_norm_"+str(w)
        assert np.allclose(pref[c],xx.loc[pref.index,c],rtol=0,atol=1e-10)
    assert np.array_equal(source.predict_levels(x),frozen.predict_levels(x))
    assert np.array_equal(core.predict_levels(x),models["primary_control"].predict_levels(xx))
    assert np.array_equal(models["audit_control"].components(xx)[0],models["audit_selected"].components(xx)[0])
    for n,mod in models.items():
        z=mod.predict_levels(xx)
        assert np.array_equal(z,mod.predict_levels(ll))
        bad=[c for c in xx if c.startswith("gt_") or c in labels+["is_compromised","normal_event"]]
        assert np.array_equal(z,mod.predict_levels(xx.drop(columns=bad)))
        sub=xx.sample(n=200,random_state=42)
        assert np.array_equal(z[sub.index.to_numpy()],mod.predict_levels(sub))
    heldout={};per_seed={};masters={};meta={}
    for k,ls in TEST.items():
        x,m,md=group(ls,cal,k,causal);check_counts(x);x=ref.prepare(x)
        meta[k]=md;heldout[k]={};per_seed[k]={}
        for n,mod in models.items():
            v,r=pair_metrics(mod,x,n.startswith("audit_"))
            heldout[k][n]=v;per_seed[k][n]=r
        masters[k]=master_metrics(master_audit,m)
        print("COUNT_HELDOUT",json.dumps({"cohort":k,**{n:short(v) for n,v in heldout[k].items()},"master":masters[k]}),flush=True)
    paired={}
    for family,ks in (("primary",NATIVE),("audit",NATIVE+POLICY)):
        for metric in ("confirmed_recall","confirmed_alert_FPR"):
            diffs=[]
            for k in ks:
                u=per_seed[k][family+"_selected"];v=per_seed[k][family+"_control"]
                diffs.append(np.array([a[metric]-b[metric] for a,b in zip(u,v)]))
            d=np.concatenate(diffs);rng=np.random.default_rng(13001)
            boot=np.mean([g[rng.integers(0,len(g),(4000,len(g)))].mean(1) for g in diffs],axis=0)
            paired[family+"_"+metric]={"delta_selected_minus_control":float(d.mean()),"n":len(d),
               "CI95":np.quantile(boot,[.025,.975]).tolist(),"p_unadjusted":1. if np.all(d==0) else float(wilcoxon(d).pvalue)}
    prev=0.
    for i,v in enumerate(sorted(paired.values(),key=lambda x:x["p_unadjusted"])):
        prev=max(prev,min(1.,(len(paired)-i)*v["p_unadjusted"]));v["p_holm"]=prev
    summary={}
    for n in models:
        rs=[r for k in TEST for r in per_seed[k][n]]
        summary[n]={"max_FPR":max(r["confirmed_alert_FPR"] for r in rs),"mean_FPR":float(np.mean([r["confirmed_alert_FPR"] for r in rs])),
           "FPR_pass":sum(r["confirmed_alert_FPR"]<=.01 for r in rs),"FWR_pass":sum(r["false_warning_rate"]<=.005 for r in rs)}
        print("COUNT_SUMMARY",json.dumps({"model":n,**summary[n]}),flush=True)
    path=a.out/"full_count_ticd_models.pkl"
    with path.open("wb") as f:pickle.dump({"models":models,"reference":ref,"master_audit":master_audit,"calibration":cal},f,pickle.HIGHEST_PROTOCOL)
    with path.open("rb") as f:reload=pickle.load(f)
    for n,mod in models.items():assert np.array_equal(mod.predict_levels(xx),reload["models"][n].predict_levels(xx))
    report={"scope":"synthetic SIP count evidence and separately simulated audit","source_run":45377337545,
       "all_original_invariants_preserved":True,"real_field_validation":False,"no_native_SBC_compromise_model":True,
       "external_audit_is_synthetic":True,"test_used_for_selection":False,
       "training_seeds":TRAIN,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
       "count_reference":ref.manifest(),"candidates":{n:m.manifest() for n,m in candidates.items()},
       "utilities":utilities,"selected_primary":primary,"selected_audit":hybrid,
       "development":dev,"development_by_seed":dev_per,"heldout":heldout,"per_seed":per_seed,
       "master_audit":masters,"metadata":meta,"paired":paired,"summary":summary,
       "counter_consistency":True,"prefix_causality":True,"observational_equivalence_verified":True,
       "label_and_batch_invariance":True,"reload_verified":True}
    (a.out/"full_count_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("COUNT_FINAL",json.dumps({"main_runs":len(seeds),"test_runs":125,"primary":primary,"hybrid":hybrid,"paired":paired,"audits_passed":True}),flush=True)
if __name__=="__main__":main()
