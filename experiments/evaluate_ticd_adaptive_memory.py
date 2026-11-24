#!/usr/bin/env python3
"""Independent evaluation of recovery-aware full TICD and sequential evidence."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from ticd_adaptive_memory import AdaptiveMemoryReference,AdaptiveMemoryTICD,SPECS
from evaluate_ticd_observability_factorial import group,short,master_metrics
from evaluate_ticd_full_count_evidence import NATIVE,POLICY,DEV_KINDS,LOSSES,pair_metrics
from evaluate_ticd_separate_robust_fusion import quick
from simulate_ticd_extended_observation import KINDS

TRAIN={m:[141001+100*i+j for j in range(3)] for i,m in enumerate(LOSSES)}
VAL={m:[142001+100*i+j for j in range(5)] for i,m in enumerate(LOSSES)}
DEV={k:[144001+100*i+j for j in range(4)] for i,k in enumerate(DEV_KINDS)}
TEST={k:[147001+100*i+j for j in range(6)] for i,k in enumerate(KINDS)}
BUDGETS=((.003,.005,0.),(.002,.006,0.),(.003,.003,.002),(.002,.003,.003))
CAL_MODES=("pooled","p90run")
WEAK=("weak_0025","weak_005")
HIGH=tuple(k for k in NATIVE if k.startswith("role_") and k.endswith("loss25"))

def selection(n,dev,per,hybrid):
    base="prior_audit" if hybrid else "prior_primary"
    ks=list(NATIVE)+["clean_loss7","clean_loss25"]+(list(POLICY) if hybrid else [])
    rec=lambda k,m:dev[k][m]["confirmed_recall"]
    mean=lambda keys,m:float(np.mean([rec(k,m) for k in keys]))
    weak=mean(WEAK,n);weak0=mean(WEAK,base)
    high=mean(HIGH,n);high0=mean(HIGH,base)
    native=mean(NATIVE,n);native0=mean(NATIVE,base)
    policy=mean(POLICY,n) if hybrid else None
    policy0=mean(POLICY,base) if hybrid else None
    f=np.array([v["confirmed_alert_FPR"] for k in ks for v in per[k][n]])
    w=np.array([v["false_warning_rate"] for k in ks for v in per[k][n]])
    default=max(v["confirmed_alert_FPR"] for v in per["default"][n])
    # Prespecify desired simultaneous constraints; never claim feasibility
    # if none meet them. Require a nonnegative weak-evidence gain and
    # no more than 3 percentage points loss on high-loss or native averages.
    rshort=max(0.,weak0-weak)+max(0.,high0-.03-high)+max(0.,native0-.03-native)
    if hybrid:rshort+=max(0.,policy0-.03-policy)
    fshort=max(0.,float(f.max())-.01)
    wshort=max(0.,float(w.max())-.005)
    feasible=rshort<=1e-12 and fshort<=1e-12 and wshort<=1e-12
    utility=.40*weak+.25*high+.35*native
    if hybrid:utility=.8*utility+.2*policy
    utility-=2*rshort+20*max(0.,default-.01)+10*fshort+8*wshort
    utility-=15*np.maximum(f-.01,0).mean()+8*np.maximum(w-.005,0).mean()+.5*f.mean()
    return {"feasible":bool(feasible),"utility":float(utility),"weak_recall":weak,"prior_weak_recall":weak0,
       "high_loss_recall":high,"prior_high_loss_recall":high0,"native_recall":native,"prior_native_recall":native0,
       "policy_recall":policy,"prior_policy_recall":policy0,"recall_shortfall":rshort,
       "max_FPR":float(f.max()),"max_FWR":float(w.max()),"default_max_FPR":default,
       "FPR_pass":int((f<=.01).sum()),"FWR_pass":int((w<=.005).sum())}

def diagnostic(df,model):
    # Evaluation only: labels used for measuring persistence following
    # a positive interval. No labels are used for the model's gate.
    y=model.predict_levels(df);x=df[["simulation_seed","node_id","interval","node_type","is_anomaly"]].copy()
    x["alert"]=y==2;x["warning"]=y==1
    for k,v in model.channels(df).items():x[k]=v
    x["gate_active"]=(df["recovery"+str(model.recovery)].to_numpy()<1.) if isinstance(model,AdaptiveMemoryTICD) and model.recovery else False
    x=x.sort_values(["simulation_seed","node_id","interval"],kind="stable")
    x["last"]=x.interval.where(x.is_anomaly>0)
    x["last"]=x.groupby(["simulation_seed","node_id"],sort=False)["last"].ffill()
    x["post"]=(x.interval-x["last"]).between(1,48)
    x=x[x.is_anomaly==0]
    def counts(g):
        return {"normal":len(g),"FP":int(g.alert.sum()),"FW":int(g.warning.sum()),
           "core_FP":int((g.alert&g.core).sum()),"count_FP":int((g.alert&g["count"]).sum()),
           "audit_FP":int((g.alert&g.audit).sum()),"gate_rows":int(g.gate_active.sum())}
    return {"post_attack_48":counts(x[x.post]),"other":counts(x[~x.post]),
       "roles":{str(k):counts(g) for k,g in x.groupby("node_type")}}

def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"separate_robust_models.pkl").open("rb") as f:pkg=pickle.load(f)
    old=pkg["models"];core=pkg["core"];audit=pkg["audit"];cal=pkg["calibration"]
    assert core.variant=="full" and core.model.mode=="weighted"
    assert core.zonal.mode=="load"
    assert old["selected_primary"].count_mode==old["selected_audit"].count_mode=="trim8_48"
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    # The active warning is load-conditioned and does not use the auxiliary
    # zone causal median; using the frozen refitted core to prepare that
    # unused field cannot alter its decisions.
    causal=core.causal
    seeds=[s for part in (TRAIN,VAL,DEV,TEST) for xs in part.values() for s in xs]
    assert len(seeds)==len(set(seeds))==298 and min(seeds)>131901
    train=pd.concat([group(ls,cal,"clean_loss"+str(m),causal)[0] for m,ls in TRAIN.items()],ignore_index=True)
    ref=AdaptiveMemoryReference(pkg["reference"]).fit(train)
    prepare=lambda x:ref.prepare(x,core,audit)
    val={m:prepare(group(ls,cal,"clean_loss"+str(m),causal)[0]) for m,ls in VAL.items()}
    candidates={}
    for r,s in SPECS:
        for b in BUDGETS:
            for cm in CAL_MODES:
                name=("audit" if b[2]>0 else "primary")+"__"+str(r)+"_"+s+"__"+"_".join(str(int(x*1000)) for x in b)+"__"+cm
                candidates[name]=AdaptiveMemoryTICD(r,s,b,cm).calibrate(val)
    controls={"prior_primary":old["selected_primary"],"prior_audit":old["selected_audit"]}
    print("MEMORY_REFERENCE",json.dumps({"candidates":len(candidates),"reference":ref.manifest(),
           "manifests":{n:m.manifest() for n,m in candidates.items()}}),flush=True)
    dev={};dev_per={}
    for k,ls in DEV.items():
        x,_,_=group(ls,cal,k,causal);x=prepare(x);dev[k]={};dev_per[k]={}
        for n,m in {**controls,**candidates}.items():
            v,r=quick(m,x,m.hybrid);dev[k][n]=v;dev_per[k][n]=r
        print("MEMORY_DEV",json.dumps({"cohort":k,"candidates":len(candidates),
           "prior":{n:dev[k][n] for n in controls}}),flush=True)
    select={n:selection(n,dev,dev_per,m.hybrid) for n,m in candidates.items()}
    def choose(h,control=False):
        return max((n for n,m in candidates.items() if m.hybrid==h and (not control or (m.recovery==0 and m.sequence=="none"))),
            key=lambda n:(select[n]["feasible"],select[n]["utility"],-list(candidates).index(n)))
    primary=choose(False);hybrid=choose(True)
    cp=choose(False,True);ch=choose(True,True)
    chosen={**controls,"recalibrated_primary":candidates[cp],"recalibrated_audit":candidates[ch],
            "selected_primary":candidates[primary],"selected_audit":candidates[hybrid]}
    print("MEMORY_SELECTED",json.dumps({"primary":primary,"audit":hybrid,"recalibrated_primary":cp,
       "recalibrated_audit":ch,"primary_selection":select[primary],"audit_selection":select[hybrid],
       "feasible_primary_count":sum(v["feasible"] for n,v in select.items() if n.startswith("primary__")),
       "feasible_audit_count":sum(v["feasible"] for n,v in select.items() if n.startswith("audit__"))}),flush=True)
    # Independent clean and observational-equivalence counterfactual,
    # prefix causal preparation and prepared-row invariance.
    x,_,_=group([151901],cal,"clean_loss10",causal)
    l,_,_=group([151901],cal,"latent_identical",causal)
    labels=["is_anomaly","attack_type","attacker_sophistication"]
    pd.testing.assert_frame_equal(x.drop(columns=labels),l.drop(columns=labels))
    xx=prepare(x);ll=prepare(l);pref=prepare(x.loc[x.interval<180])
    cols=["recovery3","recovery6","seq_fast","seq_slow","memory_instant","memory_history"]
    cols += ["count_"+str(r)+"_"+s for r,s in SPECS]
    for c in cols:assert np.allclose(pref[c],xx.loc[pref.index,c],rtol=0,atol=1e-10)
    assert np.array_equal(xx.core_0,xx.split_core) or np.allclose(xx.core_0,xx.split_core,atol=1e-12)
    for r in (3,6):
        assert (xx["core_"+str(r)]>=xx.memory_instant).all()
        assert (xx["recovery"+str(r)]>=.25).all()
    bad=[c for c in x if c.startswith("gt_") or c in labels+["is_compromised","normal_event"]]
    stripped=prepare(x.drop(columns=bad))
    for n,m in chosen.items():
        z=m.predict_levels(xx)
        assert np.array_equal(z,m.predict_levels(ll))
        assert np.array_equal(z,m.predict_levels(stripped))
        sub=xx.sample(n=200,random_state=42);assert np.array_equal(z[sub.index.to_numpy()],m.predict_levels(sub))
    heldout={};per_seed={};masters={};meta={};diagnostics={}
    for k,ls in TEST.items():
        x,m,md=group(ls,cal,k,causal);x=prepare(x)
        meta[k]=md;heldout[k]={};per_seed[k]={}
        for n,mod in chosen.items():
            v,r=pair_metrics(mod,x,mod.hybrid);heldout[k][n]=v;per_seed[k][n]=r
        if k=="default":
            diagnostics={n:diagnostic(x,mod) for n,mod in chosen.items()}
            print("MEMORY_DIAG",json.dumps(diagnostics),flush=True)
        masters[k]=master_metrics(pkg["master_audit"],m)
        print("MEMORY_HELDOUT",json.dumps({"cohort":k,**{n:short(v) for n,v in heldout[k].items()}}),flush=True)
    paired={}
    for family,ks in (("primary",NATIVE),("audit",NATIVE+POLICY)):
        for metric,cohorts in (("confirmed_recall",ks),("confirmed_recall_weak",WEAK),
              ("confirmed_alert_FPR",ks),("false_warning_rate",ks)):
            key=metric.removesuffix("_weak")
            ds=[]
            for k in cohorts:
                u=per_seed[k]["selected_"+family];v=per_seed[k]["prior_"+family]
                ds.append(np.array([a[key]-b[key] for a,b in zip(u,v)]))
            d=np.concatenate(ds);rng=np.random.default_rng(7611)
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
           "max_FPR":max(r["confirmed_alert_FPR"] for r in rs),
           "max_FWR":max(r["false_warning_rate"] for r in rs),
           "FPR_pass":sum(r["confirmed_alert_FPR"]<=.01 for r in rs),
           "FWR_pass":sum(r["false_warning_rate"]<=.005 for r in rs)}
        print("MEMORY_SUMMARY",json.dumps({"model":n,**summary[n]}),flush=True)
    path=a.out/"adaptive_memory_models.pkl"
    with path.open("wb") as f:pickle.dump({"models":chosen,"reference":ref,"core":core,"audit":audit,
             "calibration":cal,"master_audit":pkg["master_audit"]},f,pickle.HIGHEST_PROTOCOL)
    with path.open("rb") as f:reload=pickle.load(f)
    for n,m in chosen.items():assert np.array_equal(m.predict_levels(xx),reload["models"][n].predict_levels(xx))
    report={"source_run":45441388811,"scope":"synthetic full TICD; observed recovery and sequential evidence",
        "real_field_validation":False,"external_audit_is_synthetic":True,"all_original_invariants_preserved":True,
        "test_used_for_selection":False,"training_seeds":TRAIN,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
        "reference":ref.manifest(),"candidates":{n:m.manifest() for n,m in candidates.items()},"selection":select,
        "selected":{"primary":primary,"audit":hybrid,"recalibrated_primary":cp,"recalibrated_audit":ch},
        "development":dev,"development_by_seed":dev_per,"heldout":heldout,"per_seed":per_seed,"summary":summary,
        "diagnostics":diagnostics,"paired":paired,"master_audit":masters,"metadata":meta,
        "prefix_causality":True,"label_and_batch_invariance":True,"observational_equivalence_verified":True,
        "instant_not_attenuated":True,"reload_verified":True}
    (a.out/"adaptive_memory_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("MEMORY_FINAL",json.dumps({"main_runs":len(seeds),"test_runs":150,"candidates":len(candidates),
         "selected":report["selected"],"paired":paired,"audits_passed":True}),flush=True)
if __name__=="__main__":main()
