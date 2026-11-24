#!/usr/bin/env python3
"""Compare physical loss compensation with matched full-detector controls."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from ticd_independent_loss import IndependentLossReference,LossFusionTICD,SOURCES,WINDOWS
from evaluate_ticd_block_confirmation import sources
from evaluate_ticd_observability_factorial import short,master_metrics
from evaluate_ticd_full_count_evidence import NATIVE as BASE_NATIVE,POLICY,LOSSES,pair_metrics
from evaluate_ticd_separate_robust_fusion import quick
from simulate_ticd_independent_loss import group,KINDS,STRESS
STRESS_ATTACK=tuple(k for k in STRESS if k.endswith("_weak"))
NATIVE=BASE_NATIVE+("shift_weak_0025","shift_weak_005","shift_redirect")+STRESS_ATTACK

NORMAL=tuple("clean_loss"+str(m) for m in LOSSES)+("shift_clean_loss","shift_clean_load")
TRAIN={k:[221001+100*i+j for j in range(3 if i<8 else 4)] for i,k in enumerate(NORMAL)}
VAL={k:[223001+100*i+j for j in range(4)] for i,k in enumerate(NORMAL)}
DEV={k:[225001+100*i+j for j in range(3)] for i,k in enumerate(KINDS)}
TEST={k:[230001+100*i+j for j in range(5)] for i,k in enumerate(KINDS)}
BUDGETS=((.0015,.0015,.002,.003,0.),(.001,.001,.002,.004,0.),
         (.0015,.001,.0015,.0015,.0025),(.001,.001,.001,.002,.003))
CAL_MODES=("p90run","worst_run")
WEAK=("weak_0025","weak_005","shift_weak_0025","shift_weak_005")+STRESS_ATTACK
SHIFT_ATTACK=("shift_weak_0025","shift_weak_005","shift_redirect")+STRESS_ATTACK
HIGH=tuple(k for k in NATIVE if k.startswith("role_") and k.endswith("loss25"))

def selection(n,dev,per,hybrid,base=None):
    if base is None:base="prior_audit" if hybrid else "prior_primary"
    ks=list(KINDS)
    rec=lambda k,m:dev[k][m]["confirmed_recall"]
    mean=lambda keys,m:float(np.mean([rec(k,m) for k in keys]))
    weak=mean(WEAK,n);weak0=mean(WEAK,base)
    high=mean(HIGH,n);high0=mean(HIGH,base)
    native=mean(NATIVE,n);native0=mean(NATIVE,base)
    shifted=mean(SHIFT_ATTACK,n);shifted0=mean(SHIFT_ATTACK,base)
    policy=mean(POLICY,n) if hybrid else None
    policy0=mean(POLICY,base) if hybrid else None
    f=np.array([v["confirmed_alert_FPR"] for k in ks for v in per[k][n]])
    w=np.array([v["false_warning_rate"] for k in ks for v in per[k][n]])
    default=max(v["confirmed_alert_FPR"] for v in per["default"][n])
    # Preserve the previous conditional gain within three percentage points.
    # Prioritize tail stability; no candidate is assumed feasible.
    rshort=max(0.,weak0-.03-weak)+max(0.,high0-.03-high)+max(0.,native0-.03-native)+max(0.,shifted0-.03-shifted)
    if hybrid:rshort+=max(0.,policy0-.03-policy)
    fshort=max(0.,float(f.max())-.01)
    wshort=max(0.,float(w.max())-.005)
    feasible=rshort<=1e-12 and fshort<=1e-12 and wshort<=1e-12
    utility=.35*weak+.20*high+.30*native+.15*shifted
    if hybrid:utility=.8*utility+.2*policy
    utility-=3*rshort+20*max(0.,default-.01)+20*fshort+12*wshort
    utility-=15*np.maximum(f-.01,0).mean()+8*np.maximum(w-.005,0).mean()+.5*f.mean()
    return {"feasible":bool(feasible),"utility":float(utility),"weak_recall":weak,"prior_weak_recall":weak0,
       "high_loss_recall":high,"prior_high_loss_recall":high0,"native_recall":native,"prior_native_recall":native0,
       "policy_recall":policy,"prior_policy_recall":policy0,"shifted_recall":shifted,"prior_shifted_recall":shifted0,"recall_shortfall":rshort,
       "max_FPR":float(f.max()),"max_FWR":float(w.max()),"default_max_FPR":default,
       "FPR_pass":int((f<=.01).sum()),"FWR_pass":int((w<=.005).sum())}

def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"commissioned_reference_models.pkl").open("rb") as f:pkg=pickle.load(f)
    old=pkg["models"];core=pkg["core"];audit=pkg["audit"];cal=pkg["calibration"]
    assert core.variant=="full" and core.model.mode=="weighted"
    assert core.zonal.mode=="load"
    assert old["selected_primary"].mix==old["selected_audit"].mix==.75
    assert old["selected_primary"].window==old["selected_audit"].window==144
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    # The active warning is load-conditioned and does not use the auxiliary
    # zone causal median; using the frozen refitted core to prepare that
    # unused field cannot alter its decisions.
    causal=core.causal
    seeds=[s for part in (TRAIN,VAL,DEV,TEST) for xs in part.values() for s in xs]
    assert len(seeds)==len(set(seeds))==360 and min(seeds)>212901
    train=pd.concat([group(ls,cal,k,causal)[0] for k,ls in TRAIN.items()],ignore_index=True)
    ref=IndependentLossReference(pkg["reference"]).fit(train)
    prepare=ref.prepare
    val={}
    for k,ls in VAL.items():
        x,_,_,c=group(ls,cal,k,causal)
        val[k]=prepare(x,c)
    del train
    candidates={}
    for src in ("none",*SOURCES):
        for w in ((96,) if src=="none" else WINDOWS):
            for b0 in BUDGETS:
                b=(b0[0],b0[1],b0[2]+b0[3],0.,b0[4]) if src=="none" else b0
                for cm in CAL_MODES:
                    name=("audit" if b[4]>0 else "primary")+"__"+src+"_"+str(w)+"__"+"_".join(str(round(x*10000)) for x in b)+"__"+cm
                    candidates[name]=LossFusionTICD(src,w,b,cm).calibrate(val)
    controls={"prior_primary":old["selected_primary"],"prior_audit":old["selected_audit"]}
    print("LOSS_REFERENCE",json.dumps({"candidates":len(candidates),"reference":ref.manifest(),
           "manifests":{n:m.manifest() for n,m in candidates.items()}}),flush=True)
    dev={};dev_per={}
    for k,ls in DEV.items():
        x,_,_,c=group(ls,cal,k,causal);x=prepare(x,c);dev[k]={};dev_per[k]={}
        for n,m in {**controls,**candidates}.items():
            v,r=quick(m,x,m.hybrid);dev[k][n]=v;dev_per[k][n]=r
        print("LOSS_DEV",json.dumps({"cohort":k,"candidates":len(candidates),
           "prior":{n:dev[k][n] for n in controls}}),flush=True)
    initial={n:selection(n,dev,dev_per,m.hybrid) for n,m in candidates.items() if m.source=="none"}
    def choose_control(h):
        return max((n for n in initial if candidates[n].hybrid==h),
            key=lambda n:(initial[n]["feasible"],initial[n]["utility"],-list(candidates).index(n)))
    cp=choose_control(False);ch=choose_control(True)
    # Match the recall target to the newly calibrated control, not to an
    # earlier model with unacceptably high FPR under shifted conditions.
    select={n:selection(n,dev,dev_per,m.hybrid,ch if m.hybrid else cp) for n,m in candidates.items()}
    def choose(h,srcs):
        return max((n for n,m in candidates.items() if m.hybrid==h and m.source in srcs),
            key=lambda n:(select[n]["feasible"],select[n]["utility"],-list(candidates).index(n)))
    # Control is eligible: a new observation must earn its inclusion.
    primary=choose(False,("none",*SOURCES));hybrid=choose(True,("none",*SOURCES))
    cup=choose(False,("cu",));cuh=choose(True,("cu",))
    pp=choose(False,("probe128","probe512","guard512"));ph=choose(True,("probe128","probe512","guard512"))
    chosen={**controls,"recalibrated_primary":candidates[cp],"recalibrated_audit":candidates[ch],
            "cu_primary":candidates[cup],"cu_audit":candidates[cuh],"probe_primary":candidates[pp],"probe_audit":candidates[ph],
            "selected_primary":candidates[primary],"selected_audit":candidates[hybrid]}
    print("LOSS_SELECTED",json.dumps({"primary":primary,"audit":hybrid,"recalibrated_primary":cp,
       "recalibrated_audit":ch,"cu_primary":cup,"cu_audit":cuh,"probe_primary":pp,"probe_audit":ph,"primary_selection":select[primary],"audit_selection":select[hybrid],
       "feasible_primary_count":sum(v["feasible"] for n,v in select.items() if n.startswith("primary__")),
       "feasible_audit_count":sum(v["feasible"] for n,v in select.items() if n.startswith("audit__"))}),flush=True)
    # Independent clean and observational-equivalence counterfactual,
    # prefix causal preparation and prepared-row invariance.
    x,_,_,c=group([235901],cal,"clean_loss10",causal)
    l,_,_,lc=group([235901],cal,"latent_identical",causal)
    pd.testing.assert_frame_equal(c,lc)
    labels=["is_anomaly","attack_type","attacker_sophistication"]
    pd.testing.assert_frame_equal(x.drop(columns=labels),l.drop(columns=labels))
    xx=prepare(x,c);ll=prepare(l,c);pref=prepare(x.loc[x.interval<180],c)
    cols=["conditional_expected_rate","core_0","core_0_role","baseline","baseline_role","commissioned_offset","personal_144_75"]
    cols += ["loss_"+src+"_"+str(w) for src in SOURCES for w in WINDOWS]
    for col in cols:assert np.allclose(pref[col],xx.loc[pref.index,col],rtol=0,atol=1e-10)
    assert np.allclose(xx.core_0,xx.split_core,atol=1e-12)
    assert np.array_equal(xx.baseline,xx.count_0_fast)
    absent=x.copy()
    for col in ("probe128_sent","probe128_lost","probe512_sent","probe512_lost"):absent[col]=0
    absent["probe_age"]=-1
    empty=prepare(absent,c)
    for src in ("probe128","probe512","guard512"):
        for w in WINDOWS:assert (empty["loss_"+src+"_"+str(w)]==0).all()
    for col in ("core_0","baseline","personal_144_75","split_audit"):
        assert np.array_equal(empty[col],xx[col])
    for m in candidates.values():
        if m.source.startswith("probe") or m.source=="guard512":
            assert not m.channels(empty)["loss"].any()
    bad=[c for c in x if c.startswith("gt_") or c in labels+["is_compromised","normal_event"]]
    stripped=prepare(x.drop(columns=bad),c.drop(columns=[v for v in bad if v in c]))
    for n,m in chosen.items():
        z=m.predict_levels(xx)
        assert np.array_equal(z,m.predict_levels(ll))
        assert np.array_equal(z,m.predict_levels(stripped))
        sub=xx.sample(n=200,random_state=42);assert np.array_equal(z[sub.index.to_numpy()],m.predict_levels(sub))
    heldout={};per_seed={};masters={};meta={};diagnostics={}
    for k,ls in TEST.items():
        x,m,md,c=group(ls,cal,k,causal);x=prepare(x,c)
        meta[k]=md;heldout[k]={};per_seed[k]={}
        for n,mod in chosen.items():
            v,r=pair_metrics(mod,x,mod.hybrid);heldout[k][n]=v;per_seed[k][n]=r
        diagnostics[k]={n:sources(x,mod,ref) for n,mod in chosen.items()}
        diagnostics[k]["profile_coverage"]={"available_rows":int(x.commissioned_available.sum()),"rows":len(x),
            "probe_available_rows":int((x.probe512_sent>0).sum()),
            "usable_probe_rows":int(x.loss_available_probe512.sum())}
        print("LOSS_DIAG",json.dumps({"cohort":k,"models":{n:v["overall"] for n,v in diagnostics[k].items() if n!="profile_coverage"}}),flush=True)
        masters[k]=master_metrics(pkg["master_audit"],m)
        print("LOSS_HELDOUT",json.dumps({"cohort":k,**{n:short(v) for n,v in heldout[k].items()}}),flush=True)
    paired={}
    for family,ks in (("primary",NATIVE),("audit",NATIVE+POLICY)):
        for contrast,left,right,items in (
          ("selected_vs_control","selected_","recalibrated_",(("confirmed_recall",ks),("confirmed_recall_weak",WEAK),("confirmed_recall_shift",SHIFT_ATTACK),("confirmed_alert_FPR",ks),("false_warning_rate",ks))),
          ("probe_vs_cu","probe_","cu_",(("confirmed_recall",ks),("confirmed_recall_weak",WEAK),("confirmed_recall_shift",SHIFT_ATTACK),("confirmed_alert_FPR",ks)))):
            for metric,cohorts in items:
                key=metric.removesuffix("_weak").removesuffix("_shift")
                ds=[]
                for k in cohorts:
                    u=per_seed[k][left+family];v=per_seed[k][right+family]
                    ds.append(np.array([a[key]-b[key] for a,b in zip(u,v)]))
                d=np.concatenate(ds);rng=np.random.default_rng(7611)
                boot=np.mean([g[rng.integers(0,len(g),(4000,len(g)))].mean(1) for g in ds],axis=0)
                paired[contrast+"_"+family+"_"+metric]={"delta":float(d.mean()),"n":len(d),
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
        print("LOSS_SUMMARY",json.dumps({"model":n,**summary[n]}),flush=True)
    path=a.out/"independent_loss_models.pkl"
    with path.open("wb") as f:pickle.dump({"models":chosen,"reference":ref,"core":core,"audit":audit,
             "calibration":cal,"master_audit":pkg["master_audit"]},f,pickle.HIGHEST_PROTOCOL)
    with path.open("rb") as f:reload=pickle.load(f)
    for n,m in chosen.items():assert np.array_equal(m.predict_levels(xx),reload["models"][n].predict_levels(xx))
    report={"source_run":45468373652,"scope":"synthetic full TICD; finite independent path probes and analytic CU comparator",
        "real_field_validation":False,"external_audit_is_synthetic":True,"all_original_invariants_preserved":True,
        "test_used_for_selection":False,"training_seeds":TRAIN,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
        "reference":ref.manifest(),"candidates":{n:m.manifest() for n,m in candidates.items()},"selection":select,
        "selected":{"primary":primary,"audit":hybrid,"recalibrated_primary":cp,"recalibrated_audit":ch,"cu_primary":cup,"cu_audit":cuh,"probe_primary":pp,"probe_audit":ph},
        "development":dev,"development_by_seed":dev_per,"heldout":heldout,"per_seed":per_seed,"summary":summary,
        "diagnostics":diagnostics,"paired":paired,"master_audit":masters,"metadata":meta,
        "prefix_causality":True,"label_and_batch_invariance":True,"observational_equivalence_verified":True,
        "old_count_preserved":True,"actual_transport_loss_not_observed":True,"independent_commissioning":True,"commissioning_runs":len(seeds),"missing_probe_fallback_verified":True,"probe_is_synthetic":True,"probe_load_not_modeled":True,"paired_comparator":"selected_vs_recalibrated_no_loss_channel_and_probe_vs_cu","reload_verified":True}
    (a.out/"independent_loss_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("LOSS_FINAL",json.dumps({"main_runs":len(seeds),"test_runs":180,"commissioning_runs":len(seeds),"candidates":len(candidates),
         "selected":report["selected"],"paired":paired,"audits_passed":True}),flush=True)
if __name__=="__main__":main()
