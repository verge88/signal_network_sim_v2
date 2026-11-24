#!/usr/bin/env python3
"""Test independent commissioning references and changed operating regimes."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from ticd_commissioned_reference import CommissionedReference,CommissionedFusionTICD,MIXES,SCOPES,WINDOWS
from evaluate_ticd_block_confirmation import sources
from evaluate_ticd_observability_factorial import short,master_metrics
from evaluate_ticd_full_count_evidence import NATIVE as BASE_NATIVE,POLICY,LOSSES,pair_metrics
from evaluate_ticd_separate_robust_fusion import quick
from simulate_ticd_commissioned_reference import observed_group as group,KINDS,SHIFT_KINDS
NATIVE=BASE_NATIVE+("shift_weak_0025","shift_weak_005","shift_redirect")

TRAIN={m:[201001+100*i+j for j in range(4)] for i,m in enumerate(LOSSES)}
VAL={m:[202001+100*i+j for j in range(6)] for i,m in enumerate(LOSSES)}
DEV={k:[204001+100*i+j for j in range(3)] for i,k in enumerate(KINDS)}
TEST={k:[208001+100*i+j for j in range(6)] for i,k in enumerate(KINDS)}
BUDGETS=((.002,.003,.003,0.),(.002,.002,.004,0.),(.002,.002,.002,.002),(.0015,.0015,.002,.003))
CAL_MODES=("p90run","worst_run")
WEAK=("weak_0025","weak_005","shift_weak_0025","shift_weak_005")
SHIFT_ATTACK=("shift_weak_0025","shift_weak_005","shift_redirect")
HIGH=tuple(k for k in NATIVE if k.startswith("role_") and k.endswith("loss25"))

def selection(n,dev,per,hybrid):
    base="prior_audit" if hybrid else "prior_primary"
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
    with (a.prior/"block_confirmation_models.pkl").open("rb") as f:pkg=pickle.load(f)
    old=pkg["models"];core=pkg["core"];audit=pkg["audit"];cal=pkg["calibration"]
    assert core.variant=="full" and core.model.mode=="weighted"
    assert core.zonal.mode=="load"
    assert old["selected_primary"].mode==old["selected_audit"].mode=="supported"
    assert old["selected_primary"].window==old["selected_audit"].window==144
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    # The active warning is load-conditioned and does not use the auxiliary
    # zone causal median; using the frozen refitted core to prepare that
    # unused field cannot alter its decisions.
    causal=core.causal
    seeds=[s for part in (TRAIN,VAL,DEV,TEST) for xs in part.values() for s in xs]
    assert len(seeds)==len(set(seeds))==350 and min(seeds)>191901
    train_groups=[group(ls,cal,"clean_loss"+str(m),causal) for m,ls in TRAIN.items()]
    train=pd.concat([v[0] for v in train_groups],ignore_index=True)
    commissions=pd.concat([v[3] for v in train_groups],ignore_index=True)
    ref=CommissionedReference(pkg["reference"]).fit(train,commissions)
    prepare=ref.prepare
    val={}
    for m,ls in VAL.items():
        x,_,_,c=group(ls,cal,"clean_loss"+str(m),causal)
        val[m]=prepare(x,c)
    del train_groups,train,commissions
    candidates={}
    for w in WINDOWS:
        for mix in MIXES:
            for scope in SCOPES:
                for b in BUDGETS:
                    for cm in CAL_MODES:
                        name=("audit" if b[3]>0 else "primary")+"__"+str(w)+"_"+str(int(100*mix))+"__"+scope+"__"+"_".join(str(round(x*10000)) for x in b)+"__"+cm
                        candidates[name]=CommissionedFusionTICD(w,mix,scope,b,cm).calibrate(val)
    controls={"prior_primary":old["selected_primary"],"prior_audit":old["selected_audit"]}
    print("NODE_REFERENCE",json.dumps({"candidates":len(candidates),"reference":ref.manifest(),
           "manifests":{n:m.manifest() for n,m in candidates.items()}}),flush=True)
    dev={};dev_per={}
    for k,ls in DEV.items():
        x,_,_,c=group(ls,cal,k,causal);x=prepare(x,c);dev[k]={};dev_per[k]={}
        for n,m in {**controls,**candidates}.items():
            v,r=quick(m,x,m.hybrid);dev[k][n]=v;dev_per[k][n]=r
        print("NODE_DEV",json.dumps({"cohort":k,"candidates":len(candidates),
           "prior":{n:dev[k][n] for n in controls}}),flush=True)
    select={n:selection(n,dev,dev_per,m.hybrid) for n,m in candidates.items()}
    def choose(h,control=False):
        return max((n for n,m in candidates.items() if m.hybrid==h and (not control or m.mix==0.)),
            key=lambda n:(select[n]["feasible"],select[n]["utility"],-list(candidates).index(n)))
    primary=choose(False);hybrid=choose(True)
    cp=choose(False,True);ch=choose(True,True)
    chosen={**controls,"recalibrated_primary":candidates[cp],"recalibrated_audit":candidates[ch],
            "selected_primary":candidates[primary],"selected_audit":candidates[hybrid]}
    print("NODE_SELECTED",json.dumps({"primary":primary,"audit":hybrid,"recalibrated_primary":cp,
       "recalibrated_audit":ch,"primary_selection":select[primary],"audit_selection":select[hybrid],
       "feasible_primary_count":sum(v["feasible"] for n,v in select.items() if n.startswith("primary__")),
       "feasible_audit_count":sum(v["feasible"] for n,v in select.items() if n.startswith("audit__"))}),flush=True)
    # Independent clean and observational-equivalence counterfactual,
    # prefix causal preparation and prepared-row invariance.
    x,_,_,c=group([212901],cal,"clean_loss10",causal)
    l,_,_,lc=group([212901],cal,"latent_identical",causal)
    pd.testing.assert_frame_equal(c,lc)
    labels=["is_anomaly","attack_type","attacker_sophistication"]
    pd.testing.assert_frame_equal(x.drop(columns=labels),l.drop(columns=labels))
    xx=prepare(x,c);ll=prepare(l,c);pref=prepare(x.loc[x.interval<180],c)
    cols=["conditional_expected_rate","core_0","core_0_role","baseline","baseline_role","commissioned_offset"]
    cols += ["personal_"+str(w)+"_"+str(int(100*mix)) for w in WINDOWS for mix in MIXES]
    for col in cols:assert np.allclose(pref[col],xx.loc[pref.index,col],rtol=0,atol=1e-10)
    assert np.allclose(xx.core_0,xx.split_core,atol=1e-12)
    assert np.array_equal(xx.baseline,xx.count_0_fast)
    absent=prepare(x,c.iloc[:0])
    assert not absent.commissioned_available.any()
    for w in WINDOWS:
        assert np.array_equal(xx["personal_"+str(w)+"_0"],xx["supported_"+str(w)])
        for mix in MIXES:
            col="personal_"+str(w)+"_"+str(int(100*mix))
            assert np.array_equal(absent[col],xx["supported_"+str(w)])
            assert (xx[col]>=(1-mix)*xx["supported_"+str(w)]-1e-12).all()
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
        diagnostics[k]["profile_coverage"]={"available_rows":int(x.commissioned_available.sum()),"rows":len(x)}
        print("NODE_DIAG",json.dumps({"cohort":k,"models":{n:v["overall"] for n,v in diagnostics[k].items() if n!="profile_coverage"}}),flush=True)
        masters[k]=master_metrics(pkg["master_audit"],m)
        print("NODE_HELDOUT",json.dumps({"cohort":k,**{n:short(v) for n,v in heldout[k].items()}}),flush=True)
    paired={}
    for family,ks in (("primary",NATIVE),("audit",NATIVE+POLICY)):
        for metric,cohorts in (("confirmed_recall",ks),("confirmed_recall_weak",WEAK),("confirmed_recall_shift",SHIFT_ATTACK),
              ("confirmed_alert_FPR",ks),("false_warning_rate",ks)):
            key=metric.removesuffix("_weak").removesuffix("_shift")
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
        print("NODE_SUMMARY",json.dumps({"model":n,**summary[n]}),flush=True)
    path=a.out/"commissioned_reference_models.pkl"
    with path.open("wb") as f:pickle.dump({"models":chosen,"reference":ref,"core":core,"audit":audit,
             "calibration":cal,"master_audit":pkg["master_audit"]},f,pickle.HIGHEST_PROTOCOL)
    with path.open("rb") as f:reload=pickle.load(f)
    for n,m in chosen.items():assert np.array_equal(m.predict_levels(xx),reload["models"][n].predict_levels(xx))
    report={"source_run":45461470242,"scope":"synthetic full TICD; clean independent commissioning and shifted regimes",
        "real_field_validation":False,"external_audit_is_synthetic":True,"all_original_invariants_preserved":True,
        "test_used_for_selection":False,"training_seeds":TRAIN,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
        "reference":ref.manifest(),"candidates":{n:m.manifest() for n,m in candidates.items()},"selection":select,
        "selected":{"primary":primary,"audit":hybrid,"recalibrated_primary":cp,"recalibrated_audit":ch},
        "development":dev,"development_by_seed":dev_per,"heldout":heldout,"per_seed":per_seed,"summary":summary,
        "diagnostics":diagnostics,"paired":paired,"master_audit":masters,"metadata":meta,
        "prefix_causality":True,"label_and_batch_invariance":True,"observational_equivalence_verified":True,
        "old_count_preserved":True,"actual_transport_loss_not_observed":True,"independent_commissioning":True,"commissioning_runs":len(seeds),"missing_profile_fallback_verified":True,"reload_verified":True}
    (a.out/"commissioned_reference_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("NODE_FINAL",json.dumps({"main_runs":len(seeds),"test_runs":180,"commissioning_runs":len(seeds),"candidates":len(candidates),
         "selected":report["selected"],"paired":paired,"audits_passed":True}),flush=True)
if __name__=="__main__":main()
