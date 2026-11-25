#!/usr/bin/env python3
"""Matched oracle and measured-loss full TICD experiment."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from ticd_measured_loss import FullMeasuredReference,MeasuredFusion,measured_frame,labels,MODES,WINDOWS
from simulate_ticd_measured_loss import group,KINDS,STRESS
from evaluate_ticd_full_count_evidence import NATIVE as ORIGINAL_NATIVE,POLICY,LOSSES,pair_metrics
from evaluate_ticd_separate_robust_fusion import quick
from evaluate_ticd_observability_factorial import short,master_metrics
from ticd_observability_factorial import PolicyEvidence

NORMAL=tuple("clean_loss"+str(m) for m in LOSSES)+("shift_clean_loss","shift_clean_load")
TRAIN={k:[241001+100*i+j for j in range(3 if i<8 else 4)] for i,k in enumerate(NORMAL)}
VAL={k:[243001+100*i+j for j in range(4)] for i,k in enumerate(NORMAL)}
DEV={k:[245001+100*i+j for j in range(3)] for i,k in enumerate(KINDS)}
TEST={k:[250001+100*i+j for j in range(5)] for i,k in enumerate(KINDS)}
STRESS_ATTACK=tuple(k for k in STRESS if k.endswith("_weak"))
NATIVE=ORIGINAL_NATIVE+("shift_weak_0025","shift_weak_005","shift_redirect")+STRESS_ATTACK
WEAK=("weak_0025","weak_005","shift_weak_0025","shift_weak_005")+STRESS_ATTACK
HIGH=tuple(k for k in NATIVE if k.startswith("role_") and k.endswith("loss25"))
BUDGETS=((.0015,.0015,.002,.003,0.),(.001,.001,.002,.004,0.),
         (.0015,.001,.0015,.0015,.0025),(.001,.001,.001,.002,.003))
CAL_MODES=("p90run","worst_run")

def select(n,dev,per,hybrid):
    avg=lambda ks:float(np.mean([dev[k][n]["confirmed_recall"] for k in ks]))
    f=np.array([r["confirmed_alert_FPR"] for k in KINDS for r in per[k][n]])
    w=np.array([r["false_warning_rate"] for k in KINDS for r in per[k][n]])
    util=.4*avg(WEAK)+.25*avg(HIGH)+.35*avg(NATIVE)
    if hybrid:util=.8*util+.2*avg(POLICY)
    fshort=max(0.,float(f.max())-.01);wshort=max(0.,float(w.max())-.005)
    util-=20*fshort+12*wshort+15*np.maximum(f-.01,0).mean()+8*np.maximum(w-.005,0).mean()+.5*f.mean()
    return {"feasible":bool(fshort==0 and wshort==0),"utility":float(util),
            "weak_recall":avg(WEAK),"high_loss_recall":avg(HIGH),"native_recall":avg(NATIVE),
            "policy_recall":avg(POLICY) if hybrid else None,"max_FPR":float(f.max()),"max_FWR":float(w.max()),
            "FPR_pass":int((f<=.01).sum()),"FWR_pass":int((w<=.005).sum())}

def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"independent_loss_models.pkl").open("rb") as f:pkg=pickle.load(f)
    source=pkg["core"];cal=pkg["calibration"];causal=source.causal
    assert source.variant=="full" and source.model.mode=="weighted"
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    seeds=[s for p in (TRAIN,VAL,DEV,TEST) for v in p.values() for s in v]
    assert len(seeds)==len(set(seeds))==360 and min(seeds)>235901
    train_groups=[group(ls,cal,k,causal) for k,ls in TRAIN.items()]
    train=pd.concat([x[0] for x in train_groups],ignore_index=True)
    commission=pd.concat([x[3] for x in train_groups],ignore_index=True)
    mt=pd.concat([x[1] for x in train_groups],ignore_index=True)
    del train_groups
    val_groups={k:group(ls,cal,k,causal) for k,ls in VAL.items()}
    raw_val={k:x[0] for k,x in val_groups.items()}
    master_audit=PolicyEvidence().fit(mt,{k:x[1] for k,x in val_groups.items()})
    refs={};candidates={}
    for mode in MODES:
        ref=FullMeasuredReference(mode,source).fit(train,commission,raw_val)
        refs[mode]=ref
        val={k:labels(ref.prepare(x[0],x[3]),x[0]) for k,x in val_groups.items()}
        for w in WINDOWS:
            for b in BUDGETS:
                for cm in CAL_MODES:
                    n=mode+"__"+("audit" if b[4]>0 else "primary")+"__"+str(w)+"__"+"_".join(str(round(t*10000)) for t in b)+"__"+cm
                    candidates[n]=MeasuredFusion(mode,w,b,cm).calibrate(val)
        print("MEASURE_REFERENCE",json.dumps({"mode":mode,"manifest":ref.manifest(),"candidates":len(candidates)}),flush=True)
        del val
    del train,commission,mt,val_groups,raw_val
    assert len(candidates)==80
    dev={};dp={}
    for k,ls in DEV.items():
        raw,_,_,c=group(ls,cal,k,causal)
        dev[k]={};dp[k]={}
        for mode,ref in refs.items():
            x=labels(ref.prepare(raw,c),raw)
            for n,m in candidates.items():
                if m.observation_mode!=mode:continue
                v,r=quick(m,x,m.hybrid);dev[k][n]=v;dp[k][n]=r
        print("MEASURE_DEV",json.dumps({"cohort":k,"candidates":len(dev[k])}),flush=True)
    selection={n:select(n,dev,dp,m.hybrid) for n,m in candidates.items()}
    selected={};chosen={}
    for mode in MODES:
        for hybrid in (False,True):
            n=max((n for n,m in candidates.items() if m.observation_mode==mode and m.hybrid==hybrid),
                key=lambda n:(selection[n]["feasible"],selection[n]["utility"],-list(candidates).index(n)))
            name=mode+"_"+("audit" if hybrid else "primary")
            chosen[name]=candidates[n];selected[name]=n
    print("MEASURE_SELECTED",json.dumps({"selected":selected,
          "selection":{n:selection[k] for n,k in selected.items()},
          "feasible_count":sum(v["feasible"] for v in selection.values())}),flush=True)
    # Identical observations/different labels, explicit raw boundary, no
    # derived oracle statistics or future records admitted to measured modes.
    raw,_,_,c=group([255901],cal,"clean_loss10",causal)
    latent,_,_,lc=group([255901],cal,"latent_identical",causal)
    pd.testing.assert_frame_equal(c,lc)
    bad=["cu_combined_loss_prob","master_ack_shortfall_adjusted","dev_master_ack_shortfall_vs_zone",
         "ack_shortfall_ewma_6","ack_shortfall_ewma_16","ack_shortfall_ewma_32","zone_causal_median","zone_observed_loss_median"]
    bad += [col for col in raw if col.startswith("gt_") or col in ("is_anomaly","attack_type","attacker_sophistication","normal_event","is_compromised")]
    audit_cache={}
    for mode,ref in refs.items():
        x=ref.prepare(raw,c);l=ref.prepare(latent,lc)
        pd.testing.assert_frame_equal(x,l)
        prefix=ref.prepare(raw.loc[raw.interval<180],c)
        cols=["core_0","baseline","personal_144_75","loss_cu_48","loss_cu_96","warn_0","split_audit","measured_oneway_loss","measured_loss_variance"]
        for col in cols:assert np.allclose(prefix[col],x.loc[prefix.index,col],rtol=0,atol=1e-10)
        if mode!="oracle":
            # Remove rather than perturb oracle and all its derived features.
            stripped=ref.prepare(raw.drop(columns=bad,errors="ignore"),c.drop(columns=bad,errors="ignore"))
            pd.testing.assert_frame_equal(x,stripped)
            # Probe outage must use measured CU fallback, not oracle.
            miss=raw.copy()
            for col in ("active_sent","active_fwd_lost","active_rev_lost","passive_sent","passive_received"):miss[col]=0
            miss["active_age"]=-1
            z=measured_frame(miss,mode);cu=measured_frame(miss,"cu")
            assert np.array_equal(z.measured_oneway_loss,cu.measured_oneway_loss)
        for n,m in chosen.items():
            if m.observation_mode!=mode:continue
            z=m.predict_levels(x);sub=x.sample(n=200,random_state=42)
            assert np.array_equal(z,m.predict_levels(l))
            assert np.array_equal(z[sub.index.to_numpy()],m.predict_levels(sub))
            audit_cache[n]=(x,z)
    held={};per={};meta={};masters={};measurements={}
    for k,ls in TEST.items():
        raw,m,md,c=group(ls,cal,k,causal)
        held[k]={};per[k]={};meta[k]=md;measurements[k]={}
        masters[k]=master_metrics(master_audit,m)
        for mode,ref in refs.items():
            x=labels(ref.prepare(raw,c),raw)
            actual=1-np.sqrt(1-raw.cu_combined_loss_prob.to_numpy(dtype=float))
            estimate=x.measured_oneway_loss.to_numpy()
            measurements[k][mode]={"mae_vs_simulator_cu_proxy":float(np.mean(np.abs(estimate-actual))),
                "available_rows":int(x.measurement_available.sum()),"fallback_rows":int(x.measurement_fallback.sum())}
            for n,mod in chosen.items():
                if mod.observation_mode!=mode:continue
                v,r=pair_metrics(mod,x,mod.hybrid);held[k][n]=v;per[k][n]=r
        print("MEASURE_HELDOUT",json.dumps({"cohort":k,**{n:short(v) for n,v in held[k].items()}}),flush=True)
    summary={}
    for n in chosen:
        rs=[r for k in KINDS for r in per[k][n]]
        summary[n]={"mean_FPR":float(np.mean([r["confirmed_alert_FPR"] for r in rs])),
          "max_FPR":max(r["confirmed_alert_FPR"] for r in rs),"max_FWR":max(r["false_warning_rate"] for r in rs),
          "FPR_pass":sum(r["confirmed_alert_FPR"]<=.01 for r in rs),"FWR_pass":sum(r["false_warning_rate"]<=.005 for r in rs)}
        print("MEASURE_SUMMARY",json.dumps({"model":n,**summary[n]}),flush=True)
    paired={}
    for fam,ks in (("primary",NATIVE),("audit",NATIVE+POLICY)):
        for left,right in [(m,"oracle") for m in MODES if m!="oracle"]+[("hybrid","active")]:
            for key,cohorts in (("confirmed_recall",ks),("weak_recall",WEAK),("confirmed_alert_FPR",ks)):
                metric="confirmed_recall" if key=="weak_recall" else key
                ds=[np.array([a[metric]-b[metric] for a,b in zip(per[k][left+"_"+fam],per[k][right+"_"+fam])]) for k in cohorts]
                d=np.concatenate(ds);rng=np.random.default_rng(7611)
                boot=np.mean([g[rng.integers(0,len(g),(4000,len(g)))].mean(1) for g in ds],axis=0)
                paired[left+"_vs_"+right+"_"+fam+"_"+key]={"delta":float(d.mean()),"n":len(d),"CI95":np.quantile(boot,[.025,.975]).tolist(),
                    "p_unadjusted":1. if np.all(d==0) else float(wilcoxon(d).pvalue)}
    prev=0.
    for i,v in enumerate(sorted(paired.values(),key=lambda x:x["p_unadjusted"])):
        prev=max(prev,min(1.,(len(paired)-i)*v["p_unadjusted"]));v["p_holm"]=prev
    path=a.out/"measured_loss_models.pkl"
    with path.open("wb") as f:pickle.dump({"models":chosen,"references":refs,"calibration":cal,"master_audit":master_audit},f,pickle.HIGHEST_PROTOCOL)
    with path.open("rb") as f:reload=pickle.load(f)
    for n,(x,z) in audit_cache.items():assert np.array_equal(z,reload["models"][n].predict_levels(x))
    report={"source_run":45479473008,"scope":"matched full TICD refit with measured and isolated oracle loss inputs",
        "real_field_validation":False,"external_audit_is_synthetic":True,"all_original_invariants_preserved":True,
        "test_used_for_selection":False,"training_seeds":TRAIN,"validation_seeds":VAL,"development_seeds":DEV,"test_seeds":TEST,
        "references":{m:r.manifest() for m,r in refs.items()},"candidates":{n:m.manifest() for n,m in candidates.items()},
        "selection":selection,"selected":selected,"development":dev,"development_by_seed":dp,"heldout":held,"per_seed":per,
        "summary":summary,"paired":paired,"metadata":meta,"master_audit":masters,"measurement_diagnostics":measurements,
        "oracle_only_control":True,"all_active_references_refit":True,"oracle_removal_verified":True,"prefix_causality":True,
        "label_and_batch_invariance":True,"observational_equivalence_verified":True,"measured_fallback_verified":True,
        "commissioning_runs":len(seeds),"probe_load_accounted":True,"passive_sensor_assumed":True,"reload_verified":True}
    (a.out/"measured_loss_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("MEASURE_FINAL",json.dumps({"main_runs":len(seeds),"commissioning_runs":len(seeds),"test_runs":180,
        "candidates":len(candidates),"selected":selected,"paired":paired,"audits_passed":True}),flush=True)
if __name__=="__main__":main()
