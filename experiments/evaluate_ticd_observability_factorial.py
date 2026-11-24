#!/usr/bin/env python3
"""Prespecified factorial and observation study; no selection on final tests."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
from ticd_observability_factorial import FactorialTICD,FACTORS,PolicyEvidence,MultiObserverTICD
from simulate_ticd_extended_observation import build,KINDS
from evaluate_ticd_invariant_ablation import CachedDecision
from evaluate_ticd_two_tier import metrics
LOSSES=(1,5,7,10,12,15,20,25)
TRAIN={m:[91001+100*i+j for j in range(3)] for i,m in enumerate(LOSSES)}
VAL={m:[92001+100*i+j for j in range(3)] for i,m in enumerate(LOSSES)}
TEST={k:[94001+100*i+j for j in range(5)] for i,k in enumerate(KINDS)}
def group(seeds,cal,kind,causal):
    parts=[build(s,cal,kind,causal) for s in seeds]
    d=pd.concat([p[0] for p in parts],ignore_index=True)
    masters=pd.concat([p[1] for p in parts],ignore_index=True)
    print("OBS_RUNS",json.dumps({"kind":kind,"seeds":seeds,"rows":len(d),"master_rows":len(masters),
       "slave_attack_rows":int(d.is_anomaly.sum()),"master_policy_rows":int(masters.is_anomaly.sum())}),flush=True)
    return d,masters,[p[2] for p in parts]
def rate(a,b):return float(a/b) if b else None
def evaluation(model,df,hybrid=False):
    valid=df.primary_observer_valid.to_numpy(dtype=bool)
    # No method sees inaccessible primary data at decision time.
    if hybrid:y=model.predict_levels(df)
    else:
        y=np.zeros(len(df),dtype=np.int8)
        if valid.any():y[valid]=model.predict_levels(df.loc[valid])
    cache=CachedDecision(df,y);out=metrics(cache,df)
    a=y==2;w=y==1;attack=df.is_anomaly.to_numpy(dtype=bool)
    audit=df.audit_observed_count.to_numpy()>0
    complete=valid&audit if hybrid else valid
    observed=(valid|audit) if hybrid else valid
    out.update({"complete_rows":int(complete.sum()),"partial_rows":int((observed&~complete).sum()),
         "unobservable_rows":int((~observed).sum()),"unobservable_attack_rows":int((~observed&attack).sum()),
         "observed_attack_recall":rate(int((a&attack&observed).sum()),int((attack&observed).sum())),
         "observed_normal_FPR":rate(int((a&~attack&observed).sum()),int((~attack&observed).sum())),
         "unobservable_no_signal_rows":int((~observed&~a&~w).sum())})
    if "eval_active_deficit" in df:
        active=df.eval_active_deficit.fillna(False).to_numpy(dtype=bool)&attack
        out["active_deficit_recall"]=rate(int((a&active).sum()),int(active.sum()))
        out["quiet_compromise_recall"]=rate(int((a&attack&~active).sum()),int((attack&~active).sum()))
    # Time until first alert for each affected node; undetected nodes remain
    # right-censored. Delay is not the mean over only successful detections.
    delays=[]
    for (s,n),g in df.loc[attack].groupby(["simulation_seed","node_id"]):
        hit=g.loc[cache.levels.loc[g.index].to_numpy()==2,"interval"]
        first=int(g.interval.min());last=int(g.interval.max())
        delays.append({"seed":int(s),"node":int(n),"detected":bool(len(hit)),
           "delay_seconds":int((hit.min()-first)*20) if len(hit) else None,
           "censor_seconds":int((last-first+1)*20)})
    out["node_delay"]=delays
    return out,cache
def short(x):
    return {k:x.get(k) for k in ("confirmed_alert_FPR","false_warning_rate","confirmed_recall",
          "combined_attack_coverage","observed_attack_recall","unobservable_rows",
          "unobservable_attack_rows","active_deficit_recall","quiet_compromise_recall")}
def master_metrics(mod,df):
    a=mod.status(df);y=df.is_anomaly.to_numpy(dtype=bool)
    return {"rows":len(df),"positive":int(y.sum()),"normal":int((~y).sum()),"TP":int((a&y).sum()),
       "FP":int((a&~y).sum()),"recall":rate(int((a&y).sum()),int(y.sum())),
       "FPR":rate(int((a&~y).sum()),int((~y).sum())),"unavailable_rows":int((df.audit_observed_count==0).sum())}
def factor_name(f):return "+".join(f) if f else "none"
def factorial_effects(by_seed,policy):
    # Average each factor's paired effect over the four settings of the
    # other two factors on each independent run. Conditional scenarios are
    # intentionally kept separate. No row-level significance tests.
    out={}
    for k,pairs in by_seed.items():
        out[k]={}
        for subset in [s for order in (1,2,3) for s in combinations(("I1","I2","D"),order)]:
            factor=":".join(subset)
            out[k][factor]={}
            for metric in ("confirmed_recall","confirmed_alert_FPR"):
                differences=[]
                for idx in range(len(TEST[k])):
                    vals=[pairs[policy+"__"+factor_name(f)][idx][metric] for f in FACTORS]
                    if any(v is None for v in vals):continue
                    # Mean signed difference over the remaining factors.
                    signed=[((-1)**(len(subset)-len(set(subset)&set(f))))*v for f,v in zip(FACTORS,vals)]
                    differences.append(float(sum(signed)/(2**(3-len(subset)))))
                if differences:
                    out[k][factor][metric]={"mean_paired_effect":float(np.mean(differences)),
                        "per_seed":differences}
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.prior/"role_adaptive_causal_ticd.pkl").open("rb") as f:pkg=pickle.load(f)
    assert pkg["selected"]=="weighted_w0.5_g0.0_b0.008"
    source=pkg["model"];cal=pkg["calibration"]
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    causal=source.prior.prior.prior.source_model.causal_model
    seeds=[s for p in (TRAIN,VAL,TEST) for xs in p.values() for s in xs]
    assert len(seeds)==len(set(seeds)) and min(seeds)>85901 and 97901 not in seeds
    train_data={m:group(ls,cal,"clean_loss"+str(m),causal) for m,ls in TRAIN.items()}
    tr=pd.concat([v[0] for v in train_data.values()],ignore_index=True)
    mt=pd.concat([v[1] for v in train_data.values()],ignore_index=True)
    val_data={m:group(ls,cal,"clean_loss"+str(m),causal) for m,ls in VAL.items()}
    va={m:x[0] for m,x in val_data.items()};mv={m:x[1] for m,x in val_data.items()}
    models={};references={}
    for policy in ("frozen","refit"):
        for f in FACTORS:
            name=policy+"__"+factor_name(f)
            mod=FactorialTICD(source,f,policy=="refit")
            if policy=="refit":mod.fit_normal(tr,va)
            models[name]=mod;references[name]=mod.manifest()
            print("FACTOR_REFERENCE",json.dumps({"model":name,"manifest":mod.manifest()}),flush=True)
    policy=PolicyEvidence().fit(tr,va)
    master_policy=PolicyEvidence().fit(mt,mv)
    hybrids={"hybrid_full":MultiObserverTICD(models["refit__I1+I2+D"],policy).calibrate(va),
             "hybrid_D":MultiObserverTICD(models["refit__D"],policy).calibrate(va)}
    models.update(hybrids)
    for n,mod in hybrids.items():
        references[n]={"threshold":mod.threshold,"audit_threshold":policy.threshold,"audit_reference":policy.refs}
        for x in va.values():
            assert (mod.predict_levels(x)==2).mean()<=.008+1e-12
    # Counterfactual exact equality: same RNG and same observations, different
    # labels only. This is deliberately NOT an independent second test run.
    x,m,_=build(97901,cal,"clean_loss10",causal)
    l,lm,_=build(97901,cal,"latent_identical",causal)
    labels=["is_anomaly","attack_type","attacker_sophistication"]
    pd.testing.assert_frame_equal(x.drop(columns=labels),l.drop(columns=labels))
    assert l.is_anomaly.sum()>0 and not x.is_anomaly.any()
    assert np.array_equal(source.predict_levels(x),models["frozen__I1+I2+D"].predict_levels(x))
    for name,mod in models.items():
        z=mod.predict_levels(x)
        assert np.array_equal(z,mod.predict_levels(l))
        cols=[c for c in x if c.startswith("gt_") or c in labels+["is_compromised","normal_event"]]
        assert np.array_equal(z,mod.predict_levels(x.drop(columns=cols)))
        sub=x.sample(n=200,random_state=19)
        assert np.array_equal(z[sub.index.to_numpy()],mod.predict_levels(sub))
    for name in ("frozen__none","refit__none"):assert not models[name].predict_levels(x).any()
    heldout={};by_seed={};masters={};metadata={}
    for k,ls in TEST.items():
        d,md,meta=group(ls,cal,k,causal);metadata[k]=meta
        heldout[k]={};by_seed[k]={}
        for name,mod in models.items():
            v,cache=evaluation(mod,d,name in hybrids)
            heldout[k][name]=v;by_seed[k][name]=[]
            for s,g in d.groupby("simulation_seed"):
                # Already frozen predictions indexed to the exact common run.
                v2=metrics(cache,g);by_seed[k][name].append({"seed":int(s),**v2})
            for c in ("confirmed_FP","confirmed_TP","normal","attack"):
                assert sum(z[c] for z in by_seed[k][name])==v[c]
        masters[k]=master_metrics(master_policy,md)
        print("OBS_HELDOUT",json.dumps({"cohort":k,"master":masters[k],**{n:short(v) for n,v in heldout[k].items()}}),flush=True)
    effects={p:factorial_effects(by_seed,p) for p in ("frozen","refit")}
    # Paired run-level descriptive comparison of the full and D-only
    # architecture on native attack scenarios. No post-test model selection.
    native_attack=[k for k in KINDS if k=="default" or k.startswith(("role_","group_","weak_")) or k=="variable_deficit"]
    comparisons={}
    for policy in ("frozen","refit"):
        for metric in ("confirmed_recall","confirmed_alert_FPR"):
            ds=[]
            for k in native_attack:
                u=by_seed[k][policy+"__I1+I2+D"];v=by_seed[k][policy+"__D"]
                ds.append(np.array([a[metric]-b[metric] for a,b in zip(u,v)]))
            flat=np.concatenate(ds);rng=np.random.default_rng(1981)
            boot=np.mean([d[rng.integers(0,len(d),(4000,len(d)))].mean(1) for d in ds],axis=0)
            comparisons[policy+"__"+metric]={"difference_full_minus_D":float(flat.mean()),"n":len(flat),
              "CI95":np.quantile(boot,[.025,.975]).tolist(),
              "p_unadjusted":1. if np.all(flat==0) else float(wilcoxon(flat).pvalue)}
    for i,(n,v) in enumerate(sorted(comparisons.items(),key=lambda x:x[1]["p_unadjusted"])):
        p=min(1.,(len(comparisons)-i)*v["p_unadjusted"])
        if i==0:prev=0.
        prev=max(prev,p);v["p_holm"]=prev
    report={"scope":"native SIP simulation plus separate stylized external policy-audit model",
      "real_field_validation":False,"observational_equivalence_verified":True,
      "source_run":45377337545,"training_seeds":TRAIN,"validation_seeds":VAL,"test_seeds":TEST,
      "references":references,"heldout":heldout,"per_seed":by_seed,"master_audit":masters,
      "metadata":metadata,"factorial_main_and_interaction_effects":effects,"paired_full_vs_D":comparisons,
      "test_used_for_selection":False,"external_audit_is_synthetic":True,
      "no_native_SBC_compromise_model":True,"label_and_batch_invariance":True,
      "normal_means_no_alert_not_known_benign":True}
    (a.out/"observability_factorial_report.json").write_text(json.dumps(report,indent=2)+"\n")
    with (a.out/"observability_factorial_models.pkl").open("wb") as f:pickle.dump({"models":models,"master_policy":master_policy,"calibration":cal},f,pickle.HIGHEST_PROTOCOL)
    print("OBS_FINAL",json.dumps({"scenarios":len(KINDS),"test_runs":sum(map(len,TEST.values())),
         "main_runs":len(seeds),"factorial_variants":16,"hybrids":2,
         "counterfactual_verified":True,"paired_full_vs_D":comparisons}),flush=True)
if __name__=="__main__":main()
