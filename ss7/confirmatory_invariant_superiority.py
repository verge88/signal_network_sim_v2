"""Fixed-sample confirmation of incremental detection value.

Preregistered here before evaluating any of the new worlds:
  seeds 8801..8824; frozen full and strictly local HGB models;
  primary endpoint: macro episode recall at a COMMON retrospective budget
  of <=20 false-positive node-windows per network-day, full vs strict_local.
  This is an equal-budget discrimination diagnostic, not a deployable
  threshold. Ties are handled conservatively and realized budgets reported.
  The operational secondary endpoint uses joint_robust thresholds from
  independent compromise-free commissioning simulations only.
  Two-sided paired sign-flip test, independent simulation world as unit.
  One primary contrast, Holm adjustment for the secondary family.
  No exclusion/replacement of seeds, optional stopping or selection of
  the best policy on these results. All generated worlds are reported.
  Exact test for <=18 nonzero paired differences, otherwise fixed-seed
  Monte Carlo (200000 draws, add-one correction).
  Statistical superiority at equal retrospective budgets does not prove
  operating budget compliance, physical-layer validity or real-network
  generalization.
"""
from pathlib import Path
import argparse
import itertools
import json
import joblib
import numpy as np
import pandas as pd
from ss7 import fix as F
from ss7.threshold_transfer_study import world,score
from ss7.incident_calibration_study import (elapsed,triple,bucket_peaks,
     threshold,metrics,RAW_BUDGET,INCIDENT_BUDGET,WARMUP)

SEEDS=tuple(range(8801,8825))
MODELS=("strict_local","strict_local_plus_inv","report_full")
POLICIES=("equal_budget_diagnostic","joint_robust")
PRIMARY=("report_full","strict_local","equal_budget_diagnostic","recall")
N_BOOT=20000
N_RANDOM=200000
ALPHA=.05

def test(seed,localpath,frozenpath,out):
    if seed not in SEEDS:raise ValueError("not a registered test world")
    d,sim,eps=world(seed,1.5,32,64,10)
    eligible={}
    for i,e in enumerate(eps):
        if e.kind!="compromise" or e.t0<WARMUP:continue
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        if len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q):eligible[i]=e
    if len(eligible)<12:
        raise RuntimeError(f"Registered world {seed} has only {len(eligible)} complete episodes; no seed substitution")
    # Independent commissioning periods; same topology and fixed parameters
    # as the test, no compromise episodes. No test labels enter these cutoffs.
    a=world(seed+10000,3.,0,128,20,sim)[0]
    b=world(seed+20000,3.,0,192,32,sim)[0]
    assert not a.label.any() and not b.label.any()
    assert set(a.nid)==set(b.nid)==set(d.nid)
    frozen=joblib.load(frozenpath);local=joblib.load(localpath)
    models={**frozen,**local}
    for name in ("strict_local","strict_local_plus_inv"):
        assert all(not c.endswith("__dev_zone") for c in models[name][0])
    assert all(any(c==base or c.startswith(base+"__") for base in F.OWN)
               for c in models["strict_local"][0])
    rows=[];events=[];controls=[]
    for name in MODELS:
        st,sa,sb=[score(models[name],x) for x in (d,a,b)]
        zt,za,zb=[triple(x,s) for x,s in ((d,st),(a,sa),(b,sb))]
        tau=max(threshold(sa,RAW_BUDGET,elapsed(a)),
                threshold(sb,RAW_BUDGET,elapsed(b)),
                threshold(bucket_peaks(a,za),INCIDENT_BUDGET,elapsed(a)),
                threshold(bucket_peaks(b,zb),INCIDENT_BUDGET,elapsed(b)))
        # Label-based test threshold is *only* the registered diagnostic.
        q=threshold(st[d.label.to_numpy()==0],RAW_BUDGET,elapsed(d))
        for policy,thr in (("joint_robust",tau),("equal_budget_diagnostic",q)):
            m,det=metrics(d,st,zt,thr,eligible)
            fp=int(((st>thr)&(d.label.to_numpy()==0)).sum())
            if policy=="equal_budget_diagnostic":
                assert fp<=int(np.floor(RAW_BUDGET*elapsed(d)))
            rows.append(dict(seed=seed,model=name,policy=policy,false_windows=fp,**m))
            for i,e in eligible.items():
                events.append(dict(seed=seed,model=name,policy=policy,episode=i,
                     detected=int(i in det),delay_seconds=det.get(i),
                     hidden_rate=e.hidden_rate,mask_family=e.family))
        for phase,x,s,z in (("matched",a,sa,za),("stress",b,sb,zb)):
            m,_=metrics(x,s,z,tau)
            controls.append(dict(seed=seed,model=name,phase=phase,**m))
        print("evaluated",seed,name,flush=True)
    for name,values in (("metrics",rows),("episodes",events),("controls",controls)):
        pd.DataFrame(values).to_csv(out/(name+".csv"),index=False)
    (out/"manifest.json").write_text(json.dumps(dict(seed=seed,complete=len(eligible),
       generated=32,raw_budget=RAW_BUDGET,notification_budget=INCIDENT_BUDGET,
       model_runs=[45421952080,45440533210],primary=PRIMARY,
       features={n:models[n][0] for n in MODELS}),indent=2))
    print(pd.DataFrame(rows)[["model","policy","episodes","recall","false_windows_day","false_notifications_day"]].to_string(index=False))

def collect(root,name):
    paths=sorted(root.rglob(name+".csv"))
    if len(paths)!=len(SEEDS):raise ValueError(f"{name}: incomplete registered worlds")
    d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    if set(d.seed)!=set(SEEDS):raise ValueError("registered seed mismatch")
    return d

def paired(x,y,rng):
    z=np.asarray(x,dtype=float)-np.asarray(y,dtype=float)
    if len(z)!=len(SEEDS) or not np.isfinite(z).all():raise ValueError("invalid world differences")
    nonzero=z[np.abs(z)>1e-14]
    observed=abs(z.sum())
    k=len(nonzero)
    if k<=18:
        sums=np.fromiter((abs(np.dot(s,nonzero)) for s in itertools.product((-1,1),repeat=k)),float)
        p=float(np.mean(sums>=observed-1e-12));method="exact_sign_flip"
    else:
        count=0
        for _ in range(N_RANDOM//10000):
            signs=rng.integers(0,2,size=(10000,k),dtype=np.int8)*2-1
            count+=int((np.abs(signs@nonzero)>=observed-1e-12).sum())
        p=(count+1)/(N_RANDOM+1);method="monte_carlo_sign_flip"
    boot=rng.choice(z,size=(N_BOOT,len(z)),replace=True).mean(axis=1)
    return dict(n=len(z),n_nonzero=k,delta=float(z.mean()),
        low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),
        p_raw=p,test=method,positive_worlds=int((z>1e-14).sum()),
        negative_worlds=int((z<-1e-14).sum()))

def aggregate(root,out):
    m=collect(root,"metrics");e=collect(root,"episodes");c=collect(root,"controls")
    keys=["seed","model","policy"]
    if m.duplicated(keys).any() or len(m)!=len(SEEDS)*len(MODELS)*len(POLICIES):
        raise ValueError("incomplete registered factorial")
    if set(m.model)!=set(MODELS) or set(m.policy)!=set(POLICIES):raise ValueError("unexpected variants")
    if e.duplicated(keys+["episode"]).any():raise ValueError("duplicate episode results")
    for seed in SEEDS:
        g=e[e.seed==seed]
        supports=[set(a.episode) for _,a in g.groupby(["model","policy"])]
        if len(supports)!=len(MODELS)*len(POLICIES) or not all(a==supports[0] for a in supports):
            raise ValueError("unpaired episode support")
    summary=m.groupby(["model","policy"]).agg(worlds=("seed","nunique"),
        episodes=("episodes","sum"),detected=("detected","sum"),recall=("recall","mean"),
        fpr=("fpr","mean"),raw_day=("false_windows_day","mean"),
        notices_day=("false_notifications_day","mean"),
        mixed_notices=("mixed_notifications","sum"),
        worlds_raw_budget=("false_windows_day",lambda x:int((x<=RAW_BUDGET).sum())),
        worlds_notice_budget=("false_notifications_day",lambda x:int((x<=INCIDENT_BUDGET).sum()))).reset_index()
    rng=np.random.default_rng(8800)
    secondary=[
       ("strict_local_plus_inv","strict_local","equal_budget_diagnostic","recall"),
       ("report_full","strict_local","joint_robust","recall"),
       ("report_full","strict_local","joint_robust","false_windows_day"),
       ("report_full","strict_local","joint_robust","false_notifications_day"),
       ("strict_local_plus_inv","strict_local","joint_robust","recall")]
    tests=[]
    for a,b,pol,metric in [PRIMARY]+secondary:
        x=m[(m.model==a)&(m.policy==pol)].set_index("seed").sort_index()
        y=m[(m.model==b)&(m.policy==pol)].set_index("seed").sort_index()
        assert x.index.equals(y.index)
        tests.append(dict(a=a,b=b,policy=pol,metric=metric,
                          family="primary" if (a,b,pol,metric)==PRIMARY else "secondary",
                          **paired(x[metric],y[metric],rng)))
    tests[0]["p_adjusted"]=tests[0]["p_raw"]
    running=0.
    for i,t in enumerate(sorted(tests[1:],key=lambda t:t["p_raw"])):
        running=max(running,min(1.,(len(secondary)-i)*t["p_raw"]))
        t["p_adjusted"]=running
    diagnostic=m[m.policy=="equal_budget_diagnostic"].pivot(index="seed",columns="model",values="false_windows")
    same=int((diagnostic.report_full==diagnostic.strict_local).sum())
    max_diff=int((diagnostic.report_full-diagnostic.strict_local).abs().max())
    primary=tests[0]
    op=summary[(summary.model=="report_full")&(summary.policy=="joint_robust")].iloc[0]
    result=dict(primary=primary,diagnostic_exact_budget_matches=same,
        max_false_window_count_difference=max_diff,
        diagnostic_superiority=bool(primary["delta"]>0 and primary["p_adjusted"]<ALPHA),
        operational_full_raw_budget_all_worlds=bool(op.worlds_raw_budget==len(SEEDS)),
        operational_full_notice_budget_all_worlds=bool(op.worlds_notice_budget==len(SEEDS)),
        note="Common retrospective upper budget is not a deployable threshold; operational superiority requires separate evidence. No guarantee of matched false-positive count with ties.")
    for name,d in (("summary",summary),("paired_tests",pd.DataFrame(tests)),
       ("all_metrics",m),("episodes",e),("controls",c),("diagnostic_budget_audit",diagnostic.reset_index())):
        d.to_csv(out/(name+".csv"),index=False)
    (out/"conclusion.json").write_text(json.dumps(result,indent=2))
    print(summary.to_string(index=False));print(pd.DataFrame(tests).to_string(index=False))
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["test","aggregate"])
    p.add_argument("--seed",type=int);p.add_argument("--local",type=Path)
    p.add_argument("--frozen",type=Path);p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="test":test(a.seed,a.local,a.frozen,a.output)
    else:aggregate(a.input,a.output)
