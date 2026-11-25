"""Development: fast path plus bounded causal CUSUM, frozen SS7 models.

Independent labelled synthetic commissioning references only. Test labels
are used exclusively for evaluation, never state resets or threshold fitting.
Keep the inherited three-consecutive-window and 30-minute bucket endpoint.
No production or statistical superiority claim from this screening.
"""
import argparse
import itertools
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ss7.threshold_transfer_study import world,score
from ss7.incident_calibration_study import (
    elapsed,triple,bucket_peaks,threshold,metrics,
    RAW_BUDGET,INCIDENT_BUDGET,WARMUP,
)

SEEDS=tuple(range(9101,9109))
# model, memory (0 = direct baseline), reference quantile, fast raw budget share
VARIANTS=(
    ("strict_local",0,0.,1.),
    ("report_full",0,0.,1.),
    ("strict_local",6,.99,.5),
    ("report_full",6,.99,.5),
    ("report_full",12,.99,.5),
    ("report_full",6,.95,.5),
    ("report_full",12,.95,.5),
    ("report_full",6,.99,.75),
)
FRACTION=.5

def name(v):
    model,k,q,a=v
    return model if not k else f"{model}_cusum{k}_q{q}_a{a}"

def logit(s):
    s=np.asarray(s,dtype=float)
    if not np.isfinite(s).all():raise ValueError("nonfinite scores")
    s=np.clip(s,1e-9,1-1e-9)
    return np.log(s)-np.log1p(-s)

def bounded_cusum(d,s,drift,k):
    """Maximum positive trailing sum over 1..k consecutive intervals.
    Vectorized finite-memory Page statistic with clipped log-odds increments.
    Never cross node boundaries or missing intervals.
    """
    if k<1 or len(d)!=len(s) or not np.isfinite(drift):
        raise ValueError("invalid CUSUM input")
    inc=np.clip(logit(s)-drift,-3.,3.)
    if not len(inc):return inc
    n=d.nid.to_numpy();t=d.t.to_numpy()
    start=np.r_[True,(n[1:]!=n[:-1])|(t[1:]!=t[:-1]+1)]
    group=np.cumsum(start)
    best=np.maximum(inc,0.)
    accum=inc.copy()
    for lag in range(1,min(k,len(inc))):
        valid=group[lag:]==group[:-lag]
        accum[lag:]+=inc[:-lag]
        best[lag:]=np.maximum(best[lag:],np.where(valid,accum[lag:],0.))
    return best/k

def raw_cut(refs,ss,budget):
    return max(threshold(s[d.label.to_numpy()==0],budget,elapsed(d))
               for d,s in zip(refs,ss))

def joint_cut(refs,ss,raw,notice):
    th=[]
    for d,s in zip(refs,ss):
        neg=d.label.to_numpy()==0
        b=bucket_peaks(d,triple(d,np.where(neg,s,-np.inf)))
        if not neg.any():raise ValueError("missing negative reference")
        # With no finite negative triples, no false confirmed candidate can
        # occur in this reference. Still enforce its raw-window constraint.
        th.append(threshold(s[neg],raw,elapsed(d)))
        if np.isfinite(b).any():th.append(threshold(b,notice,elapsed(d)))
    return max(th)

def fit(refs,ss,k,q,share):
    """Fit only on the three independent reference frames."""
    if k==0:
        return dict(k=0,fast=joint_cut(refs,ss,RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION),
                    extra=0.)
    # Conservative drift across reference regimes, strictly from negatives.
    drift=max(float(np.quantile(logit(s[d.label.to_numpy()==0]),q)) for d,s in zip(refs,ss))
    cs=[bounded_cusum(d,s,drift,k) for d,s in zip(refs,ss)]
    fast=raw_cut(refs,ss,RAW_BUDGET*FRACTION*share)
    slow=raw_cut(refs,cs,RAW_BUDGET*FRACTION*(1-share))
    u=[np.maximum(s-fast,c-slow) for s,c in zip(ss,cs)]
    # Joint calibration counts mixed-path triples too. Never lower either
    # branch threshold; no independence assumption between channels.
    extra=max(0.,joint_cut(refs,u,RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION))
    return dict(k=k,fast=fast,slow=slow,drift=drift,extra=extra)

def apply(d,s,rule):
    f=s-rule["fast"]-rule["extra"]
    c=(bounded_cusum(d,s,rule["drift"],rule["k"])-rule["slow"]-rule["extra"]
       if rule["k"] else np.full(len(s),-np.inf))
    return np.maximum(f,c),f,c

def test(seed,localpath,frozenpath,out):
    if seed not in SEEDS:raise ValueError("unregistered development seed")
    d,sim,eps=world(seed,1.5,32,64,10)
    eligible={}
    for i,e in enumerate(eps):
        if e.kind!="compromise" or e.t0<WARMUP:continue
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        if len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q):eligible[i]=e
    if len(eligible)<12:raise ValueError("insufficient complete episodes; no replacement")
    refs=(world(seed+10000,3.,0,128,20,sim)[0],
          world(seed+20000,3.,0,192,32,sim)[0],
          world(seed+40000,3.,64,128,20,sim)[0])
    assert not refs[0].label.any() and not refs[1].label.any() and refs[2].label.any()
    assert all(set(x.nid)==set(d.nid) for x in refs)
    models={"strict_local":joblib.load(localpath)["strict_local"],
            "report_full":joblib.load(frozenpath)["report_full"]}
    scores={n:[score(m,x) for x in (*refs,d)] for n,m in models.items()}
    rows=[];events=[];rules=[];controls=[]
    y=d.label.to_numpy();ids=d.episode_id.to_numpy();nids=d.nid.to_numpy()
    for v in VARIANTS:
        model,k,q,share=v
        key=name(v);ss=scores[model]
        rule=fit(refs,ss[:3],k,q,share)
        rules.append(dict(seed=seed,variant=key,**rule))
        for j,(x,s) in enumerate(zip(refs,ss[:3])):
            u,_,_=apply(x,s,rule)
            m,_=metrics(x,u,triple(x,u),0.)
            controls.append(dict(seed=seed,variant=key,reference=j,**m))
        u,f,c=apply(d,ss[3],rule)
        z=triple(d,u)
        m,det=metrics(d,u,z,0.,eligible)
        rows.append(dict(seed=seed,variant=key,**m,
            fast_false_day=float(((f>0)&(y==0)).sum()/elapsed(d)),
            slow_only_false_day=float(((c>0)&(f<=0)&(y==0)).sum()/elapsed(d))))
        for i,e in eligible.items():
            ix=(ids==i)&(nids==e.nid)
            events.append(dict(seed=seed,variant=key,episode=i,detected=int(i in det),
                delay_seconds=det.get(i),hidden_rate=e.hidden_rate,mask_family=e.family,
                ntype=sim.topo.nodes[e.nid].ntype.value,duration_windows=int(ix.sum()),
                fast_windows=int((f[ix]>0).sum()),slow_only_windows=int(((c[ix]>0)&(f[ix]<=0)).sum()),
                candidate=int((z[ix]>0).any())))
        print(seed,key,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    for k,table in (("metrics",rows),("episodes",events),("rules",rules),("controls",controls)):
        pd.DataFrame(table).to_csv(out/(k+".csv"),index=False)
    (out/"manifest.json").write_text(json.dumps(dict(seed=seed,complete=len(eligible),generated=32,
        reference_seeds=[seed+10000,seed+20000,seed+40000],model_runs=[45421952080,45440533210],
        fraction=FRACTION,scope="development only, labelled simulated mixed reference"),indent=2))

def aggregate(root,out):
    def collect(k):
        p=sorted(root.rglob(k+".csv"))
        if len(p)!=len(SEEDS):raise ValueError("incomplete registered worlds")
        return pd.concat([pd.read_csv(x) for x in p],ignore_index=True)
    m,e,r,c=[collect(k) for k in ("metrics","episodes","rules","controls")]
    keys=["seed","variant"]
    expected={(s,name(v)) for s in SEEDS for v in VARIANTS}
    if m.duplicated(keys).any() or set(m[keys].itertuples(index=False,name=None))!=expected:
        raise ValueError("invalid factorial")
    if e.duplicated(keys+["episode"]).any():raise ValueError("duplicate episodes")
    for seed in SEEDS:
        support=[set(x.episode) for _,x in e[e.seed==seed].groupby("variant")]
        if len(support)!=len(VARIANTS) or not all(x==support[0] for x in support):
            raise ValueError("unpaired episode support")
    m["both_budgets"]=(m.false_windows_day<=RAW_BUDGET)&(m.false_notifications_day<=INCIDENT_BUDGET)
    summary=m.groupby("variant").agg(worlds=("seed","nunique"),episodes=("episodes","sum"),
        detected=("detected","sum"),recall=("recall","mean"),recall_sd=("recall","std"),
        fpr=("fpr","mean"),raw_day=("false_windows_day","mean"),worst_raw=("false_windows_day","max"),
        notice_day=("false_notifications_day","mean"),worst_notice=("false_notifications_day","max"),
        both_budgets=("both_budgets","sum"),median_world_delay=("median_delay_seconds","median")).reset_index()
    e["rate_band"]=pd.cut(e.hidden_rate,[-np.inf,.01,.05,.1,np.inf])
    breakdown=e.groupby(["variant","mask_family","ntype","rate_band"],observed=True).agg(
        episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean")).reset_index()
    base=e[e.variant=="report_full"][["seed","episode","detected"]].rename(columns={"detected":"base_detected"})
    transition=e.merge(base,on=["seed","episode"],validate="many_to_one")
    transition["rescued"]=(transition.detected==1)&(transition.base_detected==0)
    transition["lost"]=(transition.detected==0)&(transition.base_detected==1)
    gains=transition.groupby("variant").agg(rescued=("rescued","sum"),lost=("lost","sum")).reset_index()
    for k,table in (("all_metrics",m),("episodes",e),("rules",r),("controls",c),
                    ("summary",summary),("breakdown",breakdown),("transitions",gains)):
        table.to_csv(out/(k+".csv"),index=False)
    print(summary.to_string(index=False));print(gains.to_string(index=False))
    print("Development only; independent confirmation required.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["test","aggregate"])
    p.add_argument("--seed",type=int)
    p.add_argument("--local",type=Path)
    p.add_argument("--frozen",type=Path)
    p.add_argument("--input",type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="test":test(a.seed,a.local,a.frozen,a.output)
    else:aggregate(a.input,a.output)
