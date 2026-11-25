"""Development-only comparison of role thresholds and causal score pooling.

Frozen classifiers; independent labelled mixed simulated references.
No test labels or future windows enter a score or decision. Classification
of notifications follows the inherited three-window endpoint, even when
the score incorporates earlier history. Not a real-network validation.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ss7.threshold_transfer_study import world, score
from ss7.incident_calibration_study import (
    elapsed, triple, bucket_peaks, threshold, metrics,
    RAW_BUDGET, INCIDENT_BUDGET, WARMUP,
)

SEEDS=tuple(range(9001,9009))
VARIANTS=(
    ("strict_local","global",1,.5),
    ("report_full","global",1,.5),
    ("report_full","role",1,.5),
    ("report_full","global",3,.5),
    ("report_full","role",3,.5),
    ("report_full","global",6,.5),
    ("report_full","role",6,.5),
    ("report_full","role",6,.25),
)
def variant(v):
    name,mode,k,f=v
    return f"{name}_{mode}_k{k}_f{f}"

def causal(d,s,k):
    """Trailing log-odds mean; restart at a node boundary or time gap."""
    s=np.asarray(s,dtype=float)
    if k<1 or len(s)!=len(d) or not np.isfinite(s).all():
        raise ValueError("invalid causal input")
    if k==1:return s.copy()
    n=d.nid.to_numpy();t=d.t.to_numpy()
    if len(s)==0:return s.copy()
    x=np.clip(s,1e-9,1-1e-9)
    l=np.log(x)-np.log1p(-x)
    breaks=np.r_[True,(n[1:]!=n[:-1])|(t[1:]!=t[:-1]+1)]
    groups=np.cumsum(breaks)
    z=pd.Series(l).groupby(groups,sort=False).rolling(k,min_periods=1).mean().droplevel(0).to_numpy()
    return 1/(1+np.exp(-z))

def cutoffs(d,s,raw,notice):
    """Return separate thresholds for raw windows and confirmed notifications."""
    neg=d.label.to_numpy()==0
    z=triple(d,np.where(neg,s,-np.inf))
    b=bucket_peaks(d,z)
    if not neg.any():raise ValueError("missing negative reference")
    nr=threshold(s[neg],raw,elapsed(d))
    # No all-negative consecutive windows means no estimable incident tail;
    # fail closed rather than silently skipping the constraint.
    ni=threshold(b,notice,elapsed(d)) if np.isfinite(b).any() else 1.
    return nr,ni

def cutoff(d,s,raw,notice):
    return max(cutoffs(d,s,raw,notice))

def calibrate(refs,ss,mode,fraction):
    """Allocate the network-wide budgets proportionally to monitored nodes."""
    a=refs[0]
    if mode=="global":
        return {"all":max(cutoff(d,s,RAW_BUDGET*fraction,INCIDENT_BUDGET*fraction)
                          for d,s in zip(refs,ss))}
    types=a[["nid","ntype"]].drop_duplicates()
    if types.nid.duplicated().any():raise ValueError("changing node type")
    result={}
    for role,count in types.groupby("ntype").size().items():
        share=count/len(types)
        th=[]
        for d,s in zip(refs,ss):
            mask=d.ntype.to_numpy()==role
            q=d.loc[mask].reset_index(drop=True)
            if q.nid.nunique()!=count:raise ValueError("role topology mismatch")
            th.append(cutoff(q,s[mask],RAW_BUDGET*fraction*share,INCIDENT_BUDGET*fraction*share))
        result[role]=max(th)
    return result

def margin(d,s,th):
    if "all" in th:return s-th["all"]
    tau=d.ntype.map(th).to_numpy()
    if not np.isfinite(tau).all():raise ValueError("unknown monitored node type")
    return s-tau

def test(seed,localpath,frozenpath,out):
    if seed not in SEEDS:raise ValueError("unregistered development seed")
    d,sim,eps=world(seed,1.5,32,64,10)
    eligible={}
    for i,e in enumerate(eps):
        if e.kind!="compromise" or e.t0<WARMUP:continue
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        if len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q):eligible[i]=e
    if len(eligible)<12:raise ValueError("insufficient episodes; no replacement")
    a=world(seed+10000,3.,0,128,20,sim)[0]
    b=world(seed+20000,3.,0,192,32,sim)[0]
    c=world(seed+40000,3.,64,128,20,sim)[0]
    assert not a.label.any() and not b.label.any() and c.label.any()
    assert all(set(x.nid)==set(d.nid) for x in (a,b,c))
    frozen=joblib.load(frozenpath);local=joblib.load(localpath)
    models={"strict_local":local["strict_local"],"report_full":frozen["report_full"]}
    frames=(a,b,c,d)
    scores={n:[score(m,x) for x in frames] for n,m in models.items()}
    cache={}
    rows=[];events=[];thresholds=[]
    for v in VARIANTS:
        name,mode,k,f=v
        key=variant(v)
        if (name,k) not in cache:
            cache[name,k]=[causal(x,s,k) for x,s in zip(frames,scores[name])]
        ss=cache[name,k]
        th=calibrate(frames[:3],ss[:3],mode,f)
        u=margin(d,ss[3],th)
        m,det=metrics(d,u,triple(d,u),0.,eligible)
        rows.append(dict(seed=seed,variant=key,model=name,mode=mode,k=k,fraction=f,**m))
        thresholds.extend(dict(seed=seed,variant=key,role=role,cutoff=tau) for role,tau in th.items())
        z=triple(d,u)
        for i,e in eligible.items():
            ix=(d.episode_id.to_numpy()==i)&(d.nid.to_numpy()==e.nid)
            events.append(dict(seed=seed,variant=key,episode=i,
                detected=int(i in det),delay_seconds=det.get(i),hidden_rate=e.hidden_rate,
                mask_family=e.family,ntype=sim.topo.nodes[e.nid].ntype.value,
                duration_windows=int(ix.sum()),peak_margin=float(u[ix].max()),
                confirmed_candidate=int(np.any(z[ix]>0))))
        print(seed,key,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    for name,table in (("metrics",rows),("episodes",events),("thresholds",thresholds)):
        pd.DataFrame(table).to_csv(out/(name+".csv"),index=False)
    (out/"manifest.json").write_text(json.dumps(dict(seed=seed,complete=len(eligible),generated=32,
        reference_seeds=[seed+10000,seed+20000,seed+40000],model_runs=[45421952080,45440533210],
        scope="development only; no test threshold fit; labelled synthetic mixed reference"),indent=2))

def aggregate(root,out):
    def collect(name):
        p=sorted(root.rglob(name+".csv"))
        if len(p)!=len(SEEDS):raise ValueError("incomplete registered worlds")
        return pd.concat([pd.read_csv(x) for x in p],ignore_index=True)
    m,e,t=collect("metrics"),collect("episodes"),collect("thresholds")
    keys=["seed","variant"]
    expected={(s,variant(v)) for s in SEEDS for v in VARIANTS}
    if m.duplicated(keys).any() or set(m[keys].itertuples(index=False,name=None))!=expected:
        raise ValueError("invalid factorial")
    if e.duplicated(keys+["episode"]).any():raise ValueError("duplicate episodes")
    for seed in SEEDS:
        supports=[set(x.episode) for _,x in e[e.seed==seed].groupby("variant")]
        if len(supports)!=len(VARIANTS) or not all(x==supports[0] for x in supports):
            raise ValueError("unpaired episode support")
    m["both_budgets"]=(m.false_windows_day<=RAW_BUDGET)&(m.false_notifications_day<=INCIDENT_BUDGET)
    summary=m.groupby("variant").agg(worlds=("seed","nunique"),episodes=("episodes","sum"),
        detected=("detected","sum"),recall=("recall","mean"),fpr=("fpr","mean"),
        raw_day=("false_windows_day","mean"),worst_raw=("false_windows_day","max"),
        notice_day=("false_notifications_day","mean"),worst_notice=("false_notifications_day","max"),
        both_budgets=("both_budgets","sum"),median_world_delay=("median_delay_seconds","median")).reset_index()
    e["rate_band"]=pd.cut(e.hidden_rate,[-np.inf,.01,.05,.1,np.inf],include_lowest=True)
    breakdown=e.groupby(["variant","mask_family","ntype","rate_band"],observed=True).agg(
        episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean")).reset_index()
    for name,table in (("all_metrics",m),("episodes",e),("thresholds",t),("summary",summary),("breakdown",breakdown)):
        table.to_csv(out/(name+".csv"),index=False)
    print(summary.to_string(index=False))
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
