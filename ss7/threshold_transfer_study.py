"""Threshold-transfer experiment, with frozen classifiers from run 45421952080.
New worlds only; no test-label tuning in operational threshold variants.
An explicitly retrospective, label-based equal-budget diagnostic is separate.
OWN in the inherited models includes peer-zone deviations, so it is not a
strictly local-only baseline. False alarms refer to node compromise, not all
security events. Calibration is a simulated compromise-free commissioning period.
"""
from pathlib import Path
import argparse
import itertools
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from ss7 import fix as F
from ss7.report_only_invariant_study import NEW,both,LEGACY

SEEDS=tuple(range(8601,8609))
MODELS=("own","legacy_inv","report_inv","legacy_full","report_full","report_no_od","report_link_only")
POLICIES=("global_sparse","global_matched","node_pooled","node_robust")
BUDGET=20.
DAYS=1.5
WARMUP=50

def world(seed,days,comp,normal,external,template=None):
    cfg=F.SimConfig(seed=seed,days=days)
    F.all_invariants=both
    try:
        df,sim,eps=F.build_dataset(cfg,episodes_kw=dict(
          n_compromise=comp,n_normal=normal,n_ext=external),
          topo=template.topo if template is not None else None,
          sim_template=template)
    finally:
        F.all_invariants=LEGACY
    df=F.temporal_block(df,NEW,cfg.feat_window)
    df=F.zone_deviation(df,NEW,{n:v.zone for n,v in sim.topo.nodes.items()})
    df=df.loc[df.t>=WARMUP].sort_values(["nid","t"]).reset_index(drop=True)
    if df.duplicated(["nid","t"]).any():raise ValueError("duplicate windows")
    return df,sim,eps

def quantile(x,alpha):
    # Strict > threshold; no test-label feedback.
    if not len(x) or not np.isfinite(x).all():raise ValueError("bad score sample")
    return float(np.quantile(x,1-alpha,method="higher"))

def thresholds(cal,scores,nids,alpha):
    # Global sparse reproduces earlier event density. Matched and stressed
    # references have independent seeds but the same topology/flows.
    sparse,matched,stress=cal
    a,b,c=scores
    result={"global_sparse":{n:quantile(a,alpha) for n in nids},
            "global_matched":{n:quantile(b,alpha) for n in nids}}
    result["node_pooled"]={}
    result["node_robust"]={}
    for n in nids:
        bv=b[matched.nid.to_numpy()==n]
        cv=c[stress.nid.to_numpy()==n]
        if min(len(bv),len(cv))<2000:raise ValueError("insufficient calibration")
        result["node_pooled"][n]=quantile(np.r_[bv,cv],alpha)
        result["node_robust"][n]=max(quantile(bv,alpha),quantile(cv,alpha))
    return result

def score(model,df):
    cols,med,est=model
    x=df[cols].replace([np.inf,-np.inf],np.nan).fillna(med)
    s=est.predict_proba(x)[:,1]
    if not np.isfinite(s).all():raise ValueError("nonfinite predictions")
    return s

def incident_starts(d,fire):
    # Only three consecutive negative labelled windows on the same node
    # confirm a false incident. A gap/positive window resets the streak.
    n=d.nid.to_numpy();t=d.t.to_numpy();y=d.label.to_numpy()
    active=fire & (y==0)
    starts=0;run=0
    for i in range(len(d)):
        if not active[i]:run=0;continue
        run=run+1 if i and n[i]==n[i-1] and t[i]==t[i-1]+1 else 1
        if run==3:starts+=1
    return starts

def evaluate(seed,name,policy,d,s,fire,eligible,days,events):
    y=d.label.to_numpy();t=d.t.to_numpy();ids=d.episode_id.to_numpy()
    neg=y==0
    hits=[]
    for k,e in eligible.items():
        ix=np.flatnonzero(ids==k)
        run=0;delay=None
        for j,i in enumerate(ix):
            run=run+1 if fire[i] and (j==0 or t[i]==t[ix[j-1]]+1) else int(fire[i])
            if run>=3:
                delay=int((t[i]-e.t0)*30);break
        hits.append(delay)
        events.append(dict(seed=seed,model=name,policy=policy,episode=k,
              detected=int(delay is not None),delay_seconds=delay,
              hidden_rate=e.hidden_rate,mask_family=e.family))
    good=[x for x in hits if x is not None]
    fp=int(fire[neg].sum())
    return dict(seed=seed,model=name,policy=policy,episodes=len(hits),
         detected=len(good),recall=len(good)/len(hits),
         auc=roc_auc_score(y,s),ap=average_precision_score(y,s),
         fpr=fp/int(neg.sum()),false_windows=fp,days=days,
         false_windows_day=fp/days,false_incidents_day=incident_starts(d,fire)/days,
         median_delay_seconds=float(np.median(good)) if good else np.nan)

def test(seed,modelpath,out):
    if seed not in SEEDS:raise ValueError("unregistered seed")
    d,sim,eps=world(seed,DAYS,32,64,10)
    eligible={}
    for i,e in enumerate(eps):
        if e.kind!="compromise" or e.t0<WARMUP:continue
        r=d[(d.episode_id==i)&(d.nid==e.nid)]
        if len(r)==e.t1-e.t0+1 and r.t.nunique()==len(r):eligible[i]=e
    if len(eligible)<12:raise ValueError("too few complete episodes")
    # Exclude incomplete positives, never relabel them as negative.
    d=d[(d.label==0)|d.episode_id.isin(eligible)].reset_index(drop=True)
    cal=[world(seed+offset,days,0,normal,ext,sim)[0]
         for offset,days,normal,ext in ((10000,1.,24,8),(20000,DAYS,64,10),(30000,DAYS,96,16))]
    assert all(not x.label.any() for x in cal)
    nids=sorted(d.nid.unique())
    assert all(set(x.nid)==set(nids) for x in cal)
    alpha=BUDGET/(len(nids)*F.INTERVALS_PER_DAY)
    days=(sim.cfg.n_intervals-WARMUP)/F.INTERVALS_PER_DAY
    models=joblib.load(modelpath)
    assert set(models)==set(MODELS)
    rows=[];events=[];calrows=[];retro=[]
    for name in MODELS:
        model=models[name];s=score(model,d);cs=[score(model,x) for x in cal]
        ts=thresholds(cal,cs,nids,alpha)
        for policy,th in ts.items():
            fire=s>d.nid.map(th).to_numpy()
            rows.append(evaluate(seed,name,policy,d,s,fire,eligible,days,events))
            for phase,x,ss in zip(("sparse","matched","stress"),cal,cs):
                alarms=ss>x.nid.map(th).to_numpy()
                dur=x.t.nunique()/F.INTERVALS_PER_DAY
                calrows.append(dict(seed=seed,model=name,policy=policy,phase=phase,
                     false_windows_day=alarms.sum()/dur))
        # Retrospective performance at <= B false windows/network-day.
        # Scores sorted with threshold ties handled conservatively.
        # Never use this threshold for operational results or model selection.
        sn=s[d.label.to_numpy()==0]
        for budget in (10.,20.,40.,100.,200.):
            k=min(len(sn)-1,int(np.floor(budget*days)))
            thr=float(np.sort(sn)[::-1][k])
            rr=evaluate(seed,name,"retrospective_global",d,s,s>thr,eligible,days,[])
            rr["budget"]=budget;retro.append(rr)
        print("evaluated",seed,name,flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(events).to_csv(out/"episodes.csv",index=False)
    pd.DataFrame(calrows).to_csv(out/"calibration.csv",index=False)
    pd.DataFrame(retro).to_csv(out/"retrospective.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(seed=seed,model_run=45421952080,
         nodes=len(nids),test_generated=32,complete=len(eligible),target=BUDGET,
         calibration=[dict(days=1.,normal=24,external=8),dict(days=DAYS,normal=64,external=10),
         dict(days=DAYS,normal=96,external=16)],scope="synthetic, node-compromise only"),indent=2))

def collect(root,name):
    paths=sorted(root.rglob(name))
    if len(paths)!=len(SEEDS):raise ValueError(f"expected {len(SEEDS)} files: {name}")
    d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    if set(d.seed)!=set(SEEDS):raise ValueError("incomplete seeds")
    return d

def aggregate(root,out):
    m=collect(root,"metrics.csv");r=collect(root,"retrospective.csv")
    c=collect(root,"calibration.csv")
    keys=["seed","model","policy"]
    if m.duplicated(keys).any() or len(m)!=len(SEEDS)*len(MODELS)*len(POLICIES):
        raise ValueError("incomplete factorial matrix")
    def summarize(d,groups):
        return d.groupby(groups).agg(worlds=("seed","nunique"),
          episodes=("episodes","sum"),detected=("detected","sum"),
          recall=("recall","mean"),fpr=("fpr","mean"),
          windows_day=("false_windows_day","mean"),
          worst_windows_day=("false_windows_day","max"),
          incidents_day=("false_incidents_day","mean"),
          worlds_within_budget=("false_windows_day",lambda s:int((s<=BUDGET).sum()))).reset_index()
    s=summarize(m,["model","policy"])
    rs=summarize(r,["model","budget"])
    rng=np.random.default_rng(8600);tests=[]
    # Predetermined paired contrasts, no best-of selection.
    for model in ("own","report_inv","report_full"):
        p=m[m.model==model].pivot(index="seed",columns="policy",values="false_windows_day")
        for metric in ("false_windows_day","recall"):
            p=m[m.model==model].pivot(index="seed",columns="policy",values=metric)
            d=(p.node_robust-p.global_sparse).to_numpy()
            null=np.array([np.mean(d*np.array(x)) for x in itertools.product((-1,1),repeat=len(d))])
            boot=rng.choice(d,size=(20000,len(d)),replace=True).mean(1)
            tests.append(dict(model=model,metric=metric,contrast="node_robust-global_sparse",
               delta=d.mean(),low=np.quantile(boot,.025),high=np.quantile(boot,.975),
               p_raw=np.mean(np.abs(null)>=abs(d.mean())-1e-12)))
    running=0.
    for i,x in enumerate(sorted(tests,key=lambda x:x["p_raw"])):
        running=max(running,min(1.,(len(tests)-i)*x["p_raw"]));x["p_holm"]=running
    for name,table in (("all_metrics",m),("summary",s),("retrospective",r),
         ("retrospective_summary",rs),("calibration",c),("paired_tests",pd.DataFrame(tests))):
        table.to_csv(out/(name+".csv"),index=False)
    print(s.to_string(index=False));print(pd.DataFrame(tests).to_string(index=False))
    print("Retrospective equal-budget diagnostic (NOT deployable):")
    print(rs[rs.budget==20].to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["test","aggregate"])
    p.add_argument("--seed",type=int);p.add_argument("--models",type=Path)
    p.add_argument("--input",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="test":test(a.seed,a.models,a.output)
    else:aggregate(a.input,a.output)
