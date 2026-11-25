"""Report-only directional and neighborhood consistency, with hard-example training.

All features derive from current/past reports and known link adjacency.
The simulator counts offered MSUs at both ends; this is not physical-layer
delivery reconciliation or end-to-end route validation. No latent loss,
hidden traffic, episode metadata, integrity flags or test labels in features.
Independent train worlds and separate mixed reference simulations calibrate
the frozen decision rule. All variants are development comparisons.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from ss7 import fix as F
from ss7.report_only_invariant_study import NEW, LEGACY, both, columns
from ss7.threshold_transfer_study import score
from ss7.incident_calibration_study import elapsed,triple,metrics,RAW_BUDGET,INCIDENT_BUDGET,WARMUP
from ss7.cusum_cascade_development import joint_cut

TRAIN=(9201,9202,9203,9204)
TEST=tuple(range(9301,9309))
DIRECTION=("dir_out_res","dir_in_res","dir_out_abs","dir_in_abs",
           "dir_out_deficit","dir_in_deficit","dir_flow_gap")
PEER=("peer_other_mean","peer_other_max","peer_excess","peer_support")
BASE=("frozen_local","frozen_full")
FITTED=("refit_full","direction_only","peer_only","enhanced","enhanced_hard")
MODELS=BASE+FITTED
FRACTION=.5

def directional(reps,topo):
    """Compare each reported TX/RX counter with the opposite endpoint."""
    edge={n:[] for n in reps}
    for a,b in topo.linksets:
        if a not in reps or b not in reps:continue
        k=(a,b);A=reps[a].state;B=reps[b].state
        for source,dest,x,y in ((a,b,A.link_out.get(k,0),B.link_in.get(k,0)),
                                 (b,a,B.link_out.get(k,0),A.link_in.get(k,0))):
            if x+y<=0:continue
            edge[source].append((dest,0,float(x),float(y)))
            edge[dest].append((source,1,float(y),float(x)))
    # Summarize residual magnitude per neighbor (both directions).
    adj={}
    for n,items in edge.items():
        by={}
        for peer,_,x,y in items:
            by.setdefault(peer,[]).append(abs(2*(x-y)/max(x+y,1.)))
        adj[n]={p:float(np.mean(v)) for p,v in by.items()}
    out={}
    for n,items in edge.items():
        row={}
        for flag,prefix in ((0,"dir_out"),(1,"dir_in")):
            a=np.asarray([(x,y) for _,j,x,y in items if j==flag],float)
            if not len(a):
                row.update({prefix+"_res":np.nan,prefix+"_abs":np.nan,prefix+"_deficit":np.nan})
                continue
            x,y=a.T;scale=np.maximum(x+y,1.)
            row[prefix+"_res"]=float(2*(x-y).sum()/max((x+y).sum(),1.))
            row[prefix+"_abs"]=float(2*np.abs(x-y).sum()/max((x+y).sum(),1.))
            row[prefix+"_deficit"]=float(np.min(2*(x-y)/scale))
        # Balance inferred from neighbors versus balance in the own report.
        own=sum((1 if j==0 else -1)*x for _,j,x,y in items)
        remote=sum((1 if j==0 else -1)*y for _,j,x,y in items)
        volume=sum(x+y for _,j,x,y in items)
        row["dir_flow_gap"]=2*(own-remote)/max(volume,1.) if items else np.nan
        others=[];excess=[]
        for p,v in adj[n].items():
            values=[z for q,z in adj[p].items() if q!=n]
            if values:
                b=float(np.mean(values));others.append(b);excess.append(v-b)
        row["peer_other_mean"]=float(np.mean(others)) if others else np.nan
        row["peer_other_max"]=float(np.max(others)) if others else np.nan
        row["peer_excess"]=float(np.max(excess)) if excess else np.nan
        row["peer_support"]=len(others)/max(1,len(adj[n]))
        out[n]=row
    return out

def augmented(reps,topo,lstate):
    out=both(reps,topo,lstate)
    for n,v in directional(reps,topo).items():out[n].update(v)
    return out

def add_history(d):
    """Strictly causal changes, no interpolation across missing intervals."""
    d=d.sort_values(["nid","t"]).reset_index(drop=True)
    g=d.groupby("nid",sort=False)
    out={}
    for lag in (1,5):
        valid=(d.t-g.t.shift(lag))==lag
        # Ensure all intermediate intervals exist, not merely endpoints.
        for c in DIRECTION+PEER:
            out[f"{c}__delta{lag}"]=(d[c]-g[c].shift(lag)).where(valid)
    return pd.concat([d,pd.DataFrame(out,index=d.index)],axis=1)

def generate(seed,days,comp,normal,external,template=None):
    cfg=F.SimConfig(seed=seed,days=days)
    F.all_invariants=augmented
    try:
        d,sim,eps=F.build_dataset(cfg,episodes_kw=dict(n_compromise=comp,n_normal=normal,n_ext=external),
            topo=template.topo if template is not None else None,sim_template=template)
    finally:
        F.all_invariants=LEGACY
    d=F.temporal_block(d,NEW,cfg.feat_window)
    d=F.zone_deviation(d,NEW,{n:v.zone for n,v in sim.topo.nodes.items()})
    d=add_history(d)
    d=d[d.t>=WARMUP].reset_index(drop=True)
    if d.duplicated(["nid","t"]).any():raise ValueError("duplicate observations")
    return d,sim,eps

def specs(d):
    base=columns(d,F.OWN+NEW)
    def group(bases):
        return [b+s for b in bases for s in ("","__delta1","__delta5")]
    a,b=group(DIRECTION),group(PEER)
    return {"refit_full":base,"direction_only":base+a,"peer_only":base+b,
            "enhanced":base+a+b,"enhanced_hard":base+a+b}

def train(frozenpath,out):
    frozen=joblib.load(frozenpath)["report_full"]
    parts=[];audits=[]
    for seed in TRAIN:
        d,_,_=generate(seed,1.5,32,64,10)
        y=d.label.to_numpy();s=score(frozen,d)
        pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
        rng=np.random.default_rng(seed)
        # Common random support for every refit; all positive windows retained.
        ix=np.sort(np.r_[pos,rng.choice(neg,min(40000,len(neg)),replace=False)])
        q=float(np.quantile(s[neg],.98))
        hard=np.where(y==1,s<.5,s>=q)
        z=d.iloc[ix].copy()
        z["hard_weight_flag"]=hard[ix]
        parts.append(z)
        audits.append(dict(seed=seed,rows=len(ix),positives=len(pos),hard_positive=int(hard[pos].sum()),
            hard_negative_sample=int(hard[ix][y[ix]==0].sum()),negative_cutoff=q))
        print("train world",seed,len(ix),flush=True)
    d=pd.concat(parts,ignore_index=True);y=d.label.to_numpy()
    models={}
    for name,cols in specs(d).items():
        X=d[cols].replace([np.inf,-np.inf],np.nan)
        med=X.median().fillna(0.)
        X=X.fillna(med);cols=[c for c in cols if X[c].nunique()>1]
        if not cols:raise ValueError("no varying features")
        w=np.ones(len(y))
        if name=="enhanced_hard":w*=np.where(d.hard_weight_flag,4.,1.)
        # Keep total positive and negative training weights identical.
        for k in (0,1):w[y==k]*=len(y)/(2*w[y==k].sum())
        model=HistGradientBoostingClassifier(max_iter=300,learning_rate=.06,
            min_samples_leaf=30,l2_regularization=2.,early_stopping=False,
            random_state=9200).fit(X[cols],y,sample_weight=w)
        models[name]=(cols,med[cols],model)
        print("trained",name,len(cols),flush=True)
    joblib.dump(models,out/"specialist.joblib",compress=3)
    (out/"train_manifest.json").write_text(json.dumps(dict(train=TRAIN,test=TEST,
        features={n:v[0] for n,v in models.items()},audit=audits),indent=2))

def test(seed,localpath,frozenpath,modelpath,out):
    if seed not in TEST:raise ValueError("unregistered test world")
    d,sim,eps=generate(seed,1.5,32,64,10)
    eligible={}
    for i,e in enumerate(eps):
        if e.kind!="compromise" or e.t0<WARMUP:continue
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        if len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q):eligible[i]=e
    if len(eligible)<12:raise ValueError("insufficient complete episodes; no replacement")
    refs=(generate(seed+10000,3.,0,128,20,sim)[0],
          generate(seed+20000,3.,0,192,32,sim)[0],
          generate(seed+40000,3.,64,128,20,sim)[0])
    assert not refs[0].label.any() and not refs[1].label.any() and refs[2].label.any()
    assert all(set(x.nid)==set(d.nid) for x in refs)
    models={"frozen_local":joblib.load(localpath)["strict_local"],
            "frozen_full":joblib.load(frozenpath)["report_full"]}
    models.update(joblib.load(modelpath));assert set(models)==set(MODELS)
    rows=[];events=[]
    for name,model in models.items():
        ss=[score(model,x) for x in (*refs,d)]
        tau=joint_cut(refs,ss[:3],RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION)
        u=ss[3];m,det=metrics(d,u,triple(d,u),tau,eligible)
        rows.append(dict(seed=seed,model=name,**m))
        for i,e in eligible.items():
            ix=(d.episode_id.to_numpy()==i)&(d.nid.to_numpy()==e.nid)
            events.append(dict(seed=seed,model=name,episode=i,detected=int(i in det),
                delay_seconds=det.get(i),hidden_rate=e.hidden_rate,mask_family=e.family,
                ntype=sim.topo.nodes[e.nid].ntype.value,duration_windows=int(ix.sum()),
                max_score=float(u[ix].max()),threshold=tau))
        print(seed,name,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(events).to_csv(out/"episodes.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(seed=seed,train=TRAIN,complete=len(eligible),
        reference_seeds=[seed+10000,seed+20000,seed+40000],scope="development; simulated report-only"),indent=2))

def aggregate(root,out):
    def collect(k):
        p=sorted(root.rglob(k+".csv"))
        if len(p)!=len(TEST):raise ValueError("incomplete worlds")
        return pd.concat([pd.read_csv(x) for x in p],ignore_index=True)
    m,e=collect("metrics"),collect("episodes")
    keys=["seed","model"]
    if m.duplicated(keys).any() or set(m[keys].itertuples(index=False,name=None))!={(s,n) for s in TEST for n in MODELS}:
        raise ValueError("incomplete factorial")
    if e.duplicated(keys+["episode"]).any():raise ValueError("duplicate episodes")
    for s in TEST:
        support=[set(x.episode) for _,x in e[e.seed==s].groupby("model")]
        if len(support)!=len(MODELS) or not all(x==support[0] for x in support):
            raise ValueError("unpaired episodes")
    m["both_budgets"]=(m.false_windows_day<=RAW_BUDGET)&(m.false_notifications_day<=INCIDENT_BUDGET)
    summary=m.groupby("model").agg(worlds=("seed","nunique"),episodes=("episodes","sum"),
        detected=("detected","sum"),recall=("recall","mean"),recall_sd=("recall","std"),
        auc=("auc","mean"),ap=("ap","mean"),fpr=("fpr","mean"),
        raw_day=("false_windows_day","mean"),worst_raw=("false_windows_day","max"),
        notice_day=("false_notifications_day","mean"),both_budgets=("both_budgets","sum"),
        median_world_delay=("median_delay_seconds","median")).reset_index()
    e["rate_band"]=pd.cut(e.hidden_rate,[-np.inf,.01,.05,.1,np.inf])
    breakdown=e.groupby(["model","mask_family","ntype","rate_band"],observed=True).agg(
        episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean")).reset_index()
    b=e[e.model=="frozen_full"][["seed","episode","detected"]].rename(columns={"detected":"baseline"})
    t=e.merge(b,on=["seed","episode"],validate="many_to_one")
    t["rescued"]=(t.detected==1)&(t.baseline==0);t["lost"]=(t.detected==0)&(t.baseline==1)
    transitions=t.groupby("model").agg(rescued=("rescued","sum"),lost=("lost","sum")).reset_index()
    for k,table in (("all_metrics",m),("episodes",e),("summary",summary),("breakdown",breakdown),("transitions",transitions)):
        table.to_csv(out/(k+".csv"),index=False)
    print(summary.to_string(index=False));print(transitions.to_string(index=False))
    print("Development only; no final model selected or operational claim.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["train","test","aggregate"])
    p.add_argument("--seed",type=int);p.add_argument("--local",type=Path)
    p.add_argument("--frozen",type=Path);p.add_argument("--models",type=Path)
    p.add_argument("--input",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="train":train(a.frozen,a.output)
    elif a.mode=="test":test(a.seed,a.local,a.frozen,a.models,a.output)
    else:aggregate(a.input,a.output)
