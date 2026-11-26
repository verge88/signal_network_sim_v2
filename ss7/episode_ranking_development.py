"""Episode-ranking development: smooth best consecutive triple versus hard negatives.

Frozen base and weak classifiers. Disjoint rank training, validation selection,
test worlds; reference labels are simulated commissioning only. No test labels
select models, shares or calibration margins. Ranking objective is a surrogate
for episode recall at small FPR; exact metrics use inherited bucket endpoint.
"""
import argparse
import json
import itertools
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from ss7.report_feature_specialist import generate
from ss7.meta_episode_development import base_models
from ss7.weak_signal_confirmation import scores,matrix,weak_predict
from ss7.threshold_transfer_study import score
from ss7.cusum_cascade_development import logit,joint_cut,raw_cut
from ss7.incident_calibration_study import elapsed,triple,metrics,RAW_BUDGET,INCIDENT_BUDGET,WARMUP

TRAIN=tuple(range(11001,11009))
VAL=tuple(range(11101,11107))
TEST=tuple(range(11201,11209))
SIZES=(2,4,8)
BASE=("local","frozen_hgb","weak")
SHARES=(.25,.5,.75)
CANDIDATES=("frozen_hgb","weak")+tuple(f"rank{k}" for k in SIZES)+tuple(
    f"union_{m}_{s}" for m in ("weak","rank8") for s in SHARES)
FRACTION=.5
TEMP=.5
TOP=32

def load(local,frozen,special,meta,weak):
    return (base_models(frozen,special,local),joblib.load(meta)["meta_hgb"],
        joblib.load(weak)["weak"])

def ranking_features(a,b,h,w):
    x,y,z,v=map(logit,(a,b,h,w))
    return np.column_stack([x,y,z,v,x-y,z-v,np.abs(x-y),np.abs(z-v),
                            np.minimum(z,v),np.maximum(z,v)])

def infer(d,models):
    base,meta,weak=models
    a,b,h=scores(d,base,meta)
    w=weak_predict(weak,matrix(d,a,b,h,False)[0])
    return h,w,ranking_features(a,b,h,w)

def episode_indices(d):
    n=d.nid.to_numpy();t=d.t.to_numpy();y=d.label.to_numpy();e=d.episode_id.to_numpy()
    end=np.flatnonzero((n[2:]==n[1:-1])&(n[1:-1]==n[:-2])&
        (t[2:]==t[1:-1]+1)&(t[1:-1]==t[:-2]+1)&
        (y[2:]==1)&(y[1:-1]==1)&(y[:-2]==1)&
        (e[2:]==e[1:-1])&(e[1:-1]==e[:-2]))+2
    return [np.column_stack([j-2,j-1,j]) for k in np.unique(e[end]) for j in [end[e[end]==k]]]

def world_samples(seed,models):
    d,_,_=generate(seed,1.5,48,96,16)
    h,w,x=infer(d,models)
    neg=np.flatnonzero(d.label.to_numpy()==0)
    hard=neg[np.argsort(np.maximum(h,w)[neg])[-min(2500,len(neg)):]]
    rest=np.setdiff1d(neg,hard)
    rng=np.random.default_rng(seed)
    selected=np.r_[hard,rng.choice(rest,min(1500,len(rest)),replace=False)]
    ep=[x[i] for i in episode_indices(d)]
    if len(ep)<12 or len(selected)<TOP:raise ValueError("insufficient training support")
    print("rank train",seed,len(ep),len(selected),flush=True)
    return ep,x[selected]

def smooth_episode(x,b):
    # Normalized soft min within a triple and soft max across triples.
    s=x@b
    q=-s/TEMP
    softmin=-TEMP*(logsumexp(q,axis=1)-np.log(3.))
    gmin=np.einsum("ij,ijk->ik",np.exp(q-logsumexp(q,axis=1,keepdims=True)),x)
    a=softmin/TEMP
    z=TEMP*(logsumexp(a)-np.log(len(a)))
    grad=np.exp(a-logsumexp(a))@gmin
    return z,grad

def objective(b,worlds):
    loss=0.;grad=np.zeros_like(b)
    for eps,neg in worlds:
        s=neg@b
        idx=np.argsort(s,kind="stable")[-TOP:]
        n=s[idx];nx=neg[idx]
        v=0.;g=np.zeros_like(b)
        for x in eps:
            z,dz=smooth_episode(x,b)
            diff=n-z+1.
            v+=np.logaddexp(0.,diff).mean()
            g+=(expit(diff)[:,None]*(nx-dz)).mean(axis=0)
        loss+=v/len(eps);grad+=g/len(eps)
    return loss/len(worlds)+.02*(b@b),grad/len(worlds)+.04*b

def fit(samples):
    x=np.concatenate([neg for _,neg in samples])
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-6)
    worlds=[([(a-mean)/scale for a in ep],(n-mean)/scale) for ep,n in samples]
    init=np.zeros(len(mean));init[2:4]=.5
    res=minimize(objective,init,args=(worlds,),jac=True,method="L-BFGS-B",
                 bounds=[(-4.,4.)]*len(mean),options={"maxiter":100,"ftol":1e-9})
    if not np.isfinite(res.fun) or not np.isfinite(res.x).all():raise ValueError("rank fit failed")
    print("rank optimization",len(samples),res.success,res.message,res.fun,flush=True)
    return dict(mean=mean,scale=scale,coef=res.x,loss=float(res.fun),converged=bool(res.success),
                message=str(res.message))

def rank_predict(m,x):
    return expit(((x-m["mean"])/m["scale"])@m["coef"])

def train(models,out):
    worlds=[world_samples(seed,models) for seed in TRAIN]
    ranks={f"rank{k}":fit(worlds[:k]) for k in SIZES}
    joblib.dump(ranks,out/"ranks.joblib",compress=3)
    (out/"train_manifest.json").write_text(json.dumps(dict(train=TRAIN,val=VAL,test=TEST,
        objective="smooth episode max-min versus top-32 sampled negative scores, pairwise logistic loss",
        fits={n:dict(loss=m["loss"],converged=m["converged"],message=m["message"],
                     coef=m["coef"].tolist()) for n,m in ranks.items()}),indent=2))

def union(refs,a,b,share):
    x=raw_cut(refs,a,RAW_BUDGET*FRACTION*share)
    y=raw_cut(refs,b,RAW_BUDGET*FRACTION*(1-share))
    u=[np.maximum(p-x,q-y) for p,q in zip(a,b)]
    extra=max(0.,joint_cut(refs,u,RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION))
    return x,y,extra

def evaluate(seed,models,ranks,out,selection=None):
    if seed not in VAL+TEST:raise ValueError("unregistered seed")
    val=seed in VAL
    if val!=(selection is None):raise ValueError("test requires frozen validation selection")
    d,sim,eps=generate(seed,1.5,32,64,10)
    eligible={}
    for i,e in enumerate(eps):
        if e.kind!="compromise" or e.t0<WARMUP:continue
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        if len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q):eligible[i]=e
    if len(eligible)<12:raise ValueError("too few complete episodes")
    refs=(generate(seed+10000,3.,0,128,20,sim)[0],
          generate(seed+20000,3.,0,192,32,sim)[0],
          generate(seed+40000,3.,64,128,20,sim)[0])
    assert not refs[0].label.any() and not refs[1].label.any() and refs[2].label.any()
    assert all(set(x.nid)==set(d.nid) for x in refs)
    frames=(*refs,d);z=[infer(x,models) for x in frames]
    ss={"frozen_hgb":[x[0] for x in z],"weak":[x[1] for x in z],
        "local":[score(models[0]["local"],x) for x in frames]}
    requested=("local",)+CANDIDATES if val else tuple(dict.fromkeys(BASE+(selection["selected"],)))
    for n,m in ranks.items():
        if n in requested or any(x.startswith("union_"+n+"_") for x in requested):
            ss[n]=[rank_predict(m,x[2]) for x in z]
    for n in requested:
        if n.startswith("union_"):
            _,b,share=n.split("_")
            a=ss["frozen_hgb"];q=ss[b]
            x,y,extra=union(refs,a[:3],q[:3],float(share))
            ss[n]=[np.maximum(u-x,v-y)-extra for u,v in zip(a,q)]
    rows=[];events=[]
    for n in requested:
        s=ss[n];u=s[3]
        tau=0. if n.startswith("union_") else joint_cut(refs,s[:3],RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION)
        m,det=metrics(d,u,triple(d,u),tau,eligible)
        rows.append(dict(seed=seed,model=n,**m))
        for i,e in eligible.items():
            ix=(d.episode_id.to_numpy()==i)&(d.nid.to_numpy()==e.nid)
            events.append(dict(seed=seed,model=n,episode=i,detected=int(i in det),
                hidden_rate=e.hidden_rate,mask_family=e.family,ntype=sim.topo.nodes[e.nid].ntype.value,
                delay_seconds=det.get(i),max_margin=float(u[ix].max()-tau)))
        print(seed,n,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(events).to_csv(out/"episodes.csv",index=False)

def collect(root,seeds,names):
    def read(k):
        p=sorted(root.rglob(k+".csv"))
        if len(p)!=len(seeds):raise ValueError("incomplete registered worlds")
        return pd.concat([pd.read_csv(x) for x in p],ignore_index=True)
    m,e=read("metrics"),read("episodes")
    keys=["seed","model"]
    if m.duplicated(keys).any() or set(m[keys].itertuples(index=False,name=None))!={(s,n) for s in seeds for n in names}:
        raise ValueError("invalid factorial")
    if e.duplicated(keys+["episode"]).any():raise ValueError("duplicate episodes")
    for s in seeds:
        groups=[set(x.episode) for _,x in e[e.seed==s].groupby("model")]
        if len(groups)!=len(names) or not all(x==groups[0] for x in groups):raise ValueError("unpaired episode support")
    m["both_budgets"]=(m.false_windows_day<=RAW_BUDGET)&(m.false_notifications_day<=INCIDENT_BUDGET)
    summary=m.groupby("model").agg(worlds=("seed","nunique"),episodes=("episodes","sum"),detected=("detected","sum"),
        recall=("recall","mean"),recall_sd=("recall","std"),auc=("auc","mean"),ap=("ap","mean"),
        fpr=("fpr","mean"),raw_day=("false_windows_day","mean"),worst_raw=("false_windows_day","max"),
        notice_day=("false_notifications_day","mean"),worst_notice=("false_notifications_day","max"),
        both_budgets=("both_budgets","sum"),median_world_delay=("median_delay_seconds","median")).reset_index()
    return m,e,summary

def select(root,out):
    m,e,s=collect(root,VAL,("local",)+CANDIDATES)
    # Fixed lexicographic criterion: compliance first, then episode recall,
    # lower worst false windows, finally stable name tie break.
    table=s[s.model.isin(CANDIDATES)].sort_values(
        ["both_budgets","recall","worst_raw","model"],ascending=[False,False,True,True])
    chosen=str(table.iloc[0].model)
    (out/"selection.json").write_text(json.dumps(dict(selected=chosen,
        validation_seeds=VAL,candidates=CANDIDATES,rule="maximize budget-compliant worlds, then recall, then minimize worst raw, then name"),indent=2))
    s.to_csv(out/"validation_summary.csv",index=False)
    m.to_csv(out/"validation_metrics.csv",index=False)
    e.to_csv(out/"validation_episodes.csv",index=False)
    s[s.model.isin([f"rank{k}" for k in SIZES])].assign(
        training_worlds=lambda d:d.model.str.replace("rank","").astype(int)).to_csv(out/"learning_curve.csv",index=False)
    print(s.to_string(index=False));print("SELECTED",chosen,flush=True)

def aggregate(root,out,selection):
    chosen=selection["selected"]
    m,e,s=collect(root,TEST,tuple(dict.fromkeys(BASE+(chosen,))))
    base=e[e.model=="frozen_hgb"][["seed","episode","detected"]].rename(columns={"detected":"base"})
    t=e.merge(base,on=["seed","episode"],validate="many_to_one")
    t["rescued"]=(t.detected==1)&(t.base==0);t["lost"]=(t.detected==0)&(t.base==1)
    transitions=t.groupby("model").agg(rescued=("rescued","sum"),lost=("lost","sum")).reset_index()
    e["rate_band"]=pd.cut(e.hidden_rate,[-np.inf,.01,.05,.1,np.inf])
    strata=e.groupby(["model","mask_family","ntype","rate_band"],observed=True).agg(
        episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean")).reset_index()
    p=m.pivot(index="seed",columns="model",values="recall")
    rng=np.random.default_rng(11200);comparisons=[]
    for b in ("local","frozen_hgb"):
        if chosen==b:continue
        v=(p[chosen]-p[b]).to_numpy()
        null=np.array([np.mean(v*np.array(a)) for a in itertools.product((-1,1),repeat=len(v))])
        boot=rng.choice(v,size=(20000,len(v)),replace=True).mean(axis=1)
        comparisons.append(dict(a=chosen,b=b,delta=v.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),
            p_raw=np.mean(np.abs(null)>=abs(v.mean())-1e-12)))
    prev=0.
    for i,x in enumerate(sorted(comparisons,key=lambda z:z["p_raw"])):
        prev=max(prev,min(1.,(len(comparisons)-i)*x["p_raw"]));x["p_holm"]=prev
    for k,table in (("all_metrics",m),("episodes",e),("summary",s),("transitions",transitions),
        ("strata",strata),("paired",pd.DataFrame(comparisons))):
        table.to_csv(out/(k+".csv"),index=False)
    print("VALIDATION-SELECTED",chosen);print(s.to_string(index=False));print(transitions.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print("Development evaluation; independent larger confirmation required.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["train","validate","select","test","aggregate"])
    p.add_argument("--seed",type=int);p.add_argument("--local",type=Path);p.add_argument("--frozen",type=Path)
    p.add_argument("--special",type=Path);p.add_argument("--meta",type=Path);p.add_argument("--weak",type=Path)
    p.add_argument("--ranks",type=Path);p.add_argument("--selection",type=Path);p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="select":select(a.input,a.output)
    elif a.mode=="aggregate":aggregate(a.input,a.output,json.loads(a.selection.read_text()))
    else:
        models=load(a.local,a.frozen,a.special,a.meta,a.weak)
        if a.mode=="train":train(models,a.output)
        else:evaluate(a.seed,models,joblib.load(a.ranks),a.output,
            None if a.mode=="validate" else json.loads(a.selection.read_text()))
