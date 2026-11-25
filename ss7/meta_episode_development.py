"""Independent stacking and route-proxy development, not confirmation.

Frozen base models from 45421952080 and 45640796971. Train meta-models
on new worlds, calibrate on separate synthetic references, test on new
worlds. No labels, latent traffic, actual failed-link state, or future
observations in inference. Route proxy uses static advertised shortest
paths and aggregate OD reports, not measured per-route delivered MSUs.
"""
import argparse
import itertools
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ss7 import fix as F
from ss7.report_only_invariant_study import NEW, LEGACY
from ss7.report_feature_specialist import augmented,add_history
from ss7.threshold_transfer_study import score
from ss7.incident_calibration_study import elapsed,triple,metrics,RAW_BUDGET,INCIDENT_BUDGET,WARMUP
from ss7.cusum_cascade_development import joint_cut,raw_cut,logit

TRAIN=(9401,9402,9403,9404)
TEST=tuple(range(9501,9509))
ROUTE=("route_transit_res","route_transit_abs","route_expected_log","route_observed_log")
MODELS=("local","primary","peer","max_pool","budget_union","meta_logistic","meta_hgb","meta_route")
FRACTION=.5

class RouteProxy:
    """Fixed-topology expected transit transactions from paired endpoint reports.
    One OD total per unordered pair; shortest-path shares are a proxy only.
    Neither actual path nor hidden events are consulted.
    """
    def __init__(self,topo):
        self.nodes=tuple(sorted(topo.nodes))
        idx={n:i for i,n in enumerate(self.nodes)}
        self.pairs=tuple(itertools.combinations(self.nodes,2))
        row=[];col=[];val=[]
        for j,(a,b) in enumerate(self.pairs):
            route=topo.route(a,b)
            for path,weight in zip(route.paths,route.shares):
                for n in path[1:-1]:
                    row.append(idx[n]);col.append(j);val.append(float(weight))
        self.matrix=csr_matrix((val,(row,col)),shape=(len(self.nodes),len(self.pairs)))
    def features(self,reps):
        # Missing endpoint report means the full OD estimate is unavailable,
        # not a zero transaction count. Fail closed for this optional feature.
        if any(n not in reps for n in self.nodes):
            return {n:{k:np.nan for k in ROUTE} for n in reps}
        st={n:r.state for n,r in reps.items()}
        x=np.fromiter((.5*(st[a].dest.get(b,0)+st[b].dest.get(a,0))
                       for a,b in self.pairs),float,count=len(self.pairs))
        expected=np.asarray(self.matrix@x).ravel()
        out={}
        for n,e in zip(self.nodes,expected):
            o=float(st[n].transited)
            r=2*(o-e)/max(o+e,1.)
            out[n]=dict(route_transit_res=r,route_transit_abs=abs(r),
                        route_expected_log=np.log1p(e),route_observed_log=np.log1p(o))
        return out

def generate(seed,days,comp,normal,external,template=None):
    cfg=F.SimConfig(seed=seed,days=days)
    cache={}
    def inv(reps,topo,lstate):
        out=augmented(reps,topo,lstate)
        if "proxy" not in cache:cache["proxy"]=RouteProxy(topo)
        for n,row in cache["proxy"].features(reps).items():out[n].update(row)
        return out
    F.all_invariants=inv
    try:
        d,sim,eps=F.build_dataset(cfg,episodes_kw=dict(n_compromise=comp,n_normal=normal,n_ext=external),
            topo=template.topo if template is not None else None,sim_template=template)
    finally:
        F.all_invariants=LEGACY
    d=F.temporal_block(d,NEW,cfg.feat_window)
    d=F.zone_deviation(d,NEW,{n:v.zone for n,v in sim.topo.nodes.items()})
    d=add_history(d)
    d=d[d.t>=WARMUP].reset_index(drop=True)
    if d.duplicated(["nid","t"]).any():raise ValueError("duplicate node windows")
    return d,sim,eps

def features(d,a,b,route=False):
    """Current scores and causal lagged minimum; no metadata or labels."""
    x,y=logit(a),logit(b)
    ta,tb=triple(d,x),triple(d,y)
    ta=np.where(np.isfinite(ta),ta,x);tb=np.where(np.isfinite(tb),tb,y)
    arr=np.column_stack([x,y,x-y,np.abs(x-y),np.minimum(x,y),np.maximum(x,y),ta,tb])
    if route:arr=np.column_stack([arr,d[list(ROUTE)].to_numpy(float)])
    return arr

def base_models(frozen,special,local=None):
    models={"primary":joblib.load(frozen)["report_full"],"peer":joblib.load(special)["peer_only"]}
    if local is not None:models["local"]=joblib.load(local)["strict_local"]
    return models

def fit_meta(X,y,w,kind):
    med=pd.DataFrame(X).median().fillna(0.).to_numpy()
    X=np.where(np.isfinite(X),X,med)
    if kind=="logistic":
        est=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,random_state=9400))
        est.fit(X,y,logisticregression__sample_weight=w)
    else:
        est=HistGradientBoostingClassifier(max_iter=150,learning_rate=.05,max_leaf_nodes=15,
            min_samples_leaf=50,l2_regularization=5.,early_stopping=False,random_state=9400)
        est.fit(X,y,sample_weight=w)
    return med,est

def predict(model,X):
    med,est=model
    s=est.predict_proba(np.where(np.isfinite(X),X,med))[:,1]
    if not np.isfinite(s).all():raise ValueError("nonfinite meta prediction")
    return s

def train(frozen,special,out):
    base=base_models(frozen,special)
    parts=[];audit=[]
    for seed in TRAIN:
        d,_,_=generate(seed,1.5,40,80,13)
        a,b=[score(base[n],d) for n in ("primary","peer")]
        y=d.label.to_numpy();pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
        u=np.maximum(a,b)
        q=float(np.quantile(u[neg],.98))
        tail=neg[u[neg]>=q];rest=neg[u[neg]<q]
        rng=np.random.default_rng(seed)
        ix=np.sort(np.r_[pos,tail,rng.choice(rest,min(30000,len(rest)),replace=False)])
        # Uniform total positive weight per episode (rather than per window).
        ep=d.episode_id.to_numpy()
        counts=pd.Series(ep[pos]).value_counts()
        w=np.ones(len(ix))
        p=y[ix]==1
        w[p]=1./pd.Series(ep[ix][p]).map(counts).to_numpy(float)
        parts.append((features(d,a,b,True)[ix],y[ix],w))
        audit.append(dict(seed=seed,positive=len(pos),episodes=len(counts),hard_negative=len(tail),
                          random_negative=len(ix)-len(pos)-len(tail),tail_cutoff=q))
        print("meta train",seed,audit[-1],flush=True)
    X=np.concatenate([p[0] for p in parts]);y=np.concatenate([p[1] for p in parts]);w=np.concatenate([p[2] for p in parts])
    for k in (0,1):w[y==k]*=len(y)/(2*w[y==k].sum())
    models={"meta_logistic":fit_meta(X[:,:8],y,w,"logistic"),
            "meta_hgb":fit_meta(X[:,:8],y,w,"hgb"),
            "meta_route":fit_meta(X,y,w,"hgb")}
    joblib.dump(models,out/"meta.joblib",compress=3)
    (out/"train_manifest.json").write_text(json.dumps(dict(train=TRAIN,test=TEST,
        base_runs=[45421952080,45640796971],audit=audit,
        objective="episode-balanced positives, oversampled high-score negatives, weighted classification surrogate"),indent=2))

def union(refs,aa,bb):
    # 75/25 split of raw budget; calibrate union jointly without independence.
    x=raw_cut(refs,aa,RAW_BUDGET*FRACTION*.75)
    y=raw_cut(refs,bb,RAW_BUDGET*FRACTION*.25)
    u=[np.maximum(a-x,b-y) for a,b in zip(aa,bb)]
    extra=max(0.,joint_cut(refs,u,RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION))
    return x,y,extra

def test(seed,local,frozen,special,metapath,out):
    if seed not in TEST:raise ValueError("unregistered test world")
    d,sim,eps=generate(seed,1.5,32,64,10)
    eligible={}
    for i,e in enumerate(eps):
        if e.kind!="compromise" or e.t0<WARMUP:continue
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        if len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q):eligible[i]=e
    if len(eligible)<12:raise ValueError("too few complete episodes; no replacement")
    refs=(generate(seed+10000,3.,0,128,20,sim)[0],
          generate(seed+20000,3.,0,192,32,sim)[0],
          generate(seed+40000,3.,64,128,20,sim)[0])
    assert not refs[0].label.any() and not refs[1].label.any() and refs[2].label.any()
    assert all(set(x.nid)==set(d.nid) for x in refs)
    frames=(*refs,d)
    base=base_models(frozen,special,local);meta=joblib.load(metapath)
    ss={n:[score(m,x) for x in frames] for n,m in base.items()}
    aa,bb=ss["primary"],ss["peer"]
    ss["max_pool"]=[np.maximum(a,b) for a,b in zip(aa,bb)]
    x,y,extra=union(refs,aa[:3],bb[:3])
    ss["budget_union"]=[np.maximum(a-x,b-y)-extra for a,b in zip(aa,bb)]
    for n,m in meta.items():
        ss[n]=[predict(m,features(x,a,b,n=="meta_route")) for x,a,b in zip(frames,aa,bb)]
    assert set(ss)==set(MODELS)
    rows=[];events=[]
    for n in MODELS:
        s=ss[n]
        tau=0. if n=="budget_union" else joint_cut(refs,s[:3],RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION)
        u=s[3];z=triple(d,u);m,det=metrics(d,u,z,tau,eligible)
        rows.append(dict(seed=seed,model=n,**m))
        for i,e in eligible.items():
            ix=(d.episode_id.to_numpy()==i)&(d.nid.to_numpy()==e.nid)
            events.append(dict(seed=seed,model=n,episode=i,detected=int(i in det),delay_seconds=det.get(i),
                hidden_rate=e.hidden_rate,mask_family=e.family,ntype=sim.topo.nodes[e.nid].ntype.value,
                duration_windows=int(ix.sum()),max_score=float(u[ix].max()),threshold=tau,
                raw_windows=int((u[ix]>tau).sum()),confirmed_candidate=int((z[ix]>tau).any())))
        print(seed,n,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(events).to_csv(out/"episodes.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(seed=seed,complete=len(eligible),train=TRAIN,
        reference_seeds=[seed+10000,seed+20000,seed+40000],union=[x,y,extra],
        scope="development, simulated labelled reference, no exact path data"),indent=2))

def aggregate(root,out):
    def collect(k):
        p=sorted(root.rglob(k+".csv"))
        if len(p)!=len(TEST):raise ValueError("incomplete worlds")
        return pd.concat([pd.read_csv(x) for x in p],ignore_index=True)
    m,e=collect("metrics"),collect("episodes");keys=["seed","model"]
    if m.duplicated(keys).any() or set(m[keys].itertuples(index=False,name=None))!={(s,n) for s in TEST for n in MODELS}:
        raise ValueError("invalid factorial")
    if e.duplicated(keys+["episode"]).any():raise ValueError("duplicate episodes")
    for s in TEST:
        support=[set(x.episode) for _,x in e[e.seed==s].groupby("model")]
        if len(support)!=len(MODELS) or not all(x==support[0] for x in support):
            raise ValueError("unpaired episode support")
    m["both_budgets"]=(m.false_windows_day<=RAW_BUDGET)&(m.false_notifications_day<=INCIDENT_BUDGET)
    summary=m.groupby("model").agg(worlds=("seed","nunique"),episodes=("episodes","sum"),
        detected=("detected","sum"),recall=("recall","mean"),recall_sd=("recall","std"),
        auc=("auc","mean"),ap=("ap","mean"),fpr=("fpr","mean"),raw_day=("false_windows_day","mean"),
        worst_raw=("false_windows_day","max"),notice_day=("false_notifications_day","mean"),
        worst_notice=("false_notifications_day","max"),both_budgets=("both_budgets","sum"),
        mixed_notifications=("mixed_notifications","sum"),median_world_delay=("median_delay_seconds","median")).reset_index()
    e["rate_band"]=pd.cut(e.hidden_rate,[-np.inf,.01,.05,.1,np.inf])
    breakdown=e.groupby(["model","mask_family","ntype","rate_band"],observed=True).agg(
        episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean")).reset_index()
    p=e[e.model=="primary"][["seed","episode","detected"]].rename(columns={"detected":"base"})
    t=e.merge(p,on=["seed","episode"],validate="many_to_one")
    t["rescued"]=(t.detected==1)&(t.base==0);t["lost"]=(t.detected==0)&(t.base==1)
    transitions=t.groupby("model").agg(rescued=("rescued","sum"),lost=("lost","sum")).reset_index()
    missed=e[e.detected==0].copy()
    missed["reason"] = np.select([missed.raw_windows==0,missed.confirmed_candidate==0],
          ["no_threshold_crossing","no_consecutive_confirmation"],default="candidate_not_credited")
    reasons=missed.groupby(["model","reason"]).size().rename("episodes").reset_index()
    # Paired world-level exploratory comparisons, not confirmatory inference.
    pivot=m.pivot(index="seed",columns="model",values="recall")
    rng=np.random.default_rng(9500);comparisons=[]
    for n in MODELS:
        if n=="primary":continue
        v=(pivot[n]-pivot["primary"]).to_numpy()
        null=np.array([np.mean(v*np.array(sign)) for sign in itertools.product((-1,1),repeat=len(v))])
        boot=rng.choice(v,size=(20000,len(v)),replace=True).mean(axis=1)
        comparisons.append(dict(model=n,delta=v.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),
                                p_raw=np.mean(np.abs(null)>=abs(v.mean())-1e-12)))
    prev=0.
    for i,x in enumerate(sorted(comparisons,key=lambda x:x["p_raw"])):
        prev=max(prev,min(1.,(len(comparisons)-i)*x["p_raw"]));x["p_holm"]=prev
    for k,table in (("all_metrics",m),("episodes",e),("summary",summary),("breakdown",breakdown),
        ("transitions",transitions),("missed_reasons",reasons),("paired_exploratory",pd.DataFrame(comparisons))):
        table.to_csv(out/(k+".csv"),index=False)
    print(summary.to_string(index=False));print(transitions.to_string(index=False))
    print(reasons.to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False))
    print("Development only; separate confirmation required.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["train","test","aggregate"])
    p.add_argument("--seed",type=int);p.add_argument("--local",type=Path)
    p.add_argument("--frozen",type=Path);p.add_argument("--special",type=Path)
    p.add_argument("--meta",type=Path);p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="train":train(a.frozen,a.special,a.output)
    elif a.mode=="test":test(a.seed,a.local,a.frozen,a.special,a.meta,a.output)
    else:aggregate(a.input,a.output)
