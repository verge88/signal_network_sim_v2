"""Weak-signal development and separate frozen-HGB confirmation.

External nodes have no reports in Ss7Sim.step_reports. The old RouteProxy
required all nodes and consequently returned NaNs. PartialRoute uses only
observed internal endpoints, with explicit coverage; it estimates a
component of total transit, not a measured path or full flow balance.
No hidden states, labels or episode attributes are used during inference.
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
from sklearn.metrics import roc_auc_score
from ss7 import fix as F
from ss7.report_only_invariant_study import NEW,LEGACY,columns
from ss7.report_feature_specialist import augmented,add_history,generate as plain_generate,DIRECTION,PEER
from ss7.meta_episode_development import RouteProxy,ROUTE,features,base_models,predict
from ss7.threshold_transfer_study import score
from ss7.incident_calibration_study import elapsed,triple,metrics,RAW_BUDGET,INCIDENT_BUDGET,WARMUP
from ss7.cusum_cascade_development import raw_cut,joint_cut

TRAIN=tuple(range(9601,9605))
DEV=tuple(range(9701,9709))
CONF=tuple(range(9801,9833))
DEVMODELS=("local","primary","frozen_hgb","weak","weak_route","union_weak","union_route")
CONFMODELS=("local","primary","frozen_hgb")
R=("partial_transit_res","partial_transit_abs","partial_expected_log",
   "partial_observed_log","partial_path_coverage","partial_pair_coverage")
FRACTION=.5

class PartialRoute:
    def __init__(self,topo):
        self.nodes=tuple(sorted(topo.nodes))
        ix={n:i for i,n in enumerate(self.nodes)}
        self.pairs=tuple(itertools.combinations(self.nodes,2))
        row=[];col=[];val=[]
        for j,(a,b) in enumerate(self.pairs):
            rt=topo.route(a,b)
            for path,w in zip(rt.paths,rt.shares):
                for n in path[1:-1]:
                    row.append(ix[n]);col.append(j);val.append(float(w))
        self.mat=csr_matrix((val,(row,col)),shape=(len(self.nodes),len(self.pairs)))
        self.total=np.asarray(self.mat.sum(axis=1)).ravel()
    def features(self,reps):
        st={n:r.state for n,r in reps.items()}
        valid=np.array([a in st and b in st for a,b in self.pairs],bool)
        # Unknown pairs are excluded, never interpreted as measured zeros.
        x=np.fromiter((.5*(st[a].dest.get(b,0)+st[b].dest.get(a,0))
                         if ok else 0. for (a,b),ok in zip(self.pairs,valid)),float,count=len(valid))
        e=np.asarray(self.mat@x).ravel()
        support=np.asarray(self.mat@valid.astype(float)).ravel()
        out={}
        for i,n in enumerate(self.nodes):
            if n not in st:continue
            o=float(st[n].transited)
            # The total observed transit may include unobserved external OD.
            # Treat this as a proxy, not conservation equality.
            res=2*(o-e[i])/max(o+e[i],1.) if support[i]>0 else np.nan
            out[n]=dict(partial_transit_res=res,partial_transit_abs=abs(res),
                partial_expected_log=np.log1p(e[i]) if support[i]>0 else np.nan,
                partial_observed_log=np.log1p(o),
                partial_path_coverage=support[i]/self.total[i] if self.total[i]>0 else np.nan,
                partial_pair_coverage=float(valid.mean()) if len(valid) else np.nan)
        return out

def generate(seed,days,comp,normal,external,template=None):
    cfg=F.SimConfig(seed=seed,days=days)
    cache={}
    def inv(reps,topo,lstate):
        out=augmented(reps,topo,lstate)
        if "proxy" not in cache:
            cache["proxy"]=PartialRoute(topo);cache["legacy"]=RouteProxy(topo)
        old=cache["legacy"].features(reps)
        for n,row in cache["proxy"].features(reps).items():
            out[n].update(row);out[n].update(old[n])
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
    if d.duplicated(["nid","t"]).any():raise ValueError("duplicate node/time")
    return d,sim,eps

def scores(d,base,meta):
    a=score(base["primary"],d);b=score(base["peer"],d)
    h=predict(meta,features(d,a,b))
    return a,b,h

def matrix(d,a,b,h,route):
    cols=columns(d,F.OWN+NEW)+[c+s for c in DIRECTION+PEER for s in ("","__delta1","__delta5")]
    if route:cols+=list(R)
    x=d[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    x=np.column_stack([x,features(d,a,b),h])
    return x,cols+["meta_"+str(i) for i in range(8)]+["frozen_hgb_score"]

def fit_model(X,y,w,names):
    med=pd.DataFrame(X).median().fillna(0.).to_numpy()
    X=np.where(np.isfinite(X),X,med)
    keep=np.ptp(X,axis=0)>0
    if not keep.any():raise ValueError("constant inputs")
    model=HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=15,
        min_samples_leaf=40,l2_regularization=5.,early_stopping=False,random_state=9600)
    model.fit(X[:,keep],y,sample_weight=w)
    return dict(med=med,keep=keep,model=model,names=names)
def weak_predict(model,X):
    s=model["model"].predict_proba(np.where(np.isfinite(X),X,model["med"])[:,model["keep"]])[:,1]
    if not np.isfinite(s).all():raise ValueError("nonfinite specialist prediction")
    return s

def train(frozen,special,meta_path,out):
    base=base_models(frozen,special);meta=joblib.load(meta_path)["meta_hgb"]
    parts=[];audit=[];names={}
    for seed in TRAIN:
        d,_,_=generate(seed,1.5,48,96,16)
        a,b,h=scores(d,base,meta)
        y=d.label.to_numpy();ep=d.episode_id.to_numpy();pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
        # Mark weak episodes on independent TRAIN only: no three-window
        # score crossing at a reference negative tail cutoff. Never use
        # attack parameters as predictors.
        tau=raw_cut((d,),(h,),RAW_BUDGET*FRACTION)
        peaks=triple(d,h)
        weak_ids=set(ep[pos])-set(ep[pos[peaks[pos]>tau]])
        weak=np.isin(ep,list(weak_ids))&(y==1)
        q=float(np.quantile(np.maximum.reduce([a[neg],b[neg],h[neg]]),.98))
        high=np.maximum.reduce([a[neg],b[neg],h[neg]])>=q
        tail=neg[high];rest=neg[~high]
        rng=np.random.default_rng(seed)
        ix=np.sort(np.r_[pos,tail,rng.choice(rest,min(30000,len(rest)),replace=False)])
        w=np.ones(len(ix));p=y[ix]==1
        counts=pd.Series(ep[pos]).value_counts()
        w[p]=np.where(weak[ix][p],6.,1.)/pd.Series(ep[ix][p]).map(counts).to_numpy(float)
        data={}
        for n,route in (("weak",False),("weak_route",True)):
            X,names[n]=matrix(d,a,b,h,route);data[n]=X[ix]
        parts.append((data,y[ix],w))
        audit.append(dict(seed=seed,episodes=len(counts),weak_episodes=len(weak_ids),tail_negative=len(tail),
            cutoff=tau,old_finite=float(d[list(ROUTE)].notna().to_numpy().mean()),
            partial_finite=float(d[list(R)].notna().to_numpy().mean())))
        print("weak training",audit[-1],flush=True)
    y=np.concatenate([p[1] for p in parts]);w=np.concatenate([p[2] for p in parts])
    for k in (0,1):w[y==k]*=len(y)/(2*w[y==k].sum())
    models={n:fit_model(np.concatenate([p[0][n] for p in parts]),y,w,names[n]) for n in names}
    joblib.dump(models,out/"weak.joblib",compress=3)
    (out/"train_manifest.json").write_text(json.dumps(dict(train=TRAIN,dev=DEV,confirm=CONF,
        audit=audit,features={n:[c for c,k in zip(m["names"],m["keep"]) if k] for n,m in models.items()}),indent=2))

def combine(refs,aa,bb):
    # Reserve one quarter of the calibration budget for the specialist.
    x=raw_cut(refs,aa,RAW_BUDGET*FRACTION*.75)
    y=raw_cut(refs,bb,RAW_BUDGET*FRACTION*.25)
    z=[np.maximum(a-x,b-y) for a,b in zip(aa,bb)]
    extra=max(0.,joint_cut(refs,z,RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION))
    return x,y,extra

def test(seed,local,frozen,special,meta_path,weak_path,out):
    if seed not in DEV+CONF:raise ValueError("unregistered evaluation seed")
    dev=seed in DEV;gen=generate if dev else plain_generate
    d,sim,eps=gen(seed,1.5,32,64,10)
    eligible={}
    for i,e in enumerate(eps):
        if e.kind!="compromise" or e.t0<WARMUP:continue
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        if len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q):eligible[i]=e
    if len(eligible)<12:raise ValueError("too few complete episodes; no replacement")
    refs=(gen(seed+10000,3.,0,128,20,sim)[0],
          gen(seed+20000,3.,0,192,32,sim)[0],
          gen(seed+40000,3.,64,128,20,sim)[0])
    assert not refs[0].label.any() and not refs[1].label.any() and refs[2].label.any()
    assert all(set(x.nid)==set(d.nid) for x in refs)
    frames=(*refs,d);base=base_models(frozen,special,local)
    meta=joblib.load(meta_path)["meta_hgb"]
    abc=[scores(x,base,meta) for x in frames]
    ss={"local":[score(base["local"],x) for x in frames],"primary":[p[0] for p in abc],
        "frozen_hgb":[p[2] for p in abc]}
    if dev:
        models=joblib.load(weak_path)
        for n,m in models.items():
            ss[n]=[weak_predict(m,matrix(x,*p,n=="weak_route")[0]) for x,p in zip(frames,abc)]
        for n,b in (("union_weak","weak"),("union_route","weak_route")):
            a=ss["frozen_hgb"];z=ss[b];x,y,extra=combine(refs,a[:3],z[:3])
            ss[n]=[np.maximum(u-x,v-y)-extra for u,v in zip(a,z)]
    assert set(ss)==set(DEVMODELS if dev else CONFMODELS)
    rows=[];events=[];diagnostics=[]
    for n,s in ss.items():
        tau=0. if n.startswith("union_") else joint_cut(refs,s[:3],RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION)
        u=s[3];z=triple(d,u);m,det=metrics(d,u,z,tau,eligible)
        rows.append(dict(seed=seed,model=n,**m))
        for i,e in eligible.items():
            ix=(d.episode_id.to_numpy()==i)&(d.nid.to_numpy()==e.nid)
            events.append(dict(seed=seed,model=n,episode=i,detected=int(i in det),delay_seconds=det.get(i),
                hidden_rate=e.hidden_rate,mask_family=e.family,ntype=sim.topo.nodes[e.nid].ntype.value,
                duration_windows=int(ix.sum()),max_score=float(u[ix].max()),threshold=tau,
                raw_windows=int((u[ix]>tau).sum()),confirmed_candidate=int((z[ix]>tau).any())))
        print(seed,n,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    if dev:
        for c in ROUTE+R:
            v=d[c].to_numpy(float);ok=np.isfinite(v);y=d.label.to_numpy()
            auc=float(roc_auc_score(y[ok],v[ok])) if ok.any() and len(np.unique(y[ok]))==2 else np.nan
            diagnostics.append(dict(seed=seed,feature=c,finite_fraction=float(ok.mean()),
                distinct=int(pd.Series(v[ok]).nunique()),auc=auc,
                sd=float(v[ok].std()) if ok.any() else np.nan))
        pd.DataFrame(diagnostics).to_csv(out/"route_diagnostics.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(events).to_csv(out/"episodes.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(seed=seed,complete=len(eligible),
        reference_seeds=[seed+10000,seed+20000,seed+40000],scope="development" if dev else "frozen HGB confirmation",
        frozen_meta_run=45659800298),indent=2))

def aggregate(root,out,stage):
    seeds,models=(DEV,DEVMODELS) if stage=="dev" else (CONF,CONFMODELS)
    def collect(k):
        p=sorted(root.rglob(k+".csv"))
        if len(p)!=len(seeds):raise ValueError("incomplete worlds")
        return pd.concat([pd.read_csv(x) for x in p],ignore_index=True)
    m,e=collect("metrics"),collect("episodes");keys=["seed","model"]
    if m.duplicated(keys).any() or set(m[keys].itertuples(index=False,name=None))!={(s,n) for s in seeds for n in models}:
        raise ValueError("incomplete factorial")
    if e.duplicated(keys+["episode"]).any():raise ValueError("duplicate episodes")
    for seed in seeds:
        support=[set(x.episode) for _,x in e[e.seed==seed].groupby("model")]
        if len(support)!=len(models) or not all(x==support[0] for x in support):raise ValueError("unpaired episodes")
    m["both_budgets"]=(m.false_windows_day<=RAW_BUDGET)&(m.false_notifications_day<=INCIDENT_BUDGET)
    summary=m.groupby("model").agg(worlds=("seed","nunique"),episodes=("episodes","sum"),detected=("detected","sum"),
        recall=("recall","mean"),recall_sd=("recall","std"),auc=("auc","mean"),ap=("ap","mean"),
        fpr=("fpr","mean"),raw_day=("false_windows_day","mean"),worst_raw=("false_windows_day","max"),
        notice_day=("false_notifications_day","mean"),worst_notice=("false_notifications_day","max"),
        both_budgets=("both_budgets","sum"),mixed=("mixed_notifications","sum"),
        median_world_delay=("median_delay_seconds","median")).reset_index()
    base=e[e.model=="frozen_hgb"][["seed","episode","detected"]].rename(columns={"detected":"base"})
    t=e.merge(base,on=["seed","episode"],validate="many_to_one")
    t["rescued"]=(t.detected==1)&(t.base==0);t["lost"]=(t.detected==0)&(t.base==1)
    transitions=t.groupby("model").agg(rescued=("rescued","sum"),lost=("lost","sum")).reset_index()
    missed=e[e.detected==0].copy()
    missed["reason"]=np.select([missed.raw_windows==0,missed.confirmed_candidate==0],
          ["no_threshold_crossing","no_consecutive_confirmation"],default="candidate_not_credited")
    reasons=missed.groupby(["model","reason"]).size().rename("episodes").reset_index()
    e["rate_band"]=pd.cut(e.hidden_rate,[-np.inf,.01,.05,.1,np.inf])
    strata=e.groupby(["model","mask_family","ntype","rate_band"],observed=True).agg(
        episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean")).reset_index()
    p=m.pivot(index="seed",columns="model",values="recall")
    rng=np.random.default_rng(9800);comparisons=[]
    pairs=[("frozen_hgb","local"),("frozen_hgb","primary")] if stage=="confirm" else [(n,"frozen_hgb") for n in models if n!="frozen_hgb"]
    for a,b in pairs:
        v=(p[a]-p[b]).to_numpy();n=len(v)
        boot=rng.choice(v,size=(20000,n),replace=True).mean(axis=1)
        # Exact sign flips on 8 worlds; fixed-seed Monte Carlo on 32.
        if n<=16:
            null=np.array([np.mean(v*np.array(s)) for s in itertools.product((-1,1),repeat=n)])
            pv=float(np.mean(np.abs(null)>=abs(v.mean())-1e-12))
        else:
            null=(rng.choice([-1,1],size=(100000,n))*v).mean(axis=1)
            pv=float((1+(np.abs(null)>=abs(v.mean())-1e-12).sum())/(1+len(null)))
        comparisons.append(dict(a=a,b=b,delta=v.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),p_raw=pv))
    prev=0.
    for i,x in enumerate(sorted(comparisons,key=lambda z:z["p_raw"])):
        prev=max(prev,min(1.,(len(comparisons)-i)*x["p_raw"]));x["p_holm"]=prev
    for k,table in (("all_metrics",m),("episodes",e),("summary",summary),("transitions",transitions),
        ("missed_reasons",reasons),("strata",strata),("paired",pd.DataFrame(comparisons))):
        table.to_csv(out/(k+".csv"),index=False)
    print(summary.to_string(index=False));print(transitions.to_string(index=False))
    print(reasons.to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False))
    if stage=="dev":
        q=collect("route_diagnostics");q.to_csv(out/"route_diagnostics.csv",index=False)
        print(q.groupby("feature").agg(finite=("finite_fraction","mean"),sd=("sd","mean"),auc=("auc","mean")).to_string())
    print("Frozen HGB confirmation only; new specialists remain development candidates." if stage=="confirm" else "Development only; no specialist selected for production.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["train","test","aggregate"])
    p.add_argument("--seed",type=int);p.add_argument("--local",type=Path);p.add_argument("--frozen",type=Path)
    p.add_argument("--special",type=Path);p.add_argument("--meta",type=Path);p.add_argument("--weak",type=Path)
    p.add_argument("--input",type=Path);p.add_argument("--stage",choices=["dev","confirm"])
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="train":train(a.frozen,a.special,a.meta,a.output)
    elif a.mode=="test":test(a.seed,a.local,a.frozen,a.special,a.meta,a.weak,a.output)
    else:aggregate(a.input,a.output,a.stage)
