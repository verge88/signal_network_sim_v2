"""Multiscale report consistency and role-normalized fusion.

Frozen Weak confirmation is separate from new training/selection.
Old test episodes are analyzed descriptively only. New features use current
and past report residuals, reset on missing windows, never true hidden state.
Role means configured node type (not attack label). Reference labels are
simulated commissioning labels; no production transfer is asserted.
"""
import argparse,itertools,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ss7.episode_ranking_development import load,infer,rank_predict,collect,union
from ss7.report_feature_specialist import generate
from ss7.weak_signal_confirmation import matrix,fit_model,weak_predict,scores
from ss7.threshold_transfer_study import score
from ss7.cusum_cascade_development import joint_cut,raw_cut
from ss7.incident_calibration_study import elapsed,triple,metrics,RAW_BUDGET,INCIDENT_BUDGET,WARMUP

TRAIN=tuple(range(12001,12007))
VAL=tuple(range(12101,12107))
TEST=tuple(range(12201,12209))
CONF=tuple(range(12301,12333))
BASE=("local","frozen_hgb","weak","rank2")
CANDIDATES=("weak","rank2","refit","multiscale","union_weak_rank","role_weak_rank",
            "union_weak_multi","role_weak_multi")
INVARIANTS=("pair_mean","pair_rms","od_pair_mean","od_pair_absmax",
            "relay_balance","dir_flow_gap","peer_excess")
WINDOWS=(3,10,30)
FRACTION=.5

def multiscale(d):
    n=d.nid.to_numpy();t=d.t.to_numpy()
    group=np.cumsum(np.r_[True,(n[1:]!=n[:-1])|(t[1:]!=t[:-1]+1)])
    result={}
    for c in INVARIANTS:
        s=d[c].astype(float)
        for k in WINDOWS:
            for kind,v in (("mean",s),("abs",s.abs())):
                # Inclusive trailing window; no crossing node/time gaps.
                a=v.groupby(group,sort=False).rolling(k,min_periods=k).mean()
                result[f"{c}_{kind}{k}"]=a.reset_index(level=0,drop=True).reindex(d.index).to_numpy()
    return pd.DataFrame(result,index=d.index)

def inputs(d,models,extra):
    a,b,h=scores(d,models[0],models[1])
    x,names=matrix(d,a,b,h,False)
    if extra:
        q=multiscale(d);x=np.column_stack([x,q.to_numpy()]);names+=list(q.columns)
    return x,names

def train(models,out):
    parts=[];names={};audit=[]
    for seed in TRAIN:
        d,_,_=generate(seed,1.5,48,96,16)
        h,w,_=infer(d,models)
        y=d.label.to_numpy();ep=d.episode_id.to_numpy()
        pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
        tau=raw_cut((d,),(w,),RAW_BUDGET*FRACTION)
        z=triple(d,w)
        weak=set(ep[pos])-set(ep[pos[z[pos]>tau]])
        hard=np.maximum(h,w)[neg];q=np.quantile(hard,.98)
        tail=neg[hard>=q];rest=neg[hard<q]
        rng=np.random.default_rng(seed)
        ix=np.sort(np.r_[pos,tail,rng.choice(rest,min(30000,len(rest)),replace=False)])
        p=y[ix]==1;weights=np.ones(len(ix))
        counts=pd.Series(ep[pos]).value_counts()
        weights[p]=np.where(np.isin(ep[ix][p],list(weak)),6.,1.)/pd.Series(ep[ix][p]).map(counts).to_numpy()
        data={}
        for name,extra in (("refit",False),("multiscale",True)):
            x,names[name]=inputs(d,models,extra);data[name]=x[ix]
        parts.append((data,y[ix],weights))
        audit.append(dict(seed=seed,episodes=len(counts),weak_episodes=len(weak),negative_tail=len(tail)))
        print("training",audit[-1],flush=True)
    y=np.concatenate([p[1] for p in parts]);w=np.concatenate([p[2] for p in parts])
    for k in (0,1):w[y==k]*=len(y)/(2*w[y==k].sum())
    result={n:fit_model(np.concatenate([p[0][n] for p in parts]),y,w,names[n]) for n in names}
    joblib.dump(result,out/"multiscale.joblib",compress=3)
    (out/"manifest.json").write_text(json.dumps(dict(train=TRAIN,val=VAL,test=TEST,confirm=CONF,
        audit=audit,features={n:[c for c,k in zip(m["names"],m["keep"]) if k] for n,m in result.items()}),indent=2))

def role_tail(refs,ss,frames):
    """Empirical negative survival score by node type, with pooled fallback.
    Fit only on references. Worst survival over regimes is conservative.
    Use smoothed ranks for ties; no test labels or role-specific test fit.
    """
    all_roles=sorted(set().union(*(set(d.ntype) for d in refs)))
    fitted=[]
    for d,s in zip(refs,ss):
        neg=d.label.to_numpy()==0;roles=d.ntype.to_numpy()
        pooled=np.sort(s[neg])
        if not len(pooled):raise ValueError("no negative reference")
        by={}
        for role in all_roles:
            v=np.sort(s[neg&(roles==role)])
            by[role]=v if len(v)>=1000 else pooled
        fitted.append((pooled,by))
    out=[]
    for d,s in frames:
        roles=d.ntype.to_numpy();p=np.zeros(len(d))
        for pooled,by in fitted:
            v=np.ones(len(d))
            for role in np.unique(roles):
                ix=roles==role;a=by.get(role,pooled)
                # inclusive survival, ties handled conservatively.
                v[ix]=(len(a)-np.searchsorted(a,s[ix],side="left")+1)/(len(a)+1)
            p=np.maximum(p,v)
        out.append(-np.log10(p))
    return out

def evaluate(seed,models,rank,extra,out,selection=None):
    if seed not in VAL+TEST+CONF:raise ValueError("unregistered world")
    confirm=seed in CONF;val=seed in VAL
    if seed in TEST and selection is None:raise ValueError("selection missing")
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
    ss={"local":[score(models[0]["local"],x) for x in frames],
        "frozen_hgb":[x[0] for x in z],"weak":[x[1] for x in z],
        "rank2":[rank_predict(rank,x[2]) for x in z]}
    requested=BASE if confirm else tuple(dict.fromkeys(BASE+(CANDIDATES if val else (selection["selected"],))))
    if not confirm:
        for name,m in extra.items():
            if name in requested or (name=="multiscale" and any("multi" in v for v in requested)):
                ss[name]=[weak_predict(m,inputs(x,models,name=="multiscale")[0]) for x in frames]
        for name,b in (("union_weak_rank","rank2"),("role_weak_rank","rank2"),
                       ("union_weak_multi","multiscale"),("role_weak_multi","multiscale")):
            if name not in requested:continue
            a=ss["weak"];q=ss[b]
            if name.startswith("role"):
                a=role_tail(refs,a[:3],list(zip(frames,a)))
                q=role_tail(refs,q[:3],list(zip(frames,q)))
            x,y,ex=union(refs,a[:3],q[:3],.5)
            ss[name]=[np.maximum(u-x,v-y)-ex for u,v in zip(a,q)]
    rows=[];events=[]
    for name in requested:
        s=ss[name];u=s[3]
        tau=0. if name.startswith(("union_","role_")) else joint_cut(refs,s[:3],RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION)
        z=triple(d,u);m,det=metrics(d,u,z,tau,eligible)
        rows.append(dict(seed=seed,model=name,**m))
        for i,e in eligible.items():
            ix=(d.episode_id.to_numpy()==i)&(d.nid.to_numpy()==e.nid)
            events.append(dict(seed=seed,model=name,episode=i,detected=int(i in det),
                hidden_rate=e.hidden_rate,mask_family=e.family,ntype=sim.topo.nodes[e.nid].ntype.value,
                delay_seconds=det.get(i),duration_windows=int(ix.sum()),max_margin=float(u[ix].max()-tau),
                raw_windows=int((u[ix]>tau).sum()),confirmed_candidate=int((z[ix]>tau).any())))
        print(seed,name,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(events).to_csv(out/"episodes.csv",index=False)

def overlap(e,out):
    a=e[e.model=="weak"].copy()
    b=e[e.model=="rank2"][["seed","episode","detected"]].rename(columns={"detected":"rank_detected"})
    t=a.merge(b,on=["seed","episode"],validate="one_to_one")
    if len(t)!=len(a) or len(t)!=len(b):raise ValueError("unpaired overlap")
    t["pattern"]=np.select([(t.detected==1)&(t.rank_detected==1),t.detected==1,t.rank_detected==1],
        ["both","weak_only","rank_only"],default="neither")
    t["rate_band"]=pd.cut(t.hidden_rate,[-np.inf,.01,.05,.1,np.inf])
    t.to_csv(out/"weak_rank_overlap.csv",index=False)
    t[t.detected==0].to_csv(out/"weak_missed.csv",index=False)
    s=t.groupby(["ntype","pattern"],observed=True).size().rename("episodes").reset_index()
    strata=t.groupby(["ntype","mask_family","rate_band","pattern"],observed=True).size().rename("episodes").reset_index()
    s.to_csv(out/"overlap_by_type.csv",index=False);strata.to_csv(out/"overlap_strata.csv",index=False)
    print("WEAK/RANK paired overlap",t.pattern.value_counts().to_string())
    print(s.to_string(index=False))

def select(root,out):
    m,e,s=collect(root,VAL,tuple(dict.fromkeys(BASE+CANDIDATES)))
    q=s[s.model.isin(CANDIDATES)].sort_values(["both_budgets","recall","worst_raw","model"],ascending=[False,False,True,True])
    chosen=str(q.iloc[0].model)
    (out/"selection.json").write_text(json.dumps(dict(selected=chosen,validation_seeds=VAL,
        rule="maximize compliant worlds, then recall, then minimize worst raw, then name"),indent=2))
    s.to_csv(out/"validation_summary.csv",index=False);m.to_csv(out/"validation_metrics.csv",index=False)
    e.to_csv(out/"validation_episodes.csv",index=False)
    overlap(e,out);print(s.to_string(index=False));print("SELECTED",chosen)

def aggregate(root,out,stage,selection=None):
    confirm=stage=="confirm"
    chosen="weak" if confirm else selection["selected"]
    m,e,s=collect(root,CONF if confirm else TEST,BASE if confirm else tuple(dict.fromkeys(BASE+(chosen,))))
    overlap(e,out)
    pivot=m.pivot(index="seed",columns="model",values="recall")
    rng=np.random.default_rng(12300);tests=[]
    for b in ("local","frozen_hgb") if confirm else ("local","frozen_hgb","weak"):
        if b==chosen:continue
        v=(pivot[chosen]-pivot[b]).to_numpy();n=len(v)
        boot=rng.choice(v,size=(20000,n),replace=True).mean(axis=1)
        if n<=16:
            null=np.array([np.mean(v*np.array(sign)) for sign in itertools.product((-1,1),repeat=n)])
            p=float(np.mean(np.abs(null)>=abs(v.mean())-1e-12))
        else:
            null=(rng.choice([-1,1],size=(100000,n))*v).mean(axis=1)
            p=float((1+(np.abs(null)>=abs(v.mean())-1e-12).sum())/(len(null)+1))
        tests.append(dict(a=chosen,b=b,delta=v.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),p_raw=p))
    prev=0.
    for i,x in enumerate(sorted(tests,key=lambda z:z["p_raw"])):
        prev=max(prev,min(1.,(len(tests)-i)*x["p_raw"]));x["p_holm"]=prev
    missed=e[e.detected==0].copy()
    missed["reason"]=np.select([missed.raw_windows==0,missed.confirmed_candidate==0],
        ["no_threshold_crossing","no_consecutive_confirmation"],default="candidate_not_credited")
    reasons=missed.groupby(["model","reason"]).size().rename("episodes").reset_index()
    for k,tab in (("summary",s),("all_metrics",m),("episodes",e),("paired",pd.DataFrame(tests)),("missed_reasons",reasons)):
        tab.to_csv(out/(k+".csv"),index=False)
    print("EVALUATED",chosen);print(s.to_string(index=False));print(reasons.to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))
    print("Frozen Weak confirmation; role candidates not evaluated here." if confirm else "Development only.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["train","validate","select","test","confirm","aggregate","old_overlap"])
    p.add_argument("--seed",type=int);p.add_argument("--local",type=Path);p.add_argument("--frozen",type=Path)
    p.add_argument("--special",type=Path);p.add_argument("--meta",type=Path);p.add_argument("--weak",type=Path)
    p.add_argument("--ranks",type=Path);p.add_argument("--extra",type=Path);p.add_argument("--selection",type=Path)
    p.add_argument("--input",type=Path);p.add_argument("--stage",choices=["dev","confirm"])
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    selection=json.loads(a.selection.read_text()) if a.selection else None
    if a.mode=="old_overlap":overlap(pd.read_csv(a.input/"episodes.csv"),a.output)
    elif a.mode=="select":select(a.input,a.output)
    elif a.mode=="aggregate":aggregate(a.input,a.output,a.stage,selection)
    else:
        models=load(a.local,a.frozen,a.special,a.meta,a.weak)
        if a.mode=="train":train(models,a.output)
        else:evaluate(a.seed,models,joblib.load(a.ranks)["rank2"],joblib.load(a.extra) if a.extra else {},a.output,selection)
