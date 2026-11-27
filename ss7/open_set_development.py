"""Open-set SS7 detection: five leave-one-attack-type-out folds and
one independent synthetic-profile novelty fold. No withheld scenario
is simulated in supervised fit, validation or commissioning references.
The novelty heads train only on benign worlds (including normal faults).
New profiles are synthetic service-load anomalies, not demonstrated exploits.
"""
import argparse,itertools,json,math
from pathlib import Path
from collections import defaultdict
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score,average_precision_score
from ss7 import five_specialist_development as S
from ss7 import integrated_cu_five_class_development as I
from ss7 import fix as F
from ss7.cusum_cascade_development import raw_cut,joint_cut
from ss7.incident_calibration_study import elapsed,triple,metrics,notify,RAW_BUDGET,INCIDENT_BUDGET,WARMUP

CLASSES=tuple(S.CLASSES)
NOVEL=("novel_auth_profile","novel_mixed_map_profile")
FOLDS=CLASSES+("new_profiles",)
NORMAL_TRAIN=(18001,18002,18003,18004)
TRAIN=(18101,18102,18103,18104)
VAL=(18201,18202)
TEST=tuple(range(18301,18309))
METHODS=("known_only","iforest","vae","invariant","novelty_ensemble","open_iforest","open_vae","open_invariant","open_ensemble")
PRIMARY="open_ensemble"
NOVEL_SHARE=.25
FRACTION=.5
ORIG_EP=I.make_episodes
ORIG_PATH=I.IntegratedSim.attack_paths
ORIG_OTHER=I.OTHER
ORIG_CLASSES=I.CLASSES
INV=tuple(x for x in list(I.NEW)+list(I.DIRECTION)+list(I.PEER)+list(I.O.EXTRA) if
          any(s in x for s in ("abs","rms","deficit","gap","balance","excess","res","mean","max")) and "support" not in x)
def known(fold):return tuple(c for c in CLASSES if c!=fold)
def novel(fold):return NOVEL if fold=="new_profiles" else (fold,)

def paths(self,kind,n):
    if kind not in NOVEL:return ORIG_PATH(self,kind,n)
    cache=getattr(self,"_novel_paths",{})
    self._novel_paths=cache
    if (kind,n) in cache:return cache[kind,n]
    allowed=(F.Service.AUTH,) if kind==NOVEL[0] else (F.Service.LOCATION,F.Service.MOBILITY,F.Service.SMS_MT)
    out=defaultdict(list)
    for f in self.flows:
        if f.dst!=n or f.service not in allowed:continue
        for p in self.topo.route(f.src,f.dst).paths:
            if n==p[-1]:out[f.service].append((f,p))
    if not out:raise ValueError("no synthetic profile route")
    cache[kind,n]=dict(out)
    return cache[kind,n]

def generate(seed,fold,phase,balanced=True,template=None,stress=False):
    # Filter episodes BEFORE physical simulation, not merely after generating
    # windows. This also excludes indirect effects on neighbors and histories.
    if fold not in FOLDS:raise ValueError(fold)
    def episodes(sim,cfg,rng,balanced,stress=False):
        eps=ORIG_EP(sim,cfg,rng,balanced,stress)
        if phase=="normal":return [e for e in eps if e.kind not in CLASSES and e.kind!="ext_attack"]
        if phase!="test" and fold!="new_profiles":eps=[e for e in eps if e.kind!=fold]
        if phase=="test" and fold=="new_profiles":
            occ=defaultdict(list)
            for e in eps:
                if e.kind in CLASSES:occ[e.nid].append((e.t0,e.t1))
            candidates=[n for n in sim.topo.slaves if sim.topo.nodes[n].ntype.value=="SCP"]
            for kind in NOVEL:
                for _ in range(25):
                    length=int(rng.integers(60,121))
                    for attempt in range(5000):
                        n=int(rng.choice(candidates));a=int(rng.integers(WARMUP+5,cfg.n_intervals-length-5));b=a+length-1
                        if all(b+3<c or a>d+3 for c,d in occ[n]):break
                    else:raise ValueError("novel episode placement")
                    occ[n].append((a,b))
                    eps.append(F.Episode(kind=kind,nid=n,t0=a,t1=b,mult=float(rng.uniform(2.,3.))))
        return eps
    I.make_episodes=episodes
    # Do NOT extend OTHER while calling original scheduler, which indexes
    # fixed class dictionaries. Only extend inside step_true and labeling.
    old_step=I.IntegratedSim.step_true
    def step(self,t,eps):
        before=I.OTHER;I.OTHER=ORIG_OTHER+NOVEL
        try:return old_step(self,t,eps)
        finally:I.OTHER=before
    if phase=="test" and fold=="new_profiles":
        I.IntegratedSim.attack_paths=paths;I.IntegratedSim.step_true=step
        I.CLASSES=ORIG_CLASSES+NOVEL
    try:d,sim,eps,tr=S.generate(seed,balanced,template,stress)
    finally:
        I.make_episodes=ORIG_EP;I.IntegratedSim.attack_paths=ORIG_PATH
        I.IntegratedSim.step_true=old_step;I.OTHER=ORIG_OTHER;I.CLASSES=ORIG_CLASSES
    present=set(d.attack)-{"normal"}
    if phase=="normal":assert not present and all(e.kind not in CLASSES+NOVEL+("ext_attack",) for e in eps)
    elif phase!="test":assert present<=set(known(fold)) and all(e.kind not in novel(fold) for e in eps)
    else:assert set(novel(fold))<=present
    return d,sim,eps,tr

def matrix(d,names):
    x=d[names].to_numpy(dtype=np.float32,copy=True)
    x[~np.isfinite(x)]=np.nan
    return x

def scale_fit(x):
    med=np.nanmedian(x,axis=0);med=np.nan_to_num(med)
    lo,hi=np.nanpercentile(x,[10,90],axis=0)
    width=np.nan_to_num(hi-lo);width=np.maximum(width,.05*np.maximum(np.abs(med),1.))
    return med.astype(np.float32),width.astype(np.float32)

def scale(x,med,width):
    # Missing observations map to median; missingness is separately
    # observable through CU delivery/staleness and support features.
    return np.clip((np.where(np.isfinite(x),x,med)-med)/width,-12,12).astype(np.float32)

def vae_fit(x,seed):
    # Small VAE, NumPy implementation, Gaussian reconstruction + KL.
    # Reparameterized minibatches and Adam, fixed epoch count, no attack data.
    rng=np.random.default_rng(seed);p=x.shape[1];h=64;l=12
    def w(a,b):return (rng.normal(size=(a,b))*np.sqrt(2/(a+b))).astype(np.float32)
    P=[w(p,h),np.zeros(h,np.float32),w(h,l),np.zeros(l,np.float32),
       w(h,l),np.zeros(l,np.float32),w(l,h),np.zeros(h,np.float32),w(h,p),np.zeros(p,np.float32)]
    m=[np.zeros_like(v) for v in P];v=[np.zeros_like(v) for v in P];step=0;beta=.05
    for epoch in range(20):
        order=rng.permutation(len(x));loss=[]
        for start in range(0,len(x),256):
            a=x[order[start:start+256]];N=len(a)
            W,b,U,c,V,f,G,g,H,j=P
            h1=np.tanh(a@W+b);mu=h1@U+c;raw=h1@V+f
            lv=np.clip(raw,-6,3);std=np.exp(.5*lv);noise=rng.normal(size=mu.shape).astype(np.float32)
            z=mu+std*noise;h2=np.tanh(z@G+g);rec=h2@H+j
            diff=rec-a;loss.append(float(np.mean(diff*diff)+beta*.5*np.mean(np.exp(lv)+mu*mu-1-lv)))
            dr=2*diff/(N*p);dh2=(dr@H.T)*(1-h2*h2);dz=dh2@G.T
            dmu=dz+beta*mu/(N*l)
            dlv=(dz*noise*std*.5+beta*.5*(np.exp(lv)-1)/(N*l))*((raw>=-6)&(raw<=3))
            dh1=(dmu@U.T+dlv@V.T)*(1-h1*h1)
            grads=[a.T@dh1,dh1.sum(0),h1.T@dmu,dmu.sum(0),h1.T@dlv,dlv.sum(0),
                   z.T@dh2,dh2.sum(0),h2.T@dr,dr.sum(0)]
            step+=1
            for k,gg in enumerate(grads):
                gg=np.clip(gg,-5,5);m[k]=.9*m[k]+.1*gg;v[k]=.999*v[k]+.001*gg*gg
                P[k]-=.001*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        if not np.isfinite(loss).all():raise ValueError("VAE divergence")
    return P

def vae_score(x,P):
    W,b,U,c,V,f,G,g,H,j=P
    h1=np.tanh(x@W+b);mu=h1@U+c;lv=np.clip(h1@V+f,-6,3)
    rec=np.tanh(mu@G+g)@H+j
    # Deterministic mean reconstruction + latent KL diagnostic.
    return np.mean((rec-x)**2,axis=1)+.025*np.mean(np.exp(lv)+mu*mu-1-lv,axis=1)

def train_normal(out):
    parts=[]
    for seed in NORMAL_TRAIN:
        d,sim,eps,tr=generate(seed,"new_profiles","normal",False)
        rng=np.random.default_rng(seed)
        for role,q in d.groupby("ntype"):
            ix=rng.choice(len(q),min(16000,len(q)),replace=False)
            parts.append(q.iloc[ix][list(dict.fromkeys(S.cols(d,True)+["ntype"]))].copy())
        print("normal fit",seed,len(d),flush=True)
    d=pd.concat(parts,ignore_index=True);models={}
    names=S.cols(d,True)
    inv=[names.index(c) for c in INV if c in names]
    assert inv
    for i,(role,q) in enumerate(d.groupby("ntype")):
        x=matrix(q,names);med,width=scale_fit(x);z=scale(x,med,width)
        est=IsolationForest(n_estimators=160,max_samples=1024,n_jobs=2,random_state=18000+i).fit(z)
        P=vae_fit(z,18000+i)
        # Robust deviation of observable invariants. Compute on unclipped
        # input, omit undefined comparisons rather than treating as violation.
        models[role]=dict(med=med,width=width,forest=est,vae=P,inv=inv)
        print("trained normal-only",role,len(q),flush=True)
    joblib.dump(dict(names=names,models=models),out/"normal.joblib",compress=3)

def normal_scores(d,pack):
    out={k:np.zeros(len(d)) for k in ("iforest","vae","invariant")}
    for role,idx in d.groupby("ntype").indices.items():
        m=pack["models"][role];x=matrix(d.iloc[idx],pack["names"]);z=scale(x,m["med"],m["width"])
        out["iforest"][idx]=-m["forest"].score_samples(z)
        out["vae"][idx]=np.concatenate([vae_score(z[j:j+4096],m["vae"]) for j in range(0,len(z),4096)])
        a=np.abs((x[:,m["inv"]]-m["med"][m["inv"]])/m["width"][m["inv"]])
        a=np.where(np.isfinite(a),a,0.)
        out["invariant"][idx]=np.sort(a,axis=1)[:,-min(3,a.shape[1]):].mean(1)
    assert all(np.isfinite(x).all() for x in out.values())
    return out

def train_fold(fold,out):
    parts=[]
    for seed in TRAIN:
        d,sim,eps,tr=generate(seed,fold,"train")
        d["uid"]=str(seed)+":"+d.episode_id.astype(str)
        pos=np.flatnonzero(d.label.to_numpy()==1);neg=np.flatnonzero(d.label.to_numpy()==0)
        ix=np.sort(np.r_[pos,np.random.default_rng(seed).choice(neg,min(50000,len(neg)),replace=False)])
        parts.append(d.iloc[ix][list(dict.fromkeys(S.cols(d,True)+["uid","label","attack"]))].copy())
        print("fit",fold,seed,d.attack.value_counts().to_dict(),flush=True)
    d=pd.concat(parts,ignore_index=True)
    assert not set(d.attack)&set(novel(fold))
    models={c:S.fit(d,c,True) for c in known(fold)}
    joblib.dump(models,out/"known.joblib",compress=3)
    (out/"manifest.json").write_text(json.dumps(dict(fold=fold,known=list(models),excluded=novel(fold),train=TRAIN),indent=2))

def fuse(refs,channels,weights):
    # Different score units (IF, VAE, probability, invariants) require
    # normalization before taking a joint maximum. Reference tail spread
    # only, no held-out labels or scores used to determine cuts/scales.
    cuts=[];scales=[]
    for a,w in zip(channels,weights):
        cut=raw_cut(refs,a[:3],RAW_BUDGET*FRACTION*w)
        spread=max(cut-max(np.quantile(s[d.label.to_numpy()==0],.99) for d,s in zip(refs,a[:3])),1e-6)
        cuts.append(cut);scales.append(spread)
    margins=[np.stack([(a[j]-c)/s for a,c,s in zip(channels,cuts,scales)],axis=1) for j in range(4)]
    extra=max(0.,joint_cut(refs,[x.max(1) for x in margins[:3]],RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION))
    return margins[3]-extra

def evaluate(fold,seed,normal,models,out,phase):
    assert set(models)==set(known(fold))
    assert seed in (VAL if phase=="validate" else TEST)
    d,sim,eps,tr=generate(seed,fold,phase)
    valid=set(known(fold))|(set(novel(fold)) if phase=="test" else set())
    eligible={i:e for i,e in enumerate(eps) if e.kind in valid}
    assert set(d.attack)-{"normal"}==valid
    for i,e in eligible.items():
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        assert len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q)
    refs=(generate(seed+10000,fold,"reference",False,sim)[0],
          generate(seed+20000,fold,"reference",False,sim,True)[0],
          generate(seed+40000,fold,"reference",True,sim)[0])
    assert not refs[0].label.any() and not refs[1].label.any()
    assert all(not set(q.attack)&set(novel(fold)) for q in refs)
    frames=(*refs,d);ks=[[S.score(q,models[c]) for q in frames] for c in models]
    nn=[normal_scores(q,normal) for q in frames];ns={c:[q[c] for q in nn] for c in nn[0]}
    y=d.label.to_numpy();ids=d.episode_id.to_numpy()
    rows=[];events=[];alerts_out=[]
    for method in METHODS:
        kc=list(models);nc=[]
        if method=="known_only":nc=[]
        elif method in ns:kc=[];nc=[method]
        elif method=="novelty_ensemble":kc=[];nc=list(ns)
        elif method=="open_ensemble":nc=list(ns)
        else:nc=[method.removeprefix("open_")]
        kshare=1. if not nc else 0. if not kc else 1.-NOVEL_SHARE
        channels=[ks[list(models).index(c)] for c in kc]+[ns[c] for c in nc]
        weights=[kshare/len(kc) for c in kc]+[(1-kshare)/len(nc) for c in nc]
        m=fuse(refs,channels,weights);u=m.max(1)
        zall=np.stack([triple(d,v) for v in m.T],axis=1);z=zall.max(1)
        # Known label takes precedence ONLY when a known head confirms.
        # Otherwise a confirmed novelty head raises unknown anomaly.
        k=len(kc);winner=zall.argmax(1)
        if k:
            w=zall[:,:k].argmax(1);winner=np.where(zall[:,:k].max(1)>0,w,winner)
        labels=kc+["unknown"]*len(nc)
        met,det=metrics(d,u,z,0.,eligible);first={}
        for j in notify(d,z,0.):
            ep=ids[j-2:j+1]
            if (y[j-2:j+1]==1).all() and (ep==ep[-1]).all() and int(ep[-1]) in eligible:
                first.setdefault(int(ep[-1]),j)
            alerts_out.append(dict(fold=fold,seed=seed,model=method,predicted=labels[winner[j]],
                false=int((y[j-2:j+1]==0).all())))
        assert set(first)==set(det)
        for i,e in eligible.items():
            is_unknown=e.kind in novel(fold);pred=labels[winner[first[i]]] if i in det else "missed"
            events.append(dict(fold=fold,seed=seed,model=method,episode=i,attack=e.kind,
                unknown=int(is_unknown),detected=int(i in det),predicted=pred,
                correct=int(pred==("unknown" if is_unknown else e.kind)),delay_seconds=det.get(i),
                ntype=sim.topo.nodes[e.nid].ntype.value))
        rows.append(dict(fold=fold,seed=seed,model=method,**met))
    e=pd.DataFrame(events);m=pd.DataFrame(rows)
    q=e.groupby(["model","unknown"]).agg(episodes=("episode","size"),recall=("detected","mean"),correct=("correct","mean")).reset_index()
    print(fold,seed,phase);print(q.to_string(index=False));print(m[["model","false_windows_day","false_notifications_day"]].to_string(index=False))
    for n,v in (("metrics",m),("episodes",e),("alerts",pd.DataFrame(alerts_out,columns=["fold","seed","model","predicted","false"]))):v.to_csv(out/(n+".csv"),index=False)

def aggregate(root,out,phase):
    seeds=VAL if phase=="validate" else TEST
    def read(name):
        p=sorted(root.rglob(name+".csv"))
        if len(p)!=len(FOLDS)*len(seeds):raise ValueError("incomplete "+name)
        return pd.concat([pd.read_csv(x) for x in p],ignore_index=True)
    m,e,a=read("metrics"),read("episodes"),read("alerts")
    keys=["fold","seed","model"]
    assert not m.duplicated(keys).any() and set(m[keys].itertuples(index=False,name=None))==set(itertools.product(FOLDS,seeds,METHODS))
    assert not e.duplicated(keys+["episode"]).any()
    assert all(len(q)==len(METHODS) and q.episode.nunique()==1 for _,q in e.groupby(["fold","seed","episode"]))
    assert (e.unknown==0).all() if phase=="validate" else (e.unknown==1).any()
    m["compliant"]=(m.false_windows_day<=RAW_BUDGET)&(m.false_notifications_day<=INCIDENT_BUDGET)
    s=m.groupby(["fold","model"]).agg(worlds=("seed","nunique"),recall=("recall","mean"),auc=("auc","mean"),
        raw_day=("false_windows_day","mean"),worst_raw=("false_windows_day","max"),
        notice_day=("false_notifications_day","mean"),worst_notice=("false_notifications_day","max"),both_budgets=("compliant","sum")).reset_index()
    at=e.groupby(["fold","model","unknown","attack"]).agg(episodes=("episode","size"),detected=("detected","sum"),
        recall=("detected","mean"),correct=("correct","sum"),correct_recall=("correct","mean"),delay=("delay_seconds","median")).reset_index()
    u=e.groupby(["fold","model","unknown"]).agg(episodes=("episode","size"),recall=("detected","mean"),
        correct_recall=("correct","mean")).reset_index()
    conf=e.groupby(["fold","model","attack","predicted"]).size().rename("episodes").reset_index()
    # Fixed primary open_ensemble, no test selection. Paired inference
    # uses seeds as clusters because folds reuse test worlds.
    tests=[]
    if phase=="test":
        for endpoint,subset,col in (("unknown_detection",e[e.unknown==1],"detected"),("unknown_attribution",e[e.unknown==1],"correct"),("known_detection",e[e.unknown==0],"detected")):
            p=subset.groupby(["seed","fold","model"])[col].mean().groupby(["seed","model"]).mean().unstack()
            diff=(p[PRIMARY]-p["known_only"]).to_numpy()
            rng=np.random.default_rng(18000);boot=rng.choice(diff,(20000,len(diff)),replace=True).mean(1)
            null=np.array([np.mean(diff*np.array(z)) for z in itertools.product((-1,1),repeat=len(diff))])
            tests.append(dict(endpoint=endpoint,delta=diff.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),p_raw=np.mean(np.abs(null)>=abs(diff.mean())-1e-12)))
        prev=0.
        for i,x in enumerate(sorted(tests,key=lambda x:x["p_raw"])):
            prev=max(prev,min(1.,(len(tests)-i)*x["p_raw"]));x["p_holm"]=prev
    for n,q in (("summary",s),("attacks",at),("unknown_known",u),("confusion",conf),("paired",pd.DataFrame(tests)),("all_metrics",m),("episodes",e),("alerts",a)):q.to_csv(out/(n+".csv"),index=False)
    print("PRIMARY",PRIMARY);print(s.to_string(index=False));print(u.to_string(index=False));print(pd.DataFrame(tests).to_string(index=False))
    print("Unknown anomaly is not proof of malicious intent. Novel profiles are synthetic; shared seeds across folds are not independent replicates.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["normal","train","validate","test","aggregate"])
    p.add_argument("--fold",choices=FOLDS);p.add_argument("--seed",type=int);p.add_argument("--normal",type=Path)
    p.add_argument("--models",type=Path);p.add_argument("--input",type=Path);p.add_argument("--phase",choices=["validate","test"])
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="normal":train_normal(a.output)
    elif a.mode=="train":train_fold(a.fold,a.output)
    elif a.mode=="aggregate":aggregate(a.input,a.output,a.phase)
    else:evaluate(a.fold,a.seed,joblib.load(a.normal),joblib.load(a.models),a.output,a.mode)
