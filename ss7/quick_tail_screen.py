"""Short screening of role-conditioned tail calibration and causal score smoothing.

Frozen models, unchanged integrated physical simulation. Short duration and
reduced factorial schedule are deliberate screening limitations; no final
performance claim. No test tuning, no retraining, no oracle inputs.
"""
import argparse,itertools,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
from ss7 import five_specialist_development as S
from ss7 import integrated_cu_five_class_development as I
from ss7 import fix as F
from ss7.incident_calibration_study import elapsed,triple,metrics,notify,RAW_BUDGET,INCIDENT_BUDGET,WARMUP
from ss7.episode_ranking_development import union,collect

DAYS=.30
SEEDS=(20101,20102)
CLASSES=S.CLASSES
METHODS=("dual_75","special_equal","smooth5","smooth15","role_tail","role_tail_smooth5","role_tail_smooth15")
ORIG=I.make_episodes
def schedule(sim,cfg,rng,balanced,stress=False):
    # 20 compromise episodes cover every role x masking scenario once.
    # Rotate intensity across cells by seed; not the full 100-cell factorial.
    controls=F.make_episodes(sim,cfg,rng,n_compromise=0,
        n_normal=19 if stress else 13,n_ext=3 if stress else 2)
    if not balanced:return controls
    occ=defaultdict(list);eps=[]
    def place(role):
        nodes=[n for n in sim.topo.slaves if sim.topo.nodes[n].ntype.value==role]
        length=int(rng.integers(60,121))
        for _ in range(5000):
            n=int(rng.choice(nodes));a=int(rng.integers(WARMUP+5,cfg.n_intervals-length-5));b=a+length-1
            if all(b+3<c or a>d+3 for c,d in occ[n]):
                occ[n].append((a,b));return n,a,b
        raise ValueError("short schedule infeasible")
    cells=list(itertools.product(list(I.B.scenarios("corrected")),I.B.ROLES))
    rng.shuffle(cells)
    rates=np.array(list(I.B.RATES)*4);rng.shuffle(rates)
    for ((soph,family,_),role),rate in zip(cells,rates):
        n,a,b=place(role)
        eps.append(F.Episode(kind="compromise",nid=n,t0=a,t1=b,hidden_rate=float(rate),family=family,
            sophistication=soph,budget=float(rng.uniform(.7,1.)),params=rng.random(2)))
    for c in I.OTHER:
        for _ in range(2):
            n,a,b=place(I.ATTACK_ROLE[c]);eps.append(F.Episode(kind=c,nid=n,t0=a,t1=b,mult=float(rng.uniform(*I.RANGES[c]))))
    return eps+controls

def generate(seed,balanced=True,template=None,stress=False):
    old=I.DAYS;I.DAYS=DAYS;I.make_episodes=schedule
    try:return S.generate(seed,balanced,template,stress)
    finally:I.DAYS=old;I.make_episodes=ORIG

def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))
def smooth(d,p,k):
    n=d.nid.to_numpy();t=d.t.to_numpy()
    g=np.cumsum(np.r_[True,(n[1:]!=n[:-1])|(t[1:]!=t[:-1]+1)])
    x=pd.Series(logit(p),index=d.index)
    m=x.groupby(g,sort=False).rolling(k,min_periods=k).mean().reset_index(level=0,drop=True).reindex(d.index)
    # First k-1 windows use the original score, never future observations.
    return 1/(1+np.exp(-m.fillna(x).to_numpy()))
def role_transform(frames,ss):
    # Estimate role-specific upper-tail location and spread on the two
    # benign commissioning realizations only. No test or attack labels.
    cuts={}
    for role in sorted(set(frames[0].ntype)):
        a=np.concatenate([logit(s[(d.ntype.to_numpy()==role)&(d.label.to_numpy()==0)])
                          for d,s in zip(frames[:2],ss[:2])])
        if len(a)<100:raise ValueError("role reference too small")
        q95,q995=np.quantile(a,[.95,.995]);cuts[role]=(q995,max(q995-q95,.25))
    out=[]
    for d,s in zip(frames,ss):
        z=np.empty(len(d))
        for role,(q,w) in cuts.items():
            idx=d.ntype.to_numpy()==role;z[idx]=(logit(s[idx])-q)/w
        assert np.isfinite(z).all();out.append(z)
    return out

def evaluate(seed,frozen,special,out):
    start=time.monotonic();d,sim,eps,tr=generate(seed)
    eligible={i:e for i,e in enumerate(eps) if e.kind in CLASSES}
    assert len(eligible)==28 and sum(e.kind=="compromise" for e in eligible.values())==20
    for i,e in eligible.items():
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        assert len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q)
    print("test generation",seed,len(d),round(time.monotonic()-start,1),flush=True)
    refs=[]
    for offset,balanced,stress in ((10000,False,False),(20000,False,True),(40000,True,False)):
        q,_,_,_=generate(seed+offset,balanced,sim,stress);refs.append(q)
        print("reference generation",offset,len(q),round(time.monotonic()-start,1),flush=True)
    assert not refs[0].label.any() and not refs[1].label.any() and refs[2].label.any()
    frames=(*refs,d);ss={}
    a=[I.predict(q,frozen,"expanded_cu") for q in frames];b=[I.predict(q,frozen,"expanded_special") for q in frames]
    x,y,z=union(refs,a[:3],b[:3],.75);dual=np.maximum(a[3]-x,b[3]-y)-z
    ss["special_equal"]=[[S.score(q,special["context_"+c]) for q in frames] for c in CLASSES]
    for k in (5,15):ss[f"smooth{k}"]=[[smooth(q,p,k) for q,p in zip(frames,head)] for head in ss["special_equal"]]
    for key in ("special_equal","smooth5","smooth15"):
        dest="role_tail" if key=="special_equal" else "role_tail_"+key
        ss[dest]=[role_transform(frames,head) for head in ss[key]]
    rows=[];events=[];diagnostics=[]
    y=d.label.to_numpy();ids=d.episode_id.to_numpy()
    for name in METHODS:
        if name=="dual_75":u=dual;z=triple(d,u);win=None;mm=None
        else:
            margins,cut,extra=S.fuse(refs,ss[name],np.full(5,.2))
            mm=margins[3];u=mm.max(1);z,win=S.confirm(d,mm)
        m,det=metrics(d,u,z,0.,eligible)
        first={}
        for j in notify(d,z,0.):
            ep=ids[j-2:j+1]
            if (y[j-2:j+1]==1).all() and (ep==ep[-1]).all() and int(ep[-1]) in eligible:first.setdefault(int(ep[-1]),j)
        assert set(first)==set(det)
        for i,e in eligible.items():
            mask=(ids==i)&(d.nid.to_numpy()==e.nid);hit=i in det
            pred=CLASSES[win[first[i]]] if hit and win is not None else "unspecified" if hit else "missed"
            peak=float(np.max(u[mask]));confirmed=float(np.max(z[mask]))
            # Separate insufficient score, failure to sustain, and cooldown.
            reason="detected" if hit else "below_threshold" if peak<=0 else "no_streak" if confirmed<=0 else "notification_or_onset"
            events.append(dict(seed=seed,model=name,episode=i,attack=e.kind,detected=int(hit),predicted=pred,
                correct_type=int(pred==e.kind),delay_seconds=det.get(i),reason=reason,peak=peak,confirmed_peak=confirmed,
                ntype=sim.topo.nodes[e.nid].ntype.value))
        rows.append(dict(seed=seed,model=name,**m))
        print(seed,name,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
        if mm is not None:
            for j,c in enumerate(CLASSES):
                for role in sorted(set(d.ntype)):
                    pos=(d.attack.to_numpy()==c)&(d.ntype.to_numpy()==role)
                    neg=(y==0)&(d.ntype.to_numpy()==role)
                    if pos.any() and neg.any():
                        diagnostics.append(dict(seed=seed,model=name,attack=c,role=role,
                            pos_median=float(np.median(mm[pos,j])),pos_q95=float(np.quantile(mm[pos,j],.95)),
                            benign_q999=float(np.quantile(mm[neg,j],.999)),benign_max=float(mm[neg,j].max())))
    for n,q in (("metrics",rows),("episodes",events),("tail_diagnostics",diagnostics)):pd.DataFrame(q).to_csv(out/(n+".csv"),index=False)
    print("SCREEN seconds",round(time.monotonic()-start,1),flush=True)

def aggregate(root,out):
    m,e,s=collect(root,SEEDS,METHODS)
    a=e.groupby(["model","attack"]).agg(episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean"),
        typed_recall=("correct_type","mean")).reset_index()
    miss=e.groupby(["model","attack","reason"]).size().rename("episodes").reset_index()
    s=s.merge(a.groupby("model").agg(macro_recall=("recall","mean"),worst_class=("recall","min")).reset_index(),on="model")
    for n,q in (("summary",s),("attacks",a),("miss_reasons",miss),("episodes",e),("metrics",m)):
        q.to_csv(out/(n+".csv"),index=False)
    pd.concat([pd.read_csv(x) for x in sorted(root.rglob("tail_diagnostics.csv"))],ignore_index=True).to_csv(out/"tail_diagnostics.csv",index=False)
    print(s.to_string(index=False));print(a.to_string(index=False));print(miss.to_string(index=False))
    print("SCREEN ONLY: short overnight simulated windows, 2 worlds; budgets extrapolated, no significance or deployment claim.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["screen","aggregate"]);p.add_argument("--seed",type=int)
    p.add_argument("--frozen",type=Path);p.add_argument("--special",type=Path);p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="screen":
        assert a.seed in SEEDS
        evaluate(a.seed,joblib.load(a.frozen),joblib.load(a.special),a.output)
    else:aggregate(a.input,a.output)
