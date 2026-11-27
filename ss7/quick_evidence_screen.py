"""Screen causal K-of-N and finite-memory CUSUM confirmation.
Same two short worlds as quick_tail_screen, for paired diagnostic comparison
only. All thresholds fitted on independent commissioning references.
"""
import argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from ss7 import quick_tail_screen as Q
from ss7 import five_specialist_development as S
from ss7 import integrated_cu_five_class_development as I
from ss7.incident_calibration_study import (elapsed,triple,metrics,notify,threshold,bucket_peaks,
    RAW_BUDGET,INCIDENT_BUDGET)
from ss7.cusum_cascade_development import raw_cut
from ss7.episode_ranking_development import collect,union

SEEDS=Q.SEEDS
CLASSES=Q.CLASSES
METHODS=("dual_75","special_equal","k2of4","k3of5","comp_k2of4",
         "pooled_triple","pooled_k2of4","cusum5","cusum15","comp_cusum5","comp_cusum15")
FRACTION=.5

def groups(d):
    n=d.nid.to_numpy();t=d.t.to_numpy()
    return np.cumsum(np.r_[True,(n[1:]!=n[:-1])|(t[1:]!=t[:-1]+1)])

def kofn(d,s,k,n):
    # Current score must exceed the threshold, plus K-1 of preceding N-1.
    # Require at least 3 current contiguous windows so that evaluation
    # uses the same last-three ground-truth convention as baseline.
    g=groups(d);z=np.full(len(s),-np.inf)
    if n>len(s):return z
    a=np.lib.stride_tricks.sliding_window_view(s,n)
    kth=np.partition(a,n-k,axis=1)[:,n-k]
    valid=g[n-1:]==g[:len(s)-n+1]
    z[n-1:]=np.where(valid,np.minimum(kth,s[n-1:]),-np.inf)
    return z

def cusum(d,p,center,scale,n):
    # Bounded-memory one-sided CUSUM: maximum positive suffix sum
    # of normalized log-odds minus fixed drift. Reset across gaps.
    a=np.clip((Q.logit(p)-center)/scale,-8,8)-.25
    g=groups(d);s=np.zeros(len(a));tot=np.zeros(len(a))
    for k in range(n):
        if k>=len(a):break
        v=np.zeros(len(a));v[k:]=np.where(g[k:]==g[:len(a)-k],a[:len(a)-k],0.)
        tot+=v;s=np.maximum(s,tot)
    return s

def tail_params(refs,ss):
    x=np.concatenate([Q.logit(a[d.label.to_numpy()==0]) for d,a in zip(refs[:2],ss[:2])])
    q95,q99,q999=np.quantile(x,[.95,.99,.999])
    return q95,q99,q999,max(q99-q95,.25)

def transform(frames,refs,heads,mode):
    result=[]
    for j,h in enumerate(heads):
        q95,q99,q999,width=tail_params(refs,h)
        if mode.startswith("cusum") or (mode.startswith("comp_cusum") and j==0):
            n=int(mode.split("cusum")[-1]);result.append([cusum(d,p,q95,width,n) for d,p in zip(frames,h)])
        elif mode.startswith("pooled") or mode.startswith("comp_cusum"):
            # Dimensionless tail excess, avoids pooling unlike score units.
            result.append([(Q.logit(p)-q999)/width for p in h])
        else:result.append(h)
    return result

def confirm(d,m,mode):
    z=[]
    for j,s in enumerate(m.T):
        if mode in ("k2of4","pooled_k2of4") or (mode=="comp_k2of4" and j==0):v=kofn(d,s,2,4)
        elif mode=="k3of5":v=kofn(d,s,3,5)
        else:v=triple(d,s)
        z.append(v)
    zz=np.stack(z,axis=1)
    return zz.max(1),zz.argmax(1)

def calibrate(refs,heads,mode):
    # Each head has a raw cutoff; a single global margin is then calibrated.
    # For pooled modes estimate tail baselines instead of allocating <1
    # allowed false observation per head on short commissioning windows.
    pooled=mode.startswith(("pooled","cusum","comp_cusum"))
    if pooled:
        cuts=np.array([max(np.quantile(s[d.label.to_numpy()==0],.999) for d,s in zip(refs,h[:3])) for h in heads])
        scales=np.array([max(c-max(np.quantile(s[d.label.to_numpy()==0],.95) for d,s in zip(refs,h[:3])),.25) for c,h in zip(cuts,heads)])
    else:
        cuts=np.array([raw_cut(refs,h[:3],RAW_BUDGET*FRACTION/5) for h in heads]);scales=np.ones(5)
    margins=[np.stack([(h[j]-c)/w for h,c,w in zip(heads,cuts,scales)],axis=1) for j in range(4)]
    raw=max(threshold(m.max(1)[d.label.to_numpy()==0],RAW_BUDGET*FRACTION,elapsed(d)) for d,m in zip(refs,margins))
    if not pooled:raw=max(0.,raw)
    margins=[m-raw for m in margins]
    # Match the baseline last-three attribution convention. Actual scores
    # on contaminated references are retained, including historical attack
    # evidence persisting into a benign window. Conservative bucket cap.
    nc=[]
    for d,m in zip(refs,margins):
        z,_=confirm(d,m,mode)
        benign=triple(d,np.where(d.label.to_numpy()==0,1.,-np.inf))>0
        b=bucket_peaks(d,np.where(benign,z,-np.inf))
        if np.isfinite(b).any():nc.append(threshold(b,INCIDENT_BUDGET*FRACTION,elapsed(d)))
    extra=max([0.]+nc)
    # Keep raw score independent of the additional notification threshold;
    # every confirmed notification still requires a raw exceedance.
    return margins,extra

def evaluate(seed,frozen,special,out):
    start=time.monotonic();d,sim,eps,tr=Q.generate(seed)
    eligible={i:e for i,e in enumerate(eps) if e.kind in CLASSES};assert len(eligible)==28
    refs=[]
    for off,b,stress in ((10000,False,False),(20000,False,True),(40000,True,False)):
        q,_,_,_=Q.generate(seed+off,b,sim,stress);refs.append(q)
        print("reference",off,"seconds",round(time.monotonic()-start,1),flush=True)
    assert not refs[0].label.any() and not refs[1].label.any() and refs[2].label.any()
    frames=(*refs,d);heads=[[S.score(q,special["context_"+c]) for q in frames] for c in CLASSES]
    a=[I.predict(q,frozen,"expanded_cu") for q in frames];b=[I.predict(q,frozen,"expanded_special") for q in frames]
    x,y,z=union(refs,a[:3],b[:3],.75);dual=np.maximum(a[3]-x,b[3]-y)-z
    rows=[];events=[];cal=[]
    y=d.label.to_numpy();ids=d.episode_id.to_numpy();nid=d.nid.to_numpy()
    for name in METHODS:
        extra=0.
        if name=="dual_75":u=dual;z=triple(d,u);win=None
        elif name=="special_equal":
            margins,_,_=S.fuse(refs,heads,np.full(5,.2));u=margins[3].max(1);z,win=S.confirm(d,margins[3])
        else:
            ss=transform(frames,refs,heads,name);margins,extra=calibrate(refs,ss,name)
            u=margins[3].max(1);z,win=confirm(d,margins[3],name);z=z-extra
            for q,m in zip(refs,margins[:3]):
                v,_=confirm(q,m,name)
                met,_=metrics(q,m.max(1),v-extra,0.)
                assert met["false_windows_day"]<=RAW_BUDGET*FRACTION+1e-8
                assert met["false_notifications_day"]<=INCIDENT_BUDGET*FRACTION+1e-8
            cal.append(dict(seed=seed,model=name,notice_extra=extra))
        m,det=metrics(d,u,z,0.,eligible);first={}
        for j in notify(d,z,0.):
            ep=ids[j-2:j+1]
            if (y[j-2:j+1]==1).all() and (ep==ep[-1]).all() and int(ep[-1]) in eligible:first.setdefault(int(ep[-1]),j)
        assert set(first)==set(det)
        for i,e in eligible.items():
            mask=(ids==i)&(nid==e.nid);hit=i in det
            pred=CLASSES[win[first[i]]] if hit and win is not None else "unspecified" if hit else "missed"
            peak=float(np.max(u[mask]));confirmed=float(np.max(z[mask]))
            reason="detected" if hit else "below_threshold" if peak<=0 else "no_confirmation" if confirmed<=0 else "notification_or_onset"
            # Additional stricter support check: full lookback wholly inside
            # the episode. Primary uses the same last-three convention.
            support=3
            if name in ("k2of4","pooled_k2of4") or (name=="comp_k2of4" and pred=="compromise"):support=4
            if name=="k3of5":support=5
            if name.startswith("cusum") or (name.startswith("comp_cusum") and pred=="compromise"):support=int(name.split("cusum")[-1])+2
            strict=int(hit and det[i]>=30*(support-1))
            events.append(dict(seed=seed,model=name,episode=i,attack=e.kind,detected=int(hit),predicted=pred,
                correct_type=int(pred==e.kind),strict_first_detection=strict,delay_seconds=det.get(i),reason=reason,
                peak=peak,confirmed_peak=confirmed,ntype=sim.topo.nodes[e.nid].ntype.value))
        rows.append(dict(seed=seed,model=name,**m))
        print(seed,name,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    for name,v in (("metrics",rows),("episodes",events),("calibration",cal)):
        pd.DataFrame(v).to_csv(out/(name+".csv"),index=False)
    print("SCREEN seconds",round(time.monotonic()-start,1),flush=True)

def aggregate(root,out):
    m,e,s=collect(root,SEEDS,METHODS)
    a=e.groupby(["model","attack"]).agg(episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean"),
        typed_recall=("correct_type","mean"),strict_first_recall=("strict_first_detection","mean")).reset_index()
    miss=e.groupby(["model","attack","reason"]).size().rename("episodes").reset_index()
    s=s.merge(a.groupby("model").agg(macro_recall=("recall","mean"),worst_class=("recall","min")).reset_index(),on="model")
    for n,q in (("summary",s),("attacks",a),("miss_reasons",miss),("episodes",e),("metrics",m)):
        q.to_csv(out/(n+".csv"),index=False)
    print(s.to_string(index=False));print(a.to_string(index=False));print(miss.to_string(index=False))
    print("Diagnostic reuse of two short worlds, not independent confirmation; first detection full support is a secondary conservative diagnostic.")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["screen","aggregate"]);p.add_argument("--seed",type=int)
    p.add_argument("--frozen",type=Path);p.add_argument("--special",type=Path);p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="screen":
        assert a.seed in SEEDS
        evaluate(a.seed,joblib.load(a.frozen),joblib.load(a.special),a.output)
    else:aggregate(a.input,a.output)
