"""Causal robust multi-sensor loss-state filter, no true-loss inputs.

Two fixed process-noise configurations are evaluated, not tuned online
against labels. Each raw counter batch is consumed once. Variance is a
heuristic uncertainty scale, not a calibrated posterior credible interval.
"""
from __future__ import annotations
import numpy as np

SETTINGS={"adaptive_stable":(.0007,.15),"adaptive_fast":(.0015,.25)}

def estimate(x,pc,vc,mode):
    q,cu_gain=SETTINGS[mode]
    valid=x.primary_observer_valid.to_numpy(dtype=bool)
    n=x.active_sent.to_numpy(dtype=float)
    f=x.active_fwd_lost.to_numpy(dtype=float);r=x.active_rev_lost.to_numpy(dtype=float)
    age=x.active_age.to_numpy(dtype=float)
    pn=x.passive_sent.to_numpy(dtype=float);pl=pn-x.passive_received.to_numpy(dtype=float)
    assert ((f>=0)&(r>=0)&(f<=n)&(r<=n)&(pl>=0)&(pl<=pn)).all()
    av=(n>0)&(age>=0)&(age<=3)&valid
    pv=(pn>0)&valid
    pa=(f+r+.5)/(2*n+1.)
    pp=(pl+.25)/(pn+.5)
    va=np.maximum(pa*(1-pa)/np.maximum(2*n+2,1.),1e-7)*2
    # Direction mismatch and stale observations increase uncertainty.
    va+=.25*((f-r)/np.maximum(n,1.))**2+np.maximum(age,0)*(.002**2)
    vp=np.maximum(pp*(1-pp)/np.maximum(pn+1,1.),1e-7)*3
    p=np.zeros(len(x));v=np.zeros(len(x));wa=np.zeros(len(x));wp=np.zeros(len(x))
    held=np.zeros(len(x),dtype=int)
    for ix in x.groupby(["simulation_seed","node_id"],sort=False).indices.values():
        mean=float(pc[ix[0]]);var=max(float(vc[ix[0]]),.005**2)
        missing=0;last_a=last_p=0
        for i in ix:
            mean=float(np.clip(mean,1e-6,.45))
            prior=min(.02**2,var+q*q*(1+min(missing,20)))
            last_a=min(last_a+1,12) if av[i] else 0
            last_p=min(last_p+1,12) if pv[i] else 0
            if not (av[i] or pv[i]):
                missing+=1
                # Do not abruptly replace the accurate previous estimate
                # with sparse CU frequency. Gradual bounded pull; variance
                # grows to cover both drift and discrepancy with CU.
                delta=float(pc[i]-mean)
                gain=min(cu_gain,prior/max(prior+vc[i]+.003**2,1e-12))
                if not valid[i]:gain=0.
                mean+=gain*float(np.clip(delta,-.004,.004))
                var=min(.02**2,prior+gain*delta*delta)
            else:
                missing=0
                a=float(va[i])*(1+3/(last_a+1))
                b=float(vp[i])*(1+3/(last_p+1))
                if av[i] and pv[i]:
                    diff=float(pa[i]-pp[i])
                    # Agreement of independent observation channels supports
                    # faster tracking of a real regime shift. Disagreement
                    # has no guaranteed attribution; penalize each innovation
                    # relative to the previous causal state.
                    agree=abs(diff)<=3*np.sqrt(a+b)+.001
                    if agree:
                        jump=min(abs(float((pa[i]+pp[i])/2-mean)),.015)
                        prior=min(.02**2,prior+.5*jump*jump)
                    else:
                        a+=.25*diff*diff+max(0.,abs(pa[i]-mean)-2*np.sqrt(prior+a))**2
                        b+=.25*diff*diff+max(0.,abs(pp[i]-mean)-2*np.sqrt(prior+b))**2
                # Huber-style soft bound for an isolated inconsistent sample.
                # Positive process noise permits recovery; no sensor is
                # permanently blacklisted based on an inferred attack label.
                scores=[]
                for ok,z,noise in ((av[i],pa[i],a),(pv[i],pp[i],b)):
                    if ok:
                        innovation=float(z-mean)
                        s=np.sqrt(prior+noise)
                        noise*=max(1.,abs(innovation)/max(3*s,1e-6))**2
                        scores.append(1/max(noise,1e-9))
                    else:scores.append(0.)
                sa,sb=scores
                # Conservative covariance inflation because the two sources
                # share the same network state. Weights sum to at most one.
                total=1/prior+.5*(sa+sb)
                ka=.5*sa/total;kb=.5*sb/total
                mean+=ka*(pa[i]-mean)+kb*(pp[i]-mean)
                var=max(1/total,1e-8)
                if av[i] and pv[i]:var+=ka*kb*(pa[i]-pp[i])**2
                var=min(var,.02**2)
                wa[i]=ka;wp[i]=kb
            p[i]=np.clip(mean,1e-6,.45);v[i]=var;held[i]=missing
    assert np.isfinite(p).all() and np.isfinite(v).all() and (v>=0).all()
    assert ((wa+wp)<=1+1e-12).all()
    return p,v,av|pv,wa,wp,held
