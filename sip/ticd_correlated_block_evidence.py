"""Causal multi-scale physical evidence with frozen sensor-conditioned offsets.

The upstream 200 OK counter is an existing trusted two-point sensor
assumption, not an additional oracle feature. Conditional normal
approximation is simulator-specific. Long-run scales are empirical, not
p-values or guarantees of independence.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from ticd_measured_loss import measured_frame
from ticd_full_count_evidence import CountReference
from ticd_sequential_physical import signed,KEYS

WINDOWS=(16,32,48)
SCORE_COLS=tuple(f"block_{kind}_{w}" for kind in ("plain","context","supported","plain_supported") for w in WINDOWS)

def inputs(ref,x):
    z,n,e,reliability,good,p=signed(ref,x)
    # When the trusted upstream count m is visible, condition on the
    # observed 200 OK count n rather than approximating m by n/(1-p).
    _,k=CountReference.counts(z)
    m=z.passive_sent.to_numpy(dtype=float)
    v=z.measured_loss_variance.to_numpy(dtype=float)
    has=(m>=n)&(m>0)&good
    a=(1-p)**2
    mu=n-m*a
    variance=m*a*(1-a)
    sd=np.sqrt(np.maximum(variance,1e-8))
    q=mu/sd
    expected=sd*np.exp(-.5*q*q)/np.sqrt(2*np.pi)+mu*ndtr(q)
    # Conditional variance of positive-part normal is approximated
    # conservatively by the untruncated variance.
    scale=np.sqrt(np.maximum(variance+4*m*m*(1-p)**2*v,1.))
    direct=(k-expected)/scale
    # Fall back to the old physical residual with no upstream observation.
    residual=np.where(has,np.clip(direct,-4,4),e)
    # Sensor disagreement and loss/volume context are measured independently
    # of ACK. No operational ACK-based adaptation of the baseline.
    av=(z.active_sent.to_numpy()>0)&(z.active_age.to_numpy()>=0)&(z.active_age.to_numpy()<=3)
    pa=(z.active_fwd_lost.to_numpy()+z.active_rev_lost.to_numpy()+.5)/(2*z.active_sent.to_numpy()+1.)
    pn=z.passive_sent.to_numpy(dtype=float)
    pp=(pn-z.passive_received.to_numpy()+.25)/(pn+.5)
    disagreement=np.where(av&has,np.clip((pa-pp)/.01,-4,4),0.)
    design=np.column_stack((np.ones(len(z)),np.clip(p/.02,0,4),np.log1p(n)/6,
                            disagreement,has.astype(float),av.astype(float)))
    return z,residual,design,good,reliability

def fit_profile(y,d,g,r,w):
    # Shrunk ridge, centered against the independent commissioning only.
    if g.sum()<80:return np.zeros(d.shape[1])
    a=d[g];b=y[g];weights=w[g]
    penalty=np.diag([40.,80.,80.,80.,80.,80.])
    coef=np.linalg.solve(a.T@(weights[:,None]*a)+penalty,a.T@(weights*b))
    return coef

def long_scale(e,valid):
    # Bartlett long-run variance of the frozen benign centered stream.
    # Invalid samples contribute neither residuals nor lag pairs.
    v=e[valid]
    if len(v)<80:return 1.
    center=float(v.mean());r=np.where(valid,e-center,0.)
    var=max(float(np.mean(v*v)-center*center),.05)
    total=var
    for lag in range(1,17):
        mask=valid[lag:]&valid[:-lag]
        if mask.any():
            cov=float(np.mean(r[lag:][mask]*r[:-lag][mask]))
            total+=2*(1-lag/17)*cov
    return float(np.sqrt(np.clip(total,.25,16.)))

def block_features(x,commission,reference):
    assert reference.mode=="adaptive_fast"
    ref=reference.reference
    c=measured_frame(commission,reference.mode,True)
    cz,ce,cd,cgood,cw=inputs(ref,c)
    z,e,d,good,w=inputs(ref,x)
    profiles={}
    for key,ix in cz.groupby(KEYS,sort=False).indices.items():
        v=cgood[ix];coef=fit_profile(ce[ix],cd[ix],v,None,cw[ix])
        residual=np.clip(ce[ix]-np.clip(cd[ix]@coef,-2,2),-4,4)*cw[ix]
        plain=np.clip(ce[ix],-4,4)*cw[ix]
        profiles[key]=(coef,long_scale(residual,v),long_scale(plain,v))
    out=x.copy()
    scores={name:np.zeros(len(z)) for name in SCORE_COLS}
    for key,ix in z.groupby(KEYS,sort=False).indices.items():
        coef,scale,pscale=profiles.get(key,(np.zeros(d.shape[1]),1.,1.))
        g=good[ix]
        plain=np.where(g,np.clip(e[ix],-4,4)*w[ix],0.)
        context=np.where(g,np.clip(e[ix]-np.clip(d[ix]@coef,-2,2),-4,4)*w[ix],0.)
        # Prefix sums; each block uses distinct intervals. Correlation
        # between adjacent blocks is NOT assumed zero; final tails are
        # calibrated empirically on independent normal runs.
        count=np.r_[0,np.cumsum(g)]
        for kind,r,s in (("plain",plain,pscale),("context",context,scale)):
            sums=np.r_[0.,np.cumsum(r)]
            for width in WINDOWS:
                for t,i in enumerate(ix):
                    if not g[t]:continue
                    end=t+1;parts=[]
                    for j in range(3):
                        hi=end-j*width;lo=hi-width
                        if lo<0:break
                        num=count[hi]-count[lo]
                        if num<.6*width:break
                        parts.append((sums[hi]-sums[lo])/max(s*np.sqrt(num),1e-9))
                    if not parts:continue
                    # Multiscale within each width: one, two and three
                    # adjacent non-overlapping blocks; no future data.
                    pooled=max(0.,max(sum(parts[:j])/np.sqrt(j) for j in range(1,len(parts)+1)))
                    scores[f"block_{kind}_{width}"][i]=pooled
                    support=max(0.,float(np.median(parts))) if len(parts)==3 else 0.
                    skind="supported" if kind=="context" else "plain_supported"
                    scores[f"block_{skind}_{width}"][i]=.4*pooled+.6*min(pooled,support)
    for name,values in scores.items():
        out[name]=0.
        out.loc[z.index,name]=values
    return out

class BlockAugmentedFusion:
    """Preserve the frozen Fast and sequential branches, recalibrate warning
    without conditioning on which intervals already trigger strong alerts."""
    def __init__(self,base,old,score,mult,budget,warning_budget):
        self.base=base;self.old=old;self.score=score;self.mult=mult
        self.budget=budget;self.warning_budget=warning_budget
        self.hybrid=False;self.observation_mode="adaptive_fast"
        self.threshold=None;self.warning_threshold=None
    def channels(self,x):
        parts={k:x[c].to_numpy()>self.mult*self.base.thresholds[k]
               for k,c in self.base.columns().items() if self.base.thresholds[k] is not None}
        # Previous sequential channel retained even when new block removed.
        parts["sequence"]=x[self.old.sequence].to_numpy()>self.mult*self.old.thresholds["sequence"]
        if self.score is not None:parts["block"]=x[self.score].to_numpy()>self.threshold
        return parts
    def strong(self,x):return np.logical_or.reduce(list(self.channels(x).values()))
    def calibrate(self,val):
        assert all(not x.is_anomaly.any() for x in val.values())
        runs=[g for x in val.values() for _,g in x.groupby("simulation_seed")]
        if self.score is not None:
            self.threshold=max(1e-9,max(float(np.quantile(g[self.score],1-self.budget,method="higher")) for g in runs))
        self.warning_threshold=max(self.base.warning_threshold,self.old.warning_threshold,
            max(float(np.quantile(np.where(g.split_eligible.to_numpy(dtype=bool),g.warn_0.to_numpy(),0.),
                   1-self.warning_budget,method="higher")) for g in runs))
        return self
    def predict_levels(self,x):
        a=self.strong(x)
        w=x.split_eligible.to_numpy(dtype=bool)&(x.warn_0.to_numpy()>self.warning_threshold)&~a
        z=np.zeros(len(x),dtype=np.int8);z[w]=1;z[a]=2
        return z
    def predict_masks(self,x):
        z=self.predict_levels(x)
        return {"normal":z==0,"warning_only":z==1,"confirmed_alert":z==2,"any_signal":z>0}
    def manifest(self):
        return {"all_original_invariants_preserved":True,"frozen_base":self.base.manifest(),
                "previous_sequence":self.old.manifest(),"block_score":self.score,
                "legacy_threshold_multiplier":self.mult,"block_budget":self.budget,
                "warning_budget":self.warning_budget,"warning_threshold":self.warning_threshold,
                "block_threshold":self.threshold,"warning_calibrated_independent_of_alerts":True,
                "no_operational_reference_update":True,"trusted_upstream_counter_assumed":True}
