"""Predictive censored-count likelihood evidence for the existing SIP simulator.

Conditional independence of transit 200 and uncensored ACK given m,p is
a property of this simulator, not a universal SIP law. The observed ACK
is min(A,N), so its endpoint is a survival probability, not a point mass.
Only primitive measured counts are used; the posterior never uses ACK.
Beta posteriors use fractional historical counts as a discounted working
model, not an exact posterior for a time-varying latent process.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import betaincinv,logsumexp
from scipy.stats import binom

SOURCES=("active","matched","mixture")
TIMES=("scan","cusum")
DELTAS=(.0025,.005,.01,.025,.04)
WINDOWS=(16,48,96,144)
KEYS=["simulation_seed","node_id"]
Q=(np.arange(12)+.5)/12.

def posterior(x):
    m=x.passive_sent.to_numpy(dtype=float)
    n=x.cu_n_200_transit.to_numpy(dtype=float)
    a=x.cu_n_ack_transit.to_numpy(dtype=float)
    valid=x.primary_observer_valid.to_numpy(dtype=bool)&(x.cu_delivered.to_numpy()>0)
    valid &= (m>0)&(n<=m)&(n>=a)&(a>=0)
    ap=x.active_sent.to_numpy(dtype=float)*2
    al=x.active_fwd_lost.to_numpy(dtype=float)+x.active_rev_lost.to_numpy(dtype=float)
    age=x.active_age.to_numpy(dtype=int)
    av=(ap>0)&(age==0)&x.primary_observer_valid.to_numpy(dtype=bool)
    # Matched receiver count N replaces passive_received. They measure the
    # same simulated population; never add them as independent observations.
    mp=np.where(valid,m,0.);ml=np.where(valid,m-n,0.)
    ap=np.where(av,ap,0.);al=np.where(av,al,0.)
    pa=np.empty((len(x),2));pb=np.empty((len(x),2))
    for ix in x.groupby(KEYS,sort=False).indices.values():
        ca=la=cm=lm=0.
        for i in ix:
            # Prior counts from earlier intervals only. Discount transport
            # history to permit changes and avoid pretending long-run iid.
            ca*=.8;la*=.8;cm*=.85;lm*=.85
            # Active measurement must refer to current interval. Stale
            # measurements are ignored in this branch; old TICD still sees
            # them. Cap active effective mass to avoid overconfidence.
            na=min(ca+ap[i],4096.)
            ka=(la+al[i])*na/max(ca+ap[i],1.)
            nm=min(cm,1024.);km=lm*nm/max(cm,1.)
            # Discount active branch when contemporaneous matched history
            # disagrees; compare using binomial-scale uncertainty.
            if na>0 and nm>20:
                aa=(ka+.5)/(na+1);bb=(km+.5)/(nm+1)
                noise=aa*(1-aa)/(na+1)+bb*(1-bb)/(nm+1)+.001**2
                factor=min(1.,9*noise/max((aa-bb)**2,1e-12))
                na*=factor;ka*=factor
            # Both branches condition on current N out of M. No ACK,
            # no simulated loss truth, no labels are used in inference.
            for j,(trials,failures) in enumerate(((na,ka),(nm,km))):
                pa[i,j]=.5+failures+ml[i]
                pb[i,j]=.5+trials-failures+mp[i]-ml[i]
            ca+=ap[i];la+=al[i];cm+=mp[i];lm+=ml[i]
    assert (pa>0).all() and (pb>0).all()
    return m,n,a,valid,pa,pb

def predictive_logprob(m,n,a,p,delta):
    """Array shapes [row,quadrature]. Integrate outside this function."""
    prob=np.clip((1-p)**2*(1-delta),1e-10,1-1e-10)
    # A_obs=N is right-censored. N=0 is a valid but uninformative row.
    z=np.where(a[:,None]<n[:,None],
               binom.logpmf(a[:,None],m[:,None],prob),
               binom.logsf(n[:,None]-1,m[:,None],prob))
    return np.maximum(z,-700.)

def increments(x):
    m,n,a,valid,alpha,beta=posterior(x)
    out=np.zeros((len(x),len(SOURCES),len(DELTAS)))
    ix=np.flatnonzero(valid)
    if not len(ix):return out,valid
    allp=[]
    for j in range(2):
        p=betaincinv(alpha[ix,j,None],beta[ix,j,None],Q[None,:])
        allp.append(np.clip(p,1e-9,.5))
    logp=np.empty((len(ix),3,len(DELTAS)+1))
    for j,p in enumerate(allp):
        for d,delta in enumerate((0.,*DELTAS)):
            z=predictive_logprob(m[ix],n[ix],a[ix],p,delta)
            logp[:,j,d]=logsumexp(z,axis=1)-np.log(len(Q))
    logp[:,2,:]=np.logaddexp(logp[:,0,:],logp[:,1,:])-np.log(2.)
    out[ix]=np.clip(logp[:,:,1:]-logp[:,:,:1],-20.,20.)
    assert np.isfinite(out).all()
    return out,valid

def features(x):
    z=x.sort_values(KEYS+["interval"],kind="stable")
    assert not z.duplicated(KEYS+["interval"]).any()
    assert (z.groupby(KEYS,sort=False).interval.diff().dropna()==1).all()
    inc,valid=increments(z)
    scores={f"pred_{s}_{t}":np.zeros(len(z)) for s in SOURCES for t in TIMES}
    for ix in z.groupby(KEYS,sort=False).indices.values():
        a=inc[ix];v=valid[ix];sums=np.concatenate([np.zeros((1,len(SOURCES),len(DELTAS))),np.cumsum(a,axis=0)])
        count=np.r_[0,np.cumsum(v)]
        state=np.zeros((len(SOURCES),len(DELTAS)))
        for t,i in enumerate(ix):
            state=np.maximum(.995*state+a[t],0.)
            if not v[t]:continue
            pooled=np.zeros(len(SOURCES))
            for width in WINDOWS:
                lo=max(0,t+1-width)
                if count[t+1]-count[lo]<max(4,.5*(t+1-lo)):continue
                pooled=np.maximum(pooled,(sums[t+1]-sums[lo]).max(axis=1))
            for j,source in enumerate(SOURCES):
                scores[f"pred_{source}_scan"][i]=pooled[j]
                scores[f"pred_{source}_cusum"][i]=state[j].max()
    out=x.copy()
    for k,v in scores.items():out.loc[z.index,k]=v
    out["pred_valid"]=pd.Series(valid,index=z.index,dtype=bool).reindex(x.index)
    assert out["pred_valid"].dtype==bool and not out["pred_valid"].isna().any()
    return out

class PredictiveFusion:
    """All previous joint channels retained via the frozen primary score."""
    def __init__(self,source,time,weight,budget,base,warning,audit=False):
        self.source=source;self.time=time;self.weight=float(weight);self.budget=budget
        self.base=base;self.warning_threshold=warning;self.hybrid=bool(audit)
        self.threshold=None
    def score(self,x):
        valid=x.primary_observer_valid.to_numpy(dtype=bool)
        a=np.where(valid,self.base.score(x)/self.base.threshold,0.)
        if self.weight:
            name=f"pred_{self.source}_{self.time}"
            b=np.where(valid,x[name].to_numpy()/self.scale,0.)
            a=np.maximum(a,self.weight*b)
        if self.hybrid:a=np.maximum(a,x.pred_audit.to_numpy())
        return a
    def calibrate(self,runs,scales):
        self.scale=scales.get(f"pred_{self.source}_{self.time}",1.)
        self.threshold=max(1e-9,max(float(np.quantile(self.score(g),1-self.budget,method="higher")) for g in runs))
        return self
    def predict_levels(self,x):
        valid=x.primary_observer_valid.to_numpy(dtype=bool)
        a=self.score(x)>self.threshold
        w=valid&x.split_eligible.to_numpy(dtype=bool)&(x.warn_0.to_numpy()>self.warning_threshold)&~a
        z=np.zeros(len(x),dtype=np.int8);z[w]=1;z[a]=2;return z
    def predict_masks(self,x):
        z=self.predict_levels(x)
        return {"normal":z==0,"warning_only":z==1,"confirmed_alert":z==2,"any_signal":z>0}
    def manifest(self):
        return {"source":self.source,"time":self.time,"weight":self.weight,"scale":self.scale,
                "threshold":self.threshold,"budget":self.budget,"independent_policy_audit":self.hybrid,
                "legacy_channels_preserved":True,"no_oracle_at_inference":True,
                "working_predictive_likelihood_not_anytime_guarantee":True}
