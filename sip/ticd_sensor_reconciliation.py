"""Causal reconciliation of sampled transport loss, no ACK or attack labels.

The receiver 200 OK counter and delivered CU transit 200 OK count refer
to the SAME population in this simulator. Reconciliation is conditional
on this explicit instrumentation contract, not an independent third
loss measurement. Active probe bias is identifiable only relative to
that trusted matched count. Complete outages remain unidentifiable.
"""
from __future__ import annotations
import numpy as np

MODES=("reconciled_raw","reconciled_cal")

def estimate_reconciled(x,pc,vc,reconcile=True,delay_correct=True):
    valid=x.primary_observer_valid.to_numpy(dtype=bool)
    sent=x.active_sent.to_numpy(dtype=float)
    lost=x.active_fwd_lost.to_numpy(dtype=float)+x.active_rev_lost.to_numpy(dtype=float)
    age=x.active_age.to_numpy(dtype=int)
    m=x.passive_sent.to_numpy(dtype=float);received=x.passive_received.to_numpy(dtype=float)
    n=x.cu_n_200_transit.to_numpy(dtype=float)
    paired=valid&(x.cu_delivered.to_numpy()>0)&(m>0)&(n>=0)&(n<=m)
    av=valid&(sent>0)&(age>=0)&(age<=3);pv=valid&(m>0)
    # When CU is delivered the transport count is trusted and matches the
    # upstream observation population. Never count it twice.
    if reconcile:received=np.where(paired,n,received)
    pa=(lost+.25)/(2*sent+.5)
    pp=(m-received+.25)/(m+.5)
    va=np.maximum(pa*(1-pa)/(2*sent+1.5),2e-8)
    vp=np.maximum(pp*(1-pp)/(m+1.5),2e-8)
    p=np.zeros(len(x));v=np.zeros(len(x));wa=np.zeros(len(x));wp=np.zeros(len(x))
    held=np.zeros(len(x),dtype=int)
    for ix in x.groupby(["simulation_seed","node_id"],sort=False).indices.values():
        mean=float(np.clip(pc[ix[0]],1e-6,.45));var=max(float(vc[ix[0]]),.003**2)
        bias=0.;bias_var=.002**2;missing=0
        past_mean=[];past_passive=[];past_pvar=[];past_trusted=[]
        for j,i in enumerate(ix):
            mean=float(np.clip(mean,1e-6,.45))
            prior=min(.10**2,var+.0012**2*(1+min(missing,20)))
            a_ok=bool(av[i]) and j>=age[i]
            b_ok=bool(pv[i])
            if a_ok:
                t=j-age[i]
                # Compare only to passive observations of the same interval.
                # Bias is estimated from transport measurements, not ACK.
                match=bool(paired[i]) if age[i]==0 else past_trusted[t]
                other=pp[i] if age[i]==0 else past_passive[t]
                other_v=vp[i] if age[i]==0 else past_pvar[t]
                if reconcile and match:
                    diff=float(pa[i]-other)
                    innovation=diff-bias
                    noise=float(va[i]+other_v)
                    gain=min(.12,(bias_var+.00015**2)/(bias_var+.00015**2+noise))
                    bias=float(np.clip(bias+gain*np.clip(innovation,-3*np.sqrt(noise+bias_var),3*np.sqrt(noise+bias_var)),-.03,.03))
                    bias_var=max(1e-8,(1-gain)*(bias_var+.00015**2))
                obs=float(pa[i]-bias) if reconcile else float(pa[i])
                ar=float(va[i]+(bias_var if reconcile else 0.))
                if age[i]>0:
                    # Delayed observation is applied as an innovation about
                    # the stored contemporaneous prediction, not as a new
                    # observation of the current state. Uncertainty grows.
                    if delay_correct:
                        obs=mean+(obs-past_mean[t])
                    ar+=age[i]*(.0012**2+prior*.25)
            else:obs=0.;ar=1.
            # Store the prediction at this interval for future delayed data.
            past_mean.append(mean);past_passive.append(float(pp[i]))
            past_pvar.append(float(vp[i]));past_trusted.append(bool(paired[i]))
            if not(a_ok or b_ok):
                missing+=1
                # CU frequency is a weak, correlated fallback, not an
                # independent observation each interval.
                gain=min(.06,prior/(prior+vc[i]+.005**2)) if valid[i] else 0.
                delta=float(pc[i]-mean)
                mean+=gain*np.clip(delta,-.003,.003)
                var=min(.10**2,prior+gain*delta*delta)
            else:
                missing=0
                br=float(vp[i])*(1. if paired[i] and reconcile else 4.)
                if a_ok and b_ok:
                    diff=float(obs-pp[i])
                    if abs(diff)>3*np.sqrt(ar+br):
                        # Attribution is possible only for the matched
                        # receiver count. Otherwise downweight both sources.
                        if paired[i] and reconcile:ar+=diff*diff
                        else:ar+=.5*diff*diff;br+=.5*diff*diff
                sa=1/max(ar,1e-9) if a_ok else 0.
                sb=1/max(br,1e-9) if b_ok else 0.
                # Modest covariance inflation; current passive count is
                # reused in the physical residual, not treated as independent.
                den=1/prior+.8*(sa+sb)
                ka=.8*sa/den;kb=.8*sb/den
                mean+=ka*(obs-mean)+kb*(pp[i]-mean)
                var=max(1/den,1e-9)
                if a_ok and b_ok:var+=ka*kb*(obs-pp[i])**2
                wa[i]=ka;wp[i]=kb
            p[i]=np.clip(mean,1e-6,.45);v[i]=var;held[i]=missing
    assert np.isfinite(p).all() and np.isfinite(v).all() and (v>=0).all()
    return p,v,av|pv,wa,wp,held

def regime(x):
    av=(x.active_sent.to_numpy()>0)&(x.active_age.to_numpy()>=0)&(x.active_age.to_numpy()<=3)
    pv=x.passive_sent.to_numpy()>0
    return av.astype(int)+2*pv.astype(int)+4*(av&(x.active_age.to_numpy()>0)).astype(int)

class VarianceCalibration:
    """Simulation-only calibration, fitted on independent clean runs.
    Frozen factors are applied using observed availability/age only.
    No true transport probability is needed at inference.
    """
    def __init__(self):self.factors={};self.pooled=1.
    def fit(self,items):
        values=[];groups=[]
        for x,truth in items:
            good=x.primary_observer_valid.to_numpy(dtype=bool)
            r=(x.measured_oneway_loss.to_numpy()-truth)**2/np.maximum(x.measured_loss_variance.to_numpy(),1e-9)
            values.append(r[good]);groups.append(regime(x)[good])
        z=np.concatenate(values);g=np.concatenate(groups)
        # Moment calibration, bounded conservatively; not a confidence
        # interval guarantee. Validate empirical coverage separately.
        self.pooled=float(np.clip(z.mean(),.25,4.))
        for k in np.unique(g):
            a=z[g==k]
            self.factors[int(k)]=float(np.clip((a.sum()+1000*self.pooled)/(len(a)+1000),.25,4.))
        return self
    def apply(self,x,v):
        return v*np.array([self.factors.get(int(k),self.pooled) for k in regime(x)])
    def manifest(self):return {"pooled":self.pooled,"factors":self.factors,"simulation_truth_calibration_only":True}
