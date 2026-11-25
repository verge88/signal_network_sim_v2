"""Additional causal evidence on signed measured physical ACK deficits.

Frozen benign commissioning, no operational baseline updates or oracle inputs.
The reliability factor is heuristic; empirical run-wise calibration is required.
"""
from __future__ import annotations
import numpy as np
from ticd_measured_loss import MeasuredFusion, measured_frame

SPECS=((.98,.10),(.995,.05),(.998,.025))
KEYS=["simulation_seed","node_id"]

def signed(ref, x):
    z,n,e,v,good,av,g,r=ref.inputs(x,"cu")
    bias=np.array([ref.center.get(("cu",k),ref.center[("cu",role)]) for k,role in zip(g,r)])
    residual=np.clip(e,-4.,4.)-bias
    p=z.measured_oneway_loss.to_numpy(dtype=float)
    # Fraction of the variance scale attributable to actual count noise,
    # rather than uncertainty in the transport measurement.
    measurement=4*n*n*z.measured_loss_variance.to_numpy(dtype=float)
    reliability=np.sqrt(np.clip(1-measurement/np.maximum(v,1.),.05,1.))
    usable=good & z.primary_observer_valid.to_numpy(dtype=bool)
    return z,n,residual,reliability,usable,p

def sequence_features(x, commission, reference):
    assert reference.mode!="oracle"
    ref=reference.reference
    c=reference.frame(commission,True)
    cz,cn,ce,cw,cgood,cp=signed(ref,c)
    z,n,e,w,good,p=signed(ref,x)
    # Per-installation, per-node benign offset with shrinkage toward the
    # frozen population reference. Do not use the operational sequence.
    profiles={}
    for key,ix in cz.groupby(KEYS,sort=False).indices.items():
        ix=ix[cgood[ix]]
        if len(ix)<80:continue
        weight=cw[ix]
        center=float(np.clip(np.sum(weight*ce[ix])/(weight.sum()+80),-.75,.75))
        profiles[key]=center
    out=x.copy()
    for j,(decay,drift) in enumerate(SPECS):
        for weighted in (False,True):
            scores=np.zeros(len(z))
            for key,ix in z.groupby(KEYS,sort=False).indices.items():
                state=0.
                offset=profiles.get(key,0.)
                for i in ix:
                    if good[i]:
                        factor=w[i] if weighted else 1.
                        state=max(0.,decay*state+factor*(e[i]-offset-drift))
                        scores[i]=state
                    else:
                        state*=decay
            col=f"phys_seq_{j}_{int(weighted)}"
            out[col]=0.
            out.loc[z.index,col]=scores
    return out

class SequentialPhysicalFusion(MeasuredFusion):
    def __init__(self,mode,window,budgets,sequence):
        assert len(budgets)==6 and budgets[4]==0 and sum(budgets)<=.008+1e-12
        super().__init__(mode,window,budgets[:5],"worst_run")
        self.budgets=tuple(budgets)
        self.sequence=sequence
    def columns(self):
        return {**super().columns(),"sequence":self.sequence}
    def manifest(self):
        return {**super().manifest(),"sequence":self.sequence,
                "causal_signed_physical_residual":True,
                "commissioned_offset":True,"no_operational_reference_update":True}
