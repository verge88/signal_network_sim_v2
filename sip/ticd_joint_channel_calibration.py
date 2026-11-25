"""Joint empirical calibration of existing TICD evidence streams.

No new oracle features, no independence assumption, no operational
baseline updates. All original channels remain positively weighted.
"""
from __future__ import annotations
import numpy as np

NAMES=("core","count","conditional","loss","sequence")
BLOCKS=("none","plain","context","supported","plain_supported")
WIDTHS=(16,32,48)
PROFILES={
    "balanced":(1.,1.,1.,1.,1.,1.),
    "weak_physical":(.8,.8,1.15,1.4,1.35,1.),
    "instant":(1.45,1.25,.85,.95,1.,1.),
    "conditional":(.85,.85,1.5,1.1,1.,1.),
    "sequence":(.85,.85,1.,1.,1.5,1.25),
    "legacy_fast":(1.,1.,1.,1.,.7,.7),
}
OPERATING=((.004,1.),(.006,1.),(.008,1.),(.006,.95))
RULES=("max","support")

class JointReference:
    def __init__(self,base,old):
        self.columns={k:base.columns()[k] for k in NAMES[:-1]}
        self.columns["sequence"]=old.sequence
        self.scales={k:float(base.thresholds[k]) for k in NAMES[:-1]}
        self.scales["sequence"]=float(old.thresholds["sequence"])
        assert all(np.isfinite(v) and v>0 for v in self.scales.values())
        self.block_scales={}
    def fit(self,normal):
        assert all(not x.is_anomaly.any() for x in normal.values())
        runs=[g for x in normal.values() for _,g in x.groupby("simulation_seed")]
        for kind in BLOCKS[1:]:
            for w in WIDTHS:
                col=f"block_{kind}_{w}"
                # Fixed robust reference scale; final combined score gets
                # a separate empirical joint threshold on fresh clean runs.
                q=[float(np.quantile(g[col],.995,method="higher")) for g in runs]
                self.block_scales[col]=max(float(np.median(q)),.1)
        return self
    def prepare(self,x):
        out=x.copy()
        for k in NAMES:
            out["joint_"+k]=np.maximum(x[self.columns[k]].to_numpy(dtype=float),0.)/self.scales[k]
        for kind in BLOCKS[1:]:
            out["joint_block_"+kind]=np.maximum.reduce([
                np.maximum(x[f"block_{kind}_{w}"].to_numpy(dtype=float),0.)/
                self.block_scales[f"block_{kind}_{w}"] for w in WIDTHS])
        return out
    def manifest(self):
        return {"columns":self.columns,"scales":self.scales,"block_scales":self.block_scales,
                "block_widths":WIDTHS,"all_original_channels_positive":True}

class JointFusion:
    def __init__(self,profile,block,rule,budget,run_quantile,warning_threshold):
        assert profile in PROFILES and block in BLOCKS and rule in RULES
        assert budget>0 and 0<run_quantile<=1
        self.profile=profile;self.block=block;self.rule=rule
        self.budget=budget;self.run_quantile=run_quantile
        self.warning_threshold=float(warning_threshold)
        self.threshold=None;self.hybrid=False;self.observation_mode="adaptive_fast"
    def columns(self):
        return ["joint_"+k for k in NAMES]+([] if self.block=="none" else ["joint_block_"+self.block])
    def components(self,x):
        weights=PROFILES[self.profile][:len(self.columns())]
        return np.column_stack([x[c].to_numpy(dtype=float)*w for c,w in zip(self.columns(),weights)])
    def score(self,x):
        values=self.components(x)
        high=values.max(axis=1)
        if self.rule=="max":return high
        second=np.partition(values,-2,axis=1)[:,-2]
        # Every original channel can still independently trigger an alert.
        # The second largest score supplies bounded corroboration. The
        # correlated scores are NOT treated as independent probabilities.
        return high+.25*second
    def calibrate(self,runs):
        assert all(not x.is_anomaly.any() for x in runs)
        tails=[float(np.quantile(self.score(g),1-self.budget,method="higher")) for g in runs]
        self.threshold=max(1e-9,float(np.quantile(tails,self.run_quantile,method="higher")))
        return self
    def strong(self,x):
        return x.primary_observer_valid.to_numpy(dtype=bool)&(self.score(x)>self.threshold)
    def predict_levels(self,x):
        valid=x.primary_observer_valid.to_numpy(dtype=bool)
        a=self.strong(x)
        w=valid&x.split_eligible.to_numpy(dtype=bool)&(x.warn_0.to_numpy()>self.warning_threshold)&~a
        z=np.zeros(len(x),dtype=np.int8);z[w]=1;z[a]=2
        return z
    def predict_masks(self,x):
        z=self.predict_levels(x)
        return {"normal":z==0,"warning_only":z==1,"confirmed_alert":z==2,"any_signal":z>0}
    def channels(self,x):
        return {c:v>self.threshold for c,v in zip(self.columns(),self.components(x).T)}
    def manifest(self):
        return {"profile":self.profile,"weights":PROFILES[self.profile],"block":self.block,"rule":self.rule,
                "budget":self.budget,"run_quantile":self.run_quantile,"threshold":self.threshold,
                "warning_threshold":self.warning_threshold,"effective_channel_thresholds":{
                    c:float(self.threshold/w) for c,w in zip(self.columns(),PROFILES[self.profile])},
                "all_original_invariants_preserved":True,"joint_empirical_calibration":True,
                "independence_assumed":False,"no_operational_reference_update":True}

def warning_bound(runs,budget=.001):
    # Warning calibration does not depend on which rows get strong alerts.
    return max(float(np.quantile(np.where(g.primary_observer_valid.to_numpy(dtype=bool)&
          g.split_eligible.to_numpy(dtype=bool),g.warn_0.to_numpy(),0.),1-budget,method="higher")) for g in runs)
