"""Factorial TICD ablation and independent policy-observation channel.

The new channel assumes a separately protected approved-state registry and
a collector of actual actions. It consumes sampled counts of matched and
unmatched actions, not compromise labels. The synthetic evaluator models
that assumption explicitly; this is not field validation or a claim that
a cryptographic signature proves host integrity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ticd_invariant_ablation import InvariantAblationTICD

FACTORS=(("I1","I2","D"),("I1","I2"),("I1","D"),("I2","D"),("I1",),("I2",),("D",),())

class FactorialTICD(InvariantAblationTICD):
    def __init__(self,source,factors,refit=False):
        self.factors=tuple(factors)
        if self.factors not in FACTORS:raise ValueError(factors)
        super().__init__(source,"full" if factors else "no_all_invariants",refit=refit)
    def transform(self,df):
        if self.factors==FACTORS[0]:return df
        # These are already prepared evidence features; no change to the
        # simulator, no recomputation from hidden traffic or attack labels.
        x=df.copy()
        if "I1" not in self.factors:
            x["obs_dialog_completion_ratio"]=.5
            x["obs_dangling_dialog_ratio"]=.5
            x["mean_txn_state_consistency_error"]=0.
            x["dev_txn_consistency_vs_zone"]=0.
            x["cu_txn_state_consistency_error"]=0.
        if "I2" not in self.factors:
            x["cu_ack_closure_mismatch"]=0.
            if "dev_ack_mismatch_vs_zone" in x:x["dev_ack_mismatch_vs_zone"]=0.
        if "D" not in self.factors:
            for c in ("master_ack_shortfall_adjusted","dev_master_ack_shortfall_vs_zone",
                      "ack_shortfall_ewma_6","ack_shortfall_ewma_16","ack_shortfall_ewma_32","zone_causal_median"):
                x[c]=0.
        return x

def prepare_policy_history(df):
    """Streaming 6-sample causal ratio of observed unauthorized actions.
    Missing audit intervals contribute neither observations nor violations.
    No policy score is emitted for an unavailable current audit.
    """
    x=df.copy()
    keys=["simulation_seed","node_id"]
    if x.duplicated(keys+["interval"]).any():raise ValueError("duplicate audit record")
    x=x.sort_values(keys+["interval"],kind="stable")
    n=x["audit_observed_count"].to_numpy(dtype=float)
    k=x["audit_mismatch_count"].to_numpy(dtype=float)
    if not(np.isfinite(n).all() and np.isfinite(k).all() and (0<=k).all() and (k<=n).all()):
        raise ValueError("invalid independent audit counts")
    if (n<0).any():raise ValueError("invalid audit denominator")
    for col in ("audit_observed_count","audit_mismatch_count"):
        x[col+"_ewma"]=x.groupby(keys,sort=False)[col].transform(lambda v:v.ewm(span=6,adjust=False).mean())
    denom=x["audit_observed_count_ewma"].to_numpy()
    x["audit_ratio"]=np.divide(x["audit_mismatch_count_ewma"],denom,out=np.zeros(len(x)),where=denom>0)
    return x.sort_index()

class PolicyEvidence:
    def __init__(self):self.refs={};self.threshold=None
    def fit(self,train,val,budget=.008):
        assert not train.is_anomaly.any()
        self.refs={}
        for role,g in train.groupby("node_type"):
            a=g.loc[g.audit_observed_count>0,"audit_ratio"].to_numpy()
            if len(a)<100:raise ValueError("insufficient policy reference for "+str(role))
            self.refs[str(role)]=max(float(np.quantile(a,.99,method="higher")),1e-3)
        self.threshold=max(float(np.quantile(self.score(x),1-budget,method="higher")) for x in val.values())
        assert self.threshold>0
        return self
    def score(self,df):
        q=np.array([self.refs[str(r)] for r in df.node_type])
        a=df.audit_ratio.to_numpy(dtype=float)/q
        assert np.isfinite(a).all()
        return np.where(df.audit_observed_count.to_numpy()>0,a,0.)
    def status(self,df):
        return self.score(df)>self.threshold

class MultiObserverTICD:
    """Primary SIP + independent policy audit. Missing evidence is not normal.
    Levels follow existing API; partial/unobserved flags must be reported
    separately, not counted as security alerts or known benign states.
    """
    def __init__(self,core,audit):
        self.core=core;self.audit=audit;self.threshold=None
        assert core.threshold>0 and audit.threshold>0
    def components(self,df):
        valid=df.primary_observer_valid.to_numpy(dtype=bool)
        a=np.zeros(len(df),dtype=float)
        if valid.any():a[valid]=self.core.score(df.loc[valid])/self.core.threshold
        b=self.audit.score(df)/self.audit.threshold
        return a,b
    def score(self,df):
        a,b=self.components(df);return np.maximum(a,b)
    def calibrate(self,val,budget=.008):
        assert all(not x.is_anomaly.any() for x in val.values())
        self.threshold=max(float(np.quantile(self.score(x),1-budget,method="higher")) for x in val.values())
        assert self.threshold>0
        return self
    def predict_levels(self,df):
        strong=self.score(df)>self.threshold
        valid=df.primary_observer_valid.to_numpy(dtype=bool)
        warn=np.zeros(len(df),dtype=bool)
        if valid.any():
            w,e=self.core.warning(df.loc[valid])
            warn[valid]=e&(w>self.core.warning_threshold)
        warn &= ~strong
        z=np.zeros(len(df),dtype=np.int8);z[warn]=1;z[strong]=2
        return z
    def predict_masks(self,df):
        z=self.predict_levels(df)
        return {"normal":z==0,"warning_only":z==1,"confirmed_alert":z==2,"any_signal":z>0}
    def observation_masks(self,df):
        p=df.primary_observer_valid.to_numpy(dtype=bool)
        q=df.audit_observed_count.to_numpy()>0
        return {"complete":p&q,"partial":p^q,"unobservable":~(p|q)}
