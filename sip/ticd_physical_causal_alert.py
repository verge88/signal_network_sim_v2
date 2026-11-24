"""SIP TICD: normal-calibrated fusion of instant and causal physical evidence.

No attack labels or true simulated loss multipliers enter inference.
The existing per-node EWMA of master-observed ACK closure deficit is
causal and must be prepared before scoring. Its new conditional benign
quantiles are fitted on fresh clean traffic, including high loss.
Both channels share underlying evidence; fusion is not proof of independent
corroboration. All SIP roles retain strong alarms. The previous warning
score, threshold and role scope remain frozen.
"""
from __future__ import annotations
import numpy as np
from ticd_causal_ack import HISTORY_FIELDS

class CausalPhysicalReference:
    def __init__(self,prior,history):
        if history not in HISTORY_FIELDS: raise ValueError(history)
        self.prior=prior
        self.history=history
        self.column=HISTORY_FIELDS[history]
        self.groups_q={}
        self.counts={}
        self.fitted=False
    def raw(self,df):
        x=df[self.column].to_numpy(dtype=float)
        if not np.isfinite(x).all() or (x<0).any(): raise ValueError("Invalid master history")
        return x
    def fit(self,df):
        if (df["is_anomaly"]!=0).any(): raise ValueError("Nonbenign reference")
        x=self.raw(df)
        groups=self.prior.groups(df)
        self.fallback=max(float(np.quantile(x,.993,method="higher")),1e-3)
        for key in np.unique(groups):
            v=x[groups==key]
            self.counts[str(key)]=len(v)
            if len(v)>=400:
                self.groups_q[str(key)]=max(float(np.quantile(v,.993,method="higher")),1e-3)
        self.fitted=True
        return self
    def score(self,df):
        if not self.fitted: raise RuntimeError("Unfitted causal reference")
        q=np.array([self.groups_q.get(str(g),self.fallback) for g in self.prior.groups(df)])
        return self.raw(df)/q
    def manifest(self):
        return {"history":self.history,"column":self.column,"conditional_q993":self.groups_q,
                "fallback":self.fallback,"counts":self.counts}

class PhysicalCausalAlertTICD:
    def __init__(self,prior,reference=None,mode="instant",weight=1.):
        if not prior.fitted: raise ValueError("Frozen prior required")
        if mode not in ("instant","max","gated","blend"): raise ValueError(mode)
        if mode!="instant" and (reference is None or not reference.fitted): raise ValueError("Need fitted reference")
        self.prior=prior
        self.reference=reference
        self.mode=mode
        self.weight=float(weight)
        self.threshold=None
    def score(self,df,channels=None):
        if channels is None: channels=self.prior.prior.score_channels(df)
        instant=self.prior.normalized(df,channels["instant"])/self.prior.scale
        if self.mode=="instant": return instant
        hist=self.reference.score(df)*self.weight
        if self.mode=="max": return np.maximum(instant,hist)
        if self.mode=="gated":
            return np.maximum(instant,np.minimum(hist,instant/.5))
        # Persistent deficit supplements moderately elevated instantaneous evidence.
        return np.maximum(instant,.5*instant+.5*hist)
    def calibrate(self,normal,budget):
        if not 0<budget<.01: raise ValueError("Invalid FPR target")
        if any((x["is_anomaly"]!=0).any() for x in normal.values()):
            raise ValueError("Contaminated validation")
        self.threshold=max(float(np.quantile(self.score(x),1-budget,method="higher"))
                           for x in normal.values())
        return self
    def predict_levels(self,df):
        if self.threshold is None: raise RuntimeError("Uncalibrated")
        role_model=self.prior.prior
        channels=role_model.score_channels(df)
        a=self.score(df,channels)>self.threshold
        w=(channels["warning"]>role_model.warning_threshold)&role_model.eligible_mask(df)&(~a)
        out=np.zeros(len(df),dtype=np.int8)
        out[w]=1;out[a]=2
        return out
    def predict_masks(self,df):
        x=self.predict_levels(df)
        return {"normal":x==0,"warning_only":x==1,"confirmed_alert":x==2,"any_signal":x>=1}
    def manifest(self):
        return {"mode":self.mode,"weight":self.weight,"threshold":self.threshold,
                "reference":None if self.reference is None else self.reference.manifest(),
                "base_high_loss_reference_frozen":True,"all_roles_strong_alert":True,
                "warning_reference_frozen":True,"correlated_evidence_not_independent_confirmation":True}
