"""Role-adaptive persistent ACK score for SIP TICD strong alarms.

Frozen instantaneous and causal references from the prior experiment.
No role is excluded from strong alerts. UA_GW history remains available:
positive weight, or corroboration by the current physical score. Inventory
role must come from trusted configuration, not a reporting node.
No attack labels, simulator multipliers, or online fitting at inference.
"""
from __future__ import annotations
import numpy as np
from ticd_role_scoped_warning import KNOWN_ROLES

class RoleAdaptiveCausalAlertTICD:
    def __init__(self,prior,mode="uniform",gateway_weight=1.,gateway_gate=0.):
        if prior.mode!="max" or prior.reference is None or prior.threshold is None:
            raise ValueError("Require frozen max-fusion model")
        if mode not in ("uniform","weighted","gated","joint"):
            raise ValueError(mode)
        if not 0.<gateway_weight<=1. or not 0.<=gateway_gate<=1.:
            raise ValueError("Invalid gateway parameters")
        if mode in ("gated","joint") and gateway_gate<=0.:
            raise ValueError("Positive gate required")
        self.prior=prior
        self.mode=mode
        self.gateway_weight=float(gateway_weight)
        self.gateway_gate=float(gateway_gate)
        self.threshold=None
    def components(self,df,channels=None):
        role_model=self.prior.prior.prior
        if channels is None: channels=role_model.score_channels(df)
        a=self.prior.prior.normalized(df,channels["instant"])/self.prior.prior.scale
        h=self.prior.reference.score(df)*self.prior.weight
        roles=df["node_type"].astype(str).to_numpy()
        if not set(roles)<=KNOWN_ROLES: raise ValueError("Unknown inventory role")
        if not np.isfinite(a).all() or not np.isfinite(h).all():
            raise ValueError("Nonfinite evidence")
        gw=roles=="UA_GW"
        h=h.copy()
        if self.mode in ("weighted","joint"): h[gw]*=self.gateway_weight
        if self.mode in ("gated","joint"): h[gw]=np.minimum(h[gw],a[gw]/self.gateway_gate)
        return a,h
    def score(self,df,channels=None):
        a,h=self.components(df,channels)
        return np.maximum(a,h)
    def calibrate(self,normal,budget):
        if not 0.<budget<.01: raise ValueError("Invalid normal budget")
        if any((x["is_anomaly"]!=0).any() for x in normal.values()):
            raise ValueError("Nonbenign calibration")
        self.threshold=max(float(np.quantile(self.score(x),1-budget,method="higher"))
                           for x in normal.values())
        return self
    def predict_levels(self,df):
        if self.threshold is None: raise RuntimeError("Uncalibrated")
        role_model=self.prior.prior.prior
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
        return {"mode":self.mode,"gateway_weight":self.gateway_weight,
                "gateway_gate":self.gateway_gate,"threshold":self.threshold,
                "prior_references_frozen":True,"instant_all_roles":True,
                "history_not_disabled_for_any_role":True,
                "warning_reference_frozen":True,"inventory_role_required":True}
