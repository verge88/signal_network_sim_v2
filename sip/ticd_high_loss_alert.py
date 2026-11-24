"""Frozen SIP TICD high-loss strong-alert calibration.

Only the normal-reference decision boundary is changed. All network roles
retain the strong-alert channel, including UA_GW. Warning score, threshold,
and eligible-role inventory are frozen from the preceding experiment.
The master CU loss estimate is an imperfect proxy for SIP transit loss.
No online adaptation, attack labels, or simulator loss multipliers enter
decision scores. A demoted strong alert may become a warning, so evaluate
both tiers and their union separately.
"""
from __future__ import annotations
import numpy as np

class HighLossAlertTICD:
    def __init__(self,prior,mode="fixed",bins=6,quantile=.993):
        if not prior.fitted: raise ValueError("Require frozen preceding model")
        if mode not in ("fixed","loss","loss_role"): raise ValueError(mode)
        self.prior=prior
        self.mode=mode
        self.bins=int(bins)
        self.quantile=float(quantile)
        self.fitted=False
        self.edges=np.array([],dtype=float)
        self.thresholds={}
        self.group_counts={}
        self.scale=1.
    @staticmethod
    def loss(df):
        x=df["cu_combined_loss_prob"].to_numpy(dtype=float)
        if not np.isfinite(x).all() or (x<0).any() or (x>1).any():
            raise ValueError("Invalid master CU loss probability")
        return x
    def groups(self,df):
        idx=np.searchsorted(self.edges,self.loss(df),side="right")
        if self.mode=="loss_role":
            roles=df["node_type"].astype(str).to_numpy()
            return np.array([str(r)+"|"+str(b) for r,b in zip(roles,idx)])
        return np.array([str(b) for b in idx])
    def fit(self,normal):
        if (normal["is_anomaly"]!=0).any(): raise ValueError("Nonbenign reference")
        x=self.loss(normal)
        a=self.prior.score_channels(normal)["instant"]
        if self.mode!="fixed":
            self.edges=np.unique(np.quantile(x,np.linspace(0,1,self.bins+1)[1:-1]))
        self.fallback=max(float(np.quantile(a,self.quantile,method="higher")),1e-6)
        groups=self.groups(normal)
        for key in np.unique(groups):
            rows=a[groups==key]
            self.group_counts[str(key)]=len(rows)
            if len(rows)>=400:
                self.thresholds[str(key)]=max(float(np.quantile(rows,self.quantile,method="higher")),1e-6)
        self.fitted=True
        return self
    def reference(self,df):
        if not self.fitted: raise RuntimeError("Not fitted")
        return np.array([self.thresholds.get(str(k),self.fallback) for k in self.groups(df)])
    def normalized(self,df,instant=None):
        # Use the identical floating-point operation for calibration and inference.
        # Comparing A > lambda*q can round differently at an exact quantile tie.
        if instant is None:
            instant=self.prior.score_channels(df)["instant"]
        return np.asarray(instant,dtype=float)/self.reference(df)
    def calibrate(self,validation,budget):
        self.scale=max(float(np.quantile(self.normalized(df),1.-budget,method="higher"))
                       for df in validation.values())
        return self
    def predict_levels(self,df):
        channels=self.prior.score_channels(df)
        a=self.normalized(df,channels["instant"])>self.scale
        w=(channels["warning"]>self.prior.warning_threshold)&self.prior.eligible_mask(df)&(~a)
        levels=np.zeros(len(df),dtype=np.int8)
        levels[w]=1
        levels[a]=2
        return levels
    def predict_masks(self,df):
        x=self.predict_levels(df)
        return {"normal":x==0,"warning_only":x==1,"confirmed_alert":x==2,"any_signal":x>=1}
    def manifest(self):
        return {"mode":self.mode,"edges":self.edges.tolist(),"thresholds":self.thresholds,
                "fallback":self.fallback,"group_counts":self.group_counts,"scale":self.scale,
                "all_roles_strong_alert":True,"warning_reference_frozen":True}
