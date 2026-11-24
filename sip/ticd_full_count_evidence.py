"""Add count-aware causal physical evidence to the FULL TICD.

Counts are the existing transport observations, not new oracle features.
Signed residuals are centered on a benign role/loss reference. Aggregation
preserves the denominator, so a few dialogs are not treated like hundreds.
All I1/I2/D evidence and the legacy warning branch are retained.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WINDOWS=(16,48,96)
ROLES=("PROXY","REGISTRAR","UA_GW","REDIRECT")

class CountReference:
    def __init__(self):
        self.edges=None;self.p={};self.q={};self.fitted=False
    @staticmethod
    def counts(df):
        n=df.cu_n_200_transit.to_numpy(dtype=float)
        a=df.cu_n_ack_transit.to_numpy(dtype=float)
        available=df.primary_observer_valid.to_numpy(dtype=bool)&(df.cu_delivered.to_numpy()>0)
        if not (np.isfinite(n).all() and np.isfinite(a).all() and (n>=a).all() and (a>=0).all()):
            raise ValueError("Invalid independently observed transit counts")
        return np.where(available,n,0.),np.where(available,n-a,0.)
    def groups(self,df):
        loss=df.cu_combined_loss_prob.to_numpy(dtype=float)
        role=df.node_type.astype(str).to_numpy()
        if not np.isfinite(loss).all() or not set(role)<=set(ROLES):raise ValueError("invalid inventory/loss")
        # Only current measured CU loss, no scenario multiplier.
        b=np.searchsorted(self.edges,loss,side="right")
        return np.array([r+"|"+str(k) for r,k in zip(role,b)])
    def fit(self,train):
        assert not train.is_anomaly.any()
        n,k=self.counts(train)
        self.edges=np.unique(np.quantile(train.cu_combined_loss_prob,[.2,.4,.6,.8]))
        groups=self.groups(train)
        roles=train.node_type.astype(str).to_numpy()
        self.p={}
        for r in ROLES:
            ix=roles==r
            self.p[r]=float((k[ix].sum()+.5)/(n[ix].sum()+1.))
        for g in np.unique(groups):
            ix=groups==g;r=g.split("|")[0]
            # Shrink sparse group estimates toward role-specific normal rate.
            self.p[g]=float((k[ix].sum()+500*self.p[r])/(n[ix].sum()+500))
        prepared=self.prepare(train,normalize=False)
        self.q={}
        for w in WINDOWS:
            z=prepared["count_raw_"+str(w)].to_numpy()
            for r in ROLES:
                self.q[(w,r)]=max(float(np.quantile(z[roles==r],.99,method="higher")),1.)
            for g in np.unique(groups):
                ix=groups==g
                if ix.sum()>=400:self.q[(w,g)]=max(float(np.quantile(z[ix],.99,method="higher")),1.)
        self.fitted=True
        return self
    def prepare(self,df,normalize=True):
        if normalize and not self.fitted:raise RuntimeError("Unfitted count reference")
        x=df.copy()
        keys=["simulation_seed","node_id"]
        if x.duplicated(keys+["interval"]).any():raise ValueError("duplicate observation")
        x=x.sort_values(keys+["interval"],kind="stable")
        # Refuse gaps in input preparation: prepare complete causal runs.
        gap=x.groupby(keys,sort=False).interval.diff().dropna()
        if not (gap==1).all():raise ValueError("Incomplete time sequence")
        n,k=self.counts(x)
        groups=self.groups(x);roles=x.node_type.astype(str).to_numpy()
        p=np.clip(np.array([self.p.get(g,self.p[r]) for g,r in zip(groups,roles)]),1e-5,1-1e-5)
        # Do not discard negative residuals, which would create a positive
        # drift under benign conditions. Binomial variance is only a scale;
        # final empirical calibration accommodates correlated observations.
        x["_cnt_r"]=k-n*p
        x["_cnt_v"]=n*p*(1-p)
        x["_cnt_n"]=n
        for w in WINDOWS:
            a=x.groupby(keys,sort=False)["_cnt_r"].transform(lambda v:v.rolling(w,min_periods=1).sum()).to_numpy()
            v=x.groupby(keys,sort=False)["_cnt_v"].transform(lambda v:v.rolling(w,min_periods=1).sum()).to_numpy()
            total=x.groupby(keys,sort=False)["_cnt_n"].transform(lambda v:v.rolling(w,min_periods=1).sum()).to_numpy()
            z=np.where((n>0)&(total>=30),np.maximum(a,0.)/np.sqrt(np.maximum(v,1.)),0.)
            x["count_raw_"+str(w)]=z
            if normalize:
                q=np.array([self.q.get((w,g),self.q[(w,r)]) for g,r in zip(groups,roles)])
                x["count_norm_"+str(w)]=z/q
        return x.drop(columns=["_cnt_r","_cnt_v","_cnt_n"]).sort_index()
    def manifest(self):
        return {"edges":self.edges.tolist(),"rates":self.p,
                "q99":{str(w)+"|"+r:v for (w,r),v in self.q.items()},
                "windows":list(WINDOWS),"min_accumulated_dialogs":30}

class FullCountTICD:
    def __init__(self,core,windows=(),weight=0.,audit=None):
        assert core.variant=="full" and core.threshold>0
        assert weight>=0 and (not windows or weight>0)
        self.core=core;self.windows=tuple(windows);self.weight=float(weight);self.audit=audit
        self.threshold=None;self.warning_threshold=None
    def components(self,df):
        valid=df.primary_observer_valid.to_numpy(dtype=bool)
        core=np.zeros(len(df));count=np.zeros(len(df))
        if valid.any():
            core[valid]=self.core.score(df.loc[valid])/self.core.threshold
            if self.windows:
                count[valid]=self.weight*np.max(np.column_stack([df.loc[valid,"count_norm_"+str(w)].to_numpy() for w in self.windows]),axis=1)
        policy=np.zeros(len(df)) if self.audit is None else self.audit.score(df)/self.audit.threshold
        return core,count,policy
    def score(self,df):
        return np.maximum.reduce(self.components(df))
    def warning(self,df):
        valid=df.primary_observer_valid.to_numpy(dtype=bool)
        w=np.zeros(len(df));e=np.zeros(len(df),dtype=bool)
        if valid.any():w[valid],e[valid]=self.core.warning(df.loc[valid])
        return w,e
    def calibrate(self,val,budget=.008,warning_budget=.005):
        assert all(not x.is_anomaly.any() for x in val.values())
        self.threshold=max(float(np.quantile(self.score(x),1-budget,method="higher")) for x in val.values())
        qs=[]
        for x in val.values():
            a=self.score(x)>self.threshold
            w,e=self.warning(x);qs.append(float(np.quantile(np.where(e&~a,w,0.),1-warning_budget,method="higher")))
        self.warning_threshold=max(qs)
        assert self.threshold>0
        for x in val.values():
            z=self.predict_levels(x)
            assert (z==2).mean()<=budget+1e-12 and (z==1).mean()<=warning_budget+1e-12
        return self
    def predict_levels(self,df):
        strong=self.score(df)>self.threshold
        w,e=self.warning(df);warn=e&(w>self.warning_threshold)&~strong
        z=np.zeros(len(df),dtype=np.int8);z[warn]=1;z[strong]=2
        return z
    def predict_masks(self,df):
        z=self.predict_levels(df)
        return {"normal":z==0,"warning_only":z==1,"confirmed_alert":z==2,"any_signal":z>0}
    def manifest(self):
        return {"all_original_invariants_preserved":True,"windows":self.windows,"count_weight":self.weight,
                "audit":self.audit is not None,"threshold":self.threshold,"warning_threshold":self.warning_threshold}
