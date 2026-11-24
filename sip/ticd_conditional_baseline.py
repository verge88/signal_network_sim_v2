"""Conditional benign count reference and full TICD fusion.

No simulator loss parameters or attacker-controlled report fields enter
the new reference. CU loss is still an imperfect transport proxy.
All previous I1/I2/D, trimmed S48 and fast sequence scores are retained.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from ticd_full_count_evidence import ROLES
from ticd_separate_robust_fusion import upper_reference

WINDOWS=(48,96,144)
MODES=("baseline","cond48","cond96","cond144","condmulti")
SCOPES=("global","role")

class ConditionalCountReference:
    def __init__(self,previous):
        self.previous=previous;self.regressors={};self.center={};self.scale={};self.q={}
        self.volume_edges={};self.loss_edges=None;self.channel_q={};self.fitted=False
    @property
    def prior(self):return self.previous.bounded.prior
    def inputs(self,df):
        x=df.sort_values(["simulation_seed","node_id","interval"],kind="stable").copy()
        keys=["simulation_seed","node_id"]
        assert not x.duplicated(keys+["interval"]).any()
        assert (x.groupby(keys,sort=False).interval.diff().dropna()==1).all()
        n,k=self.prior.counts(x)
        loss=x.cu_combined_loss_prob.to_numpy(dtype=float)
        role=x.node_type.astype(str).to_numpy()
        if not np.isfinite(loss).all() or not set(role)<=set(ROLES):raise ValueError("invalid conditional features")
        features=np.column_stack((loss,np.log1p(n)))
        return x,n,k,loss,role,features
    def groups(self,n,loss,role):
        lb=np.searchsorted(self.loss_edges,loss,side="right")
        return np.array([r+"|"+str(b)+"|"+str(np.searchsorted(self.volume_edges[r],v,side="right"))
                         for v,b,r in zip(n,lb,role)])
    def prediction(self,n,role,features):
        p=np.zeros(len(n))
        for r in ROLES:
            mask=role==r
            if mask.any():p[mask]=self.regressors[r].predict(features[mask])
        return np.clip(p,1e-5,1-1e-5)
    def fit(self,train,core,audit):
        assert not train.is_anomaly.any() and core.variant=="full"
        x,n,k,loss,role,f=self.inputs(train)
        available=n>0
        self.loss_edges=np.unique(np.quantile(loss[available],[.2,.4,.6,.8]))
        for r in ROLES:
            m=available&(role==r);assert m.sum()>=1000
            self.volume_edges[r]=np.unique(np.quantile(n[m],[.25,.5,.75]))
            mod=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=8,min_samples_leaf=200,
                  learning_rate=.06,l2_regularization=5.,early_stopping=False,random_state=1729)
            mod.fit(f[m],k[m]/n[m],sample_weight=n[m]/np.mean(n[m]))
            self.regressors[r]=mod
        p=self.prediction(n,role,f)
        z=(k-n*p)/np.sqrt(np.maximum(n*p*(1-p),1.))
        g=self.groups(n,loss,role)
        # Robust conditional mean and scale of observed standardized residuals.
        # Shrink group moments toward role; no use of scenario/attack labels.
        good=available&(np.abs(z)<=8.)
        bounded=np.clip(z,-4.,4.)
        for r in ROLES:
            a=bounded[good&(role==r)];assert len(a)>=400
            self.center[r]=float(a.mean());self.scale[r]=max(float(a.std()),.5)
        for key in np.unique(g):
            a=bounded[good&(g==key)];r=key.split("|")[0]
            if len(a)>=200:
                w=len(a)/(len(a)+500.)
                self.center[key]=float(w*a.mean()+(1-w)*self.center[r])
                self.scale[key]=max(float(np.sqrt(w*a.var()+(1-w)*self.scale[r]**2)),.5)
        raw=self.raw(train)
        r=raw.node_type.astype(str).to_numpy()
        for w in WINDOWS:
            c="conditional_raw_"+str(w);v=raw[c].to_numpy()
            for role_name in ROLES:
                self.q[(w,role_name)]=max(float(np.quantile(v[r==role_name],.99,method="higher")),1.)
        # Fit optional role-scale calibration from benign scores only.
        x=self.previous.prepare(train,core,audit)
        x=self.add(x)
        for c in ("core_0",*MODES):
            v=x[c].to_numpy();r=x.node_type.astype(str).to_numpy()
            for role_name in ROLES:
                self.channel_q[(c,role_name)]=max(float(np.quantile(v[r==role_name],.99,method="higher")),1e-3)
        self.fitted=True;return self
    def raw(self,df):
        x,n,k,loss,r,f=self.inputs(df)
        p=self.prediction(n,r,f);g=self.groups(n,loss,r)
        z=(k-n*p)/np.sqrt(np.maximum(n*p*(1-p),1.))
        center=np.array([self.center.get(a,self.center[b]) for a,b in zip(g,r)])
        scale=np.array([self.scale.get(a,self.scale[b]) for a,b in zip(g,r)])
        valid=(n>0)&(np.abs(z)<=8.)
        # Weight by modeled information; trim gross transients only in the
        # supplementary channel. Keep signed residuals and missing masks.
        v=np.maximum(n*p*(1-p),1.)
        e=(np.clip(z,-4.,4.)-center)/scale
        x["_cond_num"]=np.where(valid,e*np.sqrt(v),0.)
        x["_cond_var"]=np.where(valid,v,0.)
        x["_cond_n"]=np.where(valid,n,0.)
        for w in WINDOWS:
            t=x.groupby(["simulation_seed","node_id"],sort=False)[["_cond_num","_cond_var","_cond_n"]].transform(lambda a:a.rolling(w,min_periods=1).sum())
            x["conditional_raw_"+str(w)]=np.where(valid&(t._cond_n>=30),np.maximum(t._cond_num,0.)/np.sqrt(np.maximum(t._cond_var,1.)),0.)
        x["conditional_expected_rate"]=p
        return x.drop(columns=["_cond_num","_cond_var","_cond_n"]).sort_index()
    def add(self,prepared):
        x=self.raw(prepared)
        r=x.node_type.astype(str).to_numpy()
        x["baseline"]=x.count_0_fast
        for w in WINDOWS:
            v=x["conditional_raw_"+str(w)].to_numpy()
            q=np.array([self.q[(w,role)] for role in r])
            # Preserve all of the previously selected accumulation channel.
            x["cond"+str(w)]=np.maximum(x.baseline,v/q)
        x["condmulti"]=np.maximum.reduce([x["cond"+str(w)].to_numpy() for w in WINDOWS])
        return x
    def prepare(self,df,core,audit):
        assert self.fitted
        x=self.add(self.previous.prepare(df,core,audit))
        r=x.node_type.astype(str).to_numpy()
        for c in ("core_0",*MODES):
            q=np.array([self.channel_q[(c,role)] for role in r])
            x[c+"_role"]=x[c].to_numpy()/q
        return x
    def manifest(self):
        return {"windows":WINDOWS,"loss_edges":self.loss_edges.tolist(),
            "volume_edges":{r:v.tolist() for r,v in self.volume_edges.items()},
            "regression_features":["cu_combined_loss_prob","log1p(cu_n_200_transit)"],
            "role_models":list(self.regressors),"center":self.center,"scale":self.scale,
            "q99":{str(w)+"|"+r:v for (w,r),v in self.q.items()},
            "role_q99":{c+"|"+r:v for (c,r),v in self.channel_q.items()},
            "old_count_preserved":True,"actual_transport_loss_not_observed":True}

class ConditionalFusionTICD:
    def __init__(self,mode,scope,budgets,calibration_mode,warning_budget=.004):
        assert mode in MODES and scope in SCOPES
        assert len(budgets)==3 and min(budgets[:2])>0 and budgets[2]>=0 and sum(budgets)<=.008+1e-12
        self.mode=mode;self.scope=scope;self.budgets=tuple(budgets)
        self.calibration_mode=calibration_mode;self.warning_budget=warning_budget
        self.thresholds={};self.warning_threshold=None
    @property
    def hybrid(self):return self.budgets[2]>0
    def columns(self):
        suffix="_role" if self.scope=="role" else ""
        return {"core":"core_0"+suffix,"count":self.mode+suffix,"audit":"split_audit"}
    def channels(self,df):
        return {k:df[c].to_numpy()>self.thresholds[k] if self.thresholds[k] is not None else np.zeros(len(df),dtype=bool)
                for k,c in self.columns().items()}
    def strong(self,df):return np.logical_or.reduce(list(self.channels(df).values()))
    def calibrate(self,val):
        assert all(not x.is_anomaly.any() for x in val.values())
        def bound(c,b):
            if self.calibration_mode!="worst_run":return upper_reference(val,c,b,self.calibration_mode)
            return max(1e-9,max(float(np.quantile(g[c],1-b,method="higher"))
                  for x in val.values() for _,g in x.groupby("simulation_seed")))
        self.thresholds={k:bound(c,b) if b>0 else None for (k,c),b in zip(self.columns().items(),self.budgets)}
        self.warning_threshold=max(float(np.quantile(np.where(
              g.split_eligible.to_numpy(dtype=bool)&~self.strong(g),g.warn_0.to_numpy(),0.),
              1-self.warning_budget,method="higher"))
              for x in val.values() for _,g in x.groupby("simulation_seed"))
        for x in val.values():
            z=self.predict_levels(x)
            assert (z==2).mean()<=sum(self.budgets)+1e-12
            assert (z==1).mean()<=self.warning_budget+1e-12
        return self
    def predict_levels(self,df):
        a=self.strong(df)
        w=df.split_eligible.to_numpy(dtype=bool)&(df.warn_0.to_numpy()>self.warning_threshold)&~a
        z=np.zeros(len(df),dtype=np.int8);z[w]=1;z[a]=2
        return z
    def predict_masks(self,df):
        z=self.predict_levels(df)
        return {"normal":z==0,"warning_only":z==1,"confirmed_alert":z==2,"any_signal":z>0}
    def manifest(self):
        return {"all_original_invariants_preserved":True,"mode":self.mode,"scope":self.scope,
            "budgets":self.budgets,"calibration_mode":self.calibration_mode,
            "warning_budget":self.warning_budget,"thresholds":self.thresholds,
            "warning_threshold":self.warning_threshold,"audit_independent":self.hybrid}
