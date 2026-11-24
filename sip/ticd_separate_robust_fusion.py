"""Full TICD with independently calibrated channels and bounded count memory.

The original I1/I2/D score and role-scoped warning remain active. The audit
threshold is never multiplied by the threshold of a different channel.
Robust count variants bound single-interval influence, not the score of
the original detector. All references are learned on benign data only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ticd_full_count_evidence import ROLES

MODES=("raw48","cap3_48","cap6_48","trim8_48","cap3_96","cap3_multi")
ROBUST=(("cap3",3.,False),("cap6",6.,False),("trim8",8.,True))

class BoundedCountReference:
    def __init__(self,prior):
        self.prior=prior;self.q={};self.fitted=False
    def raw(self,df):
        x=df.copy().sort_values(["simulation_seed","node_id","interval"],kind="stable")
        keys=["simulation_seed","node_id"]
        assert not x.duplicated(keys+["interval"]).any()
        assert (x.groupby(keys,sort=False).interval.diff().dropna()==1).all()
        n,k=self.prior.counts(x)
        groups=self.prior.groups(x);roles=x.node_type.astype(str).to_numpy()
        p=np.clip(np.array([self.prior.p.get(g,self.prior.p[r]) for g,r in zip(groups,roles)]),1e-5,1-1e-5)
        v=n*p*(1-p);r=k-n*p
        # Robust influence per interval, symmetric about the benign rate.
        # A gross shock can be omitted from the supplementary weak-evidence
        # channel; the original detector still evaluates it without trimming.
        scale=np.sqrt(np.maximum(v,1.))
        for name,c,trim in ROBUST:
            keep=(np.abs(r)<=c*scale) if trim else np.ones(len(x),dtype=bool)
            x["_r"]=np.where(keep,np.clip(r,-c*scale,c*scale),0.)
            x["_v"]=np.where(keep,v,0.)
            x["_n"]=np.where(keep,n,0.)
            for w in ((48,96) if name=="cap3" else (48,)):
                sums=x.groupby(keys,sort=False)[["_r","_v","_n"]].transform(lambda z:z.rolling(w,min_periods=1).sum())
                x[name+"_raw_"+str(w)]=np.where((n>0)&(sums._n>=30),
                    np.maximum(sums._r,0)/np.sqrt(np.maximum(sums._v,1)),0.)
        return x.drop(columns=["_r","_v","_n"]).sort_index()
    def fit(self,train):
        assert not train.is_anomaly.any()
        x=self.raw(train);g=self.prior.groups(x);r=x.node_type.astype(str).to_numpy()
        self.q={}
        for c in [v for v in x if "_raw_" in v and v.startswith(("cap3","cap6","trim8"))]:
            y=x[c].to_numpy()
            for role in ROLES:self.q[(c,role)]=max(float(np.quantile(y[r==role],.99,method="higher")),1.)
            for key in np.unique(g):
                if (g==key).sum()>=400:self.q[(c,key)]=max(float(np.quantile(y[g==key],.99,method="higher")),1.)
        self.fitted=True
        return self
    def prepare(self,df):
        assert self.fitted
        x=self.raw(self.prior.prepare(df))
        g=self.prior.groups(x);r=x.node_type.astype(str).to_numpy()
        for c in [v for v in x if "_raw_" in v and v.startswith(("cap3","cap6","trim8"))]:
            q=np.array([self.q.get((c,k),self.q[(c,role)]) for k,role in zip(g,r)])
            x[c.replace("_raw_","_norm_")]=x[c].to_numpy()/q
        x["raw48"]=x.count_norm_48
        x["cap3_48"]=x.cap3_norm_48;x["cap6_48"]=x.cap6_norm_48
        x["trim8_48"]=x.trim8_norm_48;x["cap3_96"]=x.cap3_norm_96
        x["cap3_multi"]=np.maximum(x.cap3_norm_48,x.cap3_norm_96)
        return x
    def manifest(self):
        return {"modes":MODES,"q99":{c+"|"+r:v for (c,r),v in self.q.items()},
                "base_reference_frozen":True,"minimum_dialogs":30}

def prepare_channel_scores(df,core,audit):
    """Cache only inference-derived components for efficient paired scoring."""
    x=df.copy();valid=x.primary_observer_valid.to_numpy(dtype=bool)
    a=np.zeros(len(x));w=np.zeros(len(x));e=np.zeros(len(x),dtype=bool)
    if valid.any():
        d=x.loc[valid]
        a[valid]=core.score(d)/core.threshold
        w[valid],e[valid]=core.warning(d)
    x["split_core"]=a;x["split_warning"]=w;x["split_eligible"]=e
    x["split_audit"]=audit.score(x)/audit.threshold
    return x

def upper_reference(val,col,budget,mode):
    assert budget>0 and mode in ("pooled","p90run")
    out=[]
    for x in val.values():
        z=x[col].to_numpy()
        q=float(np.quantile(z,1-budget,method="higher"))
        if mode=="p90run":
            per=[float(np.quantile(g[col],1-budget,method="higher")) for _,g in x.groupby("simulation_seed")]
            q=max(q,float(np.quantile(per,.9)))
        out.append(q)
    return max(max(out),1e-9)

class SeparateChannelTICD:
    def __init__(self,count_mode,budgets,calibration_mode):
        assert count_mode in MODES
        self.count_mode=count_mode;self.budgets=tuple(budgets)
        self.calibration_mode=calibration_mode
        assert len(budgets)==3 and budgets[0]>0 and budgets[1]>0 and 0<=budgets[2]
        assert sum(budgets)<=.008+1e-12
        self.thresholds={};self.warning_threshold=None
    @property
    def hybrid(self):return self.budgets[2]>0
    def channels(self,df):
        return {"core":df.split_core.to_numpy()>self.thresholds["core"],
                "count":df[self.count_mode].to_numpy()>self.thresholds["count"],
                "audit":df.split_audit.to_numpy()>self.thresholds["audit"] if self.hybrid else np.zeros(len(df),dtype=bool)}
    def strong(self,df):return np.logical_or.reduce(list(self.channels(df).values()))
    def calibrate(self,val):
        assert all(not d.is_anomaly.any() for d in val.values())
        self.thresholds={"core":upper_reference(val,"split_core",self.budgets[0],self.calibration_mode),
            "count":upper_reference(val,self.count_mode,self.budgets[1],self.calibration_mode),
            "audit":upper_reference(val,"split_audit",self.budgets[2],self.calibration_mode) if self.hybrid else None}
        qs=[]
        for d in val.values():
            w=np.where(d.split_eligible.to_numpy(dtype=bool)&~self.strong(d),d.split_warning.to_numpy(),0.)
            qs.append(float(np.quantile(w,.995,method="higher")))
            if self.calibration_mode=="p90run":
                qs.append(float(np.quantile([np.quantile(w[d.simulation_seed.to_numpy()==s],.995,method="higher")
                    for s in d.simulation_seed.unique()],.9)))
        self.warning_threshold=max(qs)
        for d in val.values():
            z=self.predict_levels(d)
            assert (z==2).mean()<=sum(self.budgets)+1e-12
            assert (z==1).mean()<=.005+1e-12
        return self
    def predict_levels(self,df):
        a=self.strong(df)
        w=(df.split_warning.to_numpy()>self.warning_threshold)&df.split_eligible.to_numpy(dtype=bool)&~a
        z=np.zeros(len(df),dtype=np.int8);z[w]=1;z[a]=2
        return z
    def predict_masks(self,df):
        y=self.predict_levels(df)
        return {"normal":y==0,"warning_only":y==1,"confirmed_alert":y==2,"any_signal":y>0}
    def manifest(self):
        return {"all_original_invariants_preserved":True,"count_mode":self.count_mode,
                "budgets":self.budgets,"calibration_mode":self.calibration_mode,
                "thresholds":self.thresholds,"warning_threshold":self.warning_threshold,
                "independent_audit_threshold":self.hybrid,"total_benign_budget":sum(self.budgets)}
