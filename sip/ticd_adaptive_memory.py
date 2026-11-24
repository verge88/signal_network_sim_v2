"""Causal recovery-aware memory and supplementary weak-evidence accumulation.

All original instant invariants, role weights, history and warning are retained.
Recovery attenuates only historical evidence, never instantaneous evidence.
The independent audit is unchanged. No attack labels or scenario schedule
participate in preparation/inference. Complete run prefixes required.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ticd_full_count_evidence import ROLES
from ticd_separate_robust_fusion import prepare_channel_scores,upper_reference

RECOVERY=(0,3,6)
SEQUENCES=("none","fast","slow")
# No structural simplification: baseline, recovery-only, sequential-only,
# and joint options. All retain trimmed S48.
SPECS=((0,"none"),(3,"none"),(6,"none"),(0,"fast"),(0,"slow"),
       (3,"fast"),(3,"slow"),(6,"slow"))

class AdaptiveMemoryReference:
    def __init__(self,bounded):
        self.bounded=bounded;self.center={};self.q={};self.fitted=False
    def inputs(self,df):
        x=df.sort_values(["simulation_seed","node_id","interval"],kind="stable").copy()
        keys=["simulation_seed","node_id"]
        assert not x.duplicated(keys+["interval"]).any()
        assert (x.groupby(keys,sort=False).interval.diff().dropna()==1).all()
        ref=self.bounded.prior;n,k=ref.counts(x)
        groups=ref.groups(x);roles=x.node_type.astype(str).to_numpy()
        p=np.clip(np.array([ref.p.get(g,ref.p[r]) for g,r in zip(groups,roles)]),1e-5,1-1e-5)
        avail=n>0
        z=(k-n*p)/np.sqrt(np.maximum(n*p*(1-p),1.))
        shock=avail&(np.abs(z)>8.)
        # Trim outlying residuals only in the weak-evidence stream. The
        # original detector retains full instantaneous evidence.
        usable=avail&~shock
        return x,groups,roles,avail,z,shock,usable
    def fit(self,train):
        assert not train.is_anomaly.any()
        x,g,r,avail,z,shock,u=self.inputs(train)
        self.center={}
        for role in ROLES:
            m=u&(r==role);assert m.sum()>=100
            self.center[role]=float(np.mean(np.clip(z[m],-3.,3.)))
        for key in np.unique(g):
            m=u&(g==key)
            if m.sum()>=400:
                self.center[key]=float(np.mean(np.clip(z[m],-3.,3.)))
        y=self.raw(train)
        # raw restores original row order: align reference groups to it.
        g=self.bounded.prior.groups(y);r=y.node_type.astype(str).to_numpy()
        self.q={}
        for c in ("seq_fast_raw","seq_slow_raw"):
            v=y[c].to_numpy()
            for role in ROLES:
                self.q[(c,role)]=max(float(np.quantile(v[r==role],.99,method="higher")),1.)
            for key in np.unique(g):
                m=g==key
                if m.sum()>=400:self.q[(c,key)]=max(float(np.quantile(v[m],.99,method="higher")),1.)
        self.fitted=True;return self
    def raw(self,df):
        x,g,r,avail,z,shock,u=self.inputs(df)
        centers=np.array([self.center.get(k,self.center[role]) for k,role in zip(g,r)])
        residual=np.clip(z,-3.,3.)-centers
        # Sequence states are updated in order per node. Missing observations
        # decay sequence evidence, break recovery streaks, and do not trigger
        # new evidence. No state is carried across independent runs.
        seq1=np.zeros(len(x));seq2=np.zeros(len(x))
        gate3=np.ones(len(x));gate6=np.ones(len(x))
        for ix in x.groupby(["simulation_seed","node_id"],sort=False).indices.values():
            a=b=0.;age=10**6;streak=0
            for i in ix:
                age+=1
                if shock[i]:age=0;streak=0
                elif u[i] and residual[i]<=.25:streak+=1
                else:streak=0
                # Temporary attenuation following a gross observed shock,
                # only after consecutive low current residuals. Never zero.
                if age<=48:
                    if streak>=3:gate3[i]=.25
                    if streak>=6:gate6[i]=.25
                if u[i]:
                    a=max(0.,.98*a+residual[i]-.10)
                    b=max(0.,.995*b+residual[i]-.05)
                else:
                    a*=.98;b*=.995
                if avail[i] and not shock[i]:
                    seq1[i]=a;seq2[i]=b
        x["recovery3"]=gate3;x["recovery6"]=gate6
        x["seq_fast_raw"]=seq1;x["seq_slow_raw"]=seq2
        return x.sort_index()
    def prepare(self,df,core,audit):
        assert self.fitted and core.variant=="full"
        x=self.raw(self.bounded.prepare(df))
        g=self.bounded.prior.groups(x);r=x.node_type.astype(str).to_numpy()
        for c in ("seq_fast_raw","seq_slow_raw"):
            q=np.array([self.q.get((c,k),self.q[(c,role)]) for k,role in zip(g,r)])
            x[c.replace("_raw","")]=x[c].to_numpy()/q
        x=prepare_channel_scores(x,core,audit)
        valid=x.primary_observer_valid.to_numpy(dtype=bool)
        instant=np.zeros(len(x));hist=np.zeros(len(x))
        if valid.any():
            a,h=core.model.components(x.loc[valid])
            instant[valid]=a/core.threshold;hist[valid]=h/core.threshold
        assert np.allclose(np.maximum(instant,hist),x.split_core,atol=1e-12,rtol=1e-12)
        x["memory_instant"]=instant;x["memory_history"]=hist
        # Precompute candidate scores using only observed causal state.
        for n in RECOVERY:
            gate=np.ones(len(x)) if n==0 else x["recovery"+str(n)].to_numpy()
            x["core_"+str(n)]=np.maximum(instant,gate*hist)
            x["warn_"+str(n)]=gate*x.split_warning.to_numpy()
            for seq in SEQUENCES:
                c=x.trim8_48.to_numpy()
                if seq!="none":c=np.maximum(c,x["seq_"+seq].to_numpy())
                x["count_"+str(n)+"_"+seq]=gate*c
        return x
    def manifest(self):
        return {"center":self.center,"q99":{c+"|"+g:v for (c,g),v in self.q.items()},
                "recovery_streaks":RECOVERY,"recovery_factor":.25,
                "sequence_decay":{"fast":.98,"slow":.995},"sequence_drift":{"fast":.10,"slow":.05},
                "all_original_invariants_preserved":True}

class AdaptiveMemoryTICD:
    def __init__(self,recovery,sequence,budgets,calibration_mode,warning_budget=.004):
        assert (recovery,sequence) in SPECS and len(budgets)==3 and min(budgets[:2])>0
        assert 0<=budgets[2] and sum(budgets)<=.008+1e-12
        self.recovery=recovery;self.sequence=sequence;self.budgets=tuple(budgets)
        self.calibration_mode=calibration_mode;self.warning_budget=warning_budget
        self.thresholds={};self.warning_threshold=None
    @property
    def hybrid(self):return self.budgets[2]>0
    def columns(self):
        return {"core":"core_"+str(self.recovery),
                "count":"count_"+str(self.recovery)+"_"+self.sequence,"audit":"split_audit"}
    def channels(self,df):
        return {k:df[c].to_numpy()>self.thresholds[k] if self.thresholds[k] is not None else np.zeros(len(df),dtype=bool)
                for k,c in self.columns().items()}
    def strong(self,df):return np.logical_or.reduce(list(self.channels(df).values()))
    def calibrate(self,val):
        assert all(not d.is_anomaly.any() for d in val.values())
        self.thresholds={k:upper_reference(val,c,b,self.calibration_mode) if b>0 else None
                         for (k,c),b in zip(self.columns().items(),self.budgets)}
        qs=[]
        for d in val.values():
            w=np.where(d.split_eligible.to_numpy(dtype=bool)&~self.strong(d),d["warn_"+str(self.recovery)].to_numpy(),0.)
            qs.append(float(np.quantile(w,1-self.warning_budget,method="higher")))
            # Conservative warning calibration on each normal run.
            qs.extend(float(np.quantile(w[d.simulation_seed.to_numpy()==s],1-self.warning_budget,method="higher")) for s in d.simulation_seed.unique())
        self.warning_threshold=max(qs)
        for d in val.values():
            z=self.predict_levels(d)
            assert (z==2).mean()<=sum(self.budgets)+1e-12
            assert (z==1).mean()<=self.warning_budget+1e-12
        return self
    def predict_levels(self,df):
        a=self.strong(df)
        w=(df["warn_"+str(self.recovery)].to_numpy()>self.warning_threshold)&df.split_eligible.to_numpy(dtype=bool)&~a
        z=np.zeros(len(df),dtype=np.int8);z[w]=1;z[a]=2
        return z
    def predict_masks(self,df):
        z=self.predict_levels(df)
        return {"normal":z==0,"warning_only":z==1,"confirmed_alert":z==2,"any_signal":z>0}
    def manifest(self):
        return {"all_original_invariants_preserved":True,"recovery":self.recovery,"sequence":self.sequence,
                "budgets":self.budgets,"calibration_mode":self.calibration_mode,"warning_budget":self.warning_budget,
                "thresholds":self.thresholds,"warning_threshold":self.warning_threshold,"audit_independent":self.hybrid}
