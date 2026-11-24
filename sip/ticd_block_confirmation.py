"""Contextual reference and disjoint-block support for full TICD.

No online baseline updates on potentially compromised traffic. All previous
instant invariants, historical score, trimmed S48, sequential score and the
long conditional channel remain present. Its separate threshold prevents
an extreme long score from changing the audit/legacy thresholds.
"""
from __future__ import annotations
import numpy as np
from ticd_full_count_evidence import ROLES
from ticd_separate_robust_fusion import upper_reference

WINDOWS=(96,144)
MODES=("raw","context","supported")
SCOPES=("global","role")

class BlockReference:
    def __init__(self,previous):
        self.previous=previous
        self.loss_edges={};self.volume_edges={};self.q={};self.fitted=False
    def context(self,df):
        x=self.previous.prepare(df,self.core,self.audit)
        x=x.sort_values(["simulation_seed","node_id","interval"],kind="stable")
        keys=["simulation_seed","node_id"]
        assert not x.duplicated(keys+["interval"]).any()
        assert (x.groupby(keys,sort=False).interval.diff().dropna()==1).all()
        n,_=self.previous.prior.counts(x)
        loss=x.cu_combined_loss_prob.to_numpy(dtype=float)
        x["_bn"]=n;x["_bl"]=n*loss
        for w in WINDOWS:
            sums=x.groupby(keys,sort=False)[["_bn","_bl"]].transform(lambda v:v.rolling(w,min_periods=1).sum())
            x["block_loss_"+str(w)]=np.divide(sums._bl,sums._bn,out=np.zeros(len(x)),where=sums._bn>0)
            x["block_volume_"+str(w)]=sums._bn
            role=x.node_type.astype(str).to_numpy()
            q=np.array([self.previous.q[(w,r)] for r in role])
            x["raw_"+str(w)]=x["conditional_raw_"+str(w)].to_numpy()/q
        return x.drop(columns=["_bn","_bl"]).sort_index()
    def group(self,x,w):
        role=x.node_type.astype(str).to_numpy()
        loss=x["block_loss_"+str(w)].to_numpy()
        volume=x["block_volume_"+str(w)].to_numpy()
        b=np.searchsorted(self.loss_edges[w],loss,side="right")
        return np.array([r+"|"+str(k)+"|"+str(np.searchsorted(self.volume_edges[(w,r)],v,side="right"))
                         for r,k,v in zip(role,b,volume)])
    def fit(self,train,core,audit):
        assert not train.is_anomaly.any() and core.variant=="full"
        self.core=core;self.audit=audit
        x=self.context(train);r=x.node_type.astype(str).to_numpy()
        for w in WINDOWS:
            l=x["block_loss_"+str(w)].to_numpy()
            n=x["block_volume_"+str(w)].to_numpy()
            good=n>=30
            self.loss_edges[w]=np.unique(np.quantile(l[good],[.25,.5,.75]))
            for role in ROLES:
                self.volume_edges[(w,role)]=np.unique(np.quantile(n[good&(r==role)],[.33,.67]))
            g=self.group(x,w);v=x["raw_"+str(w)].to_numpy()
            for role in ROLES:
                self.q[(w,role)]=max(1.,float(np.quantile(v[r==role],.99,method="higher")))
            for key in np.unique(g):
                m=g==key;role=key.split("|")[0]
                if m.sum()>=400:
                    # Conditional tail scale cannot amplify a sparse context.
                    # Use a conservative pooled tail and between-run tail
                    # estimate; the latter is based on clean runs only.
                    sub=x.loc[m,["simulation_seed"]].copy();sub["v"]=v[m]
                    qs=[float(np.quantile(t.v,.99,method="higher")) for _,t in sub.groupby("simulation_seed") if len(t)>=80]
                    robust=float(np.quantile(qs,.9)) if len(qs)>=4 else 0.
                    self.q[(w,key)]=max(self.q[(w,role)],float(np.quantile(v[m],.99,method="higher")),robust)
        self.fitted=True;return self
    def prepare(self,df):
        assert self.fitted
        x=self.context(df).sort_values(["simulation_seed","node_id","interval"],kind="stable")
        role=x.node_type.astype(str).to_numpy()
        q48=np.array([self.previous.q[(48,r)] for r in role])
        x["_block48"]=x.conditional_raw_48.to_numpy()/q48
        by=x.groupby(["simulation_seed","node_id"],sort=False)["_block48"]
        a=x["_block48"].to_numpy()
        b=by.shift(48,fill_value=0.).to_numpy()
        c=by.shift(96,fill_value=0.).to_numpy()
        for w in WINDOWS:
            groups=self.group(x,w)
            q=np.array([self.q.get((w,g),self.q[(w,r)]) for g,r in zip(groups,role)])
            long=x["raw_"+str(w)].to_numpy()
            context=long/q
            # Disjoint blocks [t-47,t], [t-95,t-48], [t-143,t-96].
            # Two-block minimum or three-block median. Never fully disable
            # long evidence, even in startup or weak corroboration.
            support=np.minimum(a,b) if w==96 else np.median(np.column_stack((a,b,c)),axis=1)
            x["context_"+str(w)]=context
            x["supported_"+str(w)]=.25*context+.75*np.minimum(context,support)
        return x.drop(columns="_block48").sort_index()
    def manifest(self):
        return {"windows":WINDOWS,"context_inputs":["rolling observed 200 OK volume","volume-weighted observed CU loss","role"],
            "loss_edges":{str(w):v.tolist() for w,v in self.loss_edges.items()},
            "volume_edges":{str(w)+"|"+r:v.tolist() for (w,r),v in self.volume_edges.items()},
            "context_scale":{str(w)+"|"+r:v for (w,r),v in self.q.items()},
            "minimum_long_weight":.25,"no_online_reference_update":True}

class BlockFusionTICD:
    def __init__(self,window,mode,scope,budgets,calibration_mode):
        assert window in WINDOWS and mode in MODES and scope in SCOPES
        assert len(budgets)==4 and min(budgets[:3])>0 and budgets[3]>=0 and sum(budgets)<=.008+1e-12
        self.window=window;self.mode=mode;self.scope=scope;self.budgets=tuple(budgets)
        self.calibration_mode=calibration_mode;self.warning_budget=.0035
        self.thresholds={};self.warning_threshold=None
    @property
    def hybrid(self):return self.budgets[3]>0
    def columns(self):
        s="_role" if self.scope=="role" else ""
        return {"core":"core_0"+s,"count":"baseline"+s,"conditional":self.mode+"_"+str(self.window),"audit":"split_audit"}
    def channels(self,x):
        return {k:x[c].to_numpy()>self.thresholds[k] if self.thresholds[k] is not None else np.zeros(len(x),dtype=bool)
                for k,c in self.columns().items()}
    def strong(self,x):return np.logical_or.reduce(list(self.channels(x).values()))
    def calibrate(self,val):
        assert all(not x.is_anomaly.any() for x in val.values())
        def bound(c,b):
            if self.calibration_mode=="p90run":return upper_reference(val,c,b,"p90run")
            assert self.calibration_mode=="worst_run"
            return max(1e-9,max(float(np.quantile(g[c],1-b,method="higher")) for x in val.values() for _,g in x.groupby("simulation_seed")))
        self.thresholds={k:bound(c,b) if b else None for (k,c),b in zip(self.columns().items(),self.budgets)}
        self.warning_threshold=max(float(np.quantile(np.where(
              g.split_eligible.to_numpy(dtype=bool)&~self.strong(g),g.warn_0.to_numpy(),0.),
              1-self.warning_budget,method="higher"))
              for x in val.values() for _,g in x.groupby("simulation_seed"))
        for x in val.values():
            z=self.predict_levels(x)
            assert (z==2).mean()<=sum(self.budgets)+1e-12
            assert (z==1).mean()<=self.warning_budget+1e-12
        return self
    def predict_levels(self,x):
        a=self.strong(x)
        w=x.split_eligible.to_numpy(dtype=bool)&(x.warn_0.to_numpy()>self.warning_threshold)&~a
        z=np.zeros(len(x),dtype=np.int8);z[w]=1;z[a]=2
        return z
    def predict_masks(self,x):
        z=self.predict_levels(x)
        return {"normal":z==0,"warning_only":z==1,"confirmed_alert":z==2,"any_signal":z>0}
    def manifest(self):
        return {"window":self.window,"mode":self.mode,"scope":self.scope,
            "budgets":self.budgets,"calibration_mode":self.calibration_mode,"warning_budget":self.warning_budget,
            "thresholds":self.thresholds,"warning_threshold":self.warning_threshold,
            "all_original_invariants_preserved":True,"legacy_count_preserved":True,"audit_independent":self.hybrid}
