"""Frozen individual offsets learned during independently clean commissioning."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ticd_full_count_evidence import ROLES
from ticd_block_confirmation import BlockFusionTICD,WINDOWS,SCOPES

MIXES=(0.,.5,.75)

class CommissionedReference:
    def __init__(self,prior):
        self.prior=prior;self.q={};self.fitted=False
    def residual(self,df):
        ref=self.prior.previous
        x,n,k,loss,r,f=ref.inputs(df)
        p=ref.prediction(n,r,f);g=ref.groups(n,loss,r)
        z=(k-n*p)/np.sqrt(np.maximum(n*p*(1-p),1.))
        mu=np.array([ref.center.get(a,ref.center[b]) for a,b in zip(g,r)])
        sd=np.array([ref.scale.get(a,ref.scale[b]) for a,b in zip(g,r)])
        good=(n>0)&(np.abs(z)<=8.)
        e=(np.clip(z,-4.,4.)-mu)/sd
        v=np.maximum(n*p*(1-p),1.)
        b=np.searchsorted(ref.loss_edges,loss,side="right")
        return x,n,e,v,good,b
    def profiles(self,commission):
        x,n,e,v,good,b=self.residual(commission)
        # Each profile is specific to a commissioned installation + node.
        # Simulation seed is an opaque installation key, not a regressor.
        a=x[["simulation_seed","node_id","node_type"]].copy()
        a["bin"]=b;a["e"]=e;a["weight"]=np.sqrt(v)
        a=a.loc[good].copy()
        profiles={}
        for key,t in a.groupby(["simulation_seed","node_id","node_type"]):
            if len(t)<120:continue
            m=float(np.average(t.e,weights=t.weight))
            m=float(np.clip(m*len(t)/(len(t)+120.),-.75,.75))
            d={"overall":m,"bins":{},"observations":len(t)}
            for b0,s in t.groupby("bin"):
                if len(s)<25:continue
                w=len(s)/(len(s)+100.)
                q=float(np.average(s.e,weights=s.weight))
                d["bins"][int(b0)]=float(np.clip(w*q+(1-w)*m,-.75,.75))
            profiles[key]=d
        return profiles
    def raw(self,df,commission):
        # Commissioning is distinct from every operational record; callers
        # must provide trusted prior measurements for the same installation.
        x,n,e,v,good,b=self.residual(df)
        profiles=self.profiles(commission)
        keys=list(zip(x.simulation_seed,x.node_id,x.node_type))
        refs=[profiles.get(k) for k in keys]
        found=np.array([d is not None for d in refs],dtype=bool)
        offset=np.array([d["bins"].get(int(i),d["overall"]) if d else 0. for d,i in zip(refs,b)])
        x["commissioned_available"]=found
        x["commissioned_offset"]=offset
        x["_personal_num"]=np.where(good,(e-offset)*np.sqrt(v),0.)
        x["_personal_var"]=np.where(good,v,0.)
        x["_personal_n"]=np.where(good,n,0.)
        for w in WINDOWS:
            t=x.groupby(["simulation_seed","node_id"],sort=False)[["_personal_num","_personal_var","_personal_n"]].transform(lambda a:a.rolling(w,min_periods=1).sum())
            x["personal_raw_"+str(w)]=np.where(good&(t._personal_n>=30),np.maximum(t._personal_num,0.)/np.sqrt(np.maximum(t._personal_var,1.)),0.)
        cols=["commissioned_available","commissioned_offset"]+["personal_raw_"+str(w) for w in WINDOWS]
        return x[cols].sort_index()
    def fit(self,train,commission):
        assert not train.is_anomaly.any() and not commission.is_anomaly.any()
        assert set(train.simulation_seed)==set(commission.simulation_seed)
        a=self.raw(train,commission);r=train.node_type.astype(str).to_numpy()
        for w in WINDOWS:
            z=a["personal_raw_"+str(w)].to_numpy()
            for role in ROLES:
                self.q[(w,role)]=max(float(np.quantile(z[r==role],.99,method="higher")),1.)
        self.fitted=True;return self
    def prepare(self,df,commission):
        assert self.fitted
        x=self.prior.prepare(df)
        a=self.raw(df,commission)
        for c in a:x[c]=a[c]
        role=x.node_type.astype(str).to_numpy()
        found=x.commissioned_available.to_numpy(dtype=bool)
        for w in WINDOWS:
            old=x["supported_"+str(w)].to_numpy()
            q=np.array([self.q[(w,r)] for r in role])
            personal=x["personal_raw_"+str(w)].to_numpy()/q
            for mix in MIXES:
                # Preserve a positive contribution of the entire previous
                # supported conditional channel; no change to core/count/audit.
                val=(1-mix)*old+mix*personal
                x["personal_"+str(w)+"_"+str(int(100*mix))]=np.where(found,val,old)
        return x
    def manifest(self):
        return {"commissioning_required_for_personalization":True,"fallback":"previous supported conditional score",
            "mixes":MIXES,"minimum_old_conditional_weight":.25,
            "offset_limit":.75,"min_node_observations":120,"min_bin_observations":25,
            "node_shrink":120,"bin_shrink":100,"q99":{str(w)+"|"+r:v for (w,r),v in self.q.items()},
            "no_operational_reference_update":True}

class CommissionedFusionTICD(BlockFusionTICD):
    def __init__(self,window,mix,scope,budgets,calibration_mode):
        assert mix in MIXES
        super().__init__(window,"supported",scope,budgets,calibration_mode)
        self.mix=mix
    def columns(self):
        out=super().columns()
        out["conditional"]="personal_"+str(self.window)+"_"+str(int(100*self.mix))
        return out
    def manifest(self):
        return {**super().manifest(),"mix":self.mix,"minimum_old_conditional_weight":1-self.mix}
