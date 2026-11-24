"""Physical loss compensation using sampled path observations.

This is a model of the simulator's transit-count observation, not a
universal SIP packet-loss law. The conditional positive-part Gaussian
approximation accounts for clamping ACK counts to observed 200 OK.
All existing TICD scores remain separate and positively budgeted.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from ticd_full_count_evidence import ROLES
from ticd_commissioned_reference import CommissionedFusionTICD

SOURCES=("cu","probe128","probe512","guard512")
WINDOWS=(48,96)
EDGES=(.001,.003,.008,.015,.03,.06)

class IndependentLossReference:
    def __init__(self,previous):
        self.previous=previous;self.center={};self.q={};self.fitted=False
    def inputs(self,df,mode):
        x=df.sort_values(["simulation_seed","node_id","interval"],kind="stable").copy()
        keys=["simulation_seed","node_id"]
        assert not x.duplicated(keys+["interval"]).any()
        assert (x.groupby(keys,sort=False).interval.diff().dropna()==1).all()
        ref=self.previous.prior.previous.prior
        n,k=ref.counts(x)
        cu=x.cu_combined_loss_prob.to_numpy(dtype=float)
        assert np.isfinite(cu).all() and ((cu>=0)&(cu<=1)).all()
        pcu=1-np.sqrt(np.clip(1-cu,0.,1.))
        if mode=="cu":
            p=pcu;varp=np.zeros(len(x));available=n>0
        else:
            num=128 if mode=="probe128" else 512
            sent=x["probe"+str(num)+"_sent"].to_numpy(dtype=float)
            lost=x["probe"+str(num)+"_lost"].to_numpy(dtype=float)
            age=x.probe_age.to_numpy(dtype=float)
            assert (sent>=lost).all() and (lost>=0).all()
            available=(n>0)&(sent>0)&(age>=0)&(age<=3)
            p=(lost+.25)/(sent+.5)
            p=np.where(sent>0,p,pcu)
            varp=p*(1-p)/np.maximum(sent+1,1.)
            if mode=="guard512":
                # Compare against the existing CU-derived probability;
                # disagreement increases uncertainty rather than creating
                # stronger evidence. This is not independent probe-only.
                delta=p-pcu
                varp=.25**2*varp+.25*delta**2
                p=.75*pcu+.25*p
            # Latency increases measurement uncertainty. No future samples.
            varp+=np.maximum(age,0)*(.002**2)
        p=np.clip(p,1e-6,.45)
        # Approximate m underlying responses using observed N. In the
        # simulator U~Bin(m,1-p), V~Bin(m,(1-p)^2), conditionally independent
        # before clipping; observed missing K=max(U-V,0). Approximate
        # E[K] with the positive-part normal mean and empirical centering.
        m=n/np.maximum(1-p,.01)
        a=1-p;b=a*a
        mu=n*p
        var=m*(a*(1-a)+b*(1-b))
        sd=np.sqrt(np.maximum(var,1e-8))
        z=mu/sd
        expected=sd*np.exp(-.5*z*z)/np.sqrt(2*np.pi)+mu*ndtr(z)
        total_var=np.maximum(var+n*n*varp,1.)
        residual=(k-expected)/np.sqrt(total_var)
        good=available&(np.abs(residual)<=8.)
        role=x.node_type.astype(str).to_numpy()
        bins=np.searchsorted(EDGES,p,side="right")
        groups=np.array([r+"|"+str(b0) for r,b0 in zip(role,bins)])
        return x,n,residual,total_var,good,available,groups,role
    def fit(self,train):
        assert not train.is_anomaly.any()
        for mode in SOURCES:
            x,n,e,v,good,available,g,r=self.inputs(train,mode)
            z=np.clip(e,-4.,4.)
            for role in ROLES:
                mask=good&(r==role)
                assert mask.sum()>=400
                self.center[(mode,role)]=float(np.average(z[mask],weights=np.sqrt(v[mask])))
            for key in np.unique(g):
                mask=good&(g==key)
                if mask.sum()>=200:
                    a=mask.sum()/(mask.sum()+500.)
                    prior=self.center[(mode,key.split("|")[0])]
                    self.center[(mode,key)]=float(a*np.average(z[mask],weights=np.sqrt(v[mask]))+(1-a)*prior)
            y=self.raw(train,mode);roles=train.node_type.astype(str).to_numpy()
            for w in WINDOWS:
                q=y["loss_raw_"+mode+"_"+str(w)].to_numpy()
                for role in ROLES:
                    self.q[(mode,w,role)]=max(1.,float(np.quantile(q[roles==role],.99,method="higher")))
        self.fitted=True;return self
    def raw(self,df,mode):
        x,n,e,v,good,available,g,r=self.inputs(df,mode)
        bias=np.array([self.center.get((mode,k),self.center[(mode,role)]) for k,role in zip(g,r)])
        x["_loss_num"]=np.where(good,(np.clip(e,-4.,4.)-bias)*np.sqrt(v),0.)
        x["_loss_var"]=np.where(good,v,0.)
        x["_loss_n"]=np.where(good,n,0.)
        cols=[]
        for w in WINDOWS:
            s=x.groupby(["simulation_seed","node_id"],sort=False)[["_loss_num","_loss_var","_loss_n"]].transform(lambda a:a.rolling(w,min_periods=1).sum())
            col="loss_raw_"+mode+"_"+str(w)
            x[col]=np.where(good&(s._loss_n>=30),np.maximum(s._loss_num,0.)/np.sqrt(np.maximum(s._loss_var,1.)),0.)
            cols.append(col)
        c="loss_available_"+mode;x[c]=available;cols.append(c)
        return x[cols].sort_index()
    def prepare(self,df,commission):
        assert self.fitted
        x=self.previous.prepare(df,commission)
        roles=x.node_type.astype(str).to_numpy()
        for mode in SOURCES:
            a=self.raw(df,mode)
            for c in a:x[c]=a[c]
            for w in WINDOWS:
                q=np.array([self.q[(mode,w,r)] for r in roles])
                x["loss_"+mode+"_"+str(w)]=x["loss_raw_"+mode+"_"+str(w)].to_numpy()/q
        x["loss_none"]=0.
        return x
    def manifest(self):
        return {"sources":SOURCES,"windows":WINDOWS,"empirical_center":{m+"|"+g:v for (m,g),v in self.center.items()},
            "q99":{m+"|"+str(w)+"|"+r:v for (m,w,r),v in self.q.items()},
            "cu_analytic_is_simulator_probability_proxy":True,"probe_counts_only":True,
            "max_probe_age":3,"probe_load_not_modeled":True,"no_online_reference_update":True}

class LossFusionTICD(CommissionedFusionTICD):
    def __init__(self,source,window,budgets,calibration_mode):
        assert source in ("none",*SOURCES) and window in WINDOWS
        assert len(budgets)==5 and min(budgets[:3])>0 and sum(budgets)<=.008+1e-12
        assert (budgets[3]>0)==(source!="none") and budgets[4]>=0
        super().__init__(144,.75,"global",(*budgets[:3],budgets[4]),calibration_mode)
        self.source=source;self.loss_window=window;self.budgets=tuple(budgets)
    @property
    def hybrid(self):return self.budgets[4]>0
    def columns(self):
        d=super().columns()
        return {"core":d["core"],"count":d["count"],"conditional":d["conditional"],
                "loss":"loss_none" if self.source=="none" else "loss_"+self.source+"_"+str(self.loss_window),
                "audit":d["audit"]}
    def manifest(self):
        return {**super().manifest(),"loss_source":self.source,"loss_window":self.loss_window,
                "existing_conditional_preserved":True,"synthetic_measurement":self.source.startswith("probe") or self.source=="guard512"}
