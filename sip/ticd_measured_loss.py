"""Full TICD refitted on measured loss; explicit telemetry boundary."""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from ticd_invariant_ablation import InvariantAblationTICD
from ticd_physical_closure import prepare_master_closure_features
from ticd_causal_ack import add_causal_ack_history
from ticd_zonal_warnings import prepare_zonal_warning_context
from ticd_full_count_evidence import CountReference,ROLES
from ticd_separate_robust_fusion import BoundedCountReference
from ticd_adaptive_memory import AdaptiveMemoryReference
from ticd_conditional_baseline import ConditionalCountReference
from ticd_block_confirmation import BlockReference
from ticd_commissioned_reference import CommissionedReference
from ticd_independent_loss import IndependentLossReference,LossFusionTICD,WINDOWS
from ticd_observability_factorial import prepare_policy_history,PolicyEvidence

MODES=("oracle","cu","active","passive","hybrid")
# Exact allowed primitive observations. No simulator loss, calculated delay,
# ground truth, precomputed loss-dependent statistics or attack labels here.
RAW=("simulation_seed","node_id","node_type","interval","zone_id","primary_observer_valid",
     "cu_delivered","cu_dcr_transit","cu_n_200_transit","cu_n_ack_transit",
     "cu_ack_closure_mismatch","cu_txn_state_consistency_error",
     "obs_dialog_completion_ratio","obs_dangling_dialog_ratio","obs_total_messages",
     "mean_txn_state_consistency_error","dev_txn_consistency_vs_zone",
     "active_sent","active_fwd_lost","active_rev_lost","active_age",
     "passive_sent","passive_received")
LABELS=("is_anomaly","attack_type","attacker_sophistication","eval_active_deficit")

def measured_frame(df,mode,commission=False):
    assert mode in MODES
    cols=list(RAW)
    if not commission:cols+=["audit_observed_count","audit_mismatch_count"]
    x=df[cols].copy().sort_values(["simulation_seed","node_id","interval"],kind="stable")
    keys=["simulation_seed","node_id"]
    assert not x.duplicated(keys+["interval"]).any()
    assert (x.groupby(keys,sort=False).interval.diff().dropna()==1).all()
    valid=x.primary_observer_valid.to_numpy(dtype=bool)
    def roll(a,w):
        x["_obs_tmp"]=a
        return x.groupby(keys,sort=False)["_obs_tmp"].transform(lambda z:z.rolling(w,min_periods=1).sum()).to_numpy()
    # The observed success of a request/response exchange is not an
    # independent observation of each direction. Use a symmetric conversion.
    trials=roll(valid.astype(float),64)
    failures=roll(valid&(x.cu_delivered.to_numpy()==0),64)
    l=(failures+.25)/(trials+16.)
    pc=1-np.sqrt(1-l)
    vc=l*(1-l)/np.maximum(trials+17.,1.)/(4*np.maximum(1-l,.01))
    # Current probe outcomes; direction-specific observed counts retained.
    # Four-interval estimate reduces binomial noise at low rates.
    sent=x.active_sent.to_numpy(dtype=float)
    af=x.active_fwd_lost.to_numpy(dtype=float);ar=x.active_rev_lost.to_numpy(dtype=float)
    assert (sent>=af).all() and (sent>=ar).all()
    assert (af>=0).all() and (ar>=0).all()
    an=roll(sent,4)
    pf=(roll(af,4)+.25)/(an+.5)
    pr=(roll(ar,4)+.25)/(an+.5)
    pa=(pf+pr)/2
    va=(pf*(1-pf)+pr*(1-pr))/(4*np.maximum(an+1.,1.))
    va+=np.maximum(x.active_age.to_numpy(),0)*.002**2
    av=(sent>0)&(x.active_age.to_numpy()<=3)&valid
    pn=x.passive_sent.to_numpy(dtype=float)
    pl=pn-x.passive_received.to_numpy(dtype=float)
    assert (pl>=0).all()
    ptotal=roll(pn,8)
    pp=(roll(pl,8)+.25)/(ptotal+.5)
    vp=pp*(1-pp)/np.maximum(ptotal+1.,1.)
    pv=(pn>0)&valid
    if mode=="oracle":
        # Explicit isolated ideal comparator only.
        p=1-np.sqrt(1-df.loc[x.index,"cu_combined_loss_prob"].to_numpy(dtype=float))
        v=np.zeros(len(x));available=valid
    elif mode=="cu":
        p=pc;v=vc;available=valid&(trials>0)
    elif mode=="active":
        p=np.where(av,pa,pc);v=np.where(av,va,vc);available=av
    elif mode=="passive":
        p=np.where(pv,pp,pc);v=np.where(pv,vp,vc);available=pv
    else:
        # Conservative combination; do not assume passive/probe independence.
        wa=np.where(av,.7,0.);wp=np.where(pv,.3,0.);den=wa+wp
        p=np.divide(wa*pa+wp*pp,den,out=pc.copy(),where=den>0)
        v=np.divide(wa*(va+(pa-p)**2)+wp*(vp+(pp-p)**2),den,out=vc.copy(),where=den>0)
        available=av|pv
    p=np.clip(p,1e-6,.45);v=np.maximum(v,0.)
    # All legacy branches receive the same replacement covariate.
    # The name is retained for compatibility, not an oracle probability.
    x["cu_combined_loss_prob"]=1-(1-p)**2
    x["measured_oneway_loss"]=p;x["measured_loss_variance"]=v
    x["measurement_available"]=available
    x["measurement_fallback"]=~available&valid
    x=x.drop(columns="_obs_tmp")
    # No unavailable primary observations enter history or zone comparisons.
    x.loc[~valid,["cu_delivered","cu_n_200_transit","cu_n_ack_transit","cu_ack_closure_mismatch",
        "cu_txn_state_consistency_error","mean_txn_state_consistency_error","dev_txn_consistency_vs_zone",
        "obs_total_messages"]]=0
    x.loc[~valid,"cu_dcr_transit"]=1.
    x.loc[~valid,["obs_dialog_completion_ratio","obs_dangling_dialog_ratio"]]=.5
    # Rebuild I1 zone feature after masking, without using old derived input.
    x["dev_txn_consistency_vs_zone"]=x.mean_txn_state_consistency_error-x.groupby(
        ["simulation_seed","interval","zone_id"],sort=False).mean_txn_state_consistency_error.transform("mean")
    x=add_causal_ack_history(prepare_master_closure_features(x))
    x["zone_causal_median"]=0.
    grp=x.groupby(["simulation_seed","interval","zone_id"],sort=False)
    x["zone_reported_load_median"]=grp.obs_total_messages.transform("median")
    x["zone_observed_loss_median"]=grp.cu_combined_loss_prob.transform("median")
    x["zone_member_count"]=grp.node_id.transform("size")
    if not commission:x=prepare_policy_history(x)
    return x.sort_index()

class InactiveContext:
    """Old contextual model is inactive; fail if any code tries to score it."""
    fitted=True
    def score(self,df):raise RuntimeError("Inactive contextual model must not be used")

def labels(x,raw):
    x=x.copy()
    for c in LABELS:
        if c in raw:x[c]=raw[c]
    return x

class MeasuredPhysicalReference(IndependentLossReference):
    def inputs(self,df,mode):
        assert mode=="cu"
        x=df.sort_values(["simulation_seed","node_id","interval"],kind="stable").copy()
        n,k=CountReference.counts(x)
        p=x.measured_oneway_loss.to_numpy(dtype=float)
        varp=x.measured_loss_variance.to_numpy(dtype=float)
        a=1-p;b=a*a;m=n/np.maximum(a,.01)
        mu=n*p;var=m*(a*(1-a)+b*(1-b))
        sd=np.sqrt(np.maximum(var,1e-8));z=mu/sd
        expected=sd*np.exp(-.5*z*z)/np.sqrt(2*np.pi)+mu*ndtr(z)
        # Rolling estimates share samples; variance is a conservative scale,
        # not an exact probability model. Empirical tails refitted below.
        total=np.maximum(var+n*n*varp*4.,1.)
        e=(k-expected)/np.sqrt(total)
        available=n>0;good=available&(np.abs(e)<=8.)
        r=x.node_type.astype(str).to_numpy()
        g=np.array([role+"|"+str(i) for role,i in zip(r,np.searchsorted([.001,.003,.008,.015,.03,.06],p,side="right"))])
        return x,n,e,total,good,available,g,r
    def fit(self,train):
        assert not train.is_anomaly.any()
        mode="cu";x,n,e,v,good,av,g,r=self.inputs(train,mode)
        z=np.clip(e,-4.,4.)
        for role in ROLES:
            m=good&(r==role);assert m.sum()>=400
            self.center[(mode,role)]=float(np.average(z[m],weights=np.sqrt(v[m])))
        for key in np.unique(g):
            m=good&(g==key)
            if m.sum()>=200:
                w=m.sum()/(m.sum()+500.)
                self.center[(mode,key)]=float(w*np.average(z[m],weights=np.sqrt(v[m]))+(1-w)*self.center[(mode,key.split("|")[0])])
        y=self.raw(train,mode);r=train.loc[y.index,"node_type"].astype(str).to_numpy()
        for w in WINDOWS:
            q=y["loss_raw_cu_"+str(w)].to_numpy()
            for role in ROLES:self.q[(mode,w,role)]=max(1.,float(np.quantile(q[r==role],.99,method="higher")))
        self.fitted=True;return self
    def prepare(self,df,commission):
        assert self.fitted
        x=self.previous.prepare(df,commission);a=self.raw(df,"cu")
        for c in a:x[c]=a[c]
        r=x.node_type.astype(str).to_numpy()
        for w in WINDOWS:
            q=np.array([self.q[("cu",w,role)] for role in r])
            x["loss_cu_"+str(w)]=x["loss_raw_cu_"+str(w)].to_numpy()/q
        return x

class FullMeasuredReference:
    def __init__(self,mode,source):
        self.mode=mode
        # Retain architecture and coefficients, not trained normal statistics.
        self.core=InvariantAblationTICD(source.model,"full",refit=True)
        assert self.core.physical.mode=="temporal"
        assert self.core.physical.physical_mode in ("physical","physical_zone")
        assert self.core.zonal.mode=="load"
        self.core.physical.base_model=InactiveContext()
        self.audit=PolicyEvidence()
    def fit(self,train,commission,val):
        t=labels(measured_frame(train,self.mode),train)
        c=labels(measured_frame(commission,self.mode,True),commission)
        v={k:labels(measured_frame(x,self.mode),x) for k,x in val.items()}
        # Load-warning mode does not use zone_causal_median; placeholders
        # are replaced with the newly refitted causal model after fitting.
        self.core.fit_normal(t,v)
        t=prepare_zonal_warning_context(t,self.core.causal)
        v={k:prepare_zonal_warning_context(x,self.core.causal) for k,x in v.items()}
        self.audit.fit(t,v)
        count=CountReference().fit(t)
        bounded=BoundedCountReference(count).fit(t)
        memory=AdaptiveMemoryReference(bounded).fit(t)
        conditional=ConditionalCountReference(memory).fit(t,self.core,self.audit)
        block=BlockReference(conditional).fit(t,self.core,self.audit)
        personal=CommissionedReference(block).fit(t,c)
        self.reference=MeasuredPhysicalReference(personal).fit(t)
        return self
    def prepare(self,df,commission):
        x=measured_frame(df,self.mode)
        x=prepare_zonal_warning_context(x,self.core.causal)
        c=measured_frame(commission,self.mode,True)
        return self.reference.prepare(x,c)
    def manifest(self):
        return {"mode":self.mode,"full_refit":True,"oracle_only":self.mode=="oracle",
                "core":self.core.manifest(),"all_original_invariants_preserved":True,
                "raw_observation_columns":RAW,"inactive_old_context_removed":True,
                "fallback":"causal CU delivery frequency","measurement_variance_used":True}

class MeasuredFusion(LossFusionTICD):
    def __init__(self,mode,window,budgets,calibration_mode):
        super().__init__("cu",window,budgets,calibration_mode)
        self.observation_mode=mode
        self.warning_budget=.002
    def manifest(self):
        d=super().manifest()
        d.update({"observation_mode":self.observation_mode,"oracle_only":self.observation_mode=="oracle",
                  "synthetic_measurement":self.observation_mode in ("active","passive","hybrid")})
        return d
