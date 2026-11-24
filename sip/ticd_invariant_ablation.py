"""Controlled leave-one-component-out ablation of the frozen role-adaptive TICD.

Input features are already causal and zone-aligned. Mask only the explicit
evidence path(s), never the simulated traffic or attack labels. The D-family
ablation removes D, its spatial residual and every ACK EWMA downstream.
I1-family ablation removes signed I1, its historical and spatial derivatives.
I2 is distinct from master-only D; the all-ACK ablation removes both families.
Both fixed-reference and identically refitted normal-reference policies are
reported. No variant selection on the test set.
"""
from __future__ import annotations
import copy
import numpy as np
from ticd_causal_ack import HISTORY_FIELDS

VARIANTS=("full","no_i1_deficit","no_i1_excess","no_i1_instant","no_i1_history","no_i1_all","no_i2",
          "no_d_instant","no_d_zone","no_d_history","no_d_all","no_ack_all","no_all_invariants",
          "no_gateway_weight","no_warning_load","no_warning_scope","no_warning")

class InvariantAblationTICD:
    def __init__(self,source,variant="full",refit=False):
        if variant not in VARIANTS: raise ValueError(variant)
        self.model=copy.deepcopy(source)
        self.variant=variant
        self.refit=bool(refit)
        self.high=self.model.prior.prior
        self.role=self.high.prior
        self.zonal=self.role.source_model
        self.causal=self.zonal.causal_model
        self.loss=self.causal.base_loss_model
        self.physical=self.loss.physical_model
        # The contextual alternative includes raw report features and would
        # require a different dependency audit. Fail rather than silently
        # retain an indirect invariant path.
        if self.physical.physical_mode not in ("physical","physical_zone"):
            raise RuntimeError("Unexpected active physical mode: "+self.physical.physical_mode)
        if self.model.prior.mode!="max" or self.causal.history!="short":
            raise RuntimeError("Unexpected fusion architecture")
        if variant=="no_gateway_weight":self.model.gateway_weight=1.
        if variant=="no_warning_load":self.zonal.mode="baseline"
        if variant=="no_warning_scope":self.role.eligible_roles=None
        self.threshold=self.model.threshold
        self.warning_threshold=self.role.warning_threshold
        self.degenerate_instant_reference=False
    def transform(self,df):
        v=self.variant
        if v in ("full","no_gateway_weight","no_warning_load","no_warning_scope","no_warning"):
            return df
        x=df.copy()
        if v in ("no_i1_deficit","no_i1_excess"):
            signed=1.-x["obs_dialog_completion_ratio"]-x["obs_dangling_dialog_ratio"]
            mask=(signed>0) if v=="no_i1_deficit" else (signed<0)
            x.loc[mask,["obs_dialog_completion_ratio","obs_dangling_dialog_ratio"]]=.5
        if v in ("no_i1_instant","no_i1_all","no_all_invariants"):
            # Set both report fractions to exact complementary values; retain the
            # precomputed I2 mismatch. Avoid floating point cancellation by
            # setting both report fractions to exact binary halves.
            x["obs_dialog_completion_ratio"]=.5
            x["obs_dangling_dialog_ratio"]=.5
        if v in ("no_i1_history","no_i1_all","no_all_invariants"):
            x["mean_txn_state_consistency_error"]=0.
        if v in ("no_i1_all","no_all_invariants"):
            x["dev_txn_consistency_vs_zone"]=0.
            x["cu_txn_state_consistency_error"]=0.
        if v in ("no_i2","no_ack_all","no_all_invariants"):
            x["cu_ack_closure_mismatch"]=0.
            if "dev_ack_mismatch_vs_zone" in x:x["dev_ack_mismatch_vs_zone"]=0.
        if v in ("no_d_instant","no_d_all","no_ack_all","no_all_invariants"):
            x["master_ack_shortfall_adjusted"]=0.
        if v in ("no_d_zone","no_d_all","no_ack_all","no_all_invariants"):
            x["dev_master_ack_shortfall_vs_zone"]=0.
        if v in ("no_d_history","no_d_all","no_ack_all","no_all_invariants"):
            for c in HISTORY_FIELDS.values():x[c]=0.
            x["zone_causal_median"]=0.
        return x
    def score(self,df):
        return self.model.score(self.transform(df))
    def warning(self,df):
        x=self.transform(df)
        if self.variant=="no_warning":
            return np.zeros(len(x)),np.zeros(len(x),dtype=bool)
        return self.role.score_channels(x)["warning"],self.role.eligible_mask(x)
    def fit_normal(self,train,validation,alert_budget=.008,warning_budget=.005):
        if not self.refit:raise RuntimeError("Frozen policy cannot fit")
        if (train.is_anomaly!=0).any() or any((x.is_anomaly!=0).any() for x in validation.values()):
            raise ValueError("Nonbenign reference")
        t=self.transform(train)
        val={m:self.transform(x) for m,x in validation.items()}
        self.physical.fit(t)
        self.loss.fit(t)
        self.causal.fit(t)
        self.zonal.fit(t)
        # These estimators assume a fresh object: discard inherited groups
        # before refitting to avoid stale sparse-group references.
        self.high.thresholds.clear()
        self.high.group_counts.clear()
        self.high.fit(t)
        self.high.calibrate(val,alert_budget)
        self.degenerate_instant_reference=False
        if self.high.scale==0.:
            # The all-invariant null has identically zero evidence, hence
            # its empirical quantile is zero. A unit divisor represents the
            # zero score exactly, without fabricating an anomaly signal.
            # Fail closed on any unexpected degenerate non-null variant.
            assert self.variant=="no_all_invariants", (self.variant,self.high.scale)
            assert np.all(self.high.normalized(t)==0.)
            assert all(np.all(self.high.normalized(x)==0.) for x in val.values())
            self.high.scale=1.
            self.degenerate_instant_reference=True
        assert np.isfinite(self.high.scale) and self.high.scale>0.
        self.model.prior.reference.groups_q.clear()
        self.model.prior.reference.counts.clear()
        self.model.prior.reference.fit(t)
        self.model.calibrate(val,alert_budget)
        self.threshold=self.model.threshold
        # Match the warning budget using all normal records as denominator.
        # Eligible non-alert warning scores are zeroed outside the mask.
        qs=[]
        for x in validation.values():
            a=self.score(x)>self.threshold
            w,e=self.warning(x)
            masked=np.where(e&~a,w,0.)
            qs.append(float(np.quantile(masked,1-warning_budget,method="higher")))
        self.warning_threshold=max(qs)
        self.role.warning_threshold=self.warning_threshold
        if self.variant=="no_all_invariants":
            assert self.degenerate_instant_reference
            assert self.threshold==self.warning_threshold==0.
            assert not self.score(train).any()
            assert not self.warning(train)[0].any()
        for x in validation.values():
            p=self.predict_masks(x)
            assert p["confirmed_alert"].mean()<=alert_budget+1e-12
            assert p["warning_only"].mean()<=warning_budget+1e-12
        return self
    def predict_levels(self,df):
        a=self.score(df)>self.threshold
        w,e=self.warning(df)
        warn=(w>self.warning_threshold)&e&~a
        y=np.zeros(len(df),dtype=np.int8)
        y[warn]=1;y[a]=2
        return y
    def predict_masks(self,df):
        y=self.predict_levels(df)
        return {"normal":y==0,"warning_only":y==1,"confirmed_alert":y==2,"any_signal":y>0}
    def manifest(self):
        return {"variant":self.variant,"refit":self.refit,"threshold":float(self.threshold),
                "warning_threshold":float(self.warning_threshold),"degenerate_instant_reference":self.degenerate_instant_reference,"instant_scale":float(self.high.scale),"physical_mode":self.physical.physical_mode,
                "gateway_weight":self.model.gateway_weight,"warning_mode":self.zonal.mode,
                "warning_roles":self.role.eligible_roles,
                "zone_D_active_in_full":self.physical.physical_mode=="physical_zone",
                "all_ack_removed":self.variant in ("no_ack_all","no_all_invariants")}
