"""Zonal-context warning branch for two-tier SIP TICD.

Confirmed alarms remain the frozen master-observable LossConditionedTICD
channel. Warnings use the frozen causal ACK persistence channel, but can be
discounted when neighboring nodes share a similar causal disturbance or
when robust zone-level REPORTED SIP load increases. The load median is
possibly adversary-influenced: it is NOT a trusted physical counter.
A compromised majority of a small zone can poison the load comparator.

The same-time zonal reference is a complete snapshot prepared BEFORE scoring.
Scoring a row does not derive any references from the test batch or future.
No attack labels, gt_* fields, compromised flags, event schedules or ground
truth are used in this file. Conditional score calibration must be from
independent benign normal datasets (in the experimental workflow).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ticd_causal_ack import CausalPersistentACKTICD

def prepare_zonal_warning_context(df:pd.DataFrame,causal_model:CausalPersistentACKTICD):
    """Precompute simultaneous zone reference using only in-band observations.

    A production collector must assemble a complete aligned zone snapshot
    (same timestamp and zone), then store the result alongside each record.
    Calling this on an incomplete test subset would change the zone reference
    and is explicitly forbidden for the experiment.
    """
    needed=("simulation_seed","interval","node_id","zone_id",
            "obs_total_messages","cu_combined_loss_prob")
    missing=[k for k in needed if k not in df.columns]
    if missing:raise ValueError("Missing telemetry to form zone reference: "+repr(missing))
    if df.duplicated(["simulation_seed","node_id","interval"]).any():
        raise ValueError("Duplicate node-time observation")
    if not isinstance(causal_model,CausalPersistentACKTICD) or not causal_model.fitted:
        raise TypeError("Expected a frozen normal-trained master ACK model")
    parts=causal_model.score_components(df)
    causal=np.asarray(parts["causal"],dtype=float)
    reported=df["obs_total_messages"].to_numpy(dtype=float)
    if not np.isfinite(causal).all() or not np.isfinite(reported).all():
        raise ValueError("Non-finite zone source")
    if (reported<0).any():raise ValueError("Negative reported SIP load")
    output=df.copy()
    output["_zone_causal_score"]=causal
    keys=["simulation_seed","interval","zone_id"]
    grp=output.groupby(keys,sort=False)
    output["zone_causal_median"]=grp["_zone_causal_score"].transform("median")
    output["zone_reported_load_median"]=grp["obs_total_messages"].transform("median")
    output["zone_observed_loss_median"]=grp["cu_combined_loss_prob"].transform("median")
    output["zone_member_count"]=grp["node_id"].transform("size")
    output.drop(columns=["_zone_causal_score"],inplace=True)
    return output

class ZoneContextTwoTierTICD:
    """Mutually exclusive alert (2), early warning (1), normal (0)."""
    VALID_MODES=("baseline","peer","load","joint")
    REQUIRED=("zone_causal_median","zone_reported_load_median",
              "zone_observed_loss_median","zone_member_count")
    def __init__(self,causal_model,mode="baseline",strength=0.5):
        if not isinstance(causal_model,CausalPersistentACKTICD) or not causal_model.fitted:
            raise TypeError("Expected frozen causal ACK TICD")
        if mode not in self.VALID_MODES:raise ValueError(mode)
        if not (0.<=float(strength)<=2.):
            raise ValueError("Invalid zone context coefficient")
        self.causal_model=causal_model
        self.mode=mode
        self.strength=float(strength)
        self.reference_load=None
        self.confirmed_threshold=None
        self.warning_threshold=None
        self.fitted=False

    def fit(self,clean_train):
        if "is_anomaly" in clean_train and (clean_train["is_anomaly"]!=0).any():
            raise ValueError("Zone normal reference must not contain attack records")
        if len(clean_train)<1000:raise ValueError("Too few normal reference rows")
        for key in self.REQUIRED:
            if key not in clean_train:raise ValueError("Missing zone context "+key)
        vals=clean_train["zone_reported_load_median"].to_numpy(dtype=float)
        if not np.isfinite(vals).all() or (vals<0).any():
            raise ValueError("Invalid reference zone load")
        self.reference_load=max(float(np.median(vals)),1.0)
        self.fitted=True
        return self

    def score_channels(self,df):
        if not self.fitted:raise RuntimeError("Fit normal-only zone reference first")
        for col in self.REQUIRED:
            if col not in df:raise ValueError("Zone snapshot field required: "+col)
        parts=self.causal_model.score_components(df)
        instant=np.asarray(parts["instant"],dtype=float)
        causal=np.asarray(parts["causal"],dtype=float)
        med=df["zone_causal_median"].to_numpy(dtype=float)
        load=df["zone_reported_load_median"].to_numpy(dtype=float)
        members=df["zone_member_count"].to_numpy(dtype=int)
        if not np.isfinite(med).all() or not np.isfinite(load).all():
            raise ValueError("Nonfinite normalizable zone context")
        # One-node zones have no meaningful cross-node consensus.
        peer_discount=np.maximum(causal-self.strength*np.maximum(med,0.),0.)
        peer_adjusted=np.where(members>=2,peer_discount,causal)
        # Only excess above normal median reported load is discounted;
        # robust median relies on fewer than half zone nodes compromised.
        load_excess=np.maximum(load/self.reference_load-1.,0.)
        load_scale=1.+self.strength*load_excess
        if self.mode=="baseline":
            warning=causal
        elif self.mode=="peer":
            warning=peer_adjusted
        elif self.mode=="load":
            warning=causal/load_scale
        else:
            warning=peer_adjusted/load_scale
        if not np.isfinite(warning).all():
            raise ValueError("Nonfinite warning scores")
        return {"instant":instant,"warning":warning}

    def set_thresholds(self,alert_threshold,warning_threshold):
        if not self.fitted:raise RuntimeError("Fit before freezing thresholds")
        for x in (alert_threshold,warning_threshold):
            if not np.isfinite(float(x)) or float(x)<0.:
                raise ValueError("Threshold must be finite and nonnegative")
        self.confirmed_threshold=float(alert_threshold)
        self.warning_threshold=float(warning_threshold)
        return self

    def predict_levels(self,df):
        if self.confirmed_threshold is None or self.warning_threshold is None:
            raise RuntimeError("Decision thresholds are not frozen")
        s=self.score_channels(df)
        alert=s["instant"]>self.confirmed_threshold
        warn=(~alert)&(s["warning"]>self.warning_threshold)
        levels=np.zeros(len(df),dtype=np.int8)
        levels[warn]=1
        levels[alert]=2
        return levels

    def predict_masks(self,df):
        levels=self.predict_levels(df)
        return {"normal":levels==0,"warning_only":levels==1,
                "confirmed_alert":levels==2,"any_signal":levels>=1}

    def manifest(self):
        return {
          "mode":self.mode,"strength":self.strength,
          "reference_load_fitted_only_on_clean_train":self.reference_load,
          "confirmed_threshold":self.confirmed_threshold,
          "warning_threshold":self.warning_threshold,
          "warn_status_is_not_confirmed":True,
          "context_from_complete_same_time_zone_snapshot":True,
          "zone_reported_load_can_be_poisoned":True,
          "underlying_loss_and_causal_models_frozen":True
        }
