"""Independent master ACK-closure anomaly channel for SIP TICD.

The original deterministic I1 / I2 misreporting invariants cannot detect a
compromised proxy that reports its changed traffic state truthfully.
This optional *statistical* path compares independently counted physical
ACK closure with a benign reference and simultaneously observed zone peers.
The physical missing-ACK rate is NOT an RFC conservation identity; legitimate
loss and normal abnormal call endings can change it. Use normal-only fitted
scales, network-impairment adjustment and an independent validation FPR.

The zone feature is prepared at the SAME interval from master-side transit
observations. Never use compromised node oracle, gt_, true_traffic, labels
or a future timestamp to compute any score.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ticd_temporal_invariants import MultiScaleInvariantTICD

def prepare_master_closure_features(df:pd.DataFrame):
    """Prepare same-time master zone comparator without looking at labels.

    Assumes one row per node and interval per simulation (or observation run).
    In a deployment this function must be evaluated on complete, time-aligned
    zone snapshots before individual rows are passed to the detector.
    """
    required=("cu_dcr_transit","cu_combined_loss_prob","cu_delivered",
              "zone_id","interval","node_id")
    absent=[c for c in required if c not in df]
    if absent:raise ValueError("Missing master observation columns: "+repr(absent))
    out=df.copy()
    keys=(["simulation_seed"] if "simulation_seed" in out else [])+["interval","zone_id"]
    if out.duplicated((["simulation_seed"] if "simulation_seed" in out else [])+["interval","node_id"]).any():
        raise ValueError("Duplicate same-time node observations")
    observed=np.nan_to_num(out["cu_dcr_transit"].to_numpy(dtype=float),nan=1.)
    loss=np.maximum(np.nan_to_num(out["cu_combined_loss_prob"].to_numpy(dtype=float),nan=0.),0.)
    delivered=(np.nan_to_num(out["cu_delivered"].to_numpy(dtype=float),nan=0.)>0)
    # Subtract master-observed link loss as a conservative traffic-path
    # allowance. The coefficient is fixed ex ante (=1.0) not learned from
    # attack labels or optimized using test metrics.
    deficit=np.maximum(0.,1.-observed-loss)*delivered
    out["master_ack_shortfall_adjusted"]=deficit
    med=out.groupby(keys,sort=False)["master_ack_shortfall_adjusted"].transform("median")
    out["dev_master_ack_shortfall_vs_zone"]=(out["master_ack_shortfall_adjusted"]-med).clip(lower=0.)
    return out

class PhysicalClosureTICD(MultiScaleInvariantTICD):
    PHYSICAL_MODES=("physical","physical_zone","physical_zone_context")
    EXTRA=("master_ack_shortfall_adjusted","dev_master_ack_shortfall_vs_zone")

    def __init__(self,base_model,mode="physical_zone",normal_quantile=99.0):
        # Invariant/temporal channel's same normal-only calibration.
        super().__init__(base_model=base_model,mode="temporal",normal_quantile=normal_quantile)
        if mode not in self.PHYSICAL_MODES:raise ValueError(mode)
        self.physical_mode=mode

    def fit(self,train_df):
        super().fit(train_df)
        for name in self.EXTRA:
            if name not in train_df:
                raise ValueError(f"Missing precomputed trusted master-zone feature: {name}")
            v=np.maximum(np.nan_to_num(train_df[name].to_numpy(dtype=float),nan=0.),0.)
            self.scales[name]=max(float(np.percentile(v,self.normal_quantile)),1e-3)
        return self

    def score(self,df):
        for c in self.EXTRA:
            if c not in df:raise ValueError("Master physical closure snapshot missing: "+c)
        temporal=super().score(df)
        physical=np.clip(np.maximum(np.nan_to_num(df[self.EXTRA[0]].to_numpy(dtype=float),nan=0.),0.)
                         /self.scales[self.EXTRA[0]],0.,50.)
        if self.physical_mode=="physical":
            return np.maximum(temporal,physical)
        zone=np.clip(np.maximum(np.nan_to_num(df[self.EXTRA[1]].to_numpy(dtype=float),nan=0.),0.)
                     /self.scales[self.EXTRA[1]],0.,50.)
        if self.physical_mode=="physical_zone":
            return np.maximum.reduce((temporal,physical,0.9*zone))
        contextual=np.clip(self.base_model.score(df),0.,1.)
        return np.maximum.reduce((temporal,physical,0.9*zone,0.8*contextual))
