"""Two-tier SIP TICD: separate *confirmed alerts* and *early warnings*.

Confirmed alert: independently fitted, frozen LossConditionedTICD physical
ACK/invariant score, with a conservative benign validation threshold.
Early warning: separately thresholded, *causal-only* ACK history evidence
from a previously frozen CausalPersistentACKTICD. It is reported only when
no confirmed alert exists. No warning is counted as a confirmed detection.

Both thresholds are fitted to SEPARATE BENIGN validation simulations and
never changed by the contents of the scored test batch. Warning selection is
based exclusively on SYNTHETIC development attacks. All required network
observations must be master-observable with online causal history.

The detector cannot infer a real compromise label from isolated SIP PCAP.
"""
from __future__ import annotations
import numpy as np
from ticd_causal_ack import CausalPersistentACKTICD

class TwoTierSIPTICD:
    LEVELS={0:"normal",1:"warning",2:"confirmed_alert"}

    def __init__(self,causal_model,confirmed_threshold,warning_threshold):
        if not isinstance(causal_model,CausalPersistentACKTICD) or not causal_model.fitted:
            raise TypeError("Expected already normal-trained frozen causal TICD")
        if not np.isfinite(float(confirmed_threshold)) or float(confirmed_threshold)<0:
            raise ValueError("Confirmed threshold must be nonnegative and finite")
        if not np.isfinite(float(warning_threshold)) or float(warning_threshold)<0:
            raise ValueError("Warning threshold must be nonnegative and finite")
        self.causal_model=causal_model
        self.confirmed_model=causal_model.base_loss_model
        self.confirmed_threshold=float(confirmed_threshold)
        self.warning_threshold=float(warning_threshold)
        self.fitted=True

    def score_channels(self,df):
        components=self.causal_model.score_components(df)
        instant=np.asarray(components["instant"],dtype=float)
        persistent=np.asarray(components["causal"],dtype=float)
        if len(instant)!=len(df) or len(persistent)!=len(df):
            raise RuntimeError("Unexpected score-channel dimensions")
        if not np.isfinite(instant).all() or not np.isfinite(persistent).all():
            raise ValueError("Non-finite trusted master-side evidence")
        return {"confirmed_evidence":instant, "warning_evidence":persistent}

    def predict_levels(self,df):
        parts=self.score_channels(df)
        confirmed=parts["confirmed_evidence"]>self.confirmed_threshold
        warn=(parts["warning_evidence"]>self.warning_threshold)&(~confirmed)
        levels=np.zeros(len(df),dtype=np.int8)
        levels[warn]=1
        levels[confirmed]=2
        return levels

    def predict_masks(self,df):
        levels=self.predict_levels(df)
        return {
          "normal":levels==0,
          "warning_only":levels==1,
          "confirmed_alert":levels==2,
          "any_signal":levels>=1,
        }

    def manifest(self):
        return {
          "confirmed_threshold":self.confirmed_threshold,
          "warning_threshold":self.warning_threshold,
          "confirmed_score":"frozen LossConditionedTICD physical/instantaneous",
          "warning_score":"normal-fitted causal ACK EWMA only",
          "warning_does_not_count_as_confirmed_alert":True,
          "online_retraining":False,
          "test_batch_scoring_normalization":False,
          "levels":self.LEVELS
        }
