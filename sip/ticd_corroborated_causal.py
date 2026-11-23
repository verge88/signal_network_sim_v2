"""SIP TICD dual-evidence decision gate.

The previous causal ACK persistence model improved weak compromise recall
in the simulator, but caused nearly 2% false alarms in an independent normal
default scenario. Require corroborating instantaneous physical evidence
before a large persistent score can trigger an alert.

Given a pre-fitted CausalPersistentACKTICD, no new feature weights or
normal reference distributions are learnt. The gate bounds the persistent
contribution by instantaneous master-observed evidence:
  combined = max(instant, min(causal, instant / gate)).
Higher gate => stricter agreement. 'instant' is the frozen loss-conditioned
physical TICD score, 'causal' its independently observed 6-window history.

The gate is selected using separately labelled SYNTHETIC development data.
The alert threshold is calibrated on independent BENIGN validation.
Neither is adapted from the held-out test batch or test labels.
"""
from __future__ import annotations
import numpy as np
from ticd_causal_ack import CausalPersistentACKTICD

class CorroboratedCausalTICD:
    def __init__(self,causal_model,gate=0.6):
        if not isinstance(causal_model,CausalPersistentACKTICD):
            raise TypeError("Expected a pre-fitted causal ACK model")
        if not causal_model.fitted:
            raise ValueError("Fit the causal model on benign training data first")
        if not (0.05<=float(gate)<=1.):
            raise ValueError("Evidence agreement gate must lie between 0.05 and 1.0")
        self.causal_model=causal_model
        self.gate=float(gate)
        self.fitted=True
    def score(self,df):
        components=self.causal_model.score_components(df)
        instant=components["instant"]
        causal=components["causal"]
        return np.maximum(instant,np.minimum(causal,instant/self.gate))
