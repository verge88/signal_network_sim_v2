"""Causal persistence of independently master-observed SIP ACK closure defects.

Requires the usual trusted transit 200 OK/ACK counters from
prepare_master_closure_features, with no access to attack labels, gt_ fields,
future timestamps or reported compromised-node state. Per-node exponential
moving average is generated in chronological order and can be recreated in
a streaming online detector; scoring assumes this feature is already present.

The normalisation is fitted ONLY on benign simulations via the existing
LossConditionedTICD (CU path loss bin, known operator node type). The original
physical TICD and its previously frozen weights are not changed.

All parameters and scores are functions solely of master-observable inputs.
No statistical promise about live SIP networks follows from these simulated
ACK/200 event profiles.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ticd_loss_conditioned import LossConditionedTICD

HISTORY_FIELDS={"short":"ack_shortfall_ewma_6",
                "medium":"ack_shortfall_ewma_16",
                "long":"ack_shortfall_ewma_32"}
SPAN={"short":6,"medium":16,"long":32}

def add_causal_ack_history(df:pd.DataFrame)->pd.DataFrame:
    required=["master_ack_shortfall_adjusted","interval","node_id",
              "cu_delivered"]
    missing=[c for c in required if c not in df]
    if missing:
        raise ValueError("Cannot construct master-side causal history: "+str(missing))
    keys=(["simulation_seed"] if "simulation_seed" in df else [])+["node_id"]
    chronological=df.sort_values(keys+["interval"],kind="stable")
    if chronological.duplicated(keys+["interval"]).any():
        raise ValueError("Duplicate time-node observations in ACK history")
    output=chronological.copy()
    raw=output["master_ack_shortfall_adjusted"]
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("Master shortfall non-finite")
    if not (raw>=0).all():
        raise ValueError("Master shortfall negative")
    # Missing reports are recorded as zero evidence; the absence of CU data
    # cannot by itself fabricate a positive ACK anomaly. Data availability
    # must be separately monitored in operational telemetry.
    for label,column in HISTORY_FIELDS.items():
        window=SPAN[label]
        output[column]=(output.groupby(keys,sort=False)["master_ack_shortfall_adjusted"]
                        .transform(lambda v:v.ewm(span=window,adjust=False).mean()))
    return output.sort_index()

class FrozenColumnEvidence:
    """Pickle-safe source for a precomputed, master-only causal statistic."""
    fitted=True
    def __init__(self,column):
        if column not in HISTORY_FIELDS.values():
            raise ValueError("Unsupported historical signal "+column)
        self.column=column
    def score(self,df):
        if self.column not in df:
            raise ValueError("Missing causal master ACK history: "+self.column)
        values=df[self.column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite streaming history")
        return np.maximum(values,0.)

class CausalPersistentACKTICD:
    """Max evidence across existing frozen TICD and one causal ACK channel."""
    CANDIDATES=("short","medium","long")
    def __init__(self,base_loss_model,history="medium",weight=0.8):
        if history not in HISTORY_FIELDS:
            raise ValueError("Invalid causal history span")
        if not 0.1<=float(weight)<=2.5:
            raise ValueError("Invalid history evidence weight")
        self.base_loss_model=base_loss_model
        self.history=history
        self.weight=float(weight)
        self.persistence_ref=LossConditionedTICD(
            FrozenColumnEvidence(HISTORY_FIELDS[history]),
            mode="loss_node",bins=8,min_group=120)
        self.fitted=False
    def fit(self,normal_train):
        if "is_anomaly" in normal_train and (normal_train["is_anomaly"]!=0).any():
            raise ValueError("Cannot fit causal ACK reference on attacks")
        if not self.base_loss_model.fitted:
            raise RuntimeError("Original loss conditioned model must be frozen")
        self.persistence_ref.fit(normal_train)
        self.fitted=True
        return self
    def score_components(self,df):
        if not self.fitted:
            raise RuntimeError("Fit normal-only causal reference before scoring")
        base=np.asarray(self.base_loss_model.score(df),dtype=float)
        history=np.asarray(self.persistence_ref.score(df),dtype=float)
        if not np.isfinite(base).all() or not np.isfinite(history).all():
            raise ValueError("Nonfinite TICD evidence")
        return {"instant":base,"causal":self.weight*history}
    def score(self,df):
        c=self.score_components(df)
        return np.maximum(c["instant"],c["causal"])
    def reference_manifest(self):
        return {
            "original_physical_model_unchanged":True,
            "history":self.history,
            "ewma_span":SPAN[self.history],
            "weight":self.weight,
            "causal_feature":HISTORY_FIELDS[self.history],
            "reference":self.persistence_ref.reference_manifest(),
            "training_labels":False}
