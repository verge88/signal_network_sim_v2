"""Multi-timescale SIP transaction-invariant TICD (normal-reference fit).

Architecture:
  * signed I1 closure deficit and excess paths (normal RFC 3261 state);
  * independently observed positive I2 ACK-closure discrepancy;
  * 30-observation causal integrity-error average computed by the master;
  * zone-centered I1 error to discount network-wide changes;
  * original fixed-reference TICD as a separate contextual channel.

No is_anomaly, attack_type, gt_ fields or future samples are ever read
during scoring. All scales are fitted on normal TRAIN simulations. "Temporal"
uses a row-level causal running summary already calculated by the original
SIP simulator (mean_txn_state_consistency_error), not batch rolling;
therefore scores do not depend on what other test records are supplied.

Original TICD can be scored as an independent baseline without alterations.
"""
import numpy as np

class MultiScaleInvariantTICD:
    REQUIRED=("obs_dialog_completion_ratio","obs_dangling_dialog_ratio",
              "cu_ack_closure_mismatch","mean_txn_state_consistency_error",
              "dev_txn_consistency_vs_zone","cu_delivered")

    CANDIDATES=("instant","temporal","temporal_zone","temporal_hybrid")

    def __init__(self,base_model,mode="temporal_hybrid",normal_quantile=99.0):
        if mode not in self.CANDIDATES:
            raise ValueError("Unsupported mode: "+mode)
        self.base_model=base_model
        self.mode=mode
        self.normal_quantile=float(normal_quantile)
        self.scales={}
        self.fitted=False

    def _raw(self,df):
        absent=[c for c in self.REQUIRED if c not in df.columns]
        if absent:
            raise ValueError("Missing master-side SIP observations: "+str(absent))
        dcr=np.nan_to_num(df["obs_dialog_completion_ratio"].to_numpy(dtype=float),
                          nan=0.,posinf=0.,neginf=0.)
        ddr=np.nan_to_num(df["obs_dangling_dialog_ratio"].to_numpy(dtype=float),
                          nan=0.,posinf=0.,neginf=0.)
        delivery=(np.nan_to_num(df["cu_delivered"].to_numpy(dtype=float))>0)
        signed=1.-dcr-ddr
        # Distinguish missing closure from double-accounted closure;
        # naive fake report is often excess, statistical/adaptive deficit.
        return {
          "i1_deficit":np.maximum(signed,0.)*delivery,
          "i1_excess":np.maximum(-signed,0.)*delivery,
          "i2_ack":np.maximum(np.nan_to_num(
               df["cu_ack_closure_mismatch"].to_numpy(dtype=float),nan=0.),0.)*delivery,
          "temporal_i1":np.maximum(np.nan_to_num(
               df["mean_txn_state_consistency_error"].to_numpy(dtype=float),nan=0.),0.),
          "zone_i1":np.maximum(np.nan_to_num(
               df["dev_txn_consistency_vs_zone"].to_numpy(dtype=float),nan=0.),0.)*delivery,
        }

    def fit(self,train_df):
        if not self.base_model.fitted:
            raise ValueError("Frozen-reference base detector must be fitted first")
        if len(train_df)==0:raise ValueError("Empty reference")
        if "is_anomaly" in train_df and (train_df["is_anomaly"]!=0).any():
            raise ValueError("Non-normal examples forbidden in fitting")
        for name,value in self._raw(train_df).items():
            v=value[np.isfinite(value)]
            if v.size==0:raise ValueError(f"No data for {name}")
            # Safeguard tiny benign I2 quantiles, avoid division by zero.
            self.scales[name]=max(float(np.percentile(v,self.normal_quantile)),1e-3)
        self.fitted=True
        return self

    def evidence_components(self,df):
        if not self.fitted:raise RuntimeError("Fit before score")
        r=self._raw(df)
        return {k:np.clip(v/self.scales[k],0.,50.) for k,v in r.items()}

    def score(self,df):
        r=self.evidence_components(df)
        instant=np.maximum.reduce((r["i1_deficit"],r["i1_excess"],r["i2_ack"]))
        if self.mode=="instant":
            return instant
        # Temporal persistence reduces one-frame noise; stronger persistent
        # I1 departures are compared directly to independent normal windows.
        temporal=np.maximum(instant,0.85*r["temporal_i1"])
        if self.mode=="temporal":
            return temporal
        zone=np.maximum(temporal,0.80*r["zone_i1"])
        if self.mode=="temporal_zone":
            return zone
        # Preserve contextual anomalies from original TICD, but physical
        # evidence dominates when sustained state/ACK invariants break.
        baseline=np.clip(self.base_model.score(df),0.,1.)
        return np.maximum(zone,0.8*baseline)
