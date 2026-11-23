"""TICD with *frozen training-reference score normalization*.

Keeps the original published TICD invariant, robust context and consensus
components, but fixes a reproducibility flaw: the previous score() used
percentiles from the batch being scored, and _consensus_score used its
standard deviation, so each unseen test batch changed its own score scale.

All additional parameters below are fitted on NORMAL training data only.
Attack labels and validation/test data never enter the score scale.
"""
import numpy as np
from ticd import TICDetector

class FrozenReferenceTICD(TICDetector):
    CONSENSUS_FEATURES=(
        "dev_total_messages_vs_zone","dev_io_ratio_vs_zone",
        "dev_var_total_vs_zone"
    )

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.consensus_scales={}
        self.component_reference={}
        self.min_covariance_rows=1000
        self.max_covariance_rows=2200

    def fit(self,train_df,val_df=None):
        norm=train_df[train_df[self.label_col]==0]
        if len(norm)<200:raise ValueError("Need at least 200 normal training rows")
        # Original MinCovDet is robust but costly on very large arrays.
        # Subsample only *normal training rows*, with deterministic seed.
        cov_rows=norm.sample(min(len(norm),self.max_covariance_rows),
                             random_state=self.seed)
        super().fit(cov_rows)
        # Recompute robust invariant scale from the WHOLE normal training corpus.
        for c in self.inv_cols:
            v=norm[c].to_numpy(dtype=float)
            v=v[np.isfinite(v)]
            self.inv_norm[c]=max(float(np.percentile(np.abs(v),99)) if len(v) else 1.,1e-6)
        for c in self.CONSENSUS_FEATURES:
            if c in norm.columns:
                v=norm[c].to_numpy(dtype=float)
                v=v[np.isfinite(v)]
                self.consensus_scales[c]=max(float(np.std(v)),1e-6) if v.size else 1.
        # All reference quantiles are frozen, never recomputed on validation/test.
        for name,value in self._components(norm).items():
            value=np.nan_to_num(value,nan=0.,posinf=1e12,neginf=0.)
            self.component_reference[name]=(
                float(np.percentile(value,1)),float(np.percentile(value,99)))
        if not self.inv_cols or "cu_ack_closure_mismatch" not in self.inv_cols:
            raise ValueError("Missing the SIP I2 invariant")
        self.fitted=True
        return self

    def _consensus_score(self,df):
        cols=[c for c in self.CONSENSUS_FEATURES if c in df.columns]
        if not cols:return np.zeros(len(df))
        total=np.zeros(len(df),dtype=float)
        for c in cols:
            v=np.nan_to_num(df[c].to_numpy(dtype=float),nan=0.)
            total+=np.clip(np.abs(v)/self.consensus_scales.get(c,1.),0,5.)
        return total/len(cols)

    def _components(self,df):
        return {
            "invariant":self._invariant_score(df),
            "context":self._context_score(df),
            "consensus":self._consensus_score(df)
        }

    def score_components(self,df):
        return self._components(df)

    def score(self,df):
        if not self.fitted:raise RuntimeError("Fit before scoring")
        def frozen(name,arr):
            lo,hi=self.component_reference[name]
            arr=np.nan_to_num(arr,nan=0.,posinf=1e12,neginf=0.)
            return np.clip((arr-lo)/max(hi-lo,1e-9),0,1)
        s=self._components(df)
        return (self.w_inv*frozen("invariant",s["invariant"])+
                self.w_ctx*frozen("context",s["context"])+
                self.w_cons*frozen("consensus",s["consensus"]))
