"""Loss-conditional, NORMAL-reference calibration for the SIP TICD score.

Purpose: the previous fixed physical closure threshold, calibrated on benign
low-loss traffic, had 4.12% FPR under a 5x loss shift in the SIP simulator.

This adapter learns normal-score 99th percentiles conditional on the current
MASTER-observed SIP-proxy loss covariate (cu_combined_loss_prob), optionally
also on node type. It does NOT refit or inspect compromised node reports,
ground truth, attack labels, or production/test score batches. The upstream
PhysicalClosureTICD is a separately frozen detector. This adapter does not
learn from unlabelled test data, since an attacker could poison it.

Caveat: cu_combined_loss_prob is a proxy, based on the CU polling path, not
an independent measurement of SIP media loss; the simulator makes these paths
related. Scores therefore require calibration with varied benign CU path loss,
and independent field validation before claiming operational robustness.
"""
from __future__ import annotations
import numpy as np

class LossConditionedTICD:
    MODES=("loss", "loss_node", "loss_node_shrink")

    def __init__(self, physical_model, mode="loss_node_shrink",
                 normal_quantile=99., bins=8, min_group=120):
        if mode not in self.MODES:
            raise ValueError(f"Unsupported model adaptation {mode}")
        self.physical_model=physical_model
        self.mode=mode
        self.normal_quantile=float(normal_quantile)
        self.bins=int(bins)
        self.min_group=int(min_group)
        self.edges=None
        self.global_q99=None
        self.bin_q99={}
        self.node_bin_q99={}
        self.node_bin_counts={}
        self.fitted=False

    @staticmethod
    def _get_loss(df):
        if "cu_combined_loss_prob" not in df:
            raise ValueError("Missing master-side CU path loss evidence")
        v=df["cu_combined_loss_prob"].to_numpy(dtype=float)
        if not np.isfinite(v).all():
            raise ValueError("Non-finite master loss observations")
        return np.clip(v,0.,1.)

    @staticmethod
    def _get_node(df):
        if "node_type" not in df:
            raise ValueError("Missing normal operator-provisioned node type")
        vals=df["node_type"].astype(str).to_numpy()
        return vals

    def _index(self,loss):
        return np.searchsorted(self.edges,loss,side="right").astype(int)

    def fit(self,normal_train):
        if len(normal_train)<500:
            raise ValueError("Need >=500 benign calibration rows")
        if "is_anomaly" in normal_train and (normal_train["is_anomaly"]!=0).any():
            raise ValueError("Loss reference may only use benign normal rows")
        if not getattr(self.physical_model,"fitted",False):
            raise ValueError("Physical TICD must be trained and frozen beforehand")
        loss=self._get_loss(normal_train)
        node=self._get_node(normal_train)
        raw=np.asarray(self.physical_model.score(normal_train),dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("Upstream TICD returned non-finite benign score")
        if len(raw)!=len(loss):
            raise ValueError("Unexpected upstream score length")
        edges=np.quantile(loss,np.linspace(0,1,self.bins+1)[1:-1])
        self.edges=np.unique(edges)
        b=self._index(loss)
        self.global_q99=max(float(np.percentile(raw,self.normal_quantile)),1e-3)
        self.bin_q99={}
        self.node_bin_q99={}
        self.node_bin_counts={}
        for k in np.unique(b):
            mask=b==k
            q=float(np.percentile(raw[mask],self.normal_quantile))
            self.bin_q99[int(k)]=max(q,1e-3)
        if self.mode!="loss":
            for label in sorted(set(node)):
                for k in np.unique(b):
                    mask=(node==label)&(b==k)
                    count=int(mask.sum())
                    if count>=self.min_group:
                        self.node_bin_q99[(label,int(k))]=max(
                            float(np.percentile(raw[mask],self.normal_quantile)),1e-3)
                        self.node_bin_counts[(label,int(k))]=count
        self.fitted=True
        return self

    def _q99(self,loss,node):
        b=self._index(loss)
        q=np.array([self.bin_q99.get(int(i),self.global_q99) for i in b],
                   dtype=float)
        if self.mode=="loss":
            return q
        for i,(label,k) in enumerate(zip(node,b)):
            idx=(label,int(k))
            if idx not in self.node_bin_q99:
                continue
            group=self.node_bin_q99[idx]
            if self.mode=="loss_node":
                q[i]=group
            else:
                # Fixed benign-only shrinkage against sparse bin overfitting.
                n=self.node_bin_counts[idx]
                alpha=n/(n+500.)
                q[i]=alpha*group+(1.-alpha)*q[i]
        return np.maximum(q,1e-3)

    def score(self,df):
        if not self.fitted:
            raise RuntimeError("Fit normal loss-conditioned TICD before scoring")
        loss=self._get_loss(df)
        node=self._get_node(df)
        raw=np.asarray(self.physical_model.score(df),dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("Non-finite upstream TICD score")
        return np.clip(raw/self._q99(loss,node),0.,1e4)

    def reference_manifest(self):
        if not self.fitted:
            raise RuntimeError("Not fitted")
        return {
          "mode":self.mode,
          "loss_bin_edges":[float(v) for v in self.edges],
          "bin_normal_q99":{str(k):float(v) for k,v in self.bin_q99.items()},
          "node_bin_counts":{f"{k[0]}|{k[1]}":v for k,v in self.node_bin_counts.items()},
          "global_normal_q99":float(self.global_q99),
          "reference_only_benign":True,
          "retrained_online":False,
          "underlying_physical_TICD_frozen":True
        }
