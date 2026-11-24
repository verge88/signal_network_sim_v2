"""Run-level robust calibration of a frozen SIP TICD strong-alert score.

A run is a complete benign two-hour simulation, not an independent row.
The calibration threshold is computed offline from disjoint normal runs.
No seed, run identity, attack labels or future observations enter inference.
Empirical run-level budgets are not population or deployment guarantees.
"""
from __future__ import annotations
import numpy as np

class RunRobustAlertTICD:
    def __init__(self,prior,mode="worst_run",budget=.008):
        if prior.threshold is None: raise ValueError("Require calibrated frozen prior")
        if mode not in ("pooled","p90_run","worst_run"): raise ValueError(mode)
        if not 0.<budget<.01: raise ValueError("Invalid reference budget")
        self.prior=prior
        self.mode=mode
        self.budget=float(budget)
        self.threshold=None
        self.reference={}
    def score(self,df,channels=None):
        return self.prior.score(df,channels)
    def calibrate(self,normal):
        regime_thresholds={}
        for loss,df in normal.items():
            if (df.is_anomaly!=0).any(): raise ValueError("Nonbenign calibration")
            scores=self.score(df)
            if not np.isfinite(scores).all(): raise ValueError("Nonfinite score")
            seeds=df.simulation_seed.to_numpy()
            qs={}
            for seed in sorted(set(seeds)):
                x=scores[seeds==seed]
                if len(x)<500: raise ValueError("Incomplete reference run")
                qs[str(int(seed))]=float(np.quantile(x,1-self.budget,method="higher"))
            pooled=float(np.quantile(scores,1-self.budget,method="higher"))
            # Use a linear 90th percentile across run thresholds; with few
            # runs this is close to the maximum, but not a tolerance bound.
            p90=float(np.quantile(list(qs.values()),.90,method="linear"))
            worst=max(qs.values())
            chosen={"pooled":pooled,"p90_run":max(pooled,p90),"worst_run":max(pooled,worst)}[self.mode]
            regime_thresholds[str(loss)]={"pooled":pooled,"p90_run":p90,"worst_run":worst,
                                          "per_run_quantile":qs,"chosen":chosen}
        self.threshold=max(x["chosen"] for x in regime_thresholds.values())
        self.reference=regime_thresholds
        return self
    def predict_levels(self,df):
        if self.threshold is None: raise RuntimeError("Uncalibrated")
        role_model=self.prior.prior.prior.prior
        channels=role_model.score_channels(df)
        a=self.score(df,channels)>self.threshold
        w=(channels["warning"]>role_model.warning_threshold)&role_model.eligible_mask(df)&(~a)
        out=np.zeros(len(df),dtype=np.int8)
        out[w]=1;out[a]=2
        return out
    def predict_masks(self,df):
        x=self.predict_levels(df)
        return {"normal":x==0,"warning_only":x==1,"confirmed_alert":x==2,"any_signal":x>=1}
    def manifest(self):
        return {"mode":self.mode,"budget":self.budget,"threshold":self.threshold,
                "reference":self.reference,"prior_score_frozen":True,"warning_reference_frozen":True,
                "run_id_used_only_offline":True,"all_roles_strong_alert":True,
                "no_population_FPR_guarantee":True}
