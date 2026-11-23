#!/usr/bin/env python3
"""Non-regression tests for loss conditional TICD, no generator required."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
from ticd_loss_conditioned import LossConditionedTICD

class FrozenDummyTICD:
    fitted=True
    def score(self,df):
        return df["_frozen_raw_score"].to_numpy(dtype=float)

def run_tests():
    rng=np.random.default_rng(9)
    n=4000
    loss=np.linspace(0.001,0.06,n)
    dt=pd.DataFrame({"cu_combined_loss_prob":loss,
        "node_type":np.where(np.arange(n)%2==0,"PROXY","UA_GW"),
        "_frozen_raw_score":np.clip(0.3+25*loss+rng.normal(0,.18,n),0,6),
        "is_anomaly":np.zeros(n,dtype=int)})
    for mode in LossConditionedTICD.MODES:
        m=LossConditionedTICD(FrozenDummyTICD(),mode=mode).fit(dt)
        observed=m.score(dt)
        assert len(observed)==n
        assert np.isfinite(observed).all()
        sub=dt.sample(150,random_state=23)
        assert np.array_equal(m.score(sub),observed[sub.index])
        changed=dt.copy();changed["is_anomaly"]=1
        try:
            LossConditionedTICD(FrozenDummyTICD(),mode=mode).fit(changed)
            raise AssertionError("Attack-contaminated normal calibration not rejected")
        except ValueError:
            pass
        try:
            m.score(sub.drop(columns=["cu_combined_loss_prob"]))
            raise AssertionError("Missing loss observation should be rejected")
        except ValueError:
            pass
    print("LOSS_CONDITIONED_TICD_TESTS_OK",list(LossConditionedTICD.MODES))

if __name__=="__main__":
    run_tests()
