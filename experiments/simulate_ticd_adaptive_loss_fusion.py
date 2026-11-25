"""Additional measurement-domain stress tests; no hidden detector inputs."""
from __future__ import annotations
import numpy as np
from simulate_ticd_measured_loss import group as base_group,KINDS as BASE_KINDS,STRESS as BASE_STRESS

EXTRA={"sensor_outage_clean":("shift_clean_loss","joint_outage"),
       "sensor_outage_weak":("shift_weak_005","joint_outage"),
       "passive_bias_clean":("shift_clean_loss","passive_bias"),
       "passive_bias_weak":("shift_weak_005","passive_bias")}
KINDS=BASE_KINDS+tuple(EXTRA)
STRESS={**BASE_STRESS,**EXTRA}

def group(seeds,cal,kind,causal):
    base,stress=EXTRA.get(kind,(kind,None))
    x,m,md,c=base_group(seeds,cal,base,causal)
    # Stress is operational only; trusted commissioning is unchanged.
    if stress=="joint_outage":
        mask=x.interval.between(150,209)
        x.loc[mask,["active_sent","active_fwd_lost","active_rev_lost","passive_sent","passive_received"]]=0
        x.loc[mask,"active_age"]=-1
    elif stress=="passive_bias":
        # Model counter-pair mismatch/undercapture at the receiver.
        # Extra missing observations in alternating phases, not physical loss.
        for seed in seeds:
            rng=np.random.default_rng(seed+7100001)
            mask=(x.simulation_seed==seed)&((x.interval//90)%2==1)
            received=x.loc[mask,"passive_received"].to_numpy(dtype=int)
            x.loc[mask,"passive_received"]=received-rng.binomial(received,.01)
    for a in md:a["additional_measurement_stress"]=stress
    return x,m,md,c
