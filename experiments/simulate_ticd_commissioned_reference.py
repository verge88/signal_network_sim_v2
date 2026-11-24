"""Separate trusted commissioning observations and shifted benign regimes.

The topology and fixed node/link characteristics are shared within an
installation. Commissioning uses independent traffic/CU RNG streams and
no attacks. Operational data are never used to learn node offsets.
The generator knows scenario schedules; the detector receives only
observed counters, inventory and CU telemetry.
"""
from __future__ import annotations
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
from sip_sim_v4_claude import SIPSimulator
from retrain_sip_ticd_calibrated import simulation
from simulate_ticd_extended_observation import build,KINDS as ORIGINAL_KINDS

SHIFT_KINDS=("shift_clean_loss","shift_clean_load","shift_weak_0025","shift_weak_005","shift_redirect")
KINDS=ORIGINAL_KINDS+SHIFT_KINDS
BASE={"shift_clean_loss":"clean_loss10","shift_clean_load":"clean_loss10",
      "shift_weak_0025":"weak_0025","shift_weak_005":"weak_005","shift_redirect":"role_redirect_loss25"}
_ORIGINAL_RUN=SIPSimulator.run

def run_with_regime(sim,seed,commission=False,shift=None):
    # One random but persistent link factor per installation and link.
    # This is a new simulator domain, common to every compared detector.
    rng=np.random.default_rng(seed+3100001)
    links=sorted(sim.topology.links.items())
    for _,link in links:link.base_loss_prob*=float(rng.uniform(.8,1.2))
    original_losses={key:link.base_loss_prob for key,link in links}
    original_rates={nid:node.base_rate for nid,node in sim.topology.nodes.items()}
    if commission:
        assert not sim.attacks
        # Topology still uses the installation seed, while all dynamic
        # randomness is independent of its operational realization.
        sim.rng=np.random.default_rng(seed+4100001)
        sim.injector.rng=sim.rng
        sim.traffic_gen.rng=np.random.default_rng(seed+4100002)
        sim.traffic_gen.txn_model.rng=sim.traffic_gen.rng
        sim.cu_engine.rng=np.random.default_rng(seed+4100003)
    reset=sim.topology.reset_link_loads
    t=-1
    def step():
        nonlocal t
        t+=1
        reset()
        phase=min(3,t//90)
        if commission:
            loss_mult=(1.,5.,10.,25.)[phase]
            load_mult=(1.,1.,1.,1.)[phase]
        elif shift:
            # Nominal baseline is 10 or 25. Multipliers never encode an
            # observed attack label and are not exported as detector features.
            loss_mult=(.2,1.,2.,.5)[phase] if shift!="shift_clean_load" and shift!="shift_redirect" else 1.
            if shift=="shift_redirect":loss_mult=(.2,.4,1.,.6)[phase]
            load_mult=(1.,1.5,.65,1.2)[phase] if shift!="shift_clean_loss" else 1.
        else:loss_mult=load_mult=1.
        for key,link in links:link.base_loss_prob=original_losses[key]*loss_mult
        for nid,node in sim.topology.nodes.items():node.base_rate=original_rates[nid]*load_mult
    sim.topology.reset_link_loads=step
    return _ORIGINAL_RUN(sim)

def commissioning(seed,cal):
    with patch.object(SIPSimulator,"run",lambda sim:run_with_regime(sim,seed,commission=True)):
        d,_=simulation(seed,cal,2.,False,loss_multiplier=1.)
    assert len(d)==3960 and not d.is_anomaly.any()
    d["primary_observer_valid"]=True
    return d

def observed_group(seeds,cal,kind,causal):
    parts=[];masters=[];meta=[];references=[]
    for seed in seeds:
        # Separate normal records are acquired first, before operation.
        c=commissioning(seed,cal)
        shift=kind if kind in SHIFT_KINDS else None
        with patch.object(SIPSimulator,"run",lambda sim:run_with_regime(sim,seed,shift=shift)):
            x,m,md=build(seed,cal,BASE.get(kind,kind),causal)
        assert len(x)==len(c)==3960
        assert x.groupby("node_id").node_type.first().to_dict()==c.groupby("node_id").node_type.first().to_dict()
        md.update({"commissioning_rows":len(c),"commissioning_clean":True,
                   "independent_dynamic_rng":True,"persistent_link_heterogeneity":True,
                   "within_run_shift":shift})
        references.append(c);parts.append(x);masters.append(m);meta.append(md)
    print("NODE_RUNS",json.dumps({"kind":kind,"seeds":seeds,"operational_rows":sum(map(len,parts)),
          "commissioning_rows":sum(map(len,references))}),flush=True)
    return (pd.concat(parts,ignore_index=True),pd.concat(masters,ignore_index=True),meta,
            pd.concat(references,ignore_index=True))
