"""Synthetic independent loss measurements, not an implemented network probe.

The generator samples independent Bernoulli losses on the simulated SIP
path at the time traffic is generated. The detector receives only counts.
Probe traffic does not affect simulated capacity; overhead is reported.
"""
from __future__ import annotations
from unittest.mock import patch
import json
import numpy as np
from sip_sim_v4_claude import SIPTrafficGenerator
from simulate_ticd_commissioned_reference import observed_group,KINDS as OLD_KINDS

STRESS={
    "probe_delay_clean":("shift_clean_loss","delay"),
    "probe_delay_weak":("shift_weak_005","delay"),
    "probe_bias_clean":("shift_clean_loss","bias"),
    "probe_bias_weak":("shift_weak_005","bias"),
    "probe_outage_clean":("shift_clean_loss","outage"),
    "probe_outage_weak":("shift_weak_005","outage"),
}
KINDS=OLD_KINDS+tuple(STRESS)
ORIGINAL=SIPTrafficGenerator.generate_node_traffic

def group(seeds,cal,kind,causal):
    base,mode=STRESS.get(kind,(kind,"nominal"))
    parts=[];masters=[];meta=[];comm=[]
    import pandas as pd
    for seed in seeds:
        seen={}
        # Separate stream; neither transaction nor CU RNG is changed.
        rng=np.random.default_rng(seed+5300001)
        def generate(self,node,t,override_params=None):
            p=float(self._path_loss_to_master(node))
            # The sampler's route may differ from the SIP route. Deliberate
            # bias is an observation-domain stress, not an attacker model.
            if mode=="bias":p=np.clip(p*(.7 if (t//90)%2==0 else 1.3),0.,1.)
            first=int(rng.binomial(128,p));extra=int(rng.binomial(384,p))
            valid=rng.random()>=.01
            if mode=="outage" and 150<=t<210:valid=False
            seen[(int(node.node_id),int(t))]=(128 if valid else 0,first if valid else 0,
                    512 if valid else 0,first+extra if valid else 0)
            return ORIGINAL(self,node,t,override_params)
        # observed_group runs commissioning first; operational samples replace
        # those keys. The raw commissioning profile remains unchanged.
        with patch.object(SIPTrafficGenerator,"generate_node_traffic",generate):
            x,m,md,c=observed_group([seed],cal,base,causal)
        assert len(seen)==len(x)==3960
        delay=3 if mode=="delay" else 0
        obs=np.array([seen.get((int(n),int(t)-delay),(0,0,0,0))
                      for n,t in zip(x.node_id,x.interval)],dtype=int)
        for i,col in enumerate(("probe128_sent","probe128_lost","probe512_sent","probe512_lost")):
            x[col]=obs[:,i]
        x["probe_age"]=np.where(obs[:,2]>0,delay,-1)
        assert (x.probe128_lost<=x.probe128_sent).all() and (x.probe512_lost<=x.probe512_sent).all()
        assert (x.probe128_lost<=x.probe512_lost).all()
        md[0].update({"probe_mode":mode,"probe_delay_intervals":delay,"probe_is_synthetic":True,
                      "probe_packets_per_interval":512,"probe_load_not_modeled":True,
                      "probe_available_rows":int((obs[:,2]>0).sum())})
        parts.append(x);masters.append(m);meta+=md;comm.append(c)
    print("LOSS_RUNS",json.dumps({"kind":kind,"seeds":seeds,"probe_mode":mode,
        "rows":sum(map(len,parts))}),flush=True)
    return pd.concat(parts,ignore_index=True),pd.concat(masters,ignore_index=True),meta,pd.concat(comm,ignore_index=True)
