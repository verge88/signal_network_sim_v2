"""Measured-loss experiment: explicit synthetic sent/lost counters.

All detector variants see the same underlying SIP run and probe overhead.
The passive sensor is an ideal trusted two-point 200 OK counter pair,
not a deployed packet collector. The physical SIP model remains symmetric.
"""
from __future__ import annotations
from unittest.mock import patch
import numpy as np
import pandas as pd
from sip_sim_v4_claude import SIPSimulator,SIPTrafficGenerator,SIPNetworkTopology,NodeRole
from simulate_ticd_commissioned_reference import observed_group
from simulate_ticd_independent_loss import KINDS,STRESS

GEN=SIPTrafficGenerator.generate_node_traffic
REC=SIPSimulator._build_record
RESET=SIPNetworkTopology.reset_link_loads
PACKETS=512
WIRE_BYTES=128

def group(seeds,cal,kind,causal):
    base,mode=STRESS.get(kind,(kind,"nominal"))
    rows=[];masters=[];metadata=[];comm=[]
    for seed in seeds:
        rng=np.random.default_rng(seed+6100001)
        def reset(self):
            RESET(self)
            # Reserve bidirectional probe traffic on each measured path.
            # Aggregate load model, not a discrete packet queue simulator.
            mbps=2*PACKETS*WIRE_BYTES*8/(self.config.interval_s*1e6)
            for node in self.nodes.values():
                if node.role==NodeRole.MASTER:continue
                path=self.get_path(node.node_id,node.master_id)
                for a,b in zip(path,path[1:]):
                    link=self.get_link(a,b)
                    if link is not None:link.current_load_mbps+=mbps
        def gen(self,node,t,override_params=None):
            p=float(self._path_loss_to_master(node))
            # Model the same physical route with independent packet outcomes.
            # An observation mismatch stress changes only probe probability.
            p_obs=float(np.clip(p*(.7 if (t//90)%2==0 else 1.3),0,1)) if mode=="bias" else p
            a=int(rng.binomial(PACKETS,p_obs));b=int(rng.binomial(PACKETS,p_obs))
            valid=rng.random()>=.01 and not(mode=="outage" and 150<=t<210)
            traffic=GEN(self,node,t,override_params)
            traffic["_measurement"]=(PACKETS if valid else 0,a if valid else 0,b if valid else 0)
            return traffic
        def record(self,t,sid,node,master_id,true_traffic,cu_result,attack_type,event_type):
            x=REC(self,t,sid,node,master_id,true_traffic,cu_result,attack_type,event_type)
            sent,fwd,rev=true_traffic["_measurement"]
            x.update({"active_sent":sent,"active_fwd_lost":fwd,"active_rev_lost":rev,
                # Independent trusted upstream/downstream capture assumption.
                # Counts refer to the actual simulated 200 OK population.
                "passive_sent":int(true_traffic["n_invite_200"]),
                "passive_received":int(true_traffic["n_200_transit"])})
            return x
        with patch.object(SIPNetworkTopology,"reset_link_loads",reset),patch.object(SIPTrafficGenerator,"generate_node_traffic",gen),patch.object(SIPSimulator,"_build_record",record):
            x,m,md,c=observed_group([seed],cal,base,causal)
        for d in (x,c):
            for col in ("active_sent","active_fwd_lost","active_rev_lost"):
                if mode=="delay":
                    d[col]=d.groupby("node_id",sort=False)[col].shift(3,fill_value=0)
            d["active_age"]=np.where(d.active_sent>0,3 if mode=="delay" else 0,-1)
            # The primary outage disables access to passive and active
            # observations here; availability of the policy audit is separate.
            mask=~d.primary_observer_valid.to_numpy(dtype=bool)
            d.loc[mask,["active_sent","active_fwd_lost","active_rev_lost","passive_sent","passive_received"]]=0
            d.loc[mask,"active_age"]=-1
            assert (d.passive_received<=d.passive_sent).all()
        md[0].update({"measurement_mode":mode,"probe_packets_per_direction":PACKETS,
                      "wire_bytes_assumed":WIRE_BYTES,"probe_load_accounted":True,
                      "passive_two_point_sensor_assumed":True})
        rows.append(x);masters.append(m);metadata+=md;comm.append(c)
    return pd.concat(rows,ignore_index=True),pd.concat(masters,ignore_index=True),metadata,pd.concat(comm,ignore_index=True)
