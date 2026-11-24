"""Defensive expanded SIP stress simulation and stylized external policy audit.

Native SIP scenarios exercise the existing transaction simulator on all
slave roles, different fractions and groups. The external audit is a
SEPARATE synthetic observation model: sampled approved/actual state
comparisons with noise, missed samples and delayed benign consistency.
It is not an implemented network collector, field trace or host attestation.
A latent compromise with no changed observation is an explicit null test.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sip_sim_v4_claude import (SIPSimulator,SimulationConfig,AttackScenario,NormalScenario,
    AttackType,NormalEventType,AttackerSophistication,NodeRole,SIP_METHODS,add_zone_deviation_features)
from retrain_sip_ticd_calibrated import simulation
from ticd_physical_closure import prepare_master_closure_features
from ticd_causal_ack import add_causal_ack_history
from ticd_zonal_warnings import prepare_zonal_warning_context
from ticd_observability_factorial import prepare_policy_history

ROLES=("PROXY","REGISTRAR","UA_GW","REDIRECT")
KINDS=("clean_loss1","clean_loss7","clean_loss25","default",
       *(f"role_{r.lower()}_loss{m}" for r in ROLES for m in (1,25)),
       "group_majority","group_zone","weak_0025","weak_005","weak_01","variable_deficit",
       "policy_route","policy_binding","policy_group","latent_identical",
       "observer_outage","observer_outage_policy","master_policy")
POLICY_CASES={"policy_route","policy_binding","policy_group","latent_identical",
              "observer_outage","observer_outage_policy","master_policy"}

def native(seed,cal,kind):
    if kind.startswith("clean_loss"):
        m=float(kind.removeprefix("clean_loss"))
        d,_=simulation(seed,cal,2.,False,loss_multiplier=m)
        return d,{"loss":m,"native":False,"targets":[]}
    if kind=="default":
        d,_=simulation(seed,cal,2.,True)
        return d,{"loss":1.,"native":True,"targets":[]}
    if kind in POLICY_CASES:
        d,_=simulation(seed,cal,2.,False,loss_multiplier=10.)
        return d,{"loss":10.,"native":False,"targets":[]}
    cfg=SimulationConfig(seed=seed,duration_hours=2.,interval_s=20.,
        n_sbc=2,n_proxy=3,n_registrar=2,n_ua_gw=4,n_redirect=2,master_node_ids=[0,1])
    n=360;start=95+(seed%17)
    s=lambda f:int(n*f)
    events=[NormalScenario(NormalEventType.TRAFFIC_SPIKE,[7,8],s(.06),s(.11),intensity=3.),
        NormalScenario(NormalEventType.MAINTENANCE,[6],s(.15),s(.22),intensity=2.),
        NormalScenario(NormalEventType.REROUTE,[9,10],s(.80),s(.87),intensity=2.),
        NormalScenario(NormalEventType.PRESENCE_SURGE,[5],s(.69),s(.77),intensity=2.)]
    sim=SIPSimulator(cfg,[],events)
    slaves=[x for x in sim.topology.nodes.values() if x.role==NodeRole.SLAVE]
    frac=.04;m=10.;variable=False
    if kind.startswith("role_"):
        role,loss=kind.removeprefix("role_").rsplit("_",1)
        pool=[x.node_id for x in slaves if x.node_type.name==role.upper()]
        targets=[pool[seed%len(pool)]];m=float(loss.removeprefix("loss"))
    elif kind.startswith("group_"):
        zones={z:sorted(x.node_id for x in slaves if x.zone_id==z) for z in (0,1)}
        z=sorted(zones,key=lambda x:len(zones[x]),reverse=True)[0]
        pool=zones[z];assert len(pool)>=3
        targets=pool if kind=="group_zone" else pool[:len(pool)//2+1]
        m=25. if kind=="group_zone" else 10.
    else:
        pool=[x.node_id for x in slaves if x.node_type.name=="PROXY"]
        targets=[pool[seed%len(pool)]]
        if kind=="variable_deficit":
            variable=True;frac=.04
        else:frac={"weak_0025":.0025,"weak_005":.005,"weak_01":.01}[kind]
    sim.attacks=[AttackScenario(AttackType.PROXY_COMPROMISE,targets,start,n,intensity=1.,
        sophistication=AttackerSophistication.STATISTICAL,covert_redirect_fraction=frac)]
    for node in slaves:
        freq={k:float(node.method_dist[k])*float(cal["method_weights"].get(k,1.)) for k in SIP_METHODS}
        total=sum(freq.values());node.method_dist={k:v/total for k,v in freq.items()}
    for link in sim.topology.links.values():link.base_loss_prob*=m
    if variable:
        # A persistent compromised-state simulation with externally prescribed
        # time-varying physical deficit. Quiet intervals are still labelled
        # compromised; activity and compromise metrics are reported separately.
        original=sim.traffic_gen.generate_node_traffic
        def generate(node,t,override_params=None):
            p=dict(override_params or {})
            if node.node_id in targets and t>=start:
                phase=(t-start)//15%4
                p["covert_redirect"]=(0.,.01,.04,.02)[phase]
            return original(node,t,p)
        sim.traffic_gen.generate_node_traffic=generate
        sim.cu_engine._masked_report=lambda true,node,sid,t:sim.cu_engine._honest_report(true)
    sim.run()
    d=add_zone_deviation_features(sim.to_dataframe());d["simulation_seed"]=seed
    assert len(d)==3960 and set(d.loc[d.is_anomaly>0,"node_id"])==set(targets)
    assert not ((d.is_compromised>0)&(d.is_anomaly==0)).any()
    if variable:
        d["eval_active_deficit"]=(d.node_id.isin(targets)&(d.interval>=start)&(((d.interval-start)//15%4)!=0))
    return d,{"loss":m,"native":True,"targets":targets,"start":start,"fraction":frac,"variable":variable}

def audit_counts(df,seed,loss,changed,rate,offline):
    # Simulate independent audit opportunities. Approved state and actual
    # state are represented by categorical tokens (not packet contents).
    # Authorized transitions may be temporarily inconsistent even when clean.
    # Observed mismatch is computed AFTER sampling and measurement noise.
    # All latent scenario flags remain in the generator; not in model inputs.
    rng=np.random.default_rng(seed+170003)
    n=len(df);t=df.interval.to_numpy(dtype=int)
    changed=np.asarray(changed,dtype=bool);offline=np.asarray(offline,dtype=bool)
    observations=np.zeros(n,dtype=int);mismatches=np.zeros(n,dtype=int)
    for i in range(n):
        # Heterogeneous counts and temporarily stale registry entries.
        opportunities=int(rng.poisson(60 if df.node_type.iloc[i]!="SBC" else 45))
        if opportunities==0:continue
        approved=rng.integers(0,16,size=opportunities)
        actual=approved.copy()
        benign_rate=.003+(.008 if 285<=t[i]<315 else 0.)
        benign=rng.random(opportunities)<benign_rate
        actual[benign]=(actual[benign]+1)%16
        policy_delta=rng.random(opportunities)<(rate if changed[i] else 0.)
        actual[policy_delta]=(approved[policy_delta]+2)%16
        measured=actual.copy()
        sensor_error=rng.random(opportunities)<.002
        measured[sensor_error]=(measured[sensor_error]+1)%16
        keep=rng.random(opportunities)>min(.35,.03+.004*loss)
        if offline[i] or rng.random()<.005:keep[:]=False
        observations[i]=int(keep.sum());mismatches[i]=int(((approved!=measured)&keep).sum())
    x=df.copy();x["audit_observed_count"]=observations;x["audit_mismatch_count"]=mismatches
    return prepare_policy_history(x)

def build(seed,cal,kind,causal):
    d,meta=native(seed,cal,kind)
    d["simulation_seed"]=seed
    d=prepare_zonal_warning_context(add_causal_ack_history(prepare_master_closure_features(d)),causal)
    # Preserve baseline telemetry exactly in policy-only and latent cases.
    legacy_labels=d[["is_anomaly","attack_type","attacker_sophistication"]].copy()
    t=d.interval.to_numpy();start=100+(seed%13)
    changes=np.zeros(len(d),dtype=bool);offline=np.zeros(len(d),dtype=bool)
    primary=np.ones(len(d),dtype=bool)
    pool=lambda role:sorted(d.loc[d.node_type==role,"node_id"].unique().tolist())
    if kind in ("policy_route","policy_binding","latent_identical"):
        role="REGISTRAR" if kind=="policy_binding" else "PROXY"
        ids=pool(role);targets=[ids[seed%len(ids)]]
    elif kind in ("policy_group","observer_outage_policy"):
        z=int(d.zone_id.value_counts().index[0]);targets=sorted(d.loc[d.zone_id==z,"node_id"].unique().tolist())
    else:targets=[]
    active=d.node_id.isin(targets).to_numpy()&(t>=start)
    if kind in ("policy_route","policy_binding","policy_group","observer_outage_policy"):changes=active
    if kind in ("observer_outage","observer_outage_policy"):
        z=int(d.zone_id.value_counts().index[0])
        primary=~((d.zone_id.to_numpy()==z)&(t>=start))
        # No primary recovery after outage; thus no later valid history
        # can inadvertently depend on unobserved primary samples.
        offline=(~primary)&(t>=start+60)&(t<start+90)
    d["primary_observer_valid"]=primary
    d=audit_counts(d,seed,meta["loss"],changes,.04,offline)
    # The labels are assigned only AFTER independent sensor data generation.
    # The legacy SIP trace is unchanged for all policy-only conditions.
    if targets:
        d.loc[active,"is_anomaly"]=1
        d.loc[active,"attack_type"]="proxy_compromise"
        d.loc[active,"attacker_sophistication"]="statistical"
    masters=pd.DataFrame([(seed,j,i,"SBC") for j in (0,1) for i in range(360)],
        columns=["simulation_seed","node_id","interval","node_type"])
    changed=(masters.node_id.to_numpy()==seed%2)&(masters.interval.to_numpy()>=start)&(kind=="master_policy")
    masters["is_anomaly"]=0
    masters=audit_counts(masters,seed+700001,meta["loss"],changed,.04,np.zeros(len(masters),dtype=bool))
    masters["is_anomaly"]=changed.astype(int)
    meta.update({"policy_targets":targets,"policy_start":start,"audit_is_synthetic":True,
                 "master_policy_only":kind=="master_policy","primary_outage":kind.startswith("observer_outage"),
                 "legacy_labels_changed":bool(np.any(legacy_labels.is_anomaly.to_numpy()!=d.is_anomaly.to_numpy())),
                 "latent_identical":kind=="latent_identical","rows":len(d),"master_rows":len(masters)})
    return d,masters,meta
