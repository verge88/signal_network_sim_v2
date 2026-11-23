#!/usr/bin/env python3
"""Frozen-model robustness checks of the improved SIP TICD (no retuning).

Three independently simulated stress conditions, all outside architecture
selection and threshold calibration:
  weak_covert: lower covert redirect fraction & shifted onset;
  weak_covert_loss5: same but 5x network link loss;
  truthful_compromised_reports: breached node reports true SIP state
      (different attacker model; the I1/I2 dilemma need not occur).

This is a model stress test, not independent real attack validation.
"""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
sys.path.insert(0,str(ROOT/"experiments"))
from sip_sim_v4_claude import (
  SIPSimulator,SimulationConfig,AttackScenario,NormalScenario,AttackType,
  NormalEventType,AttackerSophistication,NodeRole,SIP_METHODS,
  add_zone_deviation_features
)
from retrain_sip_ticd_calibrated import evaluate

STRESS_GROUPS=("weak_covert","weak_covert_loss5","truthful_compromised_reports")

def simulate_stress(seed,cal,condition,hours=2.,loss_multiplier=None):
    cfg=SimulationConfig(seed=seed,duration_hours=hours,interval_s=20.,
        n_sbc=2,n_proxy=3,n_registrar=2,n_ua_gw=4,n_redirect=2,
        master_node_ids=[0,1])
    n=int(hours*3600/cfg.interval_s)
    def s(frac):
        k=int(round(n*frac))+(seed%17-8)
        return max(1,min(n-1,k))
    if condition=="truthful_compromised_reports":
        redirect_fractions=(.10,.15,.20)
    else:
        redirect_fractions=(.02,.04,.06)
    attacks=[
      AttackScenario(AttackType.PROXY_COMPROMISE,[2],s(.17),n,
        intensity=1.0,sophistication=AttackerSophistication.NAIVE,
        covert_redirect_fraction=redirect_fractions[0]),
      AttackScenario(AttackType.PROXY_COMPROMISE,[3],s(.39),n,
        intensity=1.0,sophistication=AttackerSophistication.STATISTICAL,
        covert_redirect_fraction=redirect_fractions[1]),
      AttackScenario(AttackType.PROXY_COMPROMISE,[4],s(.56),n,
        intensity=1.0,sophistication=AttackerSophistication.ADAPTIVE,
        covert_redirect_fraction=redirect_fractions[2]),
    ]
    normal=[
       NormalScenario(NormalEventType.TRAFFIC_SPIKE,[7,8],s(.08),s(.14),intensity=4.0),
       NormalScenario(NormalEventType.MAINTENANCE,[6],s(.23),s(.32),intensity=3.0),
       NormalScenario(NormalEventType.REROUTE,[9,10],s(.70),s(.78),intensity=3.0),
       NormalScenario(NormalEventType.PRESENCE_SURGE,[5],s(.82),s(.89),intensity=4.0),
    ]
    sim=SIPSimulator(cfg,attacks,normal)
    weights=cal["method_weights"]
    for node in sim.topology.nodes.values():
        if node.role!=NodeRole.SLAVE:continue
        freq={m:float(node.method_dist[m])*float(weights.get(m,1.)) for m in SIP_METHODS}
        total=sum(freq.values())
        node.method_dist={k:v/total for k,v in freq.items()}
    effective_loss_multiplier = (5.0 if condition=="weak_covert_loss5" else 1.0)
    if loss_multiplier is not None:
        effective_loss_multiplier=float(loss_multiplier)
    if not 0.1 <= effective_loss_multiplier <= 25.0:
        raise ValueError("Invalid independent network-loss multiplier")
    if effective_loss_multiplier!=1.0:
        for link in sim.topology.links.values():
            link.base_loss_prob*=effective_loss_multiplier
    if condition=="truthful_compromised_reports":
        # Deliberately change threat assumption. _honest_report reports actual
        # traffic and so no intentional tampering with CU; attack still modifies
        # actual ACK/dangling traffic via covert redirect.
        engine=sim.cu_engine
        engine._masked_report=lambda true,node,slave_id,interval_idx:engine._honest_report(true)
    sim.run()
    d=add_zone_deviation_features(sim.to_dataframe())
    d["simulation_seed"]=seed
    if not ((d.is_anomaly==1)&(d.attack_type=="proxy_compromise")).sum():
        raise RuntimeError("No compromise labels in stress output")
    return d,{"seed":seed,"condition":condition,"redirect_fractions":redirect_fractions,
              "link_loss_multiplier":effective_loss_multiplier,
              "corrupted_reports":condition!="truthful_compromised_reports",
              "attack_schedule":[{"node":a.target_nodes[0],"start":a.start_interval,
                                  "sophistication":a.sophistication.value,
                                  "redirect":a.covert_redirect_fraction} for a in attacks]}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prior-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,default=Path("results_ticd_stress"))
    args=parser.parse_args();args.out_dir.mkdir(exist_ok=True,parents=True)
    with (args.prior_dir/"ticd_multiscale_model.pkl").open("rb") as f:
        p=pickle.load(f)
    a=p["model"];b=p["baseline_model"]
    thr_a=p["decision_threshold"];thr_b=p["baseline_threshold"]
    cal=p["calibration"]
    assert p["selected_mode"]=="temporal"
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    independent_seeds={"weak_covert":[1601,1602,1603],
                       "weak_covert_loss5":[1701,1702,1703],
                       "truthful_compromised_reports":[1801,1802,1803]}
    out={"scope":"out-of-selection fixed-model synthetic stress tests",
         "model_frozen":True,"thresholds_frozen":True,
         "thresholds":{"baseline":thr_b,"multiscale":thr_a},
         "seed_partitions":independent_seeds,
         "real_ticd_validation":False,"groups":{}}
    all_seen=[]
    for condition,seeds in independent_seeds.items():
        frames=[];meta=[]
        for seed in seeds:
            d,details=simulate_stress(seed,cal,condition)
            frames.append(d);meta.append(details)
        d=pd.concat(frames,ignore_index=True)
        assert set(d.simulation_seed).isdisjoint(
            set(sum(p["partitions"].values(),[])))
        x=evaluate(a,d,thr_a)
        y=evaluate(b,d,thr_b)
        group={"model":x,"baseline":y,"simulation_metadata":meta}
        out["groups"][condition]=group
        print("STRESS_RESULT",json.dumps({"condition":condition,
            "model":{"fpr":x["fpr"],"recall":x["tpr_recall"],
                     "sophistication":x["by_compromise_sophistication"]},
            "baseline":{"fpr":y["fpr"],"recall":y["tpr_recall"],
                        "sophistication":y["by_compromise_sophistication"]}},
            ensure_ascii=False),flush=True)
    (args.out_dir/"untouched_sensitivity_report.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print("STRESS_TEST_COMPLETE",json.dumps({
        x:{"base_fpr":v["baseline"]["fpr"],
           "model_fpr":v["model"]["fpr"],
           "base_recall":v["baseline"]["tpr_recall"],
           "model_recall":v["model"]["tpr_recall"]}
        for x,v in out["groups"].items()}),flush=True)

if __name__=="__main__":main()
