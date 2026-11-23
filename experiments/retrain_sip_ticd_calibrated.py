#!/usr/bin/env python3
"""Retrain TICD from simulated SIP data calibrated on *real benign LTE* PCAP.

Three independent seed partitions:
 - train: simulated clean network with legitimate transient events
 - threshold validation: separate clean simulation seeds (never test)
 - test: separate simulator seeds, with seeded attacks + compromise reports

Only normal examples are used to fit TICD or select a 1%-FPR threshold.
No synthetic label enters detector training. Evaluation labels are generated
by the simulator and DO NOT constitute external attack validation.

Unlike sip/train_ticd.py, train and test are NOT random rows from the same
simulation and threshold is NEVER learned from test normal samples.
"""
from __future__ import annotations
import argparse,hashlib,json,pickle,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_score,recall_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
from sip_sim_v4_claude import (SIPSimulator,SimulationConfig,AttackScenario,
    NormalScenario,AttackType,NormalEventType,AttackerSophistication,
    SIP_METHODS,add_zone_deviation_features,NodeRole)
from ticd_recalibrated import FrozenReferenceTICD

def simulation(seed:int,calibration:dict,duration_hours:float,attacks_enabled:bool,loss_multiplier:float=1.0):
    cfg=SimulationConfig(seed=seed,duration_hours=duration_hours,interval_s=20.,
                         n_sbc=2,n_proxy=3,n_registrar=2,n_ua_gw=4,n_redirect=2,
                         master_node_ids=[0,1])
    n=int(cfg.duration_hours*3600/cfg.interval_s)
    s=lambda q:max(1,min(n-1,int(n*q)))
    a=[]
    if attacks_enabled:
        a=[
        AttackScenario(AttackType.SIGNALING_DOS,[7],s(.12),s(.25),intensity=3.),
        AttackScenario(AttackType.SIGNALING_DOS,[8],s(.45),s(.58),intensity=4.),
        AttackScenario(AttackType.REGISTER_HIJACK,[5],s(.20),s(.48),intensity=3.),
        AttackScenario(AttackType.REGISTER_BRUTEFORCE,[6],s(.50),s(.77),intensity=3.),
        AttackScenario(AttackType.SPIT,[9],s(.28),s(.63),intensity=3.),
        # IMPORTANT: simulator's compromised state persists after an attack
        # starts; so these attacks run to n, avoiding compromised-labelled-normal.
        AttackScenario(AttackType.PROXY_COMPROMISE,[2],s(.23),n,
                       intensity=3.,sophistication=AttackerSophistication.NAIVE,
                       covert_redirect_fraction=.10),
        AttackScenario(AttackType.PROXY_COMPROMISE,[3],s(.40),n,
                       intensity=3.,sophistication=AttackerSophistication.STATISTICAL,
                       covert_redirect_fraction=.15),
        AttackScenario(AttackType.PROXY_COMPROMISE,[4],s(.57),n,
                       intensity=3.,sophistication=AttackerSophistication.ADAPTIVE,
                       covert_redirect_fraction=.20)]
    events=[
        NormalScenario(NormalEventType.TRAFFIC_SPIKE,[7,8],s(.06),s(.11),intensity=3.),
        NormalScenario(NormalEventType.MAINTENANCE,[6],s(.15),s(.22),intensity=2.),
        NormalScenario(NormalEventType.REROUTE,[9,10],s(.80),s(.87),intensity=2.),
        NormalScenario(NormalEventType.PRESENCE_SURGE,[5],s(.69),s(.77),intensity=2.)]
    sim=SIPSimulator(cfg,a,events)
    ws=calibration["method_weights"]
    orig={}
    for node in sim.topology.nodes.values():
        if node.role!=NodeRole.SLAVE:continue
        dist={m:float(node.method_dist[m])*float(ws.get(m,1.)) for m in SIP_METHODS}
        total=sum(dist.values())
        node.method_dist={m:float(v/total) for m,v in dist.items()}
        orig[node.node_id]=node.method_dist
    if not (0.1 <= float(loss_multiplier) <= 25.0):
        raise ValueError("Loss multiplier must be within [0.1,25]")
    if float(loss_multiplier)!=1.0:
        for link in sim.topology.links.values():
            link.base_loss_prob*=float(loss_multiplier)
    sim.run()
    df=add_zone_deviation_features(sim.to_dataframe())
    if df.empty:raise RuntimeError("Simulated data empty")
    if df.duplicated(["interval","node_id"]).any():
        raise ValueError("Duplicated interval/node pairs")
    if attacks_enabled and df["is_anomaly"].sum()<100:
        raise ValueError("Expected simulated attack labels")
    if not attacks_enabled and df["is_anomaly"].sum():
        raise ValueError("Train/validation simulation contains attacks")
    if not attacks_enabled and df["is_compromised"].sum():
        raise ValueError("Train/validation simulation contains compromised nodes")
    df["simulation_seed"]=seed
    return df,{"seed":seed,"hours":duration_hours,"intervals":n,"rows":len(df),
       "attack_count":int(df["is_anomaly"].sum()),
       "normal_count":int((df["is_anomaly"]==0).sum()),
       "link_loss_multiplier":float(loss_multiplier),
       "attack_schedules":[{"type":x.attack_type.value,"target_nodes":x.target_nodes,
                            "start":x.start_interval,"end":x.end_interval,
                            "sophistication":x.sophistication.value} for x in a],
       "normal_schedules":[{"type":x.event_type.value,"start":x.start_interval,
                            "end":x.end_interval,"nodes":x.target_nodes} for x in events]}

def evaluate(model,df,threshold):
    y=df["is_anomaly"].to_numpy(dtype=int)
    score=model.score(df)
    yp=(score>threshold).astype(int)
    tn,fp,fn,tp=map(int,confusion_matrix(y,yp,labels=[0,1]).ravel())
    def frac(a,b):return float(a/b) if b else None
    def rec(mask):
        n=int(mask.sum())
        return {"n":n,"recall":frac(int(yp[mask].sum()),n) if n else None}
    by_attack={name:rec((df["attack_type"]==name).to_numpy())
         for name in ["signaling_dos","register_hijack","register_bruteforce","spit","proxy_compromise"]}
    by_soph={name:rec(((df["attack_type"]=="proxy_compromise")&
                         (df["attacker_sophistication"]==name)).to_numpy())
         for name in ["naive","statistical","adaptive"]}
    benign_events={name:{"n":int(mask.sum()),"fpr":frac(int(yp[mask].sum()),int(mask.sum()))}
           for name in ["none","traffic_spike","maintenance","reroute","presence_surge"]
           for mask in [(df["normal_event"].astype(str).to_numpy()==name)&(y==0)]}
    return {"rows":len(df),"normal":tn+fp,"attacks":tp+fn,
       "threshold":float(threshold),
       "tp":tp,"fp":fp,"fn":fn,"tn":tn,
       "tpr_recall":frac(tp,tp+fn),"fpr":frac(fp,fp+tn),
       "precision":frac(tp,tp+fp),"f1":float(f1_score(y,yp,zero_division=0)),
       "auc_roc":float(roc_auc_score(y,score)) if len(set(y))>1 else None,
       "auc_pr":float(average_precision_score(y,score)) if len(set(y))>1 else None,
       "by_attack":by_attack,"by_compromise_sophistication":by_soph,
       "by_benign_event":benign_events,
       "score_quantiles":[float(x) for x in np.quantile(score,[0,0.01,0.5,0.95,0.99,1])]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--calibration",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,default=Path("results_ticd_calibrated"))
    ap.add_argument("--duration-hours",type=float,default=2.)
    a=ap.parse_args()
    if a.duration_hours<1.5:raise ValueError("Need >=1.5 hours for scenario scheduling")
    a.out_dir.mkdir(exist_ok=True,parents=True)
    cal=json.loads(a.calibration.read_text())
    assert cal["calibration_kind"]=="partial_benign_SIP_method_profile"
    assert cal["source_archive_sha256"]=="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
    seeds={"train":[401,402],"threshold_validation":[503],"test":[601,602,603]}
    data={};manifests=[];started=time.time()
    for group,ss in seeds.items():
        frames=[]
        for seed in ss:
            d,m=simulation(seed,cal,a.duration_hours,attacks_enabled=(group=="test"))
            frames.append(d);m["split"]=group;manifests.append(m)
            print("SIM_RUN",json.dumps(m,ensure_ascii=False)[:4000],flush=True)
        data[group]=pd.concat(frames,ignore_index=True)
        data[group].to_csv(a.out_dir/(group+".csv.gz"),index=False,compression="gzip")
    for x in ("train","threshold_validation"):
        if data[x]["is_anomaly"].any():raise ValueError("Attack labels in training or validation")
    if len(set(seeds["train"]) & set(seeds["test"]))>0:raise ValueError("Non-independent seeds")
    if len(set(seeds["threshold_validation"]) & set(seeds["test"]))>0:
        raise ValueError("Validation/test seed overlap")
    model=FrozenReferenceTICD(seed=42)
    model.fit(data["train"])
    val_scores=model.score(data["threshold_validation"])
    if not np.isfinite(val_scores).all():raise RuntimeError("Validation scores not finite")
    # Strict > threshold; an exact tied score is not a false positive.
    thr=float(np.quantile(val_scores,.99,method="higher"))
    validation_fpr=float((val_scores>thr).mean())
    if validation_fpr>.011:raise RuntimeError("Validation FPR exceeds threshold budget")
    model_pack={"detector":model,"threshold":thr,"calibration":cal,
        "split_seeds":seeds,"simulator":"sip_sim_v4_claude.py",
        "score_normalization":"fit_normal_only_frozen_P1_P99",
        "decision_rule":"score > threshold"}
    with (a.out_dir/"ticd_model.pkl").open("wb") as f:pickle.dump(model_pack,f,pickle.HIGHEST_PROTOCOL)
    test=data["test"];per_seed=[]
    for seed in seeds["test"]:
        d=test[test["simulation_seed"]==seed]
        stats=evaluate(model,d,thr)
        stats["simulation_seed"]=seed
        per_seed.append(stats)
        print("EVAL_SEED",json.dumps({k:v for k,v in stats.items()
              if not isinstance(v,dict)},ensure_ascii=False),flush=True)
    overall=evaluate(model,test,thr)
    measures=("auc_roc","auc_pr","f1","fpr","tpr_recall","precision")
    aggregate={}
    for m in measures:
        q=[x[m] for x in per_seed if x[m] is not None]
        aggregate[m]={"mean":float(np.mean(q)) if q else None,
                      "sd":float(np.std(q,ddof=1)) if len(q)>1 else None,
                      "n_simulation_seeds":len(q)}
    simulator_path=ROOT/"sip"/"sip_sim_v4_claude.py"
    digest=hashlib.sha256(simulator_path.read_bytes()).hexdigest()
    report={
       "scope":"retrained original SIP TICD on simulated clean data with partial real PCAP method-profile calibration",
       "not_external_ticd_validation":True,
       "calibration_kind":cal["calibration_kind"],
       "real_calibration_source_sha256":cal["source_archive_sha256"],
       "simulator_source_sha256":digest,
       "train_seed_partition":seeds,
       "duration_hours_per_seed":a.duration_hours,
       "attack_labels_used_for_training":False,
       "training_rows":len(data["train"]),
       "threshold_validation_rows":len(data["threshold_validation"]),
       "test_rows":len(test),
       "normal_train_count":int((data["train"]["is_anomaly"]==0).sum()),
       "normal_validation_count":int((data["threshold_validation"]["is_anomaly"]==0).sum()),
       "test_attack_count":int(test["is_anomaly"].sum()),
       "test_normal_count":int((test["is_anomaly"]==0).sum()),
       "model":"sip.ticd_recalibrated.FrozenReferenceTICD",
       "component_weights":{"invariant":model.w_inv,"context":model.w_ctx,"consensus":model.w_cons},
       "invariant_features":model.inv_cols,"context_features":model.ctx_cols,
       "context_covariance_fitted":model.robust_cov is not None,
       "frozen_component_reference":model.component_reference,
       "frozen_consensus_sd":model.consensus_scales,
       "normal_validation_threshold":thr,
       "validation_fpr_at_threshold":validation_fpr,
       "overall_pooled_synthetic_test":overall,
       "by_independent_test_seed":per_seed,
       "mean_and_sd_over_independent_test_seeds":aggregate,
       "simulation_schedules":manifests,
       "wall_seconds":time.time()-started,
       "external_real_attack_recall":None,"external_real_fpr":None,
       "scientific_limitation":"Training and detection evaluation are simulation-to-simulation; real-world SIP only calibrates selected conditional benign method shares. Original 'gt_' labels not used for score or threshold."}
    (a.out_dir/"retraining_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    (a.out_dir/"model_manifest.json").write_text(json.dumps({
      "detector_class":"FrozenReferenceTICD",
      "threshold":thr,"features":model.inv_cols+model.ctx_cols+list(model.CONSENSUS_FEATURES),
      "seed_partitions":seeds,
      "calibration_archive_sha256":cal["source_archive_sha256"],
      "simulator_source_sha256":digest,
      "pickle_load_warning":"Only load trusted pickles in a matching Python environment"},indent=2)+"\n")
    print("TRAINING_COMPLETE",json.dumps({k:v for k,v in report.items() if k in
      ("training_rows","threshold_validation_rows","test_rows","normal_validation_threshold",
       "validation_fpr_at_threshold","context_covariance_fitted","mean_and_sd_over_independent_test_seeds",
       "test_attack_count","test_normal_count")},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
