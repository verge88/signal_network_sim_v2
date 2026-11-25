#!/usr/bin/env python3
"""Read-only diagnostic experiment on frozen TICD. Oracle interventions are
attribution controls only, never candidate detector configurations."""
from __future__ import annotations
import argparse,json,pickle,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"sip"),str(ROOT/"experiments")]
import simulate_ticd_measured_loss as sim
import ticd_measured_loss as measured
from evaluate_ticd_adaptive_loss_fusion import group
from ticd_measured_loss import labels
from ticd_sequential_physical import sequence_features
from ticd_correlated_block_evidence import block_features
from ticd_full_count_evidence import CountReference
from evaluate_ticd_full_count_evidence import pair_metrics

STRESS={"probe_delay_weak":"shift_weak_005","probe_bias_weak":"shift_weak_005",
        "probe_outage_weak":"shift_weak_005","sensor_outage_weak":"shift_weak_005",
        "passive_bias_weak":"shift_weak_005",
        "probe_delay_clean":"shift_clean_loss","probe_bias_clean":"shift_clean_loss",
        "probe_outage_clean":"shift_clean_loss","sensor_outage_clean":"shift_clean_loss",
        "passive_bias_clean":"shift_clean_loss"}
NEW_SEEDS=list(range(420001,420009))
EXTRA=("weak_0025","weak_005","weak_01","shift_weak_0025","role_redirect_loss25")

def observed(seeds,cal,kind,causal):
    gen0,rec0=sim.GEN,sim.REC
    def gen(self,node,t,override_params=None):
        p=float(self._path_loss_to_master(node))
        d=gen0(self,node,t,override_params)
        d["_diagnostic_path_p"]=p
        return d
    def record(self,t,sid,node,master_id,true_traffic,cu_result,attack_type,event_type):
        d=rec0(self,t,sid,node,master_id,true_traffic,cu_result,attack_type,event_type)
        d.update({"_diagnostic_path_p":true_traffic["_diagnostic_path_p"],
          "_diagnostic_200":true_traffic["n_200_transit"],"_diagnostic_ack":true_traffic["n_ack_transit"],
          "_diagnostic_upstream":true_traffic["n_invite_200"]})
        return d
    with patch.object(sim,"GEN",gen),patch.object(sim,"REC",record):
        raw,_,_,c=group(seeds,cal,kind,causal)
    assert all(k in raw for k in ("_diagnostic_path_p","_diagnostic_200","_diagnostic_ack","_diagnostic_upstream"))
    return raw,c

def qs(v):
    v=np.asarray(v,dtype=float)
    if not len(v):return None
    assert np.isfinite(v).all()
    return {"n":len(v),"mean":float(v.mean()),"sd":float(v.std()),"q05":float(np.quantile(v,.05)),
            "median":float(np.median(v)),"q95":float(np.quantile(v,.95))}

def metrics_mask(z,y,m):
    a=z==2
    return {"n":int(m.sum()),"attack":int((m&y).sum()),"normal":int((m&~y).sum()),
        "recall":float(a[m&y].mean()) if (m&y).any() else None,
        "FPR":float(a[m&~y].mean()) if (m&~y).any() else None}

def main():
    p=argparse.ArgumentParser();p.add_argument("--prior",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    report=json.loads((a.prior/"joint_channel_report.json").read_text())
    with (a.prior/"joint_channel_models.pkl").open("rb") as f:pkg=pickle.load(f)
    model=pkg["models"]["proposed"];fast=pkg["models"]["frozen_adaptive_fast"]
    oracle=pkg["models"]["frozen_oracle"];jr=pkg["joint_reference"]
    ref=pkg["references"]["adaptive_fast"];cal=pkg["calibration"];causal=ref.core.causal
    # Read all original per-seed outcomes, not rounded scenario averages.
    failures=[]
    for kind,mods in report["per_seed"].items():
        for r in mods["proposed"]:
            if r["confirmed_alert_FPR"]>.01 or r["false_warning_rate"]>.005:
                failures.append({"kind":kind,**r})
    print("DIAG_ORIGINAL_FAILURES",json.dumps(failures),flush=True)
    (a.out/"original_violations.json").write_text(json.dumps(failures,indent=2))
    original_frame=measured.measured_frame;original_estimate=measured.estimate
    def prepare(raw,c,intervention="none"):
        def frame(df,mode,commission=False):
            if commission or mode!="adaptive_fast" or intervention=="none":
                return original_frame(df,mode,commission)
            def estimate(x,pc,vc,mode):
                est=list(original_estimate(x,pc,vc,mode))
                est[0]=df.loc[x.index,"_diagnostic_path_p"].to_numpy(dtype=float)
                if intervention=="true_p_zero_var":est[1]=np.zeros(len(x))
                return tuple(est)
            with patch.object(measured,"estimate",estimate):
                return original_frame(df,mode,commission)
        # Only operational physical loss input is intervened on.
        # All fitted references, commissioning and thresholds stay frozen.
        with patch.object(measured,"measured_frame",frame):
            x=ref.prepare(raw,c)
        x=sequence_features(x,c,ref);x=block_features(x,c,ref)
        return labels(jr.prepare(x),raw)
    def diagnose(raw,c,kind):
        x=prepare(raw,c)
        y=raw.is_anomaly.to_numpy(dtype=bool)
        valid=x.primary_observer_valid.to_numpy(dtype=bool)
        cu=raw.cu_delivered.to_numpy()>0
        n,k=CountReference.counts(x)
        true=raw._diagnostic_path_p.to_numpy(dtype=float)
        p=x.measured_oneway_loss.to_numpy(dtype=float);err=p-true
        z=model.predict_levels(x)
        zfast=np.zeros(len(x),dtype=np.int8)
        zfast[valid]=fast.predict_levels(x.loc[valid])
        # Direct replay with unchanged reference and threshold isolates
        # estimated probability, then its uncertainty, separately.
        xp=prepare(raw,c,"true_p");zp=model.predict_levels(xp)
        xv=prepare(raw,c,"true_p_zero_var");zv=model.predict_levels(xv)
        xo=labels(pkg["references"]["oracle"].prepare(raw,c),raw)
        zo=np.zeros(len(x),dtype=np.int8);zo[valid]=oracle.predict_levels(xo.loc[valid])
        scores=model.components(x)
        names=model.columns()
        winner=np.argmax(scores,axis=1)
        a=z==2;fp=a&~y;fn=~a&y;tp=a&y
        sensor=(raw.active_sent.to_numpy()>0).astype(int)+2*(raw.passive_sent.to_numpy()>0).astype(int)
        diagnostic={"kind":kind,"rows":len(x),"attack_rows":int(y.sum()),
          "primary_missing_attack":int((y&~valid).sum()),"cu_missing_attack":int((y&valid&~cu).sum()),
          "attack_no_observed_200":int((y&(n==0)).sum()),"attack_no_observed_deficit":int((y&(k==0)).sum()),
          "true_upstream_attack":int(raw.loc[y,"_diagnostic_upstream"].sum()),
          "true_transit_deficit_attack":int((raw.loc[y,"_diagnostic_200"]-raw.loc[y,"_diagnostic_ack"]).sum()),
          "observed_transit_deficit_attack":int(k[y].sum()),
          "measured_loss_attack":qs(p[y&valid]),"true_loss_attack":qs(true[y&valid]),
          "loss_error_attack":qs(err[y&valid]),"loss_error_normal":qs(err[~y&valid]),
          "loss_error_FN":qs(err[fn&valid]),"loss_error_TP":qs(err[tp&valid]),
          "loss_error_FP":qs(err[fp&valid]),
          "loss_rmse_attack":float(np.sqrt(np.mean(err[y&valid]**2))) if (y&valid).any() else None,
          "FP_winning_channel":{key:int((fp&(winner==j)).sum()) for j,key in enumerate(names)},
          "TP_winning_channel":{key:int((tp&(winner==j)).sum()) for j,key in enumerate(names)},
          "FN_max_score_over_threshold":qs(model.score(x)[fn]/model.threshold),
          "channels":{key:{"TP":int((y&valid&(scores[:,j]>model.threshold)).sum()),
                          "FP":int((~y&valid&(scores[:,j]>model.threshold)).sum()),
                          "sole_TP":int((y&valid&(scores[:,j]>model.threshold)&((scores>model.threshold).sum(1)==1)).sum())}
                      for j,key in enumerate(names)},
          "interventions":{"measured":metrics_mask(z,y,np.ones(len(y),bool)),
            "true_p_same_uncertainty":metrics_mask(zp,y,np.ones(len(y),bool)),
            "true_p_zero_uncertainty":metrics_mask(zv,y,np.ones(len(y),bool)),
            "refitted_oracle":metrics_mask(zo,y,np.ones(len(y),bool)),
            "fast":metrics_mask(zfast,y,np.ones(len(y),bool))},
          "FN_rescued_by_true_p":int((fn&(zp==2)).sum()),"FN_rescued_by_true_p_zero_var":int((fn&(zv==2)).sum()),
          "FP_removed_by_true_p":int((fp&(zp!=2)).sum()),"FP_removed_by_true_p_zero_var":int((fp&(zv!=2)).sum()),
          "FP_new_by_true_p":int((~y&~a&(zp==2)).sum()),"TP_lost_by_true_p":int((tp&(zp!=2)).sum()),
          "by_sensor":{str(i):metrics_mask(z,y,sensor==i) for i in range(4)},
          "by_phase":{str(i):{"loss_error_attack":qs(err[y&valid&(raw.interval.to_numpy()//90==i)]),
                    **metrics_mask(z,y,raw.interval.to_numpy()//90==i)} for i in range(4)},
          "by_seed":[]}
        # Compare exact FP rows and channels with the frozen earlier run.
        for seed in raw.simulation_seed.unique():
            ix=raw.simulation_seed.to_numpy()==seed
            diagnostic["by_seed"].append({"seed":int(seed),**metrics_mask(z,y,ix),
                "FP_channels":{key:int((ix&fp&(winner==j)).sum()) for j,key in enumerate(names)},
                "FP_removed_true_p":int((ix&fp&(zp!=2)).sum()),
                "FP_removed_true_p_zero_var":int((ix&fp&(zv!=2)).sum()),
                "FP_rows":[{"node":int(raw.node_id.iloc[i]),"interval":int(raw.interval.iloc[i]),
                             "winner":names[winner[i]],"loss_error":float(err[i]),
                             "normalized_score":float(model.score(x.iloc[[i]])[0]/model.threshold)}
                            for i in np.flatnonzero(ix&fp)]})
        return diagnostic,z
    # First reproduce exact violation seeds, using no test data to tune anything.
    replay={}
    for kind in sorted({r["kind"] for r in failures}):
        ss=[r["seed"] for r in failures if r["kind"]==kind]
        raw,c=observed(ss,cal,kind,causal)
        d,_=diagnose(raw,c,kind)
        for row in d["by_seed"]:
            prior=next(r for r in failures if r["kind"]==kind and r["seed"]==row["seed"])
            assert abs(prior["confirmed_alert_FPR"]-row["FPR"])<1e-12, (kind,row["seed"],prior["confirmed_alert_FPR"],row["FPR"])
        replay[kind]=d
        print("DIAG_VIOLATION_REPLAY",json.dumps(d),flush=True)
    # Fresh paired sensor interventions on identical physical realizations.
    cases={};pairs={}
    base_cache={}
    fields=["simulation_seed","node_id","interval","is_anomaly","_diagnostic_path_p","_diagnostic_200","_diagnostic_ack","_diagnostic_upstream","cu_delivered"]
    for kind in ("shift_weak_005","shift_clean_loss",*EXTRA):
        raw,c=observed(NEW_SEEDS,cal,kind,causal)
        d,z=diagnose(raw,c,kind);cases[kind]=d
        if kind in set(STRESS.values()):base_cache[kind]=(raw[fields].copy(),z.copy())
        print("DIAG_COHORT",json.dumps({k:v for k,v in d.items() if k!="by_seed"}),flush=True)
    for kind,baseline in STRESS.items():
        raw,c=observed(NEW_SEEDS,cal,kind,causal)
        d,z=diagnose(raw,c,kind);cases[kind]=d
        nominal,zn=base_cache[baseline]
        pd.testing.assert_frame_equal(raw[fields],nominal)
        y=raw.is_anomaly.to_numpy(dtype=bool)
        pairs[kind]={"matched_baseline":baseline,"physical_rows_identical":True,
            "TP_lost_to_sensor_stress":int((y&(zn==2)&(z!=2)).sum()),
            "TP_gained_under_stress":int((y&(zn!=2)&(z==2)).sum()),
            "FP_created_by_stress":int((~y&(zn!=2)&(z==2)).sum()),
            "FP_removed_under_stress":int((~y&(zn==2)&(z!=2)).sum()),
            "nominal":metrics_mask(zn,y,np.ones(len(y),bool)),"stress":metrics_mask(z,y,np.ones(len(y),bool))}
        print("DIAG_PAIR",json.dumps({"kind":kind,**pairs[kind]}),flush=True)
        print("DIAG_COHORT",json.dumps({k:v for k,v in d.items() if k!="by_seed"}),flush=True)
    result={"source_run":45626804800,"frozen_model":model.manifest(),"original_violations":failures,
          "exact_violation_replay":replay,"fresh_paired_cases":cases,"sensor_pairs":pairs,
          "fresh_seeds":NEW_SEEDS,"oracle_controls_diagnostic_only":True,
          "no_training_or_threshold_changes":True,"real_field_validation":False}
    (a.out/"root_cause_report.json").write_text(json.dumps(result,indent=2)+"\n")
    print("DIAG_FINAL",json.dumps({"reproduced_violations":len(failures),"fresh_paired_scenarios":len(pairs),"cases":len(cases),
          "no_training_or_threshold_changes":True}),flush=True)
if __name__=="__main__":main()
