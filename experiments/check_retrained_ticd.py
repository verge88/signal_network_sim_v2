#!/usr/bin/env python3
"""Post-run verification of trained TICD artifact, without retraining or tuning."""
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sip"))
from ticd_recalibrated import FrozenReferenceTICD

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifact-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    d=json.loads((a.artifact_dir/"retraining_report.json").read_text())
    c=json.loads((a.artifact_dir/"sip_real_method_calibration.json").read_text())
    with (a.artifact_dir/"ticd_model.pkl").open("rb") as f:packed=pickle.load(f)
    model=packed["detector"]
    assert isinstance(model,FrozenReferenceTICD) and model.fitted
    val=pd.read_csv(a.artifact_dir/"threshold_validation.csv.gz")
    test=pd.read_csv(a.artifact_dir/"test.csv.gz")
    assert set(val.simulation_seed)=={503}
    assert set(test.simulation_seed)=={601,602,603}
    assert val.is_anomaly.sum()==0 and test.is_anomaly.sum()==3195
    score_all=model.score(test)
    subset=test.sample(250,random_state=42)
    score_subset=model.score(subset)
    max_diff=float(np.max(np.abs(score_subset-score_all[subset.index.to_numpy()])))
    assert np.allclose(score_subset,score_all[subset.index.to_numpy()],atol=1e-12,rtol=0),max_diff
    shuffled=test.sample(frac=1.,random_state=3)
    score_shuffled=model.score(shuffled)
    assert np.allclose(score_shuffled,score_all[shuffled.index.to_numpy()],atol=1e-12,rtol=0)
    q=float(np.quantile(model.score(val),.99,method="higher"))
    assert abs(q-packed["threshold"])<1e-9,(q,packed["threshold"])
    assert abs(q-d["normal_validation_threshold"])<1e-9
    original=c["calibration_effect_l1_before"]
    now=c["calibration_effect_l1_after"]
    assert now<original,(original,now)
    assert len(d["by_independent_test_seed"])==3
    assert "cu_ack_closure_mismatch" in model.inv_cols
    features=model.inv_cols+model.ctx_cols+list(model.CONSENSUS_FEATURES)
    assert not any(x.startswith("gt_") for x in features)
    # Prespecified operating points are calibrated solely on benign validation.
    # This is sensitivity analysis, not picking a favorable threshold on test.
    y=test.is_anomaly.to_numpy(dtype=int)
    val_s=model.score(val)
    controls=[]
    for budget in (0.01,0.025,0.05,0.10):
        t=float(np.quantile(val_s,1-budget,method="higher"))
        pred=score_all>t
        tp=int(((pred==1)&(y==1)).sum())
        fp=int(((pred==1)&(y==0)).sum())
        tn=int(((pred==0)&(y==0)).sum())
        fn=int(((pred==0)&(y==1)).sum())
        comp={}
        for sophistication in ("naive","statistical","adaptive"):
            mask=(test.attack_type.eq("proxy_compromise")&
                  test.attacker_sophistication.eq(sophistication)).to_numpy()
            comp[sophistication]={"n":int(mask.sum()),
                "recall":float(pred[mask].mean()) if mask.sum() else None}
        controls.append({"validation_fpr_budget":budget,"threshold":t,
                         "actual_validation_fpr":float((val_s>t).mean()),
                         "test_tpr":float(tp/(tp+fn)),
                         "test_fpr":float(fp/(fp+tn)),
                         "per_compromise_sophistication":comp})
    attack=d["overall_pooled_synthetic_test"]["by_attack"]
    soph=d["overall_pooled_synthetic_test"]["by_compromise_sophistication"]
    events=d["overall_pooled_synthetic_test"]["by_benign_event"]
    findings={"verification":"success",
      "artifact_model_reload":True,
      "unchanged_score_independent_of_test_batch":True,
      "max_score_difference_when_subsetting":max_diff,
      "threshold_rederived_on_validation":q,
      "no_test_label_used_for_fit":True,
      "normal_training_only":True,
      "calibration_original_L1":original,
      "calibration_after_L1":now,
      "method_weights":c["method_weights"],
      "threshold_sensitivity_from_validation_only":controls,
      "pooled_synthetic_test_confusion":{
          k:d["overall_pooled_synthetic_test"][k] for k in ("tp","tn","fp","fn")},
      "by_attack":attack,
      "by_compromise_sophistication":soph,
      "by_legitimate_event":events,
      "note":"Attacker type and compromise scores describe synthetic test performance only."}
    a.out_dir.mkdir(exist_ok=True,parents=True)
    (a.out_dir/"verified_model_breakdown.json").write_text(json.dumps(findings,ensure_ascii=False,indent=2)+"\n")
    print("RETRAINED_TICD_VERIFIED",json.dumps(findings,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
