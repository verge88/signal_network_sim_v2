"""Frozen paired evaluation on the 40 SIP scenarios used by the previous study."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sip"), str(ROOT / "experiments")]
from evaluate_ticd_adaptive_loss_fusion import KINDS, NATIVE, WEAK, group
from evaluate_ticd_full_count_evidence import POLICY
from ticd_measured_loss import RAW
from ticd_report_consistency import ReportConsistencyTICD, RATIOS, extra_evidence

COLUMNS = list(dict.fromkeys([*RAW, *RATIOS, "obs_method_dominance_ratio",
                             "audit_observed_count", "audit_mismatch_count"]))
MODELS = ("block", "block_report", "full", "normalized")
REPLICATES = 10


def counts(levels, labels):
    attack = np.asarray(labels, dtype=bool)
    alert = levels == 2
    return dict(attack=int(attack.sum()), normal=int((~attack).sum()),
                tp=int((alert & attack).sum()), fp=int((alert & ~attack).sum()),
                fw=int(((levels == 1) & ~attack).sum()))


def evaluate(package, shard, out):
    prior = package["frozen_package"]
    ref = prior["references"]["adaptive_fast"]
    model = ReportConsistencyTICD(ref, prior["models"]["frozen_block"])
    rows = []
    for i, kind in enumerate(KINDS):
        seed = 610001 + i * 100 + shard
        raw, _, _, commissioning = group([seed], prior["calibration"], kind, ref.core.causal)
        obs = raw[COLUMNS]
        commissioning = commissioning[list(RAW)]
        base = model.baseline(obs, commissioning)
        evidence = extra_evidence(obs)
        norm = extra_evidence(obs, normalize=True)
        levels = {"block": base, "block_report": np.where(evidence.report, 2, base),
                  "full": np.where(evidence.any(axis=1), 2, base),
                  "normalized": np.where(norm.any(axis=1), 2, base)}
        blind = ~obs.primary_observer_valid & obs.audit_observed_count.eq(0)
        assert not np.any(levels["full"][blind] > 0)
        if i == 0:
            cut = obs.interval.lt(150)
            np.testing.assert_array_equal(model.predict_levels(obs.loc[cut], commissioning),
                                          levels["full"][cut])
            pd.testing.assert_frame_equal(evidence, extra_evidence(obs.sample(frac=1, random_state=19)).loc[obs.index])
        if kind == "latent_identical":
            twin, _, _, c = group([seed], prior["calibration"], "clean_loss10", ref.core.causal)
            pd.testing.assert_frame_equal(obs, twin[COLUMNS])
            np.testing.assert_array_equal(levels["full"], model.predict_levels(twin[COLUMNS], c[list(RAW)]))
        row = {"seed": seed, "kind": kind, "models": {m: counts(v, raw.is_anomaly)
                                                      for m, v in levels.items()},
               "blind_attack": int((blind & raw.is_anomaly.astype(bool)).sum())}
        rows.append(row)
        print("TICD_COHORT", json.dumps(row), flush=True)
    (out / f"part-{shard}.json").write_text(json.dumps(rows))


def summarize(folder):
    rows = [r for j in range(REPLICATES) for r in json.loads((folder / f"part-{j}.json").read_text())]
    assert len(rows) == len({r["seed"] for r in rows}) == len(KINDS) * REPLICATES
    assert all(sum(r["kind"] == k for r in rows) == REPLICATES for k in KINDS)
    all_attack = [k for k in KINDS if any(r["models"]["full"]["attack"] for r in rows if r["kind"] == k)]
    results = {}
    for model in MODELS:
        by_kind = {}
        for k in KINDS:
            trials = [r["models"][model] for r in rows if r["kind"] == k]
            total = {key: sum(t[key] for t in trials) for key in trials[0]}
            by_kind[k] = {**total, "recall": total["tp"] / total["attack"] if total["attack"] else None}
        all_trials = [r["models"][model] for r in rows]
        fpr = [t["fp"] / t["normal"] for t in all_trials]
        fwr = [t["fw"] / t["normal"] for t in all_trials]
        d = {"cohorts": by_kind, "max_FPR": max(fpr), "max_FWR": max(fwr),
             "FPR_pass": sum(x <= .01 for x in fpr), "FWR_pass": sum(x <= .005 for x in fwr)}
        for label, kinds in (("native", NATIVE), ("weak", WEAK), ("policy", POLICY), ("all_attack", all_attack)):
            d[label] = float(np.mean([by_kind[k]["recall"] for k in kinds]))
        total = {key: sum(t[key] for t in all_trials) for key in all_trials[0]}
        d["micro"] = total["tp"] / total["attack"]
        d["pooled_FPR"] = total["fp"] / total["normal"]
        d["counts"] = total
        results[model] = d
    d = results["full"]
    report = {"results": results, "trials": rows, "scope": "all slave rows including unobservable attacks",
              "target_pass": min(d["native"], d["weak"], d["all_attack"], d["micro"]) >= .90
              and min(d["FPR_pass"], d["FWR_pass"]) == len(rows)}
    (folder / "report.json").write_text(json.dumps(report, indent=2))
    for name, metrics in results.items():
        print(name, json.dumps({k: v for k, v in metrics.items() if k != "cohorts"}))
    print("TARGET_PASS", report["target_pass"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path)
    parser.add_argument("--shard", type=int, choices=range(REPLICATES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.shard is None:
        summarize(args.out)
    else:
        with args.package.open("rb") as f:
            evaluate(pickle.load(f), args.shard, args.out)
