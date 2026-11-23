"""Hard-negative and feature-robust validation for the SS7 INV detector.

Purpose
-------
Previous calibrated validation showed:
* OWN OR INV did not improve episode recall and increased false positives.
* context recognition was too sparse to provide reliable per-window suppression.
* INV Random Forest remained the useful attack-scoring channel, but temporal
  test FPR drifted above the validation target on some topologies.

This experiment therefore keeps a single INV attack channel and changes only
training/calibration:
1. normal operational events are explicit hard negatives and may receive extra
   training weight;
2. several invariant feature families are compared, because temporal z-features
   can react strongly to benign load changes;
3. a global validation quantile is compared with a block-robust threshold;
4. architecture selection uses development seeds only;
5. final metrics are reported once on disjoint evaluation seeds.

No evaluation-seed test window participates in model/threshold selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from ss7.fix import (
    INV,
    MASK_FAMILIES,
    NORMAL_EVENTS,
    RngBook,
    SimConfig,
    blocked_split,
    build_dataset,
    episode_metrics,
)
from ss7.validate_calibrated_detectors import (
    INV_CAL_RF,
    add_episode_parameters,
    all_derived,
    detector_metrics,
    episode_recall_by_group,
    fpr_threshold,
    prep,
    tx,
)
from ss7.validate_context_aware_detector import (
    annotate_context,
    audit_reroute_controls,
    purge_cross_split_benign_events,
)


OPERATING_FPR = 0.01
DEVELOPMENT_FPR_TOLERANCE = 0.015
DEVELOPMENT_EVENT_FPR_TOLERANCE = 0.05
MIN_BLOCK_NEGATIVES = 400

HARD_NEGATIVE_WEIGHTS = (1.0, 2.0, 4.0, 8.0)
CALIBRATION_TARGETS = (0.005, 0.01)
THRESHOLD_MODES = ("global", "block_robust")


def invariant_feature_families(frame: pd.DataFrame) -> Dict[str, List[str]]:
    raw = [c for c in INV if c in frame.columns]
    mean_std = []
    z = []
    zone = []
    for c in raw:
        for suffix, target in [
            ("__mean", mean_std),
            ("__std", mean_std),
            ("__z", z),
            ("__dev_zone", zone),
        ]:
            name = c + suffix
            if name in frame.columns:
                target.append(name)

    return {
        "inv_raw": raw,
        "inv_raw_history": raw + mean_std,
        "inv_raw_zone": raw + zone,
        "inv_raw_history_zone": raw + mean_std + zone,
        "inv_all_derived": all_derived(frame, INV),
    }


def contiguous_time_blocks(frame: pd.DataFrame) -> List[np.ndarray]:
    times = np.array(sorted(frame["t"].unique()), dtype=int)
    if len(times) == 0:
        return []
    cuts = np.where(np.diff(times) > 1)[0] + 1
    return [x for x in np.split(times, cuts) if len(x)]


def calibrate_threshold(
    validation: pd.DataFrame,
    scores: np.ndarray,
    target_fpr: float,
    mode: str,
) -> Tuple[float, dict]:
    y = validation["label"].to_numpy(int)
    neg = y == 0
    global_thr = fpr_threshold(scores[neg], target_fpr)
    if mode == "global":
        return float(global_thr), {
            "mode": "global",
            "global_threshold": float(global_thr),
            "selected_threshold": float(global_thr),
            "threshold_lift": 0.0,
            "eligible_blocks": 0,
        }

    if mode != "block_robust":
        raise ValueError(mode)

    candidates = [float(global_thr)]
    block_rows = []
    for idx, times in enumerate(contiguous_time_blocks(validation), start=1):
        mask = neg & validation["t"].isin(times).to_numpy()
        n = int(mask.sum())
        if n < MIN_BLOCK_NEGATIVES:
            block_rows.append({
                "block": idx,
                "negative_rows": n,
                "eligible": False,
                "q": None,
            })
            continue
        q = fpr_threshold(scores[mask], target_fpr)
        candidates.append(float(q))
        block_rows.append({
            "block": idx,
            "negative_rows": n,
            "eligible": True,
            "q": float(q),
        })

    selected = float(max(candidates))
    return selected, {
        "mode": "block_robust",
        "global_threshold": float(global_thr),
        "selected_threshold": selected,
        "threshold_lift": float(selected - global_thr),
        "eligible_blocks": int(sum(x["eligible"] for x in block_rows)),
        "blocks": block_rows,
    }


def training_sample_weights(frame: pd.DataFrame, hard_negative_weight: float) -> np.ndarray:
    y = frame["label"].to_numpy(int)
    context = frame["benign_event_kind_truth"].astype(str).to_numpy()
    w = np.ones(len(frame), dtype=float)
    hard = (y == 0) & (context != "background")
    w[hard] = float(hard_negative_weight)
    return w


def fit_inv_model(
    train: pd.DataFrame,
    cols: Sequence[str],
    seed: int,
    hard_negative_weight: float,
):
    imp, sc, xtr = prep(train, cols)
    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=int(seed),
        n_jobs=-1,
        **INV_CAL_RF,
    )
    model.fit(
        xtr,
        train["label"].to_numpy(int),
        sample_weight=training_sample_weights(train, hard_negative_weight),
    )
    return imp, sc, model


def attack_metrics(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict:
    return detector_metrics(frame, scores, threshold)


def event_fpr_rows(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> List[dict]:
    y = frame["label"].to_numpy(int)
    event = frame["benign_event_kind_truth"].astype(str).to_numpy()
    fire = np.asarray(scores) > float(threshold)
    rows = []
    for kind in sorted(set(event[y == 0])):
        mask = (y == 0) & (event == kind)
        rows.append({
            "normal_event": str(kind),
            "rows": int(mask.sum()),
            "alerts": int(np.sum(fire[mask])),
            "window_fpr": float(np.mean(fire[mask])) if mask.any() else float("nan"),
        })
    return rows


def max_reliable_event_fpr(rows: Sequence[dict], min_rows: int = 20) -> float:
    vals = [
        float(r["window_fpr"])
        for r in rows
        if int(r["rows"]) >= min_rows
        and r["normal_event"] != "background"
        and np.isfinite(float(r["window_fpr"]))
    ]
    return float(max(vals)) if vals else 0.0


def build_world(seed: int, days: float, n_compromise: int, n_normal: int):
    cfg = SimConfig(seed=int(seed), days=float(days))
    frame, sim, episodes = build_dataset(
        cfg,
        episodes_kw={
            "n_compromise": int(n_compromise),
            "n_normal": int(n_normal),
            "n_ext": 6,
        },
    )
    reroute = audit_reroute_controls(sim, episodes)
    frame = annotate_context(frame, episodes)
    frame = add_episode_parameters(frame, episodes)
    tr, va, te = blocked_split(frame, cfg, RngBook(seed)["split"])
    tr, va, te, split_audit = purge_cross_split_benign_events(tr, va, te)
    return frame, cfg, episodes, tr, va, te, {
        "reroute": reroute,
        "benign_split": split_audit,
    }


def evaluate_candidate_on_seed(
    seed: int,
    days: float,
    candidate: dict,
    *,
    n_compromise: int,
    n_normal: int,
) -> dict:
    frame, cfg, episodes, tr, va, te, audit = build_world(
        seed, days, n_compromise, n_normal
    )
    families = invariant_feature_families(frame)
    cols = families[candidate["feature_family"]]

    imp, sc, model = fit_inv_model(
        tr,
        cols,
        seed,
        candidate["hard_negative_weight"],
    )
    va_scores = model.predict_proba(tx(va, cols, imp, sc))[:, 1]
    te_scores = model.predict_proba(tx(te, cols, imp, sc))[:, 1]
    threshold, threshold_diag = calibrate_threshold(
        va,
        va_scores,
        candidate["target_fpr"],
        candidate["threshold_mode"],
    )

    metrics = attack_metrics(te, te_scores, threshold)
    event_rows = event_fpr_rows(te, te_scores, threshold)
    rate_rows = episode_recall_by_group(
        te,
        te_scores - threshold,
        0.0,
        "episode_hidden_rate",
    )
    family_rows = episode_recall_by_group(
        te,
        te_scores - threshold,
        0.0,
        "episode_family_truth",
    )
    return {
        "seed": int(seed),
        "feature_family": candidate["feature_family"],
        "n_features": int(len(cols)),
        "hard_negative_weight": float(candidate["hard_negative_weight"]),
        "target_fpr": float(candidate["target_fpr"]),
        "threshold_mode": candidate["threshold_mode"],
        "threshold": float(threshold),
        "threshold_diagnostics": threshold_diag,
        "metrics": metrics,
        "event_fpr": event_rows,
        "max_reliable_event_fpr": max_reliable_event_fpr(event_rows),
        "rate_rows": rate_rows,
        "family_rows": family_rows,
        "test_rows": int(len(te)),
        "test_episodes": int(te.loc[te["label"] == 1, "episode_id"].nunique()),
        "audit": audit,
    }


def candidate_key(candidate: dict) -> str:
    return (
        f'{candidate["feature_family"]}'
        f'/w{candidate["hard_negative_weight"]:g}'
        f'/fpr{candidate["target_fpr"]:g}'
        f'/{candidate["threshold_mode"]}'
    )


def candidate_grid() -> List[dict]:
    rows = []
    for feature_family in [
        "inv_raw",
        "inv_raw_history",
        "inv_raw_zone",
        "inv_raw_history_zone",
        "inv_all_derived",
    ]:
        for weight in HARD_NEGATIVE_WEIGHTS:
            for target in CALIBRATION_TARGETS:
                for mode in THRESHOLD_MODES:
                    rows.append({
                        "feature_family": feature_family,
                        "hard_negative_weight": float(weight),
                        "target_fpr": float(target),
                        "threshold_mode": mode,
                    })
    return rows


def summarize_development(candidate: dict, results: Sequence[dict]) -> dict:
    recall = np.asarray(
        [r["metrics"]["episode_recall"] for r in results], dtype=float
    )
    fpr = np.asarray(
        [r["metrics"]["window_fpr_direct"] for r in results], dtype=float
    )
    ap = np.asarray([r["metrics"]["ap"] for r in results], dtype=float)
    event_fpr = np.asarray(
        [r["max_reliable_event_fpr"] for r in results], dtype=float
    )

    max_fpr = float(np.nanmax(fpr))
    mean_fpr = float(np.nanmean(fpr))
    worst_event = float(np.nanmax(event_fpr))
    min_recall = float(np.nanmin(recall))
    mean_recall = float(np.nanmean(recall))
    mean_ap = float(np.nanmean(ap))

    fpr_violation = max(
        0.0,
        max_fpr - DEVELOPMENT_FPR_TOLERANCE,
    )
    event_violation = max(
        0.0,
        worst_event - DEVELOPMENT_EVENT_FPR_TOLERANCE,
    )
    feasible = bool(fpr_violation == 0.0 and event_violation == 0.0)

    # Lexicographic selection: first satisfy robust false-positive constraints;
    # then maximize worst-seed and mean episode recall, then AP and compactness.
    selection_key = (
        0 if feasible else 1,
        float(fpr_violation + event_violation),
        -min_recall,
        -mean_recall,
        -mean_ap,
        int(results[0]["n_features"]),
    )

    return {
        "candidate": candidate_key(candidate),
        **candidate,
        "n_features": int(results[0]["n_features"]),
        "development_seeds": [int(r["seed"]) for r in results],
        "mean_episode_recall": mean_recall,
        "min_episode_recall": min_recall,
        "mean_test_fpr": mean_fpr,
        "max_test_fpr": max_fpr,
        "worst_reliable_event_fpr": worst_event,
        "mean_ap": mean_ap,
        "fpr_constraint": DEVELOPMENT_FPR_TOLERANCE,
        "event_fpr_constraint": DEVELOPMENT_EVENT_FPR_TOLERANCE,
        "feasible": feasible,
        "selection_key": list(selection_key),
    }


def aggregate_metrics(rows: pd.DataFrame) -> dict:
    out = {}
    for col in [
        "episode_recall",
        "window_fpr_direct",
        "auc",
        "ap",
        "median_ttd_seconds",
        "window_tpr",
        "alerts_per_day_network",
    ]:
        x = rows[col].to_numpy(float)
        x = x[np.isfinite(x)]
        out[col + "_mean"] = float(np.mean(x)) if len(x) else float("nan")
        out[col + "_std"] = (
            float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        )
        out[col + "_min"] = float(np.min(x)) if len(x) else float("nan")
        out[col + "_max"] = float(np.max(x)) if len(x) else float("nan")
    return out


def aggregate_event_rows(rows: pd.DataFrame) -> List[dict]:
    out = []
    for event, g in rows.groupby("normal_event"):
        n = int(g["rows"].sum())
        alerts = int(g["alerts"].sum())
        out.append({
            "normal_event": event,
            "rows": n,
            "alerts": alerts,
            "pooled_window_fpr": float(alerts / n) if n else float("nan"),
            "mean_seed_fpr": float(g["window_fpr"].mean()),
            "max_seed_fpr": float(g["window_fpr"].max()),
        })
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--development-seeds",
        nargs="+",
        type=int,
        default=[42, 1337, 5151],
    )
    ap.add_argument(
        "--evaluation-seeds",
        nargs="+",
        type=int,
        default=[7001, 7002, 7003, 7004, 7005],
    )
    ap.add_argument("--development-days", type=float, default=1.0)
    ap.add_argument("--evaluation-days", type=float, default=1.25)
    ap.add_argument(
        "--output-dir",
        default="results/ss7_hard_negative_inv",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    development_rows = []
    development_detail = {}
    for candidate in candidate_grid():
        key = candidate_key(candidate)
        per_seed = []
        for seed in args.development_seeds:
            result = evaluate_candidate_on_seed(
                int(seed),
                float(args.development_days),
                candidate,
                n_compromise=24,
                n_normal=72,
            )
            per_seed.append(result)
        summary = summarize_development(candidate, per_seed)
        development_rows.append(summary)
        development_detail[key] = per_seed

    development_rows.sort(key=lambda x: tuple(x["selection_key"]))
    selected = development_rows[0]
    selected_candidate = {
        "feature_family": selected["feature_family"],
        "hard_negative_weight": selected["hard_negative_weight"],
        "target_fpr": selected["target_fpr"],
        "threshold_mode": selected["threshold_mode"],
    }

    pd.DataFrame(development_rows).to_csv(
        outdir / "development_candidate_summary.csv",
        index=False,
    )

    eval_metric_rows = []
    eval_event_rows = []
    eval_rate_rows = []
    eval_family_rows = []
    eval_seed_reports = []
    for seed in args.evaluation_seeds:
        result = evaluate_candidate_on_seed(
            int(seed),
            float(args.evaluation_days),
            selected_candidate,
            n_compromise=18,
            n_normal=42,
        )
        m = {
            "seed": int(seed),
            **result["metrics"],
            "threshold": result["threshold"],
            "test_rows": result["test_rows"],
            "test_episodes": result["test_episodes"],
        }
        eval_metric_rows.append(m)
        for row in result["event_fpr"]:
            eval_event_rows.append({"seed": int(seed), **row})
        for row in result["rate_rows"]:
            eval_rate_rows.append({"seed": int(seed), **row})
        for row in result["family_rows"]:
            eval_family_rows.append({"seed": int(seed), **row})
        eval_seed_reports.append({
            "seed": int(seed),
            "threshold": result["threshold"],
            "threshold_diagnostics": result["threshold_diagnostics"],
            "n_features": result["n_features"],
            "max_reliable_event_fpr": result["max_reliable_event_fpr"],
            "test_rows": result["test_rows"],
            "test_episodes": result["test_episodes"],
            "audit": result["audit"],
        })

    metrics_df = pd.DataFrame(eval_metric_rows)
    events_df = pd.DataFrame(eval_event_rows)
    rates_df = pd.DataFrame(eval_rate_rows)
    families_df = pd.DataFrame(eval_family_rows)

    metrics_df.to_csv(outdir / "evaluation_multiseed.csv", index=False)
    events_df.to_csv(outdir / "evaluation_normal_event_fpr.csv", index=False)
    rates_df.to_csv(outdir / "evaluation_recall_by_hidden_rate.csv", index=False)
    families_df.to_csv(outdir / "evaluation_recall_by_mask_family.csv", index=False)

    report = {
        "status": "completed",
        "architecture": {
            "attack_channel": "INV Random Forest only",
            "own_channel": "not used in attack decision; retained for drift monitoring",
            "hard_negative_training": (
                "benign operational-event windows receive extra sample weight"
            ),
            "feature_family_selection": (
                "raw/temporal/zone invariant families selected on development seeds"
            ),
            "threshold_selection": (
                "global or max-over-validation-block quantile, development-selected"
            ),
            "test_used_for_architecture_selection": False,
        },
        "development": {
            "seeds": [int(x) for x in args.development_seeds],
            "days": float(args.development_days),
            "candidate_count": int(len(development_rows)),
            "selected": selected,
            "top_candidates": development_rows[:10],
        },
        "evaluation": {
            "seeds": [int(x) for x in args.evaluation_seeds],
            "days": float(args.evaluation_days),
            "aggregate_metrics": aggregate_metrics(metrics_df),
            "normal_event_fpr": aggregate_event_rows(events_df),
            "seed_reports": eval_seed_reports,
        },
        "limitations": [
            "Attack labels and benign-event labels are simulation-generated.",
            "Development seeds are used for architecture selection; evaluation seeds are disjoint.",
            "Thresholds remain locally calibrated per topology/realization and are not universal constants.",
            "Real SS7 attack recall remains unmeasured.",
        ],
    }
    (outdir / "ss7_hard_negative_inv_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "completed",
        "selected": selected,
        "evaluation_aggregate": report["evaluation"]["aggregate_metrics"],
        "evaluation_normal_events": report["evaluation"]["normal_event_fpr"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
