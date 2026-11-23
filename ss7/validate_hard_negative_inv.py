"""Hard-negative and feature-robust validation for the SS7 INV detector.

Architecture selection is performed only on development seeds.  Evaluation seeds
are disjoint and their test blocks are touched once after the architecture is frozen.

The experiment addresses two failures found by the earlier full validation:
(1) temporal/zone derived INV features may overreact to benign operational changes;
(2) benign events such as traffic spikes/campaigns must be treated as explicit hard
negative controls rather than runtime context labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ss7.fix import INV, RngBook, SimConfig, blocked_split, build_dataset
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


DEVELOPMENT_FPR_TOLERANCE = 0.015
DEVELOPMENT_EVENT_FPR_TOLERANCE = 0.05
MIN_BLOCK_NEGATIVES = 400

HARD_NEGATIVE_WEIGHTS = (1.0, 4.0, 8.0)
CALIBRATION_TARGETS = (0.005, 0.01)
THRESHOLD_MODES = ("global", "block_robust")
FEATURE_FAMILIES = (
    "inv_raw",
    "inv_raw_history",
    "inv_raw_history_zone",
    "inv_all_derived",
)


def invariant_feature_families(frame: pd.DataFrame) -> Dict[str, List[str]]:
    raw = [c for c in INV if c in frame.columns]
    mean_std, zone = [], []
    for c in raw:
        for suffix in ("__mean", "__std"):
            name = c + suffix
            if name in frame.columns:
                mean_std.append(name)
        zone_name = c + "__dev_zone"
        if zone_name in frame.columns:
            zone.append(zone_name)
    return {
        "inv_raw": raw,
        "inv_raw_history": raw + mean_std,
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
            "mode": mode,
            "global_threshold": float(global_thr),
            "selected_threshold": float(global_thr),
            "threshold_lift": 0.0,
            "eligible_blocks": 0,
        }

    candidates = [float(global_thr)]
    blocks = []
    for idx, times in enumerate(contiguous_time_blocks(validation), start=1):
        mask = neg & validation["t"].isin(times).to_numpy()
        n = int(mask.sum())
        if n < MIN_BLOCK_NEGATIVES:
            blocks.append({
                "block": idx, "negative_rows": n,
                "eligible": False, "quantile_threshold": None,
            })
            continue
        q = fpr_threshold(scores[mask], target_fpr)
        candidates.append(float(q))
        blocks.append({
            "block": idx, "negative_rows": n,
            "eligible": True, "quantile_threshold": float(q),
        })
    selected = float(max(candidates))
    return selected, {
        "mode": mode,
        "global_threshold": float(global_thr),
        "selected_threshold": selected,
        "threshold_lift": float(selected - global_thr),
        "eligible_blocks": int(sum(x["eligible"] for x in blocks)),
        "blocks": blocks,
    }


def training_sample_weights(frame: pd.DataFrame, hard_weight: float) -> np.ndarray:
    y = frame["label"].to_numpy(int)
    context = frame["benign_event_kind_truth"].astype(str).to_numpy()
    w = np.ones(len(frame), dtype=float)
    w[(y == 0) & (context != "background")] = float(hard_weight)
    return w


def fit_model(
    train: pd.DataFrame,
    cols: Sequence[str],
    seed: int,
    hard_weight: float,
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
        sample_weight=training_sample_weights(train, hard_weight),
    )
    return imp, sc, model


def build_prepared_world(seed: int, days: float, n_compromise: int, n_normal: int) -> dict:
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
    return {
        "seed": int(seed),
        "frame": frame,
        "train": tr,
        "validation": va,
        "test": te,
        "feature_families": invariant_feature_families(frame),
        "audit": {
            "reroute": reroute,
            "benign_split": split_audit,
        },
    }


def event_fpr_rows(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> List[dict]:
    y = frame["label"].to_numpy(int)
    event = frame["benign_event_kind_truth"].astype(str).to_numpy()
    fire = np.asarray(scores, dtype=float) > float(threshold)
    out = []
    for kind in sorted(set(event[y == 0])):
        mask = (y == 0) & (event == kind)
        out.append({
            "normal_event": str(kind),
            "rows": int(mask.sum()),
            "alerts": int(np.sum(fire[mask])),
            "window_fpr": float(np.mean(fire[mask])) if mask.any() else float("nan"),
        })
    return out


def max_event_fpr(rows: Sequence[dict], min_rows: int = 20) -> float:
    vals = [
        float(r["window_fpr"])
        for r in rows
        if int(r["rows"]) >= min_rows
        and r["normal_event"] != "background"
        and np.isfinite(float(r["window_fpr"]))
    ]
    return float(max(vals)) if vals else 0.0


def model_config_key(feature_family: str, hard_weight: float) -> str:
    return f"{feature_family}/w{hard_weight:g}"


def candidate_key(
    feature_family: str,
    hard_weight: float,
    target_fpr: float,
    threshold_mode: str,
) -> str:
    return (
        f"{model_config_key(feature_family, hard_weight)}"
        f"/fpr{target_fpr:g}/{threshold_mode}"
    )


def score_model_config(world: dict, feature_family: str, hard_weight: float) -> dict:
    cols = world["feature_families"][feature_family]
    tr, va, te = world["train"], world["validation"], world["test"]
    imp, sc, model = fit_model(tr, cols, world["seed"], hard_weight)
    return {
        "cols": cols,
        "validation_scores": model.predict_proba(tx(va, cols, imp, sc))[:, 1],
        "test_scores": model.predict_proba(tx(te, cols, imp, sc))[:, 1],
    }


def evaluate_threshold_variant(
    world: dict,
    scored: dict,
    target_fpr: float,
    threshold_mode: str,
) -> dict:
    va, te = world["validation"], world["test"]
    threshold, diag = calibrate_threshold(
        va, scored["validation_scores"], target_fpr, threshold_mode
    )
    te_scores = scored["test_scores"]
    metrics = detector_metrics(te, te_scores, threshold)
    events = event_fpr_rows(te, te_scores, threshold)
    return {
        "seed": int(world["seed"]),
        "threshold": float(threshold),
        "threshold_diagnostics": diag,
        "metrics": metrics,
        "event_fpr": events,
        "max_reliable_event_fpr": max_event_fpr(events),
        "test_rows": int(len(te)),
        "test_episodes": int(te.loc[te["label"] == 1, "episode_id"].nunique()),
        "rate_rows": episode_recall_by_group(
            te, te_scores - threshold, 0.0, "episode_hidden_rate"
        ),
        "family_rows": episode_recall_by_group(
            te, te_scores - threshold, 0.0, "episode_family_truth"
        ),
    }


def summarize_candidate(candidate: dict, seed_results: Sequence[dict], n_features: int) -> dict:
    recall = np.asarray([x["metrics"]["episode_recall"] for x in seed_results], float)
    fpr = np.asarray([x["metrics"]["window_fpr_direct"] for x in seed_results], float)
    ap = np.asarray([x["metrics"]["ap"] for x in seed_results], float)
    event = np.asarray([x["max_reliable_event_fpr"] for x in seed_results], float)

    max_fpr = float(np.nanmax(fpr))
    max_event = float(np.nanmax(event))
    fpr_violation = max(0.0, max_fpr - DEVELOPMENT_FPR_TOLERANCE)
    event_violation = max(0.0, max_event - DEVELOPMENT_EVENT_FPR_TOLERANCE)
    feasible = bool(fpr_violation == 0.0 and event_violation == 0.0)

    mean_recall = float(np.nanmean(recall))
    min_recall = float(np.nanmin(recall))
    mean_ap = float(np.nanmean(ap))
    selection_key = (
        0 if feasible else 1,
        float(fpr_violation + event_violation),
        -min_recall,
        -mean_recall,
        -mean_ap,
        int(n_features),
    )
    return {
        "candidate": candidate_key(
            candidate["feature_family"],
            candidate["hard_negative_weight"],
            candidate["target_fpr"],
            candidate["threshold_mode"],
        ),
        **candidate,
        "n_features": int(n_features),
        "mean_episode_recall": mean_recall,
        "min_episode_recall": min_recall,
        "mean_test_fpr": float(np.nanmean(fpr)),
        "max_test_fpr": max_fpr,
        "worst_reliable_event_fpr": max_event,
        "mean_ap": mean_ap,
        "feasible": feasible,
        "selection_key": list(selection_key),
    }


def aggregate_metrics(df: pd.DataFrame) -> dict:
    out = {}
    for col in (
        "episode_recall", "window_fpr_direct", "auc", "ap",
        "median_ttd_seconds", "window_tpr", "alerts_per_day_network",
    ):
        x = df[col].to_numpy(float)
        x = x[np.isfinite(x)]
        out[col + "_mean"] = float(np.mean(x)) if len(x) else float("nan")
        out[col + "_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        out[col + "_min"] = float(np.min(x)) if len(x) else float("nan")
        out[col + "_max"] = float(np.max(x)) if len(x) else float("nan")
    return out


def aggregate_events(df: pd.DataFrame) -> List[dict]:
    out = []
    for event, g in df.groupby("normal_event"):
        n = int(g["rows"].sum())
        alerts = int(g["alerts"].sum())
        out.append({
            "normal_event": str(event),
            "rows": n,
            "alerts": alerts,
            "pooled_window_fpr": float(alerts / n) if n else float("nan"),
            "mean_seed_fpr": float(g["window_fpr"].mean()),
            "max_seed_fpr": float(g["window_fpr"].max()),
        })
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--development-seeds", nargs="+", type=int, default=[42, 1337, 5151])
    ap.add_argument("--evaluation-seeds", nargs="+", type=int, default=[7001, 7002, 7003, 7004, 7005])
    ap.add_argument("--development-days", type=float, default=0.9)
    ap.add_argument("--evaluation-days", type=float, default=1.25)
    ap.add_argument("--output-dir", default="results/ss7_hard_negative_inv")
    args = ap.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    development_worlds = [
        build_prepared_world(int(seed), float(args.development_days), 20, 60)
        for seed in args.development_seeds
    ]

    candidate_summaries = []
    candidate_detail = {}
    for feature_family in FEATURE_FAMILIES:
        for hard_weight in HARD_NEGATIVE_WEIGHTS:
            config_scores = {
                w["seed"]: score_model_config(w, feature_family, hard_weight)
                for w in development_worlds
            }
            n_features = len(next(iter(config_scores.values()))["cols"])
            for target_fpr in CALIBRATION_TARGETS:
                for threshold_mode in THRESHOLD_MODES:
                    candidate = {
                        "feature_family": feature_family,
                        "hard_negative_weight": float(hard_weight),
                        "target_fpr": float(target_fpr),
                        "threshold_mode": str(threshold_mode),
                    }
                    per_seed = [
                        evaluate_threshold_variant(
                            w,
                            config_scores[w["seed"]],
                            float(target_fpr),
                            str(threshold_mode),
                        )
                        for w in development_worlds
                    ]
                    summary = summarize_candidate(candidate, per_seed, n_features)
                    candidate_summaries.append(summary)
                    candidate_detail[summary["candidate"]] = per_seed

    candidate_summaries.sort(key=lambda x: tuple(x["selection_key"]))
    selected = candidate_summaries[0]
    selected_model = {
        "feature_family": selected["feature_family"],
        "hard_negative_weight": float(selected["hard_negative_weight"]),
    }
    selected_threshold = {
        "target_fpr": float(selected["target_fpr"]),
        "threshold_mode": selected["threshold_mode"],
    }

    pd.DataFrame(candidate_summaries).to_csv(
        outdir / "development_candidate_summary.csv", index=False
    )

    metric_rows, event_rows, rate_rows, family_rows, seed_reports = [], [], [], [], []
    for seed in args.evaluation_seeds:
        world = build_prepared_world(int(seed), float(args.evaluation_days), 18, 42)
        scored = score_model_config(
            world,
            selected_model["feature_family"],
            selected_model["hard_negative_weight"],
        )
        result = evaluate_threshold_variant(
            world,
            scored,
            selected_threshold["target_fpr"],
            selected_threshold["threshold_mode"],
        )
        metric_rows.append({
            "seed": int(seed),
            **result["metrics"],
            "threshold": result["threshold"],
            "test_rows": result["test_rows"],
            "test_episodes": result["test_episodes"],
        })
        event_rows.extend({"seed": int(seed), **x} for x in result["event_fpr"])
        rate_rows.extend({"seed": int(seed), **x} for x in result["rate_rows"])
        family_rows.extend({"seed": int(seed), **x} for x in result["family_rows"])
        seed_reports.append({
            "seed": int(seed),
            "n_features": int(len(scored["cols"])),
            "threshold": result["threshold"],
            "threshold_diagnostics": result["threshold_diagnostics"],
            "max_reliable_event_fpr": result["max_reliable_event_fpr"],
            "test_rows": result["test_rows"],
            "test_episodes": result["test_episodes"],
            "audit": world["audit"],
        })

    metrics_df = pd.DataFrame(metric_rows)
    events_df = pd.DataFrame(event_rows)
    rates_df = pd.DataFrame(rate_rows)
    families_df = pd.DataFrame(family_rows)
    metrics_df.to_csv(outdir / "evaluation_multiseed.csv", index=False)
    events_df.to_csv(outdir / "evaluation_normal_event_fpr.csv", index=False)
    rates_df.to_csv(outdir / "evaluation_recall_by_hidden_rate.csv", index=False)
    families_df.to_csv(outdir / "evaluation_recall_by_mask_family.csv", index=False)

    report = {
        "status": "completed",
        "architecture": {
            "attack_channel": "INV Random Forest only",
            "own_channel": "drift monitoring only; excluded from attack decision",
            "hard_negative_training": "normal operational-event windows are upweighted",
            "feature_family": selected["feature_family"],
            "hard_negative_weight": selected["hard_negative_weight"],
            "threshold_mode": selected["threshold_mode"],
            "threshold_target_fpr": selected["target_fpr"],
            "architecture_selected_on_evaluation_seeds": False,
        },
        "development": {
            "seeds": [int(x) for x in args.development_seeds],
            "days": float(args.development_days),
            "candidate_count": int(len(candidate_summaries)),
            "selected": selected,
            "top_candidates": candidate_summaries[:10],
        },
        "evaluation": {
            "seeds": [int(x) for x in args.evaluation_seeds],
            "days": float(args.evaluation_days),
            "aggregate_metrics": aggregate_metrics(metrics_df),
            "normal_event_fpr": aggregate_events(events_df),
            "seed_reports": seed_reports,
        },
        "limitations": [
            "Attack and benign-event labels are simulation-generated.",
            "Architecture selection uses development seeds only; evaluation seeds are disjoint.",
            "Thresholds are locally calibrated per realization and are not universal constants.",
            "Real SS7 attack recall remains unmeasured.",
        ],
    }
    (outdir / "ss7_hard_negative_inv_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
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
