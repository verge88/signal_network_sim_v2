"""Context-aware validation for the calibrated SS7 detector.

Architecture:
1. INV Random Forest is the only attack-scoring channel.
2. OWN features train a benign-context classifier (background / known normal event).
3. A global INV threshold is calibrated on validation negatives at target FPR.
4. For confidently predicted benign contexts, a context-specific threshold is
   estimated from validation negatives and can only RAISE the global threshold.
5. Test labels are never used to select thresholds or context gating parameters.

This directly tests whether known benign traffic modes can suppress false alarms
without the recall/FPR penalty of unconditional OWN OR INV fusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ss7.fix import (
    FEATURE_SETS,
    INTERVAL_S,
    NORMAL_EVENTS,
    RngBook,
    SimConfig,
    blocked_split,
    build_dataset,
    episode_metrics,
)
from ss7.validate_calibrated_detectors import (
    INV_CAL_RF,
    all_derived,
    detector_metrics,
    episode_recall_by_group,
    fpr_threshold,
    prep,
    tx,
)


TARGET_FPR = 0.01
CONTEXT_CONFIDENCE = 0.70
MIN_CONTEXT_VALIDATION = 40
CONTEXT_SHRINKAGE_PRIOR = 100.0

CONTEXT_RF = {
    "n_estimators": 500,
    "min_samples_leaf": 8,
    "max_features": "sqrt",
}


def annotate_context(frame: pd.DataFrame, episodes: Sequence) -> pd.DataFrame:
    """Attach benign-event truth for training/evaluation only.

    If several benign events overlap on the same node/time, label as multi_event.
    The deployed test decision never reads this column; it uses predicted context.
    """
    out = frame.copy()
    labels = np.full(len(out), "background", dtype=object)
    event_ids = np.full(len(out), -1, dtype=int)
    hits = np.zeros(len(out), dtype=int)
    nid = out["nid"].to_numpy(int)
    t = out["t"].to_numpy(int)

    for event_id, e in enumerate(episodes):
        if e.kind not in NORMAL_EVENTS or e.nid is None:
            continue
        mask = (nid == int(e.nid)) & (t >= int(e.t0)) & (t <= int(e.t1))
        first = mask & (hits == 0)
        labels[first] = str(e.kind)
        event_ids[first] = int(event_id)
        overlap = mask & (hits > 0)
        labels[overlap] = "multi_event"
        event_ids[overlap] = -2
        hits[mask] += 1

    out["benign_context_truth"] = labels
    out["benign_event_id"] = event_ids
    out["benign_context_overlap_count"] = hits
    return out


def purge_cross_split_benign_events(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    te: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Ensure a benign event instance cannot occur in more than one partition.

    Rows belonging to overlapping normal events (benign_event_id=-2) are removed
    as ambiguous context examples.  Any normal event represented in multiple
    partitions is removed from all of them. Background rows are retained.
    """
    parts = {"train": tr.copy(), "validation": va.copy(), "test": te.copy()}
    membership: Dict[int, set] = {}
    for name, part in parts.items():
        for eid in part.loc[part["benign_event_id"] >= 0, "benign_event_id"].unique():
            membership.setdefault(int(eid), set()).add(name)
    crossing = {eid for eid, names in membership.items() if len(names) > 1}

    removed = {}
    clean = {}
    for name, part in parts.items():
        ambiguous = part["benign_event_id"] == -2
        crossed = part["benign_event_id"].isin(crossing)
        drop = ambiguous | crossed
        removed[name] = {
            "rows": int(drop.sum()),
            "ambiguous_overlap_rows": int(ambiguous.sum()),
            "cross_split_event_rows": int(crossed.sum()),
        }
        clean[name] = part.loc[~drop].copy()

    return (
        clean["train"],
        clean["validation"],
        clean["test"],
        {
            "crossing_event_ids": sorted(int(x) for x in crossing),
            "n_crossing_events": int(len(crossing)),
            "removed": removed,
        },
    )


def add_attack_parameters(frame: pd.DataFrame, episodes: Sequence) -> pd.DataFrame:
    out = frame.copy()
    rate = {
        i: float(e.hidden_rate)
        for i, e in enumerate(episodes)
        if e.kind == "compromise"
    }
    fam = {
        i: str(e.family)
        for i, e in enumerate(episodes)
        if e.kind == "compromise"
    }
    out["episode_hidden_rate"] = out["episode_id"].map(rate)
    out["episode_family_truth"] = out["episode_id"].map(fam)
    return out


def fit_inv_attack_model(
    tr: pd.DataFrame,
    inv_cols: Sequence[str],
    seed: int,
) -> Tuple[SimpleImputer, StandardScaler, RandomForestClassifier]:
    imp, sc, xtr = prep(tr, inv_cols)
    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
        **INV_CAL_RF,
    ).fit(xtr, tr["label"].to_numpy(int))
    return imp, sc, model


def fit_context_model(
    tr: pd.DataFrame,
    own_cols: Sequence[str],
    seed: int,
) -> Tuple[SimpleImputer, StandardScaler, RandomForestClassifier]:
    benign = tr[tr["label"] == 0].copy()
    imp, sc, xtr = prep(benign, own_cols)
    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=seed + 10000,
        n_jobs=-1,
        **CONTEXT_RF,
    ).fit(xtr, benign["benign_context_truth"].astype(str))
    return imp, sc, model


def context_predictions(
    frame: pd.DataFrame,
    own_cols: Sequence[str],
    imp: SimpleImputer,
    sc: StandardScaler,
    model: RandomForestClassifier,
) -> Tuple[np.ndarray, np.ndarray]:
    x = tx(frame, own_cols, imp, sc)
    proba = model.predict_proba(x)
    idx = np.argmax(proba, axis=1)
    labels = model.classes_[idx].astype(str)
    conf = proba[np.arange(len(frame)), idx]
    return labels, conf


def calibrate_context_thresholds(
    va: pd.DataFrame,
    inv_scores: np.ndarray,
    predicted_context: np.ndarray,
    confidence: np.ndarray,
    target_fpr: float,
) -> Tuple[float, Dict[str, float], Dict[str, dict]]:
    """Estimate suppression-only context thresholds from validation negatives.

    The global threshold remains the q(1-FPR) operating point.  Context samples
    are too small for reliable per-context q99 estimates, so we estimate only a
    location shift in the *median* INV score and shrink it toward zero:
        lift = max(0, median_context - median_reference) * n/(n + prior).
    A context can therefore only raise the global threshold, never lower it.
    """
    neg = va["label"].to_numpy(int) == 0
    global_thr = fpr_threshold(inv_scores[neg], target_fpr)
    thresholds: Dict[str, float] = {}
    diagnostics: Dict[str, dict] = {}

    ref_mask = (
        neg
        & (predicted_context == "background")
        & (confidence >= CONTEXT_CONFIDENCE)
    )
    if int(ref_mask.sum()) < MIN_CONTEXT_VALIDATION:
        ref_mask = neg
    reference_median = float(np.median(inv_scores[ref_mask]))

    for context in sorted(set(predicted_context[neg])):
        eligible = (
            neg
            & (predicted_context == context)
            & (confidence >= CONTEXT_CONFIDENCE)
        )
        n = int(eligible.sum())
        if context == "background" or n < MIN_CONTEXT_VALIDATION:
            thresholds[context] = float(global_thr)
            diagnostics[context] = {
                "validation_rows": n,
                "specific_threshold_used": False,
                "threshold": float(global_thr),
                "reference_median": reference_median,
            }
            continue

        context_median = float(np.median(inv_scores[eligible]))
        raw_shift = max(0.0, context_median - reference_median)
        shrinkage = float(n / (n + CONTEXT_SHRINKAGE_PRIOR))
        lift = raw_shift * shrinkage
        threshold = float(global_thr + lift)
        thresholds[context] = threshold
        diagnostics[context] = {
            "validation_rows": n,
            "specific_threshold_used": bool(lift > 0.0),
            "reference_median": reference_median,
            "context_median": context_median,
            "raw_median_shift": raw_shift,
            "shrinkage_factor": shrinkage,
            "threshold": threshold,
            "threshold_lift": lift,
        }

    return float(global_thr), thresholds, diagnostics


def per_row_threshold(
    predicted_context: np.ndarray,
    confidence: np.ndarray,
    global_thr: float,
    context_thresholds: Dict[str, float],
) -> np.ndarray:
    threshold = np.full(len(predicted_context), float(global_thr), dtype=float)
    for context, value in context_thresholds.items():
        if context == "background":
            continue
        use = (
            (predicted_context == context)
            & (confidence >= CONTEXT_CONFIDENCE)
        )
        threshold[use] = max(float(global_thr), float(value))
    return threshold


def variable_threshold_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """Operational metrics when each row has a context-conditioned threshold."""
    margin = np.asarray(scores, dtype=float) - np.asarray(thresholds, dtype=float)
    # fire iff margin > 0. AUC/AP of the margin quantify ranking after context penalty.
    m = episode_metrics(frame, margin, 0.0, k_consec=3)
    m["median_ttd_seconds"] = (
        float(m["median_ttd_intervals"] * INTERVAL_S)
        if np.isfinite(m["median_ttd_intervals"]) else float("nan")
    )
    y = frame["label"].to_numpy(int)
    if len(np.unique(y)) > 1:
        m["auc"] = float(roc_auc_score(y, margin))
        m["ap"] = float(average_precision_score(y, margin))
    else:
        m["auc"] = float("nan")
        m["ap"] = float("nan")
    neg = y == 0
    pos = y == 1
    fire = margin > 0.0
    m["window_fpr_direct"] = float(np.mean(fire[neg])) if neg.any() else float("nan")
    m["window_tpr"] = float(np.mean(fire[pos])) if pos.any() else float("nan")
    m["threshold_mean"] = float(np.mean(thresholds))
    m["threshold_max"] = float(np.max(thresholds))
    return m


def benign_context_fpr(
    frame: pd.DataFrame,
    fire: np.ndarray,
    predicted_context: np.ndarray,
    confidence: np.ndarray,
) -> List[dict]:
    rows = []
    neg = frame["label"].to_numpy(int) == 0
    truth = frame["benign_context_truth"].astype(str).to_numpy()
    for context in sorted(set(truth[neg])):
        mask = neg & (truth == context)
        rows.append({
            "true_context": context,
            "rows": int(mask.sum()),
            "alerts": int(np.sum(fire[mask])),
            "window_fpr": float(np.mean(fire[mask])) if mask.any() else float("nan"),
            "predicted_as_same_fraction": float(
                np.mean(predicted_context[mask] == context)
            ) if mask.any() else float("nan"),
            "mean_prediction_confidence": float(
                np.mean(confidence[mask])
            ) if mask.any() else float("nan"),
        })
    return rows


def attack_group_rows(
    frame: pd.DataFrame,
    margins: np.ndarray,
    group_col: str,
) -> List[dict]:
    return episode_recall_by_group(frame, margins, 0.0, group_col)


def bootstrap_ci(values: Sequence[float], seed: int = 321, n_boot: int = 3000) -> List[float]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return [float("nan"), float("nan")]
    if len(x) == 1:
        return [float(x[0]), float(x[0])]
    rng = np.random.default_rng(seed)
    bs = np.array([
        rng.choice(x, len(x), replace=True).mean()
        for _ in range(n_boot)
    ])
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def evaluate_seed(seed: int, days: float) -> Tuple[dict, List[dict], List[dict], List[dict], dict]:
    cfg = SimConfig(seed=int(seed), days=float(days))
    frame, _, episodes = build_dataset(
        cfg,
        episodes_kw={
            "n_compromise": 18,
            "n_normal": 42,
            "n_ext": 6,
        },
    )
    frame = annotate_context(frame, episodes)
    frame = add_attack_parameters(frame, episodes)

    tr, va, te = blocked_split(frame, cfg, RngBook(seed)["split"])
    tr, va, te, benign_split_audit = purge_cross_split_benign_events(tr, va, te)
    own_cols = all_derived(frame, FEATURE_SETS["OWN_only"])
    inv_cols = all_derived(frame, FEATURE_SETS["INV_only"])

    imp_i, sc_i, inv = fit_inv_attack_model(tr, inv_cols, seed)
    inv_va = inv.predict_proba(tx(va, inv_cols, imp_i, sc_i))[:, 1]
    inv_te = inv.predict_proba(tx(te, inv_cols, imp_i, sc_i))[:, 1]

    imp_c, sc_c, context = fit_context_model(tr, own_cols, seed)
    pred_va, conf_va = context_predictions(va, own_cols, imp_c, sc_c, context)
    pred_te, conf_te = context_predictions(te, own_cols, imp_c, sc_c, context)

    global_thr, ctx_thr, ctx_diag = calibrate_context_thresholds(
        va, inv_va, pred_va, conf_va, TARGET_FPR
    )
    row_thr = per_row_threshold(
        pred_te, conf_te, global_thr, ctx_thr
    )

    global_metrics = detector_metrics(te, inv_te, global_thr)
    context_metrics = variable_threshold_metrics(te, inv_te, row_thr)

    global_fire = inv_te > global_thr
    context_fire = inv_te > row_thr
    global_context = benign_context_fpr(
        te, global_fire, pred_te, conf_te
    )
    context_context = benign_context_fpr(
        te, context_fire, pred_te, conf_te
    )

    margin = inv_te - row_thr
    rate_rows = attack_group_rows(te, margin, "episode_hidden_rate")
    family_rows = attack_group_rows(te, margin, "episode_family_truth")

    benign_te = te["label"].to_numpy(int) == 0
    true_context = te["benign_context_truth"].astype(str).to_numpy()
    context_accuracy = float(
        accuracy_score(true_context[benign_te], pred_te[benign_te])
    ) if benign_te.any() else float("nan")

    summary = {
        "seed": int(seed),
        "rows": int(len(frame)),
        "train_rows": int(len(tr)),
        "validation_rows": int(len(va)),
        "test_rows": int(len(te)),
        "test_episodes": int(
            te.loc[te["label"] == 1, "episode_id"].nunique()
        ),
        "own_context_features": int(len(own_cols)),
        "inv_features": int(len(inv_cols)),
        "global_inv": global_metrics,
        "context_aware_inv": context_metrics,
        "context_classifier_accuracy_benign_test": context_accuracy,
        "global_threshold": global_thr,
        "context_thresholds": ctx_thr,
        "context_threshold_diagnostics": ctx_diag,
        "benign_event_split_audit": benign_split_audit,
    }
    comparison_rows = [
        {
            "seed": int(seed),
            "detector": "inv_global",
            **global_metrics,
        },
        {
            "seed": int(seed),
            "detector": "inv_context_aware",
            **context_metrics,
        },
    ]

    event_rows = []
    for architecture, rows in [
        ("inv_global", global_context),
        ("inv_context_aware", context_context),
    ]:
        for row in rows:
            event_rows.append({
                "seed": int(seed),
                "detector": architecture,
                **row,
            })

    for r in rate_rows:
        r["seed"] = int(seed)
    for r in family_rows:
        r["seed"] = int(seed)

    return summary, comparison_rows, event_rows, rate_rows, {
        "family_rows": family_rows,
        "validation_predicted_context_counts": pd.Series(pred_va).value_counts().to_dict(),
        "test_predicted_context_counts": pd.Series(pred_te).value_counts().to_dict(),
    }


def aggregate_comparison(df: pd.DataFrame) -> List[dict]:
    rows = []
    for detector, g in df.groupby("detector"):
        item = {"detector": detector, "n_seeds": int(g["seed"].nunique())}
        for col in [
            "episode_recall", "window_fpr_direct", "auc", "ap",
            "median_ttd_seconds", "alerts_per_day_network", "window_tpr",
        ]:
            vals = g[col].to_numpy(float)
            finite = vals[np.isfinite(vals)]
            item[f"{col}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            item[f"{col}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            item[f"{col}_min"] = float(np.min(finite)) if len(finite) else float("nan")
            item[f"{col}_max"] = float(np.max(finite)) if len(finite) else float("nan")
            item[f"{col}_ci95"] = bootstrap_ci(finite)
        rows.append(item)
    return rows


def aggregate_events(df: pd.DataFrame) -> List[dict]:
    rows = []
    for (detector, context), g in df.groupby(["detector", "true_context"]):
        total_rows = int(g["rows"].sum())
        alerts = int(g["alerts"].sum())
        rows.append({
            "detector": detector,
            "true_context": context,
            "rows": total_rows,
            "alerts": alerts,
            "pooled_window_fpr": float(alerts / total_rows) if total_rows else float("nan"),
            "mean_seed_fpr": float(g["window_fpr"].mean()),
            "same_context_prediction_fraction_mean": float(
                g["predicted_as_same_fraction"].mean()
            ),
        })
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds", nargs="+", type=int,
        default=[6001, 6002, 6003, 6004, 6005],
    )
    ap.add_argument("--days", type=float, default=1.25)
    ap.add_argument(
        "--output-dir",
        default="results/ss7_context_aware_validation",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    seed_reports = []
    comparison = []
    event_rows = []
    rate_rows = []
    family_rows = []
    diagnostics = []

    for seed in args.seeds:
        summary, comp, events, rates, extra = evaluate_seed(
            int(seed), float(args.days)
        )
        seed_reports.append(summary)
        comparison.extend(comp)
        event_rows.extend(events)
        rate_rows.extend(rates)
        family_rows.extend(extra["family_rows"])
        diagnostics.append({
            "seed": int(seed),
            "validation_predicted_context_counts": extra[
                "validation_predicted_context_counts"
            ],
            "test_predicted_context_counts": extra[
                "test_predicted_context_counts"
            ],
        })

    comparison_df = pd.DataFrame(comparison)
    events_df = pd.DataFrame(event_rows)
    rates_df = pd.DataFrame(rate_rows)
    family_df = pd.DataFrame(family_rows)

    comparison_df.to_csv(
        outdir / "context_aware_detector_multiseed.csv", index=False
    )
    events_df.to_csv(
        outdir / "context_aware_normal_event_fpr.csv", index=False
    )
    rates_df.to_csv(
        outdir / "context_aware_recall_by_hidden_rate.csv", index=False
    )
    family_df.to_csv(
        outdir / "context_aware_recall_by_mask_family.csv", index=False
    )

    comparison_agg = aggregate_comparison(comparison_df)
    event_agg = aggregate_events(events_df)

    report = {
        "status": "completed",
        "architecture": {
            "attack_channel": "INV Random Forest only",
            "context_channel": (
                "OWN Random Forest predicts benign context; it never raises "
                "attack score"
            ),
            "global_target_fpr": TARGET_FPR,
            "context_confidence": CONTEXT_CONFIDENCE,
            "min_context_validation_rows": MIN_CONTEXT_VALIDATION,
            "context_shrinkage_prior_rows": CONTEXT_SHRINKAGE_PRIOR,
            "context_threshold_rule": (
                "context threshold = global q99 + positive shrunk median "
                "INV-score shift; low-confidence/unknown -> global threshold"
            ),
            "test_used_for_tuning": False,
        },
        "seeds": [int(s) for s in args.seeds],
        "days_per_seed": float(args.days),
        "seed_reports": seed_reports,
        "multiseed_comparison": comparison_agg,
        "normal_event_fpr": event_agg,
        "diagnostics": diagnostics,
        "limitations": [
            "Context labels used to train the context classifier are simulation-generated normal-event labels.",
            "The context-aware architecture must be externally validated before claiming real SS7 attack performance.",
            "Context thresholds are locally calibrated on validation negatives and are not universal numeric constants.",
        ],
    }

    (outdir / "ss7_context_aware_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "completed",
        "multiseed_comparison": comparison_agg,
        "normal_event_fpr": event_agg,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
