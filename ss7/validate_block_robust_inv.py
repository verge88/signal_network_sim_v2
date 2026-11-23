"""Block-robust threshold validation for the calibrated SS7 INV detector.

The attack model is unchanged: Random Forest over derived INV features.
Only the operational threshold calibration is compared:

1. global_q99: q(1-target_fpr) over all validation negatives.
2. block_robust_q99: maximum q(1-target_fpr) across contiguous validation
   time blocks with enough negative windows, never lower than global_q99.

No test score or test label participates in threshold selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ss7.fix import (
    FEATURE_SETS,
    RngBook,
    SimConfig,
    blocked_split,
    build_dataset,
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
from ss7.validate_context_aware_detector import (
    add_attack_parameters,
    annotate_context,
    audit_reroute_controls,
    benign_context_fpr,
    purge_cross_split_benign_events,
)


TARGET_FPR = 0.01
MIN_BLOCK_NEGATIVES = 500


def contiguous_time_blocks(frame: pd.DataFrame) -> List[np.ndarray]:
    """Return contiguous groups of unique t values.

    blocked_split creates disjoint temporal blocks separated by purge gaps.
    A new block starts whenever adjacent observed t values differ by >1.
    """
    times = np.array(sorted(frame["t"].unique()), dtype=int)
    if len(times) == 0:
        return []
    cuts = np.where(np.diff(times) > 1)[0] + 1
    return [x for x in np.split(times, cuts) if len(x)]


def robust_threshold(
    va: pd.DataFrame,
    scores: np.ndarray,
    target_fpr: float,
) -> Tuple[float, dict]:
    y = va["label"].to_numpy(int)
    neg = y == 0
    global_thr = fpr_threshold(scores[neg], target_fpr)

    block_rows = []
    candidates = [float(global_thr)]
    for idx, times in enumerate(contiguous_time_blocks(va), start=1):
        mask = neg & va["t"].isin(times).to_numpy()
        n = int(mask.sum())
        if n < MIN_BLOCK_NEGATIVES:
            block_rows.append({
                "block": idx,
                "t_min": int(times.min()),
                "t_max": int(times.max()),
                "negative_rows": n,
                "eligible": False,
                "q99": None,
                "fpr_at_selected": None,
            })
            continue
        q = fpr_threshold(scores[mask], target_fpr)
        candidates.append(float(q))
        block_rows.append({
            "block": idx,
            "t_min": int(times.min()),
            "t_max": int(times.max()),
            "negative_rows": n,
            "eligible": True,
            "q99": float(q),
            "fpr_at_selected": None,
        })

    selected = float(max(candidates))
    for row in block_rows:
        if row["eligible"]:
            times = np.arange(row["t_min"], row["t_max"] + 1, dtype=int)
            mask = neg & va["t"].isin(times).to_numpy()
            row["fpr_at_selected"] = float(np.mean(scores[mask] > selected))

    return selected, {
        "global_q99": float(global_thr),
        "selected_threshold": selected,
        "threshold_lift": float(selected - global_thr),
        "eligible_blocks": int(sum(r["eligible"] for r in block_rows)),
        "blocks": block_rows,
    }


def fit_inv(
    tr: pd.DataFrame,
    cols: Sequence[str],
    seed: int,
):
    imp, sc, xtr = prep(tr, cols)
    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
        **INV_CAL_RF,
    ).fit(xtr, tr["label"].to_numpy(int))
    return imp, sc, model


def bootstrap_ci(values, seed=881, n_boot=3000):
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
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def evaluate_seed(seed: int, days: float):
    cfg = SimConfig(seed=int(seed), days=float(days))
    frame, sim, episodes = build_dataset(
        cfg,
        episodes_kw={
            "n_compromise": 18,
            "n_normal": 42,
            "n_ext": 6,
        },
    )
    reroute_audit = audit_reroute_controls(sim, episodes)
    frame = annotate_context(frame, episodes)
    frame = add_attack_parameters(frame, episodes)

    tr, va, te = blocked_split(frame, cfg, RngBook(seed)["split"])
    tr, va, te, benign_audit = purge_cross_split_benign_events(tr, va, te)

    inv_cols = all_derived(frame, FEATURE_SETS["INV_only"])
    imp, sc, model = fit_inv(tr, inv_cols, seed)
    va_scores = model.predict_proba(tx(va, inv_cols, imp, sc))[:, 1]
    te_scores = model.predict_proba(tx(te, inv_cols, imp, sc))[:, 1]

    yva = va["label"].to_numpy(int)
    global_thr = fpr_threshold(va_scores[yva == 0], TARGET_FPR)
    block_thr, block_diag = robust_threshold(va, va_scores, TARGET_FPR)

    rows = []
    event_rows = []
    rate_rows = []
    family_rows = []
    for name, threshold in [
        ("inv_global_q99", global_thr),
        ("inv_block_robust_q99", block_thr),
    ]:
        m = detector_metrics(te, te_scores, threshold)
        rows.append({
            "seed": int(seed),
            "detector": name,
            **m,
        })
        fire = te_scores > threshold
        for e in benign_context_fpr(
            te,
            fire,
            np.full(len(te), "not_used", dtype=object),
            np.zeros(len(te), dtype=float),
        ):
            event_rows.append({
                "seed": int(seed),
                "detector": name,
                **e,
            })
        margin = te_scores - threshold
        for r in episode_recall_by_group(
            te, margin, 0.0, "episode_hidden_rate"
        ):
            rate_rows.append({"seed": int(seed), "detector": name, **r})
        for r in episode_recall_by_group(
            te, margin, 0.0, "episode_family_truth"
        ):
            family_rows.append({"seed": int(seed), "detector": name, **r})

    meta = {
        "seed": int(seed),
        "rows": int(len(frame)),
        "train_rows": int(len(tr)),
        "validation_rows": int(len(va)),
        "test_rows": int(len(te)),
        "test_episodes": int(
            te.loc[te["label"] == 1, "episode_id"].nunique()
        ),
        "inv_features": int(len(inv_cols)),
        "global_threshold": float(global_thr),
        "block_robust": block_diag,
        "reroute_control_audit": reroute_audit,
        "benign_event_split_audit": benign_audit,
    }
    return meta, rows, event_rows, rate_rows, family_rows


def aggregate(df: pd.DataFrame):
    out = []
    for detector, g in df.groupby("detector"):
        row = {"detector": detector, "n_seeds": int(g["seed"].nunique())}
        for col in [
            "episode_recall", "window_fpr_direct", "auc", "ap",
            "median_ttd_seconds", "alerts_per_day_network", "window_tpr",
        ]:
            x = g[col].to_numpy(float)
            x = x[np.isfinite(x)]
            row[f"{col}_mean"] = float(np.mean(x)) if len(x) else float("nan")
            row[f"{col}_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
            row[f"{col}_min"] = float(np.min(x)) if len(x) else float("nan")
            row[f"{col}_max"] = float(np.max(x)) if len(x) else float("nan")
            row[f"{col}_ci95"] = bootstrap_ci(x)
        out.append(row)
    return out


def aggregate_events(df: pd.DataFrame):
    rows = []
    for (detector, context), g in df.groupby(["detector", "true_context"]):
        n = int(g["rows"].sum())
        a = int(g["alerts"].sum())
        rows.append({
            "detector": detector,
            "true_context": context,
            "rows": n,
            "alerts": a,
            "pooled_window_fpr": float(a / n) if n else float("nan"),
            "mean_seed_fpr": float(g["window_fpr"].mean()),
        })
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds", nargs="+", type=int,
        default=[6201, 6202, 6203, 6204, 6205],
    )
    ap.add_argument("--days", type=float, default=1.25)
    ap.add_argument(
        "--output-dir",
        default="results/ss7_block_robust_validation",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    metas, metrics, events, rates, families = [], [], [], [], []
    for seed in args.seeds:
        meta, m, e, r, f = evaluate_seed(int(seed), float(args.days))
        metas.append(meta)
        metrics.extend(m)
        events.extend(e)
        rates.extend(r)
        families.extend(f)

    metrics_df = pd.DataFrame(metrics)
    events_df = pd.DataFrame(events)
    rates_df = pd.DataFrame(rates)
    families_df = pd.DataFrame(families)

    metrics_df.to_csv(outdir / "block_robust_multiseed.csv", index=False)
    events_df.to_csv(outdir / "block_robust_normal_event_fpr.csv", index=False)
    rates_df.to_csv(outdir / "block_robust_recall_by_hidden_rate.csv", index=False)
    families_df.to_csv(outdir / "block_robust_recall_by_mask_family.csv", index=False)

    report = {
        "status": "completed",
        "protocol": {
            "attack_model": "calibrated INV Random Forest",
            "target_fpr": TARGET_FPR,
            "global_threshold": "validation-negative q99",
            "block_robust_threshold": (
                "max validation-negative q99 over contiguous validation blocks; "
                "never below global q99"
            ),
            "min_block_negatives": MIN_BLOCK_NEGATIVES,
            "test_used_for_tuning": False,
        },
        "seeds": [int(s) for s in args.seeds],
        "days_per_seed": float(args.days),
        "seed_meta": metas,
        "multiseed_comparison": aggregate(metrics_df),
        "normal_event_fpr": aggregate_events(events_df),
        "limitations": [
            "Attack labels are simulation-generated.",
            "The threshold is calibrated per topology/realization and is not universal.",
            "Taking the maximum validation-block q99 is deliberately conservative and may reduce recall.",
        ],
    }
    (outdir / "ss7_block_robust_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "completed",
        "multiseed_comparison": report["multiseed_comparison"],
        "normal_event_fpr": report["normal_event_fpr"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
