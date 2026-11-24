"""Comprehensive validation of calibrated SS7 detectors on canonical simulated data.

The detector hyperparameters are frozen from the hybrid calibration stage.  Every
simulation seed regenerates topology and traffic.  Thresholds are calibrated only on
validation negatives of that realization; test windows and test episodes are never
used to select a threshold or hyperparameter.

Outputs:
- multiseed detector comparison at a common 1% false-positive budget;
- calibrated hybrid OWN/INV channel metrics;
- recall by hidden-rate and masking family;
- leave-one-masking-family-out generalisation;
- false alarms during benign operational events;
- h=0 counterfactual/null test for generator leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ss7.fix import (
    INV,
    MASK_FAMILIES,
    NORMAL_EVENTS,
    RngBook,
    SimConfig,
    Sophistication,
    blocked_split,
    build_dataset,
    episode_metrics,
    null_test,
    rule_baseline,
)


INTERVAL_S = 30.0
OWN_CAL_FEATURES = ["mtype_entropy", "map_share", "in_out_ratio"]
OWN_CAL_IF = {
    "n_estimators": 500,
    "max_samples": 1024,
    "max_features": 0.75,
}
INV_CAL_RF = {
    "n_estimators": 400,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
}
OVERALL_FPR = 0.01
CHANNEL_FPR = OVERALL_FPR / 2.0


def all_derived(frame: pd.DataFrame, bases: Sequence[str]) -> List[str]:
    return [
        c for c in frame.columns
        if any(c == b or c.startswith(b + "__") for b in bases)
    ]


def prep(
    train: pd.DataFrame, cols: Sequence[str]
) -> Tuple[SimpleImputer, StandardScaler, np.ndarray]:
    cols = list(cols)
    imp = SimpleImputer(strategy="median").fit(train[cols])
    scaler = StandardScaler().fit(imp.transform(train[cols]))
    return imp, scaler, scaler.transform(imp.transform(train[cols]))


def tx(
    frame: pd.DataFrame,
    cols: Sequence[str],
    imp: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    return scaler.transform(imp.transform(frame[list(cols)]))


def fpr_threshold(scores: np.ndarray, target: float) -> float:
    if len(scores) == 0:
        raise ValueError("No validation-negative scores available")
    return float(np.quantile(np.asarray(scores, dtype=float), 1.0 - target))


def robust_scale(scores: np.ndarray) -> float:
    x = np.asarray(scores, dtype=float)
    if len(x) == 0:
        return 1.0
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return float(max(q3 - q1, np.std(x), 1e-6))


def detector_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    y = frame["label"].to_numpy(int)
    m = episode_metrics(frame, scores, threshold, k_consec=3)
    m["median_ttd_seconds"] = (
        float(m["median_ttd_intervals"] * INTERVAL_S)
        if np.isfinite(m["median_ttd_intervals"]) else float("nan")
    )
    if len(np.unique(y)) > 1:
        m["auc"] = float(roc_auc_score(y, scores))
        m["ap"] = float(average_precision_score(y, scores))
    else:
        m["auc"] = float("nan")
        m["ap"] = float("nan")
    neg = y == 0
    pos = y == 1
    m["window_tpr"] = float(np.mean(scores[pos] > threshold)) if pos.any() else float("nan")
    m["window_fpr_direct"] = float(np.mean(scores[neg] > threshold)) if neg.any() else float("nan")
    m["threshold"] = float(threshold)
    return m


def add_episode_parameters(
    frame: pd.DataFrame, episodes: Sequence
) -> pd.DataFrame:
    out = frame.copy()
    rate = {i: float(e.hidden_rate) for i, e in enumerate(episodes) if e.kind == "compromise"}
    family = {i: str(e.family) for i, e in enumerate(episodes) if e.kind == "compromise"}
    out["episode_hidden_rate"] = out["episode_id"].map(rate)
    out["episode_family_truth"] = out["episode_id"].map(family)
    return out


def episode_recall_by_group(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    group_col: str,
) -> List[dict]:
    d = frame.copy()
    d["_score"] = np.asarray(scores, dtype=float)
    rows = []
    pos = d[(d["label"] == 1) & d[group_col].notna()]
    for value, g in pos.groupby(group_col):
        episode_ids = sorted(int(x) for x in g["episode_id"].unique() if x >= 0)
        det = []
        ttd = []
        for eid in episode_ids:
            e = g[g["episode_id"] == eid].sort_values("t")
            fire = (e["_score"].to_numpy(float) > threshold).astype(int)
            run = 0
            hit = None
            for i, v in enumerate(fire):
                run = run + 1 if v else 0
                if run >= 3:
                    hit = i - 2
                    break
            det.append(hit is not None)
            if hit is not None:
                ttd.append(int(e["t"].iloc[hit] - e["t"].iloc[0]))
        rows.append({
            "group": str(value),
            "episodes": int(len(episode_ids)),
            "episode_recall": float(np.mean(det)) if det else float("nan"),
            "median_ttd_intervals": float(np.median(ttd)) if ttd else float("nan"),
            "median_ttd_seconds": float(np.median(ttd) * INTERVAL_S) if ttd else float("nan"),
        })
    return rows


def fit_common_detectors(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    te: pd.DataFrame,
    seed: int,
) -> Tuple[List[dict], Dict[str, np.ndarray], Dict[str, float], dict]:
    own_cols = [c for c in OWN_CAL_FEATURES if c in tr.columns]
    inv_cols = all_derived(tr, INV)
    full_cols = sorted(set(own_cols + inv_cols))
    if len(own_cols) != len(OWN_CAL_FEATURES):
        raise RuntimeError(f"Missing OWN calibrated features: have {own_cols}")
    if len(inv_cols) < 20:
        raise RuntimeError(f"Too few invariant features: {len(inv_cols)}")

    ytr = tr["label"].to_numpy(int)
    yva = va["label"].to_numpy(int)
    neg_tr = ytr == 0
    neg_va = yva == 0

    results = []
    test_scores: Dict[str, np.ndarray] = {}
    thresholds: Dict[str, float] = {}
    fitted: dict = {}

    # Mechanistic baseline at the same 1% validation-negative operating point.
    s_va = rule_baseline(va)
    s_te = rule_baseline(te)
    thr = fpr_threshold(s_va[neg_va], OVERALL_FPR)
    results.append({"detector": "rule_baseline", **detector_metrics(te, s_te, thr)})
    test_scores["rule_baseline"] = s_te
    thresholds["rule_baseline"] = thr

    # Canonical all-channel Isolation Forest.
    imp, sc, xtr = prep(tr, full_cols)
    xva = tx(va, full_cols, imp, sc)
    xte = tx(te, full_cols, imp, sc)
    iso = IsolationForest(
        n_estimators=400,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    ).fit(xtr[neg_tr])
    s_va = -iso.score_samples(xva)
    s_te = -iso.score_samples(xte)
    thr = fpr_threshold(s_va[neg_va], OVERALL_FPR)
    results.append({"detector": "isolation_forest_all", **detector_metrics(te, s_te, thr)})
    test_scores["isolation_forest_all"] = s_te
    thresholds["isolation_forest_all"] = thr

    # Canonical all-channel Random Forest.
    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    ).fit(xtr, ytr)
    s_va = rf.predict_proba(xva)[:, 1]
    s_te = rf.predict_proba(xte)[:, 1]
    thr = fpr_threshold(s_va[neg_va], OVERALL_FPR)
    results.append({"detector": "random_forest_all", **detector_metrics(te, s_te, thr)})
    test_scores["random_forest_all"] = s_te
    thresholds["random_forest_all"] = thr

    # Calibrated OWN channel: frozen feature family and IF hyperparameters.
    imp_o, sc_o, xo_tr = prep(tr[tr["label"] == 0], own_cols)
    own = IsolationForest(
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
        **OWN_CAL_IF,
    ).fit(xo_tr)
    own_va = -own.score_samples(tx(va, own_cols, imp_o, sc_o))
    own_te = -own.score_samples(tx(te, own_cols, imp_o, sc_o))
    own_thr = fpr_threshold(own_va[neg_va], CHANNEL_FPR)
    results.append({"detector": "calibrated_own_if", **detector_metrics(te, own_te, own_thr)})
    test_scores["calibrated_own_if"] = own_te
    thresholds["calibrated_own_if"] = own_thr

    # Calibrated INV channel: frozen 45-ish derived family and RF hyperparameters.
    imp_i, sc_i, xi_tr = prep(tr, inv_cols)
    inv = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
        **INV_CAL_RF,
    ).fit(xi_tr, ytr)
    inv_va = inv.predict_proba(tx(va, inv_cols, imp_i, sc_i))[:, 1]
    inv_te = inv.predict_proba(tx(te, inv_cols, imp_i, sc_i))[:, 1]
    inv_thr = fpr_threshold(inv_va[neg_va], CHANNEL_FPR)
    results.append({"detector": "calibrated_inv_rf", **detector_metrics(te, inv_te, inv_thr)})
    test_scores["calibrated_inv_rf"] = inv_te
    thresholds["calibrated_inv_rf"] = inv_thr

    # Fusion by OR.  Continuous score is a threshold-normalised margin so AUC/AP
    # remain interpretable; alert iff fused margin > 0.
    own_scale = robust_scale(own_va[neg_va])
    inv_scale = robust_scale(inv_va[neg_va])
    fused_te = np.maximum(
        (own_te - own_thr) / own_scale,
        (inv_te - inv_thr) / inv_scale,
    )
    fused_va = np.maximum(
        (own_va - own_thr) / own_scale,
        (inv_va - inv_thr) / inv_scale,
    )
    fused_thr = 0.0
    fm = detector_metrics(te, fused_te, fused_thr)
    fm["validation_window_fpr"] = float(np.mean(fused_va[neg_va] > fused_thr))
    fm["bonferroni_bound"] = float(2 * CHANNEL_FPR)
    results.append({"detector": "calibrated_hybrid_or", **fm})
    test_scores["calibrated_hybrid_or"] = fused_te
    thresholds["calibrated_hybrid_or"] = fused_thr

    fitted["own"] = (own_cols, imp_o, sc_o, own, own_thr, own_scale)
    fitted["inv"] = (inv_cols, imp_i, sc_i, inv, inv_thr, inv_scale)
    fitted["full_cols"] = full_cols
    return results, test_scores, thresholds, fitted


def bootstrap_ci(values: Sequence[float], seed: int = 123, n_boot: int = 3000) -> List[float]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return [float("nan"), float("nan")]
    if len(x) == 1:
        return [float(x[0]), float(x[0])]
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate_multiseed(rows: pd.DataFrame) -> List[dict]:
    out = []
    for detector, g in rows.groupby("detector"):
        item = {"detector": detector, "n_seeds": int(g["seed"].nunique())}
        for col in [
            "episode_recall", "auc", "ap", "window_fpr_direct",
            "alerts_per_day_network", "median_ttd_seconds", "window_tpr",
        ]:
            vals = g[col].to_numpy(float)
            item[f"{col}_mean"] = float(np.nanmean(vals))
            item[f"{col}_std"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else 0.0
            item[f"{col}_ci95"] = bootstrap_ci(vals)
            item[f"{col}_min"] = float(np.nanmin(vals))
            item[f"{col}_max"] = float(np.nanmax(vals))
        out.append(item)
    return out


def normal_event_audit(
    cfg: SimConfig,
    sim_template,
    fitted: dict,
) -> List[dict]:
    # Counterfactual benign world with many normal operational events and no compromise.
    normal_df, _, episodes = build_dataset(
        cfg,
        episodes_kw={"n_compromise": 0, "n_normal": 60, "n_ext": 0},
        sim_template=sim_template,
    )
    d = normal_df.copy()
    d["normal_event"] = "baseline"
    for e in episodes:
        if e.kind not in NORMAL_EVENTS or e.nid is None:
            continue
        mask = (d["nid"] == e.nid) & (d["t"] >= e.t0) & (d["t"] <= e.t1)
        d.loc[mask, "normal_event"] = e.kind

    own_cols, imp_o, sc_o, own, own_thr, own_scale = fitted["own"]
    inv_cols, imp_i, sc_i, inv, inv_thr, inv_scale = fitted["inv"]
    own_s = -own.score_samples(tx(d, own_cols, imp_o, sc_o))
    inv_s = inv.predict_proba(tx(d, inv_cols, imp_i, sc_i))[:, 1]
    fused = np.maximum((own_s - own_thr) / own_scale, (inv_s - inv_thr) / inv_scale)
    d["_fire"] = fused > 0

    rows = []
    for event, g in d.groupby("normal_event"):
        rows.append({
            "normal_event": str(event),
            "rows": int(len(g)),
            "cells_or_nodes": int(g["nid"].nunique()),
            "alert_rate": float(g["_fire"].mean()),
            "alerts": int(g["_fire"].sum()),
        })
    return rows


def lomfo_calibrated(
    cfg: SimConfig,
    frame: pd.DataFrame,
    seed: int,
) -> List[dict]:
    rows = []
    inv_cols = all_derived(frame, INV)
    for family in MASK_FAMILIES:
        tr, va, te = blocked_split(frame, cfg, RngBook(seed + 100 + len(rows))["split"])
        tr2 = tr[(tr["label"] == 0) | (tr["mask_family"] != family)].copy()
        va2 = va[(va["label"] == 0) | (va["mask_family"] != family)].copy()
        te2 = te[(te["label"] == 0) | (te["mask_family"] == family)].copy()
        if tr2["label"].sum() < 20 or te2["label"].sum() < 20:
            continue
        ytr = tr2["label"].to_numpy(int)
        yva = va2["label"].to_numpy(int)
        imp, sc, xtr = prep(tr2, inv_cols)
        model = RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
            **INV_CAL_RF,
        ).fit(xtr, ytr)
        sva = model.predict_proba(tx(va2, inv_cols, imp, sc))[:, 1]
        ste = model.predict_proba(tx(te2, inv_cols, imp, sc))[:, 1]
        thr = fpr_threshold(sva[yva == 0], CHANNEL_FPR)
        rows.append({
            "holdout_family": family,
            **detector_metrics(te2, ste, thr),
        })
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2026, 7, 31415])
    ap.add_argument("--days", type=float, default=1.25)
    ap.add_argument("--output-dir", default="results/ss7_calibrated_detector_validation")
    ap.add_argument("--skip-null-test", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    rate_rows = []
    family_rows = []
    seed_meta = []
    primary_frame = primary_cfg = primary_sim = primary_fitted = None

    for seed_idx, seed in enumerate(args.seeds):
        cfg = SimConfig(seed=int(seed), days=float(args.days))
        frame, sim, episodes = build_dataset(
            cfg,
            episodes_kw={
                "n_compromise": 15,
                "n_normal": 30,
                "n_ext": 6,
                "soph": Sophistication.ADAPTIVE,
            },
        )
        frame = add_episode_parameters(frame, episodes)
        tr, va, te = blocked_split(frame, cfg, RngBook(seed)["split"])
        results, test_scores, thresholds, fitted = fit_common_detectors(
            tr, va, te, int(seed)
        )
        for r in results:
            r["seed"] = int(seed)
            r["test_rows"] = int(len(te))
            r["test_positive_windows"] = int(te["label"].sum())
            r["test_episodes"] = int(te.loc[te["label"] == 1, "episode_id"].nunique())
            all_rows.append(r)

        hybrid_scores = test_scores["calibrated_hybrid_or"]
        hybrid_thr = thresholds["calibrated_hybrid_or"]
        for row in episode_recall_by_group(
            te, hybrid_scores, hybrid_thr, "episode_hidden_rate"
        ):
            rate_rows.append({"seed": int(seed), **row})
        for row in episode_recall_by_group(
            te, hybrid_scores, hybrid_thr, "episode_family_truth"
        ):
            family_rows.append({"seed": int(seed), **row})

        seed_meta.append({
            "seed": int(seed),
            "rows": int(len(frame)),
            "nodes": int(frame["nid"].nunique()),
            "positive_windows": int(frame["label"].sum()),
            "attack_episodes": int(frame.loc[frame["label"] == 1, "episode_id"].nunique()),
            "train_rows": int(len(tr)),
            "validation_rows": int(len(va)),
            "test_rows": int(len(te)),
        })
        if seed_idx == 0:
            primary_frame, primary_cfg, primary_sim, primary_fitted = frame, cfg, sim, fitted

    results_df = pd.DataFrame(all_rows)
    rates_df = pd.DataFrame(rate_rows)
    families_df = pd.DataFrame(family_rows)
    results_df.to_csv(outdir / "detector_multiseed.csv", index=False)
    rates_df.to_csv(outdir / "hybrid_recall_by_hidden_rate.csv", index=False)
    families_df.to_csv(outdir / "hybrid_recall_by_family.csv", index=False)

    aggregate = aggregate_multiseed(results_df)

    lomfo_rows = lomfo_calibrated(
        primary_cfg, primary_frame, int(args.seeds[0])
    )
    pd.DataFrame(lomfo_rows).to_csv(outdir / "inv_lomfo.csv", index=False)

    normal_rows = normal_event_audit(
        primary_cfg, primary_sim, primary_fitted
    )
    pd.DataFrame(normal_rows).to_csv(outdir / "normal_event_false_alarms.csv", index=False)

    null_report = {"skipped": True}
    if not args.skip_null_test:
        union_cols = sorted(set(
            OWN_CAL_FEATURES + all_derived(primary_frame, INV)
        ))
        # Use the calibrated feature space.  null_test itself evaluates h=0 against
        # a paired honest counterfactual and checks diagnostic separability.
        null_report = null_test(
            primary_cfg,
            union_cols,
            auc_tol=0.08,
            delta=0.35,
            n_boot=100,
            min_pos=40,
            log=lambda *_: None,
        )

    # Aggregates by attack difficulty.
    rate_agg = []
    if not rates_df.empty:
        for group, g in rates_df.groupby("group"):
            vals = g["episode_recall"].to_numpy(float)
            rate_agg.append({
                "hidden_rate": group,
                "n_seed_groups": int(len(g)),
                "episode_recall_mean": float(np.nanmean(vals)),
                "episode_recall_std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "episode_recall_ci95": bootstrap_ci(vals),
                "episodes_total": int(g["episodes"].sum()),
            })

    family_agg = []
    if not families_df.empty:
        for group, g in families_df.groupby("group"):
            vals = g["episode_recall"].to_numpy(float)
            family_agg.append({
                "mask_family": group,
                "n_seed_groups": int(len(g)),
                "episode_recall_mean": float(np.nanmean(vals)),
                "episode_recall_std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "episode_recall_ci95": bootstrap_ci(vals),
                "episodes_total": int(g["episodes"].sum()),
            })

    lomfo_summary = []
    if lomfo_rows:
        for r in lomfo_rows:
            lomfo_summary.append({
                "holdout_family": r["holdout_family"],
                "episode_recall": r["episode_recall"],
                "auc": r["auc"],
                "ap": r["ap"],
                "window_fpr": r["window_fpr_direct"],
                "median_ttd_seconds": r["median_ttd_seconds"],
            })

    report = {
        "status": "completed",
        "protocol": {
            "model": "canonical ss7/fix.py",
            "detector_hyperparameters": "frozen from full hybrid calibration",
            "seeds": [int(s) for s in args.seeds],
            "days_per_seed": float(args.days),
            "topology_regenerated_per_seed": True,
            "split": "blocked temporal train/validation/test with purge gap",
            "threshold_selection": "validation negatives only",
            "overall_target_fpr": OVERALL_FPR,
            "hybrid_channel_target_fpr": CHANNEL_FPR,
            "test_used_for_tuning": False,
        },
        "detectors": {
            "rule_baseline": "mechanistic invariant score, 1% validation-negative threshold",
            "isolation_forest_all": "canonical unsupervised detector over calibrated OWN + INV space",
            "random_forest_all": "canonical supervised detector over calibrated OWN + INV space",
            "calibrated_own_if": {
                "features": OWN_CAL_FEATURES,
                "params": OWN_CAL_IF,
                "channel_target_fpr": CHANNEL_FPR,
            },
            "calibrated_inv_rf": {
                "feature_family": "all derived INV features",
                "params": INV_CAL_RF,
                "channel_target_fpr": CHANNEL_FPR,
            },
            "calibrated_hybrid_or": "OWN alert OR INV alert; Bonferroni 0.5% + 0.5%",
        },
        "seed_meta": seed_meta,
        "multiseed_aggregate": aggregate,
        "hybrid_recall_by_hidden_rate": rate_agg,
        "hybrid_recall_by_mask_family": family_agg,
        "inv_leave_one_mask_family_out": lomfo_summary,
        "normal_event_false_alarm_audit": normal_rows,
        "null_test": null_report,
        "limitations": [
            "All attack labels in this validation are generated by the canonical simulation model.",
            "This validates detector behaviour on calibrated simulation data; it is not evidence of real-attack recall.",
            "The OWN detector is re-fitted to simulated normal data with frozen real-calibrated hyperparameters; its real CDR FPR is reported separately by the external-validation workflow.",
            "The INV threshold is calibrated per simulated realization because prior cross-topology testing rejected a universal fixed numeric threshold.",
        ],
    }
    (outdir / "ss7_calibrated_detector_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "status": "completed",
        "multiseed_aggregate": aggregate,
        "hybrid_recall_by_hidden_rate": rate_agg,
        "hybrid_recall_by_mask_family": family_agg,
        "lomfo": lomfo_summary,
        "normal_events": normal_rows,
        "null_test_passed": bool(
            null_report.get("passed", False)
            and null_report.get("role_diagnostic_passed", False)
        ) if not args.skip_null_test else None,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
