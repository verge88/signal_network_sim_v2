"""Two-stage SS7 detector validation with hard-negative benign suppression.

Architecture
------------
Stage 1: INV Random Forest produces the attack-evidence score.
Stage 2: a hard-negative suppressor sees OWN derived features plus Stage-1 score
and learns attack vs benign among difficult windows. It never uses test labels.

Training protocol
-----------------
* Base INV model: train blocks from development seeds 42 and 1337.
* Suppressor: out-of-sample validation blocks from the same development seeds.
  All attack windows are included, plus the top 10% hardest benign INV scores.
* Evaluation: independent regenerated seeds. Model weights are frozen.
* Deployment threshold: calibrated only on each evaluation seed's validation
  negatives at 1% FPR; test is touched once.
* A separate frozen-threshold diagnostic from seed 2026 is reported only to show
  cross-topology threshold transfer and is not the recommended operating mode.
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
from sklearn.preprocessing import StandardScaler

from ss7.fix import (
    FEATURE_SETS,
    INV,
    MASK_FAMILIES,
    NORMAL_EVENTS,
    RngBook,
    SimConfig,
    blocked_split,
    build_dataset,
    rule_baseline,
)
from ss7.validate_calibrated_detectors import (
    INV_CAL_RF,
    add_episode_parameters,
    all_derived,
    bootstrap_ci,
    detector_metrics,
    episode_recall_by_group,
    fpr_threshold,
    prep,
    tx,
)
from ss7.validate_context_aware_detector import (
    annotate_context,
    purge_cross_split_benign_events,
)


TARGET_FPR = 0.01
HARD_NEGATIVE_Q = 0.90
SUPPRESSOR_RF = {
    "n_estimators": 600,
    "min_samples_leaf": 8,
    "max_features": "sqrt",
}
FULL_RF = {
    "n_estimators": 500,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
}


def namespace_rows(frame: pd.DataFrame, seed_index: int) -> pd.DataFrame:
    out = frame.copy()
    out["source_seed"] = int(seed_index)
    out["nid"] = out["nid"].astype(int) + seed_index * 10000
    pos = out["episode_id"] >= 0
    out.loc[pos, "episode_id"] = (
        out.loc[pos, "episode_id"].astype(int) + seed_index * 1000
    )
    return out


def build_seed(seed: int, days: float, n_compromise: int, n_normal: int):
    cfg = SimConfig(seed=int(seed), days=float(days))
    frame, sim, episodes = build_dataset(
        cfg,
        episodes_kw={
            "n_compromise": int(n_compromise),
            "n_normal": int(n_normal),
            "n_ext": 6,
        },
    )
    frame = annotate_context(frame, episodes)
    frame = add_episode_parameters(frame, episodes)
    tr, va, te = blocked_split(frame, cfg, RngBook(seed)["split"])
    tr, va, te, audit = purge_cross_split_benign_events(tr, va, te)
    return frame, tr, va, te, episodes, sim, audit


def fit_base_inv(
    train: pd.DataFrame,
    inv_cols: Sequence[str],
    seed: int,
) -> Tuple[SimpleImputer, StandardScaler, RandomForestClassifier]:
    imp, sc, x = prep(train, inv_cols)
    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=int(seed),
        n_jobs=-1,
        **INV_CAL_RF,
    ).fit(x, train["label"].to_numpy(int))
    return imp, sc, model


def inv_score(
    frame: pd.DataFrame,
    inv_cols: Sequence[str],
    fitted,
) -> np.ndarray:
    imp, sc, model = fitted
    return model.predict_proba(tx(frame, inv_cols, imp, sc))[:, 1]


def fit_full_rf(
    train: pd.DataFrame,
    full_cols: Sequence[str],
    seed: int,
):
    imp, sc, x = prep(train, full_cols)
    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=int(seed) + 17,
        n_jobs=-1,
        **FULL_RF,
    ).fit(x, train["label"].to_numpy(int))
    return imp, sc, model


def full_rf_score(frame: pd.DataFrame, cols: Sequence[str], fitted) -> np.ndarray:
    imp, sc, model = fitted
    return model.predict_proba(tx(frame, cols, imp, sc))[:, 1]


def suppressor_matrix(
    frame: pd.DataFrame,
    own_cols: Sequence[str],
    inv_scores: np.ndarray,
    imp: SimpleImputer | None = None,
    sc: StandardScaler | None = None,
):
    base = frame[list(own_cols)].copy()
    base["stage1_inv_score"] = np.asarray(inv_scores, dtype=float)
    cols = list(own_cols) + ["stage1_inv_score"]
    if imp is None:
        imp = SimpleImputer(strategy="median").fit(base[cols])
    x = imp.transform(base[cols])
    if sc is None:
        sc = StandardScaler().fit(x)
    return imp, sc, sc.transform(x), cols


def fit_suppressor(
    frame: pd.DataFrame,
    inv_scores: np.ndarray,
    own_cols: Sequence[str],
    seed: int,
) -> Tuple[SimpleImputer, StandardScaler, RandomForestClassifier, List[str], dict]:
    y = frame["label"].to_numpy(int)
    neg = y == 0
    pos = y == 1
    if not neg.any() or not pos.any():
        raise RuntimeError("Suppressor development set requires both classes")

    hard_thr = float(np.quantile(inv_scores[neg], HARD_NEGATIVE_Q))
    keep = pos | (neg & (inv_scores >= hard_thr))
    hard = frame.loc[keep].copy()
    hard_scores = np.asarray(inv_scores)[keep]
    hy = hard["label"].to_numpy(int)

    imp, sc, x, cols = suppressor_matrix(
        hard, own_cols, hard_scores
    )
    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=int(seed) + 1000,
        n_jobs=-1,
        **SUPPRESSOR_RF,
    ).fit(x, hy)
    meta = {
        "hard_negative_quantile": HARD_NEGATIVE_Q,
        "hard_negative_inv_threshold": hard_thr,
        "rows": int(len(hard)),
        "positive_windows": int(hy.sum()),
        "negative_windows": int((hy == 0).sum()),
        "features": cols,
    }
    return imp, sc, model, cols, meta


def suppressor_score(
    frame: pd.DataFrame,
    inv_scores: np.ndarray,
    own_cols: Sequence[str],
    fitted,
) -> np.ndarray:
    imp, sc, model, _, _ = fitted
    _, _, x, _ = suppressor_matrix(frame, own_cols, inv_scores, imp, sc)
    return model.predict_proba(x)[:, 1]


def two_stage_score(inv_scores: np.ndarray, suppressor_scores: np.ndarray) -> np.ndarray:
    # Product keeps INV evidence mandatory. A benign suppressor can only reduce,
    # never create, attack evidence.
    return np.asarray(inv_scores, float) * np.asarray(suppressor_scores, float)


def development_models(
    seeds: Sequence[int],
    days: float,
):
    train_parts = []
    gate_parts = []
    gate_scores = []
    template = None
    audits = []

    # First pass constructs train/validation blocks.
    seed_data = []
    for idx, seed in enumerate(seeds):
        frame, tr, va, _, _, _, audit = build_seed(
            seed, days, n_compromise=28, n_normal=60
        )
        seed_data.append((idx, seed, frame, tr, va))
        train_parts.append(namespace_rows(tr, idx))
        audits.append({"seed": int(seed), "split_audit": audit})
        if template is None:
            template = frame

    dev_train = pd.concat(train_parts, ignore_index=True, sort=False)
    inv_cols = all_derived(template, FEATURE_SETS["INV_only"])
    own_cols = all_derived(template, FEATURE_SETS["OWN_only"])
    full_cols = all_derived(template, FEATURE_SETS["OWN+INV"])

    base_inv = fit_base_inv(dev_train, inv_cols, int(seeds[0]))
    full_rf = fit_full_rf(dev_train, full_cols, int(seeds[0]))

    # Stage-2 training uses base-model out-of-sample validation blocks.
    for idx, seed, _, _, va in seed_data:
        s = inv_score(va, inv_cols, base_inv)
        gate_parts.append(namespace_rows(va, idx))
        gate_scores.append(s)
    gate_frame = pd.concat(gate_parts, ignore_index=True, sort=False)
    gate_score_vec = np.concatenate(gate_scores)

    suppressor = fit_suppressor(
        gate_frame, gate_score_vec, own_cols, int(seeds[0])
    )

    return {
        "inv_cols": inv_cols,
        "own_cols": own_cols,
        "full_cols": full_cols,
        "base_inv": base_inv,
        "full_rf": full_rf,
        "suppressor": suppressor,
        "development_rows": int(len(dev_train)),
        "development_positive_windows": int(dev_train["label"].sum()),
        "suppressor_meta": suppressor[-1],
        "audits": audits,
    }


def normal_event_rows(
    frame: pd.DataFrame,
    fire_map: Dict[str, np.ndarray],
    seed: int,
) -> List[dict]:
    truth = frame["benign_event_kind_truth"].astype(str).to_numpy()
    neg = frame["label"].to_numpy(int) == 0
    rows = []
    for detector, fire in fire_map.items():
        for kind in sorted(set(truth[neg])):
            mask = neg & (truth == kind)
            if not mask.any():
                continue
            rows.append({
                "seed": int(seed),
                "detector": detector,
                "normal_event": kind,
                "rows": int(mask.sum()),
                "alerts": int(np.sum(fire[mask])),
                "window_fpr": float(np.mean(fire[mask])),
            })
    return rows


def evaluate_seed(
    seed: int,
    days: float,
    models: dict,
    frozen_threshold: float | None = None,
):
    frame, tr, va, te, _, _, split_audit = build_seed(
        seed, days, n_compromise=20, n_normal=50
    )
    inv_cols = models["inv_cols"]
    own_cols = models["own_cols"]
    full_cols = models["full_cols"]

    inv_va = inv_score(va, inv_cols, models["base_inv"])
    inv_te = inv_score(te, inv_cols, models["base_inv"])
    sup_va = suppressor_score(va, inv_va, own_cols, models["suppressor"])
    sup_te = suppressor_score(te, inv_te, own_cols, models["suppressor"])
    two_va = two_stage_score(inv_va, sup_va)
    two_te = two_stage_score(inv_te, sup_te)

    full_va = full_rf_score(va, full_cols, models["full_rf"])
    full_te = full_rf_score(te, full_cols, models["full_rf"])
    rule_va = rule_baseline(va)
    rule_te = rule_baseline(te)

    yva = va["label"].to_numpy(int)
    neg_va = yva == 0

    thr_inv = fpr_threshold(inv_va[neg_va], TARGET_FPR)
    thr_two = fpr_threshold(two_va[neg_va], TARGET_FPR)
    thr_full = fpr_threshold(full_va[neg_va], TARGET_FPR)
    thr_rule = fpr_threshold(rule_va[neg_va], TARGET_FPR)

    scores = {
        "rule_baseline": (rule_te, thr_rule),
        "inv_rf": (inv_te, thr_inv),
        "full_rf": (full_te, thr_full),
        "two_stage_inv_own": (two_te, thr_two),
    }

    metrics = []
    fires = {}
    for name, (s, thr) in scores.items():
        m = detector_metrics(te, s, thr)
        metrics.append({"seed": int(seed), "detector": name, **m})
        fires[name] = np.asarray(s) > float(thr)

    # Frozen-threshold diagnostic for the two-stage score.
    frozen_metric = None
    if frozen_threshold is not None:
        frozen_metric = {
            "seed": int(seed),
            "detector": "two_stage_frozen_threshold_diagnostic",
            **detector_metrics(te, two_te, float(frozen_threshold)),
        }

    normal = normal_event_rows(te, fires, seed)

    rate_rows = []
    family_rows = []
    for row in episode_recall_by_group(
        te, two_te, thr_two, "episode_hidden_rate"
    ):
        rate_rows.append({"seed": int(seed), **row})
    for row in episode_recall_by_group(
        te, two_te, thr_two, "episode_family_truth"
    ):
        family_rows.append({"seed": int(seed), **row})

    return {
        "metrics": metrics,
        "normal": normal,
        "rates": rate_rows,
        "families": family_rows,
        "two_stage_validation_threshold": float(thr_two),
        "frozen_metric": frozen_metric,
        "split_audit": split_audit,
        "rows": {
            "train": int(len(tr)),
            "validation": int(len(va)),
            "test": int(len(te)),
            "test_episodes": int(te.loc[te["label"] == 1, "episode_id"].nunique()),
        },
    }


def aggregate_metrics(df: pd.DataFrame) -> List[dict]:
    rows = []
    for detector, g in df.groupby("detector"):
        item = {"detector": detector, "n_seeds": int(g["seed"].nunique())}
        for col in [
            "episode_recall",
            "window_fpr_direct",
            "auc",
            "ap",
            "median_ttd_seconds",
            "window_tpr",
            "alerts_per_day_network",
        ]:
            vals = g[col].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            item[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            item[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            item[f"{col}_min"] = float(np.min(vals)) if len(vals) else float("nan")
            item[f"{col}_max"] = float(np.max(vals)) if len(vals) else float("nan")
            item[f"{col}_ci95"] = bootstrap_ci(vals)
        rows.append(item)
    return rows


def aggregate_normal(df: pd.DataFrame) -> List[dict]:
    rows = []
    for (detector, event), g in df.groupby(["detector", "normal_event"]):
        n = int(g["rows"].sum())
        a = int(g["alerts"].sum())
        rows.append({
            "detector": detector,
            "normal_event": event,
            "rows": n,
            "alerts": a,
            "pooled_fpr": float(a / n) if n else float("nan"),
            "mean_seed_fpr": float(g["window_fpr"].mean()),
            "max_seed_fpr": float(g["window_fpr"].max()),
        })
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--development-seeds", nargs="+", type=int, default=[42, 1337])
    ap.add_argument("--calibration-seed", type=int, default=2026)
    ap.add_argument(
        "--evaluation-seeds",
        nargs="+",
        type=int,
        default=[7001, 7002, 7003, 7004, 7005],
    )
    ap.add_argument("--development-days", type=float, default=1.1)
    ap.add_argument("--evaluation-days", type=float, default=1.25)
    ap.add_argument(
        "--output-dir",
        default="results/ss7_two_stage_validation",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    models = development_models(args.development_seeds, args.development_days)

    # Independent seed supplies a frozen-threshold diagnostic only.
    calibration = evaluate_seed(
        int(args.calibration_seed),
        float(args.evaluation_days),
        models,
        frozen_threshold=None,
    )
    frozen_threshold = float(calibration["two_stage_validation_threshold"])

    all_metrics = []
    all_normal = []
    all_rates = []
    all_families = []
    eval_meta = []
    frozen_rows = []

    for seed in args.evaluation_seeds:
        result = evaluate_seed(
            int(seed),
            float(args.evaluation_days),
            models,
            frozen_threshold=frozen_threshold,
        )
        all_metrics.extend(result["metrics"])
        all_normal.extend(result["normal"])
        all_rates.extend(result["rates"])
        all_families.extend(result["families"])
        if result["frozen_metric"] is not None:
            frozen_rows.append(result["frozen_metric"])
        eval_meta.append({
            "seed": int(seed),
            "rows": result["rows"],
            "split_audit": result["split_audit"],
            "local_two_stage_threshold": result["two_stage_validation_threshold"],
        })

    metrics_df = pd.DataFrame(all_metrics)
    normal_df = pd.DataFrame(all_normal)
    rates_df = pd.DataFrame(all_rates)
    families_df = pd.DataFrame(all_families)
    frozen_df = pd.DataFrame(frozen_rows)

    metrics_df.to_csv(outdir / "two_stage_multiseed.csv", index=False)
    normal_df.to_csv(outdir / "two_stage_normal_event_fpr.csv", index=False)
    rates_df.to_csv(outdir / "two_stage_recall_by_hidden_rate.csv", index=False)
    families_df.to_csv(outdir / "two_stage_recall_by_mask_family.csv", index=False)
    frozen_df.to_csv(outdir / "two_stage_frozen_threshold.csv", index=False)

    report = {
        "status": "completed",
        "architecture": {
            "stage1": "frozen cross-seed INV Random Forest",
            "stage2": (
                "hard-negative RF on OWN derived features + stage1 INV score; "
                "all attacks + top 10% benign INV scores"
            ),
            "fusion_score": "stage1_inv_probability * stage2_attack_probability",
            "model_development_seeds": [int(s) for s in args.development_seeds],
            "frozen_threshold_calibration_seed": int(args.calibration_seed),
            "recommended_threshold_policy": (
                "local validation-negative q99 on each target realization"
            ),
            "target_fpr": TARGET_FPR,
            "test_used_for_tuning": False,
        },
        "development": {
            "rows": models["development_rows"],
            "positive_windows": models["development_positive_windows"],
            "suppressor": models["suppressor_meta"],
            "split_audits": models["audits"],
        },
        "calibration_seed": {
            "seed": int(args.calibration_seed),
            "frozen_two_stage_threshold": frozen_threshold,
            "rows": calibration["rows"],
        },
        "evaluation_meta": eval_meta,
        "multiseed": aggregate_metrics(metrics_df),
        "normal_event_fpr": aggregate_normal(normal_df),
        "frozen_threshold_diagnostic": aggregate_metrics(frozen_df)
        if not frozen_df.empty else [],
        "limitations": [
            "Attack and benign-event labels used to train the suppressor are simulation-generated.",
            "The experiment validates simulated detector architecture, not real SS7 attack recall.",
            "The recommended operating threshold remains locally calibrated because a universal INV threshold previously failed cross-topology FPR transfer.",
        ],
    }
    (outdir / "ss7_two_stage_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
