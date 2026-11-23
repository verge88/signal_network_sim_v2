"""Full hybrid calibration for the canonical SS7 detector.

This module calibrates every part that can be calibrated with the data currently
available without fabricating observables:

1. OWN observable channel: trained on independent non-synthetic Telecom Italia CDR
   normal traffic from one day, thresholded on the next day, tested once on a third
   day.  Only scale-robust proxy features defensibly reconstructable from CDR are
   eligible.
2. INV cross-node channel: calibrated on canonical SS7 simulations because public
   CDR has no link/OD/TCAP reconciliation observables.  Hyperparameters and invariant
   feature family are selected on a simulation validation split, then evaluated both
   on a held-out split and on a different seed/topology with the threshold frozen.
3. Fusion: Bonferroni OR rule.  A 1% network-window false-positive budget is split
   into 0.5% for OWN and 0.5% for INV, requiring no independence assumption.

The procedure does not claim real-attack sensitivity.  It produces deployable
calibration artifacts plus explicit provenance and scope limitations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ss7.fix import (
    INV,
    RngBook,
    SimConfig,
    blocked_split,
    build_dataset,
    episode_metrics,
)
from ss7.validate_real_cdr import (
    MIRROR_REPO,
    MIRROR_REVISION,
    download_prefix,
    load_real_cdr_features,
    mirror_url,
    rejection_summary,
    temporal_block_fpr,
    threshold_for_fpr,
)


OWN_FEATURE_CANDIDATES: Dict[str, List[str]] = {
    "ratio4": ["mtype_entropy", "map_share", "isup_share", "in_out_ratio"],
    # MAP + ISUP shares are algebraically close to redundant in the CDR proxy.
    "ratio3_no_redundant_isup": ["mtype_entropy", "map_share", "in_out_ratio"],
    "ratio3_no_direction": ["mtype_entropy", "map_share", "isup_share"],
}

IF_GRID = (
    {"n_estimators": 300, "max_samples": "auto", "max_features": 1.0},
    {"n_estimators": 500, "max_samples": 512, "max_features": 1.0},
    {"n_estimators": 500, "max_samples": 1024, "max_features": 0.75},
)


def fit_if(
    train: pd.DataFrame,
    cols: Sequence[str],
    seed: int,
    params: dict,
) -> Tuple[SimpleImputer, StandardScaler, IsolationForest]:
    imp = SimpleImputer(strategy="median").fit(train[list(cols)])
    scaler = StandardScaler().fit(imp.transform(train[list(cols)]))
    x = scaler.transform(imp.transform(train[list(cols)]))
    model = IsolationForest(
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
        **params,
    ).fit(x)
    return imp, scaler, model


def scores(
    frame: pd.DataFrame,
    cols: Sequence[str],
    imp: SimpleImputer,
    scaler: StandardScaler,
    model: IsolationForest,
) -> np.ndarray:
    x = scaler.transform(imp.transform(frame[list(cols)]))
    return -model.score_samples(x)


def finite_metric(v: float, default: float) -> float:
    return float(v) if np.isfinite(v) else float(default)


def score_resolution(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"unique": 0, "unique_fraction": 0.0, "iqr": float("nan")}
    rounded = np.round(np.asarray(x, dtype=float), 12)
    return {
        "unique": int(np.unique(rounded).size),
        "unique_fraction": float(np.unique(rounded).size / len(rounded)),
        "iqr": float(np.quantile(rounded, 0.75) - np.quantile(rounded, 0.25)),
    }


def load_real_day(
    date: str,
    cache: Path,
    max_bytes: int,
    max_windows: int,
    seed: int,
) -> Tuple[pd.DataFrame, dict]:
    path = cache / f"{date}.csv"
    downloaded, prefix_sha = download_prefix(mirror_url(date), path, max_bytes)
    frame, meta = load_real_cdr_features(path, max_windows, seed)
    meta = {
        **meta,
        "date": date,
        "downloaded_prefix_bytes": int(downloaded),
        "downloaded_prefix_sha256": prefix_sha,
    }
    return frame, meta


def own_candidate_objective(
    validation: dict,
    stability: dict,
    resolution: dict,
    target: float,
) -> Tuple[float, float, float]:
    block_rates = np.array(
        [b["fpr"] for b in stability["blocks"]], dtype=float
    )
    max_block_dev = (
        float(np.max(np.abs(block_rates - target))) if len(block_rates) else 1.0
    )
    aggregate_dev = abs(float(validation["fpr"]) - target)
    # Tie/resolution penalty discourages a degenerate threshold with score saturation.
    resolution_penalty = max(0.0, 0.02 - float(resolution["unique_fraction"]))
    objective = (
        aggregate_dev
        + float(stability["std_fpr"])
        + max_block_dev
        + resolution_penalty
    )
    return (
        float(objective),
        -float(resolution["unique_fraction"]),
        -float(resolution["iqr"]) if np.isfinite(resolution["iqr"]) else 0.0,
    )


def calibrate_own(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    channel_fpr: float,
) -> Tuple[dict, dict, pd.DataFrame]:
    candidates = []
    fitted = {}

    for feature_name, cols in OWN_FEATURE_CANDIDATES.items():
        for grid_idx, params in enumerate(IF_GRID):
            imp, scaler, model = fit_if(train, cols, seed, params)
            val_scores = scores(validation, cols, imp, scaler, model)
            threshold = threshold_for_fpr(val_scores, channel_fpr)
            val_summary = rejection_summary(val_scores, threshold)
            stability = temporal_block_fpr(validation, val_scores, threshold)
            resolution = score_resolution(val_scores)
            objective = own_candidate_objective(
                val_summary, stability, resolution, channel_fpr
            )
            key = f"{feature_name}/if{grid_idx}"
            row = {
                "candidate": key,
                "feature_family": feature_name,
                "features": list(cols),
                "params": params,
                "threshold": float(threshold),
                "validation": val_summary,
                "validation_temporal_stability": stability,
                "score_resolution": resolution,
                "selection_objective": list(objective),
            }
            candidates.append(row)
            fitted[key] = (imp, scaler, model, list(cols), params, threshold)

    selected = min(candidates, key=lambda x: tuple(x["selection_objective"]))
    imp, scaler, model, cols, params, threshold = fitted[selected["candidate"]]

    test_scores = scores(test, cols, imp, scaler, model)
    test_summary = rejection_summary(test_scores, threshold)
    test_stability = temporal_block_fpr(test, test_scores, threshold)

    result = {
        "selected_candidate": selected["candidate"],
        "selected_feature_family": selected["feature_family"],
        "features": cols,
        "params": params,
        "threshold": float(threshold),
        "channel_target_fpr": float(channel_fpr),
        "validation": selected["validation"],
        "validation_temporal_stability": selected["validation_temporal_stability"],
        "validation_score_resolution": selected["score_resolution"],
        "test": test_summary,
        "test_temporal_stability": test_stability,
        "selection_used_test": False,
    }
    artifact = {
        "kind": "ss7_own_real_normal_isolation_forest",
        "features": cols,
        "imputer": imp,
        "scaler": scaler,
        "model": model,
        "threshold": float(threshold),
        "channel_target_fpr": float(channel_fpr),
    }

    flat = []
    for c in candidates:
        flat.append({
            "channel": "own",
            "candidate": c["candidate"],
            "feature_family": c["feature_family"],
            "n_features": len(c["features"]),
            "n_estimators": c["params"]["n_estimators"],
            "max_samples": c["params"]["max_samples"],
            "max_features": c["params"]["max_features"],
            "threshold": c["threshold"],
            "validation_fpr": c["validation"]["fpr"],
            "validation_block_fpr_std": c["validation_temporal_stability"]["std_fpr"],
            "validation_score_unique_fraction": c["score_resolution"]["unique_fraction"],
            "objective": c["selection_objective"][0],
            "selected": c["candidate"] == selected["candidate"],
        })
    return result, artifact, pd.DataFrame(flat)


def invariant_feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
    raw = [c for c in INV if c in df.columns]
    raw_z = raw + [f"{c}__z" for c in raw if f"{c}__z" in df.columns]
    raw_zone = raw + [
        f"{c}__dev_zone" for c in raw if f"{c}__dev_zone" in df.columns
    ]
    all_derived = [
        c for c in df.columns
        if any(c == base or c.startswith(base + "__") for base in raw)
    ]
    return {
        "inv_raw": raw,
        "inv_raw_plus_z": raw_z,
        "inv_raw_plus_zone": raw_zone,
        "inv_all_derived": all_derived,
    }


def attack_summary(frame: pd.DataFrame, s: np.ndarray, threshold: float) -> dict:
    m = episode_metrics(frame, s, threshold, k_consec=3)
    y = frame["label"].to_numpy(int)
    m["auc"] = (
        float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    )
    m["ap"] = (
        float(average_precision_score(y, s))
        if len(np.unique(y)) > 1 else float("nan")
    )
    neg = y == 0
    m["negative_window_fpr_direct"] = (
        float(np.mean(s[neg] > threshold)) if neg.any() else float("nan")
    )
    return m


def invariant_rank_key(row: dict, target: float) -> Tuple[float, float, float, float, int]:
    m = row["validation_attack_metrics"]
    recall = finite_metric(m["episode_recall"], -1.0)
    ap = finite_metric(m["ap"], -1.0)
    auc = finite_metric(m["auc"], -1.0)
    fpr_dev = abs(finite_metric(m["negative_window_fpr_direct"], 1.0) - target)
    # Select exclusively on validation: maximize incident recall, then AP/AUC,
    # then operational FPR fidelity, finally prefer the smaller representation.
    return (-recall, -ap, -auc, fpr_dev, int(row["n_features"]))


def make_synthetic_dataset(seed: int, days: float) -> Tuple[pd.DataFrame, dict]:
    cfg = SimConfig(seed=seed, days=days)
    df, sim, episodes = build_dataset(
        cfg,
        episodes_kw={
            "n_compromise": 12,
            "n_normal": 24,
            "n_ext": 6,
        },
    )
    meta = {
        "seed": int(seed),
        "days": float(days),
        "rows": int(len(df)),
        "nodes": int(df["nid"].nunique()),
        "positive_windows": int(df["label"].sum()),
        "compromise_episodes": int(
            len({e for e in df.loc[df["label"] == 1, "episode_id"].unique() if e >= 0})
        ),
        "topology_nodes": int(len(sim.topo.nodes)),
        "episodes_total": int(len(episodes)),
    }
    return df, meta


def calibrate_invariant(
    seed: int,
    confirm_seed: int,
    days: float,
    channel_fpr: float,
) -> Tuple[dict, dict, pd.DataFrame]:
    cfg = SimConfig(seed=seed, days=days)
    df, primary_meta = make_synthetic_dataset(seed, days)
    tr, va, te = blocked_split(df, cfg, RngBook(seed)["split"])
    families = invariant_feature_families(df)

    candidates = []
    fitted = {}
    yva = va["label"].to_numpy(int)

    for family_name, cols in families.items():
        if not cols:
            continue
        for grid_idx, params in enumerate(IF_GRID):
            imp, scaler, model = fit_if(tr[tr["label"] == 0], cols, seed, params)
            val_scores = scores(va, cols, imp, scaler, model)
            neg_scores = val_scores[yva == 0]
            threshold = threshold_for_fpr(neg_scores, channel_fpr)
            metrics = attack_summary(va, val_scores, threshold)
            key = f"{family_name}/if{grid_idx}"
            row = {
                "candidate": key,
                "feature_family": family_name,
                "features": list(cols),
                "n_features": int(len(cols)),
                "params": params,
                "threshold": float(threshold),
                "validation_attack_metrics": metrics,
                "validation_negative_score_resolution": score_resolution(neg_scores),
            }
            row["selection_key"] = list(invariant_rank_key(row, channel_fpr))
            candidates.append(row)
            fitted[key] = (imp, scaler, model, list(cols), params, threshold)

    if not candidates:
        raise RuntimeError("No invariant calibration candidate could be constructed")
    selected = min(candidates, key=lambda x: tuple(x["selection_key"]))
    imp, scaler, model, cols, params, threshold = fitted[selected["candidate"]]

    same_seed_scores = scores(te, cols, imp, scaler, model)
    same_seed_metrics = attack_summary(te, same_seed_scores, threshold)

    confirm_df, confirm_meta = make_synthetic_dataset(confirm_seed, days)
    confirm_scores = scores(confirm_df, cols, imp, scaler, model)
    confirm_metrics = attack_summary(confirm_df, confirm_scores, threshold)

    result = {
        "selected_candidate": selected["candidate"],
        "selected_feature_family": selected["feature_family"],
        "features": cols,
        "params": params,
        "threshold": float(threshold),
        "channel_target_fpr": float(channel_fpr),
        "validation_attack_metrics": selected["validation_attack_metrics"],
        "same_seed_heldout_test": same_seed_metrics,
        "cross_seed_frozen_threshold_test": confirm_metrics,
        "primary_synthetic_meta": primary_meta,
        "primary_split_rows": {
            "train": int(len(tr)),
            "validation": int(len(va)),
            "test": int(len(te)),
        },
        "cross_seed_meta": confirm_meta,
        "selection_used_primary_test": False,
        "selection_used_cross_seed_test": False,
    }
    artifact = {
        "kind": "ss7_invariant_isolation_forest",
        "features": cols,
        "imputer": imp,
        "scaler": scaler,
        "model": model,
        "threshold": float(threshold),
        "channel_target_fpr": float(channel_fpr),
        "synthetic_calibration_seed": int(seed),
    }

    flat = []
    for c in candidates:
        m = c["validation_attack_metrics"]
        flat.append({
            "channel": "invariant",
            "candidate": c["candidate"],
            "feature_family": c["feature_family"],
            "n_features": c["n_features"],
            "n_estimators": c["params"]["n_estimators"],
            "max_samples": c["params"]["max_samples"],
            "max_features": c["params"]["max_features"],
            "threshold": c["threshold"],
            "validation_fpr": m["negative_window_fpr_direct"],
            "validation_episode_recall": m["episode_recall"],
            "validation_auc": m["auc"],
            "validation_ap": m["ap"],
            "selected": c["candidate"] == selected["candidate"],
        })
    return result, artifact, pd.DataFrame(flat)


def jsonable_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-train-date", default="2013-11-01")
    ap.add_argument("--real-validation-date", default="2013-11-02")
    ap.add_argument("--real-test-date", default="2013-11-03")
    ap.add_argument("--max-bytes-per-day", type=int, default=64 * 1024 * 1024)
    ap.add_argument("--max-real-windows-per-day", type=int, default=120000)
    ap.add_argument("--overall-target-fpr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--confirm-seed", type=int, default=2026)
    ap.add_argument("--synthetic-days", type=float, default=1.0)
    ap.add_argument("--cache-dir", default=".cache/full_calibration")
    ap.add_argument("--output-dir", default="results/full_calibration")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not (0.0 < args.overall_target_fpr < 0.2):
        raise ValueError("overall-target-fpr must be in (0, 0.2)")
    dates = [
        args.real_train_date,
        args.real_validation_date,
        args.real_test_date,
    ]
    if len(set(dates)) != 3:
        raise ValueError("Real train/validation/test dates must be distinct")

    channel_fpr = float(args.overall_target_fpr / 2.0)
    cache = Path(args.cache_dir)
    outdir = Path(args.output_dir)
    cache.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    real_train, train_meta = load_real_day(
        args.real_train_date, cache, args.max_bytes_per_day,
        args.max_real_windows_per_day, args.seed,
    )
    real_validation, validation_meta = load_real_day(
        args.real_validation_date, cache, args.max_bytes_per_day,
        args.max_real_windows_per_day, args.seed,
    )
    real_test, test_meta = load_real_day(
        args.real_test_date, cache, args.max_bytes_per_day,
        args.max_real_windows_per_day, args.seed,
    )

    own, own_artifact, own_candidates = calibrate_own(
        real_train, real_validation, real_test, args.seed, channel_fpr
    )
    inv, inv_artifact, inv_candidates = calibrate_invariant(
        args.seed, args.confirm_seed, args.synthetic_days, channel_fpr
    )

    joblib.dump(own_artifact, outdir / "ss7_own_calibrator.joblib")
    joblib.dump(inv_artifact, outdir / "ss7_invariant_calibrator.joblib")

    candidates = pd.concat(
        [own_candidates, inv_candidates], ignore_index=True, sort=False
    )
    candidates.to_csv(outdir / "ss7_calibration_candidates.csv", index=False)

    deployment = {
        "version": 1,
        "protocol": "ss7",
        "fusion": {
            "rule": "OR",
            "calibration": "Bonferroni allocation; no channel independence assumed",
            "overall_target_fpr": float(args.overall_target_fpr),
            "own_channel_target_fpr": channel_fpr,
            "invariant_channel_target_fpr": channel_fpr,
            "alert_rule": "own_score > own_threshold OR invariant_score > invariant_threshold",
            "joint_real_fpr_empirically_verified": False,
        },
        "own_channel": {
            "feature_family": own["selected_feature_family"],
            "features": own["features"],
            "isolation_forest": jsonable_params(own["params"]),
            "threshold": own["threshold"],
            "calibration_source": args.real_validation_date,
            "training_source": args.real_train_date,
        },
        "invariant_channel": {
            "feature_family": inv["selected_feature_family"],
            "features": inv["features"],
            "isolation_forest": jsonable_params(inv["params"]),
            "threshold": inv["threshold"],
            "training_source": f"canonical synthetic seed {args.seed}",
            "external_synthetic_confirmation_seed": int(args.confirm_seed),
        },
        "integrity_channel": {
            "default_key_custody": "node_software",
            "active_in_calibrated_profile": False,
            "reason": (
                "With node-held software keys the adversary can produce valid report "
                "MACs; integrity_fail is not treated as an independent calibrated "
                "detection channel."
            ),
        },
    }
    (outdir / "ss7_calibrated_profile.json").write_text(
        json.dumps(deployment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "status": "completed",
        "scope": {
            "name": "full_hybrid_calibration",
            "meaning": (
                "Full calibration of all currently calibratable channels. OWN is "
                "calibrated on real benign CDR; INV is calibrated on the canonical "
                "SS7 generator because public CDR does not expose cross-node SS7 "
                "invariants. Real attack sensitivity is not claimed."
            ),
        },
        "source": {
            "real_dataset": "Telecom Italia Milan telecommunications activity",
            "original_doi": "10.7910/DVN/EGZHFV",
            "mirror": f"https://huggingface.co/datasets/{MIRROR_REPO}",
            "mirror_revision": MIRROR_REVISION,
            "real_train": train_meta,
            "real_validation": validation_meta,
            "real_test": test_meta,
        },
        "calibration_protocol": {
            "real_split": "three distinct calendar days; no test-day tuning",
            "real_train_date": args.real_train_date,
            "real_validation_date": args.real_validation_date,
            "real_test_date": args.real_test_date,
            "overall_target_fpr": float(args.overall_target_fpr),
            "channel_target_fpr": channel_fpr,
            "fusion": "Bonferroni OR",
            "test_leakage": False,
        },
        "own_channel": own,
        "invariant_channel": inv,
        "deployment_profile": deployment,
        "limitations": [
            "Public CDR is not packet-level SS7 and has no real attack labels.",
            "Cross-node invariant calibration remains simulation-based.",
            "The 1% fused real-world FPR is a design budget, not a jointly measured "
            "real-data FPR, because no public dataset exposes OWN and INV channels together.",
            "Real attack recall/precision/AUC cannot be estimated from this CDR dataset.",
        ],
    }
    (outdir / "ss7_full_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "own_selected": own["selected_candidate"],
        "own_validation_fpr": own["validation"]["fpr"],
        "own_real_test_fpr": own["test"]["fpr"],
        "own_real_test_ci95": own["test"]["fpr_wilson_95"],
        "inv_selected": inv["selected_candidate"],
        "inv_validation_episode_recall": inv["validation_attack_metrics"]["episode_recall"],
        "inv_same_seed_test_episode_recall": inv["same_seed_heldout_test"]["episode_recall"],
        "inv_cross_seed_episode_recall": inv["cross_seed_frozen_threshold_test"]["episode_recall"],
        "inv_cross_seed_normal_fpr": inv["cross_seed_frozen_threshold_test"]["negative_window_fpr_direct"],
        "overall_target_fpr": float(args.overall_target_fpr),
        "channel_target_fpr": channel_fpr,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
