#!/usr/bin/env python3
"""Independent Diameter validation on external wire captures.

This script deliberately separates two questions:
1. Does the simulator-trained Diameter traffic model retain discrimination when
   restricted to features that can be reconstructed from an ordinary PCAP?
2. Does that portable feature branch falsely flag independent, benign lab wire
   captures produced by a real freeDiameter/Open5GS peer?

The full master-slave model also uses CU, zone and temporal features that are not
present in a standalone PCAP; those features are therefore NOT fabricated here.
The resulting result is a domain-transfer validation of the PCAP-observable
feature branch, not a claim of full external validation of every model feature.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diameter.diameter_sim_v1 import (
    AttackScenario,
    AttackType,
    DiameterSimulator,
    NormalEventType,
    NormalScenario,
    SimulationConfig,
    add_zone_deviation_features,
)

S6A_COMMANDS = ["AIR", "ULR", "CLR", "IDR", "DSR", "PUR", "RSR", "NOR"]
S6A_CODES = {
    318: "AIR", 316: "ULR", 317: "CLR", 319: "IDR",
    320: "DSR", 321: "PUR", 322: "RSR", 323: "NOR",
}
GX_COMMANDS = ["CCR_I", "CCR_U", "CCR_T", "RAR"]

APP_S6A = 16777251
APP_GX = 16777238
APP_RX = 16777236
APP_S13 = 16777252

PORTABLE_FEATURES = [
    "obs_entropy",
    "obs_s6a_dominance_ratio",
    "obs_s6a_top2_ratio",
    "obs_s6a_gini",
] + [f"obs_ratio_s6a_{c.lower()}" for c in S6A_COMMANDS] + [
    f"obs_ratio_gx_{c.lower()}" for c in GX_COMMANDS
]

FULL_FEATURES = [
    "cu_delivered", "cu_rtt_delay_ms", "cu_req_delay_ms",
    "cu_resp_delay_ms", "cu_combined_loss_prob", "integrity_check",
    "consecutive_cu_losses", "cu_processing_delay_ms",
    "cu_response_time_jitter", "cu_staleness", "cu_consistency_error",
    "obs_total_messages", "obs_s6a_count", "obs_gx_count", "obs_rx_count",
    "obs_s13_count", "obs_other_count", "obs_entropy",
    "obs_inbound_outbound_ratio", "obs_international_fraction",
    "obs_n_unique_peers", "obs_s6a_dominance_ratio", "obs_s6a_top2_ratio",
    "obs_s6a_gini",
] + [f"obs_ratio_s6a_{c.lower()}" for c in S6A_COMMANDS] + [
    f"obs_ratio_gx_{c.lower()}" for c in GX_COMMANDS
] + [
    "z_obs_total_messages", "z_obs_s6a_count",
    "z_obs_s6a_dominance_ratio", "z_obs_ratio_s6a_ulr",
    "z_obs_ratio_s6a_idr", "dev_total_messages_vs_zone",
    "dev_s6a_count_vs_zone", "dev_gx_count_vs_zone", "zone_io_ratio_rank",
    "dev_io_ratio_vs_zone", "dev_integrity_fail_rate_vs_zone",
    "dev_var_total_vs_zone", "var_obs_total_messages", "var_obs_entropy",
    "var_obs_inbound_outbound_ratio", "var_obs_s6a_dominance_ratio",
    "autocorr_obs_total_messages", "integrity_fail_rate", "obs_peers_cv",
    "node_type_enc",
]


def _entropy(counts: Iterable[int]) -> float:
    vals = np.asarray([x for x in counts if x > 0], dtype=float)
    if vals.size == 0:
        return 0.0
    p = vals / vals.sum()
    return float(-(p * np.log2(p)).sum())


def _gini(values: Iterable[int]) -> float:
    x = np.sort(np.asarray(list(values), dtype=float))
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    n = x.size
    return float((2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n)


def build_training_frame(seed: int = 42) -> pd.DataFrame:
    config = SimulationConfig(
        duration_hours=8.0,
        interval_s=15.0,
        n_mme=4, n_hss=2, n_pcrf=2, n_sgw=2, n_pgw=2,
        n_dra=3, n_dea=2, master_node_ids=[0, 1], seed=seed,
    )
    attacks = [
        AttackScenario(AttackType.SUBSCRIBER_LOCATION_TRACKING, [7], 180, 250, 6.0),
        AttackScenario(AttackType.SMS_DATA_INTERCEPTION, [8, 9], 360, 450, 7.0),
        AttackScenario(AttackType.SIGNALING_DOS, [13, 14], 570, 650, 8.0),
        AttackScenario(AttackType.FRAUD_PROFILE_MANIPULATION, [11], 760, 860, 7.0),
        AttackScenario(AttackType.NODE_COMPROMISE, [4], 950, 1350, 5.0),
        AttackScenario(AttackType.NODE_COMPROMISE, [9, 10], 1450, 1850, 6.0),
    ]
    normal_events = [
        NormalScenario(NormalEventType.TRAFFIC_SPIKE, [7, 8], 80, 130, 5.0),
        NormalScenario(NormalEventType.MAINTENANCE, [6], 700, 760, 3.0),
        NormalScenario(NormalEventType.REROUTE, [13, 14], 1250, 1310, 3.0),
        NormalScenario(NormalEventType.ROAMING_SURGE, [3, 4], 1050, 1120, 5.0),
    ]
    sim = DiameterSimulator(config, attacks, normal_events)
    sim.run()
    df = add_zone_deviation_features(sim.to_dataframe())
    missing = [c for c in PORTABLE_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"Simulator did not produce portable features: {missing}")
    return df



def _positive_median(values: pd.Series, fallback: float = 10.0) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return float(fallback)
    return float(np.clip(np.median(arr), 5.0, 50.0))


def fit_robust_feature_transform(train_rows: pd.DataFrame) -> Dict[str, object]:
    """Fit an availability-aware transform on benign TRAINING rows only.

    Ratio features estimated from a small number of protocol messages are noisy.
    A missing S6a/Gx family is also not evidence of an attack.  We therefore
    shrink each family-specific ratio towards a benign empirical prior with a
    pseudo-count determined only from the benign training distribution.

    The transform deliberately does not inspect external PCAP labels or scores.
    """
    normal = train_rows[train_rows["is_anomaly"].astype(int) == 0].copy()
    if normal.empty:
        raise RuntimeError("No benign rows available for robust IF transform")

    priors = {
        c: float(np.nanmedian(
            normal[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        ))
        for c in PORTABLE_FEATURES
    }

    def weighted_prior(ratio_col: str, count_col: str) -> float:
        p = normal[ratio_col].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        n = pd.to_numeric(normal[count_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        if n.sum() <= 0:
            return float(np.median(p))
        return float(np.sum(p * n) / np.sum(n))

    total_n = pd.to_numeric(normal["obs_total_messages"], errors="coerce").fillna(0.0).clip(lower=0.0)
    s6a_n = pd.to_numeric(normal["obs_s6a_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    gx_n = pd.to_numeric(normal["obs_gx_count"], errors="coerce").fillna(0.0).clip(lower=0.0)

    total_sum = float(total_n.sum())
    if total_sum > 0:
        priors["obs_s6a_dominance_ratio"] = float(
            np.sum(
                normal["obs_s6a_dominance_ratio"].fillna(0.0).to_numpy(dtype=float)
                * total_n.to_numpy(dtype=float)
            ) / total_sum
        )

    for c in S6A_COMMANDS:
        col = f"obs_ratio_s6a_{c.lower()}"
        priors[col] = weighted_prior(col, "obs_s6a_count")
    for c in GX_COMMANDS:
        col = f"obs_ratio_gx_{c.lower()}"
        priors[col] = weighted_prior(col, "obs_gx_count")

    params: Dict[str, object] = {
        "priors": priors,
        "alpha_total": _positive_median(total_n, 20.0),
        "alpha_s6a": _positive_median(s6a_n, 20.0),
        "alpha_gx": _positive_median(gx_n, 20.0),
    }

    # Learn very wide winsorization limits from benign TRAINING data only.
    # This protects the forest from representation artifacts while preserving
    # almost all genuine internal extremes.
    transformed = apply_robust_feature_transform(normal, params, clip=False)
    params["clip_low"] = {
        c: float(transformed[c].quantile(0.001)) for c in PORTABLE_FEATURES
    }
    params["clip_high"] = {
        c: float(transformed[c].quantile(0.999)) for c in PORTABLE_FEATURES
    }
    return params


def apply_robust_feature_transform(
    frame: pd.DataFrame,
    params: Dict[str, object],
    clip: bool = True,
) -> pd.DataFrame:
    X = frame[PORTABLE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
    priors = params["priors"]

    def count_series(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
        return pd.to_numeric(frame[name], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)

    total_n = count_series("obs_total_messages")
    s6a_n = count_series("obs_s6a_count")
    gx_n = count_series("obs_gx_count")

    def shrink(col: str, n: pd.Series, alpha: float) -> None:
        prior = float(priors[col])
        w = n / (n + float(alpha))
        X[col] = prior + w.to_numpy(dtype=float) * (X[col].to_numpy(dtype=float) - prior)

    shrink("obs_entropy", total_n, float(params["alpha_total"]))
    shrink("obs_s6a_dominance_ratio", total_n, float(params["alpha_total"]))
    shrink("obs_s6a_top2_ratio", s6a_n, float(params["alpha_s6a"]))
    shrink("obs_s6a_gini", s6a_n, float(params["alpha_s6a"]))

    for c in S6A_COMMANDS:
        shrink(f"obs_ratio_s6a_{c.lower()}", s6a_n, float(params["alpha_s6a"]))
    for c in GX_COMMANDS:
        shrink(f"obs_ratio_gx_{c.lower()}", gx_n, float(params["alpha_gx"]))

    if clip and "clip_low" in params and "clip_high" in params:
        low = params["clip_low"]
        high = params["clip_high"]
        for c in PORTABLE_FEATURES:
            lo, hi = float(low[c]), float(high[c])
            if hi < lo:
                lo, hi = hi, lo
            X[c] = X[c].clip(lower=lo, upper=hi)
    return X


def train_portable_models(df: pd.DataFrame, seed: int = 42):
    y = df["is_anomaly"].astype(int).to_numpy()
    all_idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.40, stratify=y, random_state=seed
    )
    calib_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=seed + 1
    )

    train_rows = df.iloc[train_idx].copy()
    calib_rows = df.iloc[calib_idx].copy()
    test_rows = df.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_calib = y[calib_idx]
    y_test = y[test_idx]

    X_train_raw = train_rows[PORTABLE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_calib_raw = calib_rows[PORTABLE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test_raw = test_rows[PORTABLE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    rf.fit(X_train_raw, y_train)
    rf_score = rf.predict_proba(X_test_raw)[:, 1]
    rf_pred = (rf_score >= 0.5).astype(int)

    # Legacy Isolation Forest: retained as a baseline for the robustness study.
    normal_train_raw = X_train_raw[y_train == 0]
    iso_legacy = IsolationForest(
        n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1
    )
    iso_legacy.fit(normal_train_raw)
    legacy_calib_scores = -iso_legacy.score_samples(X_calib_raw)
    legacy_threshold = float(np.percentile(legacy_calib_scores[y_calib == 0], 95.0))
    legacy_test_scores = -iso_legacy.score_samples(X_test_raw)
    legacy_pred = (legacy_test_scores >= legacy_threshold).astype(int)

    # Robust Isolation Forest:
    # 1) availability-aware empirical-Bayes shrinkage;
    # 2) benign-train-only wide winsorization;
    # 3) random feature subspaces + bootstrap;
    # 4) threshold calibration on a disjoint benign calibration partition.
    robust_transform = fit_robust_feature_transform(train_rows)
    X_train_robust = apply_robust_feature_transform(train_rows, robust_transform)
    X_calib_robust = apply_robust_feature_transform(calib_rows, robust_transform)
    X_test_robust = apply_robust_feature_transform(test_rows, robust_transform)

    iso_robust = IsolationForest(
        n_estimators=500,
        contamination="auto",
        max_features=0.75,
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    )
    iso_robust.fit(X_train_robust[y_train == 0])
    robust_calib_scores = -iso_robust.score_samples(X_calib_robust)
    robust_threshold = float(np.percentile(robust_calib_scores[y_calib == 0], 95.0))
    robust_test_scores = -iso_robust.score_samples(X_test_robust)
    robust_pred = (robust_test_scores >= robust_threshold).astype(int)

    benign_test = y_test == 0
    metrics = {
        "internal_rows": int(len(df)),
        "internal_anomaly_rate": float(y.mean()),
        "split_strategy": "60/20/20 stratified train/calibration/test",
        "portable_feature_count": len(PORTABLE_FEATURES),
        "full_feature_count": len(FULL_FEATURES),
        "feature_coverage_fraction": len(PORTABLE_FEATURES) / len(FULL_FEATURES),
        "rf_internal_auc": float(roc_auc_score(y_test, rf_score)),
        "rf_internal_f1": float(f1_score(y_test, rf_pred)),
        "iso_legacy_internal_auc": float(roc_auc_score(y_test, legacy_test_scores)),
        "iso_legacy_internal_f1": float(f1_score(y_test, legacy_pred)),
        "iso_legacy_internal_benign_fpr": float(legacy_pred[benign_test].mean()),
        "iso_legacy_threshold": legacy_threshold,
        "iso_robust_internal_auc": float(roc_auc_score(y_test, robust_test_scores)),
        "iso_robust_internal_f1": float(f1_score(y_test, robust_pred)),
        "iso_robust_internal_benign_fpr": float(robust_pred[benign_test].mean()),
        "iso_robust_threshold": robust_threshold,
        "iso_internal_benign_fpr": float(robust_pred[benign_test].mean()),
        "iso_threshold": robust_threshold,
        "robust_alpha_total": float(robust_transform["alpha_total"]),
        "robust_alpha_s6a": float(robust_transform["alpha_s6a"]),
        "robust_alpha_gx": float(robust_transform["alpha_gx"]),
        "robustness_method": (
            "benign-train-only empirical-Bayes availability smoothing + "
            "0.1/99.9% winsorization + feature-subsampled bootstrap IsolationForest + "
            "disjoint benign calibration"
        ),
    }
    return (
        rf,
        iso_legacy,
        legacy_threshold,
        iso_robust,
        robust_threshold,
        robust_transform,
        metrics,
    )


def evaluate_full_feature_isolation_forest(
    df: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate legacy vs robust IF on the complete 56-feature simulator view.

    This is an internal sensitivity guardrail for the robustness change.  It
    does not turn the external PCAP test into a full-feature validation.
    """
    missing = [c for c in FULL_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing full-model features: {missing}")

    y = df["is_anomaly"].astype(int).to_numpy()
    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.40, stratify=y, random_state=seed
    )
    calib_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=seed + 1
    )

    train_rows = df.iloc[train_idx]
    calib_rows = df.iloc[calib_idx]
    test_rows = df.iloc[test_idx]
    y_train, y_calib, y_test = y[train_idx], y[calib_idx], y[test_idx]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        imputer.fit_transform(
            train_rows[FULL_FEATURES].replace([np.inf, -np.inf], np.nan)
        )
    )
    X_calib = scaler.transform(
        imputer.transform(
            calib_rows[FULL_FEATURES].replace([np.inf, -np.inf], np.nan)
        )
    )
    X_test = scaler.transform(
        imputer.transform(
            test_rows[FULL_FEATURES].replace([np.inf, -np.inf], np.nan)
        )
    )

    normal_train = X_train[y_train == 0]
    legacy = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    ).fit(normal_train)
    legacy_calib = -legacy.score_samples(X_calib)
    legacy_threshold = float(np.percentile(legacy_calib[y_calib == 0], 95.0))
    legacy_test = -legacy.score_samples(X_test)
    legacy_pred = (legacy_test >= legacy_threshold).astype(int)

    low = np.nanquantile(normal_train, 0.001, axis=0)
    high = np.nanquantile(normal_train, 0.999, axis=0)

    def winsor(X):
        return np.minimum(np.maximum(X, low), high)

    robust = IsolationForest(
        n_estimators=500,
        contamination="auto",
        max_features=0.75,
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    ).fit(winsor(normal_train))
    robust_calib = -robust.score_samples(winsor(X_calib))
    robust_threshold = float(np.percentile(robust_calib[y_calib == 0], 95.0))
    robust_test = -robust.score_samples(winsor(X_test))
    robust_pred = (robust_test >= robust_threshold).astype(int)

    benign = y_test == 0
    return {
        "full_iso_legacy_internal_auc": float(roc_auc_score(y_test, legacy_test)),
        "full_iso_legacy_internal_f1": float(f1_score(y_test, legacy_pred)),
        "full_iso_legacy_internal_benign_fpr": float(legacy_pred[benign].mean()),
        "full_iso_legacy_threshold": legacy_threshold,
        "full_iso_robust_internal_auc": float(roc_auc_score(y_test, robust_test)),
        "full_iso_robust_internal_f1": float(f1_score(y_test, robust_pred)),
        "full_iso_robust_internal_benign_fpr": float(robust_pred[benign].mean()),
        "full_iso_robust_threshold": robust_threshold,
    }

def _run_tshark(path: Path) -> List[Dict[str, str]]:
    # Ubuntu 26.04 confines tshark with AppArmor. A user-owned copy in /tmp
    # is readable while an identical file under the GitHub workspace may not be.
    with tempfile.TemporaryDirectory(prefix="diameter-pcap-", dir="/tmp") as td:
        safe_path = Path(td) / path.name
        shutil.copy2(path, safe_path)
        cmd = [
            "tshark", "-r", str(safe_path), "-Y", "diameter", "-T", "fields",
            "-E", "occurrence=f",
            "-e", "frame.time_epoch", "-e", "diameter.cmd.code",
            "-e", "diameter.applicationId", "-e", "diameter.CC-Request-Type",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        proc.check_returncode()
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        parts += [""] * (4 - len(parts))
        if not parts[1].strip():
            continue
        rows.append({
            "time": parts[0].strip(),
            "cmd": parts[1].strip(),
            "app": parts[2].strip(),
            "cc_type": parts[3].strip(),
        })
    return rows


def pcap_to_portable_features(path: Path) -> Dict[str, float]:
    rows = _run_tshark(path)
    if not rows:
        raise RuntimeError(f"No Diameter PDUs decoded from {path}")

    app_counts = Counter()
    command_counts = Counter()
    s6a_counts = Counter({c: 0 for c in S6A_COMMANDS})
    gx_counts = Counter({c: 0 for c in GX_COMMANDS})

    for row in rows:
        try:
            code = int(row["cmd"].split(",")[0], 0)
        except ValueError:
            continue
        try:
            app = int(row["app"].split(",")[0], 0) if row["app"] else 0
        except ValueError:
            app = 0
        app_counts[app] += 1
        command_counts[(app, code)] += 1

        if app == APP_S6A and code in S6A_CODES:
            s6a_counts[S6A_CODES[code]] += 1
        if app == APP_GX:
            if code == 258:
                gx_counts["RAR"] += 1
            elif code == 272:
                try:
                    typ = int(row["cc_type"].split(",")[0], 0)
                except ValueError:
                    typ = 0
                gx_counts[{1: "CCR_I", 2: "CCR_U", 3: "CCR_T"}.get(typ, "CCR_I")] += 1

    total = max(1, sum(app_counts.values()))
    s6a_total = sum(s6a_counts.values())
    gx_total = sum(gx_counts.values())
    s6a_vals = [s6a_counts[c] for c in S6A_COMMANDS]
    top2 = sum(sorted(s6a_vals, reverse=True)[:2])

    out: Dict[str, float] = {
        "obs_entropy": _entropy(command_counts.values()),
        "obs_s6a_dominance_ratio": app_counts[APP_S6A] / total,
        "obs_s6a_top2_ratio": (top2 / s6a_total) if s6a_total else 0.0,
        "obs_s6a_gini": _gini(s6a_vals),
    }
    for c in S6A_COMMANDS:
        out[f"obs_ratio_s6a_{c.lower()}"] = s6a_counts[c] / s6a_total if s6a_total else 0.0
    for c in GX_COMMANDS:
        out[f"obs_ratio_gx_{c.lower()}"] = gx_counts[c] / gx_total if gx_total else 0.0

    out["_decoded_pdus"] = float(sum(app_counts.values()))
    out["_s6a_pdus"] = float(app_counts[APP_S6A])
    out["_gx_pdus"] = float(app_counts[APP_GX])
    out["_rx_pdus"] = float(app_counts[APP_RX])
    out["_s13_pdus"] = float(app_counts[APP_S13])
    return out



def evaluate_external(
    pcaps: List[Path],
    rf,
    iso_legacy,
    legacy_threshold: float,
    iso_robust,
    robust_threshold: float,
    robust_transform: Dict[str, object],
) -> pd.DataFrame:
    records = []
    for pcap in pcaps:
        feat = pcap_to_portable_features(pcap)
        raw_row = {k: feat[k] for k in PORTABLE_FEATURES}
        raw_row.update({
            "obs_total_messages": feat["_decoded_pdus"],
            "obs_s6a_count": feat["_s6a_pdus"],
            "obs_gx_count": feat["_gx_pdus"],
        })
        frame = pd.DataFrame([raw_row])
        X_raw = frame[PORTABLE_FEATURES]
        X_robust = apply_robust_feature_transform(frame, robust_transform)

        rf_score = float(rf.predict_proba(X_raw)[:, 1][0])
        legacy_score = float(-iso_legacy.score_samples(X_raw)[0])
        robust_score = float(-iso_robust.score_samples(X_robust)[0])

        records.append({
            "capture": pcap.name,
            "expected_label": "benign_wire_capture",
            "decoded_pdus": int(feat["_decoded_pdus"]),
            "s6a_pdus": int(feat["_s6a_pdus"]),
            "gx_pdus": int(feat["_gx_pdus"]),
            "rx_pdus": int(feat["_rx_pdus"]),
            "rf_anomaly_score": rf_score,
            "rf_pred_anomaly": int(rf_score >= 0.5),
            "iso_legacy_anomaly_score": legacy_score,
            "iso_legacy_pred_anomaly": int(legacy_score >= legacy_threshold),
            "iso_robust_anomaly_score": robust_score,
            "iso_robust_pred_anomaly": int(robust_score >= robust_threshold),
        })
    return pd.DataFrame(records)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pcaps", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("results_external_diameter"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = build_training_frame(args.seed)
    full_iso_metrics = evaluate_full_feature_isolation_forest(train_df, args.seed)
    (
        rf,
        iso_legacy,
        legacy_threshold,
        iso_robust,
        robust_threshold,
        robust_transform,
        metrics,
    ) = train_portable_models(train_df, args.seed)
    ext = evaluate_external(
        args.pcaps,
        rf,
        iso_legacy,
        legacy_threshold,
        iso_robust,
        robust_threshold,
        robust_transform,
    )

    metrics.update(full_iso_metrics)
    metrics.update({
        "external_capture_count": int(len(ext)),
        "external_total_pdus": int(ext["decoded_pdus"].sum()),
        "rf_external_benign_fpr": float(ext["rf_pred_anomaly"].mean()),
        "iso_legacy_external_benign_fpr": float(ext["iso_legacy_pred_anomaly"].mean()),
        "iso_robust_external_benign_fpr": float(ext["iso_robust_pred_anomaly"].mean()),
        "iso_external_benign_fpr": float(ext["iso_robust_pred_anomaly"].mean()),
        "scope": "PCAP-observable feature branch only; CU/zone/temporal features are not fabricated",
    })

    ext.to_csv(args.out_dir / "external_capture_predictions.csv", index=False)
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("=== Independent Diameter external validation ===")
    print(json.dumps(metrics, indent=2))
    print("\nPer-capture predictions:")
    print(ext.to_string(index=False))

    if metrics["feature_coverage_fraction"] < 0.25:
        raise SystemExit("Portable feature coverage unexpectedly too small")


if __name__ == "__main__":
    main()
