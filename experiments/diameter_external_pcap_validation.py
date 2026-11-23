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


def train_portable_models(df: pd.DataFrame, seed: int = 42):
    X = df[PORTABLE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["is_anomaly"].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_score = rf.predict_proba(X_test)[:, 1]
    rf_pred = (rf_score >= 0.5).astype(int)

    normal_train = X_train[y_train == 0]
    iso = IsolationForest(
        n_estimators=300, contamination=0.05, random_state=seed, n_jobs=-1
    )
    iso.fit(normal_train)
    iso_test_scores = -iso.score_samples(X_test)
    benign_scores = iso_test_scores[y_test == 0]
    iso_threshold = float(np.percentile(benign_scores, 95.0))

    metrics = {
        "internal_rows": int(len(df)),
        "internal_anomaly_rate": float(y.mean()),
        "portable_feature_count": len(PORTABLE_FEATURES),
        "full_feature_count": len(FULL_FEATURES),
        "feature_coverage_fraction": len(PORTABLE_FEATURES) / len(FULL_FEATURES),
        "rf_internal_auc": float(roc_auc_score(y_test, rf_score)),
        "rf_internal_f1": float(f1_score(y_test, rf_pred)),
        "iso_internal_benign_fpr": float((benign_scores >= iso_threshold).mean()),
        "iso_threshold": iso_threshold,
    }
    return rf, iso, iso_threshold, metrics


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


def evaluate_external(pcaps: List[Path], rf, iso, iso_threshold: float) -> pd.DataFrame:
    records = []
    for pcap in pcaps:
        feat = pcap_to_portable_features(pcap)
        X = pd.DataFrame([{k: feat[k] for k in PORTABLE_FEATURES}])
        rf_score = float(rf.predict_proba(X)[:, 1][0])
        iso_score = float(-iso.score_samples(X)[0])
        records.append({
            "capture": pcap.name,
            "expected_label": "benign_wire_capture",
            "decoded_pdus": int(feat["_decoded_pdus"]),
            "s6a_pdus": int(feat["_s6a_pdus"]),
            "gx_pdus": int(feat["_gx_pdus"]),
            "rx_pdus": int(feat["_rx_pdus"]),
            "rf_anomaly_score": rf_score,
            "rf_pred_anomaly": int(rf_score >= 0.5),
            "iso_anomaly_score": iso_score,
            "iso_pred_anomaly": int(iso_score >= iso_threshold),
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
    rf, iso, iso_threshold, metrics = train_portable_models(train_df, args.seed)
    ext = evaluate_external(args.pcaps, rf, iso, iso_threshold)

    metrics.update({
        "external_capture_count": int(len(ext)),
        "external_total_pdus": int(ext["decoded_pdus"].sum()),
        "rf_external_benign_fpr": float(ext["rf_pred_anomaly"].mean()),
        "iso_external_benign_fpr": float(ext["iso_pred_anomaly"].mean()),
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
