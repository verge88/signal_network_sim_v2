#!/usr/bin/env python3
"""Evaluate Diameter Isolation Forest on independently replayed S6a attacks.

The PCAPs consumed here are generated at CI time by a pinned third-party
Diameter implementation over a real TCP loopback connection and captured by
tcpdump.  The attack driver lives in this repository only as scenario
orchestration; Diameter encoding, peer state, HSS processing and answers are
provided by KellyKinyama/diameter-protocol at a pinned upstream commit.

This is a reproduced-lab external attack validation, not production-operator
attack telemetry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from diameter_external_pcap_validation import (
    PORTABLE_FEATURES,
    apply_robust_feature_transform,
    build_training_frame,
    pcap_to_portable_features,
    train_portable_models,
)


def score_capture(
    path: Path,
    rf,
    iso_legacy,
    legacy_threshold: float,
    iso_robust,
    robust_threshold: float,
    robust_transform: Dict[str, object],
) -> Dict[str, float]:
    feat = pcap_to_portable_features(path)
    raw_row = {k: feat[k] for k in PORTABLE_FEATURES}
    raw_row.update(
        {
            "obs_total_messages": feat["_decoded_pdus"],
            "obs_s6a_count": feat["_s6a_pdus"],
            "obs_gx_count": feat["_gx_pdus"],
        }
    )
    frame = pd.DataFrame([raw_row])
    X_raw = frame[PORTABLE_FEATURES]
    X_robust = apply_robust_feature_transform(frame, robust_transform)

    rf_score = float(rf.predict_proba(X_raw)[:, 1][0])
    legacy_score = float(-iso_legacy.score_samples(X_raw)[0])
    robust_score = float(-iso_robust.score_samples(X_robust)[0])

    return {
        "decoded_pdus": int(feat["_decoded_pdus"]),
        "s6a_pdus": int(feat["_s6a_pdus"]),
        "gx_pdus": int(feat["_gx_pdus"]),
        "rx_pdus": int(feat["_rx_pdus"]),
        "air_ratio": float(feat.get("obs_ratio_s6a_air", 0.0)),
        "ulr_ratio": float(feat.get("obs_ratio_s6a_ulr", 0.0)),
        "s6a_dominance": float(feat.get("obs_s6a_dominance_ratio", 0.0)),
        "rf_anomaly_score": rf_score,
        "rf_pred_anomaly": int(rf_score >= 0.5),
        "iso_legacy_anomaly_score": legacy_score,
        "iso_legacy_pred_anomaly": int(legacy_score >= legacy_threshold),
        "iso_robust_anomaly_score": robust_score,
        "iso_robust_pred_anomaly": int(robust_score >= robust_threshold),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pcap-dir", type=Path, required=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results_external_diameter_attacks"),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    required = {"capture", "expected_label", "scenario"}
    missing = required.difference(manifest.columns)
    if missing:
        raise SystemExit(f"manifest missing columns: {sorted(missing)}")
    if not set(manifest["expected_label"]).issubset({"benign", "attack"}):
        raise SystemExit("expected_label must contain only benign/attack")

    train_df = build_training_frame(args.seed)
    (
        rf,
        iso_legacy,
        legacy_threshold,
        iso_robust,
        robust_threshold,
        robust_transform,
        train_metrics,
    ) = train_portable_models(train_df, args.seed)

    rows: List[Dict[str, object]] = []
    for rec in manifest.to_dict("records"):
        path = args.pcap_dir / str(rec["capture"])
        if not path.exists():
            raise SystemExit(f"missing capture: {path}")
        scored = score_capture(
            path,
            rf,
            iso_legacy,
            legacy_threshold,
            iso_robust,
            robust_threshold,
            robust_transform,
        )
        rows.append({**rec, **scored})

    out = pd.DataFrame(rows)
    y = (out["expected_label"] == "attack").astype(int).to_numpy()

    summary: Dict[str, object] = {
        "scope": (
            "independent third-party S6a implementation, real TCP/tcpdump "
            "lab replay; not production operator telemetry"
        ),
        "capture_count": int(len(out)),
        "attack_capture_count": int(y.sum()),
        "benign_control_count": int((1 - y).sum()),
        "portable_feature_count": len(PORTABLE_FEATURES),
        "legacy_threshold": float(legacy_threshold),
        "robust_threshold": float(robust_threshold),
    }

    for prefix, col in [
        ("rf", "rf_pred_anomaly"),
        ("iso_legacy", "iso_legacy_pred_anomaly"),
        ("iso_robust", "iso_robust_pred_anomaly"),
    ]:
        pred = out[col].astype(int).to_numpy()
        summary[f"{prefix}_external_attack_tpr"] = float(
            recall_score(y, pred, zero_division=0)
        )
        summary[f"{prefix}_external_benign_fpr"] = float(
            pred[y == 0].mean() if np.any(y == 0) else np.nan
        )
        summary[f"{prefix}_external_f1"] = float(
            f1_score(y, pred, zero_division=0)
        )
        summary[f"{prefix}_external_balanced_accuracy"] = float(
            balanced_accuracy_score(y, pred)
        )

    scenario_metrics: Dict[str, object] = {}
    for scenario, grp in out.groupby("scenario"):
        gy = (grp["expected_label"] == "attack").astype(int).to_numpy()
        entry: Dict[str, object] = {
            "captures": int(len(grp)),
            "expected_label": str(grp["expected_label"].iloc[0]),
            "decoded_pdus": int(grp["decoded_pdus"].sum()),
        }
        for prefix, col in [
            ("rf", "rf_pred_anomaly"),
            ("iso_legacy", "iso_legacy_pred_anomaly"),
            ("iso_robust", "iso_robust_pred_anomaly"),
        ]:
            gp = grp[col].astype(int).to_numpy()
            if gy[0] == 1:
                entry[f"{prefix}_tpr"] = float(gp.mean())
            else:
                entry[f"{prefix}_fpr"] = float(gp.mean())
        scenario_metrics[str(scenario)] = entry
    summary["per_scenario"] = scenario_metrics

    # Retain relevant internal metrics so an external recall gain cannot hide
    # a collapse in the simulator holdout performance.
    for key in [
        "iso_legacy_internal_auc",
        "iso_legacy_internal_f1",
        "iso_robust_internal_auc",
        "iso_robust_internal_f1",
        "iso_legacy_internal_benign_fpr",
        "iso_robust_internal_benign_fpr",
    ]:
        summary[key] = train_metrics[key]

    out.to_csv(args.out_dir / "external_attack_predictions.csv", index=False)
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== Independent Diameter reproduced-attack validation ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nPer-capture predictions:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
