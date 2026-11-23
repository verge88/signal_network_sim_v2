#!/usr/bin/env python3
"""Verify that an externally calibrated profile propagated into a simulator dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

S6A_COMMANDS = ["AIR", "ULR", "CLR", "IDR", "DSR", "PUR", "RSR", "NOR"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    with open(args.profile, "r", encoding="utf-8") as f:
        profile = json.load(f)

    benign = df[df["is_anomaly"].astype(int) == 0].copy()
    if benign.empty:
        raise SystemExit("No benign simulator rows available")

    weights = pd.to_numeric(
        benign["obs_s6a_count"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0).to_numpy(float)

    simulated = {}
    for cmd in S6A_COMMANDS:
        col = f"obs_ratio_s6a_{cmd.lower()}"
        vals = pd.to_numeric(benign[col], errors="coerce").fillna(0.0).to_numpy(float)
        simulated[cmd] = float(np.average(vals, weights=weights)) if weights.sum() > 0 else 0.0

    sim_sum = sum(simulated.values())
    if sim_sum > 0:
        simulated = {k: v / sim_sum for k, v in simulated.items()}

    target = {k: float(profile["s6a_command_dist"][k]) for k in S6A_COMMANDS}
    l1 = float(sum(abs(simulated[k] - target[k]) for k in S6A_COMMANDS))

    io = pd.to_numeric(
        benign["obs_inbound_outbound_ratio"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()
    io_median = float(io.median())
    io_target = float(profile["io_ratio_center"])
    io_abs_error = abs(io_median - io_target)

    result = {
        "dataset_rows": int(len(df)),
        "benign_rows": int(len(benign)),
        "anomaly_rate": float(df["is_anomaly"].mean()),
        "feature_count": int(len(df.columns)),
        "target_s6a_command_dist": target,
        "simulated_benign_s6a_command_dist": simulated,
        "s6a_l1_distance": l1,
        "target_io_ratio_center": io_target,
        "simulated_benign_io_ratio_median": io_median,
        "io_ratio_abs_error": io_abs_error,
        "calibrated_avg_message_size_bytes": int(profile["avg_message_size_bytes"]),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("=== Calibrated Diameter dataset verification ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Loose propagation checks, not performance tuning gates.
    if l1 > 0.08:
        raise SystemExit(f"Calibrated S6a mix did not propagate: L1={l1:.4f}")
    if io_abs_error > max(0.08, float(profile["io_ratio_halfspan"])):
        raise SystemExit(
            f"Calibrated request/answer balance did not propagate: error={io_abs_error:.4f}"
        )


if __name__ == "__main__":
    main()
