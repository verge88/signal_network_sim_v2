#!/usr/bin/env python3
"""Calibrate PCAP-observable Diameter simulator parameters on independent traces.

Calibration data are intentionally separate from the KellyKinyama holdout set
used by the repository's external-validation workflows.  This script estimates
only quantities that short laboratory PCAPs can support defensibly:

* Diameter message-size distribution;
* request/answer balance;
* conditional S6a request command mixture.

Operator-scale node rates, topology, time-of-day profile, CU behaviour and
long-range temporal correlation remain literature/model priors because short
public traces cannot identify them.

S6a command probabilities use a Dirichlet empirical-Bayes update around the
simulator prior so absent commands in a small calibration corpus are not forced
to zero.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

APP_S6A = 16777251
S6A_CODES = {
    318: "AIR",
    316: "ULR",
    317: "CLR",
    319: "IDR",
    320: "DSR",
    321: "PUR",
    322: "RSR",
    323: "NOR",
}
S6A_COMMANDS = ["AIR", "ULR", "CLR", "IDR", "DSR", "PUR", "RSR", "NOR"]
PRIOR_S6A = {
    "AIR": 0.30,
    "ULR": 0.25,
    "CLR": 0.05,
    "IDR": 0.12,
    "DSR": 0.03,
    "PUR": 0.05,
    "RSR": 0.02,
    "NOR": 0.18,
}


def _first_int(value: str, default: int = 0) -> int:
    value = (value or "").split(",")[0].strip()
    if not value:
        return default
    try:
        return int(value, 0)
    except ValueError:
        return default


def _first_float(value: str) -> float:
    value = (value or "").split(",")[0].strip()
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _request_flag(value: str) -> int | None:
    value = (value or "").split(",")[0].strip().lower()
    if value in {"1", "true"}:
        return 1
    if value in {"0", "false"}:
        return 0
    return None


def _run_tshark(path: Path) -> List[Dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="diameter-cal-", dir="/tmp") as td:
        safe = Path(td) / path.name
        shutil.copy2(path, safe)
        cmd = [
            "tshark", "-r", str(safe), "-Y", "diameter", "-T", "fields",
            "-E", "occurrence=f",
            "-e", "frame.time_epoch",
            "-e", "frame.len",
            "-e", "diameter.length",
            "-e", "diameter.cmd.code",
            "-e", "diameter.applicationId",
            "-e", "diameter.flags.request",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        proc.check_returncode()

    rows: List[Dict[str, str]] = []
    for line in proc.stdout.splitlines():
        p = line.split("\t")
        p += [""] * (6 - len(p))
        if not p[3].strip():
            continue
        rows.append({
            "time": p[0].strip(),
            "frame_len": p[1].strip(),
            "diameter_len": p[2].strip(),
            "cmd": p[3].strip(),
            "app": p[4].strip(),
            "request": p[5].strip(),
        })
    return rows


def summarize_capture(path: Path) -> Dict[str, object]:
    rows = _run_tshark(path)
    if not rows:
        raise RuntimeError(f"No Diameter PDUs decoded from {path}")

    lengths: List[float] = []
    times: List[float] = []
    req = ans = 0
    s6a_req = Counter({c: 0 for c in S6A_COMMANDS})

    for row in rows:
        dlen = _first_float(row["diameter_len"])
        flen = _first_float(row["frame_len"])
        if np.isfinite(dlen) and dlen > 0:
            lengths.append(dlen)
        elif np.isfinite(flen) and flen > 54:
            lengths.append(max(20.0, flen - 54.0))

        t = _first_float(row["time"])
        if np.isfinite(t):
            times.append(t)

        is_req = _request_flag(row["request"])
        if is_req == 1:
            req += 1
        elif is_req == 0:
            ans += 1

        app = _first_int(row["app"])
        code = _first_int(row["cmd"])
        if app == APP_S6A and is_req == 1 and code in S6A_CODES:
            s6a_req[S6A_CODES[code]] += 1

    duration = max(times) - min(times) if len(times) >= 2 else 0.0
    io_ratio = req / max(1, ans)
    return {
        "capture": path.name,
        "decoded_pdus": len(rows),
        "duration_s": duration,
        "request_pdus": req,
        "answer_pdus": ans,
        "request_answer_ratio": io_ratio,
        "message_size_median": float(np.median(lengths)) if lengths else np.nan,
        "message_size_mean": float(np.mean(lengths)) if lengths else np.nan,
        **{f"s6a_req_{k.lower()}": int(v) for k, v in s6a_req.items()},
        "_lengths": lengths,
    }


def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(d.values()))
    if total <= 0:
        raise ValueError("Cannot normalize empty distribution")
    return {k: float(v / total) for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pcaps", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("diameter_calibration"))
    ap.add_argument(
        "--prior-strength",
        type=float,
        default=40.0,
        help="Dirichlet equivalent prior S6a request count",
    )
    ap.add_argument(
        "--source",
        default="Orange-OpenSource/diafuzzer",
        help="human-readable provenance string",
    )
    ap.add_argument("--source-commit", default="")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [summarize_capture(p) for p in args.pcaps]
    lengths = [
        x
        for rec in summaries
        for x in rec.pop("_lengths")
        if np.isfinite(x) and x > 0
    ]
    if not lengths:
        raise SystemExit("No Diameter message lengths available for calibration")

    # Robust message-size estimate.  Median is stable for small heterogeneous
    # protocol traces and is clipped only to a broad valid Diameter range.
    avg_message_size = int(round(float(np.clip(np.median(lengths), 100, 2000))))

    ratios = np.asarray(
        [r["request_answer_ratio"] for r in summaries if r["answer_pdus"] > 0],
        dtype=float,
    )
    io_center = float(np.clip(np.median(ratios) if ratios.size else 1.0, 0.8, 1.2))
    if ratios.size >= 2:
        mad = float(np.median(np.abs(ratios - np.median(ratios))))
        io_halfspan = float(np.clip(1.4826 * mad, 0.03, 0.20))
    else:
        io_halfspan = 0.10

    counts = Counter({c: 0 for c in S6A_COMMANDS})
    for rec in summaries:
        for cmd in S6A_COMMANDS:
            counts[cmd] += int(rec[f"s6a_req_{cmd.lower()}"])
    n_s6a = int(sum(counts.values()))
    alpha = float(max(0.0, args.prior_strength))
    posterior = {
        cmd: alpha * PRIOR_S6A[cmd] + counts[cmd] for cmd in S6A_COMMANDS
    }
    calibrated_s6a = _normalize(posterior)

    prior_vec = np.asarray([PRIOR_S6A[c] for c in S6A_COMMANDS], dtype=float)
    cal_vec = np.asarray([calibrated_s6a[c] for c in S6A_COMMANDS], dtype=float)
    empirical_vec = np.asarray([counts[c] for c in S6A_COMMANDS], dtype=float)
    if empirical_vec.sum() > 0:
        empirical_vec /= empirical_vec.sum()
        l1_prior_to_external = float(np.abs(prior_vec - empirical_vec).sum())
        l1_calibrated_to_external = float(np.abs(cal_vec - empirical_vec).sum())
    else:
        l1_prior_to_external = float("nan")
        l1_calibrated_to_external = float("nan")

    profile = {
        "schema_version": 1,
        "calibration_scope": "PCAP-observable parameters only",
        "source": args.source,
        "source_commit": args.source_commit,
        "capture_count": len(summaries),
        "decoded_pdu_count": int(sum(int(r["decoded_pdus"]) for r in summaries)),
        "avg_message_size_bytes": avg_message_size,
        "io_ratio_center": io_center,
        "io_ratio_halfspan": io_halfspan,
        "s6a_command_dist": calibrated_s6a,
        "s6a_request_evidence_count": n_s6a,
        "dirichlet_prior_strength": alpha,
        "prior_s6a_command_dist": PRIOR_S6A,
        "raw_s6a_request_counts": dict(counts),
        "fit_diagnostics": {
            "l1_prior_to_external_s6a": l1_prior_to_external,
            "l1_calibrated_to_external_s6a": l1_calibrated_to_external,
        },
        "not_calibrated_from_short_pcaps": [
            "node topology and node counts",
            "operator-scale base message rates",
            "time-of-day profile",
            "AR(1) phi and sigma",
            "CU delivery/integrity process",
            "attack intensities",
            "Gx command mixture (no representative Gx calibration capture)",
        ],
    }

    pd.DataFrame(summaries).to_csv(args.out_dir / "calibration_evidence.csv", index=False)
    with open(args.out_dir / "calibration_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print("=== Diameter independent-data calibration ===")
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    print("\nPer-capture evidence:")
    print(pd.DataFrame(summaries).to_string(index=False))

    if n_s6a <= 0:
        raise SystemExit("Calibration corpus contains no S6a request evidence")
    if not np.isclose(sum(calibrated_s6a.values()), 1.0, atol=1e-9):
        raise SystemExit("Calibrated S6a distribution is not normalized")
    if np.isfinite(l1_prior_to_external):
        if l1_calibrated_to_external > l1_prior_to_external + 1e-12:
            raise SystemExit("Calibration moved S6a distribution away from evidence")


if __name__ == "__main__":
    main()
