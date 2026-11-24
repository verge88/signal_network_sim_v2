#!/usr/bin/env python3
"""PCAP-native Diameter detector with source-separated domain transfer.

This module implements the portable branch of the two-mode detector:
- the full 56-feature detector remains the preferred mode when CU/zone telemetry
  is available;
- this PCAP-native branch uses only features reconstructible from a wire
  capture and is trained with domain randomisation across independent simulator
  seeds.

Source separation:
  base fit       -> simulator seeds 42, 43
  meta fit       -> simulator seed 44
  threshold fit  -> simulator seed 45 + optional independent benign PCAPs
  internal test  -> simulator seed 46
  final external -> supplied separately; Kelly traces must never be passed to
                    the train command.

The script deliberately treats node_compromise as out of scope for the PCAP
branch because its defining evidence is CU/report inconsistency that is absent
from a standalone capture.  Node compromise remains the responsibility of the
full telemetry mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from diameter_external_pcap_validation import (  # noqa: E402
    APP_GX,
    APP_RX,
    APP_S13,
    APP_S6A,
    GX_COMMANDS,
    PORTABLE_FEATURES,
    S6A_CODES,
    S6A_COMMANDS,
    _run_tshark,
    build_training_frame,
)

SUPPORTED_PORTABLE_ATTACKS = {
    "location_tracking",
    "sms_data_interception",
    "signaling_dos",
    "fraud_profile",
}

EXTRA_FEATURES = [
    "pcap_log_total_pdus",
    "pcap_log_duration_s",
    "pcap_log_rate_hz",
    "pcap_s6a_fraction",
    "pcap_gx_fraction",
    "pcap_rx_fraction",
    "pcap_s13_fraction",
    "pcap_other_fraction",
    "pcap_iat_cv",
    "pcap_bin_count_cv",
    "pcap_bin_count_peak_ratio",
    "pcap_bin_count_autocorr",
    "pcap_bin_entropy_std",
    "pcap_bin_s6a_fraction_std",
    "pcap_air_ratio_std",
    "pcap_ulr_ratio_std",
    "pcap_active_command_fraction",
    "pcap_log_burst_mass",
    "pcap_log_pdus_per_active_command",
]
PCAP_NATIVE_FEATURES = PORTABLE_FEATURES + EXTRA_FEATURES

PROFILE_FEATURES = PORTABLE_FEATURES + [
    "pcap_s6a_fraction",
    "pcap_gx_fraction",
    "pcap_rx_fraction",
    "pcap_s13_fraction",
    "pcap_other_fraction",
    "pcap_active_command_fraction",
]
VOLUME_FEATURES = [
    "pcap_log_total_pdus",
    "pcap_log_duration_s",
    "pcap_log_rate_hz",
    "pcap_iat_cv",
    "pcap_bin_count_cv",
    "pcap_bin_count_peak_ratio",
    "pcap_bin_count_autocorr",
    "pcap_log_burst_mass",
    "pcap_log_pdus_per_active_command",
]
WIRE_FEATURES = PROFILE_FEATURES + [
    "pcap_log_total_pdus",
    "pcap_log_duration_s",
    "pcap_log_rate_hz",
    "pcap_log_burst_mass",
    "pcap_log_pdus_per_active_command",
]

TEMPORAL_FEATURES = [
    "pcap_iat_cv",
    "pcap_bin_count_cv",
    "pcap_bin_count_peak_ratio",
    "pcap_bin_count_autocorr",
    "pcap_bin_entropy_std",
    "pcap_bin_s6a_fraction_std",
    "pcap_air_ratio_std",
    "pcap_ulr_ratio_std",
]
META_RAW_FEATURES = [
    "pcap_log_total_pdus",
    "pcap_log_duration_s",
    "pcap_log_rate_hz",
    "pcap_log_burst_mass",
    "pcap_log_pdus_per_active_command",
    "pcap_active_command_fraction",
]


def _entropy(values: Iterable[float]) -> float:
    x = np.asarray([float(v) for v in values if float(v) > 0], dtype=float)
    if x.size == 0:
        return 0.0
    p = x / x.sum()
    return float(-(p * np.log2(p)).sum())


def _gini(values: Iterable[float]) -> float:
    x = np.sort(np.asarray(list(values), dtype=float))
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    n = x.size
    return float(
        2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())
        - (n + 1) / n
    )


def _cv(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    m = float(np.mean(x))
    return float(np.std(x) / (abs(m) + 1e-9))


def _autocorr(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    if x.size < 3 or np.std(x) <= 1e-9:
        return 0.0
    m = float(np.mean(x))
    den = float(np.sum((x - m) ** 2))
    if den <= 1e-12:
        return 0.0
    return float(np.sum((x[1:] - m) * (x[:-1] - m)) / den)


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _normalised_command_features(
    total: float,
    s6a_count: float,
    gx_count: float,
    rx_count: float,
    s13_count: float,
    other_count: float,
    s6a_cmd_counts: Dict[str, float],
    gx_cmd_counts: Dict[str, float],
) -> Dict[str, float]:
    total = max(float(total), 1.0)
    s6a_total = max(float(sum(s6a_cmd_counts.values())), 0.0)
    gx_total = max(float(sum(gx_cmd_counts.values())), 0.0)
    s6a_vals = [float(s6a_cmd_counts.get(c, 0.0)) for c in S6A_COMMANDS]
    gx_vals = [float(gx_cmd_counts.get(c, 0.0)) for c in GX_COMMANDS]
    top2 = sum(sorted(s6a_vals, reverse=True)[:2])

    # Match the simulator semantics exactly.  The legacy portable branch used
    # command entropy and S6a application fraction under these names, which
    # created a silent representation mismatch at domain transfer time.
    interface_counts = [
        float(s6a_count), float(gx_count), float(rx_count),
        float(s13_count), float(other_count),
    ]
    s6a_rest = max(s6a_total - max(s6a_vals, default=0.0), 1e-3)
    s6a_top2_rest = max(s6a_total - top2, 1e-3)

    out = {
        "obs_entropy": _entropy(interface_counts),
        "obs_s6a_dominance_ratio": (
            max(s6a_vals, default=0.0) / s6a_rest if s6a_total > 0 else 0.0
        ),
        "obs_s6a_top2_ratio": (
            top2 / s6a_top2_rest if s6a_total > 0 else 0.0
        ),
        "obs_s6a_gini": _gini(s6a_vals),
    }
    for c in S6A_COMMANDS:
        out[f"obs_ratio_s6a_{c.lower()}"] = (
            float(s6a_cmd_counts.get(c, 0.0)) / s6a_total
            if s6a_total > 0 else 0.0
        )
    for c in GX_COMMANDS:
        out[f"obs_ratio_gx_{c.lower()}"] = (
            float(gx_cmd_counts.get(c, 0.0)) / gx_total
            if gx_total > 0 else 0.0
        )
    return out


def _aggregate_sim_window(win: pd.DataFrame, interval_s: float = 15.0):
    attacks = set(str(x) for x in win["attack_type"].astype(str))
    if "node_compromise" in attacks:
        return None
    non_none = attacks - {"none"}
    if len(non_none) > 1:
        return None
    if non_none and not non_none.issubset(SUPPORTED_PORTABLE_ATTACKS):
        return None

    label = int(bool(non_none))
    attack_type = next(iter(non_none)) if non_none else "none"

    total_rows = pd.to_numeric(
        win["obs_total_messages"], errors="coerce").fillna(0.0).to_numpy(float)
    s6a_rows = pd.to_numeric(
        win["obs_s6a_count"], errors="coerce").fillna(0.0).to_numpy(float)
    gx_rows = pd.to_numeric(
        win["obs_gx_count"], errors="coerce").fillna(0.0).to_numpy(float)
    rx_rows = pd.to_numeric(
        win["obs_rx_count"], errors="coerce").fillna(0.0).to_numpy(float)
    s13_rows = pd.to_numeric(
        win["obs_s13_count"], errors="coerce").fillna(0.0).to_numpy(float)
    other_rows = pd.to_numeric(
        win["obs_other_count"], errors="coerce").fillna(0.0).to_numpy(float)

    total = float(total_rows.sum())
    s6a = float(s6a_rows.sum())
    gx = float(gx_rows.sum())
    rx = float(rx_rows.sum())
    s13 = float(s13_rows.sum())
    other = float(other_rows.sum())

    s6a_cmd_counts = {}
    for c in S6A_COMMANDS:
        ratios = pd.to_numeric(
            win[f"obs_ratio_s6a_{c.lower()}"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        s6a_cmd_counts[c] = float(np.sum(ratios * s6a_rows))
    gx_cmd_counts = {}
    for c in GX_COMMANDS:
        ratios = pd.to_numeric(
            win[f"obs_ratio_gx_{c.lower()}"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        gx_cmd_counts[c] = float(np.sum(ratios * gx_rows))

    out = _normalised_command_features(
        total, s6a, gx, rx, s13, other, s6a_cmd_counts, gx_cmd_counts)

    duration = max(interval_s * len(win), 1e-3)
    rate = total / duration
    mean_iat = duration / max(total, 1.0)

    row_ent = pd.to_numeric(
        win["obs_entropy"], errors="coerce").fillna(0.0).to_numpy(float)
    row_s6a_frac = s6a_rows / np.maximum(total_rows, 1.0)
    air_rows = pd.to_numeric(
        win["obs_ratio_s6a_air"], errors="coerce").fillna(0.0).to_numpy(float)
    ulr_rows = pd.to_numeric(
        win["obs_ratio_s6a_ulr"], errors="coerce").fillna(0.0).to_numpy(float)

    active = sum(v > 0.5 for v in s6a_cmd_counts.values()) + sum(
        v > 0.5 for v in gx_cmd_counts.values())

    out.update({
        "pcap_log_total_pdus": float(np.log1p(max(total, 0.0))),
        "pcap_log_duration_s": float(np.log1p(duration)),
        "pcap_log_rate_hz": float(np.log1p(max(rate, 0.0))),
        "pcap_s6a_fraction": s6a / max(total, 1.0),
        "pcap_gx_fraction": gx / max(total, 1.0),
        "pcap_rx_fraction": rx / max(total, 1.0),
        "pcap_s13_fraction": s13 / max(total, 1.0),
        "pcap_other_fraction": other / max(total, 1.0),
        # With aggregated 15 s bins we do not observe message-level IAT.
        # Count variation is used only as a weak proxy and is aggressively
        # domain-randomised below.
        "pcap_iat_cv": _cv(total_rows),
        "pcap_bin_count_cv": _cv(total_rows),
        "pcap_bin_count_peak_ratio": (
            float(np.max(total_rows) / (np.mean(total_rows) + 1e-9))
            if len(total_rows) else 0.0
        ),
        "pcap_bin_count_autocorr": _autocorr(total_rows),
        "pcap_bin_entropy_std": float(np.std(row_ent)),
        "pcap_bin_s6a_fraction_std": float(np.std(row_s6a_frac)),
        "pcap_air_ratio_std": float(np.std(air_rows)),
        "pcap_ulr_ratio_std": float(np.std(ulr_rows)),
        "pcap_active_command_fraction": float(active / (len(S6A_COMMANDS) + len(GX_COMMANDS))),
        "pcap_log_burst_mass": float(np.log1p(max(total * rate, 0.0))),
        "pcap_log_pdus_per_active_command": float(
            np.log1p(total / max(active, 1))),
        "is_anomaly": label,
        "attack_type": attack_type,
    })
    return out


def simulation_to_native_windows(
    df: pd.DataFrame,
    window_sizes: Sequence[int] = (2, 4, 8),
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for _, node_df in df.groupby("node_id", sort=False):
        node_df = node_df.sort_values("interval").reset_index(drop=True)
        for width in window_sizes:
            stride = max(width * 2, 1)
            for start in range(0, max(0, len(node_df) - width + 1), stride):
                win = node_df.iloc[start:start + width]
                rec = _aggregate_sim_window(win)
                if rec is not None:
                    records.append(rec)
    if not records:
        raise RuntimeError("No PCAP-native simulator windows were generated")
    return pd.DataFrame(records)


def domain_randomise(
    frame: pd.DataFrame,
    seed: int,
    variants: int = 4,
) -> pd.DataFrame:
    """Create domain variants without conditioning randomisation on labels."""
    rng = np.random.default_rng(seed)
    outs = []
    s6a_cols = [f"obs_ratio_s6a_{c.lower()}" for c in S6A_COMMANDS]
    gx_cols = [f"obs_ratio_gx_{c.lower()}" for c in GX_COMMANDS]

    for variant in range(variants):
        x = frame.copy()
        # One domain-level scale per variant preserves relative attack/benign
        # intensity while spanning tiny lab captures through network-scale load.
        traffic_scale = float(np.exp(rng.uniform(np.log(0.002), np.log(2.0))))
        time_scale = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))

        total = np.expm1(x["pcap_log_total_pdus"].to_numpy(float))
        duration = np.expm1(x["pcap_log_duration_s"].to_numpy(float))
        rate = np.expm1(x["pcap_log_rate_hz"].to_numpy(float))
        x["pcap_log_total_pdus"] = np.log1p(np.maximum(0.0, total * traffic_scale))
        x["pcap_log_duration_s"] = np.log1p(np.maximum(1e-3, duration * time_scale))
        scaled_total = np.maximum(0.0, total * traffic_scale)
        scaled_rate = np.maximum(
            0.0, rate * traffic_scale / time_scale)
        x["pcap_log_rate_hz"] = np.log1p(scaled_rate)
        x["pcap_log_burst_mass"] = np.log1p(
            np.maximum(0.0, scaled_total * scaled_rate))
        active_frac = np.maximum(
            x["pcap_active_command_fraction"].to_numpy(float), 1e-6)
        active_count = active_frac * (
            len(S6A_COMMANDS) + len(GX_COMMANDS) + 4)
        x["pcap_log_pdus_per_active_command"] = np.log1p(
            scaled_total / np.maximum(active_count, 1.0))

        # Small domain-level shifts in command mixtures emulate vendor and
        # deployment differences while keeping attack structure intact.
        for cols in (s6a_cols, gx_cols):
            arr = x[cols].to_numpy(float)
            bias = rng.normal(0.0, 0.025, size=(1, len(cols)))
            arr = np.clip(arr + bias, 0.0, None)
            den = arr.sum(axis=1, keepdims=True)
            arr = np.divide(arr, den, out=np.zeros_like(arr), where=den > 0)
            x.loc[:, cols] = arr

        for col in [
            "pcap_iat_cv", "pcap_bin_count_cv",
            "pcap_bin_count_peak_ratio", "pcap_bin_entropy_std",
            "pcap_bin_s6a_fraction_std", "pcap_air_ratio_std",
            "pcap_ulr_ratio_std",
        ]:
            mult = float(np.exp(rng.normal(0.0, 0.30)))
            x[col] = np.clip(x[col].to_numpy(float) * mult, 0.0, None)

        x["domain_variant"] = variant
        outs.append(x)
    return pd.concat(outs, ignore_index=True)


def _parse_raw_diameter_from_tcp(path: Path):
    """Fallback RFC 6733 header parser over TCP payload.

    Some Wireshark builds refuse to label deliberately minimal 20-byte
    third-party Diameter messages as Diameter even with decode-as.  The
    transport bytes are still valid RFC 6733 frames.  This fallback asks
    tshark only for TCP payload/reassembly metadata and parses the fixed
    Diameter header directly.  It changes neither model inputs nor thresholds.
    """
    with tempfile.TemporaryDirectory(prefix="diameter-raw-", dir="/tmp") as td:
        safe_path = Path(td) / path.name
        shutil.copy2(path, safe_path)
        cmd = [
            "tshark", "-r", str(safe_path),
            "-Y", "tcp.port == 3868 && tcp.payload",
            "-T", "fields", "-E", "occurrence=f",
            "-e", "frame.time_epoch",
            "-e", "tcp.stream",
            "-e", "ip.src", "-e", "tcp.srcport",
            "-e", "ip.dst", "-e", "tcp.dstport",
            "-e", "tcp.payload",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        proc.check_returncode()

    buffers: Dict[Tuple[str, str, str, str, str], bytearray] = {}
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        parts += [""] * (7 - len(parts))
        ts_s, stream, src, sport, dst, dport, payload_hex = (
            p.strip() for p in parts[:7])
        if not payload_hex:
            continue
        try:
            ts = float(ts_s)
            chunk = bytes.fromhex(payload_hex.replace(":", ""))
        except Exception:
            continue
        key = (stream, src, sport, dst, dport)
        buf = buffers.setdefault(key, bytearray())
        buf.extend(chunk)

        while len(buf) >= 20:
            # Diameter Version must be 1.  If a retransmission/odd byte gets
            # ahead of a frame, resynchronise conservatively on the next 0x01.
            if buf[0] != 1:
                try:
                    nxt = buf.index(1, 1)
                    del buf[:nxt]
                except ValueError:
                    buf.clear()
                    break
                if len(buf) < 20:
                    break

            length = int.from_bytes(buf[1:4], "big")
            if length < 20 or length > 1_048_576:
                del buf[0]
                continue
            if len(buf) < length:
                break

            frame = bytes(buf[:length])
            del buf[:length]
            code = int.from_bytes(frame[5:8], "big")
            app = int.from_bytes(frame[8:12], "big")
            # CC-Request-Type is an AVP and is absent from the frozen
            # aiodiameter harness.  Keep zero for fallback-only Gx decoding.
            rows.append((ts, app, code, 0))

    rows.sort(key=lambda x: x[0])
    return rows


def _parse_pcap_rows(path: Path):
    raw = _run_tshark(path)
    rows = []
    for rec in raw:
        try:
            ts = float(rec.get("time", ""))
        except Exception:
            continue
        try:
            code = int(str(rec.get("cmd", "")).split(",")[0], 0)
        except Exception:
            continue
        try:
            app = int(str(rec.get("app", "")).split(",")[0], 0) if rec.get("app") else 0
        except Exception:
            app = 0
        try:
            cc = int(str(rec.get("cc_type", "")).split(",")[0], 0) if rec.get("cc_type") else 0
        except Exception:
            cc = 0
        rows.append((ts, app, code, cc))

    if not rows:
        rows = _parse_raw_diameter_from_tcp(path)
    if not rows:
        raise RuntimeError(f"No Diameter PDUs decoded from {path}")
    rows.sort(key=lambda x: x[0])
    return rows


def pcap_to_native_features(path: Path) -> Dict[str, float]:
    rows = _parse_pcap_rows(path)
    times = np.asarray([r[0] for r in rows], dtype=float)
    duration = max(float(times[-1] - times[0]), 1e-3)
    total = len(rows)

    app_counts = {APP_S6A: 0, APP_GX: 0, APP_RX: 0, APP_S13: 0}
    s6a_counts = {c: 0.0 for c in S6A_COMMANDS}
    gx_counts = {c: 0.0 for c in GX_COMMANDS}
    command_pairs = set()

    for _, app, code, cc in rows:
        if app in app_counts:
            app_counts[app] += 1
        command_pairs.add((app, code, cc if app == APP_GX and code == 272 else 0))
        if app == APP_S6A and code in S6A_CODES:
            s6a_counts[S6A_CODES[code]] += 1.0
        if app == APP_GX:
            if code == 258:
                gx_counts["RAR"] += 1.0
            elif code == 272:
                gx_counts[{1: "CCR_I", 2: "CCR_U", 3: "CCR_T"}.get(cc, "CCR_I")] += 1.0

    # Four equal time bins preserve burst information and work on short traces.
    bin_counts = np.zeros(4, dtype=float)
    bin_entropy = np.zeros(4, dtype=float)
    bin_s6a_frac = np.zeros(4, dtype=float)
    bin_air = np.zeros(4, dtype=float)
    bin_ulr = np.zeros(4, dtype=float)

    if duration <= 1e-3:
        bin_ids = np.minimum(
            (np.arange(total) * 4 // max(total, 1)).astype(int), 3)
    else:
        rel = (times - times[0]) / duration
        bin_ids = np.minimum((rel * 4).astype(int), 3)

    for b in range(4):
        sub = [rows[i] for i in np.flatnonzero(bin_ids == b)]
        if not sub:
            continue
        bin_counts[b] = len(sub)
        pairs = {}
        s6a_b = 0
        air_b = 0
        ulr_b = 0
        for _, app, code, cc in sub:
            key = (app, code, cc if app == APP_GX and code == 272 else 0)
            pairs[key] = pairs.get(key, 0) + 1
            if app == APP_S6A:
                s6a_b += 1
                if code == 318:
                    air_b += 1
                elif code == 316:
                    ulr_b += 1
        bin_entropy[b] = _entropy(pairs.values())
        bin_s6a_frac[b] = s6a_b / len(sub)
        if s6a_b:
            bin_air[b] = air_b / s6a_b
            bin_ulr[b] = ulr_b / s6a_b

    iat = np.diff(times)
    iat = iat[np.isfinite(iat) & (iat >= 0)]
    iat_cv = _cv(iat) if len(iat) else 0.0

    known = sum(app_counts.values())
    other = max(0, total - known)

    # Recompute the legacy-named features with the exact simulator semantics.
    # This is intentionally local to the new PCAP-native branch; the robust IF
    # baseline keeps its frozen legacy transform for comparability.
    semantic = _normalised_command_features(
        total,
        app_counts[APP_S6A],
        app_counts[APP_GX],
        app_counts[APP_RX],
        app_counts[APP_S13],
        other,
        s6a_counts,
        gx_counts,
    )
    out = {k: float(semantic[k]) for k in PORTABLE_FEATURES}
    out.update({
        "pcap_log_total_pdus": float(np.log1p(total)),
        "pcap_log_duration_s": float(np.log1p(duration)),
        "pcap_log_rate_hz": float(np.log1p(total / duration)),
        "pcap_s6a_fraction": app_counts[APP_S6A] / total,
        "pcap_gx_fraction": app_counts[APP_GX] / total,
        "pcap_rx_fraction": app_counts[APP_RX] / total,
        "pcap_s13_fraction": app_counts[APP_S13] / total,
        "pcap_other_fraction": other / total,
        "pcap_iat_cv": float(iat_cv),
        "pcap_bin_count_cv": _cv(bin_counts),
        "pcap_bin_count_peak_ratio": float(
            np.max(bin_counts) / (np.mean(bin_counts) + 1e-9)),
        "pcap_bin_count_autocorr": _autocorr(bin_counts),
        "pcap_bin_entropy_std": float(np.std(bin_entropy)),
        "pcap_bin_s6a_fraction_std": float(np.std(bin_s6a_frac)),
        "pcap_air_ratio_std": float(np.std(bin_air)),
        "pcap_ulr_ratio_std": float(np.std(bin_ulr)),
        "pcap_active_command_fraction": float(
            len(command_pairs) / (len(S6A_COMMANDS) + len(GX_COMMANDS) + 4)
        ),
        "pcap_log_burst_mass": float(
            np.log1p(total * (total / duration))),
        "pcap_log_pdus_per_active_command": float(
            np.log1p(total / max(len(command_pairs), 1))),
        "_decoded_pdus": float(total),
        "_duration_s": float(duration),
    })
    return out


def protocol_archetype_frame(
    seed: int,
    n_benign: int = 1200,
    n_attack: int = 1200,
) -> pd.DataFrame:
    """Generate protocol-valid PCAP archetypes independent of Kelly holdout.

    Benign examples include ordinary attach, re-authentication, location update
    and detach procedures.  Attack examples reuse valid Diameter commands but
    at broadly randomised flood/burst volumes and rates.  Exact Kelly counts,
    timings and scores are never used.
    """
    rng = np.random.default_rng(seed)
    records = []

    benign_profiles = [
        ("benign_attach", {"AIR": 0.50, "ULR": 0.50}),
        ("benign_reauth", {"AIR": 1.00}),
        ("benign_location_update", {"ULR": 0.75, "AIR": 0.15, "NOR": 0.10}),
        ("benign_detach", {"PUR": 0.45, "CLR": 0.35, "NOR": 0.20}),
    ]
    attack_profiles = [
        ("air_flood", {"AIR": 1.00}),
        ("ulr_flood", {"ULR": 1.00}),
        ("mixed_flood", {"AIR": 0.50, "ULR": 0.50}),
        ("tracking_burst", {"AIR": 0.45, "ULR": 0.35, "IDR": 0.20}),
    ]

    def sample(profile_name: str, mix: Dict[str, float], attack: bool):
        if attack:
            s6a_total = float(
                np.exp(rng.uniform(np.log(30.0), np.log(2500.0))))
            duration = float(np.exp(rng.uniform(np.log(0.05), np.log(12.0))))
        else:
            s6a_total = float(
                np.exp(rng.uniform(np.log(2.0), np.log(48.0))))
            # Orange calibration includes very short benign request/answer
            # exchanges, so benign timing must span sub-millisecond-scale
            # captures as well as ordinary paced procedures.
            duration = float(np.exp(
                rng.uniform(np.log(0.001), np.log(40.0))))

        # Diameter peer/session overhead (CER/CEA, DWR/DWA, DPR/DPA) is a
        # protocol property, not a Kelly-specific observation.  It dominates
        # short benign captures but is amortised by floods.  Model it broadly
        # rather than assuming every decoded PDU belongs to S6a.
        base_overhead = float(rng.integers(2, 9))
        base_active_commands = int(rng.integers(1, 4))

        # Add broad overlap so the classifier cannot reduce the task to one
        # hard count cutoff.
        if rng.random() < 0.15:
            if attack:
                s6a_total *= rng.uniform(0.35, 0.75)
                duration *= rng.uniform(1.5, 4.0)
            else:
                s6a_total *= rng.uniform(1.5, 3.0)
                duration *= rng.uniform(0.3, 0.8)

        total = s6a_total + base_overhead
        mix_vec = np.array([mix.get(c, 0.0) for c in S6A_COMMANDS], dtype=float)
        concentration = 80.0 if not attack else 55.0
        alpha = np.maximum(mix_vec * concentration, 0.02)
        noisy = rng.dirichlet(alpha)
        s6a_cmd = {
            c: float(noisy[i] * s6a_total) for i, c in enumerate(S6A_COMMANDS)
        }
        gx_cmd = {c: 0.0 for c in GX_COMMANDS}
        out = _normalised_command_features(
            total, s6a_total, 0.0, 0.0, 0.0, base_overhead,
            s6a_cmd, gx_cmd)

        rate = total / max(duration, 1e-3)
        count_cv = float(rng.uniform(0.10, 0.95 if attack else 0.75))
        peak = float(rng.uniform(1.05, 3.0 if attack else 2.2))
        active_s6a = int(np.sum(noisy > 0.02))
        active_commands = active_s6a + base_active_commands
        out.update({
            "pcap_log_total_pdus": float(np.log1p(total)),
            "pcap_log_duration_s": float(np.log1p(duration)),
            "pcap_log_rate_hz": float(np.log1p(rate)),
            "pcap_s6a_fraction": s6a_total / max(total, 1.0),
            "pcap_gx_fraction": 0.0,
            "pcap_rx_fraction": 0.0,
            "pcap_s13_fraction": 0.0,
            "pcap_other_fraction": base_overhead / max(total, 1.0),
            "pcap_iat_cv": float(rng.uniform(0.15, 1.4)),
            "pcap_bin_count_cv": count_cv,
            "pcap_bin_count_peak_ratio": peak,
            "pcap_bin_count_autocorr": float(rng.uniform(-0.35, 0.75)),
            "pcap_bin_entropy_std": float(rng.uniform(0.0, 0.25)),
            "pcap_bin_s6a_fraction_std": float(rng.uniform(0.0, 0.08)),
            "pcap_air_ratio_std": float(rng.uniform(0.0, 0.18)),
            "pcap_ulr_ratio_std": float(rng.uniform(0.0, 0.18)),
            "pcap_active_command_fraction": float(
                active_commands /
                (len(S6A_COMMANDS) + len(GX_COMMANDS) + 4)
            ),
            "pcap_log_burst_mass": float(np.log1p(total * rate)),
            "pcap_log_pdus_per_active_command": float(
                np.log1p(total / max(active_commands, 1))),
            "is_anomaly": int(attack),
            "attack_type": profile_name if attack else "none",
            "protocol_archetype": profile_name,
        })
        return out

    for i in range(n_benign):
        name, mix = benign_profiles[i % len(benign_profiles)]
        records.append(sample(name, mix, False))
    for i in range(n_attack):
        name, mix = attack_profiles[i % len(attack_profiles)]
        records.append(sample(name, mix, True))
    return pd.DataFrame(records)


def _make_hgb(seed: int):
    return HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=31,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )


def _branch_scores(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    scores = []
    for name, cols in bundle["branch_features"].items():
        model = bundle["branches"][name]
        scores.append(model.predict_proba(X[cols])[:, 1])
    return np.column_stack(scores)


def _meta_matrix(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    branch = _branch_scores(bundle, X)
    raw_cols = bundle.get("meta_raw_features", META_RAW_FEATURES)
    raw = X[raw_cols].to_numpy(dtype=float)
    return np.column_stack([branch, raw])


def _meta_score(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    return bundle["meta"].predict_proba(_meta_matrix(bundle, X))[:, 1]


def _metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (score >= threshold).astype(int)
    benign = y == 0
    return {
        "auc_roc": float(roc_auc_score(y, score)),
        "auc_pr": float(average_precision_score(y, score)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "benign_fpr": float(pred[benign].mean()) if benign.any() else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def train_detector(
    calibration_pcaps: Sequence[Path],
    base_seeds: Sequence[int] = (42, 43),
    meta_seed: int = 44,
    threshold_seed: int = 45,
    test_seed: int = 46,
) -> Tuple[dict, dict]:
    frames = {}
    all_seeds = list(base_seeds) + [meta_seed, threshold_seed, test_seed]
    for seed in all_seeds:
        sim = build_training_frame(seed)
        native = simulation_to_native_windows(sim)
        archetypes = protocol_archetype_frame(
            5000 + seed, n_benign=1200, n_attack=1200)
        native = pd.concat([native, archetypes], ignore_index=True)
        native["source_seed"] = seed
        frames[seed] = native
        print(
            f"seed={seed}: windows={len(native)}, "
            f"positive={native['is_anomaly'].mean():.4f}, "
            f"protocol_archetypes={len(archetypes)}")

    train = pd.concat(
        [domain_randomise(frames[s], 1000 + s, variants=4)
         for s in base_seeds],
        ignore_index=True,
    )
    meta_fit = domain_randomise(frames[meta_seed], 1000 + meta_seed, variants=2)
    threshold_fit = domain_randomise(
        frames[threshold_seed], 1000 + threshold_seed, variants=2)
    test = domain_randomise(frames[test_seed], 1000 + test_seed, variants=1)

    branch_features = {
        "profile": PROFILE_FEATURES,
        "volume": VOLUME_FEATURES,
        "temporal": TEMPORAL_FEATURES,
        "wire_protocol": WIRE_FEATURES,
        "full_rf": PCAP_NATIVE_FEATURES,
    }
    branches = {
        "profile": _make_hgb(101),
        "volume": _make_hgb(102),
        "temporal": _make_hgb(103),
        "wire_protocol": _make_hgb(106),
        "full_rf": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            max_depth=14,
            min_samples_leaf=4,
            min_samples_split=8,
            max_features="sqrt",
            bootstrap=True,
            max_samples=0.85,
            random_state=104,
            n_jobs=-1,
        ),
    }

    y_train = train["is_anomaly"].astype(int).to_numpy()
    for name, model in branches.items():
        if name == "wire_protocol":
            # Train the wire expert only on protocol-valid low-volume
            # archetypes.  Other branches retain simulator-scale evidence.
            mask = train["protocol_archetype"].notna().to_numpy()
            model.fit(
                train.loc[mask, branch_features[name]],
                y_train[mask],
            )
        else:
            model.fit(train[branch_features[name]], y_train)

    temp_bundle = {
        "branches": branches,
        "branch_features": branch_features,
    }
    temp_bundle["meta_raw_features"] = META_RAW_FEATURES
    meta_X = _meta_matrix(temp_bundle, meta_fit[PCAP_NATIVE_FEATURES])
    y_meta = meta_fit["is_anomaly"].astype(int).to_numpy()
    meta = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=3000,
            random_state=105,
        ),
    ).fit(meta_X, y_meta)

    bundle = {
        "version": 1,
        "features": PCAP_NATIVE_FEATURES,
        "branch_features": branch_features,
        "branches": branches,
        "meta": meta,
        "meta_raw_features": META_RAW_FEATURES,
        "threshold": None,
        "threshold_method": None,
        "seed_roles": {
            "base_fit": list(base_seeds),
            "meta_fit": meta_seed,
            "threshold_fit": threshold_seed,
            "internal_test": test_seed,
        },
    }

    calib_score = _meta_score(bundle, threshold_fit[PCAP_NATIVE_FEATURES])
    y_calib = threshold_fit["is_anomaly"].astype(int).to_numpy()
    synth_benign = calib_score[y_calib == 0]
    synth_threshold = float(np.quantile(synth_benign, 0.975))

    external_benign_scores = []
    external_benign_rows = []
    for p in calibration_pcaps:
        feat = pcap_to_native_features(p)
        X = pd.DataFrame([{k: feat[k] for k in PCAP_NATIVE_FEATURES}])
        score = float(_meta_score(bundle, X)[0])
        external_benign_scores.append(score)
        external_benign_rows.append({
            "capture": p.name,
            "score": score,
            "decoded_pdus": int(feat["_decoded_pdus"]),
        })

    # Orange/diafuzzer is calibration-only.  Taking the maximum benign score
    # is conservative because the corpus is small; Kelly remains untouched.
    external_floor = max(external_benign_scores) if external_benign_scores else 0.0
    threshold = max(synth_threshold, float(external_floor))
    bundle["threshold"] = threshold
    bundle["threshold_method"] = (
        "max(97.5th percentile synthetic benign calibration seed, "
        "maximum independent Orange benign PCAP score)"
    )

    test_score = _meta_score(bundle, test[PCAP_NATIVE_FEATURES])
    y_test = test["is_anomaly"].astype(int).to_numpy()
    test_metrics = _metrics(y_test, test_score, threshold)

    # Per-attack cross-seed internal recall.
    pred_test = test_score >= threshold
    per_attack = {}
    for at in sorted(set(test["attack_type"].astype(str)) - {"none"}):
        mask = test["attack_type"].astype(str).to_numpy() == at
        per_attack[at] = {
            "count": int(mask.sum()),
            "recall": float(pred_test[mask].mean()) if mask.any() else 0.0,
        }

    metrics = {
        "scope": (
            "PCAP-native portable detector; node_compromise excluded because "
            "standalone PCAP lacks CU/report inconsistency evidence"
        ),
        "feature_count": len(PCAP_NATIVE_FEATURES),
        "meta_raw_features": META_RAW_FEATURES,
        "features": PCAP_NATIVE_FEATURES,
        "seed_roles": bundle["seed_roles"],
        "domain_randomisation": {
            "traffic_scale_range": [0.002, 2.0],
            "time_scale_range": [0.5, 2.0],
            "command_mix_sigma": 0.025,
            "temporal_scale_log_sigma": 0.30,
            "protocol_benign_archetypes": [
                "attach", "re-authentication", "location update", "detach"
            ],
            "protocol_attack_archetypes": [
                "AIR flood", "ULR flood", "mixed flood", "tracking burst"
            ],
            "holdout_specific_parameters_used": False,
        },
        "threshold": threshold,
        "threshold_method": bundle["threshold_method"],
        "synthetic_benign_threshold_97_5pct": synth_threshold,
        "independent_calibration_pcaps": external_benign_rows,
        "internal_cross_seed_test": test_metrics,
        "internal_cross_seed_per_attack": per_attack,
        "train_windows": int(len(train)),
        "meta_fit_windows": int(len(meta_fit)),
        "threshold_fit_windows": int(len(threshold_fit)),
        "test_windows": int(len(test)),
    }
    return bundle, metrics


def score_pcaps(
    bundle: dict,
    manifest: pd.DataFrame,
    pcap_dir: Path,
    threshold: float | None = None,
) -> pd.DataFrame:
    decision_threshold = (
        float(bundle["threshold"]) if threshold is None else float(threshold)
    )
    rows = []
    for rec in manifest.to_dict("records"):
        path = pcap_dir / str(rec["capture"])
        feat = pcap_to_native_features(path)
        X = pd.DataFrame([{k: feat[k] for k in PCAP_NATIVE_FEATURES}])
        branch = _branch_scores(bundle, X)[0]
        score = float(_meta_score(bundle, X)[0])
        out = {
            **rec,
            "pcap_native_score": score,
            "pcap_native_pred_anomaly": int(score >= decision_threshold),
            "decoded_pdus": int(feat["_decoded_pdus"]),
            "duration_s": float(feat["_duration_s"]),
        }
        for name, value in zip(bundle["branch_features"].keys(), branch):
            out[f"pcap_native_{name}_score"] = float(value)
        rows.append(out)
    return pd.DataFrame(rows)


def summarise_external(
    pred: pd.DataFrame,
    threshold: float,
    threshold_source: str = "frozen_global",
) -> Dict[str, object]:
    y = (pred["expected_label"].astype(str) == "attack").astype(int).to_numpy()
    p = pred["pcap_native_pred_anomaly"].astype(int).to_numpy()
    summary = {
        "capture_count": int(len(pred)),
        "attack_capture_count": int(y.sum()),
        "benign_capture_count": int((1 - y).sum()),
        "feature_count": len(PCAP_NATIVE_FEATURES),
        "threshold": float(threshold),
        "threshold_source": str(threshold_source),
        "external_attack_tpr": float(
            recall_score(y, p, zero_division=0)) if y.sum() else None,
        "external_benign_fpr": float(
            p[y == 0].mean()) if np.any(y == 0) else None,
        "external_f1": float(
            f1_score(y, p, zero_division=0)) if len(np.unique(y)) > 1 else None,
        "external_balanced_accuracy": float(
            balanced_accuracy_score(y, p)) if len(np.unique(y)) > 1 else None,
    }
    per_scenario = {}
    if "scenario" in pred.columns:
        for scenario, grp in pred.groupby("scenario"):
            gy = (grp["expected_label"].astype(str) == "attack").astype(int).to_numpy()
            gp = grp["pcap_native_pred_anomaly"].astype(int).to_numpy()
            per_scenario[str(scenario)] = {
                "captures": int(len(grp)),
                "expected_label": str(grp["expected_label"].iloc[0]),
                "detection_rate": float(gp.mean()),
                "decoded_pdus": int(grp["decoded_pdus"].sum()),
            }
    summary["per_scenario"] = per_scenario
    return summary



LOCAL_ADAPT_FEATURES = list(dict.fromkeys(
    PORTABLE_FEATURES + [
        "pcap_log_total_pdus",
        "pcap_log_duration_s",
        "pcap_log_rate_hz",
        "pcap_s6a_fraction",
        "pcap_gx_fraction",
        "pcap_rx_fraction",
        "pcap_s13_fraction",
        "pcap_other_fraction",
        "pcap_iat_cv",
        "pcap_bin_count_cv",
        "pcap_bin_count_peak_ratio",
        "pcap_bin_count_autocorr",
        "pcap_bin_entropy_std",
        "pcap_bin_s6a_fraction_std",
        "pcap_air_ratio_std",
        "pcap_ulr_ratio_std",
        "pcap_active_command_fraction",
        "pcap_log_burst_mass",
        "pcap_log_pdus_per_active_command",
    ]
))


def _robust_scale_matrix(X: np.ndarray) -> np.ndarray:
    q25 = np.quantile(X, 0.25, axis=0)
    q75 = np.quantile(X, 0.75, axis=0)
    iqr_scale = (q75 - q25) / 1.349
    std = np.std(X, axis=0)
    # A finite floor prevents a feature that is constant in a small benign
    # calibration set from dominating the local distance by numerical noise.
    scale = np.maximum.reduce([
        iqr_scale,
        0.25 * std,
        np.full(X.shape[1], 0.05, dtype=float),
    ])
    return scale


def _topk_rms(delta: np.ndarray, k: int = 8) -> float:
    d = np.clip(np.abs(np.asarray(delta, dtype=float)), 0.0, 12.0)
    if d.size == 0:
        return 0.0
    kk = min(k, d.size)
    top = np.partition(d, d.size - kk)[-kk:]
    return float(np.sqrt(np.mean(top ** 2)))


def _nn_distance(x: np.ndarray, ref: np.ndarray, scale: np.ndarray) -> float:
    if len(ref) == 0:
        return float("inf")
    z = (ref - x.reshape(1, -1)) / scale.reshape(1, -1)
    distances = np.array([_topk_rms(row) for row in z], dtype=float)
    return float(np.min(distances))


def _envelope_distance(
    x: np.ndarray,
    ref: np.ndarray,
    scale: np.ndarray,
) -> float:
    if len(ref) == 0:
        return float("inf")
    lo = np.min(ref, axis=0)
    hi = np.max(ref, axis=0)
    below = np.maximum((lo - x) / scale, 0.0)
    above = np.maximum((x - hi) / scale, 0.0)
    return _topk_rms(np.maximum(below, above), k=4)


def _conformal_upper(values: np.ndarray, alpha: float) -> float:
    vals = np.sort(np.asarray(values, dtype=float))
    if len(vals) == 0:
        raise RuntimeError("Cannot calibrate local detector on an empty set")
    k = int(np.ceil((len(vals) + 1) * (1.0 - alpha)))
    idx = min(max(k - 1, 0), len(vals) - 1)
    return float(np.nextafter(vals[idx], np.inf))


def _local_reference_score(
    x: np.ndarray,
    ref: np.ndarray,
    scale: np.ndarray,
    nn_threshold: float,
    envelope_threshold: float,
) -> Dict[str, float]:
    nn = _nn_distance(x, ref, scale)
    env = _envelope_distance(x, ref, scale)
    nn_norm = nn / max(float(nn_threshold), 1e-9)
    env_norm = env / max(float(envelope_threshold), 1e-9)
    return {
        "nn_distance": float(nn),
        "envelope_distance": float(env),
        "local_score": float(max(nn_norm, env_norm)),
    }


def fit_local_benign_calibration(
    bundle: dict,
    manifest: pd.DataFrame,
    pcap_dir: Path,
    target_fpr: float,
) -> Dict[str, object]:
    labels = manifest["expected_label"].astype(str).str.lower().str.strip()
    if set(labels) != {"benign"}:
        raise SystemExit(
            "Local one-class calibration accepts benign captures only")
    if len(manifest) < 12:
        raise SystemExit(
            "At least 12 benign PCAPs are required for local one-class calibration")
    if not (0.0 < target_fpr < 1.0):
        raise SystemExit("target_fpr must be strictly between 0 and 1")

    rows = []
    meta_rows = []
    roles = []
    has_roles = "role" in manifest.columns
    for rec in manifest.to_dict("records"):
        path = pcap_dir / str(rec["capture"])
        feat = pcap_to_native_features(path)
        rows.append([float(feat[c]) for c in LOCAL_ADAPT_FEATURES])

        role = str(rec.get("role", "")).lower().strip()
        roles.append(role)
        X = pd.DataFrame([{k: feat[k] for k in PCAP_NATIVE_FEATURES}])
        branch_scores = _branch_scores(bundle, X)[0]
        meta_rows.append({
            "capture": str(rec["capture"]),
            "role": role if has_roles else "loo_calibration",
            "decoded_pdus": int(feat["_decoded_pdus"]),
            "duration_s": float(feat["_duration_s"]),
            "frozen_supervised_score": float(_meta_score(bundle, X)[0]),
            **{
                f"branch_{name}": float(value)
                for name, value in zip(
                    bundle["branch_features"].keys(), branch_scores)
            },
        })

    Xall = np.asarray(rows, dtype=float)

    if has_roles:
        allowed = {"reference", "calibration"}
        unknown = sorted(set(roles) - allowed)
        if unknown:
            raise SystemExit(
                "Local calibration role must be reference/calibration only; "
                f"unknown roles: {unknown}")
        ref_mask = np.asarray([r == "reference" for r in roles], dtype=bool)
        cal_mask = np.asarray([r == "calibration" for r in roles], dtype=bool)
        Xref = Xall[ref_mask]
        Xconf = Xall[cal_mask]
        if len(Xref) < 12:
            raise SystemExit(
                f"At least 12 benign reference PCAPs required; got {len(Xref)}")
        if len(Xconf) < 8:
            raise SystemExit(
                f"At least 8 benign conformal PCAPs required; got {len(Xconf)}")

        scale = _robust_scale_matrix(Xref)
        nn_scores = np.asarray(
            [_nn_distance(x, Xref, scale) for x in Xconf], dtype=float)
        env_scores = np.asarray(
            [_envelope_distance(x, Xref, scale) for x in Xconf], dtype=float)
        method = "deployment_benign_local_manifold_split_conformal"
        reference_count = int(len(Xref))
        conformal_count = int(len(Xconf))
    else:
        # Backwards-compatible leave-one-out mode for small development sets.
        # A production deployment should prefer explicit reference/calibration
        # roles so the manifold fit and conformal threshold are source-separated.
        Xref = Xall
        scale = _robust_scale_matrix(Xref)
        nn_scores = []
        env_scores = []
        for i in range(len(Xref)):
            ref = np.delete(Xref, i, axis=0)
            nn_scores.append(_nn_distance(Xref[i], ref, scale))
            env_scores.append(_envelope_distance(Xref[i], ref, scale))
        nn_scores = np.asarray(nn_scores, dtype=float)
        env_scores = np.asarray(env_scores, dtype=float)
        method = "deployment_benign_local_manifold_conformal"
        reference_count = int(len(Xref))
        conformal_count = int(len(Xref))

    nn_threshold = _conformal_upper(nn_scores, target_fpr)
    envelope_threshold = _conformal_upper(env_scores, target_fpr)

    profiles = {}
    if has_roles and "scenario" in manifest.columns:
        scenarios = manifest["scenario"].astype(str).to_numpy()
        for scenario in sorted(set(scenarios)):
            p_ref_mask = ref_mask & (scenarios == scenario)
            p_cal_mask = cal_mask & (scenarios == scenario)
            Xp_ref = Xall[p_ref_mask]
            Xp_cal = Xall[p_cal_mask]
            # Profile-specific manifolds are deliberately modest in size; the
            # global manifold remains available when a deployment cannot supply
            # enough examples for a procedure family.
            if len(Xp_ref) < 4 or len(Xp_cal) < 6:
                continue
            p_scale = _robust_scale_matrix(Xp_ref)
            p_nn = np.asarray(
                [_nn_distance(x, Xp_ref, p_scale) for x in Xp_cal],
                dtype=float)
            p_env = np.asarray(
                [_envelope_distance(x, Xp_ref, p_scale) for x in Xp_cal],
                dtype=float)
            profiles[str(scenario)] = {
                "reference_vectors": Xp_ref.tolist(),
                "scale": p_scale.tolist(),
                "reference_capture_count": int(len(Xp_ref)),
                "conformal_capture_count": int(len(Xp_cal)),
                "nn_threshold": float(_conformal_upper(p_nn, target_fpr)),
                "envelope_threshold": float(
                    _conformal_upper(p_env, target_fpr)),
                "calibration_nn_scores": p_nn.tolist(),
                "calibration_envelope_scores": p_env.tolist(),
            }
        if profiles:
            method = "deployment_benign_mixture_manifold_split_conformal"

    return {
        "version": 3,
        "method": method,
        "scope": (
            "deployment-local one-class detector; frozen global model; "
            "benign PCAPs only; no attack labels"
        ),
        "model_sha256": None,
        "feature_names": LOCAL_ADAPT_FEATURES,
        "reference_vectors": Xref.tolist(),
        "scale": scale.tolist(),
        "target_fpr": float(target_fpr),
        "calibration_capture_count": int(len(Xall)),
        "reference_capture_count": reference_count,
        "conformal_capture_count": conformal_count,
        "nn_threshold": float(nn_threshold),
        "envelope_threshold": float(envelope_threshold),
        "calibration_nn_scores": nn_scores.tolist(),
        "calibration_envelope_scores": env_scores.tolist(),
        "profiles": profiles,
        "captures": meta_rows,
    }


def score_with_local_calibration(
    bundle: dict,
    calibration: Dict[str, object],
    feat: Dict[str, float],
) -> Dict[str, float]:
    names = list(calibration["feature_names"])
    x = np.asarray([float(feat[c]) for c in names], dtype=float)
    ref = np.asarray(calibration["reference_vectors"], dtype=float)
    scale = np.asarray(calibration["scale"], dtype=float)
    global_score = _local_reference_score(
        x, ref, scale,
        float(calibration["nn_threshold"]),
        float(calibration["envelope_threshold"]),
    )

    profile_scores = {}
    for name, profile in calibration.get("profiles", {}).items():
        p_ref = np.asarray(profile["reference_vectors"], dtype=float)
        p_scale = np.asarray(profile["scale"], dtype=float)
        profile_scores[name] = _local_reference_score(
            x, p_ref, p_scale,
            float(profile["nn_threshold"]),
            float(profile["envelope_threshold"]),
        )

    if profile_scores:
        best_name, best = min(
            profile_scores.items(), key=lambda kv: kv[1]["local_score"])
        return {
            "nn_distance": float(best["nn_distance"]),
            "envelope_distance": float(best["envelope_distance"]),
            "local_score": float(best["local_score"]),
            "matched_benign_profile": str(best_name),
            "global_local_score": float(global_score["local_score"]),
            "profile_scores": {
                str(k): float(v["local_score"])
                for k, v in profile_scores.items()
            },
        }
    return {
        **global_score,
        "matched_benign_profile": "global",
        "global_local_score": float(global_score["local_score"]),
        "profile_scores": {},
    }


def calibrate_local_cli(args) -> None:
    with args.model.open("rb") as f:
        bundle = pickle.load(f)
    manifest = _load_benign_calibration_manifest(args.manifest)
    calibration = fit_local_benign_calibration(
        bundle, manifest, args.pcap_dir, float(args.target_fpr))
    calibration["model_sha256"] = _sha256_file(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(calibration, ensure_ascii=False, indent=2))


def train_cli(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle, metrics = train_detector(args.calibration_pcaps)
    with (args.out_dir / "pcap_native_bundle.pkl").open("wb") as f:
        pickle.dump(bundle, f)
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    test = metrics["internal_cross_seed_test"]
    if test["recall"] < 0.70:
        raise SystemExit(
            f"PCAP-native internal cross-seed recall below 70%: {test['recall']:.3f}")
    if test["benign_fpr"] > 0.05:
        raise SystemExit(
            f"PCAP-native internal cross-seed FPR above 5%: {test['benign_fpr']:.3f}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_benign_calibration_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"capture", "expected_label"}
    missing = required.difference(manifest.columns)
    if missing:
        raise SystemExit(
            f"benign calibration manifest missing columns: {sorted(missing)}")
    labels = manifest["expected_label"].astype(str).str.lower().str.strip()
    bad = manifest.loc[labels != "benign", ["capture", "expected_label"]]
    if not bad.empty:
        raise SystemExit(
            "Deployment calibration accepts benign captures only; "
            f"non-benign rows found: {bad.to_dict('records')}")
    if len(manifest) < 8:
        raise SystemExit(
            "At least 8 independent benign captures are required for "
            "deployment-local calibration")
    return manifest


def calibrate_cli(args) -> None:
    """Fit a deployment-local threshold from benign PCAP scores only.

    The frozen classifier is never refit.  This calibration changes only the
    operating threshold and rejects any manifest containing attack labels.
    The order-statistic rule is the split-conformal upper quantile; with small
    benign calibration sets it naturally reduces to max(score)+ULP.
    """
    with args.model.open("rb") as f:
        bundle = pickle.load(f)
    manifest = _load_benign_calibration_manifest(args.manifest)

    scored = score_pcaps(
        bundle, manifest, args.pcap_dir,
        threshold=float("inf"),
    )
    scores = np.sort(scored["pcap_native_score"].to_numpy(dtype=float))
    n = len(scores)
    alpha = float(args.target_fpr)
    if not (0.0 < alpha < 1.0):
        raise SystemExit("--target-fpr must be strictly between 0 and 1")

    # Split-conformal upper quantile index.  For n too small to resolve the
    # requested alpha, use the maximum benign score.  nextafter prevents the
    # maximum calibration example itself from being classified as anomalous
    # under the detector's >= comparison.
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    idx = min(max(k - 1, 0), n - 1)
    base_threshold = float(scores[idx])
    threshold = float(np.nextafter(base_threshold, np.inf))

    model_sha = _sha256_file(args.model)
    calibration = {
        "version": 1,
        "method": "deployment_benign_split_conformal_upper_quantile",
        "scope": (
            "threshold-only deployment adaptation; frozen model weights; "
            "benign captures only; no attack labels"
        ),
        "model_sha256": model_sha,
        "feature_count": len(PCAP_NATIVE_FEATURES),
        "target_fpr": alpha,
        "calibration_capture_count": int(n),
        "conformal_rank_1_based": int(k),
        "selected_order_index_0_based": int(idx),
        "threshold": threshold,
        "global_frozen_threshold": float(bundle["threshold"]),
        "score_min": float(scores[0]),
        "score_median": float(np.median(scores)),
        "score_max": float(scores[-1]),
        "captures": [
            {
                "capture": str(row["capture"]),
                "score": float(row["pcap_native_score"]),
                "decoded_pdus": int(row["decoded_pdus"]),
                "duration_s": float(row["duration_s"]),
            }
            for _, row in scored.iterrows()
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(calibration, ensure_ascii=False, indent=2))


def score_cli(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.model.open("rb") as f:
        bundle = pickle.load(f)
    manifest = pd.read_csv(args.manifest)
    required = {"capture", "expected_label"}
    missing = required.difference(manifest.columns)
    if missing:
        raise SystemExit(f"manifest missing columns: {sorted(missing)}")

    threshold = float(bundle["threshold"])
    threshold_source = "frozen_global"
    calibration_meta = None
    if args.calibration is not None:
        calibration_meta = json.loads(
            args.calibration.read_text(encoding="utf-8"))
        expected_sha = str(calibration_meta.get("model_sha256", ""))
        actual_sha = _sha256_file(args.model)
        if not expected_sha or expected_sha != actual_sha:
            raise SystemExit(
                "Deployment calibration was produced for a different model "
                f"(expected {expected_sha}, actual {actual_sha})")
        if calibration_meta.get("method") != (
                "deployment_benign_split_conformal_upper_quantile"):
            raise SystemExit("Unsupported deployment calibration method")
        threshold = float(calibration_meta["threshold"])
        threshold_source = "deployment_benign_only"

    pred = score_pcaps(
        bundle, manifest, args.pcap_dir, threshold=threshold)

    local_meta = None
    if args.local_calibration is not None:
        local_meta = json.loads(
            args.local_calibration.read_text(encoding="utf-8"))
        expected_sha = str(local_meta.get("model_sha256", ""))
        actual_sha = _sha256_file(args.model)
        if expected_sha != actual_sha:
            raise SystemExit(
                "Local one-class calibration was produced for a different model")
        if local_meta.get("method") not in {
                "deployment_benign_local_manifold_conformal",
                "deployment_benign_local_manifold_split_conformal",
                "deployment_benign_mixture_manifold_split_conformal",
        }:
            raise SystemExit("Unsupported local calibration method")

        local_scores = []
        global_scores = []
        matched_profiles = []
        matched_profile_scores = []
        local_preds = []
        for rec in manifest.to_dict("records"):
            feat = pcap_to_native_features(
                args.pcap_dir / str(rec["capture"]))
            scored = score_with_local_calibration(bundle, local_meta, feat)
            local_scores.append(float(scored["local_score"]))
            global_scores.append(float(scored.get(
                "global_local_score", scored["local_score"])))
            matched_profiles.append(str(scored.get(
                "matched_benign_profile", "global")))
            pvals = scored.get("profile_scores", {})
            matched_profile_scores.append(
                float(min(pvals.values())) if pvals else float("nan"))
            local_preds.append(int(scored["local_score"] > 1.0))
        pred["frozen_supervised_pred_anomaly"] = (
            pred["pcap_native_pred_anomaly"].astype(int))
        pred["local_oneclass_score"] = local_scores
        pred["global_local_oneclass_score"] = global_scores
        pred["matched_benign_profile"] = matched_profiles
        pred["matched_benign_profile_score"] = matched_profile_scores
        pred["local_oneclass_pred_anomaly"] = local_preds
        # Detection decision in deployment-local mode is one-class.  The
        # frozen supervised score remains available as explanatory evidence
        # and possible attack-type classification, not as a hard gate.
        pred["pcap_native_pred_anomaly"] = local_preds
        threshold_source = "deployment_benign_local_oneclass"

    summary = summarise_external(
        pred, threshold, threshold_source=threshold_source)
    if local_meta is not None:
        summary["local_oneclass_calibration"] = {
            "method": local_meta["method"],
            "target_fpr": local_meta["target_fpr"],
            "calibration_capture_count": local_meta[
                "calibration_capture_count"],
            "reference_capture_count": local_meta.get(
                "reference_capture_count"),
            "conformal_capture_count": local_meta.get(
                "conformal_capture_count"),
            "profile_count": len(local_meta.get("profiles", {})),
            "profile_names": sorted(local_meta.get("profiles", {}).keys()),
            "nn_threshold": local_meta["nn_threshold"],
            "envelope_threshold": local_meta["envelope_threshold"],
        }
    if calibration_meta is not None:
        summary["deployment_calibration"] = {
            "method": calibration_meta["method"],
            "target_fpr": calibration_meta["target_fpr"],
            "calibration_capture_count": calibration_meta[
                "calibration_capture_count"],
            "score_max": calibration_meta["score_max"],
            "global_frozen_threshold": calibration_meta[
                "global_frozen_threshold"],
        }
    pred.to_csv(args.out_dir / "predictions.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pred.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument(
        "--calibration-pcaps", nargs="*", type=Path, default=[])
    tr.add_argument(
        "--out-dir", type=Path, default=Path("results_pcap_native_train"))
    tr.set_defaults(func=train_cli)

    cal = sub.add_parser("calibrate")
    cal.add_argument("--model", type=Path, required=True)
    cal.add_argument("--manifest", type=Path, required=True)
    cal.add_argument("--pcap-dir", type=Path, required=True)
    cal.add_argument("--target-fpr", type=float, default=0.05)
    cal.add_argument(
        "--out", type=Path,
        default=Path("deployment_benign_calibration.json"))
    cal.set_defaults(func=calibrate_cli)

    local = sub.add_parser("calibrate-local")
    local.add_argument("--model", type=Path, required=True)
    local.add_argument("--manifest", type=Path, required=True)
    local.add_argument("--pcap-dir", type=Path, required=True)
    local.add_argument("--target-fpr", type=float, default=0.05)
    local.add_argument(
        "--out", type=Path,
        default=Path("deployment_local_oneclass_calibration.json"))
    local.set_defaults(func=calibrate_local_cli)

    sc = sub.add_parser("score")
    sc.add_argument("--model", type=Path, required=True)
    sc.add_argument("--manifest", type=Path, required=True)
    sc.add_argument("--pcap-dir", type=Path, required=True)
    sc.add_argument(
        "--calibration", type=Path, default=None,
        help="Optional benign-only supervised threshold calibration JSON")
    sc.add_argument(
        "--local-calibration", type=Path, default=None,
        help="Optional benign-only deployment local one-class calibration JSON")
    sc.add_argument(
        "--out-dir", type=Path, default=Path("results_pcap_native_external"))
    sc.set_defaults(func=score_cli)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
