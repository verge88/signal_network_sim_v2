"""External validity check for the canonical SS7 model on independent real telecom data.

The real-data side uses the public Hugging Face mirror of the Telecom Italia Milan
CDR dataset.  These are measured, non-synthetic network activity records.  The mirror
is already aggregated by grid square and time, so this check uses only SS7 own-report
features that can be reconstructed from observed SMS/call activity without inventing
DPCs, TCAP state, transit traffic, cross-node invariants, integrity failures, or attack
labels.

This is therefore an external normal-traffic transfer / rejection-rate test.  It does
NOT estimate attack recall, precision, or AUC on real attacks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ss7.fix import SimConfig, build_dataset


ORIGINAL_DATASET_DOI = "10.7910/DVN/EGZHFV"
MIRROR_REPO = "GreatLove/CellularTrafficPrediction"
MIRROR_REVISION = "main"
MIRROR_FILE_SHA256 = {
    "2013-11-01": "eeaf3352af1bb076ba05fe20cfd964bfd9a55b3a1ce82019e6502714ef2b8fa9",
}

# The mirror exposes one row per grid square/time with these measured fields:
# datetime, squareid, sms-in, sms-out, call-in, call-out, internet.
# Only features defensibly derivable from SMS/call activity are evaluated.
FULL_PROXY_FEATURES = [
    "total_messages",
    "mtype_entropy",
    "map_share",
    "isup_share",
    "in_out_ratio",
]
RATIO_PROXY_FEATURES = [
    "mtype_entropy",
    "map_share",
    "isup_share",
    "in_out_ratio",
]


def mirror_url(date: str) -> str:
    return (
        "https://huggingface.co/datasets/"
        f"{MIRROR_REPO}/resolve/{MIRROR_REVISION}/"
        f"sms-call-internet-mi/{date}.csv"
    )


def download_prefix(url: str, dst: Path, max_bytes: int) -> Tuple[int, str]:
    """Download at most max_bytes and trim the final partial CSV line.

    The client stops the stream after max_bytes.  No rows are generated or modified;
    only a possibly incomplete final record is discarded.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = Request(
        url,
        headers={"User-Agent": "signal-network-sim-v2-external-validation/1.0"},
    )
    remaining = int(max_bytes)
    h = hashlib.sha256()
    with urlopen(req, timeout=120) as response, dst.open("wb") as out:
        while remaining > 0:
            chunk = response.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            remaining -= len(chunk)

    with dst.open("rb+") as f:
        data = f.read()
        end = data.rfind(b"\n")
        if end <= 0:
            raise RuntimeError("Downloaded mirror prefix contains no complete CSV rows")
        f.seek(0)
        f.write(data[: end + 1])
        f.truncate()

    return int(dst.stat().st_size), h.hexdigest()


def _shannon_rows(parts: np.ndarray) -> np.ndarray:
    total = parts.sum(axis=1)
    p = np.divide(
        parts,
        total[:, None],
        out=np.zeros_like(parts, dtype=float),
        where=total[:, None] > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log2(p), 0.0)
    return terms.sum(axis=1)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "datetime": "datetime",
        "date": "datetime",
        "time": "datetime",
        "squareid": "cell_id",
        "square_id": "cell_id",
        "cellid": "cell_id",
        "cell_id": "cell_id",
        "sms-in": "sms_in",
        "sms_in": "sms_in",
        "smsin": "sms_in",
        "sms-out": "sms_out",
        "sms_out": "sms_out",
        "smsout": "sms_out",
        "call-in": "call_in",
        "call_in": "call_in",
        "callin": "call_in",
        "call-out": "call_out",
        "call_out": "call_out",
        "callout": "call_out",
        "internet": "internet",
    }
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in aliases:
            rename[c] = aliases[key]
    out = df.rename(columns=rename)
    required = {"datetime", "cell_id", "sms_in", "sms_out", "call_in", "call_out"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise RuntimeError(
            f"Unexpected mirror schema; missing {missing}; columns={list(df.columns)}"
        )
    return out


def load_real_cdr_features(
    path: Path, max_windows: int, seed: int
) -> Tuple[pd.DataFrame, dict]:
    raw = _normalise_columns(pd.read_csv(path, low_memory=False))
    raw = raw[["datetime", "cell_id", "sms_in", "sms_out", "call_in", "call_out"]].copy()

    activity_cols = ["sms_in", "sms_out", "call_in", "call_out"]
    for c in activity_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    raw["cell_id"] = pd.to_numeric(raw["cell_id"], errors="coerce")
    raw = raw.dropna(subset=["cell_id", "datetime"]).copy()
    raw["cell_id"] = raw["cell_id"].astype("int64")

    sms_in = raw["sms_in"].to_numpy(float)
    sms_out = raw["sms_out"].to_numpy(float)
    call_in = raw["call_in"].to_numpy(float)
    call_out = raw["call_out"].to_numpy(float)
    call_total = call_in + call_out

    # Signalling-equivalent units preserve the canonical service templates only:
    # one call contributes five ISUP message-type units; an outgoing SMS one MAP
    # operation; an incoming SMS two MAP operations.  Telecom Italia activity values
    # remain anonymised activity measures, not literal MSU/message counts.
    parts = np.column_stack([
        call_total,
        call_total,
        call_total,
        call_total,
        call_total,
        sms_out,
        sms_in,
        sms_in,
    ])
    total = parts.sum(axis=1)
    map_units = sms_out + 2.0 * sms_in
    isup_units = 5.0 * call_total
    denom = np.maximum(total, 1e-12)

    out = pd.DataFrame({
        "cell_id": raw["cell_id"].to_numpy(),
        "datetime": raw["datetime"].astype(str).to_numpy(),
        "total_messages": total,
        "mtype_entropy": _shannon_rows(parts),
        "map_share": map_units / denom,
        "isup_share": isup_units / denom,
        "in_out_ratio": (
            5.0 * call_in + 2.0 * sms_in
        ) / np.maximum(5.0 * call_out + sms_out, 1e-12),
    })

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=FULL_PROXY_FEATURES)
    out = out[out["total_messages"] > 0].copy()
    if len(out) < 100:
        raise RuntimeError(f"Only {len(out)} usable real CDR windows; need at least 100")

    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=True)
    out = out.dropna(subset=["datetime"]).copy()

    # Preserve complete cell trajectories.  Random row sampling would leak the same
    # timestamps into train/validation/test and make temporal calibration optimistic.
    selection = "all_cells"
    if len(out) > max_windows:
        counts = out.groupby("cell_id").size()
        cells = counts.index.to_numpy()
        rng = np.random.default_rng(seed)
        cells = cells[rng.permutation(len(cells))]
        chosen = []
        used = 0
        for cell in cells:
            n = int(counts.loc[cell])
            if chosen and used + n > max_windows:
                continue
            chosen.append(int(cell))
            used += n
            if used >= int(max_windows * 0.98):
                break
        if not chosen:
            raise RuntimeError("Cannot select complete cell trajectories under max_windows")
        out = out[out["cell_id"].isin(chosen)].copy()
        selection = "complete_cell_trajectories"

    out = out.sort_values(["datetime", "cell_id"]).reset_index(drop=True)
    meta = {
        "raw_rows_in_prefix": int(len(raw)),
        "usable_windows": int(len(out)),
        "unique_cells": int(out["cell_id"].nunique()),
        "unique_timestamps": int(out["datetime"].nunique()),
        "timestamp_min": str(out["datetime"].min()),
        "timestamp_max": str(out["datetime"].max()),
        "sampling": selection,
    }
    return out, meta


def synthetic_normal(seed: int, days: float) -> pd.DataFrame:
    cfg = SimConfig(seed=seed, days=days)
    df, _, _ = build_dataset(
        cfg,
        episodes_kw={"n_compromise": 0, "n_normal": 0, "n_ext": 0},
    )
    need = ["nid", "t"] + FULL_PROXY_FEATURES
    out = df[need].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(out) < 1000:
        raise RuntimeError(f"Synthetic normal sample too small: {len(out)}")
    return out


def time_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = np.sort(df["t"].unique())
    if len(ts) < 10:
        raise RuntimeError("Synthetic sample has too few time intervals")
    cut1 = ts[max(1, int(len(ts) * 0.60)) - 1]
    cut2 = ts[max(2, int(len(ts) * 0.80)) - 1]
    tr = df[df["t"] <= cut1]
    va = df[(df["t"] > cut1) & (df["t"] <= cut2)]
    te = df[df["t"] > cut2]
    if min(len(tr), len(va), len(te)) < 100:
        raise RuntimeError("Synthetic temporal split is too small")
    return tr, va, te


def real_time_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Strict 60/20/20 split by unique real timestamps."""
    times = np.array(sorted(df["datetime"].dropna().unique()))
    if len(times) < 15:
        raise RuntimeError(
            f"Real-data temporal split requires >=15 timestamps, got {len(times)}"
        )
    i1 = max(1, int(len(times) * 0.60))
    i2 = max(i1 + 1, int(len(times) * 0.80))
    i2 = min(i2, len(times) - 1)
    train_times = set(times[:i1])
    val_times = set(times[i1:i2])
    test_times = set(times[i2:])

    tr = df[df["datetime"].isin(train_times)].copy()
    va = df[df["datetime"].isin(val_times)].copy()
    te = df[df["datetime"].isin(test_times)].copy()
    if min(len(tr), len(va), len(te)) < 100:
        raise RuntimeError(
            f"Real temporal split too small: train={len(tr)}, val={len(va)}, test={len(te)}"
        )

    def desc(part: pd.DataFrame) -> dict:
        return {
            "rows": int(len(part)),
            "cells": int(part["cell_id"].nunique()),
            "timestamps": int(part["datetime"].nunique()),
            "start": str(part["datetime"].min()),
            "end": str(part["datetime"].max()),
        }

    return tr, va, te, {
        "method": "strict_unique_timestamp_60_20_20",
        "train": desc(tr),
        "validation": desc(va),
        "test": desc(te),
    }


def feature_shift(
    train: pd.DataFrame,
    real: pd.DataFrame,
    cols: List[str],
    imp: SimpleImputer,
    scaler: StandardScaler,
) -> Dict[str, dict]:
    sx = scaler.transform(imp.transform(train[cols]))
    rx = scaler.transform(imp.transform(real[cols]))
    out: Dict[str, dict] = {}
    for j, col in enumerate(cols):
        s = sx[:, j]
        r = rx[:, j]
        ks = ks_2samp(s, r, alternative="two-sided", mode="auto")
        lo, hi = np.quantile(s, [0.005, 0.995])
        out[col] = {
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "median_shift_sigma": float(abs(np.median(r) - np.median(s))),
            "real_inside_synth_99pct": float(np.mean((r >= lo) & (r <= hi))),
        }
    return out


def fit_detector(
    train: pd.DataFrame, cols: List[str], seed: int
) -> Tuple[SimpleImputer, StandardScaler, IsolationForest]:
    imp = SimpleImputer(strategy="median").fit(train[cols])
    scaler = StandardScaler().fit(imp.transform(train[cols]))
    x = scaler.transform(imp.transform(train[cols]))
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    ).fit(x)
    return imp, scaler, model


def detector_scores(
    frame: pd.DataFrame,
    cols: List[str],
    imp: SimpleImputer,
    scaler: StandardScaler,
    model: IsolationForest,
) -> np.ndarray:
    x = scaler.transform(imp.transform(frame[cols]))
    return -model.score_samples(x)


def threshold_for_fpr(scores: np.ndarray, target_fpr: float) -> float:
    return float(np.quantile(scores, 1.0 - target_fpr))


def wilson_interval(k: int, n: int, z: float = 1.96) -> List[float]:
    if n <= 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt((p * (1.0 - p) / n) + z * z / (4.0 * n * n)) / den
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def rejection_summary(scores: np.ndarray, threshold: float) -> dict:
    rejected = scores > threshold
    k = int(rejected.sum())
    n = int(len(rejected))
    return {
        "rows": n,
        "rejections": k,
        "fpr": float(k / n) if n else float("nan"),
        "fpr_wilson_95": wilson_interval(k, n),
    }


def temporal_block_fpr(
    frame: pd.DataFrame, scores: np.ndarray, threshold: float, n_blocks: int = 4
) -> dict:
    tmp = frame[["datetime"]].copy()
    tmp["score"] = scores
    unique_times = np.array(sorted(tmp["datetime"].unique()))
    blocks = []
    for idx, times in enumerate(np.array_split(unique_times, n_blocks), start=1):
        if len(times) == 0:
            continue
        sub = tmp[tmp["datetime"].isin(set(times))]
        summary = rejection_summary(sub["score"].to_numpy(float), threshold)
        summary.update({
            "block": idx,
            "timestamps": int(len(times)),
            "start": str(sub["datetime"].min()),
            "end": str(sub["datetime"].max()),
        })
        blocks.append(summary)
    rates = np.array([b["fpr"] for b in blocks], dtype=float)
    return {
        "blocks": blocks,
        "mean_fpr": float(np.mean(rates)) if len(rates) else float("nan"),
        "std_fpr": float(np.std(rates)) if len(rates) else float("nan"),
        "min_fpr": float(np.min(rates)) if len(rates) else float("nan"),
        "max_fpr": float(np.max(rates)) if len(rates) else float("nan"),
    }


def evaluate_profile(
    syn_tr: pd.DataFrame,
    syn_va: pd.DataFrame,
    syn_te: pd.DataFrame,
    real_tr: pd.DataFrame,
    real_va: pd.DataFrame,
    real_te: pd.DataFrame,
    cols: List[str],
    seed: int,
    target_fpr: float,
) -> dict:
    # Domain-shift diagnostics are measured only against real TRAIN so the held-out
    # real test block remains untouched until final evaluation.
    simp = SimpleImputer(strategy="median").fit(syn_tr[cols])
    ssc = StandardScaler().fit(simp.transform(syn_tr[cols]))
    shift = feature_shift(syn_tr, real_tr, cols, simp, ssc)
    ks_values = np.array([v["ks_stat"] for v in shift.values()], dtype=float)
    coverage = np.array(
        [v["real_inside_synth_99pct"] for v in shift.values()], dtype=float
    )

    # Mode 1: original synthetic-only calibration.
    imp_s, sc_s, model_s = fit_detector(syn_tr, cols, seed)
    score_syn_val = detector_scores(syn_va, cols, imp_s, sc_s, model_s)
    score_syn_test = detector_scores(syn_te, cols, imp_s, sc_s, model_s)
    score_real_val_s = detector_scores(real_va, cols, imp_s, sc_s, model_s)
    score_real_test_s = detector_scores(real_te, cols, imp_s, sc_s, model_s)

    thr_syn = threshold_for_fpr(score_syn_val, target_fpr)
    mode_synthetic = {
        "model_fit": "synthetic_train",
        "threshold_fit": "synthetic_validation",
        "threshold": thr_syn,
        "synthetic_validation": rejection_summary(score_syn_val, thr_syn),
        "synthetic_test": rejection_summary(score_syn_test, thr_syn),
        "real_test": rejection_summary(score_real_test_s, thr_syn),
        "real_test_temporal_stability": temporal_block_fpr(
            real_te, score_real_test_s, thr_syn
        ),
    }

    # Mode 2: keep the synthetic model unchanged, but calibrate its operational
    # threshold on a disjoint real NORMAL validation block.
    thr_real = threshold_for_fpr(score_real_val_s, target_fpr)
    mode_threshold = {
        "model_fit": "synthetic_train",
        "threshold_fit": "real_validation",
        "threshold": thr_real,
        "real_validation": rejection_summary(score_real_val_s, thr_real),
        "real_test": rejection_summary(score_real_test_s, thr_real),
        "synthetic_test_at_real_threshold": rejection_summary(score_syn_test, thr_real),
        "real_test_temporal_stability": temporal_block_fpr(
            real_te, score_real_test_s, thr_real
        ),
    }

    # Mode 3: learn the normal-domain preprocessing and Isolation Forest exclusively
    # on real TRAIN, set the threshold on real VALIDATION, and touch real TEST once.
    imp_r, sc_r, model_r = fit_detector(real_tr, cols, seed)
    score_real_train_r = detector_scores(real_tr, cols, imp_r, sc_r, model_r)
    score_real_val_r = detector_scores(real_va, cols, imp_r, sc_r, model_r)
    score_real_test_r = detector_scores(real_te, cols, imp_r, sc_r, model_r)
    thr_real_model = threshold_for_fpr(score_real_val_r, target_fpr)
    mode_real_normal = {
        "model_fit": "real_train_normal_only",
        "threshold_fit": "real_validation",
        "threshold": thr_real_model,
        "real_train_at_validation_threshold": rejection_summary(
            score_real_train_r, thr_real_model
        ),
        "real_validation": rejection_summary(score_real_val_r, thr_real_model),
        "real_test": rejection_summary(score_real_test_r, thr_real_model),
        "real_test_temporal_stability": temporal_block_fpr(
            real_te, score_real_test_r, thr_real_model
        ),
    }

    return {
        "features": cols,
        "domain_shift_train_only": {
            "median_ks_stat": float(np.median(ks_values)),
            "median_real_inside_synth_99pct": float(np.median(coverage)),
            "per_feature": shift,
        },
        "modes": {
            "synthetic_only": mode_synthetic,
            "threshold_calibration": mode_threshold,
            "real_normal_calibration": mode_real_normal,
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2013-11-01")
    ap.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    ap.add_argument("--max-real-windows", type=int, default=120000)
    ap.add_argument("--synthetic-days", type=float, default=0.5)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default=".cache/real_cdr")
    ap.add_argument("--output", default="results/ss7_real_cdr_calibration.json")
    ap.add_argument("--summary-csv", default="results/ss7_real_cdr_calibration.csv")
    args = ap.parse_args(list(argv) if argv is not None else None)

    url = mirror_url(args.date)
    cache = Path(args.cache_dir)
    raw_path = cache / f"{args.date}.csv"
    downloaded, prefix_sha256 = download_prefix(url, raw_path, args.max_bytes)

    real, real_meta = load_real_cdr_features(
        raw_path, args.max_real_windows, args.seed
    )
    real_tr, real_va, real_te, real_split = real_time_split(real)

    syn = synthetic_normal(args.seed, args.synthetic_days)
    syn_tr, syn_va, syn_te = time_split(syn)

    profiles = {
        "full_proxy": evaluate_profile(
            syn_tr, syn_va, syn_te,
            real_tr, real_va, real_te,
            FULL_PROXY_FEATURES, args.seed, args.target_fpr,
        ),
        "ratio_proxy": evaluate_profile(
            syn_tr, syn_va, syn_te,
            real_tr, real_va, real_te,
            RATIO_PROXY_FEATURES, args.seed, args.target_fpr,
        ),
    }

    rows = []
    for profile_name, profile in profiles.items():
        for mode_name, mode in profile["modes"].items():
            test = mode["real_test"]
            stability = mode["real_test_temporal_stability"]
            rows.append({
                "profile": profile_name,
                "mode": mode_name,
                "target_fpr": args.target_fpr,
                "real_test_rows": test["rows"],
                "real_test_rejections": test["rejections"],
                "real_test_fpr": test["fpr"],
                "real_test_fpr_ci_low": test["fpr_wilson_95"][0],
                "real_test_fpr_ci_high": test["fpr_wilson_95"][1],
                "temporal_block_fpr_std": stability["std_fpr"],
                "temporal_block_fpr_min": stability["min_fpr"],
                "temporal_block_fpr_max": stability["max_fpr"],
            })

    ratio_modes = profiles["ratio_proxy"]["modes"]
    target = float(args.target_fpr)
    real_normal_fpr = ratio_modes["real_normal_calibration"]["real_test"]["fpr"]
    threshold_only_fpr = ratio_modes["threshold_calibration"]["real_test"]["fpr"]
    synthetic_only_fpr = ratio_modes["synthetic_only"]["real_test"]["fpr"]

    report = {
        "status": "completed",
        "interpretation": (
            "Three-way calibration experiment on independent non-synthetic normal "
            "Telecom Italia CDR traffic. Real data are split strictly by timestamp "
            "60/20/20 before calibration. No real attack recall/precision/AUC is "
            "claimed because the public CDR mirror has no SS7 attack labels and no "
            "packet-level/cross-node SS7 invariants."
        ),
        "source": {
            "original_dataset": "Telecom Italia Milan telecommunications activity",
            "original_doi": ORIGINAL_DATASET_DOI,
            "mirror": f"https://huggingface.co/datasets/{MIRROR_REPO}",
            "mirror_file": f"sms-call-internet-mi/{args.date}.csv",
            "mirror_revision": MIRROR_REVISION,
            "full_file_sha256_from_mirror": MIRROR_FILE_SHA256.get(args.date),
            "downloaded_prefix_bytes": downloaded,
            "downloaded_prefix_sha256": prefix_sha256,
            "non_synthetic": True,
            "independent_of_repository_generator": True,
        },
        "protocol": {
            "target_fpr": target,
            "real_split": real_split,
            "modes": {
                "synthetic_only": "fit synthetic train; threshold synthetic validation",
                "threshold_calibration": (
                    "fit synthetic train; threshold real validation; no model refit"
                ),
                "real_normal_calibration": (
                    "fit real normal train; threshold real validation; evaluate once on real test"
                ),
            },
        },
        "proxy_mapping": {
            "signalling_equivalent_units": (
                "call activity -> five ISUP message-type units; "
                "SMS-out -> one MAP unit; SMS-in -> two MAP units"
            ),
            "excluded_as_unobservable_in_mirror": [
                "dest_entropy", "n_unique_dest", "dest_top1_ratio",
                "dest_top2_ratio", "transit_share", "dlg_closure_ratio",
                "recon_mean", "recon_min", "recon_absmax", "recon_rms",
                "od_mean", "od_min", "od_absmax", "closure_resid",
                "transit_resid", "integrity_fail",
            ],
        },
        "real_data": real_meta,
        "synthetic_reference": {
            "seed": args.seed,
            "days": args.synthetic_days,
            "rows": int(len(syn)),
            "train_rows": int(len(syn_tr)),
            "validation_rows": int(len(syn_va)),
            "test_rows": int(len(syn_te)),
        },
        "profiles": profiles,
        "primary_ratio_profile": {
            "synthetic_only_real_test_fpr": synthetic_only_fpr,
            "threshold_calibration_real_test_fpr": threshold_only_fpr,
            "real_normal_calibration_real_test_fpr": real_normal_fpr,
            "real_normal_within_2x_target": bool(real_normal_fpr <= 2.0 * target),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = Path(args.summary_csv)
    summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary, index=False)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nSUMMARY")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
