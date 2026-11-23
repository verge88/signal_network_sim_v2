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

    if len(out) > max_windows:
        out = out.sample(n=max_windows, random_state=seed)

    meta = {
        "raw_rows_in_prefix": int(len(raw)),
        "usable_windows": int(len(out)),
        "unique_cells": int(out["cell_id"].nunique()),
        "timestamp_min": str(out["datetime"].min()),
        "timestamp_max": str(out["datetime"].max()),
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
    for j, c in enumerate(cols):
        s = sx[:, j]
        r = rx[:, j]
        ks = ks_2samp(s, r, alternative="two-sided", mode="auto")
        lo, hi = np.quantile(s, [0.005, 0.995])
        out[c] = {
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "median_shift_sigma": float(abs(np.median(r) - np.median(s))),
            "real_inside_synth_99pct": float(np.mean((r >= lo) & (r <= hi))),
        }
    return out


def validate_profile(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    te: pd.DataFrame,
    real: pd.DataFrame,
    cols: List[str],
    seed: int,
    target_fpr: float,
) -> dict:
    imp = SimpleImputer(strategy="median").fit(tr[cols])
    scaler = StandardScaler().fit(imp.transform(tr[cols]))

    def tx(frame: pd.DataFrame) -> np.ndarray:
        return scaler.transform(imp.transform(frame[cols]))

    xtr, xva, xte, xreal = tx(tr), tx(va), tx(te), tx(real)
    model = IsolationForest(
        n_estimators=400,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    ).fit(xtr)

    score_val = -model.score_samples(xva)
    score_test = -model.score_samples(xte)
    score_real = -model.score_samples(xreal)
    threshold = float(np.quantile(score_val, 1.0 - target_fpr))

    synth_test_fpr = float(np.mean(score_test > threshold))
    real_rejection = float(np.mean(score_real > threshold))
    shift = feature_shift(tr, real, cols, imp, scaler)

    ks_values = np.array([v["ks_stat"] for v in shift.values()], dtype=float)
    coverage = np.array(
        [v["real_inside_synth_99pct"] for v in shift.values()], dtype=float
    )
    med_shift = np.array(
        [v["median_shift_sigma"] for v in shift.values()], dtype=float
    )

    transfer_warning = bool(
        real_rejection > max(0.05, 5.0 * target_fpr)
        or float(np.median(ks_values)) > 0.50
        or float(np.median(coverage)) < 0.50
    )
    top_shift = sorted(
        ({"feature": c, **vals} for c, vals in shift.items()),
        key=lambda x: (x["ks_stat"], x["median_shift_sigma"]),
        reverse=True,
    )

    return {
        "features": cols,
        "target_synthetic_fpr": float(target_fpr),
        "threshold": threshold,
        "synthetic_test_fpr": synth_test_fpr,
        "real_rejection_rate_unlabeled": real_rejection,
        "median_ks_stat": float(np.median(ks_values)),
        "median_real_inside_synth_99pct": float(np.median(coverage)),
        "median_abs_shift_sigma": float(np.median(med_shift)),
        "transfer_warning": transfer_warning,
        "top_feature_shift": top_shift,
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2013-11-01")
    ap.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    ap.add_argument("--max-real-windows", type=int, default=60000)
    ap.add_argument("--synthetic-days", type=float, default=0.5)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default=".cache/real_cdr")
    ap.add_argument("--output", default="results/ss7_real_cdr_validation.json")
    args = ap.parse_args(list(argv) if argv is not None else None)

    url = mirror_url(args.date)
    cache = Path(args.cache_dir)
    raw_path = cache / f"{args.date}.csv"
    downloaded, prefix_sha256 = download_prefix(url, raw_path, args.max_bytes)

    real, real_meta = load_real_cdr_features(
        raw_path, args.max_real_windows, args.seed
    )
    syn = synthetic_normal(args.seed, args.synthetic_days)
    tr, va, te = time_split(syn)

    profiles = {
        "full_proxy": validate_profile(
            tr, va, te, real, FULL_PROXY_FEATURES, args.seed, args.target_fpr
        ),
        "ratio_proxy": validate_profile(
            tr, va, te, real, RATIO_PROXY_FEATURES, args.seed, args.target_fpr
        ),
    }

    report = {
        "status": "completed",
        "interpretation": (
            "Independent non-synthetic normal-traffic external-validity check. "
            "No real attack recall/precision/AUC is claimed because this public CDR "
            "mirror contains no SS7 attack labels and no packet-level/cross-node "
            "SS7 invariants."
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
        "proxy_mapping": {
            "signalling_equivalent_units": (
                "call activity -> five ISUP message-type units; "
                "SMS-out -> one MAP unit; SMS-in -> two MAP units"
            ),
            "excluded_as_unobservable_in_mirror": [
                "dest_entropy",
                "n_unique_dest",
                "dest_top1_ratio",
                "dest_top2_ratio",
                "transit_share",
                "dlg_closure_ratio",
                "recon_mean",
                "recon_min",
                "recon_absmax",
                "recon_rms",
                "od_mean",
                "od_min",
                "od_absmax",
                "closure_resid",
                "transit_resid",
                "integrity_fail",
            ],
        },
        "real_data": real_meta,
        "synthetic_reference": {
            "seed": args.seed,
            "days": args.synthetic_days,
            "rows": int(len(syn)),
            "train_rows": int(len(tr)),
            "validation_rows": int(len(va)),
            "test_rows": int(len(te)),
        },
        "profiles": profiles,
        "overall_transfer_warning": bool(
            any(p["transfer_warning"] for p in profiles.values())
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
