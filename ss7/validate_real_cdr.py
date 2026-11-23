"""External validity check for the canonical SS7 model on independent real telecom data.

This module intentionally does NOT claim that Telecom Italia CDRs are packet-level SS7
traces or that they contain attack labels.  It uses the public, non-synthetic Milan CDR
dataset only for the subset of SS7 observables that can be reconstructed without
inventing packet-level events.  The check therefore measures synthetic-to-real
distribution shift and real-data rejection/false-alarm propensity of an unsupervised
detector.  Attack recall cannot be estimated from this dataset.

Source:
  Telecom Italia (2015), Telecommunications - SMS, Call, Internet - MI
  Harvard Dataverse, DOI 10.7910/DVN/EGZHFV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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


DATASET_DOI = "doi:10.7910/DVN/EGZHFV"
DATAVERSE = "https://dataverse.harvard.edu"
RAW_COLUMNS = [
    "cell_id", "timestamp_ms", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet",
]

# These are the canonical SS7 own-report features for which the CDR release has a
# defensible external proxy.  transit_share, dlg_closure_ratio and all cross-node
# invariants are deliberately excluded because CDRs do not observe them.
FULL_PROXY_FEATURES = [
    "total_messages",
    "mtype_entropy",
    "dest_entropy",
    "n_unique_dest",
    "dest_top1_ratio",
    "dest_top2_ratio",
    "map_share",
    "isup_share",
    "in_out_ratio",
]
RATIO_PROXY_FEATURES = [
    "mtype_entropy",
    "dest_entropy",
    "dest_top1_ratio",
    "dest_top2_ratio",
    "map_share",
    "isup_share",
    "in_out_ratio",
]


def _get_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "signal-network-sim-v2-external-validation/1.0"})
    with urlopen(req, timeout=60) as r:
        return json.load(r)


def dataverse_file_id(filename: str) -> int:
    url = (
        f"{DATAVERSE}/api/datasets/:persistentId/"
        f"?persistentId={DATASET_DOI.replace('/', '%2F').replace(':', '%3A')}"
    )
    meta = _get_json(url)
    files = meta["data"]["latestVersion"]["files"]
    for item in files:
        df = item.get("dataFile", {})
        if df.get("filename") == filename or df.get("label") == filename:
            return int(df["id"])
    available = sorted(
        (x.get("dataFile", {}).get("filename") or x.get("dataFile", {}).get("label") or "")
        for x in files
    )
    raise RuntimeError(f"Dataverse file {filename!r} not found; first entries={available[:5]}")


def download_prefix(file_id: int, dst: Path, max_bytes: int) -> Tuple[int, str]:
    """Download at most max_bytes from the real Dataverse file and trim partial line.

    A byte prefix is used so CI does not need to transfer a 300+ MB daily file.
    Harvard Dataverse rejects Range requests for this endpoint, so the client opens
    the normal stream and stops reading after max_bytes.  No rows are generated or
    altered; only the final incomplete line is discarded.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{DATAVERSE}/api/access/datafile/{file_id}"
    req = Request(
        url,
        headers={"User-Agent": "signal-network-sim-v2-external-validation/1.0"},
    )
    remaining = int(max_bytes)
    h = hashlib.sha256()
    written = 0
    with urlopen(req, timeout=120) as r, dst.open("wb") as out:
        while remaining > 0:
            chunk = r.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            written += len(chunk)
            remaining -= len(chunk)

    # Ensure pandas never sees a truncated final record.
    with dst.open("rb+") as f:
        data = f.read()
        end = data.rfind(b"\n")
        if end <= 0:
            raise RuntimeError("Downloaded prefix contains no complete CDR rows")
        f.seek(0)
        f.write(data[: end + 1])
        f.truncate()
    trimmed = dst.stat().st_size
    return trimmed, h.hexdigest()


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


def load_real_cdr_features(path: Path, max_windows: int, seed: int) -> Tuple[pd.DataFrame, dict]:
    usecols = RAW_COLUMNS[:7]
    dtypes = {
        "cell_id": "Int32",
        "timestamp_ms": "Int64",
        "country_code": "Float64",
        "sms_in": "float64",
        "sms_out": "float64",
        "call_in": "float64",
        "call_out": "float64",
    }
    raw = pd.read_csv(
        path,
        sep="\t",
        names=RAW_COLUMNS,
        usecols=usecols,
        dtype=dtypes,
        header=None,
        on_bad_lines="skip",
        low_memory=False,
    )
    raw = raw.dropna(subset=["cell_id", "timestamp_ms"]).copy()
    activity_cols = ["sms_in", "sms_out", "call_in", "call_out"]
    raw[activity_cols] = raw[activity_cols].fillna(0.0).clip(lower=0.0)
    raw["country_code"] = raw["country_code"].fillna(-1.0)
    raw["dest_weight"] = raw[activity_cols].sum(axis=1)

    keys = ["cell_id", "timestamp_ms"]
    agg = raw.groupby(keys, sort=False)[activity_cols].sum()

    # Country code is NOT an SS7 DPC.  It is used only as an independently observed
    # destination-diversity proxy, and this limitation is carried into the report.
    dest = (
        raw.groupby(keys + ["country_code"], sort=False)["dest_weight"]
        .sum()
        .reset_index()
    )
    dest = dest[dest["dest_weight"] > 0].copy()
    if dest.empty:
        raise RuntimeError("No positive call/SMS activity in downloaded CDR prefix")
    dest["dest_total"] = dest.groupby(keys)["dest_weight"].transform("sum")
    dest["p"] = dest["dest_weight"] / dest["dest_total"]
    dest["entropy_term"] = -dest["p"] * np.log2(dest["p"])
    dest["rank"] = dest.groupby(keys)["dest_weight"].rank(method="first", ascending=False)

    dest_metrics = pd.DataFrame({
        "dest_entropy": dest.groupby(keys)["entropy_term"].sum(),
        "n_unique_dest": dest.groupby(keys).size().astype(float),
        "dest_top1_ratio": dest.loc[dest["rank"] == 1].set_index(keys)["p"],
        "dest_top2_ratio": dest.loc[dest["rank"] <= 2].groupby(keys)["p"].sum(),
    })

    out = agg.join(dest_metrics, how="inner").reset_index()
    call_total = (out["call_in"] + out["call_out"]).to_numpy(float)
    sms_in = out["sms_in"].to_numpy(float)
    sms_out = out["sms_out"].to_numpy(float)

    # CDR activity values are anonymised activity measures, not literal MSU counts.
    # We therefore construct "signalling-equivalent units" only to preserve the
    # canonical service template proportions: ISUP call = 5 message types,
    # MO-SMS = 1 MAP operation, MT-SMS = 2 MAP operations.
    parts = np.column_stack([
        call_total, call_total, call_total, call_total, call_total,
        sms_out, sms_in, sms_in,
    ])
    total = parts.sum(axis=1)
    out["total_messages"] = total
    out["mtype_entropy"] = _shannon_rows(parts)
    map_units = sms_out + 2.0 * sms_in
    isup_units = 5.0 * call_total
    denom = np.maximum(total, 1.0)
    out["map_share"] = map_units / denom
    out["isup_share"] = isup_units / denom
    incoming = 5.0 * out["call_in"].to_numpy(float) + 2.0 * sms_in
    outgoing = 5.0 * out["call_out"].to_numpy(float) + sms_out
    out["in_out_ratio"] = incoming / np.maximum(outgoing, 1.0)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=FULL_PROXY_FEATURES)
    out = out[out["total_messages"] > 0].copy()
    if len(out) < 100:
        raise RuntimeError(f"Only {len(out)} usable real CDR windows; need at least 100")

    if len(out) > max_windows:
        out = out.sample(n=max_windows, random_state=seed).sort_values(keys)

    meta = {
        "raw_rows": int(len(raw)),
        "usable_windows": int(len(out)),
        "unique_cells": int(out["cell_id"].nunique()),
        "unique_intervals": int(out["timestamp_ms"].nunique()),
        "timestamp_min_ms": int(out["timestamp_ms"].min()),
        "timestamp_max_ms": int(out["timestamp_ms"].max()),
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


def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = np.sort(df["t"].unique())
    if len(ts) < 10:
        raise RuntimeError("Synthetic sample has too few time intervals")
    a = ts[max(1, int(len(ts) * 0.60)) - 1]
    b = ts[max(2, int(len(ts) * 0.80)) - 1]
    tr = df[df["t"] <= a]
    va = df[(df["t"] > a) & (df["t"] <= b)]
    te = df[df["t"] > b]
    if min(len(tr), len(va), len(te)) < 100:
        raise RuntimeError("Synthetic temporal split is too small")
    return tr, va, te


def _feature_shift(
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
        coverage = float(np.mean((r >= lo) & (r <= hi)))
        out[c] = {
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "median_shift_sigma": float(abs(np.median(r) - np.median(s))),
            "real_inside_synth_99pct": coverage,
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
    sva = -model.score_samples(xva)
    ste = -model.score_samples(xte)
    sreal = -model.score_samples(xreal)
    threshold = float(np.quantile(sva, 1.0 - target_fpr))
    synth_test_fpr = float(np.mean(ste > threshold))
    real_rejection = float(np.mean(sreal > threshold))

    shift = _feature_shift(tr, real, cols, imp, scaler)
    ks_values = np.array([v["ks_stat"] for v in shift.values()], dtype=float)
    cover = np.array([v["real_inside_synth_99pct"] for v in shift.values()], dtype=float)
    med_shift = np.array([v["median_shift_sigma"] for v in shift.values()], dtype=float)

    # Diagnostic, not a theorem: it flags a clear external-validity problem when
    # unlabeled real windows are rejected far above the calibration rate or when
    # the common-feature distributions barely overlap.
    transfer_warning = bool(
        real_rejection > max(0.05, 5.0 * target_fpr)
        or float(np.median(ks_values)) > 0.50
        or float(np.median(cover)) < 0.50
    )

    top_shift = sorted(
        (
            {
                "feature": c,
                **vals,
            }
            for c, vals in shift.items()
        ),
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
        "median_real_inside_synth_99pct": float(np.median(cover)),
        "median_abs_shift_sigma": float(np.median(med_shift)),
        "transfer_warning": transfer_warning,
        "top_feature_shift": top_shift[:5],
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

    filename = f"sms-call-internet-mi-{args.date}.txt"
    file_id = dataverse_file_id(filename)
    cache = Path(args.cache_dir)
    raw_path = cache / filename
    downloaded, sha256 = download_prefix(file_id, raw_path, args.max_bytes)

    real, real_meta = load_real_cdr_features(raw_path, args.max_real_windows, args.seed)
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
            "External non-synthetic normal-traffic check only. "
            "No attack recall/precision is claimed because the public CDR dataset "
            "has no SS7 attack labels and does not expose cross-node SS7 invariants."
        ),
        "source": {
            "dataset": "Telecom Italia Milan telecommunications activity",
            "doi": "10.7910/DVN/EGZHFV",
            "provider": "Harvard Dataverse",
            "file": filename,
            "dataverse_file_id": file_id,
            "downloaded_prefix_bytes": downloaded,
            "download_sha256_before_final_line_trim": sha256,
            "non_synthetic": True,
            "independent_of_repository_generator": True,
        },
        "proxy_mapping": {
            "country_code_as_destination_proxy": True,
            "signalling_equivalent_units": (
                "call activity -> five ISUP message-type units; "
                "SMS-out -> one MAP unit; SMS-in -> two MAP units"
            ),
            "excluded_as_unobservable": [
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
        "overall_transfer_warning": bool(any(p["transfer_warning"] for p in profiles.values())),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
