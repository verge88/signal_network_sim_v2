"""Freeze the historical Diameter cascades and score independent CU exports offline."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from diameter_report_recurrence import MODEL_NAME, PROFILE_COLUMNS, report_recurrence
from diameter_training_v1 import (
    CascadeCompromiseDetector,
    DataPipeline,
    TrainingConfig,
    calibrate_cascade_across_domains,
    combine_cascade_scores,
    select_compromise_aware_thresholds,
)


KEY = ["node_id", "interval"]
WINDOW_SECONDS = 15
HISTORICAL_RUN = 45599697858
PCAP_MAGIC = {
    bytes.fromhex(magic) for magic in (
        "a1b2c3d4", "d4c3b2a1", "a1b23c4d", "4d3cb2a1", "0a0d0d0a",
    )
}


def checksum(path, decompress=False):
    opener = gzip.open if decompress and path.suffix == ".gz" else open
    with opener(path, "rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def freeze(args):
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    paths = {"train": args.train, "validation": args.validation, "stack": args.stack}
    digests = {role: checksum(path, decompress=True) for role, path in paths.items()}
    if (reference["run"] != 1
            or any(reference["dataset_sha256"][role] != digest for role, digest in digests.items())
            or checksum(args.profile) != reference["calibration_sha256"]
            or len(set(digests.values())) != len(digests)):
        raise ValueError("Training inputs do not match the historical repetition 1 hashes")
    config = TrainingConfig(seed=reference["model_seed"])
    pipeline = DataPipeline(config)
    train = pd.read_csv(args.train)
    pipeline._build_feature_cols(train)
    if len(pipeline.feature_cols) != 56:
        raise ValueError("Expected the original 56-feature cascade")
    x, y = pipeline.get_features_labels(train, fit=True)
    cascade = CascadeCompromiseDetector(config, config.seed)
    cascade.fit(x, y, None, train["is_compromised"].to_numpy())
    recurrence_domains = []
    for role in ("validation", "stack"):
        frame = pd.read_csv(paths[role])
        x, y = pipeline.get_features_labels(frame)
        compromised = frame["attack_type"].eq("node_compromise").to_numpy()
        general, specific = cascade.predict_components(x, None)
        if role == "validation":
            baseline = select_compromise_aware_thresholds(
                general, specific, y, compromised,
                general_fpr_budget=config.compromise_general_fpr_budget,
                total_fpr_budget=config.compromise_total_fpr_budget,
                target_recall=config.compromise_target_recall,
            )[:2]
        recurrence_domains.append((role, general, report_recurrence(frame), y, compromised))
    recurrence = calibrate_cascade_across_domains(
        recurrence_domains, config, config.recurrence_calibration_budget)["thresholds"]
    diagnostics = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    original = diagnostics["validation"]
    cutoffs = {
        "Cascade": (original["general_threshold"], original["compromise_threshold"]),
        MODEL_NAME: tuple(diagnostics["recurrence_validation"]["thresholds"]),
    }
    for name, computed in (("Cascade", baseline), (MODEL_NAME, recurrence)):
        if not np.allclose(computed, cutoffs[name], rtol=0, atol=1e-12):
            raise ValueError(f"{name} thresholds differ from the historical run")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "cascade.joblib"
    joblib.dump({"pipeline": pipeline, "cascade": cascade, "cutoffs": cutoffs},
                path, compress=3)
    write_json(args.out / "model.json", {
        "historical_actions_run": HISTORICAL_RUN,
        "historical_repetition": reference["run"],
        "seed": config.seed,
        "training_sha256": digests,
        "profile_sha256": reference["calibration_sha256"],
        "model_sha256": checksum(path),
        "feature_columns": pipeline.feature_cols,
        "node_type_encoding": {name: int(code) for name, code in
                               train[["node_type", "node_type_enc"]]
                               .drop_duplicates().itertuples(index=False, name=None)},
        "thresholds": {name: [value if np.isfinite(value) else None for value in values]
                       for name, values in cutoffs.items()},
    })


def read_windows(path):
    frame = pd.read_csv(path, dtype={"node_id": "string"}, keep_default_na=False,
                        na_values=[""])
    if frame.empty or not set(KEY).issubset(frame):
        raise ValueError(f"{path.name}: expected node_id and interval on each row")
    intervals = pd.to_numeric(frame["interval"], errors="raise").to_numpy(dtype=float)
    if (frame["node_id"].isna().any() or frame["node_id"].str.strip().eq("").any()
            or not np.isfinite(intervals).all() or np.any(intervals < 0)
            or np.any(intervals != np.floor(intervals)) or np.any(intervals >= 2**53)):
        raise ValueError(f"{path.name}: invalid node or interval ID")
    frame["interval"] = intervals.astype("int64")
    if frame.duplicated(KEY).any():
        raise ValueError(f"{path.name}: duplicate node and interval")
    return frame


def load_session(root, entry, columns):
    if not isinstance(entry.get("source"), str) or not entry["source"].strip():
        raise ValueError("Each session needs a provenance description")
    paths = {name: (root / entry[name]).resolve() for name in ("capture", "features", "labels")}
    if any(not path.is_relative_to(root) for path in paths.values()):
        raise ValueError("Session files must be located under the manifest directory")
    if len(set(paths.values())) != len(paths):
        raise ValueError("The capture, telemetry and labels must be separate files")
    with paths["capture"].open("rb") as stream:
        if stream.read(4) not in PCAP_MAGIC or len(stream.read(24)) < 24:
            raise ValueError("Expected a PCAP/PCAPNG capture")
    features = read_windows(paths["features"])
    labels = read_windows(paths["labels"])
    if set(labels) != {*KEY, "label"} or not labels["label"].isin(
            ["benign", "compromise", "exclude"]).all():
        raise ValueError("Labels must contain only node_id, interval, label")
    if {"label", "is_anomaly", "is_compromised", "attack_type"} & set(features):
        raise ValueError("The feature export must be separate from ground truth")
    missing = set(columns) - set(features)
    if missing:
        raise ValueError(f"Missing cascade inputs: {', '.join(sorted(missing))}")
    features[columns] = features[columns].apply(pd.to_numeric, errors="raise")
    if np.isinf(features[columns].to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature value")
    if not features["cu_delivered"].isin([0, 1]).all():
        raise ValueError("cu_delivered must be 0 or 1")
    profiles = features.loc[features["cu_delivered"].eq(1), list(PROFILE_COLUMNS)]
    if profiles.empty or not profiles.ge(0).all().all() or not profiles.le(1).all().all():
        raise ValueError("Delivered CU reports must contain valid command ratios")
    missing_columns = features.loc[features["cu_delivered"].eq(1), columns].isna().all()
    if missing_columns.any():
        raise ValueError("Entire features unavailable in delivered reports: "
                         + ", ".join(missing_columns.index[missing_columns]))
    frame = features.merge(labels, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not frame["_merge"].eq("both").all():
        raise ValueError("Ground truth must cover exactly the exported windows")
    frame = frame.sort_values(KEY, ignore_index=True)
    steps = frame.groupby("node_id")["interval"].diff().dropna()
    if not steps.eq(1).all():
        raise ValueError("Export every scheduled poll, including failed polls")
    return frame, {name: checksum(path, decompress=True) for name, path in paths.items()}


def counts(frame, alarms):
    active = frame["label"].eq("compromise")
    benign = frame["label"].eq("benign")
    transitions = frame["label"].ne(frame["label"].shift()) | frame["node_id"].ne(
        frame["node_id"].shift()).fillna(True)
    episodes = frame.loc[active, KEY].assign(
        episode=transitions.cumsum()[active], alarm=alarms[active])
    delays = []
    for _, episode in episodes.groupby("episode", sort=False):
        detected = episode.loc[episode["alarm"], "interval"]
        delays.append(int(detected.iloc[0] - episode["interval"].iloc[0]) * WINDOW_SECONDS
                      if len(detected) else None)
    return {
        "compromise_windows": int(active.sum()), "hits": int((active & alarms).sum()),
        "benign_windows": int(benign.sum()), "false_alarms": int((benign & alarms).sum()),
        "episode_delays_s": delays,
    }


def aggregate(rows):
    result = {key: sum(row[key] for row in rows) for key in (
        "compromise_windows", "hits", "benign_windows", "false_alarms")}
    delays = [d for row in rows for d in row["episode_delays_s"]]
    found = [d for d in delays if d is not None]
    result.update({
        "window_recall": result["hits"] / result["compromise_windows"]
        if result["compromise_windows"] else None,
        "benign_fpr": result["false_alarms"] / result["benign_windows"]
        if result["benign_windows"] else None,
        "episodes": len(delays), "episodes_detected": len(found),
        "episode_recall": len(found) / len(delays) if delays else None,
        "median_delay_detected_s": float(np.median(found)) if found else None,
    })
    return result


def paired_intervals(sessions):
    """Resample independent sessions, preserving the models' pairing."""
    rng = np.random.default_rng(42)
    out = {}
    for metric, numerator, denominator in (
        ("window_recall", "hits", "compromise_windows"),
        ("benign_fpr", "false_alarms", "benign_windows"),
    ):
        eligible = [s for s in sessions if s["models"]["Cascade"][denominator]]
        if len(eligible) < 5:
            out[metric] = None
            continue
        differences = np.array([s["models"][MODEL_NAME][numerator]
                                - s["models"]["Cascade"][numerator] for s in eligible])
        totals = np.array([s["models"]["Cascade"][denominator] for s in eligible])
        draw = rng.integers(len(eligible), size=(5000, len(eligible)))
        samples = differences[draw].sum(axis=1) / totals[draw].sum(axis=1)
        out[metric] = np.quantile(samples, [0.025, 0.975]).tolist()
    return out


def evaluate(args):
    model_info = json.loads(args.model.with_name("model.json").read_text(encoding="utf-8"))
    if checksum(args.model) != model_info["model_sha256"]:
        raise ValueError("Frozen model hash mismatch")
    # Only load a bundle produced by the trusted freeze step.
    bundle = joblib.load(args.model)
    pipeline, cascade = bundle["pipeline"], bundle["cascade"]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (manifest.get("window_seconds") != WINDOW_SECONDS
            or manifest.get("data_origin") not in ("independent_testbed", "operator")
            or not manifest.get("sessions")):
        raise ValueError("Expected a manifest for an independent capture and 15-second CU windows")
    sessions = []
    ids, seen_hashes = set(), set(model_info["training_sha256"].values())
    for entry in manifest["sessions"]:
        session_id = entry.get("id")
        if not isinstance(session_id, str) or not session_id.strip() or session_id in ids:
            raise ValueError("Session IDs must be nonempty and unique")
        frame, hashes = load_session(args.manifest.resolve().parent, entry, pipeline.feature_cols)
        observable_hashes = {hashes["capture"], hashes["features"]}
        if seen_hashes.intersection(observable_hashes):
            raise ValueError("A training input or an external session input was reused")
        ids.add(session_id)
        seen_hashes.update(observable_hashes)
        x = pipeline.scaler.transform(pipeline.imputer.transform(frame[pipeline.feature_cols]))
        general, specific = cascade.predict_components(x, None)
        models = {}
        for name, raw in (("Cascade", specific), (MODEL_NAME, report_recurrence(frame))):
            scores = combine_cascade_scores(general, raw, *bundle["cutoffs"][name])
            models[name] = counts(frame, scores >= 1)
        sessions.append({"id": session_id, "source": entry["source"], "sha256": hashes,
                         "models": models, "excluded_windows": int(frame["label"].eq("exclude").sum())})
    summary = {name: aggregate([s["models"][name] for s in sessions])
               for name in bundle["cutoffs"]}
    if not summary["Cascade"]["benign_windows"]:
        raise ValueError("At least one labelled benign window is required")
    write_json(args.out, {
        "data_origin": manifest["data_origin"],
        "manifest_sha256": checksum(args.manifest),
        "model_sha256": model_info["model_sha256"],
        "summary": summary,
        "paired_delta_recurrence_minus_cascade_ci95": paired_intervals(sessions),
        "sessions": sessions,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    freeze_parser = sub.add_parser("freeze", help="Recover the fixed development models")
    for name in ("reference", "diagnostics", "profile", "train", "validation", "stack", "out"):
        freeze_parser.add_argument(f"--{name}", required=True, type=Path)
    freeze_parser.set_defaults(run=freeze)
    eval_parser = sub.add_parser("evaluate", help="Score an independent CU export")
    for name in ("model", "manifest", "out"):
        eval_parser.add_argument(f"--{name}", required=True, type=Path)
    eval_parser.set_defaults(run=evaluate)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
