"""Compare the two Diameter cascade configurations on separate simulation runs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from diameter_report_recurrence import (
    MODEL_NAME, report_recurrence, validate_compromise_state,
)
from diameter_training_v1 import (
    CascadeCompromiseDetector, DataPipeline, TrainingConfig, combine_cascade_scores,
    evaluate_model, select_compromise_aware_thresholds,
)


def load_partition(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    frame = pd.read_csv(path)
    validate_compromise_state(frame)
    if not frame["is_compromised"].eq(1).any() or not frame["attack_type"].eq("none").any():
        raise ValueError(f"Missing compromise or benign controls in {path}")
    return frame, checksum


def trial_metrics(frame, scores, model):
    y = frame["is_anomaly"].to_numpy()
    labels = frame["attack_type"].to_numpy()
    metrics = evaluate_model(model, scores, y, labels, threshold=1.0)
    per_type = {label: {"alarms": int(hit), "windows": int(total)}
                for label, (hit, total, _) in metrics["per_attack"].items()}
    compromise = per_type["node_compromise"]
    episodes = []
    for node_id, group in frame.loc[
        frame["attack_type"].eq("node_compromise")
    ].assign(alarm=metrics["y_pred"][labels == "node_compromise"]).groupby("node_id"):
        ordered = group.sort_values("interval")
        starts = ordered["interval"].diff().ne(1).cumsum()
        for _, episode in ordered.groupby(starts):
            active = episode.loc[episode["alarm"].eq(1), "time_s"]
            episodes.append({
                "node_id": int(node_id),
                "windows": len(episode),
                "delay_seconds": (float(active.iloc[0] - episode["time_s"].iloc[0])
                                  if not active.empty else None),
            })
    return {
        "compromise_hits": compromise["alarms"], "compromise_windows": compromise["windows"],
        "compromise_recall": compromise["alarms"] / compromise["windows"],
        "false_alarms": int(metrics["fp"]), "benign_windows": int(metrics["tn"] + metrics["fp"]),
        "benign_fpr": float(metrics["fpr"]), "auc_pr": float(metrics["auc_pr"]),
        "f1": float(metrics["f1"]), "per_attack": per_type, "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train, train_hash = load_partition(args.train)
    val, val_hash = load_partition(args.validation)
    seen = {train_hash, val_hash}
    if len(seen) != 2 or train.equals(val):
        raise ValueError("Training and validation simulations must differ")
    config = TrainingConfig()
    pipeline = DataPipeline(config)
    pipeline._build_feature_cols(train)
    X, y = pipeline.get_features_labels(train, fit=True)
    Xval, yval = pipeline.get_features_labels(val)
    cascade = CascadeCompromiseDetector(config, config.seed)
    cascade.fit(X, y, None, train["is_compromised"].to_numpy())
    general, hist = cascade.predict_components(Xval, None)
    validation = {}
    for model, score in (("Cascade", hist), (MODEL_NAME, report_recurrence(val))):
        t1, t2, _, diag = select_compromise_aware_thresholds(
            general, score, yval, val["is_compromised"].to_numpy(),
            general_fpr_budget=config.compromise_general_fpr_budget,
            total_fpr_budget=config.compromise_total_fpr_budget,
            target_recall=config.compromise_target_recall,
        )
        validation[model] = {"thresholds": [t1, t2], **diag}

    # Both decision rules are fixed before any test dataset is read.
    tests = []
    for path in args.test:
        frame, checksum = load_partition(path)
        if checksum in seen or frame.equals(train) or frame.equals(val):
            raise ValueError(f"Simulation reused across partitions: {path}")
        seen.add(checksum)
        Xt, _ = pipeline.get_features_labels(frame)
        stage1, stage2 = cascade.predict_components(Xt, None)
        evaluated = {}
        for model, raw in (("Cascade", stage2), (MODEL_NAME, report_recurrence(frame))):
            combined = combine_cascade_scores(stage1, raw, *validation[model]["thresholds"])
            evaluated[model] = trial_metrics(frame, combined, model)
        tests.append({"simulation": path.stem.removesuffix(".csv"), "sha256": checksum,
                      "models": evaluated})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"train_sha256": train_hash, "validation_sha256": val_hash,
              "validation": validation, "tests": tests}
    as_json = json.dumps(result, indent=2, allow_nan=False)
    args.out.write_text(as_json + "\n", encoding="utf-8")
    print(as_json)


if __name__ == "__main__":
    main()
