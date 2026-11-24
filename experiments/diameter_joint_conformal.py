"""Joint calibration of the frozen Diameter local anomaly score.

Fit the base score on reference + inner calibration only; calibrate its
FINAL scalar on a disjoint benign set. Do not tune using evaluation labels.
The finite-sample bound is marginal, conditional on the fitted base score,
and requires exchangeability of joint-calibration and future benign scores.
"""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path


def conformal_p(scores, score):
    scores = list(scores)
    if not scores or not all(math.isfinite(x) for x in [*scores, score]):
        raise ValueError("Nonempty finite calibration and test scores required")
    return (1 + sum(x >= score for x in scores)) / (len(scores) + 1)


def decision(scores, score, alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    return int(conformal_p(scores, score) <= alpha)


def require_resolution(n, alpha):
    if n < 1 or not 0 < alpha < 1 or 1 / (n + 1) > alpha:
        raise ValueError("Insufficient independent calibration units for alpha")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_captures(calibration, evaluation, calibration_root, evaluation_root, decode):
    import pandas as pd

    rows = []
    for dataset, manifest, root in (
        ("calibration", calibration, calibration_root),
        ("evaluation", evaluation, evaluation_root),
    ):
        for record in manifest.to_dict("records"):
            capture = str(record["capture"])
            path = root / capture
            row = {**record, "dataset": dataset, "sha256": None,
                   "decoded_pdus": 0, "status": "missing", "retained_capture": None}
            if path.is_file():
                row.update(sha256=digest(path), status="pending")
            rows.append(row)

    audit = pd.DataFrame(rows).sort_values(["dataset", "capture"], kind="stable")
    candidates = audit[audit.status == "pending"]
    for _, group in candidates.groupby("sha256", sort=False):
        first = group.iloc[0]
        root = calibration_root if first.dataset == "calibration" else evaluation_root
        try:
            decoded = int(decode(root / first.capture)["_decoded_pdus"])
        except RuntimeError as error:
            if not str(error).startswith("No Diameter PDUs decoded from "):
                raise
            decoded = 0
        audit.loc[group.index, "decoded_pdus"] = decoded
        if decoded <= 0:
            audit.loc[group.index, "status"] = "undecodable"
        elif group.dataset.nunique() > 1:
            audit.loc[group.index, "status"] = "cross_dataset_overlap"
        elif any(group[key].nunique(dropna=False) > 1
                 for key in ("expected_label", "role", "scenario")):
            audit.loc[group.index, "status"] = "conflicting_metadata"
        else:
            audit.loc[group.index, ["status", "retained_capture"]] = (
                "duplicate", first.capture)
            audit.loc[group.index[0], "status"] = "retained"
    return audit


def split_development(manifest):
    # The whole archived calibration set is already development data.
    # Prefer former references for the new fit, then fill by capture name.
    if set(manifest.expected_label) != {"benign"}:
        raise ValueError("Only benign calibration captures allowed")
    if not set(manifest.role) <= {"reference", "calibration"}:
        raise ValueError("Expected explicit reference/calibration roles")

    import pandas as pd

    inner, outer = [], []
    for scenario, group in manifest.groupby("scenario", dropna=False):
        if pd.isna(scenario) or len(group) < 10:
            raise ValueError(f"Each archived procedure needs 10 usable fit captures: {scenario}")
        ordered = group.assign(
            archive_role=group.role, reference_first=group.role.ne("reference"),
        ).sort_values(["reference_first", "capture"]).drop(columns="reference_first")
        inner.extend([
            ordered.iloc[:4].assign(role="reference"),
            ordered.iloc[4:10].assign(role="calibration"),
        ])
        outer.append(ordered.iloc[10:].assign(role="joint_calibration"))
    return pd.concat(inner, ignore_index=True), pd.concat(outer, ignore_index=True)


def score_manifest(bundle, base, manifest, root):
    import pandas as pd
    from diameter_pcap_native_detector import (
        pcap_to_native_features, score_with_local_calibration,
    )
    rows = []
    for rec in manifest.to_dict("records"):
        feat = pcap_to_native_features(root / str(rec["capture"]))
        if int(feat["_decoded_pdus"]) <= 0:
            raise ValueError("Undecodable capture: " + str(rec["capture"]))
        s = score_with_local_calibration(bundle, base, feat)
        rows.append({**rec, "score": float(s["local_score"]),
                     "decoded_pdus": int(feat["_decoded_pdus"])})
    return pd.DataFrame(rows)


def wilson(k, n):
    if n == 0:
        return None
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0., center-half), min(1., center+half)]


def metrics(frame, column):
    out = {}
    for label, name in [("attack", "tpr"), ("benign", "fpr")]:
        a = frame[frame.expected_label == label][column]
        n, k = len(a), int(a.sum())
        out[name] = {"n": n, "positive": k, "rate": k/n if n else None,
                     "wilson95": wilson(k,n)}
    return out


def paired_changes(frame, candidate, baseline):
    return {
        label: {
            "added_alarms": int(((group[candidate] == 1) & (group[baseline] == 0)).sum()),
            "removed_alarms": int(((group[candidate] == 0) & (group[baseline] == 1)).sum()),
        }
        for label, group in frame.groupby("expected_label")
    }


def main():
    import pandas as pd
    from diameter_pcap_native_detector import (
        fit_local_benign_calibration, pcap_to_native_features,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument("--historical-predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=.05)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cal = pd.read_csv(args.calibration_dir / "manifest.csv")
    ev = pd.read_csv(args.evaluation_dir / "manifest.csv")
    if set(ev.expected_label) != {"benign", "attack"}:
        raise ValueError("Both evaluation classes required")
    audit = audit_captures(cal, ev, args.calibration_dir, args.evaluation_dir,
                           pcap_to_native_features)
    audit.to_csv(args.out / "capture_audit.csv", index=False)
    counts = (audit.groupby(["dataset", "expected_label", "status"]).size()
              .rename("count").reset_index().to_dict("records"))
    print(json.dumps({"capture_audit": counts}, indent=2), flush=True)
    if audit.status.isin(["missing", "cross_dataset_overlap", "conflicting_metadata"]).any():
        raise ValueError("Missing, overlapping or inconsistently labelled PCAPs; see capture_audit.csv")
    retained = audit[audit.status == "retained"]
    cal = retained.loc[retained.dataset == "calibration", cal.columns]
    ev = retained.loc[retained.dataset == "evaluation", ev.columns]
    if set(ev.expected_label) != {"benign", "attack"}:
        raise ValueError("Both evaluation classes need a decodable capture")
    inner, outer = split_development(cal)
    require_resolution(len(outer), args.alpha)
    print(json.dumps({"reference_count": int((inner.role == "reference").sum()),
                      "inner_calibration_count": int((inner.role == "calibration").sum()),
                      "joint_count": len(outer)}), flush=True)
    with args.model.open("rb") as f:
        bundle = pickle.load(f)
    base = fit_local_benign_calibration(bundle, inner, args.calibration_dir,
                                        args.alpha)
    joint = score_manifest(bundle, base, outer, args.calibration_dir)
    scores = joint.score.tolist()
    # Freeze all model/threshold data before evaluating historical captures.
    frozen = {"alpha": args.alpha, "model_sha256": digest(args.model),
              "base": base, "joint_scores": scores,
              "inner_capture_sha256": sorted(retained.loc[
                  (retained.dataset == "calibration") & retained.capture.isin(inner.capture), "sha256"]),
              "joint_capture_sha256": sorted(retained.loc[
                  (retained.dataset == "calibration") & retained.capture.isin(outer.capture), "sha256"])}
    (args.out / "joint_calibration.json").write_text(json.dumps(frozen, indent=2, allow_nan=False))
    inner.to_csv(args.out / "inner_manifest.csv", index=False)
    joint.to_csv(args.out / "joint_manifest.csv", index=False)
    pred = score_manifest(bundle, base, ev, args.evaluation_dir)
    pred["p_value"] = [conformal_p(scores, s) for s in pred.score]
    pred["joint_pred"] = [decision(scores, s, args.alpha) for s in pred.score]
    pred["same_base_uncalibrated_pred"] = (pred.score > 1).astype(int)
    old = pd.read_csv(args.historical_predictions)
    cols = ["capture", "expected_label", "pcap_native_pred_anomaly"]
    pred = pred.merge(old[cols], on=["capture", "expected_label"], how="left",
                      validate="one_to_one")
    if not pred.pcap_native_pred_anomaly.isin([0, 1]).all():
        raise ValueError("Historical baseline and evaluation mismatch")
    pred.to_csv(args.out / "predictions.csv", index=False)
    comparisons = ["joint_pred", "same_base_uncalibrated_pred",
                   "pcap_native_pred_anomaly"]
    result = {
        "role": "historical development replay, NOT independent confirmation",
        "alpha": args.alpha, "inner_count": len(inner),
        "capture_audit": counts,
        "paired_changes": {
            baseline: paired_changes(pred, "joint_pred", baseline)
            for baseline in comparisons[1:]
        },
        "joint_count": len(outer), "min_p_value": 1/(len(outer)+1),
        "guarantee": "marginal benign FPR under exchangeability; not established by stratified historical replay",
        "metrics": {c: metrics(pred, c) for c in comparisons},
        "per_scenario": {str(s): {c: metrics(g,c) for c in comparisons}
                         for s,g in pred.groupby("scenario")},
        "limitations": ["historical holdout reused only as development",
                        "capture independence not established by hashes",
                        "no operator traffic or prospective final evaluation"]
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
