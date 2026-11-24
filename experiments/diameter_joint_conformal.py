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


def hashes(manifest, root):
    result = [digest(root / str(x)) for x in manifest["capture"]]
    if len(set(result)) != len(result):
        raise ValueError("Duplicate PCAP content")
    return set(result)


def split_development(manifest):
    # Historical development-only repartition: fixed by names, not scores.
    # Prospective deployment should collect separate exchangeable units.
    if set(manifest.expected_label) != {"benign"}:
        raise ValueError("Only benign calibration captures allowed")
    if set(manifest.role) != {"reference", "calibration"}:
        raise ValueError("Expected explicit reference/calibration roles")
    inner = [manifest[manifest.role == "reference"]]
    outer = []
    for _, group in manifest[manifest.role == "calibration"].groupby("scenario"):
        group = group.sort_values("capture")
        cut = len(group) // 2
        inner.append(group.iloc[:cut])
        outer.append(group.iloc[cut:])
    import pandas as pd
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


def main():
    import pandas as pd
    from diameter_pcap_native_detector import fit_local_benign_calibration
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
    if hashes(cal, args.calibration_dir) & hashes(ev, args.evaluation_dir):
        raise ValueError("Calibration/evaluation PCAP overlap")
    inner, outer = split_development(cal)
    require_resolution(len(outer), args.alpha)
    with args.model.open("rb") as f:
        bundle = pickle.load(f)
    base = fit_local_benign_calibration(bundle, inner, args.calibration_dir,
                                        args.alpha)
    joint = score_manifest(bundle, base, outer, args.calibration_dir)
    scores = joint.score.tolist()
    # Freeze all model/threshold data before evaluating historical captures.
    frozen = {"alpha": args.alpha, "model_sha256": digest(args.model),
              "base": base, "joint_scores": scores,
              "inner_capture_sha256": sorted(hashes(inner, args.calibration_dir)),
              "joint_capture_sha256": sorted(hashes(outer, args.calibration_dir))}
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
    if pred.pcap_native_pred_anomaly.isna().any():
        raise ValueError("Historical baseline and evaluation mismatch")
    pred.to_csv(args.out / "predictions.csv", index=False)
    comparisons = ["joint_pred", "same_base_uncalibrated_pred",
                   "pcap_native_pred_anomaly"]
    result = {
        "role": "historical development replay, NOT independent confirmation",
        "alpha": args.alpha, "inner_count": len(inner),
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
