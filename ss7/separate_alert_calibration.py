"""Paired SS7 development study of window and notification thresholds."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ss7.causal_temporal_calibration import cutoffs
from ss7.incident_calibration_study import (
    INCIDENT_BUDGET, RAW_BUDGET, WARMUP, metrics, triple,
)
from ss7.threshold_transfer_study import score, world


SEEDS = tuple(range(9101, 9109))
MODELS = ("strict_local", "report_full")
POLICIES = ("shared", "separate")
REFERENCE_FRACTION = 0.5
MODEL_RUNS = (45440533210, 45421952080)


def thresholds(references, scores):
    """Use the strictest reference independently for each output."""
    pairs = [
        cutoffs(d, s, RAW_BUDGET * REFERENCE_FRACTION,
                INCIDENT_BUDGET * REFERENCE_FRACTION)
        for d, s in zip(references, scores, strict=True)
    ]
    raw, notice = map(max, zip(*pairs, strict=True))
    shared = max(raw, notice)
    return {"shared": (shared, shared), "separate": (raw, notice)}


def evaluate(seed, local_path, report_path, out):
    if seed not in SEEDS:
        raise ValueError("unexpected development seed")

    test, sim, episodes = world(seed, 1.5, 32, 64, 10)
    eligible = {}
    for episode_id, ep in enumerate(episodes):
        if ep.kind != "compromise" or ep.t0 < WARMUP:
            continue
        rows = test[(test.episode_id == episode_id) & (test.nid == ep.nid)]
        if len(rows) == ep.t1 - ep.t0 + 1 and rows.t.nunique() == len(rows):
            eligible[episode_id] = ep
    if len(eligible) < 12:
        raise ValueError("insufficient complete episodes")

    reference_seeds = (seed + 10000, seed + 20000, seed + 40000)
    matched = world(reference_seeds[0], 3., 0, 128, 20, sim)[0]
    stress = world(reference_seeds[1], 3., 0, 192, 32, sim)[0]
    mixed = world(reference_seeds[2], 3., 64, 128, 20, sim)[0]
    references = (matched, stress, mixed)
    if matched.label.any() or stress.label.any() or not mixed.label.any():
        raise ValueError("unexpected reference labels")
    if any(set(ref.nid) != set(test.nid) for ref in references):
        raise ValueError("reference topology mismatch")

    models = {
        "strict_local": joblib.load(local_path)["strict_local"],
        "report_full": joblib.load(report_path)["report_full"],
    }
    results, events = [], []
    for name, model in models.items():
        test_score = score(model, test)
        confirmation_score = triple(test, test_score)
        ref_scores = [score(model, ref) for ref in references]
        for policy, (raw, notice) in thresholds(references, ref_scores).items():
            result, detected = metrics(
                test, test_score, confirmation_score, raw, eligible,
                notification_threshold=notice,
            )
            results.append(dict(
                seed=seed, model=name, policy=policy, notification_threshold=notice,
                **result,
            ))
            events.extend(
                dict(seed=seed, model=name, policy=policy, episode=i,
                     detected=int(i in detected), delay_seconds=detected.get(i))
                for i in eligible
            )
            print(seed, name, policy, "recall", result["recall"],
                  "false windows/day", result["false_windows_day"],
                  "false notifications/day", result["false_notifications_day"],
                  flush=True)

    pd.DataFrame(results).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(events).to_csv(out / "episodes.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(dict(
        test_seed=seed, reference_seeds=reference_seeds,
        model_runs=dict(zip(MODELS, MODEL_RUNS, strict=True)),
        eligible_episodes=len(eligible), generated_episodes=32,
        network_budgets_per_day={"windows": RAW_BUDGET, "notifications": INCIDENT_BUDGET},
        reference_fraction=REFERENCE_FRACTION,
        test_scope="development on independent synthetic worlds",
    ), indent=2) + "\n", encoding="utf-8")


def collect(root, name):
    paths = sorted(root.glob(f"separate-alert-seed-*/{name}.csv"))
    if len(paths) != len(SEEDS):
        raise ValueError(f"incomplete registered inputs for {name}")
    parts = []
    for path in paths:
        seed = int(path.parent.name.rsplit("-", 1)[-1])
        part = pd.read_csv(path)
        if set(part.seed) != {seed}:
            raise ValueError(f"seed mismatch in {path}")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def aggregate(root, out):
    results = collect(root, "metrics")
    events = collect(root, "episodes")
    keys = ["seed", "model", "policy"]
    expected = {(s, m, p) for s in SEEDS for m in MODELS for p in POLICIES}
    for frame, unique_keys in ((results, keys), (events, keys + ["episode"])):
        if frame.duplicated(unique_keys).any() or set(
            frame[keys].itertuples(index=False, name=None)
        ) != expected:
            raise ValueError("invalid paired design")
    for _, part in events.groupby("seed", sort=False):
        supports = [set(group.episode) for _, group in part.groupby(["model", "policy"])]
        if any(support != supports[0] for support in supports[1:]):
            raise ValueError("unequal episode support across paired policies")
    if not events.detected.isin([0, 1]).all():
        raise ValueError("invalid episode outcome")
    counts = events.groupby(keys).detected.agg(["size", "sum"])
    audit = results.set_index(keys).join(counts)
    if not ((audit.episodes == audit["size"]) & (audit.detected == audit["sum"])).all():
        raise ValueError("episode and world totals disagree")

    results["both_budgets"] = (
        (results.false_windows_day <= RAW_BUDGET)
        & (results.false_notifications_day <= INCIDENT_BUDGET)
    )
    summary = results.groupby(["model", "policy"]).agg(
        worlds=("seed", "size"), complete_episodes=("episodes", "sum"),
        detected=("detected", "sum"), recall=("recall", "mean"),
        raw_day=("false_windows_day", "mean"), worst_raw=("false_windows_day", "max"),
        notice_day=("false_notifications_day", "mean"),
        worst_notice=("false_notifications_day", "max"),
        both_budgets=("both_budgets", "sum"),
        median_world_delay_seconds=("median_delay_seconds", "median"),
    ).reset_index()

    worlds = results.set_index(keys)
    episodes = events.set_index(keys + ["episode"])
    resample = np.random.default_rng(9100).integers(0, len(SEEDS), size=(10000, len(SEEDS)))
    comparisons = (
        ("report_full", "separate", "report_full", "shared"),
        ("strict_local", "separate", "strict_local", "shared"),
        ("report_full", "separate", "strict_local", "separate"),
        ("report_full", "separate", "strict_local", "shared"),
    )
    paired = []
    for model, policy, comparator, baseline in comparisons:
        a = (model, policy)
        b = (comparator, baseline)
        select = ["model", "policy"]
        wa, wb = (worlds.xs(v, level=select).sort_index() for v in (a, b))
        ea, eb = (episodes.xs(v, level=select).sort_index() for v in (a, b))
        delta = (wa.recall - wb.recall).to_numpy()
        lo, hi = np.quantile(delta[resample].mean(axis=1), (0.025, 0.975))
        both_detected = ea.detected.eq(1) & eb.detected.eq(1)
        paired.append(dict(
            model=model, policy=policy, comparator=comparator, baseline=baseline,
            recall_delta=float(delta.mean()), recall_delta_lo=lo, recall_delta_hi=hi,
            raw_day_delta=(wa.false_windows_day - wb.false_windows_day).mean(),
            notice_day_delta=(wa.false_notifications_day - wb.false_notifications_day).mean(),
            candidate_only=int((ea.detected > eb.detected).sum()),
            comparator_only=int((ea.detected < eb.detected).sum()),
            both_detected=int(both_detected.sum()),
            paired_median_delay_delta_seconds=(
                ea.delay_seconds[both_detected] - eb.delay_seconds[both_detected]
            ).median(),
        ))
    comparisons_table = pd.DataFrame(paired)
    for name, frame in (
        ("world_metrics", results), ("episodes", events),
        ("summary", summary), ("paired_comparisons", comparisons_table),
    ):
        frame.to_csv(out / f"{name}.csv", index=False)
    print(summary.to_string(index=False))
    print(comparisons_table.to_string(index=False))
    print("Development only. Paired bootstrap resamples whole worlds.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    test = sub.add_parser("test")
    test.add_argument("--seed", type=int, choices=SEEDS, required=True)
    test.add_argument("--local", type=Path, required=True)
    test.add_argument("--report", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    combine = sub.add_parser("aggregate")
    combine.add_argument("--input", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "test":
        evaluate(args.seed, args.local, args.report, args.output)
    else:
        aggregate(args.input, args.output)


if __name__ == "__main__":
    main()
