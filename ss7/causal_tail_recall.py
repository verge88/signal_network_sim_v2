"""Causal accumulation of locally unusual SS7 telemetry scores."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ss7 import observability_audit_development as obs
from ss7.episode_ranking_development import collect

VALIDATE = tuple(range(15101, 15107))
CONFIRM = tuple(range(15201, 15217))
BASE = "role_expanded"
WINDOWS = {"tail_8": (1, 4, 8), "tail_32": (1, 4, 12, 32)}
CANDIDATES = (BASE, *WINDOWS)


def reference_tails(frame, scores):
    """Per-node empirical score distribution from a benign reference."""
    if frame.label.any():
        raise ValueError("reference must be benign")
    return {node: np.sort(scores[indices]) for node, indices
            in frame.groupby("nid", sort=False).indices.items()}


def accumulated(frame, scores, tails, widths):
    if len(frame) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("invalid score array")
    if frame.duplicated(["nid", "t"]).any():
        raise ValueError("duplicate node-window")
    output = np.empty(len(frame))
    for node, indices in frame.groupby("nid", sort=False).indices.items():
        times = frame.t.to_numpy()[indices]
        if np.any(np.diff(times) <= 0):
            raise ValueError("unsorted node windows")
        ref = tails[node]
        # Smoothed upper-tail probability; count ties conservatively.
        above = len(ref) - ref.searchsorted(scores[indices], side="left")
        evidence = -np.log((1.0 + above) / (1.0 + len(ref)))
        cuts = np.flatnonzero(np.diff(times) != 1) + 1
        for block in np.split(np.arange(len(indices)), cuts):
            values = evidence[block]
            pooled = values.copy()
            sums = np.r_[0., np.cumsum(values - 1.)]
            for width in widths:
                if width > 1 and len(block) >= width:
                    statistic = (sums[width:] - sums[:-width]) / np.sqrt(width)
                    pooled[width-1:] = np.maximum(pooled[width-1:], statistic)
            output[indices[block]] = pooled
    return output


def evaluate(seed, models, names, out):
    test, sim, episodes = obs.generate(seed, 1.5, 32, 64, 10)
    eligible = {}
    for i, ep in enumerate(episodes):
        if ep.kind == "compromise" and ep.t0 >= obs.WARMUP:
            rows = test[(test.episode_id == i) & (test.nid == ep.nid)]
            if len(rows) == ep.t1 - ep.t0 + 1 and rows.t.nunique() == len(rows):
                eligible[i] = ep
    if len(eligible) < 12:
        raise ValueError("insufficient complete episodes")
    specs = [(10000, 0, 128, 20), (20000, 0, 192, 32), (40000, 64, 128, 20)]
    def compact(frame):
        scores = obs.predict(frame, models, BASE)
        return frame[["nid", "t", "episode_id", "label"]].copy(), scores
    test_data = compact(test)
    del test
    references = [compact(obs.generate(seed + offset, 3., comp, normal, ext, sim)[0])
                  for offset, comp, normal, ext in specs]
    frames, raw = zip(*references, test_data)
    tails = reference_tails(frames[0], raw[0])
    rows, events = [], []
    for name in names:
        scores = raw if name == BASE else [
            accumulated(frame, values, tails, WINDOWS[name])
            for frame, values in zip(frames, raw, strict=True)]
        tau = obs.joint_cut(frames[:3], scores[:3], obs.RAW_BUDGET * obs.FRACTION,
                            obs.INCIDENT_BUDGET * obs.FRACTION)
        confirmed = obs.triple(frames[-1], scores[-1])
        metrics, hits = obs.metrics(frames[-1], scores[-1], confirmed, tau, eligible)
        rows.append(dict(seed=seed, model=name, **metrics))
        events.extend(dict(seed=seed, model=name, episode=i,
                           ntype=sim.topo.nodes[ep.nid].ntype.value,
                           detected=int(i in hits), delay_seconds=hits.get(i))
                      for i, ep in eligible.items())
        print(seed, name, metrics, flush=True)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(events).to_csv(out / "episodes.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("validate", "select", "confirm", "summary"))
    p.add_argument("--models", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--input", type=Path)
    p.add_argument("--selection", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    development = a.stage in ("validate", "select")
    seeds = VALIDATE if development else CONFIRM
    names = CANDIDATES
    if not development:
        selected = json.loads(a.selection.read_text())["selected"]
        if selected not in CANDIDATES:
            raise ValueError("unknown selected model")
        names = tuple(dict.fromkeys((BASE, selected)))
    if a.stage in ("validate", "confirm"):
        if a.seed not in seeds:
            raise ValueError("unregistered seed")
        evaluate(a.seed, joblib.load(a.models), names, a.output)
        return
    worlds, events, summary = collect(a.input, seeds, names)
    print(summary.to_string(index=False))
    summary.to_csv(a.output / "summary.csv", index=False)
    worlds.to_csv(a.output / "worlds.csv", index=False)
    events.to_csv(a.output / "episodes.csv", index=False)
    role = events.groupby(["model", "ntype"]).detected.agg(["size", "sum", "mean"])
    role.to_csv(a.output / "roles.csv")
    if development:
        selected = summary.sort_values(["both_budgets", "recall", "worst_raw", "model"],
                                       ascending=[False, False, True, True]).iloc[0].model
        result = {"selected": selected}
        filename = "selection.json"
    else:
        paired = worlds.pivot(index="seed", columns="model", values="recall")
        recall = paired[selected].to_numpy()
        sample = np.random.default_rng(15200).integers(len(seeds), size=(20000, len(seeds)))
        ci = np.quantile(recall[sample].mean(axis=1), [.025, .975])
        delta = (paired[selected] - paired[BASE]).to_numpy()
        compliant = int(worlds.loc[worlds.model == selected, "both_budgets"].sum())
        result = dict(selected=selected, recall=float(recall.mean()), recall_ci=ci.tolist(),
                      paired_gain=float(delta.mean()),
                      paired_gain_ci=np.quantile(delta[sample].mean(axis=1), [.025, .975]).tolist(),
                      compliant_worlds=compliant, worlds=len(seeds),
                      target_met=bool(recall.mean() > .90 and compliant == len(seeds)))
        filename = "result.json"
    print(json.dumps(result), flush=True)
    (a.output / filename).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
