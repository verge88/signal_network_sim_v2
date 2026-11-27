"""Evaluate a causal interval-feature forest on the existing five-class CU workload.

Adapted from Deng et al., Information Sciences 239 (2013), 142-153.
Fixed trailing intervals and ExtraTrees are used instead of random intervals
and the original TSF split criterion. The experiment changes only inference.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from ss7 import five_specialist_development as S

TRAIN = tuple(range(19001, 19005))
TEST = tuple(range(19201, 19209))
NAMES = ("dual_75", "snapshot_forest", "interval_forest")
CHANNELS = tuple(dict.fromkeys((*S.TEMPORAL, *S.CONTEXT)))
WIDTHS = (4, 12, 32)


def interval_features(d):
    """Past and current samples, with a reset at each node or time gap."""
    values = d[list(CHANNELS)].replace([np.inf, -np.inf], np.nan)
    node, time = d.nid.to_numpy(), d.t.to_numpy()
    group = np.cumsum(np.r_[True, (node[1:] != node[:-1]) | (np.diff(time) != 1)])
    blocks = []
    for width in WIDTHS:
        windows = values.groupby(group, sort=False).rolling(width, min_periods=width)
        for field, stat in (("mean", windows.mean()), ("sd", windows.std(ddof=0))):
            blocks.append(stat.droplevel(0).reindex(d.index).add_suffix(f"_{field}{width}"))
        # Difference of adjacent half-window means estimates a local trend.
        half = width // 2
        recent = values.groupby(group, sort=False).rolling(half, min_periods=half).mean()
        recent = recent.droplevel(0).reindex(d.index)
        trend = (recent - recent.groupby(group, sort=False).shift(half)) / half
        blocks.append(trend.add_suffix(f"_trend{width}"))
    return pd.concat(blocks, axis=1).astype("float32")


def features(d):
    current = d[S.cols(d, True)].astype("float32")
    return pd.concat([current, interval_features(d)], axis=1).replace([np.inf, -np.inf], np.nan)


def train(out):
    parts, labels, weights = [], [], []
    for seed in TRAIN:
        d = S.generate(seed)[0]
        positive = d.label.to_numpy() == 1
        # Equal contribution per episode within class; balance classes below.
        counts = d.loc[positive, "episode_id"].value_counts()
        neg = np.flatnonzero(~positive)
        take = np.sort(np.r_[np.flatnonzero(positive),
                              np.random.default_rng(seed).choice(neg, min(20000, len(neg)), replace=False)])
        current = S.cols(d, True)
        parts.append(features(d).iloc[take].copy())
        labels.extend(np.where(positive, d.attack.to_numpy(), "normal")[take])
        w = np.ones(len(d))
        w[positive] = 1 / d.loc[positive, "episode_id"].map(counts).to_numpy()
        weights.extend(w[take])
        print("training data", seed, len(take), flush=True)
    x = pd.concat(parts, ignore_index=True)
    y, w = np.asarray(labels), np.asarray(weights)
    if set(y) != {"normal", *S.CLASSES}:
        raise ValueError("training class missing")
    for label in np.unique(y):
        mask = y == label
        w[mask] *= len(w) / (len(np.unique(y)) * w[mask].sum())
    models = {}
    for name, cols in (("snapshot_forest", current), ("interval_forest", list(x.columns))):
        med = x[cols].median().fillna(0.)
        model = ExtraTreesClassifier(n_estimators=160, max_features="sqrt", min_samples_leaf=4,
                                     max_depth=28, random_state=19000, n_jobs=2)
        model.fit(x[cols].fillna(med).to_numpy(), y, sample_weight=w)
        models[name] = (cols, med, model)
    joblib.dump(models, out / "models.joblib", compress=3)


def observe(d, frozen, models):
    x = features(d)
    scores = {}
    for name, (cols, med, model) in models.items():
        prob = model.predict_proba(x[cols].fillna(med).to_numpy())
        scores[name] = 1 - prob[:, list(model.classes_).index("normal")]
    for name in ("expanded_cu", "expanded_special"):
        scores[name] = S.I.predict(d, frozen, name)
    return d[["t", "nid", "episode_id", "label"]].copy(), scores


def test(seed, frozen, models, out):
    if seed not in TEST:
        raise ValueError("unexpected test seed")
    d, sim, episodes, _ = S.generate(seed)
    eligible = {i: e for i, e in enumerate(episodes) if e.kind in S.CLASSES}
    if len(eligible) != 200:
        raise ValueError("wrong episode count")
    for i, e in eligible.items():
        q = d[(d.episode_id == i) & (d.nid == e.nid)]
        if len(q) != e.t1 - e.t0 + 1 or q.t.nunique() != len(q):
            raise ValueError("incomplete episode")
    test_frame, test_scores = observe(d, frozen, models)
    del d
    refs, ref_scores = [], []
    for offset, balanced, stress in ((10000, False, False), (20000, False, True), (40000, True, False)):
        frame, scores = observe(S.generate(seed+offset, balanced, sim, stress)[0], frozen, models)
        refs.append(frame)
        ref_scores.append(scores)
    def series(name):
        return [r[name] for r in ref_scores] + [test_scores[name]]
    a, b = series("expanded_cu"), series("expanded_special")
    ta, tb, ex = S.union(refs, a[:3], b[:3], .75)
    rows, events = [], []
    for name in NAMES:
        if name == "dual_75":
            u, tau = np.maximum(a[-1]-ta, b[-1]-tb)-ex, 0.
        else:
            scores = series(name)
            tau = S.joint_cut(refs, scores[:3], S.RAW_BUDGET*S.FRACTION, S.INCIDENT_BUDGET*S.FRACTION)
            u = scores[-1]
        m, hits = S.metrics(test_frame, u, S.triple(test_frame, u), tau, eligible)
        rows.append(dict(seed=seed, model=name, **m))
        events.extend(dict(seed=seed, model=name, episode=i, attack=e.kind,
                           detected=int(i in hits), delay_seconds=hits.get(i)) for i,e in eligible.items())
        print(seed, name, m, flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    pd.DataFrame(events).to_csv(out/"episodes.csv", index=False)


def aggregate(root, out):
    m, e, summary = S.collect(root, TEST, NAMES)
    classes = e.groupby(["model","attack"]).detected.agg(["size","sum","mean"])
    summary["target_met"] = (summary.recall > .90) & (summary.both_budgets == len(TEST))
    p = m.pivot(index="seed",columns="model",values="recall")
    sample = np.random.default_rng(19200).integers(len(TEST),size=(20000,len(TEST)))
    comparisons = []
    for baseline in ("snapshot_forest", "dual_75"):
        delta = (p.interval_forest-p[baseline]).to_numpy()
        lo, hi = np.quantile(delta[sample].mean(axis=1), [.025,.975])
        comparisons.append(dict(baseline=baseline, mean_delta=delta.mean(), lo=lo, hi=hi))
    classes.to_csv(out/"classes.csv")
    for name, frame in (("summary",summary),("worlds",m),("episodes",e),("paired",pd.DataFrame(comparisons))):
        frame.to_csv(out/(name+".csv"),index=False)
    print(summary.to_string(index=False), classes.to_string(), comparisons, sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train","test","aggregate"))
    parser.add_argument("--input",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--models",type=Path)
    parser.add_argument("--frozen",type=Path)
    parser.add_argument("--seed",type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        train(args.output)
    elif args.mode == "test":
        test(args.seed, joblib.load(args.frozen), joblib.load(args.models), args.output)
    else:
        aggregate(args.input,args.output)
