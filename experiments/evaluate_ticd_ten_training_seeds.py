"""Ten independent SIP experiments with refitted baselines and TICD references."""
from copy import deepcopy
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from experiments import evaluate_ticd_article_benchmark as bench
from evaluate_ticd_adaptive_loss_fusion import NORMAL
from ticd_measured_loss import FullMeasuredReference, labels
from ticd_sequential_physical import sequence_features
from ticd_correlated_block_evidence import block_features

REPEATS = 10
TRAIN_KINDS = (*bench.KINDS, *NORMAL)
VAL_KINDS = tuple(dict.fromkeys((*NORMAL, *bench.CLEAN)))


def seed_for(repeat, phase, scenario):
    return 900001 + repeat * 10000 + phase * 1000 + scenario


def prepare(repeat, reference, calibration, folder):
    references, commissioning, validation = [], [], {}
    for phase, stage, kinds in ((0, "train", TRAIN_KINDS), (1, "val", VAL_KINDS)):
        frames = {t: [] for t in bench.TRACKS}
        for i, kind in enumerate(kinds):
            seed = seed_for(repeat, phase, i)
            raw, _, _, commission = bench.group([seed], calibration, kind, reference.core.causal)
            clean = kind in VAL_KINDS
            if clean:
                assert not raw.is_anomaly.any(), kind
            if stage == "train" and kind in NORMAL:
                references.append(labels(raw[bench.COLUMNS], raw))
                assert not commission.is_anomaly.any()
                commissioning.append(labels(commission[list(bench.RAW)], commission))
            if stage == "val":
                validation[kind] = (labels(raw[bench.COLUMNS], raw), commission[list(bench.RAW)])
            features, available = bench.measurements(raw)
            for track in bench.TRACKS:
                x = features[track].assign(y=raw.is_anomaly.astype(bool), seed=seed,
                                           clean_run=clean, visible=available[track])
                frames[track].append(x)
        for track, parts in frames.items():
            pd.concat(parts, ignore_index=True).to_pickle(folder / f"{stage}-{track}-0.pkl.gz")
    return pd.concat(references, ignore_index=True), pd.concat(commissioning, ignore_index=True), validation


def refit_ticd(template, train, commissioning, validation):
    # Refit all active references and empirical cuts; keep the published
    # report/audit rules and all architecture choices fixed across repeats.
    ticd = deepcopy(template)
    ticd.reference = FullMeasuredReference(template.reference.mode, template.reference.core)
    ticd.reference.fit(train, commissioning, {k: x for k, (x, _) in validation.items()})
    prepared = {}
    for key, (raw, c) in validation.items():
        x = ticd.reference.prepare(raw, c)
        x = sequence_features(x, c, ticd.reference)
        x = block_features(x, c, ticd.reference)
        prepared[key] = labels(x, raw)
    policy = ticd.block_model
    policy.base.calibrate(prepared)
    policy.old.calibrate(prepared)
    policy.calibrate(prepared)
    ver = [ticd.predict_levels(x[bench.COLUMNS], c) for x, c in validation.values()]
    return ticd, [float((z == 2).mean()) for z in ver]


def evaluate(repeat, prior, out):
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

    template, cal = bench.load_inputs(prior)
    data = out / "data"
    weights = out / "weights"
    data.mkdir(exist_ok=True)
    weights.mkdir(exist_ok=True)
    train, commissioning, val = prepare(repeat, template.reference, cal, data)
    ticd, val_fpr = refit_ticd(template, train, commissioning, val)
    print("TICD_REFIT_OK", repeat, len(train), max(val_fpr), flush=True)
    del train, commissioning, val
    fit_seed = seed_for(repeat, 3, 0)
    fits = {f"{t}/{m}": bench.fit(m, t, data, weights, fit_seed=fit_seed, shards=(0,))
            for t in bench.TRACKS for m in bench.FAMILIES}
    sections = []
    for i, kind in enumerate(bench.KINDS):
        seed = seed_for(repeat, 2, i)
        raw, _, _, c = bench.group([seed], cal, kind, template.reference.core.causal)
        x, visible = bench.measurements(raw)
        p = raw[bench.METADATA].assign(scenario=kind)
        p["TICD"] = ticd.predict_levels(raw[bench.COLUMNS], c[list(bench.RAW)])
        for name, (imp, scale, model, manifest) in fits.items():
            track, m = name.split("/")
            assert seed not in manifest["training_seeds"] + manifest["validation_seeds"]
            s = np.asarray(bench.score(m, model, scale.transform(imp.transform(x[track][manifest["features"]]))), float)
            assert np.isfinite(s).all()
            s[~visible[track]] = -np.inf
            p[name] = s
        sections.append(p)
        print("REPEAT_TEST", repeat, seed, kind, flush=True)
    panel = pd.concat(sections, ignore_index=True)
    assert len(panel) == 158400 and panel.simulation_seed.nunique() == 40
    assert not panel.duplicated(["simulation_seed", "node_id", "interval"]).any()
    y = panel.is_anomaly.to_numpy(dtype=bool)
    group_specs = [("all", "all", np.arange(len(panel)))]
    for field in ("scenario", "attack_type", "attacker_sophistication", "simulation_seed"):
        for label, ix in panel.groupby(field).indices.items():
            if field in ("attack_type", "attacker_sophistication"):
                ix = ix[y[ix]]
                if not len(ix):
                    continue
            group_specs.append((field, str(label), ix))
    summaries, curves, history = [], [], []
    for name in ("TICD", *fits):
        s = panel[name].to_numpy(dtype=float)
        ranked = np.where(np.isfinite(s), s, s[np.isfinite(s)].min() - 1)
        auc, ap = roc_auc_score(y, ranked), average_precision_score(y, ranked)
        meta = fits[name][-1] if name != "TICD" else None
        cut_points = meta["thresholds"] if meta else {"refit": 1}
        for cut_name, cut in cut_points.items():
            positive = s >= cut if cut_name == "article" and meta["supervised"] else s > cut
            for field, label, ix in group_specs:
                row = dict(repeat=repeat + 1, model=name, operating_point=cut_name,
                           group_type=field, group=label, threshold=cut,
                           **bench.metrics(positive[ix], y[ix]))
                if field == "all":
                    row.update(ROC_AUC=auc, AP=ap)
                summaries.append(row)
        fpr, tpr, _ = roc_curve(y, ranked)
        pre, rec, _ = precision_recall_curve(y, ranked, drop_intermediate=True)
        for curve, a, b in (("ROC", fpr, tpr), ("PR", rec[::-1], pre[::-1])):
            indices = np.unique(np.linspace(0, len(a) - 1, min(len(a), 1200), dtype=int))
            curves.extend(dict(repeat=repeat + 1, model=name, curve=curve, x=a[i], y=b[i])
                          for i in indices)
        if meta and meta["training_loss"] is not None:
            history.extend(dict(repeat=repeat + 1, model=name, step=i, loss=v)
                           for i, v in enumerate(meta["training_loss"], 1))
    result = out / "results"
    result.mkdir(exist_ok=True)
    for name, rows in (("metrics", summaries), ("curves", curves), ("loss", history)):
        pd.DataFrame(rows).to_csv(result / f"{name}-{repeat}.csv.gz", index=False)
    ledger = {"repeat": repeat + 1, "model_seed": fit_seed, "ticd_validation_FPR": val_fpr,
              "train": {str(seed_for(repeat, 0, i)): k for i, k in enumerate(TRAIN_KINDS)},
              "validation": {str(seed_for(repeat, 1, i)): k for i, k in enumerate(VAL_KINDS)},
              "test": {str(seed_for(repeat, 2, i)): k for i, k in enumerate(bench.KINDS)},
              "models": {k: v[-1] for k, v in fits.items()},
              "ticd_block": ticd.block_model.manifest()}
    bench.dump(result / f"ledger-{repeat}.json", ledger)
    print("REPEAT_COMPLETE", repeat + 1, len(panel), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeat", type=int, choices=range(REPEATS), required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    evaluate(a.repeat, a.prior, a.out)
