"""Matched SIP TICD benchmark with the five model families from the original article."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.evaluate_ticd_report_consistency import COLUMNS, KINDS, NATIVE, WEAK, REPLICATES, RAW, group

FAMILIES = ("isolation_forest", "one_class_svm", "autoencoder", "random_forest", "gradient_boosting")
SUPERVISED = FAMILIES[-2:]
TRACKS = ("article", "shared")
METADATA = ["simulation_seed", "node_id", "interval", "attack_type", "attacker_sophistication", "is_anomaly"]
CLEAN = tuple(k for k in KINDS if k.startswith(("clean_", "shift_clean")) or k.endswith("_clean") or k == "observer_outage")
DEV = {
    "dev_io_ratio_vs_zone", "dev_integrity_fail_rate_vs_zone", "dev_var_total_vs_zone",
    "dev_txn_consistency_vs_zone", "dev_processing_delay_vs_zone", "dev_ack_mismatch_vs_zone",
    "dev_total_messages_vs_zone", "dev_method_dominance_ratio_vs_zone",
    "dev_ratio_invite_vs_zone", "dev_ratio_register_vs_zone", "dev_ratio_resp_401_407_vs_zone",
    "dev_dangling_dialog_ratio_vs_zone",
}
MASKED = {"cu_combined_loss_prob", "cu_req_delay_ms", "cu_resp_delay_ms", "cu_dcr_transit"}
BUDGETS = {"val_FPR_1pct": .01, "val_FPR_5pct": .05}
SAMPLE_SIZE = 28512
FIT_SEED = 8811


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_inputs(prior):
    model = load_pickle(prior / "report_consistency_ticd.pkl")
    package = load_pickle(prior / "sensor_reconciliation_models.pkl")
    return model, package["frozen_package"]["calibration"]


def measurements(raw):
    good = raw.primary_observer_valid
    observed = good | raw.audit_observed_count.gt(0)
    # Retain the article's feature families except simulator-only channels and
    # derived values that incorporate those channels.
    legacy = [c for c in raw if c.startswith(("obs_", "var_", "z_")) or c in DEV]
    legacy += [c for c in raw if c.startswith("cu_") and c not in MASKED
               and c in {*RAW, "cu_rtt_delay_ms", "cu_processing_delay_ms",
                         "cu_response_time_jitter", "cu_staleness", "cu_consistency_error"}]
    legacy += [c for c in ("zone_io_ratio_rank", "integrity_fail_rate", "autocorr_obs_total_messages",
                           "obs_destinations_cv", "mean_txn_state_consistency_error",
                           "mean_cu_processing_delay_ms") if c in raw]
    shared = [c for c in COLUMNS if c not in {*METADATA, "node_type", "zone_id"}]
    frames = {}
    for track, cols in (("article", legacy), ("shared", shared)):
        x = raw[sorted(set(cols))].astype("float32")
        audit_cols = ["audit_observed_count", "audit_mismatch_count"]
        primary_columns = x.columns.difference([*audit_cols, "primary_observer_valid"])
        x.loc[~good, primary_columns] = np.nan
        if track == "shared":
            src = raw.sort_values(["node_id", "interval"])
            group_id = src.node_id
            for width in (6, 16, 32):
                for key in audit_cols:
                    x[f"{key}_sum_{width}"] = src[key].groupby(group_id).transform(
                        lambda s: s.rolling(width, min_periods=1).sum()).astype("float32")
        x["node_type_enc"] = raw.node_type.map(
            {"PROXY": 0, "REDIRECT": 1, "REGISTRAR": 2, "UA_GW": 3}).astype("float32")
        assert x.node_type_enc.notna().all()
        # Guarantee the counterfactual pair has the same model input whenever
        # both observation streams are unavailable.
        visible = observed if track == "shared" else good
        x.loc[~visible] = np.nan
        frames[track] = x
    return frames, {"article": good.to_numpy(), "shared": observed.to_numpy()}


def train_data(prior, n, out):
    model, cal = load_inputs(prior)
    for stage, kinds, start in (("train", KINDS, 740001), ("val", CLEAN, 760001)):
        parts = {track: [] for track in TRACKS}
        for i, kind in enumerate(kinds):
            seed = start + i * 100 + n
            raw, _, _, _ = group([seed], cal, kind, model.reference.core.causal)
            if stage == "val":
                assert not raw.is_anomaly.any()
            features, available = measurements(raw)
            for track in TRACKS:
                x = features[track].copy()
                x["y"] = raw.is_anomaly.astype(bool)
                x["seed"] = seed
                x["clean_run"] = kind in CLEAN
                x["visible"] = available[track]
                parts[track].append(x)
        for track in TRACKS:
            pd.concat(parts[track], ignore_index=True).to_pickle(out / f"{stage}-{track}-{n}.pkl.gz")


def score(family, model, X):
    if family == "autoencoder":
        from train_ticd import ae_err
        return ae_err(model, X)
    if family in SUPERVISED:
        return model.predict_proba(X)[:, 1]
    return -model.score_samples(X)


def threshold(scores, budget):
    # Strict comparison implements the same budget even when scores tie.
    keep = int(np.floor(budget * len(scores)))
    return float(np.partition(scores, len(scores) - 1 - keep)[len(scores) - 1 - keep])


def fit(which, track, data, out, *, fit_seed=FIT_SEED, shards=(0, 1), sample_size=SAMPLE_SIZE):
    from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    train, val = [pd.concat([pd.read_pickle(data / f"{stage}-{track}-{i}.pkl.gz")
                            for i in shards], ignore_index=True) for stage in ("train", "val")]
    assert not set(train.seed) & set(val.seed) and not val.y.any()
    supervised = which in SUPERVISED
    features = train.columns.difference(["y", "seed", "clean_run", "visible"]).tolist()
    pool = train[train.visible & (True if supervised else train.clean_run & ~train.y)]
    # Fixed original article training size keeps the kernel SVM tractable.
    if supervised:
        sample, _ = train_test_split(pool, train_size=sample_size, stratify=pool.y, random_state=fit_seed)
        assert sample.y.any()
    else:
        sample = pool.sample(sample_size, random_state=fit_seed)
        assert not sample.y.any()
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(sample[features])
    scaler = StandardScaler().fit(imputer.transform(sample[features]))
    X = scaler.transform(imputer.transform(sample[features]))
    hist = None
    if which == "autoencoder":
        import torch
        from train_ticd import train_ae
        torch.set_num_threads(1)
        torch.manual_seed(fit_seed)
        model, hist = train_ae(X)
    else:
        constructors = {
            "isolation_forest": lambda: IsolationForest(n_estimators=300, contamination=.05, random_state=fit_seed, n_jobs=2),
            "one_class_svm": lambda: OneClassSVM(nu=.05, kernel="rbf", gamma="scale", cache_size=1024),
            "random_forest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=fit_seed, n_jobs=2),
            "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=300, learning_rate=.05, max_depth=5, random_state=fit_seed),
        }
        estimator = constructors[which]()
        model = estimator.fit(X, sample.y) if supervised else estimator.fit(X)
    if which == "gradient_boosting":
        hist = model.train_score_.tolist()
    scores = np.asarray(score(which, model, scaler.transform(imputer.transform(val[features]))), float)
    assert np.isfinite(scores).all()
    scores[~val.visible.to_numpy()] = -np.inf
    cuts = {}
    for key, fpr in BUDGETS.items():
        cuts[key] = threshold(scores, fpr)
    cuts["val_FPR_1pct_each_run"] = max(threshold(scores[ix], .01) for ix in val.groupby("seed").indices.values())
    cuts["article"] = .5 if supervised else cuts["val_FPR_5pct"]
    val_rates = {key: float((scores > cut).mean()) for key, cut in cuts.items()}
    for key, max_rate in BUDGETS.items():
        assert val_rates[key] <= max_rate
    assert all((scores[ix] > cuts["val_FPR_1pct_each_run"]).mean() <= .01
               for ix in val.groupby("seed").indices.values())
    manifest = {"model": which, "track": track, "supervised": supervised, "features": features,
                "training_rows": len(sample), "training_attack_rows": int(sample.y.sum()),
                "training_seeds": sorted(map(int, sample.seed.unique())), "training_rng": fit_seed,
                "validation_rows": len(val), "validation_seeds": sorted(map(int, val.seed.unique())),
                "thresholds": cuts, "validation_fpr": val_rates, "training_loss": hist}
    print("BASELINE_FIT", json.dumps(manifest), flush=True)
    with (out / f"{track}-{which}.pkl").open("wb") as f:
        pickle.dump((imputer, scaler, model, manifest), f, pickle.HIGHEST_PROTOCOL)
    dump(out / f"{track}-{which}.json", manifest)
    return imputer, scaler, model, manifest


def test_part(prior, models, shard, out):
    model, cal = load_inputs(prior)
    reference_report = json.loads((prior / "report.json").read_text(encoding="utf-8"))
    expected = {r["seed"]: r["models"]["full"] for r in reference_report["trials"]}
    fitted = {}
    for path in sorted(Path(models).glob("*.pkl")):
        model_data = load_pickle(path)
        manifest = model_data[-1]
        fitted[manifest["track"], manifest["model"]] = model_data
    assert set(fitted) == {(t, m) for t in TRACKS for m in FAMILIES}
    parts = []
    for i, kind in enumerate(KINDS):
        seed = 680001 + 100 * i + shard
        raw, _, _, commissioning = group([seed], cal, kind, model.reference.core.causal)
        y = raw.is_anomaly.to_numpy(dtype=bool)
        levels = model.predict_levels(raw[COLUMNS], commissioning[list(RAW)])
        hits = levels == 2
        old = expected[seed]
        assert (int(y.sum()), int((hits & y).sum()), int((hits & ~y).sum()),
                int(((levels == 1) & ~y).sum())) == (old["attack"], old["tp"], old["fp"], old["fw"])
        x, visible = measurements(raw)
        if kind == "latent_identical":
            twin, _, _, _ = group([seed], cal, "clean_loss10", model.reference.core.causal)
            xt, _ = measurements(twin)
            for track in TRACKS:
                pd.testing.assert_frame_equal(x[track], xt[track])
        p = raw[METADATA].copy()
        p["scenario"], p["TICD"] = kind, levels
        for (track, family), (imp, sc, est, manifest) in fitted.items():
            assert seed not in {*manifest["training_seeds"], *manifest["validation_seeds"]}
            s = np.asarray(score(family, est, sc.transform(imp.transform(x[track][manifest["features"]]))), float)
            assert np.isfinite(s).all()
            s[~visible[track]] = -np.inf
            p[f"{track}/{family}"] = s
        parts.append(p)
        print("BENCHMARK_TEST", seed, kind, len(p), flush=True)
    pd.concat(parts, ignore_index=True).to_pickle(out / f"test-{shard}.pkl.gz")


def ratio(a, b):
    return float(a / b) if b else None


def metrics(pred, y):
    tp, fp = int(np.sum(pred & y)), int(np.sum(pred & ~y))
    attack, normal = int(y.sum()), int((~y).sum())
    return {"TP": tp, "FP": fp, "FN": attack - tp, "TN": normal - fp,
            "recall": ratio(tp, attack), "FPR": ratio(fp, normal),
            "precision": ratio(tp, tp + fp), "F1": ratio(2 * tp, attack + tp + fp)}


def summarize(data, manifests, out):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    from scipy.stats import wilcoxon

    panel = pd.concat([pd.read_pickle(data / f"test-{i}.pkl.gz") for i in range(REPLICATES)],
                      ignore_index=True)
    assert panel.simulation_seed.nunique() == len(KINDS) * REPLICATES == 400
    assert not panel.duplicated(["simulation_seed", "node_id", "interval"]).any()
    assert len(panel) == 1584000
    y = panel.is_anomaly.to_numpy(dtype=bool)
    assert y.sum() == 168962
    summary, per_group, per_run, ranking, curves, paired = [], [], [], [], [], []
    models = {"TICD": {"thresholds": {"frozen": 1}, "supervised": False, "validation_fpr": {}}}
    for path in sorted(manifests.glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        models[f"{config['track']}/{config['model']}"] = config

    def add_groups(pred, name, operating):
        local = []
        for key in ("scenario", "attack_type", "attacker_sophistication", "simulation_seed"):
            for level, ix in panel.groupby(key, sort=True).indices.items():
                if key in ("attack_type", "attacker_sophistication"):
                    ix = ix[y[ix]]
                    if not len(ix):
                        continue
                row = {"model": name, "operating_point": operating, "group": str(level),
                       "group_type": key, **metrics(pred[ix], y[ix])}
                if key == "simulation_seed":
                    row["scenario"] = panel.scenario.iloc[ix[0]]
                    per_run.append(row)
                else:
                    per_group.append(row)
                    if key == "scenario":
                        local.append(row)
        return local

    for name, cfg in models.items():
        col = panel[name].to_numpy(dtype=float)
        ranked = np.where(np.isfinite(col), col, float(np.min(col[np.isfinite(col)]) - 1))
        rank_row = {"model": name, "score_type": "three ordinal alert levels" if name == "TICD" else "model score",
                    "AUC_ROC": float(roc_auc_score(y, ranked)), "AP": float(average_precision_score(y, ranked))}
        ranking.append(rank_row)
        for curve, points in (("ROC", roc_curve(y, ranked)[:2]), ("PR", precision_recall_curve(y, ranked, drop_intermediate=True)[:2][::-1])):
            indices = np.unique(np.linspace(0, len(points[0]) - 1, min(len(points[0]), 1500), dtype=int))
            curves.extend({"model": name, "curve": curve, "x": float(points[0][i]), "y": float(points[1][i])}
                          for i in indices)
        for point, cut in cfg["thresholds"].items():
            # Original RF/GB point used >= 0.5; budgeted cuts use strict >.
            pred = col >= cut if point == "article" and cfg["supervised"] else col > cut
            cohort = add_groups(pred, name, point)
            macro = {}
            for label, kinds in (("native", NATIVE), ("weak", WEAK), ("all_attack_scenarios", KINDS)):
                listed = [v["recall"] for v in cohort if v["group"] in kinds and v["recall"] is not None]
                macro["recall_macro_" + label] = float(np.mean(listed))
            r = {"model": name, "operating_point": point, "threshold": cut,
                 "validation_FPR": cfg["validation_fpr"].get(point), **rank_row,
                 **metrics(pred, y), **macro}
            runs = [v for v in per_run if v["model"] == name and v["operating_point"] == point]
            r.update({"max_run_FPR": max(v["FPR"] for v in runs),
                      "runs_FPR_le_1pct": sum(v["FPR"] <= .01 for v in runs)})
            summary.append(r)
            print("SIP_BENCHMARK", json.dumps(r), flush=True)
    b = pd.DataFrame(per_run)
    base = b[b.model == "TICD"].set_index("group").sort_index()
    rng = np.random.default_rng(77661)
    positions = list(base.groupby("scenario").indices.values())
    boot = np.column_stack([rng.choice(ix, size=(2000, len(ix))) for ix in positions])
    for name in models:
        if name == "TICD":
            continue
        rival = b[(b.model == name) & b.operating_point.eq("val_FPR_1pct_each_run")].set_index("group").sort_index()
        assert base.index.equals(rival.index)
        for label, count, metric in (("recall", "TP", "recall"), ("FPR", "FP", "FPR")):
            d = (base[count] - rival[count]).to_numpy()
            total = base[["TP", "FN"] if label == "recall" else ["FP", "TN"]].sum(axis=1).to_numpy()
            estimates = d[boot].sum(axis=1) / total[boot].sum(axis=1)
            values = (base[metric] - rival[metric]).dropna().to_numpy()
            paired.append({"comparator": name, "metric": label,
                           "comparator_point": "val_FPR_1pct_each_run",
                           "difference_TICD_minus_baseline": float(d.sum() / total.sum()),
                           "paired_stratified_CI95": np.quantile(estimates, [.025, .975]).tolist(),
                           "wilcoxon_p": 1. if not np.any(values) else float(wilcoxon(values).pvalue),
                           "paired_runs": len(values)})
    previous = 0.
    for i, row in enumerate(sorted(paired, key=lambda r: r["wilcoxon_p"])):
        previous = max(previous, min(1., (len(paired) - i) * row["wilcoxon_p"]))
        row["p_holm"] = previous
    summary_frame = pd.DataFrame(summary)
    for filename, rows in (("comparison", summary_frame), ("by_group", per_group),
                           ("by_run", per_run), ("paired", paired), ("curve_points", curves)):
        pd.DataFrame(rows).to_csv(out / (filename + ".csv"), index=False)
    dump(out / "benchmark.json", {"reference_run": 45688280428, "test_runs": 400,
                                  "thresholds_selected_on_test": False, "summary": summary, "paired": paired,
                                  "model_manifests": models, "ranking": ranking})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"TICD": "TICD", "isolation_forest": "Isolation Forest", "one_class_svm": "One-Class SVM",
             "autoencoder": "Автоэнкодировщик", "random_forest": "Random Forest",
             "gradient_boosting": "Gradient Boosting"}
    for track in TRACKS:
        chosen = ["TICD", *[f"{track}/{m}" for m in FAMILIES]]
        d = summary_frame[(summary_frame.operating_point.isin(["frozen", "val_FPR_1pct_each_run"]))
                          & summary_frame.model.isin(chosen)].set_index("model").loc[chosen]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        labels = [names[m.split("/")[-1]] for m in d.index]
        for ax, metric, name in zip(axes, ("recall", "FPR"), ("Полнота, %", "Ложные тревоги, %")):
            bars = ax.barh(labels, 100 * d[metric], color=["#333333"] + ["#aaaaaa"] * len(FAMILIES))
            ax.bar_label(bars, fmt="%.2f" if metric == "recall" else "%.3f", padding=4, fontsize=8)
            ax.margins(x=.20)
            ax.set_xlabel(name)
        axes[0].invert_yaxis()
        fig.tight_layout()
        for fmt in ("pdf", "png", "svg"):
            fig.savefig(out / f"comparison_{track}.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fit", "test", "summarize"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--track", choices=TRACKS)
    parser.add_argument("--family", choices=FAMILIES)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        train_data(args.prior, args.shard, args.out)
    elif args.action == "fit":
        fit(args.family, args.track, args.data, args.out)
    elif args.action == "test":
        test_part(args.prior, args.models, args.shard, args.out)
    else:
        summarize(args.data, args.models, args.out)
