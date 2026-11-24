"""Build all figures for the full SS7 all-model experiment.

The script reads GitHub Actions CSV outputs recursively, aggregates independent
simulation seeds, writes summary CSV files, and renders every publication-facing
figure as PNG and PDF.  No experimental values are hard-coded.

Example:
    python -m ss7.plot_full_model_experiment \
        --input-root results/all_models_downloaded \
        --output-dir results/all_models_figures
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


INTERVAL_SECONDS = 30.0

MODEL_LABELS = {
    "rule_baseline": "Rule baseline",
    "pca_recon": "PCA reconstruction",
    "isolation_forest": "Isolation Forest",
    "ocsvm": "One-Class SVM",
    "random_forest": "Random Forest",
    "hist_gb": "HistGradientBoosting",
    "cascade": "Cascade",
    "meta_ensemble": "Stacking ensemble",
    "autoencoder": "Autoencoder",
    "vae": "VAE",
    "lstm_ae": "LSTM autoencoder",
    "attn_ae": "Attention autoencoder",
}

MODEL_ORDER = list(MODEL_LABELS)


def pretty_model(x: str) -> str:
    return MODEL_LABELS.get(str(x), str(x))


def extract_seed(path: Path) -> int | None:
    text = str(path)
    patterns = [r"seed[_-]?(\d+)", r"_seed(\d+)"]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def read_many(root: Path, filename: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob(filename)):
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        if d.empty:
            continue
        seed = extract_seed(path)
        if "seed" not in d.columns and seed is not None:
            d["seed"] = seed
        d["_source"] = str(path)
        rows.append(d)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def numeric_mean_std(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    d = df.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    g = d.groupby("model")[metric].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    g["_order"] = g["model"].map(order).fillna(999)
    return g.sort_values(["_order", "model"]).drop(columns="_order")


def safe_save(fig, outdir: Path, stem: str, formats: Sequence[str], manifest: list, description: str):
    fig.tight_layout()
    paths = []
    for ext in formats:
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, dpi=220 if ext == "png" else None, bbox_inches="tight")
        paths.append(str(path.name))
    plt.close(fig)
    manifest.append({"figure": stem, "files": paths, "description": description})


def bar_with_error(df: pd.DataFrame, metric: str, title: str, xlabel: str, outdir: Path,
                   stem: str, formats: Sequence[str], manifest: list, scale: float = 1.0,
                   xlim=None):
    g = numeric_mean_std(df, metric)
    g = g[np.isfinite(g["mean"])]
    if g.empty:
        return
    labels = [pretty_model(x) for x in g["model"]]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.48 * len(g) + 1.5)))
    ax.barh(y, g["mean"] * scale, xerr=g["std"] * scale, capsize=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    if xlim is not None:
        ax.set_xlim(*xlim)
    safe_save(fig, outdir, stem, formats, manifest, title)


def grouped_metric(df: pd.DataFrame, metrics: Sequence[tuple[str, str, float]], title: str,
                   ylabel: str, outdir: Path, stem: str, formats: Sequence[str],
                   manifest: list, ylim=None):
    models = [m for m in MODEL_ORDER if m in set(df["model"].astype(str))]
    extras = [m for m in sorted(set(df["model"].astype(str))) if m not in models]
    models += extras
    if not models:
        return
    x = np.arange(len(models))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.85), 6))
    for j, (metric, label, scale) in enumerate(metrics):
        means = []
        for model in models:
            vals = pd.to_numeric(df.loc[df["model"] == model, metric], errors="coerce")
            means.append(float(vals.mean()) * scale)
        ax.bar(x + (j - (len(metrics)-1)/2) * width, means, width, label=label)
    ax.set_xticks(x, [pretty_model(m) for m in models], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    safe_save(fig, outdir, stem, formats, manifest, title)


def heatmap_seed_model(df: pd.DataFrame, metric: str, title: str, outdir: Path,
                       stem: str, formats: Sequence[str], manifest: list, scale: float = 1.0):
    if "seed" not in df.columns or metric not in df.columns:
        return
    p = df.pivot_table(index="model", columns="seed", values=metric, aggfunc="mean")
    order = [m for m in MODEL_ORDER if m in p.index] + [m for m in p.index if m not in MODEL_ORDER]
    p = p.reindex(order)
    if p.empty:
        return
    a = p.to_numpy(dtype=float) * scale
    fig, ax = plt.subplots(figsize=(max(8, 1.05 * len(p.columns) + 3), max(5, 0.45 * len(p.index) + 2)))
    im = ax.imshow(a, aspect="auto")
    ax.set_xticks(np.arange(len(p.columns)), [str(x) for x in p.columns])
    ax.set_yticks(np.arange(len(p.index)), [pretty_model(x) for x in p.index])
    ax.set_xlabel("Seed")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.set_ylabel("%" if scale == 100.0 else metric, rotation=270, labelpad=15)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if np.isfinite(a[i, j]):
                ax.text(j, i, f"{a[i,j]:.2f}", ha="center", va="center", fontsize=7)
    safe_save(fig, outdir, stem, formats, manifest, title)


def plot_fit_time(df, outdir, formats, manifest):
    if "fit_seconds" not in df.columns:
        return
    g = numeric_mean_std(df, "fit_seconds")
    g = g[np.isfinite(g["mean"])]
    if g.empty:
        return
    labels = [pretty_model(x) for x in g["model"]]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.48 * len(g) + 1.5)))
    ax.barh(y, g["mean"], xerr=g["std"], capsize=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Training/evaluation time, s (log scale)")
    ax.set_title("Computational cost by model")
    ax.grid(axis="x", alpha=0.25)
    safe_save(fig, outdir, "06_fit_time", formats, manifest, "Computational cost by model")


def plot_scatter(df, outdir, formats, manifest):
    g = df.groupby("model")[["auc", "episode_recall", "window_fpr"]].mean(numeric_only=True).reset_index()
    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["auc", "episode_recall"])
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(g["auc"], g["episode_recall"] * 100.0, s=55)
    for _, r in g.iterrows():
        ax.annotate(pretty_model(r["model"]), (r["auc"], r["episode_recall"] * 100.0),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean ROC AUC")
    ax.set_ylabel("Mean episode recall, %")
    ax.set_title("Detector trade-off: ranking quality vs incident recall")
    ax.grid(alpha=0.25)
    safe_save(fig, outdir, "07_auc_vs_recall", formats, manifest,
              "Detector trade-off: ranking quality vs incident recall")


def plot_ablation(df, category: str, title: str, outdir: Path, stem: str,
                  formats: Sequence[str], manifest: list):
    if df.empty or category not in df.columns:
        return
    d = df[df["model"].astype(str).isin(["random_forest", "isolation_forest", "rule_baseline"])].copy()
    if d.empty:
        return
    cats = list(dict.fromkeys(d[category].astype(str)))
    models = [m for m in ["random_forest", "isolation_forest", "rule_baseline"] if m in set(d["model"])]
    x = np.arange(len(cats))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(cats) * 1.4), 6))
    for j, model in enumerate(models):
        vals = []
        for cat in cats:
            v = pd.to_numeric(d.loc[(d[category].astype(str) == cat) & (d["model"] == model), "episode_recall"], errors="coerce")
            vals.append(float(v.mean()) * 100 if len(v) else np.nan)
        ax.bar(x + (j - (len(models)-1)/2) * width, vals, width, label=pretty_model(model))
    ax.set_xticks(x, cats, rotation=25, ha="right")
    ax.set_ylabel("Episode recall, %")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    safe_save(fig, outdir, stem, formats, manifest, title)


def plot_curve(df, xcol: str, title: str, outdir: Path, stem: str,
               formats: Sequence[str], manifest: list):
    if df.empty or xcol not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for model, g in df.groupby("model"):
        g = g.copy()
        g[xcol] = pd.to_numeric(g[xcol], errors="coerce")
        g["episode_recall"] = pd.to_numeric(g["episode_recall"], errors="coerce")
        g = g.dropna(subset=[xcol, "episode_recall"]).sort_values(xcol)
        if g.empty:
            continue
        ax.plot(g[xcol], g["episode_recall"] * 100.0, marker="o", label=pretty_model(model))
    ax.set_xlabel(xcol.replace("_", " ").capitalize())
    ax.set_ylabel("Episode recall, %")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    safe_save(fig, outdir, stem, formats, manifest, title)


def plot_sophistication(df, outdir, formats, manifest):
    if df.empty or "sophistication" not in df.columns:
        return
    cats = list(dict.fromkeys(df["sophistication"].astype(str)))
    models = [m for m in ["random_forest", "isolation_forest", "rule_baseline"] if m in set(df["model"])]
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(9, 6))
    for j, model in enumerate(models):
        vals = []
        for cat in cats:
            v = pd.to_numeric(df.loc[(df["sophistication"].astype(str) == cat) & (df["model"] == model), "episode_recall"], errors="coerce")
            vals.append(float(v.mean()) * 100 if len(v) else np.nan)
        ax.bar(x + (j-(len(models)-1)/2)*width, vals, width, label=pretty_model(model))
    ax.set_xticks(x, cats)
    ax.set_ylabel("Episode recall, %")
    ax.set_title("Sensitivity to adversary sophistication")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    safe_save(fig, outdir, "11_sophistication", formats, manifest,
              "Sensitivity to adversary sophistication")


def plot_lomfo(df, outdir, formats, manifest):
    if df.empty or "holdout_family" not in df.columns:
        return
    d = df[df["model"].astype(str).isin(["random_forest", "rule_baseline"])].copy()
    if d.empty:
        return
    families = list(dict.fromkeys(d["holdout_family"].astype(str)))
    models = [m for m in ["random_forest", "rule_baseline"] if m in set(d["model"])]
    x = np.arange(len(families))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    for j, model in enumerate(models):
        vals = []
        for fam in families:
            v = pd.to_numeric(d.loc[(d["holdout_family"].astype(str)==fam)&(d["model"]==model), "episode_recall"], errors="coerce")
            vals.append(float(v.mean())*100 if len(v) else np.nan)
        ax.bar(x + (j-(len(models)-1)/2)*width, vals, width, label=pretty_model(model))
    ax.set_xticks(x, families, rotation=25, ha="right")
    ax.set_ylabel("Episode recall, %")
    ax.set_title("Leave-one-masking-family-out generalisation")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    safe_save(fig, outdir, "12_lomfo", formats, manifest,
              "Leave-one-masking-family-out generalisation")


def read_score_exports(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("scores_*.csv")):
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        if d.empty or not {"model", "label", "score"}.issubset(d.columns):
            continue
        seed = extract_seed(path)
        if seed is not None:
            d["seed"] = seed
        d["_source"] = str(path)
        rows.append(d)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def plot_roc_curves(scores: pd.DataFrame, outdir: Path, formats: Sequence[str], manifest: list):
    if scores.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 7))
    plotted = 0
    for model in MODEL_ORDER + [m for m in sorted(scores["model"].astype(str).unique()) if m not in MODEL_ORDER]:
        d = scores[scores["model"].astype(str) == model].copy()
        d["label"] = pd.to_numeric(d["label"], errors="coerce")
        d["score"] = pd.to_numeric(d["score"], errors="coerce")
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "score"])
        if d.empty or d["label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(d["label"].astype(int), d["score"].astype(float))
        auc = float(np.trapz(tpr, fpr))
        ax.plot(fpr, tpr, label=f"{pretty_model(model)} (AUC={auc:.3f})")
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Pooled ROC curves across regenerated test seeds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    safe_save(fig, outdir, "16_roc_curves", formats, manifest,
              "Pooled ROC curves across regenerated test seeds")


def plot_pr_curves(scores: pd.DataFrame, outdir: Path, formats: Sequence[str], manifest: list):
    if scores.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 7))
    plotted = 0
    prevalence = pd.to_numeric(scores["label"], errors="coerce").mean()
    for model in MODEL_ORDER + [m for m in sorted(scores["model"].astype(str).unique()) if m not in MODEL_ORDER]:
        d = scores[scores["model"].astype(str) == model].copy()
        d["label"] = pd.to_numeric(d["label"], errors="coerce")
        d["score"] = pd.to_numeric(d["score"], errors="coerce")
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "score"])
        if d.empty or d["label"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(
            d["label"].astype(int), d["score"].astype(float)
        )
        ax.plot(recall, precision, label=pretty_model(model))
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    if np.isfinite(prevalence):
        ax.axhline(float(prevalence), linestyle="--", linewidth=1, label="Test prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Pooled precision–recall curves across regenerated test seeds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    safe_save(fig, outdir, "17_precision_recall_curves", formats, manifest,
              "Pooled precision–recall curves across regenerated test seeds")


def bootstrap_differences(models: pd.DataFrame, baseline: str = "rule_baseline",
                          n_boot: int = 5000, seed: int = 2026) -> pd.DataFrame:
    if "seed" not in models.columns:
        return pd.DataFrame()
    b = models[models["model"] == baseline].set_index("seed")["episode_recall"]
    rows = []
    rng = np.random.default_rng(seed)
    for model in sorted(set(models["model"]) - {baseline}):
        a = models[models["model"] == model].set_index("seed")["episode_recall"]
        common = a.index.intersection(b.index)
        diff = (a.loc[common] - b.loc[common]).dropna().to_numpy(float)
        if len(diff) < 2:
            continue
        bs = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)])
        rows.append({
            "model": model,
            "n": int(len(diff)),
            "mean_diff": float(diff.mean()),
            "ci_lo": float(np.quantile(bs, 0.025)),
            "ci_hi": float(np.quantile(bs, 0.975)),
        })
    return pd.DataFrame(rows)


def plot_forest(df, outdir, formats, manifest):
    if df.empty:
        return
    d = df.sort_values("mean_diff")
    y = np.arange(len(d))
    mean = d["mean_diff"].to_numpy(float) * 100
    lo = d["ci_lo"].to_numpy(float) * 100
    hi = d["ci_hi"].to_numpy(float) * 100
    fig, ax = plt.subplots(figsize=(9, max(5, 0.45*len(d)+1.5)))
    ax.errorbar(mean, y, xerr=[mean-lo, hi-mean], fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y, [pretty_model(x) for x in d["model"]])
    ax.set_xlabel("Episode recall difference vs rule baseline, percentage points")
    ax.set_title("Paired bootstrap across regenerated seeds")
    ax.grid(axis="x", alpha=0.25)
    safe_save(fig, outdir, "13_bootstrap_vs_rule", formats, manifest,
              "Paired bootstrap episode-recall differences vs rule baseline")


def build_summary(models: pd.DataFrame, outdir: Path):
    numeric = [
        "auc", "ap", "episode_recall", "median_ttd_intervals",
        "incident_precision", "false_incidents_per_day",
        "alerts_per_day_network", "window_fpr", "window_recall",
        "precision_at_op_prevalence", "coverage", "fit_seconds",
    ]
    cols = [c for c in numeric if c in models.columns]
    if not cols:
        return
    agg = models.groupby(["model", "family"])[cols].agg(["mean", "std", "count"])
    agg.columns = ["_".join(x) for x in agg.columns]
    agg.reset_index().to_csv(outdir / "all_models_summary.csv", index=False)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = Path(args.input_root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []

    models = read_many(root, "models.csv")
    if models.empty:
        raise RuntimeError(f"No models.csv found below {root}")
    models.to_csv(outdir / "all_models_multiseed.csv", index=False)
    build_summary(models, outdir)

    raw_scores = read_score_exports(root)
    if not raw_scores.empty:
        raw_scores.to_csv(outdir / "all_models_test_scores.csv", index=False)

    ab_fs = read_many(root, "ablation_feature_sets.csv")
    ab_t = read_many(root, "ablation_temporal.csv")
    lomfo = read_many(root, "lomfo.csv")
    curve = read_many(root, "detectability_curve.csv")
    soph = read_many(root, "sophistication_sweep.csv")

    bar_with_error(models, "episode_recall", "Episode-level detection by model",
                   "Episode recall, %", outdir, "01_episode_recall", args.formats, manifest,
                   scale=100.0, xlim=(0, 100))
    grouped_metric(models, [("auc", "ROC AUC", 1.0), ("ap", "Average precision", 1.0)],
                   "Ranking metrics by model", "Metric", outdir, "02_auc_ap",
                   args.formats, manifest, ylim=(0, 1))
    bar_with_error(models, "window_fpr", "Window false-positive rate by model",
                   "Window FPR, %", outdir, "03_window_fpr", args.formats, manifest, scale=100.0)
    bar_with_error(models, "median_ttd_intervals", "Median time to detection by model",
                   "Median TTD, minutes", outdir, "04_ttd", args.formats, manifest,
                   scale=INTERVAL_SECONDS / 60.0)
    grouped_metric(models,
                   [("incident_precision", "Incident precision", 1.0),
                    ("window_recall", "Window recall", 1.0)],
                   "Operational precision and recall", "Metric", outdir,
                   "05_precision_recall", args.formats, manifest, ylim=(0, 1))
    plot_fit_time(models, outdir, args.formats, manifest)
    plot_scatter(models, outdir, args.formats, manifest)
    heatmap_seed_model(models, "episode_recall", "Episode recall across regenerated seeds",
                       outdir, "08_seed_recall_heatmap", args.formats, manifest, scale=100.0)
    heatmap_seed_model(models, "window_fpr", "Window FPR across regenerated seeds",
                       outdir, "09_seed_fpr_heatmap", args.formats, manifest, scale=100.0)
    plot_ablation(ab_fs, "feature_set", "Feature-set ablation", outdir,
                  "10_feature_set_ablation", args.formats, manifest)
    plot_ablation(ab_t, "variant", "Temporal/zone feature ablation", outdir,
                  "10b_temporal_ablation", args.formats, manifest)
    plot_curve(curve, "hidden_rate", "Detectability versus hidden activity rate",
               outdir, "10c_detectability_curve", args.formats, manifest)
    plot_sophistication(soph, outdir, args.formats, manifest)
    plot_lomfo(lomfo, outdir, args.formats, manifest)

    if "coverage" in models.columns:
        bar_with_error(models, "coverage", "Score coverage by model",
                       "Fraction of test windows with a score", outdir,
                       "12b_score_coverage", args.formats, manifest, scale=1.0, xlim=(0, 1))

    boot = bootstrap_differences(models)
    if not boot.empty:
        boot.to_csv(outdir / "bootstrap_vs_rule.csv", index=False)
        plot_forest(boot, outdir, args.formats, manifest)

    # Operational alert volume is often easier to interpret than FPR.
    if "alerts_per_day_network" in models.columns:
        bar_with_error(models, "alerts_per_day_network", "Alert load by model",
                       "Alerts per day across network", outdir, "14_alerts_per_day",
                       args.formats, manifest)

    # Low-prevalence precision is typically tiny, so use a logarithmic x-axis.
    plot_roc_curves(raw_scores, outdir, args.formats, manifest)
    plot_pr_curves(raw_scores, outdir, args.formats, manifest)

    if "precision_at_op_prevalence" in models.columns:
        g = numeric_mean_std(models, "precision_at_op_prevalence")
        g = g[np.isfinite(g["mean"]) & (g["mean"] > 0)]
        if not g.empty:
            fig, ax = plt.subplots(figsize=(9.5, max(5, 0.48*len(g)+1.5)))
            y = np.arange(len(g))
            ax.barh(y, g["mean"])
            ax.set_yticks(y, [pretty_model(x) for x in g["model"]])
            ax.invert_yaxis()
            ax.set_xscale("log")
            ax.set_xlabel("Precision at operational prevalence (log scale)")
            ax.set_title("Operational-prevalence precision")
            ax.grid(axis="x", alpha=0.25)
            safe_save(fig, outdir, "15_operational_precision", args.formats, manifest,
                      "Precision recalculated at operational prevalence")

    (outdir / "plots_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "models_rows": int(len(models)),
        "models": sorted(models["model"].dropna().astype(str).unique().tolist()),
        "seeds": sorted(pd.to_numeric(models.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist()),
        "figures": manifest,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
