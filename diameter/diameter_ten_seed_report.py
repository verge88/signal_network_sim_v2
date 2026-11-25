"""Collect and aggregate independent full Diameter experiments."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import precision_recall_curve, roc_curve

from diameter_generate_plots import MAIN_MODELS, configure_style, save_figure

ATTACKS = {
    "node_compromise": "Компрометация", "signaling_dos": "Сигнальный DoS",
    "fraud_profile": "Изменение профиля", "location_tracking": "Отслеживание",
    "sms_data_interception": "Перехват SMS и данных",
}
KEY_METRICS = {
    "auc_roc": "ROC-AUC", "auc_pr": "AUC-PR", "f1": "F1",
    "fpr": "Доля ложных тревог", "recall": "Общая полнота",
    **{f"recall_{k}": "Полнота: " + v for k, v in ATTACKS.items()},
}
MODELS_TO_COMPARE = [
    "Cascade + report recurrence", "Cascade", "Gradient Boosting", "Meta-Ensemble",
]
GRID = np.linspace(0, 1, 201)


def collect(root):
    metrics = pd.read_csv(root / "full_comparison_diameter.csv")
    attacks = pd.read_csv(root / "recall_by_simulation_and_attack.csv")
    per_sim = pd.read_csv(root / "metrics_by_simulation.csv")
    names = set(metrics["name"])
    if len(metrics) != 23 or len(names) != 23:
        raise ValueError("Expected exactly 23 distinct model configurations")
    if len(per_sim) != 46 or per_sim["simulation"].nunique() != 2:
        raise ValueError("Expected all models on each of two test simulations")
    if any(set(g["name"]) != names for _, g in per_sim.groupby("simulation")):
        raise ValueError("Test simulations do not contain the same model suite")
    if len(attacks) != 230 or attacks.duplicated(["simulation", "name", "attack_type"]).any():
        raise ValueError("Invalid per-attack test coverage")
    if set(attacks["attack_type"]) != set(ATTACKS):
        raise ValueError("Missing attack class")

    for key in ("tp", "fp", "tn", "fn"):
        expected = metrics.set_index("name")[key].sort_index()
        actual = per_sim.groupby("name")[key].sum().sort_index()
        pd.testing.assert_series_equal(actual, expected)
    by_attack = attacks.groupby(["name", "attack_type"])[["detected", "windows"]].sum()
    if not (by_attack.groupby("name")["detected"].sum() == metrics.set_index("name")["tp"].sort_index()).all():
        raise ValueError("Per-attack hits disagree with the confusion matrices")
    if not (by_attack.groupby("name")["windows"].sum() ==
            (metrics.set_index("name")["tp"] + metrics.set_index("name")["fn"]).sort_index()).all():
        raise ValueError("Per-attack totals disagree with the confusion matrices")
    grouped_recall = (by_attack["detected"] / by_attack["windows"]).unstack()
    grouped_recall.columns = [f"recall_{name}" for name in grouped_recall]
    metrics = metrics.join(grouped_recall, on="name")
    metrics["recall"] = metrics["tp"] / (metrics["tp"] + metrics["fn"])
    metrics.to_csv(root / "run_metrics.csv", index=False)

    artifacts = json.loads((root / "plot_artifacts.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((root / "overfitting_diagnostics.json").read_text(encoding="utf-8"))
    curves = {}
    for result in artifacts["results_list"]:
        scores = result["scores"]
        labels = result.get("y_true", artifacts["y_test"])
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        # Upper envelopes resolve tied coordinates before gridding.
        roc = pd.Series(tpr, index=fpr).groupby(level=0).max()
        pr = pd.Series(precision, index=recall).groupby(level=0).max()
        pr = pr.iloc[::-1].cummax().iloc[::-1]
        curves[result["name"]] = {
            "roc": np.interp(GRID, roc.index, roc).tolist(),
            "pr": np.interp(GRID, pr.index, pr).tolist(),
        }
    if set(curves) != names:
        raise ValueError("ROC and precision–recall curves are incomplete")
    (root / "grid_curves.json").write_text(json.dumps(curves, allow_nan=False), encoding="utf-8")

    histories = []
    for name, history in artifacts["trainers_history"].items():
        best = diagnostics["neural"][name]["best_epoch"]
        for epoch, (train, validation) in enumerate(zip(
            history["train_eval_losses"], history["val_losses"], strict=True
        ), start=1):
            histories.append({"name": name, "epoch": epoch, "train_loss": train,
                              "validation_loss": validation, "best_epoch": best})
    if len({h["name"] for h in histories}) != 11:
        raise ValueError("Expected 11 neural training histories")
    pd.DataFrame(histories).to_csv(root / "epoch_history.csv", index=False)
    print("Verified 23 configurations, 46 model/simulation rows and 230 attack rows.")


def stats(values):
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    margin = float(t.ppf(.975, n - 1) * sd / np.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": mean - margin,
            "ci95_high": mean + margin, "min": float(np.min(values)),
            "max": float(np.max(values))}


def plot_metric_panels(summary, names, out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharey=True)
    ys = np.arange(len(names))
    for ax, metric in zip(axes.flat, ("auc_pr", "f1", "fpr", "recall_node_compromise")):
        values = summary.loc[summary.metric.eq(metric)].set_index("name").loc[names]
        scale = 100 if metric in ("fpr", "recall_node_compromise") else 1
        point = values["mean"].to_numpy()
        err = np.vstack([point - values["ci95_low"].clip(lower=0),
                         values["ci95_high"].clip(upper=1) - point])
        ax.errorbar(point * scale, ys, xerr=err * scale, fmt="o",
                    color="0.2", capsize=2, markersize=4)
        ax.set_xlabel(KEY_METRICS[metric] + (", %" if scale == 100 else ""))
        ax.set_yticks(ys, names)
        ax.grid(True, axis="x", linestyle=":")
    axes[0, 0].invert_yaxis()
    fig.suptitle("Средние значения и 95%-е доверительные интервалы по 10 прогонам")
    fig.tight_layout()
    save_figure(fig, out, "20_mean_model_metrics")


def plot_attack_summary(summary, names, out):
    data = summary.pivot(index="name", columns="metric", values="mean")
    heat = data.loc[names, [f"recall_{key}" for key in ATTACKS]].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(heat, cmap="Greys", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ATTACKS)), list(ATTACKS.values()), rotation=20, ha="right")
    ax.set_yticks(range(len(names)), names)
    for (i, j), v in np.ndenumerate(heat):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                color="white" if v > 60 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Полнота, %")
    ax.set_title("Средняя полнота по типам аномалий")
    fig.tight_layout()
    save_figure(fig, out, "21_mean_attack_recall")


def plot_run_comparison(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric in zip(axes.flat, ("auc_pr", "fpr", "recall_node_compromise",
                                     "recall_signaling_dos")):
        scale = 100 if metric != "auc_pr" else 1
        for name, style, marker in zip(MODELS_TO_COMPARE, ("-", "--", "-.", ":"),
                                       ("o", "^", "s", "x")):
            r = rows[rows["name"] == name].sort_values("run")
            ax.plot(r["run"], r[metric] * scale, linestyle=style,
                    marker=marker, color="0.15", label=name)
        ax.set_title(KEY_METRICS[metric] + (", %" if scale == 100 else ""))
        ax.set_xlabel("Независимый прогон")
        ax.set_xticks(range(1, 11))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, .90))
    save_figure(fig, out, "22_paired_runs")


def plot_curve_pages(grids, names, summary, out):
    for group in range(0, len(names), 8):
        for key, x, y, label in (("roc", "Доля ложных тревог", "Полнота", "ROC"),
                                  ("pr", "Полнота", "Точность", "Precision–Recall")):
            fig, ax = plt.subplots(figsize=(8.5, 6))
            for i, name in enumerate(names[group:group + 8]):
                values = np.array([g[name][key] for g in grids])
                mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
                m = "auc_roc" if key == "roc" else "auc_pr"
                ap = summary.loc[summary["name"].eq(name) & summary.metric.eq(m), "mean"].iloc[0]
                line, = ax.plot(GRID, mean, label=f"{name} ({ap:.3f})",
                                linestyle=("-", "--", "-.", ":")[i % 4])
                ax.fill_between(GRID, (mean - sd).clip(0, 1), (mean + sd).clip(0, 1),
                                color=line.get_color(), alpha=.07)
            ax.set(xlabel=x, ylabel=y, xlim=(0, 1), ylim=(0, 1.02),
                   title=f"Средние кривые {label}, группа {group // 8 + 1}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            save_figure(fig, out, f"23_mean_{key}_{group // 8 + 1}")


def plot_epoch_pages(histories, out):
    names = list(pd.unique(histories["name"]))
    for start in range(0, len(names), 4):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, name in zip(axes.flat, names[start:start + 4]):
            rows = histories[histories["name"].eq(name)]
            data = rows.groupby("epoch")[["train_loss", "validation_loss"]].agg(
                ["mean", "std", "count"])
            enough = data[("train_loss", "count")] >= 5
            for mode, line, label in (("train_loss", "-", "Обучение"),
                                       ("validation_loss", "--", "Валидация")):
                y = data.loc[enough, (mode, "mean")]
                sd = data.loc[enough, (mode, "std")]
                ax.plot(y.index, y, line, color="0.25", label=label)
                ax.fill_between(y.index, (y - sd).clip(lower=1e-12), y + sd,
                                color="0.4", alpha=.13)
            ax.set_yscale("log")
            ax.set(title=name, xlabel="Эпоха", ylabel="Функция потерь")
            ax.legend()
        for ax in list(axes.flat)[len(names[start:start + 4]):]:
            ax.set_visible(False)
        fig.suptitle("Кривые обучения по эпохам с данными как минимум пяти прогонов")
        fig.tight_layout()
        save_figure(fig, out, f"24_mean_training_curves_{start // 4 + 1}")


def aggregate(source, out):
    bundles = sorted(source.glob("run-*"))
    if len(bundles) != 10:
        raise ValueError("Expected exactly 10 completed run artifacts")
    seeds, metrics, grids, histories, simulations, attacks = [], [], [], [], [], []
    datasets = set()
    reference_version = None
    reference_calibration = None
    for bundle in bundles:
        info = json.loads((bundle / "seeds.json").read_text(encoding="utf-8"))
        version = (bundle / "versions.txt").read_text()
        calibration = info["calibration_sha256"]
        if reference_version is not None and version != reference_version:
            raise ValueError("Python package versions differ across runs")
        if reference_calibration is not None and calibration != reference_calibration:
            raise ValueError("Calibration profiles differ across runs")
        reference_version = version
        reference_calibration = calibration
        if any(d in datasets for d in info["dataset_sha256"].values()):
            raise ValueError("A dataset was reused")
        datasets.update(info["dataset_sha256"].values())
        run = info["run"]
        seeds.append(info)
        metrics.append(pd.read_csv(bundle / "run_metrics.csv").assign(run=run))
        grids.append(json.loads((bundle / "grid_curves.json").read_text(encoding="utf-8")))
        histories.append(pd.read_csv(bundle / "epoch_history.csv").assign(run=run))
        simulations.append(pd.read_csv(bundle / "metrics_by_simulation.csv").assign(run=run))
        attacks.append(pd.read_csv(bundle / "recall_by_simulation_and_attack.csv").assign(run=run))

    if {s["run"] for s in seeds} != set(range(1, 11)):
        raise ValueError("Replications are missing or duplicated")
    sim_seeds = [seed for run in seeds for seed in run["simulation_seeds"].values()]
    if len(set(sim_seeds)) != 50 or len({s["model_seed"] for s in seeds}) != 10 or len(datasets) != 50:
        raise ValueError("Replication seeds or data are not disjoint")
    runs = pd.concat(metrics, ignore_index=True)
    if runs.duplicated(["run", "name"]).any() or len(runs) != 230:
        raise ValueError("Incomplete or duplicated per-run model coverage")
    if not np.isfinite(runs[list(KEY_METRICS)].to_numpy()).all():
        raise ValueError("Non-finite metric")
    if not runs.groupby("name")["run"].count().eq(10).all():
        raise ValueError("All models must have one measurement in each run")
    out.mkdir(parents=True, exist_ok=True)
    reports = out / "per_run"
    for bundle in bundles:
        shutil.copytree(bundle, reports / bundle.name, dirs_exist_ok=True)
    runs.to_csv(out / "metrics_by_run.csv", index=False)
    pd.concat(simulations).to_csv(out / "metrics_by_test_simulation.csv", index=False)
    pd.concat(attacks).to_csv(out / "attack_recall_by_test_simulation.csv", index=False)
    epochs = pd.concat(histories, ignore_index=True)
    epochs.to_csv(out / "epoch_history_by_run.csv", index=False)
    (out / "seeds.json").write_text(json.dumps(seeds, indent=2), encoding="utf-8")

    summary = pd.DataFrame([
        {"name": name, "metric": metric, **stats(group[metric].to_numpy())}
        for name, group in runs.groupby("name") for metric in KEY_METRICS
    ])
    summary.to_csv(out / "metrics_mean_sd_ci95_long.csv", index=False)
    mean_sd = summary.pivot(index="name", columns="metric", values=["mean", "sd"])
    mean_sd.columns = [f"{metric}_{stat}" for stat, metric in mean_sd]
    mean_sd.sort_values("auc_pr_mean", ascending=False).to_csv(out / "metrics_mean_sd.csv")
    names = sorted(runs["name"].unique(),
                   key=lambda n: -float(mean_sd.loc[n, "auc_pr_mean"]))

    differences = []
    for metric in KEY_METRICS:
        paired = runs.pivot(index="run", columns="name", values=metric)
        for baseline in MODELS_TO_COMPARE[1:]:
            delta = paired[MODELS_TO_COMPARE[0]] - paired[baseline]
            differences.append({"metric": metric, "baseline": baseline, **stats(delta.to_numpy())})
    pd.DataFrame(differences).to_csv(out / "paired_differences.csv", index=False)
    pd.DataFrame([
        {"name": name, "epoch": epoch, "active_runs": group["run"].nunique()}
        for (name, epoch), group in epochs.groupby(["name", "epoch"])
    ]).to_csv(out / "training_curve_support.csv", index=False)

    figures = out / "summary_figures"
    figures.mkdir(exist_ok=True)
    configure_style()
    plot_metric_panels(summary, names, figures)
    plot_attack_summary(summary, names, figures)
    plot_run_comparison(runs, figures)
    plot_curve_pages(grids, names, summary, figures)
    plot_epoch_pages(epochs, figures)
    (out / "protocol.txt").write_text(
        "Replication = one model fit/validation/stacking experiment, evaluated on two disjoint "
        "test simulations pooled within that replication. Table = unweighted mean of 10 "
        "replication metrics; sample SD and two-sided Student t CI use replications, not windows. "
        "Graph CIs for proportions are clipped to [0,1]; CSV intervals are untruncated. "
        "ROC/PR figures use a common grid and mean ± one sample SD; AUCs come from raw scores. "
        "Epoch bands show mean ± SD among runs reaching each epoch, with at least five runs. "
        "Per-run figures use the standard Diameter generator. Simulated data use the same "
        "topology/scenario specification and the same frozen calibration profile.\n",
        encoding="utf-8")
    view = ["auc_roc", "auc_pr", "f1", "fpr", "recall_node_compromise", "recall_signaling_dos"]
    compact = mean_sd.loc[names, [f"{k}_mean" for k in view]]
    print("MEAN_TABLE_START")
    print(compact.to_csv(float_format="%.7f"), end="")
    print("MEAN_TABLE_END")
    print(f"Packaged {len(names)} models × 10 runs; {len(list(figures.glob('*.png')))} aggregate plots.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("collect")
    prepare.add_argument("results", type=Path)
    combine = sub.add_parser("aggregate")
    combine.add_argument("source", type=Path)
    combine.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.results)
    else:
        aggregate(args.source, args.out)


if __name__ == "__main__":
    main()
