#!/usr/bin/env python3
"""
Publication-quality figure generator for the Diameter experiment.

This file is the single plotting entry point for the calibrated Diameter
experiment. Training code saves raw plotting artifacts; this script converts
those artifacts and JSON/CSV summaries into reproducible PNG + SVG figures.

Example:
    python diameter/diameter_generate_plots.py \
        --results-dir results_diameter_v1 \
        --external-benign results_calibrated_external_benign/summary.json \
        --external-attacks results_calibrated_external_attacks/summary.json \
        --calibration-verification results_diameter_calibration/calibrated_dataset_verification.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


MAIN_MODELS = [
    "Isolation Forest",
    "One-Class SVM",
    "Autoencoder",
    "VAE",
    "LSTM-Autoencoder",
    "Temporal-Attention-AE",
    "Random Forest",
    "Gradient Boosting",
    "Cascade",
    "Meta-Ensemble",
]

ROC_MODELS = [
    "Isolation Forest",
    "One-Class SVM",
    "Autoencoder",
    "VAE",
    "Random Forest",
    "Gradient Boosting",
    "Cascade",
    "Meta-Ensemble",
]

ATTACK_ORDER = [
    "location_tracking",
    "sms_data_interception",
    "signaling_dos",
    "fraud_profile",
    "node_compromise",
]

ATTACK_LABELS = {
    "location_tracking": "Отслеживание",
    "sms_data_interception": "Перехват SMS/данных",
    "signaling_dos": "Сигнальный DoS",
    "fraud_profile": "Изменение профиля",
    "node_compromise": "Компрометация узла",
}


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 320,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.5,
    })


def load_json(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", format=ext)
        paths.append(str(path))
    plt.close(fig)
    print(f"Saved: {stem}.png / .svg")
    return paths


def get_result_map(results_list: Sequence[dict]) -> Dict[str, dict]:
    return {str(r["name"]): r for r in results_list}


def plot_metric_comparison(df: pd.DataFrame, out_dir: Path) -> List[str]:
    d = df[df["name"].isin(MAIN_MODELS)].copy()
    order = [m for m in MAIN_MODELS if m in set(d["name"])]
    d["name"] = pd.Categorical(d["name"], categories=order, ordered=True)
    d = d.sort_values("name")

    x = np.arange(len(d))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 5.4))
    bars1 = ax.bar(x - width, d["auc_roc"], width, label="AUC-ROC",
                   facecolor="0.20", edgecolor="black")
    bars2 = ax.bar(x, d["auc_pr"], width, label="AUC-PR",
                   facecolor="0.55", edgecolor="black", hatch="//")
    bars3 = ax.bar(x + width, d["f1"], width, label="F1",
                   facecolor="0.85", edgecolor="black", hatch="..")
    ax.set_xticks(x)
    ax.set_xticklabels(d["name"], rotation=35, ha="right")
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0, 1.05)
    ax.set_title("Сравнение качества моделей обнаружения аномалий Diameter")
    ax.legend(ncol=3, loc="upper left")
    for bars in (bars1, bars2, bars3):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)
    fig.tight_layout()
    return save_figure(fig, out_dir, "01_model_metrics")


def plot_roc(results_list: Sequence[dict], y_test: np.ndarray,
             out_dir: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    plotted = 0
    for idx, r in enumerate(results_list):
        name = str(r["name"])
        if name not in ROC_MODELS:
            continue
        scores = np.asarray(r.get("scores", []))
        if len(scores) != len(y_test):
            continue
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax.plot(fpr, tpr, linestyle=styles[idx % len(styles)],
                label=f"{name} (AUC={r['auc_roc']:.3f})")
        plotted += 1
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Доля ложных срабатываний (FPR)")
    ax.set_ylabel("Полнота обнаружения (TPR)")
    ax.set_title("ROC-кривые моделей Diameter")
    if plotted:
        ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, out_dir, "02_roc_curves")


def plot_pr(results_list: Sequence[dict], y_test: np.ndarray,
            out_dir: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    plotted = 0
    for idx, r in enumerate(results_list):
        name = str(r["name"])
        if name not in ROC_MODELS:
            continue
        scores = np.asarray(r.get("scores", []))
        if len(scores) != len(y_test):
            continue
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ax.plot(recall, precision, linestyle=styles[idx % len(styles)],
                label=f"{name} (AP={r['auc_pr']:.3f})")
        plotted += 1
    baseline = float(np.mean(y_test))
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1,
               label=f"Базовый уровень ({baseline:.3f})")
    ax.set_xlabel("Полнота")
    ax.set_ylabel("Точность")
    ax.set_title("Precision–Recall-кривые моделей Diameter")
    if plotted:
        ax.legend(loc="upper right")
    fig.tight_layout()
    return save_figure(fig, out_dir, "03_precision_recall")


def plot_attack_heatmap(results_list: Sequence[dict], out_dir: Path) -> List[str]:
    selected = [r for r in results_list if str(r["name"]) in MAIN_MODELS]
    if not selected:
        return []
    matrix = np.array([
        [float(r.get("per_attack", {}).get(a, (0, 1, 0))[2])
         for a in ATTACK_ORDER]
        for r in selected
    ])
    names = [str(r["name"]) for r in selected]

    fig, ax = plt.subplots(figsize=(10.5, 7))
    im = ax.imshow(matrix, cmap="Greys", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ATTACK_ORDER)))
    ax.set_xticklabels([ATTACK_LABELS[a] for a in ATTACK_ORDER],
                       rotation=28, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Полнота обнаружения по типам атак, %")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v > 55 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Полнота, %")
    fig.tight_layout()
    return save_figure(fig, out_dir, "04_attack_detection_heatmap")


def plot_feature_importance(artifacts: dict, out_dir: Path,
                            top_n: int = 20) -> List[str]:
    importances = np.asarray(artifacts.get("importances", []), dtype=float)
    names = list(artifacts.get("feature_cols", []))
    if len(importances) == 0 or len(importances) != len(names):
        return []

    idx = np.argsort(importances)[-top_n:]
    vals = importances[idx]
    labels = [names[i] for i in idx]
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    bars = ax.barh(np.arange(len(idx)), vals, facecolor="0.65",
                   edgecolor="black", hatch="//")
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Важность признака (Gini importance)")
    ax.set_title(f"Наиболее значимые признаки Random Forest (top-{top_n})")
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height()/2,
                f"{100*v:.1f}%", va="center", fontsize=7)
    fig.tight_layout()
    return save_figure(fig, out_dir, "05_rf_feature_importance")


def plot_ablation(artifacts: dict, out_dir: Path) -> List[str]:
    rows = artifacts.get("ablation_results", [])
    if not rows:
        return []
    labels = [str(r["config"]).replace("\n", " ") for r in rows]
    f1 = [float(r["f1"]) for r in rows]
    comp = [float(r["compromise_rate"]) for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    b1 = axes[0].bar(x, f1, facecolor="0.75", edgecolor="black", hatch="//")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("F1")
    axes[0].set_title("Абляция: итоговый F1")
    for b, v in zip(b1, f1):
        axes[0].text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}",
                     ha="center", fontsize=8)

    b2 = axes[1].bar(x, comp, facecolor="0.45", edgecolor="black", hatch="..")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Полнота, %")
    axes[1].set_title("Абляция: компрометация узла")
    for b, v in zip(b2, comp):
        axes[1].text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.1f}%",
                     ha="center", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, out_dir, "06_ablation")


def plot_training_curves(artifacts: dict, diagnostics: dict,
                         out_dir: Path) -> List[str]:
    histories = artifacts.get("trainers_history", {})
    if not histories:
        return []
    n = len(histories)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = np.ravel(axes)
    neural_diag = diagnostics.get("neural", {})

    for ax, (name, hist) in zip(axes, histories.items()):
        train = np.asarray(hist.get("train_losses", []), dtype=float)
        val = np.asarray(hist.get("val_losses", []), dtype=float)
        if len(train):
            ax.plot(np.arange(1, len(train)+1), train, color="black",
                    linestyle="-", label="Обучение")
        if len(val):
            ax.plot(np.arange(1, len(val)+1), val, color="0.45",
                    linestyle="--", label="Валидация")
        best_epoch = neural_diag.get(name, {}).get("best_epoch")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="0.20", linestyle=":",
                       label=f"Лучшая эпоха: {best_epoch}")
        ax.set_yscale("log")
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Ошибка реконструкции")
        ax.set_title(name)
        ax.legend()
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Кривые обучения нейросетевых детекторов Diameter", y=1.01)
    fig.tight_layout()
    return save_figure(fig, out_dir, "07_training_curves")


def plot_confusion(results_list: Sequence[dict], y_test: np.ndarray,
                   out_dir: Path) -> List[str]:
    names = ["Gradient Boosting", "Random Forest", "Cascade", "Meta-Ensemble"]
    rmap = get_result_map(results_list)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, name in zip(axes, names):
        r = rmap.get(name)
        if r is None:
            ax.axis("off")
            continue
        pred = np.asarray(r.get("y_pred", []), dtype=int)
        if len(pred) != len(y_test):
            ax.axis("off")
            continue
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="Greys")
        for i in range(2):
            for j in range(2):
                v = int(cm[i, j])
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        color="white" if v > cm.max()*0.55 else "black",
                        fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Норма", "Атака"])
        ax.set_yticklabels(["Норма", "Атака"])
        ax.set_xlabel("Предсказание")
        ax.set_ylabel("Истина")
        ax.set_title(f"{name}\nF1={r['f1']:.3f}")
    fig.suptitle("Матрицы ошибок основных управляемых моделей", y=1.03)
    fig.tight_layout()
    return save_figure(fig, out_dir, "08_confusion_matrices")


def plot_supervised_generalization(diagnostics: dict,
                                   out_dir: Path) -> List[str]:
    sup = diagnostics.get("supervised", {})
    names = ["Random Forest", "Gradient Boosting", "Cascade", "Meta-Ensemble"]
    names = [n for n in names if n in sup]
    if not names:
        return []

    train = []
    val = []
    test = []
    for n in names:
        d = sup[n]
        train.append(float(d.get("train_auc_pr", np.nan)))
        val.append(float(d.get("val_auc_pr", d.get("fit_auc_pr_on_validation", np.nan))))
        test.append(float(d.get("test_auc_pr", np.nan)))

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x-w, train, w, label="Train", facecolor="0.20", edgecolor="black")
    ax.bar(x, val, w, label="Validation", facecolor="0.55",
           edgecolor="black", hatch="//")
    ax.bar(x+w, test, w, label="Test", facecolor="0.85",
           edgecolor="black", hatch="..")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC-PR")
    ax.set_title("Обобщающая способность управляемых моделей")
    ax.legend(ncol=3)
    fig.tight_layout()
    return save_figure(fig, out_dir, "09_supervised_generalization")


def plot_fpr_vs_f1(df: pd.DataFrame, out_dir: Path) -> List[str]:
    d = df[df["name"].isin(MAIN_MODELS)].copy()
    if d.empty:
        return []

    # Fixed offsets keep labels deterministic and readable without requiring
    # an additional layout dependency such as adjustText.
    offsets = {
        "Meta-Ensemble": (7, 8),
        "Gradient Boosting": (7, 5),
        "Random Forest": (7, 5),
        "Cascade": (7, -9),
        "One-Class SVM": (7, 7),
        "Autoencoder": (7, -8),
        "Isolation Forest": (7, -1),
        "VAE": (7, 7),
        "Temporal-Attention-AE": (7, 5),
        "LSTM-Autoencoder": (7, 5),
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for _, row in d.iterrows():
        name = str(row["name"])
        x = float(row["fpr"]) * 100
        y = float(row["f1"])
        ax.scatter(x, y, facecolors="white", edgecolors="black", s=55)
        dx, dy = offsets.get(name, (6, 4))
        ax.annotate(
            name, (x, y), xytext=(dx, dy),
            textcoords="offset points", fontsize=7)
    ax.set_xlabel("FPR, %")
    ax.set_ylabel("F1")
    ax.set_title("Компромисс между ложными срабатываниями и F1")
    fig.tight_layout()
    return save_figure(fig, out_dir, "10_fpr_f1_tradeoff")


def plot_external_validation(summary: dict, out_dir: Path) -> List[str]:
    if not summary:
        return []
    models = ["Portable RF", "Legacy IF", "Robust IF"]
    tpr = [
        100*float(summary.get("rf_external_attack_tpr", 0.0)),
        100*float(summary.get("iso_legacy_external_attack_tpr", 0.0)),
        100*float(summary.get("iso_robust_external_attack_tpr", 0.0)),
    ]
    fpr = [
        100*float(summary.get("rf_external_benign_fpr", 0.0)),
        100*float(summary.get("iso_legacy_external_benign_fpr", 0.0)),
        100*float(summary.get("iso_robust_external_benign_fpr", 0.0)),
    ]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    b1 = axes[0].bar(x, tpr, facecolor="0.40", edgecolor="black", hatch="//")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("TPR, %")
    axes[0].set_title("Внешние воспроизведённые атаки")
    for b, v in zip(b1, tpr):
        axes[0].text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.0f}%",
                     ha="center", fontsize=8)

    b2 = axes[1].bar(x, fpr, facecolor="0.80", edgecolor="black", hatch="..")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("FPR, %")
    axes[1].set_title("Внешние доброкачественные трассы")
    for b, v in zip(b2, fpr):
        axes[1].text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.0f}%",
                     ha="center", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, out_dir, "11_external_validation")


def plot_calibration(verification: dict, out_dir: Path) -> List[str]:
    if not verification:
        return []
    target = (
        verification.get("target_s6a_command_dist")
        or verification.get("target_s6a_distribution")
        or verification.get("target_s6a")
    )
    simulated = (
        verification.get("simulated_benign_s6a_command_dist")
        or verification.get("simulated_benign_s6a_distribution")
        or verification.get("simulated_benign_s6a")
    )
    if not isinstance(target, dict) or not isinstance(simulated, dict):
        return []
    keys = [k for k in target.keys() if k in simulated]
    if not keys:
        return []
    x = np.arange(len(keys))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x-w/2, [100*float(target[k]) for k in keys], w,
           label="Целевой профиль", facecolor="0.30", edgecolor="black")
    ax.bar(x+w/2, [100*float(simulated[k]) for k in keys], w,
           label="Имитационная модель", facecolor="0.80",
           edgecolor="black", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Доля запросов, %")
    ax.set_title("Проверка переноса калиброванного профиля команд S6a")
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, out_dir, "12_calibration_s6a")


def build_manifest(out_dir: Path, generated: Iterable[str],
                   source_files: Iterable[Path]) -> None:
    manifest = {
        "generator": "diameter/diameter_generate_plots.py",
        "formats": ["png", "svg"],
        "generated_files": sorted(str(Path(p)) for p in generated),
        "source_files": [str(p) for p in source_files if p.exists()],
        "note": (
            "Figures are generated only from saved experiment artifacts and "
            "summaries. No values are interpolated or manually substituted."
        ),
    }
    (out_dir / "figures_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path,
                   default=Path("results_diameter_v1"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--external-benign", type=Path,
                   default=Path("results_calibrated_external_benign/summary.json"))
    p.add_argument("--external-attacks", type=Path,
                   default=Path("results_calibrated_external_attacks/summary.json"))
    p.add_argument("--calibration-verification", type=Path,
                   default=Path(
                       "results_diameter_calibration/"
                       "calibrated_dataset_verification.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()

    results_dir = args.results_dir
    out_dir = args.out_dir or (results_dir / "publication_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "full_comparison_diameter.csv"
    json_artifacts_path = results_dir / "plot_artifacts.json"
    pkl_path = results_dir / "plot_artifacts.pkl"
    diag_path = results_dir / "overfitting_diagnostics.json"

    required = [csv_path, diag_path]
    missing = [str(p) for p in required if not p.exists()]
    if not json_artifacts_path.exists() and not pkl_path.exists():
        missing.append(
            f"{json_artifacts_path} (preferred) or {pkl_path} (legacy)")
    if missing:
        raise SystemExit(f"Missing required plotting inputs: {missing}")

    metrics = pd.read_csv(csv_path)
    if json_artifacts_path.exists():
        artifacts = json.loads(
            json_artifacts_path.read_text(encoding="utf-8"))
        artifacts_source = json_artifacts_path
    else:
        with pkl_path.open("rb") as f:
            artifacts = pickle.load(f)
        artifacts_source = pkl_path
    diagnostics = load_json(diag_path)
    external_attacks = load_json(args.external_attacks)
    calibration = load_json(args.calibration_verification)

    results_list = artifacts["results_list"]
    y_test = np.asarray(artifacts["y_test"], dtype=int)

    generated: List[str] = []
    generated += plot_metric_comparison(metrics, out_dir)
    generated += plot_roc(results_list, y_test, out_dir)
    generated += plot_pr(results_list, y_test, out_dir)
    generated += plot_attack_heatmap(results_list, out_dir)
    generated += plot_feature_importance(artifacts, out_dir)
    generated += plot_ablation(artifacts, out_dir)
    generated += plot_training_curves(artifacts, diagnostics, out_dir)
    generated += plot_confusion(results_list, y_test, out_dir)
    generated += plot_supervised_generalization(diagnostics, out_dir)
    generated += plot_fpr_vs_f1(metrics, out_dir)
    generated += plot_external_validation(external_attacks, out_dir)
    generated += plot_calibration(calibration, out_dir)

    source_files = [
        csv_path,
        artifacts_source,
        diag_path,
        args.external_benign,
        args.external_attacks,
        args.calibration_verification,
    ]
    build_manifest(out_dir, generated, source_files)
    print(f"Generated {len(generated)//2} figures in {out_dir}")


if __name__ == "__main__":
    main()
