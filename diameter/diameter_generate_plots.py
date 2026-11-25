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
    "Cascade + report recurrence",
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
    "Cascade + report recurrence",
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
        labels = np.asarray(r.get("y_true", y_test))
        if len(scores) != len(labels):
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
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
        labels = np.asarray(r.get("y_true", y_test))
        if len(scores) != len(labels):
            continue
        precision, recall, _ = precision_recall_curve(labels, scores)
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


def plot_node_compromise_recall(results_list: Sequence[dict],
                                out_dir: Path,
                                target: float = 85.0) -> List[str]:
    rows = []
    for r in results_list:
        name = str(r.get("name", ""))
        if name not in MAIN_MODELS:
            continue
        rec = r.get("per_attack", {}).get("node_compromise")
        if rec is None:
            continue
        rows.append((name, float(rec[2])))
    if not rows:
        return []

    rows.sort(key=lambda x: x[1])
    names = [x[0] for x in rows]
    values = [x[1] for x in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        y, values, facecolor="0.70", edgecolor="black", hatch="//")
    ax.axvline(
        target, color="black", linestyle="--", linewidth=1.2,
        label=f"Целевой уровень {target:.0f}%")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Полнота обнаружения компрометации узла, %")
    ax.set_title("Обнаружение атаки «компрометация узла»")
    for b, v in zip(bars, values):
        ax.text(v + 1.0, b.get_y() + b.get_height()/2,
                f"{v:.1f}%", va="center", fontsize=8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, out_dir, "13_node_compromise_recall")


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
                         out_dir: Path, specialized: bool = False) -> List[str]:
    histories = {
        name: hist for name, hist in artifacts.get("trainers_history", {}).items()
        if name.startswith("Ens:") == specialized
    }
    if not histories:
        return []
    n = len(histories)
    cols = 3 if specialized else 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 3.8 * rows), squeeze=False)
    axes = axes.ravel()
    neural_diag = diagnostics.get("neural", {})

    for ax, (name, hist) in zip(axes, histories.items()):
        train = np.asarray(
            hist.get("train_eval_losses", hist.get("train_losses", [])),
            dtype=float)
        val = np.asarray(hist.get("val_losses", []), dtype=float)
        if len(train):
            ax.plot(np.arange(1, len(train)+1), train, color="black",
                    linestyle="-", label="Обучение (eval)")
        if len(val):
            ax.plot(np.arange(1, len(val)+1), val, color="0.45",
                    linestyle="--", label="Валидация")
        best_epoch = neural_diag.get(name, {}).get("best_epoch")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="0.20", linestyle=":",
                       label=f"Лучшая эпоха: {best_epoch}")
        ax.set_yscale("log")
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Ошибка реконструкции + штраф KL" if name == "VAE"
                      else "Ошибка реконструкции (MSE)")
        ax.set_title(name.removeprefix("Ens:"))
        ax.legend()
    for ax in axes[n:]:
        ax.axis("off")
    if specialized:
        fig.suptitle("Кривые обучения специализированных автоэнкодеров Diameter")
        stem = "14_specialized_training_curves"
    else:
        fig.suptitle("Кривые обучения нейросетевых детекторов Diameter")
        stem = "07_training_curves"
    fig.tight_layout()
    return save_figure(fig, out_dir, stem)


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
        train.append(float(d.get("train_auc_pr", d.get("fit_auc_pr_on_stack", np.nan))))
        val.append(float(d.get("val_auc_pr", d.get("fit_auc_pr_on_validation", np.nan))))
        test.append(float(d.get("test_auc_pr", np.nan)))

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x-w, train, w, label="Обучение (для Meta — stacking)", facecolor="0.20", edgecolor="black")
    ax.bar(x, val, w, label="Валидация", facecolor="0.55",
           edgecolor="black", hatch="//")
    ax.bar(x+w, test, w, label="Тест", facecolor="0.85",
           edgecolor="black", hatch="..")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC-PR")
    ax.set_title("Обобщающая способность управляемых моделей")
    ax.legend(ncol=3)
    fig.tight_layout()
    return save_figure(fig, out_dir, "09_supervised_generalization")


def plot_tree_regularization(diagnostics: dict, out_dir: Path) -> List[str]:
    reference = diagnostics.get("supervised_reference", {})
    candidate = diagnostics.get("supervised", {})
    names = [name for name in reference if name in candidate]
    if not names:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))
    for ax, key, title, ylabel, scale in (
        (axes[0], "train_minus_val_auc_pr", "Разрыв между обучением и валидацией", "Разрыв, п. п.", 100),
        (axes[1], "test_auc_pr", "Качество на тестовой симуляции", "AUC-PR (AP)", 1),
    ):
        ax.bar(x - 0.18, [reference[name][key] * scale for name in names],
               0.36, label="Прежние настройки", color="0.80", edgecolor="black", hatch="//")
        ax.bar(x + 0.18, [candidate[name][key] * scale for name in names],
               0.36, label="Новые настройки", color="0.35", edgecolor="black")
        ax.axhline(0, color="0.2", linewidth=0.8)
        ax.set_xticks(x, names)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[1].set_ylim(0, 1)
    axes[0].legend()
    fig.tight_layout()
    return save_figure(fig, out_dir, "15_tree_regularization")


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


def plot_results_by_simulation(metrics: pd.DataFrame, recall: pd.DataFrame,
                               out_dir: Path) -> List[str]:
    models = [model for model in MAIN_MODELS if model in set(metrics["name"])]
    runs = sorted(metrics["simulation"].unique())
    y = np.arange(len(models))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    attack_scores = recall.pivot(index=["simulation", "name"], columns="attack_type",
                                 values="recall").reset_index()
    data = metrics.merge(attack_scores, on=["simulation", "name"], how="left")
    panels = [
        ("auc_pr", "AUC-PR", 1, None),
        ("fpr", "Доля ложных тревог, %", 100, 2),
        ("node_compromise", "Полнота обнаружения компрометации, %", 100, 85),
        ("signaling_dos", "Полнота обнаружения сигнального DoS, %", 100, None),
    ]
    width = 0.76 / len(runs)
    for ax, (key, title, scale, target) in zip(axes.flat, panels):
        for index, run in enumerate(runs):
            values = data[data["simulation"] == run].set_index("name").reindex(models)
            if key not in values:
                continue
            shift = width * (index - (len(runs) - 1) / 2)
            ax.barh(y + shift, values[key] * scale, height=width, label=run,
                    facecolor=("0.35", "0.80")[index % 2], edgecolor="black",
                    hatch=("", "//", "..", "xx")[index % 4])
        if target is not None:
            ax.axvline(target, color="black", linestyle=":", linewidth=1)
        if key != "fpr":
            ax.set_xlim(0, scale)
        ax.set_yticks(y, models)
        ax.set_xlabel(title)
    axes[0, 0].invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(runs)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, out_dir, "17_metrics_by_simulation")


def plot_recurrence_benchmark(summary: dict, out_dir: Path) -> List[str]:
    trials = summary["tests"]
    x = np.arange(len(trials))
    baseline = "Cascade"
    candidate = "Cascade + report recurrence"
    validation = summary["validation"][candidate]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title, target in (
        (axes[0], "compromise_recall", "Полнота обнаружения компрометации",
         validation["target_compromise_recall"]),
        (axes[1], "benign_fpr", "Доля ложных тревог",
         validation["total_fpr_budget"]),
    ):
        for model, shift, label, color, hatch in (
            (baseline, -0.17, "Исходный каскад", "0.80", "//"),
            (candidate, 0.17, "Каскад с повторяемостью отчётов", "0.35", ""),
        ):
            rates = [100 * trial["models"][model][key] for trial in trials]
            bars = ax.bar(x + shift, rates, 0.34, label=label,
                          facecolor=color, edgecolor="black", hatch=hatch)
            ax.bar_label(bars, labels=[f"{rate:.2f}%" for rate in rates], padding=3)
        ax.axhline(100 * target, color="black", linestyle=":", linewidth=1.3)
        ax.set_xticks(x, [trial["simulation"] for trial in trials])
        ax.set_title(title)
        ax.set_ylabel("%")
    axes[0].set_ylim(0, 110)
    max_fpr = max(trial["models"][model]["benign_fpr"]
                  for trial in trials for model in (baseline, candidate))
    axes[1].set_ylim(0, 125 * max(max_fpr, validation["total_fpr_budget"]))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, .88))
    return save_figure(fig, out_dir, "16_report_recurrence_benchmark")


def plot_learning_curve_suite(reference: Path, points: Path, out_dir: Path) -> None:
    artifacts_file = reference / "plot_artifacts.json"
    diagnostics_file = reference / "overfitting_diagnostics.json"
    reference_file = reference / "full_comparison_diameter.csv"
    artifacts = load_json(artifacts_file)
    diagnostics = load_json(diagnostics_file)
    table = pd.read_csv(reference_file)
    if len(table) != 23 or not table["name"].is_unique:
        raise ValueError("The archived full benchmark must contain all 23 configurations")
    temporal_and_neural = {"Autoencoder", "VAE", "LSTM-Autoencoder", "Temporal-Attention-AE"}
    models = [name for name in table["name"] if name not in temporal_and_neural]
    rows = []
    sources = [artifacts_file, diagnostics_file, reference_file]
    for metadata in sorted(points.rglob("curve_point.json")):
        point = pd.read_csv(metadata.with_name(reference_file.name))
        if not set(models).issubset(point["name"]):
            raise ValueError(f"Incomplete learning curve point: {metadata}")
        rows.append(point[point["name"].isin(models)].assign(**load_json(metadata)))
        sources.extend([metadata, metadata.with_name(reference_file.name)])
    if len(rows) != 3:
        raise ValueError("Expected three subset fits")
    rows.append(table[table["name"].isin(models)].assign(
        fraction=1.0, train_rows=86400, test_rows=172800))
    curves = pd.concat(rows).sort_values(["name", "fraction"])
    if len(curves) != 4 * len(models) or curves.duplicated(["name", "fraction"]).any():
        raise ValueError("Expected four points per model")
    curves.to_csv(out_dir / "learning_curve_by_training_size.csv", index=False)
    histories = artifacts["trainers_history"]
    epoch_rows = [
        {"component": name, "epoch": i, "train_loss": train, "validation_loss": val,
         "best_epoch": diagnostics["neural"][name]["best_epoch"]}
        for name, history in histories.items()
        for i, (train, val) in enumerate(zip(history["train_eval_losses"],
                                             history["val_losses"]), start=1)
    ]
    pd.DataFrame(epoch_rows).to_csv(out_dir / "learning_curve_by_epoch.csv", index=False)

    generated = plot_training_curves(artifacts, diagnostics, out_dir)
    generated += plot_training_curves(artifacts, diagnostics, out_dir, specialized=True)
    for page, start in enumerate(range(0, len(models), 7), start=1):
        names = models[start:start + 7]
        fig, axes = plt.subplots((len(names) + 2) // 3, 3, figsize=(13, 8), squeeze=False)
        for ax, name in zip(axes.flat, names):
            run = curves[curves["name"] == name]
            ax.plot(run["train_rows"], run["auc_pr"], "-o", color="0.2", linewidth=1.4)
            ax.scatter(run["train_rows"].iloc[-1], run["auc_pr"].iloc[-1],
                       facecolors="white", edgecolors="black", zorder=3)
            ax.set_xscale("log", base=2)
            ax.set_xticks(run["train_rows"], [f"{f * 100:g}%" for f in run["fraction"]])
            ax.tick_params(axis="x", labelsize=8)
            ax.set_ylim(0, 1.02)
            ax.set_title(name, fontsize=10)
            ax.set_ylabel("AUC-PR на тесте")
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)
        fig.suptitle("Кривые обучения Diameter по доле обучающей выборки")
        fig.supxlabel("Доля обучающих окон. Пустой маркер — полный эксперимент")
        fig.tight_layout()
        generated += save_figure(fig, out_dir, f"18_learning_curve_size_{page}")
    (out_dir / "protocol.txt").write_text(
        "Epoch curves: training (eval mode) and validation losses from run 45579805922.\n"
        "Size curves: nested subsets stratified by training attack type (seed 42), "
        "calibration 43, stacking 48, fixed evaluation seeds 49 and 50.\n"
        "Only the training partition varies. The 100% point is archived. "
        "Meta-Ensemble stack size is held fixed. Test scores are diagnostic only.\n",
        encoding="utf-8")
    build_manifest(out_dir, generated, sources)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path,
                   default=Path("results_diameter_v1"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--learning-curve-points", type=Path)
    p.add_argument("--recurrence-benchmark", type=Path,
                   help="Generate the cascade comparison from a benchmark JSON")
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
    if args.learning_curve_points:
        plot_learning_curve_suite(results_dir, args.learning_curve_points, out_dir)
        return
    if args.recurrence_benchmark:
        summary = load_json(args.recurrence_benchmark)
        generated = plot_recurrence_benchmark(summary, out_dir)
        build_manifest(out_dir, generated, [args.recurrence_benchmark])
        return

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
    generated += plot_node_compromise_recall(results_list, out_dir)
    generated += plot_feature_importance(artifacts, out_dir)
    generated += plot_ablation(artifacts, out_dir)
    generated += plot_training_curves(artifacts, diagnostics, out_dir)
    generated += plot_training_curves(artifacts, diagnostics, out_dir, specialized=True)
    generated += plot_confusion(results_list, y_test, out_dir)
    generated += plot_supervised_generalization(diagnostics, out_dir)
    generated += plot_tree_regularization(diagnostics, out_dir)
    generated += plot_fpr_vs_f1(metrics, out_dir)
    generated += plot_external_validation(external_attacks, out_dir)
    generated += plot_calibration(calibration, out_dir)
    simulation_metrics = results_dir / "metrics_by_simulation.csv"
    attack_recall = results_dir / "recall_by_simulation_and_attack.csv"
    if simulation_metrics.is_file() and attack_recall.is_file():
        generated += plot_results_by_simulation(
            pd.read_csv(simulation_metrics), pd.read_csv(attack_recall), out_dir)

    source_files = [
        csv_path,
        artifacts_source,
        diag_path,
        args.external_benign,
        args.external_attacks,
        args.calibration_verification,
        simulation_metrics,
        attack_recall,
    ]
    build_manifest(out_dir, generated, source_files)
    print(f"Generated {len(generated)//2} figures in {out_dir}")


if __name__ == "__main__":
    main()
