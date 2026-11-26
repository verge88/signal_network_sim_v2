"""Publication tables and figures for the ten independent SIP repeats."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = {"TICD": "TICD", "autoencoder": "Автоэнкодер", "random_forest": "Random Forest",
         "gradient_boosting": "Gradient Boosting", "one_class_svm": "One-Class SVM",
         "isolation_forest": "Isolation Forest"}
MAIN = ("refit", "val_FPR_1pct_each_run")
METRICS = ["recall", "FPR", "precision", "F1", "ROC_AUC", "AP"]
STYLES = ["-", "--", "-.", ":", (0, (6, 2, 1, 2)), (0, (1, 1))]


def name(model):
    return NAMES[model.split("/")[-1]]


def save(fig, out, stem):
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def read(src, stem):
    return pd.concat([pd.read_csv(src / f"{stem}-{i}.csv.gz") for i in range(10)], ignore_index=True)


def table(rows):
    keys = ["model", "operating_point", "group_type", "group"]
    agg = rows.groupby(keys, dropna=False)[METRICS].agg(["count", "mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()


def summarize(src, out):
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ledgers = [json.loads((src / f"ledger-{i}.json").read_text(encoding="utf-8")) for i in range(10)]
    seeds = [s for ledger in ledgers for key in ("train", "validation", "test") for s in ledger[key]]
    assert len(set(seeds)) == len(seeds)
    assert len({v["model_seed"] for v in ledgers}) == 10
    assert all(len(v["test"]) == 40 for v in ledgers)
    rows, curves, loss = (read(src, k) for k in ("metrics", "curves", "loss"))
    assert len(loss) == 8000
    grouped = table(rows[~rows.group_type.eq("simulation_seed")])
    all_rows = grouped[grouped.group_type.eq("all")]
    main = all_rows[all_rows.operating_point.isin(MAIN)].copy()
    assert len(main) == 11 and main.recall_count.eq(10).all()
    main["recall_pct_mean_pm_sd"] = main.apply(
        lambda q: f"{100*q.recall_mean:.2f} ± {100*q.recall_std:.2f}", axis=1)
    main["FPR_pct_mean_pm_sd"] = main.apply(
        lambda q: f"{100*q.FPR_mean:.4f} ± {100*q.FPR_std:.4f}", axis=1)
    learning = loss.groupby(["model", "step"]).loss.agg(["mean", "std", "count"]).reset_index()
    assert learning["count"].eq(10).all()
    for stem, data in (("main_table", main), ("all_thresholds", all_rows), ("group_means_sd", grouped),
                       ("per_repeat_and_scenario", rows), ("roc_pr_points", curves),
                       ("training_loss_by_seed", loss), ("training_loss_mean_sd", learning)):
        data.to_csv(out / f"{stem}.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "SIP_TICD_10_Seeds_Tables.xlsx", engine="xlsxwriter") as writer:
        for sheet, frame in (("Mean SD n10", main), ("All thresholds", all_rows),
                             ("Individual repeats", rows[rows.group_type.eq("all")]),
                             ("Attack types", grouped[grouped.group_type.eq("attack_type")]),
                             ("Sophistication", grouped[grouped.group_type.eq("attacker_sophistication")]),
                             ("Scenarios", grouped[grouped.group_type.eq("scenario")]),
                             ("All 400 simulations", rows[rows.group_type.eq("simulation_seed")]),
                             ("Training means", learning), ("Training per seed", loss)):
            frame.to_excel(writer, sheet_name=sheet, index=False)
            w = writer.sheets[sheet]
            w.freeze_panes(1, 1)
            w.autofilter(0, 0, len(frame), len(frame.columns)-1)
            for j, col in enumerate(frame.columns):
                w.set_column(j, j, min(40, max(14, len(col) + 3)))
    md = ["| Набор признаков | Модель | Полнота, % | FPR, % |",
          "|---|---|---:|---:|"]
    for r in main.itertuples():
        track = "TICD" if r.model == "TICD" else ("Общая телеметрия" if r.model.startswith("shared/") else "Признаки статьи")
        md.append(f"| {track} | {name(r.model)} | {r.recall_pct_mean_pm_sd.replace('.', ',')} | {r.FPR_pct_mean_pm_sd.replace('.', ',')} |")
        print("TEN_SEED_RESULT", r.model, r.recall_pct_mean_pm_sd, r.FPR_pct_mean_pm_sd, flush=True)
    (out / "main_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for track in ("article", "shared"):
        chosen = main[main.model.eq("TICD") | main.model.str.startswith(track + "/")]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, key, title in zip(axes, ("recall", "FPR"), ("Полнота, %", "Ложные тревоги, %")):
            ax.barh([name(m) for m in chosen.model], 100*chosen[key+"_mean"],
                    xerr=100*chosen[key+"_std"], color="#777777", capsize=3)
            ax.set_xlabel(title)
        axes[0].invert_yaxis()
        save(fig, figures, f"performance_{track}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        grid = np.linspace(0, 1, 501)
        for ax, kind in zip(axes, ("ROC", "PR")):
            for j, model in enumerate(chosen.model):
                selected = curves[(curves.model == model) & (curves.curve == kind)]
                points = []
                for _, g in selected.groupby("repeat"):
                    g = g.groupby("x").y.max()
                    points.append(np.interp(grid, g.index, g))
                ax.plot(grid, np.mean(points, axis=0), color="black", ls=STYLES[j], label=name(model))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Доля ложных тревог" if kind == "ROC" else "Полнота")
            ax.set_ylabel("Полнота" if kind == "ROC" else "Точность")
            ax.legend(fontsize=7, loc="lower right")
        save(fig, figures, f"roc_pr_{track}")
        for group in ("attack_type", "attacker_sophistication", "scenario"):
            selected = grouped[grouped.group_type.eq(group) & grouped.model.isin(chosen.model)
                               & grouped.operating_point.isin(MAIN)]
            g = selected.pivot(index="group", columns="model", values="recall_mean").dropna(how="all")
            g = g[chosen.model]
            fig, ax = plt.subplots(figsize=(9, max(3.2, .33*len(g) + 1.4)))
            img = ax.imshow(100*g, cmap="Greys", vmin=0, vmax=100, aspect="auto")
            ax.set_yticks(range(len(g)), g.index, fontsize=8)
            ax.set_xticks(range(len(g.columns)), [name(v) for v in g.columns], rotation=25, ha="right")
            for i, row in enumerate(g.to_numpy()):
                for j, value in enumerate(row):
                    if np.isfinite(value):
                        ax.text(j, i, f"{100*value:.1f}", ha="center", va="center",
                                color="white" if value > .65 else "black", fontsize=7)
            fig.colorbar(img, ax=ax, label="Средняя полнота, %")
            save(fig, figures, f"{group}_{track}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, model in zip(axes, ("autoencoder", "gradient_boosting")):
        for j, track in enumerate(("article", "shared")):
            d = learning[learning.model.eq(f"{track}/{model}")]
            ax.plot(d.step, d["mean"], color="black", ls=STYLES[j],
                    label="Признаки статьи" if j == 0 else "Общая телеметрия")
            ax.fill_between(d.step, d["mean"]-d["std"], d["mean"]+d["std"], color="black", alpha=.10)
        ax.set_title(name(model))
        ax.set_xlabel("Эпоха" if model == "autoencoder" else "Число деревьев")
        ax.set_ylabel("MSE" if model == "autoencoder" else "Log loss")
        ax.legend(fontsize=8)
    save(fig, figures, "training_curves")
    (out / "README.md").write_text(
        "## Десять независимых повторов\n\n"
        "В каждом повторе обучение, валидация и тест используют разные seed симуляции. "
        "Модели оцениваются попарно на одной тестовой панели из 40 сценариев. "
        "Указаны среднее по десяти значениям полноты и выборочное СКО (n − 1). "
        "Внутри каждого повтора общая полнота рассчитана как TP/(TP+FN) по всем тестовым интервалам. "
        "Пороги пяти моделей выбраны на валидации с лимитом 1 % FPR в каждом её сценарии. "
        "TICD заново формирует нормальные профили и пороги физических каналов. Правила отчёта CU и аудита фиксированы. "
        "ROC/PR TICD основаны на трёх уровнях решения. Кривые ROC/PR усреднены на общей сетке. "
        "Кривые обучения показывают тренировочную ошибку с полосой ± СКО. "
        "Excel содержит полный набор метрик и разрезов. Единица метрик в числовых столбцах — доля.\n", encoding="utf-8")
    audit = out / "seeds"
    audit.mkdir(exist_ok=True)
    for f in src.glob("ledger-*.json"):
        shutil.copy2(f, audit / f.name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    summarize(a.input, a.out)
