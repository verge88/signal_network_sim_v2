"""Aggregate the ten train and validation curves and export article figures."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

MODELS = {"TICD":"TICD", "isolation_forest":"Isolation Forest", "one_class_svm":"One-Class SVM",
          "autoencoder":"Автоэнкодировщик", "random_forest":"Random Forest", "gradient_boosting":"Gradient Boosting"}
SPLITS = {"training": ("Обучающие сценарии", "-", "o"),
          "validation": ("Валидация", "--", "s")}
METRICS = {"recall": "Полнота", "F1": "F1", "FPR": "Доля ложных тревог"}


def publish(src, out):
    out.mkdir(exist_ok=True, parents=True)
    rows = pd.concat([pd.read_csv(src / f"learning-{i}.csv") for i in range(10)], ignore_index=True)
    assert len(rows) == 660 and not rows.duplicated(["repeat","model","split","fraction"]).any()
    summary = rows.groupby(["model","split","fraction","train_rows"])[list(METRICS)].agg(["mean","std","count"])
    assert summary.xs("count", level=1, axis=1).eq(10).all().all()
    summary.columns = [f"{a}_{b}" for a,b in summary.columns]
    summary = summary.reset_index()
    rows.to_csv(out / "learning_per_repeat.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "learning_mean_sd.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "SIP_Learning_Curves.xlsx", engine="xlsxwriter") as writer:
        for title, frame in (("Среднее и СКО", summary), ("По повторам", rows)):
            frame.to_excel(writer, sheet_name=title, index=False)
            writer.sheets[title].freeze_panes(1, 1)
            writer.sheets[title].autofilter(0,0,len(frame),len(frame.columns)-1)
    for track in ("shared","article"):
        for metric, title in METRICS.items():
            fig, axes = plt.subplots(2,3,figsize=(10,6.4),sharex=True,sharey=True)
            ymax = 1 if metric != "FPR" else 0
            for (family,label), ax in zip(MODELS.items(), axes.flat):
                model = "TICD" if family == "TICD" else f"{track}/{family}"
                for split,(legend,style,marker) in SPLITS.items():
                    d = summary[(summary.model == model) & (summary.split == split)].sort_values("fraction")
                    x,y,sd = d.fraction.to_numpy(),d[f"{metric}_mean"].to_numpy(),d[f"{metric}_std"].to_numpy()
                    assert len(d)==3
                    ax.plot(x,y,label=legend,color="black",ls=style,marker=marker,ms=4)
                    high=np.minimum(1,y+sd)
                    ymax=max(ymax,float(high.max())*1.08 if metric=="FPR" else 1)
                    ax.fill_between(x,np.maximum(0,y-sd),high,color="black",alpha=.10)
                ax.set_title(label,fontsize=10)
                ax.set_xticks([1/3,2/3,1],["33","67","100"])
                ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=1 if metric=="FPR" else 0))
                ax.grid(axis="y",color=".88",lw=.6)
                ax.spines[["top","right"]].set_visible(False)
            axes[0,0].set_ylim(0,ymax)
            fig.supxlabel("Доля обучающей выборки, %",y=.12)
            fig.supylabel(title)
            fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.015),ncol=2,frameon=False)
            fig.tight_layout(rect=(.02,.15,1,1))
            for fmt in ("png","svg","pdf"):
                fig.savefig(out / f"learning_{track}_{metric}.{fmt}",dpi=300,bbox_inches="tight")
            plt.close(fig)
    (out/"README.md").write_text(
        "Среднее ± выборочное СКО по десяти переобучениям. Пороги настраиваются на третьем, "
        "отдельном наборе штатных реализаций. Обе кривые рассчитаны для одинакового перечня "
        "из 11 сценариев с атаками и штатными режимами. Обучающая кривая использует реализации "
        "из обучающего пула, а валидационная — отдельные реализации тех же сценариев. "
        "Для пяти моделей объёмы обучения 9 504, 19 008, 28 512 строк. "
        "Для TICD используются первые 120, 240 или 360 интервалов каждой штатной реализации "
        "с отдельной калибровочной записью соответствующей длины. "
        "Вариант TICD одинаков в обоих наборах рисунков. "
        "Тестовая панель из 40 сценариев в расчёт кривых не входит.\n",encoding="utf-8")
    print("LEARNING_REPORT",len(rows),len(summary),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    publish(a.input,a.out)
