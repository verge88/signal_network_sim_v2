"""Summarize the Diameter loopback experiment and publish its primary figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = {"healthy": "Healthy", "frozen_joint": "Frozen S6a and Gx",
         "frozen_s6a": "Frozen S6a only"}
MODELS = ("Cascade", "Cascade + report recurrence")


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    results = {key: json.loads((args.results / f"{key}.json").read_text())
               for key in MODES}
    captures = sorted(args.data.glob("session-*/session.json"))
    sessions = [json.loads(path.read_text()) for path in captures]
    if len(sessions) != 10 or any(len(r["sessions"]) != len(sessions) for r in results.values()):
        raise ValueError("Expected ten observed sessions for each evaluation")
    if len({r["model_sha256"] for r in results.values()}) != 1:
        raise ValueError("The cases must share the same frozen bundle")
    for sid, group in enumerate(sessions):
        expected = results["healthy"]["sessions"][sid]["sha256"]["capture"]
        if any(r["sessions"][sid]["id"] != group["id"]
               or r["sessions"][sid]["sha256"]["capture"] != expected for r in results.values()):
            raise ValueError("The three evaluations must share the captured sessions")
        if group["capture_verification"]["polls"] != 288:
            raise ValueError("Missing captured CU polls")
    headings = [
        "# Diameter cascade on a local TCP/CU testbed",
        "",
        f"Ten sessions, {sum(s['wire_request_answer_pairs'] for s in sessions):,} Diameter request/answer pairs, "
        f"{sum(s['received_reports'] for s in sessions):,} CU reports received. "
        "Both fitted models and thresholds are from historical repetition 1.",
        "",
        "The three evaluations share the same benign wire captures. The two frozen-profile cases are "
        "offline sensor test fixtures applied to the observed CU profiles. Each model sees the same "
        "labels and feature export. Windows represent 15 logical seconds and were collected on accelerated loopback. "
        "The peers exercise Diameter binary encoding and TCP exchange with minimal application payloads.",
        "",
        "| Case | Model | Detected windows | Recall | Benign alarms | FPR | Episodes | Median delay, windows |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, name in MODES.items():
        for model in MODELS:
            m = results[key]["summary"][model]
            delay = m["median_delay_detected_s"]
            delay = "—" if delay is None else f"{delay / 15:g}"
            headlines = [name, model, f"{m['hits']}/{m['compromise_windows']}",
                         pct(m["window_recall"]), f"{m['false_alarms']}/{m['benign_windows']}",
                         pct(m["benign_fpr"]), f"{m['episodes_detected']}/{m['episodes']}", delay]
            headings.append("| " + " | ".join(headlines) + " |")
    headings += ["", "The paired 95% session-bootstrap intervals are in the JSON results.",
                 "Intervals and delays describe this workload and these observation fixtures."]
    (args.results / "summary.md").write_text("\n".join(headings) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, metric, keys, label in (
        (axes[0], "window_recall", list(MODES)[1:], "Recall on fixture windows"),
        (axes[1], "benign_fpr", list(MODES), "False positive rate on benign windows"),
    ):
        positions = np.arange(len(keys))
        for i, model in enumerate(MODELS):
            values = [100 * results[key]["summary"][model][metric] for key in keys]
            bars = ax.bar(positions + (i - .5) * .33, values, width=.33,
                          label=model, color=["#a1a1a1", "#242424"][i])
            ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], fontsize=8)
        ax.set_xticks(positions, [MODES[key] for key in keys], rotation=12)
        ax.set_ylim(0, 112)
        ax.set_title(label)
        ax.set_ylabel("%")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    for form in ("png", "svg"):
        fig.savefig(args.results / f"cascade-testbed.{form}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
