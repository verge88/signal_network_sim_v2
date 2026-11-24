"""High-coverage full SS7 model-zoo experiment.

This runner uses the canonical calibrated simulation but explicitly increases the
number of attack and benign-control episodes so incident metrics are not based on
1-3 test incidents.  Every model in ss7_training_v7.model_zoo is retrained from
scratch.  Per-window test scores are exported for ROC/PR/distribution plots.

Default scientific protocol:
- 4 simulated days;
- 24 compromise episodes, 48 benign operational events, 8 external perturbations;
- blocked temporal train/validation/test split with purge;
- threshold selected only from validation negatives using the canonical alert budget;
- all 12 models, including PyTorch AE/VAE/LSTM/attention-AE;
- no quick mode.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ss7.fix import (
    FEATURE_SETS,
    MASK_FAMILIES,
    NORMAL_EVENTS,
    RngBook,
    SimConfig,
    Sophistication,
    blocked_split,
    build_dataset,
)
from ss7.ss7_training_v7 import (
    TrainingConfig,
    assert_no_leakage,
    feature_set_ablation,
    gate_null_test,
    model_zoo,
    prepare_matrices,
    resolve_feature_columns,
    run_zoo,
    score_row,
    temporal_ablation,
)


def annotate_analysis_truth(df: pd.DataFrame, episodes: Sequence) -> pd.DataFrame:
    out = df.copy()
    rate = {i: float(e.hidden_rate) for i, e in enumerate(episodes) if e.kind == "compromise"}
    family = {i: str(e.family) for i, e in enumerate(episodes) if e.kind == "compromise"}
    out["episode_hidden_rate"] = out["episode_id"].map(rate)
    out["episode_family_truth"] = out["episode_id"].map(family)

    normal_kind = np.full(len(out), "background", dtype=object)
    nid = out["nid"].to_numpy(int)
    t = out["t"].to_numpy(int)
    for e in episodes:
        if e.kind not in NORMAL_EVENTS or e.nid is None:
            continue
        mask = (
            (out["label"].to_numpy(int) == 0)
            & (nid == int(e.nid))
            & (t >= int(e.t0))
            & (t <= int(e.t1))
        )
        normal_kind[mask] = str(e.kind)
    out["normal_event_truth"] = normal_kind
    return out


def run_all_models_with_scores(M, cfg: TrainingConfig, seed: int):
    rows = []
    score_frames = []
    meta = M.te[[
        "nid", "t", "label", "episode_id", "mask_family", "soph",
        "episode_hidden_rate", "episode_family_truth", "normal_event_truth",
    ]].copy()

    for name, (family, fn) in model_zoo(cfg).items():
        started = time.time()
        try:
            s_va, s_te = fn(M, cfg, seed)
            s_va = np.asarray(s_va, dtype=float)
            s_te = np.asarray(s_te, dtype=float)
            row = score_row(name, family, M, s_va, s_te, cfg)
            row["fit_seconds"] = round(time.time() - started, 3)
            row["error"] = ""
            rows.append(row)

            d = meta.copy()
            d["model"] = name
            d["score"] = s_te
            d["threshold"] = float(row["threshold"])
            d["fire"] = (np.nan_to_num(s_te, nan=-np.inf) > float(row["threshold"])).astype(int)
            score_frames.append(d)
        except Exception as exc:
            rows.append({
                "model": name,
                "family": family,
                "fit_seconds": round(time.time() - started, 3),
                "error": repr(exc),
            })

    return pd.DataFrame(rows), (
        pd.concat(score_frames, ignore_index=True, sort=False)
        if score_frames else pd.DataFrame()
    )


def lomfo_high_coverage(df: pd.DataFrame, cfg: TrainingConfig) -> pd.DataFrame:
    cols = resolve_feature_columns(df, FEATURE_SETS[cfg.feature_set])
    rows = []
    for idx, family in enumerate(MASK_FAMILIES):
        tr, va, te = blocked_split(
            df, cfg.sim_config(), RngBook(cfg.seed + 100 + idx)["split"]
        )
        tr = tr[(tr["label"] == 0) | (tr["mask_family"] != family)]
        va = va[(va["label"] == 0) | (va["mask_family"] != family)]
        te = te[(te["label"] == 0) | (te["mask_family"] == family)]
        if tr["label"].sum() < 30 or te["label"].sum() < 20:
            rows.append({
                "holdout_family": family,
                "model": "status",
                "error": "insufficient_positive_windows",
                "train_positive_windows": int(tr["label"].sum()),
                "test_positive_windows": int(te["label"].sum()),
            })
            continue
        M = prepare_matrices(tr, va, te, cols)
        r = run_zoo(
            M, cfg, cfg.seed + idx,
            only=("rule_baseline", "isolation_forest", "random_forest", "hist_gb"),
        )
        r["holdout_family"] = family
        r["test_episodes"] = int(te.loc[te["label"] == 1, "episode_id"].nunique())
        rows.append(r)
    return pd.concat(
        [x if isinstance(x, pd.DataFrame) else pd.DataFrame([x]) for x in rows],
        ignore_index=True, sort=False
    ) if rows else pd.DataFrame()


def detectability_high_coverage(cfg: TrainingConfig, rates, n_compromise: int) -> pd.DataFrame:
    rows = []
    for idx, h in enumerate(rates):
        sc = SimConfig(
            **{**cfg.sim_config().__dict__, "seed": int(cfg.seed + 4000 + idx * 137)}
        )
        df, _, eps = build_dataset(
            sc,
            episodes_kw={
                "n_compromise": int(n_compromise),
                "n_normal": 30,
                "n_ext": 6,
                "force_hidden": float(h),
                "soph": Sophistication(cfg.sophistication),
            },
        )
        df = annotate_analysis_truth(df, eps)
        cols = resolve_feature_columns(df, FEATURE_SETS[cfg.feature_set])
        tr, va, te = blocked_split(df, sc, RngBook(sc.seed)["split"])
        if tr["label"].sum() < 20 or te["label"].sum() < 20:
            continue
        M = prepare_matrices(tr, va, te, cols)
        r = run_zoo(
            M, cfg, sc.seed,
            only=("rule_baseline", "isolation_forest", "random_forest", "hist_gb"),
        )
        r["hidden_rate"] = float(h)
        r["test_episodes"] = int(te.loc[te["label"] == 1, "episode_id"].nunique())
        rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def sophistication_high_coverage(cfg: TrainingConfig, n_compromise: int) -> pd.DataFrame:
    rows = []
    for idx, sophistication in enumerate(Sophistication):
        sc = SimConfig(
            **{**cfg.sim_config().__dict__, "seed": int(cfg.seed + 7000 + idx * 173)}
        )
        df, _, eps = build_dataset(
            sc,
            episodes_kw={
                "n_compromise": int(n_compromise),
                "n_normal": 36,
                "n_ext": 6,
                "soph": sophistication,
            },
        )
        df = annotate_analysis_truth(df, eps)
        cols = resolve_feature_columns(df, FEATURE_SETS[cfg.feature_set])
        tr, va, te = blocked_split(df, sc, RngBook(sc.seed)["split"])
        if tr["label"].sum() < 20 or te["label"].sum() < 20:
            continue
        M = prepare_matrices(tr, va, te, cols)
        r = run_zoo(
            M, cfg, sc.seed,
            only=("rule_baseline", "isolation_forest", "random_forest", "hist_gb"),
        )
        r["sophistication"] = sophistication.value
        r["test_episodes"] = int(te.loc[te["label"] == 1, "episode_id"].nunique())
        rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--days", type=float, default=4.0)
    ap.add_argument("--n-compromise", type=int, default=24)
    ap.add_argument("--n-normal", type=int, default=48)
    ap.add_argument("--n-ext", type=int, default=8)
    ap.add_argument("--feature-set", default="OWN+INV", choices=list(FEATURE_SETS))
    ap.add_argument("--alerts-per-day", type=float, default=20.0)
    ap.add_argument("--op-prevalence", type=float, default=1e-4)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--diagnostics", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = TrainingConfig(
        seed=int(args.seed),
        days=float(args.days),
        sophistication=Sophistication.ADAPTIVE.value,
        feature_set=args.feature_set,
        alerts_per_day=float(args.alerts_per_day),
        op_prevalence=float(args.op_prevalence),
        use_torch=True,
        quick=False,
        use_cache=False,
        out_dir=str(outdir),
    )

    sc = cfg.sim_config()
    df, sim, episodes = build_dataset(
        sc,
        episodes_kw={
            "n_compromise": int(args.n_compromise),
            "n_normal": int(args.n_normal),
            "n_ext": int(args.n_ext),
            "soph": Sophistication.ADAPTIVE,
        },
    )
    df = annotate_analysis_truth(df, episodes)
    cols = resolve_feature_columns(df, FEATURE_SETS[cfg.feature_set])
    assert_no_leakage(cols)

    null_report = gate_null_test(cfg, cols, False)
    (outdir / "null_test.json").write_text(
        json.dumps(
            {k: v for k, v in null_report.items() if k != "per_feature"},
            ensure_ascii=False, indent=2, default=float,
        ),
        encoding="utf-8",
    )
    if not null_report.get("passed", False):
        raise RuntimeError("Null-test gate failed; refusing to train trusted models")

    tr, va, te = blocked_split(df, sc, RngBook(sc.seed)["split"])
    M = prepare_matrices(tr, va, te, cols)

    models, test_scores = run_all_models_with_scores(M, cfg, int(args.seed))
    models["seed"] = int(args.seed)
    models["test_rows"] = int(len(te))
    models["test_positive_windows"] = int(te["label"].sum())
    models["test_episodes"] = int(te.loc[te["label"] == 1, "episode_id"].nunique())
    models.to_csv(outdir / "models.csv", index=False)
    if not test_scores.empty:
        test_scores["seed"] = int(args.seed)
        test_scores.to_csv(outdir / "test_scores.csv.gz", index=False, compression="gzip")

    manifest = {
        "seed": int(args.seed),
        "days": float(args.days),
        "n_compromise_requested": int(args.n_compromise),
        "n_normal_requested": int(args.n_normal),
        "n_ext_requested": int(args.n_ext),
        "rows": int(len(df)),
        "nodes": int(df["nid"].nunique()),
        "positive_windows": int(df["label"].sum()),
        "attack_episodes_total": int(df.loc[df["label"] == 1, "episode_id"].nunique()),
        "split": {
            "train_rows": int(len(tr)),
            "validation_rows": int(len(va)),
            "test_rows": int(len(te)),
            "train_attack_episodes": int(tr.loc[tr["label"] == 1, "episode_id"].nunique()),
            "validation_attack_episodes": int(va.loc[va["label"] == 1, "episode_id"].nunique()),
            "test_attack_episodes": int(te.loc[te["label"] == 1, "episode_id"].nunique()),
        },
        "feature_count": int(len(cols)),
        "models_expected": sorted(model_zoo(cfg).keys()),
        "models_completed": sorted(models.loc[models["error"].fillna("") == "", "model"].astype(str).tolist()),
        "model_errors": models.loc[models["error"].fillna("") != "", ["model", "error"]].to_dict(orient="records"),
        "null_test_passed": bool(null_report.get("passed", False)),
        "role_diagnostic_passed": bool(null_report.get("role_diagnostic_passed", False)),
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.diagnostics:
        fs = feature_set_ablation(df, cfg, int(args.seed))
        fs.to_csv(outdir / "ablation_feature_sets.csv", index=False)

        ta = temporal_ablation(df, cfg, int(args.seed))
        ta.to_csv(outdir / "ablation_temporal.csv", index=False)

        lo = lomfo_high_coverage(df, cfg)
        lo.to_csv(outdir / "lomfo.csv", index=False)

        curve_cfg = TrainingConfig(**{**cfg.__dict__, "days": min(float(args.days), 3.0)})
        cu = detectability_high_coverage(
            curve_cfg, rates=(0.0, 0.05, 0.15, 0.4, 1.0, 2.5),
            n_compromise=max(18, int(args.n_compromise)),
        )
        cu.to_csv(outdir / "detectability_curve.csv", index=False)

        sw = sophistication_high_coverage(
            curve_cfg, n_compromise=max(18, int(args.n_compromise))
        )
        sw.to_csv(outdir / "sophistication_sweep.csv", index=False)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
