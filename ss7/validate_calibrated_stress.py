"""Controlled stress validation for the calibrated SS7 hybrid detector.

Complements multiseed validation with guaranteed coverage of:
- every masking family;
- every registered hidden-rate level;
- every adversary sophistication level;
- leave-one-masking-family-out generalisation for every family.

The detector structure/hyperparameters are frozen.  A baseline realization is used
for training/threshold calibration; stress worlds use different traffic/adversary RNG
seeds on the same fixed topology so scenario effects are not confounded by topology.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ss7.fix import (
    INV,
    MASK_FAMILIES,
    RngBook,
    SimConfig,
    Sophistication,
    blocked_split,
    build_dataset,
)
from ss7.validate_calibrated_detectors import (
    CHANNEL_FPR,
    INV_CAL_RF,
    add_episode_parameters,
    detector_metrics,
    episode_recall_by_group,
    fit_common_detectors,
    fpr_threshold,
    prep,
    robust_scale,
    tx,
)


HIDDEN_RATES = (0.05, 0.15, 0.4, 1.0, 2.5)


def inv_cols(frame: pd.DataFrame) -> List[str]:
    return [
        c for c in frame.columns
        if any(c == b or c.startswith(b + "__") for b in INV)
    ]


def score_hybrid(frame: pd.DataFrame, fitted: dict) -> tuple[np.ndarray, dict]:
    own_cols, imp_o, sc_o, own, own_thr, own_scale = fitted["own"]
    inv_features, imp_i, sc_i, inv, inv_thr, inv_scale = fitted["inv"]
    own_s = -own.score_samples(tx(frame, own_cols, imp_o, sc_o))
    inv_s = inv.predict_proba(tx(frame, inv_features, imp_i, sc_i))[:, 1]
    fused = np.maximum(
        (own_s - own_thr) / own_scale,
        (inv_s - inv_thr) / inv_scale,
    )
    return fused, {
        "own_threshold": float(own_thr),
        "inv_threshold": float(inv_thr),
        "own_scale": float(own_scale),
        "inv_scale": float(inv_scale),
    }


def build_stress_world(
    topology,
    seed: int,
    days: float,
    *,
    families=MASK_FAMILIES,
    force_hidden=None,
    sophistication=Sophistication.ADAPTIVE,
    n_compromise: int = 10,
    n_normal: int = 12,
):
    cfg = SimConfig(seed=int(seed), days=float(days))
    frame, _, episodes = build_dataset(
        cfg,
        topo=topology,
        episodes_kw={
            "n_compromise": int(n_compromise),
            "n_normal": int(n_normal),
            "n_ext": 0,
            "families": tuple(families),
            "force_hidden": force_hidden,
            "soph": sophistication,
        },
    )
    return add_episode_parameters(frame, episodes), cfg


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--base-days", type=float, default=1.25)
    ap.add_argument("--stress-days", type=float, default=0.55)
    ap.add_argument("--output-dir", default="results/ss7_calibrated_detector_stress")
    args = ap.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_cfg = SimConfig(seed=args.base_seed, days=args.base_days)
    base, sim, base_eps = build_dataset(
        base_cfg,
        episodes_kw={
            "n_compromise": 25,
            "n_normal": 35,
            "n_ext": 6,
            "soph": Sophistication.ADAPTIVE,
        },
    )
    base = add_episode_parameters(base, base_eps)
    tr, va, te = blocked_split(base, base_cfg, RngBook(args.base_seed)["split"])
    _, _, _, fitted = fit_common_detectors(tr, va, te, args.base_seed)

    family_rows = []
    family_rate_rows = []
    family_worlds = {}

    # Guaranteed family coverage with independent traffic/adversary seeds.
    for idx, family in enumerate(MASK_FAMILIES):
        frame, cfg = build_stress_world(
            sim.topo,
            9100 + idx,
            args.stress_days,
            families=(family,),
            sophistication=Sophistication.ADAPTIVE,
            n_compromise=12,
            n_normal=12,
        )
        s, _ = score_hybrid(frame, fitted)
        m = detector_metrics(frame, s, 0.0)
        family_rows.append({
            "mask_family": family,
            "seed": int(cfg.seed),
            "rows": int(len(frame)),
            "episodes": int(frame.loc[frame["label"] == 1, "episode_id"].nunique()),
            **m,
        })
        for row in episode_recall_by_group(frame, s, 0.0, "episode_hidden_rate"):
            family_rate_rows.append({
                "mask_family": family,
                "seed": int(cfg.seed),
                **row,
            })
        family_worlds[family] = frame

    rate_rows = []
    for idx, rate in enumerate(HIDDEN_RATES):
        frame, cfg = build_stress_world(
            sim.topo,
            9200 + idx,
            args.stress_days,
            families=MASK_FAMILIES,
            force_hidden=float(rate),
            sophistication=Sophistication.ADAPTIVE,
            n_compromise=12,
            n_normal=12,
        )
        s, _ = score_hybrid(frame, fitted)
        rate_rows.append({
            "hidden_rate": float(rate),
            "seed": int(cfg.seed),
            "rows": int(len(frame)),
            "episodes": int(frame.loc[frame["label"] == 1, "episode_id"].nunique()),
            **detector_metrics(frame, s, 0.0),
        })

    sophistication_rows = []
    sophs = [
        Sophistication.NAIVE,
        Sophistication.STATISTICAL,
        Sophistication.ADAPTIVE,
    ]
    for idx, soph in enumerate(sophs):
        frame, cfg = build_stress_world(
            sim.topo,
            9300 + idx,
            args.stress_days,
            families=MASK_FAMILIES,
            force_hidden=0.4,
            sophistication=soph,
            n_compromise=12,
            n_normal=12,
        )
        s, _ = score_hybrid(frame, fitted)
        sophistication_rows.append({
            "sophistication": soph.value,
            "hidden_rate": 0.4,
            "seed": int(cfg.seed),
            "rows": int(len(frame)),
            "episodes": int(frame.loc[frame["label"] == 1, "episode_id"].nunique()),
            **detector_metrics(frame, s, 0.0),
        })

    # Guaranteed leave-one-family-out.  Training and threshold calibration use the
    # baseline world with all examples of the held-out family removed.  Test is the
    # dedicated independent-seed world containing only that family.
    lomfo_rows = []
    features = inv_cols(base)
    for idx, family in enumerate(MASK_FAMILIES):
        tr2 = tr[(tr["label"] == 0) | (tr["mask_family"] != family)].copy()
        va2 = va[(va["label"] == 0) | (va["mask_family"] != family)].copy()
        if tr2["label"].sum() < 20:
            lomfo_rows.append({
                "holdout_family": family,
                "status": "insufficient_training_positives",
                "training_positive_windows": int(tr2["label"].sum()),
            })
            continue
        ytr = tr2["label"].to_numpy(int)
        yva = va2["label"].to_numpy(int)
        imp, sc, xtr = prep(tr2, features)
        model = RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=args.base_seed + idx,
            n_jobs=-1,
            **INV_CAL_RF,
        ).fit(xtr, ytr)
        sva = model.predict_proba(tx(va2, features, imp, sc))[:, 1]
        threshold = fpr_threshold(sva[yva == 0], CHANNEL_FPR)
        test = family_worlds[family]
        ste = model.predict_proba(tx(test, features, imp, sc))[:, 1]
        lomfo_rows.append({
            "holdout_family": family,
            "status": "completed",
            "training_positive_windows": int(tr2["label"].sum()),
            "validation_negative_windows": int((yva == 0).sum()),
            **detector_metrics(test, ste, threshold),
        })

    # External-traffic perturbation audit: not an attack-recall metric.  These rows
    # remain label=0 because the detector's target is compromised internal reporting.
    ext_cfg = SimConfig(seed=9400, days=args.stress_days)
    ext, _, ext_eps = build_dataset(
        ext_cfg,
        topo=sim.topo,
        episodes_kw={"n_compromise": 0, "n_normal": 0, "n_ext": 20},
    )
    ext = add_episode_parameters(ext, ext_eps)
    ext_s, _ = score_hybrid(ext, fitted)
    external_audit = {
        "seed": 9400,
        "rows": int(len(ext)),
        "external_episodes_generated": int(sum(e.kind == "ext_attack" for e in ext_eps)),
        "hybrid_alert_rate_all_internal_windows": float(np.mean(ext_s > 0.0)),
        "alerts": int(np.sum(ext_s > 0.0)),
        "interpretation": (
            "Response of the internal compromise detector during external traffic "
            "perturbations; not labelled attack recall."
        ),
    }

    pd.DataFrame(family_rows).to_csv(outdir / "stress_by_mask_family.csv", index=False)
    pd.DataFrame(family_rate_rows).to_csv(outdir / "stress_family_by_hidden_rate.csv", index=False)
    pd.DataFrame(rate_rows).to_csv(outdir / "stress_by_hidden_rate.csv", index=False)
    pd.DataFrame(sophistication_rows).to_csv(outdir / "stress_by_sophistication.csv", index=False)
    pd.DataFrame(lomfo_rows).to_csv(outdir / "guaranteed_lomfo.csv", index=False)

    report = {
        "status": "completed",
        "protocol": {
            "base_seed": int(args.base_seed),
            "base_days": float(args.base_days),
            "stress_days": float(args.stress_days),
            "fixed_topology": True,
            "independent_stress_rng_seeds": True,
            "hybrid_thresholds": "frozen from base validation",
            "test_scenario_used_for_tuning": False,
        },
        "family_stress": family_rows,
        "family_rate_detail": family_rate_rows,
        "hidden_rate_stress": rate_rows,
        "sophistication_stress": sophistication_rows,
        "guaranteed_leave_one_mask_family_out": lomfo_rows,
        "external_traffic_perturbation_audit": external_audit,
        "limitations": [
            "Stress worlds share the base topology by design but use independent traffic/adversary RNG seeds.",
            "Attack labels remain simulation-generated and do not establish real-world attack recall.",
            "The scenario sweep tests the calibrated detector operating procedure, not the external CDR calibration.",
        ],
    }
    (outdir / "ss7_calibrated_detector_stress.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
