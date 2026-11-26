"""Learning curves for every SIP model on training and independent validation scenarios."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import evaluate_ticd_ten_training_seeds as ten

bench = ten.bench
# Include routine operation, compromised states, weak deficits, observer
# failures and the matched-telemetry latent scenario.
CURVE_KINDS = ("clean_loss1", "clean_loss25", "shift_clean_loss", "probe_outage_clean",
               "default", "role_proxy_loss25", "weak_0025", "variable_deficit",
               "policy_route", "observer_outage_policy", "latent_identical")
FRACTIONS = (1, 2, 3)


def run(repeat, prior, out):
    template, cal = bench.load_inputs(prior)
    data = out / "data"
    data.mkdir()
    training = {}
    def capture(kind, raw, commission):
        if kind in CURVE_KINDS:
            training.setdefault(kind, (raw, commission))
    ref, commissioned, clean_cuts = ten.prepare(
        repeat, template.reference, cal, data, capture_training=capture)
    assert set(training) == set(CURVE_KINDS)
    validation = {}
    for i, kind in enumerate(CURVE_KINDS):
        raw, _, _, c = bench.group([ten.seed_for(repeat, 4, i)], cal, kind, template.reference.core.causal)
        validation[kind] = raw, c
    a = {int(raw.simulation_seed.iloc[0]) for raw, _ in training.values()}
    b = {int(raw.simulation_seed.iloc[0]) for raw, _ in validation.values()}
    c = {int(raw.simulation_seed.iloc[0]) for raw, _ in clean_cuts.values()}
    assert not (a & b or a & c or b & c)

    sets = {}
    for split, worlds in (("training", training), ("validation", validation)):
        raw_pairs = [(raw[bench.COLUMNS], c[list(bench.RAW)]) for raw, c in worlds.values()]
        y = np.concatenate([raw.is_anomaly.to_numpy(dtype=bool) for raw, _ in worlds.values()])
        assert y.any() and not y.all()
        input_parts = [bench.measurements(raw) for raw, _ in worlds.values()]
        features = {t: pd.concat([x[0][t] for x in input_parts], ignore_index=True) for t in bench.TRACKS}
        visible = {t: np.concatenate([x[1][t] for x in input_parts]) for t in bench.TRACKS}
        sets[split] = raw_pairs, y, features, visible
    rows = []
    for part in FRACTIONS:
        # Preserve complete 20 s bins and node timelines at every fraction.
        ref_slice = ref.loc[ref.interval.lt(120 * part)].reset_index(drop=True)
        comm_slice = commissioned.loc[commissioned.interval.lt(120 * part)].reset_index(drop=True)
        assert len(ref_slice) * 3 == len(ref) * part
        ticd, _ = ten.refit_ticd(template, ref_slice, comm_slice, clean_cuts)
        for split, (worlds, y, _, _) in sets.items():
            pred = np.concatenate([ticd.predict_levels(raw, c) == 2 for raw, c in worlds])
            rows.append(dict(repeat=repeat + 1, model="TICD", split=split, fraction=part/3,
                             train_rows=len(ref_slice), **bench.metrics(pred, y)))
        size = bench.SAMPLE_SIZE * part // 3
        for track in bench.TRACKS:
            for family in bench.FAMILIES:
                imp, scale, model, manifest = bench.fit(
                    family, track, data, out, fit_seed=ten.seed_for(repeat, 3, 0),
                    shards=(0,), sample_size=size)
                assert not (set(manifest["training_seeds"]) | set(manifest["validation_seeds"])) & b
                for split, (_, y, features, visible) in sets.items():
                    matrix = scale.transform(imp.transform(features[track][manifest["features"]]))
                    scores = bench.score(family, model, matrix)
                    pred = visible[track] & (scores > manifest["thresholds"]["val_FPR_1pct_each_run"])
                    rows.append(dict(repeat=repeat + 1, model=f"{track}/{family}", split=split,
                                     fraction=part/3, train_rows=size, **bench.metrics(pred, y)))
        print("LEARNING_FRACTION_COMPLETE", repeat + 1, part, flush=True)
    assert len(rows) == 66
    pd.DataFrame(rows).to_csv(out / f"learning-{repeat}.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeat", type=int, choices=range(10), required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    run(a.repeat, a.prior, a.out)
