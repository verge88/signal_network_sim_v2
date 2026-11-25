"""Complete ablation tables from a staged full Diameter experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diameter_training_v1 import (
    METRIC_TABLES, DataPipeline, TrainingConfig,
    collect_metric_tables, evaluate_ablation_variants,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--stack", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    config = TrainingConfig(
        data_path=args.train, validation_path=args.validation, stack_path=args.stack,
        test_path=args.test[0], additional_test_paths=tuple(args.test[1:]),
    )
    pipeline = DataPipeline(config)
    train, validation, _, test = pipeline.load_and_prepare()
    x_train = pipeline.get_features_labels(train, fit=True)
    x_validation = pipeline.get_features_labels(validation)
    x_test, y_test = pipeline.get_features_labels(test)
    attacks = test[pipeline.ATTACK_COL].to_numpy()
    variants = evaluate_ablation_variants(
        config, pipeline, x_train, x_validation, (x_test, y_test), attacks)
    artifacts_path = args.results / "plot_artifacts.json"
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    for result, prior in zip(variants, artifacts["ablation_results"][1:4]):
        recall = result["per_attack"]["node_compromise"][2]
        if not (np.isclose(result["f1"], prior["f1"]) and
                np.isclose(recall, prior["compromise_rate"])):
            raise SystemExit(f"Ablation failed to reproduce {result['name']}")
    names = {r["name"] for r in variants}
    for filename, extra in zip(METRIC_TABLES, collect_metric_tables(
            variants, test, y_test, attacks)):
        path = args.results / filename
        existing = pd.read_csv(path)
        pd.concat([existing[~existing["name"].isin(names)], extra],
                  ignore_index=True).to_csv(path, index=False)
    print("Recovered and validated all three ablation variants.")


if __name__ == "__main__":
    main()
