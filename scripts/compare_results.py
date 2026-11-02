import json
import re
from pathlib import Path

import pandas as pd


def main():
    results_dir = Path("results")
    records = []
    for metric_path in sorted(results_dir.glob("*_H*_S42/test_metrics.json")):
        run_name = metric_path.parent.name
        match = re.fullmatch(
            r"(?P<model>pacformer|pangu)_(?P<dataset>.+)_H(?P<horizon>\d+)_S(?P<seed>\d+)",
            run_name,
        )
        if match is None:
            continue
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        records.append(
            {
                "model": match["model"],
                "dataset": match["dataset"],
                "horizon": int(match["horizon"]),
                "seed": int(match["seed"]),
                **metrics,
            }
        )

    frame = pd.DataFrame(records).sort_values(["dataset", "horizon", "model"])
    if frame.empty:
        raise RuntimeError("No completed comparison results were found.")

    pivot = frame.pivot(
        index=["dataset", "horizon"], columns="model", values="nMAE_percent"
    )
    if {"pacformer", "pangu"}.issubset(pivot.columns):
        pivot["nMAE_improvement_percent"] = (
            (pivot["pangu"] - pivot["pacformer"]) / pivot["pangu"] * 100.0
        )
    frame.to_csv(results_dir / "comparison_metrics.csv", index=False)
    pivot.reset_index().to_csv(results_dir / "comparison_nMAE.csv", index=False)
    print(frame.to_string(index=False))
    print("\nnMAE comparison")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
