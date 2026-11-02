import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PATTERN = re.compile(
    r"(?P<model>pacformer|pangu)_(?P<dataset>.+)_H(?P<horizon>\d+)_S(?P<seed>\d+)"
)


def load_runs(results_dir):
    runs = []
    for directory in sorted(results_dir.iterdir()):
        match = PATTERN.fullmatch(directory.name)
        history_path = directory / "history.json"
        metric_path = directory / "test_metrics.json"
        if match is None or not history_path.exists() or not metric_path.exists():
            continue
        runs.append(
            {
                **match.groupdict(),
                "horizon": int(match["horizon"]),
                "history": json.loads(history_path.read_text(encoding="utf-8")),
                "metrics": json.loads(metric_path.read_text(encoding="utf-8")),
            }
        )
    return runs


def main():
    results_dir = Path("results")
    runs = load_runs(results_dir)
    colors = {"pacformer": "#167D8D", "pangu": "#D9822B"}
    labels = {"pacformer": "PACFormer", "pangu": "Pangu"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    settings = [("RN_PV", 48), ("RN_PV", 96), ("US_UPV", 48), ("US_UPV", 96)]
    x = np.arange(len(settings))
    width = 0.36
    for offset, model in ((-width / 2, "pacformer"), (width / 2, "pangu")):
        values = [
            next(
                run["metrics"]["nMAE_percent"]
                for run in runs
                if run["dataset"] == dataset
                and run["horizon"] == horizon
                and run["model"] == model
            )
            for dataset, horizon in settings
        ]
        axes[0].bar(x + offset, values, width, label=labels[model], color=colors[model])
    axes[0].set_xticks(x, [f"{d.replace('_', '-')}\nH={h}" for d, h in settings])
    axes[0].set_ylabel("Capacity-normalized nMAE (%)")
    axes[0].set_title("Best-checkpoint test performance")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    for run in runs:
        epochs = [record["epoch"] for record in run["history"]]
        values = [record["val_nMAE_percent"] for record in run["history"]]
        linestyle = "-" if run["horizon"] == 48 else "--"
        axes[1].plot(
            epochs,
            values,
            color=colors[run["model"]],
            linestyle=linestyle,
            alpha=0.85,
            label=f"{labels[run['model']]} {run['dataset'].replace('_', '-')} H={run['horizon']}",
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation nMAE (%)")
    axes[1].set_title("Early-stopped validation trajectories")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)

    fig.savefig(results_dir / "comparison_summary.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
