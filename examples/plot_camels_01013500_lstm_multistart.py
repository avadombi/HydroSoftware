"""Create reproducible LSTM training and partition-performance figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "camels_us_01013500" / "lstm_multistart"


def main() -> None:
    summary = json.loads(
        (RESULT_DIR / "multistart-summary.json").read_text(encoding="utf-8")
    )
    reference_seed = int(summary["reference_run"]["seed"])
    histories = {}
    rows = []
    for run in summary["runs"]:
        seed = int(run["seed"])
        archive = json.loads(
            (RESULT_DIR / run["archive"]).read_text(encoding="utf-8")
        )
        history = archive["result"]["training"]["history"]
        histories[seed] = history
        best = -np.inf
        for item in history:
            best = max(best, float(item["validation_objective"]))
            rows.append(
                {
                    "seed": seed,
                    "epoch": int(item["epoch"]),
                    "training_nse": float(item["training_objective"]),
                    "validation_nse": float(item["validation_objective"]),
                    "best_validation_nse": best,
                }
            )

    csv_path = RESULT_DIR / "training-history.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(histories)))
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=True)
    for color, (seed, history) in zip(colors, sorted(histories.items())):
        epochs = [int(item["epoch"]) for item in history]
        validation = [float(item["validation_objective"]) for item in history]
        axes[0].plot(
            epochs,
            validation,
            color=color,
            linewidth=2.2 if seed == reference_seed else 1.25,
            alpha=1.0 if seed == reference_seed else 0.82,
            label=f"{seed}" + (" · reference" if seed == reference_seed else ""),
        )
    axes[0].set_title("A  Validation NSE during training", loc="left", weight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NSE")
    axes[0].axhline(0.0, color="#87939b", linewidth=0.8)
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    partitions = ("training", "validation", "test")
    x = np.arange(len(summary["runs"]), dtype=float)
    width = 0.24
    partition_colors = ("#2d7f9d", "#7c5bb3", "#d07a38")
    for offset, partition, color in zip((-width, 0.0, width), partitions, partition_colors):
        values = [float(run[f"{partition}_nse"]) for run in summary["runs"]]
        axes[1].bar(
            x + offset,
            values,
            width,
            color=color,
            label=partition.capitalize(),
        )
    axes[1].set_title("B  Restored-checkpoint performance", loc="left", weight="bold")
    axes[1].set_xlabel("Initialization seed")
    axes[1].set_ylabel("NSE")
    axes[1].set_xticks(x, [str(run["seed"])[-2:] for run in summary["runs"]])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].text(
        0.0,
        -0.18,
        "Tick labels show the final two digits of the frozen seed.",
        transform=axes[1].transAxes,
        fontsize=8,
        color="#53636c",
    )

    for suffix in ("png", "svg"):
        figure.savefig(
            RESULT_DIR / f"lstm-multistart-diagnostics.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(f"figure={RESULT_DIR / 'lstm-multistart-diagnostics.png'}")
    print(f"history={csv_path}")


if __name__ == "__main__":
    main()
