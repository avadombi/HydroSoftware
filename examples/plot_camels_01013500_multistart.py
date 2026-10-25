"""Create convergence artifacts from the EXP-HYDRO multistart summary."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/camels_us_01013500/exp_hydro_multistart/"
            "multistart-summary.json"
        ),
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    summary_path = (
        args.summary
        if args.summary.is_absolute()
        else project_root / args.summary
    ).resolve()
    output_dir = summary_path.parent
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    runs = sorted(summary["runs"], key=lambda item: item["seed"])
    generations = list(range(len(runs[0]["best_by_generation"])))
    csv_path = output_dir / "optimizer-convergence.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", *[f"seed_{run['seed']}" for run in runs]])
        for generation in generations:
            writer.writerow(
                [
                    generation,
                    *[
                        f"{run['best_by_generation'][generation]:.15g}"
                        for run in runs
                    ],
                ]
            )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hydrosoftware-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )
    colors = ("#1261A0", "#E07A1F", "#2A8C70")
    figure, axis = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    for run, color in zip(runs, colors):
        axis.plot(
            generations,
            run["best_by_generation"],
            label=f"Seed {run['seed']}",
            color=color,
            linewidth=2.0,
        )
        axis.scatter(
            [generations[-1]],
            [run["best_by_generation"][-1]],
            color=color,
            s=24,
            zorder=3,
        )
    axis.set_title(
        "EXP-HYDRO optimizer convergence — CAMELS-US 01013500",
        loc="left",
        weight="bold",
        pad=12,
    )
    axis.set_xlabel("Differential-evolution generation")
    axis.set_ylabel("Best-so-far calibration NSE")
    axis.set_xlim(0, generations[-1])
    axis.set_ylim(0.60, 0.80)
    axis.legend(loc="lower right")
    axis.text(
        0.0,
        -0.16,
        "60 population members; 100 generations; validation and test targets excluded from selection.",
        transform=axis.transAxes,
        color="#4B5563",
        fontsize=9,
    )
    figure.savefig(
        output_dir / "optimizer-convergence.png",
        dpi=220,
        facecolor="white",
    )
    figure.savefig(
        output_dir / "optimizer-convergence.svg",
        facecolor="white",
    )
    plt.close(figure)
    print(csv_path)


if __name__ == "__main__":
    main()
