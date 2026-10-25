"""Create DeepDiscover convergence and comparison diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 22, 200, 123, 7)
MODEL_LABELS = ("EXP-HYDRO", "LSTM", "DeepDiscover", "PaML")
COLORS = ("#64748B", "#2563EB", "#C2410C", "#0F766E")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(archive, partition: str, symbol: str) -> float:
    for metric in archive["result"]["partition_metrics"][partition]:
        if metric["symbol"] == symbol:
            return float(metric["value"])
    raise KeyError(symbol)


def _write_convergence_csv(result_dir: Path, runs) -> None:
    path = result_dir / "deepdiscover-convergence.csv"
    histories = {
        seed: runs[seed]["result"]["training"]["history"] for seed in SEEDS
    }
    maximum_epochs = max(len(history) for history in histories.values())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "epoch",
                "validation_nse",
                "best_validation_nse_to_date",
                "training_nse",
            ]
        )
        for seed in SEEDS:
            best = -np.inf
            for item in histories[seed]:
                validation = float(item["validation_objective"])
                best = max(best, validation)
                writer.writerow(
                    [
                        seed,
                        int(item["epoch"]),
                        f"{validation:.15g}",
                        f"{best:.15g}",
                        f"{float(item['training_objective']):.15g}",
                    ]
                )
        if maximum_epochs < 1:
            raise ValueError("No DeepDiscover learning history was archived.")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "results/camels_us_01013500/deepdiscover_multistart"
    summary = _read(result_dir / "multistart-summary.json")
    runs = {seed: _read(result_dir / f"run-seed-{seed}.json") for seed in SEEDS}
    _write_convergence_csv(result_dir, runs)

    reference = runs[int(summary["reference_run"]["seed"])]
    comparison = summary["reference_test_comparison"]
    nse = [comparison[name]["NSE"] for name in MODEL_LABELS]
    kge = [comparison[name]["KGE"] for name in MODEL_LABELS]
    pbias = [comparison[name]["PBIAS"] for name in MODEL_LABELS]

    timestamps = np.asarray(reference["result"]["timestamps"], dtype="datetime64[D]")
    observations = np.asarray(reference["result"]["observations"], dtype=float)
    simulations = np.asarray(
        reference["result"]["best_simulation"]["outputs"], dtype=float
    )[:, 0]
    test_start = (
        int(reference["config"]["task"]["period"]["validation_end_index"])
        - int(reference["config"]["task"]["period"]["start_index"])
    )
    water_year_end = min(test_start + 365, len(timestamps))
    water_year = slice(test_start, water_year_end)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.4), constrained_layout=True)

    ax = axes[0, 0]
    seed_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(SEEDS)))
    for color, seed in zip(seed_colors, SEEDS):
        history = runs[seed]["result"]["training"]["history"]
        epochs = np.asarray([item["epoch"] for item in history], dtype=int)
        validation = np.maximum.accumulate(
            np.asarray(
                [item["validation_objective"] for item in history], dtype=float
            )
        )
        ax.plot(
            epochs,
            validation,
            label=f"seed {seed}",
            color=color,
            linewidth=1.55,
        )
    ax.set_title("A  Validation convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best validation NSE to date")
    ax.set_ylim(0.70, 0.925)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")

    ax = axes[0, 1]
    positions = np.arange(len(MODEL_LABELS))
    width = 0.36
    nse_bars = ax.bar(
        positions - width / 2,
        nse,
        width,
        color=COLORS,
        alpha=0.95,
        label="NSE",
    )
    kge_bars = ax.bar(
        positions + width / 2,
        kge,
        width,
        color=COLORS,
        alpha=0.42,
        hatch="//",
        label="KGE",
    )
    ax.set_xticks(positions, MODEL_LABELS, rotation=14, ha="right")
    ax.set_ylim(0.70, 0.94)
    ax.set_ylabel("Dimensionless efficiency")
    ax.set_title("B  Independent test efficiencies")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.bar_label(nse_bars, labels=[f"{value:.3f}" for value in nse], fontsize=7)
    ax.bar_label(kge_bars, labels=[f"{value:.3f}" for value in kge], fontsize=7)

    ax = axes[1, 0]
    bars = ax.bar(positions, pbias, color=COLORS, width=0.68)
    ax.axhline(0.0, color="#0F172A", linewidth=0.9)
    ax.set_xticks(positions, MODEL_LABELS, rotation=14, ha="right")
    ax.set_ylabel("PBIAS (%)")
    ax.set_title("C  Independent test volumetric bias")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.bar_label(
        bars,
        labels=[f"{value:+.2f}%" for value in pbias],
        padding=3,
        fontsize=8,
    )

    ax = axes[1, 1]
    dates = timestamps[water_year].astype("datetime64[ms]").astype(object)
    ax.plot(
        dates,
        observations[water_year],
        color="#111827",
        linewidth=1.0,
        label="Observed",
    )
    ax.plot(
        dates,
        simulations[water_year],
        color="#C2410C",
        linewidth=1.15,
        label="DeepDiscover",
    )
    ax.set_title("D  First independent test water year")
    ax.set_ylabel("Streamflow (mm/day)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle(
        "CAMELS-US 01013500: single-DU DeepDiscover reference",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "Reference seed 22 was selected only on validation. Panel D is the first complete test water year, fixed chronologically.",
        ha="center",
        va="top",
        fontsize=8,
        color="#475569",
    )
    for suffix in ("png", "svg"):
        fig.savefig(
            result_dir / f"deepdiscover-multistart-diagnostics.{suffix}",
            dpi=220,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
