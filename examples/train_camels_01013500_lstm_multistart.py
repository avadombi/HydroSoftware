"""Train and diagnose repeated causal LSTMs for CAMELS-US 01013500."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from hydrosoftware import __version__
from hydrosoftware.configuration import load_experiment_config
from hydrosoftware.core.evaluation import evaluate_metrics
from hydrosoftware.execution.experiments import ExperimentRunner
from hydrosoftware.execution.scientific_verification import verify_run_archive


INITIALIZATION_SEEDS = (20260811, 20260812, 20260813, 20260814, 20260815)
REFERENCE_METRIC = "NSE"
PARTITION_NAMES = ("training", "validation", "test")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_archive(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("record_type") != "hydrosoftware.run-archive":
        raise ValueError(f"{path.name} is not a HydroSoftware run archive.")
    if payload.get("state") != "completed":
        raise ValueError(f"{path.name} does not contain a completed run.")
    return payload


def _archive_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"run-seed-{seed}.json"


def _metric_value(
    archive: Mapping[str, Any], partition: str, symbol: str
) -> float | None:
    metrics = archive["result"]["partition_metrics"][partition]
    for metric in metrics:
        if metric["symbol"] == symbol:
            value = metric.get("value")
            return None if value is None else float(value)
    raise KeyError(f"Metric {symbol!r} is missing from partition {partition!r}.")


def _metric_mapping(observations: np.ndarray, simulations: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    return {
        metric.symbol: metric.value
        for metric in evaluate_metrics(observations, simulations, mask)
    }


def run_seed(
    project_root: Path,
    config_path: Path,
    output_dir: Path,
    seed: int,
) -> Path:
    destination = _archive_path(output_dir, seed)
    if destination.is_file():
        existing = _read_archive(destination)
        existing_seed = existing["config"]["task"]["training"]["seed"]
        if int(existing_seed) != seed:
            raise ValueError(
                f"{destination.name} records seed {existing_seed}, not {seed}."
            )
        print(f"seed={seed} resumed archive={destination.name}", flush=True)
        return destination

    config = load_experiment_config(config_path)
    training = config.task.training.model_copy(update={"seed": seed})
    task = config.task.model_copy(update={"training": training})
    seeded = config.model_copy(update={"task": task})

    def progress(event) -> None:
        epoch = int(event.completed_units)
        if event.stage == "neural_initialization_ensemble" and (
            epoch == 1 or (epoch > 0 and epoch % 25 == 0)
        ):
            score = (
                "unavailable"
                if event.best_score is None
                else f"{event.best_score:.8f}"
            )
            print(
                f"seed={seed} epoch={epoch:03d} "
                f"best_validation_nse={score}",
                flush=True,
            )

    task_id = f"camels-01013500-lstm-seed-{seed}"
    started = perf_counter()
    result = ExperimentRunner(project_root).run(
        seeded,
        task_id=task_id,
        on_progress=progress,
    )
    elapsed = perf_counter() - started
    if result.state.value != "completed":
        raise RuntimeError(
            f"LSTM seed {seed} ended in state {result.state.value}."
        )
    archive = {
        "record_type": "hydrosoftware.run-archive",
        "record_version": "1.0",
        "software_version": __version__,
        "task_id": task_id,
        "experiment_id": seeded.experiment_id,
        "configuration_hash": result.configuration_hash,
        "state": result.state.value,
        "saved_at": _utc_now(),
        "elapsed_seconds": elapsed,
        "config": seeded.model_dump(mode="json"),
        "result": result.as_dict(),
    }
    _write_json(destination, archive)
    validation_nse = _metric_value(archive, "validation", REFERENCE_METRIC)
    print(
        f"seed={seed} completed validation_nse={validation_nse:.8f} "
        f"elapsed_seconds={elapsed:.2f} archive={destination.name}",
        flush=True,
    )
    return destination


def _partition_masks(archive: Mapping[str, Any]) -> dict[str, np.ndarray]:
    period = archive["config"]["task"]["period"]
    warmup = int(archive["config"]["model"]["s2s_warmup"])
    start = int(period["start_index"])
    train_end = int(period["train_end_index"]) - start
    validation_end = int(period["validation_end_index"]) - start
    test_end = int(period["test_end_index"]) - start
    masks = {
        "training": np.zeros(test_end, dtype=bool),
        "validation": np.zeros(test_end, dtype=bool),
        "test": np.zeros(test_end, dtype=bool),
    }
    masks["training"][warmup:train_end] = True
    masks["validation"][train_end:validation_end] = True
    masks["test"][validation_end:test_end] = True
    return masks


def _markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# CAMELS-US 01013500 causal LSTM multi-initialization experiment",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "| Seed | Best epoch | Epochs run | Training NSE | Validation NSE | Test NSE | Validation KGE | Test KGE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            "| {seed} | {best_epoch} | {epochs_completed} | "
            "{training_nse:.6f} | {validation_nse:.6f} | {test_nse:.6f} | "
            "{validation_kge:.6f} | {test_kge:.6f} |".format(**run)
        )
    reference = summary["reference_run"]
    ensemble = summary["initialization_mean"]
    lines.extend(
        [
            "",
            f"Reference seed: `{reference['seed']}`.",
            "",
            "Reference selection: maximum validation NSE after checkpoint "
            "selection within each seed.",
            "",
            f"Reference test NSE: `{reference['test_nse']:.6f}`.",
            "",
            f"Five-member arithmetic-mean test NSE: `{ensemble['test']['NSE']:.6f}`.",
            "",
            f"Scientific verification passed: `{reference['verification_passed']}`.",
            "",
            "The test observations are evaluated only after all five members have "
            "been trained and the reference seed has been selected from validation. "
            "The arithmetic mean is a secondary initialization-sensitivity summary, "
            "not a complete predictive-uncertainty distribution.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(output_dir: Path) -> dict[str, Any]:
    paths = sorted(output_dir.glob("run-seed-*.json"))
    if len(paths) != len(INITIALIZATION_SEEDS):
        raise ValueError(
            "Exactly five completed initialization archives are required before "
            "the LSTM experiment can be finalized."
        )
    archives = [_read_archive(path) for path in paths]
    seeds = [int(item["config"]["task"]["training"]["seed"]) for item in archives]
    if tuple(sorted(seeds)) != INITIALIZATION_SEEDS:
        raise ValueError("The completed LSTM seed set differs from the frozen protocol.")

    source_hashes = {
        item["config"]["data"]["expected_sha256"] for item in archives
    }
    if len(source_hashes) != 1:
        raise ValueError("LSTM members do not share one immutable data source.")
    non_seed_configs = []
    for archive in archives:
        config = json.loads(json.dumps(archive["config"]))
        config["task"]["training"]["seed"] = 0
        non_seed_configs.append(config)
    if any(config != non_seed_configs[0] for config in non_seed_configs[1:]):
        raise ValueError("LSTM members differ in more than initialization seed.")

    runs = []
    predictions = []
    for path, archive in zip(paths, archives):
        output = np.asarray(
            archive["result"]["best_simulation"]["outputs"], dtype=float
        )[:, 0]
        predictions.append(output)
        runs.append(
            {
                "seed": int(archive["config"]["task"]["training"]["seed"]),
                "archive": path.name,
                "configuration_hash": archive["configuration_hash"],
                "best_epoch": int(archive["result"]["training"]["best_epoch"]),
                "epochs_completed": len(archive["result"]["training"]["history"]),
                "stopped_early": bool(
                    archive["result"]["training"]["stopped_early"]
                ),
                "elapsed_seconds": float(archive.get("elapsed_seconds", 0.0)),
                "training_nse": _metric_value(archive, "training", "NSE"),
                "validation_nse": _metric_value(archive, "validation", "NSE"),
                "test_nse": _metric_value(archive, "test", "NSE"),
                "training_kge": _metric_value(archive, "training", "KGE"),
                "validation_kge": _metric_value(archive, "validation", "KGE"),
                "test_kge": _metric_value(archive, "test", "KGE"),
                "test_rmse": _metric_value(archive, "test", "RMSE"),
                "test_pbias": _metric_value(archive, "test", "PBIAS"),
            }
        )

    validation_scores = np.asarray(
        [run["validation_nse"] for run in runs], dtype=float
    )
    reference_position = int(np.argmax(validation_scores))
    reference_archive = archives[reference_position]
    reference_run = runs[reference_position]
    verification = verify_run_archive(reference_archive)
    verification_path = output_dir / "scientific-verification-reference.json"
    _write_json(verification_path, verification)

    prediction_stack = np.stack(predictions)
    common_finite = np.all(np.isfinite(prediction_stack), axis=0)
    initialization_mean = np.full(prediction_stack.shape[1], np.nan)
    initialization_mean[common_finite] = np.mean(
        prediction_stack[:, common_finite], axis=0
    )
    observations = np.asarray(reference_archive["result"]["observations"], dtype=float)
    masks = _partition_masks(reference_archive)
    ensemble_metrics = {
        name: _metric_mapping(
            observations,
            initialization_mean,
            mask & common_finite & np.isfinite(observations),
        )
        for name, mask in masks.items()
    }
    test_values = prediction_stack[:, masks["test"] & common_finite]
    daily_spread = np.std(test_values, axis=0, ddof=0)

    reference = {
        "seed": reference_run["seed"],
        "archive": reference_run["archive"],
        "selection_rule": "maximum validation NSE across the five frozen initializations",
        "selection_uses_test_data": False,
        "validation_nse": reference_run["validation_nse"],
        "test_nse": reference_run["test_nse"],
        "verification_file": verification_path.name,
        "verification_passed": verification["status"] == "passed",
    }
    summary = {
        "method_id": "neural_initialization_multistart.v1",
        "software_version": __version__,
        "generated_at": _utc_now(),
        "basin_id": "01013500",
        "model_id": "lstm.tensorflow_stacked_s2s.v3",
        "source_sha256": next(iter(source_hashes)),
        "selection_metric": "validation NSE",
        "selection_uses_test_data": False,
        "seeds": list(INITIALIZATION_SEEDS),
        "runs": sorted(runs, key=lambda item: item["seed"]),
        "validation_stability": {
            "minimum_nse": float(np.min(validation_scores)),
            "maximum_nse": float(np.max(validation_scores)),
            "range": float(np.ptp(validation_scores)),
            "standard_deviation": float(np.std(validation_scores, ddof=0)),
            "interpretation": (
                "Observed sensitivity to weight initialization under one fixed "
                "data, architecture, transformation, and optimizer protocol."
            ),
        },
        "reference_run": reference,
        "initialization_mean": {
            "aggregation": "arithmetic mean of five frozen member predictions",
            "scope": "weight-initialization sensitivity only",
            **ensemble_metrics,
        },
        "test_prediction_spread_mm_per_day": {
            "mean": float(np.mean(daily_spread)),
            "median": float(np.median(daily_spread)),
            "p95": float(np.quantile(daily_spread, 0.95)),
            "maximum": float(np.max(daily_spread)),
        },
    }
    _write_json(output_dir / "multistart-summary.json", summary)
    report_path = output_dir / "multistart-report.md"
    report_path.write_text(
        _markdown_report(summary), encoding="utf-8", newline="\n"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train five causal LSTM initializations and select the reference "
            "strictly on validation NSE."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/camels_us_01013500_lstm.yml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/camels_us_01013500/lstm_multistart"),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(INITIALIZATION_SEEDS)
    )
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config_path = (
        args.config if args.config.is_absolute() else project_root / args.config
    ).resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else project_root / args.output_dir
    ).resolve()
    if args.run_only and args.finalize_only:
        raise SystemExit("--run-only and --finalize-only are mutually exclusive.")
    unknown = sorted(set(args.seeds) - set(INITIALIZATION_SEEDS))
    if unknown:
        raise SystemExit(f"Seeds outside the frozen protocol: {unknown}")

    if not args.finalize_only:
        for seed in args.seeds:
            run_seed(project_root, config_path, output_dir, seed)
    if not args.run_only:
        summary = finalize(output_dir)
        print(
            f"summary={output_dir / 'multistart-summary.json'} "
            f"reference_seed={summary['reference_run']['seed']} "
            f"reference_validation_nse="
            f"{summary['reference_run']['validation_nse']:.8f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
