"""Train and diagnose repeated single-DU DeepDiscover models for 01013500."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
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


INITIALIZATION_SEEDS = (42, 22, 200, 123, 7)
REFERENCE_METRIC = "NSE"


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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_archive(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
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
    for metric in archive["result"]["partition_metrics"][partition]:
        if metric["symbol"] == symbol:
            value = metric.get("value")
            return None if value is None else float(value)
    raise KeyError(f"Metric {symbol!r} is absent from {partition!r}.")


def _metric_mapping(
    observations: np.ndarray,
    simulations: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    return {
        metric.symbol: metric.value
        for metric in evaluate_metrics(observations, simulations, mask)
    }


def _output(archive: Mapping[str, Any], name: str = "streamflow") -> np.ndarray:
    simulation = archive["result"]["best_simulation"]
    names = list(simulation["output_names"])
    return np.asarray(simulation["outputs"], dtype=float)[:, names.index(name)]


def _partition_masks(archive: Mapping[str, Any]) -> dict[str, np.ndarray]:
    task = archive["config"]["task"]
    period = task["period"]
    start = int(period["start_index"])
    train_end = int(period["train_end_index"]) - start
    validation_end = int(period["validation_end_index"]) - start
    test_end = int(period["test_end_index"]) - start
    masks = {
        "training": np.zeros(test_end, dtype=bool),
        "validation": np.zeros(test_end, dtype=bool),
        "test": np.zeros(test_end, dtype=bool),
    }
    masks["training"][int(task["warmup_steps"]):train_end] = True
    masks["validation"][train_end:validation_end] = True
    masks["test"][validation_end:test_end] = True
    return masks


def _named_column(
    archive: Mapping[str, Any], group: str, names_group: str, name: str
) -> np.ndarray:
    simulation = archive["result"]["best_simulation"]
    names = list(simulation[names_group])
    return np.asarray(simulation[group], dtype=float)[:, names.index(name)]


def _pairwise_stability(
    values: Mapping[int, np.ndarray], mask: np.ndarray
) -> dict[str, float]:
    correlations: list[float] = []
    root_mean_square_differences: list[float] = []
    for first, second in combinations(INITIALIZATION_SEEDS, 2):
        first_values = values[first][mask]
        second_values = values[second][mask]
        correlations.append(float(np.corrcoef(first_values, second_values)[0, 1]))
        root_mean_square_differences.append(
            float(np.sqrt(np.mean(np.square(first_values - second_values))))
        )
    return {
        "minimum_correlation": min(correlations),
        "mean_correlation": float(np.mean(correlations)),
        "maximum_correlation": max(correlations),
        "minimum_root_mean_square_difference": min(
            root_mean_square_differences
        ),
        "mean_root_mean_square_difference": float(
            np.mean(root_mean_square_differences)
        ),
        "maximum_root_mean_square_difference": max(
            root_mean_square_differences
        ),
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
        existing_seed = int(existing["config"]["task"]["training"]["seed"])
        if existing_seed != seed:
            raise ValueError(
                f"{destination.name} records seed {existing_seed}, not {seed}."
            )
        print(f"seed={seed} resumed archive={destination.name}", flush=True)
        return destination

    config = load_experiment_config(config_path)
    training = config.task.training.model_copy(update={"seed": seed})
    seeded = config.model_copy(
        update={"task": config.task.model_copy(update={"training": training})}
    )

    def progress(event) -> None:
        epoch = int(event.completed_units)
        if event.stage == "deepdiscover_training" and (
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

    task_id = f"camels-01013500-deepdiscover-seed-{seed}"
    started = perf_counter()
    result = ExperimentRunner(project_root).run(
        seeded,
        task_id=task_id,
        on_progress=progress,
    )
    elapsed = perf_counter() - started
    if result.state.value != "completed":
        raise RuntimeError(
            f"DeepDiscover seed {seed} ended in state {result.state.value}."
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


def _comparison_metrics(project_root: Path) -> dict[str, dict[str, float]]:
    exp = _read_archive(
        project_root
        / "results/camels_us_01013500/exp_hydro_multistart/run-seed-20260803.json"
    )
    lstm_summary = _read_json(
        project_root
        / "results/camels_us_01013500/lstm_multistart/multistart-summary.json"
    )
    lstm = _read_archive(
        project_root
        / "results/camels_us_01013500/lstm_multistart"
        / lstm_summary["reference_run"]["archive"]
    )
    paml_summary = _read_json(
        project_root
        / "results/camels_us_01013500/paml_multistart/multistart-summary.json"
    )
    paml = _read_archive(
        project_root
        / "results/camels_us_01013500/paml_multistart"
        / paml_summary["reference_run"]["archive"]
    )
    return {
        name: {
            symbol: float(_metric_value(archive, "test", symbol))
            for symbol in ("NSE", "KGE", "RMSE", "PBIAS")
        }
        for name, archive in (("EXP-HYDRO", exp), ("LSTM", lstm), ("PaML", paml))
    }


def _markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# CAMELS-US 01013500 single-DU DeepDiscover multi-initialization",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "| Seed | Best epoch | Epochs run | Training NSE | Validation NSE | Test NSE | Test KGE | Test PBIAS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            "| {seed} | {best_epoch} | {epochs_completed} | {training_nse:.6f} | "
            "{validation_nse:.6f} | {test_nse:.6f} | {test_kge:.6f} | "
            "{test_pbias:.3f}% |".format(**run)
        )
    reference = summary["reference_run"]
    lines.extend(
        [
            "",
            f"Reference seed: `{reference['seed']}`.",
            "",
            "Selection rule: maximum validation NSE across five conventional initializations; test data are excluded from selection.",
            "",
            f"Reference test NSE: `{reference['test_nse']:.6f}`.",
            "",
            "This is the explicitly corrected, instantaneous, mass-conservative DeepDiscover variant. Numerical equivalence with the historical Keras weights is not claimed.",
            "",
            f"Scientific verification passed: `{reference['verification_passed']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(project_root: Path, output_dir: Path) -> dict[str, Any]:
    paths = [_archive_path(output_dir, seed) for seed in INITIALIZATION_SEEDS]
    if not all(path.is_file() for path in paths):
        raise ValueError(
            "Exactly five completed initialization archives are required before "
            "the DeepDiscover experiment can be finalized."
        )
    archives = [_read_archive(path) for path in paths]

    source_hashes = {item["config"]["data"]["expected_sha256"] for item in archives}
    if len(source_hashes) != 1:
        raise ValueError("DeepDiscover members do not share the same data source.")
    non_seed_configs = []
    for archive in archives:
        config = json.loads(json.dumps(archive["config"]))
        config["task"]["training"]["seed"] = 0
        non_seed_configs.append(config)
    if any(config != non_seed_configs[0] for config in non_seed_configs[1:]):
        raise ValueError(
            "DeepDiscover members differ in more than their initialization seed."
        )

    runs: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    for path, archive in zip(paths, archives):
        training = archive["result"]["training"]
        predictions.append(_output(archive))
        runs.append(
            {
                "seed": int(archive["config"]["task"]["training"]["seed"]),
                "archive": path.name,
                "configuration_hash": archive["configuration_hash"],
                "best_epoch": int(training["best_epoch"]),
                "epochs_completed": len(training["history"]),
                "stopped_early": bool(training["stopped_early"]),
                "elapsed_seconds": float(archive.get("elapsed_seconds", 0.0)),
                "training_nse": _metric_value(archive, "training", "NSE"),
                "validation_nse": _metric_value(archive, "validation", "NSE"),
                "test_nse": _metric_value(archive, "test", "NSE"),
                "test_kge": _metric_value(archive, "test", "KGE"),
                "test_rmse": _metric_value(archive, "test", "RMSE"),
                "test_pbias": _metric_value(archive, "test", "PBIAS"),
                "snow_parameters": training["snow_parameters"],
                "input_switch_probabilities": training[
                    "input_switch_probabilities"
                ],
                "process_switch_probabilities": training[
                    "process_switch_probabilities"
                ],
                "input_scale_weights": archive["result"]["best_simulation"][
                    "diagnostics"
                ]["input_scale_weights"],
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
    prediction_mean = np.full(prediction_stack.shape[1], np.nan)
    prediction_mean[common_finite] = np.mean(
        prediction_stack[:, common_finite], axis=0
    )
    observations = np.asarray(reference_archive["result"]["observations"], dtype=float)
    masks = _partition_masks(reference_archive)
    sensitivity_mean_metrics = {
        name: _metric_mapping(
            observations,
            prediction_mean,
            mask & common_finite & np.isfinite(observations),
        )
        for name, mask in masks.items()
    }
    test_spread = np.std(
        prediction_stack[:, masks["test"] & common_finite], axis=0, ddof=0
    )
    post_warmup_mask = np.arange(len(observations)) >= int(
        reference_archive["config"]["task"]["warmup_steps"]
    )
    storage_by_seed = {
        seed: _named_column(archive, "states", "state_names", "discovered_storage")[
            1:
        ]
        for seed, archive in zip(INITIALIZATION_SEEDS, archives)
    }
    aet_by_seed = {
        seed: _named_column(
            archive,
            "fluxes",
            "flux_names",
            "actual_evapotranspiration",
        )
        for seed, archive in zip(INITIALIZATION_SEEDS, archives)
    }
    snow_parameter_names = (
        "snowfall_temperature",
        "melt_temperature",
        "degree_day_factor",
        "sharpness",
    )
    snow_parameter_stability = {}
    for name in snow_parameter_names:
        values = np.asarray(
            [run["snow_parameters"][name] for run in runs], dtype=float
        )
        snow_parameter_stability[name] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "range": float(np.ptp(values)),
            "standard_deviation": float(np.std(values, ddof=0)),
        }
    tail_length = 100
    learning_tail = {}
    for run, archive in zip(runs, archives):
        history = archive["result"]["training"]["history"]
        best_values = np.maximum.accumulate(
            np.asarray(
                [item["validation_objective"] for item in history], dtype=float
            )
        )
        start_position = max(0, len(best_values) - tail_length - 1)
        learning_tail[str(run["seed"])] = {
            "epochs_compared": len(best_values) - 1 - start_position,
            "best_validation_gain": float(
                best_values[-1] - best_values[start_position]
            ),
            "best_epoch_reaches_budget": (
                run["best_epoch"] == run["epochs_completed"]
            ),
        }
    reference = {
        **reference_run,
        "selection_rule": "maximum validation NSE across five initializations",
        "selection_uses_test_data": False,
        "verification_file": verification_path.name,
        "verification_passed": verification["status"] == "passed",
    }
    comparison = _comparison_metrics(project_root)
    comparison["DeepDiscover"] = {
        symbol: float(reference_run[f"test_{symbol.lower()}"])
        for symbol in ("NSE", "KGE", "RMSE", "PBIAS")
    }
    summary = {
        "method_id": "deepdiscover_instantaneous_multistart.v1",
        "software_version": __version__,
        "generated_at": _utc_now(),
        "basin_id": "01013500",
        "model_id": "deepdiscover.numpy_instantaneous_s2s.v1",
        "historical_architecture_basis": {
            "source": "deep_discover_demo.ipynb",
            "model_type": "physical",
            "hidden_units": 8,
            "process_count": 1,
        },
        "implementation_relationship": (
            "corrected instantaneous mass-conservative variant; historical "
            "Keras weight equivalence not claimed"
        ),
        "source_sha256": next(iter(source_hashes)),
        "selection_metric": "validation NSE",
        "selection_uses_test_data": False,
        "seeds": list(INITIALIZATION_SEEDS),
        "runs": runs,
        "validation_stability": {
            "minimum_nse": float(np.min(validation_scores)),
            "maximum_nse": float(np.max(validation_scores)),
            "range": float(np.ptp(validation_scores)),
            "standard_deviation": float(np.std(validation_scores, ddof=0)),
        },
        "process_stability_after_warmup": {
            "discovered_storage_mm": _pairwise_stability(
                storage_by_seed, post_warmup_mask
            ),
            "actual_evapotranspiration_mm_per_day": _pairwise_stability(
                aet_by_seed, post_warmup_mask
            ),
            "streamflow_mm_per_day": _pairwise_stability(
                {
                    seed: values
                    for seed, values in zip(INITIALIZATION_SEEDS, predictions)
                },
                post_warmup_mask,
            ),
            "snow_parameters": snow_parameter_stability,
        },
        "optimization_tail_diagnostic": {
            "interpretation": (
                "Fixed-budget sufficiency diagnostic; a terminal best epoch is "
                "not presented as proof of mathematical convergence."
            ),
            "last_up_to_100_epochs": learning_tail,
        },
        "reference_run": reference,
        "initialization_sensitivity_mean": {
            "interpretation": (
                "secondary predictive sensitivity summary only; discovered "
                "states and processes are not averaged"
            ),
            **sensitivity_mean_metrics,
        },
        "test_prediction_spread_mm_per_day": {
            "mean": float(np.mean(test_spread)),
            "median": float(np.median(test_spread)),
            "p95": float(np.quantile(test_spread, 0.95)),
            "maximum": float(np.max(test_spread)),
        },
        "reference_test_comparison": comparison,
    }
    _write_json(output_dir / "multistart-summary.json", summary)
    (output_dir / "multistart-report.md").write_text(
        _markdown_report(summary), encoding="utf-8", newline="\n"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train five single-DU DeepDiscover initializations and select "
            "the reference strictly on validation NSE."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/camels_us_01013500_deepdiscover.yml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/camels_us_01013500/deepdiscover_multistart"),
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
        summary = finalize(project_root, output_dir)
        print(
            f"summary={output_dir / 'multistart-summary.json'} "
            f"reference_seed={summary['reference_run']['seed']} "
            f"reference_validation_nse="
            f"{summary['reference_run']['validation_nse']:.8f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
