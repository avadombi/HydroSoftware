"""Run and diagnose repeated EXP-HYDRO calibrations for CAMELS-US 01013500."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hydrosoftware import __version__
from hydrosoftware.configuration import load_experiment_config
from hydrosoftware.execution.experiments import ExperimentRunner
from hydrosoftware.execution.scientific_verification import verify_run_archive


INITIAL_SEEDS = (20260801, 20260802, 20260803)
EXTENSION_SEEDS = (20260804, 20260805)
SCORE_RANGE_TOLERANCE = 0.01
PARAMETER_NORMALIZED_RANGE_TOLERANCE = 0.10
PARAMETER_BOUNDS = {
    "recession_factor": (0.0, 0.1),
    "maximum_storage": (100.0, 1500.0),
    "maximum_subsurface_flow": (10.0, 50.0),
    "snowfall_temperature": (-3.0, 0.0),
    "melt_temperature": (0.0, 3.0),
    "degree_day_factor": (0.0, 5.0),
}


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


def run_seed(
    project_root: Path,
    config_path: Path,
    output_dir: Path,
    seed: int,
    workers: int,
) -> Path:
    destination = _archive_path(output_dir, seed)
    if destination.is_file():
        existing = _read_archive(destination)
        existing_seed = existing["config"]["task"]["search"]["seed"]
        if int(existing_seed) != seed:
            raise ValueError(
                f"{destination.name} records seed {existing_seed}, not {seed}."
            )
        print(f"seed={seed} resumed archive={destination.name}", flush=True)
        return destination

    config = load_experiment_config(config_path)
    search = config.task.search.model_copy(update={"seed": seed})
    task = config.task.model_copy(update={"search": search})
    seeded = config.model_copy(update={"task": task})
    population_size = int(search.population_size)

    def progress(event) -> None:
        completed = event.completed_units
        if (
            event.stage == "search"
            and completed > 0
            and completed % population_size == 0
        ):
            generation = completed // population_size - 1
            score = "unavailable" if event.best_score is None else f"{event.best_score:.8f}"
            print(
                f"seed={seed} generation={generation:03d} "
                f"best_calibration_nse={score}",
                flush=True,
            )

    task_id = f"camels-01013500-exp-hydro-seed-{seed}"
    result = ExperimentRunner(project_root).run(
        seeded,
        task_id=task_id,
        on_progress=progress,
        calibration_workers=workers,
    )
    if result.state.value != "completed":
        raise RuntimeError(
            f"Calibration seed {seed} ended in state {result.state.value}."
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
        "config": seeded.model_dump(mode="json"),
        "result": result.as_dict(),
    }
    _write_json(destination, archive)
    print(
        f"seed={seed} completed calibration_nse="
        f"{result.calibration.best_score:.10f} archive={destination.name}",
        flush=True,
    )
    return destination


def _metric_value(archive: Mapping[str, Any], partition: str, symbol: str) -> float | None:
    metrics = archive["result"]["partition_metrics"][partition]
    for metric in metrics:
        if metric["symbol"] == symbol:
            value = metric.get("value")
            return None if value is None else float(value)
    raise KeyError(f"Metric {symbol!r} is missing from partition {partition!r}.")


def _best_by_generation(archive: Mapping[str, Any]) -> list[float]:
    records = archive["result"]["calibration"]["records"]
    direction = archive["result"]["calibration"]["objective_direction"]
    generations: dict[int, list[float]] = {}
    for record in records:
        if record.get("score") is None:
            continue
        generation = int(record.get("generation") or 0)
        generations.setdefault(generation, []).append(float(record["score"]))
    curve = []
    best: float | None = None
    for generation in sorted(generations):
        candidate = (
            max(generations[generation])
            if direction == "maximize"
            else min(generations[generation])
        )
        if best is None:
            best = candidate
        elif direction == "maximize":
            best = max(best, candidate)
        else:
            best = min(best, candidate)
        curve.append(float(best))
    return curve


def _last_improvement_generation(curve: Sequence[float]) -> int:
    last = 0
    for generation in range(1, len(curve)):
        if curve[generation] != curve[generation - 1]:
            last = generation
    return last


def _markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# CAMELS-US 01013500 EXP-HYDRO multi-start calibration",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "| Seed | Calibration NSE | Validation NSE | Test NSE | Calibration KGE | Validation KGE | Test KGE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            "| {seed} | {calibration_nse:.6f} | {validation_nse:.6f} | "
            "{test_nse:.6f} | {calibration_kge:.6f} | {validation_kge:.6f} | "
            "{test_kge:.6f} |".format(**run)
        )
    lines.extend(
        [
            "",
            f"Calibration-NSE range: `{summary['score_stability']['range']:.10f}`.",
            "",
            f"Optimizer-score stability: `{summary['score_stability']['stable']}`.",
            "",
            f"Parameter-solution stability: `{summary['parameter_stability']['stable']}`.",
            "",
            "Parameters touching a search bound: `"
            + ", ".join(summary["parameter_stability"]["boundary_parameters"])
            + "`.",
            "",
        ]
    )
    if summary["reference_run"] is None:
        lines.append(
            "No reference run was frozen because the predeclared score-stability "
            "criterion was not met."
        )
    else:
        lines.extend(
            [
                f"Reference run: `{summary['reference_run']['archive']}`.",
                "",
                f"Reference seed: `{summary['reference_run']['seed']}`.",
                "",
                f"Scientific verification passed: `{summary['reference_run']['verification_passed']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "Parameter dispersion is diagnosed separately from objective convergence. "
            "Similar scores with different parameter sets indicate possible equifinality; "
            "they do not justify claiming parameter identifiability.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(output_dir: Path) -> dict[str, Any]:
    paths = sorted(output_dir.glob("run-seed-*.json"))
    if not paths:
        raise ValueError("No completed seed archives are available to summarize.")
    archives = []
    for path in paths:
        archive = _read_archive(path)
        if "software_version" not in archive:
            archive["software_version"] = __version__
            _write_json(path, archive)
        archives.append(archive)
    runs = []
    parameter_values: dict[str, list[float]] = {
        name: [] for name in PARAMETER_BOUNDS
    }
    for path, archive in zip(paths, archives):
        calibration = archive["result"]["calibration"]
        parameters = calibration["best_parameters"]
        seed = int(archive["config"]["task"]["search"]["seed"])
        for name in parameter_values:
            parameter_values[name].append(float(parameters[name]))
        convergence_curve = _best_by_generation(archive)
        runs.append(
            {
                "seed": seed,
                "archive": path.name,
                "configuration_hash": archive["configuration_hash"],
                "calibration_nse": float(calibration["best_score"]),
                "validation_nse": _metric_value(archive, "validation", "NSE"),
                "test_nse": _metric_value(archive, "test", "NSE"),
                "calibration_kge": _metric_value(archive, "calibration", "KGE"),
                "validation_kge": _metric_value(archive, "validation", "KGE"),
                "test_kge": _metric_value(archive, "test", "KGE"),
                "parameters": parameters,
                "best_by_generation": convergence_curve,
                "last_improvement_generation": _last_improvement_generation(
                    convergence_curve
                ),
            }
        )

    calibration_scores = np.asarray(
        [run["calibration_nse"] for run in runs], dtype=float
    )
    score_range = float(np.ptp(calibration_scores))
    score_stable = len(runs) >= 3 and score_range <= SCORE_RANGE_TOLERANCE
    parameter_diagnostics = {}
    for name, values in parameter_values.items():
        lower, upper = PARAMETER_BOUNDS[name]
        array = np.asarray(values, dtype=float)
        normalized_range = float(np.ptp(array) / (upper - lower))
        touches_lower = bool(np.any(np.isclose(array, lower, atol=1e-12, rtol=0.0)))
        touches_upper = bool(np.any(np.isclose(array, upper, atol=1e-12, rtol=0.0)))
        parameter_diagnostics[name] = {
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
            "mean": float(np.mean(array)),
            "standard_deviation": float(np.std(array, ddof=0)),
            "normalized_range": normalized_range,
            "stable": normalized_range <= PARAMETER_NORMALIZED_RANGE_TOLERANCE,
            "touches_lower_bound": touches_lower,
            "touches_upper_bound": touches_upper,
        }
    parameter_stable = all(
        item["stable"] for item in parameter_diagnostics.values()
    )
    boundary_parameters = [
        name
        for name, item in parameter_diagnostics.items()
        if item["touches_lower_bound"] or item["touches_upper_bound"]
    ]

    best_position = int(np.argmax(calibration_scores))
    reference = None
    verification_path = output_dir / "scientific-verification-reference.json"
    if score_stable:
        best_archive = archives[best_position]
        verification = verify_run_archive(best_archive)
        _write_json(verification_path, verification)
        reference = {
            "seed": runs[best_position]["seed"],
            "archive": runs[best_position]["archive"],
            "selection_rule": "maximum calibration NSE among stable optimizer repetitions",
            "verification_file": verification_path.name,
            "verification_passed": verification["status"] == "passed",
        }
    elif verification_path.exists():
        verification_path.unlink()

    completed_seeds = {run["seed"] for run in runs}
    additional_required = (
        [seed for seed in EXTENSION_SEEDS if seed not in completed_seeds]
        if not score_stable and len(runs) < 5
        else []
    )
    summary = {
        "method_id": "conceptual_multistart_stability.v1",
        "software_version": __version__,
        "generated_at": _utc_now(),
        "basin_id": "01013500",
        "model_id": "exp_hydro.mass_conservative.v1",
        "selection_uses_holdout_data": False,
        "runs": sorted(runs, key=lambda item: item["seed"]),
        "score_stability": {
            "metric": "calibration NSE",
            "range": score_range,
            "tolerance": SCORE_RANGE_TOLERANCE,
            "stable": score_stable,
            "interpretation": (
                "This is an optimizer-repeatability diagnostic, not a confidence interval."
            ),
        },
        "parameter_stability": {
            "normalized_range_tolerance": PARAMETER_NORMALIZED_RANGE_TOLERANCE,
            "stable": parameter_stable,
            "boundary_parameters": boundary_parameters,
            "parameters": parameter_diagnostics,
            "interpretation": (
                "Parameter dispersion is reported independently because stable performance "
                "does not establish hydrological parameter identifiability."
            ),
        },
        "additional_seeds_required": additional_required,
        "reference_run": reference,
    }
    _write_json(output_dir / "multistart-summary.json", summary)
    report = _markdown_report(summary)
    report_path = output_dir / "multistart-report.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact repeated EXP-HYDRO calibrations and diagnose optimizer "
            "and parameter stability."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/camels_us_01013500_exp_hydro.yml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/camels_us_01013500/exp_hydro_multistart"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(INITIAL_SEEDS))
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
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
    if args.workers < 1:
        raise SystemExit("--workers must be at least one.")
    if args.run_only and args.finalize_only:
        raise SystemExit("--run-only and --finalize-only are mutually exclusive.")

    if not args.finalize_only:
        for seed in args.seeds:
            run_seed(
                project_root,
                config_path,
                output_dir,
                seed,
                args.workers,
            )
    if not args.run_only:
        summary = finalize(output_dir)
        print(
            "summary="
            f"{output_dir / 'multistart-summary.json'} "
            f"stable={summary['score_stability']['stable']} "
            f"additional_seeds={summary['additional_seeds_required']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
