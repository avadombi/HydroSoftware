"""Scientifically guarded comparison of completed model-run archives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from hydrosoftware.configuration import (
    CalibrationTaskConfig,
    DiscoveryTrainingTaskConfig,
    ExperimentConfigV1,
    NeuralTrainingTaskConfig,
    PamlTrainingTaskConfig,
)
from hydrosoftware.core.evaluation import evaluate_metrics
from hydrosoftware.execution.serialization import to_jsonable


class RunComparisonError(ValueError):
    """Stable invalid-comparison failure returned at the service boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _ComparableRun:
    task_id: str
    experiment_id: str
    experiment_name: str
    model_id: str
    run_kind: str
    saved_at: str
    config: ExperimentConfigV1
    timestamps: tuple[str, ...] | None
    absolute_rows: tuple[int, ...]
    observations: np.ndarray
    predictions: np.ndarray
    native_evaluation_mask: np.ndarray

    @property
    def keys(self) -> tuple[str | int, ...]:
        return self.timestamps if self.timestamps is not None else self.absolute_rows


def _native_evaluation_mask(
    config: ExperimentConfigV1,
    *,
    output_length: int,
    run_kind: str,
) -> tuple[int, np.ndarray]:
    if run_kind == "forecast":
        return 0, np.ones(output_length, dtype=bool)
    task = config.task
    start = int(task.period.start_index)
    mask = np.zeros(output_length, dtype=bool)
    if isinstance(task, CalibrationTaskConfig):
        if task.period.protocol == "calibration_validation_test":
            assert task.period.validation_end_index is not None
            local_start = int(task.period.validation_end_index) - start
        elif task.period.protocol == "calibration_test":
            assert task.period.calibration_end_index is not None
            local_start = int(task.period.calibration_end_index) - start
        else:
            local_start = int(task.period.warmup_steps)
    elif isinstance(
        task,
        (
            NeuralTrainingTaskConfig,
            PamlTrainingTaskConfig,
            DiscoveryTrainingTaskConfig,
        ),
    ):
        local_start = int(task.period.validation_end_index) - start
    else:  # pragma: no cover - the versioned union is exhaustive
        raise RunComparisonError(
            "unsupported_comparison_task",
            f"Task type {type(task).__name__!r} is not comparable.",
        )
    if local_start < output_length:
        mask[max(0, local_start):] = True
    return start, mask


def _extract_run(archive: Mapping[str, Any]) -> _ComparableRun:
    try:
        config = ExperimentConfigV1.model_validate(archive["config"])
        result = archive["result"]
        simulation = result["best_simulation"]
        output_names = tuple(str(name) for name in simulation["output_names"])
        output_position = output_names.index("streamflow")
        predictions = np.asarray(simulation["outputs"], dtype=float)[:, output_position]
        raw_observations = result["observations"]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise RunComparisonError(
            "comparison_archive_invalid",
            "A selected archive does not contain a valid streamflow trajectory.",
        ) from exc
    if result.get("state") != "completed":
        raise RunComparisonError(
            "comparison_run_incomplete",
            f"Run {archive.get('task_id')!r} is not completed.",
        )
    if raw_observations is None:
        raise RunComparisonError(
            "comparison_observations_missing",
            f"Run {archive.get('task_id')!r} has no observed streamflow.",
        )
    observations = np.asarray(
        [np.nan if value is None else value for value in raw_observations],
        dtype=float,
    )
    if predictions.ndim != 1 or observations.shape != predictions.shape:
        raise RunComparisonError(
            "comparison_trajectory_mismatch",
            f"Run {archive.get('task_id')!r} has misaligned observations and predictions.",
        )
    raw_timestamps = result.get("timestamps")
    timestamps = (
        tuple(str(value) for value in raw_timestamps)
        if raw_timestamps is not None
        else None
    )
    if timestamps is not None:
        if len(timestamps) != len(predictions) or len(set(timestamps)) != len(timestamps):
            raise RunComparisonError(
                "comparison_timestamps_invalid",
                f"Run {archive.get('task_id')!r} has invalid or duplicate timestamps.",
            )
    start, native_mask = _native_evaluation_mask(
        config,
        output_length=len(predictions),
        run_kind=str(result.get("run_kind", "calibration")),
    )
    return _ComparableRun(
        task_id=str(archive["task_id"]),
        experiment_id=config.experiment_id,
        experiment_name=config.name,
        model_id=config.model.model_id,
        run_kind=str(result.get("run_kind", "calibration")),
        saved_at=str(archive.get("saved_at", "")),
        config=config,
        timestamps=timestamps,
        absolute_rows=tuple(range(start, start + len(predictions))),
        observations=observations,
        predictions=predictions,
        native_evaluation_mask=native_mask,
    )


def _validate_source_identity(runs: tuple[_ComparableRun, ...]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    hashes = tuple(run.config.data.expected_sha256 for run in runs)
    if all(hashes):
        if len(set(hashes)) != 1:
            raise RunComparisonError(
                "comparison_sources_differ",
                "Selected runs do not use the same source data.",
            )
        return warnings
    fallback = tuple(
        (
            run.config.data.path,
            run.config.data.columns.precipitation,
            run.config.data.columns.temperature,
            run.config.data.columns.potential_evapotranspiration,
            run.config.data.columns.streamflow,
        )
        for run in runs
    )
    if len(set(fallback)) != 1:
        raise RunComparisonError(
            "comparison_sources_unverified",
            "Source hashes are unavailable and the selected data mappings differ.",
        )
    warnings.append(
        {
            "code": "comparison_source_hash_missing",
            "message": (
                "The runs share a path and column mapping, but at least one legacy "
                "saved run lacks a source-data fingerprint."
            ),
        }
    )
    return warnings


def align_run_archives(
    archives: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Align 1–6 runs and recompute every metric on one common support."""

    if not 1 <= len(archives) <= 6:
        raise RunComparisonError(
            "comparison_run_count_invalid",
            "Select between one and six completed runs.",
        )
    runs = tuple(_extract_run(archive) for archive in archives)
    if len({run.task_id for run in runs}) != len(runs):
        raise RunComparisonError(
            "comparison_duplicate_run",
            "Every selected comparison run must be unique.",
        )
    warnings = _validate_source_identity(runs)
    timestamp_modes = {run.timestamps is not None for run in runs}
    if len(timestamp_modes) != 1:
        raise RunComparisonError(
            "comparison_time_axes_differ",
            "Timestamped and row-indexed runs cannot be aligned together.",
        )

    common_keys = set(runs[0].keys)
    for run in runs[1:]:
        common_keys.intersection_update(run.keys)
    ordered_keys = tuple(key for key in runs[0].keys if key in common_keys)
    if not ordered_keys:
        raise RunComparisonError(
            "comparison_support_empty",
            "The selected runs have no overlapping time support.",
        )

    aligned_predictions: list[np.ndarray] = []
    aligned_masks: list[np.ndarray] = []
    aligned_observations: list[np.ndarray] = []
    for run in runs:
        positions = {key: position for position, key in enumerate(run.keys)}
        indices = np.asarray([positions[key] for key in ordered_keys], dtype=int)
        aligned_predictions.append(run.predictions[indices])
        aligned_observations.append(run.observations[indices])
        aligned_masks.append(run.native_evaluation_mask[indices])

    reference_observations = aligned_observations[0]
    for values in aligned_observations[1:]:
        jointly_finite = np.isfinite(reference_observations) & np.isfinite(values)
        if np.any(
            ~np.isclose(
                reference_observations[jointly_finite],
                values[jointly_finite],
                rtol=1e-9,
                atol=1e-10,
            )
        ) or np.any(np.isfinite(reference_observations) != np.isfinite(values)):
            raise RunComparisonError(
                "comparison_observations_differ",
                "Observed streamflow differs across the overlapping run support.",
            )

    common_mask = np.isfinite(reference_observations)
    for values, native_mask in zip(aligned_predictions, aligned_masks):
        common_mask &= np.isfinite(values) & native_mask
    scored_count = int(np.count_nonzero(common_mask))
    if scored_count < 2:
        raise RunComparisonError(
            "comparison_scored_support_insufficient",
            "Fewer than two observations remain on the common evaluation support.",
        )

    task_types = {run.config.task.task_type for run in runs}
    if len(task_types) > 1:
        warnings.append(
            {
                "code": "comparison_fit_protocols_differ",
                "message": (
                    "The runs use different fitting protocols. Metrics share the "
                    "same evaluation rows, but this is a diagnostic comparison and "
                    "not automatically an unbiased model-ranking experiment."
                ),
            }
        )
    legacy_conceptual_runs = [
        run
        for run in runs
        if isinstance(run.config.task, CalibrationTaskConfig)
        and run.config.task.period.protocol == "legacy_calibration_only"
    ]
    if legacy_conceptual_runs:
        warnings.append(
            {
                "code": "comparison_legacy_conceptual_in_sample",
                "message": (
                    "At least one legacy conceptual run has no holdout test. Its "
                    "comparison support is in-sample and cannot be interpreted as "
                    "generalization performance."
                ),
            }
        )

    comparison_runs = []
    for run, predictions in zip(runs, aligned_predictions):
        comparison_runs.append(
            {
                "task_id": run.task_id,
                "experiment_id": run.experiment_id,
                "experiment_name": run.experiment_name,
                "model_id": run.model_id,
                "run_kind": run.run_kind,
                "saved_at": run.saved_at,
                "metrics": evaluate_metrics(
                    reference_observations,
                    predictions,
                    common_mask,
                ),
                "predictions": predictions,
            }
        )

    scored_positions = np.flatnonzero(common_mask)
    payload = {
        "alignment": "timestamp" if runs[0].timestamps is not None else "source_row",
        "source_sha256": runs[0].config.data.expected_sha256,
        "aligned_rows": len(ordered_keys),
        "scored_observations": scored_count,
        "first_scored_label": str(ordered_keys[int(scored_positions[0])]),
        "last_scored_label": str(ordered_keys[int(scored_positions[-1])]),
        "timestamps": list(ordered_keys) if runs[0].timestamps is not None else None,
        "source_rows": (
            list(ordered_keys) if runs[0].timestamps is None else None
        ),
        "observations": reference_observations,
        "score_mask": common_mask,
        "runs": comparison_runs,
        "warnings": warnings,
    }
    return to_jsonable(payload)


def compare_run_archives(
    archives: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Align 2–6 runs and recompute every metric on one common support."""

    if not 2 <= len(archives) <= 6:
        raise RunComparisonError(
            "comparison_run_count_invalid",
            "Select between two and six completed runs.",
        )
    return align_run_archives(archives)
