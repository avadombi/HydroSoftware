"""Rolling-origin temporal validation for reproducible hydrological experiments."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from statistics import median
from threading import RLock
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from hydrosoftware.configuration import (
    CalibrationTaskConfig,
    DiscoveryTrainingTaskConfig,
    ExperimentConfigV1,
    NeuralTrainingTaskConfig,
    PamlTrainingTaskConfig,
    configuration_hash,
)
from hydrosoftware.core.evaluation import EvaluationMetricResult, evaluate_metrics
from hydrosoftware.data_sources import analyze_calibration_csv
from hydrosoftware.execution.experiments import (
    ExperimentExecutionError,
    ExperimentRunner,
)
from hydrosoftware.execution.forecasting import (
    configured_full_state,
    simulate_frozen_model,
)
from hydrosoftware.execution.manager import (
    ResultNotReadyError,
    TaskNotFoundError,
    TaskSnapshot,
)
from hydrosoftware.execution.serialization import to_jsonable
from hydrosoftware.execution.tasks import (
    CancellationToken,
    ProgressEvent,
    ProgressSeverity,
    StructuredError,
    TaskState,
)


TEMPORAL_VALIDATION_METHOD_ID = "rolling_origin_temporal_validation.v1"
TERMINAL_STATES = {TaskState.COMPLETED, TaskState.CANCELLED, TaskState.FAILED}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TemporalValidationError(ValueError):
    """Stable scientific precondition failure for temporal validation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RollingOriginSettings:
    """Common chronological fold geometry for one or more experiments."""

    start_index: int = 0
    initial_training_steps: int = 1_095
    validation_steps: int = 365
    test_steps: int = 365
    step_steps: int = 365
    folds: int = 3
    window_mode: str = "expanding"
    minimum_scored_observations: int = 30

    def __post_init__(self) -> None:
        integer_fields = {
            "start_index": self.start_index,
            "initial_training_steps": self.initial_training_steps,
            "validation_steps": self.validation_steps,
            "test_steps": self.test_steps,
            "step_steps": self.step_steps,
            "folds": self.folds,
            "minimum_scored_observations": self.minimum_scored_observations,
        }
        if self.start_index < 0:
            raise TemporalValidationError(
                "temporal_validation_invalid_start",
                "The first temporal-validation row cannot be negative.",
            )
        for name, value in integer_fields.items():
            if name != "start_index" and value < 1:
                raise TemporalValidationError(
                    "temporal_validation_invalid_length",
                    f"{name} must be at least one.",
                )
        if self.folds > 10:
            raise TemporalValidationError(
                "temporal_validation_too_many_folds",
                "Temporal validation supports at most ten folds per analysis.",
            )
        if self.window_mode not in {"expanding", "sliding"}:
            raise TemporalValidationError(
                "temporal_validation_invalid_window",
                "window_mode must be 'expanding' or 'sliding'.",
            )
        if self.step_steps < self.test_steps:
            raise TemporalValidationError(
                "temporal_validation_overlapping_tests",
                "The origin step must be at least the test length so held-out "
                "test blocks do not overlap.",
            )

    @property
    def required_rows(self) -> int:
        """Exclusive source-row boundary required by the final fold."""

        return (
            self.start_index
            + self.initial_training_steps
            + (self.folds - 1) * self.step_steps
            + self.validation_steps
            + self.test_steps
        )

    def as_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class TemporalFold:
    """Absolute half-open row boundaries for one rolling-origin fold."""

    fold: int
    train_start_index: int
    train_end_index: int
    validation_end_index: int
    test_end_index: int

    @property
    def test_start_index(self) -> int:
        return self.validation_end_index

    def as_dict(self) -> dict[str, int]:
        return to_jsonable(self)


def build_temporal_folds(settings: RollingOriginSettings) -> tuple[TemporalFold, ...]:
    """Build deterministic expanding- or sliding-window folds."""

    folds: list[TemporalFold] = []
    for position in range(settings.folds):
        shift = position * settings.step_steps
        train_start = (
            settings.start_index
            if settings.window_mode == "expanding"
            else settings.start_index + shift
        )
        train_end = (
            settings.start_index + settings.initial_training_steps + shift
            if settings.window_mode == "expanding"
            else train_start + settings.initial_training_steps
        )
        validation_end = train_end + settings.validation_steps
        folds.append(
            TemporalFold(
                fold=position + 1,
                train_start_index=train_start,
                train_end_index=train_end,
                validation_end_index=validation_end,
                test_end_index=validation_end + settings.test_steps,
            )
        )
    return tuple(folds)


def _fold_config(
    config: ExperimentConfigV1,
    fold: TemporalFold,
) -> ExperimentConfigV1:
    """Return a fold-specific configuration without mutating the saved setup."""

    payload = config.model_dump(mode="json")
    payload["metadata"] = {
        **payload.get("metadata", {}),
        "temporal_validation_method": TEMPORAL_VALIDATION_METHOD_ID,
        "temporal_validation_fold": str(fold.fold),
    }
    task = payload["task"]
    if isinstance(config.task, CalibrationTaskConfig):
        task["period"] = {
            "protocol": "calibration_validation_test",
            "start_index": fold.train_start_index,
            "calibration_end_index": fold.train_end_index,
            "validation_end_index": fold.validation_end_index,
            "test_end_index": fold.test_end_index,
            "warmup_steps": config.task.period.warmup_steps,
        }
    else:
        task["period"] = {
            "start_index": fold.train_start_index,
            "train_end_index": fold.train_end_index,
            "validation_end_index": fold.validation_end_index,
            "test_end_index": fold.test_end_index,
        }
    return ExperimentConfigV1.model_validate(payload)


def _processed_source(
    config: ExperimentConfigV1,
    runner: ExperimentRunner,
) -> Any:
    analysis = analyze_calibration_csv(
        config.data,
        workspace_root=runner.workspace_root,
    )
    if not analysis.report.ready:
        first = next(
            issue for issue in analysis.report.issues if issue.severity == "error"
        )
        raise TemporalValidationError(first.code, first.explanation)
    return analysis


def _validate_common_source(
    configs: Sequence[ExperimentConfigV1],
    runner: ExperimentRunner,
    settings: RollingOriginSettings,
) -> tuple[Any, tuple[Any, ...]]:
    analyses = tuple(_processed_source(config, runner) for config in configs)
    reference = analyses[0]
    if settings.required_rows > len(reference.streamflow):
        raise TemporalValidationError(
            "temporal_validation_period_exceeds_data",
            "The requested final fold ends at source row "
            f"{settings.required_rows}, but the common source contains only "
            f"{len(reference.streamflow)} rows.",
        )
    for config, analysis in zip(configs[1:], analyses[1:]):
        if analysis.report.sha256 != reference.report.sha256:
            raise TemporalValidationError(
                "temporal_validation_source_mismatch",
                "Every selected experiment must use the same CSV content.",
            )
        if len(analysis.streamflow) != len(reference.streamflow):
            raise TemporalValidationError(
                "temporal_validation_row_mismatch",
                "Selected experiments do not expose the same number of processed rows.",
            )
        arrays = (
            ("precipitation", analysis.precipitation, reference.precipitation),
            ("temperature", analysis.temperature, reference.temperature),
            (
                "potential evapotranspiration",
                analysis.potential_evapotranspiration,
                reference.potential_evapotranspiration,
            ),
            ("streamflow", analysis.streamflow, reference.streamflow),
        )
        for name, candidate, expected in arrays:
            if not np.allclose(candidate, expected, equal_nan=True):
                raise TemporalValidationError(
                    "temporal_validation_processing_mismatch",
                    "Selected experiments do not produce identical processed "
                    f"{name} values from the common source. Align their mapping, "
                    "units, and data-quality rules before comparing them.",
                )
        if (
            (analysis.timestamps is None) != (reference.timestamps is None)
            or (
                analysis.timestamps is not None
                and tuple(analysis.timestamps) != tuple(reference.timestamps)
            )
        ):
            raise TemporalValidationError(
                "temporal_validation_timestamp_mismatch",
                "Selected experiments do not expose the same processed time axis.",
            )
    return reference, analyses


def _metric_sort_value(metric: EvaluationMetricResult) -> float:
    if metric.value is None:
        return float("inf")
    if metric.direction == "maximize":
        return -float(metric.value)
    if metric.direction == "minimize":
        return float(metric.value)
    return abs(float(metric.value) - float(metric.ideal_value))


def _positive_improvement(
    reference: EvaluationMetricResult,
    candidate: EvaluationMetricResult,
) -> float | None:
    if reference.value is None or candidate.value is None:
        return None
    if reference.direction == "maximize":
        return float(candidate.value - reference.value)
    if reference.direction == "minimize":
        return float(reference.value - candidate.value)
    return abs(float(reference.value - reference.ideal_value)) - abs(
        float(candidate.value - candidate.ideal_value)
    )


def _metric_summary(
    values: Sequence[float],
) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {
            "available_folds": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "available_folds": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(median(finite.tolist())),
        "standard_deviation": (
            float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        ),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _conceptual_fold_prediction(
    config: ExperimentConfigV1,
    fold_config: ExperimentConfigV1,
    analysis: Any,
    fold: TemporalFold,
    result: Any,
) -> np.ndarray:
    calibration = result.calibration
    if (
        calibration is None
        or not calibration.best_parameters
        or result.state is not TaskState.COMPLETED
    ):
        raise TemporalValidationError(
            "temporal_validation_fold_incomplete",
            f"Conceptual calibration did not complete in fold {fold.fold}.",
        )
    start = fold.train_start_index
    stop = fold.test_end_index
    simulation = simulate_frozen_model(
        fold_config,
        calibration.best_parameters,
        analysis.precipitation[start:stop],
        analysis.temperature[start:stop],
        analysis.potential_evapotranspiration[start:stop],
        initial_state=configured_full_state(fold_config),
    )
    test_offset = fold.test_start_index - start
    return np.asarray(simulation.outputs[test_offset:, 0], dtype=float)


def _learned_fold_prediction(
    result: Any,
    fold: TemporalFold,
) -> np.ndarray:
    if result.state is not TaskState.COMPLETED or result.best_simulation is None:
        raise TemporalValidationError(
            "temporal_validation_fold_incomplete",
            f"Training did not complete in fold {fold.fold}.",
        )
    offset = fold.test_start_index - fold.train_start_index
    return np.asarray(result.best_simulation.outputs[offset:, 0], dtype=float)


class RollingOriginValidator:
    """Refit saved experiment definitions on common chronological folds."""

    def __init__(self, runner: ExperimentRunner) -> None:
        self.runner = runner

    def run(
        self,
        configs: Sequence[ExperimentConfigV1],
        settings: RollingOriginSettings,
        *,
        task_id: str,
        cancellation_token: CancellationToken | None = None,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> dict[str, Any]:
        if not configs:
            raise TemporalValidationError(
                "temporal_validation_no_experiments",
                "Select at least one saved experiment for temporal validation.",
            )
        if len(configs) > 4:
            raise TemporalValidationError(
                "temporal_validation_too_many_experiments",
                "Temporal validation supports at most four experiment definitions.",
            )
        identifiers = [config.experiment_id for config in configs]
        if len(set(identifiers)) != len(identifiers):
            raise TemporalValidationError(
                "temporal_validation_duplicate_experiments",
                "Selected experiment identifiers must be unique.",
            )

        token = cancellation_token or CancellationToken()
        source, analyses = _validate_common_source(
            configs,
            self.runner,
            settings,
        )
        folds = build_temporal_folds(settings)
        total_units = len(folds) * len(configs)
        completed_units = 0
        fold_payloads: list[dict[str, Any]] = []
        pooled_observations: list[np.ndarray] = []
        pooled_predictions: dict[str, list[np.ndarray]] = {
            config.experiment_id: [] for config in configs
        }

        def emit(
            *,
            state: TaskState,
            stage: str,
            message: str,
            fraction: float | None = None,
            best_score: float | None = None,
        ) -> None:
            if on_progress is None:
                return
            on_progress(
                ProgressEvent(
                    task_id=task_id,
                    sequence=0,
                    state=state,
                    stage=stage,
                    message=message,
                    completed_units=completed_units,
                    total_units=total_units,
                    fraction=(
                        completed_units / total_units
                        if fraction is None
                        else fraction
                    ),
                    best_score=best_score,
                )
            )

        emit(
            state=TaskState.PREPARING,
            stage="temporal_validation_preflight",
            message=(
                "The common source and chronological fold boundaries were validated."
            ),
        )

        for fold in folds:
            if token.is_requested:
                emit(
                    state=TaskState.CANCELLED,
                    stage="temporal_validation",
                    message="Temporal validation was cancelled at a safe fold boundary.",
                )
                return {
                    "task_id": task_id,
                    "state": TaskState.CANCELLED.value,
                    "method_id": TEMPORAL_VALIDATION_METHOD_ID,
                    "settings": settings.as_dict(),
                    "folds_completed": len(fold_payloads),
                    "folds_requested": len(folds),
                }

            test_observations = np.asarray(
                source.streamflow[fold.test_start_index:fold.test_end_index],
                dtype=float,
            )
            model_predictions: dict[str, np.ndarray] = {}
            model_records: list[dict[str, Any]] = []

            for model_position, (config, analysis) in enumerate(
                zip(configs, analyses)
            ):
                fold_config = _fold_config(config, fold)
                model_label = config.name
                unit_index = completed_units

                def relay(event: ProgressEvent) -> None:
                    if on_progress is None:
                        return
                    local_fraction = event.fraction
                    if local_fraction is None:
                        local_fraction = 0.0
                    on_progress(
                        replace(
                            event,
                            task_id=task_id,
                            state=(
                                event.state
                                if event.state
                                in {TaskState.FAILED, TaskState.CANCELLED}
                                else TaskState.RUNNING
                            ),
                            stage=(
                                f"temporal_validation_fold_{fold.fold}_"
                                f"{event.stage}"
                            ),
                            message=(
                                f"Fold {fold.fold}/{len(folds)} · "
                                f"{model_label}: {event.message}"
                            ),
                            completed_units=unit_index,
                            total_units=total_units,
                            fraction=(unit_index + local_fraction) / total_units,
                        )
                    )

                try:
                    fold_result = self.runner.run(
                        fold_config,
                        task_id=(
                            f"{task_id}-fold-{fold.fold}-model-"
                            f"{model_position + 1}"
                        ),
                        cancellation_token=token,
                        on_progress=relay,
                    )
                except ExperimentExecutionError as exc:
                    raise TemporalValidationError(
                        f"temporal_validation_{exc.error.code}",
                        f"Fold {fold.fold}, {config.name}: "
                        f"{exc.error.explanation}",
                    ) from exc
                if fold_result.state is TaskState.CANCELLED:
                    emit(
                        state=TaskState.CANCELLED,
                        stage="temporal_validation",
                        message=(
                            "Temporal validation was cancelled inside a model fit."
                        ),
                    )
                    return {
                        "task_id": task_id,
                        "state": TaskState.CANCELLED.value,
                        "method_id": TEMPORAL_VALIDATION_METHOD_ID,
                        "settings": settings.as_dict(),
                        "folds_completed": len(fold_payloads),
                        "folds_requested": len(folds),
                    }
                if isinstance(config.task, CalibrationTaskConfig):
                    prediction = _conceptual_fold_prediction(
                        config,
                        fold_config,
                        analysis,
                        fold,
                        fold_result,
                    )
                    fitting_end = fold.validation_end_index
                    validation_used_for = "conceptual_parameter_calibration"
                else:
                    prediction = _learned_fold_prediction(fold_result, fold)
                    fitting_end = fold.train_end_index
                    validation_used_for = "checkpoint_selection"
                if prediction.shape != test_observations.shape:
                    raise TemporalValidationError(
                        "temporal_validation_prediction_length",
                        f"Fold {fold.fold} produced {len(prediction)} predictions "
                        f"for {len(test_observations)} test observations.",
                    )
                model_predictions[config.experiment_id] = prediction
                model_records.append(
                    {
                        "experiment_id": config.experiment_id,
                        "experiment_name": config.name,
                        "model_id": config.model.model_id,
                        "configuration_hash": configuration_hash(config),
                        "fold_configuration_hash": configuration_hash(fold_config),
                        "fitting_end_index": fitting_end,
                        "validation_used_for": validation_used_for,
                        "prediction": prediction,
                    }
                )
                completed_units += 1
                emit(
                    state=TaskState.RUNNING,
                    stage="temporal_validation",
                    message=(
                        f"Fold {fold.fold}/{len(folds)} completed for "
                        f"{config.name}."
                    ),
                )

            common_mask = np.isfinite(test_observations)
            for prediction in model_predictions.values():
                common_mask &= np.isfinite(prediction)
            scored = int(np.count_nonzero(common_mask))
            if scored < settings.minimum_scored_observations:
                raise TemporalValidationError(
                    "temporal_validation_insufficient_test_observations",
                    f"Fold {fold.fold} contains {scored} common finite test "
                    "observations; at least "
                    f"{settings.minimum_scored_observations} are required.",
                )
            pooled_observations.append(test_observations[common_mask])
            for record in model_records:
                prediction = np.asarray(record.pop("prediction"), dtype=float)
                metrics = evaluate_metrics(
                    test_observations,
                    prediction,
                    common_mask,
                )
                record["metrics"] = metrics
                record["predictions"] = prediction
                pooled_predictions[record["experiment_id"]].append(
                    prediction[common_mask]
                )

            labels = (
                tuple(
                    str(value)
                    for value in source.timestamps[
                        fold.test_start_index:fold.test_end_index
                    ]
                )
                if source.timestamps is not None
                else None
            )
            fold_payloads.append(
                {
                    **fold.as_dict(),
                    "scored_observations": scored,
                    "first_test_label": (
                        labels[0] if labels else str(fold.test_start_index + 1)
                    ),
                    "last_test_label": (
                        labels[-1] if labels else str(fold.test_end_index)
                    ),
                    "timestamps": labels,
                    "source_rows": tuple(
                        range(fold.test_start_index, fold.test_end_index)
                    ),
                    "observations": test_observations,
                    "score_mask": common_mask,
                    "models": model_records,
                }
            )

        pooled_observation_array = np.concatenate(pooled_observations)
        model_summaries: list[dict[str, Any]] = []
        metric_ids = tuple(
            item.metric_id
            for item in fold_payloads[0]["models"][0]["metrics"]
        )
        rank_accumulator: dict[str, dict[str, list[float]]] = {
            config.experiment_id: {metric_id: [] for metric_id in metric_ids}
            for config in configs
        }
        win_accumulator: dict[str, dict[str, int]] = {
            config.experiment_id: {metric_id: 0 for metric_id in metric_ids}
            for config in configs
        }

        for fold_payload in fold_payloads:
            for metric_id in metric_ids:
                candidates: list[tuple[float, str]] = []
                for model_record in fold_payload["models"]:
                    metric = next(
                        item
                        for item in model_record["metrics"]
                        if item.metric_id == metric_id
                    )
                    candidates.append(
                        (_metric_sort_value(metric), model_record["experiment_id"])
                    )
                candidates.sort(key=lambda item: item[0])
                for rank, (_, experiment_id) in enumerate(candidates, start=1):
                    rank_accumulator[experiment_id][metric_id].append(float(rank))
                if candidates and np.isfinite(candidates[0][0]):
                    best_value = candidates[0][0]
                    for value, experiment_id in candidates:
                        if np.isclose(value, best_value, rtol=1e-10, atol=1e-12):
                            win_accumulator[experiment_id][metric_id] += 1

        for config in configs:
            experiment_id = config.experiment_id
            predictions = np.concatenate(pooled_predictions[experiment_id])
            pooled_metrics = evaluate_metrics(
                pooled_observation_array,
                predictions,
                np.ones(len(predictions), dtype=bool),
            )
            fold_metric_summaries = []
            for template in pooled_metrics:
                values = []
                for fold_payload in fold_payloads:
                    record = next(
                        item
                        for item in fold_payload["models"]
                        if item["experiment_id"] == experiment_id
                    )
                    metric = next(
                        item
                        for item in record["metrics"]
                        if item.metric_id == template.metric_id
                    )
                    if metric.value is not None:
                        values.append(float(metric.value))
                fold_metric_summaries.append(
                    {
                        "metric_id": template.metric_id,
                        "symbol": template.symbol,
                        **_metric_summary(values),
                        "mean_rank": float(
                            np.mean(
                                rank_accumulator[experiment_id][
                                    template.metric_id
                                ]
                            )
                        ),
                        "best_fold_count": win_accumulator[experiment_id][
                            template.metric_id
                        ],
                    }
                )
            model_summaries.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_name": config.name,
                    "model_id": config.model.model_id,
                    "configuration_hash": configuration_hash(config),
                    "pooled_metrics": pooled_metrics,
                    "fold_metric_summaries": fold_metric_summaries,
                }
            )

        pairwise: list[dict[str, Any]] = []
        if len(configs) > 1:
            reference = configs[0]
            for candidate in configs[1:]:
                metric_payloads = []
                for metric_id in metric_ids:
                    improvements: list[float] = []
                    symbol = metric_id
                    for fold_payload in fold_payloads:
                        reference_record = next(
                            item
                            for item in fold_payload["models"]
                            if item["experiment_id"] == reference.experiment_id
                        )
                        candidate_record = next(
                            item
                            for item in fold_payload["models"]
                            if item["experiment_id"] == candidate.experiment_id
                        )
                        reference_metric = next(
                            item
                            for item in reference_record["metrics"]
                            if item.metric_id == metric_id
                        )
                        candidate_metric = next(
                            item
                            for item in candidate_record["metrics"]
                            if item.metric_id == metric_id
                        )
                        symbol = reference_metric.symbol
                        improvement = _positive_improvement(
                            reference_metric,
                            candidate_metric,
                        )
                        if improvement is not None:
                            improvements.append(improvement)
                    summary = _metric_summary(improvements)
                    metric_payloads.append(
                        {
                            "metric_id": metric_id,
                            "symbol": symbol,
                            "positive_means": "candidate_closer_to_ideal",
                            "candidate_better_fraction": (
                                float(
                                    np.mean(
                                        np.asarray(improvements, dtype=float) > 0.0
                                    )
                                )
                                if improvements
                                else None
                            ),
                            **summary,
                        }
                    )
                pairwise.append(
                    {
                        "reference_experiment_id": reference.experiment_id,
                        "reference_experiment_name": reference.name,
                        "candidate_experiment_id": candidate.experiment_id,
                        "candidate_experiment_name": candidate.name,
                        "metrics": metric_payloads,
                    }
                )

        warnings = [
            {
                "code": "temporal_validation_not_independent_folds",
                "message": (
                    "Rolling-origin folds are chronologically distinct test "
                    "blocks, but expanding training sets share historical rows. "
                    "Fold metrics are therefore repeated operational tests, not "
                    "independent replicates."
                ),
            },
            {
                "code": "temporal_validation_fixed_seed",
                "message": (
                    "Every fold retains the saved optimizer and initialization "
                    "seeds. The analysis isolates temporal-split sensitivity; "
                    "Initialization ensembles remain a separate source of "
                    "variability."
                ),
            },
            {
                "code": "temporal_validation_no_hyperparameter_search",
                "message": (
                    "Architectures, feature recipes, optimizer budgets, and other "
                    "hyperparameters are held fixed. Only model parameters or "
                    "weights are refitted inside each fold."
                ),
            },
        ]
        if any(isinstance(config.task, CalibrationTaskConfig) for config in configs):
            warnings.append(
                {
                    "code": "temporal_validation_conceptual_pretest_use",
                    "message": (
                        "Conceptual models use the complete pre-test information "
                        "set for parameter calibration because they have no "
                        "validation-selected checkpoint. Neural, PaML, and DD "
                        "models preserve their declared train/validation roles. "
                        "No model uses test targets."
                    ),
                }
            )
        if settings.window_mode == "sliding":
            warnings.append(
                {
                    "code": "temporal_validation_sliding_state_reset",
                    "message": (
                        "Each sliding fold restarts stateful models from their "
                        "configured initial state at the moving window boundary. "
                        "The saved warm-up must be long enough to make the scored "
                        "period insensitive to that reset."
                    ),
                }
            )

        emit(
            state=TaskState.COMPLETED,
            stage="temporal_validation",
            message=(
                "Every model was refitted on every chronological fold and all "
                "pooled metrics were recomputed from out-of-sample predictions."
            ),
        )
        return to_jsonable(
            {
                "task_id": task_id,
                "state": TaskState.COMPLETED.value,
                "method_id": TEMPORAL_VALIDATION_METHOD_ID,
                "settings": settings,
                "source": {
                    "sha256": source.report.sha256,
                    "rows": len(source.streamflow),
                    "timestamps_available": source.timestamps is not None,
                },
                "folds": fold_payloads,
                "pooled_scored_observations": len(pooled_observation_array),
                "models": model_summaries,
                "pairwise": pairwise,
                "warnings": warnings,
            }
        )


@dataclass
class _ManagedTemporalValidation:
    task_id: str
    experiment_id: str
    created_at: str
    updated_at: str
    state: TaskState
    cancellation_token: CancellationToken
    events: list[ProgressEvent] = field(default_factory=list)
    result: Mapping[str, Any] | None = None
    error: StructuredError | None = None
    future: Future | None = None


class TemporalValidationTaskManager:
    """Asynchronous temporal-validation manager with safe cancellation."""

    def __init__(
        self,
        validator: RollingOriginValidator,
        *,
        max_workers: int = 1,
        on_result: Callable[[str, Mapping[str, Any]], object] | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least one.")
        self._validator = validator
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="hydrosoftware-temporal-validation",
        )
        self._tasks: dict[str, _ManagedTemporalValidation] = {}
        self._lock = RLock()
        self._on_result = on_result

    def _record(self, task_id: str) -> _ManagedTemporalValidation:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise TaskNotFoundError(task_id) from exc

    @staticmethod
    def _snapshot(record: _ManagedTemporalValidation) -> TaskSnapshot:
        return TaskSnapshot(
            task_id=record.task_id,
            experiment_id=record.experiment_id,
            state=record.state,
            created_at=record.created_at,
            updated_at=record.updated_at,
            event_count=len(record.events),
            latest_event=record.events[-1],
            has_result=record.result is not None,
            error=record.error,
        )

    def _event(self, task_id: str, event: ProgressEvent) -> None:
        with self._lock:
            record = self._record(task_id)
            normalized = replace(
                event,
                task_id=task_id,
                sequence=len(record.events),
            )
            if (
                record.cancellation_token.is_requested
                and normalized.state in {TaskState.PREPARING, TaskState.RUNNING}
            ):
                normalized = replace(
                    normalized,
                    state=TaskState.CANCELLATION_REQUESTED,
                )
            record.events.append(normalized)
            record.state = normalized.state
            record.updated_at = _utc_now()
            if normalized.error is not None:
                record.error = normalized.error

    def submit(
        self,
        configs: Sequence[ExperimentConfigV1],
        settings: RollingOriginSettings,
    ) -> TaskSnapshot:
        task_id = f"temporal-validation-{uuid4()}"
        now = _utc_now()
        queued = ProgressEvent(
            task_id=task_id,
            sequence=0,
            state=TaskState.QUEUED,
            stage="queue",
            message="Temporal validation is waiting for an execution worker.",
        )
        record = _ManagedTemporalValidation(
            task_id=task_id,
            experiment_id=configs[0].experiment_id,
            created_at=now,
            updated_at=now,
            state=TaskState.QUEUED,
            cancellation_token=CancellationToken(),
            events=[queued],
        )
        with self._lock:
            self._tasks[task_id] = record
            record.future = self._executor.submit(
                self._execute,
                task_id,
                tuple(configs),
                settings,
            )
            return self._snapshot(record)

    def _execute(
        self,
        task_id: str,
        configs: tuple[ExperimentConfigV1, ...],
        settings: RollingOriginSettings,
    ) -> None:
        with self._lock:
            token = self._record(task_id).cancellation_token
        try:
            result = self._validator.run(
                configs,
                settings,
                task_id=task_id,
                cancellation_token=token,
                on_progress=lambda event: self._event(task_id, event),
            )
            archive_error: Exception | None = None
            if (
                result.get("state") == TaskState.COMPLETED.value
                and self._on_result is not None
            ):
                try:
                    self._on_result(task_id, result)
                except Exception as exc:  # pragma: no cover - platform I/O detail
                    archive_error = exc
            with self._lock:
                record = self._record(task_id)
                record.result = result
                record.state = TaskState(str(result["state"]))
                record.updated_at = _utc_now()
            if archive_error is not None:
                self._event(
                    task_id,
                    ProgressEvent(
                        task_id=task_id,
                        sequence=0,
                        state=TaskState.COMPLETED,
                        stage="persistence",
                        message=(
                            "Temporal validation completed, but its persistent "
                            "archive could not be written."
                        ),
                        severity=ProgressSeverity.WARNING,
                        error=StructuredError(
                            code="temporal_validation_archive_failed",
                            summary="The temporal-validation archive could not be saved.",
                            explanation=str(archive_error),
                            suggested_action=(
                                "Keep the service open and export the result before "
                                "closing the workspace."
                            ),
                        ),
                    ),
                )
        except TemporalValidationError as exc:
            self._fail(
                task_id,
                StructuredError(
                    code=exc.code,
                    summary="Temporal validation could not be completed.",
                    explanation=str(exc),
                    suggested_action=(
                        "Review the common source, fold lengths, and selected "
                        "experiment definitions."
                    ),
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive containment
            self._fail(
                task_id,
                StructuredError(
                    code="unexpected_temporal_validation_failure",
                    summary="Temporal validation stopped unexpectedly.",
                    explanation=str(exc),
                    suggested_action=(
                        "Inspect expert details and retry with a smaller protocol."
                    ),
                    technical_details=f"{type(exc).__name__}: {exc}",
                ),
            )

    def _fail(self, task_id: str, error: StructuredError) -> None:
        self._event(
            task_id,
            ProgressEvent(
                task_id=task_id,
                sequence=0,
                state=TaskState.FAILED,
                stage="temporal_validation",
                message=error.summary,
                severity=ProgressSeverity.ERROR,
                error=error,
            ),
        )

    def snapshot(self, task_id: str) -> TaskSnapshot:
        with self._lock:
            return self._snapshot(self._record(task_id))

    def events(
        self,
        task_id: str,
        *,
        after_sequence: int = -1,
    ) -> tuple[ProgressEvent, ...]:
        with self._lock:
            record = self._record(task_id)
            return tuple(
                event for event in record.events if event.sequence > after_sequence
            )

    def cancel(self, task_id: str) -> TaskSnapshot:
        with self._lock:
            record = self._record(task_id)
            if record.state in TERMINAL_STATES:
                return self._snapshot(record)
            record.cancellation_token.request()
        self._event(
            task_id,
            ProgressEvent(
                task_id=task_id,
                sequence=0,
                state=TaskState.CANCELLATION_REQUESTED,
                stage="cancellation",
                message=(
                    "Cancellation was requested and will occur at the next safe "
                    "fold, member, generation, or epoch boundary."
                ),
            ),
        )
        return self.snapshot(task_id)

    def result(self, task_id: str) -> Mapping[str, Any]:
        with self._lock:
            result = self._record(task_id).result
            if result is None:
                raise ResultNotReadyError(
                    "Temporal validation has not produced a result."
                )
            return result

    def wait(self, task_id: str, *, timeout: float | None = None) -> TaskSnapshot:
        with self._lock:
            future = self._record(task_id).future
        if future is not None:
            future.result(timeout=timeout)
        return self.snapshot(task_id)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)
