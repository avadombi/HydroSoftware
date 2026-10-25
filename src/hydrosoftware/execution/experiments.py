"""Configuration-driven execution of the first EXP-HYDRO vertical slice."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping

import numpy as np

from hydrosoftware.calibration import (
    CalibrationEngine,
    ConfiguredHydrologicObjective,
    DifferentialEvolutionSearchConfig,
    EvaluationRecord,
    UniformRandomSearchConfig,
    build_exp_hydro_calibration_problem,
    build_hbv_calibration_problem,
)
from hydrosoftware.calibration.definitions import CalibrationRunResult
from hydrosoftware.configuration import (
    CalibrationTaskConfig,
    Cnn1dModelConfig,
    DeepDiscoverModelConfig,
    DifferentialEvolutionSearchExperimentConfig,
    DegreeDaySnowConfig,
    DiscoveryTrainingTaskConfig,
    ExperimentConfigV1,
    GruModelConfig,
    LstmModelConfig,
    MlpModelConfig,
    NeuralTrainingTaskConfig,
    NoSnowConfig,
    PamlModelConfig,
    PamlTrainingTaskConfig,
    SmoothDegreeDaySnowConfig,
    TriangularMaxbasRoutingConfig,
    configuration_hash,
)
from hydrosoftware.core.specifications import ParameterSpec
from hydrosoftware.core.evaluation import EvaluationMetricResult, evaluate_metrics
from hydrosoftware.core.optimization import (
    objective_definition,
    objective_optimization_score,
    validate_objective_observations,
)
from hydrosoftware.core.features import (
    LaggedFeature,
    build_feature_mapping,
    feature_pipeline_mapping,
    maximum_lag,
)
from hydrosoftware.core.results import SimulationResult
from hydrosoftware.data_sources import (
    CsvCalibrationData,
    DataQualityReport,
    DataSourceError,
    analyze_calibration_csv,
    select_calibration_period,
)
from hydrosoftware.execution.ensembles import (
    aggregate_member_predictions,
    build_ensemble_artifact,
    derive_member_seeds,
)
from hydrosoftware.execution.serialization import to_jsonable
from hydrosoftware.execution.tasks import (
    CancellationToken,
    ProgressEvent,
    StructuredError,
    TaskState,
)
from hydrosoftware.processes.snow import (
    DegreeDaySnowModule,
    NoSnowModule,
    SmoothDegreeDaySnowModule,
    SnowModule,
)
from hydrosoftware.models.neural import (
    fit_cnn1d,
    fit_gru,
    fit_lstm,
    fit_mlp,
    fit_s2s,
    is_s2s_model_id,
)
from hydrosoftware.models.deepdiscover_numpy import fit_deepdiscover
from hydrosoftware.models.paml_tensorflow import fit_paml
from hydrosoftware.models.conceptual import (
    EXP_HYDRO_MANIFEST,
    HBV_MANIFEST,
)
from hydrosoftware.processes.routing import MAXBAS_STATE_NAMES


ProgressObserver = Callable[[ProgressEvent], None]


class _TrainingProgressClock:
    """Measure training time and estimate the remaining epoch budget."""

    def __init__(self, time_source: Callable[[], float] = monotonic) -> None:
        self._time_source = time_source
        self._started_at = time_source()

    def estimate(self, completed: int, total: int) -> tuple[float, float | None]:
        elapsed = max(self._time_source() - self._started_at, 0.0)
        if completed <= 0 or total <= 0:
            return elapsed, None
        if completed >= total:
            return elapsed, 0.0
        remaining = elapsed * (total - completed) / completed
        return elapsed, max(remaining, 0.0)

    def finished(self) -> tuple[float, float]:
        elapsed = max(self._time_source() - self._started_at, 0.0)
        return elapsed, 0.0


class ExperimentExecutionError(RuntimeError):
    """Execution failure already translated into a stable user-facing error."""

    def __init__(self, error: StructuredError) -> None:
        super().__init__(error.explanation)
        self.error = error


@dataclass(frozen=True)
class ExperimentValidationSummary:
    """Scientific and data checks completed before a task is submitted."""

    experiment_id: str
    configuration_hash: str
    source_rows: int
    selected_rows: int
    warmup_steps: int
    scored_observations: int
    transformed_value_count: int = 0
    data_quality_warning_count: int = 0
    primary_evaluation_partition: str = "calibration"
    partition_scored_observations: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[StructuredError, ...] = ()

    def as_dict(self) -> dict:
        return to_jsonable(self)


@dataclass(frozen=True)
class PreparedExperiment:
    """Validated arrays and components reused by execution after validation."""

    data: CsvCalibrationData
    summary: ExperimentValidationSummary
    snow_runtime: "SnowRuntimeDefinition"
    objective: ConfiguredHydrologicObjective
    evaluation_mask: np.ndarray
    partition_masks: Mapping[str, np.ndarray]
    primary_evaluation_partition: str
    data_quality: DataQualityReport


@dataclass(frozen=True)
class PreparedNeuralExperiment:
    """Validated continuous arrays and partitions for neural training."""

    analysis: Any
    summary: ExperimentValidationSummary
    data_quality: DataQualityReport


@dataclass(frozen=True)
class PreparedDiscoveryExperiment:
    """Validated continuous arrays and partitions for DD training."""

    analysis: Any
    summary: ExperimentValidationSummary
    data_quality: DataQualityReport
    snow_runtime: "SnowRuntimeDefinition"


@dataclass(frozen=True)
class ExperimentRunResult:
    """Terminal task result plus the best full scientific trajectory."""

    task_id: str
    state: TaskState
    experiment_id: str
    configuration_hash: str
    validation: ExperimentValidationSummary | None
    calibration: CalibrationRunResult | None
    best_simulation: SimulationResult | None
    timestamps: tuple[str, ...] | None
    observations: np.ndarray | None
    forcings: dict[str, np.ndarray] | None
    evaluation_metrics: tuple[EvaluationMetricResult, ...] = ()
    partition_metrics: Mapping[str, tuple[EvaluationMetricResult, ...]] = field(
        default_factory=dict
    )
    run_kind: str = "calibration"
    training: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.state not in {
            TaskState.COMPLETED,
            TaskState.CANCELLED,
            TaskState.FAILED,
        }:
            raise ValueError("ExperimentRunResult requires a terminal task state.")
        if (
            self.state is TaskState.COMPLETED
            and self.calibration is None
            and self.training is None
        ):
            raise ValueError(
                "A completed experiment must contain calibration or training results."
            )

    def as_dict(self) -> dict:
        return to_jsonable(self)


@dataclass(frozen=True)
class SnowRuntimeDefinition:
    """Runtime snow composition, including optional calibration dimensions."""

    default_module: SnowModule
    parameter_specs: tuple[ParameterSpec, ...]
    factory: Callable[[Mapping[str, float]], SnowModule]


def _snow_runtime(config: ExperimentConfigV1) -> SnowRuntimeDefinition:
    snow = config.model.snow
    if isinstance(snow, NoSnowConfig):
        module = NoSnowModule()
        return SnowRuntimeDefinition(module, (), lambda _parameters: module)

    def parameter_specs(names_and_units: tuple[tuple[str, str], ...]) -> tuple[ParameterSpec, ...]:
        specs: list[ParameterSpec] = []
        for name, unit in names_and_units:
            treatment = getattr(snow.calibration, name)
            if treatment.mode == "calibrate":
                specs.append(
                    ParameterSpec(
                        name=name,
                        lower_bound=treatment.lower_bound,
                        upper_bound=treatment.upper_bound,
                        default=float(getattr(snow, name)),
                        unit=unit,
                        description=f"Calibrated parameter of {snow.module_id}.",
                    )
                )
        return tuple(specs)

    def value(parameters: Mapping[str, float], name: str) -> float:
        return float(parameters.get(name, getattr(snow, name)))

    if isinstance(snow, DegreeDaySnowConfig):
        default_module = DegreeDaySnowModule(
            snowfall_temperature=snow.snowfall_temperature,
            melt_temperature=snow.melt_temperature,
            degree_day_factor=snow.degree_day_factor,
        )
        specs = parameter_specs((
            ("snowfall_temperature", "degC"),
            ("melt_temperature", "degC"),
            ("degree_day_factor", "mm degC^-1 day^-1"),
        ))

        def hard_factory(parameters: Mapping[str, float]) -> SnowModule:
            return DegreeDaySnowModule(
                snowfall_temperature=value(parameters, "snowfall_temperature"),
                melt_temperature=value(parameters, "melt_temperature"),
                degree_day_factor=value(parameters, "degree_day_factor"),
            )

        return SnowRuntimeDefinition(default_module, specs, hard_factory)
    if isinstance(snow, SmoothDegreeDaySnowConfig):
        default_module = SmoothDegreeDaySnowModule(
            snowfall_temperature=snow.snowfall_temperature,
            melt_temperature=snow.melt_temperature,
            degree_day_factor=snow.degree_day_factor,
            sharpness=snow.sharpness,
        )
        specs = parameter_specs((
            ("snowfall_temperature", "degC"),
            ("melt_temperature", "degC"),
            ("degree_day_factor", "mm degC^-1 day^-1"),
            ("sharpness", "1/degC"),
        ))

        def smooth_factory(parameters: Mapping[str, float]) -> SnowModule:
            return SmoothDegreeDaySnowModule(
                snowfall_temperature=value(parameters, "snowfall_temperature"),
                melt_temperature=value(parameters, "melt_temperature"),
                degree_day_factor=value(parameters, "degree_day_factor"),
                sharpness=value(parameters, "sharpness"),
            )

        return SnowRuntimeDefinition(default_module, specs, smooth_factory)
    raise TypeError(f"Unsupported snow configuration: {type(snow).__name__}")


def _objective(config: ExperimentConfigV1):
    return ConfiguredHydrologicObjective(config.task.objective.objective_id)


def _raw_predictors(analysis: Any) -> dict[str, np.ndarray]:
    """Return all user-addressable non-physical predictor channels."""

    return {
        "precipitation": analysis.precipitation,
        "temperature": analysis.temperature,
        "potential_evapotranspiration": analysis.potential_evapotranspiration,
        **{
            name: np.asarray(values, dtype=float)
            for name, values in analysis.additional_predictors.items()
        },
    }


def _lag_recipes(model: Any) -> tuple[LaggedFeature, ...]:
    return tuple(
        LaggedFeature(source=item.source, lag_steps=item.lag_steps)
        for item in model.lagged_features
    )


def _expanded_corrector(
    model: MlpModelConfig | LstmModelConfig | GruModelConfig | Cnn1dModelConfig,
) -> tuple[
    MlpModelConfig | LstmModelConfig | GruModelConfig | Cnn1dModelConfig,
    tuple[LaggedFeature, ...],
]:
    recipes = _lag_recipes(model)
    expanded_roles = tuple(
        [*model.feature_roles, *(item.name for item in recipes)]
    )
    return (
        model.model_copy(
            update={
                "feature_roles": expanded_roles,
                "lagged_features": (),
            }
        ),
        recipes,
    )


def _neural_fit_function(model: Any) -> Callable[..., Any]:
    """Resolve one neural family to its TensorFlow fitter."""

    if is_s2s_model_id(model.model_id):
        return fit_s2s
    if isinstance(model, LstmModelConfig):
        return fit_lstm
    if isinstance(model, GruModelConfig):
        return fit_gru
    if isinstance(model, Cnn1dModelConfig):
        return fit_cnn1d
    return fit_mlp


def _metric_value(
    metrics: tuple[EvaluationMetricResult, ...],
    metric_id: str,
) -> float | None:
    for metric in metrics:
        if metric.metric_id == metric_id:
            return metric.value
    return None


def _ensemble_partition_metrics(
    observations: np.ndarray,
    predictions: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, tuple[EvaluationMetricResult, ...]]:
    return {
        name: evaluate_metrics(observations, predictions, mask)
        for name, mask in masks.items()
    }


def _optimization_score(
    observations: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    objective_id: str,
) -> float:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(observations)
        & np.isfinite(predictions)
    )
    score, _ = objective_optimization_score(
        observations[valid],
        predictions[valid],
        objective_id,
    )
    return score


class ExperimentRunner:
    """Validate and execute versioned experiments without an HTTP dependency."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    @staticmethod
    def _preparation_error(exc: Exception) -> ExperimentExecutionError:
        if isinstance(exc, DataSourceError):
            code = exc.code
            summary = "The experiment data could not be prepared."
            action = "Correct the CSV source, column mapping, or selected period and retry."
        else:
            code = "invalid_experiment_inputs"
            summary = "The experiment inputs are not scientifically valid."
            action = "Review the objective, warm-up, model settings, and observations."
        return ExperimentExecutionError(
            StructuredError(
                code=code,
                summary=summary,
                explanation=str(exc) or summary,
                suggested_action=action,
                technical_details=f"{type(exc).__name__}: {exc}",
            )
        )

    def prepare(
        self, config: ExperimentConfigV1
    ) -> PreparedExperiment | PreparedNeuralExperiment | PreparedDiscoveryExperiment:
        """Load data and complete all inexpensive validation checks."""

        if isinstance(config.task, DiscoveryTrainingTaskConfig):
            return self._prepare_discovery(config)
        if isinstance(
            config.task, (NeuralTrainingTaskConfig, PamlTrainingTaskConfig)
        ):
            return self._prepare_neural(config)

        try:
            analysis = analyze_calibration_csv(
                config.data,
                workspace_root=self.workspace_root,
            )
            if not analysis.report.ready:
                first_error = next(
                    issue
                    for issue in analysis.report.issues
                    if issue.severity == "error"
                )
                raise DataSourceError(first_error.code, first_error.explanation)
            source_data = CsvCalibrationData(
                precipitation=analysis.precipitation,
                temperature=analysis.temperature,
                potential_evapotranspiration=(
                    analysis.potential_evapotranspiration
                ),
                streamflow=analysis.streamflow,
                timestamps=analysis.timestamps,
                source_path=analysis.source_path,
            )
            selected = select_calibration_period(source_data, config.task.period)
            period = config.task.period
            calibration_end_absolute = (
                period.calibration_end_index
                if period.calibration_end_index is not None
                else period.start_index + selected.n_time
            )
            calibration_end = calibration_end_absolute - period.start_index
            calibration_mask = np.zeros(selected.n_time, dtype=bool)
            calibration_mask[period.warmup_steps:calibration_end] = True
            partition_masks: dict[str, np.ndarray] = {
                "calibration": calibration_mask
            }
            if period.protocol == "calibration_validation_test":
                assert period.validation_end_index is not None
                validation_end = (
                    period.validation_end_index - period.start_index
                )
                validation_mask = np.zeros(selected.n_time, dtype=bool)
                validation_mask[calibration_end:validation_end] = True
                test_mask = np.zeros(selected.n_time, dtype=bool)
                test_mask[validation_end:] = True
                partition_masks.update(
                    {
                        "validation": validation_mask,
                        "test": test_mask,
                    }
                )
                primary_partition = "test"
            elif period.protocol == "calibration_test":
                test_mask = np.zeros(selected.n_time, dtype=bool)
                test_mask[calibration_end:] = True
                partition_masks["test"] = test_mask
                primary_partition = "test"
            else:
                primary_partition = "calibration"
            objective = _objective(config)
            objective.validate_observations(
                selected.streamflow, calibration_mask
            )
            for name, partition_mask in partition_masks.items():
                if not np.any(partition_mask & np.isfinite(selected.streamflow)):
                    raise ValueError(
                        f"The conceptual {name} partition contains no finite "
                        "streamflow observation."
                    )
            snow_runtime = _snow_runtime(config)
        except Exception as exc:
            if isinstance(exc, ExperimentExecutionError):
                raise
            raise self._preparation_error(exc) from exc

        warnings = [
            StructuredError(
                code=issue.code,
                summary=issue.summary,
                explanation=issue.explanation,
                suggested_action=issue.suggested_action,
                context={
                    "variable": issue.variable,
                    "count": issue.count,
                    "csv_rows": list(issue.csv_rows),
                },
            )
            for issue in analysis.report.issues
            if issue.severity == "warning"
        ]
        if config.task.period.protocol == "legacy_calibration_only":
            warnings.append(
                StructuredError(
                    code="legacy_calibration_only_protocol",
                    summary="This legacy conceptual experiment has no holdout test.",
                    explanation=(
                        "Its reported performance is an in-sample calibration "
                        "score, not an estimate of generalization."
                    ),
                    suggested_action=(
                        "Create a calibration-test or "
                        "calibration-validation-test experiment."
                    ),
                )
            )
        partition_counts = {
            name: int(
                np.count_nonzero(mask & np.isfinite(selected.streamflow))
            )
            for name, mask in partition_masks.items()
        }
        summary = ExperimentValidationSummary(
            experiment_id=config.experiment_id,
            configuration_hash=configuration_hash(config),
            source_rows=source_data.n_time,
            selected_rows=selected.n_time,
            warmup_steps=config.task.period.warmup_steps,
            scored_observations=partition_counts[primary_partition],
            transformed_value_count=analysis.report.transformed_value_count,
            data_quality_warning_count=analysis.report.warning_count,
            primary_evaluation_partition=primary_partition,
            partition_scored_observations=partition_counts,
            warnings=tuple(warnings),
        )
        return PreparedExperiment(
            data=selected,
            summary=summary,
            snow_runtime=snow_runtime,
            objective=objective,
            evaluation_mask=calibration_mask,
            partition_masks=partition_masks,
            primary_evaluation_partition=primary_partition,
            data_quality=analysis.report,
        )

    def _prepare_neural(self, config: ExperimentConfigV1) -> PreparedNeuralExperiment:
        """Validate data and temporal partitions for neural or PaML training."""

        standalone = isinstance(
            config.model,
            (MlpModelConfig, LstmModelConfig, GruModelConfig, Cnn1dModelConfig),
        ) and isinstance(config.task, NeuralTrainingTaskConfig)
        paml = isinstance(config.model, PamlModelConfig) and isinstance(
            config.task, PamlTrainingTaskConfig
        )
        if not (standalone or paml):
            raise ValueError("Neural preparation requires a supported training setup.")
        task = config.task
        try:
            analysis = analyze_calibration_csv(
                config.data,
                workspace_root=self.workspace_root,
            )
            if not analysis.report.ready:
                first_error = next(
                    issue for issue in analysis.report.issues if issue.severity == "error"
                )
                raise DataSourceError(first_error.code, first_error.explanation)
            end = task.period.test_end_index or len(analysis.streamflow)
            if end > len(analysis.streamflow):
                raise ValueError("The test period exceeds the available data rows.")

            if paml:
                warmup = task.warmup_steps
                required_sources = {
                    "precipitation",
                    "temperature",
                    "potential_evapotranspiration",
                }
            else:
                recipes = _lag_recipes(config.model)
                neural_warmup = (
                    config.model.s2s_warmup
                    if is_s2s_model_id(config.model.model_id)
                    else config.model.lookback
                )
                warmup = neural_warmup + maximum_lag(recipes)
                required_sources = {
                    *config.model.feature_roles,
                    *(item.source for item in recipes),
                }

            if task.period.train_end_index - task.period.start_index <= warmup:
                raise ValueError(
                    "The training period must be longer than the declared warm-up."
                )
            available_external = set(_raw_predictors(analysis))
            missing_external = sorted(
                name for name in required_sources if name not in available_external
            )
            if missing_external:
                raise ValueError(
                    "Selected predictor channels are not mapped in the data source: "
                    + ", ".join(missing_external)
                )
            partitions = {
                "training": (
                    task.period.start_index + warmup,
                    task.period.train_end_index,
                ),
                "validation": (
                    task.period.train_end_index,
                    task.period.validation_end_index,
                ),
                "test": (task.period.validation_end_index, end),
            }
            for name, (start, stop) in partitions.items():
                valid = np.isfinite(analysis.streamflow[start:stop])
                if not np.any(valid):
                    raise ValueError(
                        f"The {name} partition contains no finite streamflow target."
                    )
                if name in {"training", "validation"}:
                    validate_objective_observations(
                        analysis.streamflow[start:stop][valid],
                        task.objective.objective_id,
                    )
            if (
                standalone
                and config.model.output_activation == "softplus"
                and np.any(
                    analysis.streamflow[
                        task.period.start_index:task.period.train_end_index
                    ] < 0.0
                )
            ):
                raise ValueError(
                    "Softplus output requires non-negative training streamflow."
                )
        except Exception as exc:
            raise self._preparation_error(exc) from exc

        warnings = tuple(
            StructuredError(
                code=issue.code,
                summary=issue.summary,
                explanation=issue.explanation,
                suggested_action=issue.suggested_action,
                context={
                    "variable": issue.variable,
                    "count": issue.count,
                    "csv_rows": list(issue.csv_rows),
                },
            )
            for issue in analysis.report.issues
            if issue.severity == "warning"
        )
        test_start = task.period.validation_end_index
        scored = int(np.count_nonzero(np.isfinite(analysis.streamflow[test_start:end])))
        partition_counts = {
            name: int(np.count_nonzero(np.isfinite(analysis.streamflow[start:stop])))
            for name, (start, stop) in partitions.items()
        }
        summary = ExperimentValidationSummary(
            experiment_id=config.experiment_id,
            configuration_hash=configuration_hash(config),
            source_rows=len(analysis.streamflow),
            selected_rows=end - task.period.start_index,
            warmup_steps=warmup,
            scored_observations=scored,
            transformed_value_count=analysis.report.transformed_value_count,
            data_quality_warning_count=analysis.report.warning_count,
            primary_evaluation_partition="test",
            partition_scored_observations=partition_counts,
            warnings=warnings,
        )
        return PreparedNeuralExperiment(
            analysis=analysis,
            summary=summary,
            data_quality=analysis.report,
        )

    def _prepare_discovery(
        self, config: ExperimentConfigV1
    ) -> PreparedDiscoveryExperiment:
        if not isinstance(
            config.model,
            DeepDiscoverModelConfig,
        ) or not isinstance(config.task, DiscoveryTrainingTaskConfig):
            raise ValueError(
                "DeepDiscover preparation requires its discovery-training setup."
            )
        try:
            analysis = analyze_calibration_csv(
                config.data,
                workspace_root=self.workspace_root,
            )
            if not analysis.report.ready:
                first_error = next(
                    issue
                    for issue in analysis.report.issues
                    if issue.severity == "error"
                )
                raise DataSourceError(first_error.code, first_error.explanation)
            period = config.task.period
            end = period.test_end_index or len(analysis.streamflow)
            if end > len(analysis.streamflow):
                raise ValueError(
                    "The DeepDiscover test period exceeds the available rows."
                )
            if np.any(
                analysis.precipitation[period.start_index:end] < 0.0
            ) or np.any(
                analysis.potential_evapotranspiration[
                    period.start_index:end
                ] < 0.0
            ):
                raise ValueError(
                    "DeepDiscover precipitation and PET must be non-negative "
                    "after declared data-quality rules."
                )
            partitions = {
                "training": (
                    period.start_index + config.task.warmup_steps,
                    period.train_end_index,
                ),
                "validation": (
                    period.train_end_index,
                    period.validation_end_index,
                ),
                "test": (period.validation_end_index, end),
            }
            for name, (start, stop) in partitions.items():
                valid = np.isfinite(analysis.streamflow[start:stop])
                if not np.any(valid):
                    raise ValueError(
                        f"The DeepDiscover {name} partition has no finite target."
                    )
                if name in {"training", "validation"}:
                    validate_objective_observations(
                        analysis.streamflow[start:stop][valid],
                        config.task.objective.objective_id,
                    )
            snow_runtime = _snow_runtime(config)
        except Exception as exc:
            raise self._preparation_error(exc) from exc

        warnings = tuple(
            StructuredError(
                code=issue.code,
                summary=issue.summary,
                explanation=issue.explanation,
                suggested_action=issue.suggested_action,
                context={
                    "variable": issue.variable,
                    "count": issue.count,
                    "csv_rows": list(issue.csv_rows),
                },
            )
            for issue in analysis.report.issues
            if issue.severity == "warning"
        )
        scored = int(
            np.count_nonzero(
                np.isfinite(
                    analysis.streamflow[period.validation_end_index:end]
                )
            )
        )
        partition_counts = {
            name: int(
                np.count_nonzero(
                    np.isfinite(analysis.streamflow[start:stop])
                )
            )
            for name, (start, stop) in partitions.items()
        }
        summary = ExperimentValidationSummary(
            experiment_id=config.experiment_id,
            configuration_hash=configuration_hash(config),
            source_rows=len(analysis.streamflow),
            selected_rows=end - period.start_index,
            warmup_steps=config.task.warmup_steps,
            scored_observations=scored,
            transformed_value_count=analysis.report.transformed_value_count,
            data_quality_warning_count=analysis.report.warning_count,
            primary_evaluation_partition="test",
            partition_scored_observations=partition_counts,
            warnings=warnings,
        )
        return PreparedDiscoveryExperiment(
            analysis=analysis,
            summary=summary,
            data_quality=analysis.report,
            snow_runtime=snow_runtime,
        )

    def validate(self, config: ExperimentConfigV1) -> ExperimentValidationSummary:
        """Return the preflight summary without starting calibration."""

        return self.prepare(config).summary

    def run(
        self,
        config: ExperimentConfigV1,
        *,
        task_id: str,
        cancellation_token: CancellationToken | None = None,
        on_progress: ProgressObserver | None = None,
        calibration_workers: int = 1,
    ) -> ExperimentRunResult:
        """Execute a calibration and regenerate the best scientific trajectory."""

        token = cancellation_token or CancellationToken()

        def emit(state: TaskState, stage: str, message: str) -> None:
            if on_progress is not None:
                on_progress(
                    ProgressEvent(
                        task_id=task_id,
                        sequence=0,
                        state=state,
                        stage=stage,
                        message=message,
                    )
                )

        if token.is_requested:
            emit(TaskState.CANCELLED, "cancellation", "The queued experiment was cancelled.")
            return ExperimentRunResult(
                task_id=task_id,
                state=TaskState.CANCELLED,
                experiment_id=config.experiment_id,
                configuration_hash=configuration_hash(config),
                validation=None,
                calibration=None,
                best_simulation=None,
                timestamps=None,
                observations=None,
                forcings=None,
            )

        emit(TaskState.PREPARING, "data_loading", "Loading and validating experiment data.")
        prepared = self.prepare(config)

        if isinstance(prepared, PreparedDiscoveryExperiment):
            return self._run_discovery(
                config,
                prepared,
                task_id=task_id,
                token=token,
                on_progress=on_progress,
            )
        if isinstance(prepared, PreparedNeuralExperiment):
            if isinstance(config.model, PamlModelConfig):
                return self._run_paml(
                    config,
                    prepared,
                    task_id=task_id,
                    token=token,
                    on_progress=on_progress,
                )
            return self._run_neural(
                config,
                prepared,
                task_id=task_id,
                token=token,
                on_progress=on_progress,
            )

        if token.is_requested:
            emit(
                TaskState.CANCELLED,
                "cancellation",
                "The experiment was cancelled after data validation.",
            )
            return ExperimentRunResult(
                task_id=task_id,
                state=TaskState.CANCELLED,
                experiment_id=config.experiment_id,
                configuration_hash=prepared.summary.configuration_hash,
                validation=prepared.summary,
                calibration=None,
                best_simulation=None,
                timestamps=prepared.data.timestamps,
                observations=prepared.data.streamflow,
                forcings={
                    "precipitation": prepared.data.precipitation,
                    "temperature": prepared.data.temperature,
                    "potential_evapotranspiration": (
                        prepared.data.potential_evapotranspiration
                    ),
                },
            )

        def build_problem(stop: int | None = None):
            selected = slice(None, stop)
            forcing = (
                prepared.data.precipitation[selected],
                prepared.data.temperature[selected],
                prepared.data.potential_evapotranspiration[selected],
                prepared.data.streamflow[selected],
                prepared.snow_runtime.default_module,
            )
            common = {
                "evaluation_mask": prepared.evaluation_mask[selected],
                "initial_state": config.model.initial_state,
                "objective": prepared.objective,
                "snow_parameter_specs": prepared.snow_runtime.parameter_specs,
                "snow_module_factory": prepared.snow_runtime.factory,
            }
            if config.model.model_id == "exp_hydro.mass_conservative.v1":
                return build_exp_hydro_calibration_problem(
                    *forcing, **common, substeps=config.model.substeps
                )
            if config.model.model_id == "hbv.mass_conservative_daily.v1":
                routing = config.model.routing
                routing_kwargs: dict = {
                    "routing_process_id": routing.process_id,
                }
                if isinstance(routing, TriangularMaxbasRoutingConfig):
                    treatment = routing.calibration.maxbas
                    routing_kwargs.update(
                        {
                            "maxbas": routing.maxbas,
                            "initial_routing_state": routing.initial_state,
                            "maxbas_parameter_spec": (
                                ParameterSpec(
                                    name="maxbas",
                                    lower_bound=treatment.lower_bound,
                                    upper_bound=treatment.upper_bound,
                                    default=float(routing.maxbas),
                                    unit="day",
                                    description=(
                                        "Integer base length of the triangular "
                                        "HBV routing function."
                                    ),
                                    value_type="integer",
                                )
                                if treatment.mode == "calibrate"
                                else None
                            ),
                        }
                    )
                return build_hbv_calibration_problem(
                    *forcing, **common, **routing_kwargs
                )
            raise ValueError(
                f"Unsupported conceptual model: {config.model.model_id}"
            )

        calibration_positions = np.flatnonzero(prepared.evaluation_mask)
        calibration_stop = int(calibration_positions[-1]) + 1
        problem = build_problem(calibration_stop)
        search = config.task.search
        runtime_search = (
            DifferentialEvolutionSearchConfig(
                generations=search.generations,
                population_size=search.population_size,
                mutation_factor=search.mutation_factor,
                crossover_probability=search.crossover_probability,
                seed=search.seed,
                include_defaults=search.include_defaults,
                continue_on_candidate_error=search.continue_on_candidate_error,
            )
            if isinstance(search, DifferentialEvolutionSearchExperimentConfig)
            else UniformRandomSearchConfig(
                n_evaluations=search.n_evaluations,
                seed=search.seed,
                include_defaults=search.include_defaults,
                continue_on_candidate_error=search.continue_on_candidate_error,
            )
        )
        calibration = CalibrationEngine().run(
            problem,
            runtime_search,
            task_id=task_id,
            cancellation_token=token,
            on_progress=on_progress,
            max_workers=calibration_workers,
        )

        best_simulation = None
        evaluation_metrics: tuple[EvaluationMetricResult, ...] = ()
        partition_metrics: dict[
            str, tuple[EvaluationMetricResult, ...]
        ] = {}
        if calibration.best_parameters is not None:
            full_problem = build_problem()
            best_simulation = full_problem.simulate(calibration.best_parameters)
            partition_metrics = {
                name: evaluate_metrics(
                    prepared.data.streamflow,
                    best_simulation.outputs[:, 0],
                    mask,
                )
                for name, mask in prepared.partition_masks.items()
            }
            evaluation_metrics = partition_metrics[
                prepared.primary_evaluation_partition
            ]

        return ExperimentRunResult(
            task_id=task_id,
            state=calibration.state,
            experiment_id=config.experiment_id,
            configuration_hash=prepared.summary.configuration_hash,
            validation=prepared.summary,
            calibration=calibration,
            best_simulation=best_simulation,
            timestamps=prepared.data.timestamps,
            observations=prepared.data.streamflow,
            forcings={
                "precipitation": prepared.data.precipitation,
                "temperature": prepared.data.temperature,
                "potential_evapotranspiration": (
                    prepared.data.potential_evapotranspiration
                ),
            },
            evaluation_metrics=evaluation_metrics,
            partition_metrics=partition_metrics,
        )

    def _run_neural(
        self,
        config: ExperimentConfigV1,
        prepared: PreparedNeuralExperiment,
        *,
        task_id: str,
        token: CancellationToken,
        on_progress: ProgressObserver | None,
    ) -> ExperimentRunResult:
        """Fit and archive a leakage-safe neural or initialization ensemble."""

        if not isinstance(
            config.model,
            (MlpModelConfig, LstmModelConfig, GruModelConfig, Cnn1dModelConfig),
        ) or not isinstance(
            config.task, NeuralTrainingTaskConfig
        ):
            raise ValueError("The neural runner received an incompatible configuration.")
        analysis = prepared.analysis
        period = config.task.period
        end = period.test_end_index or len(analysis.streamflow)
        raw_features = _raw_predictors(analysis)
        runtime_model, recipes = _expanded_corrector(config.model)
        features = build_feature_mapping(
            raw_features,
            config.model.feature_roles,
            recipes,
        )
        effective_start = max(period.start_index, maximum_lag(recipes))

        if token.is_requested:
            return ExperimentRunResult(
                task_id=task_id,
                state=TaskState.CANCELLED,
                experiment_id=config.experiment_id,
                configuration_hash=prepared.summary.configuration_hash,
                validation=prepared.summary,
                calibration=None,
                best_simulation=None,
                timestamps=None,
                observations=None,
                forcings=None,
                run_kind="training",
            )

        ensemble = config.task.ensemble
        member_count = ensemble.members if ensemble.enabled else 1
        member_seeds = (
            derive_member_seeds(config.task.training.seed, member_count)
            if ensemble.enabled
            else (config.task.training.seed,)
        )
        fit_function = _neural_fit_function(runtime_model)
        training_objective = objective_definition(
            config.task.objective.objective_id
        )
        fits: list[Any] = []
        total_epoch_budget = member_count * config.task.training.epochs
        progress_clock = _TrainingProgressClock()

        for member_position, member_seed in enumerate(member_seeds):
            if token.is_requested and fits:
                break

            def epoch_progress(
                epoch: int,
                total: int,
                best_score: float,
                *,
                _member_position: int = member_position,
            ) -> None:
                if on_progress is None:
                    return
                completed = _member_position * config.task.training.epochs + epoch
                elapsed, remaining = progress_clock.estimate(
                    completed, total_epoch_budget
                )
                on_progress(
                    ProgressEvent(
                        task_id=task_id,
                        sequence=0,
                        state=TaskState.RUNNING,
                        stage="neural_initialization_ensemble",
                        message=(
                            (
                                f"Training member {_member_position + 1} of "
                                f"{member_count}, epoch {epoch} of {total}; "
                            )
                            if ensemble.enabled
                            else f"Training epoch {epoch} of {total}; "
                        )
                        + (
                            f"best member validation "
                            f"{training_objective.symbol} {best_score:.6g}."
                        ),
                        completed_units=completed,
                        total_units=total_epoch_budget,
                        fraction=completed / total_epoch_budget,
                        elapsed_seconds=elapsed,
                        estimated_remaining_seconds=remaining,
                        best_score=best_score,
                    )
                )

            fit = fit_function(
                runtime_model,
                config.task.training.model_copy(update={"seed": member_seed}),
                features,
                analysis.streamflow,
                start_index=effective_start,
                train_end_index=period.train_end_index,
                validation_end_index=period.validation_end_index,
                test_end_index=end,
                objective_id=config.task.objective.objective_id,
                stride=config.task.stride,
                cancellation_requested=lambda: token.is_requested,
                on_epoch=epoch_progress,
            )
            fits.append(fit)
            if fit.cancelled:
                break

        ensemble_predictions, ensemble_lower, ensemble_upper = (
            aggregate_member_predictions(
                [fit.predictions for fit in fits],
                lower_quantile=ensemble.lower_quantile,
                upper_quantile=ensemble.upper_quantile,
            )
        )
        start = period.start_index
        selected_predictions = ensemble_predictions[start:end]
        selected_observations = analysis.streamflow[start:end]
        partition_metrics = _ensemble_partition_metrics(
            analysis.streamflow,
            ensemble_predictions,
            fits[0].partition_masks,
        )
        validation_score = _optimization_score(
            analysis.streamflow,
            ensemble_predictions,
            fits[0].partition_masks["validation"],
            config.task.objective.objective_id,
        )
        member_records = []
        for member_position, (member_seed, fit) in enumerate(
            zip(member_seeds, fits)
        ):
            member_metrics = _ensemble_partition_metrics(
                analysis.streamflow,
                fit.predictions,
                fit.partition_masks,
            )
            member_records.append(
                {
                    "member": member_position + 1,
                    "seed": member_seed,
                    "best_epoch": fit.artifact.best_epoch,
                    "epochs_completed": len(fit.history),
                    "stopped_early": fit.stopped_early,
                    "cancelled": fit.cancelled,
                    "best_validation_rmse": _metric_value(
                        member_metrics["validation"],
                        "root_mean_square_error",
                    ),
                    "best_validation_score": _optimization_score(
                        analysis.streamflow,
                        fit.predictions,
                        fit.partition_masks["validation"],
                        config.task.objective.objective_id,
                    ),
                    "partition_metrics": member_metrics,
                }
            )
        representative_position = min(
            range(len(fits)),
            key=lambda position: (
                (
                    -member_records[position]["best_validation_score"]
                    if training_objective.direction == "maximize"
                    else member_records[position]["best_validation_score"]
                )
                if member_records[position]["best_validation_score"] is not None
                else float("inf")
            ),
        )
        representative = fits[representative_position]
        artifact_payload = (
            build_ensemble_artifact(
                [fit.artifact.as_mapping() for fit in fits],
                member_seeds[:len(fits)],
                lower_quantile=ensemble.lower_quantile,
                upper_quantile=ensemble.upper_quantile,
            )
            if ensemble.enabled
            else representative.artifact.as_mapping()
        )
        result_engine_id = (
            "neural_initialization_ensemble.v1"
            if ensemble.enabled
            else representative.artifact.engine_id
        )
        diagnostics: dict[str, Any] = {
            "engine_id": result_engine_id,
            "model_id": config.model.model_id,
            "prediction_layout": (
                "sequence_to_sequence"
                if is_s2s_model_id(config.model.model_id)
                else "sequence_to_value"
            ),
            "lookback": config.model.lookback,
            "s2s_warmup": (
                config.model.s2s_warmup
                if is_s2s_model_id(config.model.model_id)
                else None
            ),
            "s2s_final_block_context_length": (
                config.model.s2s_warmup
                + (
                    (end - period.validation_end_index)
                    % (config.model.lookback - config.model.s2s_warmup)
                )
                if is_s2s_model_id(config.model.model_id)
                else None
            ),
            "train_end_index": period.train_end_index - start,
            "validation_end_index": period.validation_end_index - start,
            "test_end_index": end - start,
            **(
                {
                    "initialization_ensemble_method": (
                        "neural_initialization_ensemble.v1"
                    ),
                    "initialization_ensemble_members": len(fits),
                    "initialization_ensemble_lower_quantile": (
                        ensemble.lower_quantile
                    ),
                    "initialization_ensemble_upper_quantile": (
                        ensemble.upper_quantile
                    ),
                    "initialization_ensemble_lower": ensemble_lower[start:end],
                    "initialization_ensemble_upper": ensemble_upper[start:end],
                }
                if ensemble.enabled
                else {}
            ),
            **(
                {
                    "convolutional_layers": len(
                        config.model.convolution_layers
                    ),
                    "receptive_field": 1
                    + sum(
                        layer.kernel_size - 1
                        for layer in config.model.convolution_layers
                    ),
                }
                if isinstance(config.model, Cnn1dModelConfig)
                else {}
            ),
        }
        simulation = SimulationResult(
            outputs=selected_predictions[:, None],
            states=np.empty((len(selected_predictions) + 1, 0), dtype=float),
            fluxes=np.empty((len(selected_predictions), 0), dtype=float),
            output_names=("streamflow",),
            state_names=(),
            flux_names=(),
            diagnostics=diagnostics,
        )
        training_result = {
            "engine_id": result_engine_id,
            "model_id": config.model.model_id,
            "artifact": artifact_payload,
            "history": representative.history,
            "best_epoch": representative.artifact.best_epoch,
            "epochs_completed": sum(len(fit.history) for fit in fits),
            "stopped_early": any(fit.stopped_early for fit in fits),
            "selection_metric": (
                f"member_validation_{training_objective.objective_id}"
                "_then_arithmetic_mean"
                if ensemble.enabled
                else f"validation_{training_objective.objective_id}"
            ),
            "best_validation_score": validation_score,
            "objective": {
                "objective_id": training_objective.objective_id,
                "symbol": training_objective.symbol,
                "name": training_objective.name,
                "direction": training_objective.direction,
                "ideal_value": training_objective.ideal_value,
                "unit": training_objective.unit,
                "requires_global_batch": (
                    training_objective.requires_global_batch
                ),
            },
            "feature_roles": representative.artifact.feature_roles,
            "base_feature_roles": config.model.feature_roles,
            "feature_pipeline": feature_pipeline_mapping(
                config.model.feature_roles, recipes
            ),
            "lookback": config.model.lookback,
            "s2s_warmup": (
                config.model.s2s_warmup
                if is_s2s_model_id(config.model.model_id)
                else None
            ),
            "recurrent_units": (
                config.model.recurrent_units
                if isinstance(config.model, (LstmModelConfig, GruModelConfig))
                else None
            ),
            "recurrent_layers": (
                tuple(config.model.recurrent_layers)
                if isinstance(config.model, (LstmModelConfig, GruModelConfig))
                else None
            ),
            "convolution_layers": (
                tuple(
                    {
                        "filters": layer.filters,
                        "kernel_size": layer.kernel_size,
                        "activation": layer.activation,
                    }
                    for layer in config.model.convolution_layers
                )
                if isinstance(config.model, Cnn1dModelConfig)
                else None
            ),
            "partition_metrics": partition_metrics,
            "initialization_ensemble": (
                {
                    "method_id": "neural_initialization_ensemble.v1",
                    "member_count_requested": member_count,
                    "member_count_completed": len(fits),
                    "member_seeds": list(member_seeds[:len(fits)]),
                    "aggregation": "arithmetic_mean",
                    "lower_quantile": ensemble.lower_quantile,
                    "upper_quantile": ensemble.upper_quantile,
                    "representative_member": representative_position + 1,
                    "members": member_records,
                    "scope": "neural_weight_initialization",
                }
                if ensemble.enabled
                else None
            ),
        }
        state = (
            TaskState.CANCELLED
            if len(fits) < member_count or any(fit.cancelled for fit in fits)
            else TaskState.COMPLETED
        )
        if on_progress is not None:
            elapsed, remaining = progress_clock.finished()
            on_progress(
                ProgressEvent(
                    task_id=task_id,
                    sequence=0,
                    state=state,
                    stage="neural_initialization_ensemble",
                    message=(
                        "Neural training was cancelled at a safe member or epoch boundary."
                        if state is TaskState.CANCELLED
                        else (
                            f"{len(fits)} initialization members were restored "
                            "and aggregated on the common evaluation support."
                            if ensemble.enabled
                            else "The best validation checkpoint was restored and evaluated."
                        )
                    ),
                    completed_units=sum(len(fit.history) for fit in fits),
                    total_units=total_epoch_budget,
                    fraction=sum(len(fit.history) for fit in fits) / total_epoch_budget,
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=remaining,
                    best_score=training_result["best_validation_score"],
                )
            )
        return ExperimentRunResult(
            task_id=task_id,
            state=state,
            experiment_id=config.experiment_id,
            configuration_hash=prepared.summary.configuration_hash,
            validation=prepared.summary,
            calibration=None,
            best_simulation=simulation,
            timestamps=(
                analysis.timestamps[start:end]
                if analysis.timestamps is not None
                else None
            ),
            observations=selected_observations,
            forcings={
                name: values[start:end] for name, values in raw_features.items()
            },
            evaluation_metrics=partition_metrics["test"],
            partition_metrics=partition_metrics,
            run_kind="training",
            training=training_result,
        )

    def _run_discovery(
        self,
        config: ExperimentConfigV1,
        prepared: PreparedDiscoveryExperiment,
        *,
        task_id: str,
        token: CancellationToken,
        on_progress: ProgressObserver | None,
    ) -> ExperimentRunResult:
        """Train instantaneous DD and archive every discovered trajectory."""

        if not isinstance(config.model, DeepDiscoverModelConfig) or not isinstance(
            config.task, DiscoveryTrainingTaskConfig
        ):
            raise ValueError(
                "The DeepDiscover runner received an incompatible configuration."
            )
        analysis = prepared.analysis
        period = config.task.period
        end = period.test_end_index or len(analysis.streamflow)
        training_objective = objective_definition(
            config.task.objective.objective_id
        )
        progress_clock = _TrainingProgressClock()

        def epoch_progress(epoch: int, total: int, best_score: float) -> None:
            if on_progress is not None:
                elapsed, remaining = progress_clock.estimate(epoch, total)
                on_progress(
                    ProgressEvent(
                        task_id=task_id,
                        sequence=0,
                        state=TaskState.RUNNING,
                        stage="deepdiscover_training",
                        message=(
                            f"Training DeepDiscover epoch {epoch} of {total}; "
                            f"best validation {training_objective.symbol} "
                            f"{best_score:.6g}."
                        ),
                        completed_units=epoch,
                        total_units=total,
                        fraction=epoch / total,
                        elapsed_seconds=elapsed,
                        estimated_remaining_seconds=remaining,
                        best_score=best_score,
                    )
                )

        if on_progress is not None:
            on_progress(
                ProgressEvent(
                    task_id=task_id,
                    sequence=0,
                    state=TaskState.RUNNING,
                    stage="deepdiscover_training",
                    message="Initializing DeepDiscover full-sequence training.",
                    completed_units=0,
                    total_units=config.task.training.epochs,
                    fraction=0.0,
                )
            )

        fit = fit_deepdiscover(
            config.model,
            config.task.training,
            prepared.snow_runtime.default_module,
            analysis.precipitation,
            analysis.temperature,
            analysis.potential_evapotranspiration,
            analysis.streamflow,
            start_index=period.start_index,
            train_end_index=period.train_end_index,
            validation_end_index=period.validation_end_index,
            test_end_index=end,
            warmup_steps=config.task.warmup_steps,
            objective_id=config.task.objective.objective_id,
            cancellation_requested=lambda: token.is_requested,
            on_epoch=epoch_progress,
        )
        start = period.start_index
        observations = analysis.streamflow[start:end]
        partition_metrics = {
            name: evaluate_metrics(
                observations,
                fit.simulation.outputs[:, 0],
                mask,
            )
            for name, mask in fit.partition_masks.items()
        }
        best_validation = _optimization_score(
            observations,
            fit.simulation.outputs[:, 0],
            fit.partition_masks["validation"],
            config.task.objective.objective_id,
        )
        input_switches = np.asarray(
            fit.simulation.diagnostics["input_switch_probabilities"],
            dtype=float,
        )
        process_switches = np.asarray(
            fit.simulation.diagnostics["process_switch_probabilities"],
            dtype=float,
        )
        training_result = {
            "engine_id": fit.artifact.engine_id,
            "model_id": config.model.model_id,
            "artifact": fit.artifact.as_mapping(),
            "history": fit.history,
            "best_epoch": fit.artifact.best_epoch,
            "epochs_completed": len(fit.history),
            "stopped_early": fit.stopped_early,
            "selection_metric": (
                f"validation_{training_objective.objective_id}"
            ),
            "best_validation_score": best_validation,
            "objective": {
                "objective_id": training_objective.objective_id,
                "symbol": training_objective.symbol,
                "name": training_objective.name,
                "direction": training_objective.direction,
                "ideal_value": training_objective.ideal_value,
                "unit": training_objective.unit,
                "requires_global_batch": True,
            },
            "prediction_layout": "sequence_to_sequence",
            "process_dynamics": "instantaneous",
            "process_count": config.model.process_count,
            "hidden_units": config.model.hidden_units,
            "input_roles": (
                "rain",
                "snowmelt",
                "potential_evapotranspiration",
                "discovered_storage",
            ),
            "input_switch_probabilities": input_switches,
            "process_switch_probabilities": process_switches,
            "snow_module_id": fit.artifact.snow_module_id,
            "snow_parameters": dict(fit.artifact.snow_parameters),
            "calibrated_snow_parameters": (
                fit.artifact.calibrated_snow_parameters
            ),
            "snow_calibration_method": fit.artifact.snow_calibration_method,
            "loss_configuration": config.model.loss.model_dump(mode="json"),
            "partition_metrics": partition_metrics,
        }
        state = TaskState.CANCELLED if fit.cancelled else TaskState.COMPLETED
        if on_progress is not None:
            elapsed, remaining = progress_clock.finished()
            on_progress(
                ProgressEvent(
                    task_id=task_id,
                    sequence=0,
                    state=state,
                    stage="deepdiscover_training",
                    message=(
                        "DeepDiscover training was cancelled at an epoch boundary."
                        if fit.cancelled
                        else (
                            "The best validation checkpoint was restored with "
                            "its states, fluxes, switches, and analysis tensors."
                        )
                    ),
                    completed_units=len(fit.history),
                    total_units=config.task.training.epochs,
                    fraction=len(fit.history) / config.task.training.epochs,
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=remaining,
                    best_score=best_validation,
                )
            )
        return ExperimentRunResult(
            task_id=task_id,
            state=state,
            experiment_id=config.experiment_id,
            configuration_hash=prepared.summary.configuration_hash,
            validation=prepared.summary,
            calibration=None,
            best_simulation=fit.simulation,
            timestamps=(
                analysis.timestamps[start:end]
                if analysis.timestamps is not None
                else None
            ),
            observations=observations,
            forcings={
                "precipitation": analysis.precipitation[start:end],
                "temperature": analysis.temperature[start:end],
                "potential_evapotranspiration": (
                    analysis.potential_evapotranspiration[start:end]
                ),
            },
            evaluation_metrics=partition_metrics["test"],
            partition_metrics=partition_metrics,
            run_kind="training",
            training=training_result,
        )

    def _run_paml(
        self,
        config: ExperimentConfigV1,
        prepared: PreparedNeuralExperiment,
        *,
        task_id: str,
        token: CancellationToken,
        on_progress: ProgressObserver | None,
    ) -> ExperimentRunResult:
        """Jointly train the selected P-RNN backbone and causal CNN."""

        if not isinstance(config.model, PamlModelConfig) or not isinstance(
            config.task, PamlTrainingTaskConfig
        ):
            raise ValueError("The PaML runner received an incompatible configuration.")
        analysis = prepared.analysis
        period = config.task.period
        end = period.test_end_index or len(analysis.streamflow)
        objective = objective_definition(config.task.objective.objective_id)
        progress_clock = _TrainingProgressClock()

        def epoch_progress(epoch: int, total: int, best_score: float) -> None:
            if on_progress is not None:
                elapsed, remaining = progress_clock.estimate(epoch, total)
                on_progress(
                    ProgressEvent(
                        task_id=task_id,
                        sequence=0,
                        state=TaskState.RUNNING,
                        stage="paml_training",
                        message=(
                            f"Training PaML epoch {epoch} of {total}; "
                            f"best validation {objective.symbol} {best_score:.6g}."
                        ),
                        completed_units=epoch,
                        total_units=total,
                        fraction=epoch / total,
                        elapsed_seconds=elapsed,
                        estimated_remaining_seconds=remaining,
                        best_score=best_score,
                    )
                )

        if on_progress is not None:
            on_progress(
                ProgressEvent(
                    task_id=task_id,
                    sequence=0,
                    state=TaskState.RUNNING,
                    stage="paml_training",
                    message="Compiling joint P-RNN and causal-CNN training graph.",
                    completed_units=0,
                    total_units=config.task.training.epochs,
                    fraction=0.0,
                )
            )

        fit = fit_paml(
            config.model,
            config.task.training,
            analysis.precipitation,
            analysis.temperature,
            analysis.potential_evapotranspiration,
            analysis.streamflow,
            start_index=period.start_index,
            train_end_index=period.train_end_index,
            validation_end_index=period.validation_end_index,
            test_end_index=end,
            warmup_steps=config.task.warmup_steps,
            objective_id=config.task.objective.objective_id,
            cancellation_requested=lambda: token.is_requested,
            on_epoch=epoch_progress,
        )
        start = period.start_index
        observations = analysis.streamflow[start:end]
        partition_metrics = {
            name: evaluate_metrics(
                observations,
                fit.simulation.outputs[:, 0],
                mask,
            )
            for name, mask in fit.partition_masks.items()
        }
        best_validation = _optimization_score(
            observations,
            fit.simulation.outputs[:, 0],
            fit.partition_masks["validation"],
            config.task.objective.objective_id,
        )
        training_result = {
            "engine_id": fit.artifact.engine_id,
            "model_id": config.model.model_id,
            "physical_model_id": config.model.physical_model.model_id,
            "training_strategy": config.model.training_strategy,
            "artifact": fit.artifact.as_mapping(),
            "history": fit.history,
            "best_epoch": fit.artifact.best_epoch,
            "epochs_completed": len(fit.history),
            "stopped_early": fit.stopped_early,
            "selection_metric": f"validation_{objective.objective_id}",
            "best_validation_score": best_validation,
            "objective": {
                "objective_id": objective.objective_id,
                "symbol": objective.symbol,
                "name": objective.name,
                "direction": objective.direction,
                "ideal_value": objective.ideal_value,
                "unit": objective.unit,
                "requires_global_batch": True,
            },
            "partition_metrics": partition_metrics,
            "physical_parameters": fit.simulation.diagnostics.get(
                "physical_parameters", {}
            ),
            "cnn_filters": config.model.cnn_filters,
            "cnn_kernel_size": config.model.cnn_kernel_size,
        }
        state = TaskState.CANCELLED if fit.cancelled else TaskState.COMPLETED
        if on_progress is not None:
            elapsed, remaining = progress_clock.finished()
            on_progress(
                ProgressEvent(
                    task_id=task_id,
                    sequence=0,
                    state=state,
                    stage="paml_training",
                    message=(
                        "PaML training was cancelled at an epoch boundary."
                        if fit.cancelled
                        else "The best validation checkpoint was restored and evaluated."
                    ),
                    completed_units=len(fit.history),
                    total_units=config.task.training.epochs,
                    fraction=len(fit.history) / config.task.training.epochs,
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=remaining,
                    best_score=best_validation,
                )
            )
        return ExperimentRunResult(
            task_id=task_id,
            state=state,
            experiment_id=config.experiment_id,
            configuration_hash=prepared.summary.configuration_hash,
            validation=prepared.summary,
            calibration=None,
            best_simulation=fit.simulation,
            timestamps=(
                analysis.timestamps[start:end]
                if analysis.timestamps is not None
                else None
            ),
            observations=observations,
            forcings={
                "precipitation": analysis.precipitation[start:end],
                "temperature": analysis.temperature[start:end],
                "potential_evapotranspiration": (
                    analysis.potential_evapotranspiration[start:end]
                ),
            },
            evaluation_metrics=partition_metrics["test"],
            partition_metrics=partition_metrics,
            run_kind="training",
            training=training_result,
        )
