"""Deterministic bounded calibration engines with safe task boundaries."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import pickle
from time import monotonic
from typing import Callable, Mapping, TypeAlias
from uuid import uuid4

import numpy as np

from hydrosoftware.calibration.definitions import (
    CalibrationProblem,
    CalibrationRunResult,
    EvaluationRecord,
)
from hydrosoftware.calibration.objectives import ObjectiveDirection
from hydrosoftware.execution.tasks import (
    CancellationToken,
    ProgressEvent,
    ProgressSeverity,
    StructuredError,
    TaskState,
)


ProgressObserver = Callable[[ProgressEvent], None]


_PROCESS_CALIBRATION_PROBLEM: CalibrationProblem | None = None


def _initialize_process_calibration(problem: CalibrationProblem) -> None:
    global _PROCESS_CALIBRATION_PROBLEM
    _PROCESS_CALIBRATION_PROBLEM = problem


def _score_process_candidate(parameters: Mapping[str, float]) -> float:
    problem = _PROCESS_CALIBRATION_PROBLEM
    if problem is None:  # pragma: no cover - protects a broken process initializer.
        raise RuntimeError("The process calibration problem was not initialized.")
    return CalibrationEngine._score_candidate(problem, parameters)


@dataclass(frozen=True)
class UniformRandomSearchConfig:
    """Bounded baseline search used to validate the calibration workflow.

    The first evaluation uses declared parameter defaults when
    ``include_defaults`` is true. Remaining candidates are sampled uniformly
    within every trainable parameter bound using a reproducible random seed.
    This algorithm is a transparent reference adapter, not the final optimizer
    offered by the future application.
    """

    n_evaluations: int = 100
    seed: int = 0
    include_defaults: bool = True
    continue_on_candidate_error: bool = True

    def __post_init__(self) -> None:
        if self.n_evaluations < 1:
            raise ValueError("n_evaluations must be at least one.")


@dataclass(frozen=True)
class DifferentialEvolutionSearchConfig:
    """Classic bounded ``DE/rand/1/bin`` search in normalized parameter space.

    Every trainable parameter is represented on ``[0, 1]`` before mutation so
    that parameters with different physical units have the same numerical
    scale. Integer parameters remain continuous inside the population and are
    rounded only when a model candidate is evaluated.
    """

    generations: int = 25
    population_size: int | None = None
    mutation_factor: float = 0.8
    crossover_probability: float = 0.9
    seed: int = 0
    include_defaults: bool = True
    continue_on_candidate_error: bool = True

    def __post_init__(self) -> None:
        if self.generations < 1:
            raise ValueError("generations must be at least one.")
        if self.population_size is not None and self.population_size < 4:
            raise ValueError("population_size must be at least four.")
        if not 0.0 < self.mutation_factor <= 2.0:
            raise ValueError("mutation_factor must be inside (0, 2].")
        if not 0.0 <= self.crossover_probability <= 1.0:
            raise ValueError("crossover_probability must be inside [0, 1].")


CalibrationSearchConfig: TypeAlias = (
    UniformRandomSearchConfig | DifferentialEvolutionSearchConfig
)


class CalibrationEngine:
    """Run calibration while emitting structured, serializable task events."""

    @staticmethod
    def _candidate_parameters(
        problem: CalibrationProblem,
        config: UniformRandomSearchConfig,
    ) -> tuple[dict[str, float], ...]:
        rng = np.random.default_rng(config.seed)
        candidates: list[dict[str, float]] = []

        if config.include_defaults:
            candidates.append(
                {parameter.name: float(parameter.default) for parameter in problem.parameter_specs}
            )

        while len(candidates) < config.n_evaluations:
            candidate: dict[str, float] = {}
            for parameter in problem.parameter_specs:
                if parameter.trainable:
                    if parameter.value_type == "integer":
                        value = rng.integers(
                            int(parameter.lower_bound),
                            int(parameter.upper_bound) + 1,
                        )
                    else:
                        value = rng.uniform(
                            parameter.lower_bound, parameter.upper_bound
                        )
                else:
                    value = parameter.default
                candidate[parameter.name] = float(value)
            candidates.append(candidate)
        return tuple(candidates[: config.n_evaluations])

    @staticmethod
    def _is_better(
        score: float,
        best_score: float | None,
        direction: ObjectiveDirection,
    ) -> bool:
        if best_score is None:
            return True
        if direction is ObjectiveDirection.MAXIMIZE:
            return score > best_score
        return score < best_score

    @staticmethod
    def _score_candidate(
        problem: CalibrationProblem, parameters: Mapping[str, float]
    ) -> float:
        simulation = problem.simulate(parameters)
        try:
            output_index = simulation.output_names.index(problem.output_name)
        except ValueError as exc:
            raise ValueError(
                f"Simulation result does not contain output {problem.output_name!r}."
            ) from exc
        simulated = simulation.outputs[:, output_index]
        if len(simulated) != len(problem.observations):
            raise ValueError(
                "Simulation output and observations do not share the time dimension."
            )
        score = problem.objective.evaluate(
            problem.observations,
            simulated,
            problem.evaluation_mask,
        )
        if not np.isfinite(score):
            raise ValueError("The objective returned a non-finite score.")
        return float(score)

    @staticmethod
    def _resolved_population_size(
        problem: CalibrationProblem,
        config: DifferentialEvolutionSearchConfig,
    ) -> int:
        trainable_count = sum(
            parameter.trainable for parameter in problem.parameter_specs
        )
        return config.population_size or max(8, 10 * trainable_count)

    @staticmethod
    def _decode_vector(
        problem: CalibrationProblem,
        vector: np.ndarray,
    ) -> dict[str, float]:
        parameters: dict[str, float] = {}
        position = 0
        for specification in problem.parameter_specs:
            if not specification.trainable:
                parameters[specification.name] = float(specification.default)
                continue
            scaled = float(np.clip(vector[position], 0.0, 1.0))
            value = specification.lower_bound + scaled * (
                specification.upper_bound - specification.lower_bound
            )
            if specification.value_type == "integer":
                value = float(
                    np.clip(
                        np.rint(value),
                        specification.lower_bound,
                        specification.upper_bound,
                    )
                )
            parameters[specification.name] = float(value)
            position += 1
        return parameters

    @staticmethod
    def _default_vector(problem: CalibrationProblem) -> np.ndarray:
        values = []
        for specification in problem.parameter_specs:
            if not specification.trainable:
                continue
            width = specification.upper_bound - specification.lower_bound
            values.append(
                np.clip(
                    (specification.default - specification.lower_bound) / width,
                    0.0,
                    1.0,
                )
            )
        return np.asarray(values, dtype=float)

    def run(
        self,
        problem: CalibrationProblem,
        config: CalibrationSearchConfig,
        *,
        task_id: str | None = None,
        cancellation_token: CancellationToken | None = None,
        on_progress: ProgressObserver | None = None,
        max_workers: int = 1,
    ) -> CalibrationRunResult:
        """Run one calibration synchronously and return a terminal result.

        Cancellation is cooperative and is inspected before each candidate.
        A model simulation already in progress is allowed to finish so that a
        partially updated hydrological state is never exposed as a valid run.
        """

        if max_workers < 1:
            raise ValueError("max_workers must be at least one.")
        resolved_task_id = task_id or f"calibration-{uuid4()}"
        token = cancellation_token or CancellationToken()
        start_time = monotonic()
        sequence = 0
        observer = on_progress
        observer_warnings: list[StructuredError] = []
        process_executor: ProcessPoolExecutor | None = None
        algorithm_id = (
            "differential_evolution.rand1bin.v1"
            if isinstance(config, DifferentialEvolutionSearchConfig)
            else "uniform_random_search.v1"
        )

        def emit(
            state: TaskState,
            stage: str,
            message: str,
            *,
            completed: int = 0,
            total: int | None = None,
            best_score: float | None = None,
            severity: ProgressSeverity = ProgressSeverity.INFO,
            error: StructuredError | None = None,
        ) -> None:
            nonlocal sequence, observer
            elapsed = max(monotonic() - start_time, 0.0)
            fraction = None if total is None or total == 0 else completed / total
            remaining = None
            if total is not None and 2 <= completed < total and elapsed > 0.0:
                remaining = elapsed / completed * (total - completed)
            event = ProgressEvent(
                task_id=resolved_task_id,
                sequence=sequence,
                state=state,
                stage=stage,
                message=message,
                completed_units=completed,
                total_units=total,
                fraction=fraction,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=remaining,
                best_score=best_score,
                severity=severity,
                error=error,
            )
            sequence += 1
            if observer is not None:
                try:
                    observer(event)
                except Exception as exc:  # Observer failure must not corrupt a run.
                    observer_warnings.append(
                        StructuredError(
                            code="progress_observer_failed",
                            summary="Progress updates could not be delivered.",
                            explanation=(
                                "The calibration can continue, but its external progress "
                                "observer stopped accepting events."
                            ),
                            suggested_action=(
                                "Inspect the progress transport before starting another run."
                            ),
                            technical_details=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    observer = None

        def terminal_result(
            state: TaskState,
            records: list[EvaluationRecord],
            best_parameters: Mapping[str, float] | None,
            best_score: float | None,
            error: StructuredError | None = None,
            *,
            termination_reason: str,
            generations_completed: int = 0,
        ) -> CalibrationRunResult:
            nonlocal process_executor
            if process_executor is not None:
                process_executor.shutdown(wait=True, cancel_futures=True)
                process_executor = None
            return CalibrationRunResult(
                task_id=resolved_task_id,
                state=state,
                objective_name=problem.objective.name,
                objective_direction=problem.objective.direction,
                records=tuple(records),
                best_parameters=best_parameters,
                best_score=best_score,
                error=error,
                warnings=tuple(observer_warnings),
                algorithm_id=algorithm_id,
                termination_reason=termination_reason,
                generations_completed=generations_completed,
            )

        try:
            problem.validate()
            if isinstance(config, DifferentialEvolutionSearchConfig):
                population_size = self._resolved_population_size(problem, config)
                total_evaluations = population_size * (config.generations + 1)
            else:
                population_size = None
                total_evaluations = config.n_evaluations
        except Exception as exc:
            error = StructuredError(
                code="invalid_calibration_problem",
                summary="The calibration could not be prepared.",
                explanation=str(exc) or "The calibration definition is invalid.",
                suggested_action=(
                    "Review the target, evaluation period, parameter bounds, and objective."
                ),
                technical_details=f"{type(exc).__name__}: {exc}",
            )
            emit(
                TaskState.FAILED,
                "validation",
                error.summary,
                total=None,
                severity=ProgressSeverity.ERROR,
                error=error,
            )
            return terminal_result(
                TaskState.FAILED,
                [],
                None,
                None,
                error,
                termination_reason="validation_failed",
            )

        if (
            max_workers > 1
            and isinstance(config, DifferentialEvolutionSearchConfig)
        ):
            try:
                pickle.dumps(problem)
            except Exception as exc:
                observer_warnings.append(
                    StructuredError(
                        code="parallel_calibration_unavailable",
                        summary="Calibration continued in one process.",
                        explanation=(
                            "This model adapter cannot be transferred safely to "
                            "independent worker processes."
                        ),
                        suggested_action=(
                            "Use a pickle-safe built-in adapter to enable exact "
                            "population-level parallelism."
                        ),
                        technical_details=f"{type(exc).__name__}: {exc}",
                    )
                )
            else:
                process_executor = ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_initialize_process_calibration,
                    initargs=(problem,),
                )

        emit(
            TaskState.PREPARING,
            "validation",
            "Validating the calibration problem and search configuration.",
            total=total_evaluations,
        )

        records: list[EvaluationRecord] = []
        best_parameters: Mapping[str, float] | None = None
        best_score: float | None = None
        generations_completed = 0

        emit(
            TaskState.RUNNING,
            "search",
            (
                "Differential-evolution calibration started."
                if isinstance(config, DifferentialEvolutionSearchConfig)
                else "Calibration search started."
            ),
            total=total_evaluations,
        )

        def cancelled_result() -> CalibrationRunResult:
            emit(
                TaskState.CANCELLATION_REQUESTED,
                "cancellation",
                "Cancellation was requested at a safe evaluation boundary.",
                completed=len(records),
                total=total_evaluations,
                best_score=best_score,
            )
            emit(
                TaskState.CANCELLED,
                "cancellation",
                "Calibration was cancelled without discarding completed evaluations.",
                completed=len(records),
                total=total_evaluations,
                best_score=best_score,
            )
            return terminal_result(
                TaskState.CANCELLED,
                records,
                best_parameters,
                best_score,
                termination_reason="cancelled",
                generations_completed=generations_completed,
            )

        def record_evaluation(
            parameters: Mapping[str, float],
            *,
            score: float | None = None,
            exception: Exception | None = None,
            generation: int | None = None,
            population_member: int | None = None,
        ) -> tuple[float | None, StructuredError | None]:
            nonlocal best_parameters, best_score
            index = len(records) + 1
            evaluation_error: StructuredError | None = None
            if exception is None:
                assert score is not None
                record = EvaluationRecord(
                    index=index,
                    parameters=dict(parameters),
                    score=score,
                    succeeded=True,
                    generation=generation,
                    population_member=population_member,
                )
                if self._is_better(score, best_score, problem.objective.direction):
                    best_score = score
                    best_parameters = dict(parameters)
            else:
                score = None
                evaluation_error = StructuredError(
                    code="candidate_evaluation_failed",
                    summary=f"Calibration evaluation {index} failed.",
                    explanation=(
                        "This parameter candidate did not produce a valid objective score."
                    ),
                    suggested_action=(
                        "Inspect the parameter values and expert details. If failures are "
                        "systematic, stop the run and revise the model configuration."
                    ),
                    technical_details=(
                        f"{type(exception).__name__}: {exception}"
                    ),
                    context={"evaluation_index": index, "parameters": dict(parameters)},
                )
                record = EvaluationRecord(
                    index=index,
                    parameters=dict(parameters),
                    score=None,
                    succeeded=False,
                    error=evaluation_error,
                    generation=generation,
                    population_member=population_member,
                )
            records.append(record)
            generation_suffix = (
                f" (generation {generation}, member {population_member})"
                if generation is not None and population_member is not None
                else ""
            )
            emit(
                TaskState.RUNNING,
                "search",
                (
                    f"Completed calibration evaluation {index} of "
                    f"{total_evaluations}{generation_suffix}."
                ),
                completed=index,
                total=total_evaluations,
                best_score=best_score,
                severity=(
                    ProgressSeverity.WARNING
                    if evaluation_error is not None
                    else ProgressSeverity.INFO
                ),
                error=evaluation_error,
            )
            return score, evaluation_error

        def evaluate(
            parameters: Mapping[str, float],
            *,
            generation: int | None = None,
            population_member: int | None = None,
        ) -> tuple[float | None, StructuredError | None]:
            try:
                score = self._score_candidate(problem, parameters)
                return record_evaluation(
                    parameters,
                    score=score,
                    generation=generation,
                    population_member=population_member,
                )
            except Exception as exc:
                return record_evaluation(
                    parameters,
                    exception=exc,
                    generation=generation,
                    population_member=population_member,
                )

        def evaluate_population(
            parameter_rows: tuple[Mapping[str, float], ...],
            *,
            generation: int,
        ) -> tuple[
            tuple[tuple[float | None, StructuredError | None], ...],
            bool,
        ]:
            futures: tuple[Future[float], ...] | None = None
            if process_executor is not None:
                futures = tuple(
                    process_executor.submit(
                        _score_process_candidate, dict(parameters)
                    )
                    for parameters in parameter_rows
                )

            outcomes = []
            for member, parameters in enumerate(parameter_rows, start=1):
                if token.is_requested:
                    if futures is not None:
                        for pending in futures[member - 1:]:
                            pending.cancel()
                    return tuple(outcomes), True
                if futures is None:
                    outcome = evaluate(
                        parameters,
                        generation=generation,
                        population_member=member,
                    )
                else:
                    try:
                        score = futures[member - 1].result()
                        outcome = record_evaluation(
                            parameters,
                            score=score,
                            generation=generation,
                            population_member=member,
                        )
                    except Exception as exc:
                        outcome = record_evaluation(
                            parameters,
                            exception=exc,
                            generation=generation,
                            population_member=member,
                        )
                outcomes.append(outcome)
                if token.is_requested:
                    if futures is not None:
                        for pending in futures[member:]:
                            pending.cancel()
                    return tuple(outcomes), True
            return tuple(outcomes), False

        def failure_result(
            error: StructuredError,
            *,
            termination_reason: str,
        ) -> CalibrationRunResult:
            emit(
                TaskState.FAILED,
                "search",
                error.summary,
                completed=len(records),
                total=total_evaluations,
                best_score=best_score,
                severity=ProgressSeverity.ERROR,
                error=error,
            )
            return terminal_result(
                TaskState.FAILED,
                records,
                best_parameters,
                best_score,
                error,
                termination_reason=termination_reason,
                generations_completed=generations_completed,
            )

        if isinstance(config, UniformRandomSearchConfig):
            candidates = self._candidate_parameters(problem, config)
            for parameters in candidates:
                if token.is_requested:
                    return cancelled_result()
                _, evaluation_error = evaluate(parameters)
                if evaluation_error is not None and not config.continue_on_candidate_error:
                    return failure_result(
                        evaluation_error,
                        termination_reason="candidate_failure",
                    )
        else:
            assert population_size is not None
            rng = np.random.default_rng(config.seed)
            trainable_count = sum(
                parameter.trainable for parameter in problem.parameter_specs
            )
            population = rng.uniform(
                0.0,
                1.0,
                size=(population_size, trainable_count),
            )
            if config.include_defaults:
                population[0] = self._default_vector(problem)
            scores = np.full(population_size, np.nan, dtype=float)

            initial_parameters = tuple(
                self._decode_vector(problem, population[member])
                for member in range(population_size)
            )
            initial_outcomes, cancelled = evaluate_population(
                initial_parameters, generation=0
            )
            for member, (score, evaluation_error) in enumerate(initial_outcomes):
                if score is not None:
                    scores[member] = score
                if evaluation_error is not None and not config.continue_on_candidate_error:
                    return failure_result(
                        evaluation_error,
                        termination_reason="candidate_failure",
                    )
            if cancelled:
                return cancelled_result()

            for generation in range(1, config.generations + 1):
                parent_population = population.copy()
                parent_scores = scores.copy()
                trial_population = np.empty_like(parent_population)
                for member in range(population_size):
                    if token.is_requested:
                        return cancelled_result()
                    pool = np.delete(np.arange(population_size), member)
                    first, second, third = rng.choice(pool, size=3, replace=False)
                    mutant = np.clip(
                        parent_population[first]
                        + config.mutation_factor
                        * (
                            parent_population[second]
                            - parent_population[third]
                        ),
                        0.0,
                        1.0,
                    )
                    crossover = (
                        rng.random(trainable_count)
                        < config.crossover_probability
                    )
                    crossover[rng.integers(0, trainable_count)] = True
                    trial = np.where(
                        crossover,
                        mutant,
                        parent_population[member],
                    )
                    trial_population[member] = trial

                trial_parameters = tuple(
                    self._decode_vector(problem, trial_population[member])
                    for member in range(population_size)
                )
                trial_outcomes, cancelled = evaluate_population(
                    trial_parameters, generation=generation
                )
                for member, (score, evaluation_error) in enumerate(trial_outcomes):
                    trial = trial_population[member]
                    if score is not None and (
                        not np.isfinite(parent_scores[member])
                        or self._is_better(
                            score,
                            float(parent_scores[member]),
                            problem.objective.direction,
                        )
                        or score == parent_scores[member]
                    ):
                        population[member] = trial
                        scores[member] = score
                    else:
                        population[member] = parent_population[member]
                        scores[member] = parent_scores[member]
                    if (
                        evaluation_error is not None
                        and not config.continue_on_candidate_error
                    ):
                        return failure_result(
                            evaluation_error,
                            termination_reason="candidate_failure",
                        )
                if cancelled:
                    return cancelled_result()
                generations_completed = generation

        if best_score is None:
            error = StructuredError(
                code="all_evaluations_failed",
                summary="No calibration candidate produced a valid score.",
                explanation=(
                    "Every model evaluation failed or returned an invalid simulation."
                ),
                suggested_action=(
                    "Inspect the first candidate error, input data, initial state, and model "
                    "configuration before retrying."
                ),
            )
            return failure_result(
                error,
                termination_reason="all_evaluations_failed",
            )

        termination_reason = (
            "generation_budget_exhausted"
            if isinstance(config, DifferentialEvolutionSearchConfig)
            else "evaluation_budget_exhausted"
        )
        emit(
            TaskState.COMPLETED,
            "finalization",
            "Calibration completed successfully.",
            completed=total_evaluations,
            total=total_evaluations,
            best_score=best_score,
        )
        return terminal_result(
            TaskState.COMPLETED,
            records,
            best_parameters,
            best_score,
            termination_reason=termination_reason,
            generations_completed=generations_completed,
        )
