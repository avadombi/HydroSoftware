"""Framework-independent definitions for deterministic model calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from hydrosoftware.calibration.objectives import ObjectiveDirection, ObjectiveMetric
from hydrosoftware.core.specifications import ParameterSpec
from hydrosoftware.core.results import SimulationResult
from hydrosoftware.execution.tasks import StructuredError, TaskState


SimulationCallable = Callable[[Mapping[str, float]], SimulationResult]


@dataclass(frozen=True)
class CalibrationProblem:
    """Complete scientific definition of one deterministic calibration."""

    parameter_specs: tuple[ParameterSpec, ...]
    simulate: SimulationCallable
    observations: NDArray[np.floating]
    objective: ObjectiveMetric
    output_name: str = "streamflow"
    evaluation_mask: NDArray[np.bool_] | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observations = np.asarray(self.observations, dtype=float)
        if observations.ndim == 2 and observations.shape[1] == 1:
            observations = observations[:, 0]
        if observations.ndim != 1:
            raise ValueError("observations must have shape [time] or [time, 1].")
        if self.evaluation_mask is None:
            mask = np.ones(len(observations), dtype=bool)
        else:
            mask = np.asarray(self.evaluation_mask, dtype=bool)
            if mask.shape != observations.shape:
                raise ValueError("evaluation_mask must have shape [time].")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "evaluation_mask", mask)

    def validate(self) -> None:
        """Validate the problem before any expensive model evaluation."""

        if not self.parameter_specs:
            raise ValueError("At least one parameter must be declared.")
        names = [parameter.name for parameter in self.parameter_specs]
        if len(set(names)) != len(names):
            raise ValueError("Calibration parameter names must be unique.")
        if not any(parameter.trainable for parameter in self.parameter_specs):
            raise ValueError("At least one calibration parameter must be trainable.")
        if not self.output_name.strip():
            raise ValueError("output_name cannot be empty.")
        if len(self.observations) == 0:
            raise ValueError("observations cannot be empty.")
        self.objective.validate_observations(
            self.observations, self.evaluation_mask
        )


@dataclass(frozen=True)
class EvaluationRecord:
    """Parameters, score, and failure details for one candidate evaluation."""

    index: int
    parameters: Mapping[str, float]
    score: float | None
    succeeded: bool
    error: StructuredError | None = None
    generation: int | None = None
    population_member: int | None = None

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError("Evaluation index must start at one.")
        if self.succeeded and self.score is None:
            raise ValueError("Successful evaluations must contain a score.")
        if not self.succeeded and self.error is None:
            raise ValueError("Failed evaluations must contain a StructuredError.")
        if self.generation is not None and self.generation < 0:
            raise ValueError("generation cannot be negative.")
        if self.population_member is not None and self.population_member < 1:
            raise ValueError("population_member must start at one.")


@dataclass(frozen=True)
class CalibrationRunResult:
    """Terminal calibration result, including partial results after cancellation."""

    task_id: str
    state: TaskState
    objective_name: str
    objective_direction: ObjectiveDirection
    records: tuple[EvaluationRecord, ...]
    best_parameters: Mapping[str, float] | None = None
    best_score: float | None = None
    error: StructuredError | None = None
    warnings: tuple[StructuredError, ...] = ()
    algorithm_id: str = "uniform_random_search.v1"
    termination_reason: str = "evaluation_budget_exhausted"
    generations_completed: int = 0

    def __post_init__(self) -> None:
        terminal_states = {TaskState.COMPLETED, TaskState.CANCELLED, TaskState.FAILED}
        if self.state not in terminal_states:
            raise ValueError("CalibrationRunResult requires a terminal task state.")
        if (self.best_parameters is None) != (self.best_score is None):
            raise ValueError("Best parameters and score must either both exist or be absent.")
        if self.state is TaskState.COMPLETED and self.best_score is None:
            raise ValueError("A completed calibration must contain a successful result.")
        if self.state is TaskState.FAILED and self.error is None:
            raise ValueError("A failed calibration must contain a StructuredError.")
        if not self.algorithm_id.strip():
            raise ValueError("algorithm_id cannot be empty.")
        if not self.termination_reason.strip():
            raise ValueError("termination_reason cannot be empty.")
        if self.generations_completed < 0:
            raise ValueError("generations_completed cannot be negative.")

