"""Objective functions with explicit direction and missing-target handling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hydrosoftware.core.optimization import (
    objective_definition,
    objective_score,
    validate_objective_observations,
)


class ObjectiveDirection(str, Enum):
    """Whether a calibration objective is minimized or maximized."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ObjectiveMetric(Protocol):
    """Protocol implemented by deterministic calibration objectives."""

    name: str
    direction: ObjectiveDirection

    def validate_observations(
        self, observations: NDArray, mask: NDArray[np.bool_]
    ) -> None:
        """Validate that the target contains enough usable information."""

    def evaluate(
        self,
        observations: NDArray,
        simulations: NDArray,
        mask: NDArray[np.bool_],
    ) -> float:
        """Evaluate one simulated target sequence."""


def _as_vector(name: str, values: NDArray) -> NDArray[np.floating]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape [time] or [time, 1].")
    return array


def _valid_observation_mask(
    observations: NDArray, mask: NDArray[np.bool_]
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    observed = _as_vector("observations", observations)
    score_mask = np.asarray(mask, dtype=bool)
    if score_mask.shape != observed.shape:
        raise ValueError("The evaluation mask must have shape [time].")
    return observed, score_mask & np.isfinite(observed)


def _paired_values(
    observations: NDArray,
    simulations: NDArray,
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    observed, valid = _valid_observation_mask(observations, mask)
    simulated = _as_vector("simulations", simulations)
    if simulated.shape != observed.shape:
        raise ValueError("Simulations and observations must share the time dimension.")
    if not np.all(np.isfinite(simulated[valid])):
        raise ValueError("Simulations contain non-finite values in the scoring period.")
    return observed[valid], simulated[valid]


@dataclass(frozen=True)
class NashSutcliffeEfficiency:
    """Nash-Sutcliffe efficiency used as a maximized objective."""

    denominator_tolerance: float = 1e-12
    name: str = "nash_sutcliffe_efficiency"
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE

    def __post_init__(self) -> None:
        if self.denominator_tolerance <= 0.0:
            raise ValueError("denominator_tolerance must be positive.")

    def validate_observations(
        self, observations: NDArray, mask: NDArray[np.bool_]
    ) -> None:
        observed, valid = _valid_observation_mask(observations, mask)
        values = observed[valid]
        if len(values) < 2:
            raise ValueError("NSE requires at least two finite scored observations.")
        denominator = float(np.square(values - values.mean()).sum())
        if denominator <= self.denominator_tolerance:
            raise ValueError("NSE is undefined for a constant scored target.")

    def evaluate(
        self,
        observations: NDArray,
        simulations: NDArray,
        mask: NDArray[np.bool_],
    ) -> float:
        self.validate_observations(observations, mask)
        observed, simulated = _paired_values(observations, simulations, mask)
        denominator = float(np.square(observed - observed.mean()).sum())
        score = 1.0 - float(np.square(simulated - observed).sum()) / denominator
        if not np.isfinite(score):
            raise ValueError("NSE evaluation produced a non-finite score.")
        return score


@dataclass(frozen=True)
class RootMeanSquareError:
    """Root mean square error used as a minimized objective."""

    name: str = "root_mean_square_error"
    direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE

    def validate_observations(
        self, observations: NDArray, mask: NDArray[np.bool_]
    ) -> None:
        observed, valid = _valid_observation_mask(observations, mask)
        if len(observed[valid]) < 1:
            raise ValueError("RMSE requires at least one finite scored observation.")

    def evaluate(
        self,
        observations: NDArray,
        simulations: NDArray,
        mask: NDArray[np.bool_],
    ) -> float:
        self.validate_observations(observations, mask)
        observed, simulated = _paired_values(observations, simulations, mask)
        score = float(np.sqrt(np.mean(np.square(simulated - observed))))
        if not np.isfinite(score):
            raise ValueError("RMSE evaluation produced a non-finite score.")
        return score


@dataclass(frozen=True)
class ConfiguredHydrologicObjective:
    """Any objective registered for calibration and learned models."""

    objective_id: str

    @property
    def name(self) -> str:
        return self.objective_id

    @property
    def direction(self) -> ObjectiveDirection:
        return (
            ObjectiveDirection.MAXIMIZE
            if objective_definition(self.objective_id).direction == "maximize"
            else ObjectiveDirection.MINIMIZE
        )

    def validate_observations(
        self, observations: NDArray, mask: NDArray[np.bool_]
    ) -> None:
        observed, valid = _valid_observation_mask(observations, mask)
        validate_objective_observations(observed[valid], self.objective_id)

    def evaluate(
        self,
        observations: NDArray,
        simulations: NDArray,
        mask: NDArray[np.bool_],
    ) -> float:
        self.validate_observations(observations, mask)
        observed, simulated = _paired_values(observations, simulations, mask)
        return objective_score(observed, simulated, self.objective_id)
