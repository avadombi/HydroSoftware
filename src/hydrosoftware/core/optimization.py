"""Shared hydrological objectives for calibration, training, and checkpoints.

Metric definitions are shared with model fitting so calibration and neural
training use the same reported objective. NumPy helpers remain here for
evaluation and analytical checks outside TensorFlow training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


ObjectiveId = Literal[
    "nash_sutcliffe_efficiency",
    "kling_gupta_efficiency_2009",
    "mean_square_error",
    "root_mean_square_error",
    "range_normalized_root_mean_square_error",
    "mean_absolute_error",
]
ObjectiveDirection = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class OptimizationObjective:
    """Stable metadata for one selectable fitting objective."""

    objective_id: ObjectiveId
    symbol: str
    name: str
    direction: ObjectiveDirection
    ideal_value: float
    unit: str
    requires_global_batch: bool = False


_TOLERANCE = 1e-12
_OBJECTIVES = (
    OptimizationObjective(
        "nash_sutcliffe_efficiency",
        "NSE",
        "Nash-Sutcliffe efficiency",
        "maximize",
        1.0,
        "dimensionless",
        requires_global_batch=True,
    ),
    OptimizationObjective(
        "kling_gupta_efficiency_2009",
        "KGE",
        "Kling-Gupta efficiency (2009)",
        "maximize",
        1.0,
        "dimensionless",
        requires_global_batch=True,
    ),
    OptimizationObjective(
        "mean_square_error",
        "MSE",
        "Mean square error",
        "minimize",
        0.0,
        "(mm/day)^2",
    ),
    OptimizationObjective(
        "root_mean_square_error",
        "RMSE",
        "Root mean square error",
        "minimize",
        0.0,
        "mm/day",
    ),
    OptimizationObjective(
        "range_normalized_root_mean_square_error",
        "NRMSE",
        "Range-normalized root mean square error",
        "minimize",
        0.0,
        "dimensionless",
        requires_global_batch=True,
    ),
    OptimizationObjective(
        "mean_absolute_error",
        "MAE",
        "Mean absolute error",
        "minimize",
        0.0,
        "mm/day",
    ),
)
OBJECTIVE_IDS: tuple[ObjectiveId, ...] = tuple(
    objective.objective_id for objective in _OBJECTIVES
)


def objective_definition(objective_id: str) -> OptimizationObjective:
    """Resolve a validated objective identifier to immutable metadata."""

    for objective in _OBJECTIVES:
        if objective.objective_id == objective_id:
            return objective
    raise ValueError(f"Unsupported optimization objective: {objective_id}")


def _vectors(
    observations: NDArray,
    simulations: NDArray | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
    observed = np.asarray(observations, dtype=float)
    if observed.ndim == 2 and observed.shape[1] == 1:
        observed = observed[:, 0]
    if observed.ndim != 1:
        raise ValueError("Objective observations must be one-dimensional.")
    simulated = None
    if simulations is not None:
        simulated = np.asarray(simulations, dtype=float)
        if simulated.ndim == 2 and simulated.shape[1] == 1:
            simulated = simulated[:, 0]
        if simulated.ndim != 1 or simulated.shape != observed.shape:
            raise ValueError(
                "Objective simulations must be one-dimensional and aligned."
            )
    if not np.all(np.isfinite(observed)):
        raise ValueError("Objective observations must be finite.")
    if simulated is not None and not np.all(np.isfinite(simulated)):
        raise ValueError("Objective simulations must be finite.")
    return observed, simulated


def validate_objective_observations(
    observations: NDArray,
    objective_id: str,
) -> None:
    """Reject targets for which the requested objective is undefined."""

    objective = objective_definition(objective_id)
    observed, _ = _vectors(observations)
    if len(observed) < (2 if objective.symbol in {"NSE", "KGE"} else 1):
        raise ValueError(
            f"{objective.symbol} requires at least "
            f"{2 if objective.symbol in {'NSE', 'KGE'} else 1} observations."
        )
    if objective.symbol == "NSE":
        centered_sum = float(np.square(observed - observed.mean()).sum())
        if centered_sum <= _TOLERANCE:
            raise ValueError(
                "NSE is undefined for constant observed streamflow."
            )
    if objective.symbol == "KGE" and float(observed.std(ddof=0)) <= _TOLERANCE:
        raise ValueError(
            "KGE is undefined for constant observed streamflow."
        )
    if objective.symbol == "KGE" and abs(float(observed.mean())) <= _TOLERANCE:
        raise ValueError(
            "KGE is undefined when mean observed streamflow is zero."
        )
    if objective.symbol == "NRMSE":
        observed_range = float(observed.max() - observed.min())
        if observed_range <= _TOLERANCE:
            raise ValueError(
                "NRMSE is undefined for zero-range observed streamflow."
            )


def objective_score(
    observations: NDArray,
    simulations: NDArray,
    objective_id: str,
) -> float:
    """Evaluate the exact user-facing score on finite paired vectors."""

    objective = objective_definition(objective_id)
    observed, simulated_raw = _vectors(observations, simulations)
    assert simulated_raw is not None
    simulated = simulated_raw
    validate_objective_observations(observed, objective_id)
    residual = simulated - observed
    mse = float(np.mean(np.square(residual)))
    if objective.symbol == "MSE":
        return mse
    if objective.symbol == "RMSE":
        return float(np.sqrt(mse))
    if objective.symbol == "NRMSE":
        return float(np.sqrt(mse)) / float(observed.max() - observed.min())
    if objective.symbol == "MAE":
        return float(np.mean(np.abs(residual)))
    if objective.symbol == "NSE":
        denominator = float(np.square(observed - observed.mean()).sum())
        return 1.0 - float(np.square(residual).sum()) / denominator

    simulated_std = float(simulated.std(ddof=0))
    if simulated_std <= _TOLERANCE:
        raise ValueError("KGE is undefined for constant simulated streamflow.")
    observed_std = float(observed.std(ddof=0))
    correlation = float(np.corrcoef(observed, simulated)[0, 1])
    variability_ratio = simulated_std / observed_std
    bias_ratio = float(simulated.mean()) / float(observed.mean())
    return 1.0 - float(
        np.sqrt(
            (correlation - 1.0) ** 2
            + (variability_ratio - 1.0) ** 2
            + (bias_ratio - 1.0) ** 2
        )
    )


def objective_loss_from_score(score: float, objective_id: str) -> float:
    """Convert a metric score to a minimized checkpoint loss."""

    objective = objective_definition(objective_id)
    loss = objective.ideal_value - score if objective.direction == "maximize" else score
    if not np.isfinite(loss):
        raise ValueError(f"{objective.symbol} produced a non-finite loss.")
    return float(loss)


def objective_optimization_score(
    observations: NDArray,
    simulations: NDArray,
    objective_id: str,
) -> tuple[float, float]:
    """Return the finite score and minimized loss used while fitting.

    For all objectives except a zero-variance simulated KGE trajectory, the
    score is exactly the public metric. KGE uses the same tiny variance floor
    as its analytical gradient during fitting so that an initially constant
    prediction can leave that degenerate state. Final reported evaluation
    metrics remain exact and explicitly unavailable when KGE is undefined.
    """

    objective = objective_definition(objective_id)
    loss, _ = objective_loss_and_gradient(
        observations,
        simulations,
        objective_id,
    )
    score = (
        objective.ideal_value - loss
        if objective.direction == "maximize"
        else loss
    )
    return float(score), float(loss)


def objective_loss_and_gradient(
    observations: NDArray,
    simulations: NDArray,
    objective_id: str,
) -> tuple[float, NDArray[np.floating]]:
    """Return a minimized loss and ``d(loss)/d(simulated streamflow)``.

    KGE uses a very small variance floor only in its training gradient so that
    an initially constant prediction can leave that degenerate state. Exact
    checkpoint and reported KGE values remain unstabilized.
    """

    objective = objective_definition(objective_id)
    observed, simulated_raw = _vectors(observations, simulations)
    assert simulated_raw is not None
    simulated = simulated_raw
    validate_objective_observations(observed, objective_id)
    count = len(observed)
    residual = simulated - observed
    mse = float(np.mean(np.square(residual)))

    if objective.symbol == "MSE":
        return mse, 2.0 * residual / count
    if objective.symbol == "MAE":
        return float(np.mean(np.abs(residual))), np.sign(residual) / count
    if objective.symbol == "RMSE":
        rmse = float(np.sqrt(mse))
        gradient = (
            np.zeros_like(residual)
            if rmse <= _TOLERANCE
            else residual / (count * rmse)
        )
        return rmse, gradient
    if objective.symbol == "NRMSE":
        observed_range = float(observed.max() - observed.min())
        rmse = float(np.sqrt(mse))
        gradient = (
            np.zeros_like(residual)
            if rmse <= _TOLERANCE
            else residual / (count * rmse * observed_range)
        )
        return rmse / observed_range, gradient
    if objective.symbol == "NSE":
        denominator = float(np.square(observed - observed.mean()).sum())
        loss = float(np.square(residual).sum()) / denominator
        return loss, 2.0 * residual / denominator

    observed_mean = float(observed.mean())
    observed_centered = observed - observed_mean
    observed_std = float(observed.std(ddof=0))
    simulated_mean = float(simulated.mean())
    simulated_centered = simulated - simulated_mean
    simulated_std = max(float(simulated.std(ddof=0)), 1e-8)
    correlation = float(
        np.mean(observed_centered * simulated_centered)
        / (observed_std * simulated_std)
    )
    variability_ratio = simulated_std / observed_std
    bias_ratio = simulated_mean / observed_mean
    distance = float(
        np.sqrt(
            (correlation - 1.0) ** 2
            + (variability_ratio - 1.0) ** 2
            + (bias_ratio - 1.0) ** 2
        )
    )
    if distance <= _TOLERANCE:
        return 0.0, np.zeros_like(simulated)
    correlation_gradient = (
        observed_centered / (count * observed_std * simulated_std)
        - correlation
        * simulated_centered
        / (count * simulated_std * simulated_std)
    )
    variability_gradient = (
        simulated_centered / (count * simulated_std * observed_std)
    )
    bias_gradient = np.full(count, 1.0 / (count * observed_mean))
    gradient = (
        (correlation - 1.0) * correlation_gradient
        + (variability_ratio - 1.0) * variability_gradient
        + (bias_ratio - 1.0) * bias_gradient
    ) / distance
    return distance, gradient
