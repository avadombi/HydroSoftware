"""Post-simulation metrics kept separate from calibration objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray


MetricDirection = Literal["maximize", "minimize", "target"]


@dataclass(frozen=True)
class EvaluationMetricResult:
    """One metric value for the best simulated trajectory."""

    metric_id: str
    symbol: str
    name: str
    value: float | None
    direction: MetricDirection
    ideal_value: float
    unit: str
    definition: str
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.metric_id.strip() or not self.symbol.strip() or not self.name.strip():
            raise ValueError("Metric identifiers and labels cannot be empty.")
        if self.value is not None and not np.isfinite(self.value):
            raise ValueError("Metric values must be finite or explicitly unavailable.")
        if not np.isfinite(self.ideal_value):
            raise ValueError("Metric ideal_value must be finite.")
        if (self.value is None) != (self.unavailable_reason is not None):
            raise ValueError(
                "Unavailable metrics require a reason and available metrics cannot have one."
            )


@dataclass(frozen=True)
class _MetricDefinition:
    metric_id: str
    symbol: str
    name: str
    direction: MetricDirection
    ideal_value: float
    unit: str
    definition: str
    evaluate: Callable[[NDArray[np.floating], NDArray[np.floating]], float]


_TOLERANCE = 1e-12


def _vectors(
    observations: NDArray,
    simulations: NDArray,
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    observed = np.asarray(observations, dtype=float)
    simulated = np.asarray(simulations, dtype=float)
    if observed.ndim == 2 and observed.shape[1] == 1:
        observed = observed[:, 0]
    if simulated.ndim == 2 and simulated.shape[1] == 1:
        simulated = simulated[:, 0]
    if observed.ndim != 1 or simulated.ndim != 1:
        raise ValueError("Observations and simulations must be one-dimensional.")
    if observed.shape != simulated.shape:
        raise ValueError("Observations and simulations must share the time dimension.")
    requested = np.asarray(mask, dtype=bool)
    if requested.shape != observed.shape:
        raise ValueError("The evaluation mask must have shape [time].")
    valid = requested & np.isfinite(observed)
    if not np.any(valid):
        raise ValueError("At least one finite scored observation is required.")
    if not np.all(np.isfinite(simulated[valid])):
        raise ValueError("Simulations contain non-finite values in the scoring period.")
    return observed[valid], simulated[valid]


def _rmse(observed: NDArray[np.floating], simulated: NDArray[np.floating]) -> float:
    return float(np.sqrt(np.mean(np.square(simulated - observed))))


def _mse(observed: NDArray[np.floating], simulated: NDArray[np.floating]) -> float:
    return float(np.mean(np.square(simulated - observed)))


def _mae(observed: NDArray[np.floating], simulated: NDArray[np.floating]) -> float:
    return float(np.mean(np.abs(simulated - observed)))


def _nse(observed: NDArray[np.floating], simulated: NDArray[np.floating]) -> float:
    if len(observed) < 2:
        raise ValueError("NSE requires at least two scored observations.")
    denominator = float(np.square(observed - observed.mean()).sum())
    if denominator <= _TOLERANCE:
        raise ValueError("NSE is undefined for constant observed streamflow.")
    return 1.0 - float(np.square(simulated - observed).sum()) / denominator


def _kge_2009(
    observed: NDArray[np.floating], simulated: NDArray[np.floating]
) -> float:
    if len(observed) < 2:
        raise ValueError("KGE requires at least two scored observations.")
    observed_mean = float(observed.mean())
    if abs(observed_mean) <= _TOLERANCE:
        raise ValueError("KGE is undefined when mean observed streamflow is zero.")
    observed_std = float(observed.std(ddof=0))
    simulated_std = float(simulated.std(ddof=0))
    if observed_std <= _TOLERANCE:
        raise ValueError("KGE is undefined for constant observed streamflow.")
    if simulated_std <= _TOLERANCE:
        raise ValueError("KGE is undefined for constant simulated streamflow.")
    correlation = float(np.corrcoef(observed, simulated)[0, 1])
    variability_ratio = simulated_std / observed_std
    bias_ratio = float(simulated.mean()) / observed_mean
    return 1.0 - float(
        np.sqrt(
            (correlation - 1.0) ** 2
            + (variability_ratio - 1.0) ** 2
            + (bias_ratio - 1.0) ** 2
        )
    )


def _pbias(observed: NDArray[np.floating], simulated: NDArray[np.floating]) -> float:
    denominator = float(observed.sum())
    if abs(denominator) <= _TOLERANCE:
        raise ValueError("PBIAS is undefined when total observed streamflow is zero.")
    return 100.0 * float((observed - simulated).sum()) / denominator


def _range_normalized_rmse(
    observed: NDArray[np.floating], simulated: NDArray[np.floating]
) -> float:
    observed_range = float(observed.max() - observed.min())
    if observed_range <= _TOLERANCE:
        raise ValueError("NRMSE is undefined for zero-range observed streamflow.")
    return _rmse(observed, simulated) / observed_range


_METRICS = (
    _MetricDefinition(
        metric_id="nash_sutcliffe_efficiency",
        symbol="NSE",
        name="Nash-Sutcliffe efficiency",
        direction="maximize",
        ideal_value=1.0,
        unit="dimensionless",
        definition="1 - sum((Qsim - Qobs)^2) / sum((Qobs - mean(Qobs))^2)",
        evaluate=_nse,
    ),
    _MetricDefinition(
        metric_id="kling_gupta_efficiency_2009",
        symbol="KGE",
        name="Kling-Gupta efficiency (2009)",
        direction="maximize",
        ideal_value=1.0,
        unit="dimensionless",
        definition="1 - sqrt((r - 1)^2 + (sigma_sim/sigma_obs - 1)^2 + (mean_sim/mean_obs - 1)^2)",
        evaluate=_kge_2009,
    ),
    _MetricDefinition(
        metric_id="mean_square_error",
        symbol="MSE",
        name="Mean square error",
        direction="minimize",
        ideal_value=0.0,
        unit="(mm/day)^2",
        definition="mean((Qsim - Qobs)^2)",
        evaluate=_mse,
    ),
    _MetricDefinition(
        metric_id="root_mean_square_error",
        symbol="RMSE",
        name="Root mean square error",
        direction="minimize",
        ideal_value=0.0,
        unit="mm/day",
        definition="sqrt(mean((Qsim - Qobs)^2))",
        evaluate=_rmse,
    ),
    _MetricDefinition(
        metric_id="range_normalized_root_mean_square_error",
        symbol="NRMSE",
        name="Range-normalized RMSE",
        direction="minimize",
        ideal_value=0.0,
        unit="dimensionless",
        definition="RMSE / (max(Qobs) - min(Qobs))",
        evaluate=_range_normalized_rmse,
    ),
    _MetricDefinition(
        metric_id="mean_absolute_error",
        symbol="MAE",
        name="Mean absolute error",
        direction="minimize",
        ideal_value=0.0,
        unit="mm/day",
        definition="mean(abs(Qsim - Qobs))",
        evaluate=_mae,
    ),
    _MetricDefinition(
        metric_id="percent_bias_moriasi",
        symbol="PBIAS",
        name="Percent bias",
        direction="target",
        ideal_value=0.0,
        unit="%",
        definition="100 * sum(Qobs - Qsim) / sum(Qobs); positive means underestimation",
        evaluate=_pbias,
    ),
)


def evaluate_metrics(
    observations: NDArray,
    simulations: NDArray,
    mask: NDArray[np.bool_],
) -> tuple[EvaluationMetricResult, ...]:
    """Evaluate the registered diagnostics on one shared paired-data mask."""

    observed, simulated = _vectors(observations, simulations, mask)
    results: list[EvaluationMetricResult] = []
    for metric in _METRICS:
        try:
            value = float(metric.evaluate(observed, simulated))
            if not np.isfinite(value):
                raise ValueError(f"{metric.symbol} produced a non-finite value.")
            reason = None
        except (FloatingPointError, ValueError, ZeroDivisionError) as exc:
            value = None
            reason = str(exc)
        results.append(
            EvaluationMetricResult(
                metric_id=metric.metric_id,
                symbol=metric.symbol,
                name=metric.name,
                value=value,
                direction=metric.direction,
                ideal_value=metric.ideal_value,
                unit=metric.unit,
                definition=metric.definition,
                unavailable_reason=reason,
            )
        )
    return tuple(results)
