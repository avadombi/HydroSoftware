"""Leakage-safe threshold-event verification for forcing ensembles.

Continuous scores answer how close a discharge ensemble is to the observation.
This module answers a different operational question: whether the ensemble
supports decisions about exceeding one hydrological threshold.  The oldest
origins define the threshold/climatology and the affine recalibration; every
reported event score is computed on later, untouched origins.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from hydrosoftware.execution.forecast_postprocessing import (
    _origin_datetime,
    postprocess_forcing_ensemble_hindcasts,
)
from hydrosoftware.execution.forecast_verification import (
    ForecastVerificationError,
    _finite_case,
    _result_payload,
)
from hydrosoftware.execution.serialization import to_jsonable


EVENT_VERIFICATION_METHOD_ID = "threshold_event_decision_verification.v1"
ThresholdMode = Literal["absolute", "training_quantile"]
_EPSILON = 1e-15


def _training_observations(
    ordered_results: Sequence[Mapping[str, Any]],
    *,
    training_case_count: int,
    member_count: int,
    horizon: int,
) -> tuple[np.ndarray, int, bool]:
    """Return one observation per unique valid time when timestamps exist."""

    by_timestamp: dict[str, float] = {}
    without_timestamp: list[float] = []
    all_timestamped = True
    repeated = 0
    for result in ordered_results[:training_case_count]:
        _, observations, timestamps = _finite_case(
            result,
            expected_members=member_count,
            expected_horizon=horizon,
        )
        if timestamps is None:
            all_timestamped = False
        for lead, observation in enumerate(observations):
            if not np.isfinite(observation):
                continue
            value = float(observation)
            if timestamps is None:
                without_timestamp.append(value)
                continue
            timestamp = timestamps[lead]
            if timestamp in by_timestamp:
                repeated += 1
                if not np.isclose(
                    by_timestamp[timestamp], value, rtol=0.0, atol=1e-12
                ):
                    raise ForecastVerificationError(
                        "event_verification_observation_conflict",
                        "Repeated valid timestamps contain conflicting streamflow "
                        "observations in the fitting archive.",
                    )
            else:
                by_timestamp[timestamp] = value
    values = [*by_timestamp.values(), *without_timestamp]
    if not values:
        raise ForecastVerificationError(
            "event_verification_training_observations_empty",
            "The fitting origins contain no finite streamflow observations.",
        )
    return np.asarray(values, dtype=float), repeated, all_timestamped


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count < 1:
        raise ValueError("A Wilson interval requires at least one event.")
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def _reliability_bins(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    bin_count: int,
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    assignments = np.minimum(
        np.floor(probabilities * bin_count).astype(int), bin_count - 1
    )
    rows: list[dict[str, Any]] = []
    for index in range(bin_count):
        mask = assignments == index
        count = int(np.count_nonzero(mask))
        if count:
            successes = int(np.count_nonzero(outcomes[mask]))
            lower, upper = _wilson_interval(successes, count)
            rows.append(
                {
                    "bin": index + 1,
                    "lower_probability": float(edges[index]),
                    "upper_probability": float(edges[index + 1]),
                    "count": count,
                    "mean_forecast_probability": float(np.mean(probabilities[mask])),
                    "observed_frequency": successes / count,
                    "wilson_lower_95": lower,
                    "wilson_upper_95": upper,
                }
            )
        else:
            rows.append(
                {
                    "bin": index + 1,
                    "lower_probability": float(edges[index]),
                    "upper_probability": float(edges[index + 1]),
                    "count": 0,
                    "mean_forecast_probability": None,
                    "observed_frequency": None,
                    "wilson_lower_95": None,
                    "wilson_upper_95": None,
                }
            )
    return rows


def _roc(probabilities: np.ndarray, outcomes: np.ndarray) -> tuple[float | None, list[dict[str, Any]]]:
    positives = probabilities[outcomes]
    negatives = probabilities[~outcomes]
    if len(positives) == 0 or len(negatives) == 0:
        return None, []
    comparisons = positives[:, None] - negatives[None, :]
    auc = float(
        (
            np.count_nonzero(comparisons > 0.0)
            + 0.5 * np.count_nonzero(comparisons == 0.0)
        )
        / comparisons.size
    )
    thresholds = [1.0 + 1e-12, *sorted(set(probabilities.tolist()), reverse=True)]
    if thresholds[-1] >= 0.0:
        thresholds.append(-1e-12)
    curve = []
    for threshold in thresholds:
        action = probabilities >= threshold
        curve.append(
            {
                "probability_threshold": float(threshold),
                "false_positive_rate": float(np.mean(action[~outcomes])),
                "true_positive_rate": float(np.mean(action[outcomes])),
            }
        )
    return auc, curve


def _economic_value(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    climatology_probability: float,
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    rows: list[dict[str, Any]] = []
    best_value: float | None = None
    best_ratio: float | None = None
    for ratio in np.linspace(0.05, 0.95, 19):
        action = probabilities >= ratio
        forecast_expense = float(
            ratio * np.mean(action) + np.mean((~action) & outcomes)
        )
        climatology_action = climatology_probability >= ratio
        climatology_expense = float(
            ratio * float(climatology_action)
            + np.mean(outcomes) * float(not climatology_action)
        )
        perfect_expense = float(ratio * np.mean(outcomes))
        denominator = climatology_expense - perfect_expense
        value = (
            (climatology_expense - forecast_expense) / denominator
            if denominator > _EPSILON
            else None
        )
        if value is not None and (best_value is None or value > best_value):
            best_value = float(value)
            best_ratio = float(ratio)
        rows.append(
            {
                "cost_loss_ratio": float(ratio),
                "forecast_expense": forecast_expense,
                "climatology_expense": climatology_expense,
                "perfect_expense": perfect_expense,
                "relative_economic_value": float(value) if value is not None else None,
            }
        )
    return rows, best_value, best_ratio


def _event_diagnostics(
    probabilities: Sequence[float],
    outcomes: Sequence[bool],
    *,
    climatology_probability: float,
    probability_bin_count: int,
) -> dict[str, Any]:
    forecast = np.asarray(probabilities, dtype=float)
    observed = np.asarray(outcomes, dtype=bool)
    if forecast.ndim != 1 or observed.ndim != 1 or forecast.shape != observed.shape:
        raise ValueError("Event probabilities and outcomes must be aligned vectors.")
    if not len(forecast) or not np.all(np.isfinite(forecast)):
        raise ValueError("Event diagnostics require finite forecast probabilities.")
    if np.any((forecast < 0.0) | (forecast > 1.0)):
        raise ValueError("Event probabilities must lie inside [0, 1].")
    observed_float = observed.astype(float)
    brier = float(np.mean(np.square(forecast - observed_float)))
    reference_brier = float(
        np.mean(np.square(climatology_probability - observed_float))
    )
    brier_skill = (
        1.0 - brier / reference_brier
        if reference_brier > _EPSILON
        else None
    )
    auc, roc_curve = _roc(forecast, observed)
    value_curve, maximum_value, maximum_value_ratio = _economic_value(
        forecast,
        observed,
        climatology_probability=climatology_probability,
    )
    return {
        "scored_cases": int(len(forecast)),
        "event_count": int(np.count_nonzero(observed)),
        "nonevent_count": int(np.count_nonzero(~observed)),
        "event_frequency": float(np.mean(observed)),
        "mean_forecast_probability": float(np.mean(forecast)),
        "brier_score": brier,
        "climatology_brier_score": reference_brier,
        "brier_skill_score": (
            float(brier_skill) if brier_skill is not None else None
        ),
        "roc_auc": auc,
        "reliability_bins": _reliability_bins(
            forecast,
            observed,
            bin_count=probability_bin_count,
        ),
        "roc_curve": roc_curve,
        "cost_loss_curve": value_curve,
        "maximum_relative_economic_value": maximum_value,
        "cost_loss_ratio_at_maximum_value": maximum_value_ratio,
    }


def verify_threshold_event_decisions(
    archives: Sequence[Mapping[str, Any]],
    *,
    training_case_count: int,
    threshold_mode: ThresholdMode,
    threshold_value: float | None = None,
    training_quantile: float = 0.9,
    probability_bin_count: int = 10,
    minimum_evaluation_cases_per_lead: int = 5,
) -> dict[str, Any]:
    """Evaluate raw and affine-adjusted exceedance probabilities out of sample."""

    if threshold_mode not in {"absolute", "training_quantile"}:
        raise ForecastVerificationError(
            "event_verification_threshold_mode_invalid",
            "The event threshold must be absolute or derived from the fitting archive.",
        )
    if threshold_mode == "absolute":
        if threshold_value is None or not np.isfinite(threshold_value) or threshold_value < 0.0:
            raise ForecastVerificationError(
                "event_verification_threshold_invalid",
                "An absolute event threshold must be a finite non-negative discharge.",
            )
    elif not 0.5 <= training_quantile <= 0.995:
        raise ForecastVerificationError(
            "event_verification_quantile_invalid",
            "The fitting-archive threshold quantile must lie inside [0.5, 0.995].",
        )
    if not 3 <= probability_bin_count <= 20:
        raise ForecastVerificationError(
            "event_verification_bins_invalid",
            "Reliability verification requires between 3 and 20 probability bins.",
        )
    if minimum_evaluation_cases_per_lead < 2:
        raise ForecastVerificationError(
            "event_verification_minimum_invalid",
            "At least two evaluation cases are required per reported lead.",
        )

    postprocessing = postprocess_forcing_ensemble_hindcasts(
        archives,
        training_case_count=training_case_count,
        minimum_training_cases_per_lead=5,
    )
    member_count = int(postprocessing["member_count"])
    horizon = int(postprocessing["horizon"])
    ordered = sorted(
        (_result_payload(archive) for archive in archives),
        key=lambda result: (_origin_datetime(result), str(result.get("task_id", ""))),
    )
    training_observed, repeated_training_times, all_training_timestamped = (
        _training_observations(
            ordered,
            training_case_count=training_case_count,
            member_count=member_count,
            horizon=horizon,
        )
    )
    resolved_threshold = (
        float(threshold_value)
        if threshold_mode == "absolute"
        else float(np.quantile(training_observed, training_quantile))
    )
    training_events = training_observed > resolved_threshold
    climatology_probability = float(np.mean(training_events))

    raw_by_lead: list[list[float]] = [[] for _ in range(horizon)]
    adjusted_by_lead: list[list[float]] = [[] for _ in range(horizon)]
    outcomes_by_lead: list[list[bool]] = [[] for _ in range(horizon)]
    raw_pooled: list[float] = []
    adjusted_pooled: list[float] = []
    outcomes_pooled: list[bool] = []
    valid_time_counts: Counter[str] = Counter()

    for case in postprocessing["evaluation_cases"]:
        raw = np.asarray(case["raw_member_predictions"], dtype=float)
        adjusted = np.asarray(case["adjusted_member_predictions"], dtype=float)
        observations = np.asarray(case["observations"], dtype=float)
        timestamps = case.get("timestamps")
        for lead in range(horizon):
            raw_members = raw[:, lead]
            adjusted_members = adjusted[:, lead]
            observation = observations[lead]
            if not (
                np.all(np.isfinite(raw_members))
                and np.all(np.isfinite(adjusted_members))
                and np.isfinite(observation)
            ):
                continue
            raw_probability = float(np.mean(raw_members > resolved_threshold))
            adjusted_probability = float(
                np.mean(adjusted_members > resolved_threshold)
            )
            event = bool(observation > resolved_threshold)
            raw_by_lead[lead].append(raw_probability)
            adjusted_by_lead[lead].append(adjusted_probability)
            outcomes_by_lead[lead].append(event)
            raw_pooled.append(raw_probability)
            adjusted_pooled.append(adjusted_probability)
            outcomes_pooled.append(event)
            if isinstance(timestamps, Sequence) and not isinstance(
                timestamps, (str, bytes)
            ):
                valid_time_counts[str(timestamps[lead])] += 1

    if not outcomes_pooled:
        raise ForecastVerificationError(
            "event_verification_evaluation_empty",
            "The later evaluation origins contain no complete finite events.",
        )
    aggregate_raw = _event_diagnostics(
        raw_pooled,
        outcomes_pooled,
        climatology_probability=climatology_probability,
        probability_bin_count=probability_bin_count,
    )
    aggregate_adjusted = _event_diagnostics(
        adjusted_pooled,
        outcomes_pooled,
        climatology_probability=climatology_probability,
        probability_bin_count=probability_bin_count,
    )
    raw_brier = float(aggregate_raw["brier_score"])
    adjusted_brier = float(aggregate_adjusted["brier_score"])
    adjusted_skill = (
        1.0 - adjusted_brier / raw_brier if raw_brier > _EPSILON else None
    )

    lead_diagnostics: list[dict[str, Any]] = []
    insufficient_leads: list[int] = []
    for lead in range(horizon):
        available = len(outcomes_by_lead[lead])
        if available < minimum_evaluation_cases_per_lead:
            insufficient_leads.append(lead + 1)
            lead_diagnostics.append(
                {
                    "lead_step": lead + 1,
                    "available_cases": available,
                    "reported": False,
                }
            )
            continue
        raw_diagnostics = _event_diagnostics(
            raw_by_lead[lead],
            outcomes_by_lead[lead],
            climatology_probability=climatology_probability,
            probability_bin_count=probability_bin_count,
        )
        adjusted_diagnostics = _event_diagnostics(
            adjusted_by_lead[lead],
            outcomes_by_lead[lead],
            climatology_probability=climatology_probability,
            probability_bin_count=probability_bin_count,
        )
        lead_raw_brier = float(raw_diagnostics["brier_score"])
        lead_diagnostics.append(
            {
                "lead_step": lead + 1,
                "available_cases": available,
                "reported": True,
                "raw": raw_diagnostics,
                "adjusted": adjusted_diagnostics,
                "adjusted_brier_skill_vs_raw": (
                    1.0
                    - float(adjusted_diagnostics["brier_score"]) / lead_raw_brier
                    if lead_raw_brier > _EPSILON
                    else None
                ),
            }
        )

    duplicate_evaluation_times = sum(
        count - 1 for count in valid_time_counts.values() if count > 1
    )
    warnings: list[dict[str, str]] = []
    evaluation_case_count = int(postprocessing["split"]["evaluation_case_count"])
    if training_case_count < 20:
        warnings.append(
            {
                "code": "event_verification_small_training_archive",
                "message": (
                    "Fewer than 20 origins define the threshold, climatology and "
                    "affine correction. Event diagnostics are exploratory."
                ),
            }
        )
    if evaluation_case_count < 20:
        warnings.append(
            {
                "code": "event_verification_small_evaluation_archive",
                "message": (
                    "Fewer than 20 later origins are available. Reliability, AUC "
                    "and economic value may be unstable."
                ),
            }
        )
    if threshold_mode == "training_quantile" and len(training_observed) < 30:
        warnings.append(
            {
                "code": "event_verification_threshold_sample_small",
                "message": (
                    "The empirical threshold uses fewer than 30 unique fitting "
                    "observations and may be sensitive to individual events."
                ),
            }
        )
    if repeated_training_times:
        warnings.append(
            {
                "code": "event_verification_training_times_deduplicated",
                "message": (
                    "Repeated fitting valid times were counted once when deriving "
                    "the threshold and climatological event probability."
                ),
            }
        )
    if not all_training_timestamped:
        warnings.append(
            {
                "code": "event_verification_training_times_unavailable",
                "message": (
                    "Some fitting cases lack timestamps, so their observations could "
                    "not be deduplicated by valid time."
                ),
            }
        )
    if duplicate_evaluation_times:
        warnings.append(
            {
                "code": "event_verification_overlapping_valid_times",
                "message": (
                    "Some later valid dates appear in more than one issue–lead event. "
                    "They are retained by lead and are not statistically independent."
                ),
            }
        )
    if insufficient_leads:
        warnings.append(
            {
                "code": "event_verification_insufficient_leads",
                "message": (
                    "Some leads have fewer evaluation cases than the reporting "
                    "minimum and therefore have no event diagnostics."
                ),
            }
        )
    if aggregate_raw["event_count"] < 10 or aggregate_raw["nonevent_count"] < 10:
        warnings.append(
            {
                "code": "event_verification_class_imbalance",
                "message": (
                    "The independent evaluation contains fewer than ten events or "
                    "non-events. Reliability and ROC discrimination are fragile."
                ),
            }
        )
    if adjusted_skill is not None and adjusted_skill < 0.0:
        warnings.append(
            {
                "code": "event_verification_adjustment_negative_skill",
                "message": (
                    "Affine recalibration has negative Brier skill relative to the "
                    "raw ensemble for this threshold and should not replace it."
                ),
            }
        )
    warnings.append(
        {
            "code": "event_verification_scope",
            "message": (
                "Brier score measures probability accuracy, AUC measures ranking, "
                "and cost–loss value depends on a user's decision ratio. None alone "
                "establishes operational suitability."
            ),
        }
    )

    return to_jsonable(
        {
            "task_id": f"forecast-event-verification-{uuid4()}",
            "state": "completed",
            "method_id": EVENT_VERIFICATION_METHOD_ID,
            "source_task_id": postprocessing["source_task_id"],
            "experiment_id": postprocessing["experiment_id"],
            "configuration_hash": postprocessing["configuration_hash"],
            "model_id": postprocessing["model_id"],
            "member_count": member_count,
            "horizon": horizon,
            "threshold": {
                "mode": threshold_mode,
                "value": resolved_threshold,
                "unit": "mm/day",
                "event_definition": "observed_or_member_streamflow > threshold",
                "training_quantile": (
                    training_quantile
                    if threshold_mode == "training_quantile"
                    else None
                ),
                "training_observation_count": int(len(training_observed)),
            },
            "split": postprocessing["split"],
            "climatology": {
                "training_event_probability": climatology_probability,
                "training_event_count": int(np.count_nonzero(training_events)),
                "training_nonevent_count": int(
                    np.count_nonzero(~training_events)
                ),
            },
            "probability_bin_count": probability_bin_count,
            "minimum_evaluation_cases_per_lead": minimum_evaluation_cases_per_lead,
            "lead_time_diagnostics": lead_diagnostics,
            "aggregate": {
                "raw": aggregate_raw,
                "adjusted": aggregate_adjusted,
                "adjusted_brier_skill_vs_raw": (
                    float(adjusted_skill) if adjusted_skill is not None else None
                ),
            },
            "postprocessing_method_id": postprocessing["method_id"],
            "warnings": warnings,
        }
    )
