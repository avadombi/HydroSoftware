"""Chronological affine recalibration of archived forcing ensembles.

The hydrological model and raw forecast files are left unchanged.  Parameters
are fitted on the oldest selected origins and evaluated only on later origins.
The discrete member ensemble is retained so empirical CRPS, ranks, and
coverage remain directly comparable with the raw forecast ensemble.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np

from hydrosoftware.core.evaluation import evaluate_metrics
from hydrosoftware.execution.forecast_verification import (
    ForecastVerificationError,
    _diagnostics,
    _finite_case,
    _result_payload,
    verify_forcing_ensemble_hindcasts,
)
from hydrosoftware.execution.serialization import to_jsonable


FORECAST_POSTPROCESSING_METHOD_ID = "affine_member_recalibration.v1"
MEAN_SLOPE_BOUNDS = (0.0, 3.0)
SPREAD_MULTIPLIER_BOUNDS = (0.0, 5.0)


def _origin_datetime(result: Mapping[str, Any]) -> datetime:
    ensemble = result.get("forecast_ensemble")
    if not isinstance(ensemble, Mapping):
        raise ForecastVerificationError(
            "forecast_postprocessing_not_ensemble",
            "Every selected case must be a meteorological forcing-ensemble forecast.",
        )
    timestamps = result.get("timestamps")
    origin = ensemble.get("issue_time")
    if origin is None and isinstance(timestamps, Sequence) and timestamps:
        origin = timestamps[0]
    if origin is None:
        raise ForecastVerificationError(
            "forecast_postprocessing_origin_required",
            "Every selected case needs an ISO issue time or first valid timestamp "
            "for a leakage-safe chronological split.",
        )
    text = str(origin).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ForecastVerificationError(
            "forecast_postprocessing_origin_invalid",
            f"Forecast origin {text!r} is not an ISO date or datetime.",
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _mean_correction(
    member_cases: np.ndarray,
    observations: np.ndarray,
) -> tuple[float, float]:
    """Fit a constrained affine map for the ensemble mean by least squares."""

    means = np.mean(member_cases, axis=1)
    centered = means - float(np.mean(means))
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-15:
        slope = 1.0
    else:
        slope = float(
            np.dot(centered, observations - float(np.mean(observations)))
            / denominator
        )
        slope = float(np.clip(slope, *MEAN_SLOPE_BOUNDS))
    intercept = float(np.mean(observations - slope * means))
    return intercept, slope


def _adjust_members(
    member_cases: np.ndarray,
    *,
    intercept: float,
    mean_slope: float,
    spread_multiplier: float,
) -> np.ndarray:
    means = np.mean(member_cases, axis=1, keepdims=True)
    corrected_mean = intercept + mean_slope * means
    adjusted = corrected_mean + spread_multiplier * (member_cases - means)
    return np.maximum(adjusted, 0.0)


def _mean_empirical_crps(
    member_cases: np.ndarray,
    observations: np.ndarray,
) -> float:
    first = np.mean(np.abs(member_cases - observations[:, None]), axis=1)
    pairwise = np.abs(
        member_cases[:, :, None] - member_cases[:, None, :]
    )
    second = 0.5 * np.mean(pairwise, axis=(1, 2))
    return float(np.mean(first - second))


def _fit_spread_multiplier(
    member_cases: np.ndarray,
    observations: np.ndarray,
    *,
    intercept: float,
    mean_slope: float,
) -> float:
    """Minimize empirical CRPS over a deterministic bounded one-dimensional grid."""

    coarse = np.linspace(
        SPREAD_MULTIPLIER_BOUNDS[0],
        SPREAD_MULTIPLIER_BOUNDS[1],
        201,
    )
    coarse_scores = np.asarray(
        [
            _mean_empirical_crps(
                _adjust_members(
                    member_cases,
                    intercept=intercept,
                    mean_slope=mean_slope,
                    spread_multiplier=float(multiplier),
                ),
                observations,
            )
            for multiplier in coarse
        ],
        dtype=float,
    )
    best = int(np.argmin(coarse_scores))
    step = float(coarse[1] - coarse[0])
    lower = max(SPREAD_MULTIPLIER_BOUNDS[0], float(coarse[best]) - step)
    upper = min(SPREAD_MULTIPLIER_BOUNDS[1], float(coarse[best]) + step)
    fine = np.linspace(lower, upper, 101)
    fine_scores = np.asarray(
        [
            _mean_empirical_crps(
                _adjust_members(
                    member_cases,
                    intercept=intercept,
                    mean_slope=mean_slope,
                    spread_multiplier=float(multiplier),
                ),
                observations,
            )
            for multiplier in fine
        ],
        dtype=float,
    )
    return float(fine[int(np.argmin(fine_scores))])


def _fit_lead(
    member_cases: np.ndarray,
    observations: np.ndarray,
) -> dict[str, Any]:
    intercept, mean_slope = _mean_correction(member_cases, observations)
    spread_multiplier = _fit_spread_multiplier(
        member_cases,
        observations,
        intercept=intercept,
        mean_slope=mean_slope,
    )
    adjusted = _adjust_members(
        member_cases,
        intercept=intercept,
        mean_slope=mean_slope,
        spread_multiplier=spread_multiplier,
    )
    return {
        "intercept": intercept,
        "mean_slope": mean_slope,
        "spread_multiplier": spread_multiplier,
        "training_case_count": int(len(observations)),
        "training_raw_mean_crps": _mean_empirical_crps(
            member_cases,
            observations,
        ),
        "training_adjusted_mean_crps": _mean_empirical_crps(
            adjusted,
            observations,
        ),
    }


def _skill_score(adjusted: float, raw: float) -> float | None:
    if raw <= 1e-15:
        return None
    return 1.0 - adjusted / raw


def postprocess_forcing_ensemble_hindcasts(
    archives: Sequence[Mapping[str, Any]],
    *,
    training_case_count: int,
    seed: int = 0,
    minimum_training_cases_per_lead: int = 5,
) -> dict[str, Any]:
    """Fit on early origins and evaluate affine member recalibration on later ones."""

    if len(archives) < 7:
        raise ForecastVerificationError(
            "forecast_postprocessing_case_count_invalid",
            "Chronological post-processing requires at least seven archived "
            "origins: five for fitting and two for independent evaluation.",
        )
    if minimum_training_cases_per_lead < 5:
        raise ForecastVerificationError(
            "forecast_postprocessing_minimum_invalid",
            "At least five finite training cases are required at every lead.",
        )
    if not (
        minimum_training_cases_per_lead
        <= training_case_count
        <= len(archives) - 2
    ):
        raise ForecastVerificationError(
            "forecast_postprocessing_split_invalid",
            "The chronological split must leave at least the declared minimum "
            "for fitting and two later origins for evaluation.",
        )

    raw_verification = verify_forcing_ensemble_hindcasts(
        archives,
        seed=seed,
        minimum_cases_per_lead=2,
    )
    ordered = sorted(
        (_result_payload(archive) for archive in archives),
        key=lambda result: (_origin_datetime(result), str(result.get("task_id", ""))),
    )
    origins = [_origin_datetime(result) for result in ordered]
    if len(set(origins)) != len(origins):
        raise ForecastVerificationError(
            "forecast_postprocessing_origin_duplicate",
            "Every selected case must have a unique chronological forecast origin.",
        )

    member_count = int(raw_verification["member_count"])
    horizon = int(raw_verification["horizon"])
    training_results = ordered[:training_case_count]
    evaluation_results = ordered[training_case_count:]
    training_predictions: list[np.ndarray] = []
    training_observations: list[np.ndarray] = []
    for result in training_results:
        predictions, observations, _ = _finite_case(
            result,
            expected_members=member_count,
            expected_horizon=horizon,
        )
        training_predictions.append(predictions)
        training_observations.append(observations)

    parameters: list[dict[str, Any]] = []
    for lead in range(horizon):
        cases = []
        observed = []
        for predictions, observations in zip(
            training_predictions,
            training_observations,
            strict=True,
        ):
            members = predictions[:, lead]
            observation = observations[lead]
            if np.all(np.isfinite(members)) and np.isfinite(observation):
                cases.append(members)
                observed.append(float(observation))
        if len(observed) < minimum_training_cases_per_lead:
            raise ForecastVerificationError(
                "forecast_postprocessing_training_cases_insufficient",
                f"Lead {lead + 1} has {len(observed)} finite training cases; "
                f"{minimum_training_cases_per_lead} are required.",
            )
        parameters.append(
            {
                "lead_step": lead + 1,
                **_fit_lead(
                    np.stack(cases),
                    np.asarray(observed, dtype=float),
                ),
            }
        )

    raw_by_lead: list[list[np.ndarray]] = [[] for _ in range(horizon)]
    adjusted_by_lead: list[list[np.ndarray]] = [[] for _ in range(horizon)]
    observations_by_lead: list[list[float]] = [[] for _ in range(horizon)]
    raw_pooled: list[np.ndarray] = []
    adjusted_pooled: list[np.ndarray] = []
    observed_pooled: list[float] = []
    evaluation_cases: list[dict[str, Any]] = []

    for result in evaluation_results:
        predictions, observations, timestamps = _finite_case(
            result,
            expected_members=member_count,
            expected_horizon=horizon,
        )
        adjusted = np.empty_like(predictions)
        for lead, parameter in enumerate(parameters):
            adjusted[:, lead] = _adjust_members(
                predictions[:, lead][None, :],
                intercept=float(parameter["intercept"]),
                mean_slope=float(parameter["mean_slope"]),
                spread_multiplier=float(parameter["spread_multiplier"]),
            )[0]
            members = predictions[:, lead]
            adjusted_members = adjusted[:, lead]
            observation = observations[lead]
            if (
                np.all(np.isfinite(members))
                and np.all(np.isfinite(adjusted_members))
                and np.isfinite(observation)
            ):
                raw_by_lead[lead].append(members)
                adjusted_by_lead[lead].append(adjusted_members)
                observations_by_lead[lead].append(float(observation))
                raw_pooled.append(members)
                adjusted_pooled.append(adjusted_members)
                observed_pooled.append(float(observation))
        ensemble = result["forecast_ensemble"]
        evaluation_cases.append(
            {
                "task_id": str(result.get("task_id")),
                "issue_id": str(ensemble.get("issue_id")),
                "issue_time": (
                    str(ensemble["issue_time"])
                    if ensemble.get("issue_time") is not None
                    else None
                ),
                "timestamps": timestamps,
                "observations": observations.tolist(),
                "raw_member_predictions": predictions.tolist(),
                "adjusted_member_predictions": adjusted.tolist(),
            }
        )

    if not observed_pooled:
        raise ForecastVerificationError(
            "forecast_postprocessing_evaluation_empty",
            "The independent evaluation origins contain no complete finite events.",
        )

    lead_diagnostics: list[dict[str, Any]] = []
    for lead in range(horizon):
        if len(observations_by_lead[lead]) < 2:
            lead_diagnostics.append(
                {
                    "lead_step": lead + 1,
                    "available_cases": len(observations_by_lead[lead]),
                    "reported": False,
                }
            )
            continue
        raw = _diagnostics(
            raw_by_lead[lead],
            observations_by_lead[lead],
            lower_quantile=float(raw_verification["lower_quantile"]),
            upper_quantile=float(raw_verification["upper_quantile"]),
            rng=np.random.default_rng(seed + lead + 1),
        )
        adjusted = _diagnostics(
            adjusted_by_lead[lead],
            observations_by_lead[lead],
            lower_quantile=float(raw_verification["lower_quantile"]),
            upper_quantile=float(raw_verification["upper_quantile"]),
            rng=np.random.default_rng(seed + lead + 1),
        )
        lead_diagnostics.append(
            {
                "lead_step": lead + 1,
                "available_cases": len(observations_by_lead[lead]),
                "reported": True,
                "raw": raw,
                "adjusted": adjusted,
                "crps_skill_score": _skill_score(
                    adjusted["mean_crps"],
                    raw["mean_crps"],
                ),
            }
        )

    raw_aggregate = _diagnostics(
        raw_pooled,
        observed_pooled,
        lower_quantile=float(raw_verification["lower_quantile"]),
        upper_quantile=float(raw_verification["upper_quantile"]),
        rng=np.random.default_rng(seed),
    )
    adjusted_aggregate = _diagnostics(
        adjusted_pooled,
        observed_pooled,
        lower_quantile=float(raw_verification["lower_quantile"]),
        upper_quantile=float(raw_verification["upper_quantile"]),
        rng=np.random.default_rng(seed),
    )
    observed_array = np.asarray(observed_pooled, dtype=float)
    mask = np.ones(len(observed_array), dtype=bool)
    raw_aggregate["deterministic_metrics"] = evaluate_metrics(
        observed_array,
        np.asarray([np.mean(values) for values in raw_pooled]),
        mask,
    )
    adjusted_aggregate["deterministic_metrics"] = evaluate_metrics(
        observed_array,
        np.asarray([np.mean(values) for values in adjusted_pooled]),
        mask,
    )
    aggregate_skill = _skill_score(
        adjusted_aggregate["mean_crps"],
        raw_aggregate["mean_crps"],
    )

    warnings: list[dict[str, str]] = []
    if training_case_count < 20:
        warnings.append(
            {
                "code": "forecast_postprocessing_small_training_archive",
                "message": (
                    "Fewer than 20 origins were used for fitting. Lead-wise "
                    "coefficients may be unstable and are exploratory."
                ),
            }
        )
    if len(evaluation_results) < 20:
        warnings.append(
            {
                "code": "forecast_postprocessing_small_evaluation_archive",
                "message": (
                    "Fewer than 20 later origins remain for evaluation. Out-of-sample "
                    "skill and reliability diagnostics may be unstable."
                ),
            }
        )
    if aggregate_skill is not None and aggregate_skill < 0:
        warnings.append(
            {
                "code": "forecast_postprocessing_negative_skill",
                "message": (
                    "The adjusted ensemble has negative CRPSS on the independent "
                    "evaluation origins and should not replace the raw forecast."
                ),
            }
        )
    if any(
        parameter["mean_slope"] in MEAN_SLOPE_BOUNDS
        or parameter["spread_multiplier"] in SPREAD_MULTIPLIER_BOUNDS
        for parameter in parameters
    ):
        warnings.append(
            {
                "code": "forecast_postprocessing_parameter_boundary",
                "message": (
                    "At least one fitted coefficient reached a declared bound. "
                    "Inspect that lead before operational use."
                ),
            }
        )
    warnings.append(
        {
            "code": "forecast_postprocessing_scope",
            "message": (
                "The affine correction addresses lead-wise mean and spread errors "
                "only. It does not model temporal dependence, extremes, structural "
                "uncertainty, or neural-initialization uncertainty."
            ),
        }
    )

    return to_jsonable(
        {
            "task_id": f"forecast-postprocessing-{uuid4()}",
            "state": "completed",
            "method_id": FORECAST_POSTPROCESSING_METHOD_ID,
            "source_task_id": raw_verification["source_task_id"],
            "experiment_id": raw_verification["experiment_id"],
            "configuration_hash": raw_verification["configuration_hash"],
            "model_id": raw_verification["model_id"],
            "member_count": member_count,
            "horizon": horizon,
            "lower_quantile": raw_verification["lower_quantile"],
            "upper_quantile": raw_verification["upper_quantile"],
            "seed": seed,
            "split": {
                "strategy": "chronological_oldest_fit_latest_evaluate",
                "training_case_count": training_case_count,
                "evaluation_case_count": len(evaluation_results),
                "training_task_ids": [
                    str(result.get("task_id")) for result in training_results
                ],
                "evaluation_task_ids": [
                    str(result.get("task_id")) for result in evaluation_results
                ],
                "last_training_origin": origins[training_case_count - 1].isoformat(),
                "first_evaluation_origin": origins[training_case_count].isoformat(),
            },
            "parameter_bounds": {
                "mean_slope": list(MEAN_SLOPE_BOUNDS),
                "spread_multiplier": list(SPREAD_MULTIPLIER_BOUNDS),
            },
            "minimum_training_cases_per_lead": minimum_training_cases_per_lead,
            "parameters": parameters,
            "evaluation_cases": evaluation_cases,
            "lead_time_diagnostics": lead_diagnostics,
            "aggregate": {
                "raw": raw_aggregate,
                "adjusted": adjusted_aggregate,
                "crps_skill_score": aggregate_skill,
            },
            "warnings": warnings,
        }
    )
