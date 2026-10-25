"""Multi-origin verification of archived meteorological ensemble forecasts.

One forcing-ensemble trajectory can provide a case-wise CRPS, but reliability,
rank and lead-dependent skill require repeated forecast origins.  This module
aggregates completed, frozen-model ensemble forecasts without refitting or
post-processing them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

import numpy as np

from hydrosoftware.core.evaluation import evaluate_metrics
from hydrosoftware.execution.forecast_ensembles import (
    FORCING_ENSEMBLE_METHOD_ID,
    empirical_crps,
)
from hydrosoftware.execution.serialization import to_jsonable


HINDCAST_VERIFICATION_METHOD_ID = (
    "meteorological_ensemble_hindcast_verification.v1"
)


class ForecastVerificationError(ValueError):
    """Stable scientific-check failure for hindcast verification."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _result_payload(archive: Mapping[str, Any]) -> Mapping[str, Any]:
    result = archive.get("result", archive)
    if not isinstance(result, Mapping):
        raise ForecastVerificationError(
            "forecast_verification_archive_invalid",
            "A selected forecast archive does not contain a result object.",
        )
    return result


def _finite_case(
    result: Mapping[str, Any],
    *,
    expected_members: int,
    expected_horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    ensemble = result.get("forecast_ensemble")
    if not isinstance(ensemble, Mapping):
        raise ForecastVerificationError(
            "forecast_verification_not_ensemble",
            "Every selected case must be a meteorological forcing-ensemble forecast.",
        )
    members = ensemble.get("members")
    if not isinstance(members, Sequence) or len(members) != expected_members:
        raise ForecastVerificationError(
            "forecast_verification_member_count_mismatch",
            "Every selected case must contain the same number of forcing members.",
        )
    member_predictions: list[np.ndarray] = []
    for member in members:
        if not isinstance(member, Mapping):
            raise ForecastVerificationError(
                "forecast_verification_archive_invalid",
                "A forcing-member record is invalid.",
            )
        prediction = np.asarray(member.get("predictions"), dtype=float)
        if prediction.ndim != 2 or prediction.shape[0] != expected_horizon:
            raise ForecastVerificationError(
                "forecast_verification_horizon_mismatch",
                "Every forcing member must share the archived forecast horizon.",
            )
        if prediction.shape[1] < 1:
            raise ForecastVerificationError(
                "forecast_verification_archive_invalid",
                "A forcing member does not contain streamflow predictions.",
            )
        member_predictions.append(prediction[:, 0])
    observations = np.asarray(result.get("observations"), dtype=float)
    if observations.ndim != 1 or len(observations) != expected_horizon:
        raise ForecastVerificationError(
            "forecast_verification_observations_required",
            "Every selected hindcast case must contain aligned streamflow observations.",
        )
    timestamps_payload = result.get("timestamps")
    timestamps = (
        [str(value) for value in timestamps_payload]
        if isinstance(timestamps_payload, Sequence)
        and not isinstance(timestamps_payload, (str, bytes))
        else None
    )
    if timestamps is not None and len(timestamps) != expected_horizon:
        raise ForecastVerificationError(
            "forecast_verification_horizon_mismatch",
            "Every timestamp series must share the archived forecast horizon.",
        )
    return np.stack(member_predictions), observations, timestamps


def _randomized_rank(
    members: np.ndarray,
    observation: float,
    rng: np.random.Generator,
) -> int:
    """Return a reproducibly randomized rank, including observation ties."""

    lower = int(np.count_nonzero(members < observation))
    tied = int(np.count_nonzero(members == observation))
    if tied == 0:
        return lower
    return int(rng.integers(lower, lower + tied + 1))


def _diagnostics(
    prediction_cases: Sequence[np.ndarray],
    observations: Sequence[float],
    *,
    lower_quantile: float,
    upper_quantile: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    predictions = np.stack(prediction_cases)
    observed = np.asarray(observations, dtype=float)
    means = np.mean(predictions, axis=1)
    errors = means - observed
    member_variances = np.var(predictions, axis=1, ddof=1)
    rms_spread = float(np.sqrt(np.mean(member_variances)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    lower = np.quantile(predictions, lower_quantile, axis=1)
    upper = np.quantile(predictions, upper_quantile, axis=1)
    coverage = float(np.mean((observed >= lower) & (observed <= upper)))
    rank_counts = np.zeros(predictions.shape[1] + 1, dtype=int)
    crps = []
    for members, observation in zip(predictions, observed, strict=True):
        crps.append(empirical_crps(members, float(observation)))
        rank_counts[_randomized_rank(members, float(observation), rng)] += 1
    spread_skill_ratio = rms_spread / rmse if rmse > 1e-15 else None
    return {
        "scored_cases": int(len(observed)),
        "mean_crps": float(np.mean(crps)),
        "mean_error_bias": float(np.mean(errors)),
        "mean_absolute_error": float(np.mean(np.abs(errors))),
        "root_mean_square_error": rmse,
        "root_mean_ensemble_variance": rms_spread,
        "spread_skill_ratio": spread_skill_ratio,
        "interval_coverage": coverage,
        "nominal_interval_coverage": upper_quantile - lower_quantile,
        "coverage_error": coverage - (upper_quantile - lower_quantile),
        "rank_counts": rank_counts.tolist(),
        "rank_relative_frequencies": (
            rank_counts.astype(float) / float(len(observed))
        ).tolist(),
    }


def verify_forcing_ensemble_hindcasts(
    archives: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
    minimum_cases_per_lead: int = 2,
) -> dict[str, Any]:
    """Verify repeated ensemble forecasts by lead without refitting the model."""

    if not 2 <= len(archives) <= 500:
        raise ForecastVerificationError(
            "forecast_verification_case_count_invalid",
            "Hindcast verification requires between 2 and 500 archived cases.",
        )
    if minimum_cases_per_lead < 2:
        raise ForecastVerificationError(
            "forecast_verification_minimum_invalid",
            "At least two scored cases are required per reported lead.",
        )

    ordered = sorted(
        (_result_payload(archive) for archive in archives),
        key=lambda result: (
            str((result.get("forecast_ensemble") or {}).get("issue_id", "")),
            str(result.get("task_id", "")),
        ),
    )
    first = ordered[0]
    first_ensemble = first.get("forecast_ensemble")
    if not isinstance(first_ensemble, Mapping):
        raise ForecastVerificationError(
            "forecast_verification_not_ensemble",
            "Every selected case must be a meteorological forcing-ensemble forecast.",
        )
    if first_ensemble.get("method_id") != FORCING_ENSEMBLE_METHOD_ID:
        raise ForecastVerificationError(
            "forecast_verification_method_mismatch",
            "The selected forecast ensemble uses an unsupported method.",
        )
    member_count = int(first_ensemble.get("member_count", 0))
    if member_count < 2:
        raise ForecastVerificationError(
            "forecast_verification_archive_invalid",
            "Archived forcing ensembles must contain at least two members.",
        )
    horizon = len(first.get("best_simulation", {}).get("outputs", []))
    if horizon < 1:
        raise ForecastVerificationError(
            "forecast_verification_archive_invalid",
            "Archived forcing ensembles must contain at least one lead.",
        )
    lower_quantile = float(first_ensemble.get("lower_quantile"))
    upper_quantile = float(first_ensemble.get("upper_quantile"))
    reference = {
        "source_task_id": str(first.get("source_task_id")),
        "experiment_id": str(first.get("experiment_id")),
        "configuration_hash": str(first.get("configuration_hash")),
        "model_id": str(first.get("model_id")),
        "member_count": member_count,
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "horizon": horizon,
    }

    cases: list[dict[str, Any]] = []
    issue_ids: set[str] = set()
    issue_times: set[str] = set()
    first_valid_times: set[str] = set()
    valid_time_counts: Counter[str] = Counter()
    predictions_by_lead: list[list[np.ndarray]] = [[] for _ in range(horizon)]
    observations_by_lead: list[list[float]] = [[] for _ in range(horizon)]
    pooled_predictions: list[np.ndarray] = []
    pooled_observations: list[float] = []

    for result in ordered:
        ensemble = result.get("forecast_ensemble")
        if not isinstance(ensemble, Mapping):
            raise ForecastVerificationError(
                "forecast_verification_not_ensemble",
                "Every selected case must be a meteorological forcing-ensemble forecast.",
            )
        candidate = {
            "source_task_id": str(result.get("source_task_id")),
            "experiment_id": str(result.get("experiment_id")),
            "configuration_hash": str(result.get("configuration_hash")),
            "model_id": str(result.get("model_id")),
            "member_count": int(ensemble.get("member_count", 0)),
            "lower_quantile": float(ensemble.get("lower_quantile")),
            "upper_quantile": float(ensemble.get("upper_quantile")),
            "horizon": len(result.get("best_simulation", {}).get("outputs", [])),
        }
        if candidate != reference:
            raise ForecastVerificationError(
                "forecast_verification_protocol_mismatch",
                "Every hindcast case must use the same frozen source model, "
                "configuration, member count, quantiles and forecast horizon.",
            )
        predictions, observations, timestamps = _finite_case(
            result,
            expected_members=member_count,
            expected_horizon=horizon,
        )
        issue_id = str(
            ensemble.get("issue_id")
            or (timestamps[0] if timestamps else result.get("task_id"))
        )
        issue_time = (
            str(ensemble["issue_time"])
            if ensemble.get("issue_time") is not None
            else None
        )
        first_valid_time = timestamps[0] if timestamps else None
        if not issue_id or issue_id in issue_ids:
            raise ForecastVerificationError(
                "forecast_verification_issue_duplicate",
                "Every hindcast case must have a unique issue identifier.",
            )
        if issue_time is not None and issue_time in issue_times:
            raise ForecastVerificationError(
                "forecast_verification_issue_time_duplicate",
                "Two selected hindcasts represent the same forecast origin.",
            )
        if (
            first_valid_time is not None
            and first_valid_time in first_valid_times
        ):
            raise ForecastVerificationError(
                "forecast_verification_first_valid_time_duplicate",
                "Two selected hindcasts start at the same valid time.",
            )
        issue_ids.add(issue_id)
        if issue_time is not None:
            issue_times.add(issue_time)
        if first_valid_time is not None:
            first_valid_times.add(first_valid_time)
        scored = 0
        for lead in range(horizon):
            members = predictions[:, lead]
            observation = observations[lead]
            if (
                np.all(np.isfinite(members))
                and np.isfinite(observation)
            ):
                predictions_by_lead[lead].append(members)
                observations_by_lead[lead].append(float(observation))
                pooled_predictions.append(members)
                pooled_observations.append(float(observation))
                scored += 1
                if timestamps is not None:
                    valid_time_counts[timestamps[lead]] += 1
        cases.append(
            {
                "task_id": str(result.get("task_id")),
                "issue_id": issue_id,
                "issue_time": issue_time,
                "first_valid_time": first_valid_time,
                "last_valid_time": timestamps[-1] if timestamps else None,
                "spinup_steps": int(ensemble.get("spinup_steps", 0)),
                "initialization": str(result.get("initialization")),
                "scored_leads": scored,
            }
        )

    rng = np.random.default_rng(seed)
    lead_diagnostics = []
    insufficient_leads: list[int] = []
    for lead in range(horizon):
        scored = len(observations_by_lead[lead])
        if scored < minimum_cases_per_lead:
            insufficient_leads.append(lead + 1)
            lead_diagnostics.append(
                {
                    "lead_step": lead + 1,
                    "available_cases": scored,
                    "reported": False,
                }
            )
            continue
        lead_diagnostics.append(
            {
                "lead_step": lead + 1,
                "available_cases": scored,
                "reported": True,
                **_diagnostics(
                    predictions_by_lead[lead],
                    observations_by_lead[lead],
                    lower_quantile=lower_quantile,
                    upper_quantile=upper_quantile,
                    rng=rng,
                ),
            }
        )
    if not pooled_observations:
        raise ForecastVerificationError(
            "forecast_verification_no_observations",
            "No finite observation is shared with complete member predictions.",
        )

    aggregate = _diagnostics(
        pooled_predictions,
        pooled_observations,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        rng=rng,
    )
    pooled_means = np.asarray(
        [np.mean(values) for values in pooled_predictions],
        dtype=float,
    )
    pooled_observed = np.asarray(pooled_observations, dtype=float)
    aggregate["deterministic_metrics"] = evaluate_metrics(
        pooled_observed,
        pooled_means,
        np.ones(len(pooled_observed), dtype=bool),
    )
    duplicate_valid_times = sum(
        count - 1 for count in valid_time_counts.values() if count > 1
    )
    warnings: list[dict[str, str]] = []
    if len(cases) < 20:
        warnings.append(
            {
                "code": "forecast_verification_small_archive",
                "message": (
                    "Fewer than 20 forecast origins are available. Rank, coverage "
                    "and spread-skill diagnostics are descriptive and may be unstable."
                ),
            }
        )
    if duplicate_valid_times:
        warnings.append(
            {
                "code": "forecast_verification_overlapping_valid_times",
                "message": (
                    "Some valid dates occur in more than one forecast case. Those "
                    "forecast events are retained by lead but are not independent."
                ),
            }
        )
    if insufficient_leads:
        warnings.append(
            {
                "code": "forecast_verification_insufficient_leads",
                "message": (
                    "Some leads have fewer finite cases than the declared reporting "
                    "minimum and therefore have no probabilistic diagnostics."
                ),
            }
        )
    warnings.append(
        {
            "code": "forecast_verification_initialization_scope",
            "message": (
                "Verification checks archived predictions and shared member spin-up, "
                "but it cannot prove that each case used a physically adequate "
                "antecedent period or initial state."
            ),
        }
    )
    warnings.append(
        {
            "code": "forecast_verification_no_postprocessing",
            "message": (
                "All diagnostics use raw frozen-model forecasts. No bias or spread "
                "correction is fitted on the verification cases."
            ),
        }
    )

    return to_jsonable(
        {
            "task_id": f"forecast-verification-{uuid4()}",
            "state": "completed",
            "method_id": HINDCAST_VERIFICATION_METHOD_ID,
            "source_task_id": reference["source_task_id"],
            "experiment_id": reference["experiment_id"],
            "configuration_hash": reference["configuration_hash"],
            "model_id": reference["model_id"],
            "seed": seed,
            "case_count": len(cases),
            "member_count": member_count,
            "horizon": horizon,
            "minimum_cases_per_lead": minimum_cases_per_lead,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "cases": cases,
            "lead_time_diagnostics": lead_diagnostics,
            "aggregate": aggregate,
            "duplicate_valid_time_events": duplicate_valid_times,
            "warnings": warnings,
        }
    )
