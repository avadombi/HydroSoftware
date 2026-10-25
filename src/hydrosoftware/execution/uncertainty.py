"""Sampling uncertainty for hydrological performance metrics."""

from __future__ import annotations

from datetime import date
from math import ceil
from typing import Any, Mapping

import numpy as np

from hydrosoftware.core.evaluation import evaluate_metrics
from hydrosoftware.execution.comparison import (
    RunComparisonError,
    align_run_archives,
)
from hydrosoftware.execution.serialization import to_jsonable


class UncertaintyAnalysisError(ValueError):
    """Stable invalid-analysis failure returned at the service boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_MINIMUM_SCORED_OBSERVATIONS = 20
_MINIMUM_VALID_REPLICATE_FRACTION = 0.8
_AUTOCORRELATION_THRESHOLD_MULTIPLIER = 1.96
_AUTOCORRELATION_STABLE_LAGS = 3
_MAXIMUM_AUTOMATIC_LAG = 365


def _numeric_array(values: list[Any]) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=float,
    )


def _contiguous_links(alignment: Mapping[str, Any]) -> np.ndarray:
    """Mark adjacent aligned rows that represent consecutive daily intervals."""

    row_count = int(alignment["aligned_rows"])
    if row_count <= 1:
        return np.zeros(0, dtype=bool)
    source_rows = alignment.get("source_rows")
    if source_rows is not None:
        rows = np.asarray(source_rows, dtype=int)
        return np.diff(rows) == 1
    raw_timestamps = alignment.get("timestamps")
    if raw_timestamps is None:
        raise UncertaintyAnalysisError(
            "uncertainty_time_axis_missing",
            "The aligned evaluation support has no timestamp or source-row axis.",
        )
    try:
        ordinals = np.asarray(
            [date.fromisoformat(str(value)[:10]).toordinal() for value in raw_timestamps],
            dtype=int,
        )
    except ValueError as exc:
        raise UncertaintyAnalysisError(
            "uncertainty_timestamps_invalid",
            "Sampling uncertainty requires ISO daily timestamps.",
        ) from exc
    return np.diff(ordinals) == 1


def _scored_segments(mask: np.ndarray, links: np.ndarray) -> tuple[np.ndarray, ...]:
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return ()
    segments: list[list[int]] = [[int(positions[0])]]
    for raw_position in positions[1:]:
        position = int(raw_position)
        previous = segments[-1][-1]
        if position == previous + 1 and bool(links[previous]):
            segments[-1].append(position)
        else:
            segments.append([position])
    return tuple(np.asarray(segment, dtype=int) for segment in segments)


def _lag_correlation(
    values: np.ndarray,
    segments: tuple[np.ndarray, ...],
    lag: int,
) -> float | None:
    scored = np.concatenate([values[segment] for segment in segments])
    finite = scored[np.isfinite(scored)]
    if finite.size < 2:
        return None
    mean = float(finite.mean())
    variance = float(np.mean(np.square(finite - mean)))
    if variance <= 1e-14:
        return None
    products: list[np.ndarray] = []
    for segment in segments:
        if len(segment) <= lag:
            continue
        left = values[segment[:-lag]]
        right = values[segment[lag:]]
        finite_pair = np.isfinite(left) & np.isfinite(right)
        if np.any(finite_pair):
            products.append(
                (left[finite_pair] - mean) * (right[finite_pair] - mean)
            )
    if not products:
        return None
    covariance = float(np.concatenate(products).mean())
    return covariance / variance


def _automatic_block_length(
    observations: np.ndarray,
    predictions: tuple[np.ndarray, ...],
    segments: tuple[np.ndarray, ...],
    scored_count: int,
) -> int:
    """Choose a transparent residual/flow autocorrelation cutoff heuristic."""

    longest_segment = max(len(segment) for segment in segments)
    maximum_lag = min(
        _MAXIMUM_AUTOMATIC_LAG,
        max(1, longest_segment // 5),
    )
    if maximum_lag < _AUTOCORRELATION_STABLE_LAGS:
        return 1

    series = (observations,) + tuple(
        prediction - observations for prediction in predictions
    )
    threshold = _AUTOCORRELATION_THRESHOLD_MULTIPLIER / np.sqrt(scored_count)
    stable_count = 0
    for lag in range(1, maximum_lag + 1):
        correlations = tuple(
            value
            for value in (
                _lag_correlation(candidate, segments, lag)
                for candidate in series
            )
            if value is not None and np.isfinite(value)
        )
        insignificant = bool(correlations) and all(
            abs(value) <= threshold for value in correlations
        )
        stable_count = stable_count + 1 if insignificant else 0
        if stable_count >= _AUTOCORRELATION_STABLE_LAGS:
            return max(1, lag - _AUTOCORRELATION_STABLE_LAGS + 1)
    return maximum_lag


def _eligible_blocks(
    segments: tuple[np.ndarray, ...],
    block_length: int,
) -> np.ndarray:
    blocks = [
        segment[start:start + block_length]
        for segment in segments
        for start in range(0, len(segment) - block_length + 1)
    ]
    if not blocks:
        raise UncertaintyAnalysisError(
            "uncertainty_block_length_invalid",
            "No complete temporal block fits inside the finite scored support.",
        )
    return np.stack(blocks, axis=0)


def _metric_values(
    observations: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    mask = np.ones(len(observations), dtype=bool)
    return {
        metric.metric_id: float(metric.value)
        for metric in evaluate_metrics(observations, predictions, mask)
        if metric.value is not None
    }


def _percentile_interval(
    values: np.ndarray,
    confidence_level: float,
    *,
    requested_replicates: int,
) -> tuple[float | None, float | None, int, str | None]:
    finite = values[np.isfinite(values)]
    valid_count = int(len(finite))
    required = max(
        50,
        int(ceil(_MINIMUM_VALID_REPLICATE_FRACTION * requested_replicates)),
    )
    if valid_count < required:
        return (
            None,
            None,
            valid_count,
            (
                f"Only {valid_count} of {requested_replicates} bootstrap replicates "
                "produced a finite value."
            ),
        )
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(finite, [tail, 1.0 - tail])
    return float(lower), float(upper), valid_count, None


def _interval_metric(
    point_metric: Mapping[str, Any],
    samples: np.ndarray,
    confidence_level: float,
    requested_replicates: int,
) -> dict[str, Any]:
    point_value = point_metric.get("value")
    if point_value is None:
        return {
            **dict(point_metric),
            "confidence_level": confidence_level,
            "lower": None,
            "upper": None,
            "bootstrap_standard_error": None,
            "valid_replicates": 0,
            "interval_unavailable_reason": point_metric.get("unavailable_reason"),
        }
    lower, upper, valid_count, reason = _percentile_interval(
        samples,
        confidence_level,
        requested_replicates=requested_replicates,
    )
    finite = samples[np.isfinite(samples)]
    standard_error = (
        float(finite.std(ddof=1))
        if reason is None and len(finite) > 1
        else None
    )
    return {
        **dict(point_metric),
        "confidence_level": confidence_level,
        "lower": lower,
        "upper": upper,
        "bootstrap_standard_error": standard_error,
        "valid_replicates": valid_count,
        "interval_unavailable_reason": reason,
    }


def analyze_run_uncertainty(
    archives: tuple[Mapping[str, Any], ...],
    *,
    bootstrap_replicates: int = 1000,
    confidence_level: float = 0.95,
    block_length: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Quantify metric and paired model-difference sampling uncertainty."""

    if not 1 <= len(archives) <= 6:
        raise UncertaintyAnalysisError(
            "uncertainty_run_count_invalid",
            "Select between one and six completed runs.",
        )
    if not 200 <= bootstrap_replicates <= 5000:
        raise UncertaintyAnalysisError(
            "uncertainty_replicates_invalid",
            "Bootstrap replicates must be between 200 and 5000.",
        )
    if not 0.8 <= confidence_level <= 0.99:
        raise UncertaintyAnalysisError(
            "uncertainty_confidence_invalid",
            "The confidence level must be between 0.80 and 0.99.",
        )
    try:
        alignment = align_run_archives(archives)
    except RunComparisonError as exc:
        raise UncertaintyAnalysisError(exc.code, str(exc)) from exc

    observations = _numeric_array(alignment["observations"])
    predictions = tuple(
        _numeric_array(run["predictions"])
        for run in alignment["runs"]
    )
    common_mask = np.asarray(alignment["score_mask"], dtype=bool)
    scored_count = int(np.count_nonzero(common_mask))
    if scored_count < _MINIMUM_SCORED_OBSERVATIONS:
        raise UncertaintyAnalysisError(
            "uncertainty_support_insufficient",
            (
                "At least 20 common finite scored observations are required for "
                "block-bootstrap uncertainty."
            ),
        )

    links = _contiguous_links(alignment)
    segments = _scored_segments(common_mask, links)
    longest_segment = max(len(segment) for segment in segments)
    if block_length is None:
        selected_block_length = _automatic_block_length(
            observations,
            predictions,
            segments,
            scored_count,
        )
        block_length_source = "autocorrelation_cutoff_heuristic"
    else:
        selected_block_length = int(block_length)
        block_length_source = "user"
    if selected_block_length < 1 or selected_block_length > longest_segment:
        raise UncertaintyAnalysisError(
            "uncertainty_block_length_invalid",
            (
                f"Block length must be between 1 and the longest contiguous "
                f"scored run ({longest_segment})."
            ),
        )

    blocks = _eligible_blocks(segments, selected_block_length)
    rng = np.random.default_rng(seed)
    metric_ids = [
        metric["metric_id"]
        for metric in alignment["runs"][0]["metrics"]
    ]
    samples = {
        run["task_id"]: {
            metric_id: np.full(bootstrap_replicates, np.nan, dtype=float)
            for metric_id in metric_ids
        }
        for run in alignment["runs"]
    }
    block_count = int(ceil(scored_count / selected_block_length))
    for replicate in range(bootstrap_replicates):
        chosen_blocks = blocks[
            rng.integers(0, len(blocks), size=block_count)
        ]
        indices = chosen_blocks.reshape(-1)[:scored_count]
        sampled_observations = observations[indices]
        for run, run_predictions in zip(alignment["runs"], predictions):
            replicate_values = _metric_values(
                sampled_observations,
                run_predictions[indices],
            )
            for metric_id, value in replicate_values.items():
                samples[run["task_id"]][metric_id][replicate] = value

    run_payloads: list[dict[str, Any]] = []
    for run in alignment["runs"]:
        run_payloads.append(
            {
                "task_id": run["task_id"],
                "experiment_id": run["experiment_id"],
                "experiment_name": run["experiment_name"],
                "model_id": run["model_id"],
                "run_kind": run["run_kind"],
                "metrics": [
                    _interval_metric(
                        metric,
                        samples[run["task_id"]][metric["metric_id"]],
                        confidence_level,
                        bootstrap_replicates,
                    )
                    for metric in run["metrics"]
                ],
            }
        )

    reference = run_payloads[0]
    comparisons: list[dict[str, Any]] = []
    for candidate in run_payloads[1:]:
        metric_comparisons: list[dict[str, Any]] = []
        for reference_metric, candidate_metric in zip(
            reference["metrics"],
            candidate["metrics"],
        ):
            metric_id = reference_metric["metric_id"]
            reference_samples = samples[reference["task_id"]][metric_id]
            candidate_samples = samples[candidate["task_id"]][metric_id]
            finite = np.isfinite(reference_samples) & np.isfinite(candidate_samples)
            raw_difference = candidate_samples - reference_samples
            direction = reference_metric["direction"]
            ideal = float(reference_metric["ideal_value"])
            if direction == "maximize":
                improvement = raw_difference
            elif direction == "minimize":
                improvement = -raw_difference
            else:
                improvement = (
                    np.abs(reference_samples - ideal)
                    - np.abs(candidate_samples - ideal)
                )
            raw_lower, raw_upper, valid_count, raw_reason = _percentile_interval(
                raw_difference[finite],
                confidence_level,
                requested_replicates=bootstrap_replicates,
            )
            lower, upper, _, reason = _percentile_interval(
                improvement[finite],
                confidence_level,
                requested_replicates=bootstrap_replicates,
            )
            finite_improvement = improvement[finite & np.isfinite(improvement)]
            fraction_better = (
                float(
                    (
                        np.count_nonzero(finite_improvement > 0)
                        + 0.5 * np.count_nonzero(finite_improvement == 0)
                    )
                    / len(finite_improvement)
                )
                if reason is None and len(finite_improvement)
                else None
            )
            if lower is not None and lower > 0:
                interpretation = "interval_favors_candidate"
            elif upper is not None and upper < 0:
                interpretation = "interval_favors_reference"
            else:
                interpretation = "interval_overlaps_zero"
            reference_point = reference_metric["value"]
            candidate_point = candidate_metric["value"]
            point_raw_difference = (
                float(candidate_point - reference_point)
                if reference_point is not None and candidate_point is not None
                else None
            )
            if point_raw_difference is None:
                point_improvement = None
            elif direction == "maximize":
                point_improvement = point_raw_difference
            elif direction == "minimize":
                point_improvement = -point_raw_difference
            else:
                point_improvement = (
                    abs(float(reference_point) - ideal)
                    - abs(float(candidate_point) - ideal)
                )
            metric_comparisons.append(
                {
                    "metric_id": metric_id,
                    "symbol": reference_metric["symbol"],
                    "name": reference_metric["name"],
                    "direction": direction,
                    "ideal_value": ideal,
                    "unit": reference_metric["unit"],
                    "reference_value": reference_point,
                    "candidate_value": candidate_point,
                    "raw_difference": point_raw_difference,
                    "raw_difference_lower": raw_lower,
                    "raw_difference_upper": raw_upper,
                    "improvement": point_improvement,
                    "improvement_lower": lower,
                    "improvement_upper": upper,
                    "bootstrap_fraction_better": fraction_better,
                    "valid_replicates": valid_count,
                    "interpretation": interpretation,
                    "interval_unavailable_reason": reason or raw_reason,
                }
            )
        comparisons.append(
            {
                "reference_task_id": reference["task_id"],
                "reference_experiment_name": reference["experiment_name"],
                "candidate_task_id": candidate["task_id"],
                "candidate_experiment_name": candidate["experiment_name"],
                "metrics": metric_comparisons,
            }
        )

    warnings = list(alignment["warnings"])
    warnings.append(
        {
            "code": "uncertainty_scope_sampling_only",
            "message": (
                "Intervals quantify sampling variability over the finite evaluation "
                "record, conditional on the fitted models. They do not include "
                "parameter, forcing, structural, or predictive uncertainty."
            ),
        }
    )
    warnings.append(
        {
            "code": "uncertainty_local_stationarity_assumption",
            "message": (
                "Moving-block bootstrap inference assumes that the evaluated daily "
                "sequence is sufficiently stationary within the selected period."
            ),
        }
    )
    if block_length_source == "autocorrelation_cutoff_heuristic":
        warnings.append(
            {
                "code": "uncertainty_block_length_heuristic",
                "message": (
                    "The automatic block length is a transparent autocorrelation "
                    "cutoff heuristic, not an estimator guaranteed to be optimal "
                    "for every metric or catchment."
                ),
            }
        )
    if longest_segment < scored_count:
        warnings.append(
            {
                "code": "uncertainty_scored_gaps_preserved",
                "message": (
                    "The scored support contains gaps. Resampled blocks never cross "
                    "a missing observation or a non-consecutive daily interval."
                ),
            }
        )
    if any(run["run_kind"] == "calibration" for run in alignment["runs"]):
        warnings.append(
            {
                "code": "uncertainty_calibration_not_holdout",
                "message": (
                    "At least one interval describes a calibration-period score. "
                    "Its sampling interval is descriptive and is not an independent "
                    "out-of-sample generalization interval."
                ),
            }
        )

    payload = {
        "schema_version": "1.0",
        "method_id": "moving_block_bootstrap_metrics.v1",
        "alignment": alignment["alignment"],
        "source_sha256": alignment["source_sha256"],
        "scored_observations": scored_count,
        "longest_contiguous_scored_run": longest_segment,
        "eligible_overlapping_blocks": len(blocks),
        "block_length": selected_block_length,
        "block_length_source": block_length_source,
        "bootstrap_replicates": bootstrap_replicates,
        "confidence_level": confidence_level,
        "interval_method": "percentile",
        "paired_resampling": len(run_payloads) > 1,
        "seed": seed,
        "reference_task_id": reference["task_id"],
        "runs": run_payloads,
        "comparisons": comparisons,
        "warnings": warnings,
    }
    return to_jsonable(payload)
