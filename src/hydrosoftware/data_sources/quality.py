"""CSV profiling and transformations for daily lumped-basin data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from hydrosoftware.configuration import CsvDataSourceConfig
from hydrosoftware.data_sources.errors import DataSourceError


Severity = Literal["warning", "error"]


@dataclass(frozen=True)
class DataQualityIssue:
    """One actionable finding produced by the data-quality engine."""

    code: str
    severity: Severity
    summary: str
    explanation: str
    suggested_action: str
    variable: str | None = None
    count: int = 0
    csv_rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class VariableQualitySummary:
    """Raw and processed statistics for one standard variable."""

    variable: str
    source_column: str
    source_unit: str
    model_unit: str
    row_count: int
    raw_valid_count: int
    raw_missing_count: int
    raw_invalid_count: int
    raw_negative_count: int
    processed_valid_count: int
    processed_missing_count: int
    transformed_count: int
    raw_minimum: float | None
    raw_maximum: float | None
    processed_minimum: float | None
    processed_maximum: float | None
    processed_mean: float | None
    processed_standard_deviation: float | None
    processed_q25: float | None
    processed_median: float | None
    processed_q75: float | None


@dataclass(frozen=True)
class TimestampQualitySummary:
    """Coverage and regularity of the optional time coordinate."""

    assigned: bool
    source_column: str | None
    first: str | None
    last: str | None
    valid_count: int
    missing_count: int
    invalid_count: int
    duplicate_or_reversed_count: int
    irregular_step_count: int
    expected_step: str


@dataclass(frozen=True)
class DataQualityReport:
    """Serializable report used by preflight validation and the interface."""

    source_path: str
    sha256: str
    row_count: int
    ready: bool
    error_count: int
    warning_count: int
    transformed_value_count: int
    variables: tuple[VariableQualitySummary, ...]
    timestamps: TimestampQualitySummary
    issues: tuple[DataQualityIssue, ...]
    timestamp_labels: tuple[str, ...] | None
    observation_valid: tuple[bool, ...] | None
    correlation_variables: tuple[str, ...]
    correlation_matrix: tuple[tuple[float | None, ...], ...]


@dataclass(frozen=True)
class DataQualityAnalysis:
    """Processed arrays plus the complete audit report."""

    precipitation: NDArray[np.floating]
    temperature: NDArray[np.floating]
    potential_evapotranspiration: NDArray[np.floating]
    streamflow: NDArray[np.floating]
    additional_predictors: Mapping[str, NDArray[np.floating]]
    timestamps: tuple[str, ...] | None
    source_path: Path
    report: DataQualityReport


def _resolve_workspace_file(relative_path: str, workspace_root: Path) -> Path:
    root = workspace_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DataSourceError(
            "data_path_outside_workspace",
            "The CSV path resolves outside the project workspace.",
        ) from exc
    if not candidate.is_file():
        raise DataSourceError(
            "data_file_not_found",
            f"The CSV data file does not exist: {relative_path}",
        )
    return candidate


def _finite_stat(values: NDArray[np.floating], operation: str) -> float | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    if operation == "min":
        return float(np.min(finite))
    if operation == "max":
        return float(np.max(finite))
    if operation == "mean":
        return float(np.mean(finite))
    if operation == "std":
        return float(np.std(finite))
    if operation == "q25":
        return float(np.quantile(finite, 0.25))
    if operation == "median":
        return float(np.median(finite))
    if operation == "q75":
        return float(np.quantile(finite, 0.75))
    raise ValueError(f"Unknown statistic: {operation}")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def _interpolate_short_interior_gaps(
    values: NDArray[np.floating],
    *,
    maximum_gap: int,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Interpolate only bounded gaps no longer than the declared maximum."""

    result = values.copy()
    changed = np.zeros(len(result), dtype=bool)
    missing = ~np.isfinite(result)
    index = 0
    while index < len(result):
        if not missing[index]:
            index += 1
            continue
        start = index
        while index < len(result) and missing[index]:
            index += 1
        end = index
        gap = end - start
        if start == 0 or end == len(result) or gap > maximum_gap:
            continue
        left = result[start - 1]
        right = result[end]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        result[start:end] = np.linspace(left, right, gap + 2)[1:-1]
        changed[start:end] = True
    return result, changed


def _convert_units(
    variable: str,
    values: NDArray[np.floating],
    source: CsvDataSourceConfig,
) -> tuple[NDArray[np.floating], str, str, bool]:
    units = source.processing.units
    if variable == "precipitation":
        source_unit = units.precipitation
        factor = 25.4 if source_unit == "inch/day" else 1.0
        return values * factor, source_unit, "mm/day", factor != 1.0
    if variable == "potential_evapotranspiration":
        source_unit = units.potential_evapotranspiration
        factor = 25.4 if source_unit == "inch/day" else 1.0
        return values * factor, source_unit, "mm/day", factor != 1.0
    if variable == "temperature":
        source_unit = units.temperature
        if source_unit == "degF":
            return (values - 32.0) * (5.0 / 9.0), source_unit, "degC", True
        if source_unit == "K":
            return values - 273.15, source_unit, "degC", True
        return values, source_unit, "degC", False
    if variable == "streamflow":
        source_unit = units.streamflow
        if source_unit == "m3/s":
            if units.basin_area_km2 is None:
                raise DataSourceError(
                    "missing_basin_area",
                    "Basin area is required to convert streamflow from m3/s to mm/day.",
                )
            return (
                values * (86.4 / units.basin_area_km2),
                source_unit,
                "mm/day",
                True,
            )
        return values, source_unit, "mm/day", False
    raise ValueError(f"Unknown standard variable: {variable}")


def _row_examples(mask: NDArray[np.bool_], maximum: int = 5) -> tuple[int, ...]:
    return tuple(int(index) + 2 for index in np.flatnonzero(mask)[:maximum])


def analyze_calibration_csv(
    source: CsvDataSourceConfig,
    *,
    workspace_root: str | Path,
    maximum_index_payload: int = 100_000,
) -> DataQualityAnalysis:
    """Parse, profile, transform, and validate one configured daily CSV source."""

    path = _resolve_workspace_file(source.path, Path(workspace_root))
    try:
        raw_bytes = path.read_bytes()
        content_hash = sha256(raw_bytes).hexdigest()
        text = raw_bytes.decode(source.encoding)
    except LookupError as exc:
        raise DataSourceError(
            "unknown_text_encoding",
            f"The declared CSV encoding is not recognized: {source.encoding}",
        ) from exc
    except UnicodeDecodeError as exc:
        raise DataSourceError(
            "csv_decoding_failed",
            f"The CSV file cannot be decoded using {source.encoding}.",
        ) from exc

    if source.expected_sha256 is not None and source.expected_sha256 != content_hash:
        raise DataSourceError(
            "data_fingerprint_mismatch",
            "The CSV content no longer matches the fingerprint stored in the experiment.",
        )

    reader = csv.DictReader(text.splitlines(), delimiter=source.delimiter)
    if reader.fieldnames is None:
        raise DataSourceError("missing_csv_header", "The CSV file has no header row.")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise DataSourceError(
            "duplicate_csv_columns", "The CSV header contains duplicate column names."
        )
    mapping = source.columns
    canonical_columns = {
        "precipitation": mapping.precipitation,
        "temperature": mapping.temperature,
        "potential_evapotranspiration": mapping.potential_evapotranspiration,
    }
    if mapping.streamflow is not None:
        canonical_columns["streamflow"] = mapping.streamflow
    predictor_configs = {
        predictor.name: predictor for predictor in mapping.additional_predictors
    }
    canonical_columns.update(
        {
            predictor.name: predictor.column
            for predictor in mapping.additional_predictors
        }
    )
    required = set(canonical_columns.values())
    if mapping.timestamp is not None:
        required.add(mapping.timestamp)
    missing_columns = sorted(required.difference(reader.fieldnames))
    if missing_columns:
        raise DataSourceError(
            "missing_csv_columns",
            f"The CSV file is missing required columns: {', '.join(missing_columns)}",
        )

    rows = list(reader)
    if not rows:
        raise DataSourceError("empty_data_file", "The CSV file contains no data rows.")

    processing = source.processing
    text_tokens = {token.casefold() for token in processing.missing.text_tokens}
    numeric_sentinels = set(processing.missing.numeric_sentinels)
    issues: list[DataQualityIssue] = []
    processed: dict[str, NDArray[np.floating]] = {}
    summaries: list[VariableQualitySummary] = []

    missing_policies = {
        "precipitation": processing.missing.precipitation,
        "temperature": processing.missing.temperature,
        "potential_evapotranspiration": (
            processing.missing.potential_evapotranspiration
        ),
        "streamflow": processing.missing.streamflow,
        **{
            predictor.name: predictor.missing
            for predictor in mapping.additional_predictors
        },
    }
    negative_policies = {
        "precipitation": processing.negative.precipitation,
        "potential_evapotranspiration": (
            processing.negative.potential_evapotranspiration
        ),
        "streamflow": processing.negative.streamflow,
    }

    for variable, column in canonical_columns.items():
        raw_values = np.full(len(rows), np.nan, dtype=float)
        raw_missing = np.zeros(len(rows), dtype=bool)
        raw_invalid = np.zeros(len(rows), dtype=bool)
        for index, row in enumerate(rows):
            raw = (row.get(column) or "").strip()
            if raw.casefold() in text_tokens:
                raw_missing[index] = True
                continue
            try:
                value = float(raw)
            except ValueError:
                raw_invalid[index] = True
                continue
            if not np.isfinite(value):
                raw_missing[index] = True
            elif value in numeric_sentinels:
                raw_missing[index] = True
            else:
                raw_values[index] = value

        original = raw_values.copy()
        original_missing = raw_missing.copy()
        raw_negative = np.isfinite(raw_values) & (raw_values < 0.0)
        changed = np.zeros(len(rows), dtype=bool)

        if np.any(raw_invalid):
            issues.append(
                DataQualityIssue(
                    code="invalid_numeric_values",
                    severity="error",
                    summary=f"{variable.replace('_', ' ').title()} contains non-numeric values.",
                    explanation=(
                        f"{int(np.count_nonzero(raw_invalid))} value(s) in column "
                        f"{column!r} are neither numeric nor declared missing tokens."
                    ),
                    suggested_action=(
                        "Correct the cells or add their exact token to the missing-value list."
                    ),
                    variable=variable,
                    count=int(np.count_nonzero(raw_invalid)),
                    csv_rows=_row_examples(raw_invalid),
                )
            )

        negative_policy = negative_policies.get(variable)
        if np.any(raw_negative) and negative_policy is not None:
            negative_count = int(np.count_nonzero(raw_negative))
            if negative_policy == "error":
                issues.append(
                    DataQualityIssue(
                        code=f"negative_{variable}",
                        severity="error",
                        summary=f"Negative {variable.replace('_', ' ')} values were found.",
                        explanation=(
                            f"{negative_count} negative value(s) remain incompatible with "
                            "the selected physical assumptions."
                        ),
                        suggested_action=(
                            "Correct the source or explicitly select a documented negative-value rule."
                        ),
                        variable=variable,
                        count=negative_count,
                        csv_rows=_row_examples(raw_negative),
                    )
                )
            elif negative_policy == "clip_to_zero":
                raw_values[raw_negative] = 0.0
                changed |= raw_negative
                issues.append(
                    DataQualityIssue(
                        code=f"negative_{variable}_clipped",
                        severity="warning",
                        summary=f"Negative {variable.replace('_', ' ')} values will be clipped.",
                        explanation=f"{negative_count} value(s) are replaced by zero.",
                        suggested_action="Confirm that clipping is scientifically justified.",
                        variable=variable,
                        count=negative_count,
                        csv_rows=_row_examples(raw_negative),
                    )
                )
            elif negative_policy in {"set_missing", "exclude"}:
                raw_values[raw_negative] = np.nan
                raw_missing |= raw_negative
                changed |= raw_negative
                issues.append(
                    DataQualityIssue(
                        code=f"negative_{variable}_excluded",
                        severity="warning",
                        summary=f"Negative {variable.replace('_', ' ')} values will be excluded.",
                        explanation=f"{negative_count} value(s) are converted to missing values.",
                        suggested_action="Confirm that exclusion is scientifically justified.",
                        variable=variable,
                        count=negative_count,
                        csv_rows=_row_examples(raw_negative),
                    )
                )
            elif negative_policy == "keep":
                issues.append(
                    DataQualityIssue(
                        code=f"negative_{variable}_kept",
                        severity="warning",
                        summary=f"Negative {variable.replace('_', ' ')} values will be kept.",
                        explanation=f"{negative_count} negative observation(s) remain unchanged.",
                        suggested_action="Verify quality flags and the meaning of negative observations.",
                        variable=variable,
                        count=negative_count,
                        csv_rows=_row_examples(raw_negative),
                    )
                )

        missing_policy = missing_policies[variable]
        missing_before_policy = ~np.isfinite(raw_values)
        if missing_policy == "zero":
            raw_values[missing_before_policy & ~raw_invalid] = 0.0
            changed |= missing_before_policy & ~raw_invalid
        elif missing_policy == "linear_interpolation":
            interpolated, interpolation_changed = _interpolate_short_interior_gaps(
                raw_values,
                maximum_gap=processing.missing.maximum_interpolation_gap,
            )
            raw_values = interpolated
            changed |= interpolation_changed

        unresolved = ~np.isfinite(raw_values)
        if variable == "streamflow":
            if np.any(unresolved):
                issues.append(
                    DataQualityIssue(
                        code="missing_streamflow_excluded",
                        severity="warning",
                        summary="Some streamflow observations will not be scored.",
                        explanation=(
                            f"{int(np.count_nonzero(unresolved))} observation(s) remain missing "
                            "after the selected rule."
                        ),
                        suggested_action=(
                            "Confirm the gaps or select bounded interpolation explicitly."
                        ),
                        variable=variable,
                        count=int(np.count_nonzero(unresolved)),
                        csv_rows=_row_examples(unresolved),
                    )
                )
        elif np.any(unresolved):
            issues.append(
                DataQualityIssue(
                    code=f"unresolved_missing_{variable}",
                    severity="error",
                    summary=f"{variable.replace('_', ' ').title()} still contains gaps.",
                    explanation=(
                        f"{int(np.count_nonzero(unresolved))} value(s) cannot be supplied to "
                        "the model under the selected missing-value rule."
                    ),
                    suggested_action=(
                        "Correct the source, choose zero filling where defensible, or use "
                        "bounded interpolation for short interior gaps."
                    ),
                    variable=variable,
                    count=int(np.count_nonzero(unresolved)),
                    csv_rows=_row_examples(unresolved),
                )
            )

        if variable in predictor_configs:
            predictor_unit = predictor_configs[variable].unit
            converted = raw_values
            source_unit = predictor_unit
            model_unit = predictor_unit
            unit_changed = False
        else:
            converted, source_unit, model_unit, unit_changed = _convert_units(
                variable, raw_values, source
            )
        if unit_changed:
            changed |= np.isfinite(converted)
        processed[variable] = converted
        summaries.append(
            VariableQualitySummary(
                variable=variable,
                source_column=column,
                source_unit=source_unit,
                model_unit=model_unit,
                row_count=len(rows),
                raw_valid_count=int(np.count_nonzero(np.isfinite(original))),
                raw_missing_count=int(np.count_nonzero(original_missing)),
                raw_invalid_count=int(np.count_nonzero(raw_invalid)),
                raw_negative_count=int(np.count_nonzero(raw_negative)),
                processed_valid_count=int(np.count_nonzero(np.isfinite(converted))),
                processed_missing_count=int(np.count_nonzero(~np.isfinite(converted))),
                transformed_count=int(np.count_nonzero(changed)),
                raw_minimum=_finite_stat(original, "min"),
                raw_maximum=_finite_stat(original, "max"),
                processed_minimum=_finite_stat(converted, "min"),
                processed_maximum=_finite_stat(converted, "max"),
                processed_mean=_finite_stat(converted, "mean"),
                processed_standard_deviation=_finite_stat(converted, "std"),
                processed_q25=_finite_stat(converted, "q25"),
                processed_median=_finite_stat(converted, "median"),
                processed_q75=_finite_stat(converted, "q75"),
            )
        )

    timestamp_labels: tuple[str, ...] | None = None
    parsed_timestamps: list[datetime] = []
    timestamp_missing = np.zeros(len(rows), dtype=bool)
    timestamp_invalid = np.zeros(len(rows), dtype=bool)
    duplicate_or_reversed = 0
    irregular_steps = 0
    if mapping.timestamp is None:
        if processing.timestamps.require_timestamp:
            issues.append(
                DataQualityIssue(
                    code="required_timestamp_not_assigned",
                    severity="error",
                    summary="A timestamp column is required but not assigned.",
                    explanation="The current data-quality rules requires explicit dates.",
                    suggested_action="Assign an ISO 8601 timestamp column or relax the requirement.",
                )
            )
        else:
            issues.append(
                DataQualityIssue(
                    code="daily_timestep_assumed",
                    severity="warning",
                    summary="Daily spacing is assumed from row order.",
                    explanation="No timestamp column is assigned, so temporal regularity cannot be verified.",
                    suggested_action="Assign a timestamp column when one is available.",
                )
            )
    else:
        labels: list[str] = []
        for index, row in enumerate(rows):
            raw = (row.get(mapping.timestamp) or "").strip()
            labels.append(raw)
            if not raw:
                timestamp_missing[index] = True
                continue
            try:
                parsed_timestamps.append(_parse_timestamp(raw))
            except ValueError:
                timestamp_invalid[index] = True
        timestamp_labels = tuple(labels)

        if np.any(timestamp_missing):
            issues.append(
                DataQualityIssue(
                    code="missing_timestamps",
                    severity="error",
                    summary="The time coordinate contains missing values.",
                    explanation=f"{int(np.count_nonzero(timestamp_missing))} timestamp(s) are empty.",
                    suggested_action="Provide one timestamp for every data row.",
                    variable="timestamp",
                    count=int(np.count_nonzero(timestamp_missing)),
                    csv_rows=_row_examples(timestamp_missing),
                )
            )
        if np.any(timestamp_invalid):
            issues.append(
                DataQualityIssue(
                    code="invalid_timestamps",
                    severity="error",
                    summary="The time coordinate contains invalid timestamps.",
                    explanation=(
                        f"{int(np.count_nonzero(timestamp_invalid))} value(s) are not valid ISO 8601 timestamps."
                    ),
                    suggested_action="Use ISO 8601 dates such as 2026-07-22.",
                    variable="timestamp",
                    count=int(np.count_nonzero(timestamp_invalid)),
                    csv_rows=_row_examples(timestamp_invalid),
                )
            )

        if not np.any(timestamp_missing | timestamp_invalid):
            awareness = {item.tzinfo is not None for item in parsed_timestamps}
            if len(awareness) > 1:
                issues.append(
                    DataQualityIssue(
                        code="inconsistent_timestamp_timezones",
                        severity="error",
                        summary="Timezone conventions change within the file.",
                        explanation="Timezone-aware and timezone-naive timestamps cannot be mixed.",
                        suggested_action="Use one timezone convention throughout the timestamp column.",
                        variable="timestamp",
                    )
                )
            else:
                for previous, current in zip(parsed_timestamps, parsed_timestamps[1:]):
                    if current <= previous:
                        duplicate_or_reversed += 1
                        continue
                    same_local_time = current.timetz().replace(tzinfo=None) == previous.timetz().replace(tzinfo=None)
                    calendar_daily = same_local_time and (current.date() - previous.date()).days == 1
                    exact_daily = current - previous == timedelta(days=1)
                    if not (calendar_daily or exact_daily):
                        irregular_steps += 1
                if duplicate_or_reversed:
                    issues.append(
                        DataQualityIssue(
                            code="nonincreasing_timestamps",
                            severity="error",
                            summary="Timestamps are duplicated or out of order.",
                            explanation=f"{duplicate_or_reversed} adjacent timestamp pair(s) are not increasing.",
                            suggested_action="Sort the data chronologically and remove duplicate timestamps.",
                            variable="timestamp",
                            count=duplicate_or_reversed,
                        )
                    )
                if irregular_steps:
                    issues.append(
                        DataQualityIssue(
                            code="irregular_daily_timestep",
                            severity="error",
                            summary="The time series is not regularly daily.",
                            explanation=f"{irregular_steps} adjacent interval(s) differ from one day.",
                            suggested_action="Restore missing dates or resample explicitly before using a daily model.",
                            variable="timestamp",
                            count=irregular_steps,
                        )
                    )

    first_timestamp = None
    last_timestamp = None
    if timestamp_labels:
        first_timestamp = timestamp_labels[0] or None
        last_timestamp = timestamp_labels[-1] or None
    timestamp_summary = TimestampQualitySummary(
        assigned=mapping.timestamp is not None,
        source_column=mapping.timestamp,
        first=first_timestamp,
        last=last_timestamp,
        valid_count=(
            len(rows) - int(np.count_nonzero(timestamp_missing | timestamp_invalid))
            if mapping.timestamp is not None
            else 0
        ),
        missing_count=int(np.count_nonzero(timestamp_missing)),
        invalid_count=int(np.count_nonzero(timestamp_invalid)),
        duplicate_or_reversed_count=duplicate_or_reversed,
        irregular_step_count=irregular_steps,
        expected_step=processing.timestamps.expected_step,
    )

    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    transformed_count = sum(summary.transformed_count for summary in summaries)
    include_index = len(rows) <= maximum_index_payload
    streamflow = processed.get("streamflow")
    observation_valid = (
        tuple(bool(value) for value in np.isfinite(streamflow))
        if include_index and streamflow is not None
        else None
    )
    correlation_variables = tuple(processed)
    correlation_rows: list[tuple[float | None, ...]] = []
    for left_name in correlation_variables:
        row: list[float | None] = []
        left = processed[left_name]
        for right_name in correlation_variables:
            right = processed[right_name]
            valid = np.isfinite(left) & np.isfinite(right)
            if (
                np.count_nonzero(valid) < 2
                or np.std(left[valid]) == 0.0
                or np.std(right[valid]) == 0.0
            ):
                row.append(None)
            else:
                row.append(float(np.corrcoef(left[valid], right[valid])[0, 1]))
        correlation_rows.append(tuple(row))

    report = DataQualityReport(
        source_path=source.path,
        sha256=content_hash,
        row_count=len(rows),
        ready=error_count == 0,
        error_count=error_count,
        warning_count=warning_count,
        transformed_value_count=transformed_count,
        variables=tuple(summaries),
        timestamps=timestamp_summary,
        issues=tuple(issues),
        timestamp_labels=timestamp_labels if include_index else None,
        observation_valid=observation_valid,
        correlation_variables=correlation_variables,
        correlation_matrix=tuple(correlation_rows),
    )
    return DataQualityAnalysis(
        precipitation=processed["precipitation"],
        temperature=processed["temperature"],
        potential_evapotranspiration=processed["potential_evapotranspiration"],
        streamflow=(
            streamflow
            if streamflow is not None
            else np.full(len(rows), np.nan, dtype=float)
        ),
        additional_predictors={
            name: processed[name] for name in predictor_configs
        },
        timestamps=timestamp_labels,
        source_path=path,
        report=report,
    )
