"""Strict CSV adapter for the first EXP-HYDRO calibration workflow."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from hydrosoftware.configuration import CalibrationPeriodConfig, CsvDataSourceConfig
from hydrosoftware.data_sources.errors import DataSourceError
from hydrosoftware.data_sources.quality import analyze_calibration_csv


@dataclass(frozen=True)
class CsvCalibrationData:
    """Standard arrays required by one EXP-HYDRO calibration."""

    precipitation: NDArray[np.floating]
    temperature: NDArray[np.floating]
    potential_evapotranspiration: NDArray[np.floating]
    streamflow: NDArray[np.floating]
    timestamps: tuple[str, ...] | None
    source_path: Path

    def __post_init__(self) -> None:
        arrays = (
            np.asarray(self.precipitation, dtype=float),
            np.asarray(self.temperature, dtype=float),
            np.asarray(self.potential_evapotranspiration, dtype=float),
            np.asarray(self.streamflow, dtype=float),
        )
        lengths = {len(array) for array in arrays}
        if len(lengths) != 1:
            raise ValueError("Calibration arrays must share the time dimension.")
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("Calibration arrays must have shape [time].")
        if self.timestamps is not None and len(self.timestamps) != len(arrays[0]):
            raise ValueError("timestamps must share the time dimension.")
        object.__setattr__(self, "precipitation", arrays[0])
        object.__setattr__(self, "temperature", arrays[1])
        object.__setattr__(self, "potential_evapotranspiration", arrays[2])
        object.__setattr__(self, "streamflow", arrays[3])

    @property
    def n_time(self) -> int:
        return len(self.precipitation)


@dataclass(frozen=True)
class CsvPreview:
    """Small, safe CSV summary used by the data-import interface."""

    relative_path: str
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, str], ...]
    row_count: int
    file_size_bytes: int
    sha256: str


def _resolve_workspace_file(relative_path: str, *, workspace_root: Path) -> Path:
    """Resolve a declared path without allowing access outside the workspace."""

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


def _resolve_source_path(
    source: CsvDataSourceConfig,
    *,
    workspace_root: Path,
) -> Path:
    return _resolve_workspace_file(source.path, workspace_root=workspace_root)


def preview_csv(
    relative_path: str,
    *,
    workspace_root: str | Path,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    preview_rows: int = 8,
) -> CsvPreview:
    """Inspect a workspace CSV without assigning hydrological variable roles."""

    if len(delimiter) != 1:
        raise DataSourceError(
            "invalid_csv_delimiter",
            "The CSV delimiter must contain exactly one character.",
        )
    if not 1 <= preview_rows <= 100:
        raise ValueError("preview_rows must lie in [1, 100].")
    path = _resolve_workspace_file(relative_path, workspace_root=Path(workspace_root))
    try:
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode(encoding)
    except LookupError as exc:
        raise DataSourceError(
            "unknown_text_encoding",
            f"The declared CSV encoding is not recognized: {encoding}",
        ) from exc
    except UnicodeDecodeError as exc:
        raise DataSourceError(
            "csv_decoding_failed",
            f"The CSV file cannot be decoded using {encoding}.",
        ) from exc

    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    if reader.fieldnames is None:
        raise DataSourceError("missing_csv_header", "The CSV file has no header row.")
    columns = tuple(reader.fieldnames)
    if not columns or any(not column.strip() for column in columns):
        raise DataSourceError(
            "empty_csv_column",
            "Every CSV column must have a non-empty name.",
        )
    if len(set(columns)) != len(columns):
        raise DataSourceError(
            "duplicate_csv_columns", "The CSV header contains duplicate column names."
        )

    sample: list[Mapping[str, str]] = []
    row_count = 0
    for row in reader:
        row_count += 1
        if len(sample) < preview_rows:
            sample.append({column: row.get(column, "") or "" for column in columns})
    if row_count == 0:
        raise DataSourceError("empty_data_file", "The CSV file contains no data rows.")

    return CsvPreview(
        relative_path=relative_path.replace("\\", "/"),
        columns=columns,
        rows=tuple(sample),
        row_count=row_count,
        file_size_bytes=len(raw_bytes),
        sha256=sha256(raw_bytes).hexdigest(),
    )


def import_csv_bytes(
    filename: str,
    content: bytes,
    *,
    workspace_root: str | Path,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    maximum_size_bytes: int = 25 * 1024 * 1024,
) -> CsvPreview:
    """Store an uploaded CSV inside the workspace and return its preview."""

    if not content:
        raise DataSourceError("empty_upload", "The selected CSV file is empty.")
    if len(content) > maximum_size_bytes:
        raise DataSourceError(
            "csv_file_too_large",
            f"The selected CSV exceeds the {maximum_size_bytes // (1024 * 1024)} MB limit.",
        )
    basename = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(basename).stem).strip("-.")
    suffix = Path(basename).suffix.lower()
    if suffix != ".csv":
        raise DataSourceError(
            "unsupported_data_file",
            "The first interface accepts CSV files only.",
        )
    if not stem:
        stem = "imported-data"

    root = Path(workspace_root).resolve()
    destination_directory = root / "data" / "imports"
    destination_directory.mkdir(parents=True, exist_ok=True)
    digest = sha256(content).hexdigest()
    destination = destination_directory / f"{stem}-{digest[:10]}.csv"
    relative_path = destination.relative_to(root).as_posix()

    if destination.exists():
        if sha256(destination.read_bytes()).hexdigest() != digest:
            raise DataSourceError(
                "import_hash_collision",
                "An imported data file has an unexpected content conflict.",
            )
    else:
        temporary = destination.with_suffix(".csv.part")
        try:
            temporary.write_bytes(content)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    try:
        return preview_csv(
            relative_path,
            workspace_root=root,
            delimiter=delimiter,
            encoding=encoding,
        )
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def load_calibration_csv(
    source: CsvDataSourceConfig,
    *,
    workspace_root: str | Path,
) -> CsvCalibrationData:
    """Load one CSV after applying its recorded quality and transformation rules."""

    analysis = analyze_calibration_csv(source, workspace_root=workspace_root)
    if not analysis.report.ready:
        first_error = next(
            issue for issue in analysis.report.issues if issue.severity == "error"
        )
        raise DataSourceError(first_error.code, first_error.explanation)
    return CsvCalibrationData(
        precipitation=analysis.precipitation,
        temperature=analysis.temperature,
        potential_evapotranspiration=analysis.potential_evapotranspiration,
        streamflow=analysis.streamflow,
        timestamps=analysis.timestamps,
        source_path=analysis.source_path,
    )


def select_calibration_period(
    data: CsvCalibrationData,
    period: CalibrationPeriodConfig,
) -> CsvCalibrationData:
    """Select the continuous simulation interval spanning every partition."""

    if period.protocol == "legacy_calibration_only":
        end = period.calibration_end_index or data.n_time
    else:
        end = period.test_end_index or data.n_time
    if period.start_index >= data.n_time:
        raise DataSourceError(
            "period_starts_after_data",
            (
                f"Calibration start_index {period.start_index} is outside a data set "
                f"with {data.n_time} rows."
            ),
        )
    if end > data.n_time:
        raise DataSourceError(
            "period_ends_after_data",
            (
                f"The selected simulation end {end} exceeds the data set length "
                f"of {data.n_time} rows."
            ),
        )
    calibration_end = period.calibration_end_index or end
    if calibration_end > end:
        raise DataSourceError(
            "calibration_ends_after_period",
            "The calibration boundary exceeds the selected simulation period.",
        )
    if (
        period.validation_end_index is not None
        and period.validation_end_index > end
    ):
        raise DataSourceError(
            "validation_ends_after_period",
            "The validation boundary exceeds the selected simulation period.",
        )
    if period.warmup_steps >= calibration_end - period.start_index:
        raise DataSourceError(
            "warmup_covers_period",
            "The warm-up must be shorter than the calibration partition.",
        )
    interval = slice(period.start_index, end)
    timestamps = None if data.timestamps is None else data.timestamps[interval]
    return CsvCalibrationData(
        precipitation=data.precipitation[interval],
        temperature=data.temperature[interval],
        potential_evapotranspiration=data.potential_evapotranspiration[interval],
        streamflow=data.streamflow[interval],
        timestamps=timestamps,
        source_path=data.source_path,
    )
