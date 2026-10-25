"""CAMELS-US time-series preparation and performance-independent basin selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import csv
import hashlib
import io
import math
from pathlib import Path, PurePosixPath
import re
from typing import Iterable, Mapping
import zipfile

import numpy as np


PROTOCOL_START = date(1980, 10, 1)
WARMUP_END = date(1981, 10, 1)
TRAINING_END = date(1995, 10, 1)
VALIDATION_END = date(2000, 10, 1)
PROTOCOL_END = date(2014, 10, 1)
SELECTION_AXES = (
    "aridity",
    "frac_snow",
    "p_seasonality",
    "slope_mean",
    "area_gages2",
    "soil_conductivity",
)
FORCING_SUFFIX = "_lump_cida_forcing_leap.txt"
STREAMFLOW_SUFFIX = "_streamflow_qc.txt"


@dataclass(frozen=True)
class CamelsForcing:
    basin_area_m2: int
    dates: tuple[date, ...]
    precipitation: np.ndarray
    temperature: np.ndarray
    day_length: np.ndarray
    solar_radiation: np.ndarray
    vapor_pressure: np.ndarray
    potential_evapotranspiration: np.ndarray


@dataclass(frozen=True)
class CamelsBasinSeries:
    gauge_id: str
    huc_02: str
    basin_area_m2: int
    dates: tuple[date, ...]
    precipitation: np.ndarray
    temperature: np.ndarray
    day_length: np.ndarray
    solar_radiation: np.ndarray
    vapor_pressure: np.ndarray
    potential_evapotranspiration: np.ndarray
    streamflow: np.ndarray


@dataclass(frozen=True)
class CamelsEligibility:
    gauge_id: str
    huc_02: str
    forcing_complete: bool
    calibration_finite_fraction: float
    validation_finite_fraction: float
    test_finite_fraction: float
    maximum_missing_run_days: int

    @property
    def eligible(self) -> bool:
        return (
            self.forcing_complete
            and min(
                self.calibration_finite_fraction,
                self.validation_finite_fraction,
                self.test_finite_fraction,
            )
            >= 0.95
            and self.maximum_missing_run_days <= 30
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _daily_dates(start: date, end: date) -> tuple[date, ...]:
    return tuple(start + timedelta(days=offset) for offset in range((end - start).days))


def hamon_potential_evapotranspiration(
    temperature: np.ndarray,
    day_length: np.ndarray,
) -> np.ndarray:
    """Return daily Hamon PET for temperature in degC and day length in days."""

    temperature = np.asarray(temperature, dtype=float)
    day_length = np.asarray(day_length, dtype=float)
    return (
        29.8
        * (day_length * 24.0)
        * 0.611
        * np.exp(17.3 * temperature / (temperature + 237.3))
        / (temperature + 273.2)
    )


def parse_forcing_text(text: str) -> CamelsForcing:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
        raise ValueError("CAMELS forcing file is incomplete.")
    try:
        basin_area_m2 = int(lines[2])
    except ValueError as exc:
        raise ValueError("The third forcing-header line must contain basin area in m2.") from exc
    if basin_area_m2 <= 0:
        raise ValueError("Basin area must be positive.")

    header = lines[3].split()
    required = {
        "Year",
        "Mnth",
        "Day",
        "dayl(s)",
        "prcp(mm/day)",
        "srad(W/m2)",
        "tmax(C)",
        "tmin(C)",
        "vp(Pa)",
    }
    if not required.issubset(header):
        raise ValueError(f"Forcing header is missing {sorted(required - set(header))}.")
    positions = {name: header.index(name) for name in required}
    dates: list[date] = []
    precipitation: list[float] = []
    maximum_temperature: list[float] = []
    minimum_temperature: list[float] = []
    day_length_seconds: list[float] = []
    solar_radiation: list[float] = []
    vapor_pressure: list[float] = []
    for line_number, line in enumerate(lines[4:], start=5):
        fields = line.split()
        if len(fields) != len(header):
            raise ValueError(
                f"Forcing line {line_number} contains {len(fields)} fields; expected {len(header)}."
            )
        try:
            current_date = date(
                int(fields[positions["Year"]]),
                int(fields[positions["Mnth"]]),
                int(fields[positions["Day"]]),
            )
            tmax = float(fields[positions["tmax(C)"]])
            tmin = float(fields[positions["tmin(C)"]])
            dates.append(current_date)
            precipitation.append(float(fields[positions["prcp(mm/day)"]]))
            maximum_temperature.append(tmax)
            minimum_temperature.append(tmin)
            day_length_seconds.append(float(fields[positions["dayl(s)"]]))
            solar_radiation.append(float(fields[positions["srad(W/m2)"]]))
            vapor_pressure.append(float(fields[positions["vp(Pa)"]]))
        except ValueError as exc:
            raise ValueError(f"Forcing line {line_number} is invalid.") from exc

    if len(set(dates)) != len(dates):
        raise ValueError("Forcing dates must be unique.")
    if any(right <= left for left, right in zip(dates, dates[1:])):
        raise ValueError("Forcing dates must be strictly increasing.")
    temperature_array = (
        np.asarray(minimum_temperature, dtype=float)
        + np.asarray(maximum_temperature, dtype=float)
    ) / 2.0
    day_length_array = np.asarray(day_length_seconds, dtype=float) / 86400.0
    return CamelsForcing(
        basin_area_m2=basin_area_m2,
        dates=tuple(dates),
        precipitation=np.asarray(precipitation, dtype=float),
        temperature=temperature_array,
        day_length=day_length_array,
        solar_radiation=np.asarray(solar_radiation, dtype=float),
        vapor_pressure=np.asarray(vapor_pressure, dtype=float),
        potential_evapotranspiration=hamon_potential_evapotranspiration(
            temperature_array,
            day_length_array,
        ),
    )


def parse_streamflow_text(text: str, *, basin_area_m2: int) -> dict[date, float]:
    if basin_area_m2 <= 0:
        raise ValueError("Basin area must be positive.")
    dates: list[date] = []
    discharge: list[float] = []
    seen: set[date] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Streamflow line {line_number} is incomplete.")
        try:
            current_date = date(int(fields[1]), int(fields[2]), int(fields[3]))
            discharge_cfs = float(fields[4])
        except ValueError as exc:
            raise ValueError(f"Streamflow line {line_number} is invalid.") from exc
        if current_date in seen:
            raise ValueError(f"Duplicate streamflow date {current_date.isoformat()}.")
        seen.add(current_date)
        dates.append(current_date)
        discharge.append(discharge_cfs)
    discharge_array = np.asarray(discharge, dtype=float)
    converted = (
        28_316_846.592
        * discharge_array
        * 86400.0
        / (basin_area_m2 * 1_000_000.0)
    )
    return dict(zip(dates, converted.tolist(), strict=True))


def assemble_basin_series(
    *,
    gauge_id: str,
    huc_02: str,
    forcing_text: str,
    streamflow_text: str,
) -> CamelsBasinSeries:
    forcing = parse_forcing_text(forcing_text)
    streamflow_by_date = parse_streamflow_text(
        streamflow_text,
        basin_area_m2=forcing.basin_area_m2,
    )
    streamflow = np.asarray(
        [streamflow_by_date.get(current_date, float("nan")) for current_date in forcing.dates],
        dtype=float,
    )
    return CamelsBasinSeries(
        gauge_id=gauge_id.zfill(8),
        huc_02=huc_02.zfill(2),
        basin_area_m2=forcing.basin_area_m2,
        dates=forcing.dates,
        precipitation=forcing.precipitation,
        temperature=forcing.temperature,
        day_length=forcing.day_length,
        solar_radiation=forcing.solar_radiation,
        vapor_pressure=forcing.vapor_pressure,
        potential_evapotranspiration=forcing.potential_evapotranspiration,
        streamflow=streamflow,
    )


def _finite_fraction(values: np.ndarray) -> float:
    return float(np.count_nonzero(np.isfinite(values)) / values.size) if values.size else 0.0


def _maximum_missing_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for finite in np.isfinite(values):
        if finite:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def evaluate_eligibility(series: CamelsBasinSeries) -> CamelsEligibility:
    expected_dates = _daily_dates(PROTOCOL_START, PROTOCOL_END)
    position = {current_date: index for index, current_date in enumerate(series.dates)}
    forcing_complete = all(current_date in position for current_date in expected_dates)
    if forcing_complete:
        indices = np.asarray([position[current_date] for current_date in expected_dates], dtype=int)
        forcing_complete = bool(
            np.all(np.isfinite(series.precipitation[indices]))
            and np.all(np.isfinite(series.temperature[indices]))
            and np.all(np.isfinite(series.potential_evapotranspiration[indices]))
            and np.all(series.precipitation[indices] >= 0.0)
            and np.all(series.potential_evapotranspiration[indices] >= 0.0)
        )
        protocol_flow = series.streamflow[indices]
    else:
        protocol_flow = np.full(len(expected_dates), np.nan, dtype=float)

    warmup_days = (WARMUP_END - PROTOCOL_START).days
    training_days = (TRAINING_END - WARMUP_END).days
    validation_days = (VALIDATION_END - TRAINING_END).days
    scored_flow = protocol_flow[warmup_days:].copy()
    scored_flow[scored_flow < 0.0] = np.nan
    calibration = scored_flow[:training_days]
    validation = scored_flow[training_days : training_days + validation_days]
    test = scored_flow[training_days + validation_days :]
    return CamelsEligibility(
        gauge_id=series.gauge_id,
        huc_02=series.huc_02,
        forcing_complete=forcing_complete,
        calibration_finite_fraction=_finite_fraction(calibration),
        validation_finite_fraction=_finite_fraction(validation),
        test_finite_fraction=_finite_fraction(test),
        maximum_missing_run_days=_maximum_missing_run(scored_flow),
    )


def write_basin_csv(path: Path, series: CamelsBasinSeries) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "date",
                "prcp(mm/day)",
                "tmean(C)",
                "dayl(day)",
                "srad(W/m2)",
                "vp(Pa)",
                "flow(mm)",
                "PET",
            )
        )
        for index, current_date in enumerate(series.dates):
            flow = series.streamflow[index]
            writer.writerow(
                (
                    current_date.isoformat(),
                    series.precipitation[index],
                    series.temperature[index],
                    series.day_length[index],
                    series.solar_radiation[index],
                    series.vapor_pressure[index],
                    "" if not np.isfinite(flow) else flow,
                    series.potential_evapotranspiration[index],
                )
            )
    return sha256(path)


def read_attribute_table(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if not reader.fieldnames or "gauge_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a gauge_id column.")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            gauge_id = str(row["gauge_id"]).strip().zfill(8)
            if gauge_id in rows:
                raise ValueError(f"Duplicate gauge_id {gauge_id} in {path}.")
            rows[gauge_id] = {str(key): str(value).strip() for key, value in row.items()}
    return rows


def _finite_attribute(value: str, *, name: str, gauge_id: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric for basin {gauge_id}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite for basin {gauge_id}.")
    return result


def _midrank_percentiles(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) < 2:
        raise ValueError("At least two eligible basins are required.")
    output: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        percentile = (0.5 * (start + end - 1)) / (len(ordered) - 1)
        for index in range(start, end):
            output[ordered[index][0]] = percentile
        start = end
    return output


def _distance(left: Iterable[float], right: Iterable[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def select_contrasting_basins(
    *,
    climate: Mapping[str, Mapping[str, str]],
    topography: Mapping[str, Mapping[str, str]],
    soil: Mapping[str, Mapping[str, str]],
    eligibility: Mapping[str, CamelsEligibility],
    pilot: str = "01013500",
    additional_count: int = 5,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, float]]:
    pilot = pilot.zfill(8)
    common_ids = set(climate) & set(topography) & set(soil) & set(eligibility)
    eligible_ids = sorted(gauge_id for gauge_id in common_ids if eligibility[gauge_id].eligible)
    if pilot not in eligible_ids:
        raise ValueError(f"Pilot basin {pilot} is not eligible under the frozen rules.")

    raw: dict[str, dict[str, float]] = {}
    regions: dict[str, str] = {}
    for gauge_id in eligible_ids:
        merged = {**climate[gauge_id], **topography[gauge_id], **soil[gauge_id]}
        missing_axes = set(SELECTION_AXES) - set(merged)
        if missing_axes:
            raise ValueError(f"Basin {gauge_id} is missing attributes {sorted(missing_axes)}.")
        raw[gauge_id] = {
            name: _finite_attribute(merged[name], name=name, gauge_id=gauge_id)
            for name in SELECTION_AXES
        }
        regions[gauge_id] = eligibility[gauge_id].huc_02

    ranked: dict[str, dict[str, float]] = {gauge_id: {} for gauge_id in eligible_ids}
    for axis in SELECTION_AXES:
        axis_ranks = _midrank_percentiles(
            {gauge_id: raw[gauge_id][axis] for gauge_id in eligible_ids}
        )
        for gauge_id, value in axis_ranks.items():
            ranked[gauge_id][axis] = value

    selected = [pilot]
    selected_regions = {regions[pilot]}
    selection_distance: dict[str, float] = {pilot: 0.0}
    while len(selected) < additional_count + 1:
        candidates: list[tuple[float, str]] = []
        for gauge_id in eligible_ids:
            if gauge_id in selected or regions[gauge_id] in selected_regions:
                continue
            vector = [ranked[gauge_id][axis] for axis in SELECTION_AXES]
            minimum = min(
                _distance(vector, [ranked[chosen][axis] for axis in SELECTION_AXES])
                for chosen in selected
            )
            candidates.append((minimum, gauge_id))
        if not candidates:
            raise ValueError("Not enough eligible unique HUC-02 regions to finish selection.")
        best_distance = max(distance for distance, _ in candidates)
        chosen = min(
            gauge_id
            for distance, gauge_id in candidates
            if math.isclose(distance, best_distance, rel_tol=0.0, abs_tol=1e-15)
        )
        selected.append(chosen)
        selected_regions.add(regions[chosen])
        selection_distance[chosen] = best_distance
    return selected, ranked, selection_distance


def _huc_from_member(member: str) -> str:
    parts = PurePosixPath(member).parts[:-1]
    for part in reversed(parts):
        if re.fullmatch(r"\d{2}", part):
            return part
    raise ValueError(f"Cannot resolve HUC-02 from {member}.")


class CamelsTimeSeriesCollection:
    """Read CAMELS-US forcing and flow members directly from a ZIP or directory."""

    def __init__(self, source: Path):
        self.source = source
        self._archive: zipfile.ZipFile | None = None
        if source.is_file() and zipfile.is_zipfile(source):
            self._archive = zipfile.ZipFile(source)
            members = [name.replace("\\", "/") for name in self._archive.namelist()]
        elif source.is_dir():
            members = [path.relative_to(source).as_posix() for path in source.rglob("*.txt")]
        else:
            raise ValueError("The CAMELS time-series source must be a ZIP archive or directory.")

        forcing: dict[str, str] = {}
        streamflow: dict[str, str] = {}
        huc: dict[str, str] = {}
        for member in members:
            name = PurePosixPath(member).name
            if name.endswith(FORCING_SUFFIX):
                gauge_id = name[: -len(FORCING_SUFFIX)].zfill(8)
                if gauge_id in forcing:
                    raise ValueError(f"Duplicate Daymet forcing for basin {gauge_id}.")
                forcing[gauge_id] = member
                huc[gauge_id] = _huc_from_member(member)
            elif name.endswith(STREAMFLOW_SUFFIX):
                gauge_id = name[: -len(STREAMFLOW_SUFFIX)].zfill(8)
                if gauge_id in streamflow:
                    raise ValueError(f"Duplicate streamflow for basin {gauge_id}.")
                streamflow[gauge_id] = member
                huc.setdefault(gauge_id, _huc_from_member(member))
        self._forcing = forcing
        self._streamflow = streamflow
        self._huc = huc

    def __enter__(self) -> "CamelsTimeSeriesCollection":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    @property
    def gauge_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._forcing) & set(self._streamflow)))

    def huc_02(self, gauge_id: str) -> str:
        return self._huc[gauge_id.zfill(8)]

    def _read_text(self, member: str) -> str:
        if self._archive is not None:
            with self._archive.open(member) as stream:
                return io.TextIOWrapper(stream, encoding="utf-8").read()
        return (self.source / member).read_text(encoding="utf-8")

    def read_basin(self, gauge_id: str) -> CamelsBasinSeries:
        gauge_id = gauge_id.zfill(8)
        if gauge_id not in self._forcing or gauge_id not in self._streamflow:
            raise KeyError(f"Basin {gauge_id} lacks Daymet forcing or USGS streamflow.")
        return assemble_basin_series(
            gauge_id=gauge_id,
            huc_02=self._huc[gauge_id],
            forcing_text=self._read_text(self._forcing[gauge_id]),
            streamflow_text=self._read_text(self._streamflow[gauge_id]),
        )
