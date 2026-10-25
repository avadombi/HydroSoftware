"""Interchangeable reference snow modules with exact stepwise water balance."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log1p
from typing import Protocol


@dataclass(frozen=True)
class SnowStepResult:
    """Snow-process fluxes and state for one forcing interval."""

    rain: float
    snowfall: float
    melt: float
    storage_before: float
    storage_after: float

    @property
    def balance_error(self) -> float:
        return (
            self.storage_before
            + self.snowfall
            - self.melt
            - self.storage_after
        )


class SnowModule(Protocol):
    """Protocol implemented by all snow-process variants."""

    module_id: str

    def step(
        self,
        precipitation: float,
        temperature: float,
        storage: float,
        *,
        dt: float = 1.0,
    ) -> SnowStepResult:
        """Advance the snow state by one forcing interval."""


def _validate_inputs(
    precipitation: float, temperature: float, storage: float, dt: float
) -> None:
    values = (precipitation, temperature, storage, dt)
    if not all(float("-inf") < value < float("inf") for value in values):
        raise ValueError("Snow-module inputs must be finite.")
    if precipitation < 0:
        raise ValueError("Precipitation cannot be negative.")
    if storage < 0:
        raise ValueError("Snow storage cannot be negative.")
    if dt <= 0:
        raise ValueError("dt must be positive.")


@dataclass(frozen=True)
class NoSnowModule:
    """Bypass snow processes and route all precipitation as rain."""

    module_id: str = "snow.none.v1"

    def step(
        self,
        precipitation: float,
        temperature: float,
        storage: float = 0.0,
        *,
        dt: float = 1.0,
    ) -> SnowStepResult:
        _validate_inputs(precipitation, temperature, storage, dt)
        if storage != 0:
            raise ValueError("NoSnowModule requires zero snow storage.")
        return SnowStepResult(
            rain=precipitation,
            snowfall=0.0,
            melt=0.0,
            storage_before=0.0,
            storage_after=0.0,
        )


@dataclass(frozen=True)
class DegreeDaySnowModule:
    """Hard-threshold degree-day snow module used by conceptual models."""

    snowfall_temperature: float
    melt_temperature: float
    degree_day_factor: float
    module_id: str = "snow.degree_day.hard.v1"

    def __post_init__(self) -> None:
        if self.degree_day_factor < 0:
            raise ValueError("degree_day_factor cannot be negative.")

    def step(
        self,
        precipitation: float,
        temperature: float,
        storage: float,
        *,
        dt: float = 1.0,
    ) -> SnowStepResult:
        _validate_inputs(precipitation, temperature, storage, dt)
        snowfall = precipitation if temperature < self.snowfall_temperature else 0.0
        rain = precipitation - snowfall
        positive_degree = max(temperature - self.melt_temperature, 0.0)
        potential_melt = self.degree_day_factor * positive_degree * dt
        # The historical EXP-HYDRO ordering melts antecedent storage only.
        melt = min(storage, potential_melt)
        storage_after = storage + snowfall - melt
        return SnowStepResult(
            rain=rain,
            snowfall=snowfall,
            melt=melt,
            storage_before=storage,
            storage_after=storage_after,
        )


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _softplus(value: float) -> float:
    return max(value, 0.0) + log1p(exp(-abs(value)))


@dataclass(frozen=True)
class SmoothDegreeDaySnowModule:
    """Smooth-threshold degree-day module for differentiable model adapters.

    Rain is computed as the exact complement of snowfall. Positive degree-days
    use softplus, which avoids the negative-melt bug in the historical Keras
    implementation when temperature is below the melt threshold.
    """

    snowfall_temperature: float
    melt_temperature: float
    degree_day_factor: float
    sharpness: float = 5.0
    module_id: str = "snow.degree_day.smooth.v1"

    def __post_init__(self) -> None:
        if self.degree_day_factor < 0:
            raise ValueError("degree_day_factor cannot be negative.")
        if self.sharpness <= 0:
            raise ValueError("sharpness must be positive.")

    def step(
        self,
        precipitation: float,
        temperature: float,
        storage: float,
        *,
        dt: float = 1.0,
    ) -> SnowStepResult:
        _validate_inputs(precipitation, temperature, storage, dt)
        snow_fraction = _sigmoid(
            self.sharpness * (self.snowfall_temperature - temperature)
        )
        snowfall = precipitation * snow_fraction
        rain = precipitation - snowfall
        positive_degree = _softplus(
            self.sharpness * (temperature - self.melt_temperature)
        ) / self.sharpness
        potential_melt = self.degree_day_factor * positive_degree * dt
        melt = min(storage, potential_melt)
        storage_after = storage + snowfall - melt
        return SnowStepResult(
            rain=rain,
            snowfall=snowfall,
            melt=melt,
            storage_before=storage,
            storage_after=storage_after,
        )

