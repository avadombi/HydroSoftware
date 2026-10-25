"""Bounded snow-parameter treatment for DeepDiscover.

The discovered mapping is optimized with Adam, but the hard degree-day snow
variant contains discontinuous thresholds.  Snow parameters are therefore
updated by a small deterministic coordinate search alternated with the mapping
updates.  This avoids pretending that hard thresholds have useful analytical
gradients while keeping every selected parameter inside its declared bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from hydrosoftware.configuration import (
    DegreeDaySnowConfig,
    NoSnowConfig,
    SmoothDegreeDaySnowConfig,
)
from hydrosoftware.processes.snow import (
    DegreeDaySnowModule,
    NoSnowModule,
    SmoothDegreeDaySnowModule,
    SnowModule,
)


SNOW_COORDINATE_METHOD = "bounded_alternating_coordinate.v1"
SNOW_FIXED_METHOD = "fixed"


@dataclass(frozen=True)
class DiscoverySnowPlan:
    """Initial values, calibrated dimensions, and bounds for one snow module."""

    module_id: str
    initial_parameters: Mapping[str, float]
    calibrated_names: tuple[str, ...]
    bounds: Mapping[str, tuple[float, float]]

    def module(self, parameters: Mapping[str, float]) -> SnowModule:
        return snow_module_from_parameters(self.module_id, parameters)


def discovery_snow_plan(
    config: NoSnowConfig | DegreeDaySnowConfig | SmoothDegreeDaySnowConfig,
) -> DiscoverySnowPlan:
    """Translate the versioned snow configuration into a training plan."""

    if isinstance(config, NoSnowConfig):
        return DiscoverySnowPlan(config.module_id, {}, (), {})

    names = [
        "snowfall_temperature",
        "melt_temperature",
        "degree_day_factor",
    ]
    if isinstance(config, SmoothDegreeDaySnowConfig):
        names.append("sharpness")
    parameters = {name: float(getattr(config, name)) for name in names}
    calibrated = tuple(
        name
        for name in names
        if getattr(config.calibration, name).mode == "calibrate"
    )
    bounds = {
        name: (
            float(getattr(config.calibration, name).lower_bound),
            float(getattr(config.calibration, name).upper_bound),
        )
        for name in calibrated
    }
    return DiscoverySnowPlan(
        module_id=config.module_id,
        initial_parameters=parameters,
        calibrated_names=calibrated,
        bounds=bounds,
    )


def snow_module_from_parameters(
    module_id: str | None,
    parameters: Mapping[str, float],
    *,
    fallback: SnowModule | None = None,
) -> SnowModule:
    """Build the exact frozen snow module stored with a discovery artifact."""

    if module_id is None:
        if fallback is None:
            raise ValueError("A legacy discovery artifact requires its configured snow module.")
        return fallback
    if module_id == "snow.none.v1":
        return NoSnowModule()

    required = (
        "snowfall_temperature",
        "melt_temperature",
        "degree_day_factor",
    )
    missing = [name for name in required if name not in parameters]
    if missing:
        raise ValueError(
            "The discovery artifact is missing snow parameters: "
            + ", ".join(missing)
        )
    shared = {
        name: float(parameters[name])
        for name in required
    }
    if module_id == "snow.degree_day.hard.v1":
        return DegreeDaySnowModule(**shared)
    if module_id == "snow.degree_day.smooth.v1":
        if "sharpness" not in parameters:
            raise ValueError(
                "The smooth snow artifact is missing transition sharpness."
            )
        return SmoothDegreeDaySnowModule(
            **shared,
            sharpness=float(parameters["sharpness"]),
        )
    raise ValueError(f"Unsupported discovery snow module: {module_id!r}.")


def snow_update_interval(epochs: int) -> int:
    """Limit snow-coordinate rounds to at most about fifty per fit."""

    return max(1, int(epochs) // 50)


def coordinate_snow_update(
    plan: DiscoverySnowPlan,
    parameters: Mapping[str, float],
    *,
    round_index: int,
    objective: Callable[[Mapping[str, float]], float],
) -> tuple[dict[str, float], float, bool]:
    """Perform one deterministic bounded coordinate-search round.

    Each coordinate compares its current value with one lower and one upper
    neighbour.  The initial step is ten percent of the declared range and
    narrows gradually, but never below half a percent of that range.
    """

    current = {
        name: float(value)
        for name, value in parameters.items()
    }
    current_loss = float(objective(current))
    if not np.isfinite(current_loss):
        raise ValueError("The current DD snow configuration produced a non-finite loss.")
    changed = False

    for name in plan.calibrated_names:
        lower, upper = plan.bounds[name]
        width = upper - lower
        step = max(0.005 * width, 0.1 * width * (0.92 ** round_index))
        center = current[name]
        candidates = tuple(
            dict.fromkeys(
                float(np.clip(value, lower, upper))
                for value in (center - step, center + step)
                if abs(float(np.clip(value, lower, upper)) - center) > 1e-15
            )
        )
        best_value = center
        best_loss = current_loss
        for candidate in candidates:
            trial = {**current, name: candidate}
            trial_loss = float(objective(trial))
            if np.isfinite(trial_loss) and trial_loss < best_loss - 1e-12:
                best_value = candidate
                best_loss = trial_loss
        if best_value != center:
            current[name] = best_value
            current_loss = best_loss
            changed = True

    return current, current_loss, changed
