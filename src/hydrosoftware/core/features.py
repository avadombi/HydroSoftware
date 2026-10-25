"""Causal, serializable feature recipes for neural and PaML models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LaggedFeature:
    """A deterministic shifted copy of one source channel."""

    source: str
    lag_steps: int

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("Lagged feature sources cannot be blank.")
        if self.lag_steps < 1:
            raise ValueError("Lagged feature steps must be positive.")

    @property
    def name(self) -> str:
        return lagged_feature_name(self.source, self.lag_steps)


def lagged_feature_name(source: str, lag_steps: int) -> str:
    """Return the stable channel name used in artifacts and exports."""

    return f"{source}__lag_{lag_steps}"


def maximum_lag(lagged_features: Sequence[LaggedFeature]) -> int:
    """Return the longest declared causal shift."""

    return max((item.lag_steps for item in lagged_features), default=0)


def required_raw_sources(
    feature_roles: Sequence[str],
    lagged_features: Sequence[LaggedFeature],
) -> tuple[str, ...]:
    """List raw source channels in deterministic first-use order."""

    return tuple(
        dict.fromkeys(
            [*feature_roles, *(item.source for item in lagged_features)]
        )
    )


def _validated_raw_mapping(
    raw_features: Mapping[str, Sequence[float]],
    required: Sequence[str],
) -> tuple[dict[str, NDArray[np.floating]], int]:
    missing = [name for name in required if name not in raw_features]
    if missing:
        raise ValueError(
            "Required feature channels are unavailable: " + ", ".join(missing)
        )
    arrays = {
        name: np.asarray(raw_features[name], dtype=float)
        for name in required
    }
    if any(values.ndim != 1 for values in arrays.values()):
        raise ValueError("Feature channels must have shape [time].")
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) > 1:
        raise ValueError("Feature channels must share the time dimension.")
    return arrays, next(iter(lengths), 0)


def build_feature_mapping(
    raw_features: Mapping[str, Sequence[float]],
    feature_roles: Sequence[str],
    lagged_features: Sequence[LaggedFeature],
) -> dict[str, NDArray[np.floating]]:
    """Build base and shifted channels without fitting data-dependent state."""

    required = required_raw_sources(feature_roles, lagged_features)
    arrays, length = _validated_raw_mapping(raw_features, required)
    result = {name: arrays[name].copy() for name in feature_roles}
    for recipe in lagged_features:
        shifted = np.full(length, np.nan, dtype=float)
        if recipe.lag_steps < length:
            shifted[recipe.lag_steps:] = arrays[recipe.source][:-recipe.lag_steps]
        result[recipe.name] = shifted
    return result


def build_forecast_feature_mapping(
    current_raw_features: Mapping[str, Sequence[float]],
    feature_roles: Sequence[str],
    lagged_features: Sequence[LaggedFeature],
    *,
    lookback: int,
    antecedent_raw_features: Mapping[str, Sequence[float]] | None = None,
) -> tuple[
    dict[str, NDArray[np.floating]],
    dict[str, NDArray[np.floating]] | None,
]:
    """Build forecast channels and the expanded antecedent lookback.

    Raw antecedent rows are concatenated before shifts are applied. This is
    essential for a lagged feature at the first forecast interval: it must use
    the archived past value, not a missing placeholder or future information.
    """

    required = required_raw_sources(feature_roles, lagged_features)
    current, current_length = _validated_raw_mapping(
        current_raw_features, required
    )
    if antecedent_raw_features is None:
        return (
            build_feature_mapping(current, feature_roles, lagged_features),
            None,
        )

    antecedent, antecedent_length = _validated_raw_mapping(
        antecedent_raw_features, required
    )
    combined = {
        name: np.concatenate((antecedent[name], current[name]))
        for name in required
    }
    expanded = build_feature_mapping(combined, feature_roles, lagged_features)
    current_expanded = {
        name: values[antecedent_length:antecedent_length + current_length]
        for name, values in expanded.items()
    }
    antecedent_expanded = {
        name: values[max(0, antecedent_length - lookback):antecedent_length]
        for name, values in expanded.items()
    }
    return current_expanded, antecedent_expanded


def feature_pipeline_mapping(
    feature_roles: Sequence[str],
    lagged_features: Sequence[LaggedFeature],
) -> dict[str, object]:
    """Return a JSON-ready description stored beside learned artifacts."""

    return {
        "base_features": tuple(feature_roles),
        "lagged_features": tuple(
            {
                "source": item.source,
                "lag_steps": item.lag_steps,
                "name": item.name,
            }
            for item in lagged_features
        ),
        "expanded_features": tuple(
            [*feature_roles, *(item.name for item in lagged_features)]
        ),
        "maximum_lag": maximum_lag(lagged_features),
        "causal": True,
    }
