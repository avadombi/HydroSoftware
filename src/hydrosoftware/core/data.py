"""Canonical in-memory representation of one lumped-basin time series."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TimeSeriesData:
    """Labelled time series with the canonical shape ``[time, variable]``."""

    values: NDArray[np.floating]
    variable_names: tuple[str, ...]
    timestamps: NDArray | None = None
    units: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError("values must have shape [time, variable].")
        if values.shape[1] != len(self.variable_names):
            raise ValueError("variable_names must match the variable dimension.")
        if len(set(self.variable_names)) != len(self.variable_names):
            raise ValueError("variable_names must be unique.")
        if self.timestamps is not None:
            timestamps = np.asarray(self.timestamps)
            if timestamps.ndim != 1 or len(timestamps) != values.shape[0]:
                raise ValueError("timestamps must have shape [time].")
            if len(timestamps) > 1 and np.any(timestamps[1:] <= timestamps[:-1]):
                raise ValueError("timestamps must be strictly increasing.")
            object.__setattr__(self, "timestamps", timestamps)
        unknown_units = set(self.units).difference(self.variable_names)
        if unknown_units:
            raise ValueError(f"Units refer to unknown variables: {sorted(unknown_units)}")
        object.__setattr__(self, "values", values)

    @property
    def n_time(self) -> int:
        return self.values.shape[0]

    def select(self, names: Sequence[str]) -> NDArray[np.floating]:
        """Return selected variables while preserving the requested order."""

        indices = []
        for name in names:
            try:
                indices.append(self.variable_names.index(name))
            except ValueError as exc:
                raise KeyError(f"Unknown variable: {name}") from exc
        return self.values[:, indices]

