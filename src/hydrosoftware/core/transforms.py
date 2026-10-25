"""Train-fitted transformations that preserve feature-axis semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Standardizer:
    """Feature-wise standardization fitted on valid training rows only."""

    mean: NDArray[np.floating]
    scale: NDArray[np.floating]

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        if mean.ndim != 1 or scale.ndim != 1 or mean.shape != scale.shape:
            raise ValueError("mean and scale must be matching one-dimensional arrays.")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("Standardizer statistics must be finite.")
        if np.any(scale <= 0):
            raise ValueError("Standardizer scales must be positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(
        cls,
        training_values: NDArray,
        *,
        valid_rows: NDArray | None = None,
        epsilon: float = 1e-12,
    ) -> "Standardizer":
        """Fit one statistic per feature from a continuous training matrix."""

        values = np.asarray(training_values, dtype=float)
        if values.ndim != 2:
            raise ValueError("training_values must have shape [time, feature].")
        if valid_rows is not None:
            mask = np.asarray(valid_rows, dtype=bool)
            if mask.shape != (len(values),):
                raise ValueError("valid_rows must have shape [time].")
            values = values[mask]
        if len(values) == 0:
            raise ValueError("No valid training rows are available.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Fit data must be finite after missing-value handling.")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale <= epsilon, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: NDArray) -> NDArray[np.floating]:
        """Transform continuous or windowed arrays along their last axis."""

        array = np.asarray(values, dtype=float)
        if array.ndim not in (2, 3):
            raise ValueError("values must have shape [time, feature] or [sample, time, feature].")
        if array.shape[-1] != len(self.mean):
            raise ValueError("Feature dimension does not match fitted statistics.")
        return (array - self.mean) / self.scale

    def inverse_transform(self, values: NDArray) -> NDArray[np.floating]:
        """Reverse standardization without changing the input rank."""

        array = np.asarray(values, dtype=float)
        if array.ndim not in (2, 3):
            raise ValueError("values must have shape [time, feature] or [sample, time, feature].")
        if array.shape[-1] != len(self.mean):
            raise ValueError("Feature dimension does not match fitted statistics.")
        return array * self.scale + self.mean

