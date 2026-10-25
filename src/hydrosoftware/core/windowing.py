"""Explicit, testable construction of S2S and S2V training samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .specifications import PredictionLayout


@dataclass(frozen=True)
class WindowSpec:
    """Alignment of input and target windows within one data partition.

    ``target_offset`` is measured from the first input index. This makes the
    same-time S2S layout used by DeepDiscover explicit instead of conflating it
    with future-horizon forecasting.
    """

    layout: PredictionLayout
    input_length: int
    target_length: int
    target_offset: int
    stride: int = 1
    warmup_length: int = 0

    def __post_init__(self) -> None:
        if self.layout is PredictionLayout.CONTINUOUS:
            raise ValueError("Continuous models do not use supervised windows.")
        for name in ("input_length", "target_length", "stride"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.target_offset < 0:
            raise ValueError("target_offset cannot be negative.")
        if not 0 <= self.warmup_length < self.target_length:
            raise ValueError("warmup_length must lie in [0, target_length).")
        if self.layout is PredictionLayout.SEQUENCE_TO_VALUE:
            if self.target_length != 1:
                raise ValueError("S2V requires target_length=1.")
            if self.warmup_length:
                raise ValueError("S2V does not support a target warm-up mask.")

    @classmethod
    def same_sequence(
        cls, length: int, *, stride: int = 1, warmup_length: int = 0
    ) -> "WindowSpec":
        """Create aligned S2S windows where each target matches an input time."""

        return cls(
            layout=PredictionLayout.SEQUENCE_TO_SEQUENCE,
            input_length=length,
            target_length=length,
            target_offset=0,
            stride=stride,
            warmup_length=warmup_length,
        )

    @classmethod
    def sequence_to_value(
        cls, lookback: int, *, lead: int = 1, stride: int = 1
    ) -> "WindowSpec":
        """Create S2V windows predicting one value after the input window."""

        if lead <= 0:
            raise ValueError("lead must be positive.")
        return cls(
            layout=PredictionLayout.SEQUENCE_TO_VALUE,
            input_length=lookback,
            target_length=1,
            target_offset=lookback + lead - 1,
            stride=stride,
        )

    @classmethod
    def future_sequence(
        cls, lookback: int, horizon: int, *, lead: int = 1, stride: int = 1
    ) -> "WindowSpec":
        """Create S2S forecast windows after an antecedent input window."""

        if lead <= 0:
            raise ValueError("lead must be positive.")
        return cls(
            layout=PredictionLayout.SEQUENCE_TO_SEQUENCE,
            input_length=lookback,
            target_length=horizon,
            target_offset=lookback + lead - 1,
            stride=stride,
        )


@dataclass(frozen=True)
class WindowBatch:
    """Windowed arrays plus exact indices and a loss/evaluation mask."""

    inputs: NDArray[np.floating]
    targets: NDArray[np.floating]
    input_indices: NDArray[np.integer]
    target_indices: NDArray[np.integer]
    target_mask: NDArray[np.bool_]
    spec: WindowSpec


def _as_2d(name: str, values: NDArray) -> NDArray[np.floating]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [time, variable].")
    return array


def build_windows(inputs: NDArray, targets: NDArray, spec: WindowSpec) -> WindowBatch:
    """Build samples inside one already-separated train, validation, or test set.

    Splitting must occur before this function is called. Consequently, a window
    can never cross a partition boundary by accident.
    """

    x = _as_2d("inputs", inputs)
    y = _as_2d("targets", targets)
    if len(x) != len(y):
        raise ValueError("inputs and targets must share the time dimension.")

    required_length = max(
        spec.input_length,
        spec.target_offset + spec.target_length,
    )
    if len(x) < required_length:
        raise ValueError(
            f"At least {required_length} time steps are required; received {len(x)}."
        )

    starts = np.arange(0, len(x) - required_length + 1, spec.stride, dtype=int)
    input_offsets = np.arange(spec.input_length, dtype=int)
    target_offsets = spec.target_offset + np.arange(spec.target_length, dtype=int)
    input_indices = starts[:, None] + input_offsets[None, :]
    target_indices = starts[:, None] + target_offsets[None, :]

    windowed_inputs = x[input_indices]
    windowed_targets = y[target_indices]
    target_mask = np.ones(target_indices.shape, dtype=bool)
    if spec.warmup_length:
        target_mask[:, : spec.warmup_length] = False

    if spec.layout is PredictionLayout.SEQUENCE_TO_VALUE:
        windowed_targets = windowed_targets[:, 0, :]
        target_indices = target_indices[:, 0]
        target_mask = target_mask[:, 0]

    return WindowBatch(
        inputs=windowed_inputs,
        targets=windowed_targets,
        input_indices=input_indices,
        target_indices=target_indices,
        target_mask=target_mask,
        spec=spec,
    )

