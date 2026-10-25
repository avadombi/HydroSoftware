"""Saved results with unambiguous state and flux timing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimulationResult:
    """Outputs, boundary states, and interval fluxes from one simulation.

    For ``n`` forcing intervals, outputs and fluxes contain ``n`` rows while
    states contain ``n + 1`` boundaries. Therefore ``states[t]`` is the state
    before interval ``t`` and ``states[t + 1]`` is the state after it.
    """

    outputs: NDArray[np.floating]
    states: NDArray[np.floating]
    fluxes: NDArray[np.floating]
    output_names: tuple[str, ...]
    state_names: tuple[str, ...]
    flux_names: tuple[str, ...]
    diagnostics: Mapping[str, NDArray | float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        outputs = np.asarray(self.outputs, dtype=float)
        states = np.asarray(self.states, dtype=float)
        fluxes = np.asarray(self.fluxes, dtype=float)
        if outputs.ndim != 2 or states.ndim != 2 or fluxes.ndim != 2:
            raise ValueError("outputs, states, and fluxes must all be two-dimensional.")
        if len(states) != len(outputs) + 1:
            raise ValueError("states must contain one more boundary than outputs.")
        if len(fluxes) != len(outputs):
            raise ValueError("fluxes and outputs must share the interval dimension.")
        if outputs.shape[1] != len(self.output_names):
            raise ValueError("output_names do not match outputs.")
        if states.shape[1] != len(self.state_names):
            raise ValueError("state_names do not match states.")
        if fluxes.shape[1] != len(self.flux_names):
            raise ValueError("flux_names do not match fluxes.")
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "fluxes", fluxes)

    @property
    def state_before(self) -> NDArray[np.floating]:
        return self.states[:-1]

    @property
    def state_after(self) -> NDArray[np.floating]:
        return self.states[1:]

