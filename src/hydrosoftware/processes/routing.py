"""Explicit, restartable runoff-routing processes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hydrosoftware.core.results import SimulationResult


MAXBAS_MAX_DAYS = 7
MAXBAS_MEMORY_SIZE = MAXBAS_MAX_DAYS - 1
MAXBAS_STATE_NAMES = tuple(
    f"routing_pending_day_{lag}" for lag in range(1, MAXBAS_MEMORY_SIZE + 1)
)


def triangular_maxbas_weights(maxbas: int) -> NDArray[np.floating]:
    """Return daily weights from the HBV-light triangular transfer function.

    ``maxbas`` is the integer base length in days. Each weight is the exact
    integral of the unit-area isosceles triangle over one daily interval, as
    defined by Seibert and Vis (2012, Eq. 6). The returned vector is padded to
    seven entries so that every configured routing process has the same
    restart-state definition.
    """

    if isinstance(maxbas, bool) or not isinstance(maxbas, (int, np.integer)):
        raise TypeError("maxbas must be an integer number of days.")
    resolved = int(maxbas)
    if not 1 <= resolved <= MAXBAS_MAX_DAYS:
        raise ValueError(f"maxbas must lie in [1, {MAXBAS_MAX_DAYS}] days.")

    def cdf(value: float) -> float:
        x = min(max(value, 0.0), float(resolved))
        midpoint = resolved / 2.0
        if x <= midpoint:
            return 2.0 * x * x / (resolved * resolved)
        return 1.0 - 2.0 * (resolved - x) ** 2 / (resolved * resolved)

    weights = np.zeros(MAXBAS_MAX_DAYS, dtype=float)
    for index in range(resolved):
        weights[index] = cdf(index + 1.0) - cdf(float(index))
    weights /= weights.sum()
    return weights


@dataclass(frozen=True)
class RoutingResult:
    """Routed discharge and pending-volume states at interval boundaries."""

    routed_streamflow: NDArray[np.floating]
    states: NDArray[np.floating]
    weights: NDArray[np.floating]
    mass_balance_error: NDArray[np.floating]


class TriangularMaxbasRouting:
    """Finite-memory HBV MAXBAS routing with an exact restart state."""

    process_id = "routing.maxbas.triangular.v1"

    def __init__(self, maxbas: int) -> None:
        self.maxbas = int(maxbas)
        self.weights = triangular_maxbas_weights(maxbas)

    def route(
        self,
        unrouted_streamflow: NDArray,
        *,
        initial_state: tuple[float, ...] = (0.0,) * MAXBAS_MEMORY_SIZE,
    ) -> RoutingResult:
        inflow = np.asarray(unrouted_streamflow, dtype=float)
        if inflow.ndim != 1 or not np.all(np.isfinite(inflow)):
            raise ValueError("unrouted_streamflow must be a finite vector.")
        if np.any(inflow < 0.0):
            raise ValueError("unrouted_streamflow cannot be negative.")

        state = np.asarray(initial_state, dtype=float)
        if (
            state.shape != (MAXBAS_MEMORY_SIZE,)
            or not np.all(np.isfinite(state))
            or np.any(state < 0.0)
        ):
            raise ValueError(
                f"initial_state must contain {MAXBAS_MEMORY_SIZE} finite "
                "non-negative pending routing volumes."
            )

        n_time = len(inflow)
        states = np.zeros((n_time + 1, MAXBAS_MEMORY_SIZE), dtype=float)
        routed = np.zeros(n_time, dtype=float)
        errors = np.zeros(n_time, dtype=float)
        states[0] = state

        for t, current_inflow in enumerate(inflow):
            before = states[t]
            routed[t] = before[0] + current_inflow * self.weights[0]
            after = np.zeros(MAXBAS_MEMORY_SIZE, dtype=float)
            after[:-1] = before[1:]
            after += current_inflow * self.weights[1:]
            states[t + 1] = after
            errors[t] = before.sum() + current_inflow - routed[t] - after.sum()

        return RoutingResult(
            routed_streamflow=routed,
            states=states,
            weights=self.weights.copy(),
            mass_balance_error=errors,
        )


def route_simulation_result(
    simulation: SimulationResult,
    routing: TriangularMaxbasRouting,
    *,
    initial_state: tuple[float, ...] = (0.0,) * MAXBAS_MEMORY_SIZE,
) -> SimulationResult:
    """Compose a hydrological trajectory with explicit triangular routing."""

    try:
        output_index = simulation.output_names.index("streamflow")
        unrouted_index = simulation.flux_names.index("unrouted_streamflow")
    except ValueError as exc:
        raise ValueError(
            "MAXBAS routing requires streamflow output and unrouted_streamflow flux."
        ) from exc

    unrouted = simulation.fluxes[:, unrouted_index]
    routed = routing.route(unrouted, initial_state=initial_state)
    outputs = simulation.outputs.copy()
    outputs[:, output_index] = routed.routed_streamflow
    states = np.column_stack((simulation.states, routed.states))
    fluxes = np.column_stack(
        (simulation.fluxes, routed.routed_streamflow)
    )

    hydrological_errors = np.asarray(
        simulation.diagnostics.get(
            "mass_balance_error", np.zeros(len(outputs), dtype=float)
        ),
        dtype=float,
    )
    combined_errors = hydrological_errors + routed.mass_balance_error
    diagnostics = dict(simulation.diagnostics)
    diagnostics.update(
        {
            "hydrological_mass_balance_error": hydrological_errors,
            "routing_mass_balance_error": routed.mass_balance_error,
            "mass_balance_error": combined_errors,
            "maximum_absolute_mass_balance_error": float(
                np.max(np.abs(combined_errors), initial=0.0)
            ),
            "maximum_absolute_routing_mass_balance_error": float(
                np.max(np.abs(routed.mass_balance_error), initial=0.0)
            ),
            "routing_process_id": routing.process_id,
            "routing_maxbas_days": float(routing.maxbas),
            "routing_weights": routed.weights,
            "routing_interpretation": "triangular_maxbas_routed_discharge",
        }
    )
    return SimulationResult(
        outputs=outputs,
        states=states,
        fluxes=fluxes,
        output_names=simulation.output_names,
        state_names=simulation.state_names + MAXBAS_STATE_NAMES,
        flux_names=simulation.flux_names + ("routed_streamflow",),
        diagnostics=diagnostics,
    )
