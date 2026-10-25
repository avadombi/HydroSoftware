"""Mass-conservative lumped EXP-HYDRO reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np
from numpy.typing import NDArray

from hydrosoftware.core.specifications import (
    ModelCapabilities,
    ModelFamily,
    ModelManifest,
    ParameterSpec,
    PredictionLayout,
    VariableSpec,
)
from hydrosoftware.core.results import SimulationResult
from hydrosoftware.processes.snow import SnowModule


@dataclass(frozen=True)
class ExpHydroParameters:
    """Soil-bucket parameters from the original EXP-HYDRO formulation."""

    recession_factor: float
    maximum_storage: float
    maximum_subsurface_flow: float

    def __post_init__(self) -> None:
        bounds = {
            "recession_factor": (self.recession_factor, 0.0, 0.1),
            "maximum_storage": (self.maximum_storage, 100.0, 1500.0),
            "maximum_subsurface_flow": (
                self.maximum_subsurface_flow,
                10.0,
                50.0,
            ),
        }
        for name, (value, lower, upper) in bounds.items():
            if not lower <= value <= upper:
                raise ValueError(f"{name} must lie in [{lower}, {upper}].")


EXP_HYDRO_MANIFEST = ModelManifest(
    model_id="exp_hydro.mass_conservative.v1",
    version="1.0.0",
    display_name="EXP-HYDRO (mass-conservative)",
    family=ModelFamily.CONCEPTUAL,
    capabilities=ModelCapabilities(
        layouts=(PredictionLayout.CONTINUOUS,),
        supports_fit=False,
        supports_calibration=True,
        differentiable=False,
        supports_initial_state=True,
        exposes_states=True,
        exposes_fluxes=True,
    ),
    inputs=(
        VariableSpec("precipitation", "forcing", "mm/interval", lower_bound=0.0),
        VariableSpec("temperature", "forcing", "degC"),
        VariableSpec(
            "potential_evapotranspiration",
            "forcing",
            "mm/interval",
            lower_bound=0.0,
        ),
    ),
    outputs=(
        VariableSpec("streamflow", "output", "mm/interval", lower_bound=0.0),
    ),
    parameters=(
        ParameterSpec("recession_factor", 0.0, 0.1, 0.05),
        ParameterSpec("maximum_storage", 100.0, 1500.0, 500.0, "mm"),
        ParameterSpec("maximum_subsurface_flow", 10.0, 50.0, 30.0, "mm/day"),
    ),
    states=(
        VariableSpec("snow_storage", "state", "mm", lower_bound=0.0),
        VariableSpec("soil_storage", "state", "mm", lower_bound=0.0),
    ),
    fluxes=(
        VariableSpec("precipitation", "flux", "mm/interval", lower_bound=0.0),
        VariableSpec("rain", "flux", "mm/interval", lower_bound=0.0),
        VariableSpec("snowfall", "flux", "mm/interval", lower_bound=0.0),
        VariableSpec("snowmelt", "flux", "mm/interval", lower_bound=0.0),
        VariableSpec(
            "actual_evapotranspiration",
            "flux",
            "mm/interval",
            lower_bound=0.0,
        ),
        VariableSpec("subsurface_flow", "flux", "mm/interval", lower_bound=0.0),
        VariableSpec("surface_flow", "flux", "mm/interval", lower_bound=0.0),
    ),
    references=(
        "Patil, S. and Stieglitz, M. (2014), Hydrological Processes 28, 1159-1169.",
    ),
    metadata={
        "integration": "mass_conservative_explicit_substeps",
        "snow_module": "selected_at_runtime",
        "legacy_equivalence": "not_claimed",
    },
)


class ExpHydroModel:
    """EXP-HYDRO soil bucket composed with a selectable snow module.

    The implementation keeps daily forcing constant and divides each forcing
    interval into explicit substeps. Potential outgoing soil fluxes are reduced
    proportionally only when they exceed the water available in that substep.
    This preserves non-negative storage and exact numerical water balance.
    """

    manifest = EXP_HYDRO_MANIFEST

    def __init__(
        self,
        parameters: ExpHydroParameters,
        snow_module: SnowModule,
        *,
        substeps: int = 4,
    ) -> None:
        if substeps <= 0:
            raise ValueError("substeps must be positive.")
        self.parameters = parameters
        self.snow_module = snow_module
        self.substeps = substeps

    @staticmethod
    def _forcing_vector(name: str, values: NDArray) -> NDArray[np.floating]:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{name} must have shape [time].")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array

    def _soil_flux_rates(
        self,
        storage: float,
        potential_evapotranspiration: float,
    ) -> tuple[float, float, float]:
        """Return potential ET, subsurface flow, and surface flow rates."""

        if storage <= 0.0:
            return 0.0, 0.0, 0.0

        p = self.parameters
        if storage > p.maximum_storage:
            return (
                potential_evapotranspiration,
                p.maximum_subsurface_flow,
                storage - p.maximum_storage,
            )

        storage_fraction = storage / p.maximum_storage
        actual_et = potential_evapotranspiration * storage_fraction
        subsurface_flow = p.maximum_subsurface_flow * exp(
            -p.recession_factor * (p.maximum_storage - storage)
        )
        return actual_et, subsurface_flow, 0.0

    def simulate(
        self,
        precipitation: NDArray,
        temperature: NDArray,
        potential_evapotranspiration: NDArray,
        *,
        initial_state: tuple[float, float] = (0.0, 0.0),
    ) -> SimulationResult:
        """Simulate one continuous forcing partition."""

        precipitation = self._forcing_vector("precipitation", precipitation)
        temperature = self._forcing_vector("temperature", temperature)
        pet = self._forcing_vector(
            "potential_evapotranspiration", potential_evapotranspiration
        )
        if not (len(precipitation) == len(temperature) == len(pet)):
            raise ValueError("All forcing vectors must share the time dimension.")
        if np.any(precipitation < 0.0):
            raise ValueError("Precipitation cannot be negative.")
        if np.any(pet < 0.0):
            raise ValueError("Potential evapotranspiration cannot be negative.")

        state = np.asarray(initial_state, dtype=float)
        if state.shape != (2,) or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must contain finite snow and soil storage.")
        if np.any(state < 0.0):
            raise ValueError("Initial storages cannot be negative.")

        n_time = len(precipitation)
        states = np.empty((n_time + 1, 2), dtype=float)
        fluxes = np.zeros((n_time, 7), dtype=float)
        outputs = np.zeros((n_time, 1), dtype=float)
        states[0] = state
        interval_balance_error = np.zeros(n_time, dtype=float)
        dt = 1.0 / self.substeps

        for t in range(n_time):
            snow_storage, soil_storage = states[t]
            initial_total_storage = snow_storage + soil_storage
            totals = np.zeros(7, dtype=float)

            for _ in range(self.substeps):
                snow = self.snow_module.step(
                    precipitation=float(precipitation[t] * dt),
                    temperature=float(temperature[t]),
                    storage=float(snow_storage),
                    dt=dt,
                )
                snow_storage = snow.storage_after
                inflow = snow.rain + snow.melt

                et_rate, qsub_rate, qsurf_rate = self._soil_flux_rates(
                    soil_storage,
                    float(pet[t]),
                )
                potential_outflows = np.array(
                    [et_rate, qsub_rate, qsurf_rate], dtype=float
                ) * dt
                available_water = soil_storage + inflow
                potential_total = float(potential_outflows.sum())
                if potential_total > available_water and potential_total > 0.0:
                    potential_outflows *= available_water / potential_total

                actual_et, qsub, qsurf = potential_outflows
                soil_storage = available_water - actual_et - qsub - qsurf
                if soil_storage < -1e-12:
                    raise RuntimeError("Internal soil-water limiter failed.")
                soil_storage = max(soil_storage, 0.0)

                totals += np.array(
                    [
                        precipitation[t] * dt,
                        snow.rain,
                        snow.snowfall,
                        snow.melt,
                        actual_et,
                        qsub,
                        qsurf,
                    ]
                )

            states[t + 1] = (snow_storage, soil_storage)
            fluxes[t] = totals
            outputs[t, 0] = totals[5] + totals[6]
            final_total_storage = snow_storage + soil_storage
            interval_balance_error[t] = (
                initial_total_storage
                + precipitation[t]
                - totals[4]
                - totals[5]
                - totals[6]
                - final_total_storage
            )

        return SimulationResult(
            outputs=outputs,
            states=states,
            fluxes=fluxes,
            output_names=("streamflow",),
            state_names=("snow_storage", "soil_storage"),
            flux_names=(
                "precipitation",
                "rain",
                "snowfall",
                "snowmelt",
                "actual_evapotranspiration",
                "subsurface_flow",
                "surface_flow",
            ),
            diagnostics={
                "mass_balance_error": interval_balance_error,
                "maximum_absolute_mass_balance_error": float(
                    np.max(np.abs(interval_balance_error), initial=0.0)
                ),
                "substeps": float(self.substeps),
                "snow_module_id": self.snow_module.module_id,
            },
        )


