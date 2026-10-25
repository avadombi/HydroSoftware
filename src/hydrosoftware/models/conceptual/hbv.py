"""Mass-conservative daily HBV two-zone response variant."""

from __future__ import annotations

from dataclasses import dataclass

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
class HbvParameters:
    """Parameters of the HBV soil and two-zone response routines."""

    field_capacity: float
    recharge_exponent: float
    evap_reduction_threshold: float
    maximum_percolation: float
    upper_zone_threshold: float
    fast_recession: float
    intermediate_recession: float
    slow_recession: float

    def __post_init__(self) -> None:
        bounds = {
            "field_capacity": (self.field_capacity, 50.0, 700.0),
            "recharge_exponent": (self.recharge_exponent, 1.0, 6.0),
            "evap_reduction_threshold": (
                self.evap_reduction_threshold,
                0.3,
                1.0,
            ),
            "maximum_percolation": (self.maximum_percolation, 0.0, 6.0),
            "upper_zone_threshold": (self.upper_zone_threshold, 0.0, 100.0),
            "fast_recession": (self.fast_recession, 0.05, 0.99),
            "intermediate_recession": (
                self.intermediate_recession,
                0.01,
                0.8,
            ),
            "slow_recession": (self.slow_recession, 0.001, 0.15),
        }
        for name, (value, lower, upper) in bounds.items():
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must lie in [{lower}, {upper}].")


HBV_MANIFEST = ModelManifest(
    model_id="hbv.mass_conservative_daily.v1",
    version="1.0.0",
    display_name="HBV (mass-conservative daily variant)",
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
        ParameterSpec("field_capacity", 50.0, 700.0, 500.0, "mm"),
        ParameterSpec("recharge_exponent", 1.0, 6.0, 2.0),
        ParameterSpec("evap_reduction_threshold", 0.3, 1.0, 0.7),
        ParameterSpec("maximum_percolation", 0.0, 6.0, 1.5, "mm/day"),
        ParameterSpec("upper_zone_threshold", 0.0, 100.0, 30.0, "mm"),
        ParameterSpec("fast_recession", 0.05, 0.99, 0.30, "1/day"),
        ParameterSpec("intermediate_recession", 0.01, 0.8, 0.10, "1/day"),
        ParameterSpec("slow_recession", 0.001, 0.15, 0.02, "1/day"),
    ),
    states=(
        VariableSpec("snow_storage", "state", "mm", lower_bound=0.0),
        VariableSpec("soil_storage", "state", "mm", lower_bound=0.0),
        VariableSpec("upper_zone_storage", "state", "mm", lower_bound=0.0),
        VariableSpec("lower_zone_storage", "state", "mm", lower_bound=0.0),
    ),
    fluxes=tuple(
        VariableSpec(name, "flux", "mm/interval", lower_bound=0.0)
        for name in (
            "precipitation",
            "rain",
            "snowfall",
            "snowmelt",
            "soil_water_input",
            "soil_excess",
            "groundwater_recharge",
            "actual_evapotranspiration",
            "percolation",
            "fast_flow",
            "intermediate_flow",
            "baseflow",
            "unrouted_streamflow",
        )
    ),
    references=(
        "Seibert and Vis (2012), Hydrology and Earth System Sciences 16, 3315-3325.",
        "Seibert and Bergstrom (2022), Hydrology and Earth System Sciences 26, 1371-1388.",
    ),
    metadata={
        "time_step": "daily",
        "integration": "mass_conservative_discrete_water_balance",
        "snow_module": "selected_at_runtime",
        "groundwater_response": "two_zones_three_linear_outflows",
        "triangular_routing": "not_applied",
        "supported_routing_process": "routing.maxbas.triangular.v1",
        "exact_hbv_light_equivalence": "not_claimed",
    },
)


class HbvModel:
    """Daily HBV soil and response routines with explicit water limiting.

    The selected snow module first partitions precipitation. Soil input is
    divided between groundwater recharge and retained soil water using the HBV
    wetness relationship. Any soil-capacity excess is transferred to the upper
    response zone. Percolation is limited by upper-zone water, and simultaneous
    upper-zone outflows are proportionally limited if their potential sum would
    empty the store below zero.

    This first variant returns the raw response-routine discharge. The optional
    HBV triangular catchment-routing transform is deliberately not applied and
    is recorded in the manifest and diagnostics.
    """

    manifest = HBV_MANIFEST

    def __init__(self, parameters: HbvParameters, snow_module: SnowModule) -> None:
        self.parameters = parameters
        self.snow_module = snow_module

    @staticmethod
    def _forcing(name: str, values: NDArray) -> NDArray[np.floating]:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite vector with shape [time].")
        return array

    def simulate(
        self,
        precipitation: NDArray,
        temperature: NDArray,
        potential_evapotranspiration: NDArray,
        *,
        initial_state: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> SimulationResult:
        precipitation = self._forcing("precipitation", precipitation)
        temperature = self._forcing("temperature", temperature)
        pet = self._forcing(
            "potential_evapotranspiration", potential_evapotranspiration
        )
        if not (len(precipitation) == len(temperature) == len(pet)):
            raise ValueError("All forcing vectors must share the time dimension.")
        if np.any(precipitation < 0.0) or np.any(pet < 0.0):
            raise ValueError(
                "Precipitation and potential evapotranspiration cannot be negative."
            )

        state = np.asarray(initial_state, dtype=float)
        if state.shape != (4,) or not np.all(np.isfinite(state)) or np.any(state < 0.0):
            raise ValueError(
                "initial_state must contain four finite non-negative storages."
            )

        n_time = len(precipitation)
        states = np.empty((n_time + 1, 4), dtype=float)
        fluxes = np.zeros((n_time, 13), dtype=float)
        outputs = np.zeros((n_time, 1), dtype=float)
        errors = np.zeros(n_time, dtype=float)
        states[0] = state
        p = self.parameters

        for t in range(n_time):
            snow_before, soil_before, upper_before, lower_before = states[t]
            snow = self.snow_module.step(
                precipitation=float(precipitation[t]),
                temperature=float(temperature[t]),
                storage=float(snow_before),
                dt=1.0,
            )

            soil_input = snow.rain + snow.melt
            wetness = min(max(soil_before / p.field_capacity, 0.0), 1.0)
            nonlinear_recharge = soil_input * wetness ** p.recharge_exponent
            soil_available = soil_before + soil_input - nonlinear_recharge
            soil_excess = max(soil_available - p.field_capacity, 0.0)
            groundwater_recharge = nonlinear_recharge + soil_excess
            soil_available -= soil_excess

            evap_factor = min(
                soil_before
                / (p.evap_reduction_threshold * p.field_capacity),
                1.0,
            )
            evap_demand = float(pet[t]) * max(evap_factor, 0.0)
            actual_et = min(evap_demand, soil_available)
            soil_after = soil_available - actual_et

            upper_available = upper_before + groundwater_recharge
            percolation = min(p.maximum_percolation, upper_available)
            upper_after_percolation = upper_available - percolation

            fast_potential = p.fast_recession * max(
                upper_after_percolation - p.upper_zone_threshold,
                0.0,
            )
            intermediate_potential = (
                p.intermediate_recession * upper_after_percolation
            )
            potential_upper_outflow = fast_potential + intermediate_potential
            if (
                potential_upper_outflow > upper_after_percolation
                and potential_upper_outflow > 0.0
            ):
                limiter = upper_after_percolation / potential_upper_outflow
                fast_flow = fast_potential * limiter
                intermediate_flow = intermediate_potential * limiter
            else:
                fast_flow = fast_potential
                intermediate_flow = intermediate_potential
            upper_after = upper_after_percolation - fast_flow - intermediate_flow

            lower_available = lower_before + percolation
            baseflow = min(p.slow_recession * lower_available, lower_available)
            lower_after = lower_available - baseflow
            streamflow = fast_flow + intermediate_flow + baseflow

            states[t + 1] = (
                snow.storage_after,
                max(soil_after, 0.0),
                max(upper_after, 0.0),
                max(lower_after, 0.0),
            )
            fluxes[t] = (
                precipitation[t],
                snow.rain,
                snow.snowfall,
                snow.melt,
                soil_input,
                soil_excess,
                groundwater_recharge,
                actual_et,
                percolation,
                fast_flow,
                intermediate_flow,
                baseflow,
                streamflow,
            )
            outputs[t, 0] = streamflow
            errors[t] = (
                snow_before
                + soil_before
                + upper_before
                + lower_before
                + precipitation[t]
                - actual_et
                - streamflow
                - states[t + 1].sum()
            )

        return SimulationResult(
            outputs=outputs,
            states=states,
            fluxes=fluxes,
            output_names=("streamflow",),
            state_names=(
                "snow_storage",
                "soil_storage",
                "upper_zone_storage",
                "lower_zone_storage",
            ),
            flux_names=(
                "precipitation",
                "rain",
                "snowfall",
                "snowmelt",
                "soil_water_input",
                "soil_excess",
                "groundwater_recharge",
                "actual_evapotranspiration",
                "percolation",
                "fast_flow",
                "intermediate_flow",
                "baseflow",
                "unrouted_streamflow",
            ),
            diagnostics={
                "mass_balance_error": errors,
                "maximum_absolute_mass_balance_error": float(
                    np.max(np.abs(errors), initial=0.0)
                ),
                "snow_module_id": self.snow_module.module_id,
                "time_step_interpretation": "daily",
                "routing_interpretation": "unrouted_response_discharge",
            },
        )
