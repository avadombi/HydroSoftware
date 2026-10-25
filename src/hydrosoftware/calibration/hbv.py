"""HBV adapter for the generic deterministic calibration engine."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from hydrosoftware.calibration.definitions import CalibrationProblem
from hydrosoftware.calibration.objectives import NashSutcliffeEfficiency, ObjectiveMetric
from hydrosoftware.core.specifications import ParameterSpec
from hydrosoftware.models.conceptual.hbv import HBV_MANIFEST, HbvModel, HbvParameters
from hydrosoftware.processes.routing import (
    MAXBAS_MEMORY_SIZE,
    TriangularMaxbasRouting,
    route_simulation_result,
)
from hydrosoftware.processes.snow import SnowModule


def build_hbv_calibration_problem(
    precipitation: NDArray,
    temperature: NDArray,
    potential_evapotranspiration: NDArray,
    observations: NDArray,
    snow_module: SnowModule,
    *,
    snow_parameter_specs: tuple[ParameterSpec, ...] = (),
    snow_module_factory: Callable[[Mapping[str, float]], SnowModule] | None = None,
    evaluation_mask: NDArray[np.bool_] | None = None,
    initial_state: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    routing_process_id: str = "routing.none.v1",
    maxbas: int = 3,
    maxbas_parameter_spec: ParameterSpec | None = None,
    initial_routing_state: tuple[float, ...] = (0.0,) * MAXBAS_MEMORY_SIZE,
    objective: ObjectiveMetric | None = None,
) -> CalibrationProblem:
    """Build a calibration problem around the conservative daily HBV variant."""

    forcing = tuple(
        np.asarray(values, dtype=float)
        for values in (
            precipitation,
            temperature,
            potential_evapotranspiration,
        )
    )

    def simulate(parameters: Mapping[str, float]):
        active_snow_module = (
            snow_module_factory(parameters)
            if snow_module_factory is not None
            else snow_module
        )
        model = HbvModel(
            HbvParameters(
                field_capacity=parameters["field_capacity"],
                recharge_exponent=parameters["recharge_exponent"],
                evap_reduction_threshold=parameters["evap_reduction_threshold"],
                maximum_percolation=parameters["maximum_percolation"],
                upper_zone_threshold=parameters["upper_zone_threshold"],
                fast_recession=parameters["fast_recession"],
                intermediate_recession=parameters["intermediate_recession"],
                slow_recession=parameters["slow_recession"],
            ),
            active_snow_module,
        )
        simulation = model.simulate(*forcing, initial_state=initial_state)
        if routing_process_id == "routing.none.v1":
            return simulation
        if routing_process_id != "routing.maxbas.triangular.v1":
            raise ValueError(f"Unsupported HBV routing process: {routing_process_id}")
        candidate_maxbas = parameters.get("maxbas", maxbas)
        if not float(candidate_maxbas).is_integer():
            raise ValueError("MAXBAS candidates must be whole days.")
        return route_simulation_result(
            simulation,
            TriangularMaxbasRouting(int(candidate_maxbas)),
            initial_state=initial_routing_state,
        )

    routing_specs = (
        (maxbas_parameter_spec,) if maxbas_parameter_spec is not None else ()
    )

    return CalibrationProblem(
        parameter_specs=(
            HBV_MANIFEST.parameters + snow_parameter_specs + routing_specs
        ),
        simulate=simulate,
        observations=observations,
        objective=objective or NashSutcliffeEfficiency(),
        output_name="streamflow",
        evaluation_mask=evaluation_mask,
        metadata={
            "model_id": HBV_MANIFEST.model_id,
            "model_version": HBV_MANIFEST.version,
            "snow_module_id": snow_module.module_id,
            "routing": routing_process_id,
        },
    )
