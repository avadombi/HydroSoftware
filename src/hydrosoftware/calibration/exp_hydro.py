"""EXP-HYDRO adapter for the generic deterministic calibration engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from hydrosoftware.calibration.definitions import CalibrationProblem
from hydrosoftware.calibration.objectives import NashSutcliffeEfficiency, ObjectiveMetric
from hydrosoftware.core.specifications import ParameterSpec
from hydrosoftware.models.conceptual.exp_hydro import (
    EXP_HYDRO_MANIFEST,
    ExpHydroModel,
    ExpHydroParameters,
)
from hydrosoftware.processes.snow import (
    DegreeDaySnowModule,
    NoSnowModule,
    SmoothDegreeDaySnowModule,
    SnowModule,
)


@dataclass(frozen=True)
class ExpHydroCalibrationSimulation:
    """Pickle-safe scalar simulator for exact parallel candidate evaluation."""

    precipitation: NDArray
    temperature: NDArray
    potential_evapotranspiration: NDArray
    snow_module: SnowModule
    initial_state: tuple[float, float]
    substeps: int

    def _snow_module(self, parameters: Mapping[str, float]) -> SnowModule:
        module = self.snow_module
        if isinstance(module, NoSnowModule):
            return module
        if isinstance(module, DegreeDaySnowModule):
            return DegreeDaySnowModule(
                snowfall_temperature=float(
                    parameters.get(
                        "snowfall_temperature", module.snowfall_temperature
                    )
                ),
                melt_temperature=float(
                    parameters.get("melt_temperature", module.melt_temperature)
                ),
                degree_day_factor=float(
                    parameters.get(
                        "degree_day_factor", module.degree_day_factor
                    )
                ),
            )
        if isinstance(module, SmoothDegreeDaySnowModule):
            return SmoothDegreeDaySnowModule(
                snowfall_temperature=float(
                    parameters.get(
                        "snowfall_temperature", module.snowfall_temperature
                    )
                ),
                melt_temperature=float(
                    parameters.get("melt_temperature", module.melt_temperature)
                ),
                degree_day_factor=float(
                    parameters.get(
                        "degree_day_factor", module.degree_day_factor
                    )
                ),
                sharpness=float(parameters.get("sharpness", module.sharpness)),
            )
        raise TypeError(
            "Parallel EXP-HYDRO calibration supports the built-in snow modules."
        )

    def __call__(self, parameters: Mapping[str, float]):
        model = ExpHydroModel(
            ExpHydroParameters(
                recession_factor=parameters["recession_factor"],
                maximum_storage=parameters["maximum_storage"],
                maximum_subsurface_flow=parameters["maximum_subsurface_flow"],
            ),
            self._snow_module(parameters),
            substeps=self.substeps,
        )
        return model.simulate(
            self.precipitation,
            self.temperature,
            self.potential_evapotranspiration,
            initial_state=self.initial_state,
        )


def build_exp_hydro_calibration_problem(
    precipitation: NDArray,
    temperature: NDArray,
    potential_evapotranspiration: NDArray,
    observations: NDArray,
    snow_module: SnowModule,
    *,
    snow_parameter_specs: tuple[ParameterSpec, ...] = (),
    snow_module_factory: Callable[[Mapping[str, float]], SnowModule] | None = None,
    evaluation_mask: NDArray[np.bool_] | None = None,
    initial_state: tuple[float, float] = (0.0, 0.0),
    substeps: int = 4,
    objective: ObjectiveMetric | None = None,
) -> CalibrationProblem:
    """Build a calibration problem around the mass-conservative EXP-HYDRO model."""

    forcing = (
        np.asarray(precipitation, dtype=float),
        np.asarray(temperature, dtype=float),
        np.asarray(potential_evapotranspiration, dtype=float),
    )

    if snow_module_factory is not None and not isinstance(
        snow_module,
        (NoSnowModule, DegreeDaySnowModule, SmoothDegreeDaySnowModule),
    ):
        def simulate(parameters: Mapping[str, float]):
            model = ExpHydroModel(
                ExpHydroParameters(
                    recession_factor=parameters["recession_factor"],
                    maximum_storage=parameters["maximum_storage"],
                    maximum_subsurface_flow=parameters["maximum_subsurface_flow"],
                ),
                snow_module_factory(parameters),
                substeps=substeps,
            )
            return model.simulate(
                forcing[0],
                forcing[1],
                forcing[2],
                initial_state=initial_state,
            )
    else:
        simulate = ExpHydroCalibrationSimulation(
            precipitation=forcing[0],
            temperature=forcing[1],
            potential_evapotranspiration=forcing[2],
            snow_module=snow_module,
            initial_state=initial_state,
            substeps=substeps,
        )

    return CalibrationProblem(
        parameter_specs=EXP_HYDRO_MANIFEST.parameters + snow_parameter_specs,
        simulate=simulate,
        observations=observations,
        objective=objective or NashSutcliffeEfficiency(),
        output_name="streamflow",
        evaluation_mask=evaluation_mask,
        metadata={
            "model_id": EXP_HYDRO_MANIFEST.model_id,
            "model_version": EXP_HYDRO_MANIFEST.version,
            "snow_module_id": snow_module.module_id,
        },
    )
