"""Calibration definitions and the first framework-independent search engine."""

from hydrosoftware.calibration.definitions import (
    CalibrationProblem,
    CalibrationRunResult,
    EvaluationRecord,
)
from hydrosoftware.calibration.engine import (
    CalibrationEngine,
    DifferentialEvolutionSearchConfig,
    UniformRandomSearchConfig,
)
from hydrosoftware.calibration.exp_hydro import build_exp_hydro_calibration_problem
from hydrosoftware.calibration.hbv import build_hbv_calibration_problem
from hydrosoftware.calibration.objectives import (
    ConfiguredHydrologicObjective,
    NashSutcliffeEfficiency,
    ObjectiveDirection,
    RootMeanSquareError,
)

__all__ = [
    "CalibrationEngine",
    "CalibrationProblem",
    "CalibrationRunResult",
    "ConfiguredHydrologicObjective",
    "DifferentialEvolutionSearchConfig",
    "EvaluationRecord",
    "NashSutcliffeEfficiency",
    "ObjectiveDirection",
    "RootMeanSquareError",
    "UniformRandomSearchConfig",
    "build_exp_hydro_calibration_problem",
    "build_hbv_calibration_problem",
]
