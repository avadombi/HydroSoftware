"""Framework-independent definitions shared by all modelling engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Mapping


class ModelFamily(str, Enum):
    """Scientific family used to classify a model implementation."""

    CONCEPTUAL = "conceptual"
    NEURAL = "neural"
    PAML = "physics_aware_machine_learning"
    DISCOVERY = "process_discovery"


class PredictionLayout(str, Enum):
    """Shape semantics exposed by a model adapter."""

    CONTINUOUS = "continuous"
    SEQUENCE_TO_SEQUENCE = "sequence_to_sequence"
    SEQUENCE_TO_VALUE = "sequence_to_value"


@dataclass(frozen=True)
class VariableSpec:
    """Semantic description of an input, target, state, flux, or output."""

    name: str
    role: str
    unit: str
    required: bool = True
    lower_bound: float | None = None
    upper_bound: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Variable name cannot be empty.")
        if not self.role.strip():
            raise ValueError("Variable role cannot be empty.")
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise ValueError("Variable lower_bound cannot exceed upper_bound.")


@dataclass(frozen=True)
class ParameterSpec:
    """Definition of one calibratable, trainable, or fixed parameter."""

    name: str
    lower_bound: float
    upper_bound: float
    default: float
    unit: str = "1"
    trainable: bool = True
    description: str = ""
    value_type: Literal["continuous", "integer"] = "continuous"

    def __post_init__(self) -> None:
        if self.lower_bound > self.upper_bound:
            raise ValueError("Parameter lower_bound cannot exceed upper_bound.")
        if not self.lower_bound <= self.default <= self.upper_bound:
            raise ValueError(
                f"Default value for {self.name!r} must lie inside its bounds."
            )
        if self.value_type == "integer" and any(
            not float(value).is_integer()
            for value in (self.lower_bound, self.upper_bound, self.default)
        ):
            raise ValueError(
                f"Integer parameter {self.name!r} requires integer bounds and default."
            )


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities declared by a model rather than inferred by the UI."""

    layouts: tuple[PredictionLayout, ...]
    supports_fit: bool
    supports_calibration: bool
    differentiable: bool
    supports_initial_state: bool
    exposes_states: bool
    exposes_fluxes: bool
    supports_ensembles: bool = False
    supports_multiple_basins: bool = False

    def __post_init__(self) -> None:
        if not self.layouts:
            raise ValueError("At least one prediction layout must be declared.")


@dataclass(frozen=True)
class ModelManifest:
    """Stable metadata used by the registry, UI, and execution engines."""

    model_id: str
    version: str
    display_name: str
    family: ModelFamily
    capabilities: ModelCapabilities
    inputs: tuple[VariableSpec, ...]
    outputs: tuple[VariableSpec, ...]
    parameters: tuple[ParameterSpec, ...] = ()
    states: tuple[VariableSpec, ...] = ()
    fluxes: tuple[VariableSpec, ...] = ()
    references: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id.strip() or " " in self.model_id:
            raise ValueError("model_id must be a non-empty, space-free identifier.")
        if not self.version.strip():
            raise ValueError("Model version cannot be empty.")
        if not self.inputs:
            raise ValueError("A model must declare at least one input.")
        if not self.outputs:
            raise ValueError("A model must declare at least one output.")
        if self.capabilities.exposes_states and not self.states:
            raise ValueError("A model exposing states must declare state variables.")
        if self.capabilities.exposes_fluxes and not self.fluxes:
            raise ValueError("A model exposing fluxes must declare flux variables.")
