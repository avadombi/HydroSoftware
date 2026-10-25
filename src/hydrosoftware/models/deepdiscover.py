"""Configuration types for the supported DeepDiscover implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProcessDynamics(str, Enum):
    """Temporal semantics of the supported discovered-process outputs."""

    INSTANTANEOUS = "instantaneous"


@dataclass(frozen=True)
class DiscoveryLossConfig:
    """Weights and thresholds for scientifically distinct regularizers."""

    decorrelation_weight: float = 0.2
    mass_balance_weight: float = 0.1
    smoothness_weight: float = 1.0
    smoothness_threshold: float = 0.1
    binary_weight: float = 1.0
    target_process_correlation: float = 0.0

    def __post_init__(self) -> None:
        nonnegative = (
            self.decorrelation_weight,
            self.mass_balance_weight,
            self.smoothness_weight,
            self.smoothness_threshold,
            self.binary_weight,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("Loss weights and thresholds cannot be negative.")
        if not -1.0 <= self.target_process_correlation <= 1.0:
            raise ValueError("target_process_correlation must lie in [-1, 1].")


@dataclass(frozen=True)
class DeepDiscoverConfig:
    """Scientific choices for the instantaneous DD process mapping."""

    snow_module_id: str
    process_count: int
    process_dynamics: ProcessDynamics = ProcessDynamics.INSTANTANEOUS
    initial_state_policy: str = "zero"
    loss: DiscoveryLossConfig = DiscoveryLossConfig()
    expose_analysis_tensors: bool = True

    def __post_init__(self) -> None:
        if not self.snow_module_id.strip():
            raise ValueError("snow_module_id cannot be empty.")
        if self.process_count <= 0:
            raise ValueError("process_count must be positive.")
        if self.initial_state_policy not in {"zero", "learned", "provided", "carried"}:
            raise ValueError("Unsupported initial_state_policy.")


def binary_relaxation_penalty(probability: float) -> float:
    """Penalize relaxed binary values after constraining them to [0, 1]."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Binary probabilities must lie in [0, 1].")
    return probability * (1.0 - probability)
