"""Mass-conservative instantaneous DeepDiscover reference engine.

This module implements the storage-only instantaneous DD variant.  It is a
deterministic NumPy engine with explicit full-sequence backpropagation through
the discovered storage.
Every displayed interval flux is the same value used by the state transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from hydrosoftware.configuration import (
    DeepDiscoverModelConfig,
    DiscoveryAdamTrainingConfig,
)
from hydrosoftware.core.specifications import (
    ModelCapabilities,
    ModelFamily,
    ModelManifest,
    PredictionLayout,
    VariableSpec,
)
from hydrosoftware.core.results import SimulationResult
from hydrosoftware.core.optimization import (
    objective_definition,
    objective_loss_and_gradient,
    objective_optimization_score,
)
from hydrosoftware.models.discovery_snow import (
    SNOW_COORDINATE_METHOD,
    SNOW_FIXED_METHOD,
    coordinate_snow_update,
    discovery_snow_plan,
    snow_module_from_parameters,
    snow_update_interval,
)
from hydrosoftware.processes.snow import SnowModule


DEEPDISCOVER_NUMPY_MANIFEST = ModelManifest(
    model_id="deepdiscover.numpy_instantaneous_s2s.v1",
    version="1.0.0",
    display_name="DeepDiscover · instantaneous NumPy S2S",
    family=ModelFamily.DISCOVERY,
    capabilities=ModelCapabilities(
        layouts=(PredictionLayout.SEQUENCE_TO_SEQUENCE,),
        supports_fit=True,
        supports_calibration=False,
        differentiable=True,
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
    states=(
        VariableSpec("snow_storage", "state", "mm", lower_bound=0.0),
        VariableSpec("discovered_storage", "state", "mm", lower_bound=0.0),
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
        VariableSpec(
            "discovered_process",
            "flux",
            "mm/interval",
            lower_bound=0.0,
            description=(
                "Runtime expands this declaration to the configured number of "
                "discovered process components."
            ),
        ),
    ),
    references=(
        "Adombi (2026), DeepDiscover: towards autonomous discovery of bucket-type conceptual models: a proof of concept applied to hydrology, Journal of Hydrology 671, 135249.",
    ),
    metadata={
        "engine": "numpy.discovery.reference.v1",
        "prediction_semantics": "same-time continuous sequence-to-sequence",
        "artifact_format": "hydrosoftware.deepdiscover.numpy.instantaneous.v1",
        "process_dynamics": "instantaneous",
        "mass_balance": "enforced_by_allocation",
    },
)


def _sigmoid(value: NDArray | float) -> NDArray:
    array = np.asarray(value, dtype=float)
    positive = array >= 0.0
    result = np.empty_like(array, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _softplus(value: NDArray | float) -> NDArray:
    array = np.asarray(value, dtype=float)
    return np.maximum(array, 0.0) + np.log1p(np.exp(-np.abs(array)))


def _inverse_softplus(value: float) -> float:
    return float(value + np.log(-np.expm1(-value)))


def _as_finite_vector(name: str, value: NDArray) -> NDArray[np.floating]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array.")
    return array


@dataclass(frozen=True)
class DeepDiscoverArtifact:
    """Frozen transformations, switches, and learned mapping for DD inference."""

    artifact_type: str
    model_id: str
    engine_id: str
    process_dynamics: str
    process_count: int
    hidden_units: int
    train_input_switches: bool
    train_process_switches: bool
    input_mean: NDArray[np.floating]
    input_scale: NDArray[np.floating]
    storage_reference: float
    input_weight_raw: NDArray[np.floating]
    input_switch_logits: NDArray[np.floating]
    hidden_kernel: NDArray[np.floating]
    hidden_bias: NDArray[np.floating]
    aet_kernel: NDArray[np.floating]
    aet_bias: NDArray[np.floating]
    process_kernel: NDArray[np.floating]
    process_bias: NDArray[np.floating]
    process_switch_logits: NDArray[np.floating]
    best_epoch: int
    snow_module_id: str | None = None
    snow_parameters: Mapping[str, float] = field(default_factory=dict)
    calibrated_snow_parameters: tuple[str, ...] = ()
    snow_calibration_method: str = SNOW_FIXED_METHOD

    def __post_init__(self) -> None:
        if self.artifact_type != "hydrosoftware.deepdiscover.numpy.instantaneous.v1":
            raise ValueError("Unsupported DeepDiscover artifact type.")
        if self.model_id != "deepdiscover.numpy_instantaneous_s2s.v1":
            raise ValueError("The artifact is not an instantaneous NumPy DD model.")
        if self.engine_id != "numpy.discovery.reference.v1":
            raise ValueError("Unsupported DeepDiscover execution engine.")
        if self.process_dynamics != "instantaneous":
            raise ValueError("This artifact cannot contain recurrent process dynamics.")
        if self.process_count < 1 or self.hidden_units < 1:
            raise ValueError("DeepDiscover dimensions must be positive.")
        if not np.isfinite(self.storage_reference) or self.storage_reference <= 0.0:
            raise ValueError("DeepDiscover storage_reference must be positive.")

        shapes = {
            "input_mean": (3,),
            "input_scale": (3,),
            "input_weight_raw": (3,),
            "input_switch_logits": (3,),
            "hidden_kernel": (4, self.hidden_units),
            "hidden_bias": (self.hidden_units,),
            "aet_kernel": (self.hidden_units,),
            "aet_bias": (1,),
            "process_kernel": (self.hidden_units, self.process_count),
            "process_bias": (self.process_count,),
            "process_switch_logits": (self.process_count,),
        }
        for name, shape in shapes.items():
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}.")
            object.__setattr__(self, name, array)
        if np.any(self.input_scale <= 0.0):
            raise ValueError("DeepDiscover input scales must be positive.")
        snow_parameters = {
            str(name): float(value)
            for name, value in self.snow_parameters.items()
        }
        if any(not np.isfinite(value) for value in snow_parameters.values()):
            raise ValueError("DeepDiscover snow parameters must be finite.")
        object.__setattr__(self, "snow_parameters", snow_parameters)
        object.__setattr__(
            self,
            "calibrated_snow_parameters",
            tuple(str(name) for name in self.calibrated_snow_parameters),
        )
        if self.snow_calibration_method not in {
            SNOW_FIXED_METHOD,
            SNOW_COORDINATE_METHOD,
        }:
            raise ValueError("Unsupported DeepDiscover snow calibration method.")
        if (
            self.calibrated_snow_parameters
            and self.snow_calibration_method != SNOW_COORDINATE_METHOD
        ):
            raise ValueError(
                "Calibrated DeepDiscover snow parameters require the coordinate method."
            )
        if self.snow_module_id is not None:
            snow_module_from_parameters(
                self.snow_module_id,
                self.snow_parameters,
            )

    def as_mapping(self) -> dict:
        return {
            "artifact_type": self.artifact_type,
            "model_id": self.model_id,
            "engine_id": self.engine_id,
            "process_dynamics": self.process_dynamics,
            "process_count": self.process_count,
            "hidden_units": self.hidden_units,
            "train_input_switches": self.train_input_switches,
            "train_process_switches": self.train_process_switches,
            "input_mean": self.input_mean,
            "input_scale": self.input_scale,
            "storage_reference": self.storage_reference,
            "input_weight_raw": self.input_weight_raw,
            "input_switch_logits": self.input_switch_logits,
            "hidden_kernel": self.hidden_kernel,
            "hidden_bias": self.hidden_bias,
            "aet_kernel": self.aet_kernel,
            "aet_bias": self.aet_bias,
            "process_kernel": self.process_kernel,
            "process_bias": self.process_bias,
            "process_switch_logits": self.process_switch_logits,
            "best_epoch": self.best_epoch,
            "snow_module_id": self.snow_module_id,
            "snow_parameters": dict(self.snow_parameters),
            "calibrated_snow_parameters": self.calibrated_snow_parameters,
            "snow_calibration_method": self.snow_calibration_method,
        }

    @classmethod
    def from_mapping(cls, value: Mapping) -> "DeepDiscoverArtifact":
        return cls(
            artifact_type=str(value["artifact_type"]),
            model_id=str(value["model_id"]),
            engine_id=str(value["engine_id"]),
            process_dynamics=str(value["process_dynamics"]),
            process_count=int(value["process_count"]),
            hidden_units=int(value["hidden_units"]),
            train_input_switches=bool(value["train_input_switches"]),
            train_process_switches=bool(value["train_process_switches"]),
            input_mean=np.asarray(value["input_mean"], dtype=float),
            input_scale=np.asarray(value["input_scale"], dtype=float),
            storage_reference=float(value["storage_reference"]),
            input_weight_raw=np.asarray(value["input_weight_raw"], dtype=float),
            input_switch_logits=np.asarray(value["input_switch_logits"], dtype=float),
            hidden_kernel=np.asarray(value["hidden_kernel"], dtype=float),
            hidden_bias=np.asarray(value["hidden_bias"], dtype=float),
            aet_kernel=np.asarray(value["aet_kernel"], dtype=float),
            aet_bias=np.asarray(value["aet_bias"], dtype=float),
            process_kernel=np.asarray(value["process_kernel"], dtype=float),
            process_bias=np.asarray(value["process_bias"], dtype=float),
            process_switch_logits=np.asarray(
                value["process_switch_logits"], dtype=float
            ),
            best_epoch=int(value["best_epoch"]),
            snow_module_id=(
                str(value["snow_module_id"])
                if value.get("snow_module_id") is not None
                else None
            ),
            snow_parameters={
                str(name): float(parameter)
                for name, parameter in (value.get("snow_parameters") or {}).items()
            },
            calibrated_snow_parameters=tuple(
                str(name)
                for name in value.get("calibrated_snow_parameters", ())
            ),
            snow_calibration_method=str(
                value.get("snow_calibration_method", SNOW_FIXED_METHOD)
            ),
        )


@dataclass(frozen=True)
class DeepDiscoverFitResult:
    artifact: DeepDiscoverArtifact
    simulation: SimulationResult
    history: tuple[Mapping[str, float | int], ...]
    partition_masks: Mapping[str, NDArray[np.bool_]]
    stopped_early: bool
    cancelled: bool


@dataclass(frozen=True)
class _SnowTrajectory:
    rain: NDArray[np.floating]
    snowfall: NDArray[np.floating]
    melt: NDArray[np.floating]
    storage: NDArray[np.floating]
    balance_residual: NDArray[np.floating]


@dataclass
class _ForwardTrajectory:
    streamflow: NDArray[np.floating]
    processes: NDArray[np.floating]
    raw_processes: NDArray[np.floating]
    actual_evapotranspiration: NDArray[np.floating]
    storage: NDArray[np.floating]
    active_inputs: NDArray[np.floating]
    balance_residual: NDArray[np.floating]
    caches: list[dict[str, NDArray | float]]


_PARAMETER_NAMES = (
    "input_weight_raw",
    "input_switch_logits",
    "hidden_kernel",
    "hidden_bias",
    "aet_kernel",
    "aet_bias",
    "process_kernel",
    "process_bias",
    "process_switch_logits",
)


def _snow_trajectory(
    snow_module: SnowModule,
    precipitation: NDArray,
    temperature: NDArray,
    *,
    initial_storage: float,
) -> _SnowTrajectory:
    n_time = len(precipitation)
    rain = np.zeros(n_time, dtype=float)
    snowfall = np.zeros(n_time, dtype=float)
    melt = np.zeros(n_time, dtype=float)
    storage = np.zeros(n_time + 1, dtype=float)
    residual = np.zeros(n_time, dtype=float)
    storage[0] = initial_storage
    for time in range(n_time):
        step = snow_module.step(
            float(precipitation[time]),
            float(temperature[time]),
            float(storage[time]),
        )
        rain[time] = step.rain
        snowfall[time] = step.snowfall
        melt[time] = step.melt
        storage[time + 1] = step.storage_after
        residual[time] = step.balance_error
    return _SnowTrajectory(rain, snowfall, melt, storage, residual)


def _switch_values(
    parameters: Mapping[str, NDArray],
    *,
    train_input_switches: bool,
    train_process_switches: bool,
) -> tuple[NDArray, NDArray]:
    input_switches = (
        _sigmoid(parameters["input_switch_logits"])
        if train_input_switches
        else np.ones(3, dtype=float)
    )
    process_switches = (
        _sigmoid(parameters["process_switch_logits"])
        if train_process_switches
        else np.ones(len(parameters["process_switch_logits"]), dtype=float)
    )
    return input_switches, process_switches


def _forward(
    parameters: Mapping[str, NDArray],
    *,
    input_mean: NDArray,
    input_scale: NDArray,
    storage_reference: float,
    rain: NDArray,
    melt: NDArray,
    potential_evapotranspiration: NDArray,
    initial_storage: float,
    train_input_switches: bool,
    train_process_switches: bool,
    retain_cache: bool,
) -> _ForwardTrajectory:
    n_time = len(rain)
    process_count = len(parameters["process_bias"])
    processes = np.zeros((n_time, process_count), dtype=float)
    raw_processes = np.zeros_like(processes)
    actual_et = np.zeros(n_time, dtype=float)
    storage = np.zeros(n_time + 1, dtype=float)
    active_inputs = np.zeros((n_time, 3), dtype=float)
    residual = np.zeros(n_time, dtype=float)
    caches: list[dict[str, NDArray | float]] = []
    storage[0] = initial_storage

    input_switches, process_switches = _switch_values(
        parameters,
        train_input_switches=train_input_switches,
        train_process_switches=train_process_switches,
    )
    input_weights = _softplus(parameters["input_weight_raw"])
    raw_features = np.column_stack(
        (rain, melt, potential_evapotranspiration)
    )
    scaled_features = (raw_features - input_mean) / input_scale

    for time in range(n_time):
        state_before = float(storage[time])
        active = scaled_features[time] * input_weights * input_switches
        active_inputs[time] = active
        storage_normalized = state_before / (state_before + storage_reference)
        context = np.concatenate((active, [storage_normalized]))
        hidden_linear = context @ parameters["hidden_kernel"] + parameters["hidden_bias"]
        hidden = np.tanh(hidden_linear)

        aet_linear = float(hidden @ parameters["aet_kernel"] + parameters["aet_bias"][0])
        aet_fraction = float(_sigmoid(aet_linear))
        available = state_before + float(rain[time]) + float(melt[time])
        pet = float(potential_evapotranspiration[time])
        pet_cap = min(pet, available)
        pet_cap_derivative = 1.0 if available < pet else 0.0
        actual_et[time] = aet_fraction * pet_cap
        remaining = available - actual_et[time]

        process_linear = hidden @ parameters["process_kernel"] + parameters["process_bias"]
        raw = _softplus(process_linear)
        raw_processes[time] = raw
        allocation_weights = raw * process_switches
        denominator = 1.0 + float(np.sum(allocation_weights))
        processes[time] = remaining * allocation_weights / denominator
        storage[time + 1] = remaining / denominator
        residual[time] = (
            available
            - actual_et[time]
            - float(np.sum(processes[time]))
            - storage[time + 1]
        )

        if retain_cache:
            caches.append(
                {
                    "scaled_features": scaled_features[time].copy(),
                    "input_weights": input_weights.copy(),
                    "input_switches": input_switches.copy(),
                    "context": context,
                    "hidden": hidden,
                    "aet_fraction": aet_fraction,
                    "pet_cap": pet_cap,
                    "pet_cap_derivative": pet_cap_derivative,
                    "process_linear": process_linear,
                    "raw_processes": raw,
                    "process_switches": process_switches.copy(),
                    "allocation_weights": allocation_weights,
                    "denominator": denominator,
                    "remaining": remaining,
                    "available": available,
                    "state_before": state_before,
                    "storage_reference": storage_reference,
                }
            )
    return _ForwardTrajectory(
        streamflow=np.sum(processes, axis=1),
        processes=processes,
        raw_processes=raw_processes,
        actual_evapotranspiration=actual_et,
        storage=storage,
        active_inputs=active_inputs,
        balance_residual=residual,
        caches=caches,
    )


def _decorrelation_loss_and_gradient(
    values: NDArray,
    *,
    target_correlation: float,
) -> tuple[float, NDArray]:
    n_time, n_process = values.shape
    gradient = np.zeros_like(values)
    if n_process < 2 or n_time < 2:
        return 0.0, gradient
    centered = values - np.mean(values, axis=0, keepdims=True)
    standard_deviation = np.sqrt(
        np.mean(np.square(centered), axis=0, keepdims=True) + 1e-8
    )
    standardized = centered / standard_deviation
    correlation = standardized.T @ standardized / n_time
    off_diagonal = np.ones_like(correlation) - np.eye(n_process)
    difference = (correlation - target_correlation) * off_diagonal
    count = n_process * (n_process - 1)
    loss = float(np.sum(np.square(difference)) / count)
    gradient_standardized = (
        4.0 / (count * n_time) * standardized @ difference
    )
    gradient = (
        gradient_standardized
        - np.mean(gradient_standardized, axis=0, keepdims=True)
        - standardized
        * np.mean(
            gradient_standardized * standardized,
            axis=0,
            keepdims=True,
        )
    ) / standard_deviation
    return loss, gradient


def _smoothness_loss_and_gradient(
    values: NDArray,
    *,
    threshold: float,
) -> tuple[float, NDArray]:
    gradient = np.zeros_like(values)
    if len(values) < 2:
        return 0.0, gradient
    differences = values[1:] - values[:-1]
    excess = np.maximum(np.abs(differences) - threshold, 0.0)
    loss = float(np.mean(np.square(excess)))
    difference_gradient = (
        2.0
        * excess
        * np.sign(differences)
        / max(1, differences.size)
    )
    gradient[1:] += difference_gradient
    gradient[:-1] -= difference_gradient
    return loss, gradient


def _empty_gradients(parameters: Mapping[str, NDArray]) -> dict[str, NDArray]:
    return {
        name: np.zeros_like(np.asarray(parameters[name], dtype=float))
        for name in _PARAMETER_NAMES
    }


def _loss_values(
    model: DeepDiscoverModelConfig,
    parameters: Mapping[str, NDArray],
    trajectory: _ForwardTrajectory,
    targets: NDArray,
    *,
    data_mask: NDArray[np.bool_],
    regularization_mask: NDArray[np.bool_],
    target_scale: float,
    objective_id: str = "root_mean_square_error",
) -> dict[str, float]:
    """Calculate the DD objective without a backward pass.

    Snow-coordinate candidates use this path so discontinuous hard thresholds
    can be compared without multiplying the cost of full-sequence BPTT.
    """

    del target_scale
    data_loss, _ = objective_loss_and_gradient(
        targets[data_mask],
        trajectory.streamflow[data_mask],
        objective_id,
    )
    regularized_processes = trajectory.processes[regularization_mask]
    decorrelation_loss, _ = _decorrelation_loss_and_gradient(
        regularized_processes,
        target_correlation=model.loss.target_process_correlation,
    )
    smoothness_loss, _ = _smoothness_loss_and_gradient(
        regularized_processes,
        threshold=model.loss.smoothness_threshold,
    )
    input_switches, process_switches = _switch_values(
        parameters,
        train_input_switches=model.train_input_switches,
        train_process_switches=model.train_process_switches,
    )
    trainable_switches: list[NDArray] = []
    if model.train_input_switches:
        trainable_switches.append(input_switches)
    if model.train_process_switches:
        trainable_switches.append(process_switches)
    switch_count = sum(len(values) for values in trainable_switches)
    binary_loss = (
        float(
            sum(np.sum(values * (1.0 - values)) for values in trainable_switches)
            / switch_count
        )
        if switch_count
        else 0.0
    )
    sparsity_loss = (
        float(
            sum(np.sum(values) for values in trainable_switches)
            / switch_count
        )
        if switch_count
        else 0.0
    )
    mass_balance_loss = float(np.mean(np.square(trajectory.balance_residual)))
    total_loss = (
        data_loss
        + model.loss.decorrelation_weight * decorrelation_loss
        + model.loss.mass_balance_weight * mass_balance_loss
        + model.loss.smoothness_weight * smoothness_loss
        + model.loss.binary_weight * binary_loss
        + model.loss.sparsity_weight * sparsity_loss
    )
    return {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "decorrelation_loss": decorrelation_loss,
        "mass_balance_loss": mass_balance_loss,
        "smoothness_loss": smoothness_loss,
        "binary_loss": binary_loss,
        "sparsity_loss": sparsity_loss,
    }


def _loss_and_gradients(
    model: DeepDiscoverModelConfig,
    parameters: Mapping[str, NDArray],
    trajectory: _ForwardTrajectory,
    targets: NDArray,
    *,
    data_mask: NDArray[np.bool_],
    regularization_mask: NDArray[np.bool_],
    target_scale: float,
    objective_id: str = "root_mean_square_error",
) -> tuple[dict[str, float], dict[str, NDArray]]:
    """Return separately logged losses and exact full-sequence gradients."""

    gradients = _empty_gradients(parameters)
    process_gradient = np.zeros_like(trajectory.processes)
    del target_scale
    data_loss, data_gradient = objective_loss_and_gradient(
        targets[data_mask],
        trajectory.streamflow[data_mask],
        objective_id,
    )
    total_gradient = np.zeros(len(targets), dtype=float)
    total_gradient[data_mask] = data_gradient
    process_gradient += total_gradient[:, None]

    regularized_processes = trajectory.processes[regularization_mask]
    decorrelation_loss, decorrelation_gradient = (
        _decorrelation_loss_and_gradient(
            regularized_processes,
            target_correlation=model.loss.target_process_correlation,
        )
    )
    smoothness_loss, smoothness_gradient = _smoothness_loss_and_gradient(
        regularized_processes,
        threshold=model.loss.smoothness_threshold,
    )
    process_gradient[regularization_mask] += (
        model.loss.decorrelation_weight * decorrelation_gradient
        + model.loss.smoothness_weight * smoothness_gradient
    )

    input_switches, process_switches = _switch_values(
        parameters,
        train_input_switches=model.train_input_switches,
        train_process_switches=model.train_process_switches,
    )
    trainable_switches: list[tuple[str, NDArray]] = []
    if model.train_input_switches:
        trainable_switches.append(("input_switch_logits", input_switches))
    if model.train_process_switches:
        trainable_switches.append(("process_switch_logits", process_switches))
    switch_count = sum(len(values) for _, values in trainable_switches)
    if switch_count:
        binary_loss = float(
            sum(np.sum(values * (1.0 - values)) for _, values in trainable_switches)
            / switch_count
        )
        sparsity_loss = float(
            sum(np.sum(values) for _, values in trainable_switches) / switch_count
        )
        for name, values in trainable_switches:
            sigmoid_gradient = values * (1.0 - values)
            gradients[name] += (
                model.loss.binary_weight
                * (1.0 - 2.0 * values)
                * sigmoid_gradient
                / switch_count
                + model.loss.sparsity_weight
                * sigmoid_gradient
                / switch_count
            )
    else:
        binary_loss = 0.0
        sparsity_loss = 0.0

    future_storage_gradient = 0.0
    for time in range(len(targets) - 1, -1, -1):
        cache = trajectory.caches[time]
        allocation_weights = np.asarray(cache["allocation_weights"], dtype=float)
        denominator = float(cache["denominator"])
        remaining = float(cache["remaining"])
        current_process_gradient = process_gradient[time]
        weighted_gradient = (
            float(current_process_gradient @ allocation_weights)
            + future_storage_gradient
        )
        remaining_gradient = weighted_gradient / denominator
        allocation_gradient = (
            remaining / denominator * current_process_gradient
            - remaining * weighted_gradient / (denominator * denominator)
        )

        raw_processes = np.asarray(cache["raw_processes"], dtype=float)
        process_switch = np.asarray(cache["process_switches"], dtype=float)
        process_linear = np.asarray(cache["process_linear"], dtype=float)
        raw_gradient = allocation_gradient * process_switch
        process_linear_gradient = raw_gradient * _sigmoid(process_linear)
        if model.train_process_switches:
            process_switch_gradient = allocation_gradient * raw_processes
            gradients["process_switch_logits"] += (
                process_switch_gradient
                * process_switch
                * (1.0 - process_switch)
            )

        hidden = np.asarray(cache["hidden"], dtype=float)
        gradients["process_kernel"] += np.outer(hidden, process_linear_gradient)
        gradients["process_bias"] += process_linear_gradient
        hidden_gradient = (
            process_linear_gradient @ parameters["process_kernel"].T
        )

        aet_fraction = float(cache["aet_fraction"])
        pet_cap = float(cache["pet_cap"])
        aet_linear_gradient = (
            -remaining_gradient
            * pet_cap
            * aet_fraction
            * (1.0 - aet_fraction)
        )
        gradients["aet_kernel"] += hidden * aet_linear_gradient
        gradients["aet_bias"][0] += aet_linear_gradient
        hidden_gradient += parameters["aet_kernel"] * aet_linear_gradient

        hidden_linear_gradient = hidden_gradient * (1.0 - hidden * hidden)
        context = np.asarray(cache["context"], dtype=float)
        gradients["hidden_kernel"] += np.outer(
            context, hidden_linear_gradient
        )
        gradients["hidden_bias"] += hidden_linear_gradient
        context_gradient = hidden_linear_gradient @ parameters["hidden_kernel"].T

        scaled_features = np.asarray(cache["scaled_features"], dtype=float)
        input_weights = np.asarray(cache["input_weights"], dtype=float)
        input_switch = np.asarray(cache["input_switches"], dtype=float)
        active_gradient = context_gradient[:3]
        gradients["input_weight_raw"] += (
            active_gradient
            * scaled_features
            * input_switch
            * _sigmoid(parameters["input_weight_raw"])
        )
        if model.train_input_switches:
            input_switch_gradient = (
                active_gradient * scaled_features * input_weights
            )
            gradients["input_switch_logits"] += (
                input_switch_gradient
                * input_switch
                * (1.0 - input_switch)
            )

        state_before = float(cache["state_before"])
        storage_gradient = (
            remaining_gradient
            * (
                1.0
                - aet_fraction * float(cache["pet_cap_derivative"])
            )
        )
        storage_gradient += (
            context_gradient[3]
            * model_storage_derivative(
                state_before,
                storage_reference=float(
                    cache.get("storage_reference", 1.0)
                ),
            )
        )
        future_storage_gradient = storage_gradient

    if not model.train_input_switches:
        gradients["input_switch_logits"].fill(0.0)
    if not model.train_process_switches:
        gradients["process_switch_logits"].fill(0.0)

    mass_balance_loss = float(np.mean(np.square(trajectory.balance_residual)))
    total_loss = (
        data_loss
        + model.loss.decorrelation_weight * decorrelation_loss
        + model.loss.mass_balance_weight * mass_balance_loss
        + model.loss.smoothness_weight * smoothness_loss
        + model.loss.binary_weight * binary_loss
        + model.loss.sparsity_weight * sparsity_loss
    )
    losses = {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "decorrelation_loss": decorrelation_loss,
        "mass_balance_loss": mass_balance_loss,
        "smoothness_loss": smoothness_loss,
        "binary_loss": binary_loss,
        "sparsity_loss": sparsity_loss,
    }
    return losses, gradients


def model_storage_derivative(
    storage: float,
    *,
    storage_reference: float,
) -> float:
    """Derivative of S / (S + Sref) used by the learned mapping."""

    return storage_reference / np.square(storage + storage_reference)


def _initialize_parameters(
    model: DeepDiscoverModelConfig,
    *,
    seed: int,
) -> dict[str, NDArray]:
    rng = np.random.default_rng(seed)
    hidden_scale = np.sqrt(1.0 / 4.0)
    output_scale = np.sqrt(1.0 / model.hidden_units)
    return {
        "input_weight_raw": np.full(3, _inverse_softplus(1.0), dtype=float),
        "input_switch_logits": np.full(3, 2.0, dtype=float),
        "hidden_kernel": rng.normal(
            0.0, hidden_scale, size=(4, model.hidden_units)
        ),
        "hidden_bias": np.zeros(model.hidden_units, dtype=float),
        "aet_kernel": rng.normal(0.0, output_scale, size=model.hidden_units),
        "aet_bias": np.array([-1.0], dtype=float),
        "process_kernel": rng.normal(
            0.0,
            output_scale,
            size=(model.hidden_units, model.process_count),
        ),
        "process_bias": np.full(model.process_count, -1.0, dtype=float),
        "process_switch_logits": np.full(model.process_count, 2.0, dtype=float),
    }


def _copy_parameters(
    parameters: Mapping[str, NDArray],
) -> dict[str, NDArray]:
    return {
        name: np.asarray(parameters[name], dtype=float).copy()
        for name in _PARAMETER_NAMES
    }


def _artifact(
    model: DeepDiscoverModelConfig,
    parameters: Mapping[str, NDArray],
    *,
    input_mean: NDArray,
    input_scale: NDArray,
    storage_reference: float,
    best_epoch: int,
    snow_module_id: str,
    snow_parameters: Mapping[str, float],
    calibrated_snow_parameters: tuple[str, ...],
) -> DeepDiscoverArtifact:
    return DeepDiscoverArtifact(
        artifact_type="hydrosoftware.deepdiscover.numpy.instantaneous.v1",
        model_id=model.model_id,
        engine_id="numpy.discovery.reference.v1",
        process_dynamics="instantaneous",
        process_count=model.process_count,
        hidden_units=model.hidden_units,
        train_input_switches=model.train_input_switches,
        train_process_switches=model.train_process_switches,
        input_mean=input_mean,
        input_scale=input_scale,
        storage_reference=storage_reference,
        input_weight_raw=parameters["input_weight_raw"],
        input_switch_logits=parameters["input_switch_logits"],
        hidden_kernel=parameters["hidden_kernel"],
        hidden_bias=parameters["hidden_bias"],
        aet_kernel=parameters["aet_kernel"],
        aet_bias=parameters["aet_bias"],
        process_kernel=parameters["process_kernel"],
        process_bias=parameters["process_bias"],
        process_switch_logits=parameters["process_switch_logits"],
        best_epoch=best_epoch,
        snow_module_id=snow_module_id,
        snow_parameters=dict(snow_parameters),
        calibrated_snow_parameters=calibrated_snow_parameters,
        snow_calibration_method=(
            SNOW_COORDINATE_METHOD
            if calibrated_snow_parameters
            else SNOW_FIXED_METHOD
        ),
    )


def _artifact_parameters(
    artifact: DeepDiscoverArtifact,
) -> dict[str, NDArray]:
    return {
        name: np.asarray(getattr(artifact, name), dtype=float).copy()
        for name in _PARAMETER_NAMES
    }


def _simulation_result(
    artifact: DeepDiscoverArtifact,
    precipitation: NDArray,
    snow: _SnowTrajectory,
    trajectory: _ForwardTrajectory,
) -> SimulationResult:
    process_names = tuple(
        f"discovered_process_{index + 1}"
        for index in range(artifact.process_count)
    )
    fluxes = np.column_stack(
        (
            precipitation,
            snow.rain,
            snow.snowfall,
            snow.melt,
            trajectory.actual_evapotranspiration,
            trajectory.processes,
        )
    )
    states = np.column_stack((snow.storage, trajectory.storage))
    input_switches, process_switches = _switch_values(
        _artifact_parameters(artifact),
        train_input_switches=artifact.train_input_switches,
        train_process_switches=artifact.train_process_switches,
    )
    diagnostics: dict[str, NDArray | float | str] = {
        "engine_id": artifact.engine_id,
        "model_id": artifact.model_id,
        "prediction_layout": "sequence_to_sequence",
        "process_dynamics": "instantaneous",
        "process_count": float(artifact.process_count),
        "snow_module_id": artifact.snow_module_id or "legacy_configuration",
        "snow_calibration_method": artifact.snow_calibration_method,
        "maximum_absolute_mass_balance_error": float(
            np.max(
                np.abs(
                    trajectory.balance_residual + snow.balance_residual
                )
            )
            if len(precipitation)
            else 0.0
        ),
        "interval_mass_balance_error": (
            trajectory.balance_residual + snow.balance_residual
        ),
        "input_switch_probabilities": input_switches,
        "input_scale_weights": _softplus(artifact.input_weight_raw),
        "process_switch_probabilities": process_switches,
        "analysis__active_rain": trajectory.active_inputs[:, 0],
        "analysis__active_snowmelt": trajectory.active_inputs[:, 1],
        "analysis__active_potential_evapotranspiration": (
            trajectory.active_inputs[:, 2]
        ),
        "analysis__balance_residual": (
            trajectory.balance_residual + snow.balance_residual
        ),
        **{
            f"snow_parameter__{name}": float(value)
            for name, value in artifact.snow_parameters.items()
        },
    }
    for index in range(artifact.process_count):
        diagnostics[f"analysis__raw_process_{index + 1}"] = (
            trajectory.raw_processes[:, index]
        )
    return SimulationResult(
        outputs=trajectory.streamflow[:, None],
        states=states,
        fluxes=fluxes,
        output_names=("streamflow",),
        state_names=("snow_storage", "discovered_storage"),
        flux_names=(
            "precipitation",
            "rain",
            "snowfall",
            "snowmelt",
            "actual_evapotranspiration",
            *process_names,
        ),
        diagnostics=diagnostics,
    )


def simulate_deepdiscover_from_artifact(
    artifact: DeepDiscoverArtifact,
    snow_module: SnowModule,
    precipitation: NDArray,
    temperature: NDArray,
    potential_evapotranspiration: NDArray,
    *,
    initial_state: tuple[float, float],
) -> SimulationResult:
    """Apply a frozen DD artifact to one continuous forcing trajectory."""

    precipitation = _as_finite_vector("precipitation", precipitation)
    temperature = _as_finite_vector("temperature", temperature)
    pet = _as_finite_vector(
        "potential_evapotranspiration", potential_evapotranspiration
    )
    if not (len(precipitation) == len(temperature) == len(pet)):
        raise ValueError("DeepDiscover forcing arrays must share the time dimension.")
    if np.any(precipitation < 0.0) or np.any(pet < 0.0):
        raise ValueError("DeepDiscover precipitation and PET cannot be negative.")
    state = np.asarray(initial_state, dtype=float)
    if state.shape != (2,) or not np.all(np.isfinite(state)) or np.any(state < 0.0):
        raise ValueError(
            "DeepDiscover initial_state requires non-negative snow and discovered storage."
        )
    frozen_snow_module = snow_module_from_parameters(
        artifact.snow_module_id,
        artifact.snow_parameters,
        fallback=snow_module,
    )
    snow = _snow_trajectory(
        frozen_snow_module,
        precipitation,
        temperature,
        initial_storage=float(state[0]),
    )
    trajectory = _forward(
        _artifact_parameters(artifact),
        input_mean=artifact.input_mean,
        input_scale=artifact.input_scale,
        storage_reference=artifact.storage_reference,
        rain=snow.rain,
        melt=snow.melt,
        potential_evapotranspiration=pet,
        initial_storage=float(state[1]),
        train_input_switches=artifact.train_input_switches,
        train_process_switches=artifact.train_process_switches,
        retain_cache=False,
    )
    return _simulation_result(artifact, precipitation, snow, trajectory)


def fit_deepdiscover(
    model: DeepDiscoverModelConfig,
    training: DiscoveryAdamTrainingConfig,
    snow_module: SnowModule,
    precipitation: NDArray,
    temperature: NDArray,
    potential_evapotranspiration: NDArray,
    targets: NDArray,
    *,
    start_index: int,
    train_end_index: int,
    validation_end_index: int,
    test_end_index: int,
    warmup_steps: int,
    objective_id: str = "root_mean_square_error",
    cancellation_requested: Callable[[], bool] | None = None,
    on_epoch: Callable[[int, int, float], None] | None = None,
) -> DeepDiscoverFitResult:
    """Fit DD on training rows and restore the validation-selected checkpoint."""

    precipitation = _as_finite_vector("precipitation", precipitation)
    temperature = _as_finite_vector("temperature", temperature)
    pet = _as_finite_vector(
        "potential_evapotranspiration", potential_evapotranspiration
    )
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 1:
        raise ValueError("DeepDiscover targets must have shape [time].")
    if not (
        len(precipitation)
        == len(temperature)
        == len(pet)
        == len(targets)
    ):
        raise ValueError("DeepDiscover forcings and targets must align.")
    if not (
        0
        <= start_index
        < train_end_index
        < validation_end_index
        < test_end_index
        <= len(targets)
    ):
        raise ValueError("DeepDiscover partition boundaries are invalid.")
    if warmup_steps >= train_end_index - start_index:
        raise ValueError("DeepDiscover warm-up must be shorter than training.")

    precipitation = precipitation[start_index:test_end_index]
    temperature = temperature[start_index:test_end_index]
    pet = pet[start_index:test_end_index]
    targets = targets[start_index:test_end_index]
    train_end = train_end_index - start_index
    validation_end = validation_end_index - start_index
    n_time = len(targets)

    snow_plan = discovery_snow_plan(model.snow)
    snow_parameters = {
        name: float(value)
        for name, value in snow_plan.initial_parameters.items()
    }

    def snow_and_transforms(
        values: Mapping[str, float],
    ) -> tuple[_SnowTrajectory, NDArray, NDArray, float]:
        current_snow = _snow_trajectory(
            snow_plan.module(values),
            precipitation,
            temperature,
            initial_storage=model.initial_state[0],
        )
        raw_inputs = np.column_stack((current_snow.rain, current_snow.melt, pet))
        current_mean = np.mean(raw_inputs[:train_end], axis=0)
        current_scale = np.std(raw_inputs[:train_end], axis=0)
        current_scale[current_scale <= 1e-12] = 1.0
        current_reference = max(
            10.0,
            365.0
            * float(
                np.mean(
                    current_snow.rain[:train_end]
                    + current_snow.melt[:train_end]
                )
            ),
        )
        return current_snow, current_mean, current_scale, current_reference

    snow, input_mean, input_scale, storage_reference = snow_and_transforms(
        snow_parameters
    )

    partition_masks = {
        "training": np.zeros(n_time, dtype=bool),
        "validation": np.zeros(n_time, dtype=bool),
        "test": np.zeros(n_time, dtype=bool),
    }
    partition_masks["training"][warmup_steps:train_end] = True
    partition_masks["validation"][train_end:validation_end] = True
    partition_masks["test"][validation_end:] = True
    for name, mask in partition_masks.items():
        mask &= np.isfinite(targets)
        if not np.any(mask):
            raise ValueError(
                f"The DeepDiscover {name} partition has no finite observations."
            )
    regularization_mask = np.zeros(n_time, dtype=bool)
    regularization_mask[warmup_steps:train_end] = True
    finite_training_targets = targets[partition_masks["training"]]
    target_scale = float(np.std(finite_training_targets))
    if target_scale <= 1e-12:
        target_scale = 1.0

    parameters = _initialize_parameters(model, seed=training.seed)
    first_moment = _empty_gradients(parameters)
    second_moment = _empty_gradients(parameters)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    best_parameters = _copy_parameters(parameters)
    best_snow_parameters = dict(snow_parameters)
    best_input_mean = input_mean.copy()
    best_input_scale = input_scale.copy()
    best_storage_reference = storage_reference
    objective = objective_definition(objective_id)
    best_validation_loss = float("inf")
    best_validation_score = (
        -float("inf") if objective.direction == "maximize" else float("inf")
    )
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[Mapping[str, float | int]] = []
    cancelled = False
    coordinate_interval = snow_update_interval(training.epochs)
    snow_round = 0

    for epoch in range(1, training.epochs + 1):
        if cancellation_requested is not None and cancellation_requested():
            cancelled = True
            break
        training_trajectory = _forward(
            parameters,
            input_mean=input_mean,
            input_scale=input_scale,
            storage_reference=storage_reference,
            rain=snow.rain[:train_end],
            melt=snow.melt[:train_end],
            potential_evapotranspiration=pet[:train_end],
            initial_storage=model.initial_state[1],
            train_input_switches=model.train_input_switches,
            train_process_switches=model.train_process_switches,
            retain_cache=True,
        )
        _, gradients = _loss_and_gradients(
            model,
            parameters,
            training_trajectory,
            targets[:train_end],
            data_mask=partition_masks["training"][:train_end],
            regularization_mask=regularization_mask[:train_end],
            target_scale=target_scale,
            objective_id=objective_id,
        )
        gradient_norm = float(
            np.sqrt(
                sum(
                    np.sum(np.square(value))
                    for value in gradients.values()
                )
            )
        )
        if gradient_norm > training.gradient_clip_norm:
            scale = training.gradient_clip_norm / (gradient_norm + 1e-12)
            gradients = {
                name: value * scale for name, value in gradients.items()
            }

        for name in _PARAMETER_NAMES:
            gradient = gradients[name]
            first_moment[name] = (
                beta1 * first_moment[name] + (1.0 - beta1) * gradient
            )
            second_moment[name] = (
                beta2 * second_moment[name]
                + (1.0 - beta2) * np.square(gradient)
            )
            corrected_first = first_moment[name] / (1.0 - beta1**epoch)
            corrected_second = second_moment[name] / (1.0 - beta2**epoch)
            parameters[name] -= (
                training.learning_rate
                * corrected_first
                / (np.sqrt(corrected_second) + epsilon)
            )

        snow_changed = False
        if (
            snow_plan.calibrated_names
            and (epoch == 1 or epoch % coordinate_interval == 0)
        ):
            def snow_objective(candidate: Mapping[str, float]) -> float:
                (
                    candidate_snow,
                    candidate_mean,
                    candidate_scale,
                    candidate_reference,
                ) = snow_and_transforms(candidate)
                candidate_trajectory = _forward(
                    parameters,
                    input_mean=candidate_mean,
                    input_scale=candidate_scale,
                    storage_reference=candidate_reference,
                    rain=candidate_snow.rain[:train_end],
                    melt=candidate_snow.melt[:train_end],
                    potential_evapotranspiration=pet[:train_end],
                    initial_storage=model.initial_state[1],
                    train_input_switches=model.train_input_switches,
                    train_process_switches=model.train_process_switches,
                    retain_cache=False,
                )
                return _loss_values(
                    model,
                    parameters,
                    candidate_trajectory,
                    targets[:train_end],
                    data_mask=partition_masks["training"][:train_end],
                    regularization_mask=regularization_mask[:train_end],
                    target_scale=target_scale,
                    objective_id=objective_id,
                )["total_loss"]

            snow_parameters, _, snow_changed = coordinate_snow_update(
                snow_plan,
                snow_parameters,
                round_index=snow_round,
                objective=snow_objective,
            )
            snow_round += 1
            if snow_changed:
                (
                    snow,
                    input_mean,
                    input_scale,
                    storage_reference,
                ) = snow_and_transforms(snow_parameters)

        evaluation = _forward(
            parameters,
            input_mean=input_mean,
            input_scale=input_scale,
            storage_reference=storage_reference,
            rain=snow.rain[:validation_end],
            melt=snow.melt[:validation_end],
            potential_evapotranspiration=pet[:validation_end],
            initial_storage=model.initial_state[1],
            train_input_switches=model.train_input_switches,
            train_process_switches=model.train_process_switches,
            retain_cache=False,
        )
        training_mask = partition_masks["training"][:validation_end]
        validation_mask = partition_masks["validation"][:validation_end]
        evaluation_targets = targets[:validation_end]
        training_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        evaluation.streamflow[training_mask]
                        - evaluation_targets[training_mask]
                    )
                )
            )
        )
        validation_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        evaluation.streamflow[validation_mask]
                        - evaluation_targets[validation_mask]
                    )
                )
            )
        )
        training_objective, _ = objective_optimization_score(
            evaluation_targets[training_mask],
            evaluation.streamflow[training_mask],
            objective_id,
        )
        validation_objective, validation_loss = objective_optimization_score(
            evaluation_targets[validation_mask],
            evaluation.streamflow[validation_mask],
            objective_id,
        )
        epoch_losses = _loss_values(
            model,
            parameters,
            evaluation,
            evaluation_targets,
            data_mask=training_mask,
            regularization_mask=regularization_mask[:validation_end],
            target_scale=target_scale,
            objective_id=objective_id,
        )
        history.append(
            {
                "epoch": epoch,
                "training_rmse": training_rmse,
                "validation_rmse": validation_rmse,
                "training_objective": training_objective,
                "validation_objective": validation_objective,
                "gradient_norm_before_clipping": gradient_norm,
                "snow_coordinate_round": snow_round,
                "snow_parameters_changed": int(snow_changed),
                **{
                    f"snow_parameter__{name}": float(value)
                    for name, value in snow_parameters.items()
                },
                **epoch_losses,
            }
        )
        if validation_loss < best_validation_loss - training.minimum_delta:
            best_validation_loss = validation_loss
            best_validation_score = validation_objective
            best_epoch = epoch
            best_parameters = _copy_parameters(parameters)
            best_snow_parameters = dict(snow_parameters)
            best_input_mean = input_mean.copy()
            best_input_scale = input_scale.copy()
            best_storage_reference = storage_reference
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if on_epoch is not None:
            on_epoch(epoch, training.epochs, best_validation_score)
        if epochs_without_improvement >= training.patience:
            break

    artifact = _artifact(
        model,
        best_parameters,
        input_mean=best_input_mean,
        input_scale=best_input_scale,
        storage_reference=best_storage_reference,
        best_epoch=best_epoch,
        snow_module_id=snow_plan.module_id,
        snow_parameters=best_snow_parameters,
        calibrated_snow_parameters=snow_plan.calibrated_names,
    )
    best_snow = _snow_trajectory(
        snow_plan.module(best_snow_parameters),
        precipitation,
        temperature,
        initial_storage=model.initial_state[0],
    )
    final_trajectory = _forward(
        best_parameters,
        input_mean=best_input_mean,
        input_scale=best_input_scale,
        storage_reference=best_storage_reference,
        rain=best_snow.rain,
        melt=best_snow.melt,
        potential_evapotranspiration=pet,
        initial_storage=model.initial_state[1],
        train_input_switches=model.train_input_switches,
        train_process_switches=model.train_process_switches,
        retain_cache=False,
    )
    return DeepDiscoverFitResult(
        artifact=artifact,
        simulation=_simulation_result(
            artifact, precipitation, best_snow, final_trajectory
        ),
        history=tuple(history),
        partition_masks=partition_masks,
        stopped_early=(
            bool(history)
            and len(history) < training.epochs
            and not cancelled
        ),
        cancelled=cancelled,
    )


def deepdiscover_loss_and_gradients(
    model: DeepDiscoverModelConfig,
    artifact: DeepDiscoverArtifact,
    snow_module: SnowModule,
    precipitation: NDArray,
    temperature: NDArray,
    potential_evapotranspiration: NDArray,
    targets: NDArray,
    *,
    data_mask: NDArray[np.bool_],
    regularization_mask: NDArray[np.bool_],
    initial_state: tuple[float, float],
    target_scale: float = 1.0,
    objective_id: str = "root_mean_square_error",
) -> tuple[dict[str, float], dict[str, NDArray]]:
    """Testing hook for analytical-versus-finite-difference gradient checks."""

    precipitation = _as_finite_vector("precipitation", precipitation)
    temperature = _as_finite_vector("temperature", temperature)
    pet = _as_finite_vector(
        "potential_evapotranspiration", potential_evapotranspiration
    )
    snow = _snow_trajectory(
        snow_module,
        precipitation,
        temperature,
        initial_storage=initial_state[0],
    )
    parameters = _artifact_parameters(artifact)
    trajectory = _forward(
        parameters,
        input_mean=artifact.input_mean,
        input_scale=artifact.input_scale,
        storage_reference=artifact.storage_reference,
        rain=snow.rain,
        melt=snow.melt,
        potential_evapotranspiration=pet,
        initial_storage=initial_state[1],
        train_input_switches=artifact.train_input_switches,
        train_process_switches=artifact.train_process_switches,
        retain_cache=True,
    )
    return _loss_and_gradients(
        model,
        parameters,
        trajectory,
        np.asarray(targets, dtype=float),
        data_mask=np.asarray(data_mask, dtype=bool),
        regularization_mask=np.asarray(regularization_mask, dtype=bool),
        target_scale=target_scale,
        objective_id=objective_id,
    )
