"""Frozen-parameter simulations on new forcing partitions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from hydrosoftware.configuration import (
    Cnn1dModelConfig,
    DeepDiscoverModelConfig,
    DegreeDaySnowConfig,
    ExperimentConfigV1,
    GruModelConfig,
    LstmModelConfig,
    MlpModelConfig,
    NoSnowConfig,
    PamlModelConfig,
    SmoothDegreeDaySnowConfig,
    TriangularMaxbasRoutingConfig,
    configuration_hash,
)
from hydrosoftware.core.evaluation import evaluate_metrics
from hydrosoftware.core.features import (
    LaggedFeature,
    build_forecast_feature_mapping,
    maximum_lag,
    required_raw_sources,
)
from hydrosoftware.core.results import SimulationResult
from hydrosoftware.data_sources.quality import DataQualityAnalysis
from hydrosoftware.execution.ensembles import (
    ENSEMBLE_ARTIFACT_TYPE,
    aggregate_member_predictions,
    member_artifact_payloads,
)
from hydrosoftware.execution.serialization import to_jsonable
from hydrosoftware.models.conceptual.exp_hydro import ExpHydroModel, ExpHydroParameters
from hydrosoftware.models.conceptual.hbv import HbvModel, HbvParameters
from hydrosoftware.models.neural import (
    S2S_ARTIFACT_TYPE,
    S2SArtifact,
    TensorFlowArtifact,
    predict_from_artifact,
    predict_s2s_from_artifact,
)
from hydrosoftware.models.paml_tensorflow import simulate_from_artifact as simulate_paml_from_artifact
from hydrosoftware.models.deepdiscover_numpy import (
    DeepDiscoverArtifact,
    simulate_deepdiscover_from_artifact,
)
from hydrosoftware.processes.routing import (
    MAXBAS_MEMORY_SIZE,
    TriangularMaxbasRouting,
    route_simulation_result,
)
from hydrosoftware.processes.snow import (
    DegreeDaySnowModule,
    NoSnowModule,
    SmoothDegreeDaySnowModule,
)


def _restore_neural_artifacts(
    model: MlpModelConfig | LstmModelConfig | GruModelConfig | Cnn1dModelConfig,
    payload: Mapping[str, Any],
) -> tuple[list[Any], Any, str]:
    """Restore one TensorFlow neural artifact or an initialization ensemble."""

    member_payloads = member_artifact_payloads(payload)
    if not member_payloads:
        raise ValueError("The saved neural artifact does not contain any model member.")
    is_s2s = str(member_payloads[0].get("artifact_type", "")) == S2S_ARTIFACT_TYPE
    artifacts = [
        (S2SArtifact if is_s2s else TensorFlowArtifact).from_mapping(member)
        for member in member_payloads
    ]
    family = model.model_id.split(".", 1)[0]
    label = {"cnn1d": "CNN-1D", "lstm": "LSTM", "gru": "GRU", "mlp": "MLP"}.get(
        family, family.upper()
    )
    return artifacts, (predict_s2s_from_artifact if is_s2s else predict_from_artifact), label

def _predict_neural_artifacts(
    payload: Mapping[str, Any],
    artifacts: Sequence[Any],
    predict: Any,
    features: Mapping[str, np.ndarray],
    *,
    antecedent_features: Mapping[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict every member and apply the archived ensemble aggregation."""

    predictions = [
        predict(
            artifact,
            features,
            antecedent_features=antecedent_features,
        )
        for artifact in artifacts
    ]
    lower_quantile = (
        float(payload.get("lower_quantile", 0.1))
        if str(payload.get("artifact_type", "")) == ENSEMBLE_ARTIFACT_TYPE
        else 0.0
    )
    upper_quantile = (
        float(payload.get("upper_quantile", 0.9))
        if str(payload.get("artifact_type", "")) == ENSEMBLE_ARTIFACT_TYPE
        else 1.0
    )
    return aggregate_member_predictions(
        predictions,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )


def _parameter(parameters: Mapping[str, float], name: str) -> float:
    try:
        value = float(parameters[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"The saved model is missing parameter {name!r}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"Saved parameter {name!r} must be finite.")
    return value


def _snow_module(config: ExperimentConfigV1, parameters: Mapping[str, float]):
    snow = config.model.snow
    if isinstance(snow, NoSnowConfig):
        return NoSnowModule()

    def value(name: str) -> float:
        return float(parameters.get(name, getattr(snow, name)))

    if isinstance(snow, DegreeDaySnowConfig):
        return DegreeDaySnowModule(
            snowfall_temperature=value("snowfall_temperature"),
            melt_temperature=value("melt_temperature"),
            degree_day_factor=value("degree_day_factor"),
        )
    if isinstance(snow, SmoothDegreeDaySnowConfig):
        return SmoothDegreeDaySnowModule(
            snowfall_temperature=value("snowfall_temperature"),
            melt_temperature=value("melt_temperature"),
            degree_day_factor=value("degree_day_factor"),
            sharpness=value("sharpness"),
        )
    raise TypeError(f"Unsupported snow configuration: {type(snow).__name__}")


def simulate_frozen_model(
    config: ExperimentConfigV1,
    parameters: Mapping[str, float],
    precipitation: np.ndarray,
    temperature: np.ndarray,
    potential_evapotranspiration: np.ndarray,
    *,
    initial_state: Sequence[float],
) -> SimulationResult:
    """Run a configured conceptual model without changing saved parameters."""

    forcing = (
        np.asarray(precipitation, dtype=float),
        np.asarray(temperature, dtype=float),
        np.asarray(potential_evapotranspiration, dtype=float),
    )
    state = tuple(float(value) for value in initial_state)
    model_state_size = len(config.model.initial_state)
    routed = isinstance(
        getattr(config.model, "routing", None), TriangularMaxbasRoutingConfig
    )
    expected_state_size = model_state_size + (MAXBAS_MEMORY_SIZE if routed else 0)
    if len(state) != expected_state_size:
        raise ValueError(
            f"The forecast initial state requires {expected_state_size} values, "
            f"but {len(state)} were supplied."
        )
    if not np.all(np.isfinite(state)) or any(value < 0.0 for value in state):
        raise ValueError("Forecast initial states must be finite and non-negative.")
    model_state = state[:model_state_size]
    snow = _snow_module(config, parameters)

    if config.model.model_id == "exp_hydro.mass_conservative.v1":
        model = ExpHydroModel(
            ExpHydroParameters(
                recession_factor=_parameter(parameters, "recession_factor"),
                maximum_storage=_parameter(parameters, "maximum_storage"),
                maximum_subsurface_flow=_parameter(
                    parameters, "maximum_subsurface_flow"
                ),
            ),
            snow,
            substeps=config.model.substeps,
        )
        return model.simulate(*forcing, initial_state=model_state)

    if config.model.model_id == "hbv.mass_conservative_daily.v1":
        model = HbvModel(
            HbvParameters(
                field_capacity=_parameter(parameters, "field_capacity"),
                recharge_exponent=_parameter(parameters, "recharge_exponent"),
                evap_reduction_threshold=_parameter(
                    parameters, "evap_reduction_threshold"
                ),
                maximum_percolation=_parameter(
                    parameters, "maximum_percolation"
                ),
                upper_zone_threshold=_parameter(
                    parameters, "upper_zone_threshold"
                ),
                fast_recession=_parameter(parameters, "fast_recession"),
                intermediate_recession=_parameter(
                    parameters, "intermediate_recession"
                ),
                slow_recession=_parameter(parameters, "slow_recession"),
            ),
            snow,
        )
        simulation = model.simulate(*forcing, initial_state=model_state)
        routing = config.model.routing
        if not isinstance(routing, TriangularMaxbasRoutingConfig):
            return simulation
        maxbas_value = parameters.get("maxbas", routing.maxbas)
        if not float(maxbas_value).is_integer():
            raise ValueError("Saved MAXBAS must be a whole number of days.")
        return route_simulation_result(
            simulation,
            TriangularMaxbasRouting(int(maxbas_value)),
            initial_state=state[model_state_size:],
        )

    raise ValueError(f"Unsupported conceptual model: {config.model.model_id}")


def configured_full_state(config: ExperimentConfigV1) -> tuple[float, ...]:
    """Return the complete configured state, including routing memory."""

    state = tuple(float(value) for value in config.model.initial_state)
    routing = getattr(config.model, "routing", None)
    if isinstance(routing, TriangularMaxbasRoutingConfig):
        state += tuple(float(value) for value in routing.initial_state)
    return state


def _raw_predictors(data: DataQualityAnalysis) -> dict[str, np.ndarray]:
    return {
        "precipitation": data.precipitation,
        "temperature": data.temperature,
        "potential_evapotranspiration": data.potential_evapotranspiration,
        **{
            name: np.asarray(values, dtype=float)
            for name, values in data.additional_predictors.items()
        },
    }


def _lag_recipes(model: Any) -> tuple[LaggedFeature, ...]:
    return tuple(
        LaggedFeature(source=item.source, lag_steps=item.lag_steps)
        for item in model.lagged_features
    )


def _physical_feature_sources(
    physical: SimulationResult,
    external: Mapping[str, Sequence[float]],
) -> dict[str, np.ndarray]:
    result = {
        name: np.asarray(values, dtype=float)
        for name, values in external.items()
    }
    result.update(
        {
            f"physical_output__{name}": physical.outputs[:, position]
            for position, name in enumerate(physical.output_names)
        }
    )
    result.update(
        {
            f"physical_state__{name}": physical.states[1:, position]
            for position, name in enumerate(physical.state_names)
        }
    )
    result.update(
        {
            f"physical_flux__{name}": physical.fluxes[:, position]
            for position, name in enumerate(physical.flux_names)
        }
    )
    return result


def run_forecast(
    source_archive: Mapping[str, Any],
    data: DataQualityAnalysis,
    *,
    initialization: str,
    observations_available: bool,
    custom_initial_state: Sequence[float] | None = None,
    antecedent_forcings: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Apply one archived calibrated model to new forcings."""

    config = ExperimentConfigV1.model_validate(source_archive.get("config"))
    source_result = source_archive.get("result") or {}
    if isinstance(config.model, DeepDiscoverModelConfig):
        return _run_discovery_forecast(
            config,
            source_archive,
            source_result,
            data,
            initialization=initialization,
            observations_available=observations_available,
            custom_initial_state=custom_initial_state,
            antecedent_forcings=antecedent_forcings,
        )
    if isinstance(config.model, PamlModelConfig):
        return _run_paml_forecast(
            config,
            source_archive,
            source_result,
            data,
            initialization=initialization,
            observations_available=observations_available,
            custom_initial_state=custom_initial_state,
            antecedent_forcings=antecedent_forcings,
        )
    if isinstance(
        config.model,
        (MlpModelConfig, LstmModelConfig, GruModelConfig, Cnn1dModelConfig),
    ):
        return _run_neural_forecast(
            config,
            source_archive,
            source_result,
            data,
            initialization=initialization,
            observations_available=observations_available,
            custom_initial_state=custom_initial_state,
            antecedent_forcings=antecedent_forcings,
        )
    if antecedent_forcings is not None:
        raise ValueError("Antecedent forcing context is supported only by neural models.")
    calibration = source_result.get("calibration") or {}
    parameters = calibration.get("best_parameters")
    source_simulation = source_result.get("best_simulation") or {}
    source_states = source_simulation.get("states") or []
    if source_result.get("state") != "completed" or not parameters or not source_states:
        raise ValueError(
            "The selected archive does not contain a completed calibrated model."
        )

    if initialization == "source_final":
        initial_state = tuple(float(value) for value in source_states[-1])
    elif initialization == "configured":
        initial_state = configured_full_state(config)
    elif initialization == "custom":
        if custom_initial_state is None:
            raise ValueError("Custom initialization requires initial_state values.")
        initial_state = tuple(float(value) for value in custom_initial_state)
    else:
        raise ValueError(f"Unsupported forecast initialization: {initialization}")

    simulation = simulate_frozen_model(
        config,
        parameters,
        data.precipitation,
        data.temperature,
        data.potential_evapotranspiration,
        initial_state=initial_state,
    )
    observations = data.streamflow if observations_available else None
    metrics = ()
    if observations is not None and np.any(np.isfinite(observations)):
        metrics = evaluate_metrics(
            observations,
            simulation.outputs[:, 0],
            np.ones(len(observations), dtype=bool),
        )

    scored = int(np.count_nonzero(np.isfinite(observations))) if observations is not None else 0
    result = {
        "task_id": f"forecast-{uuid4()}",
        "run_kind": "forecast",
        "source_task_id": str(source_archive.get("task_id")),
        "state": "completed",
        "experiment_id": config.experiment_id,
        "configuration_hash": configuration_hash(config),
        "model_id": config.model.model_id,
        "initialization": initialization,
        "initial_state": initial_state,
        "validation": {
            "experiment_id": config.experiment_id,
            "configuration_hash": configuration_hash(config),
            "source_rows": len(data.precipitation),
            "selected_rows": len(data.precipitation),
            "warmup_steps": 0,
            "scored_observations": scored,
            "transformed_value_count": data.report.transformed_value_count,
            "data_quality_warning_count": data.report.warning_count,
            "warnings": [],
        },
        # This identifies the saved source run; no calibration is performed here.
        "calibration": calibration,
        "best_simulation": simulation,
        "timestamps": data.timestamps,
        "observations": observations,
        "forcings": {
            "precipitation": data.precipitation,
            "temperature": data.temperature,
            "potential_evapotranspiration": data.potential_evapotranspiration,
            **data.additional_predictors,
        },
        "evaluation_metrics": metrics,
        "data_quality": data.report,
    }
    return to_jsonable(result)


def _run_discovery_forecast(
    config: ExperimentConfigV1,
    source_archive: Mapping[str, Any],
    source_result: Mapping[str, Any],
    data: DataQualityAnalysis,
    *,
    initialization: str,
    observations_available: bool,
    custom_initial_state: Sequence[float] | None,
    antecedent_forcings: Mapping[str, Sequence[float]] | None,
) -> dict[str, Any]:
    """Apply a frozen DD mapping with an explicit complete-state restart."""

    if not isinstance(config.model, DeepDiscoverModelConfig):
        raise TypeError("DeepDiscover forecasting requires its model configuration.")
    if antecedent_forcings is not None:
        raise ValueError(
            "DeepDiscover forecasts carry complete model states, not "
            "antecedent predictor windows."
        )
    training = source_result.get("training") or {}
    artifact_payload = training.get("artifact")
    source_simulation = source_result.get("best_simulation") or {}
    source_states = source_simulation.get("states") or []
    if (
        source_result.get("state") != "completed"
        or not artifact_payload
        or not source_states
    ):
        raise ValueError(
            "The selected archive does not contain a completed DeepDiscover artifact."
        )
    if initialization == "source_final":
        initial_state = tuple(float(value) for value in source_states[-1])
    elif initialization == "configured":
        initial_state = configured_full_state(config)
    elif initialization == "custom":
        if custom_initial_state is None:
            raise ValueError("Custom initialization requires initial_state values.")
        initial_state = tuple(float(value) for value in custom_initial_state)
    else:
        raise ValueError(
            f"Unsupported DeepDiscover forecast initialization: {initialization}"
        )
    artifact = DeepDiscoverArtifact.from_mapping(artifact_payload)
    simulation = simulate_deepdiscover_from_artifact(
        artifact,
        _snow_module(config, {}),
        data.precipitation,
        data.temperature,
        data.potential_evapotranspiration,
        initial_state=initial_state,
    )
    observations = data.streamflow if observations_available else None
    metric_mask = np.ones(len(data.precipitation), dtype=bool)
    if observations is not None:
        metric_mask &= np.isfinite(observations)
    metrics = (
        evaluate_metrics(
            observations,
            simulation.outputs[:, 0],
            metric_mask,
        )
        if observations is not None and np.any(metric_mask)
        else ()
    )
    scored = int(np.count_nonzero(metric_mask)) if observations is not None else 0
    result = {
        "task_id": f"forecast-{uuid4()}",
        "run_kind": "forecast",
        "source_task_id": str(source_archive.get("task_id")),
        "state": "completed",
        "experiment_id": config.experiment_id,
        "configuration_hash": configuration_hash(config),
        "model_id": config.model.model_id,
        "initialization": initialization,
        "initial_state": initial_state,
        "validation": {
            "experiment_id": config.experiment_id,
            "configuration_hash": configuration_hash(config),
            "source_rows": len(data.precipitation),
            "selected_rows": len(data.precipitation),
            "warmup_steps": 0,
            "scored_observations": scored,
            "transformed_value_count": data.report.transformed_value_count,
            "data_quality_warning_count": data.report.warning_count,
            "warnings": [],
        },
        "calibration": None,
        "training": training,
        "best_simulation": simulation,
        "timestamps": data.timestamps,
        "observations": observations,
        "forcings": {
            "precipitation": data.precipitation,
            "temperature": data.temperature,
            "potential_evapotranspiration": data.potential_evapotranspiration,
            **data.additional_predictors,
        },
        "evaluation_metrics": metrics,
        "data_quality": data.report,
    }
    return to_jsonable(result)


def _run_neural_forecast(
    config: ExperimentConfigV1,
    source_archive: Mapping[str, Any],
    source_result: Mapping[str, Any],
    data: DataQualityAnalysis,
    *,
    initialization: str,
    observations_available: bool,
    custom_initial_state: Sequence[float] | None,
    antecedent_forcings: Mapping[str, Sequence[float]] | None,
) -> dict[str, Any]:
    """Apply an immutable neural artifact with optional antecedent context."""

    if custom_initial_state is not None or initialization == "custom":
        raise ValueError("Neural forecasts use antecedent forcings, not storage states.")
    training = source_result.get("training") or {}
    artifact_payload = training.get("artifact")
    if source_result.get("state") != "completed" or not artifact_payload:
        raise ValueError(
            "The selected archive does not contain a completed neural model artifact."
        )
    artifacts, predict, model_label = _restore_neural_artifacts(
        config.model, artifact_payload
    )
    artifact = artifacts[0]
    artifact_warmup = int(
        getattr(artifact, "warmup_length", artifact.lookback)
    )
    current_raw = _raw_predictors(data)
    recipes = _lag_recipes(config.model)
    antecedent_raw = None
    s2s_context_length = 0
    if initialization == "source_final":
        source_forcings = (
            antecedent_forcings
            or source_result.get("neural_context")
            or source_result.get("forcings")
            or {}
        )
        required = required_raw_sources(config.model.feature_roles, recipes)
        missing = [name for name in required if name not in source_forcings]
        if missing:
            raise ValueError(
                f"The source archive lacks antecedent forcings required by the {model_label}: "
                + ", ".join(missing)
            )
        if str(artifact.artifact_type) == S2S_ARTIFACT_TYPE:
            diagnostics = (source_result.get("best_simulation") or {}).get(
                "diagnostics"
            ) or {}
            recorded_context = diagnostics.get(
                "s2s_final_block_context_length"
            )
            if recorded_context is not None:
                s2s_context_length = int(recorded_context)
            elif source_result.get("run_kind") == "training":
                period = config.task.period
                final_length = int(period.test_end_index) - int(
                    period.validation_end_index
                )
                step = artifact.lookback - artifact_warmup
                s2s_context_length = artifact_warmup + final_length % step
            else:
                source_outputs = (
                    source_result.get("best_simulation") or {}
                ).get("outputs") or []
                source_length = len(source_outputs)
                if source_length < artifact_warmup:
                    s2s_context_length = source_length
                else:
                    step = artifact.lookback - artifact_warmup
                    s2s_context_length = artifact_warmup + (
                        (source_length - artifact_warmup) % step
                    )
            if not 0 <= s2s_context_length < artifact.lookback:
                raise ValueError(
                    "The archived S2S block context is outside the model lookback."
                )
            history_length = s2s_context_length + maximum_lag(recipes)
        else:
            history_length = artifact.lookback + maximum_lag(recipes)
        antecedent_raw = {
            name: (
                np.asarray(source_forcings[name], dtype=float)[-history_length:]
                if history_length
                else np.asarray(source_forcings[name], dtype=float)[:0]
            )
            for name in required
        }
    elif initialization != "configured":
        raise ValueError(f"Unsupported neural forecast initialization: {initialization}")

    current_features, antecedent = build_forecast_feature_mapping(
        current_raw,
        config.model.feature_roles,
        recipes,
        lookback=artifact.lookback,
        antecedent_raw_features=antecedent_raw,
    )
    prediction, ensemble_lower, ensemble_upper = _predict_neural_artifacts(
        artifact_payload,
        artifacts,
        predict,
        current_features,
        antecedent_features=antecedent,
    )
    is_ensemble = len(artifacts) > 1
    final_s2s_context = None
    if str(artifact.artifact_type) == S2S_ARTIFACT_TYPE:
        total_length = s2s_context_length + len(data.precipitation)
        if total_length < artifact_warmup:
            final_s2s_context = total_length
        else:
            step = artifact.lookback - artifact_warmup
            final_s2s_context = artifact_warmup + (
                (total_length - artifact_warmup) % step
            )
    simulation = SimulationResult(
        outputs=prediction[:, None],
        states=np.empty((len(prediction) + 1, 0), dtype=float),
        fluxes=np.empty((len(prediction), 0), dtype=float),
        output_names=("streamflow",),
        state_names=(),
        flux_names=(),
        diagnostics={
            "engine_id": (
                "neural_initialization_ensemble.v1"
                if is_ensemble
                else artifact.engine_id
            ),
            "model_id": artifact.model_id,
            "prediction_layout": (
                "sequence_to_sequence"
                if str(artifact.artifact_type) == S2S_ARTIFACT_TYPE
                else "sequence_to_value"
            ),
            "lookback": artifact.lookback,
            "s2s_warmup": (
                artifact_warmup
                if str(artifact.artifact_type) == S2S_ARTIFACT_TYPE
                else None
            ),
            "s2s_final_block_context_length": final_s2s_context,
            **(
                {
                    "initialization_ensemble_method": (
                        "neural_initialization_ensemble.v1"
                    ),
                    "initialization_ensemble_members": len(artifacts),
                    "initialization_ensemble_lower_quantile": float(
                        artifact_payload.get("lower_quantile", 0.1)
                    ),
                    "initialization_ensemble_upper_quantile": float(
                        artifact_payload.get("upper_quantile", 0.9)
                    ),
                    "initialization_ensemble_lower": ensemble_lower,
                    "initialization_ensemble_upper": ensemble_upper,
                }
                if is_ensemble
                else {}
            ),
            "antecedent_context": (
                "source_final_forcings" if antecedent is not None else "forecast_rows_only"
            ),
        },
    )
    observations = data.streamflow if observations_available else None
    metric_mask = np.isfinite(prediction)
    if observations is not None:
        metric_mask &= np.isfinite(observations)
    metrics = ()
    if observations is not None and np.any(metric_mask):
        metrics = evaluate_metrics(observations, prediction, metric_mask)
    scored = int(np.count_nonzero(metric_mask)) if observations is not None else 0
    raw_context = {}
    required = required_raw_sources(config.model.feature_roles, recipes)
    history_length = artifact.lookback + maximum_lag(recipes)
    for name in required:
        current_values = np.asarray(current_raw[name], dtype=float)
        combined = (
            np.concatenate((antecedent_raw[name], current_values))
            if antecedent_raw is not None
            else current_values
        )
        raw_context[name] = combined[-history_length:]
    result = {
        "task_id": f"forecast-{uuid4()}",
        "run_kind": "forecast",
        "source_task_id": str(source_archive.get("task_id")),
        "state": "completed",
        "experiment_id": config.experiment_id,
        "configuration_hash": configuration_hash(config),
        "model_id": config.model.model_id,
        "initialization": initialization,
        "initial_state": None,
        "validation": {
            "experiment_id": config.experiment_id,
            "configuration_hash": configuration_hash(config),
            "source_rows": len(data.precipitation),
            "selected_rows": len(data.precipitation),
            "warmup_steps": 0 if antecedent is not None else artifact_warmup,
            "scored_observations": scored,
            "transformed_value_count": data.report.transformed_value_count,
            "data_quality_warning_count": data.report.warning_count,
            "warnings": [],
        },
        "calibration": None,
        "training": training,
        "best_simulation": simulation,
        "timestamps": data.timestamps,
        "observations": observations,
        "forcings": current_raw,
        "neural_context": raw_context,
        "evaluation_metrics": metrics,
        "data_quality": data.report,
    }
    return to_jsonable(result)


def _run_paml_forecast(
    config: ExperimentConfigV1,
    source_archive: Mapping[str, Any],
    source_result: Mapping[str, Any],
    data: DataQualityAnalysis,
    *,
    initialization: str,
    observations_available: bool,
    custom_initial_state: Sequence[float] | None,
    antecedent_forcings: Mapping[str, Sequence[float]] | None,
) -> dict[str, Any]:
    """Forecast with a saved joint P-RNN and causal-CNN PaML model."""

    if not isinstance(config.model, PamlModelConfig):
        raise TypeError("PaML forecasting requires a PaML configuration.")
    training = source_result.get("training") or {}
    artifact_payload = training.get("artifact")
    if source_result.get("state") != "completed" or not artifact_payload:
        raise ValueError("The selected archive does not contain a completed PaML artifact.")

    current_external = _raw_predictors(data)
    prepend = None
    if initialization == "source_final":
        source_external = {
            name: np.asarray(values, dtype=float)
            for name, values in (source_result.get("forcings") or {}).items()
        }
        required = (
            "precipitation",
            "temperature",
            "potential_evapotranspiration",
        )
        missing = [name for name in required if name not in source_external]
        if missing:
            raise ValueError(
                "The source PaML archive lacks antecedent forcing: "
                + ", ".join(missing)
            )
        prepend = {name: source_external[name] for name in required}
    elif initialization == "configured":
        prepend = None
    elif initialization == "custom":
        if custom_initial_state is not None:
            raise ValueError(
                "PaML custom state initialization is not supported; provide antecedent forcings instead."
            )
        if antecedent_forcings is None:
            raise ValueError("Custom PaML initialization requires antecedent forcings.")
        prepend = {
            name: np.asarray(antecedent_forcings[name], dtype=float)
            for name in (
                "precipitation",
                "temperature",
                "potential_evapotranspiration",
            )
        }
    else:
        raise ValueError(f"Unsupported forecast initialization: {initialization}")

    simulation = simulate_paml_from_artifact(
        artifact_payload,
        current_external["precipitation"],
        current_external["temperature"],
        current_external["potential_evapotranspiration"],
        prepend_forcings=prepend,
        enforce_nonnegative_output=config.model.enforce_nonnegative_output,
    )
    corrected = simulation.outputs[:, 0]
    observations = data.streamflow if observations_available else None
    metric_mask = np.isfinite(corrected)
    if observations is not None:
        metric_mask &= np.isfinite(observations)
    metrics = (
        evaluate_metrics(observations, corrected, metric_mask)
        if observations is not None and np.any(metric_mask)
        else ()
    )
    scored = int(np.count_nonzero(metric_mask)) if observations is not None else 0
    result = {
        "task_id": f"forecast-{uuid4()}",
        "run_kind": "forecast",
        "source_task_id": str(source_archive.get("task_id")),
        "state": "completed",
        "experiment_id": config.experiment_id,
        "configuration_hash": configuration_hash(config),
        "model_id": config.model.model_id,
        "initialization": initialization,
        "initial_state": None,
        "validation": {
            "experiment_id": config.experiment_id,
            "configuration_hash": configuration_hash(config),
            "source_rows": len(data.precipitation),
            "selected_rows": len(data.precipitation),
            "warmup_steps": 0 if prepend is not None else config.task.warmup_steps,
            "scored_observations": scored,
            "transformed_value_count": data.report.transformed_value_count,
            "data_quality_warning_count": data.report.warning_count,
            "warnings": [],
        },
        "calibration": None,
        "training": training,
        "best_simulation": simulation,
        "timestamps": data.timestamps,
        "observations": observations,
        "forcings": current_external,
        "neural_context": None,
        "evaluation_metrics": metrics,
        "data_quality": data.report,
    }
    return to_jsonable(result)
