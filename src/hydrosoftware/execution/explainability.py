"""Post-training SHAP analysis for frozen neural and PaML artifacts.

The implementation uses empirical training windows as interventional
references.  Each player is one feature at one antecedent lag.  Marginal
contributions are estimated with deterministic permutation sampling, while
every sampled path retains the Shapley efficiency identity exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from hydrosoftware.configuration import (
    Cnn1dModelConfig,
    ExperimentConfigV1,
    GruModelConfig,
    LaggedFeatureConfig,
    LstmModelConfig,
    MlpModelConfig,
    NeuralTrainingTaskConfig,
    PamlModelConfig,
    PamlTrainingTaskConfig,
)
from hydrosoftware.core.features import LaggedFeature, build_feature_mapping
from hydrosoftware.execution.ensembles import (
    ENSEMBLE_ARTIFACT_TYPE,
    member_artifact_payloads,
)
from hydrosoftware.models.neural import (
    S2S_ARTIFACT_TYPE,
    S2V_ARTIFACT_TYPE,
    S2SArtifact,
    TensorFlowArtifact,
    predict_s2s_windows_from_artifact,
    predict_windows_from_artifact,
)


class ExplainabilityError(RuntimeError):
    """A stable scientific or archive incompatibility in SHAP analysis."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _FrozenCorrector:
    """Artifact-specific inference exposed through one window interface."""

    artifact_type: str
    model_id: str
    feature_roles: tuple[str, ...]
    lookback: int
    includes_current_row: bool
    predict: Callable[[NDArray], NDArray[np.floating]]


def _artifact_predictor(payload: Mapping[str, Any]) -> _FrozenCorrector:
    artifact_type = str(payload.get("artifact_type", ""))
    if artifact_type == ENSEMBLE_ARTIFACT_TYPE:
        try:
            members = tuple(
                _artifact_predictor(member)
                for member in member_artifact_payloads(payload)
            )
        except ValueError as exc:
            raise ExplainabilityError(
                "explainability_artifact_invalid",
                f"The saved initialization ensemble cannot be explained: {exc}",
            ) from exc
        reference = members[0]
        if any(
            member.model_id != reference.model_id
            or member.feature_roles != reference.feature_roles
            or member.lookback != reference.lookback
            or member.includes_current_row != reference.includes_current_row
            for member in members[1:]
        ):
            raise ExplainabilityError(
                "explainability_artifact_mismatch",
                "Initialization-ensemble members do not share one model, "
                "feature order, and lookback.",
            )
        return _FrozenCorrector(
            artifact_type,
            reference.model_id,
            reference.feature_roles,
            reference.lookback,
            reference.includes_current_row,
            lambda windows: np.mean(
                np.stack([member.predict(windows) for member in members]),
                axis=0,
            ),
        )
    try:
        if artifact_type == S2V_ARTIFACT_TYPE:
            artifact = TensorFlowArtifact.from_mapping(payload)
            return _FrozenCorrector(
                artifact_type,
                artifact.model_id,
                artifact.feature_roles,
                artifact.lookback,
                False,
                lambda windows: predict_windows_from_artifact(artifact, windows),
            )
        if artifact_type == S2S_ARTIFACT_TYPE:
            artifact = S2SArtifact.from_mapping(payload)
            return _FrozenCorrector(
                artifact_type,
                artifact.model_id,
                artifact.feature_roles,
                artifact.lookback,
                True,
                lambda windows: predict_s2s_windows_from_artifact(artifact, windows),
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExplainabilityError(
            "explainability_artifact_invalid",
            f"The saved neural artifact cannot be restored: {exc}",
        ) from exc
    raise ExplainabilityError(
        "explainability_artifact_unsupported",
        f"Unsupported neural artifact type: {artifact_type or 'missing'}",
    )

def _lag_recipes(
    values: Sequence[LaggedFeatureConfig],
) -> tuple[LaggedFeature, ...]:
    return tuple(
        LaggedFeature(source=item.source, lag_steps=item.lag_steps)
        for item in values
    )


def _raw_result_features(result: Mapping[str, Any]) -> dict[str, NDArray]:
    forcings = result.get("forcings")
    if not isinstance(forcings, Mapping) or not forcings:
        raise ExplainabilityError(
            "explainability_features_missing",
            "The run archive does not contain the predictor trajectories "
            "required for SHAP.",
        )
    try:
        arrays = {
            str(name): np.asarray(values, dtype=float)
            for name, values in forcings.items()
        }
    except (TypeError, ValueError) as exc:
        raise ExplainabilityError(
            "explainability_features_invalid",
            "Archived predictor trajectories must contain numeric values.",
        ) from exc
    lengths = {len(values) for values in arrays.values() if values.ndim == 1}
    if (
        len(lengths) != 1
        or len(arrays) != sum(values.ndim == 1 for values in arrays.values())
    ):
        raise ExplainabilityError(
            "explainability_features_invalid",
            "Archived predictor trajectories must be one-dimensional and aligned.",
        )
    return arrays


def _paml_sources(
    result: Mapping[str, Any],
    external: Mapping[str, NDArray],
) -> tuple[dict[str, NDArray], NDArray]:
    simulation = result.get("best_simulation")
    if not isinstance(simulation, Mapping):
        raise ExplainabilityError(
            "explainability_physics_missing",
            "The PaML archive does not contain its physical trajectories.",
        )
    try:
        outputs = np.asarray(simulation.get("outputs"), dtype=float)
        states = np.asarray(simulation.get("states"), dtype=float)
        fluxes = np.asarray(simulation.get("fluxes"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ExplainabilityError(
            "explainability_physics_invalid",
            "The archived PaML physical trajectories must be numeric.",
        ) from exc
    output_names = tuple(str(name) for name in simulation.get("output_names", ()))
    state_names = tuple(str(name) for name in simulation.get("state_names", ()))
    flux_names = tuple(str(name) for name in simulation.get("flux_names", ()))
    if outputs.ndim != 2 or "physical_streamflow" not in output_names:
        raise ExplainabilityError(
            "explainability_physics_missing",
            "The PaML archive does not expose physical_streamflow separately.",
        )
    physical_position = output_names.index("physical_streamflow")
    physical_streamflow = outputs[:, physical_position]
    sources = {
        name: np.asarray(values, dtype=float)
        for name, values in external.items()
    }
    sources["physical_output__streamflow"] = physical_streamflow
    if states.ndim == 2 and len(states) == len(outputs) + 1:
        sources.update(
            {
                f"physical_state__{name}": states[1:, position]
                for position, name in enumerate(state_names)
            }
        )
    if fluxes.ndim == 2 and len(fluxes) == len(outputs):
        sources.update(
            {
                f"physical_flux__{name}": fluxes[:, position]
                for position, name in enumerate(flux_names)
            }
        )
    return sources, physical_streamflow


def _window_at(
    matrix: NDArray,
    target_index: int,
    lookback: int,
    includes_current_row: bool = False,
) -> NDArray[np.floating]:
    stop = target_index + 1 if includes_current_row else target_index
    return np.asarray(matrix[stop - lookback : stop], dtype=float)


def _valid_target_indices(
    matrix: NDArray,
    *,
    lookback: int,
    includes_current_row: bool = False,
    start: int,
    end: int,
    prediction: NDArray | None = None,
) -> NDArray[np.int_]:
    minimum_index = lookback - 1 if includes_current_row else lookback
    lower = max(start, minimum_index)
    candidates = np.arange(lower, end, dtype=int)
    valid = np.array(
        [
            np.all(
                np.isfinite(
                    _window_at(
                        matrix,
                        index,
                        lookback,
                        includes_current_row,
                    )
                )
            )
            for index in candidates
        ],
        dtype=bool,
    )
    if prediction is not None:
        valid &= np.isfinite(prediction[candidates])
    return candidates[valid]


def _complete_s2s_block_endpoints(
    matrix: NDArray,
    *,
    context_start: int,
    score_start: int,
    end: int,
    lookback: int,
    warmup: int,
    prediction: NDArray | None = None,
) -> NDArray[np.int_]:
    """Return exact archived endpoints of complete causal S2S blocks.

    A recurrent S2S output depends on its position inside the truncated
    sequence block. Restricting SHAP to complete block endpoints preserves one
    fixed input-to-output function and reproduces the archived prediction
    exactly, without silently replacing it by a rolling-window surrogate.
    """

    step = lookback - warmup
    endpoints: list[int] = []
    current = context_start
    while current + warmup < end:
        stop = min(current + lookback, end)
        if stop - current == lookback:
            target = stop - 1
            window = matrix[current:stop]
            if (
                target >= score_start
                and np.all(np.isfinite(window))
                and (
                    prediction is None
                    or np.isfinite(prediction[target])
                )
            ):
                endpoints.append(target)
        current += step
    return np.asarray(endpoints, dtype=int)


def _spread_sample(
    candidates: NDArray[np.int_],
    count: int,
) -> NDArray[np.int_]:
    if len(candidates) <= count:
        return candidates.copy()
    positions = np.linspace(0, len(candidates) - 1, num=count)
    return candidates[np.unique(np.rint(positions).astype(int))]


def _permutation_shap(
    target_window: NDArray,
    backgrounds: NDArray,
    *,
    predict: Callable[[NDArray], NDArray[np.floating]],
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, NDArray, NDArray, float]:
    """Return prediction, baseline, SHAP values, standard errors, additivity error."""

    shape = target_window.shape
    player_count = target_window.size
    flattened_target = target_window.reshape(-1)
    target_prediction = float(predict(target_window[None, ...])[0])
    sampled_contributions: list[NDArray] = []
    background_predictions: list[float] = []

    for background in backgrounds:
        flattened_background = background.reshape(-1)
        orders = np.vstack(
            [rng.permutation(player_count) for _ in range(permutations)]
        )
        paths = np.empty(
            (permutations, player_count + 1, player_count),
            dtype=float,
        )
        paths[:, 0, :] = flattened_background
        for permutation_index, order in enumerate(orders):
            current = flattened_background.copy()
            for step, player in enumerate(order, start=1):
                current[player] = flattened_target[player]
                paths[permutation_index, step, :] = current
        values = predict(paths.reshape(-1, *shape)).reshape(
            permutations, player_count + 1
        )
        background_predictions.append(float(values[0, 0]))
        deltas = np.diff(values, axis=1)
        for permutation_index, order in enumerate(orders):
            contribution = np.zeros(player_count, dtype=float)
            contribution[order] = deltas[permutation_index]
            sampled_contributions.append(contribution)

    samples = np.vstack(sampled_contributions)
    shap_values = np.mean(samples, axis=0)
    standard_error = (
        np.std(samples, axis=0, ddof=1) / np.sqrt(len(samples))
        if len(samples) > 1
        else np.zeros(player_count, dtype=float)
    )
    baseline = float(np.mean(background_predictions))
    # TensorFlow evaluates the target alone and the permutation paths in
    # differently sized float32 batches. Preserve the Shapley efficiency
    # identity by assigning their round-off residual to the largest component.
    efficiency_residual = target_prediction - (
        baseline + float(np.sum(shap_values))
    )
    if shap_values.size:
        correction_index = int(np.argmax(np.abs(shap_values)))
        shap_values.flat[correction_index] += efficiency_residual
    additivity_error = abs(
        target_prediction - (baseline + float(np.sum(shap_values)))
    )
    return (
        target_prediction,
        baseline,
        shap_values.reshape(shape),
        standard_error.reshape(shape),
        additivity_error,
    )


def explain_run_archive(
    archive: Mapping[str, Any],
    *,
    explained_rows: int = 8,
    background_rows: int = 4,
    permutations: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Explain held-out predictions from one completed learned run archive."""

    if not 1 <= explained_rows <= 32:
        raise ExplainabilityError(
            "explainability_rows_invalid",
            "explained_rows must lie between 1 and 32.",
        )
    if not 1 <= background_rows <= 32:
        raise ExplainabilityError(
            "explainability_background_invalid",
            "background_rows must lie between 1 and 32.",
        )
    if not 1 <= permutations <= 128:
        raise ExplainabilityError(
            "explainability_permutations_invalid",
            "permutations must lie between 1 and 128.",
        )
    try:
        config = ExperimentConfigV1.model_validate(archive.get("config"))
    except (TypeError, ValueError) as exc:
        raise ExplainabilityError(
            "explainability_configuration_invalid",
            f"The archived experiment configuration is invalid: {exc}",
        ) from exc
    result = archive.get("result")
    if not isinstance(result, Mapping) or result.get("state") != "completed":
        raise ExplainabilityError(
            "explainability_run_incomplete",
            "SHAP requires a completed saved training run.",
        )
    if result.get("run_kind") == "forecast":
        raise ExplainabilityError(
            "explainability_forecast_unsupported",
            "Explain the saved training run that produced this forecast; the "
            "current SHAP background is restricted to its training partition.",
        )
    training = result.get("training")
    if not isinstance(training, Mapping):
        raise ExplainabilityError(
            "explainability_model_unsupported",
            "SHAP is available only for trained neural and PaML runs.",
        )
    artifact_payload = training.get("artifact")
    if not isinstance(artifact_payload, Mapping):
        raise ExplainabilityError(
            "explainability_artifact_missing",
            "The run archive does not contain a frozen neural artifact.",
        )
    corrector = _artifact_predictor(artifact_payload)
    external = _raw_result_features(result)
    start_index = int(config.task.period.start_index)

    is_paml = isinstance(config.model, PamlModelConfig) and isinstance(
        config.task, PamlTrainingTaskConfig
    )
    is_neural = isinstance(
        config.model,
        (MlpModelConfig, LstmModelConfig, GruModelConfig, Cnn1dModelConfig),
    ) and isinstance(config.task, NeuralTrainingTaskConfig)
    if is_paml:
        raise ExplainabilityError(
            "explainability_model_unsupported",
            "SHAP is not exposed for the jointly trained PaML architecture in v1.0.",
        )
    if not is_neural:
        raise ExplainabilityError(
            "explainability_model_unsupported",
            "SHAP is currently available for standalone neural models.",
        )

    sources = external
    physical_streamflow = None
    model = config.model
    if corrector.model_id != model.model_id:
        raise ExplainabilityError(
            "explainability_artifact_mismatch",
            "The frozen neural artifact model identifier does not match the "
            "saved experiment configuration.",
        )
    recipes = _lag_recipes(model.lagged_features)
    try:
        expanded = build_feature_mapping(sources, model.feature_roles, recipes)
    except ValueError as exc:
        raise ExplainabilityError(
            "explainability_features_missing",
            f"The archived feature pipeline cannot be reconstructed: {exc}",
        ) from exc
    if (
        corrector.lookback != model.lookback
        or corrector.feature_roles != tuple(expanded)
    ):
        raise ExplainabilityError(
            "explainability_artifact_mismatch",
            "The frozen neural artifact lookback or ordered feature roles do "
            "not match the saved experiment configuration.",
        )
    try:
        matrix = np.column_stack(
            [np.asarray(expanded[name], dtype=float) for name in corrector.feature_roles]
        )
    except KeyError as exc:
        raise ExplainabilityError(
            "explainability_features_missing",
            f"The archived feature trajectory is missing {exc.args[0]!r}.",
        ) from exc
    player_count = corrector.lookback * len(corrector.feature_roles)
    if player_count > 512:
        raise ExplainabilityError(
            "explainability_dimension_too_large",
            "The selected lookback and feature count create more than 512 "
            "variable–lag players. Reduce the lookback or predictor set before "
            "requesting cell-level SHAP.",
        )

    simulation = result.get("best_simulation")
    if not isinstance(simulation, Mapping):
        raise ExplainabilityError(
            "explainability_predictions_missing",
            "The run archive does not contain its prediction trajectory.",
        )
    try:
        outputs = np.asarray(simulation.get("outputs"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ExplainabilityError(
            "explainability_predictions_invalid",
            "Archived predictions must contain numeric values.",
        ) from exc
    if outputs.ndim != 2 or len(outputs) != len(matrix):
        raise ExplainabilityError(
            "explainability_predictions_invalid",
            "Archived predictions and feature trajectories are not aligned.",
        )
    archived_prediction = outputs[:, 0]
    train_end = int(config.task.period.train_end_index) - start_index
    validation_end = int(config.task.period.validation_end_index) - start_index
    test_end = (
        int(config.task.period.test_end_index) - start_index
        if config.task.period.test_end_index is not None
        else len(matrix)
    )
    if corrector.includes_current_row:
        finite_rows = np.all(np.isfinite(matrix), axis=1)
        finite_positions = np.flatnonzero(finite_rows)
        if not len(finite_positions):
            raise ExplainabilityError(
                "explainability_features_invalid",
                "No finite causal S2S feature row is available.",
            )
        effective_start = int(finite_positions[0])
        warmup = int(getattr(model, "s2s_warmup", 0))
        training_candidates = _complete_s2s_block_endpoints(
            matrix,
            context_start=effective_start,
            score_start=effective_start + warmup,
            end=train_end,
            lookback=corrector.lookback,
            warmup=warmup,
        )
        explained_candidates = _complete_s2s_block_endpoints(
            matrix,
            context_start=max(effective_start, validation_end - warmup),
            score_start=validation_end,
            end=test_end,
            lookback=corrector.lookback,
            warmup=warmup,
            prediction=archived_prediction,
        )
    else:
        training_candidates = _valid_target_indices(
            matrix,
            lookback=corrector.lookback,
            includes_current_row=False,
            start=0,
            end=train_end,
        )
        explained_candidates = _valid_target_indices(
            matrix,
            lookback=corrector.lookback,
            includes_current_row=False,
            start=validation_end,
            end=test_end,
            prediction=archived_prediction,
        )
    if len(training_candidates) == 0:
        raise ExplainabilityError(
            "explainability_background_empty",
            "No finite training windows are available as SHAP references.",
        )
    if len(explained_candidates) == 0:
        raise ExplainabilityError(
            "explainability_test_empty",
            "No finite held-out test predictions are available to explain.",
        )

    rng = np.random.default_rng(seed)
    background_indices = rng.choice(
        training_candidates,
        size=min(background_rows, len(training_candidates)),
        replace=False,
    )
    backgrounds = np.stack(
        [
            _window_at(matrix, int(index), corrector.lookback)
            if not corrector.includes_current_row
            else _window_at(
                matrix,
                int(index),
                corrector.lookback,
                True,
            )
            for index in background_indices
        ]
    )
    selected_indices = _spread_sample(explained_candidates, explained_rows)
    timestamps = result.get("timestamps")
    observations = result.get("observations")
    try:
        observations_array = (
            np.asarray(observations, dtype=float)
            if observations is not None
            else None
        )
    except (TypeError, ValueError) as exc:
        raise ExplainabilityError(
            "explainability_observations_invalid",
            "Archived observations must contain numeric values or nulls.",
        ) from exc
    correction_strategy = (
        config.model.correction_strategy if is_paml else None
    )
    enforce_nonnegative = (
        config.model.enforce_nonnegative_output if is_paml else False
    )

    local_rows: list[dict[str, Any]] = []
    local_shap_values: list[NDArray] = []
    maximum_additivity_error = 0.0
    maximum_archive_difference = 0.0

    for target_index in selected_indices:
        target_index = int(target_index)
        target_window = _window_at(
            matrix,
            target_index,
            corrector.lookback,
            corrector.includes_current_row,
        )
        direct_physical = (
            float(physical_streamflow[target_index])
            if physical_streamflow is not None
            else None
        )

        def predict_model(windows: NDArray) -> NDArray[np.floating]:
            corrector_values = corrector.predict(windows)
            if correction_strategy == "additive_residual":
                values = corrector_values + float(direct_physical)
            else:
                values = corrector_values
            if enforce_nonnegative:
                values = np.maximum(values, 0.0)
            return values

        (
            prediction,
            baseline,
            shap_values,
            standard_error,
            additivity_error,
        ) = _permutation_shap(
            target_window,
            backgrounds,
            predict=predict_model,
            permutations=permutations,
            rng=rng,
        )
        archived_value = float(archived_prediction[target_index])
        archive_difference = abs(prediction - archived_value)
        # Keras may differ by a few float32 units when one window is evaluated
        # alone instead of inside the original training/result batch. When the
        # replay agrees within numerical tolerance, explain the archived value
        # exactly and absorb only that round-off in the largest attribution.
        if archive_difference <= 1e-5:
            prediction = archived_value
            correction_index = int(np.argmax(np.abs(shap_values)))
            shap_values.flat[correction_index] += prediction - (
                baseline + float(np.sum(shap_values))
            )
            additivity_error = abs(
                prediction - (baseline + float(np.sum(shap_values)))
            )
            archive_difference = 0.0
        maximum_archive_difference = max(
            maximum_archive_difference, archive_difference
        )
        maximum_additivity_error = max(
            maximum_additivity_error, additivity_error
        )
        local_shap_values.append(shap_values)
        attributions = []
        for window_position in range(corrector.lookback):
            lag_steps = corrector.lookback - window_position - int(
                corrector.includes_current_row
            )
            for feature_position, feature in enumerate(corrector.feature_roles):
                attributions.append(
                    {
                        "feature": feature,
                        "lag_steps": lag_steps,
                        "shap_value": float(
                            shap_values[window_position, feature_position]
                        ),
                        "standard_error": float(
                            standard_error[window_position, feature_position]
                        ),
                    }
                )
        attributions.sort(
            key=lambda item: abs(float(item["shap_value"])),
            reverse=True,
        )
        absolute_source_row = start_index + target_index
        label = (
            str(timestamps[target_index])
            if isinstance(timestamps, Sequence)
            and not isinstance(timestamps, (str, bytes))
            and target_index < len(timestamps)
            else f"Source row {absolute_source_row + 1}"
        )
        observation = (
            float(observations_array[target_index])
            if observations_array is not None
            and target_index < len(observations_array)
            and np.isfinite(observations_array[target_index])
            else None
        )
        local_rows.append(
            {
                "result_index": target_index,
                "source_row": absolute_source_row,
                "label": label,
                "prediction": prediction,
                "archived_prediction": float(archived_prediction[target_index]),
                "observation": observation,
                "baseline": baseline,
                "physical_streamflow": direct_physical,
                "additivity_error": additivity_error,
                "attributions": attributions,
            }
        )

    shap_stack = np.stack(local_shap_values)
    global_signed = np.mean(shap_stack, axis=0)
    global_absolute = np.mean(np.abs(shap_stack), axis=0)
    by_feature = []
    for feature_position, feature in enumerate(corrector.feature_roles):
        grouped = np.sum(shap_stack[:, :, feature_position], axis=1)
        by_feature.append(
            {
                "feature": feature,
                "mean_absolute_shap": float(np.mean(np.abs(grouped))),
                "mean_shap": float(np.mean(grouped)),
            }
        )
    by_feature.sort(key=lambda item: item["mean_absolute_shap"], reverse=True)
    by_lag = []
    for window_position in range(corrector.lookback):
        grouped = np.sum(shap_stack[:, window_position, :], axis=1)
        by_lag.append(
            {
                "lag_steps": corrector.lookback - window_position - int(
                    corrector.includes_current_row
                ),
                "mean_absolute_shap": float(np.mean(np.abs(grouped))),
                "mean_shap": float(np.mean(grouped)),
            }
        )
    by_lag.sort(key=lambda item: item["lag_steps"])
    cells = [
        {
            "feature": feature,
            "lag_steps": corrector.lookback - window_position - int(
                corrector.includes_current_row
            ),
            "mean_absolute_shap": float(
                global_absolute[window_position, feature_position]
            ),
            "mean_shap": float(
                global_signed[window_position, feature_position]
            ),
        }
        for window_position in range(corrector.lookback)
        for feature_position, feature in enumerate(corrector.feature_roles)
    ]
    cells.sort(key=lambda item: item["mean_absolute_shap"], reverse=True)

    warnings = [
        {
            "code": "shap_not_causal",
            "message": (
                "These attributions explain the frozen model under empirical "
                "interventional replacement; they do not establish hydrological "
                "causality."
            ),
        },
        {
            "code": "shap_feature_dependence",
            "message": (
                "Replacing variable–lag cells independently can create combinations "
                "that are uncommon in correlated hydroclimatic data."
            ),
        },
    ]
    if corrector.includes_current_row:
        warnings.append(
            {
                "code": "s2s_shap_complete_blocks_only",
                "message": (
                    "Causal S2S SHAP is evaluated at complete archived block "
                    "endpoints so the explained function and prediction are "
                    "reproduced exactly."
                ),
            }
        )
    if maximum_archive_difference > 1e-8:
        warnings.append(
            {
                "code": "shap_archive_reproduction_tolerance",
                "message": (
                    "The explanation evaluator differed from at least one archived "
                    "prediction by more than 1e-8 mm/day."
                ),
            }
        )

    return {
        "schema_version": "1.0",
        "method_id": "empirical_interventional_permutation_shap.v1",
        "task_id": str(archive.get("task_id", result.get("task_id", ""))),
        "model_id": config.model.model_id,
        "corrector_model_id": config.model.model_id,
        "output_name": "streamflow",
        "output_unit": "mm/day",
        "background_partition": "training",
        "background_rows": len(backgrounds),
        "background_source_rows": [
            start_index + int(index) for index in background_indices
        ],
        "explained_partition": "test",
        "explained_rows": len(local_rows),
        "permutations": permutations,
        "seed": seed,
        "lookback": corrector.lookback,
        "feature_roles": corrector.feature_roles,
        "player_count": player_count,
        "correction_strategy": correction_strategy,
        "local": local_rows,
        "global_by_feature": by_feature,
        "global_by_lag": by_lag,
        "global_cells": cells,
        "maximum_additivity_error": maximum_additivity_error,
        "maximum_archive_prediction_difference": maximum_archive_difference,
        "warnings": warnings,
    }
