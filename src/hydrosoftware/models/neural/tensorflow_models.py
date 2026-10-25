"""TensorFlow/Keras neural models used by HydroSoftware.

The module provides MLP, LSTM, GRU and causal CNN-1D rainfall-runoff models in
sequence-to-value (S2V) and causal sequence-to-sequence (S2S) form. NumPy is
used only for tabular preparation, archive serialization and metric reporting;
all trainable neural operations, automatic differentiation and optimization are
performed with TensorFlow/Keras.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from hydrosoftware.configuration import (
    AdamTrainingConfig,
    Cnn1dModelConfig,
    GruModelConfig,
    LstmModelConfig,
    MlpModelConfig,
)
from hydrosoftware.core.specifications import (
    ModelCapabilities,
    ModelFamily,
    ModelManifest,
    PredictionLayout,
    VariableSpec,
)
from hydrosoftware.core.transforms import Standardizer
from hydrosoftware.core.optimization import (
    objective_definition,
    objective_optimization_score,
)

TENSORFLOW_SERIES = "2.21"
S2V_ARTIFACT_TYPE = "hydrosoftware.neural.tensorflow.s2v.v1"
S2S_ARTIFACT_TYPE = "hydrosoftware.neural.tensorflow.s2s.v1"


def _tensorflow():
    """Import the supported TensorFlow series only when a neural model is used."""

    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "HydroSoftware neural models require TensorFlow 2.21.x. "
            "Install HydroSoftware with its declared dependencies."
        ) from exc
    version = str(getattr(tf, "__version__", ""))
    if not version.startswith(f"{TENSORFLOW_SERIES}."):
        raise RuntimeError(
            f"HydroSoftware v1.0 expects TensorFlow {TENSORFLOW_SERIES}.x; "
            f"the active environment provides {version or 'an unknown version'}."
        )
    return tf


def tensorflow_runtime_version() -> str:
    """Return the TensorFlow runtime version after compatibility validation."""

    return str(_tensorflow().__version__)


def _set_deterministic_seed(seed: int) -> Any:
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Some accelerator/runtime combinations already enforce deterministic
        # kernels or do not expose this switch. The seed is still fixed.
        pass
    return tf


_COMMON_INPUTS = (
    VariableSpec("precipitation", "forcing", "mm/day"),
    VariableSpec("temperature", "forcing", "degC"),
    VariableSpec("potential_evapotranspiration", "forcing", "mm/day"),
)
_COMMON_OUTPUTS = (
    VariableSpec("streamflow", "target", "mm/day", lower_bound=0.0),
)


def _manifest(
    model_id: str,
    display_name: str,
    layout: PredictionLayout,
    architecture: str,
    references: tuple[str, ...],
) -> ModelManifest:
    return ModelManifest(
        model_id=model_id,
        version="1.0.0",
        display_name=display_name,
        family=ModelFamily.NEURAL,
        capabilities=ModelCapabilities(
            layouts=(layout,),
            supports_fit=True,
            supports_calibration=False,
            differentiable=True,
            supports_initial_state=False,
            exposes_states=False,
            exposes_fluxes=False,
        ),
        inputs=_COMMON_INPUTS,
        outputs=_COMMON_OUTPUTS,
        references=references,
        metadata={
            "engine": "tensorflow.keras.2.21",
            "framework": "TensorFlow",
            "architecture": architecture,
            "artifact_format": (
                S2S_ARTIFACT_TYPE
                if layout is PredictionLayout.SEQUENCE_TO_SEQUENCE
                else S2V_ARTIFACT_TYPE
            ),
        },
    )


_MLP_REFS = ("Rumelhart, Hinton and Williams (1986), Nature 323, 533-536.",)
_LSTM_REFS = (
    "Hochreiter and Schmidhuber (1997), Neural Computation 9(8), 1735-1780.",
    "Gers, Schmidhuber and Cummins (2000), Neural Computation 12(10), 2451-2471.",
)
_GRU_REFS = ("Cho et al. (2014), arXiv:1406.1078.",)
_CNN_REFS = ("Bai, Kolter and Koltun (2018), arXiv:1803.01271.",)

MLP_TENSORFLOW_MANIFEST = _manifest(
    "mlp.tensorflow_s2v.v1", "MLP · TensorFlow S2V",
    PredictionLayout.SEQUENCE_TO_VALUE, "multilayer_perceptron", _MLP_REFS,
)
LSTM_TENSORFLOW_MANIFEST = _manifest(
    "lstm.tensorflow_s2v.v1", "LSTM · TensorFlow S2V",
    PredictionLayout.SEQUENCE_TO_VALUE, "lstm", _LSTM_REFS,
)
STACKED_LSTM_TENSORFLOW_MANIFEST = _manifest(
    "lstm.tensorflow_stacked_s2v.v2", "Stacked LSTM · TensorFlow S2V",
    PredictionLayout.SEQUENCE_TO_VALUE, "stacked_lstm", _LSTM_REFS,
)
GRU_TENSORFLOW_MANIFEST = _manifest(
    "gru.tensorflow_s2v.v1", "GRU · TensorFlow S2V",
    PredictionLayout.SEQUENCE_TO_VALUE, "gru", _GRU_REFS,
)
STACKED_GRU_TENSORFLOW_MANIFEST = _manifest(
    "gru.tensorflow_stacked_s2v.v2", "Stacked GRU · TensorFlow S2V",
    PredictionLayout.SEQUENCE_TO_VALUE, "stacked_gru", _GRU_REFS,
)
CNN1D_TENSORFLOW_MANIFEST = _manifest(
    "cnn1d.tensorflow_s2v.v1", "CNN-1D · TensorFlow S2V",
    PredictionLayout.SEQUENCE_TO_VALUE, "causal_cnn1d", _CNN_REFS,
)
MLP_S2S_TENSORFLOW_MANIFEST = _manifest(
    "mlp.tensorflow_s2s.v2", "MLP · TensorFlow S2S",
    PredictionLayout.SEQUENCE_TO_SEQUENCE, "time_distributed_mlp", _MLP_REFS,
)
LSTM_S2S_TENSORFLOW_MANIFEST = _manifest(
    "lstm.tensorflow_stacked_s2s.v3", "Stacked LSTM · TensorFlow S2S",
    PredictionLayout.SEQUENCE_TO_SEQUENCE, "stacked_lstm", _LSTM_REFS,
)
GRU_S2S_TENSORFLOW_MANIFEST = _manifest(
    "gru.tensorflow_stacked_s2s.v3", "Stacked GRU · TensorFlow S2S",
    PredictionLayout.SEQUENCE_TO_SEQUENCE, "stacked_gru", _GRU_REFS,
)
CNN1D_S2S_TENSORFLOW_MANIFEST = _manifest(
    "cnn1d.tensorflow_s2s.v2", "CNN-1D · TensorFlow S2S",
    PredictionLayout.SEQUENCE_TO_SEQUENCE, "causal_cnn1d", _CNN_REFS,
)


S2S_MODEL_IDS = {
    MLP_S2S_TENSORFLOW_MANIFEST.model_id,
    LSTM_S2S_TENSORFLOW_MANIFEST.model_id,
    GRU_S2S_TENSORFLOW_MANIFEST.model_id,
    CNN1D_S2S_TENSORFLOW_MANIFEST.model_id,
}


def is_s2s_model_id(model_id: str) -> bool:
    return model_id in S2S_MODEL_IDS


def _model_kind(model_id: str) -> str:
    return model_id.split(".", 1)[0]


@dataclass(frozen=True)
class TensorFlowArtifact:
    """Portable TensorFlow model definition, scaling and fitted weights."""

    artifact_type: str
    model_id: str
    engine_id: str
    feature_roles: tuple[str, ...]
    lookback: int
    warmup_length: int
    output_activation: str
    input_mean: NDArray[np.floating]
    input_scale: NDArray[np.floating]
    target_offset: float
    target_scale: float
    architecture: Mapping[str, Any]
    weights: tuple[NDArray[np.floating], ...]
    best_epoch: int
    gradient_clip_norm: float

    def __post_init__(self) -> None:
        if self.artifact_type not in {S2V_ARTIFACT_TYPE, S2S_ARTIFACT_TYPE}:
            raise ValueError("Unsupported TensorFlow neural artifact type.")
        if ".tensorflow_" not in self.model_id:
            raise ValueError("The artifact model identifier is not a TensorFlow model.")
        if self.engine_id != "tensorflow.keras.2.21":
            raise ValueError("Unsupported TensorFlow neural execution engine.")
        if not self.feature_roles or len(set(self.feature_roles)) != len(self.feature_roles):
            raise ValueError("Neural feature roles must be non-empty and unique.")
        if self.lookback < 1 or not 0 <= self.warmup_length < self.lookback:
            raise ValueError("Invalid neural lookback or warm-up length.")
        if self.artifact_type == S2V_ARTIFACT_TYPE and self.warmup_length != 0:
            raise ValueError("S2V artifacts cannot define an S2S warm-up.")
        if self.output_activation not in {"linear", "softplus"}:
            raise ValueError("Unsupported neural output activation.")
        mean = np.asarray(self.input_mean, dtype=float)
        scale = np.asarray(self.input_scale, dtype=float)
        if mean.shape != (len(self.feature_roles),) or scale.shape != mean.shape:
            raise ValueError("Neural input scaling does not match the feature count.")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("Neural input scaling must be finite and positive.")
        if not np.isfinite(self.target_offset) or not np.isfinite(self.target_scale) or self.target_scale <= 0:
            raise ValueError("Neural target scaling must be finite and positive.")
        if self.best_epoch < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("Invalid neural training metadata.")
        clean_weights = tuple(np.asarray(item, dtype=np.float32) for item in self.weights)
        if any(not np.all(np.isfinite(item)) for item in clean_weights):
            raise ValueError("Neural weights must be finite.")
        object.__setattr__(self, "input_mean", mean)
        object.__setattr__(self, "input_scale", scale)
        object.__setattr__(self, "weights", clean_weights)
        object.__setattr__(self, "architecture", dict(self.architecture))

    @property
    def layout(self) -> str:
        return "sequence_to_sequence" if self.artifact_type == S2S_ARTIFACT_TYPE else "sequence_to_value"

    def as_mapping(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "model_id": self.model_id,
            "engine_id": self.engine_id,
            "feature_roles": self.feature_roles,
            "lookback": self.lookback,
            "warmup_length": self.warmup_length,
            "output_activation": self.output_activation,
            "input_mean": self.input_mean,
            "input_scale": self.input_scale,
            "target_offset": self.target_offset,
            "target_scale": self.target_scale,
            "architecture": dict(self.architecture),
            "weights": self.weights,
            "best_epoch": self.best_epoch,
            "gradient_clip_norm": self.gradient_clip_norm,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TensorFlowArtifact":
        return cls(
            artifact_type=str(value["artifact_type"]),
            model_id=str(value["model_id"]),
            engine_id=str(value["engine_id"]),
            feature_roles=tuple(str(item) for item in value["feature_roles"]),
            lookback=int(value["lookback"]),
            warmup_length=int(value.get("warmup_length", 0)),
            output_activation=str(value["output_activation"]),
            input_mean=np.asarray(value["input_mean"], dtype=float),
            input_scale=np.asarray(value["input_scale"], dtype=float),
            target_offset=float(value["target_offset"]),
            target_scale=float(value["target_scale"]),
            architecture=dict(value["architecture"]),
            weights=tuple(np.asarray(item, dtype=np.float32) for item in value["weights"]),
            best_epoch=int(value["best_epoch"]),
            gradient_clip_norm=float(value.get("gradient_clip_norm", 5.0)),
        )


# Public names retained so the rest of the application has one stable Python API.
MlpArtifact = TensorFlowArtifact
LstmArtifact = TensorFlowArtifact
StackedLstmArtifact = TensorFlowArtifact
GruArtifact = TensorFlowArtifact
StackedGruArtifact = TensorFlowArtifact
Cnn1dArtifact = TensorFlowArtifact
S2SArtifact = TensorFlowArtifact


@dataclass(frozen=True)
class TensorFlowFitResult:
    artifact: TensorFlowArtifact
    predictions: NDArray[np.floating]
    history: tuple[Mapping[str, float | int], ...]
    partition_masks: Mapping[str, NDArray[np.bool_]]
    stopped_early: bool
    cancelled: bool


MlpFitResult = TensorFlowFitResult
LstmFitResult = TensorFlowFitResult
GruFitResult = TensorFlowFitResult
Cnn1dFitResult = TensorFlowFitResult
S2SFitResult = TensorFlowFitResult


def feature_matrix(feature_roles: Sequence[str], values: Mapping[str, NDArray]) -> NDArray[np.floating]:
    columns = [np.asarray(values[name], dtype=float) for name in feature_roles]
    if not columns or any(column.ndim != 1 for column in columns):
        raise ValueError("Neural feature arrays must be one-dimensional.")
    if len({len(column) for column in columns}) != 1:
        raise ValueError("Neural feature arrays must share the time dimension.")
    matrix = np.column_stack(columns)
    if np.any(np.isinf(matrix)):
        raise ValueError("Neural features cannot contain infinite values.")
    return matrix


def _window_rows(scaled_features: NDArray, target_indices: NDArray[np.integer], lookback: int) -> NDArray[np.floating]:
    offsets = np.arange(-lookback, 0, dtype=int)
    indices = target_indices[:, None] + offsets[None, :]
    return scaled_features[indices]


def _partition_indices(
    targets: NDArray,
    *,
    lookback: int,
    start: int,
    train_end: int,
    validation_end: int,
    test_end: int,
    stride: int,
) -> dict[str, NDArray[np.integer]]:
    indices = np.arange(max(start + lookback, lookback), test_end, dtype=int)
    indices = indices[np.isfinite(targets[indices])]
    partitions = {
        "training": indices[indices < train_end],
        "validation": indices[(indices >= train_end) & (indices < validation_end)],
        "test": indices[(indices >= validation_end) & (indices < test_end)],
    }
    if stride > 1:
        partitions = {name: values[::stride] for name, values in partitions.items()}
    for name, values in partitions.items():
        if len(values) == 0:
            raise ValueError(f"The {name} partition contains no valid target window.")
    return partitions


@dataclass(frozen=True)
class _SequenceBlocks:
    inputs: NDArray[np.floating]
    targets: NDArray[np.floating]
    offsets: NDArray[np.floating]
    absolute_indices: NDArray[np.integer]
    score_mask: NDArray[np.bool_]
    assignment_mask: NDArray[np.bool_]


def _build_blocks(
    features: NDArray,
    targets: NDArray,
    offsets: NDArray,
    *,
    context_start: int,
    score_start: int,
    end: int,
    length: int,
    warmup: int,
) -> _SequenceBlocks:
    if not 0 <= context_start <= score_start < end <= len(features):
        raise ValueError("Invalid causal S2S partition boundaries.")
    step = length - warmup
    starts: list[int] = []
    current = context_start
    while current + warmup < end:
        starts.append(current)
        current += step
    if not starts:
        raise ValueError("The S2S partition contains no sequence block.")
    batch = len(starts)
    inputs = np.zeros((batch, length, features.shape[1]), dtype=float)
    block_targets = np.zeros((batch, length), dtype=float)
    block_offsets = np.zeros((batch, length), dtype=float)
    indices = np.full((batch, length), -1, dtype=int)
    assignment = np.zeros((batch, length), dtype=bool)
    score = np.zeros((batch, length), dtype=bool)
    counts = np.zeros(len(features), dtype=int)
    for row, start in enumerate(starts):
        stop = min(start + length, end)
        count = stop - start
        absolute = np.arange(start, stop, dtype=int)
        if not np.all(np.isfinite(features[absolute])):
            raise ValueError("Every real S2S input row must be finite after feature preparation.")
        inputs[row, :count] = features[absolute]
        indices[row, :count] = absolute
        valid_assignment = (np.arange(count) >= warmup) & (absolute >= score_start)
        assignment[row, :count] = valid_assignment
        finite_target = np.isfinite(targets[absolute])
        score[row, :count] = valid_assignment & finite_target
        block_targets[row, :count] = np.where(finite_target, targets[absolute], 0.0)
        block_offsets[row, :count] = offsets[absolute]
        counts[absolute[valid_assignment]] += 1
    if np.any(counts[score_start:end] != 1):
        raise RuntimeError("Causal S2S blocking must assign every partition date exactly once.")
    if not np.any(score):
        raise ValueError("The S2S partition contains no finite scored target.")
    return _SequenceBlocks(inputs, block_targets, block_offsets, indices, score, assignment)


def _architecture_from_config(model: Any) -> dict[str, Any]:
    kind = _model_kind(model.model_id)
    architecture: dict[str, Any] = {"kind": kind}
    if kind == "mlp":
        architecture.update(hidden_layers=list(model.hidden_layers), activation=model.activation)
    elif kind in {"lstm", "gru"}:
        architecture.update(recurrent_layers=list(model.recurrent_layers))
    elif kind == "cnn1d":
        architecture.update(
            convolution_layers=[
                {
                    "filters": int(layer.filters),
                    "kernel_size": int(layer.kernel_size),
                    "activation": str(layer.activation),
                }
                for layer in model.convolution_layers
            ]
        )
    else:
        raise ValueError(f"Unsupported neural model family: {kind}")
    return architecture


def _build_keras_model(
    *,
    model_id: str,
    architecture: Mapping[str, Any],
    feature_count: int,
    lookback: int,
    output_activation: str,
    s2s: bool,
):
    tf = _tensorflow()
    keras = tf.keras
    kind = str(architecture["kind"])
    input_shape = (None, feature_count) if s2s else (lookback, feature_count)
    inputs = keras.Input(shape=input_shape, dtype="float32", name="forcings")
    x = inputs
    if kind == "mlp":
        if not s2s:
            x = keras.layers.Flatten(name="flatten_antecedent_window")(x)
        for index, width in enumerate(architecture["hidden_layers"]):
            x = keras.layers.Dense(
                int(width), activation=str(architecture["activation"]),
                name=f"dense_{index + 1}",
            )(x)
    elif kind in {"lstm", "gru"}:
        recurrent_cls = keras.layers.LSTM if kind == "lstm" else keras.layers.GRU
        layers = [int(value) for value in architecture["recurrent_layers"]]
        for index, width in enumerate(layers):
            return_sequences = s2s or index < len(layers) - 1
            kwargs: dict[str, Any] = {
                "units": width,
                "return_sequences": return_sequences,
                "name": f"{kind}_{index + 1}",
            }
            if kind == "gru":
                kwargs["reset_after"] = True
            x = recurrent_cls(**kwargs)(x)
    elif kind == "cnn1d":
        for index, layer in enumerate(architecture["convolution_layers"]):
            x = keras.layers.Conv1D(
                filters=int(layer["filters"]),
                kernel_size=int(layer["kernel_size"]),
                padding="causal",
                strides=1,
                activation=str(layer["activation"]),
                name=f"causal_conv_{index + 1}",
            )(x)
        if not s2s:
            x = keras.layers.Lambda(lambda value: value[:, -1, :], name="final_time_position")(x)
    else:
        raise ValueError(f"Unsupported neural model family: {kind}")
    outputs = keras.layers.Dense(1, activation=output_activation, name="streamflow")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=model_id.replace(".", "_"))


def _tf_objective_loss(observed, simulated, objective_id: str):
    tf = _tensorflow()
    simulated = tf.reshape(tf.convert_to_tensor(simulated), (-1,))
    if not simulated.dtype.is_floating:
        simulated = tf.cast(simulated, tf.float32)
    dtype = simulated.dtype
    observed = tf.reshape(tf.cast(observed, dtype), (-1,))
    residual = simulated - observed
    mse = tf.reduce_mean(tf.square(residual))
    eps = tf.cast(1e-8 if dtype == tf.float32 else 1e-12, dtype)
    if objective_id == "mean_square_error":
        return mse
    if objective_id == "root_mean_square_error":
        return tf.sqrt(tf.maximum(mse, eps * eps))
    if objective_id == "mean_absolute_error":
        return tf.reduce_mean(tf.abs(residual))
    if objective_id == "range_normalized_root_mean_square_error":
        observed_range = tf.reduce_max(observed) - tf.reduce_min(observed)
        return tf.sqrt(tf.maximum(mse, eps * eps)) / tf.maximum(observed_range, eps)
    if objective_id == "nash_sutcliffe_efficiency":
        denominator = tf.reduce_sum(tf.square(observed - tf.reduce_mean(observed)))
        return tf.reduce_sum(tf.square(residual)) / tf.maximum(denominator, eps)
    if objective_id == "kling_gupta_efficiency_2009":
        obs_mean = tf.reduce_mean(observed)
        sim_mean = tf.reduce_mean(simulated)
        obs_centered = observed - obs_mean
        sim_centered = simulated - sim_mean
        obs_std = tf.sqrt(tf.maximum(tf.reduce_mean(tf.square(obs_centered)), eps * eps))
        sim_std = tf.sqrt(tf.maximum(tf.reduce_mean(tf.square(sim_centered)), eps * eps))
        correlation = tf.reduce_mean(obs_centered * sim_centered) / tf.maximum(obs_std * sim_std, eps)
        alpha = sim_std / tf.maximum(obs_std, eps)
        safe_mean = tf.where(tf.abs(obs_mean) > eps, obs_mean, eps)
        beta = sim_mean / safe_mean
        return tf.sqrt(tf.square(correlation - 1.0) + tf.square(alpha - 1.0) + tf.square(beta - 1.0))
    raise ValueError(f"Unsupported optimization objective: {objective_id}")

def _predict_scaled(model, inputs: NDArray) -> NDArray[np.floating]:
    tf = _tensorflow()
    prediction = model(tf.convert_to_tensor(np.asarray(inputs, dtype=np.float32)), training=False)
    return np.asarray(prediction.numpy(), dtype=float)


def _artifact_from_model(
    config: Any,
    keras_model: Any,
    *,
    input_scaler: Standardizer,
    target_offset: float,
    target_scale: float,
    best_epoch: int,
) -> TensorFlowArtifact:
    return TensorFlowArtifact(
        artifact_type=S2S_ARTIFACT_TYPE if is_s2s_model_id(config.model_id) else S2V_ARTIFACT_TYPE,
        model_id=config.model_id,
        engine_id="tensorflow.keras.2.21",
        feature_roles=tuple(config.feature_roles),
        lookback=int(config.lookback),
        warmup_length=int(config.s2s_warmup if is_s2s_model_id(config.model_id) else 0),
        output_activation=str(config.output_activation),
        input_mean=np.asarray(input_scaler.mean, dtype=float),
        input_scale=np.asarray(input_scaler.scale, dtype=float),
        target_offset=float(target_offset),
        target_scale=float(target_scale),
        architecture=_architecture_from_config(config),
        weights=tuple(np.asarray(weight, dtype=np.float32) for weight in keras_model.get_weights()),
        best_epoch=int(best_epoch),
        gradient_clip_norm=float(getattr(config, "gradient_clip_norm", 5.0)),
    )


def _model_from_artifact(artifact: TensorFlowArtifact):
    model = _build_keras_model(
        model_id=artifact.model_id,
        architecture=artifact.architecture,
        feature_count=len(artifact.feature_roles),
        lookback=artifact.lookback,
        output_activation=artifact.output_activation,
        s2s=artifact.artifact_type == S2S_ARTIFACT_TYPE,
    )
    model.set_weights([np.asarray(weight, dtype=np.float32) for weight in artifact.weights])
    return model


def _fit_s2v(
    model: MlpModelConfig | LstmModelConfig | GruModelConfig | Cnn1dModelConfig,
    training: AdamTrainingConfig,
    features: Mapping[str, NDArray],
    targets: NDArray,
    *,
    start_index: int,
    train_end_index: int,
    validation_end_index: int,
    test_end_index: int,
    objective_id: str,
    stride: int,
    cancellation_requested: Callable[[], bool] | None,
    on_epoch: Callable[[int, int, float], None] | None,
    objective_offset: NDArray | None = None,
) -> TensorFlowFitResult:
    tf = _set_deterministic_seed(training.seed)
    raw_features = feature_matrix(model.feature_roles, features)
    raw_targets = np.asarray(targets, dtype=float)
    if raw_targets.ndim != 1 or len(raw_targets) != len(raw_features):
        raise ValueError("Neural targets must align with the forcing rows.")
    if test_end_index > len(raw_features):
        raise ValueError("The neural test period exceeds the available rows.")
    input_scaler = Standardizer.fit(raw_features[start_index:train_end_index])
    scaled_features = input_scaler.transform(raw_features)
    finite_training = raw_targets[start_index:train_end_index]
    finite_training = finite_training[np.isfinite(finite_training)]
    if len(finite_training) < 2:
        raise ValueError("At least two finite training targets are required.")
    target_offset = 0.0 if model.output_activation == "softplus" else float(np.mean(finite_training))
    target_scale = float(np.std(finite_training))
    if target_scale <= 1e-12:
        target_scale = 1.0
    scaled_targets = (raw_targets - target_offset) / target_scale
    offsets = np.zeros_like(raw_targets) if objective_offset is None else np.asarray(objective_offset, dtype=float)
    if offsets.shape != raw_targets.shape or not np.all(np.isfinite(offsets)):
        raise ValueError("Neural objective offsets must be finite and aligned with targets.")
    partitions = _partition_indices(
        raw_targets, lookback=model.lookback, start=start_index,
        train_end=train_end_index, validation_end=validation_end_index,
        test_end=test_end_index, stride=stride,
    )
    x = {name: _window_rows(scaled_features, idx, model.lookback) for name, idx in partitions.items()}
    y = {name: scaled_targets[idx] for name, idx in partitions.items()}
    off = {name: offsets[idx] for name, idx in partitions.items()}
    keras_model = _build_keras_model(
        model_id=model.model_id, architecture=_architecture_from_config(model),
        feature_count=len(model.feature_roles), lookback=model.lookback,
        output_activation=model.output_activation, s2s=False,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(training.learning_rate))
    objective_meta = objective_definition(objective_id)
    best_weights = [np.asarray(item).copy() for item in keras_model.get_weights()]
    best_loss = float("inf")
    best_score = -float("inf") if objective_meta.direction == "maximize" else float("inf")
    best_epoch = 0
    no_improvement = 0
    history: list[Mapping[str, float | int]] = []
    cancelled = False
    batch_size = len(x["training"]) if objective_meta.requires_global_batch else min(training.batch_size, len(x["training"]))
    clip_norm = float(getattr(model, "gradient_clip_norm", 5.0))
    for epoch in range(1, training.epochs + 1):
        if cancellation_requested is not None and cancellation_requested():
            cancelled = True
            break
        order = np.arange(len(x["training"]))
        if training.shuffle and batch_size < len(order):
            order = np.asarray(
                tf.random.experimental.stateless_shuffle(
                    tf.range(len(order), dtype=tf.int32),
                    seed=tf.constant([int(training.seed), int(epoch)], dtype=tf.int32),
                ).numpy(),
                dtype=int,
            )
        gradient_norms: list[float] = []
        for left in range(0, len(order), batch_size):
            batch = order[left:left + batch_size]
            xb = tf.convert_to_tensor(x["training"][batch], dtype=tf.float32)
            yb = tf.convert_to_tensor(y["training"][batch], dtype=tf.float32)
            ob = tf.convert_to_tensor(off["training"][batch], dtype=tf.float32)
            with tf.GradientTape() as tape:
                pred_scaled = tf.reshape(keras_model(xb, training=True), (-1,))
                pred = pred_scaled * target_scale + target_offset
                target = yb * target_scale + target_offset
                loss = _tf_objective_loss(target + ob, pred + ob, objective_id)
            gradients = tape.gradient(loss, keras_model.trainable_variables)
            gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            gradient_norms.append(float(global_norm.numpy()))
        train_pred = _predict_scaled(keras_model, x["training"])[:, 0] * target_scale + target_offset
        val_pred = _predict_scaled(keras_model, x["validation"])[:, 0] * target_scale + target_offset
        train_score, train_loss = objective_optimization_score(
            raw_targets[partitions["training"]] + off["training"], train_pred + off["training"], objective_id
        )
        val_score, val_loss = objective_optimization_score(
            raw_targets[partitions["validation"]] + off["validation"], val_pred + off["validation"], objective_id
        )
        history.append({
            "epoch": epoch,
            "training_objective": float(train_score),
            "validation_objective": float(val_score),
            "training_rmse": float(np.sqrt(np.mean(np.square(train_pred - raw_targets[partitions["training"]])))),
            "validation_rmse": float(np.sqrt(np.mean(np.square(val_pred - raw_targets[partitions["validation"]])))),
            "gradient_norm_before_clipping": float(max(gradient_norms, default=0.0)),
            "training_loss": float(train_loss),
            "validation_loss": float(val_loss),
        })
        if val_loss < best_loss - training.minimum_delta:
            best_loss = float(val_loss)
            best_score = float(val_score)
            best_epoch = epoch
            best_weights = [np.asarray(item).copy() for item in keras_model.get_weights()]
            no_improvement = 0
        else:
            no_improvement += 1
        if on_epoch is not None:
            on_epoch(epoch, training.epochs, best_score)
        if no_improvement >= training.patience:
            break
    keras_model.set_weights(best_weights)
    artifact = _artifact_from_model(
        model, keras_model, input_scaler=input_scaler,
        target_offset=target_offset, target_scale=target_scale, best_epoch=best_epoch,
    )
    predictions = np.full(len(raw_targets), np.nan, dtype=float)
    available = np.arange(max(start_index + model.lookback, model.lookback), test_end_index, dtype=int)
    finite_rows = np.all(np.isfinite(scaled_features), axis=1)
    valid = np.array([np.all(finite_rows[index - model.lookback:index]) for index in available], dtype=bool)
    available = available[valid]
    if len(available):
        windows = _window_rows(scaled_features, available, model.lookback)
        predictions[available] = _predict_scaled(keras_model, windows)[:, 0] * target_scale + target_offset
    masks = {name: np.zeros(len(raw_targets), dtype=bool) for name in partitions}
    for name, indices in partitions.items():
        masks[name][indices] = True
    return TensorFlowFitResult(
        artifact=artifact, predictions=predictions, history=tuple(history),
        partition_masks=masks,
        stopped_early=(not cancelled and bool(history) and len(history) < training.epochs),
        cancelled=cancelled,
    )


def _assign_partition_predictions(artifact: TensorFlowArtifact, blocks: _SequenceBlocks, destination: NDArray) -> None:
    model = _model_from_artifact(artifact)
    normalized = _predict_scaled(model, blocks.inputs)[:, :, 0]
    physical = normalized * artifact.target_scale + artifact.target_offset
    for row in range(len(blocks.inputs)):
        mask = blocks.assignment_mask[row]
        destination[blocks.absolute_indices[row, mask]] = physical[row, mask]


def _fit_s2s(
    model: MlpModelConfig | LstmModelConfig | GruModelConfig | Cnn1dModelConfig,
    training: AdamTrainingConfig,
    features: Mapping[str, NDArray],
    targets: NDArray,
    *,
    start_index: int,
    train_end_index: int,
    validation_end_index: int,
    test_end_index: int,
    objective_id: str,
    stride: int,
    cancellation_requested: Callable[[], bool] | None,
    on_epoch: Callable[[int, int, float], None] | None,
    objective_offset: NDArray | None = None,
) -> TensorFlowFitResult:
    if stride != 1:
        raise ValueError("Causal S2S training uses every date and requires stride=1.")
    tf = _set_deterministic_seed(training.seed)
    raw_features = feature_matrix(model.feature_roles, features)
    raw_targets = np.asarray(targets, dtype=float)
    if raw_targets.ndim != 1 or len(raw_targets) != len(raw_features):
        raise ValueError("S2S targets must align with the feature time dimension.")
    if not (0 <= start_index < train_end_index < validation_end_index < test_end_index <= len(raw_features)):
        raise ValueError("Invalid causal S2S training boundaries.")
    warmup = int(model.s2s_warmup)
    input_scaler = Standardizer.fit(raw_features[start_index:train_end_index])
    scaled_features = input_scaler.transform(raw_features)
    finite_training = raw_targets[start_index + warmup:train_end_index]
    finite_training = finite_training[np.isfinite(finite_training)]
    if len(finite_training) < 2:
        raise ValueError("At least two finite S2S training targets are required.")
    target_offset = 0.0 if model.output_activation == "softplus" else float(np.mean(finite_training))
    target_scale = float(np.std(finite_training))
    if target_scale <= 1e-12:
        target_scale = 1.0
    scaled_targets = (raw_targets - target_offset) / target_scale
    offsets = np.zeros_like(raw_targets) if objective_offset is None else np.asarray(objective_offset, dtype=float)
    if offsets.shape != raw_targets.shape or not np.all(np.isfinite(offsets)):
        raise ValueError("S2S objective offsets must be finite and aligned.")
    length = int(model.lookback)
    blocks = {
        "training": _build_blocks(scaled_features, scaled_targets, offsets, context_start=start_index, score_start=start_index + warmup, end=train_end_index, length=length, warmup=warmup),
        "validation": _build_blocks(scaled_features, scaled_targets, offsets, context_start=max(start_index, train_end_index - warmup), score_start=train_end_index, end=validation_end_index, length=length, warmup=warmup),
        "test": _build_blocks(scaled_features, scaled_targets, offsets, context_start=max(start_index, validation_end_index - warmup), score_start=validation_end_index, end=test_end_index, length=length, warmup=warmup),
    }
    keras_model = _build_keras_model(
        model_id=model.model_id, architecture=_architecture_from_config(model),
        feature_count=len(model.feature_roles), lookback=model.lookback,
        output_activation=model.output_activation, s2s=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(training.learning_rate))
    objective_meta = objective_definition(objective_id)
    best_weights = [np.asarray(item).copy() for item in keras_model.get_weights()]
    best_loss = float("inf")
    best_score = -float("inf") if objective_meta.direction == "maximize" else float("inf")
    best_epoch = 0
    no_improvement = 0
    history: list[Mapping[str, float | int]] = []
    cancelled = False
    clip_norm = float(getattr(model, "gradient_clip_norm", 5.0))
    train_blocks = blocks["training"]
    batch_size = len(train_blocks.inputs) if objective_meta.requires_global_batch else min(training.batch_size, len(train_blocks.inputs))
    for epoch in range(1, training.epochs + 1):
        if cancellation_requested is not None and cancellation_requested():
            cancelled = True
            break
        order = np.arange(len(train_blocks.inputs))
        if training.shuffle and batch_size < len(order):
            order = np.asarray(
                tf.random.experimental.stateless_shuffle(
                    tf.range(len(order), dtype=tf.int32),
                    seed=tf.constant([int(training.seed), int(epoch)], dtype=tf.int32),
                ).numpy(),
                dtype=int,
            )
        gradient_norms: list[float] = []
        for left in range(0, len(order), batch_size):
            rows = order[left:left + batch_size]
            xb = tf.convert_to_tensor(train_blocks.inputs[rows], dtype=tf.float32)
            yb = tf.convert_to_tensor(train_blocks.targets[rows], dtype=tf.float32)
            ob = tf.convert_to_tensor(train_blocks.offsets[rows], dtype=tf.float32)
            mb = tf.convert_to_tensor(train_blocks.score_mask[rows])
            with tf.GradientTape() as tape:
                pred_scaled = tf.squeeze(keras_model(xb, training=True), axis=-1)
                pred = pred_scaled * target_scale + target_offset
                target = yb * target_scale + target_offset
                observed = tf.boolean_mask(target + ob, mb)
                simulated = tf.boolean_mask(pred + ob, mb)
                loss = _tf_objective_loss(observed, simulated, objective_id)
            gradients = tape.gradient(loss, keras_model.trainable_variables)
            gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm)
            optimizer.apply_gradients(zip(gradients, keras_model.trainable_variables))
            gradient_norms.append(float(global_norm.numpy()))
        archive = _artifact_from_model(
            model, keras_model, input_scaler=input_scaler,
            target_offset=target_offset, target_scale=target_scale, best_epoch=epoch,
        )
        epoch_predictions = np.full(len(raw_targets), np.nan, dtype=float)
        for partition in blocks.values():
            _assign_partition_predictions(archive, partition, epoch_predictions)
        train_mask = np.zeros(len(raw_targets), dtype=bool)
        train_mask[start_index + warmup:train_end_index] = np.isfinite(raw_targets[start_index + warmup:train_end_index])
        val_mask = np.zeros(len(raw_targets), dtype=bool)
        val_mask[train_end_index:validation_end_index] = np.isfinite(raw_targets[train_end_index:validation_end_index])
        train_score, train_loss = objective_optimization_score(
            raw_targets[train_mask] + offsets[train_mask], epoch_predictions[train_mask] + offsets[train_mask], objective_id
        )
        val_score, val_loss = objective_optimization_score(
            raw_targets[val_mask] + offsets[val_mask], epoch_predictions[val_mask] + offsets[val_mask], objective_id
        )
        history.append({
            "epoch": epoch,
            "training_objective": float(train_score),
            "validation_objective": float(val_score),
            "training_rmse": float(np.sqrt(np.mean(np.square(epoch_predictions[train_mask] - raw_targets[train_mask])))),
            "validation_rmse": float(np.sqrt(np.mean(np.square(epoch_predictions[val_mask] - raw_targets[val_mask])))),
            "gradient_norm_before_clipping": float(max(gradient_norms, default=0.0)),
            "training_loss": float(train_loss),
            "validation_loss": float(val_loss),
        })
        if val_loss < best_loss - training.minimum_delta:
            best_loss = float(val_loss)
            best_score = float(val_score)
            best_epoch = epoch
            best_weights = [np.asarray(item).copy() for item in keras_model.get_weights()]
            no_improvement = 0
        else:
            no_improvement += 1
        if on_epoch is not None:
            on_epoch(epoch, training.epochs, best_score)
        if no_improvement >= training.patience:
            break
    keras_model.set_weights(best_weights)
    artifact = _artifact_from_model(
        model, keras_model, input_scaler=input_scaler,
        target_offset=target_offset, target_scale=target_scale, best_epoch=best_epoch,
    )
    predictions = np.full(len(raw_targets), np.nan, dtype=float)
    for partition in blocks.values():
        _assign_partition_predictions(artifact, partition, predictions)
    masks = {
        "training": np.zeros(len(raw_targets), dtype=bool),
        "validation": np.zeros(len(raw_targets), dtype=bool),
        "test": np.zeros(len(raw_targets), dtype=bool),
    }
    masks["training"][start_index + warmup:train_end_index] = np.isfinite(raw_targets[start_index + warmup:train_end_index])
    masks["validation"][train_end_index:validation_end_index] = np.isfinite(raw_targets[train_end_index:validation_end_index])
    masks["test"][validation_end_index:test_end_index] = np.isfinite(raw_targets[validation_end_index:test_end_index])
    return TensorFlowFitResult(
        artifact=artifact, predictions=predictions, history=tuple(history),
        partition_masks=masks,
        stopped_early=(not cancelled and bool(history) and len(history) < training.epochs),
        cancelled=cancelled,
    )


def fit_mlp(model: MlpModelConfig, training: AdamTrainingConfig, features: Mapping[str, NDArray], targets: NDArray, **kwargs: Any) -> TensorFlowFitResult:
    return _fit_s2s(model, training, features, targets, **kwargs) if is_s2s_model_id(model.model_id) else _fit_s2v(model, training, features, targets, **kwargs)


def fit_lstm(model: LstmModelConfig, training: AdamTrainingConfig, features: Mapping[str, NDArray], targets: NDArray, **kwargs: Any) -> TensorFlowFitResult:
    return _fit_s2s(model, training, features, targets, **kwargs) if is_s2s_model_id(model.model_id) else _fit_s2v(model, training, features, targets, **kwargs)


def fit_gru(model: GruModelConfig, training: AdamTrainingConfig, features: Mapping[str, NDArray], targets: NDArray, **kwargs: Any) -> TensorFlowFitResult:
    return _fit_s2s(model, training, features, targets, **kwargs) if is_s2s_model_id(model.model_id) else _fit_s2v(model, training, features, targets, **kwargs)


def fit_cnn1d(model: Cnn1dModelConfig, training: AdamTrainingConfig, features: Mapping[str, NDArray], targets: NDArray, **kwargs: Any) -> TensorFlowFitResult:
    return _fit_s2s(model, training, features, targets, **kwargs) if is_s2s_model_id(model.model_id) else _fit_s2v(model, training, features, targets, **kwargs)


def fit_s2s(model: MlpModelConfig | LstmModelConfig | GruModelConfig | Cnn1dModelConfig, training: AdamTrainingConfig, features: Mapping[str, NDArray], targets: NDArray, **kwargs: Any) -> TensorFlowFitResult:
    if not is_s2s_model_id(model.model_id):
        raise ValueError("fit_s2s requires a causal S2S TensorFlow model identifier.")
    return _fit_s2s(model, training, features, targets, **kwargs)


def _resolve_artifact(artifact: TensorFlowArtifact | Mapping[str, Any]) -> TensorFlowArtifact:
    return artifact if isinstance(artifact, TensorFlowArtifact) else TensorFlowArtifact.from_mapping(artifact)


def _predict_s2v_from_artifact(
    artifact: TensorFlowArtifact | Mapping[str, Any],
    features: Mapping[str, NDArray],
    *,
    antecedent_features: Mapping[str, NDArray] | None = None,
) -> NDArray[np.floating]:
    resolved = _resolve_artifact(artifact)
    if resolved.artifact_type != S2V_ARTIFACT_TYPE:
        raise ValueError("The selected artifact is not an S2V model.")
    current = feature_matrix(resolved.feature_roles, features)
    prefix_length = 0
    if antecedent_features is not None:
        antecedent = feature_matrix(resolved.feature_roles, antecedent_features)[-resolved.lookback:]
        prefix_length = len(antecedent)
        complete = np.vstack((antecedent, current))
    else:
        complete = current
    scaled = (complete - resolved.input_mean) / resolved.input_scale
    result = np.full(len(current), np.nan, dtype=float)
    target_indices = np.arange(resolved.lookback, len(complete), dtype=int)
    if len(target_indices):
        finite = np.array([np.all(np.isfinite(scaled[index - resolved.lookback:index])) for index in target_indices])
        target_indices = target_indices[finite]
    if len(target_indices):
        windows = _window_rows(scaled, target_indices, resolved.lookback)
        normalized = _predict_scaled(_model_from_artifact(resolved), windows)[:, 0]
        physical = normalized * resolved.target_scale + resolved.target_offset
        current_indices = target_indices - prefix_length
        available = (current_indices >= 0) & (current_indices < len(current))
        result[current_indices[available]] = physical[available]
    return result


def predict_windows_from_artifact(artifact: TensorFlowArtifact | Mapping[str, Any], windows: NDArray) -> NDArray[np.floating]:
    resolved = _resolve_artifact(artifact)
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[1] != resolved.lookback or values.shape[2] != len(resolved.feature_roles) or not np.all(np.isfinite(values)):
        raise ValueError("Neural windows must be finite with shape [samples, lookback, features].")
    if resolved.artifact_type == S2S_ARTIFACT_TYPE:
        return predict_s2s_sequences_from_artifact(resolved, values)[:, -1]
    scaled = (values - resolved.input_mean) / resolved.input_scale
    normalized = _predict_scaled(_model_from_artifact(resolved), scaled)[:, 0]
    return normalized * resolved.target_scale + resolved.target_offset


def predict_from_artifact(artifact: TensorFlowArtifact | Mapping[str, Any], features: Mapping[str, NDArray], *, antecedent_features: Mapping[str, NDArray] | None = None) -> NDArray[np.floating]:
    resolved = _resolve_artifact(artifact)
    if resolved.artifact_type == S2S_ARTIFACT_TYPE:
        return predict_s2s_from_artifact(resolved, features, antecedent_features=antecedent_features)
    return _predict_s2v_from_artifact(resolved, features, antecedent_features=antecedent_features)


predict_lstm_from_artifact = predict_from_artifact
predict_gru_from_artifact = predict_from_artifact
predict_cnn1d_from_artifact = predict_from_artifact
predict_lstm_windows_from_artifact = predict_windows_from_artifact
predict_gru_windows_from_artifact = predict_windows_from_artifact
predict_cnn1d_windows_from_artifact = predict_windows_from_artifact


def predict_s2s_sequences_from_artifact(artifact: TensorFlowArtifact | Mapping[str, Any], sequences: NDArray) -> NDArray[np.floating]:
    resolved = _resolve_artifact(artifact)
    if resolved.artifact_type != S2S_ARTIFACT_TYPE:
        raise ValueError("The selected artifact is not an S2S model.")
    values = np.asarray(sequences, dtype=float)
    if values.ndim != 3 or values.shape[2] != len(resolved.feature_roles) or not np.all(np.isfinite(values)):
        raise ValueError("S2S sequences must be finite with shape [batch, time, features].")
    scaled = (values - resolved.input_mean) / resolved.input_scale
    normalized = _predict_scaled(_model_from_artifact(resolved), scaled)[:, :, 0]
    return normalized * resolved.target_scale + resolved.target_offset


def predict_s2s_windows_from_artifact(artifact: TensorFlowArtifact | Mapping[str, Any], windows: NDArray) -> NDArray[np.floating]:
    resolved = _resolve_artifact(artifact)
    values = np.asarray(windows, dtype=float)
    if values.ndim != 3 or values.shape[1] != resolved.lookback:
        raise ValueError("S2S explanation windows must match the archived sequence length.")
    return predict_s2s_sequences_from_artifact(resolved, values)[:, -1]


def predict_s2s_from_artifact(
    artifact: TensorFlowArtifact | Mapping[str, Any],
    features: Mapping[str, NDArray],
    *,
    antecedent_features: Mapping[str, NDArray] | None = None,
) -> NDArray[np.floating]:
    resolved = _resolve_artifact(artifact)
    current = feature_matrix(resolved.feature_roles, features)
    prefix_length = 0
    if antecedent_features is not None and resolved.lookback > 1:
        antecedent = feature_matrix(resolved.feature_roles, antecedent_features)[-(resolved.lookback - 1):]
        prefix_length = len(antecedent)
        complete = np.vstack((antecedent, current))
    else:
        complete = current
    scaled = (complete - resolved.input_mean) / resolved.input_scale
    finite_rows = np.all(np.isfinite(scaled), axis=1)
    finite_positions = np.flatnonzero(finite_rows)
    if not len(finite_positions):
        return np.full(len(current), np.nan)
    first_finite = int(finite_positions[0])
    if not np.all(finite_rows[first_finite:]):
        raise ValueError("S2S forecasting supports leading unavailable feature rows only.")
    score_start = max(prefix_length, first_finite + resolved.warmup_length)
    result = np.full(len(complete), np.nan)
    if score_start < len(complete):
        dummy = np.zeros(len(complete))
        blocks = _build_blocks(
            scaled, dummy, dummy, context_start=first_finite, score_start=score_start,
            end=len(complete), length=resolved.lookback, warmup=resolved.warmup_length,
        )
        _assign_partition_predictions(resolved, blocks, result)
    return result[prefix_length:]


def predict_s2s_partitioned_from_artifact(
    artifact: TensorFlowArtifact | Mapping[str, Any],
    features: Mapping[str, NDArray],
    *,
    start_index: int,
    train_end_index: int,
    validation_end_index: int,
    test_end_index: int,
) -> NDArray[np.floating]:
    resolved = _resolve_artifact(artifact)
    raw = feature_matrix(resolved.feature_roles, features)
    if not (0 <= start_index < train_end_index < validation_end_index < test_end_index <= len(raw)):
        raise ValueError("Invalid partition boundaries for frozen S2S replay.")
    scaled = (raw - resolved.input_mean) / resolved.input_scale
    dummy = np.zeros(len(raw), dtype=float)
    warmup = resolved.warmup_length
    partitions = (
        _build_blocks(scaled, dummy, dummy, context_start=start_index, score_start=start_index + warmup, end=train_end_index, length=resolved.lookback, warmup=warmup),
        _build_blocks(scaled, dummy, dummy, context_start=max(start_index, train_end_index - warmup), score_start=train_end_index, end=validation_end_index, length=resolved.lookback, warmup=warmup),
        _build_blocks(scaled, dummy, dummy, context_start=max(start_index, validation_end_index - warmup), score_start=validation_end_index, end=test_end_index, length=resolved.lookback, warmup=warmup),
    )
    result = np.full(len(raw), np.nan, dtype=float)
    for blocks in partitions:
        _assign_partition_predictions(resolved, blocks, result)
    return result


__all__ = [
    "CNN1D_S2S_TENSORFLOW_MANIFEST", "CNN1D_TENSORFLOW_MANIFEST",
    "GRU_S2S_TENSORFLOW_MANIFEST", "GRU_TENSORFLOW_MANIFEST",
    "LSTM_S2S_TENSORFLOW_MANIFEST", "LSTM_TENSORFLOW_MANIFEST",
    "MLP_S2S_TENSORFLOW_MANIFEST", "MLP_TENSORFLOW_MANIFEST",
    "STACKED_GRU_TENSORFLOW_MANIFEST", "STACKED_LSTM_TENSORFLOW_MANIFEST",
    "S2S_ARTIFACT_TYPE", "S2V_ARTIFACT_TYPE", "TensorFlowArtifact",
    "MlpArtifact", "LstmArtifact", "StackedLstmArtifact", "GruArtifact",
    "StackedGruArtifact", "Cnn1dArtifact", "S2SArtifact",
    "MlpFitResult", "LstmFitResult", "GruFitResult", "Cnn1dFitResult", "S2SFitResult",
    "fit_mlp", "fit_lstm", "fit_gru", "fit_cnn1d", "fit_s2s", "is_s2s_model_id",
    "predict_from_artifact", "predict_windows_from_artifact",
    "predict_lstm_from_artifact", "predict_lstm_windows_from_artifact",
    "predict_gru_from_artifact", "predict_gru_windows_from_artifact",
    "predict_cnn1d_from_artifact", "predict_cnn1d_windows_from_artifact",
    "predict_s2s_from_artifact", "predict_s2s_sequences_from_artifact",
    "predict_s2s_windows_from_artifact", "predict_s2s_partitioned_from_artifact",
    "tensorflow_runtime_version",
]
