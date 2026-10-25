#!/usr/bin/env python3
"""Runtime verification for HydroSoftware's TensorFlow 2.21 modelling backend.

Run this script in the release environment after installing the package. It
trains small MLP, LSTM, GRU, CNN-1D and PaML examples, checks
Keras layer types, and restores saved model data.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is not installed. Install HydroSoftware dependencies first; "
        "v1.0.0 requires tensorflow==2.21.0."
    ) from exc

from hydrosoftware.configuration import (
    AdamTrainingConfig,
    Cnn1dModelConfig,
    Conv1dLayerConfig,
    ExperimentConfigV1,
    GruModelConfig,
    LstmModelConfig,
    MlpModelConfig,
)
from hydrosoftware.execution.experiments import ExperimentRunner
from hydrosoftware.models.conceptual.exp_hydro import ExpHydroModel, ExpHydroParameters
from hydrosoftware.models.neural import (
    TensorFlowArtifact,
    fit_cnn1d,
    fit_gru,
    fit_lstm,
    fit_mlp,
    predict_from_artifact,
    tensorflow_runtime_version,
)
from hydrosoftware.models.neural.tensorflow_models import _build_keras_model


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def synthetic_series(length: int = 96):
    rng = np.random.default_rng(2706)
    precipitation = rng.gamma(1.8, 1.6, size=length)
    temperature = rng.normal(7.0, 4.0, size=length)
    pet = rng.uniform(0.1, 2.0, size=length)
    streamflow = np.zeros(length, dtype=float)
    streamflow[2:] = (
        0.55 * precipitation[1:-1]
        + 0.18 * precipitation[:-2]
        + 0.06 * np.maximum(temperature[1:-1], 0.0)
        + 0.12 * pet[1:-1]
    )
    return {
        "precipitation": precipitation,
        "temperature": temperature,
        "potential_evapotranspiration": pet,
    }, streamflow


def neural_training(seed: int = 17) -> AdamTrainingConfig:
    return AdamTrainingConfig(
        epochs=3,
        batch_size=16,
        learning_rate=0.01,
        patience=3,
        seed=seed,
        shuffle=False,
    )


def verify_keras_layers_and_replay() -> None:
    features, targets = synthetic_series()
    cases = (
        (
            MlpModelConfig(
                model_id="mlp.tensorflow_s2v.v1",
                lookback=4,
                hidden_layers=(8,),
                activation="tanh",
                output_activation="linear",
            ),
            fit_mlp,
            tf.keras.layers.Dense,
        ),
        (
            LstmModelConfig(
                model_id="lstm.tensorflow_stacked_s2v.v2",
                lookback=4,
                recurrent_layers=(6, 3),
                output_activation="linear",
            ),
            fit_lstm,
            tf.keras.layers.LSTM,
        ),
        (
            GruModelConfig(
                model_id="gru.tensorflow_stacked_s2v.v2",
                lookback=4,
                recurrent_layers=(6, 3),
                output_activation="linear",
            ),
            fit_gru,
            tf.keras.layers.GRU,
        ),
        (
            Cnn1dModelConfig(
                model_id="cnn1d.tensorflow_s2v.v1",
                lookback=4,
                convolution_layers=(
                    Conv1dLayerConfig(filters=5, kernel_size=3, activation="tanh"),
                ),
                output_activation="linear",
            ),
            fit_cnn1d,
            tf.keras.layers.Conv1D,
        ),
    )
    for model_config, fitter, expected_layer in cases:
        architecture = {
            "kind": model_config.model_id.split(".", 1)[0],
        }
        if isinstance(model_config, MlpModelConfig):
            architecture.update(hidden_layers=list(model_config.hidden_layers), activation=model_config.activation)
        elif isinstance(model_config, (LstmModelConfig, GruModelConfig)):
            architecture.update(recurrent_layers=list(model_config.recurrent_layers))
        else:
            architecture.update(
                convolution_layers=[
                    {"filters": x.filters, "kernel_size": x.kernel_size, "activation": x.activation}
                    for x in model_config.convolution_layers
                ]
            )
        keras_model = _build_keras_model(
            model_id=model_config.model_id,
            architecture=architecture,
            feature_count=len(model_config.feature_roles),
            lookback=model_config.lookback,
            output_activation=model_config.output_activation,
            s2s=False,
        )
        check(any(isinstance(layer, expected_layer) for layer in keras_model.layers),
              f"{model_config.model_id} does not contain {expected_layer.__name__}.")
        fit = fitter(
            model_config,
            neural_training(),
            features,
            targets,
            start_index=0,
            train_end_index=56,
            validation_end_index=76,
            test_end_index=96,
            objective_id="root_mean_square_error",
            stride=1,
            cancellation_requested=None,
            on_epoch=None,
        )
        check(fit.artifact.engine_id == "tensorflow.keras.2.21", "Unexpected neural engine identifier.")
        restored = TensorFlowArtifact.from_mapping(fit.artifact.as_mapping())
        replay = predict_from_artifact(restored, features)
        mask = np.isfinite(fit.predictions)
        np.testing.assert_allclose(replay[mask], fit.predictions[mask], rtol=1e-6, atol=1e-6)
        print(f"PASS neural: {model_config.model_id} ({expected_layer.__name__})")

    # Repeat one fit to exercise deterministic initialization and training.
    model = cases[0][0]
    kwargs = dict(
        start_index=0,
        train_end_index=56,
        validation_end_index=76,
        test_end_index=96,
        objective_id="root_mean_square_error",
        stride=1,
        cancellation_requested=None,
        on_epoch=None,
    )
    first = fit_mlp(model, neural_training(seed=11), features, targets, **kwargs)
    second = fit_mlp(model, neural_training(seed=11), features, targets, **kwargs)
    np.testing.assert_allclose(first.predictions, second.predictions, rtol=1e-6, atol=1e-6, equal_nan=True)
    check(first.artifact.best_epoch == second.artifact.best_epoch, "Repeated MLP fit selected a different epoch.")
    print("PASS neural determinism: MLP repeated seed")


def verify_paml() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        length = 90
        index = np.arange(length, dtype=float)
        precipitation = 2.0 + 1.5 * (np.sin(index / 5.0) + 1.0)
        temperature = 4.0 + 8.0 * np.sin(index / 20.0)
        pet = 0.8 + 0.2 * np.cos(index / 9.0)
        observations = np.maximum(0.0, 0.35 * precipitation + 0.05 * np.roll(precipitation, 1))
        observations[0] = 0.35 * precipitation[0]
        csv_path = root / "paml.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["p", "t", "pet", "q"])
            writer.writerows(zip(precipitation, temperature, pet, observations))
        payload = {
            "schema_version": "1.21",
            "experiment_id": "tensorflow-paml-check",
            "name": "TensorFlow PaML runtime check",
            "data": {
                "kind": "csv",
                "path": "paml.csv",
                "columns": {
                    "precipitation": "p",
                    "temperature": "t",
                    "potential_evapotranspiration": "pet",
                    "streamflow": "q",
                },
            },
            "model": {
                "model_id": "paml.tensorflow_prnn_cnn_s2s.v1",
                "physical_model": {
                    "model_id": "exp_hydro.mass_conservative.v1",
                },
                "cnn_filters": 4,
                "cnn_kernel_size": 3,
                "training_strategy": "joint_end_to_end",
                "enforce_nonnegative_output": True,
            },
            "task": {
                "task_type": "paml_training",
                "period": {
                    "start_index": 0,
                    "train_end_index": 50,
                    "validation_end_index": 70,
                    "test_end_index": 90,
                },
                "objective": {"objective_id": "nash_sutcliffe_efficiency"},
                "warmup_steps": 5,
                "training": {
                    "optimizer_id": "adam.tensorflow.v1",
                    "epochs": 2,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "patience": 2,
                    "minimum_delta": 0.0,
                    "gradient_clip_norm": 5.0,
                    "seed": 2,
                    "shuffle": False,
                },
            },
        }
        config = ExperimentConfigV1.model_validate(payload)
        result = ExperimentRunner(root).run(config, task_id="tensorflow-paml-check")
        check(result.state.value == "completed", "PaML runtime check did not complete.")
        check(result.training["training_strategy"] == "joint_end_to_end", "PaML is not jointly trained.")
        check(result.training["physical_model_id"] == "exp_hydro.mass_conservative.v1", "PaML physical backbone is incorrect.")
        check("physical_streamflow" in result.best_simulation.output_names, "PaML physical discharge is not retained.")
        print("PASS PaML: jointly trained P-RNN + causal CNN")


def main() -> int:
    version = tensorflow_runtime_version()
    check(version == "2.21.0", f"Expected TensorFlow 2.21.0, found {version}.")
    print(f"TensorFlow runtime: {version}")
    print(f"Keras runtime: {getattr(tf.keras, '__version__', 'bundled with TensorFlow')}")
    verify_keras_layers_and_replay()
    verify_paml()
    print("TensorFlow backend verification PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
