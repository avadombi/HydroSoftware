import importlib.util
import unittest

import numpy as np

from hydrosoftware.configuration import (
    AdamTrainingConfig,
    Cnn1dModelConfig,
    Conv1dLayerConfig,
    GruModelConfig,
    LstmModelConfig,
    MlpModelConfig,
)
from hydrosoftware.models.neural import (
    S2S_ARTIFACT_TYPE,
    S2V_ARTIFACT_TYPE,
    TensorFlowArtifact,
    fit_cnn1d,
    fit_gru,
    fit_lstm,
    fit_mlp,
    fit_s2s,
    predict_from_artifact,
    predict_s2s_sequences_from_artifact,
    tensorflow_runtime_version,
)


TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


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


def training(seed: int = 17) -> AdamTrainingConfig:
    return AdamTrainingConfig(
        epochs=4,
        batch_size=16,
        learning_rate=0.01,
        patience=4,
        seed=seed,
        shuffle=False,
    )


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow 2.21 is not installed in this test environment")
class TensorFlowNeuralModelTests(unittest.TestCase):
    def test_runtime_is_tensorflow_2_21(self):
        self.assertTrue(tensorflow_runtime_version().startswith("2.21."))

    def test_s2v_families_train_save_and_replay(self):
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
            ),
            (
                LstmModelConfig(
                    model_id="lstm.tensorflow_stacked_s2v.v2",
                    lookback=4,
                    recurrent_layers=(6, 3),
                    output_activation="linear",
                ),
                fit_lstm,
            ),
            (
                GruModelConfig(
                    model_id="gru.tensorflow_stacked_s2v.v2",
                    lookback=4,
                    recurrent_layers=(6, 3),
                    output_activation="linear",
                ),
                fit_gru,
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
            ),
        )
        for model, fitter in cases:
            with self.subTest(model=model.model_id):
                fit = fitter(
                    model,
                    training(),
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
                self.assertEqual(fit.artifact.artifact_type, S2V_ARTIFACT_TYPE)
                self.assertEqual(fit.artifact.engine_id, "tensorflow.keras.2.21")
                self.assertTrue(fit.artifact.weights)
                restored = TensorFlowArtifact.from_mapping(fit.artifact.as_mapping())
                replay = predict_from_artifact(restored, features)
                mask = np.isfinite(fit.predictions)
                np.testing.assert_allclose(replay[mask], fit.predictions[mask], rtol=1e-6, atol=1e-6)
                expected_mean = np.column_stack(
                    [features[name] for name in model.feature_roles]
                )[:56].mean(axis=0)
                np.testing.assert_allclose(restored.input_mean, expected_mean)

    def test_same_seed_repeats_the_same_mlp_fit(self):
        features, targets = synthetic_series()
        model = MlpModelConfig(
            model_id="mlp.tensorflow_s2v.v1",
            lookback=4,
            hidden_layers=(6,),
            activation="tanh",
            output_activation="linear",
        )
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
        first = fit_mlp(model, training(seed=11), features, targets, **kwargs)
        second = fit_mlp(model, training(seed=11), features, targets, **kwargs)
        np.testing.assert_allclose(first.predictions, second.predictions, rtol=1e-6, atol=1e-6, equal_nan=True)
        self.assertEqual(first.artifact.best_epoch, second.artifact.best_epoch)

    def test_s2s_families_are_causal(self):
        features, targets = synthetic_series()
        models = (
            MlpModelConfig(
                model_id="mlp.tensorflow_s2s.v2",
                lookback=6,
                s2s_warmup=2,
                hidden_layers=(6,),
                activation="tanh",
                output_activation="linear",
            ),
            LstmModelConfig(
                model_id="lstm.tensorflow_stacked_s2s.v3",
                lookback=6,
                s2s_warmup=2,
                recurrent_layers=(5, 3),
                output_activation="linear",
            ),
            GruModelConfig(
                model_id="gru.tensorflow_stacked_s2s.v3",
                lookback=6,
                s2s_warmup=2,
                recurrent_layers=(5, 3),
                output_activation="linear",
            ),
            Cnn1dModelConfig(
                model_id="cnn1d.tensorflow_s2s.v2",
                lookback=6,
                s2s_warmup=2,
                convolution_layers=(
                    Conv1dLayerConfig(filters=4, kernel_size=3, activation="tanh"),
                ),
                output_activation="linear",
            ),
        )
        raw = np.column_stack(
            [features["precipitation"], features["temperature"], features["potential_evapotranspiration"]]
        )
        for model in models:
            with self.subTest(model=model.model_id):
                fit = fit_s2s(
                    model,
                    training(seed=23),
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
                self.assertEqual(fit.artifact.artifact_type, S2S_ARTIFACT_TYPE)
                window = raw[20:26][None, :, :]
                first = predict_s2s_sequences_from_artifact(fit.artifact, window)
                changed = window.copy()
                changed[:, 4:, 0] += 100.0
                second = predict_s2s_sequences_from_artifact(fit.artifact, changed)
                np.testing.assert_allclose(first[:, :4], second[:, :4], rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
