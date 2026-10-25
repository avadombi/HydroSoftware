import unittest
import importlib.util

import numpy as np

from hydrosoftware.configuration import ExperimentConfigV1
from hydrosoftware.execution.explainability import (
    ExplainabilityError,
    explain_run_archive,
)
from hydrosoftware.models.neural import MlpArtifact, predict_from_artifact


def _artifact(feature_roles=("precipitation",), lookback=2):
    width = lookback * len(feature_roles)
    return MlpArtifact(
        artifact_type="hydrosoftware.neural.tensorflow.s2v.v1",
        model_id="mlp.tensorflow_s2v.v1",
        engine_id="tensorflow.keras.2.21",
        feature_roles=feature_roles,
        lookback=lookback,
        warmup_length=0,
        output_activation="linear",
        input_mean=np.zeros(len(feature_roles)),
        input_scale=np.ones(len(feature_roles)),
        target_offset=0.0,
        target_scale=1.0,
        architecture={"kind": "mlp", "hidden_layers": [], "activation": "tanh"},
        weights=(
            np.arange(1, width + 1, dtype=float)[:, None],
            np.zeros(1),
        ),
        best_epoch=1,
        gradient_clip_norm=5.0,
    )


def _data_config():
    return {
        "kind": "csv",
        "path": "unused.csv",
        "columns": {
            "precipitation": "p",
            "temperature": "t",
            "potential_evapotranspiration": "pet",
            "streamflow": "q",
        },
    }


class ExplainabilityTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow 2.21 is not installed in this test environment")
    def test_neural_shap_is_reproducible_additive_and_training_referenced(self):
        artifact = _artifact()
        precipitation = np.arange(10, dtype=float)
        features = {"precipitation": precipitation}
        prediction = predict_from_artifact(artifact, features)
        config = ExperimentConfigV1.model_validate(
            {
                "schema_version": "1.14",
                "experiment_id": "shap-neural",
                "name": "SHAP neural",
                "data": _data_config(),
                "model": {
                    "model_id": "mlp.tensorflow_s2v.v1",
                    "feature_roles": ["precipitation"],
                    "lookback": 2,
                    "hidden_layers": [2],
                    "output_activation": "linear",
                },
                "task": {
                    "task_type": "training",
                    "period": {
                        "start_index": 0,
                        "train_end_index": 5,
                        "validation_end_index": 7,
                        "test_end_index": 10,
                    },
                    "training": {"epochs": 1, "patience": 1},
                },
            }
        )
        archive = {
            "task_id": "run-shap-neural",
            "config": config.model_dump(mode="json"),
            "result": {
                "task_id": "run-shap-neural",
                "state": "completed",
                "run_kind": "training",
                "training": {"artifact": artifact.as_mapping()},
                "forcings": {
                    "precipitation": precipitation,
                    "temperature": np.zeros(10),
                    "potential_evapotranspiration": np.zeros(10),
                },
                "best_simulation": {
                    "outputs": prediction[:, None],
                    "states": np.empty((11, 0)),
                    "fluxes": np.empty((10, 0)),
                    "output_names": ["streamflow"],
                    "state_names": [],
                    "flux_names": [],
                    "diagnostics": {},
                },
                "timestamps": [f"2000-01-{index + 1:02d}" for index in range(10)],
                "observations": prediction,
            },
        }
        first = explain_run_archive(
            archive,
            explained_rows=3,
            background_rows=2,
            permutations=6,
            seed=14,
        )
        second = explain_run_archive(
            archive,
            explained_rows=3,
            background_rows=2,
            permutations=6,
            seed=14,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["background_partition"], "training")
        self.assertEqual(first["explained_partition"], "test")
        self.assertEqual(first["player_count"], 2)
        self.assertLess(first["maximum_additivity_error"], 1e-12)
        self.assertLess(first["maximum_archive_prediction_difference"], 1e-12)
        for local in first["local"]:
            reconstructed = local["baseline"] + sum(
                item["shap_value"] for item in local["attributions"]
            )
            self.assertAlmostEqual(reconstructed, local["prediction"])
        grouped = [
            sum(item["shap_value"] for item in local["attributions"])
            for local in first["local"]
        ]
        self.assertAlmostEqual(
            first["global_by_feature"][0]["mean_absolute_shap"],
            float(np.mean(np.abs(grouped))),
        )

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow 2.21 is not installed in this test environment")

    def test_conceptual_run_is_rejected_instead_of_receiving_fake_shap(self):
        config = ExperimentConfigV1.model_validate(
            {
                "schema_version": "1.14",
                "experiment_id": "conceptual",
                "name": "Conceptual",
                "data": _data_config(),
                "model": {
                    "model_id": "exp_hydro.mass_conservative.v1",
                    "snow": {"module_id": "snow.none.v1"},
                },
                "task": {
                    "task_type": "calibration",
                    "period": {"start_index": 0, "end_index": 10},
                },
            }
        )
        with self.assertRaises(ExplainabilityError) as caught:
            explain_run_archive(
                {
                    "task_id": "conceptual",
                    "config": config.model_dump(mode="json"),
                    "result": {"state": "completed"},
                }
            )
        self.assertEqual(caught.exception.code, "explainability_model_unsupported")


if __name__ == "__main__":
    unittest.main()
