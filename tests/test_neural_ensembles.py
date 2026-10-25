import csv
from pathlib import Path
import tempfile
import unittest
import importlib.util

import numpy as np

from hydrosoftware.configuration import (
    CsvColumnMapping,
    CsvDataSourceConfig,
    ExperimentConfigV1,
)
from hydrosoftware.data_sources import analyze_calibration_csv
from hydrosoftware.execution.ensembles import (
    ENSEMBLE_ARTIFACT_TYPE,
    derive_member_seeds,
)
from hydrosoftware.execution.explainability import explain_run_archive
from hydrosoftware.execution.experiments import ExperimentRunner
from hydrosoftware.execution.forecast_ensembles import (
    run_forcing_ensemble_forecast,
)
from hydrosoftware.execution.forecasting import run_forecast
from hydrosoftware.models.neural import (
    MlpArtifact,
    StackedLstmArtifact,
    predict_from_artifact,
    predict_lstm_from_artifact,
)


@unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow 2.21 is not installed in this test environment")
class NeuralInitializationEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        rng = np.random.default_rng(1701)
        self.precipitation = rng.gamma(1.7, 2.0, 90)
        self.temperature = rng.normal(7.0, 4.0, 90)
        self.pet = rng.uniform(0.0, 2.0, 90)
        self.streamflow = np.zeros(90, dtype=float)
        self.streamflow[1:] = (
            0.52 * self.precipitation[:-1]
            + 0.07 * np.maximum(self.temperature[:-1], 0.0)
            + 0.12 * self.pet[:-1]
        )
        with (self.root / "ensemble.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["p", "t", "pet", "q"])
            writer.writerows(
                zip(
                    self.precipitation,
                    self.temperature,
                    self.pet,
                    self.streamflow,
                )
            )

    def tearDown(self):
        self.directory.cleanup()

    def config(self) -> ExperimentConfigV1:
        return ExperimentConfigV1.model_validate(
            {
                "schema_version": "1.17",
                "experiment_id": "mlp-initialization-ensemble",
                "name": "MLP initialization ensemble",
                "data": {
                    "path": "ensemble.csv",
                    "columns": {
                        "precipitation": "p",
                        "temperature": "t",
                        "potential_evapotranspiration": "pet",
                        "streamflow": "q",
                    },
                },
                "model": {
                    "model_id": "mlp.tensorflow_s2v.v1",
                    "lookback": 3,
                    "hidden_layers": [8],
                    "activation": "tanh",
                    "output_activation": "linear",
                },
                "task": {
                    "task_type": "training",
                    "period": {
                        "start_index": 0,
                        "train_end_index": 50,
                        "validation_end_index": 70,
                        "test_end_index": 90,
                    },
                    "training": {
                        "epochs": 8,
                        "batch_size": 16,
                        "learning_rate": 0.01,
                        "patience": 4,
                        "seed": 29,
                    },
                    "ensemble": {
                        "enabled": True,
                        "members": 3,
                    },
                },
            }
        )

    def test_members_are_reproducible_and_mean_matches_archived_output(self):
        config = self.config()
        first = ExperimentRunner(self.root).run(config, task_id="ensemble-first")
        second = ExperimentRunner(self.root).run(config, task_id="ensemble-second")

        np.testing.assert_allclose(
            first.best_simulation.outputs,
            second.best_simulation.outputs,
            equal_nan=True,
        )
        ensemble = first.training["initialization_ensemble"]
        self.assertEqual(ensemble["member_count_completed"], 3)
        self.assertEqual(
            ensemble["member_seeds"],
            list(derive_member_seeds(29, 3)),
        )
        artifact = first.training["artifact"]
        self.assertEqual(artifact["artifact_type"], ENSEMBLE_ARTIFACT_TYPE)
        features = {
            "precipitation": self.precipitation,
            "temperature": self.temperature,
            "potential_evapotranspiration": self.pet,
        }
        member_predictions = np.stack(
            [
                predict_from_artifact(MlpArtifact.from_mapping(member), features)
                for member in artifact["members"]
            ]
        )
        np.testing.assert_allclose(
            first.best_simulation.outputs[:, 0],
            np.mean(member_predictions, axis=0),
            equal_nan=True,
        )
        lower = np.asarray(
            first.best_simulation.diagnostics["initialization_ensemble_lower"]
        )
        upper = np.asarray(
            first.best_simulation.diagnostics["initialization_ensemble_upper"]
        )
        finite = np.isfinite(first.best_simulation.outputs[:, 0])
        self.assertTrue(
            np.all(lower[finite] <= first.best_simulation.outputs[finite, 0])
        )
        self.assertTrue(
            np.all(first.best_simulation.outputs[finite, 0] <= upper[finite])
        )

    def test_ensemble_forecast_and_shap_use_the_aggregated_model(self):
        config = self.config()
        result = ExperimentRunner(self.root).run(config, task_id="ensemble-run")
        archive = {
            "task_id": "ensemble-run",
            "config": config.model_dump(mode="json"),
            "result": result.as_dict(),
        }
        explanation = explain_run_archive(
            archive,
            explained_rows=2,
            background_rows=2,
            permutations=2,
            seed=4,
        )
        self.assertLess(explanation["maximum_additivity_error"], 1e-10)
        self.assertLess(
            explanation["maximum_archive_prediction_difference"], 1e-10
        )

        forecast_source = CsvDataSourceConfig(
            path="ensemble.csv",
            columns=CsvColumnMapping(
                precipitation="p",
                temperature="t",
                potential_evapotranspiration="pet",
                streamflow="q",
            ),
        )
        forecast_data = analyze_calibration_csv(
            forecast_source, workspace_root=self.root
        )
        forecast = run_forecast(
            archive,
            forecast_data,
            initialization="source_final",
            observations_available=True,
        )
        self.assertEqual(
            forecast["best_simulation"]["diagnostics"][
                "initialization_ensemble_members"
            ],
            3,
        )
        self.assertTrue(forecast["evaluation_metrics"])

        forcing_ensemble = run_forcing_ensemble_forecast(
            archive,
            [
                ("met-1", forecast_data, True),
                ("met-2", forecast_data, True),
            ],
            initialization="source_final",
        )
        self.assertEqual(
            forcing_ensemble["forecast_ensemble"]["member_count"],
            2,
        )
        self.assertTrue(
            forcing_ensemble["forecast_ensemble"]["members"][0][
                "initialization_ensemble_lower"
            ]
        )
        forcing_lower = np.asarray(
            forcing_ensemble["best_simulation"]["diagnostics"][
                "forcing_ensemble_lower"
            ],
            dtype=float,
        )
        forcing_mean = np.asarray(
            forcing_ensemble["best_simulation"]["outputs"],
            dtype=float,
        )[:, 0]
        np.testing.assert_allclose(
            forcing_lower,
            forcing_mean,
            equal_nan=True,
        )

    def test_stacked_lstm_members_preserve_their_architecture(self):
        payload = self.config().model_dump(mode="json")
        payload["schema_version"] = "1.19"
        payload["experiment_id"] = "stacked-lstm-initialization-ensemble"
        payload["name"] = "Stacked LSTM initialization ensemble"
        payload["model"] = {
            "model_id": "lstm.tensorflow_stacked_s2v.v2",
            "lookback": 3,
            "recurrent_layers": [6, 3],
        }
        payload["task"]["training"]["epochs"] = 3
        payload["task"]["training"]["patience"] = 2
        payload["task"]["ensemble"]["members"] = 3
        config = ExperimentConfigV1.model_validate(payload)

        result = ExperimentRunner(self.root).run(
            config,
            task_id="stacked-lstm-ensemble",
        )
        artifact = result.training["artifact"]
        self.assertEqual(artifact["artifact_type"], ENSEMBLE_ARTIFACT_TYPE)
        self.assertEqual(
            [member["architecture"]["recurrent_layers"] for member in artifact["members"]],
            [[6, 3], [6, 3], [6, 3]],
        )
        features = {
            "precipitation": self.precipitation,
            "temperature": self.temperature,
            "potential_evapotranspiration": self.pet,
        }
        member_predictions = np.stack(
            [
                predict_lstm_from_artifact(
                    StackedLstmArtifact.from_mapping(member),
                    features,
                )
                for member in artifact["members"]
            ]
        )
        np.testing.assert_allclose(
            result.best_simulation.outputs[:, 0],
            np.mean(member_predictions, axis=0),
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
