import copy
import csv
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
import importlib.util

import numpy as np

from hydrosoftware.configuration import ExperimentConfigV1
from hydrosoftware.execution.experiments import ExperimentRunner
from hydrosoftware.execution.scientific_verification import verify_run_archive
from hydrosoftware.models.conceptual.exp_hydro import (
    ExpHydroModel,
    ExpHydroParameters,
)
from hydrosoftware.processes.snow import NoSnowModule
from tests.test_experiment_execution import make_config
from tests.test_tensorflow_models import synthetic_series


class ScientificVerificationTests(unittest.TestCase):
    def _conceptual_archive(self, root: Path):
        rng = np.random.default_rng(12)
        precipitation = rng.gamma(1.4, 2.0, size=60)
        temperature = np.full(60, 8.0)
        pet = rng.uniform(0.1, 1.5, size=60)
        streamflow = ExpHydroModel(
            ExpHydroParameters(0.05, 500.0, 30.0),
            NoSnowModule(),
        ).simulate(precipitation, temperature, pet).outputs[:, 0]
        with (root / "basin.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["date", "precipitation", "temperature", "pet", "streamflow"]
            )
            for index, values in enumerate(
                zip(precipitation, temperature, pet, streamflow)
            ):
                writer.writerow(
                    [(date(2025, 1, 1) + timedelta(days=index)).isoformat(), *values]
                )
        config = make_config(path="basin.csv", evaluations=1)
        result = ExperimentRunner(root).run(config, task_id="verified-conceptual")
        return {
            "task_id": result.task_id,
            "config": config.model_dump(mode="json"),
            "result": result.as_dict(),
        }

    def test_conceptual_archive_passes_every_applicable_check(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self._conceptual_archive(Path(directory))
            report = verify_run_archive(archive)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["summary"], {
            "passed": 6,
            "failed": 0,
            "not_applicable": 0,
        })
        checks = {item["check_id"]: item for item in report["checks"]}
        self.assertLessEqual(
            checks["mass_balance"]["maximum_absolute_difference"],
            checks["mass_balance"]["tolerance"],
        )
        self.assertEqual(checks["restart_equivalence"]["status"], "passed")
        self.assertEqual(checks["future_forcing_causality"]["status"], "passed")
        self.assertEqual(
            report["implementation_source"]["relationship"],
            "mass_conservative_exp_hydro_adaptation",
        )
        self.assertEqual(report["implementation_source"]["equivalence_claim"], "not_claimed")

    def test_a_changed_archived_trajectory_fails_frozen_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self._conceptual_archive(Path(directory))
            changed = copy.deepcopy(archive)
            changed["result"]["best_simulation"]["outputs"][20][0] += 0.25
            report = verify_run_archive(changed)

        checks = {item["check_id"]: item for item in report["checks"]}
        self.assertEqual(report["status"], "failed")
        self.assertEqual(checks["deterministic_replay"]["status"], "failed")
        self.assertAlmostEqual(
            checks["deterministic_replay"]["maximum_absolute_difference"],
            0.25,
        )
        self.assertEqual(checks["restart_equivalence"]["status"], "passed")

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow 2.21 is not installed in this test environment")
    def test_neural_archive_replays_restarts_and_remains_causal(self):
        features, streamflow = synthetic_series(length=120)
        precipitation = features["precipitation"]
        temperature = features["temperature"]
        pet = features["potential_evapotranspiration"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "neural.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "p", "t", "pet", "q"])
                for index, values in enumerate(
                    zip(precipitation, temperature, pet, streamflow)
                ):
                    writer.writerow(
                        [(date(2025, 1, 1) + timedelta(days=index)).isoformat(), *values]
                    )
            config = ExperimentConfigV1.model_validate(
                {
                    "schema_version": "1.18",
                    "experiment_id": "verified-mlp",
                    "name": "Verified MLP",
                    "data": {
                        "path": "neural.csv",
                        "columns": {
                            "timestamp": "date",
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
                            "train_end_index": 70,
                            "validation_end_index": 95,
                            "test_end_index": 120,
                        },
                        "objective": {"objective_id": "mean_absolute_error"},
                        "training": {
                            "epochs": 35,
                            "batch_size": 16,
                            "learning_rate": 0.01,
                            "patience": 8,
                            "seed": 21,
                        },
                    },
                }
            )
            result = ExperimentRunner(root).run(config, task_id="verified-neural")
            archive = {
                "task_id": result.task_id,
                "config": config.model_dump(mode="json"),
                "result": result.as_dict(),
            }
            report = verify_run_archive(archive)

        checks = {item["check_id"]: item for item in report["checks"]}
        self.assertEqual(report["status"], "passed")
        self.assertEqual(checks["mass_balance"]["status"], "not_applicable")
        self.assertEqual(
            checks["physical_nonnegativity"]["status"], "not_applicable"
        )
        self.assertEqual(checks["deterministic_replay"]["status"], "passed")
        self.assertEqual(checks["restart_equivalence"]["status"], "passed")
        self.assertEqual(checks["future_forcing_causality"]["status"], "passed")
        self.assertEqual(report["implementation_source"]["relationship"], "hydrosoftware_native")

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow 2.21 is not installed in this test environment")
    def test_causal_s2s_archive_replays_partition_blocks_and_arbitrary_restart(self):
        features, streamflow = synthetic_series(length=120)
        precipitation = features["precipitation"]
        temperature = features["temperature"]
        pet = features["potential_evapotranspiration"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "s2s.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "p", "t", "pet", "q"])
                for index, values in enumerate(
                    zip(precipitation, temperature, pet, streamflow)
                ):
                    writer.writerow(
                        [(date(2025, 1, 1) + timedelta(days=index)).isoformat(), *values]
                    )
            config = ExperimentConfigV1.model_validate(
                {
                    "schema_version": "1.20",
                    "experiment_id": "verified-s2s",
                    "name": "Verified causal S2S",
                    "data": {
                        "path": "s2s.csv",
                        "columns": {
                            "timestamp": "date",
                            "precipitation": "p",
                            "temperature": "t",
                            "potential_evapotranspiration": "pet",
                            "streamflow": "q",
                        },
                    },
                    "model": {
                        "model_id": "lstm.tensorflow_stacked_s2s.v3",
                        "lookback": 8,
                        "s2s_warmup": 3,
                        "recurrent_units": 4,
                        "recurrent_layers": [4],
                        "output_activation": "softplus",
                    },
                    "task": {
                        "task_type": "training",
                        "period": {
                            "start_index": 0,
                            "train_end_index": 70,
                            "validation_end_index": 95,
                            "test_end_index": 120,
                        },
                        "lead_steps": 0,
                        "stride": 1,
                        "objective": {"objective_id": "mean_square_error"},
                        "training": {
                            "epochs": 4,
                            "batch_size": 8,
                            "learning_rate": 0.01,
                            "patience": 4,
                            "seed": 31,
                        },
                    },
                }
            )
            result = ExperimentRunner(root).run(config, task_id="verified-s2s")
            archive = {
                "task_id": result.task_id,
                "config": config.model_dump(mode="json"),
                "result": result.as_dict(),
            }
            report = verify_run_archive(archive)

        checks = {item["check_id"]: item for item in report["checks"]}
        self.assertEqual(report["status"], "passed")
        self.assertEqual(checks["deterministic_replay"]["status"], "passed")
        self.assertEqual(checks["restart_equivalence"]["status"], "passed")
        self.assertEqual(checks["future_forcing_causality"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
