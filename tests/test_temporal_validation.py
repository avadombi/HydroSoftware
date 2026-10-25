import csv
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
import importlib.util

import numpy as np

from hydrosoftware.configuration import ExperimentConfigV1
from hydrosoftware.execution.experiments import ExperimentRunner
from hydrosoftware.execution.temporal_validation import (
    RollingOriginSettings,
    RollingOriginValidator,
    TemporalValidationError,
    TemporalValidationTaskManager,
    build_temporal_folds,
)
from hydrosoftware.models.conceptual.exp_hydro import (
    ExpHydroModel,
    ExpHydroParameters,
)
from hydrosoftware.processes.snow import NoSnowModule


def conceptual_config(
    *,
    experiment_id: str = "temporal-exp-hydro",
) -> ExperimentConfigV1:
    return ExperimentConfigV1.model_validate(
        {
            "schema_version": "1.17",
            "experiment_id": experiment_id,
            "name": experiment_id.replace("-", " "),
            "data": {
                "path": "basin.csv",
                "columns": {
                    "timestamp": "date",
                    "precipitation": "precipitation",
                    "temperature": "temperature",
                    "potential_evapotranspiration": "pet",
                    "streamflow": "streamflow",
                },
            },
            "model": {
                "model_id": "exp_hydro.mass_conservative.v1",
                "snow": {"module_id": "snow.none.v1"},
                "initial_state": [0.0, 0.0],
                "substeps": 4,
            },
            "task": {
                "task_type": "calibration",
                "period": {
                    "start_index": 0,
                    "end_index": 50,
                    "warmup_steps": 5,
                },
                "objective": {
                    "objective_id": "nash_sutcliffe_efficiency",
                },
                "search": {
                    "algorithm_id": "uniform_random_search.v1",
                    "n_evaluations": 1,
                    "seed": 17,
                    "include_defaults": True,
                    "continue_on_candidate_error": True,
                },
            },
        }
    )


def neural_config() -> ExperimentConfigV1:
    return ExperimentConfigV1.model_validate(
        {
            "schema_version": "1.17",
            "experiment_id": "temporal-stacked-gru",
            "name": "Temporal stacked GRU",
            "data": conceptual_config().data.model_dump(mode="json"),
            "model": {
                "model_id": "gru.tensorflow_stacked_s2v.v2",
                "feature_roles": [
                    "precipitation",
                    "temperature",
                    "potential_evapotranspiration",
                ],
                "lookback": 3,
                "recurrent_layers": [5, 3],
                "input_transform": "standardize_training_only",
                "target_transform": "standardize_training_only",
            },
            "task": {
                "task_type": "training",
                "period": {
                    "start_index": 0,
                    "train_end_index": 40,
                    "validation_end_index": 50,
                    "test_end_index": 60,
                },
                "target_role": "streamflow",
                "lead_steps": 1,
                "stride": 1,
                "training": {
                    "optimizer_id": "adam.tensorflow.v1",
                    "epochs": 2,
                    "batch_size": 8,
                    "learning_rate": 0.005,
                    "patience": 2,
                    "minimum_delta": 0.0,
                    "seed": 23,
                    "shuffle": True,
                },
            },
        }
    )


def s2s_neural_config() -> ExperimentConfigV1:
    payload = neural_config().model_dump(mode="json")
    payload["schema_version"] = "1.20"
    payload["experiment_id"] = "temporal-causal-s2s-gru"
    payload["name"] = "Temporal causal S2S GRU"
    payload["model"] = {
        "model_id": "gru.tensorflow_stacked_s2s.v3",
        "feature_roles": [
            "precipitation",
            "temperature",
            "potential_evapotranspiration",
        ],
        "lookback": 6,
        "s2s_warmup": 2,
        "recurrent_layers": [4, 2],
        "output_activation": "linear",
    }
    payload["task"]["lead_steps"] = 0
    payload["task"]["stride"] = 1
    payload["task"]["training"]["epochs"] = 1
    payload["task"]["training"]["patience"] = 1
    return ExperimentConfigV1.model_validate(payload)


class TemporalValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        rng = np.random.default_rng(717)
        precipitation = rng.gamma(1.8, 2.2, size=80)
        temperature = 6.0 + 8.0 * np.sin(np.arange(80) / 20.0)
        pet = rng.uniform(0.1, 1.8, size=80)
        streamflow = ExpHydroModel(
            ExpHydroParameters(0.05, 500.0, 30.0),
            NoSnowModule(),
        ).simulate(precipitation, temperature, pet).outputs[:, 0]
        with (self.root / "basin.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["date", "precipitation", "temperature", "pet", "streamflow"]
            )
            for index, values in enumerate(
                zip(precipitation, temperature, pet, streamflow)
            ):
                writer.writerow(
                    [
                        (date(2020, 1, 1) + timedelta(days=index)).isoformat(),
                        *values,
                    ]
                )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_fold_geometry_is_chronological_and_non_overlapping(self):
        expanding = build_temporal_folds(
            RollingOriginSettings(
                start_index=5,
                initial_training_steps=40,
                validation_steps=5,
                test_steps=10,
                step_steps=10,
                folds=3,
                window_mode="expanding",
                minimum_scored_observations=5,
            )
        )
        self.assertEqual(
            [
                (
                    fold.train_start_index,
                    fold.train_end_index,
                    fold.validation_end_index,
                    fold.test_end_index,
                )
                for fold in expanding
            ],
            [(5, 45, 50, 60), (5, 55, 60, 70), (5, 65, 70, 80)],
        )
        sliding = build_temporal_folds(
            RollingOriginSettings(
                start_index=5,
                initial_training_steps=40,
                validation_steps=5,
                test_steps=10,
                step_steps=10,
                folds=3,
                window_mode="sliding",
                minimum_scored_observations=5,
            )
        )
        self.assertEqual(
            [
                (fold.train_start_index, fold.train_end_index)
                for fold in sliding
            ],
            [(5, 45), (15, 55), (25, 65)],
        )
        with self.assertRaisesRegex(
            TemporalValidationError,
            "do not overlap",
        ):
            RollingOriginSettings(
                initial_training_steps=20,
                validation_steps=5,
                test_steps=10,
                step_steps=5,
                folds=2,
            )

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow 2.21 is not installed in this test environment")
    def test_refits_conceptual_and_neural_models_on_common_future_blocks(self):
        settings = RollingOriginSettings(
            start_index=0,
            initial_training_steps=40,
            validation_steps=10,
            test_steps=10,
            step_steps=10,
            folds=2,
            window_mode="expanding",
            minimum_scored_observations=5,
        )
        result = RollingOriginValidator(ExperimentRunner(self.root)).run(
            [conceptual_config(), neural_config()],
            settings,
            task_id="temporal-direct",
        )

        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["pooled_scored_observations"], 20)
        self.assertEqual(len(result["folds"]), 2)
        self.assertEqual(len(result["models"]), 2)
        self.assertEqual(len(result["pairwise"]), 1)
        for fold in result["folds"]:
            self.assertEqual(fold["scored_observations"], 10)
            self.assertEqual(
                fold["source_rows"],
                list(
                    range(
                        fold["validation_end_index"],
                        fold["test_end_index"],
                    )
                ),
            )
            conceptual = next(
                item
                for item in fold["models"]
                if item["experiment_id"] == "temporal-exp-hydro"
            )
            neural = next(
                item
                for item in fold["models"]
                if item["experiment_id"] == "temporal-stacked-gru"
            )
            self.assertEqual(
                conceptual["fitting_end_index"],
                fold["validation_end_index"],
            )
            self.assertEqual(
                conceptual["validation_used_for"],
                "conceptual_parameter_calibration",
            )
            self.assertEqual(
                neural["fitting_end_index"],
                fold["train_end_index"],
            )
            self.assertEqual(
                neural["validation_used_for"],
                "checkpoint_selection",
            )
        conceptual_summary = next(
            item
            for item in result["models"]
            if item["experiment_id"] == "temporal-exp-hydro"
        )
        nse = next(
            item
            for item in conceptual_summary["pooled_metrics"]
            if item["symbol"] == "NSE"
        )
        self.assertAlmostEqual(nse["value"], 1.0, places=12)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow 2.21 is not installed in this test environment")
    def test_same_saved_seed_and_folds_are_reproducible(self):
        settings = RollingOriginSettings(
            initial_training_steps=40,
            validation_steps=10,
            test_steps=10,
            step_steps=10,
            folds=2,
            minimum_scored_observations=5,
        )
        validator = RollingOriginValidator(ExperimentRunner(self.root))
        first = validator.run(
            [neural_config()],
            settings,
            task_id="temporal-repeat-a",
        )
        second = validator.run(
            [neural_config()],
            settings,
            task_id="temporal-repeat-b",
        )
        self.assertEqual(first["models"], second["models"])
        first_predictions = [
            fold["models"][0]["predictions"] for fold in first["folds"]
        ]
        second_predictions = [
            fold["models"][0]["predictions"] for fold in second["folds"]
        ]
        self.assertEqual(first_predictions, second_predictions)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow 2.21 is not installed in this test environment")
    def test_causal_s2s_is_refitted_inside_each_chronological_fold(self):
        result = RollingOriginValidator(ExperimentRunner(self.root)).run(
            [s2s_neural_config()],
            RollingOriginSettings(
                initial_training_steps=40,
                validation_steps=10,
                test_steps=10,
                step_steps=10,
                folds=2,
                minimum_scored_observations=5,
            ),
            task_id="temporal-causal-s2s",
        )

        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["pooled_scored_observations"], 20)
        for fold in result["folds"]:
            model = fold["models"][0]
            self.assertEqual(
                model["model_id"],
                "gru.tensorflow_stacked_s2s.v3",
            )
            self.assertEqual(model["fitting_end_index"], fold["train_end_index"])
            self.assertEqual(model["validation_used_for"], "checkpoint_selection")
            self.assertEqual(len(model["predictions"]), 10)
            self.assertTrue(np.all(np.isfinite(model["predictions"])))

    def test_async_manager_exposes_ordered_progress_and_result(self):
        saved = []
        manager = TemporalValidationTaskManager(
            RollingOriginValidator(ExperimentRunner(self.root)),
            on_result=lambda task_id, result: saved.append(
                (task_id, result["state"])
            ),
        )
        try:
            submitted = manager.submit(
                [conceptual_config()],
                RollingOriginSettings(
                    initial_training_steps=40,
                    validation_steps=10,
                    test_steps=10,
                    step_steps=10,
                    folds=2,
                    minimum_scored_observations=5,
                ),
            )
            completed = manager.wait(submitted.task_id, timeout=10.0)
            events = manager.events(submitted.task_id)
            result = manager.result(submitted.task_id)
        finally:
            manager.shutdown()

        self.assertEqual(completed.state.value, "completed")
        self.assertTrue(completed.terminal)
        self.assertEqual(
            [event.sequence for event in events],
            list(range(len(events))),
        )
        self.assertEqual(result["pooled_scored_observations"], 20)
        self.assertEqual(saved, [(submitted.task_id, "completed")])


if __name__ == "__main__":
    unittest.main()
