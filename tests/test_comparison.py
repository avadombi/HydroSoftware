import unittest

from hydrosoftware.execution.comparison import (
    RunComparisonError,
    compare_run_archives,
)


def configuration(model, task, *, fingerprint="a" * 64, name="Experiment"):
    return {
        "schema_version": "1.14",
        "experiment_id": name.lower().replace(" ", "-"),
        "name": name,
        "data": {
            "path": "data.csv",
            "expected_sha256": fingerprint,
            "columns": {
                "timestamp": "date",
                "precipitation": "p",
                "temperature": "t",
                "potential_evapotranspiration": "pet",
                "streamflow": "q",
            },
        },
        "model": model,
        "task": task,
    }


def archive(task_id, config, predictions):
    observations = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0]
    return {
        "task_id": task_id,
        "saved_at": "2026-07-28T12:00:00+00:00",
        "config": config,
        "result": {
            "task_id": task_id,
            "state": "completed",
            "run_kind": (
                "training"
                if config["task"]["task_type"] != "calibration"
                else "calibration"
            ),
            "observations": observations,
            "timestamps": [f"2026-01-0{index + 1}" for index in range(6)],
            "best_simulation": {
                "outputs": [[value] for value in predictions],
                "states": [[] for _ in range(7)],
                "fluxes": [[] for _ in range(6)],
                "output_names": ["streamflow"],
                "state_names": [],
                "flux_names": [],
                "diagnostics": {},
            },
        },
    }


class RunComparisonTests(unittest.TestCase):
    def test_metrics_are_recomputed_on_one_common_native_evaluation_support(self):
        conceptual = configuration(
            {
                "model_id": "exp_hydro.mass_conservative.v1",
                "snow": {"module_id": "snow.none.v1"},
                "initial_state": [0.0, 20.0],
            },
            {
                "task_type": "calibration",
                "period": {
                    "start_index": 0,
                    "end_index": 6,
                    "warmup_steps": 1,
                },
            },
            name="Conceptual",
        )
        neural = configuration(
            {
                "model_id": "mlp.tensorflow_s2v.v1",
                "feature_roles": ["precipitation"],
                "lookback": 1,
                "hidden_layers": [3],
                "output_activation": "linear",
            },
            {
                "task_type": "training",
                "period": {
                    "start_index": 0,
                    "train_end_index": 3,
                    "validation_end_index": 4,
                    "test_end_index": 6,
                },
            },
            name="Neural",
        )
        comparison = compare_run_archives(
            (
                archive(
                    "run-conceptual",
                    conceptual,
                    [1, 2, 1.4, 2.8, 2.4, 3.8],
                ),
                archive(
                    "run-neural",
                    neural,
                    [1, 2, 1.5, 3.0, 2.6, 4.2],
                ),
            )
        )

        self.assertEqual(comparison["alignment"], "timestamp")
        self.assertEqual(comparison["scored_observations"], 2)
        self.assertEqual(comparison["first_scored_label"], "2026-01-05")
        self.assertEqual(comparison["last_scored_label"], "2026-01-06")
        self.assertEqual(len(comparison["runs"][0]["metrics"]), 7)
        warning_codes = {item["code"] for item in comparison["warnings"]}
        self.assertIn("comparison_fit_protocols_differ", warning_codes)
        self.assertIn(
            "comparison_legacy_conceptual_in_sample",
            warning_codes,
        )

    def test_different_source_fingerprints_are_rejected(self):
        task = {
            "task_type": "calibration",
            "period": {"start_index": 0, "end_index": 6, "warmup_steps": 1},
        }
        model = {
            "model_id": "exp_hydro.mass_conservative.v1",
            "snow": {"module_id": "snow.none.v1"},
            "initial_state": [0.0, 20.0],
        }
        first = configuration(model, task, fingerprint="a" * 64, name="First")
        second = configuration(model, task, fingerprint="b" * 64, name="Second")
        with self.assertRaisesRegex(
            RunComparisonError,
            "same source data",
        ):
            compare_run_archives(
                (
                    archive(
                        "run-first",
                        first,
                        [1, 2, 1.5, 3, 2.5, 4],
                    ),
                    archive(
                        "run-second",
                        second,
                        [1, 2, 1.5, 3, 2.5, 4],
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
