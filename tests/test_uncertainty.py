import math
import unittest
from datetime import date, timedelta

from hydrosoftware.execution.uncertainty import (
    UncertaintyAnalysisError,
    analyze_run_uncertainty,
)


def configuration(name: str, row_count: int) -> dict:
    return {
        "schema_version": "1.15",
        "experiment_id": name.lower().replace(" ", "-"),
        "name": name,
        "data": {
            "path": "data.csv",
            "expected_sha256": "a" * 64,
            "columns": {
                "timestamp": "date",
                "precipitation": "p",
                "temperature": "t",
                "potential_evapotranspiration": "pet",
                "streamflow": "q",
            },
        },
        "model": {
            "model_id": "exp_hydro.mass_conservative.v1",
            "snow": {"module_id": "snow.none.v1"},
            "initial_state": [0.0, 20.0],
        },
        "task": {
            "task_type": "calibration",
            "period": {
                "start_index": 0,
                "end_index": row_count,
                "warmup_steps": 0,
            },
        },
    }


def archive(
    task_id: str,
    name: str,
    observations: list[float | None],
    predictions: list[float],
) -> dict:
    row_count = len(observations)
    return {
        "task_id": task_id,
        "saved_at": "2026-07-28T12:00:00+00:00",
        "config": configuration(name, row_count),
        "result": {
            "task_id": task_id,
            "state": "completed",
            "run_kind": "calibration",
            "observations": observations,
            "timestamps": [
                (date(2026, 1, 1) + timedelta(days=index)).isoformat()
                for index in range(row_count)
            ],
            "best_simulation": {
                "outputs": [[value] for value in predictions],
                "states": [[] for _ in range(row_count + 1)],
                "fluxes": [[] for _ in range(row_count)],
                "output_names": ["streamflow"],
                "state_names": [],
                "flux_names": [],
                "diagnostics": {},
            },
        },
    }


class RunUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.observed = [
            2.0
            + 0.8 * math.sin(index / 5.0)
            + 0.015 * index
            + (1.5 if index % 17 == 0 else 0.0)
            for index in range(84)
        ]

    def test_single_run_perfect_metrics_have_exact_intervals(self):
        result = analyze_run_uncertainty(
            (
                archive(
                    "perfect-run",
                    "Perfect",
                    self.observed,
                    self.observed,
                ),
            ),
            bootstrap_replicates=200,
            seed=7,
        )

        self.assertEqual(result["method_id"], "moving_block_bootstrap_metrics.v1")
        self.assertEqual(result["scored_observations"], len(self.observed))
        self.assertFalse(result["paired_resampling"])
        expected = {
            "NSE": 1.0,
            "KGE": 1.0,
            "MSE": 0.0,
            "RMSE": 0.0,
            "NRMSE": 0.0,
            "MAE": 0.0,
            "PBIAS": 0.0,
        }
        for metric in result["runs"][0]["metrics"]:
            self.assertAlmostEqual(metric["value"], expected[metric["symbol"]])
            self.assertAlmostEqual(metric["lower"], expected[metric["symbol"]])
            self.assertAlmostEqual(metric["upper"], expected[metric["symbol"]])

    def test_paired_resampling_identifies_a_consistently_better_candidate(self):
        reference_predictions = [0.7 * value + 0.3 for value in self.observed]
        candidate_predictions = [value + 0.02 * math.sin(index) for index, value in enumerate(self.observed)]
        result = analyze_run_uncertainty(
            (
                archive(
                    "reference-run",
                    "Reference",
                    self.observed,
                    reference_predictions,
                ),
                archive(
                    "candidate-run",
                    "Candidate",
                    self.observed,
                    candidate_predictions,
                ),
            ),
            bootstrap_replicates=300,
            block_length=7,
            seed=11,
        )

        self.assertTrue(result["paired_resampling"])
        self.assertEqual(result["reference_task_id"], "reference-run")
        comparison = result["comparisons"][0]
        rmse = next(
            metric for metric in comparison["metrics"] if metric["symbol"] == "RMSE"
        )
        self.assertGreater(rmse["improvement"], 0.0)
        self.assertGreater(rmse["improvement_lower"], 0.0)
        self.assertEqual(rmse["interpretation"], "interval_favors_candidate")
        self.assertGreater(rmse["bootstrap_fraction_better"], 0.99)

    def test_same_seed_reproduces_every_bootstrap_summary(self):
        selected = (
            archive(
                "run-one",
                "One",
                self.observed,
                [0.9 * value for value in self.observed],
            ),
        )
        first = analyze_run_uncertainty(
            selected,
            bootstrap_replicates=200,
            block_length=5,
            seed=42,
        )
        second = analyze_run_uncertainty(
            selected,
            bootstrap_replicates=200,
            block_length=5,
            seed=42,
        )
        self.assertEqual(first, second)

    def test_blocks_never_cross_a_missing_scored_interval(self):
        observations: list[float | None] = list(self.observed)
        observations[41] = None
        result = analyze_run_uncertainty(
            (
                archive(
                    "gapped-run",
                    "Gapped",
                    observations,
                    self.observed,
                ),
            ),
            bootstrap_replicates=200,
            block_length=12,
            seed=3,
        )
        self.assertEqual(result["longest_contiguous_scored_run"], 42)
        warning_codes = {warning["code"] for warning in result["warnings"]}
        self.assertIn("uncertainty_scored_gaps_preserved", warning_codes)

    def test_block_longer_than_contiguous_support_is_rejected(self):
        observations: list[float | None] = list(self.observed)
        observations[41] = None
        with self.assertRaisesRegex(
            UncertaintyAnalysisError,
            "longest contiguous",
        ):
            analyze_run_uncertainty(
                (
                    archive(
                        "gapped-run",
                        "Gapped",
                        observations,
                        self.observed,
                    ),
                ),
                bootstrap_replicates=200,
                block_length=43,
            )


if __name__ == "__main__":
    unittest.main()
