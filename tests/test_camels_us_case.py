"""Regression tests for the frozen CAMELS-US 01013500 real-basin case."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest

import numpy as np

from hydrosoftware.configuration import load_experiment_config
from hydrosoftware.data_sources import analyze_calibration_csv
from hydrosoftware.execution.experiments import ExperimentRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "examples" / "camels_us_01013500_exp_hydro.yml"
LSTM_CONFIG_PATH = PROJECT_ROOT / "examples" / "camels_us_01013500_lstm.yml"
DEEPDISCOVER_CONFIG_PATH = (
    PROJECT_ROOT / "examples" / "camels_us_01013500_deepdiscover.yml"
)


class CamelsUsCaseTests(unittest.TestCase):
    @staticmethod
    def configuration():
        return load_experiment_config(CONFIG_PATH)

    def test_frozen_data_record(self) -> None:
        configuration = self.configuration()
        analysis = analyze_calibration_csv(
            configuration.data, workspace_root=PROJECT_ROOT
        )

        self.assertTrue(analysis.report.ready)
        self.assertEqual(
            analysis.report.sha256,
            "f91a50e1a21bee952a7439117213422fd25244de623a0d280eb83ef3c6cd89b0",
        )
        self.assertEqual(analysis.report.row_count, 12_784)
        self.assertEqual(analysis.report.timestamps.first, "1980-01-01")
        self.assertEqual(analysis.report.timestamps.last, "2014-12-31")
        self.assertEqual(analysis.report.timestamps.invalid_count, 0)
        self.assertEqual(analysis.report.timestamps.duplicate_or_reversed_count, 0)
        self.assertEqual(analysis.report.timestamps.irregular_step_count, 0)
        self.assertEqual(
            int(np.count_nonzero(~np.isfinite(analysis.streamflow))), 92
        )

        issue_codes = {issue.code for issue in analysis.report.issues}
        self.assertIn("negative_streamflow_excluded", issue_codes)
        self.assertIn("missing_streamflow_excluded", issue_codes)
        self.assertFalse(
            any(issue.severity == "error" for issue in analysis.report.issues)
        )

    def test_pet_reproduces_hamon_formula(self) -> None:
        configuration = self.configuration()
        analysis = analyze_calibration_csv(
            configuration.data, workspace_root=PROJECT_ROOT
        )

        temperature = analysis.temperature
        day_length = analysis.additional_predictors["day_length"]
        expected_pet = (
            29.8
            * (day_length * 24.0)
            * 0.611
            * np.exp(17.3 * temperature / (temperature + 237.3))
            / (temperature + 273.2)
        )
        np.testing.assert_allclose(
            analysis.potential_evapotranspiration,
            expected_pet,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_partition_dates_and_counts(self) -> None:
        configuration = self.configuration()
        period = configuration.task.period
        analysis = analyze_calibration_csv(
            configuration.data, workspace_root=PROJECT_ROOT
        )
        timestamps = tuple(
            date.fromisoformat(value) for value in analysis.timestamps or ()
        )

        self.assertEqual(timestamps[period.start_index], date(1980, 10, 1))
        self.assertEqual(
            timestamps[period.start_index + period.warmup_steps],
            date(1981, 10, 1),
        )
        self.assertEqual(
            timestamps[period.calibration_end_index], date(1995, 10, 1)
        )
        self.assertEqual(
            timestamps[period.validation_end_index], date(2000, 10, 1)
        )
        self.assertEqual(timestamps[period.test_end_index], date(2014, 10, 1))

        summary = ExperimentRunner(PROJECT_ROOT).validate(configuration)
        self.assertEqual(summary.source_rows, 12_784)
        self.assertEqual(summary.selected_rows, 12_418)
        self.assertEqual(summary.warmup_steps, 365)
        self.assertEqual(
            summary.partition_scored_observations,
            {"calibration": 5_113, "validation": 1_827, "test": 5_113},
        )
        self.assertEqual(summary.primary_evaluation_partition, "test")

    def test_lstm_protocol_matches_the_frozen_scored_partitions(self) -> None:
        conceptual = self.configuration()
        neural = load_experiment_config(LSTM_CONFIG_PATH)

        self.assertEqual(
            neural.data.model_dump(mode="json"),
            conceptual.data.model_dump(mode="json"),
        )
        self.assertEqual(neural.model.model_id, "lstm.tensorflow_stacked_s2s.v3")
        self.assertEqual(
            neural.model.feature_roles,
            ("precipitation", "temperature", "potential_evapotranspiration"),
        )
        self.assertEqual(neural.model.lookback, 1095)
        self.assertEqual(neural.model.s2s_warmup, 365)
        self.assertEqual(neural.model.recurrent_layers, (16,))
        self.assertEqual(neural.model.output_activation, "softplus")
        self.assertEqual(neural.task.lead_steps, 0)
        self.assertEqual(
            neural.task.objective.objective_id,
            "nash_sutcliffe_efficiency",
        )
        self.assertFalse(neural.task.ensemble.enabled)
        self.assertEqual(neural.task.training.epochs, 300)
        self.assertEqual(neural.task.training.patience, 50)

        conceptual_period = conceptual.task.period
        neural_period = neural.task.period
        self.assertEqual(neural_period.start_index, conceptual_period.start_index)
        self.assertEqual(
            neural_period.train_end_index,
            conceptual_period.calibration_end_index,
        )
        self.assertEqual(
            neural_period.validation_end_index,
            conceptual_period.validation_end_index,
        )
        self.assertEqual(
            neural_period.test_end_index,
            conceptual_period.test_end_index,
        )
        self.assertEqual(
            neural_period.start_index + neural.model.s2s_warmup,
            conceptual_period.start_index + conceptual_period.warmup_steps,
        )

    def test_deepdiscover_protocol_is_source_aligned_and_partition_matched(self) -> None:
        conceptual = self.configuration()
        discovery = load_experiment_config(DEEPDISCOVER_CONFIG_PATH)

        self.assertEqual(
            discovery.data.model_dump(mode="json"),
            conceptual.data.model_dump(mode="json"),
        )
        self.assertEqual(
            discovery.model.model_id,
            "deepdiscover.numpy_instantaneous_s2s.v1",
        )
        self.assertEqual(discovery.model.hidden_units, 8)
        self.assertEqual(discovery.model.process_count, 1)
        self.assertFalse(discovery.model.train_input_switches)
        self.assertFalse(discovery.model.train_process_switches)
        self.assertEqual(discovery.model.initial_state, (0.0, 0.0))
        self.assertEqual(
            discovery.task.objective.objective_id,
            "nash_sutcliffe_efficiency",
        )
        self.assertEqual(discovery.task.warmup_steps, 365)
        self.assertEqual(discovery.task.training.epochs, 600)
        self.assertEqual(discovery.task.training.patience, 75)

        conceptual_period = conceptual.task.period
        discovery_period = discovery.task.period
        self.assertEqual(
            discovery_period.start_index, conceptual_period.start_index
        )
        self.assertEqual(
            discovery_period.train_end_index,
            conceptual_period.calibration_end_index,
        )
        self.assertEqual(
            discovery_period.validation_end_index,
            conceptual_period.validation_end_index,
        )
        self.assertEqual(
            discovery_period.test_end_index,
            conceptual_period.test_end_index,
        )
        self.assertEqual(
            discovery_period.start_index + discovery.task.warmup_steps,
            conceptual_period.start_index + conceptual_period.warmup_steps,
        )


if __name__ == "__main__":
    unittest.main()
