import csv
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import tempfile
from threading import Event
import unittest

import numpy as np

from hydrosoftware.configuration import ExperimentConfigV1
from hydrosoftware.execution import CancellationToken, TaskState
from hydrosoftware.execution.experiments import ExperimentRunner, _TrainingProgressClock
from hydrosoftware.execution.manager import (
    ExperimentTaskManager,
    ResultNotReadyError,
    TaskSnapshot,
)
from hydrosoftware.execution.tasks import ProgressEvent
from hydrosoftware.models.conceptual.exp_hydro import (
    ExpHydroModel,
    ExpHydroParameters,
)
from hydrosoftware.models.conceptual.hbv import HbvModel, HbvParameters
from hydrosoftware.processes.routing import (
    TriangularMaxbasRouting,
    route_simulation_result,
)
from hydrosoftware.processes.snow import DegreeDaySnowModule, NoSnowModule


def make_config(path="basin.csv", evaluations=5):
    return ExperimentConfigV1.model_validate(
        {
            "schema_version": "1.0",
            "experiment_id": "synthetic-exp-hydro",
            "name": "Synthetic EXP-HYDRO calibration",
            "data": {
                "path": path,
                "columns": {
                    "timestamp": "date",
                    "precipitation": "precipitation",
                    "temperature": "temperature",
                    "potential_evapotranspiration": "pet",
                    "streamflow": "streamflow",
                },
            },
            "model": {
                "snow": {"module_id": "snow.none.v1"},
                "initial_state": [0.0, 0.0],
                "substeps": 4,
            },
            "task": {
                "period": {"start_index": 0, "warmup_steps": 5},
                "objective": {"objective_id": "nash_sutcliffe_efficiency"},
                "search": {
                    "n_evaluations": evaluations,
                    "seed": 9,
                    "include_defaults": True,
                },
            },
        }
    )


class ExperimentExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        rng = np.random.default_rng(12)
        precipitation = rng.gamma(1.4, 2.0, size=60)
        temperature = np.full(60, 8.0)
        pet = rng.uniform(0.1, 1.5, size=60)
        model = ExpHydroModel(
            ExpHydroParameters(0.05, 500.0, 30.0),
            NoSnowModule(),
        )
        streamflow = model.simulate(precipitation, temperature, pet).outputs[:, 0]
        with (self.root / "basin.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["date", "precipitation", "temperature", "pet", "streamflow"]
            )
            for index, values in enumerate(
                zip(precipitation, temperature, pet, streamflow), start=1
            ):
                writer.writerow([f"2020-01-{index:02d}" if index <= 31 else f"2020-02-{index - 31:02d}", *values])

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_training_progress_clock_reports_elapsed_and_remaining_time(self):
        timestamps = iter((100.0, 110.0, 130.0, 135.0))
        clock = _TrainingProgressClock(lambda: next(timestamps))

        self.assertEqual(clock.estimate(1, 4), (10.0, 30.0))
        self.assertEqual(clock.estimate(2, 4), (30.0, 30.0))
        self.assertEqual(clock.finished(), (35.0, 0.0))

    def test_runner_validates_and_reproduces_default_parameters(self):
        runner = ExperimentRunner(self.root)
        config = make_config()
        summary = runner.validate(config)
        events = []
        result = runner.run(config, task_id="run-direct", on_progress=events.append)

        self.assertEqual(summary.source_rows, 60)
        self.assertEqual(summary.scored_observations, 55)
        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertAlmostEqual(result.calibration.best_score, 1.0, places=12)
        metrics = {metric.symbol: metric for metric in result.evaluation_metrics}
        self.assertEqual(
            set(metrics),
            {"NSE", "KGE", "MSE", "RMSE", "NRMSE", "MAE", "PBIAS"},
        )
        self.assertAlmostEqual(metrics["NSE"].value, result.calibration.best_score)
        self.assertAlmostEqual(metrics["KGE"].value, 1.0, places=12)
        self.assertEqual(result.best_simulation.outputs.shape, (60, 1))
        self.assertEqual(events[0].stage, "data_loading")
        self.assertTrue(
            any(
                warning.code == "legacy_calibration_only_protocol"
                for warning in summary.warnings
            )
        )

    def test_conceptual_holdouts_are_scored_separately_without_parameter_leakage(self):
        with (self.root / "basin.csv").open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        with (self.root / "altered.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            for index, row in enumerate(rows):
                changed = dict(row)
                if index >= 30:
                    changed["streamflow"] = str(
                        float(changed["streamflow"]) + 20.0
                    )
                writer.writerow(changed)

        def holdout_config(path: str) -> ExperimentConfigV1:
            payload = make_config(path=path, evaluations=8).model_dump(
                mode="json"
            )
            payload["schema_version"] = "1.18"
            payload["task"]["period"] = {
                "protocol": "calibration_validation_test",
                "start_index": 0,
                "calibration_end_index": 30,
                "validation_end_index": 45,
                "test_end_index": 60,
                "warmup_steps": 5,
            }
            payload["task"]["objective"] = {
                "objective_id": "mean_absolute_error"
            }
            return ExperimentConfigV1.model_validate(payload)

        runner = ExperimentRunner(self.root)
        reference_config = holdout_config("basin.csv")
        altered_config = holdout_config("altered.csv")
        reference_summary = runner.validate(reference_config)
        reference = runner.run(reference_config, task_id="holdout-reference")
        altered = runner.run(altered_config, task_id="holdout-altered")

        self.assertEqual(
            reference_summary.primary_evaluation_partition,
            "test",
        )
        self.assertEqual(
            reference_summary.partition_scored_observations,
            {"calibration": 25, "validation": 15, "test": 15},
        )
        self.assertEqual(
            set(reference.partition_metrics),
            {"calibration", "validation", "test"},
        )
        self.assertEqual(
            reference.calibration.objective_name,
            "mean_absolute_error",
        )
        self.assertEqual(
            reference.calibration.best_parameters,
            altered.calibration.best_parameters,
        )
        reference_test = {
            metric.symbol: metric.value
            for metric in reference.partition_metrics["test"]
        }
        altered_test = {
            metric.symbol: metric.value
            for metric in altered.partition_metrics["test"]
        }
        self.assertAlmostEqual(reference_test["MAE"], 0.0, places=12)
        self.assertGreater(altered_test["MAE"], 19.0)
        self.assertEqual(
            tuple(reference.evaluation_metrics),
            tuple(reference.partition_metrics["test"]),
        )

    def test_runner_executes_differential_evolution_and_archives_generations(self):
        payload = make_config(evaluations=1).model_dump(mode="json")
        payload["schema_version"] = "1.16"
        payload["task"]["search"] = {
            "algorithm_id": "differential_evolution.rand1bin.v1",
            "generations": 2,
            "population_size": 8,
            "mutation_factor": 0.8,
            "crossover_probability": 0.9,
            "seed": 14,
            "include_defaults": True,
            "continue_on_candidate_error": True,
        }
        result = ExperimentRunner(self.root).run(
            ExperimentConfigV1.model_validate(payload),
            task_id="run-differential-evolution",
        )

        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertEqual(
            result.calibration.algorithm_id,
            "differential_evolution.rand1bin.v1",
        )
        self.assertEqual(result.calibration.generations_completed, 2)
        self.assertEqual(len(result.calibration.records), 24)
        self.assertEqual(result.calibration.records[0].generation, 0)
        self.assertEqual(result.calibration.records[-1].generation, 2)
        self.assertAlmostEqual(result.calibration.best_score, 1.0, places=12)


    def test_runner_calibrates_daily_hbv_and_returns_four_states(self):
        with (self.root / "basin.csv").open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        precipitation = np.array([float(row["precipitation"]) for row in rows])
        temperature = np.array([float(row["temperature"]) for row in rows])
        pet = np.array([float(row["pet"]) for row in rows])
        parameters = HbvParameters(
            field_capacity=500.0,
            recharge_exponent=2.0,
            evap_reduction_threshold=0.7,
            maximum_percolation=1.5,
            upper_zone_threshold=30.0,
            fast_recession=0.30,
            intermediate_recession=0.10,
            slow_recession=0.02,
        )
        streamflow = HbvModel(parameters, NoSnowModule()).simulate(
            precipitation,
            temperature,
            pet,
            initial_state=(0.0, 300.0, 20.0, 80.0),
        ).outputs[:, 0]
        with (self.root / "hbv.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["date", "precipitation", "temperature", "pet", "streamflow"]
            )
            for row, flow in zip(rows, streamflow):
                writer.writerow(
                    [
                        row["date"],
                        row["precipitation"],
                        row["temperature"],
                        row["pet"],
                        flow,
                    ]
                )

        payload = make_config(path="hbv.csv", evaluations=1).model_dump(mode="json")
        payload["schema_version"] = "1.3"
        payload["experiment_id"] = "synthetic-hbv"
        payload["model"] = {
            "model_id": "hbv.mass_conservative_daily.v1",
            "snow": {"module_id": "snow.none.v1"},
            "initial_state": [0.0, 300.0, 20.0, 80.0],
        }
        result = ExperimentRunner(self.root).run(
            ExperimentConfigV1.model_validate(payload), task_id="run-hbv"
        )
        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertAlmostEqual(result.calibration.best_score, 1.0, places=12)
        self.assertEqual(result.best_simulation.states.shape, (61, 4))
        self.assertIn("soil_excess", result.best_simulation.flux_names)
        self.assertIn("percolation", result.best_simulation.flux_names)
        self.assertIn("baseflow", result.best_simulation.flux_names)


    def test_selected_snow_parameter_is_part_of_the_real_calibration_vector(self):
        rng = np.random.default_rng(32)
        precipitation = rng.gamma(1.6, 2.5, size=60)
        temperature = np.linspace(-6.0, 6.0, 60)
        pet = rng.uniform(0.1, 1.0, size=60)
        model = ExpHydroModel(
            ExpHydroParameters(0.05, 500.0, 30.0),
            DegreeDaySnowModule(0.0, 0.0, 2.0),
        )
        streamflow = model.simulate(precipitation, temperature, pet).outputs[:, 0]
        with (self.root / "snow.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "precipitation", "temperature", "pet", "streamflow"])
            for index, values in enumerate(zip(precipitation, temperature, pet, streamflow)):
                writer.writerow([(date(2021, 1, 1) + timedelta(days=index)).isoformat(), *values])

        payload = make_config(path="snow.csv", evaluations=3).model_dump(mode="json")
        payload["schema_version"] = "1.2"
        payload["model"]["snow"] = {
            "module_id": "snow.degree_day.hard.v1",
            "snowfall_temperature": 0.0,
            "melt_temperature": 0.0,
            "degree_day_factor": 2.0,
            "calibration": {
                "degree_day_factor": {
                    "mode": "calibrate",
                    "lower_bound": 0.0,
                    "upper_bound": 5.0,
                }
            },
        }
        result = ExperimentRunner(self.root).run(
            ExperimentConfigV1.model_validate(payload), task_id="run-snow-calibration"
        )
        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertAlmostEqual(result.calibration.best_score, 1.0, places=12)
        self.assertEqual(result.calibration.best_parameters["degree_day_factor"], 2.0)
        self.assertEqual(len(result.calibration.best_parameters), 4)
        self.assertTrue(any(
            record.parameters["degree_day_factor"] != 2.0
            and record.score != result.calibration.best_score
            for record in result.calibration.records[1:]
        ))

    def test_runner_calibrates_discrete_maxbas_and_returns_routing_memory(self):
        with (self.root / "basin.csv").open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        precipitation = np.array([float(row["precipitation"]) for row in rows])
        temperature = np.array([float(row["temperature"]) for row in rows])
        pet = np.array([float(row["pet"]) for row in rows])
        parameters = HbvParameters(500.0, 2.0, 0.7, 1.5, 30.0, 0.30, 0.10, 0.02)
        base = HbvModel(parameters, NoSnowModule()).simulate(
            precipitation,
            temperature,
            pet,
            initial_state=(0.0, 300.0, 20.0, 80.0),
        )
        streamflow = route_simulation_result(
            base, TriangularMaxbasRouting(3)
        ).outputs[:, 0]
        with (self.root / "hbv-routed.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["date", "precipitation", "temperature", "pet", "streamflow"]
            )
            for row, flow in zip(rows, streamflow):
                writer.writerow(
                    [
                        row["date"],
                        row["precipitation"],
                        row["temperature"],
                        row["pet"],
                        flow,
                    ]
                )

        payload = make_config(
            path="hbv-routed.csv", evaluations=12
        ).model_dump(mode="json")
        payload["schema_version"] = "1.4"
        payload["experiment_id"] = "synthetic-hbv-routed"
        payload["model"] = {
            "model_id": "hbv.mass_conservative_daily.v1",
            "snow": {"module_id": "snow.none.v1"},
            "initial_state": [0.0, 300.0, 20.0, 80.0],
            "routing": {
                "process_id": "routing.maxbas.triangular.v1",
                "maxbas": 3,
                "calibration": {
                    "maxbas": {
                        "mode": "calibrate",
                        "lower_bound": 1,
                        "upper_bound": 7,
                    }
                },
                "initial_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        }
        result = ExperimentRunner(self.root).run(
            ExperimentConfigV1.model_validate(payload), task_id="run-hbv-routed"
        )
        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertAlmostEqual(result.calibration.best_score, 1.0, places=12)
        self.assertEqual(result.calibration.best_parameters["maxbas"], 3.0)
        self.assertTrue(all(
            record.parameters["maxbas"].is_integer()
            for record in result.calibration.records
            if record.succeeded
        ))
        self.assertEqual(result.best_simulation.states.shape, (61, 10))
        self.assertIn("routed_streamflow", result.best_simulation.flux_names)

    def test_task_manager_preserves_ordered_progress_and_result(self):
        manager = ExperimentTaskManager(ExperimentRunner(self.root), max_workers=1)
        try:
            submitted = manager.submit(make_config(evaluations=4))
            terminal = manager.wait(submitted.task_id, timeout=10.0)
            events = manager.events(submitted.task_id)
            result = manager.result(submitted.task_id)
        finally:
            manager.shutdown()

        self.assertEqual(submitted.state, TaskState.QUEUED)
        self.assertEqual(terminal.state, TaskState.COMPLETED)
        self.assertEqual(
            [event.sequence for event in events], list(range(len(events)))
        )
        self.assertEqual(events[0].state, TaskState.QUEUED)
        self.assertEqual(events[-1].state, TaskState.COMPLETED)
        self.assertEqual(len(result.configuration_hash), 64)

    def test_completed_snapshot_is_terminal_only_when_result_is_ready(self):
        event = ProgressEvent(
            task_id="race-check",
            sequence=0,
            state=TaskState.COMPLETED,
            stage="finalization",
            message="Calibration completed successfully.",
        )
        snapshot = TaskSnapshot(
            task_id="race-check",
            experiment_id="experiment",
            state=TaskState.COMPLETED,
            created_at="2026-07-21T00:00:00+00:00",
            updated_at="2026-07-21T00:00:01+00:00",
            event_count=1,
            latest_event=event,
            has_result=False,
        )

        self.assertFalse(snapshot.terminal)
        self.assertTrue(replace(snapshot, has_result=True).terminal)

    def test_terminal_snapshot_waits_for_persistent_result_observer(self):
        archive_started = Event()
        archive_release = Event()

        def archive(_config, _result):
            archive_started.set()
            archive_release.wait(timeout=2.0)

        manager = ExperimentTaskManager(
            ExperimentRunner(self.root),
            max_workers=1,
            on_result=archive,
        )
        try:
            submitted = manager.submit(make_config(evaluations=1))
            self.assertTrue(archive_started.wait(timeout=2.0))
            during_archive = manager.snapshot(submitted.task_id)
            self.assertEqual(during_archive.state, TaskState.COMPLETED)
            self.assertFalse(during_archive.has_result)
            self.assertFalse(during_archive.terminal)

            archive_release.set()
            finished = manager.wait(submitted.task_id, timeout=2.0)
            self.assertTrue(finished.has_result)
            self.assertTrue(finished.terminal)
        finally:
            archive_release.set()
            manager.shutdown()

    def test_pre_requested_task_cancels_before_reading_data(self):
        runner = ExperimentRunner(self.root)
        token = CancellationToken()
        token.request()
        result = runner.run(
            make_config(path="missing.csv"),
            task_id="cancelled-before-start",
            cancellation_token=token,
        )

        self.assertEqual(result.state, TaskState.CANCELLED)
        self.assertIsNone(result.validation)

    def test_missing_file_is_contained_as_a_failed_task(self):
        manager = ExperimentTaskManager(ExperimentRunner(self.root), max_workers=1)
        try:
            submitted = manager.submit(make_config(path="missing.csv"))
            terminal = manager.wait(submitted.task_id, timeout=10.0)
            with self.assertRaises(ResultNotReadyError):
                manager.result(submitted.task_id)
        finally:
            manager.shutdown()

        self.assertEqual(terminal.state, TaskState.FAILED)
        self.assertEqual(terminal.error.code, "data_file_not_found")


if __name__ == "__main__":
    unittest.main()
