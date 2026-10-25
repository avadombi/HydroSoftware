import csv
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hydrosoftware.configuration import (
    CsvColumnMapping,
    CsvDataSourceConfig,
    DeepDiscoverModelConfig,
    DiscoveryAdamTrainingConfig,
    DiscoveryLossConfig,
    ExperimentConfigV1,
    NoSnowConfig,
    SmoothDegreeDaySnowConfig,
)
from hydrosoftware.data_sources import analyze_calibration_csv
from hydrosoftware.execution.experiments import ExperimentRunner
from hydrosoftware.execution.forecasting import run_forecast
from hydrosoftware.execution.persistence import ProjectStore
from hydrosoftware.execution.tasks import TaskState
from hydrosoftware.models.deepdiscover_numpy import (
    DeepDiscoverArtifact,
    deepdiscover_loss_and_gradients,
    fit_deepdiscover,
    simulate_deepdiscover_from_artifact,
)
from hydrosoftware.processes.snow import NoSnowModule


def synthetic_series(length=120):
    rng = np.random.default_rng(2706)
    precipitation = rng.gamma(1.7, 2.2, size=length)
    temperature = rng.normal(7.0, 5.0, size=length)
    pet = rng.uniform(0.1, 2.1, size=length)
    streamflow = np.zeros(length, dtype=float)
    storage = 18.0
    for index in range(length):
        available = storage + precipitation[index]
        actual_et = 0.22 * min(pet[index], available)
        remaining = available - actual_et
        quick = 0.22 * remaining
        slow = 0.07 * remaining
        streamflow[index] = quick + slow
        storage = remaining - streamflow[index]
    return precipitation, temperature, pet, streamflow


def discovery_model(**updates):
    values = {
        "snow": NoSnowConfig(),
        "initial_state": (0.0, 18.0),
        "process_count": 2,
        "hidden_units": 5,
        "loss": DiscoveryLossConfig(
            decorrelation_weight=0.01,
            smoothness_weight=0.01,
            smoothness_threshold=0.05,
            binary_weight=0.005,
            sparsity_weight=0.001,
        ),
    }
    values.update(updates)
    return DeepDiscoverModelConfig(**values)


class NumpyDeepDiscoverTests(unittest.TestCase):
    def test_selected_smooth_snow_parameters_are_bounded_persisted_and_reused(self):
        precipitation, temperature, pet, streamflow = synthetic_series(36)
        model = discovery_model(
            snow=SmoothDegreeDaySnowConfig(
                snowfall_temperature=0.0,
                melt_temperature=0.0,
                degree_day_factor=2.0,
                sharpness=5.0,
                calibration={
                    "snowfall_temperature": {
                        "mode": "fixed",
                        "lower_bound": -3.0,
                        "upper_bound": 0.5,
                    },
                    "melt_temperature": {
                        "mode": "fixed",
                        "lower_bound": -1.0,
                        "upper_bound": 3.0,
                    },
                    "degree_day_factor": {
                        "mode": "calibrate",
                        "lower_bound": 0.5,
                        "upper_bound": 4.0,
                    },
                    "sharpness": {
                        "mode": "calibrate",
                        "lower_bound": 1.0,
                        "upper_bound": 10.0,
                    },
                },
            ),
            initial_state=(2.0, 18.0),
        )
        fit = fit_deepdiscover(
            model,
            DiscoveryAdamTrainingConfig(
                epochs=5,
                learning_rate=0.005,
                patience=5,
                seed=13,
            ),
            NoSnowModule(),
            precipitation,
            temperature,
            pet,
            streamflow,
            start_index=0,
            train_end_index=20,
            validation_end_index=28,
            test_end_index=36,
            warmup_steps=1,
        )
        artifact = DeepDiscoverArtifact.from_mapping(fit.artifact.as_mapping())
        self.assertEqual(
            artifact.snow_module_id,
            "snow.degree_day.smooth.v1",
        )
        self.assertEqual(
            artifact.calibrated_snow_parameters,
            ("degree_day_factor", "sharpness"),
        )
        self.assertEqual(
            artifact.snow_calibration_method,
            "bounded_alternating_coordinate.v1",
        )
        self.assertTrue(0.5 <= artifact.snow_parameters["degree_day_factor"] <= 4.0)
        self.assertTrue(1.0 <= artifact.snow_parameters["sharpness"] <= 10.0)
        frozen = simulate_deepdiscover_from_artifact(
            artifact,
            NoSnowModule(),
            precipitation,
            temperature,
            pet,
            initial_state=(2.0, 18.0),
        )
        np.testing.assert_allclose(frozen.outputs, fit.simulation.outputs)
        np.testing.assert_allclose(frozen.states, fit.simulation.states)

    def test_training_is_deterministic_leakage_safe_and_mass_conservative(self):
        precipitation, temperature, pet, streamflow = synthetic_series()
        settings = DiscoveryAdamTrainingConfig(
            epochs=45,
            learning_rate=0.01,
            patience=15,
            gradient_clip_norm=5.0,
            seed=42,
        )
        kwargs = {
            "start_index": 0,
            "train_end_index": 70,
            "validation_end_index": 95,
            "test_end_index": 120,
            "warmup_steps": 3,
        }
        first = fit_deepdiscover(
            discovery_model(),
            settings,
            NoSnowModule(),
            precipitation,
            temperature,
            pet,
            streamflow,
            **kwargs,
        )
        second = fit_deepdiscover(
            discovery_model(),
            settings,
            NoSnowModule(),
            precipitation,
            temperature,
            pet,
            streamflow,
            **kwargs,
        )

        np.testing.assert_allclose(first.simulation.outputs, second.simulation.outputs)
        self.assertEqual(first.artifact.best_epoch, second.artifact.best_epoch)
        np.testing.assert_allclose(
            first.artifact.input_mean,
            np.column_stack((precipitation, np.zeros(120), pet))[:70].mean(axis=0),
        )
        self.assertLessEqual(
            min(item["validation_rmse"] for item in first.history),
            first.history[0]["validation_rmse"],
        )
        self.assertLess(
            first.simulation.diagnostics["maximum_absolute_mass_balance_error"],
            1e-12,
        )
        process_start = first.simulation.flux_names.index("discovered_process_1")
        np.testing.assert_allclose(
            first.simulation.outputs[:, 0],
            first.simulation.fluxes[:, process_start:].sum(axis=1),
        )
        total_state = first.simulation.states.sum(axis=1)
        actual_et = first.simulation.fluxes[
            :, first.simulation.flux_names.index("actual_evapotranspiration")
        ]
        np.testing.assert_allclose(
            total_state[:-1] + precipitation
            - actual_et
            - first.simulation.outputs[:, 0]
            - total_state[1:],
            0.0,
            atol=1e-12,
        )

    def test_artifact_round_trip_and_storage_restart_are_exact(self):
        precipitation, temperature, pet, streamflow = synthetic_series()
        fit = fit_deepdiscover(
            discovery_model(),
            DiscoveryAdamTrainingConfig(
                epochs=25,
                learning_rate=0.01,
                patience=10,
                seed=7,
            ),
            NoSnowModule(),
            precipitation,
            temperature,
            pet,
            streamflow,
            start_index=0,
            train_end_index=70,
            validation_end_index=95,
            test_end_index=120,
            warmup_steps=2,
        )
        restored = DeepDiscoverArtifact.from_mapping(fit.artifact.as_mapping())
        full = simulate_deepdiscover_from_artifact(
            restored,
            NoSnowModule(),
            precipitation,
            temperature,
            pet,
            initial_state=(0.0, 18.0),
        )
        split = 73
        first = simulate_deepdiscover_from_artifact(
            restored,
            NoSnowModule(),
            precipitation[:split],
            temperature[:split],
            pet[:split],
            initial_state=(0.0, 18.0),
        )
        second = simulate_deepdiscover_from_artifact(
            restored,
            NoSnowModule(),
            precipitation[split:],
            temperature[split:],
            pet[split:],
            initial_state=tuple(first.states[-1]),
        )

        np.testing.assert_allclose(
            np.vstack((first.outputs, second.outputs)), full.outputs
        )
        np.testing.assert_allclose(
            np.vstack((first.fluxes, second.fluxes)), full.fluxes
        )
        np.testing.assert_allclose(
            np.vstack((first.states, second.states[1:])), full.states
        )

    def test_full_sequence_gradients_match_finite_differences(self):
        precipitation, temperature, pet, streamflow = synthetic_series(18)
        model = discovery_model(hidden_units=3)
        fit = fit_deepdiscover(
            model,
            DiscoveryAdamTrainingConfig(
                epochs=2,
                learning_rate=0.005,
                patience=2,
                seed=11,
            ),
            NoSnowModule(),
            precipitation,
            temperature,
            pet,
            streamflow,
            start_index=0,
            train_end_index=10,
            validation_end_index=14,
            test_end_index=18,
            warmup_steps=1,
        )
        artifact = DeepDiscoverArtifact.from_mapping(fit.artifact.as_mapping())
        data_mask = np.ones(18, dtype=bool)
        regularization_mask = np.ones(18, dtype=bool)

        def loss_and_gradients():
            return deepdiscover_loss_and_gradients(
                model,
                artifact,
                NoSnowModule(),
                precipitation,
                temperature,
                pet,
                streamflow,
                data_mask=data_mask,
                regularization_mask=regularization_mask,
                initial_state=(0.0, 18.0),
            )

        _, analytic = loss_and_gradients()
        epsilon = 1e-6
        for name, coordinate in (
            ("hidden_kernel", (3, 1)),
            ("process_bias", (0,)),
            ("input_switch_logits", (1,)),
            ("process_switch_logits", (0,)),
        ):
            parameter = getattr(artifact, name)
            original = parameter[coordinate]
            parameter[coordinate] = original + epsilon
            upper = loss_and_gradients()[0]["total_loss"]
            parameter[coordinate] = original - epsilon
            lower = loss_and_gradients()[0]["total_loss"]
            parameter[coordinate] = original
            numerical = (upper - lower) / (2.0 * epsilon)
            self.assertAlmostEqual(
                analytic[name][coordinate],
                numerical,
                places=5,
                msg=f"Gradient mismatch for {name}{coordinate}",
            )

    def test_runner_persists_and_forecasts_without_retraining(self):
        precipitation, temperature, pet, streamflow = synthetic_series(90)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "discovery.csv"
            with source_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "p", "t", "pet", "q"])
                for index, values in enumerate(
                    zip(precipitation, temperature, pet, streamflow)
                ):
                    writer.writerow(
                        [
                            (date(2026, 1, 1) + timedelta(days=index)).isoformat(),
                            *values,
                        ]
                    )
            config = ExperimentConfigV1.model_validate(
                {
                    "schema_version": "1.12",
                    "experiment_id": "synthetic-deepdiscover",
                    "name": "Synthetic DeepDiscover",
                    "data": {
                        "path": "discovery.csv",
                        "columns": {
                            "timestamp": "date",
                            "precipitation": "p",
                            "temperature": "t",
                            "potential_evapotranspiration": "pet",
                            "streamflow": "q",
                        },
                    },
                    "model": {
                        "model_id": "deepdiscover.numpy_instantaneous_s2s.v1",
                        "snow": {"module_id": "snow.none.v1"},
                        "initial_state": [0.0, 18.0],
                        "process_count": 2,
                        "hidden_units": 4,
                    },
                    "task": {
                        "task_type": "discovery_training",
                        "period": {
                            "start_index": 0,
                            "train_end_index": 50,
                            "validation_end_index": 70,
                            "test_end_index": 90,
                        },
                        "warmup_steps": 2,
                        "objective": {
                            "objective_id": "mean_absolute_error",
                        },
                        "training": {
                            "epochs": 12,
                            "learning_rate": 0.01,
                            "patience": 5,
                            "gradient_clip_norm": 5.0,
                            "seed": 9,
                        },
                    },
                }
            )
            result = ExperimentRunner(root).run(config, task_id="run-deepdiscover")
            self.assertEqual(result.state, TaskState.COMPLETED)
            self.assertEqual(result.training["process_dynamics"], "instantaneous")
            self.assertEqual(result.training["process_count"], 2)
            self.assertEqual(
                result.training["objective"]["objective_id"],
                "mean_absolute_error",
            )
            self.assertIn(
                "analysis__raw_process_1",
                result.best_simulation.diagnostics,
            )
            store = ProjectStore(root)
            store.save_run(config, result)
            persisted = store.load_run("run-deepdiscover")
            self.assertEqual(
                persisted["result"]["training"]["artifact"]["artifact_type"],
                "hydrosoftware.deepdiscover.numpy.instantaneous.v1",
            )

            forecast_path = root / "forecast.csv"
            with forecast_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["p", "t", "pet"])
                writer.writerows(
                    zip(precipitation[:8], temperature[:8], pet[:8])
                )
            analysis = analyze_calibration_csv(
                CsvDataSourceConfig(
                    path="forecast.csv",
                    columns=CsvColumnMapping(
                        precipitation="p",
                        temperature="t",
                        potential_evapotranspiration="pet",
                    ),
                ),
                workspace_root=root,
            )
            forecast = run_forecast(
                {
                    "task_id": "run-deepdiscover",
                    "config": config.model_dump(mode="json"),
                    "result": result.as_dict(),
                },
                analysis,
                initialization="source_final",
                observations_available=False,
            )
            self.assertEqual(forecast["run_kind"], "forecast")
            self.assertEqual(
                forecast["initial_state"],
                result.best_simulation.states[-1].tolist(),
            )
            self.assertIsNone(forecast["observations"])
            self.assertEqual(
                len(forecast["best_simulation"]["outputs"]), 8
            )


if __name__ == "__main__":
    unittest.main()
