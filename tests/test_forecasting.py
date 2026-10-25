import csv
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from hydrosoftware.configuration import (
    CsvColumnMapping,
    CsvDataSourceConfig,
    ExperimentConfigV1,
)
from hydrosoftware.data_sources import analyze_calibration_csv
from hydrosoftware.execution.experiments import ExperimentRunner
from hydrosoftware.execution.forecasting import (
    configured_full_state,
    run_forecast,
    simulate_frozen_model,
)
from tests.test_experiment_execution import make_config


class ForecastingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        rng = np.random.default_rng(12)
        precipitation = rng.gamma(1.4, 2.0, size=60)
        temperature = np.full(60, 8.0)
        pet = rng.uniform(0.1, 1.5, size=60)

        from hydrosoftware.models.conceptual.exp_hydro import (
            ExpHydroModel,
            ExpHydroParameters,
        )
        from hydrosoftware.processes.snow import NoSnowModule

        model = ExpHydroModel(
            ExpHydroParameters(0.05, 500.0, 30.0), NoSnowModule()
        )
        streamflow = model.simulate(precipitation, temperature, pet).outputs[:, 0]
        with (self.root / "calibration.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["date", "precipitation", "temperature", "pet", "streamflow"]
            )
            for index, row in enumerate(
                zip(precipitation, temperature, pet, streamflow)
            ):
                writer.writerow([(date(2026, 1, 1) + timedelta(days=index)).isoformat(), *row])

        config = make_config(path="calibration.csv", evaluations=1)
        result = ExperimentRunner(self.root).run(config, task_id="source-run")
        self.config = config
        self.source_result = result
        self.archive = {
            "task_id": "source-run",
            "config": config.model_dump(mode="json"),
            "result": result.as_dict(),
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _forecast_analysis(self, *, with_observations: bool):
        rows = [
            ("2026-01-25", 3.0, 7.0, 0.5, 0.1),
            ("2026-01-26", 0.0, 7.0, 0.6, ""),
            ("2026-01-27", 5.0, 9.0, 0.7, 0.3),
        ]
        with (self.root / "forecast.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "p", "t", "pet", "q"])
            writer.writerows(rows)
        source = CsvDataSourceConfig(
            path="forecast.csv",
            columns=CsvColumnMapping(
                timestamp="date",
                precipitation="p",
                temperature="t",
                potential_evapotranspiration="pet",
                streamflow="q" if with_observations else None,
            ),
        )
        return analyze_calibration_csv(source, workspace_root=self.root)

    def test_saved_parameters_continue_from_the_exact_final_state_without_target(self):
        analysis = self._forecast_analysis(with_observations=False)
        result = run_forecast(
            self.archive,
            analysis,
            initialization="source_final",
            observations_available=False,
        )

        self.assertEqual(result["run_kind"], "forecast")
        self.assertIsNone(result["observations"])
        self.assertEqual(result["evaluation_metrics"], [])
        self.assertEqual(result["validation"]["scored_observations"], 0)
        np.testing.assert_allclose(
            result["best_simulation"]["states"][0],
            self.source_result.best_simulation.states[-1],
        )
        self.assertEqual(len(result["best_simulation"]["outputs"]), 3)

    def test_optional_streamflow_enables_metrics_without_recalibration(self):
        analysis = self._forecast_analysis(with_observations=True)
        result = run_forecast(
            self.archive,
            analysis,
            initialization="configured",
            observations_available=True,
        )

        self.assertEqual(result["validation"]["scored_observations"], 2)
        self.assertEqual(result["observations"], [0.1, None, 0.3])
        self.assertEqual(
            {metric["symbol"] for metric in result["evaluation_metrics"]},
            {"NSE", "KGE", "MSE", "RMSE", "NRMSE", "MAE", "PBIAS"},
        )
        self.assertEqual(
            result["calibration"]["records"],
            self.source_result.as_dict()["calibration"]["records"],
        )

    def test_daily_conceptual_models_restart_exactly_including_maxbas_memory(self):
        precipitation = np.array([2.0, 0.0, 5.0, 1.0, 0.0, 3.0, 4.0, 0.0])
        temperature = np.full(8, 8.0)
        pet = np.linspace(0.2, 0.8, 8)
        cases = (
            (
                {
                    "model_id": "exp_hydro.mass_conservative.v1",
                    "snow": {"module_id": "snow.none.v1"},
                    "initial_state": [0.0, 300.0],
                    "substeps": 4,
                },
                {
                    "recession_factor": 0.05,
                    "maximum_storage": 500.0,
                    "maximum_subsurface_flow": 30.0,
                },
            ),
            (
                {
                    "model_id": "hbv.mass_conservative_daily.v1",
                    "snow": {"module_id": "snow.none.v1"},
                    "routing": {
                        "process_id": "routing.maxbas.triangular.v1",
                        "maxbas": 3,
                        "initial_state": [0.0] * 6,
                    },
                    "initial_state": [0.0, 300.0, 20.0, 80.0],
                },
                {
                    "field_capacity": 500.0,
                    "recharge_exponent": 2.0,
                    "evap_reduction_threshold": 0.7,
                    "maximum_percolation": 1.5,
                    "upper_zone_threshold": 30.0,
                    "fast_recession": 0.3,
                    "intermediate_recession": 0.1,
                    "slow_recession": 0.02,
                    "maxbas": 3.0,
                },
            ),
        )
        for model_config, parameters in cases:
            with self.subTest(model=model_config["model_id"]):
                payload = self.config.model_dump(mode="json")
                payload["schema_version"] = "1.6"
                payload["model"] = model_config
                config = ExperimentConfigV1.model_validate(payload)
                initial = configured_full_state(config)
                full = simulate_frozen_model(
                    config,
                    parameters,
                    precipitation,
                    temperature,
                    pet,
                    initial_state=initial,
                )
                first = simulate_frozen_model(
                    config,
                    parameters,
                    precipitation[:5],
                    temperature[:5],
                    pet[:5],
                    initial_state=initial,
                )
                second = simulate_frozen_model(
                    config,
                    parameters,
                    precipitation[5:],
                    temperature[5:],
                    pet[5:],
                    initial_state=first.states[-1],
                )
                np.testing.assert_allclose(second.outputs, full.outputs[5:])
                np.testing.assert_allclose(second.states, full.states[5:])
                np.testing.assert_allclose(second.fluxes, full.fluxes[5:])


if __name__ == "__main__":
    unittest.main()
