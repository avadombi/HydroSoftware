import csv
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
import importlib.util

import numpy as np

from hydrosoftware.configuration import (
    CsvColumnMapping,
    CsvDataSourceConfig,
    ExperimentConfigV1,
    load_experiment_config,
)
from hydrosoftware.data_sources import analyze_calibration_csv
from hydrosoftware.execution.experiments import ExperimentRunner
from hydrosoftware.execution.forecast_ensembles import (
    empirical_crps,
    run_forcing_ensemble_forecast,
)
from hydrosoftware.execution.forecasting import run_forecast
from tests.test_experiment_execution import make_config


class MeteorologicalForecastEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        rng = np.random.default_rng(301)
        precipitation = rng.gamma(1.6, 2.2, 60)
        temperature = np.full(60, 7.0)
        pet = rng.uniform(0.1, 1.4, 60)

        from hydrosoftware.models.conceptual.exp_hydro import (
            ExpHydroModel,
            ExpHydroParameters,
        )
        from hydrosoftware.processes.snow import NoSnowModule

        model = ExpHydroModel(
            ExpHydroParameters(0.05, 500.0, 30.0),
            NoSnowModule(),
        )
        streamflow = model.simulate(precipitation, temperature, pet).outputs[:, 0]
        self._write_csv(
            "calibration.csv",
            precipitation,
            temperature,
            pet,
            streamflow,
            start=date(2026, 1, 1),
        )
        config_payload = make_config(
            path="calibration.csv", evaluations=1
        ).model_dump(mode="json")
        config_payload["data"]["columns"] = {
            "timestamp": "date",
            "precipitation": "p",
            "temperature": "t",
            "potential_evapotranspiration": "pet",
            "streamflow": "q",
        }
        from hydrosoftware.configuration import ExperimentConfigV1

        config = ExperimentConfigV1.model_validate(config_payload)
        result = ExperimentRunner(self.root).run(config, task_id="source-run")
        self.archive = {
            "task_id": "source-run",
            "config": config.model_dump(mode="json"),
            "result": result.as_dict(),
        }

    def tearDown(self):
        self.directory.cleanup()

    def _write_csv(
        self,
        filename,
        precipitation,
        temperature,
        pet,
        streamflow,
        *,
        start,
    ):
        with (self.root / filename).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "p", "t", "pet", "q"])
            for index, values in enumerate(
                zip(precipitation, temperature, pet, streamflow)
            ):
                writer.writerow(
                    [
                        (start + timedelta(days=index)).isoformat(),
                        *values,
                    ]
                )

    def _analysis(self, filename, *, with_observations):
        return analyze_calibration_csv(
            CsvDataSourceConfig(
                path=filename,
                columns=CsvColumnMapping(
                    timestamp="date",
                    precipitation="p",
                    temperature="t",
                    potential_evapotranspiration="pet",
                    streamflow="q" if with_observations else None,
                ),
            ),
            workspace_root=self.root,
        )

    def test_empirical_crps_uses_member_pairwise_dispersion(self):
        self.assertAlmostEqual(empirical_crps([0.0, 2.0], 1.0), 0.5)
        self.assertAlmostEqual(empirical_crps([1.0, 1.0, 1.0], 1.0), 0.0)
        with self.assertRaisesRegex(ValueError, "at least two"):
            empirical_crps([1.0], 1.0)

    def test_frozen_model_mean_quantiles_and_scores_are_exact(self):
        temperature = np.full(5, 6.0)
        pet = np.full(5, 0.5)
        observations = np.array([0.2, 0.3, 0.25, 0.4, 0.35])
        precipitation_members = (
            np.array([0.0, 2.0, 1.0, 4.0, 0.0]),
            np.array([1.0, 3.0, 0.0, 2.0, 1.0]),
            np.array([2.0, 1.0, 3.0, 1.0, 2.0]),
        )
        member_inputs = []
        deterministic = []
        for index, precipitation in enumerate(precipitation_members, start=1):
            filename = f"member-{index}.csv"
            self._write_csv(
                filename,
                precipitation,
                temperature,
                pet,
                observations,
                start=date(2026, 3, 2),
            )
            analysis = self._analysis(
                filename,
                with_observations=index == 1,
            )
            member_inputs.append((f"member-{index}", analysis, index == 1))
            deterministic.append(
                run_forecast(
                    self.archive,
                    analysis,
                    initialization="source_final",
                    observations_available=index == 1,
                )
            )

        ensemble = run_forcing_ensemble_forecast(
            self.archive,
            member_inputs,
            initialization="source_final",
        )
        member_predictions = np.stack(
            [
                np.asarray(result["best_simulation"]["outputs"], dtype=float)[:, 0]
                for result in deterministic
            ]
        )
        np.testing.assert_allclose(
            np.asarray(ensemble["best_simulation"]["outputs"])[:, 0],
            np.mean(member_predictions, axis=0),
        )
        diagnostics = ensemble["forecast_ensemble"]
        self.assertEqual(diagnostics["member_count"], 3)
        self.assertEqual(
            diagnostics["member_ids"],
            ["member-1", "member-2", "member-3"],
        )
        self.assertEqual(diagnostics["aggregate_scores"]["scored_leads"], 5)
        for lead in diagnostics["lead_time_diagnostics"]:
            index = lead["lead_step"] - 1
            self.assertAlmostEqual(
                lead["crps"],
                empirical_crps(member_predictions[:, index], observations[index]),
            )
            self.assertAlmostEqual(
                lead["signed_error"],
                float(np.mean(member_predictions[:, index]) - observations[index]),
            )
        self.assertEqual(
            {metric["symbol"] for metric in ensemble["evaluation_metrics"]},
            {"NSE", "KGE", "MSE", "RMSE", "NRMSE", "MAE", "PBIAS"},
        )
        self.assertEqual(
            ensemble["calibration"]["records"],
            self.archive["result"]["calibration"]["records"],
        )

    def test_members_must_share_horizon_and_timestamps(self):
        temperature = np.full(3, 6.0)
        pet = np.full(3, 0.5)
        observations = np.full(3, 0.2)
        self._write_csv(
            "aligned.csv",
            np.array([1.0, 2.0, 3.0]),
            temperature,
            pet,
            observations,
            start=date(2026, 4, 1),
        )
        self._write_csv(
            "shifted.csv",
            np.array([1.5, 2.5, 3.5]),
            temperature,
            pet,
            observations,
            start=date(2026, 4, 2),
        )
        with self.assertRaisesRegex(ValueError, "identical forecast timestamps"):
            run_forcing_ensemble_forecast(
                self.archive,
                [
                    ("aligned", self._analysis("aligned.csv", with_observations=True), True),
                    ("shifted", self._analysis("shifted.csv", with_observations=False), False),
                ],
                initialization="source_final",
            )

    def test_shared_observed_spinup_updates_state_and_is_not_scored(self):
        temperature = np.full(5, 6.0)
        pet = np.full(5, 0.5)
        observations = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        member_inputs = []
        for index, precipitation in enumerate(
            (
                np.array([3.0, 1.0, 0.0, 2.0, 4.0]),
                np.array([3.0, 1.0, 1.0, 3.0, 2.0]),
            ),
            start=1,
        ):
            filename = f"spinup-{index}.csv"
            self._write_csv(
                filename,
                precipitation,
                temperature,
                pet,
                observations,
                start=date(2026, 6, 1),
            )
            member_inputs.append(
                (
                    f"member-{index}",
                    self._analysis(filename, with_observations=True),
                    True,
                )
            )
        ensemble = run_forcing_ensemble_forecast(
            self.archive,
            member_inputs,
            initialization="configured",
            spinup_steps=2,
        )
        self.assertEqual(len(ensemble["best_simulation"]["outputs"]), 3)
        self.assertEqual(len(ensemble["best_simulation"]["states"]), 4)
        self.assertEqual(ensemble["timestamps"][0], "2026-06-03")
        self.assertEqual(ensemble["observations"], [0.3, 0.4, 0.5])
        self.assertEqual(
            ensemble["forecast_ensemble"]["spinup_steps"],
            2,
        )
        self.assertEqual(
            ensemble["forecast_ensemble"]["aggregate_scores"]["scored_leads"],
            3,
        )

        changed = list(member_inputs)
        changed_analysis = changed[1][1]
        changed_analysis.precipitation[0] += 1.0
        with self.assertRaisesRegex(ValueError, "identical observed spin-up"):
            run_forcing_ensemble_forecast(
                self.archive,
                changed,
                initialization="configured",
                spinup_steps=2,
            )

    def test_observations_cannot_change_between_member_files(self):
        temperature = np.full(3, 6.0)
        pet = np.full(3, 0.5)
        self._write_csv(
            "observed-a.csv",
            np.array([1.0, 2.0, 3.0]),
            temperature,
            pet,
            np.array([0.2, 0.3, 0.4]),
            start=date(2026, 5, 1),
        )
        self._write_csv(
            "observed-b.csv",
            np.array([1.5, 2.5, 3.5]),
            temperature,
            pet,
            np.array([0.2, 9.0, 0.4]),
            start=date(2026, 5, 1),
        )
        with self.assertRaisesRegex(ValueError, "Observed streamflow differs"):
            run_forcing_ensemble_forecast(
                self.archive,
                [
                    ("member-a", self._analysis("observed-a.csv", with_observations=True), True),
                    ("member-b", self._analysis("observed-b.csv", with_observations=True), True),
                ],
                initialization="source_final",
            )

    def test_memberwise_cycle_matches_continuous_member_simulations(self):
        temperature_first = np.full(3, 6.0)
        temperature_second = np.full(4, 7.0)
        pet_first = np.full(3, 0.5)
        pet_second = np.full(4, 0.7)
        observations_first = np.full(3, 0.2)
        observations_second = np.full(4, 0.3)
        precipitation = {
            "member-a": (
                np.array([0.0, 5.0, 1.0]),
                np.array([2.0, 0.0, 4.0, 1.0]),
            ),
            "member-b": (
                np.array([3.0, 0.0, 2.0]),
                np.array([0.0, 6.0, 1.0, 3.0]),
            ),
        }
        first_inputs = []
        second_inputs = []
        continuous_predictions = {}
        for member_id, (first_p, second_p) in precipitation.items():
            first_name = f"{member_id}-cycle-1.csv"
            second_name = f"{member_id}-cycle-2.csv"
            full_name = f"{member_id}-continuous.csv"
            self._write_csv(
                first_name,
                first_p,
                temperature_first,
                pet_first,
                observations_first,
                start=date(2026, 3, 2),
            )
            self._write_csv(
                second_name,
                second_p,
                temperature_second,
                pet_second,
                observations_second,
                start=date(2026, 3, 5),
            )
            self._write_csv(
                full_name,
                np.concatenate((first_p, second_p)),
                np.concatenate((temperature_first, temperature_second)),
                np.concatenate((pet_first, pet_second)),
                np.concatenate((observations_first, observations_second)),
                start=date(2026, 3, 2),
            )
            first_inputs.append(
                (member_id, self._analysis(first_name, with_observations=True), True)
            )
            second_inputs.append(
                (member_id, self._analysis(second_name, with_observations=True), True)
            )
            continuous = run_forecast(
                self.archive,
                self._analysis(full_name, with_observations=True),
                initialization="source_final",
                observations_available=True,
            )
            continuous_predictions[member_id] = np.asarray(
                continuous["best_simulation"]["outputs"], dtype=float
            )[-4:]

        first = run_forcing_ensemble_forecast(
            self.archive,
            first_inputs,
            initialization="source_final",
            issue_id="cycle-1",
        )
        second = run_forcing_ensemble_forecast(
            self.archive,
            list(reversed(second_inputs)),
            initialization="source_final",
            continuation_result=first,
            issue_id="cycle-2",
        )

        self.assertEqual(second["continuation_task_id"], first["task_id"])
        self.assertEqual(second["initialization"], "member_final")
        self.assertEqual(second["forecast_ensemble"]["cycle_index"], 2)
        self.assertEqual(
            second["forecast_ensemble"]["parent_issue_id"],
            "cycle-1",
        )
        archived = {
            member["member_id"]: member
            for member in second["forecast_ensemble"]["members"]
        }
        for member_id, expected in continuous_predictions.items():
            np.testing.assert_allclose(
                np.asarray(archived[member_id]["predictions"], dtype=float),
                expected,
                rtol=0.0,
                atol=1e-12,
            )
        np.testing.assert_allclose(
            np.asarray(second["best_simulation"]["outputs"], dtype=float),
            np.mean(np.stack(list(continuous_predictions.values())), axis=0),
            rtol=0.0,
            atol=1e-12,
        )

        wrong_identity = [
            ("member-a", second_inputs[0][1], True),
            ("member-c", second_inputs[1][1], True),
        ]
        with self.assertRaisesRegex(ValueError, "exact meteorological member identities"):
            run_forcing_ensemble_forecast(
                self.archive,
                wrong_identity,
                initialization="source_final",
                continuation_result=first,
            )

    def test_memberwise_cycle_rejects_a_temporal_gap(self):
        values = np.ones(2)
        first_inputs = []
        gap_inputs = []
        for member_id in ("member-a", "member-b"):
            first_name = f"{member_id}-first.csv"
            gap_name = f"{member_id}-gap.csv"
            self._write_csv(
                first_name,
                values,
                values * 6.0,
                values * 0.5,
                values * 0.2,
                start=date(2026, 3, 2),
            )
            self._write_csv(
                gap_name,
                values,
                values * 6.0,
                values * 0.5,
                values * 0.2,
                start=date(2026, 3, 5),
            )
            first_inputs.append(
                (member_id, self._analysis(first_name, with_observations=True), True)
            )
            gap_inputs.append(
                (member_id, self._analysis(gap_name, with_observations=True), True)
            )
        first = run_forcing_ensemble_forecast(
            self.archive,
            first_inputs,
            initialization="source_final",
        )
        with self.assertRaisesRegex(ValueError, "next daily interval"):
            run_forcing_ensemble_forecast(
                self.archive,
                gap_inputs,
                initialization="source_final",
                continuation_result=first,
            )

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow 2.21 is not installed in this test environment")
    def test_neural_members_continue_with_their_own_antecedent_context(self):
        length = 90
        index = np.arange(length, dtype=float)
        precipitation = 2.0 + np.sin(index / 4.0)
        temperature = 7.0 + np.cos(index / 9.0)
        pet = 0.5 + 0.1 * np.sin(index / 7.0)
        streamflow = np.zeros(length)
        streamflow[1:] = 0.6 * precipitation[:-1] + 0.05 * temperature[:-1]
        self._write_csv(
            "neural-source.csv",
            precipitation,
            temperature,
            pet,
            streamflow,
            start=date(2026, 1, 1),
        )
        config = ExperimentConfigV1.model_validate(
            {
                "schema_version": "1.19",
                "experiment_id": "ensemble-neural-context",
                "name": "Ensemble neural context",
                "data": {
                    "path": "neural-source.csv",
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
                    "hidden_layers": [6],
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
                        "epochs": 5,
                        "batch_size": 16,
                        "learning_rate": 0.01,
                        "patience": 5,
                        "seed": 17,
                    },
                },
            }
        )
        trained = ExperimentRunner(self.root).run(config, task_id="neural-source")
        archive = {
            "task_id": "neural-source",
            "config": config.model_dump(mode="json"),
            "result": trained.as_dict(),
        }
        first_inputs = []
        second_inputs = []
        continuous_predictions = {}
        for position, member_id in enumerate(("member-a", "member-b")):
            first_p = np.array([1.0, 3.0, 0.0, 2.0]) + position
            second_p = np.array([4.0, 0.0, 1.0]) + 0.5 * position
            first_t = np.full(4, 8.0 + position)
            second_t = np.full(3, 9.0 + position)
            first_pet = np.full(4, 0.6)
            second_pet = np.full(3, 0.7)
            first_name = f"neural-{member_id}-1.csv"
            second_name = f"neural-{member_id}-2.csv"
            full_name = f"neural-{member_id}-full.csv"
            self._write_csv(
                first_name,
                first_p,
                first_t,
                first_pet,
                np.full(4, np.nan),
                start=date(2026, 4, 1),
            )
            self._write_csv(
                second_name,
                second_p,
                second_t,
                second_pet,
                np.full(3, np.nan),
                start=date(2026, 4, 5),
            )
            self._write_csv(
                full_name,
                np.concatenate((first_p, second_p)),
                np.concatenate((first_t, second_t)),
                np.concatenate((first_pet, second_pet)),
                np.full(7, np.nan),
                start=date(2026, 4, 1),
            )
            first_inputs.append(
                (member_id, self._analysis(first_name, with_observations=False), False)
            )
            second_inputs.append(
                (member_id, self._analysis(second_name, with_observations=False), False)
            )
            continuous = run_forecast(
                archive,
                self._analysis(full_name, with_observations=False),
                initialization="source_final",
                observations_available=False,
            )
            continuous_predictions[member_id] = np.asarray(
                continuous["best_simulation"]["outputs"], dtype=float
            )[-3:]

        first = run_forcing_ensemble_forecast(
            archive,
            first_inputs,
            initialization="source_final",
        )
        for member in first["forecast_ensemble"]["members"]:
            self.assertEqual(member["final_state"], [])
            self.assertEqual(
                set(member["neural_context"]),
                {
                    "precipitation",
                    "temperature",
                    "potential_evapotranspiration",
                },
            )
        second = run_forcing_ensemble_forecast(
            archive,
            list(reversed(second_inputs)),
            initialization="source_final",
            continuation_result=first,
        )
        by_id = {
            member["member_id"]: np.asarray(member["predictions"], dtype=float)
            for member in second["forecast_ensemble"]["members"]
        }
        for member_id, expected in continuous_predictions.items():
            np.testing.assert_allclose(
                by_id[member_id],
                expected,
                rtol=0.0,
                atol=1e-12,
            )


class BundledForecastEnsembleExampleTests(unittest.TestCase):
    def test_three_csv_members_run_from_bundled_calibration(self):
        project_root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(
            project_root / "examples/exp_hydro_calibration.yml"
        )
        source_result = ExperimentRunner(project_root).run(
            config,
            task_id="bundled-ensemble-source",
        )
        archive = {
            "task_id": "bundled-ensemble-source",
            "config": config.model_dump(mode="json"),
            "result": source_result.as_dict(),
        }
        members = []
        for filename in (
            "member_dry.csv",
            "member_control.csv",
            "member_wet.csv",
        ):
            path = f"examples/data/forcing_ensemble/{filename}"
            analysis = analyze_calibration_csv(
                CsvDataSourceConfig(
                    path=path,
                    columns=CsvColumnMapping(
                        timestamp="date",
                        precipitation="precipitation",
                        temperature="temperature",
                        potential_evapotranspiration="pet",
                        streamflow="streamflow",
                    ),
                ),
                workspace_root=project_root,
            )
            members.append((filename.removesuffix(".csv"), analysis, True))
        result = run_forcing_ensemble_forecast(
            archive,
            members,
            initialization="source_final",
            issue_id="bundled-ensemble-example",
        )
        self.assertEqual(
            result["forecast_ensemble"]["member_count"],
            3,
        )
        self.assertEqual(
            result["forecast_ensemble"]["aggregate_scores"]["scored_leads"],
            7,
        )
        self.assertEqual(
            result["timestamps"],
            [f"2020-02-{day:02d}" for day in range(10, 17)],
        )


if __name__ == "__main__":
    unittest.main()
