import csv
import base64
from datetime import date, timedelta
from pathlib import Path
import tempfile
import time
import unittest
import importlib.util

try:
    from fastapi.testclient import TestClient
    from hydrosoftware.service.api import create_app
except ImportError:
    TestClient = None
    create_app = None

from tests.test_experiment_execution import make_config
from tests.test_forecast_postprocessing import biased_cases
from tests.test_forecast_events import event_cases
from tests.test_temporal_validation import conceptual_config
from hydrosoftware.configuration import load_experiment_config
from hydrosoftware.execution.persistence import ProjectStore


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are not installed.")
class ServiceApiTests(unittest.TestCase):
    def test_successful_experiment_round_trip(self):
        project_root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(
            project_root / "examples/exp_hydro_calibration.yml"
        )
        app = create_app(workspace_root=project_root, max_workers=1)
        with TestClient(app) as client:
            quality = client.post(
                "/api/v1/data/quality",
                json=config.model_dump(mode="json"),
            )
            self.assertEqual(quality.status_code, 200)
            self.assertTrue(quality.json()["ready"])

            validation = client.post(
                "/api/v1/experiments/validate",
                json=config.model_dump(mode="json"),
            )
            self.assertEqual(validation.status_code, 200)
            self.assertEqual(validation.json()["scored_observations"], 20)
            self.assertEqual(
                validation.json()["partition_scored_observations"],
                {"calibration": 10, "validation": 5, "test": 20},
            )
            self.assertEqual(
                validation.json()["primary_evaluation_partition"],
                "test",
            )

            response = client.post(
                "/api/v1/runs",
                json=config.model_dump(mode="json"),
            )
            self.assertEqual(response.status_code, 202)
            task_id = response.json()["task_id"]
            for _ in range(200):
                task = client.get(f"/api/v1/tasks/{task_id}").json()
                if task["terminal"]:
                    break
                time.sleep(0.01)

            self.assertEqual(task["state"], "completed")
            result = client.get(f"/api/v1/tasks/{task_id}/result")
            self.assertEqual(result.status_code, 200)
            payload = result.json()
            self.assertAlmostEqual(payload["calibration"]["best_score"], 1.0)
            self.assertEqual(
                [metric["symbol"] for metric in payload["evaluation_metrics"]],
                ["NSE", "KGE", "MSE", "RMSE", "NRMSE", "MAE", "PBIAS"],
            )
            self.assertEqual(len(payload["best_simulation"]["outputs"]), 40)
            self.assertEqual(len(payload["best_simulation"]["states"]), 41)
            self.assertEqual(len(payload["observations"]), 40)
            self.assertEqual(len(payload["forcings"]["precipitation"]), 40)

            events = client.get(f"/api/v1/tasks/{task_id}/events").json()["events"]
            self.assertEqual(
                [event["sequence"] for event in events], list(range(len(events)))
            )

    def test_health_schema_and_failed_task_are_available_over_http(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(workspace_root=Path(directory), max_workers=1)
            with TestClient(app) as client:
                self.assertEqual(client.get("/api/v1/health").status_code, 200)
                schema = client.get("/api/v1/schemas/experiment").json()
                self.assertIn("1.10", schema["properties"]["schema_version"]["enum"])
                self.assertIn("1.12", schema["properties"]["schema_version"]["enum"])
                self.assertIn("1.13", schema["properties"]["schema_version"]["enum"])
                self.assertIn("1.14", schema["properties"]["schema_version"]["enum"])
                self.assertIn("1.15", schema["properties"]["schema_version"]["enum"])
                self.assertIn("1.17", schema["properties"]["schema_version"]["enum"])
                self.assertIn("1.18", schema["properties"]["schema_version"]["enum"])
                self.assertIn("1.19", schema["properties"]["schema_version"]["enum"])
                self.assertIn("1.20", schema["properties"]["schema_version"]["enum"])
                models = client.get("/api/v1/models").json()["models"]
                self.assertEqual(len(models), 14)
                self.assertIn(
                    "hbv.mass_conservative_daily.v1",
                    [model["model_id"] for model in models],
                )
                self.assertIn(
                    "mlp.tensorflow_s2v.v1",
                    [model["model_id"] for model in models],
                )
                self.assertIn(
                    "lstm.tensorflow_s2v.v1",
                    [model["model_id"] for model in models],
                )
                self.assertIn(
                    "lstm.tensorflow_stacked_s2v.v2",
                    [model["model_id"] for model in models],
                )
                self.assertIn(
                    "gru.tensorflow_s2v.v1",
                    [model["model_id"] for model in models],
                )
                self.assertIn(
                    "gru.tensorflow_stacked_s2v.v2",
                    [model["model_id"] for model in models],
                )
                self.assertIn(
                    "cnn1d.tensorflow_s2v.v1",
                    [model["model_id"] for model in models],
                )
                self.assertIn(
                    "paml.tensorflow_prnn_cnn_s2s.v1",
                    [model["model_id"] for model in models],
                )
                for model_id in (
                    "mlp.tensorflow_s2s.v2",
                    "lstm.tensorflow_stacked_s2s.v3",
                    "gru.tensorflow_stacked_s2s.v3",
                    "cnn1d.tensorflow_s2s.v2",
                ):
                    self.assertIn(
                        model_id,
                        [model["model_id"] for model in models],
                    )
                self.assertIn(
                    "deepdiscover.numpy_instantaneous_s2s.v1",
                    [model["model_id"] for model in models],
                )

                response = client.post(
                    "/api/v1/runs",
                    json=make_config(path="missing.csv").model_dump(mode="json"),
                )
                self.assertEqual(response.status_code, 202)
                task_id = response.json()["task_id"]
                for _ in range(100):
                    task = client.get(f"/api/v1/tasks/{task_id}").json()
                    if task["terminal"]:
                        break
                    time.sleep(0.01)
                self.assertEqual(task["state"], "failed")
                self.assertEqual(task["error"]["code"], "data_file_not_found")

    def test_csv_can_be_imported_and_previewed_for_column_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(workspace_root=Path(directory), max_workers=1)
            content = b"date,p,t,pet,q\n2020-01-01,1,-2,0.2,0.1\n"
            with TestClient(app) as client:
                imported = client.post(
                    "/api/v1/data/imports",
                    json={
                        "filename": "test basin.csv",
                        "content_base64": base64.b64encode(content).decode("ascii"),
                    },
                )
                self.assertEqual(imported.status_code, 201)
                payload = imported.json()
                self.assertEqual(payload["columns"], ["date", "p", "t", "pet", "q"])
                self.assertEqual(payload["row_count"], 1)

                preview = client.post(
                    "/api/v1/data/preview",
                    json={"path": payload["relative_path"]},
                )
                self.assertEqual(preview.status_code, 200)
                self.assertEqual(preview.json()["sha256"], payload["sha256"])

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow 2.21 is not installed in this test environment")
    def test_saved_neural_run_can_be_explained_over_http(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with (workspace / "neural.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "p", "t", "pet", "q"])
                for index in range(60):
                    precipitation = float((index * 3) % 7)
                    writer.writerow(
                        [
                            f"2024-{1 + index // 28:02d}-{1 + index % 28:02d}",
                            precipitation,
                            4.0 + 0.1 * index,
                            0.5,
                            0.35 * precipitation + 0.02 * index,
                        ]
                    )
            config = {
                "schema_version": "1.15",
                "experiment_id": "http-shap",
                "name": "HTTP SHAP",
                "data": {
                    "path": "neural.csv",
                    "columns": {
                        "precipitation": "p",
                        "temperature": "t",
                        "potential_evapotranspiration": "pet",
                        "streamflow": "q",
                    },
                },
                "model": {
                    "model_id": "mlp.tensorflow_s2v.v1",
                    "feature_roles": ["precipitation"],
                    "lookback": 3,
                    "hidden_layers": [4],
                    "activation": "tanh",
                    "output_activation": "linear",
                },
                "task": {
                    "task_type": "training",
                    "period": {
                        "start_index": 0,
                        "train_end_index": 30,
                        "validation_end_index": 45,
                        "test_end_index": 60,
                    },
                    "training": {
                        "epochs": 4,
                        "batch_size": 8,
                        "patience": 4,
                        "seed": 9,
                    },
                },
            }
            app = create_app(workspace_root=workspace, max_workers=1)
            with TestClient(app) as client:
                response = client.post("/api/v1/runs", json=config)
                self.assertEqual(response.status_code, 202)
                task_id = response.json()["task_id"]
                for _ in range(300):
                    task = client.get(f"/api/v1/tasks/{task_id}").json()
                    if task["terminal"]:
                        break
                    time.sleep(0.01)
                self.assertEqual(task["state"], "completed")

                explanation = client.post(
                    "/api/v1/explanations",
                    json={
                        "task_id": task_id,
                        "explained_rows": 2,
                        "background_rows": 2,
                        "permutations": 3,
                        "seed": 5,
                    },
                )
                self.assertEqual(explanation.status_code, 200)
                payload = explanation.json()
                self.assertEqual(
                    payload["method_id"],
                    "empirical_interventional_permutation_shap.v1",
                )
                self.assertEqual(payload["explained_partition"], "test")
                self.assertEqual(payload["background_partition"], "training")
                self.assertLess(payload["maximum_additivity_error"], 1e-10)
                self.assertLess(
                    payload["maximum_archive_prediction_difference"], 1e-10
                )

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow 2.21 is not installed in this test environment")
    def test_causal_s2s_run_and_explanation_round_trip_over_http(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with (workspace / "s2s.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["p", "t", "pet", "q"])
                for index in range(60):
                    precipitation = float((index * 5) % 9)
                    temperature = 3.0 + 0.2 * index
                    writer.writerow(
                        [
                            precipitation,
                            temperature,
                            0.4,
                            0.4 * precipitation + 0.03 * temperature,
                        ]
                    )
            config = {
                "schema_version": "1.20",
                "experiment_id": "http-causal-s2s",
                "name": "HTTP causal S2S",
                "data": {
                    "path": "s2s.csv",
                    "columns": {
                        "precipitation": "p",
                        "temperature": "t",
                        "potential_evapotranspiration": "pet",
                        "streamflow": "q",
                    },
                },
                "model": {
                    "model_id": "mlp.tensorflow_s2s.v2",
                    "feature_roles": ["precipitation", "temperature"],
                    "lookback": 6,
                    "s2s_warmup": 2,
                    "hidden_layers": [4],
                    "activation": "tanh",
                    "output_activation": "linear",
                },
                "task": {
                    "task_type": "training",
                    "period": {
                        "start_index": 0,
                        "train_end_index": 30,
                        "validation_end_index": 45,
                        "test_end_index": 60,
                    },
                    "lead_steps": 0,
                    "stride": 1,
                    "training": {
                        "epochs": 2,
                        "batch_size": 8,
                        "patience": 2,
                        "seed": 13,
                    },
                },
            }
            app = create_app(workspace_root=workspace, max_workers=1)
            with TestClient(app) as client:
                response = client.post("/api/v1/runs", json=config)
                self.assertEqual(response.status_code, 202)
                task_id = response.json()["task_id"]
                for _ in range(300):
                    task = client.get(f"/api/v1/tasks/{task_id}").json()
                    if task["terminal"]:
                        break
                    time.sleep(0.01)
                self.assertEqual(task["state"], "completed")
                result = client.get(
                    f"/api/v1/tasks/{task_id}/result"
                ).json()
                self.assertEqual(
                    result["best_simulation"]["diagnostics"][
                        "prediction_layout"
                    ],
                    "sequence_to_sequence",
                )
                self.assertEqual(
                    result["training"]["artifact"]["artifact_type"],
                    "hydrosoftware.neural.tensorflow.s2s.v1",
                )
                explanation = client.post(
                    "/api/v1/explanations",
                    json={
                        "task_id": task_id,
                        "explained_rows": 2,
                        "background_rows": 1,
                        "permutations": 1,
                        "seed": 3,
                    },
                )
                self.assertEqual(explanation.status_code, 200)
                payload = explanation.json()
                self.assertEqual(
                    payload["maximum_archive_prediction_difference"],
                    0.0,
                )
                self.assertIn(
                    0,
                    [
                        item["lag_steps"]
                        for item in payload["global_by_lag"]
                    ],
                )

    def test_saved_experiment_and_run_archive_survive_service_restart(self):
        project_root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(
            project_root / "examples/exp_hydro_calibration.yml"
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            data_dir = workspace / "examples" / "data"
            data_dir.mkdir(parents=True)
            source = project_root / config.data.path
            (data_dir / source.name).write_bytes(source.read_bytes())

            first_app = create_app(workspace_root=workspace, max_workers=1)
            with TestClient(first_app) as client:
                saved = client.put(
                    f"/api/v1/experiments/{config.experiment_id}",
                    json=config.model_dump(mode="json"),
                )
                self.assertEqual(saved.status_code, 200)
                response = client.post(
                    "/api/v1/runs",
                    json=config.model_dump(mode="json"),
                )
                task_id = response.json()["task_id"]
                for _ in range(200):
                    task = client.get(f"/api/v1/tasks/{task_id}").json()
                    if task["terminal"]:
                        break
                    time.sleep(0.01)
                self.assertEqual(task["state"], "completed")

            second_app = create_app(workspace_root=workspace, max_workers=1)
            with (workspace / "forecast.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["date", "p", "t", "pet"])
                writer.writerow(["2030-01-01", 2.0, 5.0, 0.3])
                writer.writerow(["2030-01-02", 0.0, 6.0, 0.4])
                writer.writerow(["2030-01-03", 4.0, 7.0, 0.5])
            for member, precipitation in (
                ("ensemble-1.csv", [2.0, 0.0, 4.0]),
                ("ensemble-2.csv", [1.0, 1.0, 5.0]),
                ("ensemble-3.csv", [3.0, 0.0, 2.0]),
                ("ensemble-4.csv", [2.0, 2.0, 3.0]),
                ("ensemble-cycle-1.csv", [0.0, 4.0, 1.0]),
                ("ensemble-cycle-2.csv", [2.0, 0.0, 5.0]),
            ):
                with (workspace / member).open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["date", "p", "t", "pet", "q"])
                    for index in range(3):
                        writer.writerow(
                            [
                                (
                                    f"2030-02-0{index + 1}"
                                    if member in {"ensemble-3.csv", "ensemble-4.csv"}
                                    else (
                                        f"2030-01-0{index + 4}"
                                        if member.startswith("ensemble-cycle-")
                                        else f"2030-01-0{index + 1}"
                                    )
                                ),
                                precipitation[index],
                                5.0 + index,
                                0.3 + 0.1 * index,
                                [0.2, 0.3, 0.5][index],
                            ]
                        )
            with TestClient(second_app) as client:
                experiments = client.get("/api/v1/experiments").json()["experiments"]
                self.assertEqual(len(experiments), 1)
                self.assertEqual(experiments[0]["latest_run_id"], task_id)

                loaded = client.get(
                    f"/api/v1/experiments/{config.experiment_id}"
                )
                self.assertEqual(loaded.status_code, 200)
                self.assertEqual(loaded.json()["config"]["name"], config.name)

                archive = client.get(f"/api/v1/run-archives/{task_id}")
                self.assertEqual(archive.status_code, 200)
                self.assertAlmostEqual(
                    archive.json()["result"]["calibration"]["best_score"],
                    1.0,
                )

                repeated = client.post(
                    "/api/v1/runs",
                    json=config.model_dump(mode="json"),
                )
                self.assertEqual(repeated.status_code, 202)
                repeated_task_id = repeated.json()["task_id"]
                for _ in range(200):
                    repeated_task = client.get(
                        f"/api/v1/tasks/{repeated_task_id}"
                    ).json()
                    if repeated_task["terminal"]:
                        break
                    time.sleep(0.01)
                self.assertEqual(repeated_task["state"], "completed")

                comparison = client.post(
                    "/api/v1/comparisons",
                    json={"task_ids": [task_id, repeated_task_id]},
                )
                self.assertEqual(comparison.status_code, 200)
                comparison_payload = comparison.json()
                self.assertEqual(comparison_payload["alignment"], "timestamp")
                self.assertEqual(len(comparison_payload["runs"]), 2)
                self.assertEqual(
                    comparison_payload["scored_observations"],
                    20,
                )
                self.assertEqual(
                    [metric["symbol"] for metric in comparison_payload["runs"][0]["metrics"]],
                    ["NSE", "KGE", "MSE", "RMSE", "NRMSE", "MAE", "PBIAS"],
                )

                uncertainty = client.post(
                    "/api/v1/uncertainty",
                    json={
                        "task_ids": [task_id, repeated_task_id],
                        "bootstrap_replicates": 200,
                        "confidence_level": 0.95,
                        "block_length": 3,
                        "seed": 9,
                        "reference_task_id": repeated_task_id,
                    },
                )
                self.assertEqual(uncertainty.status_code, 200)
                uncertainty_payload = uncertainty.json()
                self.assertEqual(
                    uncertainty_payload["method_id"],
                    "moving_block_bootstrap_metrics.v1",
                )
                self.assertEqual(
                    uncertainty_payload["reference_task_id"],
                    repeated_task_id,
                )
                self.assertEqual(uncertainty_payload["block_length"], 3)
                self.assertEqual(len(uncertainty_payload["runs"]), 2)
                self.assertEqual(len(uncertainty_payload["comparisons"]), 1)
                self.assertEqual(
                    [metric["symbol"] for metric in uncertainty_payload["runs"][0]["metrics"]],
                    ["NSE", "KGE", "MSE", "RMSE", "NRMSE", "MAE", "PBIAS"],
                )

                forecast_data = config.data.model_dump(mode="json")
                forecast_data["path"] = "forecast.csv"
                forecast_data.pop("expected_sha256", None)
                forecast_data["columns"] = {
                    "timestamp": "date",
                    "precipitation": "p",
                    "temperature": "t",
                    "potential_evapotranspiration": "pet",
                }
                forecast = client.post(
                    "/api/v1/forecasts",
                    json={
                        "source_task_id": task_id,
                        "initialization": "source_final",
                        "data": forecast_data,
                    },
                )
                self.assertEqual(forecast.status_code, 200)
                forecast_payload = forecast.json()
                self.assertEqual(forecast_payload["run_kind"], "forecast")
                self.assertIsNone(forecast_payload["observations"])
                self.assertEqual(forecast_payload["evaluation_metrics"], [])
                self.assertEqual(len(forecast_payload["best_simulation"]["outputs"]), 3)
                self.assertEqual(
                    forecast_payload["best_simulation"]["states"][0],
                    archive.json()["result"]["best_simulation"]["states"][-1],
                )

                ensemble_members = []
                for index, member in enumerate(
                    ("ensemble-1.csv", "ensemble-2.csv"),
                    start=1,
                ):
                    member_data = dict(forecast_data)
                    member_data["path"] = member
                    member_data["columns"] = {
                        **forecast_data["columns"],
                        "streamflow": "q",
                    }
                    ensemble_members.append(
                        {
                            "member_id": f"member-{index:02d}",
                            "data": member_data,
                        }
                    )
                ensemble_forecast = client.post(
                    "/api/v1/forecast-ensembles",
                    json={
                        "source_task_id": task_id,
                        "issue_id": "hindcast-2030-01-01",
                        "initialization": "source_final",
                        "members": ensemble_members,
                        "lower_quantile": 0.1,
                        "upper_quantile": 0.9,
                    },
                )
                self.assertEqual(ensemble_forecast.status_code, 200)
                ensemble_payload = ensemble_forecast.json()
                self.assertEqual(
                    ensemble_payload["forecast_ensemble"]["method_id"],
                    "meteorological_forcing_ensemble.v1",
                )
                self.assertEqual(
                    ensemble_payload["forecast_ensemble"]["member_count"],
                    2,
                )
                self.assertEqual(
                    ensemble_payload["forecast_ensemble"]["aggregate_scores"][
                        "scored_leads"
                    ],
                    3,
                )
                self.assertIsNotNone(
                    ensemble_payload["forecast_ensemble"]["lead_time_diagnostics"][
                        0
                    ]["crps"]
                )
                archived_forecasts = client.get(
                    "/api/v1/forecast-ensembles"
                ).json()["forecast_ensembles"]
                self.assertEqual(len(archived_forecasts), 1)
                self.assertEqual(
                    archived_forecasts[0]["issue_id"],
                    "hindcast-2030-01-01",
                )
                reopened_ensemble = client.get(
                    f"/api/v1/forecast-ensembles/{ensemble_payload['task_id']}"
                )
                self.assertEqual(reopened_ensemble.status_code, 200)

                continued_members = []
                for index, member in enumerate(
                    ("ensemble-cycle-2.csv", "ensemble-cycle-1.csv"),
                    start=1,
                ):
                    member_data = dict(forecast_data)
                    member_data["path"] = member
                    member_data["columns"] = {
                        **forecast_data["columns"],
                        "streamflow": "q",
                    }
                    continued_members.append(
                        {
                            "member_id": f"member-{3 - index:02d}",
                            "data": member_data,
                        }
                    )
                continued_ensemble = client.post(
                    "/api/v1/forecast-ensembles",
                    json={
                        "source_task_id": task_id,
                        "continuation_task_id": ensemble_payload["task_id"],
                        "issue_id": "cycle-2030-01-04",
                        "initialization": "source_final",
                        "members": continued_members,
                        "lower_quantile": 0.1,
                        "upper_quantile": 0.9,
                    },
                )
                self.assertEqual(continued_ensemble.status_code, 200)
                continued_payload = continued_ensemble.json()
                self.assertEqual(
                    continued_payload["continuation_task_id"],
                    ensemble_payload["task_id"],
                )
                self.assertEqual(continued_payload["initialization"], "member_final")
                self.assertEqual(
                    continued_payload["forecast_ensemble"]["cycle_index"],
                    2,
                )
                self.assertEqual(
                    continued_payload["forecast_ensemble"]["member_ids"],
                    ["member-02", "member-01"],
                )

                second_members = []
                for index, member in enumerate(
                    ("ensemble-3.csv", "ensemble-4.csv"),
                    start=1,
                ):
                    member_data = dict(forecast_data)
                    member_data["path"] = member
                    member_data["columns"] = {
                        **forecast_data["columns"],
                        "streamflow": "q",
                    }
                    second_members.append(
                        {
                            "member_id": f"member-{index:02d}",
                            "data": member_data,
                        }
                    )
                second_ensemble = client.post(
                    "/api/v1/forecast-ensembles",
                    json={
                        "source_task_id": task_id,
                        "issue_id": "hindcast-2030-02-01",
                        "initialization": "source_final",
                        "members": second_members,
                        "lower_quantile": 0.1,
                        "upper_quantile": 0.9,
                    },
                )
                self.assertEqual(second_ensemble.status_code, 200)
                verification = client.post(
                    "/api/v1/forecast-verifications",
                    json={
                        "task_ids": [
                            ensemble_payload["task_id"],
                            second_ensemble.json()["task_id"],
                        ],
                        "seed": 11,
                        "minimum_cases_per_lead": 2,
                    },
                )
                self.assertEqual(verification.status_code, 200)
                verification_payload = verification.json()
                self.assertEqual(
                    verification_payload["method_id"],
                    "meteorological_ensemble_hindcast_verification.v1",
                )
                self.assertEqual(verification_payload["case_count"], 2)
                self.assertEqual(
                    len(verification_payload["lead_time_diagnostics"]),
                    3,
                )
                saved_verifications = client.get(
                    "/api/v1/forecast-verifications"
                ).json()["forecast_verifications"]
                self.assertEqual(len(saved_verifications), 1)
                reopened_verification = client.get(
                    "/api/v1/forecast-verifications/"
                    f"{verification_payload['task_id']}"
                )
                self.assertEqual(reopened_verification.status_code, 200)

    def test_forecast_postprocessing_is_created_listed_and_reopened(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            cases = biased_cases()
            task_ids = []
            for case in cases:
                store.save_forecast_ensemble(case)
                task_ids.append(case["task_id"])

            app = create_app(workspace_root=Path(directory), max_workers=1)
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/forecast-postprocessings",
                    json={
                        "task_ids": list(reversed(task_ids)),
                        "training_case_count": 6,
                        "minimum_training_cases_per_lead": 5,
                        "seed": 13,
                    },
                )
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(
                    payload["method_id"],
                    "affine_member_recalibration.v1",
                )
                self.assertEqual(payload["split"]["training_case_count"], 6)
                self.assertEqual(payload["split"]["evaluation_case_count"], 4)
                self.assertGreater(
                    payload["aggregate"]["crps_skill_score"],
                    0.0,
                )

                summaries = client.get(
                    "/api/v1/forecast-postprocessings"
                ).json()["forecast_postprocessings"]
                self.assertEqual(len(summaries), 1)
                self.assertEqual(summaries[0]["training_case_count"], 6)
                reopened = client.get(
                    "/api/v1/forecast-postprocessings/"
                    f"{payload['task_id']}"
                )
                self.assertEqual(reopened.status_code, 200)
                self.assertEqual(
                    reopened.json()["result"]["parameters"],
                    payload["parameters"],
                )

    def test_event_decision_verification_is_created_listed_and_reopened(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            task_ids = []
            for case in event_cases():
                store.save_forecast_ensemble(case)
                task_ids.append(case["task_id"])

            app = create_app(workspace_root=Path(directory), max_workers=1)
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/forecast-event-verifications",
                    json={
                        "task_ids": list(reversed(task_ids)),
                        "training_case_count": 20,
                        "threshold_mode": "absolute",
                        "threshold_value": 2.0,
                        "training_quantile": 0.9,
                        "probability_bin_count": 10,
                        "minimum_evaluation_cases_per_lead": 5,
                    },
                )
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(
                    payload["method_id"],
                    "threshold_event_decision_verification.v1",
                )
                self.assertEqual(payload["threshold"]["value"], 2.0)
                self.assertGreater(
                    payload["aggregate"]["adjusted_brier_skill_vs_raw"],
                    0.0,
                )

                summaries = client.get(
                    "/api/v1/forecast-event-verifications"
                ).json()["forecast_event_verifications"]
                self.assertEqual(len(summaries), 1)
                self.assertEqual(summaries[0]["threshold_value"], 2.0)
                reopened = client.get(
                    "/api/v1/forecast-event-verifications/"
                    f"{payload['task_id']}"
                )
                self.assertEqual(reopened.status_code, 200)
                self.assertEqual(
                    reopened.json()["result"]["aggregate"],
                    payload["aggregate"],
                )

    def test_temporal_validation_runs_asynchronously_and_is_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with (workspace / "basin.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "date",
                        "precipitation",
                        "temperature",
                        "pet",
                        "streamflow",
                    ]
                )
                for index in range(80):
                    writer.writerow(
                        [
                            (
                                date(2024, 1, 1) + timedelta(days=index)
                            ).isoformat(),
                            1.0 + (index % 5),
                            5.0 + 0.05 * index,
                            0.4,
                            0.2 + 0.1 * (index % 7),
                        ]
                    )

            config = conceptual_config()
            app = create_app(workspace_root=workspace, max_workers=1)
            with TestClient(app) as client:
                saved = client.put(
                    f"/api/v1/experiments/{config.experiment_id}",
                    json=config.model_dump(mode="json"),
                )
                self.assertEqual(saved.status_code, 200)

                started = client.post(
                    "/api/v1/temporal-validations",
                    json={
                        "experiment_ids": [config.experiment_id],
                        "start_index": 0,
                        "initial_training_steps": 40,
                        "validation_steps": 10,
                        "test_steps": 10,
                        "step_steps": 10,
                        "folds": 2,
                        "window_mode": "expanding",
                        "minimum_scored_observations": 5,
                    },
                )
                self.assertEqual(started.status_code, 202)
                task_id = started.json()["task_id"]
                for _ in range(300):
                    task = client.get(
                        f"/api/v1/temporal-validations/tasks/{task_id}"
                    ).json()
                    if task["terminal"]:
                        break
                    time.sleep(0.01)
                self.assertEqual(task["state"], "completed")

                result = client.get(
                    f"/api/v1/temporal-validations/tasks/{task_id}/result"
                )
                self.assertEqual(result.status_code, 200)
                self.assertEqual(
                    result.json()["method_id"],
                    "rolling_origin_temporal_validation.v1",
                )
                self.assertEqual(
                    result.json()["pooled_scored_observations"],
                    20,
                )

                archives = client.get(
                    "/api/v1/temporal-validations"
                ).json()["temporal_validations"]
                self.assertEqual(len(archives), 1)
                self.assertEqual(archives[0]["task_id"], task_id)
                reopened = client.get(
                    f"/api/v1/temporal-validations/archives/{task_id}"
                )
                self.assertEqual(reopened.status_code, 200)
                self.assertEqual(
                    reopened.json()["result"]["settings"]["folds"],
                    2,
                )


if __name__ == "__main__":
    unittest.main()
