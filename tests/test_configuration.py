import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from hydrosoftware.configuration import (
    ExperimentConfigV1,
    configuration_hash,
    experiment_json_schema,
    load_experiment_config,
)


def valid_payload(data_path="data/basin.csv"):
    return {
        "schema_version": "1.0",
        "experiment_id": "exp-hydro-demo",
        "name": "EXP-HYDRO demonstration",
        "data": {
            "kind": "csv",
            "path": data_path,
            "columns": {
                "timestamp": "date",
                "precipitation": "prcp",
                "temperature": "temp",
                "potential_evapotranspiration": "pet",
                "streamflow": "qobs",
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
            "period": {"start_index": 0, "end_index": 20, "warmup_steps": 2},
            "objective": {"objective_id": "nash_sutcliffe_efficiency"},
            "search": {
                "algorithm_id": "uniform_random_search.v1",
                "n_evaluations": 10,
                "seed": 7,
                "include_defaults": True,
                "continue_on_candidate_error": True,
            },
        },
    }


class ConfigurationTests(unittest.TestCase):
    def test_valid_configuration_has_a_stable_hash(self):
        first = ExperimentConfigV1.model_validate(valid_payload())
        second = ExperimentConfigV1.model_validate(
            json.loads(json.dumps(valid_payload(), sort_keys=True))
        )

        self.assertEqual(configuration_hash(first), configuration_hash(second))
        self.assertEqual(len(configuration_hash(first)), 64)

    def test_schema_version_and_unknown_fields_are_rejected(self):
        wrong_version = valid_payload()
        wrong_version["schema_version"] = "2.0"
        with self.assertRaises(ValidationError):
            ExperimentConfigV1.model_validate(wrong_version)

        unknown = valid_payload()
        unknown["silent_option"] = True
        with self.assertRaises(ValidationError):
            ExperimentConfigV1.model_validate(unknown)

    def test_data_path_must_remain_portable_and_inside_workspace(self):
        for path in ("../outside.csv", "/absolute.csv", r"C:\\absolute.csv"):
            with self.subTest(path=path), self.assertRaises(ValidationError):
                ExperimentConfigV1.model_validate(valid_payload(path))

    def test_no_snow_rejects_nonzero_initial_snow_storage(self):
        payload = valid_payload()
        payload["model"]["initial_state"] = [1.0, 0.0]
        with self.assertRaisesRegex(ValidationError, "zero initial snow"):
            ExperimentConfigV1.model_validate(payload)

    def test_nonfinite_numeric_configuration_is_rejected(self):
        payload = valid_payload()
        payload["model"]["initial_state"] = [0.0, float("nan")]
        with self.assertRaises(ValidationError):
            ExperimentConfigV1.model_validate(payload)

    def test_snow_parameters_can_be_fixed_or_calibrated_with_explicit_bounds(self):
        payload = valid_payload()
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
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(config.model.snow.calibration.degree_day_factor.mode, "calibrate")
        self.assertEqual(config.model.snow.calibration.snowfall_temperature.mode, "fixed")

        payload["model"]["snow"]["degree_day_factor"] = 6.0
        with self.assertRaisesRegex(ValidationError, "inside its calibration bounds"):
            ExperimentConfigV1.model_validate(payload)

    def test_streamflow_unit_conversion_requires_basin_area(self):
        payload = valid_payload()
        payload["schema_version"] = "1.1"
        payload["data"]["processing"] = {
            "units": {"streamflow": "m3/s"}
        }

        with self.assertRaisesRegex(ValidationError, "basin_area_km2"):
            ExperimentConfigV1.model_validate(payload)

    def test_yaml_and_json_loaders_share_the_same_contract(self):
        payload = valid_payload()
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "experiment.json"
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_experiment_config(json_path)

        self.assertEqual(loaded.experiment_id, "exp-hydro-demo")
        self.assertEqual(loaded.task.period.warmup_steps, 2)

    def test_exported_json_schema_declares_supported_versions(self):
        schema = experiment_json_schema()
        self.assertEqual(
            schema["properties"]["schema_version"]["enum"],
            [
                "1.0",
                "1.1",
                "1.2",
                "1.3",
                "1.4",
                "1.5",
                "1.6",
                "1.7",
                "1.8",
                "1.9",
                "1.10",
                "1.11",
                "1.12",
                "1.13",
                "1.14",
                "1.15",
                "1.16",
                "1.17",
                "1.18",
                "1.19",
                "1.20",
                "1.21",
            ],
        )

    def test_neural_initialization_ensemble_has_bounded_explicit_scope(self):
        payload = valid_payload()
        payload["schema_version"] = "1.17"
        payload["model"] = {
            "model_id": "mlp.tensorflow_s2v.v1",
            "lookback": 3,
            "hidden_layers": [8],
        }
        payload["task"] = {
            "task_type": "training",
            "period": {
                "start_index": 0,
                "train_end_index": 10,
                "validation_end_index": 15,
                "test_end_index": 20,
            },
            "ensemble": {
                "enabled": True,
                "members": 5,
                "aggregation": "arithmetic_mean",
                "lower_quantile": 0.1,
                "upper_quantile": 0.9,
            },
        }
        config = ExperimentConfigV1.model_validate(payload)
        self.assertTrue(config.task.ensemble.enabled)
        self.assertEqual(config.task.ensemble.members, 5)

        payload["task"]["ensemble"]["members"] = 2
        with self.assertRaises(ValidationError):
            ExperimentConfigV1.model_validate(payload)

        payload["task"]["ensemble"]["members"] = 5
        payload["task"]["ensemble"]["lower_quantile"] = 0.95
        with self.assertRaisesRegex(ValidationError, "lower_quantile"):
            ExperimentConfigV1.model_validate(payload)

    def test_schema_118_preserves_explicit_holdout_boundaries_without_protocol(self):
        payload = valid_payload()
        payload["schema_version"] = "1.18"
        payload["task"]["period"] = {
            "start_index": 0,
            "calibration_end_index": 10,
            "validation_end_index": 15,
            "test_end_index": 20,
            "warmup_steps": 2,
        }
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(
            config.task.period.protocol,
            "calibration_validation_test",
        )
        self.assertEqual(config.task.period.calibration_end_index, 10)
        self.assertEqual(config.task.period.validation_end_index, 15)
        self.assertEqual(config.task.period.test_end_index, 20)

    def test_differential_evolution_search_has_bounded_explicit_controls(self):
        payload = valid_payload()
        payload["schema_version"] = "1.16"
        payload["task"]["search"] = {
            "algorithm_id": "differential_evolution.rand1bin.v1",
            "generations": 30,
            "population_size": 24,
            "mutation_factor": 0.7,
            "crossover_probability": 0.85,
            "seed": 11,
        }
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(
            config.task.search.algorithm_id,
            "differential_evolution.rand1bin.v1",
        )
        self.assertEqual(config.task.search.population_size, 24)

        payload["task"]["search"]["population_size"] = 3
        with self.assertRaises(ValidationError):
            ExperimentConfigV1.model_validate(payload)

        payload["task"]["search"]["population_size"] = 24
        payload["task"]["search"]["mutation_factor"] = 2.1
        with self.assertRaises(ValidationError):
            ExperimentConfigV1.model_validate(payload)

    def test_legacy_configuration_without_search_keeps_random_baseline(self):
        payload = valid_payload()
        payload["schema_version"] = "1.15"
        del payload["task"]["search"]
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(
            config.task.search.algorithm_id,
            "uniform_random_search.v1",
        )

        payload["schema_version"] = "1.16"
        current = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(
            current.task.search.algorithm_id,
            "differential_evolution.rand1bin.v1",
        )


    def test_hbv_requires_four_non_negative_initial_storages(self):
        payload = valid_payload()
        payload["schema_version"] = "1.3"
        payload["model"] = {
            "model_id": "hbv.mass_conservative_daily.v1",
            "snow": {"module_id": "snow.none.v1"},
            "initial_state": [0.0, 300.0, 20.0, 80.0],
        }
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(len(config.model.initial_state), 4)

        payload["model"]["initial_state"] = [0.0, 300.0, -1.0, 80.0]
        with self.assertRaises(ValidationError):
            ExperimentConfigV1.model_validate(payload)

    def test_hbv_maxbas_routing_is_explicit_discrete_and_restartable(self):
        payload = valid_payload()
        payload["schema_version"] = "1.4"
        payload["model"] = {
            "model_id": "hbv.mass_conservative_daily.v1",
            "snow": {"module_id": "snow.none.v1"},
            "initial_state": [0.0, 300.0, 20.0, 80.0],
            "routing": {
                "process_id": "routing.maxbas.triangular.v1",
                "maxbas": 4,
                "calibration": {
                    "maxbas": {
                        "mode": "calibrate",
                        "lower_bound": 1,
                        "upper_bound": 7,
                    }
                },
                "initial_state": [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
            },
        }
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(config.model.routing.maxbas, 4)
        self.assertEqual(config.model.routing.initial_state[:2], (0.1, 0.2))

        payload["model"]["routing"]["maxbas"] = 3.5
        with self.assertRaises(ValidationError):
            ExperimentConfigV1.model_validate(payload)

    def test_pre_routing_hbv_configuration_defaults_to_no_routing(self):
        payload = valid_payload()
        payload["schema_version"] = "1.3"
        payload["model"] = {
            "model_id": "hbv.mass_conservative_daily.v1",
            "snow": {"module_id": "snow.none.v1"},
            "initial_state": [0.0, 300.0, 20.0, 80.0],
        }
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(config.model.routing.process_id, "routing.none.v1")

    def test_mlp_requires_training_task_and_explicit_ordered_partitions(self):
        payload = valid_payload()
        payload["schema_version"] = "1.7"
        payload["model"] = {
            "model_id": "mlp.tensorflow_s2v.v1",
            "feature_roles": ["precipitation", "temperature"],
            "lookback": 7,
            "hidden_layers": [16, 8],
        }
        with self.assertRaisesRegex(ValidationError, "task_type='training'"):
            ExperimentConfigV1.model_validate(payload)

        payload["task"] = {
            "task_type": "training",
            "period": {
                "start_index": 0,
                "train_end_index": 30,
                "validation_end_index": 40,
                "test_end_index": 50,
            },
        }
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(config.model.lookback, 7)
        self.assertEqual(config.task.period.validation_end_index, 40)

        payload["task"]["period"]["validation_end_index"] = 30
        with self.assertRaises(ValidationError):
            ExperimentConfigV1.model_validate(payload)

    def test_lstm_requires_training_and_persists_recurrent_width_and_clipping(self):
        payload = valid_payload()
        payload["schema_version"] = "1.8"
        payload["model"] = {
            "model_id": "lstm.tensorflow_s2v.v1",
            "feature_roles": ["precipitation", "temperature"],
            "lookback": 10,
            "recurrent_units": 24,
            "gradient_clip_norm": 3.5,
            "output_activation": "linear",
        }
        payload["task"] = {
            "task_type": "training",
            "period": {
                "start_index": 0,
                "train_end_index": 30,
                "validation_end_index": 40,
                "test_end_index": 50,
            },
        }
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(config.model.recurrent_units, 24)
        self.assertEqual(config.model.gradient_clip_norm, 3.5)

        payload["model"]["recurrent_units"] = 0
        with self.assertRaises(ValidationError):
            ExperimentConfigV1.model_validate(payload)

    def test_gru_requires_training_and_persists_recurrent_width_and_clipping(self):
        payload = valid_payload()
        payload["schema_version"] = "1.9"
        payload["model"] = {
            "model_id": "gru.tensorflow_s2v.v1",
            "feature_roles": ["precipitation", "potential_evapotranspiration"],
            "lookback": 12,
            "recurrent_units": 20,
            "gradient_clip_norm": 4.0,
            "output_activation": "linear",
        }
        payload["task"] = {
            "task_type": "training",
            "period": {
                "start_index": 0,
                "train_end_index": 30,
                "validation_end_index": 40,
                "test_end_index": 50,
            },
        }
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(config.model.recurrent_units, 20)
        self.assertEqual(config.model.gradient_clip_norm, 4.0)

        payload["model"]["feature_roles"] = []
        with self.assertRaises(ValidationError):
            ExperimentConfigV1.model_validate(payload)

    def test_stacked_recurrent_variants_validate_every_layer_width(self):
        for model_id in (
            "lstm.tensorflow_stacked_s2v.v2",
            "gru.tensorflow_stacked_s2v.v2",
        ):
            payload = valid_payload()
            payload["schema_version"] = "1.19"
            payload["model"] = {
                "model_id": model_id,
                "feature_roles": ["precipitation", "temperature"],
                "lookback": 8,
                "recurrent_layers": [24, 12, 6],
                "output_activation": "linear",
            }
            payload["task"] = {
                "task_type": "training",
                "period": {
                    "start_index": 0,
                    "train_end_index": 30,
                    "validation_end_index": 40,
                    "test_end_index": 50,
                },
            }
            config = ExperimentConfigV1.model_validate(payload)
            self.assertEqual(config.model.recurrent_layers, (24, 12, 6))
            self.assertEqual(config.model.recurrent_units, 24)

            payload["model"]["recurrent_layers"] = [24, 0]
            with self.assertRaises(ValidationError):
                ExperimentConfigV1.model_validate(payload)

    def test_cnn1d_validates_the_layer_stack_and_receptive_field(self):
        payload = valid_payload()
        payload["schema_version"] = "1.10"
        payload["model"] = {
            "model_id": "cnn1d.tensorflow_s2v.v1",
            "feature_roles": ["precipitation", "temperature"],
            "lookback": 8,
            "convolution_layers": [
                {"filters": 12, "kernel_size": 4, "activation": "relu"},
                {"filters": 6, "kernel_size": 3, "activation": "tanh"},
            ],
            "output_activation": "linear",
        }
        payload["task"] = {
            "task_type": "training",
            "period": {
                "start_index": 0,
                "train_end_index": 10,
                "validation_end_index": 15,
                "test_end_index": 20,
            },
        }
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(len(config.model.convolution_layers), 2)

        payload["model"]["lookback"] = 5
        with self.assertRaisesRegex(ValidationError, "receptive field"):
            ExperimentConfigV1.model_validate(payload)


    def test_deepdiscover_requires_discovery_training_and_accepts_snow_calibration(self):
        payload = valid_payload()
        payload["schema_version"] = "1.12"
        payload["model"] = {
            "model_id": "deepdiscover.numpy_instantaneous_s2s.v1",
            "snow": {"module_id": "snow.none.v1"},
            "initial_state": [0.0, 18.0],
            "process_count": 3,
            "hidden_units": 12,
            "train_input_switches": True,
            "train_process_switches": True,
            "loss": {
                "decorrelation_weight": 0.02,
                "mass_balance_weight": 0.1,
                "smoothness_weight": 0.01,
                "smoothness_threshold": 0.1,
                "binary_weight": 0.01,
                "sparsity_weight": 0.001,
                "target_process_correlation": 0.0,
            },
        }
        with self.assertRaisesRegex(ValidationError, "discovery_training"):
            ExperimentConfigV1.model_validate(payload)

        payload["task"] = {
            "task_type": "discovery_training",
            "period": {
                "start_index": 0,
                "train_end_index": 10,
                "validation_end_index": 15,
                "test_end_index": 20,
            },
            "warmup_steps": 2,
            "training": {
                "optimizer_id": "adam.numpy.full_sequence.v1",
                "epochs": 20,
                "learning_rate": 0.005,
                "patience": 5,
                "gradient_clip_norm": 4.0,
                "seed": 7,
            },
        }
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(config.model.process_count, 3)
        self.assertEqual(
            config.task.training.optimizer_id,
            "adam.numpy.full_sequence.v1",
        )

        payload["model"]["snow"] = {
            "module_id": "snow.degree_day.hard.v1",
            "calibration": {
                "degree_day_factor": {
                    "mode": "calibrate",
                    "lower_bound": 0.0,
                    "upper_bound": 5.0,
                }
            },
        }
        config = ExperimentConfigV1.model_validate(payload)
        self.assertEqual(
            config.model.snow.calibration.degree_day_factor.mode,
            "calibrate",
        )



if __name__ == "__main__":
    unittest.main()
