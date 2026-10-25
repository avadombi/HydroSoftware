from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from hydrosoftware.configuration import ExperimentConfigV1
from hydrosoftware.models.paml import PAML_PRNN_CNN_MANIFEST
from hydrosoftware.models.paml_tensorflow import build_model


def payload(backbone: str = "exp_hydro.mass_conservative.v1") -> dict:
    physical = {"model_id": backbone}
    return {
        "schema_version": "1.21",
        "experiment_id": "paml-check",
        "name": "PaML check",
        "data": {
            "kind": "csv",
            "path": "data.csv",
            "columns": {
                "precipitation": "p",
                "temperature": "t",
                "potential_evapotranspiration": "pet",
                "streamflow": "q",
            },
        },
        "model": {
            "model_id": "paml.tensorflow_prnn_cnn_s2s.v1",
            "physical_model": physical,
            "cnn_filters": 8,
            "cnn_kernel_size": 5,
            "training_strategy": "joint_end_to_end",
            "enforce_nonnegative_output": True,
        },
        "task": {
            "task_type": "paml_training",
            "period": {
                "start_index": 0,
                "train_end_index": 40,
                "validation_end_index": 50,
                "test_end_index": 60,
            },
            "target_role": "streamflow",
            "lead_steps": 0,
            "stride": 1,
            "warmup_steps": 5,
            "objective": {"objective_id": "nash_sutcliffe_efficiency"},
            "training": {
                "optimizer_id": "adam.tensorflow.v1",
                "epochs": 2,
                "batch_size": 16,
                "learning_rate": 0.001,
                "patience": 2,
                "minimum_delta": 0.0,
                "gradient_clip_norm": 5.0,
                "seed": 7,
                "shuffle": False,
            },
        },
    }


class PamlConfigurationTests(unittest.TestCase):
    def test_manifest_declares_joint_prnn_cnn_model(self):
        self.assertEqual(PAML_PRNN_CNN_MANIFEST.model_id, "paml.tensorflow_prnn_cnn_s2s.v1")
        self.assertEqual(PAML_PRNN_CNN_MANIFEST.metadata["training_strategy"], "joint_end_to_end")

    def test_exp_hydro_and_hbv_backbones_validate(self):
        for backbone in ("exp_hydro.mass_conservative.v1", "hbv.mass_conservative_daily.v1"):
            with self.subTest(backbone=backbone):
                config = ExperimentConfigV1.model_validate(payload(backbone))
                self.assertEqual(config.model.physical_model.model_id, backbone)
                self.assertEqual(config.task.task_type, "paml_training")

    def test_unused_conceptual_backbone_options_are_rejected(self):
        invalid = payload()
        invalid["model"]["physical_model"]["substeps"] = 8
        with self.assertRaisesRegex(ValueError, "Extra inputs are not permitted"):
            ExperimentConfigV1.model_validate(invalid)


@unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow is not installed")
class PamlTensorFlowTests(unittest.TestCase):
    def test_both_backbones_build_and_emit_aligned_sequences(self):
        center = np.zeros(4, dtype=np.float32)
        scale = np.ones(4, dtype=np.float32)
        x = np.zeros((1, 12, 3), dtype=np.float32)
        for backbone in ("exp_hydro.mass_conservative.v1", "hbv.mass_conservative_daily.v1"):
            with self.subTest(backbone=backbone):
                model, _ = build_model(backbone, center, scale, filters=4, kernel_size=3, seed=4)
                q, physical = model(x, training=False)
                self.assertEqual(tuple(q.shape), (1, 12, 1))
                self.assertEqual(physical.shape[1], 12)


class PamlSourceRegressionTests(unittest.TestCase):
    def test_physical_discharge_statistics_drive_paml_physical_channel_scaling(self):
        from pathlib import Path
        source = (Path(__file__).parents[1] / "src" / "hydrosoftware" / "models" / "paml_tensorflow.py").read_text(encoding="utf-8")
        self.assertIn("_,physical0=provisional_model(train_prefix_tf,training=False)", source)
        self.assertIn("qphys0=np.asarray(physical0[0,:,0],dtype=np.float32)", source)
        self.assertIn("validation_x_tf=tf.convert_to_tensor(x[None,:va,:])", source)
        self.assertIn("full_x_tf=tf.convert_to_tensor(x[None,...])", source)

    def test_paml_receives_the_selected_training_objective(self):
        from pathlib import Path
        source = (Path(__file__).parents[1] / "src" / "hydrosoftware" / "execution" / "experiments.py").read_text(encoding="utf-8")
        block = source.split("fit = fit_paml(", 1)[1].split(")", 1)[0]
        self.assertIn("objective_id=config.task.objective.objective_id", block)


if __name__ == "__main__":
    unittest.main()
