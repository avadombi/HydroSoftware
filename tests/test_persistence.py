from pathlib import Path
import json
import tempfile
import unittest

from hydrosoftware.configuration import load_experiment_config
from hydrosoftware.execution.experiments import ExperimentRunner
from hydrosoftware.execution.persistence import ProjectStore, ProjectStoreError


class ProjectStoreTests(unittest.TestCase):
    def test_experiment_and_completed_run_survive_a_new_store_instance(self):
        project_root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(project_root / "examples/exp_hydro_calibration.yml")

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            data_dir = workspace / "examples" / "data"
            data_dir.mkdir(parents=True)
            source = project_root / config.data.path
            (data_dir / source.name).write_bytes(source.read_bytes())

            store = ProjectStore(workspace)
            saved = store.save_experiment(config)
            self.assertEqual(saved.experiment_id, config.experiment_id)
            self.assertEqual(saved.run_count, 0)

            result = ExperimentRunner(workspace).run(config, task_id="run-persisted")
            store.save_run(config, result)
            store.save_run(config, result)

            reopened = ProjectStore(workspace)
            experiments = reopened.list_experiments()
            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments[0].latest_run_id, "run-persisted")
            self.assertEqual(experiments[0].run_count, 1)

            archive = reopened.load_run("run-persisted")
            self.assertAlmostEqual(archive["result"]["calibration"]["best_score"], 1.0)
            self.assertEqual(
                [metric["symbol"] for metric in archive["result"]["evaluation_metrics"]],
                ["NSE", "KGE", "MSE", "RMSE", "NRMSE", "MAE", "PBIAS"],
            )
            self.assertEqual(len(reopened.list_runs(config.experiment_id)), 1)

    def test_unsafe_identifiers_are_rejected_before_path_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            with self.assertRaises(ProjectStoreError) as context:
                store.load_run("../outside")
            self.assertEqual(context.exception.code, "invalid_run_identifier")



if __name__ == "__main__":
    unittest.main()
