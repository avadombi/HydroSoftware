import copy
import tempfile
import unittest

import numpy as np

from hydrosoftware.execution.forecast_postprocessing import (
    FORECAST_POSTPROCESSING_METHOD_ID,
    postprocess_forcing_ensemble_hindcasts,
)
from hydrosoftware.execution.forecast_verification import (
    ForecastVerificationError,
)
from hydrosoftware.execution.persistence import ProjectStore
from tests.test_forecast_verification import ensemble_case


def biased_cases(count: int = 10) -> list[dict]:
    cases = []
    for index in range(count):
        observations = [
            1.5 + 0.15 * index,
            2.0 + 0.08 * index,
        ]
        predictions = []
        for anomaly in (-0.3, 0.0, 0.3):
            predictions.append(
                [
                    observations[0] + 0.8 + anomaly,
                    observations[1] + 0.6 + 1.4 * anomaly,
                ]
            )
        case = ensemble_case(
            f"2026-01-{1 + index:02d}",
            predictions,
            observations,
        )
        case["forecast_ensemble"]["issue_time"] = f"2026-01-{1 + index:02d}"
        cases.append(case)
    return cases


class ForecastPostprocessingTests(unittest.TestCase):
    def test_chronological_fit_improves_independent_biased_cases(self):
        result = postprocess_forcing_ensemble_hindcasts(
            list(reversed(biased_cases())),
            training_case_count=6,
            seed=7,
        )
        self.assertEqual(result["method_id"], FORECAST_POSTPROCESSING_METHOD_ID)
        self.assertEqual(result["split"]["training_case_count"], 6)
        self.assertEqual(result["split"]["evaluation_case_count"], 4)
        self.assertEqual(
            result["split"]["training_task_ids"][0],
            "forecast-ensemble-2026-01-01",
        )
        self.assertEqual(
            result["split"]["evaluation_task_ids"][0],
            "forecast-ensemble-2026-01-07",
        )
        self.assertGreater(result["aggregate"]["crps_skill_score"], 0.0)
        self.assertLess(
            abs(result["aggregate"]["adjusted"]["mean_error_bias"]),
            abs(result["aggregate"]["raw"]["mean_error_bias"]),
        )
        self.assertEqual(len(result["parameters"]), 2)
        self.assertTrue(
            all(lead["reported"] for lead in result["lead_time_diagnostics"])
        )
        adjusted = np.asarray(
            result["evaluation_cases"][0]["adjusted_member_predictions"],
            dtype=float,
        )
        self.assertTrue(np.all(adjusted >= 0.0))

    def test_evaluation_observations_cannot_change_fitted_parameters(self):
        original = biased_cases()
        changed = copy.deepcopy(original)
        for case in changed[6:]:
            case["observations"] = [99.0, 101.0]
        first = postprocess_forcing_ensemble_hindcasts(
            original,
            training_case_count=6,
        )
        second = postprocess_forcing_ensemble_hindcasts(
            changed,
            training_case_count=6,
        )
        self.assertEqual(first["parameters"], second["parameters"])
        self.assertNotEqual(
            first["aggregate"]["adjusted"]["mean_crps"],
            second["aggregate"]["adjusted"]["mean_crps"],
        )

    def test_case_count_split_and_iso_origin_are_guarded(self):
        with self.assertRaisesRegex(
            ForecastVerificationError,
            "at least seven",
        ):
            postprocess_forcing_ensemble_hindcasts(
                biased_cases(6),
                training_case_count=5,
            )
        with self.assertRaisesRegex(
            ForecastVerificationError,
            "leave at least",
        ):
            postprocess_forcing_ensemble_hindcasts(
                biased_cases(8),
                training_case_count=7,
            )
        invalid = biased_cases()
        invalid[0]["forecast_ensemble"]["issue_time"] = "not-a-date"
        with self.assertRaisesRegex(
            ForecastVerificationError,
            "not an ISO",
        ):
            postprocess_forcing_ensemble_hindcasts(
                invalid,
                training_case_count=6,
            )

    def test_postprocessing_survives_store_restart(self):
        result = postprocess_forcing_ensemble_hindcasts(
            biased_cases(),
            training_case_count=6,
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            store.save_forecast_postprocessing(result)
            reopened = ProjectStore(directory)
            summaries = reopened.list_forecast_postprocessings()
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["training_case_count"], 6)
            self.assertEqual(summaries[0]["evaluation_case_count"], 4)
            archive = reopened.load_forecast_postprocessing(result["task_id"])
            self.assertEqual(
                archive["result"]["aggregate"]["crps_skill_score"],
                result["aggregate"]["crps_skill_score"],
            )


if __name__ == "__main__":
    unittest.main()
