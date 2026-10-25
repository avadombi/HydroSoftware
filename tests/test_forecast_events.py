import copy
from datetime import date, timedelta
import tempfile
import unittest

from hydrosoftware.execution.forecast_events import (
    EVENT_VERIFICATION_METHOD_ID,
    verify_threshold_event_decisions,
)
from hydrosoftware.execution.forecast_verification import ForecastVerificationError
from hydrosoftware.execution.persistence import ProjectStore
from tests.test_forecast_verification import ensemble_case


def event_cases(count: int = 30) -> list[dict]:
    cases = []
    for index in range(count):
        observations = [
            1.0 if index % 2 == 0 else 3.0,
            3.0 if index % 3 == 0 else 1.0,
        ]
        predictions = [
            [value + 0.6 + anomaly for value in observations]
            for anomaly in (-0.1, 0.3, 0.7)
        ]
        issue = (date(2026, 1, 1) + timedelta(days=index)).isoformat()
        case = ensemble_case(issue, predictions, observations)
        case["forecast_ensemble"]["issue_time"] = issue
        cases.append(case)
    return cases


class ForecastEventVerificationTests(unittest.TestCase):
    def test_event_probabilities_are_evaluated_only_on_later_origins(self):
        result = verify_threshold_event_decisions(
            list(reversed(event_cases())),
            training_case_count=20,
            threshold_mode="absolute",
            threshold_value=2.0,
            probability_bin_count=10,
            minimum_evaluation_cases_per_lead=5,
        )
        self.assertEqual(result["method_id"], EVENT_VERIFICATION_METHOD_ID)
        self.assertEqual(result["threshold"]["value"], 2.0)
        self.assertEqual(result["split"]["training_case_count"], 20)
        self.assertEqual(result["split"]["evaluation_case_count"], 10)
        raw = result["aggregate"]["raw"]
        adjusted = result["aggregate"]["adjusted"]
        self.assertEqual(raw["scored_cases"], 20)
        self.assertEqual(raw["event_count"] + raw["nonevent_count"], 20)
        self.assertAlmostEqual(raw["roc_auc"], 1.0)
        self.assertLess(adjusted["brier_score"], raw["brier_score"])
        self.assertGreater(result["aggregate"]["adjusted_brier_skill_vs_raw"], 0.0)
        self.assertEqual(
            sum(row["count"] for row in raw["reliability_bins"]),
            20,
        )
        self.assertEqual(len(raw["cost_loss_curve"]), 19)
        self.assertTrue(
            all(lead["reported"] for lead in result["lead_time_diagnostics"])
        )

    def test_evaluation_observations_cannot_change_threshold_or_climatology(self):
        original = event_cases()
        changed = copy.deepcopy(original)
        for case in changed[20:]:
            case["observations"] = [10.0, 10.0]
        first = verify_threshold_event_decisions(
            original,
            training_case_count=20,
            threshold_mode="training_quantile",
            training_quantile=0.75,
            minimum_evaluation_cases_per_lead=5,
        )
        second = verify_threshold_event_decisions(
            changed,
            training_case_count=20,
            threshold_mode="training_quantile",
            training_quantile=0.75,
            minimum_evaluation_cases_per_lead=5,
        )
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertEqual(first["climatology"], second["climatology"])
        self.assertNotEqual(first["aggregate"], second["aggregate"])

    def test_threshold_and_reporting_guards_are_explicit(self):
        cases = event_cases(10)
        with self.assertRaisesRegex(ForecastVerificationError, "absolute"):
            verify_threshold_event_decisions(
                cases,
                training_case_count=6,
                threshold_mode="absolute",
            )
        with self.assertRaisesRegex(ForecastVerificationError, "quantile"):
            verify_threshold_event_decisions(
                cases,
                training_case_count=6,
                threshold_mode="training_quantile",
                training_quantile=0.2,
            )
        with self.assertRaisesRegex(ForecastVerificationError, "probability bins"):
            verify_threshold_event_decisions(
                cases,
                training_case_count=6,
                threshold_mode="absolute",
                threshold_value=2.0,
                probability_bin_count=2,
            )

    def test_event_analysis_survives_store_restart(self):
        result = verify_threshold_event_decisions(
            event_cases(),
            training_case_count=20,
            threshold_mode="absolute",
            threshold_value=2.0,
            minimum_evaluation_cases_per_lead=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            summary = store.save_forecast_event_verification(result)
            reopened = ProjectStore(directory)
            summaries = reopened.list_forecast_event_verifications()
            archive = reopened.load_forecast_event_verification(
                result["task_id"]
            )
        self.assertEqual(summary, summaries[0])
        self.assertEqual(archive["result"], result)
        self.assertEqual(summary["threshold_value"], 2.0)


if __name__ == "__main__":
    unittest.main()
