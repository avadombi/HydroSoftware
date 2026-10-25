import copy
import tempfile
import unittest

import numpy as np

from hydrosoftware.execution.forecast_ensembles import empirical_crps
from hydrosoftware.execution.forecast_verification import (
    ForecastVerificationError,
    verify_forcing_ensemble_hindcasts,
)
from hydrosoftware.execution.persistence import ProjectStore


def ensemble_case(
    issue: str,
    member_predictions: list[list[float]],
    observations: list[float],
    *,
    source_task_id: str = "source-run",
) -> dict:
    horizon = len(observations)
    timestamps = [f"{issue}-day-{lead + 1}" for lead in range(horizon)]
    members = [
        {
            "member_id": f"member-{index + 1:02d}",
            "predictions": [[value] for value in prediction],
            "final_state": [],
            "evaluation_metrics": [],
            "initialization_ensemble_lower": None,
            "initialization_ensemble_upper": None,
        }
        for index, prediction in enumerate(member_predictions)
    ]
    means = np.mean(np.asarray(member_predictions, dtype=float), axis=0)
    return {
        "task_id": f"forecast-ensemble-{issue}",
        "state": "completed",
        "run_kind": "forecast",
        "source_task_id": source_task_id,
        "experiment_id": "verification-test",
        "configuration_hash": "same-config",
        "model_id": "exp_hydro.mass_conservative.v1",
        "best_simulation": {
            "outputs": [[value] for value in means],
            "states": [[] for _ in range(horizon + 1)],
            "fluxes": [[] for _ in range(horizon)],
            "output_names": ["streamflow"],
            "state_names": [],
            "flux_names": [],
            "diagnostics": {},
        },
        "timestamps": timestamps,
        "observations": observations,
        "forecast_ensemble": {
            "method_id": "meteorological_forcing_ensemble.v1",
            "issue_id": issue,
            "issue_time": issue,
            "scope": "meteorological_forcing_scenarios",
            "member_count": len(members),
            "member_ids": [member["member_id"] for member in members],
            "aggregation": "arithmetic_mean",
            "lower_quantile": 0.1,
            "upper_quantile": 0.9,
            "members": members,
            "lead_time_diagnostics": [],
            "aggregate_scores": {"scored_leads": horizon},
            "warnings": [],
        },
    }


class ForecastVerificationTests(unittest.TestCase):
    def setUp(self):
        self.first = ensemble_case(
            "2026-01-01",
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
            [1.0, 2.5],
        )
        self.second = ensemble_case(
            "2026-01-08",
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            [2.0, 2.5],
        )

    def test_lead_scores_coverage_spread_and_ranks_are_reproducible(self):
        result = verify_forcing_ensemble_hindcasts(
            [self.second, self.first],
            seed=17,
        )
        repeated = verify_forcing_ensemble_hindcasts(
            [self.first, self.second],
            seed=17,
        )
        self.assertEqual(
            result["method_id"],
            "meteorological_ensemble_hindcast_verification.v1",
        )
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["member_count"], 3)
        self.assertEqual(result["horizon"], 2)
        lead_one = result["lead_time_diagnostics"][0]
        expected_crps = np.mean(
            [
                empirical_crps([0.0, 1.0, 2.0], 1.0),
                empirical_crps([1.0, 2.0, 3.0], 2.0),
            ]
        )
        self.assertAlmostEqual(lead_one["mean_crps"], expected_crps)
        self.assertAlmostEqual(lead_one["mean_error_bias"], 0.0)
        self.assertAlmostEqual(lead_one["root_mean_square_error"], 0.0)
        self.assertIsNone(lead_one["spread_skill_ratio"])
        self.assertEqual(sum(lead_one["rank_counts"]), 2)
        self.assertEqual(
            result["aggregate"]["rank_counts"],
            repeated["aggregate"]["rank_counts"],
        )
        self.assertEqual(
            result["lead_time_diagnostics"][0]["rank_counts"],
            repeated["lead_time_diagnostics"][0]["rank_counts"],
        )
        self.assertEqual(result["aggregate"]["scored_cases"], 4)
        self.assertEqual(
            len(result["aggregate"]["deterministic_metrics"]),
            7,
        )

    def test_nonfinite_observations_are_excluded_by_lead(self):
        second = copy.deepcopy(self.second)
        second["observations"][1] = float("nan")
        result = verify_forcing_ensemble_hindcasts(
            [self.first, second],
            minimum_cases_per_lead=2,
        )
        self.assertTrue(result["lead_time_diagnostics"][0]["reported"])
        self.assertFalse(result["lead_time_diagnostics"][1]["reported"])
        self.assertEqual(
            result["lead_time_diagnostics"][1]["available_cases"],
            1,
        )

    def test_protocol_and_issue_guards_reject_ambiguous_cases(self):
        changed_source = copy.deepcopy(self.second)
        changed_source["source_task_id"] = "another-source"
        with self.assertRaisesRegex(
            ForecastVerificationError,
            "same frozen source model",
        ):
            verify_forcing_ensemble_hindcasts([self.first, changed_source])

        duplicate_issue = copy.deepcopy(self.second)
        duplicate_issue["forecast_ensemble"]["issue_id"] = "2026-01-01"
        with self.assertRaisesRegex(
            ForecastVerificationError,
            "unique issue identifier",
        ):
            verify_forcing_ensemble_hindcasts([self.first, duplicate_issue])

    def test_forecasts_and_verification_survive_store_restart(self):
        result = verify_forcing_ensemble_hindcasts(
            [self.first, self.second],
            seed=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            store.save_forecast_ensemble(self.first)
            store.save_forecast_ensemble(self.second)
            store.save_forecast_verification(result)

            reopened = ProjectStore(directory)
            forecasts = reopened.list_forecast_ensembles()
            self.assertEqual(len(forecasts), 2)
            self.assertEqual({item["issue_id"] for item in forecasts}, {
                "2026-01-01",
                "2026-01-08",
            })
            verifications = reopened.list_forecast_verifications()
            self.assertEqual(len(verifications), 1)
            self.assertEqual(verifications[0]["case_count"], 2)
            archive = reopened.load_forecast_verification(result["task_id"])
            self.assertEqual(
                archive["result"]["aggregate"]["scored_cases"],
                4,
            )


if __name__ == "__main__":
    unittest.main()
