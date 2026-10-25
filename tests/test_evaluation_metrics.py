import unittest

import numpy as np

from hydrosoftware.core.evaluation import evaluate_metrics


class EvaluationMetricTests(unittest.TestCase):
    def test_perfect_simulation_has_expected_registered_metrics(self):
        observed = np.array([1.0, 2.0, 4.0, 8.0])
        results = {
            result.metric_id: result
            for result in evaluate_metrics(observed, observed.copy(), np.ones(4, dtype=bool))
        }

        self.assertEqual(
            set(results),
            {
                "nash_sutcliffe_efficiency",
                "root_mean_square_error",
                "mean_square_error",
                "mean_absolute_error",
                "kling_gupta_efficiency_2009",
                "percent_bias_moriasi",
                "range_normalized_root_mean_square_error",
            },
        )
        self.assertAlmostEqual(results["nash_sutcliffe_efficiency"].value, 1.0)
        self.assertAlmostEqual(results["kling_gupta_efficiency_2009"].value, 1.0)
        self.assertAlmostEqual(results["root_mean_square_error"].value, 0.0)
        self.assertAlmostEqual(results["mean_square_error"].value, 0.0)
        self.assertAlmostEqual(results["mean_absolute_error"].value, 0.0)
        self.assertAlmostEqual(results["percent_bias_moriasi"].value, 0.0)
        self.assertAlmostEqual(
            results["range_normalized_root_mean_square_error"].value, 0.0
        )

    def test_warmup_and_missing_observations_share_one_pairing_mask(self):
        observed = np.array([1000.0, np.nan, 1.0, 3.0])
        simulated = np.array([-1000.0, 500.0, 2.0, 2.0])
        mask = np.array([False, True, True, True])
        results = {item.symbol: item for item in evaluate_metrics(observed, simulated, mask)}

        self.assertAlmostEqual(results["RMSE"].value, 1.0)
        self.assertAlmostEqual(results["NRMSE"].value, 0.5)
        self.assertAlmostEqual(results["PBIAS"].value, 0.0)

    def test_pbias_sign_is_positive_for_model_underestimation(self):
        observed = np.array([2.0, 4.0])
        simulated = np.array([1.0, 2.0])
        results = {item.symbol: item for item in evaluate_metrics(observed, simulated, np.ones(2, dtype=bool))}

        self.assertAlmostEqual(results["PBIAS"].value, 50.0)

    def test_original_kge_uses_correlation_variability_and_mean_ratios(self):
        observed = np.array([1.0, 2.0, 3.0])
        simulated = np.array([1.0, 2.0, 4.0])
        results = {
            item.symbol: item
            for item in evaluate_metrics(
                observed, simulated, np.ones(3, dtype=bool)
            )
        }

        self.assertAlmostEqual(results["KGE"].value, 0.44647913321266675)

    def test_degenerate_metrics_are_explicitly_unavailable(self):
        observed = np.array([2.0, 2.0, 2.0])
        simulated = np.array([1.0, 2.0, 3.0])
        results = {item.symbol: item for item in evaluate_metrics(observed, simulated, np.ones(3, dtype=bool))}

        self.assertIsNone(results["NSE"].value)
        self.assertIn("constant", results["NSE"].unavailable_reason)
        self.assertIsNone(results["NRMSE"].value)
        self.assertIn("zero-range", results["NRMSE"].unavailable_reason)
        self.assertIsNotNone(results["RMSE"].value)


if __name__ == "__main__":
    unittest.main()
