import unittest

import numpy as np

from hydrosoftware.core.optimization import (
    OBJECTIVE_IDS,
    objective_definition,
    objective_loss_and_gradient,
    objective_loss_from_score,
    objective_optimization_score,
    objective_score,
    validate_objective_observations,
)


class OptimizationObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.observed = np.array([1.0, 2.5, 4.0, 7.5, 5.0], dtype=float)
        self.simulated = np.array([1.2, 2.1, 4.8, 6.7, 5.4], dtype=float)

    def test_registry_exposes_six_fitting_metrics_with_explicit_directions(self):
        self.assertEqual(
            OBJECTIVE_IDS,
            (
                "nash_sutcliffe_efficiency",
                "kling_gupta_efficiency_2009",
                "mean_square_error",
                "root_mean_square_error",
                "range_normalized_root_mean_square_error",
                "mean_absolute_error",
            ),
        )
        self.assertEqual(
            objective_definition("nash_sutcliffe_efficiency").direction,
            "maximize",
        )
        self.assertEqual(
            objective_definition("kling_gupta_efficiency_2009").direction,
            "maximize",
        )
        self.assertTrue(
            objective_definition("nash_sutcliffe_efficiency").requires_global_batch
        )
        self.assertTrue(
            objective_definition(
                "kling_gupta_efficiency_2009"
            ).requires_global_batch
        )
        self.assertTrue(
            objective_definition(
                "range_normalized_root_mean_square_error"
            ).requires_global_batch
        )
        for objective_id in OBJECTIVE_IDS[2:]:
            self.assertEqual(
                objective_definition(objective_id).direction,
                "minimize",
            )

    def test_analytical_gradients_match_finite_differences_for_every_objective(self):
        epsilon = 1e-6
        for objective_id in OBJECTIVE_IDS:
            with self.subTest(objective_id=objective_id):
                loss, gradient = objective_loss_and_gradient(
                    self.observed,
                    self.simulated,
                    objective_id,
                )
                numerical = np.zeros_like(self.simulated)
                for index in range(len(self.simulated)):
                    upper = self.simulated.copy()
                    lower = self.simulated.copy()
                    upper[index] += epsilon
                    lower[index] -= epsilon
                    upper_loss = objective_loss_and_gradient(
                        self.observed,
                        upper,
                        objective_id,
                    )[0]
                    lower_loss = objective_loss_and_gradient(
                        self.observed,
                        lower,
                        objective_id,
                    )[0]
                    numerical[index] = (upper_loss - lower_loss) / (2.0 * epsilon)
                self.assertTrue(np.isfinite(loss))
                np.testing.assert_allclose(
                    gradient,
                    numerical,
                    rtol=2e-5,
                    atol=2e-6,
                )

    def test_reported_scores_and_minimized_losses_share_one_contract(self):
        for objective_id in OBJECTIVE_IDS:
            with self.subTest(objective_id=objective_id):
                score = objective_score(
                    self.observed,
                    self.simulated,
                    objective_id,
                )
                loss, _ = objective_loss_and_gradient(
                    self.observed,
                    self.simulated,
                    objective_id,
                )
                self.assertAlmostEqual(
                    objective_loss_from_score(score, objective_id),
                    loss,
                    places=10,
                )

    def test_degenerate_efficiency_and_normalized_targets_are_rejected(self):
        constant = np.ones(4, dtype=float)
        for objective_id in (
            "nash_sutcliffe_efficiency",
            "kling_gupta_efficiency_2009",
            "range_normalized_root_mean_square_error",
        ):
            with self.subTest(objective_id=objective_id):
                with self.assertRaises(ValueError):
                    validate_objective_observations(constant, objective_id)

    def test_kge_training_score_remains_finite_for_constant_initial_prediction(self):
        constant_prediction = np.full_like(
            self.observed,
            self.observed.mean(),
        )
        score, loss = objective_optimization_score(
            self.observed,
            constant_prediction,
            "kling_gupta_efficiency_2009",
        )
        self.assertTrue(np.isfinite(score))
        self.assertTrue(np.isfinite(loss))
        with self.assertRaisesRegex(ValueError, "constant simulated"):
            objective_score(
                self.observed,
                constant_prediction,
                "kling_gupta_efficiency_2009",
            )


if __name__ == "__main__":
    unittest.main()
