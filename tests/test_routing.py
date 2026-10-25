import unittest

import numpy as np

from hydrosoftware.models.conceptual.hbv import HbvModel, HbvParameters
from hydrosoftware.processes.routing import (
    MAXBAS_MEMORY_SIZE,
    TriangularMaxbasRouting,
    route_simulation_result,
    triangular_maxbas_weights,
)
from hydrosoftware.processes.snow import NoSnowModule


class MaxbasRoutingTests(unittest.TestCase):
    def test_published_triangle_is_integrated_over_daily_intervals(self):
        np.testing.assert_allclose(
            triangular_maxbas_weights(3),
            np.array([2 / 9, 5 / 9, 2 / 9, 0, 0, 0, 0]),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            triangular_maxbas_weights(4),
            np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8, 0, 0, 0]),
            atol=1e-14,
        )

    def test_impulse_is_conservative_and_maxbas_one_is_identity(self):
        impulse = np.array([1.0, 0.0, 0.0, 0.0])
        routed = TriangularMaxbasRouting(3).route(impulse)
        np.testing.assert_allclose(routed.routed_streamflow, [2 / 9, 5 / 9, 2 / 9, 0])
        self.assertAlmostEqual(routed.routed_streamflow.sum(), 1.0)
        self.assertLess(np.max(np.abs(routed.mass_balance_error)), 1e-14)

        identity = TriangularMaxbasRouting(1).route(impulse)
        np.testing.assert_allclose(identity.routed_streamflow, impulse)
        np.testing.assert_allclose(identity.states, 0.0)

    def test_routing_restart_preserves_pending_contributions_exactly(self):
        inflow = np.array([2.0, 0.0, 3.0, 1.0, 0.0, 4.0, 0.0])
        routing = TriangularMaxbasRouting(7)
        full = routing.route(inflow)
        first = routing.route(inflow[:3])
        second = routing.route(
            inflow[3:], initial_state=tuple(first.states[-1])
        )
        np.testing.assert_allclose(
            np.concatenate((first.routed_streamflow, second.routed_streamflow)),
            full.routed_streamflow,
        )
        np.testing.assert_allclose(
            np.vstack((first.states, second.states[1:])), full.states
        )

    def test_composed_hbv_result_has_restart_states_and_combined_balance(self):
        parameters = HbvParameters(500.0, 2.0, 0.7, 1.5, 30.0, 0.3, 0.1, 0.02)
        base = HbvModel(parameters, NoSnowModule()).simulate(
            np.array([8.0, 0.0, 4.0, 1.0]),
            np.full(4, 5.0),
            np.full(4, 1.0),
            initial_state=(0.0, 300.0, 20.0, 80.0),
        )
        routed = route_simulation_result(base, TriangularMaxbasRouting(3))
        self.assertEqual(routed.states.shape, (5, 4 + MAXBAS_MEMORY_SIZE))
        self.assertIn("routed_streamflow", routed.flux_names)
        self.assertEqual(
            routed.diagnostics["routing_process_id"],
            "routing.maxbas.triangular.v1",
        )
        self.assertLess(
            routed.diagnostics["maximum_absolute_mass_balance_error"], 1e-10
        )


if __name__ == "__main__":
    unittest.main()
