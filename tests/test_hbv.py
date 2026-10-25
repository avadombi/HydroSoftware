import unittest

import numpy as np

from hydrosoftware.models.conceptual.hbv import HbvModel, HbvParameters
from hydrosoftware.processes.snow import DegreeDaySnowModule, NoSnowModule


class HbvTests(unittest.TestCase):
    def setUp(self):
        self.parameters = HbvParameters(
            field_capacity=500.0,
            recharge_exponent=2.0,
            evap_reduction_threshold=0.7,
            maximum_percolation=1.5,
            upper_zone_threshold=30.0,
            fast_recession=0.30,
            intermediate_recession=0.10,
            slow_recession=0.02,
        )

    def test_mass_balance_and_non_negative_states(self):
        rng = np.random.default_rng(2026)
        n = 500
        result = HbvModel(self.parameters, NoSnowModule()).simulate(
            rng.gamma(1.2, 5.0, n),
            np.full(n, 8.0),
            rng.uniform(0.0, 5.0, n),
            initial_state=(0.0, 250.0, 20.0, 80.0),
        )
        self.assertEqual(result.states.shape, (n + 1, 4))
        self.assertTrue(np.all(result.states >= 0.0))
        self.assertTrue(np.all(result.outputs >= 0.0))
        self.assertLess(
            result.diagnostics["maximum_absolute_mass_balance_error"],
            1e-10,
        )

    def test_percolation_is_limited_by_upper_zone_water(self):
        result = HbvModel(self.parameters, NoSnowModule()).simulate(
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            initial_state=(0.0, 0.0, 0.4, 0.0),
        )
        percolation = result.fluxes[0, result.flux_names.index("percolation")]
        self.assertAlmostEqual(percolation, 0.4)
        self.assertAlmostEqual(result.states[1, 2], 0.0)
        self.assertGreater(result.states[1, 3], 0.0)

    def test_soil_capacity_excess_reaches_upper_response_zone(self):
        parameters = HbvParameters(
            field_capacity=100.0,
            recharge_exponent=2.0,
            evap_reduction_threshold=0.7,
            maximum_percolation=0.0,
            upper_zone_threshold=100.0,
            fast_recession=0.05,
            intermediate_recession=0.01,
            slow_recession=0.001,
        )
        result = HbvModel(parameters, NoSnowModule()).simulate(
            np.array([100.0]),
            np.array([10.0]),
            np.array([0.0]),
            initial_state=(0.0, 99.0, 0.0, 0.0),
        )
        excess = result.fluxes[0, result.flux_names.index("soil_excess")]
        recharge = result.fluxes[
            0, result.flux_names.index("groundwater_recharge")
        ]
        self.assertGreater(excess, 0.0)
        self.assertGreater(recharge, excess)
        self.assertLessEqual(result.states[1, 1], parameters.field_capacity)
        self.assertGreater(result.states[1, 2], 0.0)

    def test_restart_is_exact(self):
        precipitation = np.array([0.0, 12.0, 4.0, 0.0, 1.0, 8.0])
        temperature = np.array([-4.0, -2.0, 3.0, 5.0, 1.0, 6.0])
        pet = np.full(6, 2.0)
        model = HbvModel(
            self.parameters,
            DegreeDaySnowModule(0.0, 0.0, 2.0),
        )
        initial = (10.0, 200.0, 40.0, 80.0)
        full = model.simulate(
            precipitation,
            temperature,
            pet,
            initial_state=initial,
        )
        first = model.simulate(
            precipitation[:3],
            temperature[:3],
            pet[:3],
            initial_state=initial,
        )
        second = model.simulate(
            precipitation[3:],
            temperature[3:],
            pet[3:],
            initial_state=tuple(first.states[-1]),
        )
        np.testing.assert_allclose(
            np.vstack((first.states, second.states[1:])),
            full.states,
        )
        np.testing.assert_allclose(
            np.vstack((first.outputs, second.outputs)),
            full.outputs,
        )


if __name__ == "__main__":
    unittest.main()
