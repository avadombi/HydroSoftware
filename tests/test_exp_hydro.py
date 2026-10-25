import unittest

import numpy as np

from hydrosoftware.models.conceptual.exp_hydro import (
    EXP_HYDRO_MANIFEST,
    ExpHydroModel,
    ExpHydroParameters,
)
from hydrosoftware.processes.snow import DegreeDaySnowModule, NoSnowModule


class ExpHydroTests(unittest.TestCase):
    def make_model(self, snow_module=None, substeps=4):
        return ExpHydroModel(
            ExpHydroParameters(0.05, 500.0, 30.0),
            snow_module or NoSnowModule(),
            substeps=substeps,
        )

    def test_manifest_exposes_states_fluxes_and_continuous_layout(self):
        self.assertTrue(EXP_HYDRO_MANIFEST.capabilities.exposes_states)
        self.assertTrue(EXP_HYDRO_MANIFEST.capabilities.exposes_fluxes)
        self.assertEqual(EXP_HYDRO_MANIFEST.metadata["legacy_equivalence"], "not_claimed")

    def test_dry_zero_state_has_zero_flow(self):
        result = self.make_model().simulate(
            np.zeros(5), np.zeros(5), np.zeros(5)
        )
        np.testing.assert_array_equal(result.outputs, np.zeros((5, 1)))
        np.testing.assert_array_equal(result.states, np.zeros((6, 2)))

    def test_interval_and_global_mass_balance(self):
        rng = np.random.default_rng(42)
        precipitation = rng.gamma(1.5, 4.0, size=100)
        temperature = rng.normal(2.0, 8.0, size=100)
        pet = rng.uniform(0.0, 5.0, size=100)
        snow = DegreeDaySnowModule(-1.0, 1.0, 2.0)

        result = self.make_model(snow).simulate(
            precipitation,
            temperature,
            pet,
            initial_state=(10.0, 100.0),
        )

        self.assertLess(
            result.diagnostics["maximum_absolute_mass_balance_error"], 1e-10
        )
        self.assertTrue(np.all(result.states >= 0.0))
        self.assertTrue(np.all(result.fluxes >= 0.0))
        global_balance = (
            result.states[0].sum()
            + precipitation.sum()
            - result.fluxes[:, 4:7].sum()
            - result.states[-1].sum()
        )
        self.assertAlmostEqual(global_balance, 0.0, places=10)

    def test_restart_from_boundary_state_is_exact(self):
        precipitation = np.array([0.0, 5.0, 2.0, 0.0, 8.0, 1.0])
        temperature = np.full(6, 5.0)
        pet = np.full(6, 1.0)
        model = self.make_model()

        full = model.simulate(precipitation, temperature, pet)
        first = model.simulate(precipitation[:3], temperature[:3], pet[:3])
        second = model.simulate(
            precipitation[3:],
            temperature[3:],
            pet[3:],
            initial_state=tuple(first.states[-1]),
        )

        np.testing.assert_allclose(full.outputs[:3], first.outputs)
        np.testing.assert_allclose(full.outputs[3:], second.outputs)
        np.testing.assert_allclose(full.states[3:], second.states)

    def test_snow_module_is_a_real_model_choice(self):
        precipitation = np.array([10.0, 0.0])
        temperature = np.array([-5.0, 5.0])
        pet = np.zeros(2)
        no_snow = self.make_model(NoSnowModule()).simulate(
            precipitation, temperature, pet
        )
        with_snow = self.make_model(
            DegreeDaySnowModule(-1.0, 1.0, 2.0)
        ).simulate(precipitation, temperature, pet)

        self.assertEqual(no_snow.states[1, 0], 0.0)
        self.assertGreater(with_snow.states[1, 0], 0.0)
        self.assertNotEqual(no_snow.outputs[0, 0], with_snow.outputs[0, 0])

    def test_parameter_bounds_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "recession_factor"):
            ExpHydroParameters(0.2, 500.0, 30.0)


if __name__ == "__main__":
    unittest.main()

