import unittest

import numpy as np

from hydrosoftware.core.specifications import PredictionLayout
from hydrosoftware.core.results import SimulationResult
from hydrosoftware.models.deepdiscover import binary_relaxation_penalty


class SpecificationTests(unittest.TestCase):
    def test_simulation_result_uses_boundary_states(self):
        result = SimulationResult(
            outputs=np.zeros((3, 1)),
            states=np.arange(4, dtype=float).reshape(4, 1),
            fluxes=np.zeros((3, 2)),
            output_names=("streamflow",),
            state_names=("storage",),
            flux_names=("input", "output"),
        )
        np.testing.assert_array_equal(result.state_before[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(result.state_after[:, 0], [1, 2, 3])


    def test_binary_penalty_rejects_unconstrained_values(self):
        self.assertAlmostEqual(binary_relaxation_penalty(0.5), 0.25)
        with self.assertRaises(ValueError):
            binary_relaxation_penalty(1.2)


if __name__ == "__main__":
    unittest.main()

