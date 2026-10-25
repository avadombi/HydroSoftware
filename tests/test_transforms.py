import unittest

import numpy as np

from hydrosoftware.core.transforms import Standardizer


class TransformTests(unittest.TestCase):
    def test_statistics_are_fitted_once_on_continuous_training_rows(self):
        training = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        scaler = Standardizer.fit(training)

        np.testing.assert_allclose(scaler.mean, [3.0, 20.0])
        transformed = scaler.transform(training)
        np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0], atol=1e-12)

    def test_one_fitted_statistic_set_applies_to_any_window_batch_size(self):
        training = np.array([[0.0], [2.0], [4.0]])
        scaler = Standardizer.fit(training)
        test_windows = np.array([[[2.0], [4.0]]])

        transformed = scaler.transform(test_windows)
        self.assertEqual(transformed.shape, (1, 2, 1))
        restored = scaler.inverse_transform(transformed)
        np.testing.assert_allclose(restored, test_windows)

    def test_constant_feature_uses_unit_scale(self):
        scaler = Standardizer.fit(np.ones((4, 1)))
        np.testing.assert_array_equal(scaler.scale, [1.0])


if __name__ == "__main__":
    unittest.main()

