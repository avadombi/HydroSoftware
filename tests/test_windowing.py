import unittest

import numpy as np

from hydrosoftware.core.specifications import PredictionLayout
from hydrosoftware.core.windowing import WindowSpec, build_windows


class WindowingTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(20, dtype=float).reshape(10, 2)
        self.y = np.arange(10, dtype=float)

    def test_same_sequence_s2s_alignment_and_warmup(self):
        spec = WindowSpec.same_sequence(4, stride=3, warmup_length=1)
        batch = build_windows(self.x, self.y, spec)

        self.assertEqual(batch.inputs.shape, (3, 4, 2))
        self.assertEqual(batch.targets.shape, (3, 4, 1))
        np.testing.assert_array_equal(batch.input_indices, batch.target_indices)
        np.testing.assert_array_equal(
            batch.target_mask,
            np.array(
                [
                    [False, True, True, True],
                    [False, True, True, True],
                    [False, True, True, True],
                ]
            ),
        )

    def test_s2v_alignment(self):
        spec = WindowSpec.sequence_to_value(lookback=3, lead=2, stride=2)
        batch = build_windows(self.x, self.y, spec)

        self.assertEqual(batch.spec.layout, PredictionLayout.SEQUENCE_TO_VALUE)
        np.testing.assert_array_equal(batch.input_indices[0], [0, 1, 2])
        self.assertEqual(batch.target_indices[0], 4)
        self.assertEqual(batch.targets.shape, (3, 1))

    def test_future_s2s_alignment(self):
        spec = WindowSpec.future_sequence(lookback=3, horizon=2, lead=1)
        batch = build_windows(self.x, self.y, spec)

        np.testing.assert_array_equal(batch.input_indices[0], [0, 1, 2])
        np.testing.assert_array_equal(batch.target_indices[0], [3, 4])

    def test_short_partition_is_rejected(self):
        spec = WindowSpec.sequence_to_value(lookback=10)
        with self.assertRaisesRegex(ValueError, "At least 11"):
            build_windows(self.x, self.y, spec)

    def test_deepdiscover_notebook_window_count_is_reproducible(self):
        inputs = np.zeros((7305, 5))
        targets = np.zeros((7305, 1))
        spec = WindowSpec.same_sequence(2190, stride=365, warmup_length=365)
        batch = build_windows(inputs, targets, spec)

        self.assertEqual(batch.inputs.shape, (15, 2190, 5))
        self.assertEqual(batch.targets.shape, (15, 2190, 1))
        self.assertEqual(batch.target_mask.sum(axis=1)[0], 1825)


if __name__ == "__main__":
    unittest.main()
