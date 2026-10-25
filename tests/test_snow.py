import unittest

from hydrosoftware.processes.snow import (
    DegreeDaySnowModule,
    NoSnowModule,
    SmoothDegreeDaySnowModule,
)


class SnowModuleTests(unittest.TestCase):
    def assert_balanced(self, result):
        self.assertAlmostEqual(result.balance_error, 0.0, places=12)
        self.assertGreaterEqual(result.rain, 0.0)
        self.assertGreaterEqual(result.snowfall, 0.0)
        self.assertGreaterEqual(result.melt, 0.0)
        self.assertGreaterEqual(result.storage_after, 0.0)

    def test_no_snow_routes_all_precipitation_to_rain(self):
        result = NoSnowModule().step(precipitation=5.0, temperature=-10.0)
        self.assertEqual(result.rain, 5.0)
        self.assertEqual(result.snowfall, 0.0)
        self.assert_balanced(result)

    def test_hard_degree_day_accumulation(self):
        module = DegreeDaySnowModule(-1.0, 1.0, 2.0)
        result = module.step(5.0, -2.0, 3.0)
        self.assertEqual(result.snowfall, 5.0)
        self.assertEqual(result.melt, 0.0)
        self.assertEqual(result.storage_after, 8.0)
        self.assert_balanced(result)

    def test_hard_degree_day_melt_is_limited_by_antecedent_storage(self):
        module = DegreeDaySnowModule(-1.0, 1.0, 2.0)
        result = module.step(0.0, 10.0, 3.0)
        self.assertEqual(result.melt, 3.0)
        self.assertEqual(result.storage_after, 0.0)
        self.assert_balanced(result)

    def test_smooth_module_never_generates_negative_melt(self):
        module = SmoothDegreeDaySnowModule(-1.0, 1.0, 2.0)
        result = module.step(2.0, -20.0, 4.0)
        self.assertGreaterEqual(result.melt, 0.0)
        self.assertAlmostEqual(result.rain + result.snowfall, 2.0)
        self.assert_balanced(result)


if __name__ == "__main__":
    unittest.main()

