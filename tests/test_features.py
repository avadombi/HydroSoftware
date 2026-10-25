import unittest

import numpy as np
from pydantic import ValidationError

from hydrosoftware.core.features import (
    LaggedFeature,
    build_feature_mapping,
    build_forecast_feature_mapping,
)
from hydrosoftware.service.schemas import NeuralAntecedentForcings


class FeatureEngineeringTests(unittest.TestCase):
    def test_lagged_features_are_causal_and_named_stably(self):
        features = build_feature_mapping(
            {"rain": np.array([1.0, 2.0, 3.0, 4.0])},
            ("rain",),
            (LaggedFeature("rain", 2),),
        )

        np.testing.assert_allclose(features["rain"], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            features["rain__lag_2"],
            [np.nan, np.nan, 1.0, 2.0],
            equal_nan=True,
        )

    def test_forecast_lags_use_only_archived_antecedent_rows(self):
        current, antecedent = build_forecast_feature_mapping(
            {"rain": np.array([10.0, 11.0])},
            ("rain",),
            (LaggedFeature("rain", 2),),
            lookback=2,
            antecedent_raw_features={"rain": np.array([6.0, 7.0, 8.0, 9.0])},
        )

        np.testing.assert_allclose(current["rain__lag_2"], [8.0, 9.0])
        np.testing.assert_allclose(antecedent["rain__lag_2"], [6.0, 7.0])

    def test_service_accepts_named_predictor_and_physical_context_channels(self):
        context = NeuralAntecedentForcings.model_validate(
            {
                "relative_humidity": [60.0, 61.0],
                "physical_state__soil_storage": [300.0, 301.0],
            }
        )
        self.assertEqual(
            context.model_dump(),
            {
                "relative_humidity": (60.0, 61.0),
                "physical_state__soil_storage": (300.0, 301.0),
            },
        )

        with self.assertRaises(ValidationError):
            NeuralAntecedentForcings.model_validate(
                {
                    "relative_humidity": [60.0],
                    "physical_state__soil_storage": [300.0, 301.0],
                }
            )


if __name__ == "__main__":
    unittest.main()
