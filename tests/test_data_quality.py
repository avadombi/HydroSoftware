import csv
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hydrosoftware.configuration import (
    AdditionalPredictorColumnConfig,
    CsvColumnMapping,
    CsvDataSourceConfig,
)
from hydrosoftware.data_sources import (
    DataSourceError,
    analyze_calibration_csv,
    load_calibration_csv,
)


class DataQualityTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.path = self.root / "quality.csv"
        self.source = CsvDataSourceConfig(
            path="quality.csv",
            columns=CsvColumnMapping(
                timestamp="date",
                precipitation="p",
                temperature="t",
                potential_evapotranspiration="pet",
                streamflow="q",
            ),
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "p", "t", "pet", "q"])
            writer.writerows(rows)

    def configured_source(self, **processing_updates):
        payload = self.source.model_dump(mode="json")
        for section, values in processing_updates.items():
            payload["processing"][section].update(values)
        return CsvDataSourceConfig.model_validate(payload)

    def test_short_interior_forcing_gap_is_interpolated_and_audited(self):
        self.write_rows(
            [
                ["2020-01-01", 1.0, 0.0, 0.2, 0.1],
                ["2020-01-02", 2.0, "", 0.3, 0.2],
                ["2020-01-03", 3.0, 2.0, 0.4, 0.3],
            ]
        )
        source = self.configured_source(
            missing={"temperature": "linear_interpolation"}
        )

        analysis = analyze_calibration_csv(source, workspace_root=self.root)

        self.assertTrue(analysis.report.ready)
        self.assertAlmostEqual(analysis.temperature[1], 1.0)
        temperature = next(
            item for item in analysis.report.variables if item.variable == "temperature"
        )
        self.assertEqual(temperature.transformed_count, 1)

    def test_edge_gap_is_not_silently_extrapolated(self):
        self.write_rows(
            [
                ["2020-01-01", 1.0, "", 0.2, 0.1],
                ["2020-01-02", 2.0, 1.0, 0.3, 0.2],
            ]
        )
        source = self.configured_source(
            missing={"temperature": "linear_interpolation"}
        )

        analysis = analyze_calibration_csv(source, workspace_root=self.root)

        self.assertFalse(analysis.report.ready)
        self.assertIn(
            "unresolved_missing_temperature",
            {issue.code for issue in analysis.report.issues},
        )

    def test_streamflow_conversion_requires_and_uses_basin_area(self):
        self.write_rows([["2020-01-01", 1.0, 0.0, 0.2, 1.0]])
        source = self.configured_source(
            units={"streamflow": "m3/s", "basin_area_km2": 100.0}
        )

        data = load_calibration_csv(source, workspace_root=self.root)

        np.testing.assert_allclose(data.streamflow, [0.864])

    def test_negative_forcing_is_changed_only_by_an_explicit_rule(self):
        self.write_rows([["2020-01-01", -1.0, 0.0, 0.2, 0.1]])
        source = self.configured_source(
            negative={"precipitation": "clip_to_zero"}
        )

        analysis = analyze_calibration_csv(source, workspace_root=self.root)

        self.assertTrue(analysis.report.ready)
        self.assertEqual(analysis.precipitation[0], 0.0)
        self.assertIn(
            "negative_precipitation_clipped",
            {issue.code for issue in analysis.report.issues},
        )

    def test_irregular_daily_timestamps_block_execution(self):
        self.write_rows(
            [
                ["2020-01-01", 1.0, 0.0, 0.2, 0.1],
                ["2020-01-03", 2.0, 1.0, 0.3, 0.2],
            ]
        )

        analysis = analyze_calibration_csv(self.source, workspace_root=self.root)

        self.assertFalse(analysis.report.ready)
        self.assertEqual(analysis.report.timestamps.irregular_step_count, 1)
        with self.assertRaises(DataSourceError) as context:
            load_calibration_csv(self.source, workspace_root=self.root)
        self.assertEqual(context.exception.code, "irregular_daily_timestep")

    def test_content_fingerprint_detects_file_replacement(self):
        self.write_rows([["2020-01-01", 1.0, 0.0, 0.2, 0.1]])
        digest = sha256(self.path.read_bytes()).hexdigest()
        source = self.source.model_copy(update={"expected_sha256": digest})
        self.write_rows([["2020-01-01", 9.0, 0.0, 0.2, 0.1]])

        with self.assertRaises(DataSourceError) as context:
            analyze_calibration_csv(source, workspace_root=self.root)

        self.assertEqual(context.exception.code, "data_fingerprint_mismatch")

    def test_additional_predictor_is_profiled_with_eda_statistics(self):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "p", "t", "pet", "rh", "q"])
            writer.writerows(
                [
                    ["2020-01-01", 1.0, 0.0, 0.2, 40.0, 0.1],
                    ["2020-01-02", 2.0, 1.0, 0.3, 50.0, 0.2],
                    ["2020-01-03", 3.0, 2.0, 0.4, 60.0, 0.3],
                ]
            )
        source = self.source.model_copy(
            update={
                "columns": self.source.columns.model_copy(
                    update={
                        "additional_predictors": (
                            AdditionalPredictorColumnConfig(
                                name="relative_humidity",
                                column="rh",
                                unit="%",
                            ),
                        )
                    }
                )
            }
        )

        analysis = analyze_calibration_csv(source, workspace_root=self.root)

        self.assertTrue(analysis.report.ready)
        np.testing.assert_allclose(
            analysis.additional_predictors["relative_humidity"],
            [40.0, 50.0, 60.0],
        )
        summary = next(
            item
            for item in analysis.report.variables
            if item.variable == "relative_humidity"
        )
        self.assertEqual(summary.processed_median, 50.0)
        self.assertAlmostEqual(summary.processed_standard_deviation, np.std([40, 50, 60]))
        self.assertIn("relative_humidity", analysis.report.correlation_variables)
        self.assertEqual(
            len(analysis.report.correlation_matrix),
            len(analysis.report.correlation_variables),
        )


if __name__ == "__main__":
    unittest.main()
