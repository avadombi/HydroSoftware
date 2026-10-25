import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hydrosoftware.configuration import (
    CalibrationPeriodConfig,
    CsvColumnMapping,
    CsvDataSourceConfig,
)
from hydrosoftware.data_sources import (
    DataSourceError,
    import_csv_bytes,
    load_calibration_csv,
    preview_csv,
    select_calibration_period,
)


class CsvDataSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.path = self.root / "basin.csv"
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "p", "t", "pet", "q"])
            writer.writerow(["2020-01-01", 1.0, -2.0, 0.2, 0.1])
            writer.writerow(["2020-01-02", 0.0, 1.0, 0.3, ""])
            writer.writerow(["2020-01-03", 2.0, 3.0, 0.4, 0.5])
        self.source = CsvDataSourceConfig(
            path="basin.csv",
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

    def test_csv_is_loaded_with_missing_observations_preserved(self):
        data = load_calibration_csv(self.source, workspace_root=self.root)

        self.assertEqual(data.n_time, 3)
        self.assertTrue(np.isnan(data.streamflow[1]))
        self.assertEqual(data.timestamps[0], "2020-01-01")

    def test_half_open_period_selection_is_explicit(self):
        data = load_calibration_csv(self.source, workspace_root=self.root)
        selected = select_calibration_period(
            data,
            CalibrationPeriodConfig(start_index=1, end_index=3, warmup_steps=1),
        )

        np.testing.assert_array_equal(selected.precipitation, [0.0, 2.0])
        self.assertEqual(selected.timestamps, ("2020-01-02", "2020-01-03"))

    def test_period_cannot_extend_past_data(self):
        data = load_calibration_csv(self.source, workspace_root=self.root)
        with self.assertRaisesRegex(DataSourceError, "exceeds"):
            select_calibration_period(
                data,
                CalibrationPeriodConfig(start_index=0, end_index=4),
            )

    def test_missing_mapped_column_has_a_stable_error_code(self):
        source = self.source.model_copy(
            update={
                "columns": self.source.columns.model_copy(
                    update={"streamflow": "missing_q"}
                )
            }
        )
        with self.assertRaises(DataSourceError) as context:
            load_calibration_csv(source, workspace_root=self.root)

        self.assertEqual(context.exception.code, "missing_csv_columns")

    def test_timestamp_timezone_awareness_cannot_change_within_a_file(self):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "p", "t", "pet", "q"])
            writer.writerow(["2020-01-01T00:00:00", 1.0, 0.0, 0.1, 0.2])
            writer.writerow(["2020-01-02T00:00:00Z", 1.0, 0.0, 0.1, 0.2])
        with self.assertRaises(DataSourceError) as context:
            load_calibration_csv(self.source, workspace_root=self.root)

        self.assertEqual(
            context.exception.code, "inconsistent_timestamp_timezones"
        )

    def test_preview_reports_columns_rows_and_stable_content_hash(self):
        preview = preview_csv("basin.csv", workspace_root=self.root, preview_rows=2)

        self.assertEqual(preview.columns, ("date", "p", "t", "pet", "q"))
        self.assertEqual(preview.row_count, 3)
        self.assertEqual(len(preview.rows), 2)
        self.assertEqual(len(preview.sha256), 64)

    def test_import_is_content_addressed_and_reuses_identical_file(self):
        content = self.path.read_bytes()
        first = import_csv_bytes("My Basin.csv", content, workspace_root=self.root)
        second = import_csv_bytes("My Basin.csv", content, workspace_root=self.root)

        self.assertEqual(first.relative_path, second.relative_path)
        self.assertTrue((self.root / first.relative_path).is_file())
        self.assertTrue(first.relative_path.startswith("data/imports/My-Basin-"))

    def test_import_rejects_non_csv_files(self):
        with self.assertRaises(DataSourceError) as context:
            import_csv_bytes("basin.txt", b"a,b\n1,2\n", workspace_root=self.root)

        self.assertEqual(context.exception.code, "unsupported_data_file")


if __name__ == "__main__":
    unittest.main()
