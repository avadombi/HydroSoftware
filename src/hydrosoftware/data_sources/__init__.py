"""Adapters that convert external data into standard in-memory arrays."""

from hydrosoftware.data_sources.csv_source import (
    CsvCalibrationData,
    CsvPreview,
    import_csv_bytes,
    load_calibration_csv,
    preview_csv,
    select_calibration_period,
)
from hydrosoftware.data_sources.errors import DataSourceError
from hydrosoftware.data_sources.quality import (
    DataQualityAnalysis,
    DataQualityIssue,
    DataQualityReport,
    TimestampQualitySummary,
    VariableQualitySummary,
    analyze_calibration_csv,
)
from hydrosoftware.data_sources.camels_us import (
    CamelsBasinSeries,
    CamelsEligibility,
    CamelsForcing,
    CamelsTimeSeriesCollection,
    assemble_basin_series,
    evaluate_eligibility,
    hamon_potential_evapotranspiration,
    read_attribute_table,
    select_contrasting_basins,
    write_basin_csv,
)

__all__ = [
    "CamelsBasinSeries",
    "CamelsEligibility",
    "CamelsForcing",
    "CamelsTimeSeriesCollection",
    "CsvCalibrationData",
    "CsvPreview",
    "DataSourceError",
    "DataQualityAnalysis",
    "DataQualityIssue",
    "DataQualityReport",
    "TimestampQualitySummary",
    "VariableQualitySummary",
    "analyze_calibration_csv",
    "assemble_basin_series",
    "evaluate_eligibility",
    "hamon_potential_evapotranspiration",
    "import_csv_bytes",
    "load_calibration_csv",
    "preview_csv",
    "read_attribute_table",
    "select_calibration_period",
    "select_contrasting_basins",
    "write_basin_csv",
]
