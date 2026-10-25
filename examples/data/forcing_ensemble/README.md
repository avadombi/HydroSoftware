# Synthetic forcing-ensemble members

These three aligned CSV files are a small interface test for the saved
`synthetic-exp-hydro` calibration:

- `member_dry.csv`
- `member_control.csv`
- `member_wet.csv`

Use **Use new forcings**, select **Meteorological ensemble**, import all three
files, and map:

| Role | Column |
| --- | --- |
| Timestamp | `date` |
| Precipitation | `precipitation` |
| Temperature | `temperature` |
| Potential evapotranspiration | `pet` |
| Observed streamflow | `streamflow` |

Choose `source_final` initialization. The files begin on 2020-02-10, the day
after the bundled calibration series ends. Their streamflow observations are
identical by design, while their meteorological forcings differ.

The data are deterministic synthetic values for integration testing only.
They are not an operational ensemble and must not be used to assess forecast
skill or calibration.
