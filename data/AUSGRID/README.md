# AUSGRID full dataset

This directory contains the full aligned AUSGRID PV--ERA5 input used by the
training script:

- `pv.csv`: 26,304 hourly timestamps and 299 site-level PV series.
- `stations.csv`: 299 station rows in exactly the same order as the PV columns.
- `nwp/`: ERA5 valid-time memmap with shape `[26304, 14, 13, 6]` in
  `time, latitude, longitude, variable` order.

The six NWP variables are direct short-wave radiation, diffuse short-wave
radiation, total cloud cover, 2-m temperature, 10-m zonal wind, and 10-m
meridional wind. Radiation accumulations were converted to hourly mean
`W/m2` during preprocessing. The included normalization statistics were
computed from the training period only.

PV and NWP contain the same 26,304-hour timeline. `NWPMemmapStore` validates
this length and retrieves historical and future NWP lazily; the full memmap is
not loaded into RAM.
