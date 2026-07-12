# AUSGRID Sample

This sample contains the first 1,000 contiguous hourly AUSGRID rows and the exactly corresponding NWP rows.

- PV: `pv.csv`, shape `(1000, 299)` excluding date
- Stations: `stations.csv`, 299 rows
- NWP disk shape: `(1000, 14, 13, 6)` = `(time, lat, lon, variable)`
- NWP model shape per sample: `(13, 14, pred_len, 6)`
- Shift: `0`
- NWP stats: generated from the first 700 sample rows on first run
