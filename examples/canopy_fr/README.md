# Example: a global canopy height map checked against airborne LiDAR in public forests (Hauts-de-France)

The 10 m global canopy height map of Lang et al. (2023, year 2020, CC BY 4.0) is compared with the IGN LiDAR HD canopy height model (0.5 m, Licence Ouverte 2.0) in the management compartments of the public forests of Hauts-de-France (ONF, Licence Ouverte 2.0). Nested levels: 10 m pixel < compartment < forest. The map was trained with GEDI and not with these LiDAR data, so its errors are out of sample. No model is fitted here.

| Step | Script | Output |
|---|---|---|
| 1. Public forests, compartments and the LiDAR HD tile index (web feature services of IGN and ONF) | `get_data.py --out data` | `forests.gpkg`, `compartments.gpkg`, `lidar_tiles.gpkg` |
| 2. Select compartments, download the LiDAR canopy height model around each of them, compute the reference canopy top height and read the map | `prepare.py --data data` | `pixels.parquet` (one row per 10 m cell) |
| 3. Coverage by scale (folds of whole forests, seeds 0–2), the map's own standard deviation, k dense compartments | `run_scalecov.py --data data --out results` | `results/coverage.csv`, `summary.json`, `k_dense.csv` |

**Selection rule, fixed before any result was computed:**
- keep compartments of 5–40 ha whose area is at least 99 % covered by LiDAR HD tiles with a canopy height model;
- keep forests with at least 10 such compartments;
- in each forest, draw at most 15 compartments at random (seed 0).

**Reference canopy top height:** as used by the map's authors for airborne LiDAR, the 0.5 m canopy height model is max-pooled within a disc of 12.5 m radius (the 25 m GEDI footprint) and read at the centre of each 10 m cell.

**Data volume:** one window of the LiDAR canopy height model per compartment, about 3 MB each (about 2 GB in total). The windows are requested on the native 0.5 m grid, so nothing is resampled. The map itself is not downloaded; only the blocks needed are read from the published cloud-optimised GeoTIFFs.

The LiDAR flights (2021–2025) are later than the map (2020), so part of the error is change in the forest. Any user who checks an existing map against newer reference data faces the same situation.

Install with `pip install scalecov -r requirements.txt`.
