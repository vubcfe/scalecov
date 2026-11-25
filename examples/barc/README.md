# Example: harvester yield maps at the USDA Beltsville Agricultural Research Center (BARC)

Prediction intervals for pixel yield, field-season, section-year and farm-year means, checked with ScaleCov on public data only.

| Step | Script | Output |
|---|---|---|
| 1. Download the BARC Gridded Yield Data Archive (Ag Data Commons, doi:10.15482/usda.adc/28950746, U.S. Public Domain) and check MD5 sums | `get_data.py --out data/raw` | `data/raw/*.zip` |
| 2. Parse field names, convert bu/ac to t/ha, average cleaned points on a 10 m grid | `prepare.py --raw data/raw --out data` | `pixels.parquet`, `records.csv`, `sites.geojson` |
| 3. Build co-registered Sentinel-2 cubes, one per season, with S2Cube | `make_cubes.py --data data` | `data/s2/cube_<year>.h5` |
| 4. Monthly median composites (April–November, 10 bands + NDVI) | `features.py --data data --cubes data/s2 --out data/features.parquet` | 88 features per pixel |
| 5. Out-of-fold predictions, folds of whole farms | `model.py --features data/features.parquet --out data/oof.parquet` | `oof.parquet` |
| 6. Coverage by scale, skill, kappa, k dense fields | `run_scalecov.py --oof data/oof.parquet --out results` | `results/*.csv`, `summary.json` |
| 7. Fold design (farm vs section folds, seeds 0–2) and ScaleCov settings | `run_sensitivity.py --features data/features.parquet --out results/sensitivity.csv` | `sensitivity.csv` |

Install the environment with `pip install scalecov -r requirements.txt`. Step 3 downloads Sentinel-2 data from Microsoft Planetary Computer and is the longest step; it can be resumed if interrupted. The yield data are not redistributed here.

The gradient-boosted tree model has fixed settings and is only a way to obtain realistic out-of-sample errors. It is not a tuned yield model. Pixel coordinates are used only to locate pixels in the cubes and for the covariance of errors. They are never model features.
