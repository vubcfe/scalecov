# Changelog

All notable changes are documented here. The project follows [Semantic Versioning](https://semver.org).

## [0.2.0] - 2026-11-25

- `coverage_interval`, `nominal_range`: Clopper-Pearson interval of an observed coverage and the range of observed coverage consistent with the nominal level for a given number of units.
- `summarise` reports `coverage_lo`, `coverage_hi` and `consistent` for every unit level.
- New example `examples/canopy_fr`: a published global canopy height map checked against public airborne LiDAR in the public forests of Hauts-de-France (compartments within forests), including the map's own per-pixel uncertainty.
- Documentation: minimum number of calibration units for `direct`; reading coverage with few units.
- CI: Ubuntu runner pinned to 24.04 (Python 3.9 is not available on newer images).

## [0.1.0] - 2026-11-12

First public release.

- `PixelConformal`: pixel intervals by hierarchical (grouped) conformal prediction, every calibration unit with equal weight.
- `UnitIntervals`: intervals for unit and group means with five constructions (`independent`, `full`, `variogram`, `variogram_re`, `direct`).
- `fit_within_unit_covariance`, `mean_error_variance`: within-unit exponential covariance with nugget and the variance of unit means it implies.
- `cross_calibrate`, `summarise`, `pixel_coverage_by_unit`: cross-fold calibration on out-of-fold residuals and coverage by scale.
- `kappa`, `unit_quantile`, `coverage_vs_kappa`, `k_dense_coverage`: pixel intervals when only aggregates or a few dense units are available.
- `r2_within`, `rho_within`, `r2_unit_means`, `block_bootstrap_r2_within`: skill inside units with block-bootstrap intervals.
- `simulate.nested_field`: simulated maps with group, unit, spatial and nugget error components.
