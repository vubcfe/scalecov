# Changelog

All notable changes are documented here. The project follows [Semantic Versioning](https://semver.org).

## [0.1.0] - 2026-11-12

First public release.

- `PixelConformal`: pixel intervals by hierarchical (grouped) conformal prediction, every calibration unit with equal weight.
- `UnitIntervals`: intervals for unit and group means with five constructions (`independent`, `full`, `variogram`, `variogram_re`, `direct`).
- `fit_within_unit_covariance`, `mean_error_variance`: within-unit exponential covariance with nugget and the variance of unit means it implies.
- `cross_calibrate`, `summarise`, `pixel_coverage_by_unit`: cross-fold calibration on out-of-fold residuals and coverage by scale.
- `kappa`, `unit_quantile`, `coverage_vs_kappa`, `k_dense_coverage`: pixel intervals when only aggregates or a few dense units are available.
- `r2_within`, `rho_within`, `r2_unit_means`, `block_bootstrap_r2_within`: skill inside units with block-bootstrap intervals.
- `simulate.nested_field`: simulated maps with group, unit, spatial and nugget error components.
