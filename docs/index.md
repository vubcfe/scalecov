# ScaleCov

**ScaleCov** (coverage across scales) calibrates and checks prediction intervals of spatial maps for pixels, units (fields, plots, stands) and groups of units (farms, districts).

A map often comes with pixel intervals, while users need statements about field or farm means. Pixel errors inside a unit share a common component, so pixel intervals do not aggregate into valid unit intervals by themselves. ScaleCov makes this explicit:

- **calibrate** intervals at every level of a nested structure;
- **compare** the usual ways of aggregating pixel uncertainty;
- **check** coverage at each scale on held-out units.

It works with any model; the input is one row per pixel with an out-of-fold residual and the identifiers of the nested units.

| Problem in practice | What ScaleCov provides |
|---|---|
| Large units dominate pixel calibration | `PixelConformal`: every calibration unit has equal weight |
| Field means need intervals, but pixel errors are correlated | `UnitIntervals`: five constructions, from independent errors to direct calibration |
| It is unclear which construction is valid | `cross_calibrate` and `summarise`: coverage by level and method on held-out folds |
| Only aggregates are reported in the target region | `kappa`, `coverage_vs_kappa`, `k_dense_coverage` |
| Pixel or field scores hide whether a map has any within-field pattern | `r2_within`, `rho_within`, `block_bootstrap_r2_within` |
| A method needs testing where the answer is known | `simulate.nested_field` |

See the [Tutorial](tutorial.md) for a complete example and [Methods](methods.md) for the definitions.
