# Recommended settings

These are starting points, checked on simulated data with known answers and on harvester yield maps from four countries. Check them on your own data with `cross_calibrate`.

## Folds

Folds must keep dependent units together: all fields of a farm, and farms that share land. Otherwise calibration and test units are not exchangeable, and coverage looks better than it is.

## Which construction to use

- **Use `direct` at the scale at which an interval is used.** It is the only construction that adapts to error shared at every level, and it needs only the aggregates at that level. It needs at least $(1-\alpha)/\alpha$ calibration units at that level (9 for 90 % intervals); with fewer, its quantile is infinite.
- **Treat `independent` as a warning, not an interval.** With any error shared within units, it under-covers badly.
- **Use `variogram_re` with care for groups.** It is valid for unit means when the unit effect is the main shared error, but it under-covers group means when units of a group share errors.
- **Use `variogram` only when the correlation range is short relative to the unit size.** Removing unit means hides correlation at the scale of the unit, so the fitted range and the variance of unit means are biased low. This happens even when there is no shared unit error.

## Pixel intervals without dense data in the target

A borrowed $\kappa$ can give pixel intervals that cover much less than the nominal level when error structures differ between regions. A few dense calibration units are more reliable: compare the spread reported by `k_dense_coverage` for `k = 1, 5, 20`.

## Size and speed

- `max_per_unit=1000` pixels per calibration unit bounds memory, with little effect on the quantile.
- The covariance fit uses up to 300 pixels per unit (`max_pixels_per_unit`) and pairs up to `max_lag`. Set `max_lag` and `bin_width` in the units of your coordinates, for example 400 and 20 m for 10 m pixels.
