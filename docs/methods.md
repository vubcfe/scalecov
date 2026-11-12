# Methods

Notation: pixel residual $e_i = y_i - \hat y_i$; unit $u$ with $n_u$ pixels; group $g$; nominal miscoverage $\alpha$.

## Conformal quantile

Without weights, the half-width is the $\lceil (n+1)(1-\alpha) \rceil$-th smallest of the $n$ absolute calibration scores. It is infinite if this exceeds $n$. With weights $w_j$, it is the smallest score $r$ such that $\sum_{r_j \le r} w_j / (\sum_j w_j + \max_j w_j) \ge 1-\alpha$, which gives the test point the largest calibration weight.

`PixelConformal` weights each calibration pixel by $1/n_u$, so that every calibration unit has the same total weight (hierarchical or grouped conformal prediction). Its guarantee is marginal over exchangeable units. With cross-fold calibration (`cross_calibrate`), the residuals come from models fitted on other folds. The scheme is then cross-conformal, so coverage should be checked empirically, which this package does.

## Intervals for unit and group means

The error of a unit mean is $\bar e_u = n_u^{-1}\sum_{i\in u} e_i$. With $q$ the pixel conformal quantile and $z = \Phi^{-1}(1-\alpha/2)$:

| Method | Half-width for unit $u$ |
|---|---|
| `independent` | $q / \sqrt{n_u}$ |
| `full` | $q$ |
| `variogram` | $z\,\sqrt{V_u}$, with $V_u$ from the within-unit covariance |
| `variogram_re` | $z\,\sqrt{V_u + \sigma^2_U}$ |
| `direct` | conformal quantile of $\lvert\bar e_u\rvert$ over calibration units |

**Within-unit covariance.** $C(h) = s\,e^{-h/a}$ for $h>0$, and $C(0) = s + \tau^2$. It is fitted by least squares to binned products of residuals centred on their unit means, over pairs of pixels in the same unit. The sum $s + \tau^2$ equals the within-unit residual variance.

**Variance of a unit mean.** $V_u = \bar C_u + (C(0) - \bar C_u)/n_u$, where $\bar C_u$ is the mean covariance over distinct pixel pairs of $u$. For large units it is estimated on a random subsample.

**Unit random effect.** $\sigma^2_U = \max\{0,\ \overline{\bar e_u^2} - \overline{V_u}\}$ over calibration units. This is the variance of unit-mean errors that the within-unit covariance does not explain.

**Groups.** For a group, $\bar e_g = \sum_u w_u \bar e_u$ with area weights $w_u = n_u / \sum_{u\in g} n_u$. `variogram` and `variogram_re` use $\sum_u w_u^2 V_u$ and $\sum_u w_u^2 (V_u + \sigma^2_U)$, so they treat units of a group as independent. `direct` calibrates on group means.

## Calibration from aggregates

With aggregates only, $\bar e_u$ is observable, so $q_U$, the conformal quantile of $\lvert\bar e_u\rvert$, can be calibrated, but the pixel quantile $q$ cannot. $\kappa = q / q_U$ is estimated where dense data exist (`kappa`). The coverage that a borrowed $\kappa$ gives in a target region is read from `coverage_vs_kappa`. `k_dense_coverage` calibrates $q$ on $k$ randomly drawn dense units, repeatedly, and reports the spread of the resulting coverage.

## Skill inside units

$R^2_w = 1 - \sum_i [(y_i-\bar y_u)-(\hat y_i-\bar{\hat y}_u)]^2 / \sum_i (y_i-\bar y_u)^2$. It is unchanged when predictions are shifted per unit, for example to match a known aggregate. $\rho_w$ is the median over units of the within-unit Spearman correlation. `block_bootstrap_r2_within` resamples blocks of units with replacement and recomputes the pooled $R^2_w$.
