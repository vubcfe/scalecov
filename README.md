# ScaleCov

**ScaleCov** (coverage across scales) calibrates and checks prediction intervals of spatial maps across nested scales: pixels, units (for example fields or plots) and groups of units (for example farms or districts).

A map with valid pixel intervals does not automatically give valid intervals for the means of fields or farms, because pixel errors inside a unit share a common component. `scalecov` makes this checkable. It takes predictions, observations and the identifiers of nested units, and works with any model.

## Features

- **Pixel intervals:** hierarchical conformal prediction, in which every calibration unit has the same total weight.
- **Intervals for unit and group means**, with five constructions:
  - `independent`: pixel errors treated as independent;
  - `full`: pixel errors treated as fully correlated;
  - `variogram`: a within-unit exponential covariance;
  - `variogram_re`: the same plus a unit random effect;
  - `direct`: conformal calibration at the scale of use.
- **Coverage diagnostics by scale**, with cross-fold calibration on out-of-fold residuals.
- **Calibration from aggregates only:**
  - the ratio κ of pixel to unit-mean quantiles;
  - the pixel coverage reached with a borrowed κ;
  - calibration on a few densely measured units.
- **Skill inside units:** within-unit R² and Spearman correlation, with block-bootstrap intervals.
- **Simulated maps** with a known nested error structure, for testing and teaching.

`scalecov` depends only on numpy, pandas and scipy.

## Installation

```bash
pip install scalecov            # after the first release
pip install -e ".[dev]"         # from a clone, with test tools
```

## Quick start

```python
import scalecov as sc

d = sc.simulate.nested_field(n_groups=200, units_per_group=3, sigma_group=0.4, sigma_unit=0.6, random_state=0)
d["fold"] = d.group % 5

res = sc.cross_calibrate(d, fold="fold", levels=["unit", "group"], alphas=(0.1,))
print(sc.summarise(res))
```

The input is one row per pixel, with these columns:
- the residual `error = observation - prediction`, out of fold;
- one column per nested level, finest first (here `unit`, then `group`);
- pixel coordinates `x` and `y`, needed only for the variogram constructions;
- the fold.

Folds should keep dependent units together: for example, all fields of a farm, or farms that share land.

In the simulated example, nominal 90 % intervals for unit means cover 13 % of units when pixel errors are treated as independent, 27 % with a within-unit variogram, and 90 % with direct calibration. For group means, adding a unit random effect is not enough (81 %), because units of a group share errors too. See [examples/quickstart.py](examples/quickstart.py).

## Notes on the constructions

- **`direct`** is the only construction that adapts to error shared at every level, because it is calibrated at that level. It needs only the unit aggregates (mean residuals of calibration units).
- **`variogram`** fits the covariance to residuals centred on their unit means. It cannot see an error shared by all pixels of a unit. When the correlation range is not small relative to the unit size, it is also biased low.
- **`variogram_re`** adds the unit-level variance that the within-unit covariance does not explain. It treats units of the same group as independent.
- **Conformal guarantees are marginal** over exchangeable calibration and test units. Cross-fold calibration is cross-conformal, so the finite-sample guarantee of split conformal prediction does not apply exactly. Always check coverage empirically, which is what this package is for.

## Citation

If you use ScaleCov, please cite the software (see `CITATION.cff`; DOI to be added on release).

## License

MIT
