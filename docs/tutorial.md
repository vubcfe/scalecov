# Tutorial

The full script is [`examples/quickstart.py`](https://github.com/vubcfe/scalecov/blob/main/examples/quickstart.py); it runs in a few seconds.

## 1. Input

One row per pixel with:
- an **out-of-fold residual** (`error = observation - prediction`): the model that produced the prediction must not have seen the pixel's unit;
- one column per **nested level**, finest first (for example `field`, then `farm`);
- pixel **coordinates** in a metric unit (needed only for the variogram constructions);
- a **fold** column that keeps dependent units together, for example all fields of a farm.

```python
import scalecov as sc

d = sc.simulate.nested_field(n_groups=200, units_per_group=3, unit_shape=(12, 12), sigma_group=0.4,
                             sigma_unit=0.6, sigma_spatial=0.6, range_m=30.0, nugget=0.4, random_state=0)
d["fold"] = d.group % 5
```

## 2. Coverage by scale

```python
res = sc.cross_calibrate(d, fold="fold", levels=["unit", "group"], alphas=(0.1,))
print(sc.summarise(res))
```

For each fold, everything is calibrated on the other folds and evaluated on the fold itself. The summary gives, for the pixel level and for every unit level and construction, the coverage of nominal 90 % intervals and their mean width. In this example, independent pixel errors give about 13 % coverage for unit means and a within-unit variogram about 27 %, while direct calibration reaches about 90 %.

Each unit row also has `coverage_lo` and `coverage_hi`, a Clopper-Pearson interval of the coverage, and `consistent`, which tells whether the observed coverage lies in the range expected under the nominal level for that number of units (`sc.nominal_range`). With few units this range is wide: for 18 units and 90 % intervals, any coverage from 72 % to 100 % is consistent with correct calibration, so a gap of a few points is not evidence of a problem.

## 3. Only aggregates in the target region

```python
cal, test = d[d.fold != 0], d[d.fold == 0]
k = sc.kappa(cal.error, cal.unit, alpha=0.1)             # where dense data exist
q_unit = sc.unit_quantile(cal.error, cal.unit, alpha=0.1)  # needs aggregates only
sc.coverage_vs_kappa(test.error, q_unit, [k["kappa"]])    # coverage with a borrowed kappa
sc.k_dense_coverage(cal.error, cal.unit, test.error, alpha=0.1, ks=(1, 5, 20))
```

## 4. Skill inside units

```python
r = sc.block_bootstrap_r2_within(d.truth, d.pred, d.unit, d.group)
```
