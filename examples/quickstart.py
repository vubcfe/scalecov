# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Quickstart: intervals for pixels, units and groups on a simulated map with shared errors.

Run: python examples/quickstart.py   (a few seconds)
"""

import scalecov as sc

# 1. A map of 600 units (for example fields) in 200 groups (for example farms). Prediction errors have a
#    group effect, a unit effect, a spatially correlated part inside units and a nugget.
d = sc.simulate.nested_field(n_groups=200, units_per_group=3, unit_shape=(12, 12), sigma_group=0.4,
                             sigma_unit=0.6, sigma_spatial=0.6, range_m=30.0, nugget=0.4, random_state=0)
d["fold"] = d.group % 5          # folds group whole groups, so calibration and test are exchangeable

# 2. Cross-fold calibration: every fold is evaluated with intervals calibrated on the other folds.
res = sc.cross_calibrate(d, fold="fold", levels=["unit", "group"], alphas=(0.1,))
print(sc.summarise(res).round(3).to_string(index=False))

# 3. Only aggregates in the target region: borrow kappa or use a few dense units.
cal, test = d[d.fold != 0], d[d.fold == 0]
k = sc.kappa(cal.error, cal.unit, alpha=0.1)
# the unit-mean quantile needs only aggregates; in practice it comes from the target region's own
# aggregates, here from the calibration folds for simplicity
q_unit = sc.unit_quantile(cal.error, cal.unit, alpha=0.1)
print("\nkappa =", round(k["kappa"], 2))
print(sc.coverage_vs_kappa(test.error, q_unit, [0.5 * k["kappa"], k["kappa"], 2 * k["kappa"]]).round(3))
print(sc.k_dense_coverage(cal.error, cal.unit, test.error, alpha=0.1, ks=(1, 5, 20), n_draws=100).round(3))

# 4. Skill inside units, with a block-bootstrap interval over groups.
r = sc.block_bootstrap_r2_within(d.truth, d.pred, d.unit, d.group, n_boot=500)
print(f"\nwithin-unit R2 of the simulated prediction: {r['estimate']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
