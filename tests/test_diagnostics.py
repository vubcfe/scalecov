# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
import numpy as np

from scalecov import check_folds, cross_calibrate, pixel_coverage_by_unit, summarise
from scalecov.simulate import nested_field


def test_cross_calibrate_uses_other_folds_and_pools():
    d = nested_field(n_groups=60, units_per_group=2, unit_shape=(8, 8), random_state=31)
    d["fold"] = d.group % 3
    res = cross_calibrate(d, "fold", ["unit", "group"], alphas=(0.1, 0.2), methods=["independent", "direct"])
    u = res["units"]
    # every test unit appears once per fold, alpha and method
    assert len(u[(u.level == "unit") & (u.alpha == 0.1) & (u.method == "direct")]) == d.unit.nunique()
    assert set(res["pixels"].fold) == {0, 1, 2}
    s = summarise(res)
    assert {"pixel", "unit", "group"} <= set(s.level)
    direct_unit = s[(s.level == "unit") & (s.method == "direct") & (s.alpha == 0.1)].coverage.iloc[0]
    assert 0.8 < direct_unit <= 1.0


def test_pixel_coverage_by_unit():
    cov = pixel_coverage_by_unit([0.1, 2.0, 0.1, 0.1], ["a", "a", "b", "b"], 1.0)
    assert cov["a"] == 0.5 and cov["b"] == 1.0
    assert np.isclose(cov.mean(), 0.75)


def test_units_split_across_folds_are_rejected():
    import pytest
    d = nested_field(n_groups=6, units_per_group=2, unit_shape=(4, 4), random_state=0)
    d["fold"] = d.unit % 3                     # groups are split across folds
    check_folds(d, "fold", ["unit"])           # units themselves are whole: fine
    with pytest.raises(ValueError, match="group"):
        cross_calibrate(d, "fold", ["unit", "group"], methods=["direct"])
