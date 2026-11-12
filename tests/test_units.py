# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
import numpy as np
import pytest

from scalecov import UnitIntervals
from scalecov.simulate import nested_field

ALPHA = 0.1


def split(d):
    cal = d[d.group % 2 == 0]
    test = d[d.group % 2 == 1]
    return cal, test


def coverage(d, **kw):
    cal, test = split(d)
    ui = UnitIntervals(["unit", "group"], **kw).fit(cal)
    rep = ui.coverage(test, ALPHA)
    return {(r.level, r.method): r.coverage for r in rep.itertuples()}


@pytest.fixture(scope="module")
def nested():
    # shared errors at unit and group level
    return nested_field(n_groups=240, units_per_group=3, unit_shape=(12, 12), sigma_group=0.5, sigma_unit=0.6,
                        sigma_spatial=0.6, range_m=40.0, nugget=0.4, random_state=11)


@pytest.fixture(scope="module")
def no_shared():
    # no shared error and a range short relative to units: the within-unit covariance describes everything
    return nested_field(n_groups=240, units_per_group=3, unit_shape=(12, 12), sigma_group=0.0, sigma_unit=0.0,
                        sigma_spatial=0.6, range_m=8.0, nugget=0.4, random_state=12)


def test_shared_unit_error_breaks_independent_and_variogram(nested):
    c = coverage(nested)
    assert c[("unit", "independent")] < 0.4
    assert c[("unit", "variogram")] < 0.75
    assert c[("unit", "full")] > 0.9


def test_random_effect_valid_for_units_not_groups(nested):
    c = coverage(nested)
    assert abs(c[("unit", "variogram_re")] - (1 - ALPHA)) < 0.05
    assert c[("group", "variogram_re")] < 1 - ALPHA - 0.1   # group effect ignored


def test_direct_valid_at_every_level():
    # conformal validity holds on average over calibration sets, so average over simulations
    cov = {"unit": [], "group": []}
    for s in range(12):
        d = nested_field(n_groups=120, units_per_group=3, unit_shape=(6, 6), sigma_group=0.5, sigma_unit=0.6,
                         sigma_spatial=0.3, range_m=20.0, nugget=0.4, random_state=100 + s)
        c = coverage(d, methods=["direct"])
        for lv, vals in cov.items():
            vals.append(c[(lv, "direct")])
    assert np.mean(cov["unit"]) >= 1 - ALPHA - 0.02
    assert np.mean(cov["group"]) >= 1 - ALPHA - 0.04
    assert np.mean(cov["unit"]) <= 1 - ALPHA + 0.05


def test_variogram_close_to_nominal_without_shared_error(no_shared):
    c = coverage(no_shared)
    assert abs(c[("unit", "variogram")] - (1 - ALPHA)) < 0.07
    assert abs(c[("unit", "variogram_re")] - (1 - ALPHA)) < 0.05


def test_output_columns_and_unknown_method(nested):
    cal, test = split(nested)
    iv = UnitIntervals(["unit", "group"], methods=["direct"]).fit(cal).intervals(test, ALPHA)
    assert set(iv.columns) >= {"level", "unit", "n", "method", "half_width", "error", "covered"}
    with pytest.raises(ValueError):
        UnitIntervals(["unit"], methods=["nope"])
