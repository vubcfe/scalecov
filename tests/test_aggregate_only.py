# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
import numpy as np
import pytest

from scalecov import PixelConformal, coverage_vs_kappa, k_dense_coverage, kappa, unit_quantile
from scalecov.simulate import nested_field

ALPHA = 0.1


@pytest.fixture(scope="module")
def data():
    d = nested_field(n_groups=200, units_per_group=2, unit_shape=(10, 10), sigma_group=0.3, sigma_unit=0.5,
                     sigma_spatial=0.5, range_m=15.0, nugget=0.5, random_state=21)
    return d[d.group % 2 == 0], d[d.group % 2 == 1]


def test_kappa_is_ratio_of_quantiles(data):
    cal, _ = data
    k = kappa(cal.error, cal.unit, ALPHA)
    assert k["kappa"] == pytest.approx(k["q_px"] / k["q_unit"])
    assert k["q_unit"] == pytest.approx(unit_quantile(cal.error, cal.unit, ALPHA))
    assert k["kappa"] > 1   # pixel errors exceed unit-mean errors


def test_coverage_vs_kappa_is_monotone_and_own_kappa_is_nominal(data):
    cal, test = data
    k = kappa(cal.error, cal.unit, ALPHA)
    curve = coverage_vs_kappa(test.error, k["q_unit"], np.geomspace(0.1, 10, 50))
    assert np.all(np.diff(curve.to_numpy()) >= 0)
    own = coverage_vs_kappa(test.error, k["q_unit"], [k["kappa"]]).iloc[0]
    assert abs(own - (1 - ALPHA)) < 0.03


def test_k_dense_approaches_nominal(data):
    cal, test = data
    r = k_dense_coverage(cal.error, cal.unit, test.error, ALPHA, ks=(1, 5, 20), n_draws=100,
                         test_units=test.unit).set_index("k")
    full = np.mean(np.abs(test.error) <= PixelConformal().fit(cal.error, cal.unit).quantile(ALPHA))
    assert r.coverage_mean.loc[20] == pytest.approx(full, abs=0.03)
    assert r.coverage_p05.loc[1] < r.coverage_p05.loc[20]          # spread shrinks with k
    assert "unit_p10_mean" in r.columns
    assert set(r.index) == {1, 5, 20}
