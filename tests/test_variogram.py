# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
import numpy as np
import pytest

from scalecov import ExpCovariance, fit_within_unit_covariance, mean_error_variance
from scalecov.simulate import nested_field


def test_fit_recovers_parameters_when_range_is_short_relative_to_units():
    d = nested_field(n_groups=40, units_per_group=2, unit_shape=(30, 30), sigma_group=0.0, sigma_unit=0.0,
                     sigma_spatial=0.8, range_m=20.0, nugget=0.4, random_state=3)
    c = fit_within_unit_covariance(d.error, d.unit, d.x, d.y)
    assert c.nugget == pytest.approx(0.4 ** 2, abs=0.05)
    assert c.sill == pytest.approx(0.8 ** 2, abs=0.1)
    assert 14 < c.range_ < 28


def test_fit_is_biased_when_range_is_long_relative_to_units():
    # documented limitation: demeaning inside small units shortens the apparent range
    d = nested_field(n_groups=60, units_per_group=2, unit_shape=(12, 12), sigma_group=0.0, sigma_unit=0.0,
                     sigma_spatial=0.8, range_m=60.0, nugget=0.4, random_state=4)
    c = fit_within_unit_covariance(d.error, d.unit, d.x, d.y)
    assert c.range_ < 60.0
    assert c.nugget + c.sill == pytest.approx(d.error.sub(d.groupby("unit").error.transform("mean")).var())


def test_mean_variance_matches_monte_carlo():
    cov = ExpCovariance(nugget=0.1, sill=0.5, range_=30.0)
    yy, xx = np.meshgrid(np.arange(8) * 10.0, np.arange(8) * 10.0, indexing="ij")
    x, y = xx.ravel(), yy.ravel()
    h = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    full = cov(h)
    exact = full.sum() / len(x) ** 2
    assert mean_error_variance(x, y, cov) == pytest.approx(exact, rel=1e-9)
    # subsampled estimate stays close
    yy, xx = np.meshgrid(np.arange(30) * 10.0, np.arange(30) * 10.0, indexing="ij")
    x2, y2 = xx.ravel(), yy.ravel()
    h2 = np.hypot(x2[:, None] - x2[None], y2[:, None] - y2[None])
    exact2 = cov(h2).sum() / len(x2) ** 2
    assert mean_error_variance(x2, y2, cov, max_pixels=400) == pytest.approx(exact2, rel=0.1)


def test_single_pixel():
    cov = ExpCovariance(0.2, 0.3, 10.0)
    assert mean_error_variance([0.0], [0.0], cov) == pytest.approx(0.5)
