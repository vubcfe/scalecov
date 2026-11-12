# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
import numpy as np
import pytest

from scalecov import PixelConformal, conformal_quantile, unit_weights


def test_finite_sample_coverage_is_valid_and_tight():
    rng = np.random.default_rng(1)
    n, alpha, trials = 49, 0.1, 20000
    cal = rng.standard_normal((trials, n))
    test = rng.standard_normal(trials)
    q = np.array([conformal_quantile(c, alpha) for c in cal])
    cov = np.mean(np.abs(test) <= q)
    # exchangeable data: 1 - alpha <= coverage <= 1 - alpha + 1 / (n + 1), up to Monte Carlo error
    assert 1 - alpha - 0.006 <= cov <= 1 - alpha + 1 / (n + 1) + 0.006


def test_uniform_weights_match_unweighted():
    r = np.random.default_rng(2).standard_normal(37)
    for alpha in (0.05, 0.1, 0.2, 0.5):
        assert conformal_quantile(r, alpha) == conformal_quantile(r, alpha, np.ones_like(r))


def test_too_few_scores_give_infinite_quantile():
    assert conformal_quantile([0.1, 0.2, 0.3], 0.1) == float("inf")


def test_invalid_inputs():
    with pytest.raises(ValueError):
        conformal_quantile([1.0], 1.5)
    with pytest.raises(ValueError):
        conformal_quantile([1.0, 2.0], 0.1, [1.0])


def test_unit_weights_sum_to_one_per_unit():
    u = np.array(["a"] * 3 + ["b"] * 7 + ["c"])
    w = unit_weights(u)
    for k in "abc":
        assert w[u == k].sum() == pytest.approx(1.0)


def test_pixel_conformal_subsample_and_interval():
    rng = np.random.default_rng(3)
    units = np.repeat(np.arange(30), 50)
    pc = PixelConformal(max_per_unit=20).fit(rng.standard_normal(len(units)), units)
    assert len(pc.residuals_) == 30 * 20
    lo, hi = pc.interval([0.0, 1.0], 0.1)
    assert np.allclose(hi - lo, 2 * pc.quantile(0.1))


def test_equal_unit_weight_resists_one_large_unit():
    # one huge unit with tiny residuals must not shrink the interval for the other units
    rng = np.random.default_rng(4)
    big = rng.normal(0, 0.01, 100000)
    small = rng.normal(0, 1.0, (40, 50)).ravel()
    units = np.r_[np.zeros(100000), np.repeat(np.arange(1, 41), 50)]
    res = np.r_[big, small]
    q_eq = PixelConformal(max_per_unit=None).fit(res, units).quantile(0.1)
    q_px = PixelConformal(max_per_unit=None, equal_unit_weight=False).fit(res, units).quantile(0.1)
    assert q_eq > 1.0 > q_px
