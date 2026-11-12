# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Pixel intervals when only aggregates, or only a few dense units, are available.

With unit aggregates only, the errors of unit means are observable (prediction mean against the
reported aggregate), so a unit-scale quantile ``q_unit`` can be calibrated, but the pixel-scale
quantile ``q_px`` cannot. A pixel interval then needs ``kappa = q_px / q_unit`` from elsewhere,
for example from another region or crop with dense data. This module estimates ``kappa`` where
dense data exist, gives the pixel coverage a transferred ``kappa`` would reach, and evaluates the
alternative of calibrating pixel intervals on a few densely measured units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .conformal import PixelConformal, conformal_quantile, unit_weights

__all__ = ["coverage_vs_kappa", "k_dense_coverage", "kappa", "unit_quantile"]


def unit_quantile(residuals, units, alpha: float) -> float:
    """Conformal quantile of the mean residuals of units: the unit-scale quantile that aggregates
    alone can calibrate."""
    m = pd.Series(np.asarray(residuals, float)).groupby(np.asarray(units)).mean()
    return conformal_quantile(m.to_numpy(), alpha)


def kappa(residuals, units, alpha: float, max_per_unit: int | None = 1000, random_state: int = 0) -> dict:
    """Ratio of the pixel to the unit-mean conformal quantile on dense calibration data.

    Returns
    -------
    dict
        ``kappa``, ``q_px`` and ``q_unit``.
    """
    q_px = PixelConformal(max_per_unit, random_state=random_state).fit(residuals, units).quantile(alpha)
    q_u = unit_quantile(residuals, units, alpha)
    return {"kappa": q_px / q_u, "q_px": q_px, "q_unit": q_u}


def coverage_vs_kappa(test_residuals, q_unit: float, kappas) -> pd.Series:
    """Pixel coverage of the interval ``+/- kappa * q_unit`` on test pixels, for each ``kappa``.

    ``q_unit`` is calibrated from aggregates of the target; reading the returned curve at a
    ``kappa`` borrowed from elsewhere gives the coverage that the transfer would achieve.
    """
    e = np.abs(np.asarray(test_residuals, float))
    k = np.asarray(kappas, float)
    return pd.Series([float(np.mean(e <= g * q_unit)) for g in k], index=pd.Index(k, name="kappa"),
                     name="coverage")


def k_dense_coverage(cal_residuals, cal_units, test_residuals, alpha: float, ks=(1, 2, 5, 10, 20),
                     n_draws: int = 200, test_units=None, random_state: int = 0) -> pd.DataFrame:
    """Pixel coverage when the pixel quantile is calibrated on ``k`` randomly drawn dense units.

    Each calibration unit has equal total weight. For every ``k`` (not larger than the number of
    calibration units), ``n_draws`` random sets of units are drawn.

    Returns
    -------
    pandas.DataFrame
        One row per ``k`` with the mean, 5th and 95th percentile of test coverage over draws; with
        ``test_units``, also the mean over draws of the 10th percentile of per-unit coverage.
    """
    rng = np.random.default_rng(random_state)
    e = np.asarray(cal_residuals, float)
    u = np.asarray(cal_units)
    t = np.abs(np.asarray(test_residuals, float))
    uniq = np.unique(u)
    idx = {k: np.flatnonzero(u == k) for k in uniq}
    rows = []
    for k in ks:
        if k > len(uniq):
            continue
        covs, lows = [], []
        for _ in range(n_draws):
            pick = rng.choice(uniq, k, replace=False)
            sel = np.concatenate([idx[p] for p in pick])
            q = conformal_quantile(e[sel], alpha, unit_weights(u[sel]))
            inside = t <= q
            covs.append(float(inside.mean()))
            if test_units is not None:
                lows.append(float(pd.Series(inside).groupby(np.asarray(test_units)).mean().quantile(0.1)))
        r = {"k": k, "coverage_mean": float(np.mean(covs)), "coverage_p05": float(np.quantile(covs, 0.05)),
             "coverage_p95": float(np.quantile(covs, 0.95))}
        if lows:
            r["unit_p10_mean"] = float(np.mean(lows))
        rows.append(r)
    return pd.DataFrame(rows)
