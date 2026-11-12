# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Within-unit covariance of residuals and the variance of unit means it implies.

The covariance is fitted only to pairs of pixels inside the same unit, after removing each unit's
mean residual. It therefore describes how errors co-vary within a unit, and by construction it
cannot see an error component shared by all pixels of a unit; that component is estimated
separately from unit means (see :class:`scalecov.UnitIntervals`).

Removing unit means also removes the part of a spatial field that varies on scales comparable to
the unit: when the correlation range is not small relative to the unit size, the fitted range and
the implied variance of unit means are biased low. This is the same blind spot as the shared unit
error, and the reason why intervals for unit means should be calibrated at the unit scale
(``direct``) or include a unit random effect (``variogram_re``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["ExpCovariance", "fit_within_unit_covariance", "mean_error_variance"]


@dataclass
class ExpCovariance:
    """Exponential covariance with nugget: ``C(0) = nugget + sill``, ``C(h) = sill * exp(-h / range_)``
    for ``h > 0``. ``3 * range_`` is the practical range."""

    nugget: float
    sill: float
    range_: float

    def __call__(self, h):
        h = np.asarray(h, dtype=float)
        return np.where(h > 0, self.sill * np.exp(-h / self.range_), self.nugget + self.sill)


def fit_within_unit_covariance(residuals, units, x, y, max_lag: float | None = None, bin_width: float | None = None,
                               max_pixels_per_unit: int = 300, min_pixels: int = 30, min_pairs_per_bin: int = 50,
                               random_state: int = 0) -> ExpCovariance:
    """Fit an exponential covariance with nugget to residuals centred on their unit means.

    Parameters
    ----------
    residuals, units, x, y : array-like
        Pixel residuals, unit identifiers and coordinates (any metric unit, consistent within units).
    max_lag : float, optional
        Largest distance used; default half the largest within-unit extent.
    bin_width : float, optional
        Width of the distance bins of the empirical covariance; default ``max_lag / 20``.
    max_pixels_per_unit : int
        Random subsample of pixels per unit for the pair computations.
    min_pixels : int
        Units with fewer pixels are ignored.
    min_pairs_per_bin : int
        Bins with fewer pairs are ignored.

    Returns
    -------
    ExpCovariance
        ``nugget + sill`` equals the within-unit residual variance.
    """
    rng = np.random.default_rng(random_state)
    d = pd.DataFrame({"e": np.asarray(residuals, float), "u": np.asarray(units), "x": np.asarray(x, float),
                      "y": np.asarray(y, float)})
    d["ec"] = d.e - d.groupby("u").e.transform("mean")
    var = float(d.ec.var())
    if max_lag is None:
        ext = d.groupby("u").agg(dx=("x", np.ptp), dy=("y", np.ptp))
        max_lag = float(np.hypot(ext.dx, ext.dy).max() / 2)
    if bin_width is None:
        bin_width = max_lag / 20
    hs, ps = [], []
    for _, g in d.groupby("u"):
        if len(g) < min_pixels:
            continue
        if len(g) > max_pixels_per_unit:
            g = g.sample(max_pixels_per_unit, random_state=int(rng.integers(1 << 30)))
        i, j = np.triu_indices(len(g), 1)
        h = np.hypot(g.x.values[i] - g.x.values[j], g.y.values[i] - g.y.values[j])
        m = (h > 0) & (h <= max_lag)
        hs.append(h[m])
        ps.append(g.ec.values[i[m]] * g.ec.values[j[m]])
    if not hs:
        raise ValueError("no unit has enough pixels to fit a covariance")
    h, p = np.concatenate(hs), np.concatenate(ps)
    b = np.floor(h / bin_width).astype(int)
    hc, cc = [], []
    for k in np.unique(b):
        m = b == k
        if m.sum() >= min_pairs_per_bin:
            hc.append(h[m].mean())
            cc.append(p[m].mean())
    hc, cc = np.array(hc), np.array(cc)
    best = None
    for a in np.geomspace(bin_width / 4, max_lag * 20, 80):
        g = np.exp(-hc / a)
        s = min(max(float(g @ cc / (g @ g)), 0.0), var)
        sse = float(np.sum((cc - s * g) ** 2))
        if best is None or sse < best[0]:
            best = (sse, s, a)
    _, s, a = best
    return ExpCovariance(nugget=var - s, sill=s, range_=float(a))


def mean_error_variance(x, y, cov: ExpCovariance, max_pixels: int = 400, random_state: int = 0) -> float:
    """Variance of the mean of within-unit errors over the pixels of one unit.

    For more than ``max_pixels`` pixels, the mean pair covariance is estimated on a random
    subsample, and the diagonal term is scaled to the full number of pixels.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n_all = len(x)
    if n_all == 0:
        return float("nan")
    if n_all > max_pixels:
        k = np.random.default_rng(random_state).choice(n_all, max_pixels, replace=False)
        x, y = x[k], y[k]
    m = len(x)
    if m == 1:
        return float(cov.nugget + cov.sill)
    h = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    c = cov.sill * np.exp(-h / cov.range_)
    np.fill_diagonal(c, 0.0)
    off = c.sum() / (m * (m - 1))
    return float(off + (cov.nugget + cov.sill - off) / n_all)
