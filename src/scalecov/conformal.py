# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Conformal quantiles and pixel-level intervals calibrated with grouped (nested) data.

The pixel-level procedure follows hierarchical (grouped) conformal prediction: every calibration
unit (for example a field) receives the same total weight, whatever its number of pixels, so that
large units do not dominate the quantile. Its guarantee is marginal over exchangeable units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["PixelConformal", "conformal_quantile", "unit_weights"]


def conformal_quantile(scores, alpha: float, weights=None) -> float:
    """Conformal ``1 - alpha`` quantile of absolute scores.

    Parameters
    ----------
    scores : array-like
        Calibration nonconformity scores; absolute values are taken.
    alpha : float
        Miscoverage level in (0, 1).
    weights : array-like, optional
        Non-negative weights of the calibration scores. Without weights, the finite-sample
        corrected quantile ``ceil((n + 1)(1 - alpha))``-th smallest score is returned. With weights,
        the weighted quantile is computed with the test point given the largest calibration weight,
        a conservative choice.

    Returns
    -------
    float
        The quantile, or ``inf`` when there are too few calibration scores for the level.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    r = np.abs(np.asarray(scores, dtype=float))
    if r.size == 0:
        return float("inf")
    if weights is None:
        n = r.size
        k = int(np.ceil((n + 1) * (1 - alpha)))
        return float("inf") if k > n else float(np.sort(r)[k - 1])
    w = np.asarray(weights, dtype=float)
    if w.shape != r.shape or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, not all zero, and match scores")
    o = np.argsort(r)
    cw = np.cumsum(w[o]) / (w.sum() + w.max())
    i = int(np.searchsorted(cw, 1 - alpha))
    return float("inf") if i >= r.size else float(r[o][i])


def unit_weights(units) -> np.ndarray:
    """Weights that give every unit the same total weight: ``1 / n_pixels(unit)``."""
    s = pd.Series(np.asarray(units))
    return 1.0 / s.map(s.value_counts()).to_numpy(dtype=float)


class PixelConformal:
    """Pixel prediction intervals calibrated on residuals of calibration units.

    Parameters
    ----------
    max_per_unit : int or None
        At most this many pixels per calibration unit are used (random subsample), which bounds
        memory for large maps. ``None`` uses all pixels.
    equal_unit_weight : bool
        Give every calibration unit the same total weight (hierarchical conformal). If False, every
        pixel has the same weight.
    random_state : int
        Seed of the subsample.
    """

    def __init__(self, max_per_unit: int | None = 1000, equal_unit_weight: bool = True, random_state: int = 0):
        self.max_per_unit = max_per_unit
        self.equal_unit_weight = equal_unit_weight
        self.random_state = random_state

    def fit(self, residuals, units) -> PixelConformal:
        """Store calibration residuals ``y - prediction`` and their unit identifiers."""
        d = pd.DataFrame({"e": np.asarray(residuals, dtype=float), "u": np.asarray(units)})
        if self.max_per_unit is not None:
            d = d.sample(frac=1.0, random_state=self.random_state)
            d = d[d.groupby("u").cumcount().to_numpy() < self.max_per_unit]
        self.residuals_ = d.e.to_numpy()
        self.units_ = d.u.to_numpy()
        self.weights_ = unit_weights(self.units_) if self.equal_unit_weight else None
        return self

    def quantile(self, alpha: float) -> float:
        """Half-width of the ``1 - alpha`` pixel interval."""
        return conformal_quantile(self.residuals_, alpha, self.weights_)

    def interval(self, prediction, alpha: float):
        """Lower and upper bounds ``prediction -/+ quantile(alpha)``."""
        q = self.quantile(alpha)
        p = np.asarray(prediction, dtype=float)
        return p - q, p + q
