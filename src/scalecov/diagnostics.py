# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Coverage diagnostics across scales and cross-fold calibration.

:func:`cross_calibrate` reproduces the protocol used with out-of-fold predictions: for each fold,
everything is calibrated on the residuals of the other folds and evaluated on the fold itself,
so that no test residual is used for calibration. Folds should group dependent units (for example
farms that share land) to keep calibration and test exchangeable.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta, binom

from .conformal import PixelConformal
from .units import METHODS, UnitIntervals

__all__ = ["check_folds", "cross_calibrate", "pixel_coverage_by_unit", "summarise"]


def pixel_coverage_by_unit(residuals, units, half_width: float) -> pd.Series:
    """Share of each unit's pixels whose absolute residual is within ``half_width``."""
    inside = np.abs(np.asarray(residuals, float)) <= half_width
    return pd.Series(inside).groupby(np.asarray(units)).mean()


def check_folds(data: pd.DataFrame, fold: str, levels: Sequence[str]) -> None:
    """Raise ``ValueError`` if a unit of any level has pixels in more than one fold.

    A unit split across folds would be evaluated as several partial units, and its calibration and
    test parts would not be independent. Folds must therefore be formed from whole units of the
    coarsest level that is evaluated (for example whole farms when farm means are evaluated).
    """
    for lv in levels:
        n = data.groupby(lv)[fold].nunique()
        bad = n[n > 1]
        if len(bad):
            raise ValueError(f"{len(bad)} unit(s) of level '{lv}' span several folds (for example "
                             f"{bad.index[0]!r}); build folds from whole units of the coarsest level")


def cross_calibrate(data: pd.DataFrame, fold: str, levels: Sequence[str], alphas=(0.1,), residual: str = "error",
                    x: str = "x", y: str = "y", methods: Sequence[str] = METHODS, max_per_unit: int | None = 1000,
                    random_state: int = 0, covariance_kwargs: dict | None = None) -> dict:
    """Cross-fold calibration and evaluation of pixel and unit intervals.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per pixel with out-of-fold residuals, the nested unit columns ``levels`` (finest first),
        coordinates and the fold column.
    fold : str
        Column with the fold of each pixel.
    alphas : sequence of float
        Miscoverage levels.

    Returns
    -------
    dict
        ``units``: one row per test unit, level, method and alpha (``half_width``, ``error``,
        ``covered``, ``fold``); ``pixels``: one row per fold and alpha with pixel quantile and coverage,
        and the 10th percentile of per-unit pixel coverage; ``fits``: fitted covariance and unit
        variance per fold.
    """
    check_folds(data, fold, levels)
    unit_rows, px_rows, fits = [], [], []
    for k in sorted(data[fold].unique()):
        cal, test = data[data[fold] != k], data[data[fold] == k]
        ui = UnitIntervals(levels, x=x, y=y, residual=residual, methods=methods, max_per_unit=max_per_unit,
                           random_state=random_state, covariance_kwargs=covariance_kwargs).fit(cal)
        f = {"fold": k, "unit_var": ui.unit_var_}
        if hasattr(ui, "cov_"):
            f.update(nugget=ui.cov_.nugget, sill=ui.cov_.sill, range=ui.cov_.range_)
        fits.append(f)
        px = PixelConformal(max_per_unit, random_state=random_state).fit(cal[residual], cal[levels[0]])
        for a in alphas:
            iv = ui.intervals(test, a)
            iv["fold"], iv["alpha"] = k, a
            unit_rows.append(iv)
            q = px.quantile(a)
            by_unit = pixel_coverage_by_unit(test[residual], test[levels[0]], q)
            px_rows.append({"fold": k, "alpha": a, "q_px": q, "n_pixels": len(test),
                            "coverage": float(np.mean(np.abs(test[residual]) <= q)),
                            "unit_p10_coverage": float(by_unit.quantile(0.1))})
    return {"units": pd.concat(unit_rows, ignore_index=True), "pixels": pd.DataFrame(px_rows),
            "fits": pd.DataFrame(fits)}


def coverage_interval(n_covered, n, conf: float = 0.95):
    """Clopper-Pearson interval for a coverage estimated from ``n_covered`` of ``n`` test units.

    Units are treated as independent trials. Units of one fold share their calibration, so the
    interval is approximate and, if anything, too narrow; it shows how much of a gap to the nominal
    level can be sampling noise.

    Returns
    -------
    tuple of float
        Lower and upper bound (``nan`` when ``n`` is 0).
    """
    k, n = np.asarray(n_covered, float), np.asarray(n, float)
    a = (1 - conf) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.where(k > 0, beta.ppf(a, k, n - k + 1), 0.0)
        hi = np.where(k < n, beta.ppf(1 - a, k + 1, n - k), 1.0)
    lo, hi = np.where(n > 0, lo, np.nan), np.where(n > 0, hi, np.nan)
    return (float(lo), float(hi)) if lo.ndim == 0 else (lo, hi)


def nominal_range(n, alpha: float, conf: float = 0.95):
    """Range of observed coverage that is consistent with the nominal level ``1 - alpha`` for ``n``
    independent test units: the central ``conf`` interval of Binomial(n, 1 - alpha) / n."""
    n = np.asarray(n, float)
    lo = binom.ppf((1 - conf) / 2, n, 1 - alpha) / n
    hi = binom.ppf(1 - (1 - conf) / 2, n, 1 - alpha) / n
    return (float(lo), float(hi)) if lo.ndim == 0 else (lo, hi)


def summarise(result: dict, conf: float = 0.95) -> pd.DataFrame:
    """Coverage and mean width pooled over folds, by alpha, level and method; pixel rows included
    (pooled pixel coverage weighted by the number of test pixels of each fold).

    For unit levels, ``coverage_lo`` and ``coverage_hi`` give a Clopper-Pearson interval of the
    coverage (see :func:`coverage_interval`) and ``consistent`` tells whether the observed coverage lies
    in the range expected under the nominal level (see :func:`nominal_range`). With few units these
    ranges are wide: 18 units at 90 % are consistent with an observed coverage from 72 % to 100 %.
    """
    u = result["units"]
    t = u.groupby(["alpha", "level", "method"], sort=False).agg(
        coverage=("covered", "mean"), mean_width=("half_width", lambda s: 2 * s.mean()),
        n_units=("unit", "size"), n_covered=("covered", "sum")).reset_index()
    t["coverage_lo"], t["coverage_hi"] = coverage_interval(t.n_covered.to_numpy(), t.n_units.to_numpy(), conf)
    lo, hi = nominal_range(t.n_units.to_numpy(), t.alpha.to_numpy(), conf)
    t["consistent"] = (t.coverage >= lo - 1e-12) & (t.coverage <= hi + 1e-12)
    t = t.drop(columns="n_covered")
    p = result["pixels"]
    pr = p.groupby("alpha").apply(lambda g: pd.Series({
        "coverage": float(np.average(g.coverage, weights=g.n_pixels)),
        "mean_width": float(np.average(2 * g.q_px, weights=g.n_pixels)),
        "n_units": np.nan}), include_groups=False).reset_index()
    pr["level"], pr["method"] = "pixel", "conformal"
    for c in ("coverage_lo", "coverage_hi", "consistent"):
        pr[c] = np.nan
    return pd.concat([pr[t.columns], t.astype({"consistent": object})], ignore_index=True)
