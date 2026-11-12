# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Intervals for the mean error of units and groups of units, from pixel-level information.

The error of a unit mean is the mean of its pixel errors. Five constructions are provided:

``independent``
    pixel errors independent: half-width ``q_px / sqrt(n)``;
``full``
    pixel errors fully correlated: half-width ``q_px``;
``variogram``
    a within-unit exponential covariance (see :mod:`scalecov.variogram`) gives the variance of the
    unit mean from the unit's pixel geometry; Gaussian interval. For a group, unit means are combined
    with area weights and treated as independent;
``variogram_re``
    ``variogram`` plus a unit random effect, the variance of calibration unit-mean errors that the
    within-unit covariance does not explain; unit effects are independent within a group;
``direct``
    split conformal on the mean residuals of calibration units at the same level.

Only ``direct`` adapts to error shared at every level, because it is calibrated at that level.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .conformal import PixelConformal, conformal_quantile
from .variogram import ExpCovariance, fit_within_unit_covariance, mean_error_variance

__all__ = ["METHODS", "UnitIntervals"]

METHODS = ("independent", "full", "variogram", "variogram_re", "direct")


class UnitIntervals:
    """Calibrate and evaluate intervals for unit and group means.

    Parameters
    ----------
    levels : sequence of str
        Column names of the nested units, finest first, for example ``["field", "farm"]``. Every finer
        unit must lie in exactly one coarser unit.
    x, y : str
        Columns with pixel coordinates (needed for the variogram methods only).
    residual : str
        Column with residuals ``observation - prediction``.
    methods : sequence of str
        Subset of :data:`METHODS`.
    max_per_unit : int or None
        Pixel subsample per unit for the pixel quantile.
    random_state : int
        Seed for subsampling.
    covariance_kwargs : dict, optional
        Options passed to :func:`scalecov.variogram.fit_within_unit_covariance`.
    """

    def __init__(self, levels: Sequence[str], x: str = "x", y: str = "y", residual: str = "error",
                 methods: Sequence[str] = METHODS, max_per_unit: int | None = 1000, random_state: int = 0,
                 covariance_kwargs: dict | None = None):
        unknown = set(methods) - set(METHODS)
        if unknown:
            raise ValueError(f"unknown methods: {sorted(unknown)}")
        self.levels = list(levels)
        self.x, self.y, self.residual = x, y, residual
        self.methods = list(methods)
        self.max_per_unit = max_per_unit
        self.random_state = random_state
        self.covariance_kwargs = dict(covariance_kwargs or {})

    # ------------------------------------------------------------------ helpers
    def _needs_cov(self) -> bool:
        return any(m in self.methods for m in ("variogram", "variogram_re"))

    def _unit_table(self, d: pd.DataFrame) -> pd.DataFrame:
        """One row per finest unit: mean error, size, variance of the within-unit mean error, parents."""
        u = self.levels[0]
        t = d.groupby(u).agg(error=(self.residual, "mean"), n=(self.residual, "size"),
                             **{lv: (lv, "first") for lv in self.levels[1:]})
        if self._needs_cov():
            v = {k: mean_error_variance(g[self.x], g[self.y], self.cov_, random_state=self.random_state)
                 for k, g in d.groupby(u)}
            t["v_within"] = pd.Series(v)
        return t

    # ------------------------------------------------------------------ API
    def fit(self, data: pd.DataFrame) -> UnitIntervals:
        """Calibrate on pixels of calibration units (residuals must be out-of-sample)."""
        d = data
        u = self.levels[0]
        self.pixel_ = PixelConformal(self.max_per_unit, random_state=self.random_state).fit(d[self.residual], d[u])
        if self._needs_cov():
            self.cov_: ExpCovariance = fit_within_unit_covariance(d[self.residual], d[u], d[self.x], d[self.y],
                                                                  random_state=self.random_state,
                                                                  **self.covariance_kwargs)
        t = self._unit_table(d)
        self.unit_var_ = (max(float(np.mean(t.error ** 2) - t.v_within.mean()), 0.0)
                          if self._needs_cov() else float("nan"))
        # mean residuals of calibration units at every level, for the direct construction
        self.cal_means_ = {}
        for lv in self.levels:
            self.cal_means_[lv] = d.groupby(lv)[self.residual].mean().to_numpy()
        return self

    def intervals(self, data: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
        """Half-widths for every unit at every level and method.

        Returns
        -------
        pandas.DataFrame
            Columns ``level``, ``unit``, ``n``, ``method``, ``half_width`` and, when the residual column
            is present, ``error`` (mean residual of the unit) and ``covered``.
        """
        z = norm.ppf(1 - alpha / 2)
        q_px = self.pixel_.quantile(alpha)
        has_err = self.residual in data
        d = data if has_err else data.assign(**{self.residual: np.nan})
        t = self._unit_table(d)
        out = []
        for li, lv in enumerate(self.levels):
            if li == 0:
                g = t.assign(unit=t.index)
                v_vario = t.v_within if self._needs_cov() else None
            else:
                w = t.n / t.groupby(lv).n.transform("sum")
                g = t.groupby(lv).agg(n=("n", "sum"))
                g["error"] = (w * t.error).groupby(t[lv]).sum(min_count=1)
                g["unit"] = g.index
                if self._needs_cov():
                    v_vario = (w ** 2 * t.v_within).groupby(t[lv]).sum()
                    v_re = (w ** 2 * (t.v_within + self.unit_var_)).groupby(t[lv]).sum()
            hw = {}
            if "independent" in self.methods:
                hw["independent"] = q_px / np.sqrt(g.n.to_numpy())
            if "full" in self.methods:
                hw["full"] = np.full(len(g), q_px)
            if "variogram" in self.methods:
                hw["variogram"] = z * np.sqrt(v_vario.reindex(g.index).to_numpy())
            if "variogram_re" in self.methods:
                v2 = (t.v_within + self.unit_var_) if li == 0 else v_re
                hw["variogram_re"] = z * np.sqrt(v2.reindex(g.index).to_numpy())
            if "direct" in self.methods:
                hw["direct"] = np.full(len(g), conformal_quantile(self.cal_means_[lv], alpha))
            for m, h in hw.items():
                r = pd.DataFrame({"level": lv, "unit": g.unit.to_numpy(), "n": g.n.to_numpy(), "method": m,
                                  "half_width": h})
                if has_err:
                    r["error"] = g.error.to_numpy()
                    r["covered"] = np.abs(r.error) <= r.half_width
                out.append(r)
        return pd.concat(out, ignore_index=True)

    def coverage(self, data: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
        """Coverage and mean width by level and method (requires the residual column)."""
        iv = self.intervals(data, alpha)
        rep = iv.groupby(["level", "method"], sort=False).agg(coverage=("covered", "mean"),
                                                               mean_width=("half_width", lambda s: 2 * s.mean()),
                                                               n_units=("unit", "size")).reset_index()
        return rep
