# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Simulated maps with a known nested error structure, for tests, examples and teaching.

The prediction error of pixel ``i`` in unit ``u`` of group ``g`` is

    e_i = a_g + b_u + s_i + n_i,

with a group effect ``a_g ~ N(0, sigma_group^2)``, a unit effect ``b_u ~ N(0, sigma_unit^2)``, a
spatially correlated within-unit component ``s`` with exponential covariance
``sigma_spatial^2 exp(-h / range_m)``, and a nugget ``n_i ~ N(0, nugget^2)``. Groups and units are
independent; spatial components of different units are independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["nested_field"]


def nested_field(n_groups: int = 30, units_per_group: int = 4, unit_shape=(20, 20), pixel_size: float = 10.0,
                 sigma_group: float = 0.3, sigma_unit: float = 0.6, sigma_spatial: float = 0.8,
                 range_m: float = 50.0, nugget: float = 0.4, mean: float = 5.0, signal_sd: float = 1.0,
                 random_state: int = 0) -> pd.DataFrame:
    """Simulate truth, prediction and a nested error structure on rectangular units.

    Returns
    -------
    pandas.DataFrame
        One row per pixel with columns ``group``, ``unit``, ``x``, ``y`` (metres, local to the
        unit), ``truth``, ``pred`` and ``error = truth - pred``.
    """
    rng = np.random.default_rng(random_state)
    ny, nx = unit_shape
    yy, xx = np.meshgrid(np.arange(ny) * pixel_size, np.arange(nx) * pixel_size, indexing="ij")
    x, y = xx.ravel(), yy.ravel()
    chol = None
    if sigma_spatial > 0:
        h = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
        c = sigma_spatial ** 2 * np.exp(-h / range_m) + 1e-10 * np.eye(len(x))
        chol = np.linalg.cholesky(c)
    rows = []
    for g in range(n_groups):
        a = rng.normal(0, sigma_group)
        for k in range(units_per_group):
            b = rng.normal(0, sigma_unit)
            s = chol @ rng.standard_normal(len(x)) if chol is not None else 0.0
            e = a + b + s + rng.normal(0, nugget, len(x))
            pred = mean + rng.normal(0, signal_sd, len(x))
            rows.append(pd.DataFrame({"group": g, "unit": g * units_per_group + k, "x": x, "y": y,
                                      "truth": pred + e, "pred": pred, "error": e}))
    return pd.concat(rows, ignore_index=True)
