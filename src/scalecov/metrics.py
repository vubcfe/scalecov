# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Skill inside units and its uncertainty.

A map can score well at the pixel or unit level only because it gets the level of each unit
right. The within-unit coefficient of determination removes the unit means of both truth and
prediction and measures the spatial pattern inside units alone; it is unchanged when the
prediction is shifted so that each unit mean matches a known aggregate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

__all__ = ["block_bootstrap_r2_within", "r2_unit_means", "r2_within", "rho_within", "within_sums"]


def within_sums(y, pred, units) -> pd.DataFrame:
    """Per-unit sums ``sse`` and ``sst`` of the within-unit R2 (truth and prediction centred on their
    own unit means). ``R2_w = 1 - sum(sse) / sum(sst)`` for any set of units."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(pred, float), "u": np.asarray(units)})
    g = d.groupby("u")
    yc = d.y - g.y.transform("mean")
    pc = d.p - g.p.transform("mean")
    return pd.DataFrame({"sse": (yc - pc) ** 2, "sst": yc ** 2, "u": d.u}).groupby("u")[["sse", "sst"]].sum()


def r2_within(y, pred, units) -> float:
    """Within-unit R2: ``1 - sum((y - ybar_u) - (p - pbar_u))^2 / sum((y - ybar_u)^2)``."""
    s = within_sums(y, pred, units)
    return float(1 - s.sse.sum() / s.sst.sum())


def rho_within(y, pred, units, min_pixels: int = 20) -> float:
    """Median over units (with at least ``min_pixels`` pixels) of the Spearman correlation between
    truth and prediction inside the unit."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(pred, float), "u": np.asarray(units)})
    rs = []
    for _, g in d.groupby("u"):
        if len(g) >= min_pixels and g.y.nunique() > 1 and g.p.nunique() > 1:
            rs.append(spearmanr(g.y, g.p).statistic)
    return float(np.median(rs)) if rs else float("nan")


def r2_unit_means(y, pred, units) -> float:
    """R2 of unit means of the prediction against unit means of the truth."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(pred, float), "u": np.asarray(units)})
    m = d.groupby("u")[["y", "p"]].mean()
    return float(1 - ((m.y - m.p) ** 2).sum() / ((m.y - m.y.mean()) ** 2).sum())


def block_bootstrap_r2_within(y, pred, units, blocks, n_boot: int = 2000, level: float = 0.95,
                              random_state: int = 0) -> dict:
    """Percentile interval of the within-unit R2 under resampling of blocks with replacement.

    Blocks are groups of units that are dependent (for example farms that share land); each
    replicate draws blocks with replacement and recomputes the pooled within-unit R2.

    Returns
    -------
    dict
        ``estimate``, ``lo``, ``hi`` and the replicate values ``boot``.
    """
    s = within_sums(y, pred, units)
    ub = pd.Series(np.asarray(blocks), index=np.asarray(units)).groupby(level=0).first()
    s["block"] = ub.reindex(s.index).to_numpy()
    sb = s.groupby("block")[["sse", "sst"]].sum()
    rng = np.random.default_rng(random_state)
    nb = len(sb)
    w = rng.multinomial(nb, np.full(nb, 1 / nb), size=n_boot).astype(float)
    boot = 1 - (w @ sb.sse.to_numpy()) / (w @ sb.sst.to_numpy())
    a = (1 - level) / 2
    return {"estimate": float(1 - sb.sse.sum() / sb.sst.sum()), "lo": float(np.quantile(boot, a)),
            "hi": float(np.quantile(boot, 1 - a)), "boot": boot}
