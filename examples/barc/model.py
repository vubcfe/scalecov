# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Out-of-fold yield predictions from monthly Sentinel-2 composites, one model per crop.

A gradient-boosted tree regressor with fixed settings is trained on pixels; this is a simple model to
produce realistic prediction errors for ScaleCov, not a tuned yield model. Folds hold out whole
farms (``--fold-level farm``, the default) over all years, so that every unit evaluated later
(record, section-year, farm-year) lies entirely in one fold and calibration units are not neighbours of
test units. Each prediction comes from the model of the fold in which its farm was held out.

    python model.py --features data/features.parquet --out data/oof.parquet --crops CORN SOYBEAN
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

MAX_TRAIN_PER_RECORD = 1000


def group_folds(groups: pd.Series, k: int, seed: int) -> pd.Series:
    """Assign whole groups to k folds, balancing pixel counts (largest groups first)."""
    sizes = groups.value_counts().sample(frac=1.0, random_state=seed).sort_values(ascending=False, kind="stable")
    load, fold = np.zeros(k), {}
    for s, n in sizes.items():
        j = int(np.argmin(load))
        fold[s] = j
        load[j] += n
    return groups.map(fold)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/features.parquet")
    ap.add_argument("--out", default="data/oof.parquet")
    ap.add_argument("--crops", nargs="+", default=["CORN", "SOYBEAN"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold-level", default="farm", choices=["farm", "section", "field"])
    a = ap.parse_args()
    d = pd.read_parquet(a.features)
    feats = [c for c in d.columns if "_m" in c and c.split("_m")[-1].isdigit()]
    out = []
    for crop in a.crops:
        dc = d[(d.crop == crop) & d.yield_t_ha.notna()].reset_index(drop=True)
        dc = dc[dc[feats].notna().mean(axis=1) > 0.5].reset_index(drop=True)   # pixels with usable imagery
        k_eff = min(a.k, dc[a.fold_level].nunique())     # never more folds than groups
        dc["fold"] = group_folds(dc[a.fold_level], k_eff, a.seed)
        dc["pred"] = np.nan
        for k in range(k_eff):
            tr = dc[dc.fold != k]
            tr = tr.groupby("record", group_keys=False).apply(
                lambda g: g.sample(min(len(g), MAX_TRAIN_PER_RECORD), random_state=a.seed))
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                              min_samples_leaf=50, l2_regularization=1.0, random_state=a.seed)
            m.fit(tr[feats], tr.yield_t_ha)
            te = dc.fold == k
            dc.loc[te, "pred"] = m.predict(dc.loc[te, feats])
        dc["error"] = dc.yield_t_ha - dc.pred
        out.append(dc[["record", "farm", "section", "field", "year", "crop", "x", "y", "yield_t_ha", "pred",
                       "error", "fold"]])
        print(f"{crop}: {len(dc)} pixels, {dc.record.nunique()} records, {dc.section.nunique()} sections; "
              f"pixel RMSE {np.sqrt(np.mean(dc.error ** 2)):.2f} t/ha")
    pd.concat(out, ignore_index=True).to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
