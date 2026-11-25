# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""ScaleCov on a public canopy height map checked against airborne LiDAR (public forests, Hauts-de-France).

Nested levels: 10 m pixel < forest compartment < forest. The map was not fitted to these LiDAR data, so its
errors are out of sample; the map is evaluated as it is, without any model fitting here. Folds hold out whole
forests (5 folds, balanced by pixels, seeds 0-2 change the assignment of forests to folds).

Besides the five constructions of ScaleCov, the map's own per-pixel standard deviation is used as users would
use it: pixel interval +/- z * sd; for a compartment mean, +/- z * sqrt(mean(sd^2) / N) if pixel errors are
taken as independent, and +/- z * mean(sd) if they are taken as fully correlated.

    python run_scalecov.py --data data --out results
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm

import scalecov as sc

LEVELS = ["compartment", "forest"]


def forest_folds(px: pd.DataFrame, k: int, seed: int) -> pd.Series:
    """Assign whole forests to k folds, balancing pixel counts (largest first, random order of ties)."""
    sizes = px.forest.value_counts().sample(frac=1.0, random_state=seed).sort_values(ascending=False, kind="stable")
    load, fold = np.zeros(k), {}
    for f, n in sizes.items():
        j = int(np.argmin(load))
        fold[f] = j
        load[j] += n
    return px.forest.map(fold)


def sd_coverage(px: pd.DataFrame, alpha: float) -> dict:
    z = norm.ppf(1 - alpha / 2)
    g = px.groupby("compartment").agg(err=("error", "mean"), n=("error", "size"), msd2=("sd", lambda s: np.mean(s ** 2)),
                                      msd=("sd", "mean"))
    return {"pixel": float(np.mean(np.abs(px.error) <= z * px.sd)),
            "compartment_independent": float(np.mean(np.abs(g.err) <= z * np.sqrt(g.msd2 / g.n))),
            "compartment_full": float(np.mean(np.abs(g.err) <= z * g.msd)),
            "n_compartments": len(g)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    px = pd.read_parquet(pathlib.Path(a.data) / "pixels.parquet")
    summary = {"n_pixels": len(px), "n_compartments": int(px.compartment.nunique()),
               "n_forests": int(px.forest.nunique()), "flight_years": sorted(px.flight.str[:4].unique().tolist()),
               "mean_error": float(px.error.mean()), "rmse": float(np.sqrt(np.mean(px.error ** 2))),
               "r2_within": sc.r2_within(px.ref, px.pred, px.compartment),
               "r2_compartment_means": sc.r2_unit_means(px.ref, px.pred, px.compartment),
               "kappa": sc.kappa(px.error, px.compartment, a.alpha)["kappa"],
               "sd_intervals": sd_coverage(px, a.alpha)}
    tabs = []
    for seed in (0, 1, 2):
        px["fold"] = forest_folds(px, 5, seed)
        res = sc.cross_calibrate(px, fold="fold", levels=LEVELS, alphas=(a.alpha,), x="x", y="y",
                                 random_state=seed, covariance_kwargs={"max_lag": 200.0, "bin_width": 10.0})
        t = sc.summarise(res)
        t["seed"] = seed
        tabs.append(t)
        res["fits"].assign(seed=seed).to_csv(out / f"fits_seed{seed}.csv", index=False)
        if seed == 0:
            kd = []
            for k in sorted(px.fold.unique()):
                cal, test = px[px.fold != k], px[px.fold == k]
                r = sc.k_dense_coverage(cal.error, cal.compartment, test.error, alpha=a.alpha, ks=(1, 2, 5, 10, 20),
                                        n_draws=200, random_state=seed)
                kd.append(r.assign(fold=k))
            pd.concat(kd).groupby("k").mean(numeric_only=True).drop(columns="fold").reset_index().to_csv(
                out / "k_dense.csv", index=False)
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(out / "coverage.csv", index=False)
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print(tab[tab.seed == 0].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
