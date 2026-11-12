# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Sensitivity of the BARC coverage results to the fold design, the seed and the ScaleCov settings.

1. Fold design and seed: folds of whole farms or of whole sections, seeds 0-2 (the seed changes the
   assignment of farms or sections to folds and the training subsample). With section folds, farms
   span several folds, so only records and section-years are evaluated (ScaleCov refuses the farm level).
2. Settings, with farm folds and seed 0: pixels per calibration unit (``max_per_unit``) and the
   distance bins of the within-unit covariance (``bin_width``, ``max_lag``).

    python run_sensitivity.py --features data/features.parquet --out results/sensitivity.csv
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import pandas as pd

import scalecov as sc

HERE = pathlib.Path(__file__).resolve().parent


def oof(features: str, fold_level: str, seed: int, workdir: pathlib.Path) -> pd.DataFrame:
    dst = workdir / f"oof_{fold_level}_seed{seed}.parquet"
    if not dst.exists():
        subprocess.run([sys.executable, str(HERE / "model.py"), "--features", features, "--out", str(dst),
                        "--fold-level", fold_level, "--seed", str(seed)], check=True)
    d = pd.read_parquet(dst)
    d["section_year"] = d.section + " " + d.year.astype(str)
    d["farm_year"] = d.farm + " " + d.year.astype(str)
    return d


def coverage(d, levels, alpha=0.1, **kw) -> pd.DataFrame:
    rows = []
    for crop, dc in d.groupby("crop"):
        res = sc.cross_calibrate(dc, fold="fold", levels=levels, alphas=(alpha,), x="x", y="y", **kw)
        t = sc.summarise(res)
        t["crop"] = crop
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/features.parquet")
    ap.add_argument("--out", default="results/sensitivity.csv")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    work = out.parent / "sensitivity_oof"
    work.mkdir(parents=True, exist_ok=True)
    cov = {"max_lag": 200.0, "bin_width": 10.0}
    rows = []
    for fold_level, levels in (("farm", ["record", "section_year", "farm_year"]),
                               ("section", ["record", "section_year"])):
        for seed in (0, 1, 2):
            d = oof(a.features, fold_level, seed, work)
            t = coverage(d, levels, covariance_kwargs=cov)
            t["experiment"], t["fold_level"], t["seed"], t["setting"] = "folds", fold_level, seed, "default"
            rows.append(t)
            print(fold_level, seed, "done", flush=True)
    d = oof(a.features, "farm", 0, work)
    for mpu in (250, 1000, None):
        t = coverage(d, ["record", "section_year", "farm_year"], max_per_unit=mpu, covariance_kwargs=cov)
        t["experiment"], t["fold_level"], t["seed"], t["setting"] = "max_per_unit", "farm", 0, str(mpu)
        rows.append(t)
    for bw, ml in ((5.0, 100.0), (10.0, 200.0), (20.0, 400.0)):
        t = coverage(d, ["record", "section_year", "farm_year"], covariance_kwargs={"max_lag": ml, "bin_width": bw},
                     methods=["variogram", "variogram_re"])
        t["experiment"], t["fold_level"], t["seed"], t["setting"] = "covariance_bins", "farm", 0, f"{bw:g}/{ml:g}"
        rows.append(t)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out, index=False)
    print(f"wrote {out} ({len(res)} rows)")


if __name__ == "__main__":
    main()
