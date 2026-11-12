# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""ScaleCov on the BARC out-of-fold predictions: coverage by scale, kappa and within-field skill.

Nested levels: 10 m pixel < harvest record (field, year, crop) < section-year < farm-year.
Units of the same section and year share weather and management, so their errors are expected to be
correlated; this is what the unit-level constructions are tested against.

    python run_scalecov.py --oof data/oof.parquet --out results
"""
from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

import scalecov as sc

LEVELS = ["record", "section_year", "farm_year"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", default="data/oof.parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2])
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(a.oof)
    d["section_year"] = d.section + " " + d.year.astype(str)
    d["farm_year"] = d.farm + " " + d.year.astype(str)
    summary = {}
    for crop, dc in d.groupby("crop"):
        res = sc.cross_calibrate(dc, fold="fold", levels=LEVELS, alphas=tuple(a.alphas), x="x", y="y",
                                 covariance_kwargs={"max_lag": 200.0, "bin_width": 10.0})
        tab = sc.summarise(res)
        tab.to_csv(out / f"coverage_{crop.lower()}.csv", index=False)
        res["fits"].to_csv(out / f"fits_{crop.lower()}.csv", index=False)
        bb = sc.block_bootstrap_r2_within(dc.yield_t_ha, dc.pred, dc.record, dc.section, n_boot=2000)
        kd = []
        for k in sorted(dc.fold.unique()):
            cal, test = dc[dc.fold != k], dc[dc.fold == k]
            r = sc.k_dense_coverage(cal.error, cal.record, test.error, alpha=0.1, ks=(1, 2, 5, 10), n_draws=200)
            r["fold"] = k
            kd.append(r)
        kd = pd.concat(kd).groupby("k").mean(numeric_only=True).drop(columns="fold").reset_index()
        kd.to_csv(out / f"k_dense_{crop.lower()}.csv", index=False)
        summary[crop] = {
            "n_pixels": len(dc), "n_records": int(dc.record.nunique()),
            "n_section_years": int(dc.section_year.nunique()), "n_farm_years": int(dc.farm_year.nunique()),
            "r2_within": bb["estimate"], "r2_within_lo": bb["lo"], "r2_within_hi": bb["hi"],
            "rho_within": sc.rho_within(dc.yield_t_ha, dc.pred, dc.record),
            "r2_record_means": sc.r2_unit_means(dc.yield_t_ha, dc.pred, dc.record),
            "kappa": sc.kappa(dc.error, dc.record, 0.1)["kappa"],
        }
        print(crop, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[crop].items()}))
        print(tab[tab.alpha == 0.1].round(3).to_string(index=False))
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
