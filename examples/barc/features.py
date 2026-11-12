# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Monthly Sentinel-2 composites for every yield pixel, from the S2Cube cubes of each season.

For each harvest record of ``pixels.parquet`` (from ``prepare.py``) and the cube of its year, the
median reflectance of clear observations (Scene Classification Layer 4 vegetation or 5 bare soil) is
computed per month from April to November, for the ten bands and NDVI. Pixel coordinates are used only
to find the pixel in the cube; they are not features.

    python features.py --data data --cubes data/s2 --out data/features.parquet
"""
from __future__ import annotations

import argparse
import pathlib

import h5py
import numpy as np
import pandas as pd

MONTHS = range(4, 12)
CLEAR = (4, 5)


def site_features(f: h5py.File, sid: str, months: np.ndarray, bands: list[str]):
    """(H, W, n_features) monthly medians and the grid origin of one site."""
    g = f["sites"][sid]
    X = g["X"][()].astype(np.float32) / 10000.0          # (T, B, H, W)
    scl = g["SCL"][()]
    ok = np.isin(scl, CLEAR) & (X.min(axis=1) > 0)       # (T, H, W)
    b4, b8 = bands.index("B04"), bands.index("B08")
    ndvi = (X[:, b8] - X[:, b4]) / np.maximum(X[:, b8] + X[:, b4], 1e-6)
    stack = np.concatenate([X, ndvi[:, None]], axis=1)    # (T, B+1, H, W)
    stack = np.where(ok[:, None], stack, np.nan)
    feats = []
    for m in MONTHS:
        sel = months == m
        with np.errstate(all="ignore"):
            feats.append(np.nanmedian(stack[sel], axis=0) if sel.any() else np.full(stack.shape[1:], np.nan))
    F = np.stack(feats, 0)                                # (M, B+1, H, W)
    F = F.reshape(-1, *F.shape[2:]).transpose(1, 2, 0)    # (H, W, M*(B+1))
    return F, float(g.attrs["x0"]), float(g.attrs["y1"]), float(g.attrs["res_m"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--cubes", default="data/s2")
    ap.add_argument("--out", default="data/features.parquet")
    a = ap.parse_args()
    pix = pd.read_parquet(pathlib.Path(a.data) / "pixels.parquet")
    out = []
    for year, py in pix.groupby("year"):
        path = pathlib.Path(a.cubes) / f"cube_{year}.h5"
        if not path.exists():
            print(f"{year}: no cube, skipped")
            continue
        with h5py.File(path, "r") as f:
            b = f.attrs["bands"]
            bands = b.split(",") if isinstance(b, str) else [x.decode() if isinstance(x, bytes) else x for x in b]
            dates = pd.to_datetime([d.decode() for d in f["dates"][()]])
            months = dates.month.to_numpy()
            names = [f"{b}_m{m:02d}" for m in MONTHS for b in [*bands, "NDVI"]]
            for field, pf in py.groupby("field"):
                sid = field.replace(" ", "_")
                if sid not in f["sites"]:
                    print(f"{year} {field}: no site, skipped")
                    continue
                F, x0, y1, res = site_features(f, sid, months, bands)
                c = np.round((pf.x.to_numpy() - x0) / res - 0.5).astype(int)
                r = np.round((y1 - pf.y.to_numpy()) / res - 0.5).astype(int)
                inside = (r >= 0) & (r < F.shape[0]) & (c >= 0) & (c < F.shape[1])
                vals = np.full((len(pf), F.shape[2]), np.nan, np.float32)
                vals[inside] = F[r[inside], c[inside]]
                out.append(pd.concat([pf.reset_index(drop=True), pd.DataFrame(vals, columns=names)], axis=1))
        print(f"{year}: done")
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(a.out, index=False)
    print(len(res), "pixels,", res.filter(like="_m").shape[1], "features")


if __name__ == "__main__":
    main()
