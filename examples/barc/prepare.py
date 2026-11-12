# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Grid the cleaned BARC yield-monitor points onto the 10 m Sentinel-2 grid.

Input: the yearly archives downloaded by ``get_data.py``. Only the cleaned point files
(``3B--Cleaned_Yield_Data_CSV``: UTM 18N easting, northing and dry yield in bushels per acre) are read.
The kriged grids of the archive are smoothed and are not used.

Output (``--out``):
- ``pixels.parquet``: one row per 10 m pixel and harvest record with ``record``, ``farm``, ``section``,
  ``field``, ``year``, ``crop``, ``x``, ``y`` (pixel centre, EPSG:32618), ``n_points``, ``yield_bu_ac`` and
  ``yield_t_ha``;
- ``records.csv``: one row per harvest record (field, year and crop; double crops are separate records);
- ``sites.geojson``: one square site per field (bounding box of all its pixels over all years, plus a margin),
  in EPSG:32618, for S2Cube.

Field identifiers are normalised across years (for example ``Central_Farm_1_7`` and ``CentralFarm_1_7``).
The nested levels are field < section (first token after the farm name, for example ``Central 1``) < farm.

    python prepare.py --raw data/raw --out data
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import re
import zipfile

import numpy as np
import pandas as pd

PIXEL = 10.0
MIN_POINTS = 3
# kg per bushel (USDA standard test weights), to convert bu/ac to t/ha
KG_PER_BU = {"CORN": 25.4012, "SORGHUM": 25.4012, "RYE": 25.4012, "SOYBEAN": 27.2155, "WHEAT": 27.2155,
             "BARLEY": 21.7724}
HA_PER_AC = 0.40468564
CROP_ALIASES = {"SOYBEANS": "SOYBEAN"}
FARM_ALIASES = {"CENTRALFARM": "CENTRAL", "EASTFARM": "EAST", "NORTHFARM": "NORTH", "SOUTHFARM": "SOUTH",
                "LINKAGEFARM": "LINKAGE"}


def parse_name(name: str) -> dict | None:
    """Farm, section, field, year and crop from a cleaned-point file name, or None if not recognised."""
    stem = re.sub(r"_(pt_out|xyz_final)\.csv$", "", pathlib.Path(name).name, flags=re.IGNORECASE)
    tokens = stem.split("_")
    if tokens[0].upper() == "USDA":
        tokens = tokens[1:]
    years = [i for i, t in enumerate(tokens) if re.fullmatch(r"(19|20)\d\d", t)]
    if not years or years[0] + 1 >= len(tokens):
        return None
    iy = years[0]
    head, year, crop = tokens[:iy], int(tokens[iy]), tokens[iy + 1].upper()
    if len(head) >= 2 and head[1].upper() == "FARM":
        farm, rest = head[0].upper(), head[2:]
    else:
        farm, rest = FARM_ALIASES.get(head[0].upper(), head[0].upper()), head[1:]
    if not rest:
        return None
    rest = [r.upper().replace("&", "") for r in rest]   # e.g. SC&D and SCD are the same block
    section = f"{farm} {rest[0]}" if len(rest) > 1 else farm
    field = f"{farm} {' '.join(rest)}"
    crop = CROP_ALIASES.get(crop, crop)
    return {"farm": farm, "section": section, "field": field, "year": year, "crop": crop}


def read_points(raw: bytes) -> pd.DataFrame:
    """Cleaned points; some years have no header line."""
    first = raw.split(b"\n", 1)[0]
    has_header = any(c.isalpha() for c in first.decode("utf-8", "ignore"))
    d = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    d = d.iloc[:, :3]
    d.columns = ["x", "y", "yield_bu_ac"]
    return d.apply(pd.to_numeric, errors="coerce").dropna()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--margin", type=float, default=40.0, help="site margin around each field (m)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pix, recs, skipped = [], [], []
    for zpath in sorted(pathlib.Path(a.raw).glob("[12][0-9][0-9][0-9].zip")):
        with zipfile.ZipFile(zpath) as z:
            for name in sorted(z.namelist()):
                if "3B--" not in name or not name.lower().endswith(".csv"):
                    continue
                meta = parse_name(name)
                if meta is None:
                    skipped.append(name)
                    continue
                pts = read_points(z.read(name))
                pts = pts[pts.yield_bu_ac > 0]
                if pts.empty:
                    skipped.append(name)
                    continue
                ix = np.floor(pts.x / PIXEL).astype(np.int64)
                iy = np.floor(pts.y / PIXEL).astype(np.int64)
                g = pts.assign(ix=ix, iy=iy).groupby(["ix", "iy"]).yield_bu_ac.agg(["mean", "size"]).reset_index()
                g = g[g["size"] >= MIN_POINTS]
                record = f"{meta['field']} {meta['year']} {meta['crop']}"
                kg = KG_PER_BU.get(meta["crop"])
                p = pd.DataFrame({"record": record, **meta, "x": (g.ix + 0.5) * PIXEL, "y": (g.iy + 0.5) * PIXEL,
                                  "n_points": g["size"].to_numpy(), "yield_bu_ac": g["mean"].to_numpy()})
                p["yield_t_ha"] = p.yield_bu_ac * kg / HA_PER_AC / 1000 if kg else np.nan
                pix.append(p)
                recs.append({"record": record, **meta, "source": name, "n_points": len(pts), "n_pixels": len(p)})
    pixels = pd.concat(pix, ignore_index=True)
    records = pd.DataFrame(recs)
    pixels.to_parquet(out / "pixels.parquet", index=False)
    records.to_csv(out / "records.csv", index=False)
    feats = []
    for field, d in pixels.groupby("field"):
        x0, x1 = d.x.min() - PIXEL / 2 - a.margin, d.x.max() + PIXEL / 2 + a.margin
        y0, y1 = d.y.min() - PIXEL / 2 - a.margin, d.y.max() + PIXEL / 2 + a.margin
        feats.append({"type": "Feature", "properties": {"id": field.replace(" ", "_")},
                      "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}})
    with open(out / "sites.geojson", "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    print(f"{len(records)} records, {records.field.nunique()} fields, {len(pixels)} pixels; skipped {len(skipped)}")
    for s in skipped:
        print("  skipped:", s)


if __name__ == "__main__":
    main()
