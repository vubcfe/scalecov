# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Select compartments, download their LiDAR HD canopy height windows and build the 10 m pixel table.

Selection rule (fixed before any result was computed):
  * compartments of 5 to 40 ha whose area is at least 99 % covered by LiDAR HD tiles with a canopy height model;
  * forests with at least 10 such compartments;
  * in each forest, a random sample of at most 15 compartments (seed 0).

Reference canopy top height, as in Lang et al. (2023) for airborne LiDAR: the 0.5 m canopy height model is
max-pooled to 2.5 m and then max-pooled in a disc of 12.5 m radius (the 25 m GEDI footprint), and read at the
centres of a 10 m grid (Lambert-93). Cells whose centre lies inside the compartment are kept; cells with any
missing LiDAR value in the footprint are dropped. The prediction and its standard deviation are read from the
global map at the cell centre (nearest 10 m cell of the map).

    python prepare.py --data data
Output: data/pixels.parquet (one row per 10 m cell) and data/compartments_selected.gpkg.
"""
from __future__ import annotations

import argparse
import pathlib
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from get_data import fetch
from pyproj import Transformer
from scipy.ndimage import maximum_filter
from shapely import contains_xy

MIN_HA, MAX_HA, MIN_PER_FOREST, PER_FOREST, SEED = 5, 40, 10, 15, 0
RADIUS = 12.5          # m, half of the GEDI footprint
BLOCK = 5              # 0.5 m pixels per 2.5 m block
LANG_TILES = ("N48E000", "N48E003")
LANG_URL = ("https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=%2F3deg_cogs&files="
            "ETH_GlobalCanopyHeight_10m_2020_{tile}_{kind}.tif")


def open_lang(tile: str, kind: str, local: pathlib.Path | None):
    """Open a tile of the Lang et al. (2023) map: a local copy if given, otherwise the published
    cloud-optimised GeoTIFF, of which only the blocks that are read are transferred."""
    if local is not None:
        return rasterio.open(local / f"ETH_GlobalCanopyHeight_10m_2020_{tile}_{kind}.tif")
    url = urllib.parse.quote(LANG_URL.format(tile=tile, kind=kind), safe="")
    return rasterio.open(f"/vsicurl?use_head=no&list_dir=no&url={url}")


def select(data: pathlib.Path) -> gpd.GeoDataFrame:
    comps = gpd.read_file(data / "compartments.gpkg").to_crs(2154)
    tiles = gpd.read_file(data / "lidar_tiles.gpkg").to_crs(2154)
    tiles = tiles[tiles.url_mnh.notna() & (tiles.url_mnh != "")]
    comps = comps[comps.geometry.notna()].copy()
    comps["ha"] = comps.area / 1e4
    comps = comps[(comps.ha >= MIN_HA) & (comps.ha <= MAX_HA)]
    cover = tiles.union_all()
    comps["covered"] = comps.geometry.intersection(cover).area / comps.area
    comps = comps[comps.covered >= 0.99]
    comps["compartment"] = comps.iidtn_frt + "_" + comps.ccod_prf.astype(str)
    comps = comps.drop_duplicates("compartment")
    n = comps.groupby("iidtn_frt").size()
    comps = comps[comps.iidtn_frt.isin(n[n >= MIN_PER_FOREST].index)]
    rng = np.random.default_rng(SEED)
    keep = []
    for _, g in comps.sort_values("compartment").groupby("iidtn_frt"):
        keep.append(g.iloc[rng.permutation(len(g))[:PER_FOREST]])
    return gpd.GeoDataFrame(pd.concat(keep), crs=2154)


WMS = ("https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&EXCEPTIONS=text/xml&REQUEST=GetMap"
       "&LAYERS=IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93&FORMAT=image/geotiff&STYLES=&CRS=EPSG:2154"
       "&BBOX={x0},{y0},{x1},{y1}&WIDTH={w}&HEIGHT={h}")


def chm_box(geom) -> tuple[float, float, float, float]:
    """Window around a compartment (padded by the footprint radius and one cell), with edges on the native
    0.5 m grid of the LiDAR HD canopy height model (edges at 0.25 m + k * 0.5 m), so no resampling occurs."""
    x0, y0, x1, y1 = geom.bounds
    pad = RADIUS + 10
    return (np.floor((x0 - pad) / 10) * 10 - 0.25, np.floor((y0 - pad) / 10) * 10 - 0.25,
            np.ceil((x1 + pad) / 10) * 10 - 0.25, np.ceil((y1 + pad) / 10) * 10 - 0.25)


def chm_url(x0, y0, x1, y1) -> str:
    return WMS.format(x0=x0, y0=y0, x1=x1, y1=y1, w=round((x1 - x0) / 0.5), h=round((y1 - y0) / 0.5))


def fetch_chm(url: str, dst: pathlib.Path, tries: int = 6) -> pathlib.Path:
    """Download a canopy height window; a response that is not a readable GeoTIFF is requested again."""
    for i in range(tries):
        fetch(url, dst)
        try:
            with rasterio.open(dst) as r:
                r.read(1, window=rasterio.windows.Window(0, 0, 1, 1))
            return dst
        except rasterio.errors.RasterioIOError:
            dst.unlink(missing_ok=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"no valid canopy height window for {dst.name}")


def disc_footprint(radius_cells: float) -> np.ndarray:
    r = int(np.floor(radius_cells))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2) <= radius_cells ** 2


def canopy_top_height(chm: np.ndarray) -> np.ndarray:
    """0.5 m CHM (nan = missing) -> 2.5 m GEDI-like canopy top height (nan where any input is missing)."""
    h, w = chm.shape
    b = chm[:h - h % BLOCK, :w - w % BLOCK].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    miss = np.isnan(b).any(axis=(1, 3))
    mx = np.where(miss, -np.inf, np.nanmax(np.where(np.isnan(b), -np.inf, b), axis=(1, 3)))
    fp = disc_footprint(RADIUS / (0.5 * BLOCK))
    top = maximum_filter(mx, footprint=fp, mode="constant", cval=-np.inf)
    bad = maximum_filter(miss.astype(np.uint8), footprint=fp, mode="constant", cval=1) > 0
    return np.where(bad, np.nan, top)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lang-local", default=None, help="folder with local copies of the Lang et al. tiles")
    a = ap.parse_args()
    data = pathlib.Path(a.data)
    sel = select(data)
    sel.to_file(data / "compartments_selected.gpkg")
    print(len(sel), "compartments in", sel.iidtn_frt.nunique(), "forests", flush=True)
    tiles = gpd.read_file(data / "lidar_tiles.gpkg").to_crs(2154)
    tiles = tiles[tiles.url_mnh.notna() & (tiles.url_mnh != "")]
    (data / "chm").mkdir(exist_ok=True)
    boxes = {c.compartment: chm_box(c.geometry) for c in sel.itertuples()}
    jobs = [(chm_url(*b), data / "chm" / f"{k}.tif") for k, b in boxes.items()]
    print(len(jobs), "canopy height windows to download (cached)", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(lambda j: fetch_chm(*j), jobs)):
            if i % 50 == 0:
                print("windows", i, flush=True)
    local = pathlib.Path(a.lang_local) if a.lang_local else None
    env = rasterio.Env(CPL_VSIL_CURL_USE_HEAD="NO", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_CACHEMAX=1024)
    env.__enter__()
    lang = {t: (open_lang(t, "Map", local), open_lang(t, "Map_SD", local)) for t in LANG_TILES}
    to_ll = Transformer.from_crs(2154, 4326, always_xy=True)
    rows = []
    for k, c in enumerate(sel.itertuples()):
        x0, y0, x1, y1 = boxes[c.compartment]
        with rasterio.open(data / "chm" / f"{c.compartment}.tif") as r:
            chm = r.read(1).astype(float)
            nod = r.nodata
        chm[(chm <= -9998) | (chm == nod)] = np.nan
        chm = np.clip(chm, 0, None)
        top = canopy_top_height(chm)                        # 2.5 m grid, origin (x0, y1)
        cx, cy = np.meshgrid(np.arange(x0 + 0.25 + 5, x1, 10), np.arange(y1 + 0.25 - 5, y0, -10))
        cx, cy = cx.ravel(), cy.ravel()
        inside = contains_xy(c.geometry, cx, cy)
        cx, cy = cx[inside], cy[inside]
        col = ((cx - x0) / 2.5).astype(int)
        row = ((y1 - cy) / 2.5).astype(int)
        ref = top[row, col]
        lon, lat = to_ll.transform(cx, cy)
        pred = np.full(len(cx), np.nan)
        sd = np.full(len(cx), np.nan)
        for t, (m, s) in lang.items():
            lo, la = int(t[4:7]), int(t[1:3])
            ok = (lon >= lo) & (lon < lo + 3) & (lat >= la) & (lat < la + 3)
            if ok.any():
                r_, c_ = rasterio.transform.rowcol(m.transform, lon[ok], lat[ok])
                r_, c_ = np.asarray(r_), np.asarray(c_)
                win = rasterio.windows.Window(c_.min(), r_.min(), c_.max() - c_.min() + 1, r_.max() - r_.min() + 1)
                mv, sv = m.read(1, window=win).astype(float), s.read(1, window=win).astype(float)
                pv = mv[r_ - r_.min(), c_ - c_.min()]
                svv = sv[r_ - r_.min(), c_ - c_.min()]
                pv[pv == m.nodata] = np.nan
                svv[svv == s.nodata] = np.nan
                pred[ok], sd[ok] = pv, svv
        flight = tiles.loc[tiles.intersects(c.geometry), "date_debut_acquisition"].astype(str).str[:10].max()
        rows.append(pd.DataFrame({"forest": c.iidtn_frt, "forest_name": c.llib_frt, "compartment": c.compartment,
                                  "x": cx, "y": cy, "ref": ref, "pred": pred, "sd": sd, "flight": flight}))
        if k % 50 == 0:
            print("compartment", k, c.compartment, len(cx), flush=True)
    px = pd.concat(rows, ignore_index=True)
    px = px.dropna(subset=["ref", "pred"])
    px["error"] = px.ref - px.pred
    px.to_parquet(data / "pixels.parquet", index=False)
    print(len(px), "pixels;", px.compartment.nunique(), "compartments;", px.forest.nunique(), "forests")


if __name__ == "__main__":
    main()
