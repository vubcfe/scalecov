# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Download the public inputs of the canopy height example (public forests of Hauts-de-France, France).

* Public forests (ONF, via the IGN Geoplateforme WFS) and their management compartments (ONF "parcelles
  forestieres", via the ONF WFS). Licence Ouverte 2.0 (Etalab).
* Index of the IGN LiDAR HD tiles over these forests, with the address of the canopy height model
  (MNH, 0.5 m) of each tile and its flight date. Licence Ouverte 2.0. The tiles themselves are downloaded
  by ``prepare.py`` for the selected compartments only.
The global canopy height map of Lang et al. (2023, CC BY 4.0) is not downloaded: ``prepare.py`` reads only the
windows it needs from the cloud-optimised GeoTIFFs published by the authors (``--lang-local`` uses local copies).

    python get_data.py --out data
"""
from __future__ import annotations

import argparse
import pathlib
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import pandas as pd

IGN_WFS = "https://data.geopf.fr/wfs/ows"
ONF_WFS = "http://ws.carmencarto.fr/WFS/105/ONF_Forets"
DEPARTMENTS = ("02", "59", "60", "62", "80")          # Hauts-de-France


def fetch(url: str, dst: pathlib.Path, tries: int = 5) -> pathlib.Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            tmp.rename(dst)
            return dst
        except OSError:
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"download failed: {url}")


def wfs(base: str, params: dict, dst: pathlib.Path, tries: int = 6) -> gpd.GeoDataFrame:
    """WFS request cached in ``dst``; a response that is not a readable vector file (the services sometimes
    return an error or an unrelated page) is discarded and requested again."""
    for i in range(tries):
        fetch(base + "?" + urllib.parse.urlencode(params), dst)
        try:
            return gpd.read_file(dst)
        except (OSError, ValueError, RuntimeError):     # unreadable response (pyogrio errors are RuntimeError)
            head = dst.read_bytes()[:4000].decode("utf-8", "ignore")
            if ("FeatureCollection" in head and "featureMember" not in head and '"features":[{' not in
                    head.replace(" ", "")):
                return gpd.GeoDataFrame(geometry=[], crs=2154)      # valid response without features
            dst.unlink(missing_ok=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"no valid WFS response for {dst.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    deps = ",".join(f"'{d}'" for d in DEPARTMENTS)
    forests = wfs(IGN_WFS, {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                            "TYPENAMES": "ONF.FORETS_PUBLIQUES:ONF_FORETS_PUBLIQUES", "SRSNAME": "EPSG:2154",
                            "OUTPUTFORMAT": "application/json", "CQL_FILTER": f"cinse_dep IN ({deps})"},
                  out / "raw" / "forests.json").to_crs(2154)
    forests.to_file(out / "forests.gpkg")
    print(len(forests), "public forests")
    def one(f):
        x0, y0, x1, y1 = f.geometry.bounds
        bbox = f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f},EPSG:2154"
        c = wfs(ONF_WFS, {"SERVICE": "WFS", "VERSION": "1.1.0", "REQUEST": "GetFeature", "TYPENAME": "PARC_PUBL_FR",
                          "BBOX": bbox}, out / "raw" / f"compartments_{f.iidtn_frt}.gml")
        if len(c) and "iidtn_frt" in c:
            c = c[c.iidtn_frt == f.iidtn_frt].set_crs(2154, allow_override=True)
        t = wfs(IGN_WFS, {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                          "TYPENAMES": "IGNF_LIDAR-HD_METADONNEE:metadata", "SRSNAME": "EPSG:2154",
                          "OUTPUTFORMAT": "application/json", "BBOX": bbox},
                out / "raw" / f"lidar_{f.iidtn_frt}.json")
        if len(t):
            t = t.to_crs(2154)[["coordonnees_nw", "url_mnh", "date_debut_acquisition", "code_mission", "geometry"]]
        return c, t

    comps, tiles = [], []
    with ThreadPoolExecutor(8) as ex:
        for i, (c, t) in enumerate(ex.map(one, list(forests.itertuples()))):
            if len(c):
                comps.append(c)
            if len(t):
                tiles.append(t)
            if i % 25 == 0:
                print(i, len(c), len(t), flush=True)
    comps = pd.concat(comps, ignore_index=True)
    gpd.GeoDataFrame(comps, crs=2154).to_file(out / "compartments.gpkg")
    tiles = pd.concat(tiles, ignore_index=True).drop_duplicates("coordonnees_nw")
    gpd.GeoDataFrame(tiles, crs=2154).to_file(out / "lidar_tiles.gpkg")
    print(len(comps), "compartments,", len(tiles), "LiDAR HD tiles")
    print("done")


if __name__ == "__main__":
    main()
