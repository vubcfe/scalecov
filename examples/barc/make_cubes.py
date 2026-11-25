# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Build the co-registered Sentinel-2 cubes of the BARC fields with S2Cube, one cube per growing season.

For every year, a project file ``data/s2/project_<year>.yaml`` is written (Sentinel-2 L2A from Microsoft
Planetary Computer, April to November, ten bands at 10 m in UTM 18N, sites from ``data/sites.geojson`` written
by ``prepare.py``, co-registration against the cube itself) and the cube is built and co-registered:

    s2cube build project_<year>.yaml
    s2cube coreg project_<year>.yaml

    python make_cubes.py --data data --years 2016 2017 2018 2019 2020 2021
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess

TEMPLATE = """# S2Cube project: Sentinel-2 L2A over the BARC fields, growing season {year}
name: barc_{year}
start: {year}-04-01
end: {year}-11-30
source: planetary-computer
bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
res: 10
crs: EPSG:32618
output: cube_{year}.h5
workers: {workers}
sites:
  path: ../sites.geojson
  id_field: id
  crs: EPSG:32618
coreg:
  reference: self
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--years", nargs="+", type=int, default=[2016, 2017, 2018, 2019, 2020, 2021])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    s2 = pathlib.Path(a.data) / "s2"
    s2.mkdir(parents=True, exist_ok=True)
    for y in a.years:
        cfg = s2 / f"project_{y}.yaml"
        cfg.write_text(TEMPLATE.format(year=y, workers=a.workers))
        for step in ("build", "coreg"):
            with open(s2 / f"{step}_{y}.log", "w") as log:
                subprocess.run(["s2cube", step, cfg.name], cwd=s2, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(y, "done", flush=True)


if __name__ == "__main__":
    main()
