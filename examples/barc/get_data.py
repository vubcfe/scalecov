# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the scalecov contributors
"""Download the USDA Beltsville Agricultural Research Center (BARC) Gridded Yield Data Archive.

Source: Ag Data Commons, https://doi.org/10.15482/usda.adc/28950746 (U.S. Public Domain).
Files are downloaded once into ``--out`` and checked against the MD5 sums published with the record.

    python get_data.py --out data/raw [--years 2016 2017 ...]
"""
import argparse
import hashlib
import json
import pathlib
import urllib.request

ARTICLE = "https://api.figshare.com/v2/articles/28950746"
UA = {"User-Agent": "scalecov-example/0.1"}


def md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--years", nargs="*", default=None, help="years to download (default: all)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(ARTICLE, headers=UA)) as r:
        files = json.load(r)["files"]
    for f in files:
        stem = f["name"].split(".")[0]
        if a.years and stem.isdigit() and stem not in a.years:
            continue
        dst = out / f["name"]
        expected = f.get("computed_md5") or f.get("supplied_md5")
        if dst.exists() and md5(dst) == expected:
            print(f"{f['name']}: present")
            continue
        print(f"{f['name']}: downloading {f['size'] / 1e6:.0f} MB")
        with urllib.request.urlopen(urllib.request.Request(f["download_url"], headers=UA)) as r, open(dst, "wb") as w:
            while chunk := r.read(1 << 20):
                w.write(chunk)
        got = md5(dst)
        if got != expected:
            raise SystemExit(f"{f['name']}: MD5 {got} does not match {expected}")
        print(f"{f['name']}: ok")


if __name__ == "__main__":
    main()
