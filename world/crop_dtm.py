"""Crop a 50 m x 50 m patch of the USGS Mars 2020 TRN HiRISE DTM (Jezero) without
downloading the 1.8 GB mosaic (rasterio does HTTP range reads through /vsicurl/).

    .venv/Scripts/python world/crop_dtm.py --lat 18.43910 --lon 77.40880
    .venv/Scripts/python world/crop_dtm.py --lat 18.45 --lon 77.40 --size 50 --out terrain/jezero_crop.tif

Prints slope stats so you can check the patch isn't a flat parking lot.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import rasterio
from rasterio.windows import from_bounds

URL = ("/vsicurl/https://planetarymaps.usgs.gov/mosaic/mars2020_trn/HiRISE/"
       "JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=18.43910)  # delta-front patch: plateau + scarp (landing site is too flat)
    ap.add_argument("--lon", type=float, default=77.40880)
    ap.add_argument("--size", type=float, default=50.0, help="patch side in metres")
    ap.add_argument("--pad", type=float, default=4.0, help="extra metres each side for gradients")
    ap.add_argument("--src", default=URL, help="URL or local path to the mosaic")
    ap.add_argument("--out", default="terrain/jezero_crop.tif")
    a = ap.parse_args()

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES"):
        with rasterio.open(a.src) as src:
            print("CRS:", src.crs)
            print("res:", src.res, "size:", src.width, "x", src.height)
            # equirectangular, lat_ts=0, lon_0=0: x = R*lon, y = R*lat (radians)
            R = src.crs.to_dict().get("R") or src.crs.to_dict().get("a") or 3396190.0
            cx, cy = R * math.radians(a.lon), R * math.radians(a.lat)
            h = a.size / 2 + a.pad
            win = from_bounds(cx - h, cy - h, cx + h, cy + h, src.transform).round_offsets().round_lengths()
            data = src.read(1, window=win, masked=True)
            profile = src.profile | {"height": data.shape[0], "width": data.shape[1],
                                     "transform": src.window_transform(win), "compress": "deflate",
                                     "tiled": False}
            profile.pop("blockxsize", None); profile.pop("blockysize", None)

    if data.mask.any():
        print(f"WARNING: {int(data.mask.sum())} nodata pixels in crop; pick another spot")
    z = data.filled(np.nan).astype(np.float32)
    with rasterio.open(a.out, "w", **profile) as dst:
        dst.write(z, 1)

    gy, gx = np.gradient(z, abs(profile["transform"].e), profile["transform"].a)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    print(f"saved {a.out}  shape={z.shape}  elev {np.nanmin(z):.1f}..{np.nanmax(z):.1f} m "
          f"(relief {np.nanmax(z) - np.nanmin(z):.1f} m)")
    print(f"slope deg: median {np.nanmedian(slope):.1f}  p95 {np.nanpercentile(slope, 95):.1f}  "
          f"max {np.nanmax(slope):.1f}  share>20deg {(slope > 20).mean():.1%}")


if __name__ == "__main__":
    main()
