"""Download native-grid NYC candidate crops with source metadata.

Requires rasterio/numpy. Downloads 1.28 km inspection windows, not full scenes.
No flood segmentation masks are created from point sensors. RTC signed URLs
exist only in memory and are never written to the project.
"""
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "nyc" / "candidate_chips"


def get(url):
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def main():
    manifest = json.loads((ROOT / "configs/datasets/nyc_candidates.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    half = manifest["roi_width_m"] / 2
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT=45,
                      GDAL_HTTP_MAX_RETRY=3, CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"):
        for scene in manifest["scenes"]:
            item = get(scene["item_url"])
            scene_dir = OUT / scene["kind"] / scene["date"]
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / "source_item.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
            for band in scene["bands"]:
                source_url = item["assets"][band]["href"]
                read_url = source_url
                if scene["kind"] == "s1_rtc":
                    read_url = get("https://planetarycomputer.microsoft.com/api/sas/v1/sign?" + urlencode({"href": source_url}))["href"]
                try:
                    with rasterio.open(read_url) as src:
                        if src.crs.to_epsg() != 32618:
                            raise ValueError("Unexpected CRS; this crop layout expects NYC UTM 18N")
                        for site in scene["sites"]:
                            xx, yy = transform("EPSG:4326", src.crs, [float(site["longitude"])], [float(site["latitude"])])
                            x, y = math.floor(xx[0]/20)*20, math.floor(yy[0]/20)*20
                            window = from_bounds(x-half, y-half, x+half, y+half, src.transform)
                            if not all(abs(v-round(v)) < 1e-5 for v in (window.col_off, window.row_off, window.width, window.height)):
                                raise ValueError("Window is not on native grid; explicit resampling would be required")
                            window = window.round_offsets().round_lengths()
                            values = src.read(1, window=window, boundless=True, fill_value=src.nodata)
                            site_dir = scene_dir / site["sensor_id"]
                            site_dir.mkdir(parents=True, exist_ok=True)
                            target = site_dir / (band + ".tif")
                            profile = {"driver": "GTiff", "width": values.shape[1], "height": values.shape[0],
                                       "count": 1, "dtype": str(values.dtype), "crs": src.crs,
                                       "transform": src.window_transform(window), "nodata": src.nodata,
                                       "compress": "deflate"}
                            with rasterio.open(target, "w", **profile) as dst:
                                dst.write(values, 1)
                                dst.set_band_description(1, band)
                                dst.update_tags(source_item=scene["item_id"], sensor_id=site["sensor_id"],
                                                use="inspection_only_no_pixel_labels", stored_values="unchanged_from_source")
                            invalid = ~np.isfinite(values)
                            if src.nodata is not None:
                                invalid |= values == src.nodata
                            records.append({"path": target.relative_to(OUT).as_posix(), "source_url": source_url,
                                            "item_id": scene["item_id"], "sensor_id": site["sensor_id"],
                                            "band": band, "pixel_size_m": abs(src.res[0]),
                                            "shape": list(values.shape), "nodata": src.nodata,
                                            "invalid_fraction": float(invalid.mean()),
                                            "raster_metadata": item["assets"][band].get("raster:bands", [])})
                except Exception as exc:
                    # Raster drivers can include SAS URLs in exceptions. Do not print them.
                    raise RuntimeError(f"Failed {scene['item_id']} band {band} ({type(exc).__name__}); retry or check provider access") from None
                print(scene["date"], band, "saved", len(scene["sites"]), "native-grid crops", flush=True)
    (OUT / "download_report.json").write_text(json.dumps({"files": records, "count": len(records),
        "purpose": manifest["purpose"], "note": "S1 RTC gamma0 and S2 L2A crops are inspection inputs, not drop-in paper reproduction inputs."}, indent=2), encoding="utf-8")
    print("Completed", len(records), "GeoTIFF crops", flush=True)


if __name__ == "__main__":
    main()
