"""Download the two matched NYC L1C acquisitions from the public Google mirror.

Keep four-band 512px DN context chips and the existing central 128px SCL layer.
Product XML supplies the quantification value and per-band radiometric offset.
No accounts, paid requests, model training or full-scene downloads are needed.
"""
import json
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/nyc/l1c_chips"
BUCKET = "https://storage.googleapis.com/gcp-public-data-sentinel-2/"
BANDS = ["B4", "B3", "B2", "B8"]
SAFE_BANDS = ["B04", "B03", "B02", "B08"]
BAND_IDS = [3, 2, 1, 7]  # Sentinel-2 product XML uses zero-based spectral IDs.


def fetch_json(url):
    with urlopen(url, timeout=45) as response:
        return json.load(response)


def main():
    config = json.loads((ROOT / "configs/datasets/nyc_candidates.json").read_text(encoding="utf-8"))
    records = []
    for scene in (s for s in config["scenes"] if s["kind"] == "s2_l2a"):
        date = scene["date"]
        folder = OUT / date
        folder.mkdir(parents=True, exist_ok=True)
        item_id = scene["item_id"].replace("_L2A", "_L1C")
        item_url = f"https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c/items/{item_id}"
        item = fetch_json(item_url)
        product = item["properties"]["s2:product_uri"]
        prefix = "tiles/18/T/WL/" + product + "/"
        objects = fetch_json("https://storage.googleapis.com/storage/v1/b/gcp-public-data-sentinel-2/o?" +
                             urlencode({"prefix": prefix}))["items"]
        names = [o["name"] for o in objects]
        xml_url = BUCKET + next(n for n in names if n.endswith("/MTD_MSIL1C.xml"))
        with urlopen(xml_url, timeout=45) as response:
            xml = response.read()
        (folder / "MTD_MSIL1C.xml").write_bytes(xml)
        (folder / "source_item.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        elements = list(ET.fromstring(xml).iter())
        quantification = float(next(e.text for e in elements if e.tag.split("}")[-1] == "QUANTIFICATION_VALUE"))
        offsets_by_id = {int(e.attrib["band_id"]): float(e.text) for e in elements
                         if e.tag.split("}")[-1] == "RADIO_ADD_OFFSET"}
        offsets = [offsets_by_id[i] for i in BAND_IDS]
        urls = [BUCKET + next(n for n in names if "/IMG_DATA/" in n and n.endswith(f"_{b}.jp2"))
                for b in SAFE_BANDS]

        # Use the existing central crop's exact grid, adding 192 pixels on each side.
        targets = []
        for site in scene["sites"]:
            old = ROOT / "data/nyc/candidate_chips/s2_l2a" / date / site["sensor_id"]
            with rasterio.open(old / "red.tif") as src:
                tr = src.window_transform(Window(-192, -192, 512, 512))
                profile = dict(driver="GTiff", width=512, height=512, count=4, dtype="uint16",
                               crs=src.crs, transform=tr, nodata=0, compress="deflate")
            targets.append((site, old, profile))

        def read_band(url):
            chips = []
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT=60,
                              CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".jp2"):
                with rasterio.open(url) as src:
                    for site, old, profile in targets:
                        tr = profile["transform"]
                        bounds = rasterio.transform.array_bounds(512, 512, tr)
                        window = from_bounds(*bounds, transform=src.transform)
                        assert src.crs == profile["crs"] and src.res == (10, 10)
                        assert all(abs(v - round(v)) < 1e-5 for v in
                                   [window.col_off, window.row_off, window.width, window.height])
                        chips.append(src.read(1, window=window.round_offsets().round_lengths(),
                                              boundless=True, fill_value=0))
            print(date, url.rsplit("/", 1)[-1], "context windows:", len(chips), flush=True)
            return chips

        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(read_band, urls))
        for index, (site, old, profile) in enumerate(targets):
            site_dir = folder / site["sensor_id"]
            site_dir.mkdir(exist_ok=True)
            dn = np.stack([band[index] for band in values])
            path = site_dir / "s2_l1c_dn.tif"
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(dn)
                for i, band in enumerate(BANDS, 1):
                    dst.set_band_description(i, band)
                dst.update_tags(product=product, units="unmodified L1C DN",
                                reflectance_formula="(DN + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE")
            shutil.copyfile(old / "scl.tif", site_dir / "scl_central.tif")
            keep = ((dn != 0) & (dn != 65535)).all(axis=0)
            record = dict(date=date, site=site, item=item_id, product=product,
                          utc=item["properties"]["datetime"], bands=BANDS, quantification=quantification,
                          radio_add_offset=offsets, source_urls=urls,
                          dn_path=path.relative_to(ROOT).as_posix(),
                          scl_path=(site_dir / "scl_central.tif").relative_to(ROOT).as_posix(),
                          central_window=[192, 192, 128, 128],
                          valid_context_fraction=float(keep.mean()),
                          valid_central_fraction=float(keep[192:320, 192:320].mean()))
            records.append(record)
            print(date, site["sensor_id"], "saved", flush=True)
    (OUT / "chips.json").write_text(json.dumps(dict(records=records,
        note="Matched L1C, raw DN; use product offsets once. Model outputs are water, not flood labels."),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("Completed:", len(records), "four-band L1C contexts", flush=True)


if __name__ == "__main__":
    main()
