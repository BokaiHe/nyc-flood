"""Download selected Sen1Floods11 layers; skip files already present."""
import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "https://storage.googleapis.com"


def list_files(prefix):
    params = {"prefix": prefix, "maxResults": 1000, "fields": "nextPageToken,items(name)"}
    while True:
        with urlopen(BUCKET + "/storage/v1/b/sen1floods11/o?" + urlencode(params)) as response:
            page = json.load(response)
        for item in page.get("items", []):
            if item["name"].endswith((".tif", ".csv", ".geojson")):
                yield item["name"]
        if "nextPageToken" not in page:
            break
        params["pageToken"] = page["nextPageToken"]


def download(name, target):
    dest = target / name
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with urlopen(BUCKET + "/sen1floods11/" + quote(name, safe="/")) as response:
        with part.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    part.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=["hand", "all"], default="all")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/sen1floods11")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/datasets/sen1floods11.json").read_text())
    prefixes = config["hand"] + config["metadata"]
    if args.subset == "all":
        prefixes += config["weak"]
    names = [name for prefix in prefixes for name in list_files(prefix)]
    print(f"Selected {len(names)} files; existing files will be skipped.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(lambda name: download(name, args.data_root), names), 1):
            if i % 100 == 0 or i == len(names):
                print(f"{i}/{len(names)}", flush=True)


if __name__ == "__main__":
    main()
