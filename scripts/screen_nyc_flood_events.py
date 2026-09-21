"""Screen public NYC FloodNet events against Sentinel acquisition metadata.

Standard library only (Windows may need tzdata for zoneinfo). Cached raw responses
and results remain under data/nyc_event_screen. This does not infer flood masks.
S1 covers the full date range; S2 covers selected busy days and named events only.
"""
import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "nyc_event_screen"
NY = ZoneInfo("America/New_York")
BBOX = "-74.26,40.49,-73.70,40.92"
START, END = "2021-08-01T00:00:00", "2026-09-17T00:00:00"
OFFICIAL_DAYS = ["2021-08-21", "2021-09-01", "2022-09-13", "2022-12-23",
                 "2023-09-29", "2024-01-12", "2024-01-13", "2024-04-02",
                 "2024-04-03", "2024-04-04", "2024-08-18", "2025-07-14",
                 "2025-07-31", "2025-10-30", "2026-05-20"]


def fetch(name, url, params=None):
    path = OUT / (name + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    if params:
        url += "?" + urlencode(params)
    with urlopen(url, timeout=45) as response:
        result = json.load(response)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                                "response": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def stamp(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def in_ring(x, y, ring):
    inside = False
    for a, b in zip(ring, ring[1:] + ring[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:
            inside = not inside
    return inside


def contains(x, y, geometry):
    if geometry["type"] not in ("Polygon", "MultiPolygon"):
        raise ValueError("Unexpected footprint geometry")
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    return any(in_ring(x, y, p[0]) and not any(in_ring(x, y, h) for h in p[1:]) for p in polygons)


def matches(when, geometry, events, sensors):
    result = []
    for event in events:
        if not stamp(event["flood_start_time"]) <= when <= stamp(event["flood_end_time"]):
            continue
        sensor = sensors.get(event["sensor_id"])
        if not sensor or not contains(float(sensor["longitude"]), float(sensor["latitude"]), geometry):
            continue
        result.append({**event, "longitude": sensor["longitude"], "latitude": sensor["latitude"],
                       "tidally_influenced_site": sensor.get("tidally_influenced")})
    return result


def optical_day(day):
    first = datetime.fromisoformat(day).replace(tzinfo=NY).astimezone(timezone.utc)
    last = first + timedelta(days=1)
    # Daytime acquisitions are unaffected by the occasional DST midnight offset change.
    result = fetch("s2/" + day + "_0", "https://earth-search.aws.element84.com/v1/search", {
        "collections": "sentinel-2-l2a", "bbox": BBOX,
        "datetime": first.isoformat() + "/" + last.isoformat(), "limit": 100})
    items = list(result["features"])
    page = 0
    while True:
        link = next((l for l in result.get("links", []) if l["rel"] == "next"), None)
        if not link:
            return items
        page += 1
        if page > 10:
            raise RuntimeError("Unexpected STAC pagination")
        result = fetch("s2/" + day + "_" + str(page), link["href"])
        items.extend(result["features"])


def main():
    events = fetch("events", "https://data.cityofnewyork.us/resource/aq7i-eu5q.json", {
        "$select": "sensor_name,sensor_id,flood_start_time,flood_end_time,max_depth_inches,duration_mins",
        "$where": f"flood_start_time >= '{START}' AND flood_start_time < '{END}'",
        "$order": "flood_start_time,sensor_id", "$limit": 50000})
    sensor_rows = fetch("sensors", "https://data.cityofnewyork.us/resource/kb2e-tjy3.json", {"$limit": 5000})
    sensors = {s["sensor_id"]: s for s in sensor_rows}
    sar = fetch("sar_all", "https://api.daac.asf.alaska.edu/services/search/param", {
        "platform": "Sentinel-1", "processingLevel": "GRD_HD", "bbox": BBOX,
        "start": START + "Z", "end": END + "Z", "output": "geojson", "maxResults": 10000})["features"]
    if len(events) >= 50000 or len(sensor_rows) >= 5000 or len(sar) >= 10000:
        raise RuntimeError("Query cap reached; paginate before drawing conclusions")
    if len(sensors) != len(sensor_rows) or any(e["sensor_id"] not in sensors for e in events):
        raise RuntimeError("Sensor join is ambiguous or incomplete")
    sar_rows = []
    for scene in sar:
        props = scene["properties"]
        t = stamp(props["startTime"])
        found = matches(t, scene["geometry"], events, sensors)
        if found:
            sar_rows.append({"scene": props["sceneName"], "utc": props["startTime"],
                             "local": t.astimezone(NY).isoformat(), "orbit": props["pathNumber"],
                             "n": len(found), "matches": found})
    sar_rows.sort(key=lambda r: (-r["n"], r["utc"]))
    day_sensors = defaultdict(set)
    for event in events:
        day_sensors[stamp(event["flood_start_time"]).astimezone(NY).date().isoformat()].add(event["sensor_id"])
    days = set(OFFICIAL_DAYS) | {d for d, ids in day_sensors.items() if len(ids) >= 10}
    days.update(r["local"][:10] for r in sar_rows)
    # Also inspect the following day for residual flooding/cloud-free observations.
    days |= {(datetime.fromisoformat(d) + timedelta(days=1)).date().isoformat() for d in list(days)}
    items, errors = {}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(optical_day, d): d for d in sorted(days)}
        for future in as_completed(futures):
            day = futures[future]
            try:
                items.update({f["id"]: f for f in future.result()})
            except Exception as exc:
                errors.append({"day": day, "error": str(exc)})
    optical_rows = []
    for scene in items.values():
        props = scene["properties"]
        t = stamp(props["datetime"])
        found = matches(t, scene["geometry"], events, sensors)
        if found:
            optical_rows.append({"item": scene["id"], "utc": props["datetime"],
                                 "local": t.astimezone(NY).isoformat(),
                                 "cloud_percent_whole_tile": props.get("eo:cloud_cover"),
                                 "n": len(found), "scl": scene["assets"].get("scl", {}).get("href"),
                                 "matches": found})
    optical_rows.sort(key=lambda r: (-r["n"], r["utc"]))
    result = {"events": len(events), "s1_scenes": len(sar), "s2_days_queried": sorted(days),
              "s2_item_records": len(items), "errors": errors, "sar_matches": sar_rows,
              "optical_matches": optical_rows}
    (OUT / "screen_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, rows in [("sar", sar_rows), ("optical", optical_rows)]:
        flat = [{k: v for k, v in r.items() if k != "matches"} for r in rows]
        if flat:
            with (OUT / (label + "_matches.csv")).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sar_matches", "optical_matches", "s2_days_queried")}, indent=2))
    print("SAR matched scenes:", len(sar_rows), "S2 queried days:", len(days), "S2 matched item records:", len(optical_rows))
    for row in optical_rows[:20]:
        print(row["item"], row["local"], "n", row["n"], "cloud", row["cloud_percent_whole_tile"])


if __name__ == "__main__":
    main()
