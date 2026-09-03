import urllib.request
import json

query = """[out:json][timeout:15];
(
  way["building"](19.095,72.920,19.110,72.935);
  way["highway"](19.095,72.920,19.110,72.935);
);
out geom;"""

url = "https://overpass-api.de/api/interpreter"
req = urllib.request.Request(url, data=query.encode("utf-8"), headers={"User-Agent": "PixaMap/1.0"})
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("OSM elements returned:", len(data.get("elements", [])))
        if data.get("elements"):
            print("First element tags:", data["elements"][0].get("tags"))
except Exception as e:
    print("OSM Error:", e)
