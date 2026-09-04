"""
PixaMap Core: Existing GIS Reference Layer Fetcher (OpenStreetMap Overpass API).
Fetches verified ground-truth building footprints, road networks, and water layers
for reference and accuracy validation against AI-extracted features.
"""

from typing import Dict, List, Any, Optional
import json
import urllib.request
from shapely.geometry import Polygon, LineString


class OSMReferenceFetcher:
    """
    Queries OpenStreetMap Overpass API for ground-truth vector layers within an AOI.
    """

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    @classmethod
    def fetch_reference_layers(
        cls,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        timeout: int = 12
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Queries OSM for buildings, roads, and water within bounding box.
        """
        # Overpass query format: (south, west, north, east)
        query = f"""[out:json][timeout:{timeout}];
(
  way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["building:part"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["water"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["water"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["landuse"="reservoir"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["landuse"="basin"]({min_lat},{min_lon},{max_lat},{max_lon});
);
out geom;"""

        req = urllib.request.Request(
            cls.OVERPASS_URL,
            data=query.encode("utf-8"),
            headers={"User-Agent": "PixaMap/1.0 (GeoAI Engine)"}
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return {"buildings": [], "roads": [], "water": []}

        buildings = []
        roads = []
        water = []

        elements = data.get("elements", [])
        bid = 1
        rid = 1
        wid = 1

        for el in elements:
            geometry = el.get("geometry", [])
            if len(geometry) < 2:
                continue

            tags = el.get("tags", {})
            coords = [[pt["lon"], pt["lat"]] for pt in geometry]

            # 1. Buildings
            if "building" in tags or "building:part" in tags:
                if len(coords) >= 3:
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    # Filter out oversized campus / compound / zone boundaries (e.g. universities, military zones)
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    dx_m = (max(lons) - min(lons)) * 105000.0
                    dy_m = (max(lats) - min(lats)) * 111320.0
                    approx_area_sqm = dx_m * dy_m
                    if approx_area_sqm > 3000.0 or max(dx_m, dy_m) > 110.0:
                        continue  # Skip giant compound / campus boundaries

                    try:
                        import math
                        from shapely.geometry import Polygon
                        center_lat = sum(c[1] for c in coords) / len(coords)
                        cos_lat = math.cos(math.radians(center_lat)) if abs(center_lat) <= 90.0 else 1.0
                        metric_coords = [(c[0] * 111320.0 * cos_lat, c[1] * 111320.0) for c in coords]
                        poly = Polygon(metric_coords)
                        area_sqm = round(float(poly.area), 2)
                        perimeter_m = round(float(poly.length), 2)
                    except Exception:
                        area_sqm = 120.0
                        perimeter_m = 44.0

                    # Parse verified architectural height or building levels
                    h_val = None
                    if "height" in tags:
                        try:
                            raw_h = tags["height"].replace("m", "").replace("meters", "").strip()
                            h_val = round(float(raw_h), 1)
                        except Exception:
                            pass
                    if h_val is None:
                        levels = tags.get("building:levels") or tags.get("levels")
                        if levels:
                            try:
                                h_val = round(float(levels) * 3.3, 1)
                            except Exception:
                                pass
                    buildings.append({
                        "type": "Feature",
                        "id": bid,
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords]
                        },
                        "properties": {
                            "feature_type": "building",
                            "building_id": bid,
                            "source": "OpenStreetMap / Municipal Ground Truth",
                            "name": tags.get("name", "Building"),
                            "building_type": tags.get("building", tags.get("building:part", "yes")),
                            "area_sqm": area_sqm,
                            "perimeter_m": perimeter_m,
                            "orientation_deg": 0.0,
                            "height_max": h_val,
                            "height_min": round(h_val - 3.0, 1) if h_val else None,
                            "height_mean": h_val,
                            "roof_profile": tags.get("roof:shape", "flat"),
                            "building:levels": tags.get("building:levels") or tags.get("levels"),
                            "confidence_score": 0.98,
                            "needs_review": False
                        }
                    })
                    bid += 1

            # 2. Roads
            elif "highway" in tags:
                roads.append({
                    "type": "Feature",
                    "id": rid,
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    },
                    "properties": {
                        "feature_type": "reference_road",
                        "road_id": rid,
                        "source": "OpenStreetMap Ground Truth",
                        "highway_type": tags.get("highway", "road"),
                        "name": tags.get("name", "Unnamed Road"),
                        "confidence_score": 1.0,
                        "needs_review": False
                    }
                })
                rid += 1

            # 3. Water
            elif "waterway" in tags or tags.get("natural") == "water" or "water" in tags or tags.get("landuse") in ["reservoir", "basin"]:
                if len(coords) >= 3 and coords[0] == coords[-1]:
                    geom_type = "Polygon"
                    geom_coords = [coords]
                else:
                    geom_type = "LineString"
                    geom_coords = coords

                water.append({
                    "type": "Feature",
                    "id": wid,
                    "geometry": {
                        "type": geom_type,
                        "coordinates": geom_coords
                    },
                    "properties": {
                        "feature_type": "reference_water",
                        "water_id": wid,
                        "source": "OpenStreetMap Ground Truth",
                        "name": tags.get("name", "Water Body"),
                        "confidence_score": 1.0,
                        "needs_review": False
                    }
                })
                wid += 1

        return {
            "buildings": buildings,
            "roads": roads,
            "water": water
        }
