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
  way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
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
            if "building" in tags:
                if len(coords) >= 3:
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    buildings.append({
                        "type": "Feature",
                        "id": bid,
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords]
                        },
                        "properties": {
                            "feature_type": "reference_building",
                            "building_id": bid,
                            "source": "OpenStreetMap / Municipal Ground Truth",
                            "name": tags.get("name", "Building"),
                            "building_type": tags.get("building", "yes"),
                            "confidence_score": 1.0,
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
            elif "waterway" in tags or tags.get("natural") == "water":
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
