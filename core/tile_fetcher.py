"""
PixaMap Core: Satellite Imagery Tile Fetcher.
Downloads and stitches real-world high-resolution satellite imagery tiles (Esri World Imagery)
for any user-selected geographic Area of Interest (AOI).
"""

from typing import Tuple, List, Optional
import io
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

from core.geo_reader import GeoRaster


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon coordinates to slippy map tile x, y."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Convert slippy map tile x, y to upper-left lat/lon."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


class SatelliteTileFetcher:
    """
    Fetches and mosaics satellite tiles for an Area of Interest (AOI).
    """

    ESRI_TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    @classmethod
    def fetch_single_tile(cls, x: int, y: int, z: int) -> Optional[Image.Image]:
        url = cls.ESRI_TILE_URL.format(z=z, y=y, x=x)
        req = urllib.request.Request(url, headers={"User-Agent": "PixaMap/1.0 (GeoAI Engine)"})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = resp.read()
                return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            # Return gray tile on failure
            return Image.new("RGB", (256, 256), color=(128, 128, 128))

    @classmethod
    def fetch_aoi_raster(
        cls,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        zoom: int = 17
    ) -> GeoRaster:
        """
        Downloads all tiles covering the bounding box at the specified zoom level,
        stitches them into a single mosaic, and returns a georeferenced GeoRaster.
        """
        # Calculate tile coordinate ranges
        x_min, y_min = deg2num(max_lat, min_lon, zoom)
        x_max, y_max = deg2num(min_lat, max_lon, zoom)

        # Clamp max tiles to prevent overloading
        tile_width = max(1, x_max - x_min + 1)
        tile_height = max(1, y_max - y_min + 1)

        # If area is too large for zoom 17, adjust zoom
        if tile_width * tile_height > 25:
            zoom = max(15, zoom - 1)
            x_min, y_min = deg2num(max_lat, min_lon, zoom)
            x_max, y_max = deg2num(min_lat, max_lon, zoom)
            tile_width = max(1, x_max - x_min + 1)
            tile_height = max(1, y_max - y_min + 1)

        # Parallel tile download
        tile_coords = []
        for r, y in enumerate(range(y_min, y_max + 1)):
            for c, x in enumerate(range(x_min, x_max + 1)):
                tile_coords.append((r, c, x, y))

        mosaic_img = Image.new("RGB", (tile_width * 256, tile_height * 256))

        def download_and_paste(item):
            r, c, x, y = item
            tile = cls.fetch_single_tile(x, y, zoom)
            if tile:
                mosaic_img.paste(tile, (c * 256, r * 256))

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(download_and_paste, tile_coords))

        # Compute accurate geographic bounding box for the stitched mosaic
        ul_lat, ul_lon = num2deg(x_min, y_min, zoom)
        lr_lat, lr_lon = num2deg(x_max + 1, y_max + 1, zoom)

        # Convert PIL to NumPy (bands, height, width)
        arr = np.array(mosaic_img)
        data = np.transpose(arr, (2, 0, 1))

        geo_bounds = (ul_lon, lr_lat, lr_lon, ul_lat)
        raster = GeoRaster(data=data, bounds=geo_bounds, crs="EPSG:4326")
        raster.is_satellite_aoi = True
        return raster
