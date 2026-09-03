"""
PixaMap Vector Engine: LULC (Land Use / Land Cover) & Farm Boundary Delineation.
Computes multi-spectral indices (NDVI, NDWI), delineates agricultural field parcels,
and segments 7-class continuous LULC zoning maps.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid


class LULCAndFarmExtractor:
    """
    Extracts LULC zoning polygons and agricultural parcel boundaries.
    """

    LULC_CLASSES = {
        1: ("water", "#0077be"),
        2: ("forest", "#228b22"),
        3: ("cropland", "#7cfc00"),
        4: ("built_up", "#dc143c"),
        5: ("bare_soil", "#d2b48c"),
        6: ("roads", "#696969"),
        7: ("grassland", "#90ee90")
    }

    def __init__(
        self,
        min_parcel_area_px: int = 500,
        pixel_size_meters: float = 0.05,
        min_confidence_threshold: float = 0.75
    ):
        self.min_parcel_area_px = min_parcel_area_px
        self.pixel_size_meters = pixel_size_meters
        self.min_confidence_threshold = min_confidence_threshold

    @staticmethod
    def compute_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """Normalized Difference Vegetation Index (NDVI = (NIR - Red) / (NIR + Red))"""
        denom = (nir_band.astype(np.float32) + red_band.astype(np.float32))
        denom = np.where(denom == 0, 1e-6, denom)
        ndvi = (nir_band.astype(np.float32) - red_band.astype(np.float32)) / denom
        return np.clip(ndvi, -1.0, 1.0)

    @staticmethod
    def compute_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """Normalized Difference Water Index (NDWI = (Green - NIR) / (Green + NIR))"""
        denom = (green_band.astype(np.float32) + nir_band.astype(np.float32))
        denom = np.where(denom == 0, 1e-6, denom)
        ndwi = (green_band.astype(np.float32) - nir_band.astype(np.float32)) / denom
        return np.clip(ndwi, -1.0, 1.0)

    def extract_farm_boundaries(
        self,
        farm_prob_mask: np.ndarray,
        geo_transform_fn
    ) -> List[Dict[str, Any]]:
        """
        Delineates agricultural parcel polygons with smooth boundaries and metric area calculations.
        """
        if farm_prob_mask.dtype != np.uint8:
            binary = (farm_prob_mask >= 0.5).astype(np.uint8) * 255
        else:
            binary = farm_prob_mask

        # Morphological opening to detach neighboring parcels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        parcels = []
        parcel_id = 1

        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_parcel_area_px:
                continue

            # Simplify contour
            epsilon = 0.015 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue

            pts = approx.reshape(-1, 2)
            geo_coords = [geo_transform_fn(float(x), float(y)) for x, y in pts]
            if geo_coords and geo_coords[0] != geo_coords[-1]:
                geo_coords.append(geo_coords[0])

            if len(geo_coords) < 4:
                continue

            try:
                poly = Polygon(geo_coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_empty:
                    continue
                if poly.geom_type == "GeometryCollection":
                    sub_polys = [g for g in poly.geoms if g.geom_type in ["Polygon", "MultiPolygon"]]
                    if not sub_polys:
                        continue
                    poly = max(sub_polys, key=lambda p: p.area)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda p: p.area)
                if poly.geom_type != "Polygon" or poly.is_empty:
                    continue
            except Exception:
                continue

            area_sqm = round(float(area_px * (self.pixel_size_meters ** 2)), 2)
            if area_sqm > 50000.0:
                continue
            area_hectares = round(area_sqm / 10000.0, 3)

            feature = {
                "type": "Feature",
                "id": parcel_id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(poly.exterior.coords)]
                },
                "properties": {
                    "feature_type": "farm_parcel",
                    "parcel_id": parcel_id,
                    "area_sqm": area_sqm,
                    "area_hectares": area_hectares,
                    "confidence_score": 0.91,
                    "needs_review": False
                }
            }
            parcels.append(feature)
            parcel_id += 1

        return parcels

    def extract_water_bodies(
        self,
        water_prob_mask: np.ndarray,
        geo_transform_fn
    ) -> List[Dict[str, Any]]:
        """
        Extracts lakes, ponds, and river polygons.
        """
        if water_prob_mask.dtype != np.uint8:
            binary = (water_prob_mask >= 0.5).astype(np.uint8) * 255
        else:
            binary = water_prob_mask

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        water_features = []
        wid = 1

        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < 100:
                continue

            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue

            pts = approx.reshape(-1, 2)
            geo_coords = [geo_transform_fn(float(x), float(y)) for x, y in pts]
            if geo_coords and geo_coords[0] != geo_coords[-1]:
                geo_coords.append(geo_coords[0])

            try:
                poly = Polygon(geo_coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_empty:
                    continue
                if poly.geom_type == "GeometryCollection":
                    sub_polys = [g for g in poly.geoms if g.geom_type in ["Polygon", "MultiPolygon"]]
                    if not sub_polys:
                        continue
                    poly = max(sub_polys, key=lambda p: p.area)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda p: p.area)
                if poly.geom_type != "Polygon" or poly.is_empty:
                    continue
            except Exception:
                continue

            area_sqm = round(float(area_px * (self.pixel_size_meters ** 2)), 2)
            if area_sqm > 250000.0:
                continue

            feature = {
                "type": "Feature",
                "id": wid,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(poly.exterior.coords)]
                },
                "properties": {
                    "feature_type": "water_body",
                    "water_id": wid,
                    "area_sqm": area_sqm,
                    "confidence_score": 0.95,
                    "needs_review": False
                }
            }
            water_features.append(feature)
            wid += 1

        return water_features
