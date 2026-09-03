"""
PixaMap Unified Inference Engine.
Orchestrates multi-modal feature extraction across Buildings, Roads, Trees,
Farm Parcels, and Water Bodies with full geometric regularization.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2

from core.geo_reader import GeoRaster, read_geotiff
from core.elevation import ElevationProcessor
from core.tiler import SlidingWindowTiler
from vector.building_regularizer import BuildingRegularizer
from vector.road_centerline import RoadCenterlineExtractor
from vector.tree_inventory import TreeInventoryExtractor
from vector.lulc_farm import LULCAndFarmExtractor


class UnifiedGeoAIEngine:
    """
    Main pipeline engine orchestrating all 5 geospatial feature extraction modules.
    """

    def __init__(
        self,
        pixel_size_meters: float = 0.05,
        tile_size: int = 512,
        tile_overlap: int = 64
    ):
        self.pixel_size_meters = pixel_size_meters
        self.tiler = SlidingWindowTiler(tile_size=tile_size, overlap=tile_overlap)
        self.building_reg = BuildingRegularizer(pixel_size_meters=pixel_size_meters)
        self.road_extractor = RoadCenterlineExtractor(pixel_size_meters=pixel_size_meters)
        self.tree_extractor = TreeInventoryExtractor(pixel_size_meters=pixel_size_meters)
        self.lulc_farm_extractor = LULCAndFarmExtractor(pixel_size_meters=pixel_size_meters)

    def process_raster(
        self,
        ortho_raster: GeoRaster,
        dsm_raster: Optional[GeoRaster] = None,
        dtm_raster: Optional[GeoRaster] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-modal extraction on the input GeoRaster.
        """
        # Step 1: Compute nDSM if elevation rasters are provided
        ndsm = None
        if dsm_raster is not None and dtm_raster is not None:
            ndsm = ElevationProcessor.compute_ndsm(dsm_raster.data, dtm_raster.data)

        # Step 2: Extract or simulate probability feature maps
        # Shape: (h, w) for each layer
        prob_maps = self._generate_feature_probability_maps(ortho_raster.data, ndsm)

        # Coordinate transformation lambda
        geo_transform = ortho_raster.pixel_to_geo

        # Step 3: Run Deterministic Geometric Regularizers
        # 1. Buildings (CAD-grade 90-degree orthogonal polygons)
        buildings = self.building_reg.regularize_mask(
            prob_maps["buildings"],
            geo_transform,
            pixel_size_meters=self.pixel_size_meters
        )

        # 2. Roads (Topological centerline graph LineStrings)
        roads = self.road_extractor.extract_centerlines(
            prob_maps["roads"],
            geo_transform
        )

        # 3. Trees (Count, canopy diameter, and 3D height from nDSM)
        trees = self.tree_extractor.extract_tree_inventory(
            prob_maps["trees"],
            geo_transform,
            ndsm=ndsm
        )

        # 4. Farm Boundaries (Cadastral agricultural parcels)
        farms = self.lulc_farm_extractor.extract_farm_boundaries(
            prob_maps["farms"],
            geo_transform
        )

        # 5. Water Bodies
        water = self.lulc_farm_extractor.extract_water_bodies(
            prob_maps["water"],
            geo_transform
        )

        # Summary Metrics
        total_road_km = round(sum(f["properties"]["length_m"] for f in roads) / 1000.0, 3)
        avg_building_conf = round(
            float(np.mean([f["properties"]["confidence_score"] for f in buildings])) if buildings else 0.0, 3
        )

        return {
            "summary": {
                "building_count": len(buildings),
                "road_segment_count": len(roads),
                "total_road_km": total_road_km,
                "tree_count": len(trees),
                "farm_parcel_count": len(farms),
                "water_body_count": len(water),
                "avg_building_confidence": avg_building_conf,
                "bounds": ortho_raster.bounds,
                "crs": ortho_raster.crs
            },
            "layers": {
                "buildings": buildings,
                "roads": roads,
                "trees": trees,
                "farms": farms,
                "water": water
            }
        }

    def _generate_feature_probability_maps(
        self,
        rgb_data: np.ndarray,
        ndsm: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generates probability masks for each class using spectral indices,
        geometric feature filters, and nDSM height signals.
        """
        # Ensure 3-band RGB
        if rgb_data.shape[0] >= 3:
            r = rgb_data[0].astype(np.float32)
            g = rgb_data[1].astype(np.float32)
            b = rgb_data[2].astype(np.float32)
        else:
            r = g = b = rgb_data[0].astype(np.float32)

        h, w = r.shape

        # Spectral Indices
        green_excess = (2.0 * g - r - b) / (r + g + b + 1e-6)
        brightness = (r + g + b) / 3.0

        # Feature 1: Trees (Green + Elevated above ground via nDSM if available)
        tree_prob = np.clip((green_excess - 0.05) * 2.5, 0.0, 1.0)
        if ndsm is not None:
            # Trees must have height >= 2.0m
            tree_prob = np.where(ndsm >= 2.0, tree_prob, tree_prob * 0.2)

        # Feature 2: Buildings (Geometric brightness + elevated nDSM >= 3.0m)
        gray = cv2.cvtColor(
            np.transpose(rgb_data[:3], (1, 2, 0)).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        ) if rgb_data.shape[0] >= 3 else rgb_data[0].astype(np.uint8)

        # Detect roof edge features
        edges = cv2.Canny(gray, 50, 150)
        building_prob = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (15, 15), 0)
        building_prob = np.clip(building_prob * 3.0, 0.0, 0.95)

        if ndsm is not None:
            # Elevated structures with height between 3.0m and 60m are buildings
            building_signal = ((ndsm >= 3.0) & (tree_prob < 0.3)).astype(np.float32)
            building_prob = np.maximum(building_prob, building_signal * 0.92)

        # Feature 3: Roads (Low greenness, medium brightness, elongated structures)
        road_prob = np.clip(1.0 - (green_excess * 2.0) - (tree_prob * 1.5), 0.0, 0.9)
        if ndsm is not None:
            # Roads must be at ground level (nDSM < 1.0m)
            road_prob = np.where(ndsm < 1.0, road_prob, 0.0)

        # Feature 4: Water (Low brightness or blue-dominated absorption)
        water_prob = np.clip((b - r) / (b + r + 1e-6), 0.0, 1.0)
        water_prob = np.where((b > r) & (b > g) & (brightness < 120), 0.95, 0.0)

        # Feature 5: Farms (Vegetation at ground level)
        farm_prob = np.clip(green_excess * 1.8, 0.0, 1.0)
        if ndsm is not None:
            farm_prob = np.where(ndsm < 1.5, farm_prob, 0.0)

        return {
            "buildings": building_prob.astype(np.float32),
            "roads": road_prob.astype(np.float32),
            "trees": tree_prob.astype(np.float32),
            "farms": farm_prob.astype(np.float32),
            "water": water_prob.astype(np.float32)
        }
