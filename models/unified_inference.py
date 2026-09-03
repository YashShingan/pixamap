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

        # Spectral Indices on real RGB
        total_rgb = r + g + b + 1e-6
        norm_r = r / total_rgb
        norm_g = g / total_rgb
        norm_b = b / total_rgb

        # Excess Green Index for vegetation / trees
        green_excess = 2.0 * norm_g - norm_r - norm_b
        brightness = (r + g + b) / 3.0
        color_dev = np.std(rgb_data[:3], axis=0)

        # Feature 1: Trees, Jungles & Green Canopy
        # True chlorophyll reflectance: Green is higher than Red AND Blue
        is_green_vegetation = (g > r) & (g > b)
        tree_prob = np.where(is_green_vegetation & (green_excess > 0.01), np.clip(green_excess * 4.5, 0.4, 1.0), 0.0)
        if ndsm is not None:
            tree_prob = np.where(ndsm >= 1.8, tree_prob, tree_prob * 0.15)

        # Feature 2: Water Bodies (Inland blue lakes/ponds AND turbid coastal creeks like Thane Creek)
        # 1. Texture smoothness: Water has near-zero local gradient variance
        gray = cv2.cvtColor(
            np.transpose(rgb_data[:3], (1, 2, 0)).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        ) if rgb_data.shape[0] >= 3 else rgb_data[0].astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        local_texture = cv2.GaussianBlur(np.abs(laplacian), (15, 15), 0)

        # Clear/blue water
        is_blue_water = (b > (g - 5)) & (b > (r + 2)) & (green_excess < 0.02)
        # Turbid/creek water (Thane Creek, silt, mudflats)
        is_turbid_water = (local_texture < 4.0) & (morph_grad < 10) & (~is_green_vegetation) & (brightness > 30) & (brightness < 170) & (color_dev < 18)
        # Deep open water
        is_deep_water = (brightness < 36) & (~is_green_vegetation)

        water_prob = np.where(is_blue_water | is_turbid_water | is_deep_water, 0.95, 0.0)
        # Prevent buildings, roads, and high-texture terrain from being water
        water_prob = np.where((morph_grad > 16) | is_green_vegetation, 0.0, water_prob)

        # Non-vegetation, non-water ground mask
        non_veg_mask = (tree_prob < 0.3) & (water_prob < 0.2)

        # Feature 3: Buildings (Village Houses & Structures)
        # 1. Bright Tin/Metal Roofs: High brightness compared to surrounding soil
        is_tin_roof = (brightness > 130) & non_veg_mask
        
        # 2. Traditional Clay/Terracotta Roofs: Red-dominant (R > G and R > B) with distinct hue
        is_clay_roof = (r > (g + 10)) & (r > (b + 14)) & (brightness > 60) & (brightness < 170) & (tree_prob < 0.2)

        # 3. Structural Gradient Edges
        gray = cv2.cvtColor(
            np.transpose(rgb_data[:3], (1, 2, 0)).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        ) if rgb_data.shape[0] >= 3 else rgb_data[0].astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        roof_edges = (morph_grad > 14) & non_veg_mask & ((is_tin_roof | is_clay_roof) | (brightness > 95))

        building_raw = (is_tin_roof | is_clay_roof | roof_edges).astype(np.uint8)
        building_clean = cv2.morphologyEx(building_raw, cv2.MORPH_CLOSE, kernel)
        building_prob = cv2.GaussianBlur(building_clean.astype(np.float32), (5, 5), 0)
        building_prob = np.clip(building_prob * 1.5, 0.0, 0.95)

        if ndsm is not None:
            building_signal = ((ndsm >= 2.5) & (tree_prob < 0.3)).astype(np.float32)
            building_prob = np.maximum(building_prob, building_signal * 0.94)

        # Feature 4: Roads (Continuous neutral asphalt corridors)
        color_dev = np.std(rgb_data[:3], axis=0)  # Neutral hue (asphalt has low saturation)
        is_asphalt = (color_dev < 20) & (brightness > 40) & (brightness < 160) & non_veg_mask & (~is_clay_roof)
        
        # Multidirectional morphology for linear highway corridor
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 2))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 11))
        d1_kernel = np.eye(9, dtype=np.uint8)
        d2_kernel = np.fliplr(d1_kernel)

        h_roads = cv2.morphologyEx(is_asphalt.astype(np.uint8), cv2.MORPH_OPEN, h_kernel)
        v_roads = cv2.morphologyEx(is_asphalt.astype(np.uint8), cv2.MORPH_OPEN, v_kernel)
        d1_roads = cv2.morphologyEx(is_asphalt.astype(np.uint8), cv2.MORPH_OPEN, d1_kernel)
        d2_roads = cv2.morphologyEx(is_asphalt.astype(np.uint8), cv2.MORPH_OPEN, d2_kernel)

        road_combined = (h_roads | v_roads | d1_roads | d2_roads).astype(np.float32)
        road_prob = cv2.GaussianBlur(road_combined, (5, 5), 0)
        road_prob = np.clip(road_prob * 1.6, 0.0, 0.95)

        if ndsm is not None:
            road_prob = np.where(ndsm < 1.2, road_prob, 0.0)

        # Feature 5: Farms (Agricultural field plots - low ground vegetation distinct from forest)
        farm_prob = np.where((green_excess > 0.04) & (tree_prob < 0.35) & (building_prob < 0.2), 0.85, 0.0)
        if ndsm is not None:
            farm_prob = np.where(ndsm < 1.0, farm_prob, 0.0)

        return {
            "buildings": building_prob.astype(np.float32),
            "roads": road_prob.astype(np.float32),
            "trees": tree_prob.astype(np.float32),
            "farms": farm_prob.astype(np.float32),
            "water": water_prob.astype(np.float32)
        }
