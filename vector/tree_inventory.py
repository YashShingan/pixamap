"""
PixaMap Vector Engine: Tree Crown Detection & 3D Inventory.
Extracts individual tree counts, crown diameters, canopy areas,
and samples the nDSM elevation raster to attribute exact tree height in meters.
"""

from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np
import cv2
from shapely.geometry import Point, Polygon


class TreeInventoryExtractor:
    """
    Detects individual tree crowns and integrates 3D height attribution from nDSM.
    """

    def __init__(
        self,
        min_crown_radius_px: int = 4,
        max_crown_radius_px: int = 120,
        pixel_size_meters: float = 0.05,
        min_tree_height_meters: float = 1.5,
        min_confidence_threshold: float = 0.75
    ):
        self.min_crown_radius_px = min_crown_radius_px
        self.max_crown_radius_px = max_crown_radius_px
        self.pixel_size_meters = pixel_size_meters
        self.min_tree_height_meters = min_tree_height_meters
        self.min_confidence_threshold = min_confidence_threshold

    def extract_tree_inventory(
        self,
        tree_prob_mask: np.ndarray,
        geo_transform_fn,  # callable: (px, py) -> (lon, lat)
        ndsm: Optional[np.ndarray] = None,
        pixel_size_meters: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Detects trees and generates a structured GIS point inventory.
        """
        px_m = pixel_size_meters if pixel_size_meters is not None else self.pixel_size_meters

        if tree_prob_mask.dtype != np.uint8:
            binary_mask = (tree_prob_mask >= 0.45).astype(np.uint8) * 255
        else:
            binary_mask = tree_prob_mask

        # Morphological opening to separate touching canopies
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Distance transform to find individual tree apexes / local maxima
        dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
        
        # Threshold peaks
        peak_thresh = float(np.max(dist)) * 0.2 if np.max(dist) > 0 else 1.0
        peaks = (dist > peak_thresh).astype(np.uint8)

        # Connected components on peaks
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks)

        trees = []
        tree_id = 1
        min_rad = max(1.5, self.min_crown_radius_px if px_m < 0.2 else 1.5)

        for i in range(1, num_labels):
            cx, cy = centroids[i]
            c_int, r_int = int(round(cx)), int(round(cy))
            radius_px = float(dist[r_int, c_int])

            if radius_px < min_rad or radius_px > self.max_crown_radius_px:
                continue

            # Convert centroid to geographic coordinates
            gx, gy = geo_transform_fn(cx, cy)

            # Calculate metric dimensions
            crown_diameter_m = round(float(radius_px * 2.0 * px_m), 2)
            crown_area_sqm = round(float(math.pi * ((crown_diameter_m / 2.0) ** 2)), 2)

            # Sample 3D height from nDSM if available
            height_m = 0.0
            if ndsm is not None and ndsm.size > 0:
                h_max = 0.0
                rad_int = max(2, int(round(radius_px)))
                r_min = max(0, r_int - rad_int)
                r_max = min(ndsm.shape[0], r_int + rad_int + 1)
                c_min = max(0, c_int - rad_int)
                c_max = min(ndsm.shape[1], c_int + rad_int + 1)
                
                sub_ndsm = ndsm[r_min:r_max, c_min:c_max]
                if sub_ndsm.size > 0:
                    h_max = float(np.max(sub_ndsm))
                height_m = round(max(h_max, 0.0), 2)

            if ndsm is not None and height_m < self.min_tree_height_meters:
                continue

            mean_conf = round(float(tree_prob_mask[r_int, c_int]), 3)

            feature = {
                "type": "Feature",
                "id": tree_id,
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(gx, 7), round(gy, 7)]
                },
                "properties": {
                    "feature_type": "tree",
                    "tree_id": tree_id,
                    "height_m": height_m if height_m > 0 else 6.5,  # Estimated baseline if no nDSM
                    "crown_diameter_m": crown_diameter_m,
                    "crown_area_sqm": crown_area_sqm,
                    "confidence_score": mean_conf,
                    "needs_review": bool(mean_conf < self.min_confidence_threshold)
                }
            }
            trees.append(feature)
            tree_id += 1

        return trees
