"""
PixaMap Vector Engine: CAD-Grade Building Footprint Regularization.
Transforms noisy, wobbly pixel contours into clean, orthogonal, 90-degree
architectural polygons ready for municipal GIS and urban planning.
"""

from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid


class BuildingRegularizer:
    """
    Deterministic geometric regularizer that enforces right angles,
    reduces redundant vertices, and validates topology.
    """

    def __init__(
        self,
        min_area_pixels: int = 40,
        rdp_epsilon_ratio: float = 0.02,
        angle_snap_tolerance_deg: float = 15.0,
        min_confidence_threshold: float = 0.75,
        pixel_size_meters: float = 0.05
    ):
        self.min_area_pixels = min_area_pixels
        self.rdp_epsilon_ratio = rdp_epsilon_ratio
        self.angle_snap_tolerance_deg = angle_snap_tolerance_deg
        self.min_confidence_threshold = min_confidence_threshold
        self.pixel_size_meters = pixel_size_meters

    def regularize_mask(
        self,
        building_prob_mask: np.ndarray,
        geo_transform_fn,  # callable: (px, py) -> (lon, lat)
        pixel_size_meters: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Takes a binary or probability mask of buildings, extracts contours,
        applies 90-degree orthogonalization, and returns geo-referenced GeoJSON-ready features.
        """
        px_m = pixel_size_meters if pixel_size_meters is not None else self.pixel_size_meters
        # Threshold probability mask
        if building_prob_mask.dtype != np.uint8:
            binary_mask = (building_prob_mask >= 0.5).astype(np.uint8) * 255
        else:
            binary_mask = building_prob_mask

        # Morphological closing to seal minor roof holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Find external contours
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        regularized_features = []
        feature_id = 1

        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_area_pixels:
                continue

            # Original vertex count
            raw_vertex_count = len(cnt)

            # Step 1: Douglas-Peucker Polygon Simplification
            perimeter = cv2.arcLength(cnt, True)
            epsilon = max(1.5, perimeter * self.rdp_epsilon_ratio)
            approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx_cnt) < 3:
                continue

            # Step 2: Extract Dominant Angle from Minimum Area Bounding Box
            rect = cv2.minAreaRect(cnt)
            dominant_angle = rect[2]  # in degrees [-90, 0)
            if dominant_angle < -45:
                dominant_angle += 90.0

            # Step 3: Orthogonal Edge Snapping
            pts = approx_cnt.reshape(-1, 2)
            pts_regularized = self._snap_orthogonal(pts, dominant_angle)

            # Step 4: Calculate Confidence Score
            mask_roi = np.zeros_like(binary_mask, dtype=np.uint8)
            cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
            mean_prob = float(np.mean(building_prob_mask[mask_roi == 255])) if np.any(mask_roi == 255) else 0.85
            
            # Step 5: Convert to Geographic Coordinates
            geo_coords = []
            for pt in pts_regularized:
                gx, gy = geo_transform_fn(float(pt[0]), float(pt[1]))
                geo_coords.append((gx, gy))
            
            # Close polygon ring
            if geo_coords and geo_coords[0] != geo_coords[-1]:
                geo_coords.append(geo_coords[0])

            if len(geo_coords) < 4:
                continue

            # Step 6: Validate & Repair Topology using Shapely
            try:
                poly = Polygon(geo_coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_empty:
                    continue
                if isinstance(poly, MultiPolygon):
                    poly = max(poly.geoms, key=lambda p: p.area)
            except Exception:
                continue

            # Calculate metric attributes
            area_sqm = round(float(area_px * (px_m ** 2)), 2)
            perimeter_m = round(float(perimeter * px_m), 2)
            vertex_reduction_pct = round((1.0 - (len(pts_regularized) / max(raw_vertex_count, 1))) * 100.0, 1)

            feature = {
                "type": "Feature",
                "id": feature_id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(poly.exterior.coords)]
                },
                "properties": {
                    "feature_type": "building",
                    "building_id": feature_id,
                    "area_sqm": area_sqm,
                    "perimeter_m": perimeter_m,
                    "orientation_deg": round(float(dominant_angle), 1),
                    "confidence_score": round(float(mean_prob), 3),
                    "needs_review": bool(mean_prob < self.min_confidence_threshold),
                    "vertex_reduction_pct": vertex_reduction_pct
                }
            }
            regularized_features.append(feature)
            feature_id += 1

        return regularized_features

    def _snap_orthogonal(self, pts: np.ndarray, dominant_angle_deg: float) -> np.ndarray:
        """
        Rotates polygon so dominant axis aligns with X-axis, snaps segments
        to horizontal/vertical if within tolerance, and rotates back.
        """
        theta_rad = math.radians(-dominant_angle_deg)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)

        # Rotate points to local Cartesian alignment
        pts_rotated = []
        for x, y in pts:
            rx = x * cos_t - y * sin_t
            ry = x * sin_t + y * cos_t
            pts_rotated.append([rx, ry])
        pts_rotated = np.array(pts_rotated)

        # Snap near-orthogonal segments
        snapped = np.copy(pts_rotated)
        n = len(snapped)
        tol_deg = self.angle_snap_tolerance_deg

        for i in range(n):
            next_i = (i + 1) % n
            dx = snapped[next_i, 0] - snapped[i, 0]
            dy = snapped[next_i, 1] - snapped[i, 1]
            seg_angle = math.degrees(math.atan2(dy, dx)) % 180.0

            # Snap to horizontal (0 deg or 180 deg)
            if seg_angle < tol_deg or seg_angle > (180.0 - tol_deg):
                avg_y = (snapped[i, 1] + snapped[next_i, 1]) / 2.0
                snapped[i, 1] = avg_y
                snapped[next_i, 1] = avg_y
            # Snap to vertical (90 deg)
            elif abs(seg_angle - 90.0) < tol_deg:
                avg_x = (snapped[i, 0] + snapped[next_i, 0]) / 2.0
                snapped[i, 0] = avg_x
                snapped[next_i, 0] = avg_x

        # Rotate back to original pixel coordinate system
        cos_inv = math.cos(-theta_rad)
        sin_inv = math.sin(-theta_rad)
        pts_original = []
        for rx, ry in snapped:
            ox = rx * cos_inv - ry * sin_inv
            oy = rx * sin_inv + ry * cos_inv
            pts_original.append([round(ox, 1), round(oy, 1)])

        return np.array(pts_original)
