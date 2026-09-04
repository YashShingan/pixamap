"""
PixaMap Core: Multi-Modal Elevation Engine.
Computes Normalized Digital Surface Model (nDSM = DSM - DTM) to isolate
pure vertical height of buildings and tree canopies above terrain.
"""

from typing import Optional, Tuple
import numpy as np


class ElevationProcessor:
    """
    Processes Digital Surface Models (DSM) and Digital Terrain Models (DTM)
    into normalized height rasters invariant to topographic terrain elevation.
    """

    @staticmethod
    def compute_ndsm(dsm: np.ndarray, dtm: np.ndarray) -> np.ndarray:
        """
        Compute nDSM = DSM - DTM.
        Clamps negative values and non-finite values (NaN / Inf) to 0.0.
        """
        # Ensure 2D arrays
        dsm_2d = dsm[0] if dsm.ndim == 3 else dsm
        dtm_2d = dtm[0] if dtm.ndim == 3 else dtm

        # Sanitize non-finite inputs
        dsm_clean = np.nan_to_num(dsm_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        dtm_clean = np.nan_to_num(dtm_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Compute relative vertical height above ground
        ndsm = dsm_clean - dtm_clean
        ndsm = np.nan_to_num(ndsm, nan=0.0, posinf=0.0, neginf=0.0)
        ndsm = np.maximum(ndsm, 0.0)
        return ndsm

    @staticmethod
    def sample_height_at_point(ndsm: np.ndarray, px: int, py: int, window_radius: int = 2) -> float:
        """
        Sample maximum relative height (in meters) within a small window around (px, py).
        Ideal for finding tree crown apex or roof ridge height.
        """
        h, w = ndsm.shape
        r_min = max(0, py - window_radius)
        r_max = min(h, py + window_radius + 1)
        c_min = max(0, px - window_radius)
        c_max = min(w, px + window_radius + 1)

        window = ndsm[r_min:r_max, c_min:c_max]
        if window.size == 0:
            return 0.0
        val = float(np.max(window))
        return 0.0 if not np.isfinite(val) else max(val, 0.0)

    @staticmethod
    def classify_height_strata(ndsm: np.ndarray) -> np.ndarray:
        """
        Classify vertical height strata:
        0: Ground / Flat (< 1.5m)
        1: Low Vegetation / Shrub (1.5m - 3.0m)
        2: Single-Story / Medium Trees (3.0m - 8.0m)
        3: Multi-Story / Tall Trees (> 8.0m)
        """
        strata = np.zeros_like(ndsm, dtype=np.uint8)
        strata[(ndsm >= 1.5) & (ndsm < 3.0)] = 1
        strata[(ndsm >= 3.0) & (ndsm < 8.0)] = 2
        strata[ndsm >= 8.0] = 3
        return strata

    @staticmethod
    def estimate_dtm_from_dsm(
        dsm: np.ndarray,
        pixel_size_meters: float = 1.0,
        target_ground_filter_meters: float = 35.0,
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Estimates bare-earth Digital Terrain Model (DTM) from a DSM using
        a dynamic GSD-aware morphological ground filter (morphological opening).
        Ensures building roofs up to target_ground_filter_meters (35m) are filtered out.
        """
        import cv2

        dsm_2d = dsm[0] if dsm.ndim == 3 else dsm
        dsm_clean = np.nan_to_num(dsm_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if window_size is not None and window_size > 0:
            kernel_size = int(window_size)
        else:
            kernel_size = max(5, int(target_ground_filter_meters / max(pixel_size_meters, 0.001)))

        if kernel_size % 2 == 0:
            kernel_size += 1

        # Cap kernel size at reasonable maximum to prevent memory exhaustion
        kernel_size = min(kernel_size, 201)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dtm = cv2.morphologyEx(dsm_clean, cv2.MORPH_OPEN, kernel)
        dtm = np.nan_to_num(dtm, nan=0.0, posinf=0.0, neginf=0.0)
        return dtm

    @staticmethod
    def fetch_global_terrain_dem(
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        zoom: int = 14
    ) -> Optional[np.ndarray]:
        """
        Fetches bare-earth Digital Terrain Model (DTM) from global open elevation tiles
        (AWS Terrarium tiles: 0 API keys required, global coverage).
        Returns terrain elevation raster in meters above sea level.
        """
        import urllib.request
        import math
        import io
        from PIL import Image

        try:
            n = 2.0 ** zoom
            x_tile = int((min_lon + 180.0) / 360.0 * n)
            lat_rad = math.radians(max_lat)
            y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

            url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x_tile}/{y_tile}.png"
            req = urllib.request.Request(url, headers={"User-Agent": "PixaMap/1.0 (GeoAI Engine)"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                img = Image.open(io.BytesIO(resp.read()))
                arr = np.array(img, dtype=np.float32)
                # Terrarium decoding formula: (R * 256 + G + B / 256) - 32768
                elev_m = (arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0) - 32768.0
                return elev_m
        except Exception:
            return None

    @staticmethod
    def estimate_building_height_from_shadow(
        poly_coords: list,
        rgb_data: np.ndarray,
        geo_inv_fn,
        pixel_size_meters: float,
        sun_elevation_deg: float = 58.0
    ) -> Optional[float]:
        """
        Calculates physical building height (in meters) via photogrammetric shadow analysis.
        Traces cast shadows from building footprint into adjacent shadow mask (brightness < 45).
        """
        if not poly_coords or len(poly_coords) < 3:
            return None

        # Compute optical brightness
        brightness = (rgb_data[0].astype(float) + rgb_data[1].astype(float) + rgb_data[2].astype(float)) / 3.0
        h_img, w_img = brightness.shape
        shadow_mask = (brightness < 45)

        # Convert geographic polygon vertices to pixel coordinates
        try:
            px_coords = [geo_inv_fn(pt[0], pt[1]) for pt in poly_coords]
        except Exception:
            return None

        if not px_coords:
            return None

        # Sample points along exterior boundary
        pts = px_coords[::max(1, len(px_coords) // 16)]
        search_dirs = [(-1, -1), (-1, 0), (0, -1), (-1, 1)]
        all_shadow_rays = []

        for bx, by in pts:
            bx, by = int(bx), int(by)
            for dx, dy in search_dirs:
                steps = 0
                for s in range(1, 90):
                    nx, ny = bx + s * dx, by + s * dy
                    if 0 <= nx < w_img and 0 <= ny < h_img and shadow_mask[ny, nx]:
                        steps += 1
                    else:
                        break
                if steps >= 4:
                    all_shadow_rays.append(steps)

        if len(all_shadow_rays) >= 3:
            shadow_steps = float(np.percentile(all_shadow_rays, 75))
            shadow_length_m = shadow_steps * pixel_size_meters
            import math
            tan_elev = math.tan(math.radians(sun_elevation_deg))
            height_m = round(float(shadow_length_m * tan_elev), 1)
            return max(5.0, min(height_m, 140.0))

        return None
