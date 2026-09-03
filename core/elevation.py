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
        Clamps negative values (caused by sensor noise) to 0.0.
        """
        # Ensure 2D arrays
        dsm_2d = dsm[0] if dsm.ndim == 3 else dsm
        dtm_2d = dtm[0] if dtm.ndim == 3 else dtm

        # Compute relative vertical height above ground
        ndsm = dsm_2d.astype(np.float32) - dtm_2d.astype(np.float32)
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
        return float(np.max(window))

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
