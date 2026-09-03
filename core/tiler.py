"""
PixaMap Core: Seamless Tiling & 2D Gaussian Apodization Engine.
Splits multi-gigabyte orthophotos into overlapping windows and stitches
probability maps without tile-boundary seam cuts.
"""

from typing import List, Tuple, Generator, Dict, Any
import numpy as np


class SlidingWindowTiler:
    """
    Slices large geospatial rasters into overlapping tiles and provides
    Gaussian apodization weighting to assemble seamless prediction maps.
    """

    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self._gaussian_weight_cache = None

    def get_gaussian_weight_matrix(self) -> np.ndarray:
        """
        Creates a 2D Gaussian apodization bell curve matrix of shape (tile_size, tile_size).
        Center pixels are weighted ~1.0; boundary pixels smoothly taper toward ~0.05.
        """
        if self._gaussian_weight_cache is not None:
            return self._gaussian_weight_cache

        size = self.tile_size
        sigma = size / 4.0
        center = (size - 1) / 2.0

        y, x = np.ogrid[:size, :size]
        dist_sq = (x - center) ** 2 + (y - center) ** 2
        weight = np.exp(-dist_sq / (2.0 * (sigma ** 2)))
        weight = np.clip(weight, 0.05, 1.0)
        
        self._gaussian_weight_cache = weight.astype(np.float32)
        return self._gaussian_weight_cache

    def generate_windows(self, height: int, width: int) -> Generator[Tuple[int, int, int, int], None, None]:
        """
        Generates (row_off, col_off, h, w) window coordinates covering the raster.
        Ensures the entire raster is covered including right and bottom edges.
        """
        row_offsets = list(range(0, max(height - self.tile_size + 1, 1), self.stride))
        if row_offsets[-1] + self.tile_size < height:
            row_offsets.append(height - self.tile_size)
        elif not row_offsets:
            row_offsets = [0]

        col_offsets = list(range(0, max(width - self.tile_size + 1, 1), self.stride))
        if col_offsets[-1] + self.tile_size < width:
            col_offsets.append(width - self.tile_size)
        elif not col_offsets:
            col_offsets = [0]

        for r in row_offsets:
            for c in col_offsets:
                r_actual = max(0, r)
                c_actual = max(0, c)
                h_actual = min(self.tile_size, height - r_actual)
                w_actual = min(self.tile_size, width - c_actual)
                yield r_actual, c_actual, h_actual, w_actual

    def create_accumulator(self, num_classes: int, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates global probability accumulator and normalization weight matrices.
        """
        prob_accum = np.zeros((num_classes, height, width), dtype=np.float32)
        weight_accum = np.zeros((height, width), dtype=np.float32)
        return prob_accum, weight_accum

    def accumulate_tile(
        self,
        prob_accum: np.ndarray,
        weight_accum: np.ndarray,
        tile_probs: np.ndarray,  # (num_classes, tile_size, tile_size)
        r_off: int,
        c_off: int,
        h: int,
        w: int
    ):
        """
        Blends a single tile's predictions into the global accumulator using 2D Gaussian weights.
        """
        weight = self.get_gaussian_weight_matrix()[:h, :w]
        sub_probs = tile_probs[:, :h, :w]

        # Accumulate weighted probabilities
        for k in range(prob_accum.shape[0]):
            prob_accum[k, r_off:r_off+h, c_off:c_off+w] += sub_probs[k] * weight
        
        weight_accum[r_off:r_off+h, c_off:c_off+w] += weight

    def finalize(self, prob_accum: np.ndarray, weight_accum: np.ndarray) -> np.ndarray:
        """
        Normalizes accumulated probabilities by total weights to yield clean [0.0, 1.0] maps.
        """
        weight_safe = np.maximum(weight_accum, 1e-6)
        normalized = prob_accum / weight_safe[np.newaxis, :, :]
        return np.clip(normalized, 0.0, 1.0)
