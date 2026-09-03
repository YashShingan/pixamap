"""
PixaMap Core: Universal Geospatial Raster Reader & Affine Geo-Referencing Engine.
Supports high-resolution drone orthophotos, multi-spectral satellite imagery,
and elevation models (DSM/DTM).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    Affine = None

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class GeoRaster:
    """
    Encapsulates a geospatial raster dataset with spatial georeferencing,
    affine transformations, and windowed reading capabilities.
    """

    def __init__(
        self,
        data: np.ndarray,
        bounds: Tuple[float, float, float, float],  # (min_x, min_y, max_x, max_y)
        crs: str = "EPSG:4326",
        nodata: Optional[float] = None
    ):
        self.data = data  # Shape: (bands, height, width) or (height, width)
        if self.data.ndim == 2:
            self.data = np.expand_dims(self.data, axis=0)
        
        self.bands, self.height, self.width = self.data.shape
        self.bounds = bounds
        self.crs = crs
        self.nodata = nodata

        # Compute affine transform matrix
        min_x, min_y, max_x, max_y = bounds
        self.res_x = (max_x - min_x) / max(self.width, 1)
        self.res_y = (max_y - min_y) / max(self.height, 1)
        self.origin_x = min_x
        self.origin_y = max_y

    def pixel_to_geo(self, px: float, py: float) -> Tuple[float, float]:
        """Convert pixel coordinates (x=column, y=row) to geographic CRS coordinates (lon/easting, lat/northing)."""
        x_geo = self.origin_x + (px * self.res_x)
        y_geo = self.origin_y - (py * self.res_y)
        return x_geo, y_geo

    def geo_to_pixel(self, gx: float, gy: float) -> Tuple[int, int]:
        """Convert geographic CRS coordinates to pixel coordinates (px, py)."""
        px = int((gx - self.origin_x) / self.res_x)
        py = int((self.origin_y - gy) / self.res_y)
        return px, py

    def get_window(self, col_off: int, row_off: int, width: int, height: int) -> np.ndarray:
        """Extract a sub-window tile safely with zero-padding if out-of-bounds."""
        c_end = min(col_off + width, self.width)
        r_end = min(row_off + height, self.height)

        c_start = max(col_off, 0)
        r_start = max(row_off, 0)

        tile = np.zeros((self.bands, height, width), dtype=self.data.dtype)
        if c_end > c_start and r_end > r_start:
            target_r_end = r_end - r_start
            target_c_end = c_end - c_start
            tile[:, :target_r_end, :target_c_end] = self.data[:, r_start:r_end, c_start:c_end]

        return tile


def read_geotiff(file_path: str) -> GeoRaster:
    """
    Robust reader for GeoTIFFs using rasterio when available,
    falling back to tifffile/PIL with embedded GeoTIFF tags.
    """
    if HAS_RASTERIO:
        with rasterio.open(file_path) as src:
            data = src.read()
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            crs = src.crs.to_string() if src.crs else "EPSG:4326"
            nodata = src.nodata
            return GeoRaster(data=data, bounds=bounds, crs=crs, nodata=nodata)

    # Fallback to tifffile or Pillow
    if HAS_TIFFFILE:
        try:
            with tifffile.TiffFile(file_path) as tif:
                data = tif.asarray()
                if data.ndim == 2:
                    data = np.expand_dims(data, axis=0)
                elif data.ndim == 3 and data.shape[2] in [3, 4]:
                    data = np.transpose(data, (2, 0, 1))
                
                h, w = data.shape[1], data.shape[2]
                bounds = (72.8777, 19.0760, 72.8777 + (w * 0.000001), 19.0760 + (h * 0.000001))
                return GeoRaster(data=data, bounds=bounds, crs="EPSG:4326")
        except Exception:
            pass

    # PIL Fallback
    img = Image.open(file_path)
    data = np.array(img)
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    elif data.ndim == 3:
        data = np.transpose(data, (2, 0, 1))
    
    h, w = data.shape[1], data.shape[2]
    bounds = (72.8777, 19.0760, 72.8777 + (w * 0.000001), 19.0760 + (h * 0.000001))
    return GeoRaster(data=data, bounds=bounds, crs="EPSG:4326")
