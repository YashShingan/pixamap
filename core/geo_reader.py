"""
PixaMap Core: Universal Geospatial Raster Reader & Affine Geo-Referencing Engine.
Supports high-resolution drone orthophotos, multi-spectral satellite imagery,
and elevation models (DSM/DTM).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

import pyproj
from pyproj import Transformer

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    calculate_default_transform = None
    reproject = None
    Resampling = None
    Affine = None


def get_utm_crs_for_lonlat(lon: float, lat: float) -> str:
    """Returns the EPSG code string for the local UTM zone (e.g. EPSG:32643)."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = max(1, min(zone, 60))
    epsg = (32600 + zone) if lat >= 0 else (32700 + zone)
    return f"EPSG:{epsg}"


class GeoRaster:
    """
    Encapsulates a geospatial raster dataset with spatial georeferencing,
    automated UTM projection checking, affine transformations, and windowed reading.
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
        self.crs = crs.upper() if crs else "EPSG:4326"
        self.nodata = nodata

        # Determine if coordinates are in degrees (geographic)
        min_x, min_y, max_x, max_y = bounds
        self.is_geographic = (
            "4326" in self.crs
            or "WGS84" in self.crs
            or (-180.0 <= min_x <= 180.0 and -90.0 <= min_y <= 90.0)
        )

        # Compute centroid
        self.center_lon = (min_x + max_x) / 2.0
        self.center_lat = (min_y + max_y) / 2.0

        if self.is_geographic:
            self.utm_crs = get_utm_crs_for_lonlat(self.center_lon, self.center_lat)
            self._to_utm = Transformer.from_crs("EPSG:4326", self.utm_crs, always_xy=True)
            self._from_utm = Transformer.from_crs(self.utm_crs, "EPSG:4326", always_xy=True)

            # Project corners to metric UTM coordinates
            e0, n0 = self._to_utm.transform(min_x, min_y)
            e1, n1 = self._to_utm.transform(max_x, max_y)
            self.utm_bounds = (min(e0, e1), min(n0, n1), max(e0, e1), max(n0, n1))

            self.res_x_m = (self.utm_bounds[2] - self.utm_bounds[0]) / max(self.width, 1)
            self.res_y_m = (self.utm_bounds[3] - self.utm_bounds[1]) / max(self.height, 1)
        else:
            self.utm_crs = self.crs
            self._to_utm = None
            self._from_utm = None
            self.utm_bounds = bounds
            self.res_x_m = abs(max_x - min_x) / max(self.width, 1)
            self.res_y_m = abs(max_y - min_y) / max(self.height, 1)

        # WGS84 Geographic Affine Transform
        self.res_x = (max_x - min_x) / max(self.width, 1)
        self.res_y = (max_y - min_y) / max(self.height, 1)
        self.origin_x = min_x
        self.origin_y = max_y

    @property
    def pixel_size_meters(self) -> float:
        """Compute ground resolution (sampling distance) in meters per pixel without spherical distortion."""
        if hasattr(self, "_custom_pixel_size") and self._custom_pixel_size is not None and self._custom_pixel_size > 0:
            return float(self._custom_pixel_size)
        return float((self.res_x_m + self.res_y_m) / 2.0)

    def pixel_to_geo(self, px: float, py: float) -> Tuple[float, float]:
        """Convert pixel coordinates (x=column, y=row) to geographic CRS coordinates (lon, lat)."""
        x_geo = self.origin_x + (px * self.res_x)
        y_geo = self.origin_y - (py * self.res_y)
        return x_geo, y_geo

    def geo_to_pixel(self, gx: float, gy: float) -> Tuple[int, int]:
        """Convert geographic CRS coordinates to pixel coordinates (px, py)."""
        px = int((gx - self.origin_x) / self.res_x)
        py = int((self.origin_y - gy) / self.res_y)
        return px, py

    def pixel_to_utm(self, px: float, py: float) -> Tuple[float, float]:
        """Convert pixel coordinates (x=col, y=row) to metric UTM coordinates (easting, northing)."""
        if self._to_utm is not None:
            lon, lat = self.pixel_to_geo(px, py)
            return self._to_utm.transform(lon, lat)
        ux = self.utm_bounds[0] + (px * self.res_x_m)
        uy = self.utm_bounds[3] - (py * self.res_y_m)
        return ux, uy

    def utm_to_pixel(self, easting: float, northing: float) -> Tuple[int, int]:
        """Convert metric UTM coordinates to pixel coordinates (px, py)."""
        if self._from_utm is not None:
            lon, lat = self._from_utm.transform(easting, northing)
            return self.geo_to_pixel(lon, lat)
        px = int((easting - self.utm_bounds[0]) / self.res_x_m)
        py = int((self.utm_bounds[3] - northing) / self.res_y_m)
        return px, py

    def reproject_to_utm(self) -> "GeoRaster":
        """
        Dynamically reprojects the raster array to the local isometric metric UTM zone
        using rasterio.warp.reproject.
        """
        if not self.is_geographic or not HAS_RASTERIO:
            return self

        src_crs = self.crs
        dst_crs = self.utm_crs

        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, self.width, self.height, *self.bounds
        )

        dst_data = np.zeros((self.bands, height, width), dtype=self.data.dtype)
        src_transform = rasterio.transform.from_bounds(*self.bounds, self.width, self.height)

        reproject(
            source=self.data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )

        # UTM bounds: (min_easting, min_northing, max_easting, max_northing)
        utm_b = (
            transform.c,
            transform.f + (height * transform.e),
            transform.c + (width * transform.a),
            transform.f
        )
        norm_utm_b = (
            min(utm_b[0], utm_b[2]),
            min(utm_b[1], utm_b[3]),
            max(utm_b[0], utm_b[2]),
            max(utm_b[1], utm_b[3])
        )

        return GeoRaster(data=dst_data, bounds=norm_utm_b, crs=dst_crs, nodata=self.nodata)

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


def read_geotiff(
    file_path: str,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    crs: str = "EPSG:4326"
) -> GeoRaster:
    """
    Robust reader for GeoTIFFs, TIFFs, PNGs, and JPEGs.
    Extracts spatial georeferencing from GeoTIFF tags (33550 & 33922) or user bounds.
    """
    if bounds is None and HAS_RASTERIO:
        try:
            with rasterio.open(file_path) as src:
                data = src.read()
                b = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                c = src.crs.to_string() if src.crs else crs
                nodata = src.nodata
                return GeoRaster(data=data, bounds=b, crs=c, nodata=nodata)
        except Exception:
            pass

    # Open with PIL (handles TIFF, PNG, JPG, BMP, etc.)
    img = Image.open(file_path)
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode not in ("RGB", "L", "F", "I"):
        img = img.convert("RGB")

    data = np.array(img)
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    elif data.ndim == 3 and data.shape[2] in (3, 4):
        data = np.transpose(data[:, :, :3], (2, 0, 1))

    h, w = data.shape[1], data.shape[2]

    # Explicit user bounds
    if bounds is not None:
        return GeoRaster(data=data, bounds=bounds, crs=crs)

    # Attempt to extract GeoTIFF tags (33550 = pixel scale, 33922 = tiepoint)
    tags = getattr(img, "tag_v2", getattr(img, "tag", {}))
    if 33550 in tags and 33922 in tags:
        try:
            scale = tags[33550]
            tiepoint = tags[33922]
            scale_x = float(scale[0])
            scale_y = float(scale[1])
            origin_x = float(tiepoint[3])
            origin_y = float(tiepoint[4])
            min_x = origin_x
            max_y = origin_y
            max_x = origin_x + (w * scale_x)
            min_y = origin_y - (h * scale_y)
            extracted_bounds = (min_x, min_y, max_x, max_y)
            return GeoRaster(data=data, bounds=extracted_bounds, crs=crs)
        except Exception:
            pass

    # Default fallback georeferencing
    span_deg = 0.0050
    aspect = w / max(h, 1)
    default_bounds = (72.8700, 19.0700, 72.8700 + (span_deg * aspect), 19.0700 + span_deg)
    return GeoRaster(data=data, bounds=default_bounds, crs=crs)
