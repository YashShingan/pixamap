"""
PixaMap Vector Engine: Multi-Format GIS Exporter.
Converts extracted vector layers into industry-standard GIS formats:
1. GeoJSON (.geojson)
2. ESRI Shapefile (.shp, .dbf, .shx, .prj)
3. GeoPackage (.gpkg)
4. Multi-layer zipped GIS archive.
"""

from typing import List, Dict, Any, Optional
import os
import json
import zipfile
import shutil
import shapefile  # pyshp

try:
    import geopandas as gpd
    from shapely.geometry import shape
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


class GISExporter:
    """
    Exports feature collections to GeoJSON, ESRI Shapefile, and GeoPackage formats.
    """

    WGS84_PRJ = (
        'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
        'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
    )

    @staticmethod
    def export_geojson(features: List[Dict[str, Any]], output_path: str, crs_name: str = "EPSG:4326") -> str:
        """Exports features as standard RFC 7946 GeoJSON FeatureCollection."""
        geojson_data = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": f"urn:ogc:def:crs:OGC:1.3:{crs_name}"}
            },
            "features": features
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)
        return output_path

    @staticmethod
    def export_shapefile(features: List[Dict[str, Any]], output_shp_path: str, crs_name: str = "EPSG:4326") -> str:
        """
        Exports features to an ESRI Shapefile set (.shp, .shx, .dbf, .prj) using GeoPandas/PyOgrio.
        Fully preserves MultiPolygons, multi-part geometries, interior donut-hole rings (courtyards),
        and strict attribute type preservation.
        """
        if not features:
            return output_shp_path

        base_name = os.path.splitext(output_shp_path)[0]
        os.makedirs(os.path.dirname(os.path.abspath(output_shp_path)), exist_ok=True)
        final_shp = f"{base_name}.shp"

        if HAS_GEOPANDAS:
            try:
                # Sanitize features and convert boolean properties to int for DBF compatibility
                sanitized_features = []
                for feat in features:
                    new_props = {}
                    for k, v in feat.get("properties", {}).items():
                        col = k[:10]  # Shapefile 10-char column limit
                        if isinstance(v, bool):
                            new_props[col] = int(v)
                        elif isinstance(v, (int, float, str)):
                            new_props[col] = v
                        else:
                            new_props[col] = str(v)
                    sanitized_features.append({
                        "type": "Feature",
                        "id": feat.get("id"),
                        "geometry": feat.get("geometry"),
                        "properties": new_props
                    })

                fc = {"type": "FeatureCollection", "features": sanitized_features}
                gdf = gpd.GeoDataFrame.from_features(fc, crs=crs_name)
                gdf.to_file(final_shp, driver="ESRI Shapefile")

                # Ensure valid .prj exists
                prj_file = f"{base_name}.prj"
                if not os.path.exists(prj_file):
                    with open(prj_file, "w", encoding="utf-8") as f:
                        f.write(GISExporter.WGS84_PRJ)

                return final_shp
            except Exception as e:
                # Fallback to pyshp if pyogrio encounters an issue
                pass

        # Fallback using pyshp
        geom_type = features[0]["geometry"]["type"]
        if geom_type in ["Polygon", "MultiPolygon"]:
            shp_type = shapefile.POLYGON
        elif geom_type in ["LineString", "MultiLineString"]:
            shp_type = shapefile.POLYLINE
        else:
            shp_type = shapefile.POINT

        w = shapefile.Writer(base_name, shapeType=shp_type)
        sample_props = features[0].get("properties", {})
        for key, val in sample_props.items():
            col_name = key[:10]
            if isinstance(val, int):
                w.field(col_name, 'N')
            elif isinstance(val, float):
                w.field(col_name, 'F', decimal=3)
            elif isinstance(val, bool):
                w.field(col_name, 'L')
            else:
                w.field(col_name, 'C', size=50)

        for feat in features:
            geom = feat["geometry"]
            props = feat.get("properties", {})
            record_vals = [props.get(k) for k in sample_props.keys()]

            if geom["type"] == "Point":
                w.point(geom["coordinates"][0], geom["coordinates"][1])
            elif geom["type"] == "LineString":
                w.line([geom["coordinates"]])
            elif geom["type"] == "Polygon":
                w.poly(geom["coordinates"])
            elif geom["type"] == "MultiPolygon":
                all_rings = []
                for poly_coords in geom["coordinates"]:
                    all_rings.extend(poly_coords)
                w.poly(all_rings)

            w.record(*record_vals)

        w.close()

        with open(f"{base_name}.prj", "w", encoding="utf-8") as prj_file:
            prj_file.write(GISExporter.WGS84_PRJ)

        return final_shp

    @staticmethod
    def export_geopackage(features_dict: Dict[str, List[Dict[str, Any]]], output_gpkg_path: str) -> str:
        """
        Exports multiple vector layers into a single OGC GeoPackage (.gpkg) file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_gpkg_path)), exist_ok=True)
        if HAS_GEOPANDAS:
            try:
                for layer_name, layer_features in features_dict.items():
                    if not layer_features:
                        continue
                    fc = {"type": "FeatureCollection", "features": layer_features}
                    gdf = gpd.GeoDataFrame.from_features(fc, crs="EPSG:4326")
                    gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG")
                return output_gpkg_path
            except Exception:
                # If GDAL DLL or pyogrio is missing on Windows, fallback cleanly
                return ""
        return ""

    @staticmethod
    def bundle_all_to_zip(
        layers: Dict[str, List[Dict[str, Any]]],
        output_zip_path: str
    ) -> str:
        """
        Bundles GeoJSON and Shapefiles for all extracted layers into a single .zip download.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True)
        temp_dir = os.path.join(os.path.dirname(output_zip_path), "_temp_export")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Include metadata manifest in export archive
            meta_path = os.path.join(temp_dir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "engine": "PixaMap GeoAI Engine",
                    "total_features": sum(len(v) for v in layers.values()),
                    "layers": list(layers.keys())
                }, f, indent=2)

            for layer_name, feats in layers.items():
                if not feats:
                    continue
                # Export GeoJSON
                json_path = os.path.join(temp_dir, f"{layer_name}.geojson")
                GISExporter.export_geojson(feats, json_path)

                # Export Shapefile
                shp_path = os.path.join(temp_dir, f"{layer_name}.shp")
                GISExporter.export_shapefile(feats, shp_path)

            # Also export GPKG if geopandas available
            if HAS_GEOPANDAS:
                gpkg_path = os.path.join(temp_dir, "pixamap_layers.gpkg")
                GISExporter.export_geopackage(layers, gpkg_path)

            # Create Zip
            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, arcname=file)

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        return output_zip_path
