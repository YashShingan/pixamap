"""
Automated Unit & Integration Test Suite for PixaMap GeoAI Engine.
"""

import os
import sys
import unittest
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.geo_reader import GeoRaster
from core.elevation import ElevationProcessor
from core.tiler import SlidingWindowTiler
from vector.building_regularizer import BuildingRegularizer
from vector.road_centerline import RoadCenterlineExtractor
from vector.tree_inventory import TreeInventoryExtractor
from vector.gis_exporter import GISExporter
from models.unified_inference import UnifiedGeoAIEngine


class TestPixaMapPipeline(unittest.TestCase):

    def setUp(self):
        self.bounds = (72.8700, 19.0700, 72.8750, 19.0750)
        self.dummy_geo_fn = lambda px, py: (72.8700 + px * 0.00001, 19.0750 - py * 0.00001)

    def test_elevation_ndsm(self):
        """Verify nDSM = DSM - DTM formula isolates object height above ground."""
        dsm = np.array([[55.0, 62.0], [50.0, 70.0]], dtype=np.float32)
        dtm = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32)

        ndsm = ElevationProcessor.compute_ndsm(dsm, dtm)
        self.assertEqual(ndsm[0, 0], 5.0)   # 5m object
        self.assertEqual(ndsm[0, 1], 12.0)  # 12m building
        self.assertEqual(ndsm[1, 0], 0.0)   # Ground
        self.assertEqual(ndsm[1, 1], 20.0)  # 20m structure

    def test_tiler_gaussian(self):
        """Verify 2D Gaussian apodization bell curve properties."""
        tiler = SlidingWindowTiler(tile_size=64, overlap=16)
        weight = tiler.get_gaussian_weight_matrix()

        self.assertEqual(weight.shape, (64, 64))
        # Center should be close to 1.0
        self.assertAlmostEqual(weight[31, 31], 1.0, places=1)
        # Corners should taper down
        self.assertLess(weight[0, 0], 0.2)

    def test_building_regularization(self):
        """Verify CAD-grade 90-degree orthogonal regularization on synthetic building."""
        # Create an image with a slightly noisy rectangle
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:80] = 255
        # Add a couple of noisy pixels on border
        mask[19, 45] = 255
        mask[61, 50] = 255

        reg = BuildingRegularizer(pixel_size_meters=0.05)
        features = reg.regularize_mask(mask, self.dummy_geo_fn)

        self.assertGreater(len(features), 0)
        building = features[0]
        self.assertEqual(building["properties"]["feature_type"], "building")
        self.assertIn("area_sqm", building["properties"])
        self.assertIn("orientation_deg", building["properties"])
        # Should have significant vertex reduction
        self.assertGreater(building["properties"]["vertex_reduction_pct"], 50.0)

    def test_road_centerline_extraction(self):
        """Verify topological thinning and LineString network generation."""
        # Create a T-shaped road intersection
        mask = np.zeros((120, 120), dtype=np.uint8)
        # Horizontal corridor (10px wide)
        mask[55:65, 10:110] = 255
        # Vertical corridor (10px wide)
        mask[60:110, 55:65] = 255

        extractor = RoadCenterlineExtractor(pixel_size_meters=0.05, min_road_length_meters=0.5)
        roads = extractor.extract_centerlines(mask, self.dummy_geo_fn)

        self.assertGreater(len(roads), 0)
        self.assertEqual(roads[0]["geometry"]["type"], "LineString")
        self.assertGreater(roads[0]["properties"]["length_m"], 0.5)

    def test_tree_inventory(self):
        """Verify tree crown detection and 3D height sampling from nDSM."""
        tree_mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.circle(tree_mask, (40, 40), 12, 255, -1)

        ndsm = np.zeros((80, 80), dtype=np.float32)
        cv2.circle(ndsm, (40, 40), 12, 14.5, -1)  # 14.5m tall tree

        tree_ext = TreeInventoryExtractor(pixel_size_meters=0.05)
        trees = tree_ext.extract_tree_inventory(tree_mask, self.dummy_geo_fn, ndsm=ndsm)

        self.assertEqual(len(trees), 1)
        tree = trees[0]
        self.assertEqual(tree["geometry"]["type"], "Point")
        self.assertEqual(tree["properties"]["height_m"], 14.5)
        self.assertGreater(tree["properties"]["crown_diameter_m"], 0.5)

    def test_gis_exporter(self):
        """Verify export to GeoJSON and ESRI Shapefile."""
        test_features = [{
            "type": "Feature",
            "id": 1,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[72.871, 19.071], [72.873, 19.071], [72.873, 19.073], [72.871, 19.073], [72.871, 19.071]]]
            },
            "properties": {
                "bldg_id": 1,
                "area_sqm": 450.5,
                "height_m": 12.0
            }
        }]

        temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "temp"))
        os.makedirs(temp_dir, exist_ok=True)

        # Test GeoJSON
        json_out = os.path.join(temp_dir, "test_bldg.geojson")
        GISExporter.export_geojson(test_features, json_out)
        self.assertTrue(os.path.exists(json_out))

        # Test Shapefile
        shp_out = os.path.join(temp_dir, "test_bldg.shp")
        GISExporter.export_shapefile(test_features, shp_out)
        self.assertTrue(os.path.exists(shp_out))
        self.assertTrue(os.path.exists(shp_out.replace(".shp", ".dbf")))
        self.assertTrue(os.path.exists(shp_out.replace(".shp", ".prj")))

    def test_unified_engine_full_run(self):
        """Verify unified engine executes smoothly across all 5 layers."""
        h, w = 150, 150
        rgb = np.zeros((3, h, w), dtype=np.uint8)
        # Green field background
        rgb[1] = 130
        # Building
        rgb[:, 30:70, 30:70] = 200
        # Road
        rgb[:, 80:95, :] = 120
        # Tree
        cv2.circle(rgb[1], (110, 40), 10, 220, -1)

        dsm = np.full((h, w), 50.0, dtype=np.float32)
        dtm = np.full((h, w), 50.0, dtype=np.float32)
        dsm[30:70, 30:70] = 62.0  # +12m building
        cv2.circle(dsm, (110, 40), 10, 58.0, -1)  # +8m tree

        raster = GeoRaster(rgb, bounds=self.bounds)
        dsm_r = GeoRaster(dsm, bounds=self.bounds)
        dtm_r = GeoRaster(dtm, bounds=self.bounds)

        engine = UnifiedGeoAIEngine(pixel_size_meters=0.05)
        results = engine.process_raster(raster, dsm_r, dtm_r)

        summary = results["summary"]
        self.assertIn("building_count", summary)
        self.assertIn("road_segment_count", summary)
        self.assertIn("tree_count", summary)
        self.assertGreater(summary["building_count"], 0)

    def test_reprojection_edge_cases(self):
        """Verify UTM CRS detection and isometric GSD at equator and high latitudes."""
        # Equator test (lat 0, lon 10) -> EPSG:32632
        data_eq = np.zeros((3, 100, 100), dtype=np.uint8)
        bounds_eq = (10.0, 0.0, 10.005, 0.005)
        r_eq = GeoRaster(data_eq, bounds_eq, crs="EPSG:4326")
        self.assertEqual(r_eq.utm_crs, "EPSG:32632")
        self.assertGreater(r_eq.pixel_size_meters, 0.0)

        # High latitude test (lat 60N, lon 10E) -> EPSG:32632
        bounds_high = (10.0, 60.0, 10.005, 60.005)
        r_high = GeoRaster(data_eq, bounds_high, crs="EPSG:4326")
        self.assertEqual(r_high.utm_crs, "EPSG:32632")
        self.assertGreater(r_high.pixel_size_meters, 0.0)
        # Verify lon distance shrinks at 60 deg latitude compared to equator
        self.assertLess(r_high.pixel_size_meters, r_eq.pixel_size_meters)

    def test_orthogonalization_l_shaped_building(self):
        """Verify dominant-angle CAD orthogonalization on L-shaped building with 3D heights."""
        mask = np.zeros((120, 120), dtype=np.uint8)
        # Create an L-shaped building
        mask[20:80, 20:50] = 255
        mask[50:80, 50:90] = 255

        ndsm = np.zeros((120, 120), dtype=np.float32)
        ndsm[mask == 255] = 8.5  # 8.5m roof

        reg = BuildingRegularizer(pixel_size_meters=0.1)
        features = reg.regularize_mask(mask, self.dummy_geo_fn, ndsm=ndsm)

        self.assertGreater(len(features), 0)
        bldg = features[0]
        self.assertEqual(bldg["geometry"]["type"], "Polygon")
        self.assertIn("height_max", bldg["properties"])
        self.assertAlmostEqual(bldg["properties"]["height_max"], 8.5, delta=0.5)
        self.assertIn("roof_profile", bldg["properties"])
        self.assertEqual(bldg["properties"]["roof_profile"], "flat")

    def test_donut_hole_shapefile_export(self):
        """Verify exporting polygons with interior courtyard rings to Shapefile."""
        import geopandas as gpd

        donut_feature = [{
            "type": "Feature",
            "id": 1,
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    # Exterior ring (100x100m)
                    [[72.871, 19.071], [72.873, 19.071], [72.873, 19.073], [72.871, 19.073], [72.871, 19.071]],
                    # Interior courtyard ring
                    [[72.8715, 19.0715], [72.8725, 19.0715], [72.8725, 19.0725], [72.8715, 19.0725], [72.8715, 19.0715]]
                ]
            },
            "properties": {
                "bldg_id": 99,
                "has_court": True,
                "height_max": 14.2
            }
        }]

        temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "temp"))
        shp_path = os.path.join(temp_dir, "donut_test.shp")
        GISExporter.export_shapefile(donut_feature, shp_path)

        # Read back using GeoPandas
        gdf = gpd.read_file(shp_path)
        geom = gdf.geometry.iloc[0]
        self.assertEqual(geom.geom_type, "Polygon")
        self.assertEqual(len(geom.interiors), 1)  # Interior courtyard preserved!


if __name__ == "__main__":
    unittest.main()
