"""
API Endpoint Test Suite for PixaMap.
Verifies FastAPI endpoints: /api/health, /api/demo, /api/layers, and /api/export.
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.main import app

class TestPixaMapAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")

    def test_demo_and_layers(self):
        # 1. Run demo simulation
        res = self.client.post("/api/demo")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("task_id", data)
        task_id = data["task_id"]

        # 2. Query layers
        for layer_name in ["buildings", "roads", "trees", "farms", "water"]:
            l_res = self.client.get(f"/api/layers/{task_id}/{layer_name}")
            self.assertEqual(l_res.status_code, 200)
            fc = l_res.json()
            self.assertEqual(fc["type"], "FeatureCollection")
            self.assertIn("features", fc)

        # 3. Test GeoJSON Export
        export_json = self.client.get(f"/api/export/{task_id}/geojson")
        self.assertEqual(export_json.status_code, 200)

        # 4. Test ZIP Export (Shapefiles)
        export_zip = self.client.get(f"/api/export/{task_id}/zip")
        self.assertEqual(export_zip.status_code, 200)
        self.assertGreater(len(export_zip.content), 500)

if __name__ == "__main__":
    unittest.main()
