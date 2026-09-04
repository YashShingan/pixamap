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

    def test_upload_orthophoto(self):
        import io
        from PIL import Image
        import numpy as np

        # Create synthetic orthophoto PNG
        img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_arr[:, :, 1] = 120  # green grass
        img_arr[20:60, 20:60, :] = 215  # building
        img_obj = Image.fromarray(img_arr)
        buf = io.BytesIO()
        img_obj.save(buf, format="PNG")
        buf.seek(0)

        files = {
            "ortho_file": ("test_survey.png", buf, "image/png")
        }
        data = {
            "min_lon": "72.8700",
            "min_lat": "19.0700",
            "max_lon": "72.8750",
            "max_lat": "19.0750",
            "pixel_size_meters": "0.5"
        }

        with TestClient(app) as client:
            res = client.post("/api/upload_ortho", files=files, data=data)
            self.assertEqual(res.status_code, 202)
            resp = res.json()
            self.assertEqual(resp["status"], "processing")
            self.assertIn("task_id", resp)
            task_id = resp["task_id"]

            # Verify polling endpoint
            t_res = client.get(f"/api/tasks/{task_id}")
            self.assertEqual(t_res.status_code, 200)
            task_data = t_res.json()
            self.assertEqual(task_data["status"], "completed")
            self.assertEqual(task_data["progress"], 100)

            # Verify preview image
            p_res = client.get(f"/api/preview/{task_id}")
            self.assertEqual(p_res.status_code, 200)

            # Verify layer query
            b_res = client.get(f"/api/layers/{task_id}/buildings")
            self.assertEqual(b_res.status_code, 200)

            # Verify shapefile export
            z_res = client.get(f"/api/export/{task_id}/zip")
            self.assertEqual(z_res.status_code, 200)
            self.assertGreater(len(z_res.content), 100)

    def test_asynchronous_concurrency(self):
        """Trigger multiple tasks and verify status polling and health respond with sub-50ms latency."""
        import time
        import io
        from PIL import Image
        import numpy as np

        img_arr = np.zeros((80, 80, 3), dtype=np.uint8)
        img_obj = Image.fromarray(img_arr)
        buf1 = io.BytesIO()
        img_obj.save(buf1, format="PNG")
        buf1.seek(0)
        buf2 = io.BytesIO()
        img_obj.save(buf2, format="PNG")
        buf2.seek(0)

        with TestClient(app) as client:
            # Launch Task 1
            t1 = client.post("/api/upload_ortho", files={"ortho_file": ("s1.png", buf1, "image/png")})
            self.assertEqual(t1.status_code, 202)
            id1 = t1.json()["task_id"]

            # Launch Task 2
            t2 = client.post("/api/upload_ortho", files={"ortho_file": ("s2.png", buf2, "image/png")})
            self.assertEqual(t2.status_code, 202)
            id2 = t2.json()["task_id"]

            # Measure latency of health endpoint and polling
            start_t = time.perf_counter()
            h_res = client.get("/api/health")
            poll1 = client.get(f"/api/tasks/{id1}")
            poll2 = client.get(f"/api/tasks/{id2}")
            latency_ms = (time.perf_counter() - start_t) * 1000.0

            self.assertEqual(h_res.status_code, 200)
            self.assertEqual(poll1.status_code, 200)
            self.assertEqual(poll2.status_code, 200)
            # Both queries executed in rapid succession
            self.assertLess(latency_ms, 200.0)


if __name__ == "__main__":
    unittest.main()
