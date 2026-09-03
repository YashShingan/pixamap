"""
PixaMap REST API Service.
FastAPI backend for automated orthophoto digitization, vectorization,
and GIS-ready exports.
"""

from typing import Optional, Dict, Any
import os
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from core.geo_reader import GeoRaster, read_geotiff
from core.tile_fetcher import SatelliteTileFetcher
from core.osm_fetcher import OSMReferenceFetcher
from models.unified_inference import UnifiedGeoAIEngine
from vector.gis_exporter import GISExporter

app = FastAPI(
    title="PixaMap GeoAI Engine",
    description="Automated Digitization of Drone & Satellite Orthophotos into CAD-Grade GIS Vectors",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for tasks
TASKS_DB: Dict[str, Dict[str, Any]] = {}
ENGINE = UnifiedGeoAIEngine()

# Ensure temp directory exists
TEMP_DIR = os.path.join(os.path.dirname(__file__), "..", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PixaMap GeoAI Engine",
        "version": "1.0.0",
        "hardware": {
            "device": str(getattr(ENGINE, "device", "cpu")),
            "gpu_name": getattr(ENGINE, "gpu_name", "CPU"),
            "cuda_available": hasattr(ENGINE, "device") and getattr(ENGINE, "device", None) is not None and getattr(ENGINE.device, "type", "") == "cuda"
        }
    }


class AOIRequest(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    zoom: Optional[int] = 17


@app.post("/api/extract_aoi")
def extract_aoi(req: AOIRequest):
    """
    Downloads real-world satellite imagery for the user-selected Area of Interest (AOI),
    runs multi-modal GeoAI feature extraction, and returns structured GIS metrics.
    """
    task_id = str(uuid.uuid4())[:8]
    try:
        # Fetch actual high-res satellite imagery mosaic for the exact bounding box
        ortho_raster = SatelliteTileFetcher.fetch_aoi_raster(
            min_lon=req.min_lon,
            min_lat=req.min_lat,
            max_lon=req.max_lon,
            max_lat=req.max_lat,
            zoom=req.zoom or 17
        )

        # Run multi-feature extraction on real imagery
        results = ENGINE.process_raster(ortho_raster)

        # Fetch verified ground-truth reference GIS layers (OSM Overpass)
        osm_data = OSMReferenceFetcher.fetch_reference_layers(
            min_lon=req.min_lon,
            min_lat=req.min_lat,
            max_lon=req.max_lon,
            max_lat=req.max_lat
        )

        # Augment with verified ground-truth roads and buildings
        if osm_data.get("roads"):
            results["layers"]["roads"].extend(osm_data["roads"])
        if osm_data.get("buildings"):
            results["layers"]["buildings"].extend(osm_data["buildings"])
        if osm_data.get("water"):
            results["layers"]["water"].extend(osm_data["water"])

        # Update summary counts
        results["summary"]["building_count"] = len(results["layers"]["buildings"])
        results["summary"]["road_segment_count"] = len(results["layers"]["roads"])
        results["summary"]["total_road_km"] = round(
            sum(f["properties"].get("length_m", 45.0) for f in results["layers"]["roads"]) / 1000.0, 3
        )
        results["summary"]["water_body_count"] = len(results["layers"]["water"])

        TASKS_DB[task_id] = results

        return {
            "status": "success",
            "task_id": task_id,
            "summary": results["summary"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AOI extraction failed: {str(e)}")


@app.post("/api/demo")
def run_demo_simulation():
    """
    Generates a realistic synthetic drone survey scene (with buildings, roads,
    trees, water, and nDSM elevation model) and processes it through the pipeline.
    """
    task_id = str(uuid.uuid4())[:8]
    h, w = 600, 600

    # 1. Create realistic synthetic orthophoto
    # Base background: Green field
    ortho_rgb = np.full((3, h, w), 50, dtype=np.uint8)
    ortho_rgb[1] = 120  # Green channel dominant for grass/fields

    # Add a lake (water body in top left)
    cv2.circle(ortho_rgb[0], (100, 100), 60, 20, -1)
    cv2.circle(ortho_rgb[1], (100, 100), 60, 60, -1)
    cv2.circle(ortho_rgb[2], (100, 100), 60, 200, -1)  # Blue water

    # Add Road Network (crossing horizontal and vertical)
    # Horizontal Road
    ortho_rgb[0, 280:320, :] = 130
    ortho_rgb[1, 280:320, :] = 130
    ortho_rgb[2, 280:320, :] = 130
    # Vertical Road
    ortho_rgb[0, :, 380:420] = 130
    ortho_rgb[1, :, 380:420] = 130
    ortho_rgb[2, :, 380:420] = 130

    # Add Buildings (rotations and rectangles)
    # Building 1: Rectangular commercial building
    ortho_rgb[0, 160:240, 180:320] = 210
    ortho_rgb[1, 160:240, 180:320] = 190
    ortho_rgb[2, 160:240, 180:320] = 180

    # Building 2: L-shaped residential structure
    ortho_rgb[0, 360:460, 150:230] = 220
    ortho_rgb[1, 360:460, 150:230] = 170
    ortho_rgb[2, 360:460, 150:230] = 150
    ortho_rgb[0, 420:460, 230:310] = 220
    ortho_rgb[1, 420:460, 230:310] = 170
    ortho_rgb[2, 420:460, 230:310] = 150

    # Add Trees (small dark green circles)
    tree_locs = [(180, 480), (220, 520), (260, 490), (140, 530), (350, 480), (390, 530), (450, 500)]
    for tx, ty in tree_locs:
        cv2.circle(ortho_rgb[0], (tx, ty), 16, 20, -1)
        cv2.circle(ortho_rgb[1], (tx, ty), 16, 170, -1)
        cv2.circle(ortho_rgb[2], (tx, ty), 16, 30, -1)

    # 2. Create synthetic DSM & DTM for 3D elevation
    dtm = np.full((h, w), 50.0, dtype=np.float32)  # Ground is 50m above sea level
    dsm = dtm.copy()

    # Building heights (+12m and +8m)
    dsm[160:240, 180:320] = 62.0
    dsm[360:460, 150:230] = 58.0
    dsm[420:460, 230:310] = 58.0

    # Tree heights (+6m to +11m)
    for tx, ty in tree_locs:
        cv2.circle(dsm, (tx, ty), 16, 58.5, -1)

    # Wrap in GeoRaster with geo coordinates (e.g. Mumbai / Pune region)
    # (min_lon, min_lat, max_lon, max_lat)
    bounds = (72.8700, 19.0700, 72.8760, 19.0760)
    ortho_raster = GeoRaster(data=ortho_rgb, bounds=bounds, crs="EPSG:4326")
    dsm_raster = GeoRaster(data=dsm, bounds=bounds, crs="EPSG:4326")
    dtm_raster = GeoRaster(data=dtm, bounds=bounds, crs="EPSG:4326")

    # Run processing
    results = ENGINE.process_raster(ortho_raster, dsm_raster, dtm_raster)
    TASKS_DB[task_id] = results

    return {
        "status": "success",
        "task_id": task_id,
        "summary": results["summary"]
    }


@app.get("/api/layers/{task_id}/{layer_name}")
def get_layer(task_id: str, layer_name: str):
    """Returns a specific vector layer as a standard GeoJSON FeatureCollection."""
    if task_id not in TASKS_DB:
        raise HTTPException(status_code=404, detail="Task not found")
    
    layers = TASKS_DB[task_id]["layers"]
    if layer_name not in layers:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_name}' not found")

    features = layers[layer_name]
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }


@app.get("/api/export/{task_id}/{format_type}")
def export_dataset(task_id: str, format_type: str):
    """
    Exports extracted layers to GeoJSON, Shapefile, or zipped bundle.
    """
    if task_id not in TASKS_DB:
        raise HTTPException(status_code=404, detail="Task not found")

    layers = TASKS_DB[task_id]["layers"]

    if format_type == "zip":
        zip_path = os.path.join(TEMP_DIR, f"pixamap_export_{task_id}.zip")
        GISExporter.bundle_all_to_zip(layers, zip_path)
        return FileResponse(zip_path, filename=f"pixamap_vectors_{task_id}.zip", media_type="application/zip")

    elif format_type == "geojson":
        # Combine all into one unified GeoJSON
        all_features = []
        for l_name, feats in layers.items():
            all_features.extend(feats)
        json_path = os.path.join(TEMP_DIR, f"pixamap_{task_id}.geojson")
        GISExporter.export_geojson(all_features, json_path)
        return FileResponse(json_path, filename=f"pixamap_all_{task_id}.geojson", media_type="application/geo+json")

    elif format_type == "gpkg":
        gpkg_path = os.path.join(TEMP_DIR, f"pixamap_{task_id}.gpkg")
        GISExporter.export_geopackage(layers, gpkg_path)
        if os.path.exists(gpkg_path):
            return FileResponse(gpkg_path, filename=f"pixamap_{task_id}.gpkg", media_type="application/geopackage+sqlite3")
        else:
            raise HTTPException(status_code=500, detail="GeoPackage export requires geopandas")

    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Choose: 'zip', 'geojson', or 'gpkg'.")


# Mount static web directory
WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.exists(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
