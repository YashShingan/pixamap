# PixaMap: AI-Powered Automated Digitization of Orthophotos for GIS & 3D Digital Twin Mapping

**PixaMap** is a high-performance GeoAI platform designed to convert high-resolution drone and satellite orthophotos into structured, CAD/GIS-ready vector datasets and immersive 3D digital twins at scale.

---

## 🚀 Key Architectural Capabilities

1. **Unified Single-Engine 2D/3D Map Architecture (MapLibre GL JS)**:
   * **Single WebGL Canvas**: Eliminated dual-map sync lag; seamlessly handles both 2D ortho mapping and 3D digital twin visualization on one MapLibre GL instance.
   * **3D Fill-Extrusion**: Real-time volumetric building extrusions with authentic height rendering and procedural tree canopy cylinders.
   * **Kinematic Perspective Transitions**: Smooth camera easing between top-down 2D orthographic view (pitch $0^\circ$) and 3D oblique perspective (pitch $60^\circ$, bearing $35^\circ$).

2. **CAD-Grade Building Footprint Regularization**:
   * Uses Ramer-Douglas-Peucker (RDP) simplification and Minimum Bounding Box dominant-angle extraction.
   * Snaps wobbly raw segmentation masks into strict **90-degree orthogonal CAD corners**.
   * Reduces redundant vertex count by **>75%**, dramatically shrinking GeoJSON/Shapefile export sizes.

3. **Multi-Tier Authentic 3D Building Height Engine**:
   * **Tier 1 (OSM Verification)**: Ingests municipal floor count tags (`building:levels`) scaled at $3.3\,\text{m}$ per story.
   * **Tier 2 (Photogrammetric Shadow Estimation)**: Estimates true vertical clearance by analyzing cast shadows along the sun azimuth vector against high-resolution imagery.
   * **Tier 3 (Calibrated Physical Baselines)**: Conservative urban typology defaults avoiding exaggerated heights on large low-rise footprints (e.g. warehouses, schools).

4. **False-Positive Discrimination & Urban Infrastructure Filtering**:
   * **Open Ground / Sports Pitch Detection**: Spectral & texture-based discrimination identifying bare clay pitches (e.g., cricket pitches) and mown turf fields, preventing false building, water, or farm boundary classifications.
   * **Linear Transit Infrastructure Corridors**: Aspect-ratio filtering ($>6:1$, continuous length $>80\,\text{m}$) to classify and separate elevated metro lines, flyovers, and viaducts from building extrusions.
   * **Mutual Exclusion Topology**: Guarantees zero water bodies overlapping rooftops or road corridors.

5. **Topological Road Centerline Graph Extraction**:
   * Applies **Zhang-Suen morphological thinning** to reduce 2D road corridors into single-pixel skeletons.
   * Converts skeletons into a **NetworkX topological spatial graph**.
   * Prunes short dead-end spurs ($<25\,\text{m}$), removes building penetrations, and deduplicates against verified OpenStreetMap road centerlines (15m buffer).

6. **Tree Crown Detection & 3D Inventory**:
   * Detects individual tree crowns via local maxima canopy peak extraction.
   * Attributes tree canopy radius, area ($m^2$), and 3D vertical height from Normalized Digital Surface Models ($\text{nDSM} = \text{DSM} - \text{DTM}$).

7. **Multi-Format Industry-Standard GIS Export**:
   * One-click client/API export to **ESRI Shapefile** (`.shp`, `.dbf`, `.shx`, `.prj`), **OGC GeoPackage** (`.gpkg`), and standard **GeoJSON**.

8. **Hardware Acceleration**:
   * PyTorch CUDA acceleration support for NVIDIA RTX GPUs with automatic CPU fallback.

---

## 📁 Repository Structure

```
pixamap/
├── core/
│   ├── geo_reader.py          # Universal GeoTIFF reader & Affine CRS transformer
│   ├── elevation.py           # Multi-modal nDSM = DSM - DTM processor & shadow height estimator
│   ├── osm_fetcher.py         # Overpass API client for OSM road/building ground-truth
│   ├── tile_fetcher.py        # Dynamic satellite imagery fetcher
│   └── tiler.py               # Sliding window tiler with 2D Gaussian apodization
├── vector/
│   ├── building_regularizer.py# Douglas-Peucker + 90° right-angle orthogonalizer
│   ├── road_centerline.py     # Zhang-Suen thinning + NetworkX topological graph builder
│   ├── tree_inventory.py      # Tree crown detector with 3D height attribution
│   ├── lulc_farm.py           # Farm cadastral parcels & water body extraction
│   └── gis_exporter.py        # Exporter to Shapefile, GeoJSON, and GeoPackage
├── models/
│   └── unified_inference.py   # Multi-task GeoAI probability map synthesis & CUDA pipelines
├── api/
│   └── main.py                # FastAPI REST service, async worker tasks & export endpoints
├── web/
│   ├── index.html             # Single-canvas MapLibre GL web dashboard
│   ├── style.css              # Cyberpunk dark GIS theme & responsive controls
│   └── app.js                 # Unified MapLibre GL 2D/3D map controller & attribute inspector
├── tests/
│   ├── test_pipeline.py       # Geometric, topological, and regularization test suites
│   └── test_api.py            # API concurrency, layer extraction, and upload tests
└── run_server.py              # Local development server launcher
```

---

## ⚡ Quickstart

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```powershell
pytest tests/ -v
```

### 3. Launch Interactive GIS Dashboard
```powershell
python run_server.py
```
* Access the web dashboard at **`http://127.0.0.1:8000`**.
* Interactive API documentation is available at **`http://127.0.0.1:8000/docs`**.
