# PixaMap: AI-Powered Automated Digitization of Orthophotos for GIS & LULC Mapping

**PixaMap** is a modular GeoAI platform designed to convert high-resolution drone and satellite orthophotos into structured, CAD/GIS-ready vector datasets at scale.

---

## 🚀 Key Architectural Capabilities

1. **CAD-Grade Building Footprint Regularization**:
   * Uses Ramer-Douglas-Peucker (RDP) simplification and Minimum Bounding Box dominant-angle extraction.
   * Snaps wobbly raw segmentation masks into strict **90-degree orthogonal CAD corners**.
   * Reduces redundant vertex count by **>75%**, dramatically shrinking file sizes.

2. **Topological Road Centerline Graph Extraction**:
   * Applies **Zhang-Suen morphological thinning** to reduce 2D road corridor swaths into 1-pixel skeletons.
   * Converts skeletons into a **NetworkX topological spatial graph**.
   * Prunes short dead-end spurs (<15m) and compresses degree-2 node chains into smooth GIS `LineString` features with connected junction nodes (degree $\ge 3$).

3. **Tree Crown Detection & 3D Inventory (nDSM)**:
   * Detects individual tree crowns via local maxima canopy peak extraction.
   * Samples the **Normalized Digital Surface Model (nDSM = DSM - DTM)** to extract **exact vertical tree height in meters** alongside crown diameter and canopy area ($m^2$).

4. **Multi-Modal Ingestion & Seamless Tiling**:
   * Supports massive GeoTIFFs (5 GB+) via sliding-window windowing.
   * Employs **2D Gaussian Apodization weighting** to seamlessly blend tile overlaps, eliminating tile-boundary seams and sliced buildings.

5. **Human-in-the-Loop Quality Control (QC)**:
   * Automated confidence scoring ($0.0 - 1.0$) per feature.
   * Visual flagging of low-confidence polygons (<0.75) for rapid human verification.

6. **Industry-Standard GIS Multi-Format Export**:
   * Exports to **ESRI Shapefile** (`.shp`, `.dbf`, `.shx`, `.prj`), **OGC GeoPackage** (`.gpkg`), and standard **GeoJSON**.

---

## 📁 Repository Structure

```
C:\Code\pixamap\
├── core\
│   ├── geo_reader.py          # Universal GeoTIFF reader & Affine CRS transformer
│   ├── elevation.py           # Multi-modal nDSM = DSM - DTM processor
│   └── tiler.py               # Sliding window tiler with 2D Gaussian apodization
├── vector\
│   ├── building_regularizer.py# Douglas-Peucker + 90° right-angle orthogonalizer
│   ├── road_centerline.py     # Zhang-Suen thinning + NetworkX graph builder
│   ├── tree_inventory.py      # Tree crown detector with 3D nDSM height attribution
│   ├── lulc_farm.py           # Farm cadastral parcels & water extraction (NDVI/NDWI)
│   └── gis_exporter.py        # Exporter to Shapefile, GeoJSON, and GeoPackage
├── models\
│   └── unified_inference.py   # Multi-task pipeline coordinator
├── api\
│   └── main.py                # FastAPI REST service & layer endpoints
├── web\
│   ├── index.html             # Interactive GIS Web Dashboard
│   ├── style.css              # Modern dark GIS theme
│   └── app.js                 # Leaflet controller & attribute inspector
├── tests\
│   ├── test_pipeline.py       # Unit test suite for geometric algorithms
│   └── test_api.py            # Integration test suite for REST endpoints
└── run_server.py              # Local development launcher
```

---

## ⚡ Quickstart

### 1. Run Automated Tests
```powershell
python tests\test_pipeline.py
python tests\test_api.py
```

### 2. Launch Interactive GIS Dashboard
```powershell
python run_server.py
```
* Open your browser at **`http://localhost:8000`**.
* Click **"Run Live GeoAI Pipeline"** to simulate full multi-modal extraction with nDSM elevation, inspect the 90° building footprints and road centerlines, filter by confidence score, and export Shapefiles!
* API Documentation available at **`http://localhost:8000/docs`**.
