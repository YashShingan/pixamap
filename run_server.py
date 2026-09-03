"""
PixaMap Server Launcher.
Starts the FastAPI backend and serves the interactive Leaflet/MapLibre GIS Dashboard.
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure current directory is in path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print("\n" + "="*70)
    print(" 🚀 Starting PixaMap GeoAI Digitization & Regularization Engine")
    print(" 🌐 Web Dashboard: http://localhost:8000")
    print(" 📖 Interactive API Docs: http://localhost:8000/docs")
    print("="*70 + "\n")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
