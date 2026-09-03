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
    print(" [*] Starting PixaMap GeoAI Digitization & Regularization Engine")
    print(" [*] Web Dashboard: http://localhost:8000")
    print(" [*] Interactive API Docs: http://localhost:8000/docs")
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f" [*] Hardware Acceleration: CUDA ENABLED ({gpu} | {vram:.1f} GB VRAM)")
        else:
            print(" [*] Hardware Acceleration: CPU Mode (CUDA not available)")
    except Exception:
        print(" [*] Hardware Acceleration: Standard CPU Mode")
    print("="*70 + "\n")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
