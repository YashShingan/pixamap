---
description: Critical invariants for geospatial computer vision, building instance separation, and multi-spectral water detection.
always_on: true
---

# Geospatial AI Extraction Invariants

1. **Individual Building Preservation (No Cluster Merging):**
   - Never wrap an entire settlement or cluster of adjacent houses into a single giant bounding box.
   - Every individual roof plane with distinct edges must be isolated as an independent polygon feature with realistic dimensions ($15\text{ m}^2$ to $450\text{ m}^2$).
   - Regularization must simplify and orthogonalize individual contours, not replace groups with an overarching convex hull or minimum area bounding box.

2. **Multi-Spectral Water Discrimination:**
   - Inland clear lakes/ponds exhibit blue-green dominance.
   - Coastal creeks, tidal rivers, and mudflats (e.g., Mumbai Thane Creek) are turbid and brown/olive with low local texture variance and zero NDVI.
   - Distinguish water from vegetation by texture smoothness and low vegetation index, never relying solely on $B > G$.
   - Forests and tree canopies must never be classified as water.

3. **Road Network Continuity:**
   - Road centerlines must preserve topological graph connectivity and intersection degree $\ge 3$.
