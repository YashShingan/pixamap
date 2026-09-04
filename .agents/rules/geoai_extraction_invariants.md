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

3. **Road Network Continuity & Corridor Filtering:**
   - Road centerlines must preserve topological graph connectivity and intersection degree $\ge 3$.
   - Road detection strictly requires multi-directional linear corridor continuity ($0^\circ, 90^\circ, 45^\circ, 135^\circ$ kernels $\ge 11\,\text{px}$) to reject parking lots, courtyards, and cross-lane perpendicular zebra cuts.
   - Discard isolated stubs shorter than $45\,\text{m}$. Authoritative OSM roads form the primary clean backbone.

4. **Topological Mutual Exclusion Cascade:**
   - $\text{Trees} \cap \text{Water} = \emptyset$: Tree crowns are mathematically barred from spawning inside rivers, creeks, or water bodies.
   - $\text{Trees} \cap \text{Buildings} = \emptyset$ and $\text{Trees} \cap \text{Roads} = \emptyset$: No trees on rooftops or active highway surfaces.
   - Natural satellite spectral boundaries define rivers/creeks; do not artificially buffer LineString centerlines into rigid 33m tubes.

5. **Concrete Urban Apartment Flat Roof Detection:**
   - White, cream, and grey flat roofs with high local contrast must be captured as buildings and protected from asphalt road filtering using aspect ratio invariants ($< 3.5:1$ is always a building).
   - Tree crown visual diameters must be calibrated to realistic scales ($2.0\text{–}4.0\,\text{m}$).
