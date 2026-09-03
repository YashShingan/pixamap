/**
 * PixaMap Interactive GIS Dashboard Controller
 */

let map;
let currentTaskId = null;
let layersData = {};
let leafletLayers = {
  buildings: null,
  roads: null,
  trees: null,
  farms: null,
  water: null
};

// Initialize Map
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  setupEventListeners();
});

function initMap() {
  // Center on Mumbai coordinates (72.8777, 19.0760)
  map = L.map("map", {
    zoomControl: false
  }).setView([19.073, 72.873], 16);

  L.control.zoom({ position: "bottomright" }).addTo(map);

  // Basemap: High-Resolution Satellite & CartoDB Dark
  const satellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
    maxZoom: 19
  }).addTo(map);

  const darkLabels = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png", {
    subdomains: "abcd",
    maxZoom: 19
  }).addTo(map);
}

function setupEventListeners() {
  document.getElementById("btnRunDemo").addEventListener("click", runDemoPipeline);

  // Layer switches
  document.getElementById("chkBuildings").addEventListener("change", (e) => toggleLayer("buildings", e.target.checked));
  document.getElementById("chkRoads").addEventListener("change", (e) => toggleLayer("roads", e.target.checked));
  document.getElementById("chkTrees").addEventListener("change", (e) => toggleLayer("trees", e.target.checked));
  document.getElementById("chkFarms").addEventListener("change", (e) => toggleLayer("farms", e.target.checked));
  document.getElementById("chkWater").addEventListener("change", (e) => toggleLayer("water", e.target.checked));

  // QC Confidence Slider
  const rngConf = document.getElementById("rngConfidence");
  rngConf.addEventListener("input", (e) => {
    document.getElementById("lblConfVal").innerText = parseFloat(e.target.value).toFixed(2);
    applyConfidenceFilter(parseFloat(e.target.value));
  });

  // Export Buttons
  document.getElementById("btnExportZip").addEventListener("click", () => {
    if (currentTaskId) window.location.href = `/api/export/${currentTaskId}/zip`;
  });

  document.getElementById("btnExportGeoJSON").addEventListener("click", () => {
    if (currentTaskId) window.location.href = `/api/export/${currentTaskId}/geojson`;
  });

  // Inspector close
  document.getElementById("btnCloseInspector").addEventListener("click", () => {
    document.getElementById("inspectorPanel").classList.add("hidden");
  });
}

async function runDemoPipeline() {
  const btn = document.getElementById("btnRunDemo");
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing GeoAI Pipeline...`;

  try {
    const res = await fetch("/api/demo", { method: "POST" });
    const data = await res.json();

    if (data.status === "success") {
      currentTaskId = data.task_id;
      updateSummaryMetrics(data.summary);
      await loadAllLayers(data.task_id);

      // Enable export buttons
      document.getElementById("btnExportZip").disabled = false;
      document.getElementById("btnExportGeoJSON").disabled = false;

      // Fit map bounds
      const b = data.summary.bounds; // (min_x, min_y, max_x, max_y)
      map.fitBounds([[b[1], b[0]], [b[3], b[2]]]);
    }
  } catch (err) {
    console.error("Error executing pipeline:", err);
    alert("Pipeline execution failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Re-Run GeoAI Pipeline`;
  }
}

function updateSummaryMetrics(summary) {
  document.getElementById("metricBuildings").innerText = summary.building_count;
  document.getElementById("metricRoads").innerText = `${summary.total_road_km} km`;
  document.getElementById("metricTrees").innerText = summary.tree_count;
  document.getElementById("metricFarms").innerText = summary.farm_parcel_count;
}

async function loadAllLayers(taskId) {
  const layerNames = ["buildings", "roads", "trees", "farms", "water"];

  for (const name of layerNames) {
    try {
      const res = await fetch(`/api/layers/${taskId}/${name}`);
      const geojson = await res.json();
      layersData[name] = geojson;
      renderLayer(name, geojson);
    } catch (e) {
      console.warn(`Could not load layer ${name}:`, e);
    }
  }
}

function renderLayer(name, geojson) {
  if (leafletLayers[name]) {
    map.removeLayer(leafletLayers[name]);
  }

  let layerGroup;

  if (name === "buildings") {
    layerGroup = L.geoJSON(geojson, {
      style: (feature) => {
        const isLowConf = feature.properties.confidence_score < 0.75;
        return {
          color: isLowConf ? "#ef4444" : "#ff5722",
          weight: 2,
          fillColor: isLowConf ? "#ef4444" : "#ff5722",
          fillOpacity: 0.55
        };
      },
      onEachFeature: (feat, layer) => {
        layer.on("click", () => showInspector("Building Footprint (90° Snap)", feat.properties));
      }
    });
  } else if (name === "roads") {
    layerGroup = L.geoJSON(geojson, {
      style: {
        color: "#ffeb3b",
        weight: 4,
        opacity: 0.9,
        lineCap: "round",
        lineJoin: "round"
      },
      onEachFeature: (feat, layer) => {
        layer.on("click", () => showInspector("Road Centerline Graph", feat.properties));
      }
    });
  } else if (name === "trees") {
    layerGroup = L.geoJSON(geojson, {
      pointToLayer: (feature, latlng) => {
        const rad = Math.max(5, (feature.properties.crown_diameter_m || 2) * 2.5);
        return L.circleMarker(latlng, {
          radius: rad,
          fillColor: "#4caf50",
          color: "#2e7d32",
          weight: 2,
          opacity: 0.9,
          fillOpacity: 0.7
        });
      },
      onEachFeature: (feat, layer) => {
        layer.on("click", () => showInspector("Tree Inventory (3D Height)", feat.properties));
      }
    });
  } else if (name === "farms") {
    layerGroup = L.geoJSON(geojson, {
      style: {
        color: "#8bc34a",
        weight: 2,
        fillColor: "#8bc34a",
        fillOpacity: 0.35,
        dashArray: "4, 4"
      },
      onEachFeature: (feat, layer) => {
        layer.on("click", () => showInspector("Farm Parcel Boundary", feat.properties));
      }
    });
  } else if (name === "water") {
    layerGroup = L.geoJSON(geojson, {
      style: {
        color: "#00bcd4",
        weight: 2,
        fillColor: "#0097a7",
        fillOpacity: 0.65
      },
      onEachFeature: (feat, layer) => {
        layer.on("click", () => showInspector("Water Body", feat.properties));
      }
    });
  }

  if (layerGroup) {
    leafletLayers[name] = layerGroup;
    layerGroup.addTo(map);
  }
}

function toggleLayer(name, isVisible) {
  if (!leafletLayers[name]) return;
  if (isVisible) {
    map.addLayer(leafletLayers[name]);
  } else {
    map.removeLayer(leafletLayers[name]);
  }
}

function applyConfidenceFilter(threshold) {
  if (leafletLayers.buildings) {
    leafletLayers.buildings.eachLayer((layer) => {
      const conf = layer.feature.properties.confidence_score;
      if (conf < threshold) {
        layer.setStyle({ color: "#ef4444", fillColor: "#ef4444", fillOpacity: 0.85, weight: 3 });
      } else {
        layer.setStyle({ color: "#ff5722", fillColor: "#ff5722", fillOpacity: 0.55, weight: 2 });
      }
    });
  }
}

function showInspector(title, props) {
  document.getElementById("inspectorTitle").innerHTML = `<i class="fa-solid fa-circle-info"></i> ${title}`;
  const container = document.getElementById("inspectorContent");
  container.innerHTML = "";

  for (const [key, val] of Object.entries(props)) {
    const row = document.createElement("div");
    row.className = "attr-row";

    let valDisplay = val;
    if (key === "needs_review") {
      valDisplay = val ? `<span class="tag-review">NEEDS REVIEW</span>` : `<span class="tag-approved">APPROVED</span>`;
    } else if (key === "area_sqm") {
      valDisplay = `${val} m²`;
    } else if (key === "height_m") {
      valDisplay = `${val} m (nDSM)`;
    } else if (key === "crown_diameter_m") {
      valDisplay = `${val} m`;
    } else if (key === "length_m") {
      valDisplay = `${val} m`;
    } else if (key === "vertex_reduction_pct") {
      valDisplay = `${val}% reduced`;
    }

    row.innerHTML = `
      <span class="attr-key">${key.replace(/_/g, " ")}:</span>
      <span class="attr-val">${valDisplay}</span>
    `;
    container.appendChild(row);
  }

  document.getElementById("inspectorPanel").classList.remove("hidden");
}
