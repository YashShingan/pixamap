/**
 * PixaMap — "Nebula" controller
 * Radar splash · aurora theme (dark/light, persisted, key T) ·
 * pipeline overlay · scan sweep · animated counters · toasts ·
 * live coords + UTC clock · click-to-toggle legend · QC counter.
 * All API endpoints & element IDs unchanged.
 */

let map;
let currentTaskId = null;
let layersData = {};
let currentThreshold = 0.75;
let leafletLayers = { buildings: null, roads: null, trees: null, farms: null, water: null };

let currentAOIBounds = null;
let aoiRectangleLayer = null;
let isDrawingBox = false;
let drawStartLatLng = null;

/* ---------- Theme ---------- */
const THEME_KEY = "pixamap-theme";

function initTheme() {
  const theme = localStorage.getItem(THEME_KEY) || "dark";
  document.documentElement.setAttribute("data-theme", theme);
  paintThemeButton(theme);
}

function toggleTheme() {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem(THEME_KEY, next);
  paintThemeButton(next);
  toast("info", next === "dark" ? "Midnight mode" : "Daylight mode", "Theme switched.");
}

function paintThemeButton(theme) {
  document.getElementById("btnThemeToggle").innerHTML =
    theme === "dark" ? `<i class="fa-solid fa-sun"></i>` : `<i class="fa-solid fa-moon"></i>`;
}

/* ---------- Layer metadata ---------- */
const LAYER_META = {
  buildings: { color: "#fb923c", fill: "#fb923c", label: "Building Footprint · 90° Snap", icon: "fa-building" },
  roads:     { color: "#facc15", fill: "#facc15", label: "Road Centerline Graph",          icon: "fa-road" },
  trees:     { color: "#4ade80", fill: "#4ade80", label: "Tree Inventory · 3D Height",     icon: "fa-tree" },
  farms:     { color: "#a3e635", fill: "#a3e635", label: "Farm Parcel Boundary",           icon: "fa-wheat-awn" },
  water:     { color: "#38bdf8", fill: "#0ea5e9", label: "Water Body",                     icon: "fa-water" },
};

const PIPELINE_STAGES = [
  { icon: "fa-satellite",      label: "Fetching satellite imagery" },
  { icon: "fa-brain",          label: "Neural segmentation (Deep Learning)" },
  { icon: "fa-vector-square",  label: "Feature vectorization" },
  { icon: "fa-ruler-combined", label: "90° footprint regularization" },
  { icon: "fa-shield-halved",  label: "Confidence scoring & QC" },
  { icon: "fa-layer-group",    label: "Packaging GIS layers" },
];

/* =====================  INIT  ===================== */
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initMap();
  setupEventListeners();
  setupAOIDrawing();
  setupLegend();
  buildPipelineStages();
  paintSlider();
  startClock();
  checkHardwareAcceleration();
  setupDigitizer();
});

window.addEventListener("load", () => {
  setTimeout(() => {
    const s = document.getElementById("splashScreen");
    if (s) s.classList.add("splash-hide");
  }, 1700);
});

function initMap() {
  map = L.map("map", {
    preferCanvas: true,
    zoomControl: false
  }).setView([19.073, 72.873], 16);
  L.control.zoom({ position: "bottomright" }).addTo(map);
  L.control.scale({ imperial: false, position: "bottomright" }).addTo(map);

  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics",
    maxZoom: 19
  }).addTo(map);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png", {
    subdomains: "abcd", maxZoom: 19
  }).addTo(map);

  const coordEl = document.getElementById("coordDisplay");
  const zoomEl = document.getElementById("zoomDisplay");
  map.on("mousemove", (e) => {
    coordEl.textContent =
      `${Math.abs(e.latlng.lat).toFixed(5)}° ${e.latlng.lat >= 0 ? "N" : "S"}, ` +
      `${Math.abs(e.latlng.lng).toFixed(5)}° ${e.latlng.lng >= 0 ? "E" : "W"}`;
  });
  map.on("zoomend", () => { zoomEl.textContent = map.getZoom(); });
  zoomEl.textContent = map.getZoom();
}

function startClock() {
  const el = document.getElementById("statusClock");
  const tick = () => {
    const d = new Date();
    const p = (n) => String(n).padStart(2, "0");
    el.textContent = `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} UTC`;
  };
  tick();
  setInterval(tick, 1000);
}

/* =====================  EVENT LISTENERS  ===================== */
function setupEventListeners() {
  document.getElementById("btnRunDemo").addEventListener("click", runDemoPipeline);
  document.getElementById("btnCurrentView").addEventListener("click", selectCurrentViewAOI);
  document.getElementById("btnExtractAOI").addEventListener("click", runAOIExtraction);
  document.getElementById("btnClearAOI").addEventListener("click", clearAOI);
  document.getElementById("btnThemeToggle").addEventListener("click", toggleTheme);

  const toggles = [
    ["chkBuildings", "buildings"], ["chkRoads", "roads"], ["chkTrees", "trees"],
    ["chkFarms", "farms"], ["chkWater", "water"]
  ];
  toggles.forEach(([chk, name]) => {
    document.getElementById(chk).addEventListener("change", (e) => {
      toggleLayer(name, e.target.checked);
      syncLegend();
    });
  });

  const rngConf = document.getElementById("rngConfidence");
  rngConf.addEventListener("input", (e) => {
    document.getElementById("lblConfVal").innerText = parseFloat(e.target.value).toFixed(2);
    paintSlider();
    applyConfidenceFilter(parseFloat(e.target.value));
  });

  document.getElementById("btnExportZip").addEventListener("click", () => {
    if (currentTaskId) {
      toast("info", "Preparing package", "Building your Shapefile export…");
      window.location.href = `/api/export/${currentTaskId}/zip`;
    }
  });

  document.getElementById("btnExportGeoJSON").addEventListener("click", () => {
    if (currentTaskId) {
      toast("info", "Preparing export", "Generating GeoJSON download…");
      window.location.href = `/api/export/${currentTaskId}/geojson`;
    }
  });

  document.getElementById("btnCloseInspector").addEventListener("click", hideInspector);

  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea") return;
    if (e.key === "Escape") {
      if (isDrawingBox) resetDrawMode();
      if (typeof activeDigitizeTool !== "undefined" && activeDigitizeTool) deactivateDigitizer();
      hideInspector();
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      if (typeof selectedFeature !== "undefined" && selectedFeature) {
        deleteSelectedFeature();
      }
    }
    if (e.key === "Enter" && typeof activeDigitizeTool !== "undefined" && activeDigitizeTool === "roads") {
      finishRoadDrawing();
    }
    if (e.key.toLowerCase() === "d") document.getElementById("btnDrawBox").click();
    if (e.key.toLowerCase() === "v") selectCurrentViewAOI();
    if (e.key.toLowerCase() === "t") toggleTheme();
  });
}

function paintSlider() {
  const el = document.getElementById("rngConfidence");
  const p = ((el.value - el.min) / (el.max - el.min)) * 100;
  el.style.setProperty("--fill", p + "%");
}

/* =====================  LEGEND  ===================== */
function setupLegend() {
  document.querySelectorAll(".legend-item").forEach((item) => {
    item.addEventListener("click", () => {
      const name = item.dataset.layer;
      if (!leafletLayers[name]) {
        toast("info", "Layer not loaded", "Run an extraction first, then toggle layers.");
        return;
      }
      const chk = document.getElementById("chk" + name.charAt(0).toUpperCase() + name.slice(1));
      chk.checked = !chk.checked;
      chk.dispatchEvent(new Event("change"));
    });
  });
}

function syncLegend() {
  document.querySelectorAll(".legend-item").forEach((item) => {
    const name = item.dataset.layer;
    const chk = document.getElementById("chk" + name.charAt(0).toUpperCase() + name.slice(1));
    item.classList.toggle("off", chk && !chk.checked);
  });
}

/* =====================  AOI DRAWING  ===================== */
function setupAOIDrawing() {
  const btnDraw = document.getElementById("btnDrawBox");

  btnDraw.addEventListener("click", () => {
    isDrawingBox = !isDrawingBox;
    if (isDrawingBox) {
      btnDraw.classList.add("btn-draw-active");
      btnDraw.innerHTML = `<i class="fa-solid fa-hand"></i> Click & Drag on Map`;
      map.getContainer().style.cursor = "crosshair";
      map.dragging.disable();
    } else {
      resetDrawMode();
    }
  });

  map.on("mousedown", (e) => {
    if (!isDrawingBox) return;
    drawStartLatLng = e.latlng;
    if (aoiRectangleLayer) { map.removeLayer(aoiRectangleLayer); aoiRectangleLayer = null; }
    aoiRectangleLayer = L.rectangle(L.latLngBounds(drawStartLatLng, drawStartLatLng), {
      color: "#22d3ee", weight: 2, className: "aoi-rect-drawing",
      fillColor: "#22d3ee", fillOpacity: 0.12
    }).addTo(map);
  });

  map.on("mousemove", (e) => {
    if (!isDrawingBox || !drawStartLatLng || !aoiRectangleLayer) return;
    aoiRectangleLayer.setBounds(L.latLngBounds(drawStartLatLng, e.latlng));
  });

  map.on("mouseup", (e) => {
    if (!isDrawingBox || !drawStartLatLng) return;
    const bounds = L.latLngBounds(drawStartLatLng, e.latlng);
    if (bounds.getNorthEast().distanceTo(bounds.getSouthWest()) > 20) setAOIBounds(bounds);
    drawStartLatLng = null;
    resetDrawMode();
  });
}

function resetDrawMode() {
  isDrawingBox = false;
  const btnDraw = document.getElementById("btnDrawBox");
  btnDraw.classList.remove("btn-draw-active");
  btnDraw.innerHTML = `<i class="fa-solid fa-vector-square"></i> Drag Box`;
  map.getContainer().style.cursor = "";
  map.dragging.enable();
}

function selectCurrentViewAOI() {
  setAOIBounds(map.getBounds());
}

function setAOIBounds(bounds) {
  if (aoiRectangleLayer) map.removeLayer(aoiRectangleLayer);
  aoiRectangleLayer = L.rectangle(bounds, {
    color: "#22d3ee", weight: 2.5, className: "aoi-rect-final",
    fillColor: "#22d3ee", fillOpacity: 0.10
  }).addTo(map);

  const sw = bounds.getSouthWest();
  const ne = bounds.getNorthEast();
  currentAOIBounds = {
    min_lon: Math.min(sw.lng, ne.lng), min_lat: Math.min(sw.lat, ne.lat),
    max_lon: Math.max(sw.lng, ne.lng), max_lat: Math.max(sw.lat, ne.lat)
  };

  const latDist = (currentAOIBounds.max_lat - currentAOIBounds.min_lat) * 111320;
  const lonDist = (currentAOIBounds.max_lon - currentAOIBounds.min_lon) * 111320 *
    Math.cos((currentAOIBounds.min_lat + currentAOIBounds.max_lat) * Math.PI / 360);
  const areaSqm = Math.abs(latDist * lonDist);
  const areaHa = (areaSqm / 10000.0).toFixed(2);

  document.getElementById("aoiAreaText").innerHTML =
    `<b>${areaHa} ha</b> selected · ready to extract`;
  document.getElementById("btnExtractAOI").disabled = false;
  document.getElementById("btnClearAOI").classList.remove("hidden");
  toast("info", "AOI selected", `${areaHa} hectares ready for extraction.`);
}

function clearAOI() {
  currentAOIBounds = null;
  if (aoiRectangleLayer) { map.removeLayer(aoiRectangleLayer); aoiRectangleLayer = null; }
  document.getElementById("aoiAreaText").innerHTML =
    `Click <b>Drag Box</b> and sweep an area on the map`;
  document.getElementById("btnExtractAOI").disabled = true;
  document.getElementById("btnClearAOI").classList.add("hidden");
}

/* =====================  PIPELINE OVERLAY  ===================== */
function buildPipelineStages() {
  const list = document.getElementById("pipelineStages");
  list.innerHTML = "";
  PIPELINE_STAGES.forEach((s) => {
    const row = document.createElement("div");
    row.className = "stage pending";
    row.innerHTML = `<i class="fa-solid ${s.icon}"></i><span>${s.label}</span>`;
    list.appendChild(row);
  });
}

let stageTimer = null;
function startPipeline() {
  buildPipelineStages();
  const rows = [...document.querySelectorAll("#pipelineStages .stage")];
  document.getElementById("processingOverlay").classList.remove("hidden");
  document.getElementById("mapWrapper").classList.add("scanning");

  let idx = 0;
  const setStage = (i) => {
    rows.forEach((r, j) => {
      r.className = "stage " + (j < i ? "done" : j === i ? "active" : "pending");
      r.querySelector("i").className =
        j < i ? "fa-solid fa-check" : `fa-solid ${PIPELINE_STAGES[j].icon}`;
    });
    document.getElementById("pipelineProgress").style.width =
      `${((i + 0.5) / PIPELINE_STAGES.length) * 100}%`;
    document.getElementById("pipelineSub").textContent =
      PIPELINE_STAGES[Math.min(i, PIPELINE_STAGES.length - 1)].label + "…";
  };
  setStage(0);

  stageTimer = setInterval(() => {
    if (idx < PIPELINE_STAGES.length - 1) { idx += 1; setStage(idx); }
  }, 800);
}

function finishPipeline(success = true) {
  clearInterval(stageTimer);
  const rows = [...document.querySelectorAll("#pipelineStages .stage")];
  if (success) {
    rows.forEach((r) => { r.className = "stage done"; r.querySelector("i").className = "fa-solid fa-check"; });
    document.getElementById("pipelineProgress").style.width = "100%";
    document.getElementById("pipelineSub").textContent = "Pipeline complete ✓";
  } else {
    const active = rows.find((r) => r.classList.contains("active"));
    if (active) active.className = "stage error";
    document.getElementById("pipelineSub").textContent = "Pipeline failed — see notification.";
  }
  setTimeout(() => {
    document.getElementById("processingOverlay").classList.add("hidden");
    document.getElementById("mapWrapper").classList.remove("scanning");
  }, success ? 700 : 1300);
}

/* =====================  TOASTS  ===================== */
function toast(type, title, msg) {
  const icons = { success: "fa-circle-check", error: "fa-circle-exclamation", info: "fa-circle-info" };
  const c = document.getElementById("toastContainer");
  const t = document.createElement("div");
  t.className = `toast ${type}`;
  t.innerHTML = `
    <i class="fa-solid ${icons[type] || icons.info}"></i>
    <div class="toast-body"><b>${title}</b><span>${msg}</span></div>
    <button class="toast-close">&times;</button>
    <div class="toast-bar"></div>`;
  c.appendChild(t);
  const close = () => { t.classList.add("leaving"); setTimeout(() => t.remove(), 300); };
  t.querySelector(".toast-close").onclick = close;
  setTimeout(close, 3800);
  while (c.children.length > 3) c.firstChild.remove();
}

/* =====================  ANIMATED METRICS  ===================== */
function animateMetric(el, target, { decimals = 0, suffix = "" } = {}) {
  const start = parseFloat(el.dataset.val || 0) || 0;
  const t0 = performance.now();
  const dur = 950;
  const ease = (p) => 1 - Math.pow(1 - p, 3);
  function frame(t) {
    const p = Math.min((t - t0) / dur, 1);
    const v = start + (target - start) * ease(p);
    el.textContent = v.toLocaleString(undefined, {
      minimumFractionDigits: decimals, maximumFractionDigits: decimals
    }) + suffix;
    if (p < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  el.dataset.val = target;
}

function updateSummaryMetrics(summary) {
  animateMetric(document.getElementById("metricBuildings"), +summary.building_count || 0);
  animateMetric(document.getElementById("metricRoads"), +summary.total_road_km || 0, { decimals: 2, suffix: " km" });
  animateMetric(document.getElementById("metricTrees"), +summary.tree_count || 0);
  animateMetric(document.getElementById("metricFarms"), +summary.farm_parcel_count || 0);
}

/* =====================  EXTRACTION RUNS  ===================== */
function hideEmptyHint() {
  const h = document.getElementById("emptyHint");
  if (h) h.classList.add("gone");
}

async function runAOIExtraction() {
  if (!currentAOIBounds) return;
  const btn = document.getElementById("btnExtractAOI");
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Extracting Features…`;
  startPipeline();

  try {
    const res = await fetch("/api/extract_aoi", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        min_lon: currentAOIBounds.min_lon, min_lat: currentAOIBounds.min_lat,
        max_lon: currentAOIBounds.max_lon, max_lat: currentAOIBounds.max_lat,
        zoom: 17
      })
    });
    const data = await res.json();

    if (data.status === "success") {
      finishPipeline(true);
      currentTaskId = data.task_id;
      updateSummaryMetrics(data.summary);
      await loadAllLayers(data.task_id);
      document.getElementById("btnExportZip").disabled = false;
      document.getElementById("btnExportGeoJSON").disabled = false;
      toast("success", "Extraction complete",
        `${data.summary.building_count} buildings · ${data.summary.tree_count} trees · ${data.summary.total_road_km} km roads`);
    } else {
      finishPipeline(false);
      toast("error", "Extraction failed", data.detail || "Unknown server error.");
    }
  } catch (err) {
    finishPipeline(false);
    toast("error", "Network error", err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-bolt"></i> Extract GIS Features`;
  }
}

async function runDemoPipeline() {
  const btn = document.getElementById("btnRunDemo");
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing Demo Scene…`;
  startPipeline();

  try {
    const res = await fetch("/api/demo", { method: "POST" });
    const data = await res.json();

    if (data.status === "success") {
      finishPipeline(true);
      currentTaskId = data.task_id;
      updateSummaryMetrics(data.summary);
      await loadAllLayers(data.task_id);
      document.getElementById("btnExportZip").disabled = false;
      document.getElementById("btnExportGeoJSON").disabled = false;

      const b = data.summary.bounds;
      map.fitBounds([[b[1], b[0]], [b[3], b[2]]], { padding: [45, 45] });
      toast("success", "Demo scene ready",
        `${data.summary.building_count} buildings · ${data.summary.tree_count} trees digitized`);
    }
  } catch (err) {
    finishPipeline(false);
    toast("error", "Pipeline failed", err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Run Synthetic Demo Scene`;
  }
}

/* =====================  LAYER LOADING / RENDERING  ===================== */
async function loadAllLayers(taskId) {
  for (const name of ["buildings", "roads", "trees", "farms", "water"]) {
    try {
      const res = await fetch(`/api/layers/${taskId}/${name}`);
      const geojson = await res.json();
      layersData[name] = geojson;
      renderLayer(name, geojson);
    } catch (e) {
      console.warn(`Could not load layer ${name}:`, e);
    }
  }
  applyConfidenceFilter(currentThreshold);
  syncLegend();
  hideEmptyHint();
}

function getLayerStyle(name, props, state = "normal") {
  const m = LAYER_META[name];
  const conf = props && props.confidence_score != null ? +props.confidence_score : null;
  const low = conf !== null && conf < currentThreshold;
  const hover = state === "hover";

  if (name === "trees") {
    return {
      radius: Math.max(4, Math.min(12, ((props && props.crown_diameter_m) || 2) * 2)),
      fillColor: low ? "#f87171" : m.color,
      color: low ? "#dc2626" : "#166534",
      weight: hover ? 3 : 1.5,
      opacity: 0.95,
      fillOpacity: low ? 0.9 : 0.75
    };
  }
  if (name === "roads") {
    return { color: low ? "#f87171" : m.color, weight: hover ? 6 : 3.5, opacity: 0.95 };
  }
  if (name === "farms") {
    return {
      color: m.color, weight: hover ? 3 : 1.5,
      fillColor: m.color, fillOpacity: 0.22,
      dashArray: low ? "2, 6" : "4, 4"
    };
  }
  if (name === "water") {
    return { color: m.color, weight: hover ? 3.5 : 2, fillColor: m.fill, fillOpacity: 0.62 };
  }
  return {
    color: low ? "#f87171" : m.color, weight: hover ? 3.5 : 2,
    fillColor: low ? "#f87171" : m.color, fillOpacity: low ? 0.75 : 0.55
  };
}

function attachHover(layer, name, props, feat) {
  layer.on("mouseover", () => {
    if (typeof selectedFeature !== "undefined" && selectedFeature && selectedFeature.leafletLayer === layer) return;
    layer.setStyle(getLayerStyle(name, props, "hover"));
    if (layer.bringToFront) layer.bringToFront();
  });
  layer.on("mouseout", () => {
    if (typeof selectedFeature !== "undefined" && selectedFeature && selectedFeature.leafletLayer === layer) return;
    layer.setStyle(getLayerStyle(name, props));
  });
  layer.on("click", (e) => {
    L.DomEvent.stopPropagation(e);
    selectFeature(name, props, feat, layer);
  });
}

function renderLayer(name, geojson) {
  if (leafletLayers[name]) map.removeLayer(leafletLayers[name]);

  let layerGroup = null;

  if (name === "trees") {
    layerGroup = L.geoJSON(geojson, {
      pointToLayer: (feature, latlng) =>
        L.circleMarker(latlng, getLayerStyle("trees", feature.properties)),
      onEachFeature: (feat, layer) => attachHover(layer, "trees", feat.properties, feat)
    });
  } else {
    layerGroup = L.geoJSON(geojson, {
      style: (feature) => getLayerStyle(name, feature.properties),
      onEachFeature: (feat, layer) => attachHover(layer, name, feat.properties, feat)
    });
  }

  leafletLayers[name] = layerGroup;
  layerGroup.addTo(map);
}

function toggleLayer(name, isVisible) {
  if (!leafletLayers[name]) return;
  if (isVisible) map.addLayer(leafletLayers[name]);
  else map.removeLayer(leafletLayers[name]);
}

/* =====================  QC FILTER  ===================== */
function applyConfidenceFilter(threshold) {
  currentThreshold = threshold;
  Object.keys(leafletLayers).forEach((name) => {
    if (!leafletLayers[name]) return;
    leafletLayers[name].eachLayer((layer) => {
      if (!layer.feature) return;
      layer.setStyle(getLayerStyle(name, layer.feature.properties));
    });
  });
  updateFlaggedCount(threshold);
}

function updateFlaggedCount(threshold) {
  let flagged = 0, total = 0;
  Object.values(layersData).forEach((gj) => {
    (gj.features || []).forEach((f) => {
      const c = f.properties && f.properties.confidence_score;
      if (typeof c === "number") { total += 1; if (c < threshold) flagged += 1; }
    });
  });
  const el = document.getElementById("qcFlagCount");
  el.innerHTML = total
    ? `<i class="fa-regular fa-flag"></i> <b>${flagged}</b> of ${total} features flagged for review`
    : `<i class="fa-regular fa-flag"></i> No features loaded`;
  el.classList.toggle("has-flags", flagged > 0);
}

/* =====================  INSPECTOR  ===================== */
function showInspector(layerName, props) {
  const meta = LAYER_META[layerName] || { color: "#22d3ee", icon: "fa-circle-info", label: "Feature" };
  const panel = document.getElementById("inspectorPanel");
  panel.style.setProperty("--ins-color", meta.color);

  document.getElementById("inspectorTitle").querySelector(".feat-title").textContent =
    meta.label;

  const container = document.getElementById("inspectorContent");
  container.innerHTML = "";

  const fmt = (key, val) => {
    if (key === "needs_review")
      return val ? `<span class="tag-review">NEEDS REVIEW</span>` : `<span class="tag-approved">APPROVED</span>`;
    if (key === "area_sqm") return `${val} m²`;
    if (key === "height_m") return `${val} m (nDSM)`;
    if (key === "crown_diameter_m") return `${val} m`;
    if (key === "length_m") return `${val} m`;
    if (key === "vertex_reduction_pct") return `${val}% reduced`;
    return val;
  };

  const addRow = (key, valHTML) => {
    const row = document.createElement("div");
    row.className = "attr-row";
    row.innerHTML = `
      <span class="attr-key">${key.replace(/_/g, " ")}</span>
      <span class="attr-leader"></span>
      <span class="attr-val">${valHTML}</span>`;
    container.appendChild(row);
  };

  for (const [key, val] of Object.entries(props)) {
    if (key === "confidence_score") continue;
    addRow(key, fmt(key, val));
  }

  // Gradient confidence bar with threshold marker
  if (props.confidence_score != null) {
    const c = +props.confidence_score;
    const ok = c >= currentThreshold;
    const block = document.createElement("div");
    block.className = "conf-block";
    block.innerHTML = `
      <div class="conf-head">
        <span>Model Confidence</span>
        <b style="color:${ok ? "var(--ok)" : "var(--warn)"}">${(c * 100).toFixed(1)}%</b>
      </div>
      <div class="conf-track">
        <div class="conf-fill" style="width:${(c * 100).toFixed(1)}%;
             background:${ok ? "linear-gradient(90deg,#22d3ee,#34d399)" : "linear-gradient(90deg,#f59e0b,#f87171)"}"></div>
        <div class="conf-threshold" style="left:${(currentThreshold * 100).toFixed(0)}%"></div>
      </div>`;
    container.appendChild(block);
  }

  panel.classList.remove("hidden");
}

let selectedFeature = null;

function selectFeature(layerName, props, feat, layer) {
  if (selectedFeature && selectedFeature.leafletLayer && selectedFeature.leafletLayer !== layer) {
    try {
      selectedFeature.leafletLayer.setStyle(getLayerStyle(selectedFeature.layerName, selectedFeature.feature.properties));
    } catch (e) {}
  }

  selectedFeature = { layerName, feature: feat, leafletLayer: layer };
  if (layer && layer.setStyle) {
    layer.setStyle({ color: "#22d3ee", weight: 4.5, dashArray: "5, 5" });
  }
  showInspector(layerName, props);
}

function hideInspector() {
  if (selectedFeature && selectedFeature.leafletLayer) {
    try {
      selectedFeature.leafletLayer.setStyle(getLayerStyle(selectedFeature.layerName, selectedFeature.feature.properties));
    } catch (e) {}
  }
  selectedFeature = null;
  document.getElementById("inspectorPanel").classList.add("hidden");
}

function deleteSelectedFeature() {
  if (!selectedFeature) return;
  const { layerName, feature, leafletLayer } = selectedFeature;

  // 1. Remove from Leaflet map
  if (leafletLayers[layerName] && leafletLayer) {
    leafletLayers[layerName].removeLayer(leafletLayer);
  }

  // 2. Remove from layersData
  if (layersData[layerName] && layersData[layerName].features) {
    const idx = layersData[layerName].features.indexOf(feature);
    if (idx !== -1) {
      layersData[layerName].features.splice(idx, 1);
    } else {
      layersData[layerName].features = layersData[layerName].features.filter(
        f => f !== feature && f.id !== feature.id
      );
    }
  }

  // 3. Update summary metrics
  if (layerName === "buildings") {
    latestSummary.building_count = Math.max(0, (latestSummary.building_count || 1) - 1);
    document.getElementById("cntBuildings").innerText = latestSummary.building_count;
  } else if (layerName === "trees") {
    latestSummary.tree_count = Math.max(0, (latestSummary.tree_count || 1) - 1);
    document.getElementById("cntTrees").innerText = latestSummary.tree_count;
  } else if (layerName === "roads") {
    let totalKm = 0;
    (layersData.roads.features || []).forEach(rf => {
      const coords = rf.geometry.coordinates;
      for (let i = 0; i < coords.length - 1; i++) {
        totalKm += getDistanceKm(coords[i], coords[i+1]);
      }
    });
    latestSummary.total_road_km = +totalKm.toFixed(2);
    latestSummary.road_segment_count = layersData.roads.features.length;
    document.getElementById("cntRoadKm").innerText = `${latestSummary.total_road_km} km`;
  } else if (layerName === "farms") {
    latestSummary.farm_parcel_count = Math.max(0, (latestSummary.farm_parcel_count || 1) - 1);
    document.getElementById("cntFarms").innerText = latestSummary.farm_parcel_count;
  }

  updateFlaggedCount(currentThreshold);
  hideInspector();
  toast("warn", "Feature Removed", `Deleted ${LAYER_META[layerName]?.label || layerName} from GIS dataset.`);
  syncLayersToServer();
}

function getDistanceKm(p1, p2) {
  const dLat = (p2[1] - p1[1]) * 111.32;
  const avgLat = ((p1[1] + p2[1]) / 2) * Math.PI / 180;
  const dLon = (p2[0] - p1[0]) * 111.32 * Math.cos(avgLat);
  return Math.sqrt(dLat * dLat + dLon * dLon);
}

let syncDebounceTimer = null;
function syncLayersToServer() {
  if (!currentTaskId) return;
  clearTimeout(syncDebounceTimer);
  syncDebounceTimer = setTimeout(async () => {
    try {
      const payload = {
        layers: {
          buildings: layersData.buildings?.features || [],
          roads: layersData.roads?.features || [],
          trees: layersData.trees?.features || [],
          farms: layersData.farms?.features || [],
          water: layersData.water?.features || []
        },
        summary: latestSummary
      };
      await fetch(`/api/layers/${currentTaskId}/sync`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
    } catch (e) {
      console.warn("Sync error:", e);
    }
  }, 300);
}

function addFeatureToLayer(layerName, feat) {
  if (!layersData[layerName]) {
    layersData[layerName] = { type: "FeatureCollection", features: [] };
  }
  layersData[layerName].features.push(feat);

  // Add to Leaflet layer group
  if (!leafletLayers[layerName]) {
    leafletLayers[layerName] = L.geoJSON(layersData[layerName], {
      style: (f) => getLayerStyle(layerName, f.properties),
      pointToLayer: (f, latlng) => L.circleMarker(latlng, getLayerStyle("trees", f.properties)),
      onEachFeature: (f, layer) => attachHover(layer, layerName, f.properties, f)
    }).addTo(map);
  } else {
    if (layerName === "trees") {
      const pt = L.circleMarker([feat.geometry.coordinates[1], feat.geometry.coordinates[0]], getLayerStyle("trees", feat.properties));
      attachHover(pt, "trees", feat.properties, feat);
      leafletLayers[layerName].addLayer(pt);
    } else {
      const l = L.GeoJSON.geometryToLayer(feat, {
        style: getLayerStyle(layerName, feat.properties)
      });
      attachHover(l, layerName, feat.properties, feat);
      leafletLayers[layerName].addLayer(l);
    }
  }

  // Update metrics
  if (layerName === "buildings") {
    latestSummary.building_count = (latestSummary.building_count || 0) + 1;
    document.getElementById("cntBuildings").innerText = latestSummary.building_count;
  } else if (layerName === "trees") {
    latestSummary.tree_count = (latestSummary.tree_count || 0) + 1;
    document.getElementById("cntTrees").innerText = latestSummary.tree_count;
  } else if (layerName === "roads") {
    const coords = feat.geometry.coordinates;
    let km = 0;
    for (let i = 0; i < coords.length - 1; i++) km += getDistanceKm(coords[i], coords[i+1]);
    latestSummary.total_road_km = +((latestSummary.total_road_km || 0) + km).toFixed(2);
    latestSummary.road_segment_count = (latestSummary.road_segment_count || 0) + 1;
    document.getElementById("cntRoadKm").innerText = `${latestSummary.total_road_km} km`;
  } else if (layerName === "farms") {
    latestSummary.farm_parcel_count = (latestSummary.farm_parcel_count || 0) + 1;
    document.getElementById("cntFarms").innerText = latestSummary.farm_parcel_count;
  }

  updateFlaggedCount(currentThreshold);
  hideEmptyHint();
  document.getElementById("btnExportZip").disabled = false;
  document.getElementById("btnExportGeoJSON").disabled = false;
  syncLayersToServer();
}

/* =====================  HITL DIGITIZER  ===================== */
let activeDigitizeTool = null;
let digitizeStartLatLng = null;
let digitizePreviewLayer = null;
let digitizeRoadPoints = [];
let digitizeRoadPolyline = null;

function setupDigitizer() {
  const tools = [
    { id: "toolBuilding", name: "buildings", hint: "Click & drag a rectangle over the roof (90° CAD snap)" },
    { id: "toolRoad", name: "roads", hint: "Click points along road centerline. Double-click or press Enter to finish" },
    { id: "toolTree", name: "trees", hint: "Click on tree canopy to place 3D tree marker" },
    { id: "toolWater", name: "water", hint: "Click & drag a box over water body" },
    { id: "toolFarm", name: "farms", hint: "Click & drag a box over agricultural farm parcel" }
  ];

  tools.forEach(({ id, name, hint }) => {
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.addEventListener("click", () => {
      if (activeDigitizeTool === name) {
        deactivateDigitizer();
      } else {
        activateDigitizer(name, hint);
      }
    });
  });

  document.getElementById("btnCancelDigitize")?.addEventListener("click", deactivateDigitizer);
  document.getElementById("btnDeleteFeature")?.addEventListener("click", deleteSelectedFeature);

  // Map mouse events for digitizer
  map.on("mousedown", onDigitizerMouseDown);
  map.on("mousemove", onDigitizerMouseMove);
  map.on("mouseup", onDigitizerMouseUp);
  map.on("click", onDigitizerClick);
  map.on("dblclick", onDigitizerDblClick);
}

function activateDigitizer(toolName, hint) {
  deactivateDigitizer();
  resetDrawMode();
  activeDigitizeTool = toolName;

  document.querySelectorAll(".digitize-toolbar .btn-tool").forEach(b => b.classList.remove("active"));
  const activeBtn = document.querySelector(`.digitize-toolbar [data-tool="${toolName}"]`);
  if (activeBtn) activeBtn.classList.add("active");

  const bar = document.getElementById("digitizerFloatingBar");
  const barText = document.getElementById("digitizerBarText");
  if (bar && barText) {
    barText.innerHTML = `<b>Digitizer Active:</b> ${hint}`;
    bar.classList.remove("hidden");
  }

  map.getContainer().style.cursor = "crosshair";
  toast("info", "Digitizer Active", hint);
}

function deactivateDigitizer() {
  activeDigitizeTool = null;
  digitizeStartLatLng = null;
  if (digitizePreviewLayer) {
    map.removeLayer(digitizePreviewLayer);
    digitizePreviewLayer = null;
  }
  if (digitizeRoadPolyline) {
    map.removeLayer(digitizeRoadPolyline);
    digitizeRoadPolyline = null;
  }
  digitizeRoadPoints = [];

  document.querySelectorAll(".digitize-toolbar .btn-tool").forEach(b => b.classList.remove("active"));
  document.getElementById("digitizerFloatingBar")?.classList.add("hidden");
  map.getContainer().style.cursor = "";
  map.dragging.enable();
}

function onDigitizerMouseDown(e) {
  if (!activeDigitizeTool) return;
  if (activeDigitizeTool === "trees" || activeDigitizeTool === "roads") return;

  digitizeStartLatLng = e.latlng;
  map.dragging.disable();
}

function onDigitizerMouseMove(e) {
  if (!activeDigitizeTool) return;

  if (digitizeStartLatLng && (activeDigitizeTool === "buildings" || activeDigitizeTool === "water" || activeDigitizeTool === "farms")) {
    const bounds = L.latLngBounds(digitizeStartLatLng, e.latlng);
    const color = LAYER_META[activeDigitizeTool]?.color || "#22d3ee";
    if (!digitizePreviewLayer) {
      digitizePreviewLayer = L.rectangle(bounds, {
        color: color,
        weight: 2,
        dashArray: "4, 4",
        fillColor: color,
        fillOpacity: 0.25
      }).addTo(map);
    } else {
      digitizePreviewLayer.setBounds(bounds);
    }
  }

  if (activeDigitizeTool === "roads" && digitizeRoadPoints.length > 0) {
    const pts = [...digitizeRoadPoints, e.latlng];
    if (!digitizeRoadPolyline) {
      digitizeRoadPolyline = L.polyline(pts, {
        color: "#facc15",
        weight: 3.5,
        dashArray: "5, 5"
      }).addTo(map);
    } else {
      digitizeRoadPolyline.setLatLngs(pts);
    }
  }
}

function onDigitizerMouseUp(e) {
  if (!activeDigitizeTool || !digitizeStartLatLng) return;
  if (activeDigitizeTool === "trees" || activeDigitizeTool === "roads") return;

  map.dragging.enable();
  const start = digitizeStartLatLng;
  const end = e.latlng;
  digitizeStartLatLng = null;

  if (digitizePreviewLayer) {
    map.removeLayer(digitizePreviewLayer);
    digitizePreviewLayer = null;
  }

  const s = Math.min(start.lat, end.lat);
  const n = Math.max(start.lat, end.lat);
  const w = Math.min(start.lng, end.lng);
  const eLon = Math.max(start.lng, end.lng);

  const dx = (eLon - w) * 111320 * Math.cos(((s + n) / 2) * Math.PI / 180);
  const dy = (n - s) * 111320;
  const areaSqm = Math.abs(dx * dy);

  if (areaSqm < 4.0) return;

  const tool = activeDigitizeTool;
  const fid = "usr_" + Date.now().toString(36);
  const coords = [[[w, s], [eLon, s], [eLon, n], [w, n], [w, s]]];

  let feat = null;
  if (tool === "buildings") {
    feat = {
      type: "Feature",
      id: fid,
      geometry: { type: "Polygon", coordinates: coords },
      properties: {
        feature_type: "building",
        building_id: fid,
        confidence_score: 1.0,
        area_sqm: Math.round(areaSqm),
        perimeter_m: Math.round(2 * (Math.abs(dx) + Math.abs(dy))),
        regularization_mode: "90° CAD Orthogonal",
        source: "Human-in-the-Loop Digitization",
        needs_review: false
      }
    };
    addFeatureToLayer("buildings", feat);
    toast("success", "Building Digitized", `90° CAD footprint (${Math.round(areaSqm)} m²) added.`);
  } else if (tool === "water") {
    feat = {
      type: "Feature",
      id: fid,
      geometry: { type: "Polygon", coordinates: coords },
      properties: {
        feature_type: "water_body",
        confidence_score: 1.0,
        area_sqm: Math.round(areaSqm),
        source: "Human-in-the-Loop Digitization",
        needs_review: false
      }
    };
    addFeatureToLayer("water", feat);
    toast("success", "Water Body Added", `Added ${Math.round(areaSqm)} m² water feature.`);
  } else if (tool === "farms") {
    const areaHa = +(areaSqm / 10000).toFixed(2);
    feat = {
      type: "Feature",
      id: fid,
      geometry: { type: "Polygon", coordinates: coords },
      properties: {
        feature_type: "farm_parcel",
        confidence_score: 1.0,
        area_ha: areaHa,
        source: "Human-in-the-Loop Digitization",
        needs_review: false
      }
    };
    addFeatureToLayer("farms", feat);
    toast("success", "Farm Parcel Added", `Added ${areaHa} ha agricultural plot.`);
  }
}

function onDigitizerClick(e) {
  if (!activeDigitizeTool) return;

  if (activeDigitizeTool === "trees") {
    const fid = "usr_tree_" + Date.now().toString(36);
    const feat = {
      type: "Feature",
      id: fid,
      geometry: { type: "Point", coordinates: [e.latlng.lng, e.latlng.lat] },
      properties: {
        feature_type: "tree",
        confidence_score: 1.0,
        crown_diameter_m: 3.5,
        height_m: 6.0,
        source: "Human-in-the-Loop Digitization",
        needs_review: false
      }
    };
    addFeatureToLayer("trees", feat);
    toast("success", "Tree Added", "Placed 3D tree crown (confidence 100%).");
  } else if (activeDigitizeTool === "roads") {
    digitizeRoadPoints.push(e.latlng);
    if (!digitizeRoadPolyline) {
      digitizeRoadPolyline = L.polyline(digitizeRoadPoints, {
        color: "#facc15",
        weight: 3.5
      }).addTo(map);
    } else {
      digitizeRoadPolyline.setLatLngs(digitizeRoadPoints);
    }
  }
}

function onDigitizerDblClick(e) {
  if (activeDigitizeTool === "roads" && digitizeRoadPoints.length >= 2) {
    L.DomEvent.stopPropagation(e);
    finishRoadDrawing();
  }
}

function finishRoadDrawing() {
  if (digitizeRoadPoints.length < 2) {
    if (digitizeRoadPolyline) {
      map.removeLayer(digitizeRoadPolyline);
      digitizeRoadPolyline = null;
    }
    digitizeRoadPoints = [];
    return;
  }

  let totalMeters = 0;
  for (let i = 0; i < digitizeRoadPoints.length - 1; i++) {
    totalMeters += digitizeRoadPoints[i].distanceTo(digitizeRoadPoints[i+1]);
  }

  const fid = "usr_road_" + Date.now().toString(36);
  const coords = digitizeRoadPoints.map(p => [p.lng, p.lat]);
  const feat = {
    type: "Feature",
    id: fid,
    geometry: { type: "LineString", coordinates: coords },
    properties: {
      feature_type: "road_centerline",
      confidence_score: 1.0,
      length_m: Math.round(totalMeters),
      source: "Human-in-the-Loop Digitization",
      needs_review: false
    }
  };

  if (digitizeRoadPolyline) {
    map.removeLayer(digitizeRoadPolyline);
    digitizeRoadPolyline = null;
  }
  digitizeRoadPoints = [];

  addFeatureToLayer("roads", feat);
  toast("success", "Road Centerline Added", `Added ${(totalMeters / 1000).toFixed(2)} km road network.`);
}

async function checkHardwareAcceleration() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const el = document.getElementById("gpuDisplay");
    if (data && data.hardware && el) {
      if (data.hardware.cuda_available) {
        const cleanName = data.hardware.gpu_name.replace("NVIDIA ", "").replace(" Laptop GPU", "");
        el.innerText = `${cleanName} (CUDA)`;
        el.title = `Hardware Acceleration: ${data.hardware.gpu_name} (CUDA Active)`;
      } else {
        el.innerText = `${data.hardware.gpu_name} Mode`;
      }
    }
  } catch (e) {
    console.debug("Hardware check:", e);
  }
}