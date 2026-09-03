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
});

window.addEventListener("load", () => {
  setTimeout(() => {
    const s = document.getElementById("splashScreen");
    if (s) s.classList.add("splash-hide");
  }, 1700);
});

function initMap() {
  map = L.map("map", { zoomControl: false }).setView([19.073, 72.873], 16);
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
      hideInspector();
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

function attachHover(layer, name, props) {
  layer.on("mouseover", () => {
    layer.setStyle(getLayerStyle(name, props, "hover"));
    if (layer.bringToFront) layer.bringToFront();
  });
  layer.on("mouseout", () => layer.setStyle(getLayerStyle(name, props)));
  layer.on("click", () => showInspector(name, props));
}

function renderLayer(name, geojson) {
  if (leafletLayers[name]) map.removeLayer(leafletLayers[name]);

  let layerGroup = null;

  if (name === "trees") {
    layerGroup = L.geoJSON(geojson, {
      pointToLayer: (feature, latlng) =>
        L.circleMarker(latlng, getLayerStyle("trees", feature.properties)),
      onEachFeature: (feat, layer) => attachHover(layer, "trees", feat.properties)
    });
  } else {
    layerGroup = L.geoJSON(geojson, {
      style: (feature) => getLayerStyle(name, feature.properties),
      onEachFeature: (feat, layer) => attachHover(layer, name, feat.properties)
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

function hideInspector() {
  document.getElementById("inspectorPanel").classList.add("hidden");
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