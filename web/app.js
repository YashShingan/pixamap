/**
 * PixaMap — Unified MapLibre GL Controller
 * Single-engine architecture: MapLibre GL JS handles both 2D Ortho Inspection and 3D Extruded CAD.
 * Zero dual-map overhead. Real-time smooth camera pitch & extrusion transitions.
 * Full pipeline overlay, radar splash, AI hardware telemetry, QC confidence filter,
 * drag-box AOI selection, external orthophoto raster overlay, and GIS exports.
 */

let map;
let currentTaskId = null;
let layersData = { buildings: null, roads: null, trees: null, farms: null, water: null };
let currentThreshold = 0.75;
let is3DMode = false;

let currentAOIBounds = null;
let isDrawingBox = false;
let drawStartLngLat = null;

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
  const btn = document.getElementById("btnThemeToggle");
  if (btn) {
    btn.innerHTML = theme === "dark" ? `<i class="fa-solid fa-sun"></i>` : `<i class="fa-solid fa-moon"></i>`;
  }
}

/* ---------- Layer metadata ---------- */
const LAYER_META = {
  buildings: { color: "#fb923c", fill: "#fb923c", label: "Building Footprint · 90° Snap", icon: "fa-building" },
  roads:     { color: "#facc15", fill: "#facc15", label: "Road Centerline Graph",          icon: "fa-road" },
  trees:     { color: "#4ade80", fill: "#4ade80", label: "Tree Inventory · 3D Height",     icon: "fa-tree" },
  farms:     { color: "#a3e635", fill: "#a3e635", label: "Farm Parcel Boundary",           icon: "fa-wheat-awn" },
  water:     { color: "#38bdf8", fill: "#0ea5e9", label: "Water Body",                     icon: "fa-water" },
};

const BUILDING_COLOR_RAMP = [
  'interpolate', ['linear'], ['coalesce', ['get', 'height_max'], 10.0],
  0, '#fdba74',    // Light orange for single-story structures
  9, '#fb923c',    // Primary vibrant orange
  18, '#f97316',   // Deep orange for mid-rises
  30, '#ea580c',   // Bold architectural orange
  45, '#c2410c'    // Dark burnt orange for towers
];

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
  setupUploadModal();
  setup3DModeAndOpacity();
});

window.addEventListener("load", () => {
  setTimeout(() => {
    const s = document.getElementById("splashScreen");
    if (s) s.classList.add("splash-hide");
  }, 1700);
});

/* =====================  MAPLIBRE INITIALIZATION  ===================== */
function initMap() {
  map = new maplibregl.Map({
    container: 'map',
    maxZoom: 18.5, // Capped to prevent tile unavailable errors
    minZoom: 2,
    style: {
      version: 8,
      sources: {
        'esri-satellite': {
          type: 'raster',
          tiles: [
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
          ],
          tileSize: 256,
          maxzoom: 18,
          attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics'
        },
        'carto-labels': {
          type: 'raster',
          tiles: [
            'https://a.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png',
            'https://b.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png',
            'https://c.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png',
            'https://d.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png'
          ],
          tileSize: 256,
          maxzoom: 19
        }
      },
      layers: [
        {
          id: 'satellite-base',
          type: 'raster',
          source: 'esri-satellite'
        }
      ]
    },
    center: [72.873, 19.073],
    zoom: 16,
    pitch: 0,
    bearing: 0
  });

  map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'bottom-right');
  map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');

  const coordEl = document.getElementById("coordDisplay");
  const zoomEl = document.getElementById("zoomDisplay");

  map.on("mousemove", (e) => {
    if (coordEl && e.lngLat) {
      coordEl.textContent =
        `${Math.abs(e.lngLat.lat).toFixed(5)}° ${e.lngLat.lat >= 0 ? "N" : "S"}, ` +
        `${Math.abs(e.lngLat.lng).toFixed(5)}° ${e.lngLat.lng >= 0 ? "E" : "W"}`;
    }
  });

  map.on("zoom", () => {
    if (zoomEl) zoomEl.textContent = Math.round(map.getZoom());
  });

  map.on("load", () => {
    initGISLayers();
    initAOILayers();
    // Add carto labels on top
    if (!map.getLayer('carto-labels-layer')) {
      map.addLayer({
        id: 'carto-labels-layer',
        type: 'raster',
        source: 'carto-labels'
      });
    }
  });
}

function initAOILayers() {
  if (map.getSource('aoi-src')) return;
  map.addSource('aoi-src', {
    type: 'geojson',
    data: { type: 'FeatureCollection', features: [] }
  });

  map.addLayer({
    id: 'aoi-fill',
    type: 'fill',
    source: 'aoi-src',
    paint: {
      'fill-color': '#22d3ee',
      'fill-opacity': 0.12
    }
  });

  map.addLayer({
    id: 'aoi-stroke',
    type: 'line',
    source: 'aoi-src',
    paint: {
      'line-color': '#22d3ee',
      'line-width': 2.5,
      'line-dasharray': [3, 2]
    }
  });
}

function initGISLayers() {
  const emptyFC = { type: 'FeatureCollection', features: [] };

  // 1. Farms
  if (!map.getSource('farms-src')) {
    map.addSource('farms-src', { type: 'geojson', data: emptyFC });
    map.addLayer({
      id: 'farms-fill',
      type: 'fill',
      source: 'farms-src',
      paint: {
        'fill-color': '#a3e635',
        'fill-opacity': 0.28
      }
    });
    map.addLayer({
      id: 'farms-outline',
      type: 'line',
      source: 'farms-src',
      paint: {
        'line-color': '#65a30d',
        'line-width': 2.0,
        'line-dasharray': [3, 2]
      }
    });
  }

  // 2. Water
  if (!map.getSource('water-src')) {
    map.addSource('water-src', { type: 'geojson', data: emptyFC });
    map.addLayer({
      id: 'water-fill',
      type: 'fill',
      source: 'water-src',
      filter: ['==', '$type', 'Polygon'],
      paint: {
        'fill-color': '#38bdf8',
        'fill-opacity': 0.72
      }
    });
    map.addLayer({
      id: 'water-line',
      type: 'line',
      source: 'water-src',
      filter: ['==', '$type', 'LineString'],
      layout: { 'line-cap': 'round', 'line-join': 'round' },
      paint: {
        'line-color': '#0284c7',
        'line-width': ['interpolate', ['linear'], ['zoom'], 14, 4.0, 18, 9.0],
        'line-opacity': 0.92
      }
    });
    map.addLayer({
      id: 'water-outline',
      type: 'line',
      source: 'water-src',
      filter: ['==', '$type', 'Polygon'],
      paint: {
        'line-color': '#0284c7',
        'line-width': 2.0
      }
    });
  }

  // 3. Roads
  if (!map.getSource('roads-src')) {
    map.addSource('roads-src', { type: 'geojson', data: emptyFC });
    map.addLayer({
      id: 'roads-casing',
      type: 'line',
      source: 'roads-src',
      layout: { 'line-cap': 'round', 'line-join': 'round' },
      paint: {
        'line-color': '#78350f',
        'line-width': ['interpolate', ['linear'], ['zoom'], 14, 4.5, 18, 9.5],
        'line-opacity': 0.85
      }
    });
    map.addLayer({
      id: 'roads-line',
      type: 'line',
      source: 'roads-src',
      layout: { 'line-cap': 'round', 'line-join': 'round' },
      paint: {
        'line-color': '#facc15',
        'line-width': ['interpolate', ['linear'], ['zoom'], 14, 2.5, 18, 6.0],
        'line-opacity': 0.98
      }
    });
  }

  // 4. Buildings (Fill-Extrusion handles both 2D flat and 3D heights)
  if (!map.getSource('buildings-src')) {
    map.addSource('buildings-src', { type: 'geojson', data: emptyFC });
    map.addLayer({
      id: 'buildings-layer',
      type: 'fill-extrusion',
      source: 'buildings-src',
      paint: {
        'fill-extrusion-color': BUILDING_COLOR_RAMP,
        'fill-extrusion-height': 0, // Flat in 2D mode, dynamic in 3D mode
        'fill-extrusion-height-transition': { duration: 1000, delay: 0 },
        'fill-extrusion-base': 0,
        'fill-extrusion-opacity': 0.92,
        'fill-extrusion-opacity-transition': { duration: 400 }
      }
    });
  }

  // 5. Trees (Fill-Extrusion canopy cylinders)
  if (!map.getSource('trees-src')) {
    map.addSource('trees-src', { type: 'geojson', data: emptyFC });
    map.addLayer({
      id: 'trees-layer',
      type: 'fill-extrusion',
      source: 'trees-src',
      paint: {
        'fill-extrusion-color': '#4ade80',
        'fill-extrusion-height': 0,
        'fill-extrusion-height-transition': { duration: 1000, delay: 0 },
        'fill-extrusion-base': 0,
        'fill-extrusion-opacity': 0.88
      }
    });
  }

  // Feature inspector clicks & hover pointers
  const inspectTargets = [
    { id: 'buildings-layer', name: 'buildings' },
    { id: 'roads-line',      name: 'roads' },
    { id: 'trees-layer',      name: 'trees' },
    { id: 'farms-fill',       name: 'farms' },
    { id: 'water-fill',       name: 'water' },
    { id: 'water-line',       name: 'water' }
  ];

  inspectTargets.forEach(({ id, name }) => {
    map.on('click', id, (e) => {
      if (e.features && e.features.length) {
        showInspector(name, e.features[0].properties);
      }
    });
    map.on('mouseenter', id, () => {
      if (!isDrawingBox) map.getCanvas().style.cursor = 'pointer';
    });
    map.on('mouseleave', id, () => {
      if (!isDrawingBox) map.getCanvas().style.cursor = '';
    });
  });
}

function startClock() {
  const el = document.getElementById("statusClock");
  const tick = () => {
    const d = new Date();
    const p = (n) => String(n).padStart(2, "0");
    if (el) el.textContent = `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} UTC`;
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
  if (el) {
    const p = ((el.value - el.min) / (el.max - el.min)) * 100;
    el.style.setProperty("--fill", p + "%");
  }
}

/* =====================  LEGEND  ===================== */
function setupLegend() {
  document.querySelectorAll(".legend-item").forEach((item) => {
    item.addEventListener("click", () => {
      const name = item.dataset.layer;
      const chk = document.getElementById("chk" + name.charAt(0).toUpperCase() + name.slice(1));
      if (chk) {
        chk.checked = !chk.checked;
        chk.dispatchEvent(new Event("change"));
      }
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
    // If currently in 3D mode, smoothly ease to 2D view for unobstructed drag box sweeping
    if (is3DMode) {
      const btn2D = document.getElementById("btnMode2D");
      if (btn2D) btn2D.click();
      toast("info", "Switched to 2D View", "Click and drag to sweep your Area of Interest.");
    }

    isDrawingBox = !isDrawingBox;
    if (isDrawingBox) {
      btnDraw.classList.add("btn-draw-active");
      btnDraw.innerHTML = `<i class="fa-solid fa-hand"></i> Click & Drag on Map`;
      map.getCanvas().style.cursor = "crosshair";
      map.dragPan.disable();
    } else {
      resetDrawMode();
    }
  });

  map.on("mousedown", (e) => {
    if (!isDrawingBox) return;
    drawStartLngLat = e.lngLat;
  });

  map.on("mousemove", (e) => {
    if (!isDrawingBox || !drawStartLngLat) return;

    const minX = Math.min(drawStartLngLat.lng, e.lngLat.lng);
    const maxX = Math.max(drawStartLngLat.lng, e.lngLat.lng);
    const minY = Math.min(drawStartLngLat.lat, e.lngLat.lat);
    const maxY = Math.max(drawStartLngLat.lat, e.lngLat.lat);

    const rectGeoJSON = {
      type: 'FeatureCollection',
      features: [{
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [[[minX, minY], [maxX, minY], [maxX, maxY], [minX, maxY], [minX, minY]]]
        }
      }]
    };
    const src = map.getSource('aoi-src');
    if (src) src.setData(rectGeoJSON);
  });

  map.on("mouseup", (e) => {
    if (!isDrawingBox || !drawStartLngLat) return;

    const minX = Math.min(drawStartLngLat.lng, e.lngLat.lng);
    const maxX = Math.max(drawStartLngLat.lng, e.lngLat.lng);
    const minY = Math.min(drawStartLngLat.lat, e.lngLat.lat);
    const maxY = Math.max(drawStartLngLat.lat, e.lngLat.lat);

    const latDist = (maxY - minY) * 111320;
    const lonDist = (maxX - minX) * 111320 * Math.cos((minY + maxY) * Math.PI / 360);
    const areaSqm = Math.abs(latDist * lonDist);

    if (areaSqm > 100) {
      currentAOIBounds = { min_lon: minX, min_lat: minY, max_lon: maxX, max_lat: maxY };
      const areaHa = (areaSqm / 10000.0).toFixed(2);
      document.getElementById("aoiAreaText").innerHTML = `<b>${areaHa} ha</b> selected · ready to extract`;
      document.getElementById("btnExtractAOI").disabled = false;
      document.getElementById("btnClearAOI").classList.remove("hidden");
      toast("info", "AOI selected", `${areaHa} hectares ready for extraction.`);
    }

    drawStartLngLat = null;
    resetDrawMode();
  });
}

function resetDrawMode() {
  isDrawingBox = false;
  const btnDraw = document.getElementById("btnDrawBox");
  if (btnDraw) {
    btnDraw.classList.remove("btn-draw-active");
    btnDraw.innerHTML = `<i class="fa-solid fa-vector-square"></i> Drag Box`;
  }
  map.getCanvas().style.cursor = "";
  map.dragPan.enable();
}

function selectCurrentViewAOI() {
  const b = map.getBounds();
  const minX = b.getWest();
  const maxX = b.getEast();
  const minY = b.getSouth();
  const maxY = b.getNorth();

  currentAOIBounds = { min_lon: minX, min_lat: minY, max_lon: maxX, max_lat: maxY };

  const rectGeoJSON = {
    type: 'FeatureCollection',
    features: [{
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [[[minX, minY], [maxX, minY], [maxX, maxY], [minX, maxY], [minX, minY]]]
      }
    }]
  };
  const src = map.getSource('aoi-src');
  if (src) src.setData(rectGeoJSON);

  const latDist = (maxY - minY) * 111320;
  const lonDist = (maxX - minX) * 111320 * Math.cos((minY + maxY) * Math.PI / 360);
  const areaSqm = Math.abs(latDist * lonDist);
  const areaHa = (areaSqm / 10000.0).toFixed(2);

  document.getElementById("aoiAreaText").innerHTML = `<b>${areaHa} ha</b> selected · ready to extract`;
  document.getElementById("btnExtractAOI").disabled = false;
  document.getElementById("btnClearAOI").classList.remove("hidden");
  toast("info", "AOI selected", `${areaHa} hectares ready for extraction.`);
}

function clearAOI() {
  currentAOIBounds = null;
  const src = map.getSource('aoi-src');
  if (src) src.setData({ type: 'FeatureCollection', features: [] });
  document.getElementById("aoiAreaText").innerHTML = `Click <b>Drag Box</b> and sweep an area on the map`;
  document.getElementById("btnExtractAOI").disabled = true;
  document.getElementById("btnClearAOI").classList.add("hidden");
}

/* =====================  PIPELINE PROGRESS  ===================== */
function buildPipelineStages() {
  const list = document.getElementById("pipelineStages");
  if (!list) return;
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
  if (!el) return;
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
  if (!summary) return;
  animateMetric(document.getElementById("metricBuildings"), +summary.building_count || 0);
  animateMetric(document.getElementById("metricRoads"), +summary.total_road_km || 0, { decimals: 2, suffix: " km" });
  animateMetric(document.getElementById("metricTrees"), +summary.tree_count || 0);
  animateMetric(document.getElementById("metricFarms"), +summary.farm_parcel_count || 0);
}

function hideEmptyHint() {
  const h = document.getElementById("emptyHint");
  if (h) h.classList.add("gone");
}

/* =====================  EXTRACTION RUNS  ===================== */
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

      if (currentAOIBounds) {
        map.fitBounds([
          [currentAOIBounds.min_lon, currentAOIBounds.min_lat],
          [currentAOIBounds.max_lon, currentAOIBounds.max_lat]
        ], { padding: 40 });
      }

      document.getElementById("btnExportZip").disabled = false;
      document.getElementById("btnExportGeoJSON").disabled = false;

      toast("success", "Extraction complete",
        `${data.summary.building_count} buildings · ${data.summary.total_road_km} km roads digitized`);
    } else {
      throw new Error(data.detail || "Extraction failed");
    }
  } catch (err) {
    finishPipeline(false);
    toast("error", "Extraction failed", err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-bolt"></i> Extract GIS Features`;
  }
}

async function runDemoPipeline() {
  const btn = document.getElementById("btnRunDemo");
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Synthesizing Scene…`;
  startPipeline();

  try {
    const res = await fetch("/api/demo");
    const data = await res.json();

    if (data.status === "success") {
      finishPipeline(true);
      currentTaskId = data.task_id;
      updateSummaryMetrics(data.summary);
      await loadAllLayers(data.task_id);

      if (data.bounds && data.bounds.length === 4) {
        map.fitBounds([
          [data.bounds[0], data.bounds[1]],
          [data.bounds[2], data.bounds[3]]
        ], { padding: 40 });
      }

      document.getElementById("btnExportZip").disabled = false;
      document.getElementById("btnExportGeoJSON").disabled = false;

      toast("success", "Demo loaded",
        `${data.summary.building_count} buildings · ${data.summary.total_road_km} km roads digitized`);
    } else {
      throw new Error(data.detail || "Demo failed");
    }
  } catch (err) {
    finishPipeline(false);
    toast("error", "Demo error", err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Run Synthetic Demo Scene`;
  }
}

/* =====================  LAYER LOADING & SYNC  ===================== */
function convertTreesToPolygons(treesGeoJSON) {
  if (!treesGeoJSON || !treesGeoJSON.features) return { type: 'FeatureCollection', features: [] };
  return {
    type: 'FeatureCollection',
    features: treesGeoJSON.features.map((f, idx) => {
      const coords = f.geometry.coordinates;
      const lon = coords[0], lat = coords[1];
      const r = Math.max(0.000015, ((f.properties.crown_diameter_m || 3.0) / 2.0) * 0.000009);
      const polyCoords = [];
      const steps = 10;
      for (let s = 0; s < steps; s++) {
        const angle = (s / steps) * 2 * Math.PI;
        polyCoords.push([
          lon + r * Math.cos(angle) / Math.cos(lat * Math.PI / 180),
          lat + r * Math.sin(angle)
        ]);
      }
      polyCoords.push(polyCoords[0]);
      return {
        type: 'Feature',
        id: f.id || idx + 1,
        geometry: { type: 'Polygon', coordinates: [polyCoords] },
        properties: {
          ...f.properties,
          tree_height: f.properties.height_m || 8.0
        }
      };
    })
  };
}

async function loadAllLayers(taskId) {
  for (const name of ["buildings", "roads", "trees", "farms", "water"]) {
    try {
      const res = await fetch(`/api/layers/${taskId}/${name}`);
      const geojson = await res.json();
      layersData[name] = geojson;

      const src = map.getSource(`${name}-src`);
      if (src) {
        if (name === "trees") {
          src.setData(convertTreesToPolygons(geojson));
        } else {
          src.setData(geojson);
        }
      }
    } catch (e) {
      console.warn(`Could not load layer ${name}:`, e);
    }
  }

  // Update building extrusion height according to active 2D/3D mode
  updateExtrusionHeights();
  applyConfidenceFilter(currentThreshold);
  syncLegend();
  hideEmptyHint();
}

function updateExtrusionHeights() {
  if (!map.getLayer('buildings-layer')) return;

  if (is3DMode) {
    map.setPaintProperty('buildings-layer', 'fill-extrusion-height', ['coalesce', ['get', 'height_max'], 10.0]);
    if (map.getLayer('trees-layer')) {
      map.setPaintProperty('trees-layer', 'fill-extrusion-height', ['coalesce', ['get', 'tree_height'], 8.0]);
    }
  } else {
    map.setPaintProperty('buildings-layer', 'fill-extrusion-height', 0);
    if (map.getLayer('trees-layer')) {
      map.setPaintProperty('trees-layer', 'fill-extrusion-height', 0);
    }
  }
}

function toggleLayer(name, isVisible) {
  const val = isVisible ? 'visible' : 'none';
  const layerMap = {
    buildings: ['buildings-layer'],
    roads:     ['roads-line', 'roads-casing'],
    trees:     ['trees-layer'],
    farms:     ['farms-fill', 'farms-outline'],
    water:     ['water-fill', 'water-line', 'water-outline']
  };

  (layerMap[name] || []).forEach(lyrId => {
    if (map.getLayer(lyrId)) {
      map.setLayoutProperty(lyrId, 'visibility', val);
    }
  });
}

/* =====================  QC CONFIDENCE FILTER  ===================== */
function applyConfidenceFilter(threshold) {
  currentThreshold = threshold;

  // 1. Buildings
  if (map.getLayer('buildings-layer')) {
    map.setPaintProperty('buildings-layer', 'fill-extrusion-color', [
      'case',
      ['<', ['coalesce', ['get', 'confidence_score'], 1.0], threshold],
      '#f87171',
      BUILDING_COLOR_RAMP
    ]);
  }

  // 2. Roads
  if (map.getLayer('roads-line')) {
    map.setPaintProperty('roads-line', 'line-color', [
      'case',
      ['<', ['coalesce', ['get', 'confidence_score'], 1.0], threshold],
      '#f87171',
      '#facc15'
    ]);
  }

  // 3. Trees
  if (map.getLayer('trees-layer')) {
    map.setPaintProperty('trees-layer', 'fill-extrusion-color', [
      'case',
      ['<', ['coalesce', ['get', 'confidence_score'], 1.0], threshold],
      '#f87171',
      '#4ade80'
    ]);
  }

  // 4. Farms
  if (map.getLayer('farms-fill')) {
    map.setPaintProperty('farms-fill', 'fill-color', [
      'case',
      ['<', ['coalesce', ['get', 'confidence_score'], 1.0], threshold],
      '#f87171',
      '#a3e635'
    ]);
  }

  // 5. Water
  if (map.getLayer('water-fill')) {
    map.setPaintProperty('water-fill', 'fill-color', [
      'case',
      ['<', ['coalesce', ['get', 'confidence_score'], 1.0], threshold],
      '#f87171',
      '#38bdf8'
    ]);
  }

  updateFlaggedCount(threshold);
}

function updateFlaggedCount(threshold) {
  let flagged = 0, total = 0;
  Object.values(layersData).forEach((gj) => {
    if (gj && gj.features) {
      gj.features.forEach((f) => {
        const c = f.properties && f.properties.confidence_score;
        if (typeof c === "number") { total += 1; if (c < threshold) flagged += 1; }
      });
    }
  });
  const el = document.getElementById("qcFlagCount");
  if (el) {
    el.innerHTML = total
      ? `<i class="fa-regular fa-flag"></i> <b>${flagged}</b> of ${total} features flagged for review`
      : `<i class="fa-regular fa-flag"></i> No features loaded`;
    el.classList.toggle("has-flags", flagged > 0);
  }
}

/* =====================  2D ORTHO vs 3D EXTRUDED CAD CONTROLS  ===================== */
function setup3DModeAndOpacity() {
  const btn2D = document.getElementById("btnMode2D");
  const btn3D = document.getElementById("btnMode3D");
  const rngOpacity = document.getElementById("rngOrthoOpacity");
  const lblOpacity = document.getElementById("lblOrthoOpacity");

  if (rngOpacity) {
    rngOpacity.addEventListener("input", (e) => {
      const val = parseInt(e.target.value);
      if (lblOpacity) lblOpacity.textContent = `${val}%`;
      if (map.getLayer('uploaded-ortho-layer')) {
        map.setPaintProperty('uploaded-ortho-layer', 'raster-opacity', val / 100.0);
      }
    });
  }

  if (btn2D && btn3D) {
    btn2D.addEventListener("click", () => {
      if (!is3DMode) return;
      is3DMode = false;
      btn2D.classList.add("active");
      btn3D.classList.remove("active");

      // Smooth camera transition to top-down 2D orthophoto inspection
      map.easeTo({
        pitch: 0,
        bearing: 0,
        duration: 900
      });

      updateExtrusionHeights();
    });

    btn3D.addEventListener("click", () => {
      if (is3DMode) return;
      is3DMode = true;
      btn3D.classList.add("active");
      btn2D.classList.remove("active");

      // Smooth camera transition to 3D perspective pitch
      map.easeTo({
        pitch: 58,
        bearing: -20,
        duration: 1000
      });

      updateExtrusionHeights();
    });
  }
}

/* =====================  EXTERNAL ORTHOPHOTO UPLOAD  ===================== */
function setupUploadModal() {
  const modal = document.getElementById("uploadModal");
  const btnOpen = document.getElementById("btnOpenUploadModal");
  const btnClose = document.getElementById("btnCloseUploadModal");
  const btnCancel = document.getElementById("btnCancelUpload");
  const form = document.getElementById("uploadOrthoForm");

  const orthoInput = document.getElementById("orthoFileInput");
  const orthoDrop = document.getElementById("orthoDropzone");
  const orthoSelected = document.getElementById("orthoSelectedFile");

  const dsmInput = document.getElementById("dsmFileInput");
  const dsmDrop = document.getElementById("dsmDropzone");
  const dsmSelected = document.getElementById("dsmSelectedFile");

  if (!modal || !btnOpen || !form) return;

  btnOpen.addEventListener("click", () => modal.classList.remove("hidden"));

  const closeModal = () => modal.classList.add("hidden");
  if (btnClose) btnClose.addEventListener("click", closeModal);
  if (btnCancel) btnCancel.addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });

  const btnBrowseOrtho = document.getElementById("btnBrowseOrtho");
  if (btnBrowseOrtho) {
    btnBrowseOrtho.addEventListener("click", (e) => {
      e.stopPropagation();
      orthoInput.click();
    });
  }
  orthoDrop.addEventListener("click", (e) => {
    if (e.target !== btnBrowseOrtho) orthoInput.click();
  });

  const btnBrowseDSM = document.getElementById("btnBrowseDSM");
  if (btnBrowseDSM) {
    btnBrowseDSM.addEventListener("click", (e) => {
      e.stopPropagation();
      dsmInput.click();
    });
  }
  dsmDrop.addEventListener("click", (e) => {
    if (e.target !== btnBrowseDSM) dsmInput.click();
  });

  ["dragenter", "dragover"].forEach((evt) => {
    orthoDrop.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); orthoDrop.classList.add("dragover"); });
    dsmDrop.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); dsmDrop.classList.add("dragover"); });
  });

  ["dragleave", "drop"].forEach((evt) => {
    orthoDrop.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); orthoDrop.classList.remove("dragover"); });
    dsmDrop.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); dsmDrop.classList.remove("dragover"); });
  });

  orthoDrop.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      orthoInput.files = e.dataTransfer.files;
      updateFileInfo(orthoInput.files[0], orthoSelected);
    }
  });
  orthoInput.addEventListener("change", () => {
    if (orthoInput.files.length) updateFileInfo(orthoInput.files[0], orthoSelected);
  });

  dsmDrop.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      dsmInput.files = e.dataTransfer.files;
      updateFileInfo(dsmInput.files[0], dsmSelected);
    }
  });
  dsmInput.addEventListener("change", () => {
    if (dsmInput.files.length) updateFileInfo(dsmInput.files[0], dsmSelected);
  });

  function updateFileInfo(file, targetEl) {
    if (!file) return;
    const mb = (file.size / (1024 * 1024)).toFixed(2);
    targetEl.innerHTML = `<span><i class="fa-solid fa-file-circle-check"></i> ${file.name} (${mb} MB)</span>
      <button type="button" class="btn-icon" style="color:var(--err); font-size:16px">&times;</button>`;
    targetEl.classList.remove("hidden");
    const removeBtn = targetEl.querySelector("button");
    if (removeBtn) {
      removeBtn.onclick = (e) => {
        e.stopPropagation();
        targetEl.classList.add("hidden");
        targetEl.innerHTML = "";
      };
    }
  }

  const btnAutoFill = document.getElementById("btnAutoFillBounds");
  if (btnAutoFill) {
    btnAutoFill.addEventListener("click", () => {
      const b = map.getBounds();
      document.getElementById("inputMinLon").value = b.getWest().toFixed(5);
      document.getElementById("inputMinLat").value = b.getSouth().toFixed(5);
      document.getElementById("inputMaxLon").value = b.getEast().toFixed(5);
      document.getElementById("inputMaxLat").value = b.getNorth().toFixed(5);
      toast("info", "Coordinates filled", "Captured current map viewport bounds.");
    });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!orthoInput.files || !orthoInput.files.length) {
      toast("error", "Missing file", "Please select an orthophoto file to upload.");
      return;
    }

    closeModal();
    startPipeline();
    const btnSubmit = document.getElementById("btnSubmitUpload");
    if (btnSubmit) btnSubmit.disabled = true;

    try {
      const formData = new FormData();
      formData.append("ortho_file", orthoInput.files[0]);
      if (dsmInput.files && dsmInput.files.length) {
        formData.append("dsm_file", dsmInput.files[0]);
      }

      const minLon = document.getElementById("inputMinLon").value;
      const minLat = document.getElementById("inputMinLat").value;
      const maxLon = document.getElementById("inputMaxLon").value;
      const maxLat = document.getElementById("inputMaxLat").value;
      if (minLon && minLat && maxLon && maxLat) {
        formData.append("min_lon", minLon);
        formData.append("min_lat", minLat);
        formData.append("max_lon", maxLon);
        formData.append("max_lat", maxLat);
      }

      const gsd = document.getElementById("inputGSD").value;
      if (gsd && !isNaN(parseFloat(gsd))) {
        formData.append("pixel_size_meters", parseFloat(gsd));
      }

      const res = await fetch("/api/upload_ortho", {
        method: "POST",
        body: formData
      });
      const data = await res.json();

      let finalTask = data;
      if (res.status === 202 || data.status === "processing") {
        finalTask = await pollTaskUntilComplete(data.task_id);
      }

      if (finalTask && (finalTask.status === "completed" || finalTask.status === "success")) {
        finishPipeline(true);
        currentTaskId = finalTask.task_id;
        updateSummaryMetrics(finalTask.summary);

        // Overlay orthophoto on single MapLibre canvas
        if (finalTask.preview_url && finalTask.bounds && finalTask.bounds.length === 4) {
          const b = finalTask.bounds; // [min_lon, min_lat, max_lon, max_lat]
          if (map.getLayer('uploaded-ortho-layer')) map.removeLayer('uploaded-ortho-layer');
          if (map.getSource('uploaded-ortho-src')) map.removeSource('uploaded-ortho-src');

          map.addSource('uploaded-ortho-src', {
            type: 'image',
            url: finalTask.preview_url,
            coordinates: [
              [b[0], b[3]], // top-left
              [b[2], b[3]], // top-right
              [b[2], b[1]], // bottom-right
              [b[0], b[1]]  // bottom-left
            ]
          });

          const beforeLyr = map.getLayer('farms-fill') ? 'farms-fill' : 'carto-labels-layer';
          map.addLayer({
            id: 'uploaded-ortho-layer',
            type: 'raster',
            source: 'uploaded-ortho-src',
            paint: {
              'raster-opacity': 0.90,
              'raster-opacity-transition': { duration: 300 }
            }
          }, beforeLyr);

          map.fitBounds([
            [b[0], b[1]],
            [b[2], b[3]]
          ], { padding: 40 });

          const opControl = document.getElementById("orthoOpacityControl");
          if (opControl) opControl.classList.remove("hidden");
        }

        await loadAllLayers(finalTask.task_id);

        document.getElementById("btnExportZip").disabled = false;
        document.getElementById("btnExportGeoJSON").disabled = false;

        toast("success", "Orthophoto mapped",
          `${finalTask.summary.building_count} buildings · ${finalTask.summary.total_road_km} km roads digitized`);
      } else {
        throw new Error((finalTask && finalTask.error) || data.detail || "Upload and extraction failed");
      }
    } catch (err) {
      finishPipeline(false);
      toast("error", "Digitization failed", err.message);
    } finally {
      if (btnSubmit) btnSubmit.disabled = false;
    }
  });
}

function pollTaskUntilComplete(taskId) {
  return new Promise((resolve, reject) => {
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/tasks/${taskId}`);
        if (!res.ok) {
          clearInterval(pollInterval);
          reject(new Error("Task polling failed"));
          return;
        }
        const task = await res.json();

        if (task.progress != null) {
          const pb = document.getElementById("pipelineProgress");
          if (pb) pb.style.width = `${Math.min(98, task.progress)}%`;
        }
        if (task.step) {
          const ps = document.getElementById("pipelineSub");
          if (ps) ps.textContent = task.step;
        }

        if (task.status === "completed") {
          clearInterval(pollInterval);
          resolve(task);
        } else if (task.status === "failed") {
          clearInterval(pollInterval);
          reject(new Error(task.error || "Processing failed"));
        }
      } catch (err) {
        clearInterval(pollInterval);
        reject(err);
      }
    }, 500);
  });
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
    if (key === "area_sqm") return `<b>${val} m²</b>`;
    if (key === "height_max") return `<span class="feat-prop-badge">${val} m (apex / ridge)</span>`;
    if (key === "height_min") return `<span class="feat-prop-badge">${val} m (eave)</span>`;
    if (key === "height_mean") return `<span class="feat-prop-badge">${val} m (mean)</span>`;
    if (key === "terrain_elevation_m") return `<span class="feat-prop-badge">${val} m (ground DEM)</span>`;
    if (key === "building:levels") return `<b>${val} stories</b>`;
    if (key === "roof_profile") return `<span class="feat-prop-badge ${val === 'sloped' ? 'sloped' : ''}">${val.toUpperCase()}</span>`;
    if (key === "height_m" || key === "tree_height") return `${val} m`;
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
    if (key === "confidence_score" || key === "tree_height") continue;
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

/* =====================  HARDWARE ACCELERATION TELEMETRY  ===================== */
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