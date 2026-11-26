"use strict";
window.Atlas = (() => {
  const root = document.body.dataset.assets;
  const cache = new Map();
  const palette = {
    health: ["#e3f2e9", "#b9d7c7", "#81b19e", "#4e8b77", "#2b6051"],
    fire: ["#f8edd6", "#ecd1a5", "#dba966", "#b8783c", "#83502d"],
    social: ["#eee9f5", "#d6c8e8", "#b79acb", "#8d6ba7", "#674780"],
    diverge: ["#407da0", "#a1c9d8", "#f4f2e8", "#d8abc7", "#9a5683"],
    missing: "#d7dfdd"
  };
  const bounds = L.latLngBounds([48.96, -120.12], [60.04, -109.89]);
  const calgary = L.latLngBounds([51.02, -114.135], [51.077, -114.013]);
  const fmt = (n, places = 0) => n == null || !Number.isFinite(Number(n)) ? "Unavailable" :
    Number(n).toLocaleString("en-CA", {minimumFractionDigits: places, maximumFractionDigits: places});
  const esc = value => String(value ?? "Unavailable").replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]);

  function fetchAsset(path, type = "json") {
    if (!cache.has(path)) {
      cache.set(path, fetch(root + path).then(async response => {
        if (!response.ok) throw new Error(path + " (" + response.status + ")");
        if (type === "gzip-json") {
          const bytes = new Uint8Array(await response.arrayBuffer());
          if (bytes[0] !== 31 || bytes[1] !== 139) return JSON.parse(new TextDecoder().decode(bytes));
          const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
          return new Response(stream).json();
        }
        return type === "text" ? response.text() : response.json();
      }).catch(error => {
        cache.delete(path);
        throw error;
      }));
    }
    return cache.get(path);
  }
  const result = name => fetchAsset("results/" + name + ".json");
  const geo = (area, file) => {
    const path = "maps/" + area + "/" + file + ".geojson";
    return area === "flood" && ["coverage", "buildings"].includes(file) ?
      fetchAsset(path + ".gz", "gzip-json") : fetchAsset(path);
  };
  const model = (area, file) => fetchAsset("maps/" + area + "/" + file + ".json");
  const classify = (value, breaks, colors) => {
    if (value == null || !Number.isFinite(value)) return palette.missing;
    const index = breaks.findIndex(cut => value <= cut);
    return colors[index < 0 ? colors.length - 1 : index];
  };
  const swatches = (colors, labels) => colors.map((color, i) => [color, labels[i]]);
  const polygons = (color, fillOpacity = .65, weight = .55) => ({color, fillColor: color, fillOpacity, weight});
  const choropleth = color => ({color: "#5e726a", fillColor: color, fillOpacity: .82, opacity: .45, weight: .45});
  const line = (color, weight = 1.15) => ({color, weight, opacity: .86, fill: false});
  const locationLabels = [
    ["Calgary", 51.04, -114.08], ["Edmonton", 53.55, -113.5], ["Fort McMurray", 56.73, -111.38],
    ["Grande Prairie", 55.17, -118.79], ["Lethbridge", 49.69, -112.83]
  ];

  function setup(panel, initial = bounds) {
    const element = panel.querySelector(".map");
    panel.dataset.mapLayout = initial === calgary ? "urban" : "province";
    const style = L.canvas({padding: .3});
    const map = L.map(element, {preferCanvas: true, renderer: style, scrollWheelZoom: false, zoomSnap: .1,
      minZoom: 3, maxZoom: 17, maxBounds: [[45, -131], [65, -99]], maxBoundsViscosity: .65,
      attributionControl: false});
    let home = initial;
    let province = bounds;
    let homeCamera;
    function fit(extent) {
      const padding = panel.dataset.mapLayout === "province" ? [24, 22] : [21, 19];
      map.fitBounds(extent, {padding, animate: false});
    }
    function fitHome() {
      fit(home === bounds ? province : home);
      homeCamera = {center: map.getCenter(), zoom: map.getZoom()};
    }
    function atHome() {
      return homeCamera.zoom === map.getZoom() && homeCamera.center.equals(map.getCenter(), 1e-6);
    }
    function setHome(extent, layout = "province") {
      home = extent;
      panel.dataset.mapLayout = layout;
      base.setStyle(provinceStyle());
      map.invalidateSize({animate: false});
      fitHome();
    }
    fitHome();
    element.tabIndex = 0;
    L.control.scale({imperial: false}).addTo(map);
    const layers = L.layerGroup().addTo(map);
    function provinceStyle() {
      return {fillColor: "#f9fbf6", fillOpacity: panel.dataset.mapLayout === "urban" ? 0 : 1,
        color: "#58766a", weight: 1.6};
    }
    const base = L.geoJSON(null, {interactive: false, style: provinceStyle}).addTo(map);
    const provincePane = map.createPane("province-labels");
    provincePane.style.pointerEvents = "none";
    provincePane.style.zIndex = "625";
    const labels = locationLabels.map(([name, lat, lon]) =>
      L.marker([lat, lon], {pane: "province-labels", interactive: false, keyboard: false,
        icon: L.divIcon({className: "map-label", html: name, iconSize: null, iconAnchor: [-5, 8]})}).addTo(map));
    function showLabels() {
      const size = map.getSize(), shown = [];
      for (const marker of labels) {
        const label = marker.getElement();
        const point = map.latLngToContainerPoint(marker.getLatLng());
        const left = point.x + 5, top = point.y - 8;
        const right = left + label.offsetWidth, bottom = top + label.offsetHeight;
        const inside = map.getZoom() < 13 && left >= 8 && right <= size.x - 8 && top >= 8 && bottom <= size.y - 26;
        const overlap = shown.some(rect => left < rect.right + 5 && right + 5 > rect.left &&
          top < rect.bottom + 4 && bottom + 4 > rect.top);
        label.style.visibility = inside && !overlap ? "" : "hidden";
        if (inside && !overlap) shown.push({left, right, top, bottom});
      }
    }
    map.on("moveend", showLabels);
    showLabels();
    geo("shared", "outline").then(value => {
      base.addData(value);
      base.bringToBack();
      province = base.getBounds();
      if (home === bounds && atHome()) fitHome();
    }).catch(error => setStatus(error.message));
    let revision = 0;
    const control = name => panel.querySelector('[data-control="' + name + '"]');
    const value = name => control(name).type === "checkbox" ? control(name).checked : control(name).value;
    const setStatus = text => { panel.querySelector(".map-status").textContent = text; };
    const setLegend = entries => {
      panel.querySelector(".legend").innerHTML = entries.map(([color, label]) =>
        '<span><b aria-hidden="true" style="background:' + color + '"></b>' + esc(label) + '</span>').join("");
    };
    panel.querySelector("[data-reset]").addEventListener("click", fitHome);
    new ResizeObserver(() => {
      const fitting = atHome();
      map.invalidateSize({animate: false});
      if (fitting) fitHome();
      showLabels();
    }).observe(element);

    function draw(data, options = {}) {
      const layer = L.geoJSON(data, {
        filter: options.filter, style: options.style,
        interactive: Boolean(options.tip || options.select),
        pointToLayer: (feature, latlng) => L.circleMarker(latlng,
          options.point ? options.point(feature.properties) : {radius: 3, color: "#fefdfa",
            fillColor: "#346c58", fillOpacity: .92, weight: .75}),
        onEachFeature: (feature, target) => {
          if (options.tip) target.bindTooltip(() => options.tip(feature.properties), {sticky: true, direction: "auto"});
          if (options.select) target.on("click", () => options.select(feature, target));
        }
      }).addTo(layers);
      return layer;
    }
    function listen(handler) {
      const refresh = async () => {
        const ticket = ++revision;
        try {
          await handler(() => ticket === revision);
        } catch (error) {
          if (ticket === revision) setStatus("This layer could not load. " + error.message);
        }
      };
      panel.querySelector(".controls").addEventListener("change", refresh);
      panel.querySelector(".controls").addEventListener("input", e => {
        if (e.target.type === "range") {
          const output = panel.querySelector('[data-output="' + e.target.dataset.control + '"]');
          if (output) output.value = e.target.value;
          refresh();
        }
      });
      return refresh;
    }
    return {panel, map, clear: () => layers.clearLayers(), control, value, draw, listen, setStatus,
      setLegend, fit, setHome};
  }
  function options(select, entries, selected) {
    const previous = selected ?? select.value;
    select.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
    if (entries.some(([value]) => String(value) === previous)) select.value = previous;
  }
  return {fetchAsset, result, geo, model, palette, classify, swatches, polygons, choropleth, line, setup,
    options, fmt, esc, bounds, calgary};
})();

