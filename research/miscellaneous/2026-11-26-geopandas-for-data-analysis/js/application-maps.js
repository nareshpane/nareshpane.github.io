"use strict";
(() => {
  const A = window.Atlas;
  const {geo, model, result, setup, options, fmt, esc, palette, classify, choropleth, polygons, line, swatches} = A;
  const findChart = name => document.querySelector('[data-chart="' + name + '"]');
  const outline = "#58766a";

  AtlasLabs.wildfire = async panel => {
    const map = setup(panel);
    const summary = await result("wildfire");
    const years = summary.years, largest = summary.annual.filter(row => row.km2 != null).sort((a, b) => b.km2 - a.km2)[0];
    const yearBox = map.control("year");
    const chart = findChart("fire");
    yearBox.min = years[0]; yearBox.max = years.at(-1); yearBox.value = largest.year;
    let timer;
    const refresh = map.listen(async keep => {
      const view = map.value("view"), year = Number(yearBox.value), grid = map.value("grid"), tau = map.value("tau");
      const km = Number(map.value("km")), h = map.value("h");
      panel.querySelector('[data-output="year"]').value = year;
      for (const mark of chart.querySelectorAll("[data-year]")) {
        mark.setAttribute("aria-pressed", ["annual","grid"].includes(view) && Number(mark.dataset.year) === year);
      }
      for (const [key, enabled] of Object.entries({year: ["annual","grid"].includes(view), grid: ["grid","recurrence"].includes(view),
          tau: view === "recurrence", km: view === "proximity", h: view === "intensity"})) {
        map.control(key).disabled = !enabled;
        map.control(key).closest("label").hidden = !enabled;
      }
      for (const button of panel.querySelectorAll("[data-action]")) button.hidden = view !== "annual";
      const files = {annual: "year_" + year, unique: "unique", grid: "grid_" + grid, recurrence: "grid_" + grid,
        proximity: "proximity", intensity: "intensity", counts: "counts"};
      const data = await geo("wildfire", files[view]);
      if (!keep()) return;
      map.clear();
      let legend, status;
      if (view === "annual" || view === "unique") {
        map.draw(data, {style: () => polygons("#ad6a38", .67, .85), tip: () =>
          view === "annual" ? year + " annual mapped footprint" : "Unique footprint across the verified years"});
        legend = [["#b8783c","Mapped perimeter"],["#f9fbf6","Alberta context"]];
        const value = view === "annual" ? summary.annual.find(r => r.year === year)?.km2 : summary.unique_km2;
        status = fmt(value, 1) + " km² of " + (view === "annual" ? "unioned mapped footprint in " + year : "unique mapped footprint.");
      } else if (view === "proximity") {
        map.draw(data, {point: p => ({radius: p.km <= km ? Math.min(8, 1.3 + Math.sqrt(p.population) / 18) : 1.5, weight: 0,
          fillColor: p.km <= km ? "#a15935" : "#829b8c", fillOpacity: p.km <= km ? .9 : .45}),
          tip: p => "DA " + esc(p.uid) + " · " + fmt(p.km, 1) + " km · " + fmt(p.population) + " residents"});
        legend = [["#a15935","Within " + km + " km · size reflects 2021 residents"],["#829b8c","More than " + km + " km"]];
        status = fmt(summary.proximity[km].locations) + " populated DA points within the threshold, representing " +
          fmt(summary.proximity[km].population) + " residents.";
      } else {
        const spec = {
          grid: ["y" + year, [.001,.01,.05,.2], ["≤0.1%", "0.1–1%", "1–5%", "5–20%", ">20%"]],
          recurrence: ["frequency_" + tau, [0,.1,.2,.4], ["0 years", "1 year", "2 years", "3–4 years", "5+ years"]],
          intensity: ["h" + h, [.001,.005,.02,.05], ["≤0.001","0.001–0.005","0.005–0.02","0.02–0.05",">0.05 events/km²"]],
          counts: ["incidents", [1,5,20,50], ["1","2–5","6–20","21–50",">50 events"]]
        };
        const [field, breaks, labels] = spec[view];
        map.draw(data, {style: f => choropleth(classify(f.properties[field], breaks, palette.fire)),
          tip: p => view === "grid" || view === "recurrence" ? "Cell " + esc(p.cell) + " · " + fmt(p[field] * 100, 1) + "%" :
            fmt(p[field], view === "counts" ? 0 : 4) + (view === "counts" ? " events" : " events/km²")});
        legend = swatches(palette.fire, labels);
        status = view === "recurrence" ? fmt(summary.grid[grid].recurrent[tau]) + " cells cross " + tau +
          "% burned fraction in at least two of " + years.length + " years." :
          view === "grid" ? year + " footprint share by " + grid + " km hexagon." :
          view === "intensity" ? fmt(summary.ignitions_used) + " usable ignition points; " + h + " km bandwidth." :
          "Observed ignition counts on a 10 km square grid.";
      }
      map.setLegend(legend); map.setStatus(status);
    });
    function selectYear(year) {
      map.control("view").value = "annual";
      yearBox.value = year;
      refresh();
    }
    function pause() { window.clearInterval(timer); timer = undefined; }
    panel.querySelector('[data-action="pause"]').onclick = pause;
    panel.querySelector('[data-action="restart"]').onclick = () => {pause(); selectYear(years[0]);};
    panel.querySelector('[data-action="play"]').onclick = () => {
      pause();
      map.control("view").value = "annual";
      if (Number(yearBox.value) === years.at(-1)) selectYear(years[0]);
      else refresh();
      timer = window.setInterval(() => {
        const i = years.indexOf(Number(yearBox.value));
        if (i >= years.length - 1) pause();
        else selectYear(years[i + 1]);
      }, 1350);
    };
    map.control("view").addEventListener("change", pause);
    const rows = summary.annual;
    const maxCount = Math.ceil(Math.max(...rows.map(x=>x.incidents)) / 500) * 500;
    const maxArea = Math.ceil(Math.max(...rows.map(x=>x.km2 || 0)) / 10000) * 10000;
    const left = 58, right = 617, top = 45, bottom = 190, spacing = (right - left) / rows.length;
    const center = i => left + spacing * (i + .5);
    const scaled = (value, max) => bottom - (bottom - top) * value / max;
    const axes = [0,.25,.5,.75,1].map(relative => {
      const y = scaled(relative, 1);
      return '<path d="M'+left+' '+y+'H'+right+'" stroke="'+(relative ? "#dedfd7" : "#849488")+'"/>' +
        '<text x="'+(left-7)+'" y="'+(y+4)+'" text-anchor="end">'+fmt(relative*maxCount)+'</text>' +
        '<text x="'+(right+7)+'" y="'+(y+4)+'" class="area-axis">'+fmt(relative*maxArea)+'</text>';
    }).join("");
    const marks = rows.map((r,i) => {
      const x = center(i), y = scaled(r.incidents, maxCount);
      const label = r.year % 2 === 0 ? '<text x="' + x + '" y="211" text-anchor="middle">' + r.year + '</text>' : "";
      return '<g data-year="' + r.year + '" tabindex="0" role="button" aria-pressed="false" aria-label="' + r.year + ', ' + r.incidents +
        ' incidents"><title>' + r.year + ' · ' + fmt(r.incidents) + ' incidents · ' + (r.km2 == null ?
          "perimeter unavailable" : fmt(r.km2, 1) + " km²") + '</title><rect class="fire-bar" x="' + (x-8.5) + '" y="' + y +
        '" width="17" height="' + (bottom - y) + '"/>' + label + '</g>';
    }).join("");
    const path = rows.map((r,i) => r.km2 == null ? "" :
      (rows[i-1]?.km2 == null ? "M" : "L") + center(i) + " " + scaled(r.km2, maxArea)).join("");
    chart.innerHTML = '<svg class="plot" viewBox="0 0 682 224" role="group" aria-label="Annual fire observations with separate incident and mapped-area axes">' +
      '<text x="'+left+'" y="23">Incident count · left axis</text><text x="'+right+'" y="23" class="area-axis" text-anchor="end">Mapped footprint (km²) · right axis</text>' +
      axes + marks + '<path d="'+path+'" fill="none" stroke="#80513e" stroke-width="2.7" pointer-events="none"/></svg>';
    function plotClick(target) {
      const year = Number(target.closest("[data-year]")?.dataset.year);
      if (years.includes(year)) {pause(); selectYear(year);}
      else if (year) map.setStatus(year + " incident data are available. No mapped perimeter layer was acquired for that year.");
    }
    chart.addEventListener("click", e=>plotClick(e.target));
    chart.addEventListener("keydown", e=>{if (e.key === "Enter" || e.key === " ") {e.preventDefault(); plotClick(e.target);}});
    return refresh();
  };

  AtlasLabs.flood = async panel => {
    const map = setup(panel, A.calgary);
    const [summary, extent] = await Promise.all([result("flood"), geo("flood", "study")]);
    options(map.control("period"), summary.scenarios.map(r => [r.period,"1 in " + r.period]), "100");
    map.control("view").querySelector('[value=coverage]').disabled = !summary.study_coverage;
    let oldMode = "local";
    const refresh = map.listen(async keep => {
      const view = map.value("view"), province = view === "coverage", period = Number(map.value("period"));
      const min = Number(map.value("minimum")), roads = map.value("roads"), roofs = map.value("roofs");
      for (const field of ["period","minimum","roads","roofs"]) {
        map.control(field).disabled = province;
        map.control(field).closest("label").hidden = province;
      }
      const active = province ? {coverage: geo("flood", "coverage")} : {event: geo("flood","event_" + period)};
      if (!province && roads) {
        if (view !== "scenario") active.roads = geo("flood","roads");
        active.impact = geo("flood","road_" + period);
      }
      if (!province && roofs) {
        if (view !== "scenario") active.roofs = geo("flood","buildings");
        active.foot = geo("flood","roof_" + period);
      }
      const data = Object.fromEntries(await Promise.all(Object.entries(active).map(async ([k,p]) => [k,await p])));
      if (!keep()) return;
      map.clear();
      const viewMode = province ? "all" : "local";
      if (viewMode !== oldMode) {
        map.setHome(province ? A.bounds : A.calgary, province ? "province" : "urban");
        oldMode = viewMode;
      }
      if (province) {
        map.draw(data.coverage, {style: () => polygons("#3d7eac",.7,1.1), tip: () => "Designated mapped flood hazard"});
        map.setLegend([["#3d7eac","Mapped flood hazard"],["#f9fbf6","Coverage outside shown polygons is undetermined"]]);
        const quality = summary.coverage_quality;
        map.setStatus("Provincial regulatory hazard polygons at display resolution." + (quality ?
          " " + fmt(quality.display_features) + " polygon records are visible after generalization and repair of " +
          fmt(quality.source_features) + " source records." : ""));
        return;
      }
      map.draw(extent, {style: () => ({color: outline, weight: 1.2, dashArray:"4 5", fill: false})});
      if (data.roofs) map.draw(data.roofs, {style: () => polygons("#899496",.34,.25)});
      if (data.roads) map.draw(data.roads, {style: () => line("#879397",.8)});
      map.draw(data.event, {style: () => polygons("#598db4",.48,.9), tip: () => "1 in " + period + " open-water footprint"});
      const included = f => f.properties.fraction >= min;
      if (data.foot) map.draw(data.foot, {filter: included, style: () => polygons("#bf7254",.83,.5),
        tip: p => fmt(p.square_metres,1) + " m² · " + fmt(100 * p.fraction,1) + "% of clipped roof"});
      if (data.impact) map.draw(data.impact, {filter: included, style: () => line("#b86b38",2.3),
        tip: p => fmt(p.metres,1) + " m · " + fmt(100 * p.fraction,1) + "% of clipped road arc"});
      const stats = summary.thresholds.find(x => x.period === period && x.minimum_fraction === min);
      map.setLegend([["#598db4","Open-water flood"],["#879397","Mapped asset"],["#b86b38",view === "scenario" ? "Hypothetical closure fragments" : "Exposed road"],["#bf7254","Exposed roof"]]);
      map.setStatus("Within the " + fmt(min * 100) + "% asset screening threshold: " + fmt(stats.road_km,2) +
        " road km and " + fmt(stats.roofs) + " roof outlines (" + fmt(stats.roof_m2) +
        " m² intersected)." + (view === "scenario" ? " Potential closures are illustrative." : ""));
    });
    return refresh();
  };
})();

