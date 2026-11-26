"use strict";
window.AtlasLabs = {};
(() => {
  const A = window.Atlas;
  const {geo, result, model, setup, options, fmt, esc, palette, classify, choropleth, polygons, line, swatches} = A;
  const densityBreaks = [10, 100, 1000, 4000];
  const densityLabels = ["≤10", "10–100", "100–1,000", "1,000–4,000", ">4,000 people/km²"];
  const popupSite = p => esc(p.name);
  const popupDA = p => "DA " + esc(p.uid) + " · " + fmt(p.density, 1) + " people/km²";
  const steps = [
    [[], "Start with Alberta’s outline and its census divisions."],
    [["sites"], "Facility records give locations and attributes such as category and provider."],
    [["sites", "fire", "flood"], "Points locate hospitals, lines trace roads and polygons delineate mapped footprints."],
    [["das", "social"], "Each dissemination area has an identifier, geometry and linked Census attributes."],
    [["fire", "flood"], "Clip to a study extent, union overlapping polygons and intersect infrastructure."],
    [["sites", "social"], "Compare facility locations with the distribution of population to frame an accessibility question."]
  ];
  AtlasLabs.opening = panel => {
    const map = setup(panel);
    const features = {
      admin: () => geo("shared", "divisions").then(d => [d, {style: () => ({color: "#789188", fill: false, weight: 1.1}),
        tip: p => esc(p.CDNAME || p.uid)}]),
      das: () => geo("shared", "das").then(d => [d, {style: () => line("#9ead9b", .45), tip: popupDA}]),
      social: () => geo("shared", "das").then(d => [d, {style: f => choropleth(classify(f.properties.density,
        densityBreaks, palette.social)), tip: popupDA}]),
      fire: () => geo("wildfire", "unique").then(d => [d, {style: () => polygons("#bc793e"), tip: () => "Unique mapped wildfire footprint"}]),
      sites: () => geo("healthcare", "hospital").then(d => [d, {tip: popupSite}])
    };
    const refresh = map.listen(async keep => {
      const active = Object.keys(features).filter(key => map.value(key));
      const status = [];
      const plans = await Promise.allSettled(active.map(key => features[key]()));
      const flood = map.value("flood") ? await Promise.allSettled([geo("flood", "coverage"), geo("flood", "roads")]) : [];
      if (!keep()) return;
      map.clear();
      for (const plan of plans) {
        if (plan.status === "fulfilled") map.draw(...plan.value);
        else status.push(plan.reason.message);
      }
      flood.forEach((plan, i) => {
        if (plan.status === "fulfilled") map.draw(plan.value, {style: () => i ? line("#647782", 1.35) : polygons("#528fb6"),
          tip: () => i ? "2021 Calgary road network" : "Mapped designated flood hazard"});
        else status.push(plan.reason.message);
      });
      const colors = {admin: ["#789188", "Census division"], das: ["#adb9a4", "DA boundary"], sites: ["#346c58", "Hospital"],
        fire: ["#bc793e", "Mapped wildfire footprint"], social: ["#8d6ba7", "Population density (darker = higher)"]};
      map.setLegend([["#f9fbf6", "Alberta"], ...active.map(k => colors[k]),
        ...(map.value("flood") ? [["#528fb6","Mapped flood hazard"],["#647782","Calgary roads"]] : [])]);
      map.setStatus(status.length ? "Some layers are unavailable. " + status.join(" · ") : "");
    });
    for (const button of document.querySelectorAll("[data-step]")) {
      button.addEventListener("click", () => {
        const [keys, text] = steps[button.dataset.step];
        for (const other of document.querySelectorAll("[data-step]")) other.setAttribute("aria-pressed", other === button);
        for (const name of ["das", "sites", "fire", "flood", "social"]) map.control(name).checked = keys.includes(name);
        map.control("admin").checked = true;
        document.querySelector("#step-description").textContent = text;
        refresh();
      });
    }
    return refresh();
  };

  function chartBars(target, names, values, color, barClass = "") {
    const highest = Math.max(.01, ...values);
    const width = 600 / values.length;
    const marks = values.map((value, i) => {
      const tall = 117 * value / highest;
      return '<g><title>' + esc(names[i]) + ": " + fmt(value, 1) + '%</title><rect class="' + barClass + '" x="' + (34 + i * width) +
        '" y="' + (145 - tall) + '" width="' + (width - 18) + '" height="' + tall +
        '" fill="' + color + '"/><text x="' + (25 + i * width + width / 2) +
        '" y="166" text-anchor="middle">' + esc(names[i]) + '</text><text x="' + (25 + i * width + width / 2) +
        '" y="' + (137 - tall) + '" text-anchor="middle">' + fmt(value, 1) + '%</text></g>';
    }).join("");
    const title = target.previousElementSibling.textContent;
    target.innerHTML = '<svg class="plot" viewBox="0 0 650 184" role="img" aria-label="'+esc(title)+'">' +
      '<path d="M27 145H637" stroke="#91a39a"/><text x="22" y="149" text-anchor="end">0</text>' + marks + '</svg>';
  }

  AtlasLabs.health = async panel => {
    const map = setup(panel);
    const [data, summary, das, divisions] = await Promise.all([model("healthcare", "model"), result("healthcare"),
      geo("shared", "das"), geo("shared", "divisions")]);
    const names = Object.keys(summary.groups).sort((a, b) => a === "hospital" ? -1 : b === "hospital" ? 1 : a.localeCompare(b));
    options(map.control("category"), names.map(k => [k, ({hospital:"Hospitals · ODHF v2.0",hospital_v11:"Hospitals · ODHF v1.1",clinic:"Clinics · ODHF v2.0",ambulatory:"Ambulatory care · ODHF v2.0"})[k]]));
    const ids = [...new Set(data.division)].sort();
    options(map.control("region"), [["all", "All Alberta"], ...ids.map(k => [k, "Division " + k.slice(2)])]);
    const partCounts = new Map(), labelledParts = new Map();
    for (const {uid} of data.candidates) partCounts.set(uid, (partCounts.get(uid) || 0) + 1);
    const candidateLabels = data.candidates.map(({uid, label}) => {
      if (partCounts.get(uid) === 1) return label;
      const part = (labelledParts.get(uid) || 0) + 1;
      labelledParts.set(uid, part);
      return label + " (mapped part " + part + ")";
    });
    options(map.control("candidate"), [["best", "Best for provincial mean distance"],
      ["cover", "Best provincial coverage at this threshold"],
      ...candidateLabels.map((label, i) => [i, label])]);
    map.control("weight").querySelector('[value=age65]').disabled = !data.age65.some(p => p > 0);
    let oldRegion = "all";
    const refresh = map.listen(async keep => {
      const group = map.value("category"), view = map.value("view"), region = map.value("region");
      const threshold = Number(map.value("threshold"));
      const choice = map.value("candidate");
      const candidate = choice === "best" ? summary.groups[group].best : choice === "cover" ?
        summary.groups[group].maximum_cover[threshold].candidate : Number(choice);
      const base = data.base[group], weights = data[map.value("weight")];
      map.control("candidate").disabled = view !== "scenario";
      map.control("candidate").closest("label").hidden = view !== "scenario";
      const tasks = [geo("healthcare", group)];
      if (view === "catchments") tasks.push(geo("healthcare", "voronoi_" + group), geo("healthcare", "buffer_" + group + "_" + threshold));
      if (view === "scenario") tasks.push(geo("healthcare", "candidates"), model("healthcare", "candidate_" + candidate));
      const [sites, extra, extra2] = await Promise.all(tasks);
      if (!keep()) return;
      const distance = view === "scenario" ? base.map((d, i) => Math.min(d, extra2[i])) : base;
      const values = new Map(data.uid.map((key, i) => [key, {base: base[i], new: distance[i], population: data.population[i]}]));
      const counts = [0, 0, 0, 0, 0];
      let population = 0, oldCost = 0, newCost = 0, covered = 0;
      const breaks = [10, 25, 50, 100];
      const inRegion = uid => region === "all" || uid.startsWith(region);
      data.uid.forEach((uid, i) => {
        if (!inRegion(uid) || weights[i] == null) return;
        const w = weights[i], d = distance[i];
        population += w;
        oldCost += base[i] * w;
        newCost += d * w;
        if (d <= threshold) covered += w;
        const bin = breaks.findIndex(x => d <= x);
        counts[bin < 0 ? 4 : bin] += w;
      });
      const km = ["≤10 km", "10–25", "25–50", "50–100", ">100"];
      chartBars(document.querySelector('[data-chart="health"]'), km, counts.map(n => population ? 100 * n / population : 0), "#6b9d87");
      map.clear();
      if (view === "catchments") {
        map.draw(extra, {style: f => {
          const hash = [...String(f.properties.name)].reduce((v, c) => v + c.charCodeAt(0), 0);
          return {...choropleth(palette.health[hash % 4]), weight: .8, fillOpacity: .62};
        }, tip: p => "Nearest site · " + esc(p.name)});
        map.draw(extra2, {style: () => polygons("#397f63", .2, 1.2), tip: () => threshold + " km buffer union"});
      } else {
        map.draw(das, {style: f => {
          const p = f.properties, score = values.get(p.uid);
          const d = view === "sites" ? p.density : view === "scenario" ? score?.base - score?.new : score?.new;
          const cuts = view === "sites" ? densityBreaks : view === "scenario" ? [.01, 2, 10, 30] : breaks;
          return choropleth(inRegion(p.uid) ? classify(d, cuts, palette.health) : palette.missing);
        }, tip: p => {
          const d = values.get(p.uid);
          return "DA " + esc(p.uid) + " · " + (d ? fmt(d.new, 1) + " km · " + fmt(d.population) + " residents" : "Unavailable");
        }});
      }
      map.draw(sites, {tip: popupSite});
      if (view === "scenario") map.draw(extra, {filter: f => f.properties.candidate_index === candidate,
        point: () => ({radius: 8, color: "#fff", weight: 2, fillColor: "#af613d", fillOpacity: 1}),
        tip: () => "Hypothetical location · " + esc(candidateLabels[candidate])});
      if (oldRegion !== region) {
        const layer = L.geoJSON(divisions, {filter: f => f.properties.uid === region});
        const divisionBounds = layer.getBounds();
        map.setHome(divisionBounds.isValid() ? divisionBounds : A.bounds, divisionBounds.isValid() ? "regional" : "province");
        oldRegion = region;
      }
      const legend = view === "catchments" ? [["#a6c6b4", "Nearest catchments"],["#397f63", threshold + " km buffer"]] :
        swatches(palette.health, view === "sites" ? densityLabels : view === "scenario" ?
          ["≤0.01 km gain", "0.01–2", "2–10", "10–30", ">30"] : km);
      map.setLegend([...legend, ["#346c58", "Listed facility"], ...(view === "scenario" ? [["#af613d","Hypothetical location"]] : []),
        [palette.missing, "Unavailable / outside selection"]]);
      map.setStatus(population ? fmt(covered) + " of " + fmt(population) + " represented residents (" +
        fmt(100 * covered / population, 1) + "%) within " + threshold + " km. Weighted mean " + fmt(newCost / population, 1) + " km." +
        (view === "scenario" ? " " + candidateLabels[candidate] + " reduces the selected mean by " +
          fmt((oldCost - newCost) / population, 2) + " km." : "") : "No population observations for this selection.");
    });
    return refresh();
  };

  AtlasLabs.chartBars = chartBars;
})();


