"use strict";
(() => {
  const {fmt, esc, result, fetchAsset} = Atlas;
  for (const anchor of document.querySelectorAll("[data-output-file]")) {
    fetchAsset(anchor.dataset.outputFile, "text").then(value => {anchor.textContent=value;}).catch(error=>{
      anchor.textContent="Build output unavailable. "+error.message;
    });
  }
  const boundaryFields = document.querySelectorAll("[data-boundary]");
  fetchAsset("figures/foundations/boundary-preview.json").then(info => {
    for (const field of boundaryFields) field.textContent = info[field.dataset.boundary];
  }).catch(() => {
    // The inline preview also works without the generated metadata.
  });
  const toggle = document.querySelector(".toc-toggle");
  const toc = document.querySelector(".toc");
  const allLinks = [...toc.querySelectorAll("a[href^='#']")];
  const locations = allLinks.map(a=>[a,document.getElementById(a.hash.slice(1))]).filter(([,target])=>target);
  let active, pending;
  function updateContents() {
    pending=false;
    let next=locations[0][0], y=-Infinity;
    for(const [link,target] of locations) {
      const top=target.getBoundingClientRect().top;
      if(top<=135&&top>=y) {next=link;y=top;}
    }
    if(next===active)return;
    if(active){active.classList.remove("active");active.removeAttribute("aria-current");}
    active=next;active.classList.add("active");active.setAttribute("aria-current","location");
    const group=active.closest("details");
    if(group)group.open=true;
    if(toc.offsetWidth) {
      const top=toc.getBoundingClientRect().top, linkTop=active.getBoundingClientRect().top;
      if(linkTop<top||linkTop>top+toc.clientHeight-28)toc.scrollTop+=linkTop-top-45;
    }
  }
  window.addEventListener("scroll",()=>{
    if(!pending){pending=true;requestAnimationFrame(updateContents);}
  },{passive:true});
  toggle.addEventListener("click",()=>{
    const opened=toc.classList.toggle("open");toggle.setAttribute("aria-expanded",opened);
  });
  toc.addEventListener("click",event=>{
    if(event.target.closest("a")&&window.innerWidth<=850){
      toc.classList.remove("open");toggle.setAttribute("aria-expanded","false");
    }
  });
  updateContents();

  for (const group of document.querySelectorAll(".code-group")) {
    const labels = [...group.querySelectorAll(".script-title, .code-action")];
    const observer = new ResizeObserver(() => {
      const height = Math.max(...labels.map(label => label.getBoundingClientRect().height));
      group.style.setProperty("--summary-content-height", height + "px");
    });
    for (const label of labels) observer.observe(label);
  }

  const token=/#[^\r\n]*|"[^"\r\n]*"|'[^'\r\n]*'|\b(?:True|False|None|and|as|assert|break|class|continue|def|del|elif|else|except|finally|for|from|if|import|in|is|lambda|not|or|pass|raise|return|try|while|with|yield)\b|\b\d+(?:\.\d+)?\b/g;
  function highlight(text) {
    let html="", at=0;
    for(const found of text.matchAll(token)) {
      const part=found[0];
      const kind=part[0]==="#"?"comment":part[0]==="\""||part[0]==="'"?"str":/\d/.test(part[0])?"num":"key";
      html+=esc(text.slice(at,found.index))+'<span class="tok-'+kind+'">'+esc(part)+'</span>';
      at=found.index+part.length;
    }
    return html+esc(text.slice(at));
  }
  for(const panel of document.querySelectorAll("details[data-script], [data-inline-script]")){
    const block=panel.querySelector("code"), button=panel.querySelector("[data-copy]");
    let source = panel.hasAttribute("data-inline-script") ? block.textContent : null;
    let loading;
    if (source) block.innerHTML = highlight(source);
    async function load(){
      if(source || loading)return loading;
      loading=fetchAsset("scripts/"+panel.dataset.script,"text").then(text=>{
        const lines=text.replace(/^\uFEFF/,"").split(/\r?\n/);
        const index=prefix=>lines.findIndex(s=>s.startsWith(prefix));
        const start=panel.dataset.start?index(panel.dataset.start):0;
        const end=panel.dataset.end?index(panel.dataset.end):-1;
        source=lines.slice(Math.max(0,start),end>start?end:undefined).join("\n").trimEnd();
        block.innerHTML=highlight(source);
      }).catch(error=>{block.textContent="Python source could not load. "+error.message;})
        .finally(()=>{loading=null;});
      return loading;
    }
    panel.addEventListener("toggle",()=>{if(panel.open)load();});
    button.addEventListener("click",async()=>{
      await load();
      if(!source)return;
      try {
        if(navigator.clipboard)await navigator.clipboard.writeText(source);
        else {
          const textarea=document.createElement("textarea");textarea.value=source;
          textarea.style.position="fixed";textarea.style.left="-9999px";document.body.append(textarea);
          textarea.select();const copied=document.execCommand("copy");textarea.remove();
          if(!copied)throw new Error("Clipboard unavailable");
        }
        button.textContent="Copied";
      } catch {
        button.textContent="Select code to copy";
      }
      window.setTimeout(()=>{button.textContent="Copy Code";},1700);
    });
  }

  const num=(value,precision=1,suffix="")=>fmt(value,precision)+suffix;
  function writeEvidence(key, metrics, sentence) {
    const spot=document.querySelector('[data-metrics="'+key+'"]');
    spot.innerHTML=metrics.map(([value,label])=>'<article><strong>'+esc(value)+'</strong><span>'+esc(label)+'</span></article>').join("");
    document.querySelector('[data-evidence="'+key+'"]').textContent=sentence;
  }
  function table(key,title,headers,rows) {
    const spot=document.querySelector('[data-table="'+key+'"]');
    const quantity = /^(?:[−+-]?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|km²?|m²|MiB|ms))?|—|Unavailable)$/;
    const numeric = headers.map((_,i) => i > 0 && rows.length && rows.every(row => quantity.test(row[i])));
    const alignment = i => numeric[i] ? ' class="numeric"' : '';
    spot.innerHTML="<table><caption>"+esc(title)+"</caption><thead><tr>"+headers.map((h,i)=>'<th scope="col"'+alignment(i)+'>'+esc(h)+"</th>").join("")+
      "</tr></thead><tbody>"+rows.map(row=>"<tr>"+row.map((v,i)=>i?'<td'+alignment(i)+'>'+esc(v)+"</td>":'<th scope="row">'+esc(v)+"</th>").join("")+
      "</tr>").join("")+"</tbody></table>";
    spot.setAttribute("role", "region");
    spot.setAttribute("aria-label", title);
    new ResizeObserver(() => {
      if (spot.scrollWidth > spot.clientWidth) spot.tabIndex = 0;
      else spot.removeAttribute("tabindex");
    }).observe(spot);
  }
  const reports={
    shared: value=>{
      writeEvidence("shared",[[fmt(value.population),"2021 census residents"],[fmt(value.areas),"Alberta DA geometries"],
        [fmt(value.points["Official representative point"] || 0),"Official DA locators"],
        [fmt(value.unjoined),"DA geometries with unavailable counts"]],
      "Joined observations reconcile to the published Alberta total. "+fmt(value.repairs)+" source DA geometries required repair.");
      table("projections","Calgary–Edmonton reference-point comparison",["Measurement","Distance (km)","Difference from ellipsoid (km)"],
        Object.entries(value.distances_km).map(([crs,km])=>[crs,fmt(km,2),fmt(km-value.distances_km["ellipsoidal WGS 84"],2)]));
    },
    healthcare: value=>{
      const h=value.groups.hospital;
      writeEvidence("healthcare",[[fmt(h.count),"Alberta hospital locations"],[num(h.mean_km,1," km"),"Population-weighted mean"],
        [num(h.p90_km,1," km"),"Population-weighted 90th percentile"],[fmt(h.beyond["50"]),"Residents beyond 50 km"]],
        "Median "+num(h.median_km,1," km")+", 95th percentile "+num(h.p95_km,1," km")+". "+
        "The best single addition in the "+fmt(value.candidates)+"-candidate hospital scenario is "+h.best_name+
        ", reducing the province-wide mean to "+num(h.best_mean_km,2," km")+
        ". The greatest weighted hospital distances were in census divisions "+h.worst_divisions.slice(0,3).map(d=>d.division.slice(2)).join(", ")+
        (value.groups.hospital_v11 ? ". The 2020 inventory yields "+fmt(value.groups.hospital_v11.count)+" listed hospitals and a mean of "+
        num(value.groups.hospital_v11.mean_km,1," km")+". This change reflects inventory composition and vintage." : "."));
    },
    wildfire: value=>{
      const ranked=value.annual.filter(r=>r.km2!=null).sort((a,b)=>b.km2-a.km2), top=ranked[0];
      writeEvidence("wildfire",[[fmt(value.recorded_incidents),"Unique 2006–2025 incident keys"],
        [value.years[0]+"–"+value.years.at(-1),"Verified perimeter years"],[num(value.unique_km2,0," km²"),"Unique mapped footprint"],
        [String(top.year),"Largest mapped footprint year"]],
        "The "+top.year+" union covered "+num(top.km2,0," km²")+". The sum of yearly footprints was "+
        num(value.cumulative_km2,0," km²")+". At 10 km, "+fmt(value.proximity["10"].locations)+
        " populated DA locators represented "+fmt(value.proximity["10"].population)+" residents.");
      table("fire","Sensitivity of repeated exposure to grid width and threshold",["Nominal hexagon width","Cells","Any fraction","Over 1%","Over 5%"],
        Object.entries(value.grid).map(([size,row])=>[size+" km",fmt(row.cells),fmt(row.recurrent["0"]),fmt(row.recurrent["1"]),fmt(row.recurrent["5"])]));
      const grid=value.grid["50"];
      AtlasLabs.chartBars(document.querySelector('[data-chart="fire-fraction"]'),
        ["<0.1%","0.1–1%","1–5%","5–20%","20–50%","50–100%"],
        grid.unique_distribution.map(n=>100*n/grid.cells), "#bfe3b5", "fire-bar");
    },
    flood: value=>{
      const row=value.scenarios.find(r=>r.period===100)||value.scenarios[0];
      writeEvidence("flood",[[num(value.total_road_km,1," km"),"2021 clipped road denominator"],
        [num(row.road_km,2," km"),"Road in the 1-in-"+row.period+" footprint"],
        [fmt(row.roofs),"Mapped roofs intersecting that footprint"],
        [num(row.roof_m2,0," m²"),"Mapped roof area intersecting"]],
        fmt(row.road_pct,1)+"% of the clipped road denominator intersects the 1-in-"+row.period+" open-water footprint.");
      table("flood","Exposure by available mapped scenario",["Return period","Flood km²","Road km","Road share","Roofs","Roof m²","Studies"],
        value.scenarios.map(r=>["1 in "+r.period,fmt(r.footprint_km2,2),fmt(r.road_km,2),fmt(r.road_pct,1)+"%",
          fmt(r.roofs),fmt(r.roof_m2),r.studies.join("; ")]));
    },
    socioeconomic: value=>{
      const selected=value.indicators.unemployment||value.indicators.density, q=selected.weights.queen;
      const low=value.indicators.low_income;
      writeEvidence("socioeconomic",[[fmt(selected.n),"DAs in focal statistical sample"],[fmt(q.I,3),"Queen Moran’s I · "+(value.indicators.unemployment?"unemployment":"log-density")],
        [fmt(q.p,3),"Global permutation p-value"],[fmt(q.fdr_significant),"BH-significant local areas (q ≤ 0.05)"]],
        low?"Median DA low-income prevalence was "+fmt(low.median,1)+"%. Composite rank correlation between equal weights and "+
        (AtlasLabs.indicators[value.composite_sensitivity?.first]?.[0]||"the first component")+" alone was "+
        fmt(value.composite_sensitivity?.rank_correlation,3)+". Permutations per test: "+value.permutations+".":
        "Published census density is available for this build. Check the acquisition report for Census Profile coverage.");
      const rows=Object.entries(value.indicators).flatMap(([key,entry])=>Object.entries(entry.weights).map(([mode,d])=>
        [AtlasLabs.indicators[key][0],mode,fmt(entry.median,1),fmt(entry.missing),fmt(d.I,3),fmt(d.p,3),fmt(d.raw_significant),fmt(d.fdr_significant)]));
      table("social","Spatial association by indicator and neighbourhood",["Indicator","Weights","DA median","Missing","Moran’s I","p","Raw local","BH local"],rows);
      const ranked=[...value.division_unemployment].sort((a,b)=>a.rate-b.rate);
      table("regional-labour","Selected census divisions by aggregate unemployment rate",
        ["Census division","Unemployed","Labour force","Rate"],
        [...ranked.slice(0,3),...ranked.slice(-3)].map(x=>[x.division,fmt(x.unemployed),fmt(x.labour),fmt(x.rate,1)+"%"]));
    },
    benchmark: value=>{
      table("benchmark","Median elapsed time and file size on the build machine",["Workload","Time (ms)","Size","Notes"],
        [["Naive geometry comparisons",fmt(value.index.naive_ms,2),"—",fmt(value.index.pairs)+" pairs"],
         ["Indexed exact search",fmt(value.index.indexed_ms,2),"—",fmt(value.index.candidates)+" bounding-box candidates"],
         ...value.reads.map(r=>[r.format,fmt(r.milliseconds,2),fmt(r.bytes/1048576,2)+" MiB",fmt(value.sample)+" identical input features"]),
         ["Full vs. web DA layer","—",fmt(value.geometry.original/1048576,2)+" → "+fmt(value.geometry.web/1048576,2)+" MiB",
          value.platform+" · repeated local read timing"]]);
    }
  };
  for(const [key,render] of Object.entries(reports)) {
    result(key).then(render).catch(error=>{
      const node=document.querySelector('[data-evidence="'+key+'"], [data-table="'+key+'"]');
      if(node)node.textContent="Build output unavailable. "+error.message;
    });
  }

  function loadMap(panel) {
    if(panel.dataset.loaded)return;
    panel.dataset.loaded="true";
    Promise.resolve().then(()=>AtlasLabs[panel.dataset.lab](panel)).catch(error=>{
      panel.querySelector(".map-status").textContent="This map could not load. "+error.message;
    });
  }
  const observer=new IntersectionObserver(entries=>{
    for(const entry of entries)if(entry.isIntersecting){observer.unobserve(entry.target);loadMap(entry.target);}
  },{rootMargin:"250px"});
  for(const panel of document.querySelectorAll("[data-lab]")){
    if(panel.dataset.lab==="opening")loadMap(panel);
    else {observer.observe(panel);panel.addEventListener("focusin",()=>loadMap(panel),{once:true});}
  }
})();

