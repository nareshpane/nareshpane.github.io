"use strict";
(() => {
  const A = window.Atlas;
  const {setup, geo, result, model, options, classify, choropleth, fmt, esc, palette, swatches} = A;
  const indicators = {
    density: ["Population density", "people/km²", [10,100,1000,4000]],
    age_share: ["Population aged 65+", "%", [10,16,23,32]],
    income: ["Median household total income", "CAD (2020)", [60000,85000,110000,150000]],
    low_income: ["Low-income prevalence (LIM-AT)", "%", [3,7,12,20]],
    unemployment: ["Unemployment rate", "%", [5,9,14,22]]
  };
  const clusters = {"High–high":"#8c4b78","Low–low":"#5485a4","High–low":"#d599b9","Low–high":"#9cc7d8"};
  const extents = {calgary:[[50.88,-114.33],[51.22,-113.88]], edmonton:[[53.33,-113.75],[53.72,-113.22]]};
  AtlasLabs.indicators = indicators;

  function scatter(canvas, rows, selected, onSelect) {
    const {width:w, height:h} = canvas.getBoundingClientRect();
    if (!w || !h) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const x0 = 58, x1 = w - 18, y0 = 18, y1 = h - 47;
    const observations = Object.entries(rows);
    const xMin = Math.floor(Math.min(-1, ...observations.map(([,d])=>d[0]))), xMax = Math.ceil(Math.max(1, ...observations.map(([,d])=>d[0])));
    const yMin = Math.floor(Math.min(-1, ...observations.map(([,d])=>d[1]))), yMax = Math.ceil(Math.max(1, ...observations.map(([,d])=>d[1])));
    const x = n => x0+(x1-x0)*(n-xMin)/(xMax-xMin);
    const y = n => y1-(y1-y0)*(n-yMin)/(yMax-yMin);
    const ticks = (min, max, target) => {
      const gap = (max-min)/target;
      const base = 10**Math.floor(Math.log10(gap));
      const step = base * [1,2,2.5,5,10].find(n => n*base >= gap);
      return Array.from({length:Math.floor(max/step)-Math.ceil(min/step)+1},
        (_,i) => Number(((Math.ceil(min/step)+i)*step).toPrecision(10)));
    };
    ctx.fillStyle = "#fbfcfa"; ctx.fillRect(0,0,w,h);
    ctx.fillStyle = "#f3eef7"; ctx.fillRect(x(0),y0,x1-x(0),y(0)-y0);
    ctx.fillStyle = "#eef5f7"; ctx.fillRect(x0,y(0),x(0)-x0,y1-y(0));
    ctx.font = "12px Segoe UI";
    function guide(value, horizontal) {
      ctx.strokeStyle = value === 0 ? "#82988d" : "#dfe6e1";
      ctx.fillStyle = "#41594e";
      ctx.beginPath();
      if (horizontal) {
        const py = y(value);
        ctx.moveTo(x0,py); ctx.lineTo(x1,py);
        ctx.textAlign = "right"; ctx.fillText(String(value),x0-8,py+4);
      } else {
        const px = x(value);
        ctx.moveTo(px,y0); ctx.lineTo(px,y1);
        ctx.textAlign = "center"; ctx.fillText(String(value),px,y1+16);
      }
      ctx.stroke();
    }
    for (const value of ticks(xMin,xMax,w > 520 ? 7 : 4)) guide(value,false);
    for (const value of ticks(yMin,yMax,h > 350 ? 6 : 4)) guide(value,true);
    ctx.strokeStyle = "#c5d3cc"; ctx.strokeRect(x0,y0,x1-x0,y1-y0);
    ctx.fillStyle = "#334f43"; ctx.font = "600 13px Segoe UI"; ctx.textAlign = "center";
    ctx.fillText("Standardized value (z)", (x0+x1)/2, h-9);
    ctx.save(); ctx.translate(16,(y1+y0)/2); ctx.rotate(-Math.PI/2); ctx.fillText("Spatial lag (Wz)",0,0); ctx.restore();
    ctx.font = "12px Segoe UI"; ctx.fillStyle = "#5b6b61";
    if (x(0)-x0 > 72 && y(0)-y0 > 37) {ctx.textAlign = "left"; ctx.fillText("Low–high", x0+9, y0+16);}
    if (x(0)-x0 > 72 && y1-y(0) > 37) {ctx.textAlign = "left"; ctx.fillText("Low–low", x0+9, y1-10);}
    if (x1-x(0) > 72 && y(0)-y0 > 37) {ctx.textAlign = "right"; ctx.fillText("High–high", x1-9, y0+16);}
    if (x1-x(0) > 72 && y1-y(0) > 37) {ctx.textAlign = "right"; ctx.fillText("High–low", x1-9, y1-10);}
    const points = observations.map(([uid,d])=>[uid,x(d[0]),y(d[1])]);
    ctx.fillStyle = "#73598744";
    for (const [,px,py] of points) {
      ctx.beginPath();ctx.arc(px,py,w > 520 ? 2.1 : 1.65,0,Math.PI*2);ctx.fill();
    }
    const focal = points.find(p=>p[0]===selected);
    if (focal) {
      ctx.beginPath();ctx.arc(focal[1],focal[2],6,0,Math.PI*2);
      ctx.fillStyle = "#1b5747";ctx.fill();
      ctx.strokeStyle = "white";ctx.lineWidth = 2;ctx.stroke();
    }
    canvas.onclick = event => {
      const bound = canvas.getBoundingClientRect(), tx=(event.clientX-bound.left)*w/bound.width, ty=(event.clientY-bound.top)*h/bound.height;
      let best = 14**2, uid;
      for (const [key,px,py] of points) {
        const d = (px-tx)**2+(py-ty)**2;
        if (d < best) {best=d;uid=key;}
      }
      if (uid) onSelect(uid);
    };
  }

  AtlasLabs.social = async panel => {
    const map = setup(panel);
    const [summary, values, das] = await Promise.all([result("socioeconomic"),model("socioeconomic","values"),geo("shared","das")]);
    const names = Object.keys(summary.indicators);
    options(map.control("indicator"), names.map(k => [k,indicators[k][0]]), names.includes("unemployment") ? "unemployment" : "density");
    map.control("mode").querySelector('[value=composite]').disabled = summary.composite.length < 2;
    const weightsBox = panel.querySelector("[data-composite]");
    for (const field of summary.composite) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "range"; input.min = "0"; input.max = "10"; input.value = "5"; input.dataset.weight = field;
      const output = document.createElement("output");
      output.dataset.weightLabel = field;
      label.append(indicators[field][0],input,output); weightsBox.append(label);
    }
    const search = document.createElement("form");
    search.className = "toolbar";search.innerHTML = '<label>Find a DA by code<input name="uid" inputmode="numeric" placeholder="8-digit DA code" pattern="[0-9]{8}"></label><button>Locate</button>';
    panel.querySelector(".controls").append(search);
    const chart = document.querySelector('[data-chart="moran"]');
    chart.innerHTML = '<p class="plot-note"></p><canvas role="img" aria-label="Moran scatterplot with standardized value and spatial lag axes"></canvas>';
    const canvas = chart.querySelector("canvas");
    let observations;
    let selected, extent = "province", selectedLayer;
    const redraw = () => {
      if (observations) scatter(canvas, observations, selected, uid => locate(uid,false));
    };
    new ResizeObserver(redraw).observe(canvas);
    const refresh = map.listen(async keep => {
      const indicator = map.value("indicator"), mode = map.value("mode"), w = map.value("weights"), region = map.value("region");
      const inference = map.value("inference") === "raw" ? 3 : 4;
      weightsBox.hidden = mode !== "composite";
      map.control("inference").disabled = mode !== "clusters";
      map.control("inference").closest("label").hidden = mode !== "clusters";
      const rawWeights = [...weightsBox.querySelectorAll("[data-weight]")].map(el=>Number(el.value));
      const weightSum = rawWeights.reduce((a,b)=>a+b,0);
      const normalized = rawWeights.map(v=>weightSum ? v/weightSum : 1/rawWeights.length);
      [...weightsBox.querySelectorAll("output")].forEach((el,i)=>{el.value=fmt(normalized[i]*100,1)+"%";});
      const [label,unit,breaks] = indicators[indicator];
      const rows = await model("socioeconomic",indicator+"_"+w);
      if (!keep()) return;
      const stats = summary.indicators[indicator].weights[w];
      const neighbours = new Set(rows[selected]?.[6] || []);
      function value(uid) {return values[indicator][uid];}
      function color(p) {
        const uid = p.uid, r=rows[uid];
        if (uid===selected) return "#2a5a49";
        if (mode==="neighbours") return neighbours.has(uid) ? "#d19c69" : r ? "#eeebe4" : palette.missing;
        if (mode==="clusters") return r ? r[inference]<=.05 ? clusters[r[5]] || "#eae4ea" : "#eae4ea" : palette.missing;
        if (mode==="composite") {
          const components=values.components?.[uid];
          const score=components?.reduce((sum,v,i)=>sum+v*normalized[i],0);
          return classify(score,[-1,-.3,.3,1],palette.diverge);
        }
        return classify(value(uid),breaks,palette.social);
      }
      map.clear();
      selectedLayer = null;
      const layer=map.draw(das,{style:f=>choropleth(color(f.properties)),tip:p=>{
        const r=rows[p.uid], v=value(p.uid);
        const score=mode==="composite" ? values.components?.[p.uid]?.reduce((sum,n,i)=>sum+n*normalized[i],0) : null;
        return "DA "+esc(p.uid)+" · "+(v==null?"Unavailable":fmt(v,indicator==="income"?0:1)+" "+unit)+
          (score==null?"":" · composite "+fmt(score,2))+(mode==="clusters"&&r?" · "+esc(r[5])+" · p "+fmt(r[3],3)+" · q "+fmt(r[4],3):"");
      },select:(feature)=>{selected=feature.properties.uid;refresh();}});
      layer.eachLayer(l=>{if(l.feature.properties.uid===selected) selectedLayer=l;});
      if(selectedLayer) selectedLayer.bringToFront();
      if (extent !== region) {
        map.setHome(extents[region] || A.bounds, region === "province" ? "province" : "regional");
        extent = region;
      }
      const grade=breaks.map((cut,i)=> (i===0?"≤ ":fmt(breaks[i-1])+"–")+fmt(cut)).concat(["> "+fmt(breaks.at(-1))+" "+unit]);
      const legend=mode==="neighbours"?[["#d19c69","Selected DA’s neighbours"],["#eeebe4","Other areas in sample"]]:
        mode==="clusters"?[...Object.entries(clusters).map(([k,v])=>[v,k]),["#eae4ea","No significant local evidence"]]:
        mode==="composite"?swatches(palette.diverge,["≤ −1","−1 to −0.3","−0.3 to 0.3","0.3 to 1","> 1"]):swatches(palette.social,grade);
      map.setLegend([...legend,["#2a5a49","Selected DA"],[palette.missing,"Unavailable"]]);
      let detail=label+" · Moran’s I "+fmt(stats.I,3)+" · p "+fmt(stats.p,3)+" · "+fmt(stats.islands)+" islands.";
      if(selected) detail += " DA "+selected+(rows[selected]?" · "+neighbours.size+" neighbours · local I "+fmt(rows[selected][2],3)+".":" is outside this indicator’s sample.");
      if(mode==="composite") detail="Complete-case normalized composite. "+(selected?"Selected DA "+selected+".":"Select an area to inspect it.");
      map.setStatus(detail);
      observations = rows;
      chart.querySelector(".plot-note").textContent = label+" · "+fmt(Object.keys(rows).length)+" areas · global Moran’s I "+fmt(stats.I,3)+
        (rows[selected] ? " · Selected DA "+selected : "");
      redraw();
      if (search.dataset.locate && selectedLayer) {
        map.fit(selectedLayer.getBounds().pad(2));
        delete search.dataset.locate;
      }
    });
    function locate(uid, reflect=true){
      if (!das.features.some(f=>f.properties.uid===uid)) {
        map.setStatus("No DA in the Alberta boundary file has code "+uid+".");
        return;
      }
      if(reflect) search.elements.uid.value=uid;
      selected=uid;search.dataset.locate="true";refresh();
    }
    search.onsubmit=e=>{e.preventDefault();locate(search.elements.uid.value.trim());};
    for(const input of weightsBox.querySelectorAll("input")) input.addEventListener("input",refresh);
    return refresh();
  };
})();

