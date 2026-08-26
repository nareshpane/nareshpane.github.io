"use strict";

(() => {
  const colors = ["#1d6c9e", "#b26935", "#287352", "#a34f4f", "#6c5b99", "#a87524"];
  const regions = ["Visual", "Somatomotor", "Attention", "Limbic", "Control", "Default"];
  const matrix = [
    [1.00, 0.82, 0.44, -0.18, 0.12, -0.52],
    [0.82, 1.00, 0.57, -0.11, 0.26, -0.35],
    [0.44, 0.57, 1.00, 0.08, 0.61, -0.21],
    [-0.18, -0.11, 0.08, 1.00, 0.48, 0.31],
    [0.12, 0.26, 0.61, 0.48, 1.00, -0.64],
    [-0.52, -0.35, -0.21, 0.31, -0.64, 1.00]
  ];
  const positions = [[0.50, 0.08], [0.84, 0.30], [0.76, 0.72], [0.50, 0.92], [0.18, 0.70], [0.13, 0.27]];
  const svgNS = "http://www.w3.org/2000/svg";

  function svgElement(name, attributes = {}, text = "") {
    const element = document.createElementNS(svgNS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    if (text) element.textContent = text;
    return element;
  }

  function initVoxelGrid() {
    const values = [
      [12, 28, 43, 25, 9], [31, 72, 118, 83, 27], [52, 111, 146, 104, 41],
      [26, 78, 101, 69, 23], [8, 29, 38, 21, 6]
    ];
    const slice = document.getElementById("sliceGrid");
    const voxel = document.getElementById("voxelGrid");
    const table = document.getElementById("matrix5");
    const readout = document.getElementById("voxelReadout");
    values.flat().forEach((value, index) => {
      [slice, voxel].forEach(container => {
        const cell = document.createElement("span");
        cell.style.background = `rgb(${value + 70},${value + 65},${value + 55})`;
        cell.dataset.index = index;
        container.appendChild(cell);
      });
    });
    values.forEach((row, rowIndex) => {
      const tr = document.createElement("tr");
      row.forEach((value, colIndex) => {
        const td = document.createElement("td");
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = String(value);
        button.style.cssText = "border:0;border-radius:3px;padding:2px;background:transparent;color:inherit;font:inherit";
        button.addEventListener("click", () => select(rowIndex, colIndex, value));
        td.appendChild(button);
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    function select(row, col, value) {
      document.querySelectorAll("#voxelGrid span,#sliceGrid span").forEach(cell => cell.classList.remove("active"));
      document.querySelectorAll(`[data-index="${row * 5 + col}"]`).forEach(cell => cell.classList.add("active"));
      readout.textContent = `Selected row ${row + 1}, column ${col + 1}: intensity ${value}. In a volume, a third coordinate would identify the slice.`;
    }
    select(2, 2, values[2][2]);
  }

  function initParcellation() {
    const group = document.getElementById("parcelRegions");
    const labels = document.getElementById("parcelLabels");
    const centers = [[165,105],[270,90],[385,92],[500,90],[615,118],[175,190],[285,180],[400,175],[520,176],[625,194],[300,252],[475,250]];
    centers.forEach(([x, y], index) => {
      const width = index > 9 ? 190 : 125;
      const height = index > 9 ? 95 : 120;
      group.appendChild(svgElement("ellipse", {cx:x, cy:y, rx:width / 2, ry:height / 2, fill:colors[index % colors.length], opacity:"0.82", stroke:"#fffdf9", "stroke-width":"4"}));
      labels.appendChild(svgElement("text", {x, y:y + 5, "text-anchor":"middle", fill:"white", "font-family":"Inter, sans-serif", "font-size":"17", "font-weight":"700", class:"parcel-number"}, String(index + 1)));
    });
    const button = document.getElementById("parcelToggle");
    button.addEventListener("click", () => {
      const visible = button.getAttribute("aria-pressed") === "true";
      button.setAttribute("aria-pressed", String(!visible));
      labels.style.display = visible ? "none" : "block";
      button.textContent = visible ? "Show region labels" : "Hide region labels";
    });
  }

  const signalLength = 60;
  const baseSignal = Array.from({length: signalLength}, (_, i) => 0.74 * Math.sin(i * 0.25) + 0.28 * Math.sin(i * 0.71 + 0.4) + 0.13 * Math.cos(i * 1.47));
  const regionalSignals = [
    baseSignal,
    baseSignal.map((value, i) => 0.72 * value + 0.25 * Math.sin(i * 0.37 + 1.4)),
    baseSignal.map((_, i) => 0.65 * Math.sin(i * 0.18 + 2) + 0.22 * Math.cos(i * 0.93)),
    baseSignal.map((value, i) => -0.63 * value + 0.22 * Math.sin(i * 0.31))
  ];

  function plotLines(canvas, series, palette, marker = null, bounds = null) {
    const context = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const pad = {left:45, right:18, top:18, bottom:34};
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#fffdf9";
    context.fillRect(0, 0, width, height);
    context.strokeStyle = "#ddd5c7";
    context.lineWidth = 1;
    for (let line = 0; line <= 4; line++) {
      const y = pad.top + line * (height - pad.top - pad.bottom) / 4;
      context.beginPath(); context.moveTo(pad.left, y); context.lineTo(width - pad.right, y); context.stroke();
    }
    const all = series.flat();
    const low = bounds ? bounds[0] : Math.min(...all) - 0.12;
    const high = bounds ? bounds[1] : Math.max(...all) + 0.12;
    const xFor = index => pad.left + index * (width - pad.left - pad.right) / (series[0].length - 1);
    const yFor = value => pad.top + (high - value) * (height - pad.top - pad.bottom) / (high - low);
    series.forEach((values, seriesIndex) => {
      context.beginPath();
      values.forEach((value, index) => index ? context.lineTo(xFor(index), yFor(value)) : context.moveTo(xFor(index), yFor(value)));
      context.strokeStyle = palette[seriesIndex]; context.lineWidth = 3; context.stroke();
    });
    if (marker !== null) {
      const x = xFor(marker);
      context.strokeStyle = "#a87524"; context.lineWidth = 2; context.setLineDash([5, 4]);
      context.beginPath(); context.moveTo(x, pad.top); context.lineTo(x, height - pad.bottom); context.stroke(); context.setLineDash([]);
      context.fillStyle = "#a87524"; context.beginPath(); context.arc(x, yFor(series[0][marker]), 6, 0, 2 * Math.PI); context.fill();
    }
    context.fillStyle = "#67655f"; context.font = "14px Inter, sans-serif";
    context.fillText("time →", width - 70, height - 9);
  }

  function drawScan(time) {
    const canvas = document.getElementById("scanCanvas");
    const context = canvas.getContext("2d");
    const width = canvas.width, height = canvas.height;
    context.clearRect(0, 0, width, height); context.fillStyle = "#111"; context.fillRect(0, 0, width, height);
    const gradient = context.createRadialGradient(220, 155, 25, 220, 155, 142);
    gradient.addColorStop(0, "#3c3c3c"); gradient.addColorStop(0.38, "#c4c2bd"); gradient.addColorStop(0.72, "#77746f"); gradient.addColorStop(1, "#171717");
    context.save(); context.scale(1, 0.78); context.beginPath(); context.ellipse(220, 197, 157, 172, 0, 0, Math.PI * 2); context.fillStyle = gradient; context.fill(); context.restore();
    context.fillStyle = "#242424"; context.beginPath(); context.ellipse(190, 151, 20, 40, -0.1, 0, 2 * Math.PI); context.ellipse(250, 151, 20, 40, 0.1, 0, 2 * Math.PI); context.fill();
    const value = baseSignal[time];
    const light = Math.round(54 + (value + 1.3) / 2.6 * 42);
    context.beginPath(); context.arc(290, 115, 24, 0, 2 * Math.PI); context.fillStyle = `hsl(42 80% ${light}%)`; context.fill(); context.strokeStyle = "#fff"; context.lineWidth = 3; context.stroke();
    context.fillStyle = "#fff"; context.font = "14px Inter, sans-serif"; context.fillText(`selected region: x(v,t) = ${value.toFixed(2)}`, 18, 297);
  }

  function initSignals() {
    const slider = document.getElementById("timeSlider");
    const output = document.getElementById("timeOut");
    function update() {
      const time = Number(slider.value); output.value = String(time + 1); drawScan(time);
      plotLines(document.getElementById("boldCanvas"), [baseSignal], [colors[0]], time, [-1.4, 1.4]);
    }
    slider.addEventListener("input", update); update();
    document.querySelectorAll(".regionSignal").forEach(canvas => {
      const index = Number(canvas.dataset.signal); plotLines(canvas, [regionalSignals[index]], [colors[index]], null, [-1.4, 1.4]);
    });
  }

  function pearson(x, y) {
    const meanX = x.reduce((a, b) => a + b, 0) / x.length;
    const meanY = y.reduce((a, b) => a + b, 0) / y.length;
    let numerator = 0, sumX = 0, sumY = 0;
    x.forEach((value, i) => {const dx = value - meanX, dy = y[i] - meanY; numerator += dx * dy; sumX += dx * dx; sumY += dy * dy;});
    return numerator / Math.sqrt(sumX * sumY);
  }

  function initCorrelation() {
    const canvas = document.getElementById("correlationCanvas");
    const readout = document.getElementById("correlationReadout");
    const x = Array.from({length:60}, (_, i) => Math.sin(i * 0.27) + 0.24 * Math.sin(i * 0.83));
    const patterns = {
      positive: x.map((value, i) => 0.87 * value + 0.17 * Math.cos(i * 0.59)),
      zero: x.map((_, i) => Math.sin(i * 0.69 + 1.7) + 0.21 * Math.cos(i * 0.17)),
      negative: x.map((value, i) => -0.88 * value + 0.16 * Math.sin(i * 0.61))
    };
    function choose(name) {
      document.querySelectorAll("[data-corr]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.corr === name)));
      const value = pearson(x, patterns[name]);
      canvas.setAttribute("aria-label", `Two synthetic time series with Pearson correlation ${value.toFixed(3)}`);
      plotLines(canvas, [x, patterns[name]], [colors[0], colors[1]], null, [-1.5, 1.5]);
      const interpretation = value > .7 ? "rise and fall together" : value < -.7 ? "move in opposing directions" : "have little linear co-fluctuation";
      readout.innerHTML = `<strong>Calculated Pearson r = ${value.toFixed(3)}.</strong> These plotted signals ${interpretation}.`;
    }
    document.querySelectorAll("[data-corr]").forEach(button => button.addEventListener("click", () => choose(button.dataset.corr)));
    choose("positive");
  }

  function correlationColor(value) {
    const neutral = [247, 244, 238];
    const end = value >= 0 ? [178, 24, 43] : [33, 102, 172];
    const amount = Math.abs(value);
    return `rgb(${neutral.map((base, i) => Math.round(base + (end[i] - base) * amount)).join(",")})`;
  }

  function initMatrix() {
    const grid = document.getElementById("matrixGrid");
    const detail = document.getElementById("matrixDetail");
    const corner = document.createElement("div"); corner.className = "matrix-label"; grid.appendChild(corner);
    regions.forEach((_, i) => {const label = document.createElement("div"); label.className = "matrix-label"; label.textContent = String.fromCharCode(65 + i); grid.appendChild(label);});
    matrix.forEach((row, rowIndex) => {
      const label = document.createElement("div"); label.className = "matrix-label"; label.textContent = String.fromCharCode(65 + rowIndex); grid.appendChild(label);
      row.forEach((value, colIndex) => {
        const cell = document.createElement("div"); cell.className = "matrix-cell"; cell.tabIndex = 0; cell.setAttribute("role", "gridcell"); cell.textContent = value.toFixed(2);
        cell.style.background = correlationColor(value); cell.style.color = Math.abs(value) > .58 ? "white" : "#1b1b1b";
        const describe = () => {detail.innerHTML = `<strong>${String.fromCharCode(65 + rowIndex)} · ${regions[rowIndex]} × ${String.fromCharCode(65 + colIndex)} · ${regions[colIndex]}</strong><br>Correlation: ${value.toFixed(2)}${rowIndex === colIndex ? " (diagonal self-correlation)" : ""}`;};
        cell.addEventListener("mouseenter", describe); cell.addEventListener("focus", describe); cell.addEventListener("click", describe); grid.appendChild(cell);
      });
    });
  }

  function nodeCoordinates(index, width = 760, height = 390) {
    return [70 + positions[index][0] * (width - 140), 40 + positions[index][1] * (height - 80)];
  }

  function drawNetwork(svg, threshold, options = {}) {
    svg.replaceChildren();
    let count = 0;
    for (let i = 0; i < matrix.length; i++) for (let j = i + 1; j < matrix.length; j++) {
      const value = matrix[i][j];
      if (Math.abs(value) < threshold) continue;
      count++;
      const [x1, y1] = nodeCoordinates(i), [x2, y2] = nodeCoordinates(j);
      const importance = options.importance ? options.importance[`${i}-${j}`] || .2 : 1;
      const line = svgElement("line", {x1,y1,x2,y2,stroke:value >= 0 ? "#a34f4f" : "#1d6c9e","stroke-width":options.importance ? 1.5 + importance * 8 : 1.5 + Math.abs(value) * 6,opacity:options.importance ? .3 + importance * .7 : .74,"stroke-dasharray":value < 0 ? "9 6" : "none"});
      svg.appendChild(line);
    }
    regions.forEach((name, index) => {
      const [cx, cy] = nodeCoordinates(index);
      svg.appendChild(svgElement("circle", {cx,cy,r:"30",fill:colors[index],stroke:"#fffdf9","stroke-width":"5"}));
      svg.appendChild(svgElement("text", {x:cx,y:cy + 4,"text-anchor":"middle",fill:"white","font-family":"Inter, sans-serif","font-size":"13","font-weight":"700"}, String.fromCharCode(65 + index)));
      svg.appendChild(svgElement("text", {x:cx,y:cy + 51,"text-anchor":"middle",fill:"#333","font-family":"Inter, sans-serif","font-size":"13"}, name));
    });
    return count;
  }

  function initGraph() {
    const slider = document.getElementById("thresholdSlider"), output = document.getElementById("thresholdOut"), count = document.getElementById("edgeCount"), svg = document.getElementById("graphSvg");
    function update() {const threshold = Number(slider.value) / 100; output.value = threshold.toFixed(2); const edges = drawNetwork(svg, threshold); count.textContent = `${edges} of 15 possible edges retained`; svg.setAttribute("aria-label", `Six-node graph retaining ${edges} edges with absolute correlation at least ${threshold.toFixed(2)}`);}
    slider.addEventListener("input", update); update();
    drawNetwork(document.getElementById("importanceSvg"), 0.1, {importance:{"0-1":1,"0-5":.86,"2-4":.78,"4-5":.94,"3-4":.45}});
  }

  function initGNN() {
    const svg = document.getElementById("gnnSvg"), button = document.getElementById("gnnStep"), status = document.getElementById("gnnStatus");
    const target = 4, neighbors = [2, 3, 5];
    const messages = [
      "Step 1 of 5: choose the Control region as the receiving node.",
      "Step 2 of 5: identify its retained neighbors: Attention, Limbic, and Default.",
      "Step 3 of 5: send transformed features along weighted positive and negative edges.",
      "Step 4 of 5: sum the neighbor messages and combine them with Control's own features.",
      "Step 5 of 5: apply an activation to create Control's updated node representation. A later layer repeats the process."
    ];
    let step = 0;
    function draw() {
      svg.replaceChildren();
      for (let i = 0; i < matrix.length; i++) for (let j = i + 1; j < matrix.length; j++) {
        if (Math.abs(matrix[i][j]) < .4) continue;
        const [x1,y1] = nodeCoordinates(i), [x2,y2] = nodeCoordinates(j);
        const connected = (i === target && neighbors.includes(j)) || (j === target && neighbors.includes(i));
        const active = step >= 1 && connected;
        svg.appendChild(svgElement("line", {x1,y1,x2,y2,stroke:active ? "#a87524" : "#b8b2a8","stroke-width":active && step >= 2 ? "7" : "3",opacity:active ? "1" : ".45","stroke-dasharray":matrix[i][j] < 0 ? "9 6" : "none"}));
        if (active && step >= 2) {
          const from = i === target ? [x2,y2] : [x1,y1], to = nodeCoordinates(target);
          const tx = to[0] + (from[0] - to[0]) * .34, ty = to[1] + (from[1] - to[1]) * .34;
          svg.appendChild(svgElement("circle", {cx:tx,cy:ty,r:"8",fill:"#fffdf9",stroke:"#a87524","stroke-width":"4"}));
        }
      }
      regions.forEach((name, index) => {
        const [cx,cy] = nodeCoordinates(index), isTarget = index === target, neighbor = neighbors.includes(index);
        const highlighted = isTarget || (step >= 1 && neighbor);
        svg.appendChild(svgElement("circle", {cx,cy,r:isTarget && step === 4 ? "39" : "30",fill:colors[index],stroke:highlighted ? "#a87524" : "#fffdf9","stroke-width":highlighted ? "7" : "5"}));
        svg.appendChild(svgElement("text", {x:cx,y:cy + 4,"text-anchor":"middle",fill:"white","font-family":"Inter, sans-serif","font-size":"13","font-weight":"700"}, String.fromCharCode(65 + index)));
        svg.appendChild(svgElement("text", {x:cx,y:cy + 52,"text-anchor":"middle",fill:"#333","font-family":"Inter, sans-serif","font-size":"13"}, name));
      });
      if (step >= 3) svg.appendChild(svgElement("text", {x:"380",y:"28","text-anchor":"middle",fill:"#6d5426","font-family":"Inter, sans-serif","font-size":"15","font-weight":"700"}, step === 3 ? "aggregate: self + weighted neighbors" : "updated Control representation"));
      status.textContent = messages[step]; button.textContent = step === 4 ? "Restart layer" : "Next step";
    }
    button.addEventListener("click", () => {step = (step + 1) % messages.length; draw();}); draw();
  }

  function initProbability() {
    const slider = document.getElementById("scoreSlider"), score = document.getElementById("scoreOut"), equation = document.getElementById("sigmoidEquation"), fill = document.getElementById("probabilityFill"), text = document.getElementById("probabilityText");
    function update() {
      const z = Number(slider.value) / 100, probability = 1 / (1 + Math.exp(-z));
      score.value = z.toFixed(2); fill.style.width = `${probability * 100}%`; equation.textContent = `p = 1 / (1 + exp(-${z.toFixed(2)})) = ${probability.toFixed(3)}`; text.textContent = `Estimated class-1 probability: ${(probability * 100).toFixed(1)}%`;
    }
    slider.addEventListener("input", update); update();
  }

  function initSplits() {
    const people = document.getElementById("peopleSplit");
    for (let i = 0; i < 100; i++) {const person = document.createElement("span"); person.className = `person ${i < 70 ? "train" : i < 85 ? "validate" : "test"}`; person.setAttribute("aria-hidden", "true"); people.appendChild(person);}
    const diagram = document.getElementById("foldDiagram"), status = document.getElementById("foldStatus");
    let held = 0;
    function drawFolds() {
      diagram.replaceChildren();
      for (let row = 0; row < 5; row++) {
        const label = document.createElement("div"); label.className = "fold-label"; label.textContent = `Round ${row + 1}`; diagram.appendChild(label);
        for (let col = 0; col < 5; col++) {const cell = document.createElement("div"); const isHeld = col === (row + held) % 5; cell.className = isHeld ? "held" : ""; cell.textContent = isHeld ? "test" : "train"; diagram.appendChild(cell);}
      }
      status.textContent = `Rotation ${held + 1}: each row holds out a different subject fold.`;
    }
    document.getElementById("rotateFold").addEventListener("click", () => {held = (held + 1) % 5; drawFolds();}); drawFolds();
  }

  function initCalculator() {
    const ids = ["tpInput", "fnInput", "tnInput", "fpInput"], readout = document.getElementById("metricReadout");
    function ratio(a, b) {return b ? (a / b).toFixed(3) : "undefined";}
    function update() {
      const [tp, fn, tn, fp] = ids.map(id => Math.max(0, Number(document.getElementById(id).value) || 0));
      const total = tp + fn + tn + fp;
      readout.innerHTML = `<strong>Accuracy ${ratio(tp + tn, total)}</strong> · sensitivity ${ratio(tp, tp + fn)} · specificity ${ratio(tn, tn + fp)} · precision ${ratio(tp, tp + fp)} · false-positive rate ${ratio(fp, fp + tn)} · total n = ${total}`;
    }
    ids.forEach(id => document.getElementById(id).addEventListener("input", update)); update();
  }

  initVoxelGrid();
  initParcellation();
  initSignals();
  initCorrelation();
  initMatrix();
  initGraph();
  initGNN();
  initProbability();
  initSplits();
  initCalculator();
})();
