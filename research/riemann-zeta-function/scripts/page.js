"use strict";

(() => {
  const NS = "http://www.w3.org/2000/svg";
  const COLORS = {
    ink: "#1b1b1b",
    muted: "#66645f",
    line: "#ddd5c7",
    blue: "#1d4f91",
    orange: "#b26935",
    green: "#287352",
    red: "#a34f4f",
    violet: "#6c5b99",
    gold: "#a87524",
    paper: "#fffdf9"
  };
  const DATA_ROOT = "riemann-zeta-function/data/";
  const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199];

  function byId(id) {
    return document.getElementById(id);
  }

  function svgNode(name, attributes = {}, text = "") {
    const node = document.createElementNS(NS, name);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text) node.textContent = text;
    return node;
  }

  function resetSvg(svg, title, description) {
    svg.replaceChildren();
    svg.append(svgNode("title", {}, title), svgNode("desc", {}, description));
  }

  function format(value, digits = 6) {
    if (!Number.isFinite(value)) return "undefined";
    if (Math.abs(value) > 9999 || (Math.abs(value) > 0 && Math.abs(value) < 0.0001)) return value.toExponential(3);
    return value.toFixed(digits).replace(/\.0+$|(?<=\.[0-9]*?)0+$/u, "");
  }

  function plotFrame(svg, bounds, options = {}) {
    const width = 900;
    const height = options.height || 350;
    const pad = { left: 66, right: 24, top: 24, bottom: 48 };
    const x = value => pad.left + (value - bounds.xMin) / (bounds.xMax - bounds.xMin) * (width - pad.left - pad.right);
    const y = value => height - pad.bottom - (value - bounds.yMin) / (bounds.yMax - bounds.yMin) * (height - pad.top - pad.bottom);
    const ticks = options.ticks || 5;
    for (let index = 0; index <= ticks; index++) {
      const xValue = bounds.xMin + index / ticks * (bounds.xMax - bounds.xMin);
      const xPosition = x(xValue);
      svg.appendChild(svgNode("line", { x1: xPosition, y1: pad.top, x2: xPosition, y2: height - pad.bottom, class: "gridline" }));
      svg.appendChild(svgNode("text", { x: xPosition, y: height - 19, "text-anchor": "middle", class: "plot-label" }, options.xFormat ? options.xFormat(xValue) : format(xValue, 1)));
      const yValue = bounds.yMin + index / ticks * (bounds.yMax - bounds.yMin);
      const yPosition = y(yValue);
      svg.appendChild(svgNode("line", { x1: pad.left, y1: yPosition, x2: width - pad.right, y2: yPosition, class: "gridline" }));
      svg.appendChild(svgNode("text", { x: pad.left - 9, y: yPosition + 4, "text-anchor": "end", class: "plot-label" }, options.yFormat ? options.yFormat(yValue) : format(yValue, 1)));
    }
    svg.appendChild(svgNode("line", { x1: pad.left, y1: height - pad.bottom, x2: width - pad.right, y2: height - pad.bottom, class: "axis" }));
    svg.appendChild(svgNode("line", { x1: pad.left, y1: pad.top, x2: pad.left, y2: height - pad.bottom, class: "axis" }));
    if (options.xLabel) svg.appendChild(svgNode("text", { x: width / 2, y: height - 3, "text-anchor": "middle", class: "plot-label" }, options.xLabel));
    if (options.yLabel) svg.appendChild(svgNode("text", { x: 15, y: height / 2, transform: `rotate(-90 15 ${height / 2})`, "text-anchor": "middle", class: "plot-label" }, options.yLabel));
    return { x, y, width, height, pad };
  }

  function pathFrom(points, x, y, step = false) {
    if (!points.length) return "";
    let path = `M ${x(points[0][0]).toFixed(2)} ${y(points[0][1]).toFixed(2)}`;
    for (let index = 1; index < points.length; index++) {
      if (step) path += ` H ${x(points[index][0]).toFixed(2)}`;
      path += ` L ${x(points[index][0]).toFixed(2)} ${y(points[index][1]).toFixed(2)}`;
    }
    return path;
  }

  function acceleratedZeta(s, terms = 1200) {
    let sum = 0;
    for (let n = 1; n <= terms; n++) sum += n ** (-s);
    const N = terms;
    const tail = N ** (1 - s) / (s - 1) - 0.5 * N ** (-s) + s / 12 * N ** (-s - 1) - s * (s + 1) * (s + 2) / 720 * N ** (-s - 3);
    return sum + tail;
  }

  function partialSum(p, terms) {
    let total = 0;
    for (let n = 1; n <= terms; n++) total += n ** (-p);
    return total;
  }

  function sampledIndices(length, count) {
    if (length <= count) return Array.from({ length }, (_, index) => index);
    const result = [];
    for (let index = 0; index < count; index++) result.push(Math.round(index * (length - 1) / (count - 1)));
    return [...new Set(result)];
  }

  function initConvergence() {
    const pSlider = byId("pSlider");
    const nSlider = byId("nSlider");
    const svg = byId("convergenceSvg");
    function update() {
      const p = Number(pSlider.value) / 100;
      const N = Number(nSlider.value);
      byId("pOutput").value = p.toFixed(2);
      byId("nOutput").value = String(N);
      document.querySelectorAll("[data-preset-p]").forEach(button => button.setAttribute("aria-pressed", String(Math.abs(Number(button.dataset.presetP) - p) < 0.006)));
      const values = new Float64Array(N);
      let sum = 0;
      for (let n = 1; n <= N; n++) {
        sum += n ** (-p);
        values[n - 1] = sum;
      }
      const earlierIndex = Math.max(0, Math.floor(N * 0.9) - 1);
      const change = sum - values[earlierIndex];
      byId("partialOutput").textContent = format(sum, 8);
      byId("lastTermOutput").textContent = format(N ** (-p), 8);
      byId("tailChangeOutput").textContent = format(change, 8);
      byId("pVerdict").textContent = p > 1 ? "converges" : "diverges";
      const comparison = p > 1 ? acceleratedZeta(p, 1600) : null;
      byId("convergenceStatus").textContent = p > 1
        ? `The theorem says the infinite series converges. At N = ${N}, the partial sum is ${format(sum, 6)}; the limiting value is about ${format(comparison, 6)}.`
        : `The theorem says the infinite series diverges for p = ${p.toFixed(2)}. A finite graph can show continued growth, but cannot prove divergence by itself.`;

      resetSvg(svg, "Partial sums of a p-series", `Partial sums through ${N} terms for exponent ${p.toFixed(2)}.`);
      const indices = sampledIndices(N, 300);
      const points = indices.map(index => [Math.log10(index + 1), values[index]]);
      const yMin = Math.min(0.95, sum * 0.92);
      const yMax = Math.max(1.2, sum * 1.07, comparison ? comparison * 1.04 : 0);
      const frame = plotFrame(svg, { xMin: 0, xMax: Math.max(1, Math.log10(N)), yMin, yMax }, { xLabel: "number of terms N (logarithmic scale)", yLabel: "partial sum", xFormat: value => `10^${value.toFixed(1)}` });
      svg.appendChild(svgNode("path", { d: pathFrom(points, frame.x, frame.y), class: "series-primary" }));
      if (comparison) {
        const yTarget = frame.y(comparison);
        svg.appendChild(svgNode("line", { x1: frame.pad.left, y1: yTarget, x2: frame.width - frame.pad.right, y2: yTarget, stroke: COLORS.green, "stroke-width": 2, "stroke-dasharray": "8 6" }));
        svg.appendChild(svgNode("text", { x: frame.width - frame.pad.right - 4, y: yTarget - 7, "text-anchor": "end", fill: COLORS.green, class: "plot-label" }, "limiting value"));
      }
      svg.appendChild(svgNode("circle", { cx: frame.x(Math.log10(N)), cy: frame.y(sum), r: 6, class: "marker" }));
    }
    pSlider.addEventListener("input", update);
    nSlider.addEventListener("input", update);
    document.querySelectorAll("[data-preset-p]").forEach(button => button.addEventListener("click", () => {
      pSlider.value = String(Math.round(Number(button.dataset.presetP) * 100));
      update();
    }));
    update();
  }

  function initRealZeta() {
    const sSlider = byId("zetaSSlider");
    const termsSlider = byId("zetaTermsSlider");
    const svg = byId("realZetaSvg");
    const graphPoints = [];
    for (let index = 0; index <= 180; index++) {
      const s = 1.05 + index / 180 * 3.95;
      graphPoints.push([s, acceleratedZeta(s, 1000)]);
    }
    function update() {
      const s = Number(sSlider.value) / 100;
      const terms = Number(termsSlider.value);
      const finite = partialSum(s, terms);
      const target = acceleratedZeta(s, Math.max(1400, Math.min(terms, 3000)));
      byId("zetaSOutput").value = s.toFixed(2);
      byId("zetaTermsOutput").value = String(terms);
      byId("zetaPartialOutput").textContent = format(finite, 9);
      byId("zetaTargetOutput").textContent = format(target, 9);
      byId("zetaTailOutput").textContent = format(target - finite, 8);
      byId("zetaPoleDistance").textContent = (s - 1).toFixed(2);
      byId("realZetaStatus").textContent = s < 1.2
        ? `Only ${s - 1 < 0.1 ? "a very small" : "a small"} distance separates s from the pole at 1, so convergence is extremely slow.`
        : `For s = ${s.toFixed(2)}, terms decay like n^(-${s.toFixed(2)}), and the finite sum approaches a value near ${format(target, 7)}.`;
      resetSvg(svg, "Zeta on the real axis", "The zeta function from real s equals 1.05 to 5, with the selected point marked.");
      const frame = plotFrame(svg, { xMin: 1.05, xMax: 5, yMin: 1, yMax: 21 }, { xLabel: "real s", yLabel: "zeta(s)", yFormat: value => value.toFixed(0) });
      svg.appendChild(svgNode("path", { d: pathFrom(graphPoints, frame.x, frame.y), class: "series-primary" }));
      svg.appendChild(svgNode("line", { x1: frame.x(1.05), y1: frame.pad.top, x2: frame.x(1.05), y2: frame.height - frame.pad.bottom, stroke: COLORS.red, "stroke-dasharray": "6 5" }));
      svg.appendChild(svgNode("circle", { cx: frame.x(s), cy: frame.y(target), r: 7, class: "marker" }));
      svg.appendChild(svgNode("text", { x: frame.x(s) + 10, y: Math.max(frame.pad.top + 12, frame.y(target) - 9), fill: COLORS.ink, class: "plot-label" }, `s=${s.toFixed(2)}`));
    }
    sSlider.addEventListener("input", update);
    termsSlider.addEventListener("input", update);
    update();
  }

  function initEulerProduct() {
    const sSlider = byId("eulerSSlider");
    const countSlider = byId("primeCountSlider");
    const svg = byId("eulerSvg");
    function update() {
      const s = Number(sSlider.value) / 100;
      const count = Number(countSlider.value);
      const products = [];
      let product = 1;
      for (let index = 0; index < count; index++) {
        product *= 1 / (1 - primes[index] ** (-s));
        products.push([index + 1, product]);
      }
      const target = acceleratedZeta(s, 1500);
      byId("eulerSOutput").value = s.toFixed(2);
      byId("primeCountOutput").value = String(count);
      byId("eulerProductOutput").textContent = format(product, 9);
      byId("eulerTargetOutput").textContent = format(target, 9);
      byId("eulerDifferenceOutput").textContent = format(target - product, 8);
      byId("largestPrimeOutput").textContent = String(primes[count - 1]);
      const factors = byId("factorList");
      factors.replaceChildren(...primes.slice(0, count).map(prime => {
        const span = document.createElement("span");
        span.textContent = `(1-${prime}^(-s))^(-1)`;
        return span;
      }));
      resetSvg(svg, "Finite Euler products", `Products using the first ${count} primes for s equals ${s.toFixed(2)}.`);
      const yMin = 0.98;
      const yMax = target * 1.05;
      const frame = plotFrame(svg, { xMin: 1, xMax: Math.max(2, count), yMin, yMax }, { height: 280, xLabel: "number of prime factors", yLabel: "finite product" });
      svg.appendChild(svgNode("path", { d: pathFrom(products, frame.x, frame.y), class: "series-primary" }));
      const targetY = frame.y(target);
      svg.appendChild(svgNode("line", { x1: frame.pad.left, y1: targetY, x2: frame.width - frame.pad.right, y2: targetY, stroke: COLORS.green, "stroke-width": 2, "stroke-dasharray": "8 6" }));
      svg.appendChild(svgNode("text", { x: frame.width - frame.pad.right, y: targetY - 7, "text-anchor": "end", fill: COLORS.green, class: "plot-label" }, "zeta target"));
    }
    sSlider.addEventListener("input", update);
    countSlider.addEventListener("input", update);
    update();
  }

  function initVectors() {
    const sigmaSlider = byId("sigmaSlider");
    const tSlider = byId("tSlider");
    const termsSlider = byId("vectorTermsSlider");
    const svg = byId("vectorSvg");
    function update() {
      const sigma = Number(sigmaSlider.value) / 100;
      const t = Number(tSlider.value) / 100;
      const terms = Number(termsSlider.value);
      byId("sigmaOutput").value = sigma.toFixed(2);
      byId("tOutput").value = t.toFixed(2);
      byId("vectorTermsOutput").value = String(terms);
      const points = [[0, 0]];
      let real = 0;
      let imaginary = 0;
      for (let n = 1; n <= terms; n++) {
        const magnitude = n ** (-sigma);
        const angle = -t * Math.log(n);
        real += magnitude * Math.cos(angle);
        imaginary += magnitude * Math.sin(angle);
        points.push([real, imaginary]);
      }
      byId("vectorRealOutput").textContent = format(real, 6);
      byId("vectorImagOutput").textContent = format(imaginary, 6);
      byId("vectorMagnitudeOutput").textContent = format(Math.hypot(real, imaginary), 6);
      byId("vectorDomainOutput").textContent = sigma > 1 ? "series converges" : "partial sum only";
      byId("vectorStatus").textContent = sigma > 1
        ? `Because sigma = ${sigma.toFixed(2)} > 1, these partial sums approach zeta(s) as more arrows are added.`
        : `Because sigma = ${sigma.toFixed(2)} <= 1, the arrows show a finite Dirichlet partial sum, not the analytically continued zeta value.`;
      resetSvg(svg, "Complex zeta summand vectors", `${terms} head-to-tail vectors for sigma ${sigma.toFixed(2)} and t ${t.toFixed(2)}.`);
      const maxCoordinate = Math.max(1.1, ...points.flatMap(point => point.map(Math.abs))) * 1.15;
      const frame = plotFrame(svg, { xMin: -maxCoordinate, xMax: maxCoordinate, yMin: -maxCoordinate, yMax: maxCoordinate }, { height: 430, xLabel: "real part", yLabel: "imaginary part" });
      svg.appendChild(svgNode("line", { x1: frame.x(0), y1: frame.pad.top, x2: frame.x(0), y2: frame.height - frame.pad.bottom, stroke: COLORS.ink, opacity: 0.55 }));
      svg.appendChild(svgNode("line", { x1: frame.pad.left, y1: frame.y(0), x2: frame.width - frame.pad.right, y2: frame.y(0), stroke: COLORS.ink, opacity: 0.55 }));
      for (let index = 1; index < points.length; index++) {
        svg.appendChild(svgNode("line", { x1: frame.x(points[index - 1][0]), y1: frame.y(points[index - 1][1]), x2: frame.x(points[index][0]), y2: frame.y(points[index][1]), stroke: index % 2 ? COLORS.blue : COLORS.violet, "stroke-width": Math.max(1.2, 3 - index / terms), opacity: 0.77 }));
      }
      svg.appendChild(svgNode("line", { x1: frame.x(0), y1: frame.y(0), x2: frame.x(real), y2: frame.y(imaginary), stroke: COLORS.gold, "stroke-width": 5, opacity: 0.85 }));
      svg.appendChild(svgNode("circle", { cx: frame.x(real), cy: frame.y(imaginary), r: 7, fill: COLORS.red, stroke: "white", "stroke-width": 2 }));
    }
    [sigmaSlider, tSlider, termsSlider].forEach(control => control.addEventListener("input", update));
    update();
  }

  function initDomainButtons() {
    const messages = {
      dirichlet: "The Dirichlet series converges absolutely for Re(s) > 1.",
      eta: "The eta series converges for Re(s) > 0, but the quotient needs care where 1 - 2^(1-s) vanishes.",
      continued: "The unique meromorphic continuation covers the complex plane except for zeta's simple pole at s = 1."
    };
    document.querySelectorAll("[data-domain]").forEach(button => button.addEventListener("click", () => {
      document.querySelectorAll("[data-domain]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
      byId("domainStatus").textContent = messages[button.dataset.domain];
    }));
  }

  async function fetchJson(name) {
    const response = await fetch(`${DATA_ROOT}${name}`);
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return response.json();
  }

  function initZeroExplorer(data) {
    const ordinates = data.ordinates;
    const heightSlider = byId("zeroHeightSlider");
    const viewToggle = byId("zeroViewToggle");
    const toggle = byId("hypotheticalToggle");
    const svg = byId("zeroSvg");
    let hypothetical = false;
    let wideView = false;
    function draw() {
      const maxHeight = Number(heightSlider.value);
      const shown = ordinates.filter(value => value <= maxHeight);
      byId("zeroHeightOutput").value = String(maxHeight);
      byId("zerosShownOutput").textContent = String(shown.length);
      byId("zeroStatus").textContent = `Showing ${shown.length} positive and ${shown.length} conjugate negative zeros through |t| = ${maxHeight}${wideView ? ", together with the trivial zeros -2, -4, and -6" : ""}. Dataset values lie on beta = 1/2.`;
      resetSvg(svg, "Critical strip and zeta zeros", `Critical strip through height ${maxHeight}, with ${shown.length} listed zeros in each half-plane.`);
      const width = 900, height = 470, left = 74, right = 26, top = 24, bottom = 45;
      const xMinimum = wideView ? -6.5 : -0.25;
      const xMaximum = 1.25;
      const x = value => left + (value - xMinimum) / (xMaximum - xMinimum) * (width - left - right);
      const y = value => top + (maxHeight - value) / (2 * maxHeight) * (height - top - bottom);
      svg.appendChild(svgNode("rect", { x: x(0), y: top, width: x(1) - x(0), height: height - top - bottom, fill: "#eaf1f8" }));
      const xTicks = wideView ? [-6, -4, -2, 0, 0.5, 1] : [-0.25, 0, 0.5, 1, 1.25];
      xTicks.forEach(value => {
        svg.appendChild(svgNode("line", { x1: x(value), y1: top, x2: x(value), y2: height - bottom, class: "gridline" }));
        svg.appendChild(svgNode("text", { x: x(value), y: height - 18, "text-anchor": "middle", class: "plot-label" }, value.toFixed(2)));
      });
      [-maxHeight, -maxHeight / 2, 0, maxHeight / 2, maxHeight].forEach(value => {
        svg.appendChild(svgNode("line", { x1: left, y1: y(value), x2: width - right, y2: y(value), class: "gridline" }));
        svg.appendChild(svgNode("text", { x: left - 10, y: y(value) + 4, "text-anchor": "end", class: "plot-label" }, value.toFixed(0)));
      });
      svg.appendChild(svgNode("line", { x1: x(0.5), y1: top, x2: x(0.5), y2: height - bottom, stroke: COLORS.blue, "stroke-width": 3, "stroke-dasharray": "8 6" }));
      svg.appendChild(svgNode("text", { x: x(0.5) + 8, y: top + 17, fill: COLORS.blue, class: "plot-title" }, "critical line"));
      shown.forEach((gamma, index) => {
        [gamma, -gamma].forEach(value => svg.appendChild(svgNode("circle", { cx: x(0.5), cy: y(value), r: index < 5 ? 6 : 4.2, fill: COLORS.red, stroke: "white", "stroke-width": 1.5 })));
      });
      if (wideView) {
        [-2, -4, -6].forEach(value => {
          svg.appendChild(svgNode("circle", { cx: x(value), cy: y(0), r: 7, fill: COLORS.green, stroke: "white", "stroke-width": 1.5 }));
          svg.appendChild(svgNode("text", { x: x(value), y: y(0) + 23, "text-anchor": "middle", fill: COLORS.green, class: "plot-title" }, String(value)));
        });
        svg.appendChild(svgNode("text", { x: x(-4), y: y(0) - 14, "text-anchor": "middle", fill: COLORS.green, class: "plot-title" }, "trivial zeros"));
      }
      if (hypothetical) {
        const gamma = maxHeight * 0.63;
        [[0.72, gamma], [0.28, gamma], [0.72, -gamma], [0.28, -gamma]].forEach(([beta, ordinate]) => svg.appendChild(svgNode("rect", { x: x(beta) - 6, y: y(ordinate) - 6, width: 12, height: 12, fill: COLORS.gold, stroke: COLORS.ink, "stroke-width": 1.5 })));
        svg.appendChild(svgNode("text", { x: x(0.72) + 10, y: y(gamma) - 10, fill: COLORS.ink, class: "plot-title" }, "hypothetical quartet"));
      }
      svg.appendChild(svgNode("text", { x: width / 2, y: height - 2, "text-anchor": "middle", class: "plot-label" }, "real part beta"));
      svg.appendChild(svgNode("text", { x: 16, y: height / 2, transform: `rotate(-90 16 ${height / 2})`, "text-anchor": "middle", class: "plot-label" }, "imaginary part gamma"));
    }
    heightSlider.addEventListener("input", draw);
    viewToggle.addEventListener("click", () => {
      wideView = !wideView;
      viewToggle.setAttribute("aria-pressed", String(wideView));
      viewToggle.textContent = wideView ? "Zoom in to critical strip" : "Zoom out to trivial zeros";
      draw();
    });
    toggle.addEventListener("click", () => {
      hypothetical = !hypothetical;
      toggle.setAttribute("aria-pressed", String(hypothetical));
      toggle.textContent = hypothetical ? "Hide hypothetical off-line quartet" : "Show hypothetical off-line quartet";
      draw();
    });
    draw();
  }

  function initPrimeCounting(data) {
    const slider = byId("primeMaxSlider");
    const svg = byId("primeCountingSvg");
    function draw() {
      const maxX = Number(slider.value);
      byId("primeMaxOutput").value = String(maxX);
      const rows = data.rows.filter(row => row.x <= maxX);
      const yMax = Math.max(...rows.map(row => row.li)) * 1.08;
      resetSvg(svg, "Prime-counting staircase", `Exact prime counts and smooth approximations through ${maxX}.`);
      const frame = plotFrame(svg, { xMin: 2, xMax: maxX, yMin: 0, yMax }, { xLabel: "x", yLabel: "count or approximation", yFormat: value => value.toFixed(0) });
      const exact = rows.map(row => [row.x, row.pi]);
      const elementary = rows.map(row => [row.x, row.x_over_log_x]);
      const li = rows.map(row => [row.x, row.li]);
      svg.appendChild(svgNode("path", { d: pathFrom(exact, frame.x, frame.y, true), fill: "none", stroke: COLORS.ink, "stroke-width": 2.2 }));
      svg.appendChild(svgNode("path", { d: pathFrom(elementary, frame.x, frame.y), class: "series-secondary" }));
      svg.appendChild(svgNode("path", { d: pathFrom(li, frame.x, frame.y), class: "series-primary" }));
    }
    slider.addEventListener("input", draw);
    draw();
  }

  function initExplicit(primeData, explicitData) {
    const svg = byId("explicitSvg");
    let pairs = 30;
    function draw() {
      const exact = primeData.rows.map(row => [row.x, row.psi0]);
      const approximation = explicitData.x.map((x, index) => [x, explicitData.approximations[String(pairs)][index]]);
      const allY = [...exact.map(point => point[1]), ...approximation.map(point => point[1])];
      resetSvg(svg, "Explicit formula truncation", `Chebyshev psi compared with an approximation using ${pairs} conjugate zero pairs.`);
      const frame = plotFrame(svg, { xMin: 2, xMax: 500, yMin: Math.min(...allY) - 5, yMax: Math.max(...allY) + 5 }, { xLabel: "x", yLabel: "psi and approximation", yFormat: value => value.toFixed(0) });
      svg.appendChild(svgNode("path", { d: pathFrom(exact, frame.x, frame.y, true), fill: "none", stroke: COLORS.ink, "stroke-width": 1.8, opacity: 0.8 }));
      svg.appendChild(svgNode("path", { d: pathFrom(approximation, frame.x, frame.y), class: "series-primary" }));
      svg.appendChild(svgNode("text", { x: frame.pad.left + 10, y: frame.pad.top + 18, fill: COLORS.ink, class: "plot-label" }, "black staircase: exact midpoint psi_0(x)"));
      byId("explicitStatus").textContent = pairs === 0
        ? "Showing only the smooth pole term and elementary corrections; no nontrivial zero pair is included."
        : `Showing ${pairs} conjugate zero pair${pairs === 1 ? "" : "s"}. More pairs add finer oscillations but do not make this finite curve the full exact formula.`;
      document.querySelectorAll("[data-zero-pairs]").forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.zeroPairs) === pairs)));
    }
    document.querySelectorAll("[data-zero-pairs]").forEach(button => button.addEventListener("click", () => {
      pairs = Number(button.dataset.zeroPairs);
      draw();
    }));
    draw();
  }

  function initZFunction(data) {
    const slider = byId("zTSlider");
    const svg = byId("zFunctionSvg");
    const values = data.z;
    const ts = data.t;
    function drawBase(selectedIndex) {
      const yAbs = Math.max(...values.map(Math.abs)) * 1.08;
      resetSvg(svg, "Hardy Z function", "Computed Hardy Z samples from zero through seventy, with listed zeros and selected t marked.");
      const frame = plotFrame(svg, { xMin: 0, xMax: 70, yMin: -yAbs, yMax: yAbs }, { xLabel: "t", yLabel: "Z(t)" });
      svg.appendChild(svgNode("line", { x1: frame.pad.left, y1: frame.y(0), x2: frame.width - frame.pad.right, y2: frame.y(0), stroke: COLORS.ink, "stroke-width": 1.5 }));
      const points = sampledIndices(ts.length, 800).map(index => [ts[index], values[index]]);
      svg.appendChild(svgNode("path", { d: pathFrom(points, frame.x, frame.y), class: "series-primary" }));
      data.known_zeros_in_range.forEach((gamma, index) => {
        svg.appendChild(svgNode("circle", { cx: frame.x(gamma), cy: frame.y(0), r: 4.5, fill: COLORS.red, stroke: "white", "stroke-width": 1.2 }));
        if (index < 3) svg.appendChild(svgNode("text", { x: frame.x(gamma), y: frame.y(0) - 10, "text-anchor": "middle", fill: COLORS.red, class: "plot-label" }, gamma.toFixed(3)));
      });
      const t = ts[selectedIndex], value = values[selectedIndex];
      svg.appendChild(svgNode("line", { x1: frame.x(t), y1: frame.pad.top, x2: frame.x(t), y2: frame.height - frame.pad.bottom, stroke: COLORS.gold, "stroke-width": 2, "stroke-dasharray": "5 4" }));
      svg.appendChild(svgNode("circle", { cx: frame.x(t), cy: frame.y(value), r: 6, class: "marker" }));
    }
    function update() {
      const index = Number(slider.value);
      const t = ts[index];
      const value = values[index];
      const nearest = data.known_zeros_in_range.reduce((best, zero) => Math.abs(zero - t) < Math.abs(best - t) ? zero : best, data.known_zeros_in_range[0]);
      byId("zTOutput").value = t.toFixed(2);
      byId("zValueOutput").textContent = format(value, 7);
      byId("nearestZeroOutput").textContent = nearest.toFixed(9);
      byId("zeroDistanceOutput").textContent = Math.abs(nearest - t).toFixed(5);
      byId("zSignOutput").textContent = Math.abs(value) < 1e-8 ? "near zero" : value > 0 ? "positive" : "negative";
      byId("zFunctionStatus").textContent = `At sampled t = ${t.toFixed(2)}, Z(t) = ${format(value, 7)}. The nearest separately computed root is gamma = ${nearest.toFixed(9)}.`;
      drawBase(index);
    }
    slider.max = String(ts.length - 1);
    slider.addEventListener("input", update);
    update();
  }

  function nearestIndex(values, target) {
    let low = 0, high = values.length - 1;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (values[middle] < target) low = middle + 1;
      else high = middle;
    }
    if (low > 0 && Math.abs(values[low - 1] - target) < Math.abs(values[low] - target)) return low - 1;
    return low;
  }

  function initZeroCount(data) {
    const slider = byId("countTSlider");
    const mainSvg = byId("zeroCountSvg");
    const residualSvg = byId("zeroResidualSvg");
    const maximum = data.maximum_certified_by_dataset;
    slider.max = String(Math.floor(maximum));
    function draw() {
      const selectedT = Number(slider.value);
      const end = nearestIndex(data.t, selectedT);
      const exact = data.exact[end];
      const smooth = data.smooth[end];
      const residual = data.residual[end];
      byId("countTOutput").value = selectedT.toFixed(0);
      byId("exactCountOutput").textContent = String(exact);
      byId("smoothCountOutput").textContent = format(smooth, 3);
      byId("countResidualOutput").textContent = format(residual, 3);
      byId("meanGapOutput").textContent = selectedT > 2 * Math.PI ? format(2 * Math.PI / Math.log(selectedT / (2 * Math.PI)), 3) : "not useful yet";
      const indices = sampledIndices(end + 1, 600);
      const exactPoints = indices.map(index => [data.t[index], data.exact[index]]);
      const smoothPoints = indices.map(index => [data.t[index], data.smooth[index]]);
      const yMax = Math.max(10, ...exactPoints.map(point => point[1])) * 1.06;
      resetSvg(mainSvg, "Zero-count staircase", `Included zero count and smooth approximation through T equals ${selectedT}.`);
      const plotMaximum = Math.max(20, selectedT);
      const frame = plotFrame(mainSvg, { xMin: 10, xMax: plotMaximum, yMin: 0, yMax }, { xLabel: "height T", yLabel: "N(T)", yFormat: value => value.toFixed(0) });
      mainSvg.appendChild(svgNode("path", { d: pathFrom(exactPoints, frame.x, frame.y, true), fill: "none", stroke: COLORS.ink, "stroke-width": 2.1 }));
      mainSvg.appendChild(svgNode("path", { d: pathFrom(smoothPoints, frame.x, frame.y), class: "series-primary" }));
      mainSvg.appendChild(svgNode("text", { x: frame.pad.left + 8, y: frame.pad.top + 17, class: "plot-label" }, "black: included count; blue: smooth main term"));

      const residualPoints = indices.map(index => [data.t[index], data.residual[index]]);
      const residualAbs = Math.max(2, ...residualPoints.map(point => Math.abs(point[1]))) * 1.08;
      resetSvg(residualSvg, "Zero-count residual", "Difference between the included count and smooth main term.");
      const residualFrame = plotFrame(residualSvg, { xMin: 10, xMax: plotMaximum, yMin: -residualAbs, yMax: residualAbs }, { height: 230, xLabel: "height T", yLabel: "N(T) - main term" });
      residualSvg.appendChild(svgNode("line", { x1: residualFrame.pad.left, y1: residualFrame.y(0), x2: residualFrame.width - residualFrame.pad.right, y2: residualFrame.y(0), stroke: COLORS.ink }));
      residualSvg.appendChild(svgNode("path", { d: pathFrom(residualPoints, residualFrame.x, residualFrame.y), class: "series-red" }));
    }
    slider.addEventListener("input", draw);
    draw();
  }

  function histogram(values, bins, maximum = 3.2) {
    const counts = Array(bins).fill(0);
    const width = maximum / bins;
    values.forEach(value => {
      const index = Math.floor(value / width);
      if (index >= 0 && index < bins) counts[index]++;
    });
    const density = counts.map(count => count / values.length / width);
    return density.map((value, index) => [(index + 0.5) * width, value, width]);
  }

  function drawHistogram(svg, series, bins, title, description) {
    const allHistograms = series.map(item => ({ ...item, values: histogram(item.data, bins) }));
    const yMax = Math.max(...allHistograms.flatMap(item => item.values.map(row => row[1])), 0.5) * 1.12;
    resetSvg(svg, title, description);
    const frame = plotFrame(svg, { xMin: 0, xMax: 3.2, yMin: 0, yMax }, { xLabel: "normalized consecutive gap", yLabel: "density", yFormat: value => value.toFixed(1) });
    allHistograms.forEach((item, seriesIndex) => {
      const offset = series.length > 1 ? (seriesIndex - (series.length - 1) / 2) * item.values[0][2] * 0.18 : 0;
      item.values.forEach(([center, density, width]) => {
        const x1 = frame.x(center - width * 0.42 + offset);
        const x2 = frame.x(center + width * 0.42 + offset);
        svg.appendChild(svgNode("rect", { x: x1, y: frame.y(density), width: Math.max(1, x2 - x1), height: frame.y(0) - frame.y(density), fill: item.color, opacity: series.length > 1 ? 0.5 : 0.72, stroke: item.color, "stroke-width": 1 }));
      });
    });
  }

  function initSpacing(data) {
    const slider = byId("spacingBinsSlider");
    const svg = byId("spacingSvg");
    function draw() {
      const bins = Number(slider.value);
      byId("spacingBinsOutput").value = String(bins);
      byId("spacingSampleOutput").textContent = `${data.sample_size} gaps`;
      byId("spacingRangeOutput").textContent = `${data.height_range[0].toFixed(1)} to ${data.height_range[1].toFixed(1)}`;
      byId("spacingMeanOutput").textContent = format(data.normalized_gaps.reduce((a, b) => a + b, 0) / data.normalized_gaps.length, 4);
      drawHistogram(svg, [{ data: data.normalized_gaps, color: COLORS.blue }], bins, "Normalized zeta-zero gaps", `${data.sample_size} finite normalized gaps in ${bins} bins.`);
    }
    slider.addEventListener("input", draw);
    draw();
  }

  function initMatrixSpacing(zetaData, matrixData) {
    const svg = byId("matrixSpacingSvg");
    let selected = "both";
    function draw() {
      const series = [];
      if (selected !== "matrix") series.push({ data: zetaData.normalized_gaps, color: COLORS.blue });
      if (selected !== "zeta") series.push({ data: matrixData.normalized_gaps, color: COLORS.orange });
      drawHistogram(svg, series, 24, "Zeta and random-matrix spacings", "Finite normalized gap distributions for zeta zeros and seeded unitary matrices.");
      document.querySelectorAll("[data-spacing-series]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.spacingSeries === selected)));
      byId("matrixSpacingStatus").textContent = `Showing ${selected === "both" ? `${zetaData.sample_size} zeta gaps and ${matrixData.sample_size} simulated matrix gaps` : selected === "zeta" ? `${zetaData.sample_size} zeta gaps` : `${matrixData.sample_size} seeded matrix gaps`}. Different sample sizes and low zeta heights limit visual comparison.`;
    }
    document.querySelectorAll("[data-spacing-series]").forEach(button => button.addEventListener("click", () => {
      selected = button.dataset.spacingSeries;
      draw();
    }));
    draw();
  }

  function initTimeline() {
    document.querySelectorAll("[data-history-filter]").forEach(button => button.addEventListener("click", () => {
      const filter = button.dataset.historyFilter;
      document.querySelectorAll("[data-history-filter]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
      document.querySelectorAll(".timeline-item").forEach(item => item.classList.toggle("hidden", filter !== "all" && item.dataset.kind !== filter));
    }));
  }

  function showLoadFailure(ids, error) {
    ids.forEach(id => {
      const element = byId(id);
      if (element) element.textContent = `Interactive data could not load (${error.message}). Serve the repository through a local HTTP server; the prose and static figures remain available.`;
    });
  }

  function initSafely(name, initializer) {
    try {
      initializer();
    } catch (error) {
      console.error(`Riemann zeta visualization '${name}' failed`, error);
    }
  }

  initSafely("convergence", initConvergence);
  initSafely("real zeta", initRealZeta);
  initSafely("Euler product", initEulerProduct);
  initSafely("complex vectors", initVectors);
  initSafely("continuation domains", initDomainButtons);
  initSafely("history", initTimeline);

  Promise.all([
    fetchJson("first-zeros.json"),
    fetchJson("z-function-samples.json"),
    fetchJson("zero-counts.json"),
    fetchJson("prime-counting.json"),
    fetchJson("explicit-formula.json"),
    fetchJson("zero-spacings.json"),
    fetchJson("random-matrix-spacings.json")
  ]).then(([zeros, zFunction, zeroCounts, primeCounting, explicit, spacings, matrixSpacings]) => {
    initSafely("critical strip", () => initZeroExplorer(zeros));
    initSafely("prime counting", () => initPrimeCounting(primeCounting));
    initSafely("explicit formula", () => initExplicit(primeCounting, explicit));
    initSafely("Hardy Z", () => initZFunction(zFunction));
    initSafely("zero counting", () => initZeroCount(zeroCounts));
    initSafely("zero spacing", () => initSpacing(spacings));
    initSafely("random matrices", () => initMatrixSpacing(spacings, matrixSpacings));
  }).catch(error => {
    console.error("Riemann zeta numerical assets failed to load", error);
    showLoadFailure(["zeroStatus", "explicitStatus", "zFunctionStatus", "matrixSpacingStatus"], error);
  });
})();
