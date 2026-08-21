/* Quantization visual guide — interactivity.
   Vanilla JS in an IIFE. No globals leaked.
   Each widget reads its controls and redraws an inline SVG or HTML table. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  function fmt(n, d) {
    if (!isFinite(n)) return "–";
    if (d === undefined) d = 2;
    return n.toFixed(d);
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  /* Core affine quantization.
     mode: "asymmetric" | "symmetric" */
  function affine(x, xmin, xmax, bits, mode) {
    var L = Math.pow(2, bits);
    var qmin, qmax, scale, z;
    if (mode === "symmetric") {
      qmax = Math.pow(2, bits - 1) - 1;
      qmin = -qmax;
      scale = Math.max(Math.abs(xmin), Math.abs(xmax)) / qmax;
      if (scale === 0 || !isFinite(scale)) scale = 1;
      var q = clamp(Math.round(x / scale), qmin, qmax);
      return { qmin: qmin, qmax: qmax, scale: scale, zeropoint: 0, q: q, xhat: q * scale, err: x - q * scale, L: L };
    } else {
      qmin = 0; qmax = L - 1;
      scale = (xmax - xmin) / (qmax - qmin);
      if (scale === 0 || !isFinite(scale)) scale = 1;
      z = clamp(Math.round(qmin - xmin / scale), qmin, qmax);
      var qq = clamp(Math.round(x / scale) + z, qmin, qmax);
      return { qmin: qmin, qmax: qmax, scale: scale, zeropoint: z, q: qq, xhat: (qq - z) * scale, err: x - (qq - z) * scale, L: L };
    }
  }

  /* ---------------- 2. Number-line quantizer ---------------- */
  function drawNumline() {
    var x = parseFloat($("num-val").value);
    var step = parseFloat($("num-step").value);
    var quant = Math.round(x / step) * step;
    var err = x - quant;
    $("num-val-out").value = fmt(x, 2);
    $("num-orig").textContent = fmt(x, 2);
    $("num-quant").textContent = fmt(quant, 2);
    $("num-err").textContent = fmt(err, 2);

    var W = 760, H = 170, x0 = 40, x1 = 720, y = 110, lo = 0, hi = 10;
    var X = function (v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); };
    var s = "";
    // baseline
    s += '<line x1="' + x0 + '" y1="' + y + '" x2="' + x1 + '" y2="' + y + '" stroke="#bbb" stroke-width="2"/>';
    // allowed levels
    for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      var px = X(v);
      s += '<line x1="' + px + '" y1="' + (y - 6) + '" x2="' + px + '" y2="' + (y + 6) + '" stroke="#c9bfa9"/>';
    }
    // ticks at integers
    for (var i = lo; i <= hi; i++) {
      var t = X(i);
      s += '<line x1="' + t + '" y1="' + (y - 14) + '" x2="' + t + '" y2="' + (y + 14) + '" stroke="#999"/>';
      s += '<text x="' + t + '" y="' + (y + 30) + '" font-size="11" fill="#666" text-anchor="middle">' + i + '</text>';
    }
    // error bridge
    s += '<line x1="' + X(x) + '" y1="' + (y - 26) + '" x2="' + X(quant) + '" y2="' + (y - 26) + '" stroke="#b4632a" stroke-width="2"/>';
    // original dot
    s += '<circle cx="' + X(x) + '" cy="' + y + '" r="7" fill="#1d4f91"/>';
    // quantized square
    s += '<rect x="' + (X(quant) - 7) + '" y="' + (y - 7) + '" width="14" height="14" fill="#b4632a"/>';
    s += '<text x="' + X(x) + '" y="' + (y - 34) + '" font-size="12" fill="#1d4f91" text-anchor="middle">orig ' + fmt(x, 2) + '</text>';
    s += '<text x="' + X(quant) + '" y="' + (y + 46) + '" font-size="12" fill="#b4632a" text-anchor="middle">quant ' + fmt(quant, 2) + '</text>';
    $("numline-svg").innerHTML = s;
  }

  /* ---------------- 3. Buckets ---------------- */
  function drawBuckets() {
    var L = parseInt(document.querySelector("#bucket-seg button[aria-pressed='true']").getAttribute("data-levels"), 10);
    var lo = 0, hi = 10, W = 760, H = 220, x0 = 40, x1 = 720, top = 40, bot = 170;
    var X = function (v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); };
    var bw = (x1 - x0) / L;
    var s = "";
    for (var i = 0; i < L; i++) {
      var xa = x0 + i * bw, xb = xa + bw;
      s += '<rect x="' + xa + '" y="' + top + '" width="' + bw + '" height="' + (bot - top) + '" fill="' + (i % 2 ? "#f6f1e6" : "#fbf7ef") + '" stroke="#ddd5c7"/>';
      var center = lo + (i + 0.5) * (hi - lo) / L;
      s += '<line x1="' + xa + '" y1="' + bot + '" x2="' + xb + '" y2="' + bot + '" stroke="#bbb"/>';
      s += '<circle cx="' + X(center) + '" cy="' + (bot + 16) + '" r="5" fill="#1d4f91"/>';
      s += '<text x="' + X(center) + '" y="' + (bot + 34) + '" font-size="11" fill="#555" text-anchor="middle">' + fmt(center, 2) + '</text>';
    }
    s += '<line x1="' + x0 + '" y1="' + bot + '" x2="' + x1 + '" y2="' + bot + '" stroke="#999"/>';
    $("buckets-svg").innerHTML = s;
    $("bucket-caption").innerHTML = L + " levels across the range 0&ndash;10: each level represents a slice of width " + fmt((hi - lo) / L, 3) + ".";
  }

  /* ---------------- 5. Bits ---------------- */
  function drawBits() {
    var b = parseInt(document.querySelector("#bits-selector button[aria-pressed='true']").getAttribute("data-bits"), 10);
    var codes = Math.pow(2, b);
    $("bits-n").textContent = b;
    $("bits-codes").textContent = codes;
    var list = $("bits-codes-list");
    var pad = b;
    var html = "";
    var show = Math.min(codes, 16);
    for (var c = 0; c < show; c++) {
      var bin = c.toString(2);
      while (bin.length < pad) bin = "0" + bin;
      html += '<span class="code"><b>' + bin + '</b></span>';
    }
    if (codes > 16) html += '<span class="code">&hellip; +' + (codes - 16) + ' more</span>';
    list.innerHTML = html;
  }

  /* ---------------- 6. Quantized signal ---------------- */
  function drawSignal() {
    var L = parseInt($("sig-levels").value, 10);
    var range = parseFloat($("sig-range").value);
    var W = 760, H = 260, x0 = 40, x1 = 720, mid = 140, amp = 90;
    var N = 260;
    var step = (2 * range) / (L - 1);
    var s = "";
    // grid baseline
    s += '<line x1="' + x0 + '" y1="' + mid + '" x2="' + x1 + '" y2="' + mid + '" stroke="#eee" stroke-width="2"/>';
    var orig = "", stairs = "";
    var maxErr = 0;
    var prevX = x0, prevY = mid;
    for (var i = 0; i <= N; i++) {
      var t = i / N;
      var ph = t * Math.PI * 2 * 2; // two periods
      var yv = Math.sin(ph) * range;
      var qy = Math.round(yv / step) * step;
      var px = x0 + t * (x1 - x0);
      var py = mid - (yv / range) * amp;
      var qpy = mid - (qy / range) * amp;
      orig += (i === 0 ? "" : " ") + px + "," + py;
      if (i > 0) {
        stairs += '<line x1="' + prevX + '" y1="' + prevY + '" x2="' + px + '" y2="' + prevY + '" stroke="#b4632a" stroke-width="2"/>';
        stairs += '<line x1="' + px + '" y1="' + prevY + '" x2="' + px + '" y2="' + qpy + '" stroke="#b4632a" stroke-width="1"/>';
      }
      prevX = px; prevY = qpy;
      maxErr = Math.max(maxErr, Math.abs(yv - qy));
    }
    s += stairs;
    s += '<polyline points="' + orig + '" fill="none" stroke="#1d4f91" stroke-width="2.5"/>';
    $("signal-svg").innerHTML = s;
    $("sig-levels-out").textContent = L;
    $("sig-err").textContent = fmt(maxErr, 3);
  }

  /* ---------------- 9. Scale map ---------------- */
  function drawScale() {
    var xmin = parseFloat($("scale-xmin").value);
    var xmax = parseFloat($("scale-xmax").value);
    var bits = clamp(parseInt($("scale-bits").value, 10), 1, 8);
    var r = affine(0, xmin, xmax, bits, "asymmetric");
    var W = 760, x0 = 60, x1 = 720;
    var X = function (v, lo, hi) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); };
    var s = "";
    var yTop = 60, yBot = 160;
    // real line
    s += '<line x1="' + X(xmin, xmin, xmax) + '" y1="' + yTop + '" x2="' + X(xmax, xmin, xmax) + '" y2="' + yTop + '" stroke="#1d4f91" stroke-width="3"/>';
    s += '<text x="' + X(xmin, xmin, xmax) + '" y="' + (yTop - 12) + '" font-size="12" fill="#1d4f91" text-anchor="middle">real ' + fmt(xmin, 2) + '</text>';
    s += '<text x="' + X(xmax, xmin, xmax) + '" y="' + (yTop - 12) + '" font-size="12" fill="#1d4f91" text-anchor="middle">real ' + fmt(xmax, 2) + '</text>';
    // code line
    var qmax = r.qmax;
    s += '<line x1="' + X(0, 0, qmax) + '" y1="' + yBot + '" x2="' + X(qmax, 0, qmax) + '" y2="' + yBot + '" stroke="#b4632a" stroke-width="3"/>';
    s += '<text x="' + X(0, 0, qmax) + '" y="' + (yBot + 22) + '" font-size="12" fill="#b4632a" text-anchor="middle">code 0</text>';
    s += '<text x="' + X(qmax, 0, qmax) + '" y="' + (yBot + 22) + '" font-size="12" fill="#b4632a" text-anchor="middle">code ' + qmax + '</text>';
    // mapping arrows for sample points
    var samples = [];
    for (var i = 0; i <= 4; i++) samples.push(xmin + (xmax - xmin) * i / 4);
    for (var k = 0; k < samples.length; k++) {
      var v = samples[k];
      var qv = Math.round(v / r.scale) + r.zeropoint;
      qv = clamp(qv, 0, qmax);
      var xt = X(v, xmin, xmax), xb = X(qv, 0, qmax);
      s += '<line x1="' + xt + '" y1="' + yTop + '" x2="' + xb + '" y2="' + yBot + '" stroke="#c9bfa9" stroke-width="1"/>';
      s += '<circle cx="' + xt + '" cy="' + yTop + '" r="3" fill="#1d4f91"/>';
      s += '<circle cx="' + xb + '" cy="' + yBot + '" r="3" fill="#b4632a"/>';
    }
    $("scale-svg").innerHTML = s;
    $("scale-s").textContent = fmt(r.scale, 3);
    $("scale-z").textContent = r.zeropoint;
    $("scale-L").textContent = r.L;
  }

  /* ---------------- 11. Symmetric vs asymmetric ---------------- */
  function drawSym() {
    var mode = document.querySelector("#sym-seg button[aria-pressed='true']").getAttribute("data-mode");
    var xmin = parseFloat($("sym-xmin").value);
    var xmax = parseFloat($("sym-xmax").value);
    var bits = clamp(parseInt($("sym-bits").value, 10), 1, 8);
    var r = affine((xmin + xmax) / 2, xmin, xmax, bits, mode);
    var W = 760, x0 = 60, x1 = 720, y = 120, lo = Math.min(xmin, -0.001), hi = Math.max(xmax, 0.001);
    // map zero-centered for symmetric
    if (mode === "symmetric") { hi = Math.max(Math.abs(xmin), Math.abs(xmax)); lo = -hi; }
    var X = function (v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); };
    var s = '<line x1="' + X(lo) + '" y1="' + y + '" x2="' + X(hi) + '" y2="' + y + '" stroke="#999" stroke-width="2"/>';
    // draw levels
    if (mode === "symmetric") {
      for (var q = r.qmin; q <= r.qmax; q++) {
        var a = q * r.scale;
        if (a < lo || a > hi) continue;
        s += '<circle cx="' + X(a) + '" cy="' + y + '" r="5" fill="#1d4f91"/>';
      }
      s += '<text x="' + X(0) + '" y="' + (y + 28) + '" font-size="12" fill="#1d4f91" text-anchor="middle">0 fixed at center</text>';
      $("sym-caption").innerHTML = "Symmetric: zero maps to the center code; codes spread evenly by scale " + fmt(r.scale, 3) + ". One-sided data wastes codes near the empty side.";
    } else {
      for (var qq = 0; qq <= r.qmax; qq++) {
        var b = xmin + qq * r.scale;
        s += '<circle cx="' + X(b) + '" cy="' + y + '" r="5" fill="#b4632a"/>';
      }
      s += '<text x="' + X(0) + '" y="' + (y + 28) + '" font-size="12" fill="#b4632a" text-anchor="middle">0 at code ' + r.zeropoint + '</text>';
      $("sym-caption").innerHTML = "Asymmetric: the zero-point slides so code " + r.zeropoint + " is exactly zero; the " + r.L + " levels cover the data snugly from " + fmt(xmin, 2) + " to " + fmt(xmax, 2) + ".";
    }
    // endpoints
    s += '<text x="' + X(lo) + '" y="' + (y - 14) + '" font-size="12" fill="#555" text-anchor="middle">' + fmt(lo, 2) + '</text>';
    s += '<text x="' + X(hi) + '" y="' + (y - 14) + '" font-size="12" fill="#555" text-anchor="middle">' + fmt(hi, 2) + '</text>';
    $("sym-svg").innerHTML = s;
  }

  /* ---------------- 12. Clipping ---------------- */
  function drawClip() {
    var r = parseFloat($("clip-range").value);
    var x = parseFloat($("clip-val").value);
    var stored = clamp(x, -r, r);
    var err = x - stored;
    $("clip-orig").textContent = fmt(x, 2);
    $("clip-stored").textContent = fmt(stored, 2);
    $("clip-err").textContent = fmt(err, 2);
    var lo = -2, hi = 9, x0 = 40, x1 = 720, y = 110;
    var X = function (v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); };
    var s = '<line x1="' + X(lo) + '" y1="' + y + '" x2="' + X(hi) + '" y2="' + y + '" stroke="#999" stroke-width="2"/>';
    // representable band
    s += '<rect x="' + X(-r) + '" y="' + (y - 26) + '" width="' + (X(r) - X(-r)) + '" height="52" fill="#eef4fb" stroke="#bcd2e8"/>';
    s += '<text x="' + X(-r) + '" y="' + (y - 32) + '" font-size="11" fill="#1d4f91" text-anchor="middle">-' + fmt(r, 1) + '</text>';
    s += '<text x="' + X(r) + '" y="' + (y - 32) + '" font-size="11" fill="#1d4f91" text-anchor="middle">+' + fmt(r, 1) + '</text>';
    s += '<text x="' + ((X(-r) + X(r)) / 2) + '" y="' + (y + 40) + '" font-size="11" fill="#1d4f91" text-anchor="middle">representable range</text>';
    // original
    s += '<circle cx="' + X(x) + '" cy="' + y + '" r="7" fill="#1d4f91"/>';
    s += '<text x="' + X(x) + '" y="' + (y - 30) + '" font-size="12" fill="#1d4f91" text-anchor="middle">orig ' + fmt(x, 2) + '</text>';
    // stored
    s += '<rect x="' + (X(stored) - 7) + '" y="' + (y - 7) + '" width="14" height="14" fill="#b4632a"/>';
    s += '<text x="' + X(stored) + '" y="' + (y + 34) + '" font-size="12" fill="#b4632a" text-anchor="middle">stored ' + fmt(stored, 2) + '</text>';
    if (Math.abs(err) > 1e-9) {
      s += '<line x1="' + X(x) + '" y1="' + (y - 22) + '" x2="' + X(stored) + '" y2="' + (y - 22) + '" stroke="#b4632a" stroke-width="2"/>';
    }
    $("clip-svg").innerHTML = s;
  }

  /* ---------------- 13. Laboratory ---------------- */
  function drawLab() {
    var xmin = parseFloat($("lab-xmin").value);
    var xmax = parseFloat($("lab-xmax").value);
    var bits = clamp(parseInt($("lab-bits").value, 10), 1, 8);
    var x = parseFloat($("lab-x").value);
    var mode = document.querySelector("#lab-seg button[aria-pressed='true']").getAttribute("data-mode");
    var r = affine(x, xmin, xmax, bits, mode);
    $("lab-L").textContent = r.L;
    $("lab-s").textContent = fmt(r.scale, 3);
    $("lab-z").textContent = r.zeropoint;
    $("lab-q").textContent = r.q;
    $("lab-xhat").textContent = fmt(r.xhat, 3);
    $("lab-err").textContent = fmt(r.err, 3);
    $("lab-rel").textContent = (x !== 0 ? fmt(Math.abs(r.err / x) * 100, 1) : "∞") + "%";

    var lo = Math.min(xmin, -0.001), hi = Math.max(xmax, 0.001);
    if (mode === "symmetric") { hi = Math.max(Math.abs(xmin), Math.abs(xmax)); lo = -hi; }
    var x0 = 40, x1 = 720, y = 95;
    var X = function (v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); };
    var s = '<line x1="' + X(lo) + '" y1="' + y + '" x2="' + X(hi) + '" y2="' + y + '" stroke="#999" stroke-width="2"/>';
    // levels
    if (mode === "symmetric") {
      for (var q = r.qmin; q <= r.qmax; q++) { var a = q * r.scale; if (a < lo || a > hi) continue; s += '<line x1="' + X(a) + '" y1="' + (y - 6) + '" x2="' + X(a) + '" y2="' + (y + 6) + '" stroke="#c9bfa9"/>'; }
    } else {
      for (var qq = 0; qq <= r.qmax; qq++) { var b = xmin + qq * r.scale; s += '<line x1="' + X(b) + '" y1="' + (y - 6) + '" x2="' + X(b) + '" y2="' + (y + 6) + '" stroke="#c9bfa9"/>'; }
    }
    s += '<circle cx="' + X(x) + '" cy="' + y + '" r="7" fill="#1d4f91"/>';
    s += '<rect x="' + (X(r.xhat) - 7) + '" y="' + (y - 7) + '" width="14" height="14" fill="#b4632a"/>';
    s += '<text x="' + X(x) + '" y="' + (y - 18) + '" font-size="12" fill="#1d4f91" text-anchor="middle">x=' + fmt(x, 2) + '</text>';
    s += '<text x="' + X(r.xhat) + '" y="' + (y + 30) + '" font-size="12" fill="#b4632a" text-anchor="middle">q=' + r.q + ' → ' + fmt(r.xhat, 2) + '</text>';
    $("lab-svg").innerHTML = s;
  }

  /* ---------------- 14. Vector ---------------- */
  var VEC = [0.12, -0.87, 1.34, 0.44, 0.71, -1.19];
  function drawVector() {
    var bits = clamp(parseInt($("vec-bits").value, 10), 1, 8);
    var mode = document.querySelector("#vec-seg button[aria-pressed='true']").getAttribute("data-mode");
    var mn = Math.min.apply(null, VEC), mx = Math.max.apply(null, VEC);
    var r0 = affine(VEC[0], mn, mx, bits, mode);
    var codes = [], xhat = [], errs = [];
    for (var i = 0; i < VEC.length; i++) {
      var r = affine(VEC[i], mn, mx, bits, mode);
      codes.push(r.q); xhat.push(r.xhat); errs.push(r.err);
    }
    function row(label, vals, isCode) {
      var h = '<div style="display:flex;gap:6px;align-items:center;margin:6px 0"><div style="width:130px;font-weight:700">' + label + '</div>';
      for (var i = 0; i < vals.length; i++) {
        var bg = "#fffdf9";
        if (!isCode) bg = "#fcfaf6";
        h += '<div class="cell" style="width:64px;height:46px;display:flex;align-items:center;justify-content:center;border:1px solid var(--line);border-radius:8px;font-family:Source Code Pro,monospace;background:' + bg + '">' + (isCode ? vals[i] : fmt(vals[i], 2)) + '</div>';
      }
      return h + '</div>';
    }
    var html = row("Original", VEC, false) + row("Code q", codes, true) + row("Reconstructed", xhat, false);
    // error row with color
    var er = '<div style="display:flex;gap:6px;align-items:center;margin:6px 0"><div style="width:130px;font-weight:700">Error</div>';
    for (var j = 0; j < errs.length; j++) {
      var e = errs[j];
      var col = Math.abs(e) < 1e-9 ? "#ddd5c7" : (e < 0 ? "#b8442e" : "#2f7d4f");
      er += '<div class="cell" style="width:64px;height:46px;display:flex;align-items:center;justify-content:center;border:1px solid var(--line);border-radius:8px;font-family:Source Code Pro,monospace;background:' + col + ';color:#1b1b1b">' + fmt(e, 2) + '</div>';
    }
    er += '</div>';
    html += er;
    $("vector-tables").innerHTML = html;
    $("vec-caption").innerHTML = "Scale s=" + fmt(r0.scale, 3) + ", zero-point z=" + r0.zeropoint + ", " + r0.L + " levels (" + mode + ", " + bits + " bits). The same mapping is applied to every entry.";
  }

  /* ---------------- 15. Matrix ---------------- */
  var MAT = [
    [0.12, -0.87, 1.34],
    [0.44, 0.71, -1.19],
    [-0.32, 0.08, 0.95]
  ];
  function errColor(e, maxE) {
    if (Math.abs(e) < 1e-9) return "#ddd5c7";
    var t = clamp(Math.abs(e) / (maxE || 1), 0, 1);
    return e < 0 ? "rgba(184,68,46," + (0.25 + 0.6 * t) + ")" : "rgba(47,125,79," + (0.25 + 0.6 * t) + ")";
  }
  function divColor(v, maxV) {
    if (Math.abs(v) < 1e-9) return "#f3efe7";
    var t = clamp(Math.abs(v) / (maxV || 1), 0, 1);
    return v < 0 ? "rgba(184,68,46," + (0.2 + 0.7 * t) + ")" : "rgba(29,79,145," + (0.2 + 0.7 * t) + ")";
  }
  function drawMatrix() {
    var bits = clamp(parseInt($("mat-bits").value, 10), 1, 8);
    var mode = document.querySelector("#mat-mode-seg button[aria-pressed='true']").getAttribute("data-mode");
    var view = document.querySelector("#mat-seg button[aria-pressed='true']").getAttribute("data-view");
    var flat = [].concat.apply([], MAT);
    var mn = Math.min.apply(null, flat), mx = Math.max.apply(null, flat);
    var codes = [], xhat = [], errs = [];
    var maxE = 0;
    for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) {
      var r = affine(MAT[i][j], mn, mx, bits, mode);
      codes.push(r.q); xhat.push(r.xhat); errs.push(r.err);
      maxE = Math.max(maxE, Math.abs(r.err));
    }
    var html = '<div class="matrix">';
    var idx = 0;
    for (var i2 = 0; i2 < 3; i2++) {
      for (var j2 = 0; j2 < 3; j2++) {
        var cell;
        if (view === "original") cell = '<div class="cell" style="background:' + divColor(MAT[i2][j2], mx) + '">' + fmt(MAT[i2][j2], 2) + '</div>';
        else if (view === "quantized") cell = '<div class="cell" style="background:#fcfaf6">' + codes[idx] + '</div>';
        else if (view === "reconstructed") cell = '<div class="cell" style="background:' + divColor(xhat[idx], mx) + '">' + fmt(xhat[idx], 2) + '</div>';
        else cell = '<div class="cell" style="background:' + errColor(errs[idx], maxE) + '">' + fmt(errs[idx], 2) + '</div>';
        html += cell;
        idx++;
      }
    }
    html += '</div>';
    $("matrix-view").innerHTML = html;
    var cap = {
      original: "The raw float weights. Range " + fmt(mn, 2) + " to " + fmt(mx, 2) + ".",
      quantized: "The integer codes q stored after quantization (" + mode + ", " + bits + " bits).",
      reconstructed: "Recovered values ĥx = s(q−z). Close to the originals, but approximate.",
      error: "Per-cell quantization error e = x − ĥx. Color: red under, green over."
    };
    $("mat-caption").innerHTML = cap[view];
  }

  /* ---------------- 16-19. Per-tensor / channel / group ---------------- */
  var CH_A = [0.01, 0.03, 0.05, 0.08];
  var CH_B = [2.1, 4.3, 6.7, 9.2];
  function quantizeArray(arr, mode, bits, groupSize) {
    if (!groupSize) {
      var mn = Math.min.apply(null, arr), mx = Math.max.apply(null, arr);
      return arr.map(function (v) { var r = affine(v, mn, mx, bits, mode); return { q: r.q, xhat: r.xhat, err: r.err, scale: r.scale, z: r.zeropoint }; });
    }
    var out = [];
    for (var g = 0; g < arr.length; g += groupSize) {
      var sub = arr.slice(g, g + groupSize);
      var mn = Math.min.apply(null, sub), mx = Math.max.apply(null, sub);
      for (var k = 0; k < sub.length; k++) { var r = affine(sub[k], mn, mx, bits, mode); out.push({ q: r.q, xhat: r.xhat, err: r.err, scale: r.scale, z: r.zeropoint, grp: g / groupSize }); }
    }
    return out;
  }
  function renderAxis(containerId, captionId, title, A, B, mode, bits, groupSize, note) {
    function table(name, arr, res) {
      var h = '<div style="margin:10px 0"><strong>' + name + '</strong><div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px">';
      for (var i = 0; i < arr.length; i++) {
        h += '<div style="border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-family:Source Code Pro,monospace;background:#fffdf9;min-width:120px">';
        h += 'x=' + fmt(arr[i], 2) + '<br>q=' + res[i].q + '<br>ŷ=' + fmt(res[i].xhat, 2) + ' <span style="color:#888">(s=' + fmt(res[i].scale, 2) + ')</span></div>';
      }
      return h + '</div></div>';
    }
    var ra = quantizeArray(A, mode, bits, groupSize);
    var rb = quantizeArray(B, mode, bits, groupSize);
    var html = '<div class="axis-card"><h4>' + title + '</h4>' + table("Channel A", A, ra) + table("Channel B", B, rb) + '</div>';
    $(containerId).innerHTML = html;
    $(captionId).innerHTML = note;
  }
  function drawPerAxes() {
    var bits = 4, mode = "asymmetric";
    // per-tensor: one scale over both channels
    var all = CH_A.concat(CH_B);
    var mn = Math.min.apply(null, all), mx = Math.max.apply(null, all);
    var ra = all.slice(0, 4).map(function (v) { var r = affine(v, mn, mx, bits, mode); return { q: r.q, xhat: r.xhat, err: r.err, scale: r.scale }; });
    var rb = all.slice(4).map(function (v) { var r = affine(v, mn, mx, bits, mode); return { q: r.q, xhat: r.xhat, err: r.err, scale: r.scale }; });
    function table(name, arr, res) {
      var h = '<div style="margin:10px 0"><strong>' + name + '</strong><div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px">';
      for (var i = 0; i < arr.length; i++) {
        var collapsed = (res[i].q === res[0].q && i > 0);
        h += '<div style="border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-family:Source Code Pro,monospace;background:' + (collapsed ? '#fbe9e4' : '#fffdf9') + ';min-width:120px">';
        h += 'x=' + fmt(arr[i], 2) + '<br>q=' + res[i].q + '<br>ŷ=' + fmt(res[i].xhat, 2) + '</div>';
      }
      return h + '</div></div>';
    }
    $("pertensor-view").innerHTML = '<div class="axis-card"><h4>One global scale s=' + fmt((mx - mn) / 15, 3) + '</h4>' + table("Channel A", CH_A, ra) + table("Channel B", CH_B, rb) + '</div>';
    $("pertensor-caption").innerHTML = "Channel A's tiny values all round to the same code (highlighted) &mdash; its detail is lost under a single global scale.";

    renderAxis("perchannel-view", "perchannel-caption", "Per-channel: separate scale per channel", CH_A, CH_B, mode, bits, 0,
      "Each channel uses its own min/max, so Channel A gets a fine scale and keeps its detail. Cost: one scale per channel to store.");

    renderAxis("pergroup-view", "pergroup-caption", "Per-group (size 2): one scale per pair", CH_A, CH_B, mode, bits, 2,
      "Each group of 2 values shares a scale. This localizes outlier effects: a big value in one group does not coarsen a distant group.");
  }

  /* ---------------- 20. Network diagram ---------------- */
  function drawNet() {
    var W = 760, H = 200;
    var layers = [3, 4, 3, 2];
    var xs = [110, 300, 500, 690];
    var nodes = [];
    for (var l = 0; l < layers.length; l++) {
      nodes[l] = [];
      var n = layers[l], y0 = H / 2 - (n - 1) * 32 / 2;
      for (var i = 0; i < n; i++) nodes[l].push({ x: xs[l], y: y0 + i * 32 });
    }
    var s = "";
    // weights as orange lines
    for (var l = 0; l < layers.length - 1; l++) {
      for (var a = 0; a < nodes[l].length; a++) for (var b = 0; b < nodes[l + 1].length; b++) {
        s += '<line x1="' + nodes[l][a].x + '" y1="' + nodes[l][a].y + '" x2="' + nodes[l + 1][b].x + '" y2="' + nodes[l + 1][b].y + '" stroke="#b4632a" stroke-width="1" opacity="0.5"/>';
      }
    }
    // activations as blue circles
    for (var l2 = 0; l2 < layers.length; l2++) {
      for (var c = 0; c < nodes[l2].length; c++) {
        s += '<circle cx="' + nodes[l2][c].x + '" cy="' + nodes[l2][c].y + '" r="11" fill="#1d4f91"/>';
      }
    }
    s += '<text x="110" y="30" font-size="13" fill="#b4632a" text-anchor="middle">Weights (orange)</text>';
    s += '<text x="400" y="30" font-size="13" fill="#1d4f91" text-anchor="middle">Activations (blue)</text>';
    $("net-svg").innerHTML = s;
  }

  /* ---------------- 24/25. Weight-only / weight+act diagrams ---------------- */
  function drawFlow(svgId, actColor) {
    var s = '<rect x="40" y="40" width="160" height="40" rx="8" fill="#b4632a"/><text x="120" y="65" font-size="14" fill="#fff" text-anchor="middle">Weights</text>';
    s += '<rect x="300" y="40" width="160" height="40" rx="8" fill="' + actColor + '"/><text x="380" y="65" font-size="14" fill="#fff" text-anchor="middle">Activations</text>';
    s += '<rect x="540" y="40" width="180" height="40" rx="8" fill="#2f7d4f"/><text x="630" y="65" font-size="14" fill="#fff" text-anchor="middle">Output</text>';
    s += '<line x1="200" y1="60" x2="300" y2="60" stroke="#999" stroke-width="2"/>';
    s += '<line x1="460" y1="60" x2="540" y2="60" stroke="#999" stroke-width="2"/>';
    s += '<text x="120" y="100" font-size="12" fill="#666" text-anchor="middle">low-bit</text>';
    s += '<text x="380" y="100" font-size="12" fill="#666" text-anchor="middle">' + (actColor === "#1d4f91" ? "high precision" : "low-bit") + '</text>';
    $(svgId).innerHTML = s;
  }

  /* ---------------- 23. Memory calculator ---------------- */
  function drawMem() {
    var params = parseFloat($("mem-params").value) * 1e9; // billions
    var bits = clamp(parseInt($("mem-bits").value, 10), 1, 32);
    var bytes = params * bits / 8;
    $("mem-bytes").textContent = bytes.toExponential(2) + " B";
    $("mem-gb").textContent = fmt(bytes / 1e9, 2) + " GB";
    $("mem-gib").textContent = fmt(bytes / Math.pow(2, 30), 2) + " GiB";
  }

  /* ---------------- 29. Outliers ---------------- */
  function drawOutlier() {
    var mag = parseFloat($("out-mag").value);
    var bits = clamp(parseInt($("out-bits").value, 10), 1, 8);
    var vals = [];
    for (var i = 0; i < 40; i++) vals.push((Math.random() - 0.5) * 0.6);
    vals.push(mag);
    var mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals);
    var r = affine(0, mn, mx, bits, "asymmetric");
    var W = 760, H = 220, x0 = 30, x1 = 740, base = 190, maxH = 150;
    var s = "";
    for (var k = 0; k < vals.length; k++) {
      var v = vals[k];
      var h = maxH * (Math.abs(v) / (Math.abs(mx) || 1));
      var col = (k === vals.length - 1) ? "#b8442e" : "#1d4f91";
      s += '<rect x="' + (x0 + k * (x1 - x0) / vals.length) + '" y="' + (base - h) + '" width="' + ((x1 - x0) / vals.length - 1) + '" height="' + h + '" fill="' + col + '" opacity="0.8"/>';
    }
    s += '<line x1="' + x0 + '" y1="' + base + '" x2="' + x1 + '" y2="' + base + '" stroke="#999"/>';
    s += '<text x="' + x1 + '" y="' + (base + 18) + '" font-size="12" fill="#b8442e" text-anchor="end">outlier = ' + fmt(mag, 1) + '</text>';
    $("outlier-svg").innerHTML = s;
    $("out-scale").textContent = fmt(r.scale, 3);
    $("out-step").textContent = fmt(r.scale, 3);
  }

  /* ---------------- 31. NF4 ---------------- */
  function drawNF4() {
    var layout = document.querySelector("#nf4-seg button[aria-pressed='true']").getAttribute("data-layout");
    var lo = -1, hi = 1, x0 = 40, x1 = 720, y = 120;
    var X = function (v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); };
    var s = '<line x1="' + X(lo) + '" y1="' + y + '" x2="' + X(hi) + '" y2="' + y + '" stroke="#999" stroke-width="2"/>';
    var levels;
    if (layout === "uniform") {
      levels = [];
      for (var i = 0; i < 16; i++) levels.push(lo + i * (hi - lo) / 15);
      $("nf4-caption").innerHTML = "Uniform INT4: 16 evenly spaced levels. Good for data spread flatly across the range; wasteful where real weights cluster near zero.";
    } else {
      // NF4-like quantile levels (symmetric, bell-shaped density)
      levels = [-1.0, -0.696, -0.525, -0.395, -0.284, -0.184, -0.091, 0.0, 0.0796, 0.177, 0.275, 0.388, 0.527, 0.704, 0.947, 1.0];
      $("nf4-caption").innerHTML = "NF4: levels packed densely where trained weights are most common (near zero) and sparse in the tails. Same 16 codes, but matched to a bell-shaped distribution.";
    }
    for (var k = 0; k < levels.length; k++) {
      s += '<circle cx="' + X(levels[k]) + '" cy="' + y + '" r="5" fill="#1d4f91"/>';
      s += '<text x="' + X(levels[k]) + '" y="' + (y - 14) + '" font-size="10" fill="#555" text-anchor="middle">' + fmt(levels[k], 2) + '</text>';
    }
    $("nf4-svg").innerHTML = s;
  }

  /* ---------------- wire up controls ---------------- */
  function bind(id, fn) { var e = $(id); if (e) e.addEventListener("input", fn); }
  function bindSeg(segId, fn) {
    var seg = $(segId);
    if (!seg) return;
    seg.addEventListener("click", function (ev) {
      var btn = ev.target.closest("button");
      if (!btn) return;
      var btns = seg.querySelectorAll("button");
      for (var i = 0; i < btns.length; i++) btns[i].setAttribute("aria-pressed", "false");
      btn.setAttribute("aria-pressed", "true");
      fn();
    });
  }

  function init() {
    bind("num-val", drawNumline); bind("num-step", drawNumline); bind("num-val-out", function () { $("num-val").value = $("num-val-out").value; drawNumline(); });
    bindSeg("bucket-seg", drawBuckets);
    bindSeg("bits-selector", drawBits);
    bind("sig-levels", drawSignal); bind("sig-range", drawSignal);
    bind("scale-xmin", drawScale); bind("scale-xmax", drawScale); bind("scale-bits", drawScale);
    bindSeg("sym-seg", drawSym); bind("sym-xmin", drawSym); bind("sym-xmax", drawSym); bind("sym-bits", drawSym);
    bind("clip-range", drawClip); bind("clip-val", drawClip);
    bind("lab-xmin", drawLab); bind("lab-xmax", drawLab); bind("lab-bits", drawLab); bind("lab-x", drawLab);
    bindSeg("lab-seg", drawLab);
    bind("vec-bits", drawVector); bindSeg("vec-seg", drawVector);
    bind("mat-bits", drawMatrix); bindSeg("mat-seg", drawMatrix); bindSeg("mat-mode-seg", drawMatrix);
    bind("mem-params", drawMem); bind("mem-bits", drawMem);
    bind("out-mag", drawOutlier); bind("out-bits", drawOutlier);
    bindSeg("nf4-seg", drawNF4);

    // initial paints
    drawNumline(); drawBuckets(); drawBits(); drawSignal(); drawScale();
    drawSym(); drawClip(); drawLab(); drawVector(); drawMatrix(); drawPerAxes();
    drawNet(); drawFlow("weightonly-svg", "#1d4f91"); drawFlow("weightact-svg", "#1d4f91");
    drawMem(); drawOutlier(); drawNF4();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
