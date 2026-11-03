/* Shared numerical conventions. Browser + dependency-free Node verification. */
(function (root) {
  'use strict';
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const round = x => Math.floor(x + 0.5); // exact ties go toward +infinity
  function uniform(x, levels = 8, a = -1, b = 1) {
    const delta = (b - a) / levels;
    const index = clamp(Math.floor((x - a) / delta), 0, levels - 1);
    const value = a + (index + 0.5) * delta;
    return {index, value, error: x - value, delta};
  }
  function affine(x, xmin, xmax, bits, mode) {
    bits = clamp(Math.trunc(bits), 1, 8);
    const capacity = 2 ** bits;
    let qmin, qmax, scale, z;
    if (mode === 'symmetric' && bits > 1) {
      qmax = capacity / 2 - 1; qmin = -qmax;
      scale = Math.max(Math.abs(xmin), Math.abs(xmax)) / qmax || 1; z = 0;
    } else if (mode === 'symmetric') {
      // A zero-inclusive symmetric one-bit alphabet has only one distinct level.
      qmin = qmax = z = 0; scale = 1;
    } else {
      qmin = 0; qmax = capacity - 1;
      // Calibration includes zero, also for one-sided and constant tensors.
      const lo = Math.min(0, xmin, xmax), hi = Math.max(0, xmin, xmax);
      scale = (hi - lo) / qmax || 1;
      z = clamp(round(-lo / scale), qmin, qmax);
    }
    const q = clamp(round(x / scale) + z, qmin, qmax);
    const xhat = scale * (q - z);
    return {qmin, qmax, scale, zeropoint:z, q, xhat, err:x-xhat,
      L:qmax-qmin+1, capacity};
  }
  function staircase(x, delta, mode) {
    const offset = mode === 'rise' ? 0.5 : 0;
    const kmin = Math.ceil(-1 / delta - offset - 1e-12);
    const kmax = Math.floor(1 / delta - offset + 1e-12);
    const k = clamp(round(x / delta - offset), kmin, kmax);
    return (k + offset) * delta;
  }
  function density(x, kind) {
    if (kind === 'uniform') return 0.5;
    if (kind === 'skew') return 0.75 * (x + 1) ** 2 / 2;
    return Math.exp(-0.5 * ((x - 0.12) / 0.27) ** 2);
  }
  // Midpoint quadrature: Gaussian is explicitly truncated and normalized on [-1,1].
  function distribution(kind, n = 2400) {
    const data = Array.from({length:n}, (_,i) => {
      const x = -1 + (i + 0.5) * 2 / n;
      return {x, w:density(x, kind)};
    });
    const sum = data.reduce((s,p) => s+p.w,0);
    return data.map(p => ({x:p.x, w:p.w/sum}));
  }
  function nearest(x, levels) {
    let k = 0;
    for (let j = 1; j < levels.length; j++)
      if (Math.abs(x-levels[j]) <= Math.abs(x-levels[k])) k=j;
    return k;
  }
  const distortion = (data, levels) => data.reduce((s,p) =>
    s+p.w*(p.x-levels[nearest(p.x,levels)])**2,0);
  function lloyd(data, levels) {
    const sums = levels.map(() => 0), mass = levels.map(() => 0);
    data.forEach(p => { const k=nearest(p.x,levels); sums[k]+=p.w*p.x; mass[k]+=p.w; });
    return levels.map((c,k) => mass[k] > 1e-15 ? sums[k]/mass[k] : c);
  }
  function seeded(seed = 8713) {
    return () => { seed=(Math.imul(1664525,seed)+1013904223)>>>0; return seed/4294967296; };
  }
  const api={clamp,round,uniform,affine,staircase,density,distribution,nearest,distortion,lloyd,seeded};
  if (typeof module !== 'undefined' && module.exports) module.exports=api;
  else root.QuantizationMath=api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
