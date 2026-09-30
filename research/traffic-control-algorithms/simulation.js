/* Pure teaching models: no DOM, clocks, randomness, or network dependencies. */
(function (root) {
  'use strict';
  const mod = (x, n) => ((x % n) + n) % n;
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const movements = ['N_T', 'S_T', 'E_T', 'W_T', 'N_L', 'S_L', 'E_L', 'W_L'];
  // Conservative compatibility model documented beside the matrix.
  const compatiblePairs = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const conflicts = movements.map((_, i) => movements.map((__, j) => i !== j && !compatiblePairs.some(([a, b]) => (a === i && b === j) || (a === j && b === i)) ? 1 : 0));
  const story = [
    [0, 6, 'Arrivals', 'Four approaches. One shared space.', 'Cars arrive independently. The controller must decide which compatible movements receive permission.'],
    [6, 11, 'Conflict', 'These paths cross. Both cannot have protected green.', 'The highlighted north-to-south and west-to-east trajectories share a conflict area. All vehicles remain held while we inspect it.'],
    [11, 16, 'Compatibility', 'Opposing through movements can share a phase.', 'North-to-south and south-to-north traffic occupy separate lanes. Grouping compatible movements makes better use of the road.'],
    [16, 28, 'Green', 'Serve the north–south queues.', 'Both opposing through movements receive green. Each car follows its lane; the east–west approaches continue to wait.'],
    [28, 32, 'Yellow', 'End permission without an abrupt conflict.', 'In this conservative teaching model, no new vehicle enters on yellow. Those already beyond the stop line continue across.'],
    [32, 35, 'All-red', 'Clear first. Then transfer permission.', 'Both groups are red. This clearance separates the end of one movement from the beginning of a conflicting movement.'],
    [35, 46, 'Next green', 'Now east–west traffic receives service.', 'The waiting east–west queues discharge. At 40 seconds a pedestrian call is registered for the next safe opportunity.'],
    [46, 61, 'A pedestrian call', 'Demand changes what comes next.', 'East–west yellow and all-red finish first. At 53 s an exclusive WALK begins; at 56 s pedestrian clearance begins. Vehicles remain red.'],
    [61, 68, 'Coordination', 'One intersection → corridor → city network.', 'Neighboring signals can share timing references and selected observations. Coordination links local decisions across space; it does not require every signal to talk to every other signal.']
  ];
  function heroSignal(t) {
    if (t >= 16 && t < 28) return {phase: 0, state: 'GREEN', start: 16};
    if (t >= 28 && t < 32) return {phase: 0, state: 'YELLOW', start: 28};
    if (t >= 35 && t < 46) return {phase: 1, state: 'GREEN', start: 35};
    if (t >= 46 && t < 50) return {phase: 1, state: 'YELLOW', start: 46};
    return {phase: -1, state: 'ALL RED', start: t >= 50 ? 50 : t >= 32 ? 32 : 0};
  }
  class Hero {
    constructor() { this.reset(); }
    reset() {
      this.time = 0; this.id = 0; this.remainder = 0;
      this.lanes = Array.from({length: 4}, () => [185, 148, 111].map(p => ({p, id: this.id++})));
      this.arrivals = [3, 4.3, 2, 3.7];
      this.entries = [];
    }
    advance(seconds) {
      this.remainder += seconds;
      while (this.remainder >= 1 / 60) {
        this.remainder -= 1 / 60;
        this.tick(1 / 60);
      }
    }
    tick(dt) {
      if (this.time + dt >= 68) { this.reset(); return; }
      const signal = heroSignal(this.time);
      this.lanes.forEach((lane, i) => {
        this.arrivals[i] -= dt;
        if (this.arrivals[i] <= 0) {
          const back = lane.length ? lane[lane.length - 1].p : 0;
          lane.push({p: Math.min(-14, back - 37), id: this.id++});
          this.arrivals[i] += i < 2 ? 6.5 : 7;
        }
        const allowed = signal.state === 'GREEN' && (i < 2 ? 0 : 1) === signal.phase;
        lane.forEach((car, j) => {
          const old = car.p;
          let limit = j ? lane[j - 1].p - 37 : Infinity;
          if (car.p <= 185 && !allowed) limit = Math.min(limit, 185);
          car.p = Math.max(old, Math.min(car.p + 40 * dt, limit));
          if (old <= 185 && car.p > 185) this.entries.push({time: this.time, lane: i, state: signal.state});
        });
        this.lanes[i] = lane.filter(car => car.p < 635);
      });
      this.time += dt;
    }
    get queues() { return this.lanes.map(lane => lane.filter(car => car.p <= 185).length); }
    get stage() { return story.findIndex(s => this.time >= s[0] && this.time < s[1]); }
  }
  function fixedQueues(green = 32) {
    const points = []; let a = 0, b = 0;
    for (let t = 0; t <= 272; t++) {
      points.push([t, a, b]);
      const p = t % 68;
      a = Math.max(0, a + .20 - (p < green ? .5 : 0));
      b = Math.max(0, b + .16 - (p >= green + 6 && p < 62 ? .5 : 0));
    }
    return points;
  }
  // Deterministic, single-intersection comparison; same integer arrivals for both policies.
  function comparison(policy) {
    let q = [0, 0], phase = 0, elapsed = 0, state = 'GREEN', last = [-99, -99], served = 0, wait = 0;
    const points = [];
    for (let t = 0; t <= 180; t++) {
      const add = [t % 60 < 20 && t % 2 === 0 ? 1 : 0, t % 13 === 0 ? 1 : 0];
      add.forEach((n, i) => { q[i] += n; if (n) last[i] = t; });
      if (state === 'GREEN' && elapsed >= (policy === 'fixed' ? 24 : 8)) {
        if (policy === 'fixed' || (q[1 - phase] > 0 && (elapsed >= 28 || (q[phase] === 0 && t - last[phase] >= 3)))) { state = 'YELLOW'; elapsed = 0; }
      } else if (state === 'YELLOW' && elapsed >= 3) { state = 'ALL RED'; elapsed = 0; }
      else if (state === 'ALL RED' && elapsed >= 2) { phase = 1 - phase; state = 'GREEN'; elapsed = 0; }
      if (state === 'GREEN' && elapsed % 2 === 0 && q[phase]) { q[phase]--; served++; }
      wait += q[0] + q[1];
      points.push({t, q: [...q], phase, state, served, wait});
      elapsed++;
    }
    return points;
  }
  // Store-and-forward city. d=0 travels south; d=1 travels east. Reservations enforce finite storage.
  class City {
    constructor(options = {}) { this.options = {demand: 1, balance: .5, strategy: 'fixed', cycle: 60, offset: 8, ...options}; this.reset(); }
    reset() {
      this.time = 0; this.nextId = 0; this.generated = 0; this.exited = 0; this.stops = 0; this.wait = 0; this.visits = 0;
      this.transit = []; this.backlog = Array.from({length: 6}, () => []); this.fraction = Array(6).fill(0);
      this.nodes = Array.from({length: 9}, () => ({q: [[], []], reserved: [0, 0], phase: 0, elapsed: 0, state: 'GREEN', last: [-99, -99], service: 0}));
      this.nodes.forEach((n, i) => this.control(n, i));
    }
    downstream(index, d) { return d === 0 ? (index < 6 ? index + 3 : -1) : (index % 3 < 2 ? index + 1 : -1); }
    pressure(index, d) {
      const next = this.downstream(index, d), n = this.nodes[index];
      return n.q[d].length - (next < 0 ? 0 : this.nodes[next].q[d].length + this.nodes[next].reserved[d]);
    }
    control(n, i) {
      const o = this.options;
      if (o.strategy === 'fixed') {
        const p = mod(this.time - (Math.floor(i / 3) + i % 3) * o.offset, o.cycle), half = o.cycle / 2, g = half - 5;
        n.phase = p < half ? 0 : 1;
        n.elapsed = p % half;
        n.state = n.elapsed < g ? 'GREEN' : n.elapsed < g + 3 ? 'YELLOW' : 'ALL RED';
      } else {
        if (n.state === 'YELLOW' && n.elapsed >= 3) { n.state = 'ALL RED'; n.elapsed = 0; }
        else if (n.state === 'ALL RED' && n.elapsed >= 2) { n.phase = 1 - n.phase; n.state = 'GREEN'; n.elapsed = 0; }
        else if (n.state === 'GREEN' && n.elapsed >= 8 && n.q[1 - n.phase].length) {
          const change = n.elapsed >= 28 || (o.strategy === 'actuated'
            ? n.q[n.phase].length === 0 && this.time - n.last[n.phase] >= 3
            : this.pressure(i, 1 - n.phase) > this.pressure(i, n.phase));
          if (change) { n.state = 'YELLOW'; n.elapsed = 0; }
        }
      }
    }
    enqueue(i, d, vehicle) {
      const n = this.nodes[i];
      if (n.state !== 'GREEN' || n.phase !== d || n.q[d].length) this.stops++;
      n.q[d].push({id: vehicle.id, joined: this.time}); n.last[d] = this.time;
    }
    step(count = 1) {
      for (let tick = 0; tick < count; tick++) {
        this.nodes.forEach((n, i) => this.control(n, i));
        // Complete reserved transfers before admitting boundary demand.
        const remaining = [];
        this.transit.forEach(v => {
          if (v.at <= this.time) { this.nodes[v.to].reserved[v.d]--; this.enqueue(v.to, v.d, v); }
          else remaining.push(v);
        });
        this.transit = remaining;
        for (let b = 0; b < 6; b++) {
          const d = b < 3 ? 0 : 1, i = d === 0 ? b : (b - 3) * 3;
          const share = d === 0 ? this.options.balance : 1 - this.options.balance;
          const pulse = mod(this.time + b * 7, 50) < 18 ? 1.5 : .72;
          this.fraction[b] += .38 * this.options.demand * share * 2 * pulse;
          while (this.fraction[b] >= 1) { this.fraction[b]--; this.backlog[b].push({id: this.nextId++}); this.generated++; }
          const n = this.nodes[i];
          while (this.backlog[b].length && n.q[d].length + n.reserved[d] < 20) this.enqueue(i, d, this.backlog[b].shift());
        }
        this.nodes.forEach((n, i) => {
          if (n.state !== 'GREEN') { n.service = 0; return; }
          n.service++;
          if (n.service < 2) return;
          n.service = 0;
          const d = n.phase, next = this.downstream(i, d);
          if (!n.q[d].length || (next >= 0 && this.nodes[next].q[d].length + this.nodes[next].reserved[d] >= 20)) return;
          const v = n.q[d].shift(); this.wait += this.time - v.joined; this.visits++;
          if (next < 0) this.exited++;
          else { this.nodes[next].reserved[d]++; this.transit.push({id: v.id, from: i, to: next, d, at: this.time + 4}); }
        });
        this.nodes.forEach(n => { n.elapsed++; });
        this.time++;
      }
    }
    get metrics() {
      const queued = this.nodes.reduce((s, n) => s + n.q[0].length + n.q[1].length, 0);
      return {queued, average: queued / 18, wait: this.visits ? this.wait / this.visits : null, stops: this.stops, throughput: this.exited,
        backlog: this.backlog.reduce((s, q) => s + q.length, 0), transit: this.transit.length, generated: this.generated};
    }
  }
  // Microscopic corridor with positions in meters, deterministic spacing and admission control.
  class Corridor {
    constructor(cycle = 80, offset = 36, speed = 40) { this.configure(cycle, offset, speed); }
    configure(cycle, offset, speed) { this.cycle = cycle; this.offset = offset; this.speed = speed / 3.6; this.reset(); }
    reset() { this.time = 0; this.cars = Array.from({length: 8}, (_, i) => ({x: -8 - i * 9, passed: -1})); }
    green(i, t = this.time) { return mod(t - i * this.offset, this.cycle) < 28; }
    advance(dt) {
      // Caller uses <= 1/30 s steps, including seeks, so no signal can be skipped.
      this.cars.forEach((car, j) => {
        let limit = j ? this.cars[j - 1].x - 9 : Infinity;
        const next = car.passed + 1;
        if (next < 5 && !this.green(next)) limit = Math.min(limit, next * 400 - 8);
        car.x = Math.max(car.x, Math.min(car.x + this.speed * dt, limit));
        if (next < 5 && car.x > next * 400 - 8) car.passed = next;
      });
      this.time += dt;
    }
    seek(t) { this.reset(); while (this.time + 1/30 < t) this.advance(1/30); this.advance(Math.max(0, t - this.time)); }
    bandwidth() {
      let longest = 0, current = 0, start = 0, bestStart = 0;
      for (let departure = 0; departure < 28; departure += .1) {
        const ok = Array.from({length: 5}, (_, i) => mod(departure + i * 400 / this.speed - i * this.offset, this.cycle) < 28).every(Boolean);
        if (ok) { if (!current) start = departure; current += .1; if (current > longest) { longest = current; bestStart = start; } } else current = 0;
      }
      return {width: Math.min(28, longest), start: bestStart};
    }
  }
  function mpcPlan(q, phase, time) {
    const horizon = 5; let best = null;
    for (let mask = 0; mask < 32; mask++) {
      let queues = [...q], p = phase, cost = 0, steps = [];
      for (let k = 0; k < horizon; k++) {
        const change = (mask >> k) & 1; if (change) p = 1 - p;
        const arrivals = [mod(time + k, 6) < 3 ? 6 : 2, mod(time + k, 6) >= 3 ? 6 : 2];
        queues = queues.map((v, d) => Math.max(0, v + arrivals[d] - (d === p ? (change ? 3 : 6) : 0)));
        cost += queues[0] + queues[1];
        steps.push({phase: p, change: !!change, q: [...queues]});
      }
      if (!best || cost < best.cost) best = {cost, steps};
    }
    return best;
  }
  const api = {mod, clamp, movements, conflicts, story, heroSignal, Hero, fixedQueues, comparison, City, Corridor, mpcPlan};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TrafficModels = api;
})(typeof globalThis === 'undefined' ? this : globalThis);
