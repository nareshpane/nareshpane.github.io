/* Small, deterministic experiments. The SVG and its readout share the same edge list. */
(() => {
  'use strict';

  const { draw, stats } = window.GraphAtlas;
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

  function pairs(n) {
    const edges = [];
    for (let u = 0; u < n; u++) {
      for (let v = u + 1; v < n; v++) edges.push([u, v]);
    }
    return edges;
  }

  function ring(n) {
    return Array.from({ length: n }, (_, v) => {
      const angle = 2 * Math.PI * v / n - Math.PI / 2;
      return [120 + 86 * Math.cos(angle), 71 + 54 * Math.sin(angle)];
    });
  }

  function setup(id, diagramCaption, settings) {
    const root = document.getElementById(id);
    root.innerHTML = `<div class="lab-grid">
      <figure class="lab-diagram"><svg role="img" viewBox="0 0 240 142"></svg>
        <figcaption>${diagramCaption}</figcaption></figure>
      <div class="lab-settings">${settings}</div></div>`;
    return {
      root, svg: root.querySelector('svg'),
      find: selector => root.querySelector(selector)
    };
  }

  function triangleCount(n, edges) {
    const neighbours = Array.from({ length: n }, () => new Set());
    edges.forEach(([u, v]) => {
      neighbours[u].add(v);
      neighbours[v].add(u);
    });
    let count = 0;
    for (const [u, v] of edges) {
      for (const w of neighbours[u]) {
        if (w > Math.max(u, v) && neighbours[v].has(w)) count++;
      }
    }
    return count;
  }

  function growthExperiment() {
    const path = Array.from({ length: 7 }, (_, i) => [i, i + 1]);
    const partitions = [0, 0, 0, 0, 1, 1, 1, 1];
    const bipartitePoints = partitions.map((side, v) => [side ? 190 : 50, 20 + v % 4 * 34]);
    const scenes = {
      path: {
        name: 'Build connectivity', edges: path, points: ring(8),
        note: 'Join eight isolated vertices into a path. Each new edge merges two components.'
      },
      complete: {
        name: 'Fill the complete graph', edges: pairs(8), points: ring(8),
        note: 'Fill the 28 unordered pairs of eight vertices. The last edge completes K₈.'
      },
      bipartite: {
        name: 'Grow a bipartite graph', points: bipartitePoints, groups: partitions,
        edges: [[0,4],[0,5],[1,5],[1,6],[2,6],[2,7],[3,4],[3,7]],
        note: 'Keep vertices 0–3 on the left and 4–7 on the right. The final eight cross-edges form an even cycle.'
      },
      triangle: {
        name: 'Close the first triangle', edges: [...path.slice(0, 5), [0, 2]],
        points: [[36,82],[64,22],[102,81],[138,39],[171,82],[207,45],[80,120],[171,120]],
        note: 'Extend a path, then join 0 to 2. The final edge closes the two-edge path 0–1–2 into a triangle.'
      }
    };
    for (const [key, scene] of Object.entries(scenes)) mountGrowth(key, scene);
  }

  function mountGrowth(key, scene) {
    const { root, svg, find } = setup('growth-' + key, 'The highlighted edge is the latest addition.', `
      <p class="experiment-note">${scene.note}</p>
      <div class="controls">
        <button type="button" data-action="back">Back</button>
        <button type="button" data-action="next">Step forward</button>
        <button type="button" data-action="play">Play</button>
        <button type="button" data-action="reset">Reset</button>
      </div>
      <div role="status" aria-atomic="true">
        <p class="metrics"></p><p class="readout"></p>
      </div>`);
    const play = find('[data-action="play"]');
    let frame = {path:3, complete:7, bipartite:3, triangle:5}[key];
    let timer = null;

    function pause() {
      window.clearInterval(timer);
      timer = null;
      play.textContent = 'Play';
    }

    function render() {
      const edges = scene.edges.slice(0, frame);
      const latest = edges[edges.length - 1];
      const graph = {
        points: scene.points, edges, groups: scene.groups,
        highlightEdges: latest ? [latest] : [],
        title: `${scene.name}, step ${frame} of ${scene.edges.length}`
      };
      const { n, m, components } = stats(graph);
      draw(svg, graph);
      find('.metrics').textContent = `Step ${frame} / ${scene.edges.length} · ${m} edge${m === 1 ? '' : 's'} · ${components.length} component${components.length === 1 ? '' : 's'}`;
      let observation = 'All eight vertices are isolated.';
      if (latest) {
        const before = stats({ ...graph, edges: edges.slice(0, -1) }).components.length;
        observation = `Added ${latest.join('–')}. `;
        if (key === 'triangle') {
          const count = triangleCount(n, edges);
          observation += count ? 'Vertices 0, 1 and 2 now span the first triangle.' : 'There are still no triangles.';
        } else if (before > components.length) {
          observation += `Two components merged${components.length === 1 ? ' into a single connected graph' : ''}.`;
        } else {
          observation += 'Its endpoints were already connected, so this edge creates a cycle.';
        }
      }
      find('.readout').textContent = observation;
      find('[data-action="back"]').disabled = frame === 0;
      find('[data-action="next"]').disabled = frame === scene.edges.length;
      find('[data-action="reset"]').disabled = frame === 0 && timer === null;
      if (frame === scene.edges.length) pause();
    }

    root.querySelector('.controls').addEventListener('click', event => {
      const action = event.target.dataset.action;
      if (!action) return;
      if (action === 'play' && timer === null) {
        if (frame === scene.edges.length) frame = 0;
        play.textContent = 'Pause';
        timer = window.setInterval(() => { frame++; render(); }, 950);
      } else {
        pause();
        if (action === 'back') frame--;
        if (action === 'next') frame++;
        if (action === 'reset') frame = 0;
      }
      render();
    });
    document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
    window.addEventListener('pagehide', pause);
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver(entries => {
        if (!entries[0].isIntersecting) pause();
      });
      observer.observe(root);
    }
    render();
  }

  // A fixed stream supplies one uniform threshold for each candidate pair.
  function uniformGenerator(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6D2B79F5) >>> 0;
      let x = Math.imul(state ^ (state >>> 15), state | 1);
      x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randomExperiment() {
    const { svg, find } = setup('random-lab', 'A fixed seed keeps each pair’s draw the same as you change the sliders.', `
      <p>In <i>G</i>(<i>n</i>, <i>p</i>), each unordered pair receives an edge independently with probability <i>p</i>.</p>
      <label for="random-n">Vertices <i>n</i> <output for="random-n" id="random-n-value"></output></label>
      <input type="range" id="random-n" min="2" max="12" step="1" value="8">
      <label for="random-p">Edge probability <i>p</i> <output for="random-p" id="random-p-value"></output></label>
      <input type="range" id="random-p" min="0" max="1" step="0.05" value="0.30">
      <div class="controls"><label for="random-seed">Seed</label>
        <input type="number" id="random-seed" min="0" max="999999" value="17" step="1">
        <button type="button" id="random-resample">Next sample</button></div>
      <div role="status" aria-atomic="true">
        <p class="metrics"></p><p class="readout"></p>
      </div>`);
    const nInput = find('#random-n');
    const pInput = find('#random-p');
    const seedInput = find('#random-seed');
    const candidates = pairs(Number(nInput.max));
    let currentSeed = Number(seedInput.value);
    function render() {
      const n = Number(nInput.value);
      const p = Number(pInput.value);
      const uniform = uniformGenerator(currentSeed);
      // Consume all thresholds even when v >= n. The common vertex pairs
      // keep their thresholds when the reader adds or removes a vertex.
      const edges = candidates.filter(([, v]) => uniform() < p && v < n);
      const graph = { points: ring(n), edges, labels: [...letters.slice(0, n)],
        title: `G(${n}, ${p}) at seed ${currentSeed} with ${edges.length} edges` };
      const { m, components } = stats(graph);
      const possible = n * (n - 1) / 2;
      draw(svg, graph);
      find('#random-n-value').value = n;
      find('#random-p-value').value = p.toFixed(2);
      find('.metrics').textContent = `Observed m = ${m} · ${components.length} component${components.length === 1 ? '' : 's'}`;
      find('.readout').textContent = `Model expectation E[m] = p × n(n − 1)/2 = ${p.toFixed(2)} × ${possible} = ${Number((p * possible).toFixed(2))}.`;
    }
    nInput.addEventListener('input', render);
    pInput.addEventListener('input', render);
    seedInput.addEventListener('change', () => {
      const seed = seedInput.valueAsNumber;
      if (Number.isInteger(seed) && seedInput.validity.valid) currentSeed = seed;
      seedInput.value = currentSeed;
      render();
    });
    find('#random-resample').addEventListener('click', () => {
      currentSeed = (currentSeed + 1) % (Number(seedInput.max) + 1);
      seedInput.value = currentSeed;
      render();
    });
    render();
  }

  // Enumerate simple routes on this four-vertex network. With nonnegative
  // weights a shortest simple path always exists between connected vertices.
  function routesToTarget(graph, source, target) {
    const routes = [];
    function extend(path, cost) {
      const last = path[path.length - 1];
      if (last === target) {
        routes.push({ path, cost });
        return;
      }
      graph.edges.forEach(([u, v], i) => {
        if (u !== last && v !== last) return;
        const next = u === last ? v : u;
        if (!path.includes(next)) extend([...path, next], cost + graph.weights[i]);
      });
    }
    extend([source], 0);
    return routes;
  }

  function weightedExperiment() {
    const { svg, find } = setup('weighted-lab', 'Edge labels give travel costs. Highlighted edges belong to a cheapest route from A to D.', `
      <p>Change the cost of A–B. A route’s total cost is the sum of its edge weights.</p>
      <label for="weighted-cost">Cost of A–B <output for="weighted-cost" id="weight-value"></output></label>
      <input type="range" id="weighted-cost" min="0" max="8" step="1" value="2">
      <ul class="route-costs" aria-label="Candidate route costs"></ul>
      <p class="readout" role="status" aria-atomic="true"></p>`);
    const input = find('input');
    const graph = {
      points: [[25,71],[110,20],[137,122],[215,71]],
      edges: [[0,1],[1,3],[0,2],[2,3],[0,3]], labels: [...'ABCD']
    };
    function render() {
      graph.weights = [Number(input.value), 1, 3, 2, 7];
      const routes = routesToTarget(graph, 0, 3);
      const min = Math.min(...routes.map(r => r.cost));
      const best = routes.filter(r => r.cost === min);
      const label = route => route.path.map(v => letters[v]).join('–');
      graph.highlightEdges = best.flatMap(r =>
        r.path.slice(1).map((v, i) => [r.path[i], v]));
      graph.title = `Weighted undirected graph. Shortest distance from A to D is ${min}.`;
      draw(svg, graph);
      find('#weight-value').value = input.value;
      find('.route-costs').innerHTML = routes.map(route =>
        `<li><span>${label(route)}</span> <strong>${route.cost}</strong></li>`).join('');
      find('.readout').textContent = best.length > 1
        ? `${best.map(label).join(' and ')} tie at cost ${min}.`
        : `${label(best[0])} is cheapest at cost ${min}.`;
    }
    input.addEventListener('input', render);
    render();
  }

  function componentGroups(n, parts) {
    const groups = Array(n);
    parts.forEach((part, group) => {
      part.forEach(v => { groups[v] = group; });
    });
    return groups;
  }

  function componentNames(parts, labels) {
    return parts.map(part => {
      const names = part.map(v => labels[v]).sort((a,b) => a - b);
      return '{' + names.join(', ') + '}';
    }).join(' · ');
  }

  function treeExperiment() {
    const original = {
      points: [[114,18],[65,53],[167,49],[30,89],[90,90],[168,85],[141,124],[206,124]],
      edges: [[0,1],[0,2],[1,3],[1,4],[2,5],[5,6],[5,7]]
    };
    const chord = [4,6];
    const path = routesToTarget({...original, weights:original.edges.map(() => 1)}, ...chord)[0].path;
    const pathEdges = path.slice(1).map((v,i) => [path[i],v]);
    const {svg,find} = setup('tree-test', 'Follow the highlighted route between vertices 4 and 6.', `
      <p>Join these two vertices directly, then remove the new edge to recover the tree.</p>
      <button type="button">Add edge 4–6</button>
      <div role="status" aria-atomic="true"><p class="metrics"></p><p class="readout"></p></div>`);
    const toggle = find('button');
    let added = false;
    function render() {
      const graph = {...original,
        edges: added ? [...original.edges, chord] : original.edges,
        highlightEdges: added ? [...pathEdges, chord] : pathEdges,
        highlightVertices: chord,
        title: added ? 'One cycle in the tree plus edge 4–6' : 'The original tree with the path from 4 to 6 highlighted'
      };
      const {n,m,components} = stats(graph);
      draw(svg,graph);
      const rank = m - n + components.length;
      find('.metrics').textContent = `n = ${n} · m = ${m} · ${rank} independent cycle${rank === 1 ? '' : 's'}`;
      find('.readout').textContent = added
        ? `The unique cycle is ${[...path,path[0]].join('–')}. Every cycle must use the new edge because the original tree is acyclic.`
        : `The only path between these endpoints is ${path.join('–')}.`;
      toggle.textContent = added ? 'Remove edge 4–6' : 'Add edge 4–6';
    }
    toggle.addEventListener('click', () => { added = !added; render(); });
    render();
  }

  function bridgeExperiment() {
    const original = {
      points: [[59,24],[22,71],[59,118],[96,71],[144,71],[181,24],[218,71],[181,118]],
      edges: [[0,1],[1,2],[2,3],[3,0],[3,4],[4,5],[5,6],[6,7],[7,4]]
    };
    const options = original.edges.map(([u,v], i) => `<option value="${i}">Remove ${u}–${v}</option>`);
    const {svg,find} = setup('bridge-test', 'Vertex positions stay fixed. Colours distinguish the components after deletion.', `
      <label for="bridge-edge">Delete one edge</label>
      <select id="bridge-edge"><option value="">Keep all edges</option>${options.join('')}</select>
      <p>Compare the edge between the two squares with any edge on a square.</p>
      <div role="status" aria-atomic="true"><p class="metrics"></p><p class="readout"></p></div>`);
    const select = find('select');
    const intact = stats(original).components.length;
    select.value = '4';
    function render() {
      const removed = select.value === '' ? -1 : Number(select.value);
      const graph = {...original, edges:original.edges.filter((edge,i) => i !== removed)};
      const {m,components} = stats(graph);
      graph.groups = componentGroups(graph.points.length,components);
      graph.highlightEdges = removed < 0 ? [original.edges[4]] : [];
      graph.title = removed < 0 ? 'Two squares joined by edge 3–4' :
        `Deleting ${original.edges[removed].join('–')} leaves ${components.length} component${components.length === 1 ? '' : 's'}`;
      draw(svg, graph);
      find('.metrics').textContent = `${m} edges · components before ${intact}, after ${components.length}`;
      let result = 'The edge 3–4 is the only route between the two squares.';
      if (removed >= 0) {
        const name = original.edges[removed].join('–');
        result = components.length > intact
          ? `Edge ${name} is a bridge. The two components are ${componentNames(components,graph.points.map((_,i) => i))}.`
          : `Edge ${name} lies on a cycle. The other three sides still connect its endpoints.`;
      }
      find('.readout').textContent = result;
    }
    select.addEventListener('change',render);
    render();
  }

  function cutExperiment() {
    const original = {
      points: [[23,37],[23,106],[76,73],[126,73],[178,32],[214,66],[155,108],[216,118]],
      edges: [[0,1],[1,2],[2,0],[2,3],[3,4],[4,5],[5,3],[3,6],[6,7]]
    };
    const ids = original.points.map((_,v) => v);
    const {svg,find} = setup('cut-test', 'Each choice starts from the intact graph. Surviving vertices keep their original labels.', `
      <label for="cut-vertex">Delete one vertex</label>
      <select id="cut-vertex"><option value="">Keep all vertices</option>${ids.map(v =>
        `<option value="${v}">Remove vertex ${v}</option>`).join('')}</select>
      <p>Select a vertex in the drawing or menu to remove it along with every incident edge.</p>
      <div role="status" aria-atomic="true"><p class="metrics"></p><p class="readout"></p></div>`);
    const select = find('select');
    const intact = stats(original).components.length;
    select.value = '3';
    function render() {
      const removed = select.value === '' ? -1 : Number(select.value);
      const labels = ids.filter(v => v !== removed);
      const graph = {
        labels, points: labels.map(v => original.points[v]),
        edges: original.edges.filter(([u,v]) => u !== removed && v !== removed)
          .map(([u,v]) => [labels.indexOf(u),labels.indexOf(v)]),
        select: (index,keyboard) => {
          select.value = String(labels[index]);
          render();
          if (keyboard) select.focus({preventScroll:true});
        }
      };
      const {n,m,components} = stats(graph);
      graph.groups = componentGroups(n,components);
      graph.title = removed < 0 ? 'Original graph, all vertices present' :
        `Deleting vertex ${removed} leaves ${components.length} component${components.length === 1 ? '' : 's'}`;
      draw(svg,graph);
      svg.querySelectorAll('[data-vertex]').forEach(node => {
        node.setAttribute('aria-label','Remove vertex ' + labels[Number(node.dataset.vertex)]);
        node.removeAttribute('aria-pressed');
      });
      find('.metrics').textContent = `n = ${n} · m = ${m} · components before ${intact}, after ${components.length}`;
      let result = 'All eight vertices lie in one component.';
      if (removed >= 0) {
        result = components.length > intact
          ? `Vertex ${removed} is a cut vertex. The remaining components are ${componentNames(components,labels)}.`
          : `The seven remaining vertices stay connected after deleting ${removed}.`;
      }
      find('.readout').textContent = result;
    }
    select.addEventListener('change',render);
    render();
  }

  function matchingExperiment() {
    const graph = {points:[[27,84],[89,55],[151,84],[213,55]],edges:[[0,1],[1,2],[2,3]]};
    const {svg,find} = setup('matching-test', 'Highlighted edges are your selection. A ring marks each shared endpoint.', `
      <p>Select edges on this four-vertex path. A matching may use each vertex at most once.</p>
      <div class="controls matching-options" role="group" aria-label="Edges in the selection">
        ${graph.edges.map(([u,v],i) => `<label><input type="checkbox" value="${i}" ${i===1?'checked':''}> ${u}–${v}</label>`).join('')}
      </div>
      <div role="status" aria-atomic="true"><p class="metrics"></p><p class="readout"></p></div>`);
    const options = find('.matching-options');
    const inputs = Array.from(options.querySelectorAll('input'));
    // Three edges give eight subsets; enumerate them to find the optimum.
    let maximum = 0;
    for (let mask = 0; mask < 2 ** graph.edges.length; mask++) {
      const edges = graph.edges.filter((edge,i) => (mask & (1 << i)) !== 0);
      if (new Set(edges.flat()).size === 2 * edges.length) maximum = Math.max(maximum,edges.length);
    }
    function render() {
      const edges = inputs.filter(el => el.checked).map(el => graph.edges[Number(el.value)]);
      const {degrees} = stats({...graph, edges});
      const conflicts = degrees.flatMap((d,v) => d > 1 ? [v] : []);
      const extensions = graph.edges.filter(([u,v]) => degrees[u] === 0 && degrees[v] === 0);
      const size = edges.length;
      let result;
      if (conflicts.length) {
        result = `Invalid matching. Selected edges share ${conflicts.length === 1 ? 'vertex' : 'vertices'} ${conflicts.join(' and ')}.`;
      } else if (size === maximum && degrees.every(d => d === 1)) {
        result = 'Perfect, maximum and maximal. Both chosen edges together cover all four vertices.';
      } else if (!extensions.length) {
        result = `Maximal because no more edges fit. The two outer edges give a larger matching of size ${maximum}.`;
      } else {
        result = `This matching can grow. Add ${extensions[0].join('–')} without sharing an endpoint.`;
      }
      draw(svg,{...graph, highlightEdges:edges, highlightVertices:conflicts, title:result});
      find('.metrics').textContent = `${size} selected · maximum matching size = ${maximum}`;
      find('.readout').textContent = result;
    }
    options.addEventListener('change',render);
    render();
  }

  growthExperiment();
  randomExperiment();
  weightedExperiment();
  treeExperiment();
  bridgeExperiment();
  cutExperiment();
  matchingExperiment();
})();
