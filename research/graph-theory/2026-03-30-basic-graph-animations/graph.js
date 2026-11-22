/* All diagrams use zero-based vertex indices and one entry per edge. */
window.GraphAtlas = (() => {
  const ns = 'http://www.w3.org/2000/svg';
  let drawing = 0;

  function element(tag, attributes = {}, content) {
    const node = document.createElementNS(ns, tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    if (content !== undefined) node.textContent = content;
    return node;
  }

  function adjacency({points, edges}) {
    const lists = points.map(() => []);
    for (const [a, b] of edges) {
      lists[a].push(b);
      lists[b].push(a);
    }
    return lists.map(list => list.sort((a,b) => a-b));
  }

  // For a digraph, components here refer to the underlying undirected graph.
  function stats(graph) {
    const lists = adjacency(graph);
    const seen = new Set();
    const components = [];
    for (let v = 0; v < lists.length; v++) {
      if (seen.has(v)) continue;
      const part = [v];
      seen.add(v);
      for (let i = 0; i < part.length; i++) {
        for (const w of lists[part[i]]) {
          if (seen.has(w)) continue;
          seen.add(w);
          part.push(w);
        }
      }
      components.push(part);
    }
    return {n: lists.length, m: graph.edges.length, degrees: lists.map(row => row.length), components};
  }

  function draw(svg, graph) {
    const {points, edges, labels, groups = [], highlightEdges = [], highlightVertices = [],
      directed, weights, select, title = 'Graph diagram', radius = 7.5} = graph;
    const id = 'drawing-' + (++drawing);
    const label = v => labels ? labels[v] : String(v);
    const key = (a,b) => directed || a < b ? a + '-' + b : b + '-' + a;
    const accent = new Set(highlightEdges.map(([a,b]) => key(a,b)));
    const layer = document.createDocumentFragment();
    svg.setAttribute('viewBox','0 0 240 142');
    svg.setAttribute('role', select ? 'group' : 'img');
    svg.setAttribute('aria-labelledby', id + '-title');
    layer.append(element('title', {id:id+'-title'}, title));
    if (directed) {
      const defs = element('defs');
      for (const variant of ['normal','lit']) {
        const marker = element('marker',{id:id+variant,viewBox:'0 0 6 6',refX:5.8,refY:3,markerWidth:5,markerHeight:5,orient:'auto',markerUnits:'userSpaceOnUse'});
        marker.append(element('path',{d:'M0 0 L6 3 L0 6 Z',fill:variant === 'lit' ? '#ad4840' : '#66747c'}));
        defs.append(marker);
      }
      layer.append(defs);
    }
    const weightLabels = [];
    edges.forEach(([a,b],index) => {
      const [x1,y1] = points[a], [x2,y2] = points[b];
      const lit = accent.has(key(a,b));
      const endGap = directed ? radius + 1.3 : 0;
      const length = Math.hypot(x2-x1,y2-y1);
      const edge = element('line',{x1,y1,x2:x2-(x2-x1)*endGap/length,y2:y2-(y2-y1)*endGap/length,
        class:'edge'+(lit ? ' lit' : ''), 'data-edge':a+'-'+b});
      if (directed) edge.setAttribute('marker-end','url(#'+id+(lit ? 'lit' : 'normal')+')');
      layer.append(edge);
      if (weights) {
        const x = (x1+x2)/2, y = (y1+y2)/2;
        weightLabels.push(element('rect',{x:x-6,y:y-5.5,width:12,height:11,rx:2,class:'weight-bg'}),
          element('text',{x,y,class:'weight'},weights[index]));
      }
    });
    layer.append(...weightLabels);
    points.forEach(([x,y],v) => {
      const lit = highlightVertices.includes(v);
      const node = element('g', {transform:'translate('+x+' '+y+')',class:'vertex group-'+(groups[v] ?? 0)+(lit?' lit':''),'data-vertex':v});
      if (select) {
        node.setAttribute('role','button');
        node.setAttribute('tabindex','0');
        node.setAttribute('aria-label','Inspect vertex '+label(v));
        node.setAttribute('aria-pressed',String(graph.selected === v));
        node.addEventListener('click',() => select(v,false));
        node.addEventListener('keydown',event => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            select(v,true);
          }
        });
        node.append(element('circle',{r:radius+4,class:'hit'}));
      }
      if (lit) node.append(element('circle',{r:radius+3,class:'halo'}));
      node.append(element('circle',{r:radius,class:'disk'}));
      if (label(v) !== '') node.append(element('text',{class:'node-label'},label(v)));
      layer.append(node);
    });
    svg.replaceChildren(layer);
  }

  return {draw, adjacency, stats};
})();
