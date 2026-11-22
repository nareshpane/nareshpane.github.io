(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const {draw, stats, adjacency} = GraphAtlas;
  const labels = [...'ABCDEF'];
  const base = {
    labels, points: [[30,72],[76,24],[127,62],[181,23],[212,88],[88,116]],
    edges: [[0,1],[0,5],[1,2],[2,5],[2,3],[2,4],[3,4]]
  };
  const neighbours = adjacency(base);
  const baseStats = stats(base);
  const svg = byId('specimen-svg');
  const selector = byId('vertex');
  const lens = byId('lens');
  const letterList = vertices => vertices.map(i => labels[i]).join(', ');
  const all = labels.map((_,i) => i);

  function pathFromA(target) {
    const queue = [0], parent = new Map([[0,null]]);
    for (let i=0; i<queue.length; i++) {
      for (const v of neighbours[queue[i]]) {
        if (parent.has(v)) continue;
        parent.set(v,queue[i]);
        queue.push(v);
      }
    }
    const path = [];
    for (let i=target; i!==null; i=parent.get(i)) path.unshift(i);
    return path;
  }

  function matrix(v) {
    const cells = neighbours.map((row,i) => '<tr'+(i===v?' class="selected"':'')+'><th scope="row">'+labels[i]+'</th>'+
      all.map(j => '<td>'+Number(row.includes(j))+'</td>').join('')+'</tr>').join('');
    byId('matrix').innerHTML = '<table><caption>Six-vertex graph · A(G)</caption><thead><tr><td></td>'+
      labels.map(l=>'<th scope="col">'+l+'</th>').join('')+'</tr></thead><tbody>'+cells+'</tbody></table>';
  }
  function inspector(restoreFocus = false) {
    const v = selector.selectedIndex, name = labels[v], mode = lens.value;
    let graph = {...base}, members = all;
    let title, formula, description, instruction;
    let kind = 'Simple undirected graph';
    let near = neighbours[v];
    if (mode === 'components') {
      members = all.filter(w => w !== v);
      graph = induced(members);
      const parts = stats(graph).components;
      graph.groups = graph.points.map((_,i) => parts.findIndex(part => part.includes(i)));
      kind = 'After vertex deletion';
      title = 'Remove vertex '+name;
      formula = parts.length+' component'+(parts.length===1?'':'s');
      const sets = parts.map(part => '{'+letterList(part.map(i=>members[i]))+'}').join(' and ');
      description = 'Removing '+name+' also removes its '+neighbours[v].length+' incident edges. '+
        (parts.length>1 ? 'The graph splits, so '+name+' is an articulation vertex.' : 'The remaining vertices can still reach one another.')+
        '<span class="set">'+sets+'</span>';
      instruction = 'Choose a vertex to remove from the original graph.';
    } else if (mode === 'path') {
      const path = pathFromA(v);
      graph.highlightEdges = path.slice(1).map((w,i)=>[path[i],w]);
      graph.highlightVertices = [v];
      title = 'Distance from A';
      formula = 'd(A, '+name+') = '+(path.length-1);
      description = 'The distance is the fewest edges in a path between the two vertices. A shortest path here is '+path.map(w=>labels[w]).join(' – ')+'.';
      instruction = 'Select the destination. The route starts at A.';
    } else if (mode === 'induced') {
      members = [v,...near].sort((a,b)=>a-b);
      graph = induced(members);
      graph.highlightVertices = [members.indexOf(v)];
      kind = 'Induced subgraph';
      title = 'Closed neighbourhood';
      formula = 'N['+name+'] = {'+letterList(members)+'}';
      description = 'Keep the selected vertex and its neighbours. Their induced subgraph contains every original edge with both endpoints in this set.';
      instruction = 'Choose any original vertex using the vertex menu.';
    } else if (mode === 'directed') {
      const incoming = graph.edges.filter(([,b])=>b===v), outgoing = graph.edges.filter(([a])=>a===v);
      graph.directed = true;
      graph.highlightEdges = [...incoming,...outgoing];
      graph.highlightVertices = [v];
      kind = 'Directed graph';
      title = 'Arcs at '+name;
      formula = incoming.length+' in · '+outgoing.length+' out';
      description = 'An arc is an ordered pair. In-degree counts arrows entering '+name+' and out-degree counts arrows leaving it. Following the arrows can change which vertices are reachable.';
      instruction = 'Follow the arrowheads to read each arc.';
    } else {
      if (mode === 'complement') {
        const edges = [];
        for (const a of all) for (const b of all.slice(a+1)) {
          if (!neighbours[a].includes(b)) edges.push([a,b]);
        }
        graph.edges = edges;
        near = adjacency(graph)[v];
        kind = 'Complement graph';
      }
      graph.highlightEdges = graph.edges.filter(e=>e.includes(v));
      graph.highlightVertices = [v];
      graph.groups = all.map(w=>near.includes(w)?2:0);
      title = mode === 'complement' ? 'Complement degree' : 'Vertex degree';
      formula = 'deg('+name+') = '+near.length;
      description = mode === 'complement' ?
        'The complement joins exactly the missing pairs. The original degree '+neighbours[v].length+' and complement degree '+near.length+' sum to n − 1 = 5.' :
        'The degree counts edges incident to '+name+'. Its open neighbourhood is the set of vertices joined directly to it.';
      description += '<span class="set">N('+name+') = {'+letterList(near)+'}</span>';
      instruction = 'Select a vertex to highlight its incident edges.';
    }
    graph.radius = 8;
    graph.selected = members.indexOf(v);
    graph.select = (index,keyboard) => {
      selector.selectedIndex = members[index];
      inspector(keyboard);
    };
    graph.title = kind+'. '+title+', '+formula;
    draw(svg,graph);
    if (restoreFocus) (svg.querySelector('[aria-pressed="true"]') || selector).focus({preventScroll:true});
    byId('figure-info').textContent = kind;
    byId('specimen-caption').textContent = instruction;
    byId('inspection-result').innerHTML = '<p class="read-label">'+title+'</p><p class="big-formula">'+formula+'</p><p>'+description+'</p>';
    const state = stats(graph);
    const component = graph.directed ? 'weakly connected component' : 'connected component';
    byId('specimen-strip').innerHTML = '<span><i>n</i> = <b>'+state.n+'</b> vertices</span><span><i>m</i> = <b>'+state.m+'</b> '+(graph.directed?'arcs':'edges')+'</span><span><b>'+state.components.length+'</b> '+component+(state.components.length===1?'':'s')+'</span>';
    matrix(v);
  }
  function induced(members) {
    return {
      points: members.map(w => base.points[w]),
      labels: members.map(w => labels[w]),
      edges: base.edges.filter(edge => edge.every(w => members.includes(w)))
        .map(edge => edge.map(w => members.indexOf(w)))
    };
  }
  byId('degree-sequence').textContent = 'The degrees in vertex order are '+baseStats.degrees.join(', ')+'. Their sum is '+baseStats.degrees.reduce((a,b)=>a+b,0)+', twice the seven edges.';
  selector.addEventListener('change',()=>inspector());
  lens.addEventListener('change',()=>inspector());
  inspector();

  const prism = [[0,1],[1,2],[2,0],[3,4],[4,5],[5,3],[0,3],[1,4],[2,5]];
  const firstPoints = [[48,110],[120,13],[192,110],[86,90],[120,45],[154,90]];
  const otherPoints = [[58,23],[182,23],[203,71],[182,119],[58,119],[37,71]];
  const permutation = [2,4,0,5,1,3];
  function comparison(answer) {
    const same = byId('iso-example').value === '0';
    const target = same ? prism.map(e=>e.map(i=>permutation[i])) : [0,2,4].flatMap(a=>[1,3,5].map(b=>[a,b]));
    const witness = [[0,1],[1,2],[2,0]];
    const right = [...'123456'];
    const cards = [
      {title:'Drawing A',points:firstPoints,edges:prism,labels,highlightEdges:answer!==undefined?witness:[]},
      {title:'Drawing B',points:otherPoints,edges:target,labels:right,highlightEdges:answer!==undefined&&same?witness.map(e=>e.map(i=>permutation[i])):[],
        groups:answer!==undefined&&!same?[0,1,0,1,0,1]:[]}
    ];
    const pair = byId('iso-pair');
    pair.replaceChildren();
    cards.forEach((graph,i)=>{
      const card = document.createElement('article');
      card.className = 'plate tone-'+(i?'peach':'teal');
      card.innerHTML = '<h4>'+graph.title+'</h4><svg></svg><p class="metrics">n = 6 · m = 9 · 3-regular</p>';
      draw(card.querySelector('svg'),graph);
      pair.append(card);
    });
    byId('iso-answer').textContent = answer===undefined ? '' :
      (answer===same?'Correct. ':'Look again. ')+(same ?
      'An adjacency-preserving map sends A, B, C, D, E, F to '+permutation.map(i=>right[i]).join(', ')+', respectively. The highlighted triangle shows three corresponding edges.' :
      'A, B and C form a triangle. Drawing B is bipartite with parts {1, 3, 5} and {2, 4, 6}, so it has no odd cycle. This rules out an isomorphism.');
    for (const [id,value] of [['iso-yes',true],['iso-no',false]]) byId(id).setAttribute('aria-pressed',String(answer===value));
  }
  byId('iso-example').addEventListener('change',()=>comparison());
  byId('iso-yes').addEventListener('click',()=>comparison(true));
  byId('iso-no').addEventListener('click',()=>comparison(false));
  comparison();

  const nav = byId('guide-nav');
  const mobile = document.querySelector('.mobile-nav');
  const links = [...nav.querySelectorAll('a')];
  const entries = links.map(a=>({a,target:byId(a.hash.slice(1))})).filter(item=>item.target);
  function expand(button,open) {
    button.setAttribute('aria-expanded',String(open));
    byId(button.getAttribute('aria-controls')).hidden = !open;
  }
  const pinned = new Set();
  nav.querySelectorAll('button').forEach(button => button.addEventListener('click', () => {
    const open = button.getAttribute('aria-expanded') === 'false';
    expand(button,open);
    if (open) pinned.add(button); else pinned.delete(button);
  }));
  function keepVisible(link) {
    if (!link) return;
    const viewport = nav.getBoundingClientRect(), item = link.getBoundingClientRect();
    if (item.bottom > viewport.bottom-8) nav.scrollTop += item.bottom-viewport.bottom+8;
    if (item.top < viewport.top+8) nav.scrollTop += item.top-viewport.top-8;
  }
  function menu(open) {
    mobile.setAttribute('aria-expanded',String(open));
    nav.classList.toggle('open',open);
    if (open) keepVisible(nav.querySelector('[aria-current]'));
  }
  mobile.addEventListener('click',()=>menu(mobile.getAttribute('aria-expanded')==='false'));
  function revealTarget() {
    const oldAnchors = {anatomy:'single-edge',families:'complete-graphs',relationships:'isomorphism',
      experiments:'growth',collection:'empty-graph','archive-viewer':'empty-graph'};
    const hash = location.hash.slice(1);
    const target = byId(hash) || byId(oldAnchors[hash]);
    if (target?.tagName==='DETAILS') target.open = true;
    if (target && oldAnchors[hash]) {
      history.replaceState(null,'','#'+target.id);
      target.scrollIntoView();
    }
  }
  nav.addEventListener('click',event=>{
    const link = event.target.closest('a');
    if (!link) return;
    const target = byId(link.hash.slice(1));
    if (target.tagName==='DETAILS') target.open = true;
    menu(false);
    if (getComputedStyle(mobile).display !== 'none') {
      target.setAttribute('tabindex','-1');
      target.focus({preventScroll:true});
    }
  });
  document.addEventListener('keydown',event=>{
    if (event.key==='Escape' && nav.classList.contains('open')) {menu(false);mobile.focus();}
  });
  window.addEventListener('hashchange',revealTarget);
  revealTarget();
  let current, pending = false;
  function track() {
    pending = false;
    const threshold = getComputedStyle(mobile).display === 'none' ? 78 : 92;
    let next = entries[0];
    for (const entry of entries) {
      if (entry.target.getBoundingClientRect().top > threshold) break;
      next = entry;
    }
    if (innerHeight + scrollY >= document.documentElement.scrollHeight-2) {
      const linked = entries.find(entry => entry.a.hash === location.hash);
      const top = linked?.target.getBoundingClientRect().top;
      next = top >= 0 && top < innerHeight ? linked : entries.at(-1);
    }
    if (next===current) return;
    current?.a.removeAttribute('aria-current');
    const group = next.a.closest('.nav-group');
    const previous = nav.querySelector('.current-group');
    previous?.classList.remove('current-group');
    if (previous && previous !== group) {
      const oldToggle = previous.querySelector('button');
      if (oldToggle && !pinned.has(oldToggle)) expand(oldToggle,false);
    }
    group.classList.add('current-group');
    next.a.setAttribute('aria-current','location');
    const toggle = group.querySelector('button');
    if (toggle && (previous !== group || next.a.closest('ul'))) expand(toggle,true);
    current = next;
    byId('mobile-location').textContent = [...next.a.childNodes]
      .filter(node => node.nodeType === Node.TEXT_NODE).map(node => node.textContent).join('').trim();
    if (getComputedStyle(mobile).display==='none') {
      keepVisible(next.a);
    }
  }
  function schedule() {
    if (!pending) {pending = true;requestAnimationFrame(track);}
  }
  window.addEventListener('scroll',schedule,{passive:true});
  window.addEventListener('resize',schedule);
  window.addEventListener('hashchange',schedule);
  track();
})();
