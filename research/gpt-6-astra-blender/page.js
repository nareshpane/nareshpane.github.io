const base = new URL('.', import.meta.url);
const $ = id => document.getElementById(id);

async function graphLab() {
  const response = await fetch(new URL('data/city.json', base));
  if (!response.ok) throw new Error('City data could not be loaded.');
  const city = await response.json();
  const svg = $('graph-lab');
  const namespace = 'http://www.w3.org/2000/svg';
  const add = (name, attrs, text) => {
    const e = document.createElementNS(namespace, name);
    for (const [key,value] of Object.entries(attrs)) e.setAttribute(key,value);
    if (text) e.textContent=text;
    svg.append(e); return e;
  };
  function update() {
    const source=+$('origin').value, target=+$('destination').value;
    const failed=$('close-bridge').checked;
    const edges=city.roads.filter(e=>!(failed && e.id==='35-36'));
    const adjacency=city.nodes.map(()=>[]);
    for (const e of edges) {adjacency[e.u].push([e.v,e.seconds]);adjacency[e.v].push([e.u,e.seconds]);}
    const distance=city.nodes.map(()=>Infinity), previous=city.nodes.map(()=>null), seen=new Set();
    distance[source]=0;
    for (let step=0;step<city.nodes.length;step++) {
      let u=-1;
      for (let i=0;i<distance.length;i++) if(!seen.has(i)&&(u<0||distance[i]<distance[u]))u=i;
      if(u<0||!Number.isFinite(distance[u]))break;
      seen.add(u);
      for(const [v,cost] of adjacency[u]) if(distance[u]+cost<distance[v]){distance[v]=distance[u]+cost;previous[v]=u;}
    }
    const path=[];
    if(Number.isFinite(distance[target])) for(let at=target;at!==null;at=previous[at])path.unshift(at);
    const pathEdges=new Set(path.slice(1).map((v,i)=>[Math.min(path[i],v),Math.max(path[i],v)].join('-')));
    svg.replaceChildren();
    add('title',{},'Interactive synthetic road graph');
    add('rect',{x:-17,y:-170,width:34,height:340,fill:'#d4e7e7'});
    for(const e of edges){const a=city.nodes[e.u],b=city.nodes[e.v];add('line',{
      x1:a.x,y1:-a.y,x2:b.x,y2:-b.y,stroke:pathEdges.has(e.id)?'#b86723':'#b4c3be','stroke-width':pathEdges.has(e.id)?4:1.5});}
    if(failed){add('path',{d:'M -5 -53 L 5 -43 M -5 -43 L 5 -53',stroke:'#ba4038','stroke-width':3});}
    for(const n of city.nodes){
      const selected=n.id===source||n.id===target;
      add('circle',{cx:n.x,cy:-n.y,r:selected?6:3.8,fill:selected?'#b86723':'#286c70'});
      add('text',{x:n.x+6,y:-n.y-6,'font-size':6.4,fill:'#274448'},String(n.id));
    }
    const metres=edges.filter(e=>pathEdges.has(e.id)).reduce((sum,e)=>sum+e.length_m,0);
    $('route-output').textContent=path.length
      ? `Node ${source} → ${target}: ${metres.toFixed(2)} m · ${(distance[target]/60).toFixed(2)} minutes. Route: ${path.join(' → ')}.`
      : 'These origins are disconnected in this scenario.';
    svg.dataset.routeSeconds=String(distance[target]);
    svg.dataset.routeMetres=String(metres);
  }
  for(const id of ['origin','destination','close-bridge'])$(id).addEventListener('change',update);
  update();
}
graphLab().catch(e=>{$('route-output').textContent=e.message;});

for(const button of document.querySelectorAll('[data-access-view]'))button.addEventListener('click',()=>{
  const after=button.dataset.accessView==='after';
  $('access-image').src=new URL(`renders/access_${after?'after':'before'}.webp`,base).href;
  $('access-image').alt=after?'Building colours show accessibility after the new cross-river transit link.':'Building colours show accessibility with existing transit only.';
  $('access-caption').textContent=after?'After: the gold line adds a 90-second transit connection between nodes 33 and 38.':'Before: four stations serve the existing eastern transit corridor.';
  for(const other of document.querySelectorAll('[data-access-view]'))other.setAttribute('aria-pressed',String(other===button));
});

let viewer;
async function loadModel(kind) {
  const status=$('viewer-status');
  const targetURL=new URL(`models/${kind}_web.glb`,base).href;
  if(viewer?.src===targetURL && viewer.dataset.verifiedLoad==='true') {
    status.textContent='This model is loaded. Drag to rotate; scroll or pinch to zoom.';
    return;
  }
  status.textContent='Loading the interactive model…';
  for(const b of document.querySelectorAll('[data-model]'))b.disabled=true;
  try {
    await import('./vendor/model-viewer.min.js');
    if(!viewer){
      viewer=document.createElement('model-viewer');
      viewer.setAttribute('camera-controls','');viewer.setAttribute('shadow-intensity','0.6');
      viewer.setAttribute('camera-orbit','35deg 55deg 75%');viewer.setAttribute('environment-image','neutral');
      viewer.setAttribute('exposure','0.9');viewer.setAttribute('interaction-prompt','none');
      viewer.setAttribute('loading','eager');viewer.setAttribute('touch-action','pan-y');
      viewer.addEventListener('load',()=>{
        $('model-fallback').hidden=true;status.textContent='Model loaded. Drag to rotate; scroll or pinch to zoom. Arrow keys rotate when the model has focus.';
        viewer.dataset.verifiedLoad='true';
        for(const b of document.querySelectorAll('[data-model]'))b.disabled=false;
      });
      viewer.addEventListener('error',()=>{
        $('model-fallback').hidden=false;status.textContent='3D is unavailable in this browser. The still image and GLB downloads remain available.';
        for(const b of document.querySelectorAll('[data-model]'))b.disabled=false;
      });
      $('viewer-shell').append(viewer);
    }
    viewer.dataset.verifiedLoad='false';
    viewer.alt=kind==='city'?'Editable synthetic city with two river crossings, 204 buildings, branching trees and pedestrians.':'64 junctions and 108 road and footway edges in a graph-only 3D model.';
    viewer.src=targetURL;
    for(const b of document.querySelectorAll('[data-model]'))b.setAttribute('aria-pressed',String(b.dataset.model===kind));
  } catch(e) {status.textContent='The 3D viewer could not initialize. Use the still image or download the model.';for(const b of document.querySelectorAll('[data-model]'))b.disabled=false;}
}
for(const button of document.querySelectorAll('[data-model]'))button.addEventListener('click',()=>loadModel(button.dataset.model));
