(function(L){
 'use strict';
 const NS='http://www.w3.org/2000/svg';
 L.palette=['#96d7b0','#f4ae78','#9cbce9','#d5b0e8','#ead36f','#81cfd0','#eaa7ba','#bdca82'];
 L.escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 L.positions=function(n,layout='circle'){
   if(layout==='prism6')return [[90,55],[35,205],[150,205],[220,95],[165,255],[285,255]];
   if(layout==='paper9')return [[250,235],[260,150],[250,65],[160,250],[160,150],[160,50],[70,235],[60,150],[70,65]];
   if(layout==='prism12')return L.range(12).map(i=>{let t=-Math.PI/2+2*Math.PI*(i%6)/6,r=i<6?112:65;return [160+r*Math.cos(t),150+r*Math.sin(t)];});
   if(layout==='grid')return L.range(n).map(i=>[55+(i%4)*70,45+Math.floor(i/4)*70]);
   return L.range(n).map(i=>{let t=-Math.PI/2+2*Math.PI*i/n;return [160+113*Math.cos(t),150+113*Math.sin(t)];});
 };
 L.svg=function(g,{cells=[L.range(g.n)],selected=[],labels=null,positions=null,layout='circle',mapping=null,title='Undirected simple graph',mini=false,highlightEdges=[],interactive=false,edgeColour=null,vertexColour=null}={}){
   const p=positions||L.positions(g.n,layout),c=L.colours(cells,g.n),r=g.n>20?8:g.n>12?12:17;
   let s=`<svg viewBox="0 0 320 300" role="${interactive?'group':'img'}" aria-label="${L.escape(title)}" class="graph-svg"><title>${L.escape(title)}</title>`;
   for(const [u,v] of L.edges(g)){
     let hot=highlightEdges.some(e=>e.includes(u)&&e.includes(v)),stroke=hot?'#b43232':edgeColour?edgeColour(u,v):'#89948d',width=hot?4:g.n>20?1.1:2;
     const [x,y]=p[u],[xx,yy]=p[v],curved=layout==='grid'&&((x===xx&&Math.abs(y-yy)>71)||(y===yy&&Math.abs(x-xx)>71));
     if(curved){const bend=22+Math.hypot(xx-x,yy-y)/7,cx=(x+xx)/2+(x===xx?bend:0),cy=(y+yy)/2-(y===yy?bend:0);s+=`<path d="M${x},${y} Q${cx},${cy} ${xx},${yy}" fill="none" stroke="${stroke}" stroke-width="${width}"/>`;}
     else s+=`<line x1="${x}" y1="${y}" x2="${xx}" y2="${yy}" stroke="${stroke}" stroke-width="${width}"/>`;
   }
   for(let v=0;v<g.n;v++){let label=labels?labels[v]:mapping?mapping[v]+1:v+1;s+=`<g class="vertex" data-vertex="${v}" ${interactive?'tabindex="0" role="button"':''} aria-label="Vertex ${L.escape(label)}, cell ${c[v]+1}"><circle cx="${p[v][0]}" cy="${p[v][1]}" r="${r}" fill="${(vertexColour?vertexColour(v):L.palette[c[v]%L.palette.length])}" stroke="${selected.includes(v)?'#d28c39':'#325a46'}" stroke-width="${selected.includes(v)?4:1.6}"/><text x="${p[v][0]}" y="${p[v][1]+4}" text-anchor="middle" font-size="${g.n>20?8:12}" font-weight="650" fill="#153e2b">${L.escape(label)}</text></g>`;}
   return s+'</svg>';
 };
 L.render=function(element,g,options={}){element.innerHTML=L.svg(g,{...options,interactive:Boolean(options.onVertex)});if(options.onVertex)element.querySelectorAll('[data-vertex]').forEach(el=>{const fn=()=>options.onVertex(Number(el.dataset.vertex));el.addEventListener('click',fn);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();fn();}});});};
 L.matrix=function(g,order=L.range(g.n),selected=[]){let s='<div class="matrix-scroll"><table class="matrix"><caption>Adjacency matrix</caption><thead><tr><th scope="col">A</th>'+order.map(v=>`<th scope="col">${v+1}</th>`).join('')+'</tr></thead><tbody>';for(const u of order)s+=`<tr><th scope="row">${u+1}</th>`+order.map(v=>`<td class="${g.a[u][v]?'one ':''}${selected.includes(u)||selected.includes(v)?'picked':''}">${g.a[u][v]}</td>`).join('')+'</tr>';return s+'</tbody></table></div>';};
 L.partitionHTML=p=>'<div class="cells" aria-label="Ordered partition">'+p.map((c,i)=>`<span class="cell" style="--cell:${L.palette[i%L.palette.length]}"><small>C${i+1}</small>{${c.map(v=>v+1).join(', ')}}</span>`).join('')+'</div>';
 L.treeSVG=function(tree,{selected=0,collapsed=new Set(),visited=new Set(),interactive=false,fit=false}={}){
   let visible=[],coords={},row={},walk=(id,depth)=>{visible.push(id);(row[depth]??=[]).push(id);if(!collapsed.has(id))tree.nodes[id].children.forEach(c=>walk(c,depth+1));};walk(0,0);
   let depth=Object.keys(row).length,width=Math.max(640,...Object.values(row).map(a=>a.length*62));
   Object.entries(row).forEach(([d,ids])=>ids.forEach((id,k)=>coords[id]=[(k+0.5)*width/ids.length,35+Number(d)*90]));
   let s=`<svg viewBox="0 0 ${width} ${depth*90+15}" ${fit?'':`style="min-width:${Math.min(width,1800)}px"`} class="tree-svg${fit?' fitted-tree':''}" role="${interactive?'group':'img'}" aria-label="Individualization refinement search tree">`;
   visible.forEach(id=>{const n=tree.nodes[id],p=coords[id];if(n.parent!==null){let q=coords[n.parent];s+=`<path d="M${q[0]},${q[1]+13} C${q[0]},${q[1]+50} ${p[0]},${p[1]-45} ${p[0]},${p[1]-13}" fill="none" stroke="${n.reason?'#c07777':'#a8b7ac'}" stroke-width="2"/>`;}});
   visible.forEach(id=>{const n=tree.nodes[id],[x,y]=coords[id],fill=n.reason?'#f3bcbc':id===selected?'#f2ae76':visited.has(id)?'#9cbce9':n.leaf?'#a4dcaa':'#f4f0e8';s+=`<g data-node="${id}" ${interactive?'tabindex="0" role="button"':''} aria-label="Node ${id}, sequence ${n.seq.map(v=>v+1).join(',')||'root'}${n.reason?', pruned':n.leaf?', discrete leaf':''}"><circle cx="${x}" cy="${y}" r="15" fill="${fill}" stroke="#48664f" stroke-width="${id===selected?3:1}"/><text x="${x}" y="${y+4}" text-anchor="middle" font-size="10">${n.reason?'×':id}</text><rect x="${x-(n.seq.join(",").length+2)*3.4-4}" y="${y+20}" width="${(n.seq.join(",").length+2)*6.8+8}" height="19" rx="3" fill="#fafbf7"/><text x="${x}" y="${y+33}" text-anchor="middle" font-size="10">${n.seq.length?'('+n.seq.map(v=>v+1).join(',')+')':'()'}</text></g>`;});return s+'</svg>';
 };
})(GraphLab);
