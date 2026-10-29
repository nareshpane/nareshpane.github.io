/* Shared numerical/heatmap view; no independent mathematical state. */
(function(L){
 'use strict';
 L.matrixView=function(a,{caption='Adjacency matrix',rowOrder=L.range(a.length),colOrder=rowOrder,rowLabels=null,colLabels=null,selected=[],selectedColumns=selected,selectedEntries=[],cells=null,cellPalette=L.palette,difference=false,heatmap=a.length>8,detail=true}={}){
   const esc=L.escape,n=a.length,colours=cells?L.colours(cells,n):null;
   const colour=v=>colours?cellPalette[colours[v]%cellPalette.length]:L.palette[v%L.palette.length];
   const rl=v=>rowLabels?rowLabels[v]:v+1,cl=v=>colLabels?colLabels[v]:v+1;
   const tip=(u,v)=>`${caption}; row ${rl(u)}, column ${cl(v)}: ${a[u][v]}`;
   const entry=(u,v)=>selectedEntries.some(([i,j])=>i===u&&j===v);
   function blockStyle(u,v,i,j){
     if(!cells||cells.length>3||difference)return '';
     const cu=colours[u],cv=colours[v],tint=cu===cv?`color-mix(in srgb,${colour(u)} 32%,white)`:'#fff0dc';
     return `background:${a[u][v]?tint:'#fafbf6'};${i&&colours[rowOrder[i-1]]!==cu?'border-top:2px solid #81967d;':''}${j&&colours[colOrder[j-1]]!==cv?'border-left:2px solid #81967d;':''}`;
   }
   function table(){
     const headers=colOrder.map(v=>`<th scope="col" style="border-top:3px solid ${colour(v)}" class="${selectedColumns.includes(v)?'selected-label':''}">${esc(cl(v))}</th>`).join('');
     const rows=rowOrder.map((u,i)=>`<tr data-row="${u}" class="${selected.includes(u)?'selected-row':''}"><th scope="row" style="border-left:3px solid ${colour(u)}">${esc(rl(u))}</th>${colOrder.map((v,j)=>`<td data-row="${u}" data-col="${v}" title="${esc(tip(u,v))}" style="${blockStyle(u,v,i,j)}" class="${a[u][v]?'one ':''}${selected.includes(u)||selectedColumns.includes(v)?'picked ':''}${entry(u,v)?'entry-picked ':''}${difference?(a[u][v]?'mismatch':'equal'):''}">${a[u][v]}</td>`).join('')}</tr>`).join('');
     return `<div class="matrix-scroll"><p class="matrix-caption">${esc(caption)}</p><table class="matrix" aria-label="${esc(caption)}"><caption class="visually-hidden">${esc(caption)}</caption><thead><tr><th scope="col">${difference?'Δ':' '}</th>${headers}</tr></thead><tbody>${rows}</tbody></table></div>`;
   }
   if(!heatmap)return table();
   const size=18,offset=30,w=offset+size*colOrder.length+5,h=offset+size*rowOrder.length+5;
   let svg=`<figure class="matrix-heatmap"><figcaption>${esc(caption)}</figcaption><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(caption)}; ${n} by ${a[0].length} matrix. Hover for exact entries.">`;
   colOrder.forEach((v,j)=>{svg+=`<text x="${offset+j*size+9}" y="18" text-anchor="middle" font-size="9" fill="#254932">${esc(cl(v))}</text>`;});
   rowOrder.forEach((u,i)=>{svg+=`<text x="19" y="${offset+i*size+12}" text-anchor="end" font-size="9">${esc(rl(u))}</text>`;colOrder.forEach((v,j)=>{const pick=selected.includes(u)||selectedColumns.includes(v);svg+=`<rect data-row="${u}" data-col="${v}" x="${offset+j*size}" y="${offset+i*size}" width="${size-1}" height="${size-1}" fill="${difference?(a[u][v]?'#dc807a':'#e4f2e3'):a[u][v]?(cells&&cells.length<=3?(colours[u]===colours[v]?colour(u):'#ecc49e'):'#94c7a0'):'#f2f3eb'}" stroke="${pick?'#d38b3f':cells&&colours[u]===colours[v]?colour(u):'#e1e5d9'}" stroke-width="${pick?2:1}"><title>${esc(tip(u,v))}</title></rect>`;});});
   if(cells){let at=0;for(const c of cells){svg+=`<path d="M${offset+at*size} ${offset}V${h-5} M${offset} ${offset+at*size}H${w-5}" stroke="#8a9b83" stroke-width="1.5" fill="none"/>`;at+=c.length;}}
   return svg+'</svg></figure>'+(detail?'<details class="matrix-detail"><summary>Detailed numerical entries</summary>'+table()+'</details>':'');
 };
 L.matrix=(g,order=L.range(g.n),selected=[],options={})=>L.matrixView(g.a,{rowOrder:order,colOrder:order,selected,...options});
 L.bindMatrix=function(element,onVertex){if(!element)return;element.querySelectorAll('th[scope="row"]').forEach(th=>{th.tabIndex=0;th.setAttribute('role','button');th.setAttribute('aria-label','Select vertex '+th.textContent);const select=()=>onVertex(Number(th.parentElement.dataset.row));th.addEventListener('click',select);th.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});});};
})(GraphLab);
