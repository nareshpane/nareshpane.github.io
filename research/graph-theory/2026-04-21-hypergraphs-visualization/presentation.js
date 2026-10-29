/* Compact reader navigation, historical anchors, and data-derived uniform figures. */
(()=>{
 'use strict';
 const $=id=>document.getElementById(id),U=window.UniformExamples,S=window.ResearchSVG;
 const colours=['#aa653e','#51805b','#527da3'],fills=['#fff0e3','#edf6e4','#eaf3fc'];
 const positions=[[40,40],[145,40],[250,40],[40,170],[145,170],[250,170]],hubs=[[87,105],[145,105],[203,105]];
 function diagram(example){let lines='',nodes='';example.edges.forEach((edge,j)=>{
   if(example.k===2){const a=positions[U.vertices.indexOf(edge[0])],b=positions[U.vertices.indexOf(edge[1])];lines+=`<line class="uniform-incidence" data-edge="${j}" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${colours[j]}" stroke-width="3"/>`;}
   else {const [x,y]=hubs[j];edge.forEach(v=>{const [vx,vy]=positions[U.vertices.indexOf(v)];lines+=`<line class="uniform-incidence" data-edge="${j}" data-vertex="${v}" x1="${vx}" y1="${vy}" x2="${x}" y2="${y}" stroke="${colours[j]}" stroke-width="2"/>`;});nodes+=`<rect class="uniform-hub" data-edge="${j}" x="${x-13}" y="${y-13}" width="26" height="26" rx="4" fill="#fffdf9" stroke="${colours[j]}"/>`+S.text(x,y+5,'e'+(j+1));}
 });
 U.vertices.forEach((v,j)=>{const [x,y]=positions[j];nodes+=`<circle cx="${x}" cy="${y}" r="15" fill="#fffdf9" stroke="#49654c"/>`+S.text(x,y+5,v);});
 return S.svg(lines+nodes,example.k+'-uniform family: '+example.edges.map(e=>'{'+e.join(',')+'}').join(', '),290,210);
 }
 const descriptions=['Every hyperedge joins two vertices. Under the simple-undirected convention, this is an ordinary graph; F is isolated.','Every hyperedge has three members. Vertices may belong to several hyperedges without having degree three.','Every hyperedge has four members. Incidence spokes make the heavily overlapping memberships unambiguous.'];
 $('uniform-cards').innerHTML=U.examples.map((e,i)=>{
  const t=U.summarize(e);
  const tokens=e.edges.map((a,j)=>`<button data-edge="${j}" aria-pressed="false" aria-label="Highlight edge ${j+1}: ${a.join(',')}">{${a.join(',')}}${j<e.edges.length-1?',':''}</button>`).join('');
  return `<figure class="uniform-card" data-k="${e.k}" style="--uniform-bg:${fills[i]};--uniform-border:${colours[i]}"><h3>${e.k}-Uniform Hypergraph</h3><div>${diagram(e)}</div><div class="uniform-sets"><span>ℰ<sub>${e.k}</sub> = {</span>${tokens}<span>}</span></div><figcaption><strong>|e|=${e.k} ∀e∈ℰ<sub>${e.k}</sub></strong>${descriptions[i]}<span class="uniform-counts">Degrees (A,…,F): (${t.degrees.join(',')}).<br>Σdeg(v)=${t.degreeSum}=${e.k}×${e.edges.length}.</span></figcaption></figure>`;
 }).join('');
 $('uniform-totals').innerHTML=U.examples.map(e=>`<span>${e.k}-uniform: ${e.k}(${e.edges.length})=${U.summarize(e).incidences} incidences</span>`).join('');
 $('uniform-cards').addEventListener('click',e=>{const b=e.target.closest('button[data-edge]');if(!b)return;const card=b.closest('.uniform-card'),edge=b.dataset.edge,active=card.dataset.active===edge?null:edge;if(active===null)delete card.dataset.active;else card.dataset.active=active;card.querySelectorAll('button[data-edge]').forEach(x=>x.setAttribute('aria-pressed',x.dataset.edge===active));card.querySelectorAll('svg [data-edge]').forEach(x=>x.classList.toggle('active',x.dataset.edge===active));});

 const nav=$('reading-nav'),menu=$('section-menu'),current=$('current-section'),links=[...menu.querySelectorAll('a')],items=links.map(a=>({a,el:$(a.hash.slice(1)),id:a.hash,label:a.textContent,top:0})).filter(x=>x.el);
 let cached=[],height=52,total=1,frame=0,layoutFrame=0;
 function update(){frame=0;const point=scrollY+height+24;let lo=0,hi=cached.length;while(lo<hi){const mid=(lo+hi)>>1;if(cached[mid].top<=point)lo=mid+1;else hi=mid;}const item=cached[Math.max(0,lo-1)];if(item){current.textContent=item.label;current.href=item.id;links.forEach(a=>{if(a===item.a)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});current.setAttribute('aria-current','location');}$('reading-progress').value=Math.max(0,Math.min(100,scrollY/total*100));}
 function schedule(){if(!frame)frame=requestAnimationFrame(update);}
 function measure(){layoutFrame=0;height=nav.offsetHeight;cached=items.map(x=>({...x,top:x.el.getBoundingClientRect().top+scrollY})).sort((a,b)=>a.top-b.top);total=Math.max(1,document.documentElement.scrollHeight-innerHeight);schedule();}
 function layout(){if(!layoutFrame)layoutFrame=requestAnimationFrame(measure);}
 addEventListener('scroll',schedule,{passive:true});addEventListener('resize',layout,{passive:true});
 new ResizeObserver(layout).observe($('top'));
 new IntersectionObserver(entries=>{nav.classList.toggle('is-stuck',!entries[0].isIntersecting&&entries[0].boundingClientRect.top<0);}).observe($('nav-sentinel'));
 menu.addEventListener('keydown',e=>{if(e.key==='Escape'){menu.open=false;menu.querySelector('summary').focus();}});
 document.addEventListener('click',e=>{if(!menu.contains(e.target))menu.open=false;});
 function destination(hash){let id;try{id=decodeURIComponent(hash.slice(1));}catch{return null;}return $(id);}
 function align(hash){const el=destination(hash);if(!el)return;menu.open=false;let parent=el.closest('details');while(parent){parent.open=true;parent=parent.parentElement.closest('details');}if(['gallery','original-april-2026'].includes(el.id))$('original-april-2026').open=true;measure();scrollTo({top:Math.max(0,el.getBoundingClientRect().top+scrollY-nav.offsetHeight-12),behavior:'instant'});update();}
 document.addEventListener('click',e=>{const a=e.target.closest('a[href^="#"]');if(!a||e.button!==0||e.metaKey||e.ctrlKey||e.shiftKey||e.altKey||!destination(a.hash))return;e.preventDefault();if(location.hash!==a.hash)history.pushState(null,'',a.hash);align(a.hash);});
 addEventListener('hashchange',()=>align(location.hash||'#top'));addEventListener('popstate',()=>align(location.hash||'#top'));
 document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',layout));
 if(document.fonts)document.fonts.ready.then(layout);
 addEventListener('load',()=>{layout();if(location.hash)align(location.hash);});
 measure();if(location.hash)align(location.hash);
})();
