/* Presentation controller for the pure HyperStory timeline. One RAF loop, cached SVG nodes. */
window.HyperIntro={init(visual) {
  'use strict';
  const M=window.HyperMath,S=window.HyperStory;
  const {svg,text,line,point,diagram,subset,radialPositions,counterexampleFigures}=visual;
  const $=id=>document.getElementById(id),full=M.example(),pos=radialPositions(full);
  const colour=i=>S.colours[i],pairKey=pair=>pair.join(','),reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const token=(e,i,active=false)=>`<span class="edge-token ${active?'active':''}" style="--edge-colour:${colour(i)}">e<sub>${i+1}</sub>=${subset(e)}</span>`;
  const allPairs=M.twoSection(full);
  let pairSvg=allPairs.map(pair=>line(pos[full.vertices.indexOf(pair[0])],pos[full.vertices.indexOf(pair[1])],'#477d62',2,1,`data-pair="${pairKey(pair)}" class="projection-link" pathLength="1"`)).join('');
  pairSvg+=full.vertices.map((v,i)=>point(pos[i],v,'#fffdf9','#38664c',`class="vertex-node" data-vertex="${v}"`)).join('');
  $('intro-graph').innerHTML=svg(pairSvg,'Ordinary 2-section on the same six vertices A–F. Every visible line is a pair contained in a current hyperedge.');
  $('intro-hyper').innerHTML=diagram(full);
  $('intro-counterexample').innerHTML=counterexampleFigures();
  const pairNodes=[...$('intro-graph').querySelectorAll('[data-pair]')];
  const edgeNodes=[...$('intro-hyper').querySelectorAll('[data-edge]')];
  const vertexNodes=[...document.querySelectorAll('#intro-graph .vertex-node,#intro-hyper .vertex-node')];
  const matrix=M.incidence(full);
  $('intro-matrix').innerHTML='<table aria-label="Animation incidence matrix"><caption>B: a 1 means membership</caption><thead><tr><th scope="col">V / ℰ</th>'+full.edges.map((_,i)=>`<th scope="col" data-column="${i}" style="--edge-colour:${colour(i)}">e<sub>${i+1}</sub></th>`).join('')+'<th scope="col">deg</th></tr></thead><tbody>'+matrix.map((row,i)=>'<tr><th scope="row">'+full.vertices[i]+'</th>'+row.map((x,j)=>`<td data-column="${j}" data-value="${x}" style="--edge-colour:${colour(j)}">${x}</td>`).join('')+'<th>'+row.reduce((a,b)=>a+b,0)+'</th></tr>').join('')+'<tr><th scope="row">|e|</th>'+full.edges.map((e,i)=>`<th data-column="${i}" style="--edge-colour:${colour(i)}">${e.length}</th>`).join('')+'<th>'+full.edges.reduce((s,e)=>s+e.length,0)+'</th></tr></tbody></table>';
  const columnNodes=[...$('intro-matrix').querySelectorAll('[data-column]')];
  const duration=S.duration;
  $('timeline').max=duration;
  let time=0,playing=false,lastTime=null,raf=0,annotationKey='',lastFrame=null,visible=true;

  function expression(frame) {
    const {stageIndex,h,active,sizes,pairs,total}=frame;
    if(stageIndex===0)return '<span class="math">V='+subset(h.vertices)+', ℰ=∅.</span>';
    if(stageIndex<=3)return token(h.edges[active],active,true)+` <span class="math">|e<sub>${active+1}</sub>|=${sizes[active]}.</span>`;
    if(stageIndex===4)return '<span class="math">H=(V,ℰ), ℰ={e₁,e₂,e₃,e₄}; |V|='+h.vertices.length+', |ℰ|='+h.edges.length+'.</span>';
    if(stageIndex===5)return token(h.edges[active],active,true)+` <span class="math">B<sub>v,${active+1}</sub>=1 ⇔ v∈e<sub>${active+1}</sub>.</span>`;
    if(stageIndex===6){const es=M.combinations(h.edges[active],2);return token(h.edges[active],active,true)+' <span class="math">→ '+es.map(subset).join(', ')+'.</span>';}
    return '<span class="math">ℰ₁={{A,B,C}} ≠ ℰ₂={{A,B},{B,C},{A,C}}, but G₂(H₁)=G₂(H₂)=K₃.</span>';
  }

  function render() {
    const frame=S.at(time,reduced.matches||!playing),{stage,stageIndex,h,active,reveal}=frame;
    const present=new Set(frame.pairs.map(pairKey));
    const fresh=stageIndex>0&&stageIndex<5?new Set(M.combinations(h.edges[h.edges.length-1],2).map(pairKey)):new Set();
    const emphasized=stage.projection?new Set(M.combinations(h.edges[active],2).map(pairKey)):new Set();
    pairNodes.forEach(node=>{
      const key=node.dataset.pair,show=present.has(key),progress=fresh.has(key)?reveal:1;
      node.style.opacity=show?(stage.projection?(emphasized.has(key)?1:.25):1):0;
      node.style.strokeDasharray='1';node.style.strokeDashoffset=show?1-progress:1;
      node.style.strokeWidth=emphasized.has(key)?4:2;
    });
    edgeNodes.forEach(node=>{
      const index=+node.dataset.edge,show=index<h.edges.length,isNew=stageIndex>0&&stageIndex<5&&index===h.edges.length-1;
      node.style.opacity=show?(isNew?reveal:1):0;
      if(node.classList.contains('incidence-link')){
        node.style.strokeDasharray='1';node.style.strokeDashoffset=isNew?1-reveal:0;
        node.style.strokeWidth=index===active?4:2;
        if(show&&!isNew)node.style.opacity=active===null||index===active?1:.38;
      }
      if(node.classList.contains('edge-node'))node.querySelector('rect').setAttribute('stroke-width',index===active?3:2);
    });
    vertexNodes.forEach(node=>{
      node.style.opacity=stageIndex===0?.35+.65*reveal:1;
      node.querySelector('circle').setAttribute('fill',active!==null&&h.edges[active]?.includes(node.dataset.vertex)?'#f3e6c9':'#fffdf9');
    });
    const key=stageIndex+':'+active;
    if(key!==annotationKey){
      $('intro-stage').innerHTML=`<strong>Stage ${stageIndex+1} / ${S.stages.length} · ${stage.title}</strong><div class="story-equation">${expression(frame)}</div><p>${stage.narration}</p>`;
      $('intro-hyper').querySelector('svg').setAttribute('aria-label',`Current hypergraph: ${h.vertices.length} vertices, ${h.edges.length} edges, ${frame.total} memberships. ${h.edges.map((e,i)=>'e'+(i+1)+'='+subset(e)).join('; ')}`);
      edgeNodes.forEach(node=>node.setAttribute('aria-hidden',String(+node.dataset.edge>=h.edges.length)));
      $('intro-sets').innerHTML=h.edges.length?'<span class="set-family-label">Current ℰ:</span>'+h.edges.map((e,i)=>token(e,i,i===active)).join(''):'<span class="hint">No edge has been added yet. Use Next step to introduce the first group.</span>';
      $('intro-counts').textContent=stageIndex>=4?`Sizes (${frame.sizes.join(',')}), degrees (${frame.degrees.join(',')}). ${frame.total} spokes = ${frame.total} matrix 1s = Σ deg(v) = Σ |e|. Projection: ${frame.pairs.length} distinct pairs.`:`|V|=${h.vertices.length}; |ℰ|=${h.edges.length}; memberships=${frame.total}; projected pairs=${frame.pairs.length}.`;
      $('intro-left-title').textContent=stageIndex>=6?'Ordinary graph: the 2-section projection':'Ordinary graph · G₂(H)=(V,E₂(H))';
      $('intro-matrix-panel').hidden=!stage.matrix;
      $('intro-details').classList.toggle('with-matrix',!!stage.matrix);
      $('intro-counterexample').hidden=!stage.counterexample;
      columnNodes.forEach(node=>node.classList.toggle('active',!!stage.matrix&&+node.dataset.column===active));
      $('intro-matrix-caption').textContent=stageIndex===5?`Column e${active+1}: 1 occurs at ${h.edges[active].join(', ')}; its sum is ${h.edges[active].length}.`:`Rows sum to degrees; columns sum to edge sizes. Both totals are ${frame.total}.`;
      annotationKey=key;
    }
    $('timeline').value=time;
    $('timeline').setAttribute('aria-valuetext',`${time.toFixed(1)} of ${duration} seconds; stage ${stageIndex+1}: ${stage.title}`);
    $('intro-prev').disabled=stageIndex===0;
    $('intro-next').disabled=stageIndex===S.stages.length-1;
    $('story').dataset.time=time.toFixed(3);$('story').dataset.stage=stageIndex+1;
    $('story').dataset.edges=h.edges.length;$('story').dataset.incidences=frame.total;
    lastFrame=frame;
  }
  function pause(){playing=false;cancelAnimationFrame(raf);raf=0;lastTime=null;$('intro-play').textContent=`Play ${duration}-second story`;$('intro-play').setAttribute('aria-pressed','false');}
  function tick(now){
    if(!playing)return;
    if(lastTime!==null)time=Math.min(duration,time+(now-lastTime)/1000*Number($('intro-speed').value));
    lastTime=now;render();
    if(time>=duration)pause();else raf=requestAnimationFrame(tick);
  }
  function seek(seconds){pause();time=seconds;render();}
  $('intro-play').onclick=()=>{
    if(playing){pause();return;}
    if(!visible||document.hidden)return;
    if(time>=duration)time=0;
    playing=true;lastTime=null;$('intro-play').textContent='Pause story';$('intro-play').setAttribute('aria-pressed','true');
    raf=requestAnimationFrame(tick);
  };
  $('intro-restart').onclick=()=>seek(0);
  $('timeline').oninput=e=>seek(Number(e.target.value));
  $('intro-prev').onclick=()=>seek(S.stages[Math.max(0,lastFrame.stageIndex-1)].start);
  $('intro-next').onclick=()=>seek(S.stages[Math.min(S.stages.length-1,lastFrame.stageIndex+1)].start);
  $('intro-speed').onchange=()=>{lastTime=null;};
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(!visible)pause();},{threshold:0}).observe($('story'));
  reduced.addEventListener('change',()=>{pause();render();});
  pause();render();
}};
