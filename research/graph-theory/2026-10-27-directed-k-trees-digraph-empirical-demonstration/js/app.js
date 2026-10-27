/* UI and SVG rendering. All counts are derived from the current graph. */
'use strict';
const M=TreeMath, $=id=>document.getElementById(id);
let serial=0;
function svgGraph(n,edges,roots,options={}){
 const id='arrow'+serial++, names=options.names||Array.from({length:n},(_,i)=>String(i+1));
 const pos=options.paper?[[40,282],[145,140],[295,40],[460,140],[300,290]]:
   Array.from({length:n},(_,i)=>[250+160*Math.cos(-Math.PI/2+i*2*Math.PI/n),175+125*Math.sin(-Math.PI/2+i*2*Math.PI/n)]);
 const selected=options.selected&&new Set(options.selected.map(e=>e.join(',')));
 let s='<svg viewBox="0 0 510 345" role="img" aria-labelledby="'+id+'-title '+id+'-desc"><title id="'+id+'-title">'+(options.title||'Directed graph')+'</title><desc id="'+id+'-desc">Vertices '+names.join(', ')+'. Square roots: '+roots.map(i=>names[i]).join(', ')+'. Arcs: '+edges.map(([u,v])=>names[u]+' to '+names[v]).join('; ')+'.</desc><defs><marker id="'+id+'" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#52625b"/></marker></defs>';
 const original=new Set(M.paper.map(e=>e.join(',')));
 for(const [u,v] of edges){
   const [x1,y1]=pos[u],[x2,y2]=pos[v],dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy);
   const reciprocal=edges.some(e=>e[0]===v&&e[1]===u),bend=reciprocal?27:0;
   const nx=-dy/len,ny=dx/len,mx=(x1+x2)/2+nx*bend,my=(y1+y2)/2+ny*bend;
   const ux=mx-x1,uy=my-y1,ul=Math.hypot(ux,uy),vx=x2-mx,vy=y2-my,vl=Math.hypot(vx,vy);
   const sx=x1+ux/ul*18,sy=y1+uy/ul*18,tx=x2-vx/vl*22,ty=y2-vy/vl*22;
   let cls='edge';
   if(options.completion&&!original.has([u,v].join(',')))cls+=' added';
   if(selected)cls+=selected.has([u,v].join(','))?' selected':' muted';
   s+='<path data-arc="'+u+','+v+'" class="'+cls+'" d="M'+sx+','+sy+' Q'+mx+','+my+' '+tx+','+ty+'" marker-end="url(#'+id+')"><title>'+names[u]+' → '+names[v]+'</title></path>';
   if(options.labels){const lx=(x1+2*mx+x2)/4+nx*11,ly=(y1+2*my+y2)/4+ny*11;s+='<text class="edge-label" x="'+lx+'" y="'+ly+'" text-anchor="middle">'+(options.paper?M.labels[[u,v].join(',')]:names[u]+'→'+names[v])+'</text>';}
 }
 pos.forEach(([x,y],i)=>{
   s+=roots.includes(i)?'<rect class="root" x="'+(x-17)+'" y="'+(y-17)+'" width="34" height="34" rx="3"/>':'<circle class="node" cx="'+x+'" cy="'+y+'" r="17"/>';
   s+='<text x="'+x+'" y="'+(y+5)+'" text-anchor="middle" font-size="15">'+names[i]+'</text>';
 });
 return s+'</svg>';
}
const names=['1','2','3','4','r'];
const complete=(n)=>Array.from({length:n},(_,u)=>Array.from({length:n},(_,v)=>[u,v])).flat().filter(e=>e[0]!==e[1]);
const completion=Array.from(new Map(M.paper.flatMap(([u,v])=>[[[u,v].join(','),[u,v]],[[v,u].join(','),[v,u]]])).values());
const put=(id,n,e,r,o={})=>$(id).innerHTML=svgGraph(n,e,r,o);
put('intro-tree',5,[[0,1],[1,2],[2,3],[3,4]],[4],{names,title:'Valid directed tree toward r'});
put('intro-cycle',5,[[0,1],[1,2],[2,0],[3,4]],[4],{names,title:'Invalid selection with a directed triangle'});
put('intro-forest',5,[[0,1],[1,2],[3,4]],[2,4],{names,title:'Two components directed toward 3 and r'});
put('paper-original',5,M.paper,[4],{paper:true,names,labels:true,title:'Reconstruction of original Figure 1a'});
put('paper-completed',5,completion,[4],{paper:true,names,labels:true,completion:true,title:'Reconstruction of Figure 1b with added reverse arcs'});
let flipped=false;
function cycle(){put('algebra-triangle',4,flipped?[[1,0],[2,1],[0,2]]:[[0,1],[1,2],[2,0]],[3],{names:['1','2','3','r'],labels:true,title:'A cycle and its reversed partner'});$('cycle-caption').textContent='Orientation '+(flipped?2:1)+' of the same underlying triangle; the two products cancel.';}
cycle();$('flip-cycle').onclick=()=>{flipped=!flipped;cycle();};
$('derive-next').onclick=()=>{
 const step=document.querySelector('#tree-derivation .next-step:not(.revealed)');
 if(step)step.classList.add('revealed');
 if(!document.querySelector('#tree-derivation .next-step:not(.revealed)'))$('derive-next').disabled=true;
};
const forests=M.candidates(5,M.paper,[2,4]).filter(c=>c.valid);
function gallery(i){
 const arcs=forests[i].arcs;put('forest-gallery',5,M.paper,[2,4],{paper:true,names,labels:true,selected:arcs,title:'Source two-tree '+(i+1)});
 $('forest-product').textContent='Forest '+(i+1)+': '+arcs.map(e=>M.labels[e.join(',')]).join(' · ');
 const groups={};
 for(let u=0;u<5;u++){let v=u;while(![2,4].includes(v))v=arcs.find(e=>e[0]===v)[1];(groups[v]??=[]).push(names[u]);}
 $('forest-components').textContent='Components: '+Object.values(groups).map(g=>'{'+g.join(', ')+'}').join(' and ')+'. Three edges; exactly two roots.';
 [...$('forest-buttons').children].forEach((b,j)=>{b.classList.toggle('active',i===j);b.setAttribute('aria-pressed',String(i===j));});
}
forests.forEach((_,i)=>{const b=document.createElement('button');b.type='button';b.textContent=String(i+1);b.setAttribute('aria-label','Show source forest '+(i+1));b.onclick=()=>gallery(i);$('forest-buttons').append(b);});gallery(0);
const P=M.parity(5,M.paper,[4]),allCandidates=M.candidates(5,completion,[4]);
let step=0,timer=null;
const stages=[
 ['Begin with G','Ten original arcs on five vertices. r is the reference node; its outgoing choices are omitted.','|E| = 10; R = {r}',M.paper],
 ['Complete the missing reverse arcs','Add four double-primed arcs to obtain G′. They are dashed. Existing primed arcs stay original.','|E′| = 14',completion],
 ['Choose one outgoing arc per non-root','The Cartesian expansion has '+allCandidates.length+' assignments. Degrees are correct, but cyclic terms remain.','1 × 4 × 3 × 3 = '+allCandidates.length,completion],
 ['Reject repeated edge identifiers','Star products eliminate reciprocal 2-cycles. '+P.terms+' expansion occurrences remain before parity grouping.','Star survivors: '+P.terms,completion],
 ['Cancel even multiplicities','Group by underlying edge sets. Cycle orientations cancel; '+P.odd+' rooted trees survive in G′.','Odd groups = '+P.odd,completion],
 ['Discard trees using artificial arcs','Four surviving trees use e3″. The remaining twelve trees use only original arcs and reproduce the source list.','16 − 4 = '+P.forests.length,M.paper]
];
function algorithm(){
 const [title,text,expr,edges]=stages[step];$('algorithm-title').textContent=title;$('algorithm-text').textContent=text;$('algorithm-expression').textContent=expr;
 put('algorithm-graph',5,edges,[4],{paper:true,names,labels:true,completion:step>0&&step<5,selected:step===5?P.forests[0]:null,title:'Construction step '+(step+1)});
 $('alg-counter').textContent='Step '+(step+1)+' of '+stages.length;$('alg-progress').value=step;$('alg-prev').disabled=step===0;$('alg-next').disabled=step===5;
}
function pause(){clearInterval(timer);timer=null;$('alg-play').textContent='Play';}
function play(){if(timer){pause();return;}if(step===5)step=0;algorithm();$('alg-play').textContent='Pause';timer=setInterval(()=>{if(step===5){pause();return;}step++;algorithm();},Number($('alg-speed').value));}
$('alg-play').onclick=play;$('alg-prev').onclick=()=>{pause();step=Math.max(0,step-1);algorithm();};$('alg-next').onclick=()=>{pause();step=Math.min(5,step+1);algorithm();};$('alg-reset').onclick=()=>{pause();step=0;algorithm();};
$('alg-progress').oninput=()=>{pause();step=Number($('alg-progress').value);algorithm();};$('alg-speed').onchange=()=>{if(timer){pause();play();}};algorithm();
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
// Laboratory: small bounded exact computations, no network/data loading.
let state={n:3,edges:complete(3),primary:2,k:1,extra:null,mode:'counts',index:0},last=null;
function roots(){return (state.extra?[state.primary,...state.extra]:Array.from({length:state.k},(_,i)=>(state.primary+i)%state.n)).sort((a,b)=>a-b);}
function stateNames(){return $('preset').value==='paper'?names:Array.from({length:state.n},(_,i)=>String(i+1));}
function controls(){
 $('primary-root').innerHTML=stateNames().map((s,i)=>'<option value="'+i+'">'+s+'</option>').join('');$('primary-root').value=state.primary;
 $('root-count').innerHTML=Array.from({length:state.n},(_,i)=>'<option>'+(i+1)+'</option>').join('');$('root-count').value=state.k;
 $('size').value=state.n;$('size').disabled=['paper','triangle'].includes($('preset').value);
}
function resetPreset(){
 const p=$('preset').value;
 state.n=p==='paper'?5:p==='triangle'?3:Number($('size').value);
 state.edges=p==='paper'?M.paper.map(e=>[...e]):p==='empty'?[]:p==='cycle'?Array.from({length:state.n},(_,i)=>[i,(i+1)%state.n]):complete(state.n);
 state.primary=state.n-1;state.k=1;state.extra=null;state.index=0;controls();recompute();
}
function recompute(){
 const R=roots(),C=M.candidates(state.n,state.edges,R),good=C.filter(c=>c.valid),par=M.parity(state.n,state.edges,R),minor=M.minor(state.n,state.edges,R);
 const a=good.map(c=>M.key(c.arcs)).sort(),b=par.forests.map(M.key).sort();
 last={R,C,good,par,minor,ok:JSON.stringify(a)===JSON.stringify(b)&&BigInt(good.length)===minor.count};
 $('parity-count').textContent=par.forests.length;$('direct-count').textContent=good.length;$('minor-count').textContent=String(minor.count);
 $('agreement').classList.toggle('fail',!last.ok);
 $('agreement').textContent=(last.ok?'Agreement':'DISCREPANCY')+': '+(last.ok?'the entire forest families match, and the determinant equals their size.':'the computed methods disagree; inspect the trace.')+' '+C.length+' outgoing-choice candidates inspected.';
 $('root-help').textContent='Prescribed roots R = {'+R.map(i=>stateNames()[i]).join(', ')+'}. Choose additional roots below, or use k to choose consecutive roots from the primary root. Every root omits one outgoing factor.';
 $('root-choices').innerHTML=stateNames().map((s,i)=>'<label style="display:flex;align-items:center;gap:5px"><input type="checkbox" data-root="'+i+'" aria-label="Root vertex '+s+'" '+(R.includes(i)?'checked ':'')+(i===state.primary?'disabled':'')+'>Root '+s+'</label>').join('');
 $('root-choices').querySelectorAll('input').forEach(c=>c.onchange=()=>{const i=Number(c.dataset.root),next=roots().filter(v=>v!==i);if(c.checked)next.push(i);state.extra=next.filter(v=>v!==state.primary);state.k=next.length;state.index=0;controls();recompute();$('root-choices').querySelector('[data-root="'+i+'"]').focus();});
 $('lab-matrix').textContent=minor.matrix.length?'['+minor.matrix.map(row=>row.join('  ')).join(']\n[')+']\ndet = '+minor.count:'Empty 0 × 0 matrix; determinant = 1.';
 $('lab-matrix').style.whiteSpace='pre-wrap';$('lab-matrix').style.fontFamily='monospace';
 $('lab-trace').textContent=JSON.stringify({roots:R.map(i=>stateNames()[i]),factors:par.stages,candidates:C.length,starOccurrences:par.terms,parityGroups:par.groups.map(g=>({edgeIDs:g.term.ids,multiplicity:g.count,odd:Boolean(g.count%2)})),completionForests:par.odd,filteredForests:par.forests.length,directForests:good.length,determinant:String(minor.count),familyEquality:JSON.stringify(a)===JSON.stringify(b)},null,2);
 adjacency();renderLab();
}
function adjacency(){
 const labels=stateNames();let h='<table class="matrix" aria-label="Editable directed adjacency table"><thead><tr><th scope="col">From / to</th>'+labels.map(s=>'<th scope="col">'+s+'</th>').join('')+'</tr></thead><tbody>';
 for(let u=0;u<state.n;u++){h+='<tr><th scope="row">'+labels[u]+'</th>';for(let v=0;v<state.n;v++){const on=state.edges.some(e=>e[0]===u&&e[1]===v);h+='<td>'+(u===v?'—':'<button type="button" class="'+(on?'present':'')+'" data-u="'+u+'" data-v="'+v+'" aria-label="Arc '+labels[u]+' to '+labels[v]+'" aria-pressed="'+on+'">'+(on?1:0)+'</button>')+'</td>';}h+='</tr>';}
 $('adjacency').innerHTML=h+'</tbody></table>';
 $('adjacency').querySelectorAll('button').forEach(b=>{
   b.onclick=()=>{const u=Number(b.dataset.u),v=Number(b.dataset.v),i=state.edges.findIndex(e=>e[0]===u&&e[1]===v);if(i<0)state.edges.push([u,v]);else state.edges.splice(i,1);state.index=0;recompute();$('adjacency').querySelector('[data-u="'+u+'"][data-v="'+v+'"]').focus();};
   const highlight=on=>{$('lab-graph').querySelectorAll('[data-arc]').forEach(p=>{if(p.dataset.arc===b.dataset.u+','+b.dataset.v){p.style.stroke=on?'#a54b34':'';p.style.strokeWidth=on?'4':'';}});};
   b.onmouseenter=b.onfocus=()=>highlight(true);b.onmouseleave=b.onblur=()=>highlight(false);
 });
}
function renderLab(){
 const {R,C,good,par}=last,mode=state.mode,rows=mode==='candidates'?C:good;
 state.index=rows.length?((state.index%rows.length)+rows.length)%rows.length:0;
 const row=rows[state.index];
 const edges=mode==='completion'?Array.from(new Map(state.edges.flatMap(([u,v])=>[[[u,v].join(','),[u,v]],[[v,u].join(','),[v,u]]])).values()):state.edges;
 const orig=new Set(state.edges.map(e=>e.join(',')));
 put('lab-graph',state.n,edges,R,{names:stateNames(),paper:$('preset').value==='paper',selected:mode==='completion'?null:row?.arcs,title:'Current laboratory '+mode});
 if(mode==='completion')$('lab-graph').querySelectorAll('[data-arc]').forEach(p=>{if(!orig.has(p.dataset.arc))p.classList.add('added');});
 $('candidate-controls').classList.toggle('hidden',mode==='completion');
 $('candidate-index').textContent=rows.length?(mode==='candidates'?'Candidate ':'Valid forest ')+(state.index+1)+' of '+rows.length:'No '+(mode==='candidates'?'candidates':'valid forests');
 $('candidate-prev').disabled=$('candidate-next').disabled=rows.length<2;
 $('graph-description').textContent='Root squares: '+R.map(i=>stateNames()[i]).join(', ')+'. '+(mode==='completion'?'Dashed arcs are added reversals.':'Highlighted arrows show the current selection; faint arrows remain available in G.');
 if(mode==='completion')$('candidate-explanation').textContent='Completion has '+edges.length+' arcs. Parity leaves '+par.odd+' rooted forests before filtering; '+par.forests.length+' use only original arcs.';
 else if(!row)$('candidate-explanation').textContent=state.k===state.n?'All vertices are roots. The empty edge set is the unique null forest.':'No selected structure exists: at least one required outgoing choice is unavailable, or all candidates are cyclic.';
 else{
   let msg=row.valid?'VALID: every non-root reaches exactly one prescribed root.':'INVALID: the outgoing-degree constraints hold, but a selected directed cycle prevents root reachability.';
   msg+=' Selected arcs: '+(row.arcs.map(([u,v])=>stateNames()[u]+'→'+stateNames()[v]).join(', ')||'none (the null forest)')+'.';
   $('candidate-explanation').textContent=msg;
 }
}
$('preset').onchange=resetPreset;$('size').onchange=resetPreset;
$('root-count').onchange=()=>{state.k=Number($('root-count').value);state.extra=null;state.index=0;recompute();};
$('primary-root').onchange=()=>{state.primary=Number($('primary-root').value);state.extra=null;state.index=0;recompute();};
$('lab-reset').onclick=()=>{state.mode='counts';document.querySelectorAll('[data-mode]').forEach(b=>{b.classList.toggle('active',b.dataset.mode==='counts');b.setAttribute('aria-pressed',String(b.dataset.mode==='counts'));});resetPreset();};
$('candidate-prev').onclick=()=>{state.index--;renderLab();};$('candidate-next').onclick=()=>{state.index++;renderLab();};
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{state.mode=b.dataset.mode;state.index=0;document.querySelectorAll('[data-mode]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});renderLab();});
controls();recompute();
