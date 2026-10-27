/* Small educational examples. Pure functions also run in the Node audit. */
(function(scope){
 'use strict';
 const T=typeof module!=='undefined'?require('./math.js'):scope.TreeMath;
 function tracePath(n,edges,roots,start){
   if(!Number.isInteger(start)||start<0||start>=n)throw new RangeError('Invalid starting vertex');
   const out=new Map();
   for(const [u,v] of edges){if(out.has(u))throw new Error('Tracing requires at most one outgoing arc');out.set(u,v);}
   const rows=[{vertex:start,arc:null,status:roots.includes(start)?'root':'start'}],seen=new Set([start]);
   let current=start;
   while(!roots.includes(current)){
     if(!out.has(current)){rows[rows.length-1].status='dead-end';break;}
     const next=out.get(current),status=roots.includes(next)?'root':seen.has(next)?'cycle':'continue';
     rows.push({vertex:next,arc:[current,next],status});current=next;
     if(status==='cycle'||status==='root')break;
     seen.add(next);
   }
   return rows;
 }
 function laplacian(n,edges){
   const l=Array.from({length:n},()=>Array(n).fill(0));
   for(const [u,v] of edges){l[u][u]++;l[u][v]--;}
   return l;
 }
 const api={tracePath,laplacian};
 if(typeof module!=='undefined'){module.exports=api;return;}
 scope.LearningMath=api;
 const el=id=>document.getElementById(id);
 const triangle=Array.from({length:3},(_,u)=>Array.from({length:3},(_,v)=>[u,v])).flat().filter(([u,v])=>u!==v);
 const draw=(id,n,e,r,o={})=>{el(id).innerHTML=svgGraph(n,e,r,{width:360,height:230,radius:14,...o});};
 const tinyOptions={width:300,height:175,radius:11,names:['1','2','3','4','r']};
 const chain=[[0,1],[1,2],[2,3],[3,4]],two=[[0,1],[1,2],[3,4]],three=[[0,1],[2,3]],bad=[[0,1],[1,2],[2,0],[3,4]];
 for(const [id,e,r] of [['mini-one',chain,[4]],['mini-two',two,[2,4]],['mini-three',three,[1,3,4]],['mini-invalid',bad,[4]]]){
   draw(id,5,e,r,{...tinyOptions,components:id!=='mini-invalid',title:id==='mini-invalid'?'Invalid candidate with a directed cycle':'Directed forest with '+r.length+' components'});
   if(id==='mini-invalid')el(id).querySelectorAll('[data-arc]').forEach(p=>{if(['0,1','1,2','2,0'].includes(p.dataset.arc))p.classList.add('cycle-edge');});
 }
 draw('foundation-one',3,[[0,1],[1,2]],[2],{components:true,title:'One root with two selected arcs'});
 draw('foundation-two',3,[[0,1]],[1,2],{components:true,title:'Two roots with one selected arc and one isolated root'});
 // Follow arrows in the existing three introduction cards.
 const examples={tree:{id:'intro-tree',edges:chain,roots:[4]},cycle:{id:'intro-cycle',edges:bad,roots:[4]},forest:{id:'intro-forest',edges:two,roots:[2,4]}};
 let pathRows=[],traceIndex=0,traceTimer=null;
 function stopTrace(){clearInterval(traceTimer);traceTimer=null;el('trace-play').textContent='Play';}
 function renderTrace(){
   const ex=examples[el('trace-example').value],row=pathRows[traceIndex],names=['1','2','3','4','r'];
   for(const graph of Object.values(examples)){
     el(graph.id).querySelectorAll('.trace-node,.inspected,.cycle-edge').forEach(e=>e.classList.remove('trace-node','inspected','cycle-edge'));
   }
   const diagram=el(ex.id);
   diagram.querySelector('[data-node="'+row.vertex+'"]').classList.add('trace-node');
   const walked=pathRows.slice(1,traceIndex+1).map(r=>r.arc.join(','));
   diagram.querySelectorAll('[data-arc]').forEach(p=>{if(walked.includes(p.dataset.arc))p.classList.add(row.status==='cycle'?'cycle-edge':'inspected');});
   const route=pathRows.slice(0,traceIndex+1).map(r=>names[r.vertex]).join(' → ');
   const explanation=row.status==='root'?'Reached reference vertex '+names[row.vertex]+'. No outgoing arc is selected here.':row.status==='cycle'?'Vertex '+names[row.vertex]+' has already been visited. This walk is trapped in a directed cycle.':row.status==='dead-end'?'A non-reference vertex has no outgoing arc: the selected graph fails the definition.':'At vertex '+names[row.vertex]+'. Follow its selected outgoing arc next.';
   el('trace-status').textContent='Step '+(traceIndex+1)+' of '+pathRows.length+' · '+route+'. '+explanation;
   el('trace-prev').disabled=traceIndex===0;el('trace-next').disabled=traceIndex===pathRows.length-1;
   if(traceIndex===pathRows.length-1)stopTrace();
 }
 function resetTrace(){stopTrace();const ex=examples[el('trace-example').value];pathRows=tracePath(5,ex.edges,ex.roots,Number(el('trace-start').value));traceIndex=0;renderTrace();}
 function startTrace(){
   if(traceTimer){stopTrace();return;}
   if(traceIndex===pathRows.length-1){traceIndex=0;renderTrace();}
   if(pathRows.length<2)return;
   el('trace-play').textContent='Pause';
   traceTimer=setInterval(()=>{traceIndex=Math.min(pathRows.length-1,traceIndex+1);renderTrace();},Number(el('trace-speed').value));
 }
 el('trace-example').onchange=el('trace-start').onchange=resetTrace;
 el('trace-prev').onclick=()=>{stopTrace();traceIndex=Math.max(0,traceIndex-1);renderTrace();};
 el('trace-next').onclick=()=>{stopTrace();traceIndex=Math.min(pathRows.length-1,traceIndex+1);renderTrace();};
 el('trace-play').onclick=startTrace;el('trace-reset').onclick=resetTrace;
 el('trace-speed').onchange=()=>{if(traceTimer){stopTrace();startTrace();}};
 resetTrace();
 // Enumerate the four assignments before abstract ring-sum/star notation.
 const assignments=T.candidates(3,triangle,[2]);
 function assignment(i){
   const row=assignments[i];draw('choice-graph',3,triangle,[2],{selected:row.arcs,title:'Triangle outgoing assignment '+(i+1)});
   el('choice-expression').textContent=row.arcs.map(([u,v])=>'('+String(u+1)+' → '+String(v+1)+')').join(' × ');
   el('choice-explanation').textContent=row.valid?'Valid: two distinct underlying edge IDs, and both vertices reach root 3. This assignment contributes one rooted tree.':'Rejected by star: 1 → 2 and 2 → 1 both use edge ID {1,2}. Their product repeats that ID and forms a 2-cycle.';
   [...el('choice-buttons').children].forEach((b,j)=>{b.classList.toggle('active',i===j);b.setAttribute('aria-pressed',String(i===j));});
 }
 assignments.forEach((_,i)=>{const b=document.createElement('button');b.type='button';b.textContent='Assignment '+(i+1);b.onclick=()=>assignment(i);el('choice-buttons').append(b);});assignment(0);
 // Determinant arithmetic is intentionally evaluated independently.
 const smallMinor=T.minor(3,triangle,[2]),smallForests=assignments.filter(c=>c.valid);
 let determinantStep=0,smallIndex=0;
 const [[a,b],[c,d]]=smallMinor.matrix;
 const determinantSteps=[
   ['det [[a, b], [c, d]] = ad − bc','For a 2 × 2 matrix, multiply the two diagonal entries, then subtract the product of the off-diagonal entries.'],
   ['= ('+a+' × '+d+') − ('+b+' × '+c+')','The entries come from the retained outgoing Laplacian. Both off-diagonal entries are negative.'],
   ['= '+(a*d)+' − '+(b*c),'The off-diagonal product is positive: (−1) × (−1) = 1.'],
   ['= '+String(smallMinor.count),'The exact determinant is '+String(smallMinor.count)+'. Direct path-based enumeration independently finds '+smallForests.length+' rooted trees.']
 ];
 draw('det-small-graph',3,triangle,[2],{title:'Complete symmetric triangle with root 3'});
 function determinantView(){
   el('det-step').textContent=determinantSteps[determinantStep][0];el('det-explanation').textContent=determinantSteps[determinantStep][1];
   el('det-counter').textContent='Step '+(determinantStep+1)+' of 4';el('det-prev').disabled=determinantStep===0;el('det-next').disabled=determinantStep===3;
 }
 function smallForestView(){
   draw('det-result-graph',3,triangle,[2],{selected:smallForests[smallIndex].arcs,title:'Small rooted tree '+(smallIndex+1)});
   el('det-enumeration').textContent='Tree '+(smallIndex+1)+' of '+smallForests.length+': '+smallForests[smallIndex].arcs.map(([u,v])=>(u+1)+' → '+(v+1)).join(', ')+'. Both non-roots reach vertex 3.';
 }
 el('det-prev').onclick=()=>{determinantStep=Math.max(0,determinantStep-1);determinantView();};
 el('det-next').onclick=()=>{determinantStep=Math.min(3,determinantStep+1);determinantView();};
 el('det-reset').onclick=()=>{determinantStep=0;smallIndex=0;determinantView();smallForestView();};
 el('det-forest-prev').onclick=()=>{smallIndex=(smallIndex+2)%3;smallForestView();};
 el('det-forest-next').onclick=()=>{smallIndex=(smallIndex+1)%3;smallForestView();};
 determinantView();smallForestView();
 // Highlight exactly the entries changed by an inspected directed arc.
 let matrixEdges=triangle.map(e=>[...e]);
 el('matrix-edge').innerHTML=triangle.map(([u,v])=>'<option value="'+u+','+v+'">'+(u+1)+' → '+(v+1)+'</option>').join('');
 function matrixTable(l,indices,selected,full){
   const [u,v]=selected;
   let h='<table class="entry-grid" aria-label="'+(full?'Full outgoing Laplacian':'Reduced outgoing Laplacian')+'"><thead><tr><th scope="col">L</th>'+indices.map(j=>'<th scope="col">'+(j+1)+'</th>').join('')+'</tr></thead><tbody>';
   for(const i of indices){h+='<tr><th scope="row">'+(i+1)+'</th>';for(const j of indices){const changed=i===u&&(j===u||j===v);h+='<td class="'+(changed?'affected ':'')+(full&&(i===2||j===2)?'deleted':'')+'">'+l[i][j]+'</td>';}h+='</tr>';}
   return h+'</tbody></table>';
 }
 function matrixView(){
   const edge=el('matrix-edge').value.split(',').map(Number),[u,v]=edge,present=matrixEdges.some(e=>T.key([e])===T.key([edge]));
   draw('matrix-graph',3,matrixEdges,[2],{title:'Triangle used to construct the outgoing Laplacian'});
   el('matrix-graph').querySelector('[data-arc="'+edge.join(',')+'"]')?.classList.add('inspected');
   const l=laplacian(3,matrixEdges);el('matrix-full').innerHTML=matrixTable(l,[0,1,2],edge,true);el('matrix-reduced').innerHTML=matrixTable(l,[0,1],edge,false);
   el('matrix-contribution').textContent=(present?'Present':'Absent')+' arc '+(u+1)+' → '+(v+1)+': '+(present?'adds +1 to L'+(u+1)+(u+1)+' and −1 to L'+(u+1)+(v+1)+'.':'its +1 / −1 contributions are not included.')+(u===2?' Its entire row is deleted because its initial vertex is the root.':v===2?' The off-diagonal entry is deleted, but its +1 diagonal contribution remains.':' Both affected entries survive root deletion.');
   const count=T.minor(3,matrixEdges,[2]).count,direct=T.candidates(3,matrixEdges,[2]).filter(c=>c.valid).length;
   el('matrix-result').textContent='det Lᵤᵤ = '+count+'; direct enumeration = '+direct+'. '+(count===BigInt(direct)?'Agreement.':'DISCREPANCY.');
   el('matrix-result').classList.toggle('fail',count!==BigInt(direct));
   el('matrix-toggle').textContent=present?'Remove selected arc':'Add selected arc';
   el('matrix-graph').querySelectorAll('[data-arc]').forEach(p=>{p.style.cursor='pointer';p.addEventListener('click',()=>{el('matrix-edge').value=p.dataset.arc;matrixView();});});
 }
 el('matrix-edge').onchange=matrixView;
 el('matrix-toggle').onclick=()=>{const edge=el('matrix-edge').value.split(',').map(Number),index=matrixEdges.findIndex(e=>e[0]===edge[0]&&e[1]===edge[1]);if(index<0)matrixEdges.push(edge);else matrixEdges.splice(index,1);matrixView();};
 el('matrix-reset').onclick=()=>{matrixEdges=triangle.map(e=>[...e]);el('matrix-edge').value='0,1';matrixView();};matrixView();
 // Before the five-vertex source example, vary k in the smallest nontrivial graph.
 let tinyIndex=0;
 function tinyView(){
   const k=Number(el('tiny-k').value),roots=k===1?[2]:k===2?[1,2]:[0,1,2];
   const rows=T.candidates(3,triangle,roots).filter(c=>c.valid),minor=T.minor(3,triangle,roots);
   tinyIndex=((tinyIndex%rows.length)+rows.length)%rows.length;
   draw('tiny-forest',3,rows[tinyIndex].arcs,roots,{components:true,title:'Triangle forest with '+k+' roots'});
   el('tiny-formula').textContent='n − k = 3 − '+k+' = '+(3-k)+' selected arcs; R = {'+roots.map(i=>i+1).join(', ')+'}';
   el('tiny-count').textContent='Direct enumeration: '+rows.length+'; exact principal minor: '+minor.count+'. Forest '+(tinyIndex+1)+' of '+rows.length+'.';
   el('tiny-description').textContent=k===3?'All vertices are references. The unique selection has no arcs, three isolated components, and the empty minor has determinant one.':rows[tinyIndex].arcs.map(([u,v])=>(u+1)+' → '+(v+1)).join(', ')+'. Every selected path terminates at '+(k===1?'reference vertex 3.':'one of the '+k+' reference vertices.');
   el('tiny-prev').disabled=el('tiny-next').disabled=rows.length===1;
 }
 el('tiny-k').onchange=()=>{tinyIndex=0;tinyView();};el('tiny-prev').onclick=()=>{tinyIndex--;tinyView();};el('tiny-next').onclick=()=>{tinyIndex++;tinyView();};tinyView();
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 function motion(){if(reduced.matches){stopTrace();pause();}for(const id of ['trace-play','alg-play']){el(id).disabled=reduced.matches;el(id).title=reduced.matches?'Reduced motion: use the previous and next controls.':'';}}
 reduced.addEventListener('change',motion);motion();
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stopTrace();});
})(globalThis);
