(function(L){
 'use strict';
 L.validatePartition=function(cells,n){const xs=cells.flat();if(!cells.length||cells.some(c=>!c.length)||xs.length!==n||new Set(xs).size!==n||xs.some(v=>!Number.isInteger(v)||v<0||v>=n))throw Error('Cells must partition all vertices exactly once.');return cells;};
 L.colours=(cells,n)=>{const c=Array(n);cells.forEach((xs,i)=>xs.forEach(v=>c[v]=i));return c;};
 L.finer=function(fine,coarse,n){const a=L.colours(fine,n),b=L.colours(coarse,n);for(let v=0;v<n;v++)for(let w=0;w<n;w++)if(b[v]<b[w]&&!(a[v]<a[w]))return false;return true;};
 L.countInto=(g,v,cell)=>cell.reduce((s,w)=>s+g.a[v][w],0);
 L.signatures=(g,cells)=>L.range(g.n).map(v=>cells.map(c=>L.countInto(g,v,c)));
 L.equitable=(g,cells)=>cells.every(c=>cells.every(w=>c.every(v=>L.countInto(g,v,w)===L.countInto(g,c[0],w))));
 // Later Algorithm 1: FIFO active cells, fragments ascending by neighbour count,
 // omit first largest fragment on ties. These choices are label-equivariant.
 L.refine=function(g,input,active=input){
   L.validatePartition(input,g.n);
   let cells=input.map(c=>c.slice()),queue=active.map(c=>c.slice()),steps=[],trace=[];
   const same=(a,b)=>a.length===b.length&&a.every(x=>b.includes(x));
   while(queue.length&&cells.length<g.n){
     const W=queue.shift(),before=cells.map(c=>c.slice()),sig=L.signatures(g,before),events=[];
     for(const X of before){
       let buckets=new Map();for(const v of X){let d=L.countInto(g,v,W);if(!buckets.has(d))buckets.set(d,[]);buckets.get(d).push(v);}
       let counts=[...buckets.keys()].sort((a,b)=>a-b),frags=counts.map(d=>buckets.get(d));if(frags.length<=1)continue;
       const pos=cells.findIndex(c=>same(c,X)),qpos=queue.findIndex(c=>same(c,X));cells.splice(pos,1,...frags);
       let largest=0;for(let k=1;k<frags.length;k++)if(frags[k].length>frags[largest].length)largest=k;
       if(qpos>=0)queue.splice(qpos,1,...frags);else queue.push(...frags.filter((_,i)=>i!==largest));
       events.push({position:pos+1,sizes:frags.map(c=>c.length),counts,omitted:qpos>=0?null:largest+1});trace.push([pos+1,...frags.map(c=>c.length)]);
     }
     steps.push({W,before,cells:cells.map(c=>c.slice()),signatures:sig,active:queue.map(c=>c.slice()),events});
   }
   return {cells,steps,trace,equitable:L.equitable(g,cells)};
 };
 L.individualize=function(cells,v){let found=false;const p=cells.flatMap(c=>{if(!c.includes(v))return [c.slice()];found=true;return c.length===1?[c.slice()]:[[v],c.filter(w=>w!==v)];});if(!found)throw Error('Vertex is not in the partition.');return p;};
 L.target=(cells,strategy='smallest')=>{let cs=cells.filter(c=>c.length>1);if(!cs.length)return [];if(strategy==='first')return cs[0];let size=(strategy==='largest'?Math.max:Math.min)(...cs.map(c=>c.length));return cs.find(c=>c.length===size);};
 L.partitionText=p=>'('+p.map(c=>'{'+c.map(v=>v+1).join(',')+'}').join(' | ')+')';
 L.refineSequence=function(g,pi,seq){let r=L.refine(g,pi);for(const v of seq){if(r.cells.find(c=>c.includes(v)).length===1)throw Error('Choose a non-singleton cell.');r=L.refine(g,L.individualize(r.cells,v),[[v]]);}return r;};
})(GraphLab);
