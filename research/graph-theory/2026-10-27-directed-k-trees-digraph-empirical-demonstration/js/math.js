/* Exact bounded graph methods. CommonJS export supports independent Node audits. */
(function (scope) {
 'use strict';
 const paper=[[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,1],[1,4],[3,4],[2,4]];
 const labels={'0,1':"e7′",'1,0':'e7','1,2':'e1','2,1':"e1′",'2,3':'e2','3,2':"e2′",'3,1':'e3','1,3':'e3″','1,4':'e4','4,1':'e4″','3,4':'e5','4,3':'e5″','2,4':'e6','4,2':'e6″'};
 function valid(n,arcs,roots) {
   const out=Array(n).fill(-1);
   for(const [u,v] of arcs){if(out[u]!==-1||roots.includes(u))return false;out[u]=v;}
   for(let i=0;i<n;i++){
     if(roots.includes(i))continue;
     let j=i,seen=new Set();
     while(!roots.includes(j)){if(j<0||seen.has(j))return false;seen.add(j);j=out[j];}
   }
   return arcs.length===n-roots.length;
 }
 function candidates(n,edges,roots) {
   const vertices=Array.from({length:n},(_,i)=>i).filter(i=>!roots.includes(i));
   let rows=[[]];
   for(const i of vertices)rows=rows.flatMap(row=>edges.filter(e=>e[0]===i).map(e=>[...row,e]));
   return rows.map(arcs=>({arcs,valid:valid(n,arcs,roots)}));
 }
 function key(arcs){return arcs.map(e=>e.join(',')).sort().join(';');}
 function parity(n,edges,roots){
   const original=new Set(edges.map(e=>e.join(','))),pairs=[...new Set(edges.map(([u,v])=>[Math.min(u,v),Math.max(u,v)].join(',')))].map(s=>s.split(',').map(Number));
   const sym=pairs.flatMap(([u,v])=>[[u,v],[v,u]]);
   let terms=[{ids:[],arcs:[]}],stages=[];
   for(let i=0;i<n;i++)if(!roots.includes(i)){
     terms=terms.flatMap(t=>sym.filter(e=>e[0]===i).flatMap(e=>{
       const id=[...e].sort((a,b)=>a-b).join(',');
       return t.ids.includes(id)?[]:[{ids:[...t.ids,id],arcs:[...t.arcs,e]}];
     }));
     stages.push({vertex:i,terms:terms.length});
   }
   const groups=new Map();
   for(const t of terms){
     const id=[...t.ids].sort().join(';');
     const g=groups.get(id)||{count:0,term:t};g.count++;groups.set(id,g);
   }
   const odd=[...groups.values()].filter(g=>g.count%2);
   const forests=odd.map(g=>g.term.arcs).filter(a=>a.every(e=>original.has(e.join(','))));
   return {forests,stages,groups:[...groups.values()],terms:terms.length,odd:odd.length};
 }
 function det(a){
   if(!a.length)return 1n;
   // Leibniz formula, at most 5! permutations in this bounded laboratory.
   let sum=0n;
   function walk(p,left){
     if(!left.length){let inv=0,v=1n;for(let i=0;i<p.length;i++){v*=BigInt(a[i][p[i]]);for(let j=i+1;j<p.length;j++)if(p[i]>p[j])inv++;}sum+=(inv%2?-v:v);return;}
     left.forEach((x,i)=>walk([...p,x],left.filter((_,j)=>j!==i)));
   }
   walk([],a.map((_,i)=>i));return sum;
 }
 function minor(n,edges,roots){
   const l=Array.from({length:n},()=>Array(n).fill(0));
   edges.forEach(([u,v])=>{l[u][u]++;l[u][v]--;});
   const keep=l.map((_,i)=>i).filter(i=>!roots.includes(i));
   const matrix=keep.map(i=>keep.map(j=>l[i][j]));
   return {matrix,count:det(matrix)};
 }
 const api={paper,labels,valid,candidates,key,parity,det,minor};
 if(typeof module!=='undefined')module.exports=api;else scope.TreeMath=api;
})(globalThis);
