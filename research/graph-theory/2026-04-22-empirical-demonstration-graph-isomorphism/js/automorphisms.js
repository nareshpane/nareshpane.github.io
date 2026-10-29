(function(L){
 'use strict';
 L.automorphisms=(g,options={})=>L.isomorphisms(g,g,{...options,all:true});
 L.orbits=function(n,maps,fixed=[]){const ps=maps.filter(p=>fixed.every(v=>p[v]===v));let cells=[],seen=new Set();for(let v=0;v<n;v++)if(!seen.has(v)){let orbit=new Set([v]),queue=[v];for(let i=0;i<queue.length;i++)for(const p of ps){let w=p[queue[i]];if(!orbit.has(w)){orbit.add(w);queue.push(w);}}let c=[...orbit].sort((a,b)=>a-b);c.forEach(x=>seen.add(x));cells.push(c);}return cells;};
 L.closure=function(n,generators,limit=100000){let maps=[L.identity(n)],seen=new Set([maps[0].join(',')]);for(let i=0;i<maps.length;i++)for(const p of generators){let q=L.compose(maps[i],p),key=q.join(',');if(!seen.has(key)){if(maps.length>=limit)throw Error('Group closure limit.');seen.add(key);maps.push(q);}}return maps;};
 L.generators=function(n,maps){let gens=[],known=new Set([L.identity(n).join(',')]);for(const p of maps)if(!known.has(p.join(','))){gens.push(p);known=new Set(L.closure(n,gens).map(x=>x.join(',')));}return gens;};
})(GraphLab);
