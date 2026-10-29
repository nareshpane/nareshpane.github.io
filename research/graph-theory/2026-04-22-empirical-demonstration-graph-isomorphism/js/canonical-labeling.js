(function(L){
 'use strict';
 // Exact canonical form from every leaf of an equivariant IR tree.
 // It is NOT the lexicographic maximum over all S_n (a different convention).
 L.canonical=function(g,{limit=250000,initial=[L.range(g.n)],prune=false,automorphisms=[],record=false}={}){
   let nodes=0,leaves=0,best=null,order=null,complete=true,candidates=[];
   function dfs(cells,seq){if(++nodes>limit){complete=false;return;}let target=L.target(cells);
     if(!target.length){leaves++;let o=cells.flat(),c=L.code(g,o);if(record&&candidates.length<1000)candidates.push({order:o,code:c,seq});if(best===null||c>best){best=c;order=o;}return;}
     let done=[];for(const v of target){if(prune&&done.some(u=>automorphisms.some(p=>seq.every(w=>p[w]===w)&&p[u]===v)))continue;let r=L.refine(g,L.individualize(cells,v),[[v]]);dfs(r.cells,[...seq,v]);done.push(v);if(!complete)return;}
   }
   dfs(L.refine(g,initial).cells,[]);let p=order?L.inverse(order):null;
   return {code:best,order,permutation:p,nodes,leaves,complete,candidates,graph:order?L.relabel(g,p):null};
 };
 L.exhaustiveCanonical=function(g){if(g.n>8)throw Error('Exhaustive canonical baseline is limited to 8 vertices.');let code=null,order=null,leaves=0;for(const p of L.permutations(L.range(g.n))){leaves++;let c=L.code(g,p);if(code===null||c>code){code=c;order=p;}}return {code,order,leaves,graph:L.relabel(g,L.inverse(order)),complete:true};};
})(GraphLab);
