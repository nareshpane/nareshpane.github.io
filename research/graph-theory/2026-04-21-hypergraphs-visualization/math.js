/* Pure, bounded finite mathematics. Shared by the tutorial and Node assertions. */
(function (root) {
  'use strict';
  const choose = (n,k) => { if(k<0||k>n)return 0; k=Math.min(k,n-k);let b=1n;for(let i=1;i<=k;i++)b=b*BigInt(n-k+i)/BigInt(i);if(b>BigInt(Number.MAX_SAFE_INTEGER))throw Error('Count exceeds safe display range');return Number(b); };
  function combinations(a,k){if(!k)return [[]];if(k>a.length)return [];return a.flatMap((x,i)=>combinations(a.slice(i+1),k-1).map(t=>[x,...t]));}
  function create(vertices,edges){
    if(new Set(vertices).size!==vertices.length)throw Error('Repeated vertex');
    const seen=new Set();
    edges.forEach(e=>{const key=[...e].sort().join('|');if(!e.length||new Set(e).size!==e.length||e.some(v=>!vertices.includes(v))||seen.has(key))throw Error('Use distinct nonempty subsets');seen.add(key);});
    return {vertices:[...vertices],edges:edges.map(e=>[...e])};
  }
  const incidence=h=>h.vertices.map(v=>h.edges.map(e=>+e.includes(v)));
  const degrees=h=>incidence(h).map(row=>row.reduce((s,x)=>s+x,0));
  const transpose=b=>b.length?Array.from({length:b[0].length},(_,i)=>b.map(r=>r[i])):[];
  const dual=h=>({vertices:h.edges.map((_,i)=>'e'+(i+1)),edges:h.vertices.map(v=>h.edges.flatMap((e,i)=>e.includes(v)?['e'+(i+1)]:[]))});
  function matching(h){if(h.edges.length>16)throw Error('Exact matching capped at 16 edges');let best=[];
    function dfs(i,used,out){if(i===h.edges.length){if(out.length>best.length)best=[...out];return;}dfs(i+1,used,out);if(h.edges[i].every(v=>!used.has(v)))dfs(i+1,new Set([...used,...h.edges[i]]),[...out,i]);}dfs(0,new Set(),[]);return best;
  }
  function transversal(h){if(h.vertices.length>12)throw Error('Exact transversal capped at 12 vertices');for(let k=0;k<=h.vertices.length;k++)for(const s of combinations(h.vertices,k))if(h.edges.every(e=>e.some(v=>s.includes(v))))return s;return null;}
  function greedy(h){let left=[...h.edges],out=[];while(left.length){let v=h.vertices.reduce((a,b)=>left.filter(e=>e.includes(b)).length>left.filter(e=>e.includes(a)).length?b:a);out.push(v);left=left.filter(e=>!e.includes(v));}return out;}
  function path(h,a,b){let queue=[[a]],seen=new Set([a]);while(queue.length){const p=queue.shift(),v=p[p.length-1];if(v===b)return p;h.edges.forEach((e,i)=>{if(e.includes(v))e.forEach(w=>{if(!seen.has(w)){seen.add(w);queue.push([...p,'e'+(i+1),w]);}});});}return null;}
  const mono=(h,c)=>h.edges.flatMap((e,i)=>e.every(v=>c[h.vertices.indexOf(v)]===c[h.vertices.indexOf(e[0])])?[i]:[]);
  const colourings=h=>Array.from({length:2**h.vertices.length},(_,m)=>h.vertices.map((_,i)=>(m>>i)&1)).filter(c=>!mono(h,c).length);
  const antichain=f=>f.every((a,i)=>f.every((b,j)=>i===j||!a.every(v=>b.includes(v))));
  const intersecting=f=>f.every((a,i)=>f.slice(i+1).every(b=>a.some(v=>b.includes(v))));
  function intersectingMaximum(n,k){const es=combinations(Array.from({length:n},(_,i)=>i+1),k);if(es.length>15)throw Error('Capped at 15 candidate edges');let best=[];function visit(i,f){if(f.length+es.length-i<=best.length)return;if(i===es.length){best=[...f];return;}visit(i+1,f);if(f.every(e=>e.some(v=>es[i].includes(v))))visit(i+1,[...f,es[i]]);}visit(0,[]);return best;}
  function rng(seed){let x=seed>>>0;return ()=>{x=(Math.imul(1664525,x)+1013904223)>>>0;return x/4294967296;};}
  function random(n,k,p,seed){const vertices=Array.from({length:n},(_,i)=>String(i+1)),r=rng(seed);return create(vertices,combinations(vertices,k).filter(()=>r()<p));}
  const fano=create(['1','2','3','4','5','6','7'],[['1','2','3'],['1','4','5'],['1','6','7'],['2','4','6'],['2','5','7'],['3','4','7'],['3','5','6']]);
  const example=()=>create(['A','B','C','D','E','F'],[['A','B','C'],['B','D'],['C','D','E'],['A','E','F']]);
  const api={choose,combinations,create,incidence,degrees,transpose,dual,matching,transversal,greedy,path,mono,colourings,antichain,intersecting,intersectingMaximum,rng,random,fano,example};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.HyperMath=api;
})(typeof globalThis!=='undefined'?globalThis:this);
