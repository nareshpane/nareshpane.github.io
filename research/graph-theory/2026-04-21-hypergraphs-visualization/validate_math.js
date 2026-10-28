/* Run: node research/graph-theory/2026-04-21-hypergraphs-visualization/validate_math.js */
'use strict';
const assert=require('node:assert/strict'),M=require('./math.js');
let checks=0;const check=(b,msg)=>{assert.ok(b,msg);checks++;};
const sum=a=>a.reduce((s,x)=>s+x,0),same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const h=M.example();
check(same(M.degrees(h),[2,2,2,2,2,1]),'Default degrees');
check(same(h.edges.map(e=>e.length),[3,2,3,3]),'Default sizes');
check(sum(M.degrees(h))===11,'Default incidence total');
check(same(M.incidence(h),[[1,0,0,1],[1,1,0,0],[1,0,1,0],[0,1,1,0],[0,0,1,1],[0,0,0,1]]),'Independent default matrix');
check(M.matching(h).length===2&&M.transversal(h).length===2,'Default optima');
check(!M.path(M.create(['a','b'],[]),'a','b'),'Disconnected vertices');
assert.throws(()=>M.create(['A'],[[]]));assert.throws(()=>M.create(['A'],[['A'],['A']]));
// Independent bitmask solvers for every simple nonempty-edge hypergraph on 3 vertices.
const V=['a','b','c'],allEdges=Array.from({length:7},(_,i)=>i+1);
for(let family=0;family<128;family++){
 const masks=allEdges.filter((_,i)=>family&(1<<i)),edges=masks.map(m=>V.filter((_,i)=>m&(1<<i))),g=M.create(V,edges);
 let tau=4,nu=0;
 for(let cover=0;cover<8;cover++)if(masks.every(m=>m&cover))tau=Math.min(tau,cover.toString(2).replaceAll('0','').length);
 for(let em=0;em<2**masks.length;em++){let used=0,n=0,ok=true;for(let i=0;i<masks.length;i++)if(em&(1<<i)){if(used&masks[i])ok=false;used|=masks[i];n++;}if(ok)nu=Math.max(nu,n);}
 check(M.transversal(g).length===tau&&M.matching(g).length===nu,'Independent optimization '+family);
 check(nu<=tau&&tau<=Math.max(0,...edges.map(e=>e.length))*nu,'Rank bounds');
 check(sum(M.degrees(g))===sum(edges.map(e=>e.length)),'Handshake');
 check(same(M.incidence(M.dual(g)),M.transpose(M.incidence(g))),'Dual transpose');
 check(M.greedy(g).every(v=>V.includes(v))&&edges.every(e=>e.some(v=>M.greedy(g).includes(v))),'Greedy feasible');
 for(const a of V)for(const b of V){let reachable=new Set([a]);for(let it=0;it<3;it++)for(const e of edges)if(e.some(v=>reachable.has(v)))e.forEach(v=>reachable.add(v));const p=M.path(g,a,b);check(Boolean(p)===reachable.has(b),'Independent connectivity');if(p){const vs=p.filter((_,i)=>i%2===0),es=p.filter((_,i)=>i%2===1);check(new Set(vs).size===vs.length&&new Set(es).size===es.length,'Berge distinctness');for(let i=1;i<p.length;i+=2){const e=edges[Number(p[i].slice(1))-1];check(e.includes(p[i-1])&&e.includes(p[i+1]),'Berge membership');}}}
}
// All complete uniform examples allowed by the UI, counts checked against enumeration.
for(let n=1;n<=8;n++)for(let k=1;k<=n;k++){const v=Array.from({length:n},(_,i)=>''+i),es=M.combinations(v,k),g=M.create(v,es);check(es.length===M.choose(n,k),'Uniform count');check(M.degrees(g).every(d=>d===M.choose(n-1,k-1)),'Uniform degree');check(sum(M.degrees(g))===k*es.length,'Uniform incidences');}
// Every pair of Fano vertices occurs once; all 128 binary assignments fail.
for(const pair of M.combinations(M.fano.vertices,2))check(M.fano.edges.filter(e=>pair.every(v=>e.includes(v))).length===1,'Fano pair uniqueness');
check(M.degrees(M.fano).every(d=>d===3),'Fano regularity');
check(M.colourings(M.fano).length===0,'Fano no Property B');
check(M.colourings(h).length>0,'Running example Property B');
for(let a=0;a<=7;a++)check(M.choose(a,2)+M.choose(7-a,2)>=9,'Same-colour pairs');
// All 65,536 families on [4], with integer LYM weights (common denominator 12).
const sets=Array.from({length:16},(_,m)=>m),bits=m=>m.toString(2).replaceAll('0','').length;let antichains=0,largest=0;
for(let mask=0;mask<65536;mask++){const f=sets.filter((_,i)=>mask&(1<<i));const valid=f.every((a,i)=>f.every((b,j)=>i===j||(a&b)!==a));const arrays=f.map(a=>[1,2,3,4].filter((_,i)=>a&(1<<i)));check(M.antichain(arrays)===valid,'Independent antichain');if(valid){antichains++;largest=Math.max(largest,f.length);check(sum(f.map(a=>12/M.choose(4,bits(a))))<=12,'Exact integer LYM');}}
check(largest===6&&antichains===168,'Sperner n=4 and Dedekind count');
// Independently enumerate all families of pairs for each offered EKR instance.
for(const n of [4,5,6]){const es=M.combinations(Array.from({length:n},(_,i)=>i),2);let largest=0;for(let f=0;f<2**es.length;f++){const family=es.filter((_,i)=>f&(1<<i));if(family.length<=largest)continue;if(family.every((a,i)=>family.slice(i+1).every(b=>a.some(v=>b.includes(v)))))largest=family.length;}check(M.intersectingMaximum(n,2).length===largest&&largest===n-1,'Independent EKR n='+n);}
// Exact endpoint theorem on all triples of integer intervals with endpoints 0..3.
const intervals=M.combinations([0,1,2,3],2).concat([0,1,2,3].map(x=>[x,x]));for(const f of M.combinations(intervals,3)){const pairwise=f.every((a,i)=>f.slice(i+1).every(b=>Math.max(a[0],b[0])<=Math.min(a[1],b[1])));check(pairwise===(Math.max(...f.map(x=>x[0]))<=Math.min(...f.map(x=>x[1]))),'Interval Helly');}
check(M.random(6,3,0,42).edges.length===0,'p=0');check(M.random(6,3,1,42).edges.length===20,'p=1');check(same(M.random(6,3,.3,42),M.random(6,3,.3,42)),'Seed reproducibility');
for(let p=0;p<1;p+=.1){const a=M.random(6,3,p,42),b=M.random(6,3,p+.1,42);check(a.edges.every(e=>b.edges.some(f=>same(e,f))),'Coupled monotonicity');}
const r=M.rng(42);let edges=0,degree=0;for(let t=0;t<500;t++)for(const e of M.combinations(['1','2','3','4','5','6'],3))if(r()<.3){edges++;if(e.includes('1'))degree++;}
check(Math.abs(edges/500-6)<.4&&Math.abs(degree/500-3)<.3,'Seeded sample means near theory, not proof');
console.log(JSON.stringify({checks,allThreeVertexHypergraphs:128,allFourElementFamilies:65536,antichains,maximumAntichain:largest,fanoProperColourings:0,random500:{edges:edges/500,degree:degree/500},status:'PASS'},null,2));
