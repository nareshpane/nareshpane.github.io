/* Run: node js/validate-mathematics.cjs --write
 * Tests the exact browser kernel without a DOM or external libraries.
 */
'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const base=path.resolve(__dirname,'..');
for(const file of ['graph-core','partition-refinement','isomorphism','automorphisms','search-tree','canonical-labeling','graph-generation'])vm.runInThisContext(fs.readFileSync(path.join(__dirname,file+'.js'),'utf8'),{filename:file+'.js'});
const L=globalThis.GraphLab,checks=[],counts={},stats={};let assertions=0;
const test=(ok,message)=>{assert.ok(ok,message);assertions++;};
const norm=p=>p.map(c=>c.slice().sort((a,b)=>a-b).join(',')).join('|');
// Independent synchronous signature refinement, used only as an unordered oracle.
function synchronous(g,input){let p=input.map(c=>c.slice());while(true){let q=[];for(const c of p){let groups=new Map();for(const v of c){let key=p.map(w=>w.reduce((a,x)=>a+g.a[v][x],0)).join(',');if(!groups.has(key))groups.set(key,[]);groups.get(key).push(v);}q.push(...groups.values());}if(q.length===p.length)return q;p=q;}}
const unordered=p=>p.map(c=>c.slice().sort((a,b)=>a-b).join(',')).sort().join('|');
for(let n=1;n<=5;n++){
  const reps=[];const keys=new Set(),allSnKeys=new Set();
  for(let m=0;m<2**(n*(n-1)/2);m++){
    const g=L.fromMask(n,m),pi=[L.range(n)],r=L.refine(g,pi),c=L.canonical(g),aut=L.automorphisms(g);
    test(r.equitable,'Stable partition equitable');test(L.finer(r.cells,pi,n),'Ordered refinement');test(unordered(r.cells)===unordered(synchronous(g,pi)),'Independent closure agrees');
    test(c.complete&&aut.complete,'Small search completed');test(aut.maps.every(p=>L.checkMap(g,g,p).ok),'Every automorphism valid');
    test(L.isomorphisms(g,g).maps.length===1,'Self isomorphism');
    for(const seed of [1981,2014,42]){let p=L.shuffle(n,seed),h=L.relabel(g,p),rh=L.refine(h,L.transport(pi,p)),ch=L.canonical(h);test(c.code===ch.code,'Canonical form invariant');test(norm(rh.cells)===norm(L.transport(r.cells,p)),'Refinement equivariant');test(L.isomorphisms(g,h).maps.length>0,'Relabelling recognized');}
    const cp=L.canonical(g,{prune:true,automorphisms:aut.maps});test(cp.code===c.code&&cp.complete,'P_C preserves canonical output');
    let exhaustive=L.exhaustiveCanonical(g);allSnKeys.add(exhaustive.code);
    if(!keys.has(c.code)){keys.add(c.code);reps.push(g);}
    // Verify coloured input equivariance and incremental singleton queue.
    let colour=[L.range(n).filter(v=>v%2===0),L.range(n).filter(v=>v%2===1)].filter(c=>c.length),rc=L.refine(g,colour),p=L.shuffle(n,7);
    test(L.finer(rc.cells,colour,n),'Ordered input colours preserved');test(norm(L.refine(L.relabel(g,p),L.transport(colour,p)).cells)===norm(L.transport(rc.cells,p)),'Coloured equivariance');
    let target=L.target(r.cells);if(target.length){let v=target[0],i=L.individualize(r.cells,v),inc=L.refine(g,i,[[v]]);test(L.equitable(g,inc.cells),'Incremental refiner equitable');test(unordered(inc.cells)===unordered(synchronous(g,i)),'Singleton active queue sound');}
  }
  test(keys.size===L.knownCounts[n]&&allSnKeys.size===keys.size,'Known counts under independent canonical conventions');
  for(let i=0;i<reps.length;i++)for(let j=0;j<i;j++)test(L.isomorphisms(reps[i],reps[j]).maps.length===0,'Representatives pairwise non-isomorphic');
  for(const method of ['naive','canonical'])test(L.generate(n,method).count===L.knownCounts[n],'Both executable generation strategies');
  counts[n]=keys.size;
}
checks.push('Every labelled graph through n=5: refinement, automorphisms, relabellings, P_C, independent S_n convention, and two generation strategies');
const families={path4:2,cycle4:8,cycle6:12,complete5:120,star6:120,asymmetric7:1,prism6:12,prism12:24,paper9:8};
for(const [name,expected] of Object.entries(families)){
  const g=L.examples(name),a=L.automorphisms(g);test(a.complete&&a.maps.length===expected,name+' group order');let gens=L.generators(g.n,a.maps);test(L.closure(g.n,gens).length===expected,'Generators complete');
  let full=L.searchTree(g,{limit:2000}),pruned=L.searchTree(g,{limit:2000,prune:true,automorphisms:a.maps});
  test(full.complete&&pruned.complete,'Tree completed');
  for(const tree of [full,pruned])for(const node of tree.nodes){test(L.equitable(g,node.cells),'Node equitable');test(node.seq.every(v=>node.cells.some(c=>c.length===1&&c[0]===v)),'Individualized vertices singleton');if(node.parent!==null){let parent=tree.nodes[node.parent];test(parent.target.includes(node.seq.at(-1)),'Child belongs to target');}if(node.reason){let p=node.reason.permutation;test(L.checkMap(g,g,p).ok,'Pruning witness automorphism');test(node.seq.slice(0,-1).every(v=>p[v]===v),'Witness fixes prefix');}}
  const cf=L.canonical(g),cp=L.canonical(g,{prune:true,automorphisms:a.maps});test(cf.complete&&cp.complete&&cf.code===cp.code,'Canonical outcome preserved');
  stats[name]={automorphisms:a.maps.length,generators:gens.length,full_nodes:full.nodes.length,full_leaves:full.nodes.filter(n=>n.leaf).length,pruned_nodes_including_witnesses:pruned.nodes.length,pruned_witnesses:pruned.nodes.filter(n=>n.reason).length,canonical_nodes:cf.nodes,canonical_leaves:cf.leaves};
}
checks.push('Tree target, singleton, equitable, stabilizer-witness, and canonical preservation checks across nine graph families');
const prism=L.examples('prism6'),k33=L.examples('k33');test(L.invariants(prism).triangles===2&&L.invariants(k33).triangles===0,'Prism/K33 triangle certificate');test(!L.isomorphisms(prism,k33).maps.length,'Known non-isomorphic pair rejected');
const rg=L.examples('rook16'),sg=L.examples('shrikhande');test(L.invariants(rg).cliques4===8&&L.invariants(sg).cliques4===0,'Rook/Shrikhande certificate');
let catalogueChecks={};
for(const n of [6,7,8]){
  let lines=fs.readFileSync(path.join(base,`data/generated-graphs/graph${n}.g6`),'utf8').trim().split(/\s+/),keys=new Set();
  test(lines.length===L.knownCounts[n],'Catalogue count');
  for(let i=0;i<lines.length;i++){
    let g=L.decodeGraph6(lines[i]),c=L.canonical(g);test(g.n===n&&c.complete,'Catalogue canonical search complete');test(!keys.has(c.code),'Catalogue pairwise distinct IR keys');keys.add(c.code);
    if(i<20||i%101===0){let h=L.relabel(g,L.shuffle(n,i+2014));test(c.code===L.canonical(h).code,'Catalogue sample relabelling invariance');}
  }
  catalogueChecks[n]={representatives:lines.length,unique_exact_IR_keys:keys.size};
}
checks.push('Every official catalogue representative n=6,7,8 has a completed unique IR key; seeded sample relabellings retain keys');
test(!L.canonical(L.examples('complete5'),{limit:1}).complete,'Incomplete canonical search labelled');test(!L.automorphisms(L.examples('complete5'),{limit:1}).complete,'Incomplete group search labelled');
assert.throws(()=>L.parseGraph('0 1\n0 0',2,'matrix'));assertions++;assert.throws(()=>L.validatePartition([[0,0],[1]],2));assertions++;assert.throws(()=>L.generate(6));assertions++;
const report={date:'2026-10-29',node:process.version,assertions,checks,live_counts:counts,families:stats,catalogues:catalogueChecks};
if(process.argv.includes('--write'))fs.writeFileSync(path.join(base,'data/validation/javascript-validation.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));
