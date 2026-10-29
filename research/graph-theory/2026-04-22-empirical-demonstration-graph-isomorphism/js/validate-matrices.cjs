/* Independent finite oracles for the source-row permutation convention.
   Run node js/validate-matrices.cjs --write. No browser or packages required. */
'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
for(const f of ['graph-core','matrix-operations','partition-refinement','isomorphism','automorphisms','search-tree','canonical-labeling','graph-renderer','matrix-renderer'])vm.runInThisContext(fs.readFileSync(path.join(__dirname,f+'.js'),'utf8'),{filename:f+'.js'});
const L=globalThis.GraphLab;let assertions=0;const checks=[],test=(x,m)=>{assert.ok(x,m);assertions++;};
const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b),oracle=(a,p)=>p.map(i=>p.map(j=>a[i][j]));
const g=L.examples('prism6'),f=[0,4,5,2,1,3],h=L.relabel(g,f),c=L.matrixCheck(g,h,f);
test(c.pairs.length===15&&c.pairs.every(x=>x.ok),'All 15 pairs checked, including non-edges');
test(eq(g.a[5],[0,0,1,1,1,0]),'Specified G row 6');test(eq(h.a[3],[0,1,1,0,0,1]),'Specified H row 4');
test(eq(f.map(v=>h.a[3][v]),g.a[5]),'Reordered H row matches G row');
test(eq(c.P,f.map(w=>L.range(6).map(v=>Number(w===v)))),'P has one at source row, target column');
test(c.ok&&c.difference.flat().every(x=>x===0),'True matrix multiplication gives zero difference');
checks.push('Specified six-vertex witness, both original rows, reordered row, all 15 pairs, and explicit P convention');
let permutations=0;
for(const p of L.permutations(L.range(6))){const r=L.matrixCheck(g,h,p);test(eq(r.product,oracle(h.a,p)),'Multiplication agrees with independent direct indexing');test(r.ok===L.checkMap(g,h,p).ok,'Matrix and pair check equivalent');test(r.difference.flat().every(x=>x===0)===r.ok,'Zero difference iff isomorphism');permutations++;}
checks.push('All 720 proposals on the triangular prism: direct indexing versus multiplication versus pair check');
const prismResults={};
for(const m of [3,6]){const graph=L.examples(m===3?'prism6':'prism12'),maps=L.prismMaps(m),exact=L.automorphisms(graph);test(exact.complete,'Independent adjacency search completed');test(new Set(maps.map(o=>o.p.join(','))).size===4*m,'Constructed maps distinct');test(eq(maps.map(o=>o.p.join(',')).sort(),exact.maps.map(p=>p.join(',')).sort()),'Structural enumeration equals independently found full group');
 for(const {p} of maps){test(L.matrixCheck(graph,graph,p).ok,'Every automorphism matrix preserves A');test(eq(L.pullback(graph.a,p),oracle(graph.a,p)),'Direct-index oracle');}
 for(let i=0;i<2*m;i++)for(let j=0;j<2*m;j++){const same=Math.floor(i/m)===Math.floor(j/m),x=i%m,y=j%m;const expected=same?Number((x-y+m)%m===1||(y-x+m)%m===1):Number(x===y);test(graph.a[i][j]===expected,'Actual graph equals displayed block construction');}
 for(const [u,v] of L.edges(graph)){let squares=0;for(let x=0;x<graph.n;x++)for(let y=0;y<graph.n;y++)if(x!==v&&y!==u&&x!==y&&graph.a[u][x]&&graph.a[x][y]&&graph.a[y][v])squares++;test(squares===(Math.floor(u/m)!==Math.floor(v/m)?2:1),'Four-cycle incidence intrinsically distinguishes connectors');}
 prismResults[2*m]={vertices:graph.n,edges:L.edges(graph).length,automorphisms:maps.length};
}
const cube=L.graph(8,L.range(4).flatMap(v=>[[v,(v+1)%4],[v+4,(v+1)%4+4],[v,v+4]]));test(L.automorphisms(cube).maps.length===48,'C4 prism exception is the cube with 48 automorphisms');test(L.prismMaps(4).length===16,'Naive factor count gives only 16 for the cube');
const compositions=L.prismMaps(3).map(o=>L.compose(o.p,f));test(new Set(compositions.map(p=>p.join(','))).size===12,'Twelve h∘a maps distinct');test(compositions.every(p=>L.matrixCheck(g,h,p).ok),'Every composition is an isomorphism');test(eq(compositions.map(p=>p.join(',')).sort(),L.isomorphisms(g,h,{all:true}).maps.map(p=>p.join(',')).sort()),'Compositions exhaust all isomorphisms to fixed H');
const square=L.examples('cycle4'),auts=L.automorphisms(square).maps;test(auts.length===8,'Square has eight symmetries');test(!L.matrixCheck(square,square,[1,0,2,3]).ok,'Invalid bijection fails');
for(const p of auts)for(const q of auts){test(eq(L.multiply(L.permutationMatrix(p),L.permutationMatrix(q)),L.permutationMatrix(L.compose(p,q))),'Source-row composition order');}
checks.push('Exact prism groups, all block entries, square and invalid map, all square matrix products, twelve distinct/exhaustive h∘a witnesses');
for(const name of ['path4','prism6','prism12','rook16','shrikhande']){const graph=L.examples(name),r=L.matrixInvariants(graph),neigh=v=>L.range(graph.n).filter(w=>graph.a[v][w]);test(eq(r.degrees,L.range(graph.n).map(v=>neigh(v).length)),'Degrees agree with independently listed neighbours');for(let i=0;i<graph.n;i++)for(let j=0;j<graph.n;j++)test(r.a2[i][j]===neigh(i).filter(k=>graph.a[k][j]).length,'Two-step walk oracle');test(r.triangles===L.invariants(graph).triangles,'Trace cubes agree with independent unordered triples');}
for(const name of ['rook16','shrikhande']){const graph=L.examples(name),r=L.matrixInvariants(graph);test(r.degrees.every(d=>d===6),'Degree six');test(L.edges(graph).length===48&&r.triangles===32,'Edges and triangles');for(let i=0;i<16;i++)for(let j=i+1;j<16;j++)test(r.a2[i][j]===2,'Both common-neighbour parameters are two');test(L.fourCliques(graph).length===(name==='rook16'?8:0),'All 1820 four-subsets');
 // A² = 4I + 2J follows from the parameters; on the all-ones line λ=6,
 // and on its orthogonal complement λ=±2. Trace=0 gives multiplicities 6,9.
 for(let i=0;i<16;i++)for(let j=0;j<16;j++)test(r.a2[i][j]===(i===j?6:2),'Shared spectral identity A²=4I+2J');}
checks.push('Degree, all two-walk entries, trace triangles, strongly regular parameters, full clique enumeration, shared spectral identity');
const full=L.searchTree(square),pruned=L.searchTree(square,{prune:true,automorphisms:auts});test(full.nodes.length===13&&full.nodes.filter(n=>n.leaf).length===8,'Full C4 tree 13 states / 8 leaves');test(pruned.nodes.length===7&&pruned.nodes.filter(n=>n.reason).length===4,'Pruned C4 tree 7 displayed states / 4 witnesses');
for(const n of pruned.nodes.filter(n=>n.reason)){const p=n.reason.permutation;test(n.seq.slice(0,-1).every(v=>p[v]===v),'Witness fixes prefix pointwise');test(p[n.reason.representative]===n.reason.vertex,'Witness carries earlier target to pruned target');test(L.matrixCheck(square,square,p).ok,'Witness matrix certificate');}
test(L.canonical(square).code===L.canonical(square,{prune:true,automorphisms:auts}).code,'P_C maximum survives');
for(let v=0;v<6;v++){const html=L.matrix(g,L.range(6),[v]);test(html.includes(`data-row="${v}" class="selected-row"`),'Selected row identity rendered');const picked=[...html.matchAll(/<td[^>]*class="[^"]*picked/g)];test(picked.length===11,'Selected row and column union has 2n−1 entries');const reordered=L.matrix(h,f,[f[v]]);test(reordered.includes(`data-row="${f[v]}" class="selected-row"`),'Mapped row rendered correctly');}
test(!L.treeSVG(full,{fit:true}).includes('min-width:'),'Fit trees do not force clipping');test(L.matrixView(L.examples('prism12').a).includes('Detailed numerical entries'),'Large matrices have heatmap and optional entries');
checks.push('C4 node and witness counts, pointwise witnesses, retained canonical output, selected rows/columns, reordered matrix markup, fit trees, heatmap detail');
const report={date:'2026-10-29',assertions,checks,six_vertex_mapping:f.map(v=>v+1),permutations_checked:permutations,prisms:prismResults,pruning:{full_states:13,full_leaves:8,displayed_pruned_states:7,witnesses:4,expanded_internal_states:2,retained_leaves:1}};
if(process.argv.includes('--write'))fs.writeFileSync(path.join(__dirname,'../data/validation/matrix-validation.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
