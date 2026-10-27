'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const M=require('../js/math.js');
let checked=0;
function check(n,e,r){
 const direct=M.candidates(n,e,r).filter(x=>x.valid).map(x=>M.key(x.arcs)).sort();
 const parity=M.parity(n,e,r).forests.map(M.key).sort();
 assert.deepEqual(parity,direct);
 assert.equal(M.minor(n,e,r).count,BigInt(direct.length));
 checked++;
}
for(let n=1;n<=3;n++){
 const arcs=Array.from({length:n},(_,u)=>Array.from({length:n},(_,v)=>[u,v])).flat().filter(e=>e[0]!==e[1]);
 for(let mask=0;mask<2**arcs.length;mask++){
  const e=arcs.filter((_,i)=>mask&(1<<i));
  for(let rm=1;rm<2**n;rm++)check(n,e,Array.from({length:n},(_,i)=>i).filter(i=>rm&(1<<i)));
 }
}
const saved=JSON.parse(fs.readFileSync(path.join(__dirname,'../data/validation-results.json'),'utf8'));
for(const s of saved.samples){
 check(s.n,s.edges,s.roots);
 assert.deepEqual(M.candidates(s.n,s.edges,s.roots).filter(x=>x.valid).map(x=>M.key(x.arcs)).sort(),s.forests.map(M.key).sort());
}
assert.equal(saved.samples[0].count,12);assert.equal(saved.samples[1].count,6);
const products=[
 ['e1','e2','e5','e7′'],['e1','e2′','e6','e7′'],['e1','e3','e6','e7′'],['e1','e5','e6','e7′'],
 ['e1′','e2′','e4','e7′'],['e1′','e3','e4','e7′'],['e1′','e4','e5','e7′'],['e2','e3','e4','e7′'],
 ['e2','e4','e5','e7′'],['e2′','e4','e6','e7′'],['e3','e4','e6','e7′'],['e4','e5','e6','e7′']
];
const products2=[['e1','e2′','e7′'],['e1','e3','e7′'],['e1','e5','e7′'],['e2′','e4','e7′'],['e3','e4','e7′'],['e4','e5','e7′']];
for(const [roots,expected] of [[[4],products],[[2,4],products2]]){
 const actual=M.parity(5,M.paper,roots).forests.map(f=>f.map(e=>M.labels[e.join(',')]).sort().join('*')).sort();
 assert.deepEqual(actual,expected.map(t=>t.sort().join('*')).sort());
}
// Two disjoint triangles and an isolated root have multiplicity four.
const sixCycle=Array.from({length:6},(_,u)=>[u,u<3?(u+1)%3:3+(u-3+1)%3]);
const sym=sixCycle.flatMap(([u,v])=>[[u,v],[v,u]]);
assert.equal(M.parity(7,sym,[6]).forests.length,0);
assert.equal(M.parity(7,sym,[6]).groups.some(g=>g.count===4),true);
const report={cases:checked,sourceProducts:18,multipleCycleCancellation:'pass',failures:0};
fs.writeFileSync(path.join(__dirname,'../data/javascript-test-results.json'),JSON.stringify(report,null,2)+'\n');
console.log(report);
