'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const M=require('../js/math.js'),L=require('../js/learning.js');
const triangle=Array.from({length:3},(_,u)=>Array.from({length:3},(_,v)=>[u,v])).flat().filter(([u,v])=>u!==v);
const chain=[[0,1],[1,2],[2,3],[3,4]],forest=[[0,1],[1,2],[3,4]],cycle=[[0,1],[1,2],[2,0],[3,4]];
let traces=0;
for(const [edges,roots] of [[chain,[4]],[forest,[2,4]],[cycle,[4]]]){
 for(let start=0;start<5;start++){
  const rows=L.tracePath(5,edges,roots,start),last=rows.at(-1);
  assert.equal(last.status,edges===cycle&&start<3?'cycle':'root');
  for(const row of rows.slice(1))assert(edges.some(e=>e[0]===row.arc[0]&&e[1]===row.arc[1]));
  if(last.status==='root')assert(roots.includes(last.vertex));
  if(last.status==='cycle')assert(rows.slice(0,-1).some(r=>r.vertex===last.vertex));
  traces++;
 }
}
assert.throws(()=>L.tracePath(3,[],[2],3),RangeError);
assert.throws(()=>L.tracePath(3,[[0,1],[0,2]],[2],0));
assert.equal(L.tracePath(3,[],[2],0)[0].status,'dead-end');
assert.deepEqual(L.laplacian(3,triangle),[[2,-1,-1],[-1,2,-1],[-1,-1,2]]);
for(const [u,v] of triangle){
 const without=triangle.filter(e=>e[0]!==u||e[1]!==v),a=L.laplacian(3,triangle),b=L.laplacian(3,without);
 for(let i=0;i<3;i++)for(let j=0;j<3;j++)assert.equal(a[i][j]-b[i][j],i===u?(j===u?1:j===v?-1:0):0);
 assert.equal(M.minor(3,without,[2]).count,BigInt(M.candidates(3,without,[2]).filter(c=>c.valid).length));
}
for(const [roots,count] of [[[2],3],[[1,2],2],[[0,1,2],1]]){
 const rows=M.candidates(3,triangle,roots).filter(c=>c.valid);
 assert.equal(rows.length,count);assert.equal(M.minor(3,triangle,roots).count,BigInt(count));
 assert(rows.every(r=>r.arcs.length===3-roots.length));
}
assert.equal(M.valid(5,chain,[4]),true);assert.equal(M.valid(5,forest,[2,4]),true);
assert.equal(M.valid(5,[[0,1],[2,3]],[1,3,4]),true);assert.equal(M.valid(5,cycle,[4]),false);
const report={traceExamples:traces,edgeContributionChecks:6,rootSetCounts:[3,2,1],primerExamples:4,invalidInputs:'pass',failures:0};
fs.writeFileSync(path.join(__dirname,'../data/learning-test-results.json'),JSON.stringify(report,null,2)+'\n');
console.log(report);
