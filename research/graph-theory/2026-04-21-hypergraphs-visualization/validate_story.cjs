'use strict';
const assert=require('node:assert/strict'),M=require('./math.js'),S=require('./story.js');
let checks=0;
const equal=(a,b,label)=>{assert.deepEqual(a,b,label);checks++;};
const full=M.example(),expectedPairs=[['A','B'],['A','C'],['A','E'],['A','F'],['B','C'],['B','D'],['C','D'],['C','E'],['D','E'],['E','F']];
equal(M.twoSection(full),expectedPairs,'Independent ten-pair list');
equal(M.incidence(full),[[1,0,0,1],[1,1,0,0],[1,0,1,0],[0,1,1,0],[0,0,1,1],[0,0,0,1]],'Independent matrix');
equal(M.degrees(full),[2,2,2,2,2,1],'Independent degrees');
equal(full.edges.map(e=>e.length),[3,2,3,3],'Edge sizes');
equal(M.edgeIntersections(full),[[3,1,1,1],[1,2,1,0],[1,1,3,1],[1,0,1,3]],'B transpose B');
equal(M.twoSection(M.create(['A','B','C','D'],[['A','B','C'],['A','B','D']])).length,5,'Repeated AB deduplicated');
equal(M.twoSection(M.create(['A','B','C'],[['A','B','C']])),M.twoSection(M.create(['A','B','C'],[['A','B'],['B','C'],['A','C']])),'Different families, same triangle');
equal(S.duration,18,'Full opening-to-conclusion duration');equal(S.stages.length,8,'Eight stages');
equal(S.stages.map(s=>s.start),[0,1.5,3.5,5.5,8,10.5,13,15.5],'Storyboard starts');
equal(S.stages.map(s=>s.end),[1.5,3.5,5.5,8,10.5,13,15.5,18],'Storyboard ends');
for(let i=0;i<S.stages.length;i++){
 const stage=S.stages[i],frame=S.at(stage.start),count=[0,1,2,3,4,4,4,4][i];
 equal(frame.stageIndex,i,'Exact boundary');equal(frame.h.edges,full.edges.slice(0,count),'Prefix family');
 equal(frame.matrix,M.incidence(frame.h),'Matrix derives from current family');equal(frame.total,[0,3,5,8,11,11,11,11][i],'Incidence total by stage');
 equal(frame.pairs.length,[0,3,4,7,10,10,10,10][i],'Projection count by stage');
 equal(S.at(stage.end-.0001).stageIndex,i,'End of interval');
 equal(S.at(stage.start,true).reveal,1,'Reduced motion skips interpolation');
}
for(let t=0;t<=18;t+=.025){
 const a=S.at(t),b=S.at(18-t);
 equal(a,S.at(t),'Direct seeking is deterministic');equal(b,S.at(18-t),'Backward seeking is deterministic');
 equal(a.degrees.reduce((s,x)=>s+x,0),a.total,'Every frame degree sum');
 equal(a.matrix.flat().reduce((s,x)=>s+x,0),a.total,'Every frame matrix ones');
 assert.ok(a.reveal>=0&&a.reveal<=1);
}
for(const t of [10.5,11.125,11.75,12.375])equal(S.at(t).active,[10.5,11.125,11.75,12.375].indexOf(t),'All matrix columns scanned');
equal(S.at(18).stageIndex,7,'Stable final frame');equal(S.at(100).time,18,'Upper clamp');equal(S.at(-1).time,0,'Lower clamp');
assert.throws(()=>S.at(18).h.edges[0].push('F'));checks++;
const edited=M.example();edited.edges[0].pop();equal(S.at(18).h,full,'Builder clone cannot mutate story');
// Exhaustive independent projection verification on all three-vertex hypergraphs.
for(let family=0;family<128;family++){
 const masks=Array.from({length:7},(_,i)=>i+1).filter((_,i)=>family&(1<<i));
 const v=['A','B','C'],h=M.create(v,masks.map(m=>v.filter((_,i)=>m&(1<<i))));
 const expected=[[0,1],[0,2],[1,2]].filter(([a,b])=>masks.some(m=>(m&(1<<a))&&(m&(1<<b)))).map(pair=>pair.map(i=>v[i]));
 equal(M.twoSection(h),expected,'Independent bitmask projection');
}
console.log(JSON.stringify({status:'PASS',checks,duration:S.duration,stages:S.stages.length,pairs:expectedPairs,incidences:11,defaultSizes:[3,2,3,3],defaultDegrees:[2,2,2,2,2,1]},null,2));
