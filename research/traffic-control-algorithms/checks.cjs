/* Run: node research/traffic-control-algorithms/checks.cjs */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const M = require('./simulation.js');
let checks = 0;
function check(test, message) { assert.ok(test, message); checks++; }
// Compatibility matrix and feasible teaching stages.
M.conflicts.forEach((row,i)=>row.forEach((v,j)=>{
  check(v===M.conflicts[j][i], 'symmetric conflict matrix');
  if(i===j)check(v===0,'zero diagonal');
}));
[[0,1],[2,3],[4,5],[6,7]].forEach(([a,b])=>check(M.conflicts[a][b]===0,'opposing stage compatible'));
check(M.conflicts[0][2]===1,'perpendicular throughs conflict');
// Sample the entire opening sequence at 60 Hz. Check entry permission, separation,
// and physical occupancy, not just that phase labels have the desired values.
const hero = new M.Hero();
let nsDischarged=false,ewDischarged=false,allRedClearing=false;
for(let tick=0;tick<67*60;tick++){
  const before=hero.queues; hero.tick(1/60);
  hero.lanes.forEach(lane=>lane.forEach((car,j)=>{if(j)check(lane[j-1].p-car.p>=37-1e-8,'following separation');}));
  const occupied=hero.lanes.map(l=>l.some(v=>v.p>203&&v.p<397));
  if(M.heroSignal(hero.time).state==='ALL RED'&&occupied.some(Boolean))allRedClearing=true;
  check(!(occupied[0]||occupied[1])||!(occupied[2]||occupied[3]),'no conflicting vehicle occupancy');
  if(hero.time>=53&&hero.time<61)check(!occupied.some(Boolean),'pedestrian stage clear of vehicles');
  if(hero.time>17&&hero.time<27&&hero.queues[0]<before[0])nsDischarged=true;
  if(hero.time>36&&hero.time<45&&hero.queues[2]<before[2])ewDischarged=true;
}
check(nsDischarged&&ewDischarged,'both green phases discharge queues');
check(allRedClearing,'vehicles visibly finish clearing during all-red');
check(hero.entries.length>0&&hero.entries.every(e=>e.state==='GREEN'),'no new entry during yellow or all-red');
hero.entries.forEach(e=>check(M.heroSignal(e.time).phase===(e.lane<2?0:1),'correct entry phase'));
const initial = new M.Hero(); hero.reset();check(JSON.stringify(hero)===JSON.stringify(initial),'deterministic restart');
const h1=new M.Hero(),h2=new M.Hero();for(let i=0;i<1800;i++)h1.advance(1/60);for(let i=0;i<600;i++)h2.advance(3/60);
check(Math.abs(h1.time-h2.time)<.02,'speed only changes clock rate');
check(h1.lanes.every((l,i)=>l.every((v,j)=>Math.abs(v.p-h2.lanes[i][j].p)<1)),'frame grouping preserves trajectories');
// A receiving queue plus in-transit reservations must never exceed storage.
for(const strategy of ['fixed','actuated','pressure'])for(const demand of [.3,1,2.5]){
  const a=new M.City({strategy,demand}), b=new M.City({strategy,demand});
  let prev=a.nodes.map(n=>({state:n.state,phase:n.phase}));
  for(let t=0;t<600;t++){
    a.step(); const m=a.metrics;
    check(m.generated===m.queued+m.transit+m.backlog+m.throughput,'vehicle conservation, including external backlog');
    a.nodes.forEach((n,i)=>{
      n.q.forEach((q,d)=>check(q.length+n.reserved[d]<=20,'finite receiving storage'));
      const last=prev[i];if(last.phase!==n.phase)check(last.state==='ALL RED','phase change follows all-red');
    });
    prev=a.nodes.map(n=>({state:n.state,phase:n.phase}));
  }
  b.step(600);check(JSON.stringify(a.metrics)===JSON.stringify(b.metrics),'identical reproducible city runs');
  check(a.metrics.throughput>0,'network throughput exists');
}
// Corridor: no car passes a red stop point, even under unhelpful offsets/speeds.
for(const settings of [[80,36,40],[50,0,60],[120,60,25]]){
 const c=new M.Corridor(...settings);
 for(let t=0;t<240*30;t++){
  const old=c.cars.map(v=>({...v}));const green=Array.from({length:5},(_,i)=>c.green(i));c.advance(1/30);
  c.cars.forEach((v,i)=>{
   if(v.passed>old[i].passed)check(green[v.passed],'corridor crossing is green');
   if(i)check(c.cars[i-1].x-v.x>=9-1e-8,'corridor vehicle separation');
  });
 }
}
check(new M.Corridor().bandwidth().width>27.8,'36 second default progression');
const fixed=M.fixedQueues(32);check(fixed.every(p=>p[1]>=0&&p[2]>=0),'nonnegative fluid queues');
const arrivals=M.comparison('fixed').at(-1),act=M.comparison('actuated').at(-1);
check(arrivals.served+arrivals.q[0]+arrivals.q[1]===act.served+act.q[0]+act.q[1],'comparison uses identical arrivals');
check((15+3-8)+(7+5)===22&&(15+3)+(7+5-7)===23,'worked example arithmetic');
const plan=M.mpcPlan([15,7],0,0);check(plan.steps.length===5&&plan.steps.every(s=>s.q.every(q=>q>=0)),'valid MPC horizon');
// Local resources, anchor targets and HTML IDs.
const htmlPath=path.resolve(__dirname,'../traffic-control-algorithms.html');const html=fs.readFileSync(htmlPath,'utf8');
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]);check(new Set(ids).size===ids.length,'unique HTML IDs');
for(const [,url] of html.matchAll(/(?:src|href)="([^"]+)"/g)){
 if(/^https?:/.test(url))continue;
 if(url.startsWith('#'))check(ids.includes(url.slice(1)),`anchor exists ${url}`);
 else check(fs.existsSync(path.resolve(path.dirname(htmlPath),url)),`local path exists ${url}`);
}
const pageScript=fs.readFileSync(path.join(__dirname,'page.js'),'utf8');
for(const [,id] of pageScript.matchAll(/\$\('([^']+)'\)/g))check(ids.includes(id),`script ID exists ${id}`);
const index=fs.readFileSync(path.resolve(__dirname,'../../research.html'),'utf8');
check(/<ul class="research-list">\s*<li>\s*<a class="project-title" href="research\/traffic-control-algorithms.html">/.test(index),'research index first entry');
console.log(`PASS: ${checks.toLocaleString()} assertions (safety, queues, conservation, deterministic models, arithmetic and local links).`);
