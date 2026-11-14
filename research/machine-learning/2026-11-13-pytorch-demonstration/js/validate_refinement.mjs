// Independent parity with canonical Python flow matrices and stress-case invariants.
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {dispatch,browserPolicy,empiricalCvar} from './allocation-model.js';
const read=name=>JSON.parse(readFileSync(new URL(`../data/${name}.json`,import.meta.url),'utf8'));
const d=read('allocation-demo'),golden=read('refinement-dispatch-cases');
const near=(a,b,tol=1e-11)=>assert(Math.abs(a-b)<tol,`${a} versus ${b}`);
let parity=0,stress=0;
for(const[name,policy]of Object.entries(d.policies)){
 for(let i=0;i<d.test_scenarios.length;i++){
  const r=dispatch(d.test_scenarios[i],policy,d.capacity,d.cost);
  assert.deepEqual(r.flow,golden.policies[name].flows[i]);
  near(r.cost,golden.policies[name].losses[i]);assert.equal(r.lost,0);parity++;
 }
 for(const multiplier of [.5,1,1.5])for(const stations of [2,3])for(const extra of [0,16])for(const row of d.test_scenarios){
  const demand=[...row];demand[7]+=extra;
  const cap=d.capacity.map((v,j)=>j<stations?Math.round(v*multiplier):0),r=dispatch(demand,policy,cap,d.cost);
  assert(r.flow.every((f,i)=>f.every(Number.isInteger)&&f.every(v=>v>=0)&&f.reduce((s,v)=>s+v,0)+r.unserved[i]===demand[i]));
  assert(r.loads.every((v,j)=>v<=cap[j]));assert.equal(r.violations,0);
  assert.equal(r.served+r.lost,r.total);assert.equal(r.shortageCost,r.lost*15);
  near(r.totalCost,r.serviceCost+r.shortageCost);near(r.cost*r.total,r.totalCost);stress++;
 }
}
near(empiricalCvar([1,2,3,4],.625),11/3);
// Objective at the first browser iterate: independent reduction, before any updates.
const history=browserPolicy(d,d.capacity,.7,.25),first=history[0],a=first.policy;
assert.equal(history.length,100);assert(history.every(r=>Number.isFinite(r.objective)&&r.policy.every(row=>Math.abs(row.reduce((s,v)=>s+v,0)-1)<1e-12)));
const q=a.map((row,i)=>row.reduce((s,v,j)=>s+v*d.cost[i][j],0)),qmean=q.reduce((s,v)=>s+v,0)/8;
const losses=d.training_scenarios.map(row=>row.reduce((s,v,i)=>s+v*q[i],0)/row.reduce((s,v)=>s+v,0));
const avg=v=>v.reduce((s,x)=>s+x,0)/v.length;
const penalty=avg(d.training_scenarios.map(row=>d.capacity.reduce((s,cap,j)=>s+Math.max(0,row.reduce((v,n,i)=>v+n*a[i][j],0)-cap)**2,0)))/d.capacity.reduce((s,v)=>s+v,0);
near(first.objective,avg(losses)+.7*(2.5+avg(losses.map(v=>Math.max(0,v-2.5)))/.1)+.25*avg(q.map(v=>(v-qmean)**2))+1.5*penalty);
const report={status:'passed',canonical_python_flow_and_cost_parity:parity,integer_capacity_shortage_stress_cases:stress,browser_history_length:history.length,fractional_cvar:11/3};
writeFileSync(new URL('../data/refinement-js-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));
