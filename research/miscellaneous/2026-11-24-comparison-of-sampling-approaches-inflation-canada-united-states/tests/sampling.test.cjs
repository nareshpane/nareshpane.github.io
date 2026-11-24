'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
require('../js/sampling-engine.js');
const E=globalThis.CPISampling;
const cfg={N:128,n:48,seed:20261124,runs:250,rho:.3,association:.4,coverage:1,judgment:.55};
test('census, normalization and fixed-population reproduction',()=>{
  const pop=E.population(cfg);
  const near=(a,b)=>assert(Math.abs(a-b)<1e-12);
  near(pop.rows.reduce((a,v)=>a+v.share,0),1);
  const full=E.draw(pop,'SRS',cfg.N,cfg.seed);
  near(full.estimate,pop.truth);
  assert.equal(full.se,0);
  assert.deepEqual(E.run(cfg,['Stratified']).results,E.run(cfg,['Stratified']).results);
});
test('Monte Carlo variance, error and coverage identity',()=>{
  const {results}=E.run(cfg);
  for(const x of results){
    assert(Math.abs(x.rmse**2-x.variance-x.bias**2)<1e-9);
    assert(x.coverage>=0 && x.coverage<=1);
    assert(x.cost>0);
  }
});
test('stable design behavior under smaller frames',()=>{
  for(const scenario of ['Baseline','D · Missingness','G · Feed coverage','H · Disruption']){
    const {results}=E.run({...cfg,N:64,n:16,coverage:0,scenario});
    assert(results.every(x=>Number.isFinite(x.rmse)));
  }
});
test('donor imputation keeps the finite target and changes usable values',()=>{
  const disrupted={...cfg,scenario:'H · Disruption',loss:.2};
  const a=E.population(disrupted), b=E.population({...disrupted,imputation:'donor'});
  assert.equal(a.truth,b.truth);
  assert(a.rows.some((unit,i)=>unit.refused && unit.observed!==b.rows[i].observed));
  assert(a.rows.every((unit,i)=>unit.refused || unit.observed===b.rows[i].observed));
});
test('changing design or sample size leaves the generated population unchanged',()=>{
  const pop=E.population(cfg);
  const before=JSON.stringify(pop);
  const a=E.runPopulation(pop,{...cfg,n:24},['SRS']);
  const b=E.runPopulation(pop,{...cfg,n:40},['PPS','Stratified']);
  assert.equal(a.pop,b.pop);
  assert.equal(JSON.stringify(pop),before);
  assert.equal(a.results[0].sample.selected.length,24);
  assert.equal(b.results[0].sample.selected.length,40);
});

