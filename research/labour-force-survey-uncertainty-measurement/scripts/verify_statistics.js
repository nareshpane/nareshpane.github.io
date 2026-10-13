'use strict';const assert=require('node:assert/strict'),S=require('../js/statistics.js');
for(const n of [0,1,40,400]){const x=S.choose(1000,n,S.rng(12));assert.equal(new Set(x).size,n);assert(x.every(i=>i>=0&&i<1000));}
assert(S.interval(100,5,99).margin>S.interval(100,5,95).margin);assert(S.interval(100,5,95).margin>S.interval(100,5,90).margin);
assert.throws(()=>S.interval(0,-1));assert.equal(S.difference(1,2,3,3,1).se,0);
assert(Math.abs(S.difference(1e6,1030000,25000,25000,.7).variance-375e6)<1e-6);
for(const scheme of ['srs','stratified','cluster']){const a=S.sampling(.08,400,500,scheme,3701),b=S.sampling(.08,400,500,scheme,3701);assert.deepEqual(a,b);assert(a.values.every(x=>x>=0&&x<=1));assert(Math.abs(a.rmse**2-a.variance-a.bias**2)<1e-12);console.log(scheme,'mean',a.mean,'SD',a.sd);}
const a=S.bootstrap(1000,338);assert.equal(a.truth,.13);assert(a.variance>0);assert(Math.abs(S.adjustedCV(.04,.002,.045)-.0470185)<1e-6);
console.log('PASS: numerical implementations, deterministic seeds, sampling uniqueness, MSE identity, intervals, covariance and bootstrap.');
