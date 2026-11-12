'use strict';
const assert=require('node:assert/strict'),E=require('../js/enhancements.js');
const a=E.replicate(1000,338);assert.deepEqual(a,E.replicate(1000,338));assert.equal(a.uy,1060);assert.equal(a.lx,7250);assert(Math.abs(a.original-1060/7250)<1e-14);
for(const row of a.rows){assert(row.m.every(m=>m.reduce((s,x)=>s+x,0)===3));assert(row.l>0);assert.equal(row.rate,row.u/row.l);}
// Enumerate 64 equally probable ordered samples in each stratum: 4096 ratios.
const draws=[];for(let i=0;i<4;i++)for(let j=0;j<4;j++)for(let k=0;k<4;k++)draws.push([i,j,k]);const exact=[];
for(const d0 of draws)for(const d1 of draws){let u=0,l=0;[d0,d1].forEach((d,h)=>d.forEach(i=>{u+=E.W[h]*E.U[h][i];l+=E.W[h]*E.L[h][i];}));exact.push(u/l);}
const theory=E.moments(exact),mc=E.replicate(100000,338);assert(Math.abs(mc.variance/theory.variance-1)<.02);assert(Math.abs(mc.mean-theory.mean)<.0003);
const missing={};for(const mode of ['MCAR','MAR','MNAR']){const x=E.impute(5000,3701,mode);assert.deepEqual(x,E.impute(5000,3701,mode));assert.equal(x.failed,0);missing[mode]=x;}
assert(Math.abs(missing.MCAR.imputed.mean-.2)<.003);assert(Math.abs(missing.MAR.imputed.mean-.2)<.004);assert(missing.MAR.observed.mean<.18);assert(missing.MNAR.imputed.mean<.14);
// All synthetic table probabilities and expansion totals.
const frame=[[100,300,30],[200,300,30],[150,450,45],[300,450,45],[100,250,25],[150,250,25],[200,600,60],[400,600,60]];for(const[M,G,w]of frame)assert(Math.abs(M/G*10/M-1/w)<1e-15);assert.equal(18*30+24*45+20*25+16*60,3080);assert.equal(10*(30+45+25+60),1600);
console.log(JSON.stringify({passed:true,exactReplicateMoments:theory,monteCarloMoments:{mean:mc.mean,variance:mc.variance},original:a.original,missing},null,2));
