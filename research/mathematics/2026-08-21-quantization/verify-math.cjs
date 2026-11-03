/* Run: node research/mathematics/2026-08-21-quantization/verify-math.cjs */
'use strict';
const assert=require('node:assert/strict');
const M=require('./quantization-math.js');
let checks=0;
const near=(a,b,t=1e-10)=>{assert.ok(Math.abs(a-b)<=t,`${a} != ${b}`);checks++;};
// Explicit threshold ownership, saturation, signed errors and fine grids.
for(const [x,k,q] of [[-1.4,0,-.875],[-1,0,-.875],[-.75,1,-.625],[0,4,.125],[.37,5,.375],[1,7,.875],[1.4,7,.875]]){
  const r=M.uniform(x);assert.equal(r.index,k);near(r.value,q);near(r.error,x-q);
}
for(let b=1;b<=16;b++){
  const L=2**b;
  for(let i=0;i<101;i++){
    const x=-1+i/50,r=M.uniform(x,L);
    assert.ok(r.index>=0&&r.index<L);assert.ok(Math.abs(r.error)<=r.delta/2+1e-12);checks+=2;
  }
}
near(M.round(-1.5),-1);near(M.round(.5),1);
near(M.staircase(0,.25,'rise'),.125);near(M.staircase(0,.25,'tread'),0);
near(M.staircase(-.125,.25,'tread'),0);near(M.staircase(1.4,.25,'rise'),.875);
// Signed symmetric one-bit degeneracy, asymmetric one-sided calibration and zeros.
for(let b=1;b<=8;b++)for(const mode of ['symmetric','asymmetric'])for(const [a,z] of [[-3,3],[.01,.08],[-9,-2],[0,0],[2,2]])for(const x of [-20,-3,-.5,0,.5,3,20]){
  const r=M.affine(x,a,z,b,mode);
  assert.ok(Number.isFinite(r.scale)&&r.scale>0);assert.ok(r.q>=r.qmin&&r.q<=r.qmax);near(r.xhat,r.scale*(r.q-r.zeropoint));near(r.err,x-r.xhat);
  assert.equal(r.L,r.qmax-r.qmin+1);checks+=3;
}
assert.equal(M.affine(1,-1,1,1,'symmetric').L,1);
near(M.affine(0,1,3,4,'asymmetric').xhat,0);
// Analytic uniform midpoint MSE and monotonic alternating optimization.
for(const kind of ['uniform','gaussian','skew'])for(const L of [2,4,8]){
  const data=M.distribution(kind),centers=Array.from({length:L},(_,i)=>-1+(i+.5)*2/L);
  near(data.reduce((s,p)=>s+p.w,0),1);
  if(kind==='uniform')near(M.distortion(data,centers),(2/L)**2/12,7e-8);
  let c=centers.map(x=>-.85+(x+1)*.45),D=M.distortion(data,c);
  for(let n=0;n<60;n++){c=M.lloyd(data,c);const next=M.distortion(data,c);assert.ok(next<=D+1e-13);D=next;checks++;}
}
const c=M.lloyd(M.distribution('uniform'),[-.8,.4]);near(c[0],-.6,1e-12);near(c[1],.4,1e-12);
near(10*Math.log10(1.5*2**16),6.020599913279624*8+1.7609125905568124);
const a=M.seeded(123),b=M.seeded(123);for(let i=0;i<100;i++)near(a(),b());
// The university story uses these concrete arithmetic and affine examples.
assert.deepEqual([.7,1.2,1.8,2.4,2.9,3.3].map(M.round),[1,1,2,2,3,3]);checks++;
near(2.4-M.round(2.4),.4);
const storyWeight=M.affine(.73,-1.24,2.61,4,'asymmetric');
assert.equal(storyWeight.q,8);checks++;
assert.equal(storyWeight.zeropoint,5);checks++;
assert.equal(storyWeight.q.toString(2).padStart(4,'0'),'1000');checks++;
near(storyWeight.scale,77/300);near(storyWeight.xhat,.77);near(storyWeight.err,-.04);
console.log(`Passed ${checks} numerical checks: boundaries, overload, signed ties, 1–16 bits, affine calibration, distortion and Lloyd–Max descent.`);
