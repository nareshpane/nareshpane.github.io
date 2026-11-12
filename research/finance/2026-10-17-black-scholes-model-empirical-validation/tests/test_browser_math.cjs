const assert=require('node:assert/strict');
const B=require('../js/pricing.js');
assert.ok(Math.abs(B.price(100,100,1,.05,.2)-10.450583572)<2e-5);
for(const kind of ['call','put']){
 const args=[102,100,.7,.04,.3,.015],g=B.greeks(...args,kind),h=.01;
 const fd=(B.price(102+h,...args.slice(1),kind)-B.price(102-h,...args.slice(1),kind))/(2*h);
 assert.ok(Math.abs(fd-g.delta)<1e-5);
}
for(const s of [50,100,180]){
 assert.ok(Math.abs(B.price(s,100,.7,.04,.3,.02)-B.price(s,100,.7,.04,.3,.02,'put')-(s*Math.exp(-.014)-100*Math.exp(-.028)))<1e-10);
}
assert.equal(B.price(120,100,0,.03,.2),20);
assert.equal(B.N(0),.5);assert.ok(Math.abs(B.N(2)+B.N(-2)-1)<1e-12);
assert.throws(()=>B.price(NaN,100,1,.03,.2));
assert.deepEqual(B.paths(42),B.paths(42));assert.notDeepEqual(B.paths(42),B.paths(43));
console.log('PASS: browser reference pricing, call/put finite-difference delta, parity, limits, CDF, invalid input, seeded simulation.');
