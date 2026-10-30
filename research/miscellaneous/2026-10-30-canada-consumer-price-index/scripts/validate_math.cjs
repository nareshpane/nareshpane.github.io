/* Exercise the actual page's calculation engine without browser rendering. */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const data=JSON.parse(fs.readFileSync(path.join(root,'data/processed/cpi.json'),'utf8'));
const context={window:{CPI_DATA:data},console};vm.createContext(context);
const source=fs.readFileSync(path.join(root,'app.js'),'utf8');
assert.equal((source.match(/\ninit\(\);/g)||[]).length,1);
vm.runInContext(source.replace('\ninit();','\nwindow.TEST={rate,elementary,indices,normalize,householdBasket,rebalanceShares,pct,pctSigned,miniUniverse,MINI_STATE};'),context);
const api=context.window.TEST,checks=[];
function close(name,a,b,tol=1e-11){assert.ok(Math.abs(a-b)<=tol,`${name}: ${a} versus ${b}`);checks.push({name,actual:a,expected:b});}
close('Annual headline',api.rate(170,164.9),3.09278350515465);
close('Monthly headline',api.rate(170,169.8),.1177856301531);
close('Adjusted monthly headline',api.rate(170,169.4),.35419126328217);
for(const price of [3,5.25,8]){
 api.MINI_STATE.milkSecondPrice=price;
 const groups=api.miniUniverse(),f=groups.reduce((s,g)=>s+g.share*g.index,0);
 const cost=groups.reduce((s,g)=>s+g.quantity*g.p0[0]*g.index,0);
 close(`Fixed quantities / weighted relatives, milk ${price}`,cost/100,f);
 close(`Updated shares sum to one, milk ${price}`,groups.reduce((s,g)=>s+g.expenditure0*g.index/cost,0),1);
 const milk=groups[0],r=milk.p1.map((p,i)=>p/milk.p0[i]);
 close(`Equal explicit weights recover Jevons, milk ${price}`,Math.exp(.5*Math.log(r[0])+.5*Math.log(r[1])),milk.index);
 assert.ok(api.elementary(milk.p0,milk.p1).carli>=milk.index-1e-12);
 const level=120*f;close(`Linked level recovers growth, milk ${price}`,api.rate(level,120),100*(f-1));
}
close('Laptop adjustment: assumed $80',api.rate(1000-80,900),2.22222222222222);
const factors=[[2,3,8],[1,4,2],[3,2,-1]].map(row=>row.reduce((s,x,i)=>s+[.2,.5,.3][i]*(1+x/100),0));
close('Synthetic regional aggregation',100*(factors.reduce((s,x,i)=>s+x*[.4,.35,.25][i],0)-1),3.025);
// Authentic link-month baseline and scenario normalization, including all-weight endpoints.
const official=api.householdBasket('official'),fictional=api.householdBasket('synthetic');
close('Official food starting weight',official.raw[0],16.93);
close('Official shelter starting weight',official.raw[1],28.30);
close('Official raw published total',official.publishedTotal,100.01);
close('Official normalized total',official.shares.reduce((a,b)=>a+b,0),1);
close('Fictional food share',fictional.shares[0],.22);
close('Full-precision official food shock',official.shares[0]*5,16.93/100.01*5);
for(const basket of [official,fictional])for(let k=0;k<8;k++)for(const share of [0,.5,1]){
 const w=api.rebalanceShares(basket.shares,k,share,basket.shares);
 close(`Scenario sum / ${basket===official?'official':'fictional'} / ${k} / ${share}`,w.reduce((a,b)=>a+b,0),1);
 close(`Selected share / ${k} / ${share}`,w[k],share);
 assert.ok(w.every(v=>Number.isFinite(v)&&v>=0));
}
const allFood=api.rebalanceShares(official.shares,0,1,official.shares),reopened=api.rebalanceShares(allFood,0,.2,official.shares);
close('Reopened food endpoint',reopened[0],.2);
close('Restored other proportions',reopened[1]/reopened[2],official.shares[1]/official.shares[2]);
assert.equal(api.pct(16.83),'16.8%');assert.equal(api.pct(3.092783505),'3.1%');assert.equal(api.pctSigned(.11778563),'+0.1%');assert.equal(api.pct(NaN),'Unavailable');assert.equal(api.pctSigned(-0),'0.0%');assert.equal(api.pctSigned(-.004),'0.0%');
const text=JSON.stringify({checks:checks.length,failures:0,results:checks},null,2);
fs.writeFileSync(path.join(root,'validation/math-checks.json'),text+'\n');console.log(`${checks.length} numerical checks passed.`);
