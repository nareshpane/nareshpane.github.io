import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {defaults,historical,population,schedule,courseCost,annualCosts,budget,payer,records,cohort,monteCarlo,decomposition,rng} from '../js/model.js';
let checks=0;const close=(a,b,msg)=>{assert.ok(Number.isFinite(a)&&Math.abs(a-b)<1e-6*Math.max(1,Math.abs(b)),`${msg}: ${a} != ${b}`);checks++;};const eq=(a,b)=>{assert.deepEqual(a,b);checks++;};
const data=records();eq(data.length,1350);eq(new Set(data.map(x=>x.id)).size,1350);
eq(['D','M','N'].map(s=>data.filter(x=>x.sources.includes(s)).length),[170,1240,250]);
const intersections=['DM','DN','MN','DMN'].map(s=>data.filter(x=>[...s].every(c=>x.sources.includes(c))).length);eq(intersections,[120,70,180,60]);eq(170+1240+250-120-70-180+60,1350);
eq(['broad','repeated','strict'].map(d=>cohort(data,d).selected.length),[1350,917,270]);
const all=['resident','adult','window','severe','failed','biologic','coverage','current'];const kept=cohort(data,'broad',all);eq(kept.selected.length,data.filter(x=>all.every(k=>x[k])).length);assert.ok(kept.stages.every((s,i)=>!i||s.n<=kept.stages[i-1].n));checks++;
close(population(historical).at(-1),121.7396480468,'historical top down');close(population({...historical,diagnosed:0}).at(-1),0,'zero diagnoses');
const fixed=budget({...defaults,growth:0});eq(fixed.rows.map(r=>r.sec),[36,48,60]);eq(fixed.rows.map(r=>r.newSwitch),[36,12,12]);
const costs=annualCosts(defaults);eq([costs.secFirst.units,costs.secMaintenance.units,costs.adaFirst.units,costs.adaMaintenance.units],[32,26,54,52]);
close(costs.secFirst.total,28242.88,'q4w first year');close(costs.secMaintenance.total,22947.34,'q4w maintenance');close(costs.adaFirst.total,25448.58,'adalimumab induction');close(costs.adaMaintenance.total,24506.04,'adalimumab maintenance');
const q2=annualCosts({...defaults,frequency:2});eq([q2.secFirst.units,q2.secMaintenance.units],[56,52]);close(q2.secFirst.total,49425.04,'q2w first year');close(q2.secMaintenance.total,45894.68,'q2w maintenance');
const mix=annualCosts({...defaults,escalation:.25});close(mix.secFirst.total,costs.secFirst.total*.75+q2.secFirst.total*.25,'regimen mixture');
close(courseCost('S',0,0,104,defaults).total,costs.secFirst.total+costs.secMaintenance.total,'no duplicate year boundary');
close(courseCost('S',0,0,52,defaults).total,courseCost('S',0,0,4,defaults).total+courseCost('S',0,4,52,defaults).total,'week 4 counted once');
const perSwitchFirst=costs.secFirst.total-costs.adaMaintenance.total;
close(fixed.rows[0].impact,36*perSwitchFirst,'manual Year 1');
close(fixed.rows[1].impact,36*(costs.secMaintenance.total-costs.adaMaintenance.total)+12*perSwitchFirst,'manual Year 2');
close(fixed.rows[2].impact,48*(costs.secMaintenance.total-costs.adaMaintenance.total)+12*perSwitchFirst,'manual Year 3');
const midway=budget({...defaults,growth:0,shares:[1,1,1],delay:26});close(midway.rows[0].secPY,60,'half-year exposure');close(midway.rows[0].newGross,120*(courseCost('A',-104,0,26,defaults).total+courseCost('S',26,26,52,defaults).total),'event-based midyear switch');
const noSwitch=budget({...defaults,shares:[0,0,0]});close(noSwitch.total,0,'identical scenarios');close(budget({...defaults,n:0}).total,0,'zero population');close(budget({...defaults,publicPay:0}).total,0,'zero public share');assert.ok(budget({...defaults,discount:1}).total<0);checks++;
const grown=budget();eq(grown.rows.map(r=>Math.round(r.n*10)/10),[120,132,145.2]);
const random=rng(81);for(let i=0;i<160;i++){
 const p={...defaults,n:Math.floor(random()*10000),growth:random()*.5,shares:[random(),random(),random()],secPrice:random()*4000,adaPrice:random()*2000,discount:random(),adaDiscount:random(),publicPay:random(),fee:random()*20,markup:random()*.2,admin:random()*40,delay:Math.floor(random()*27),mortality:random()*.2,loss:random()*.3,discontinuation:random(),back:random(),initiations:random()*200,frequency:random()>.5?2:4,escalation:random(),inflation:random()*.1};
 const b=budget(p);let prev=p.n;for(const r of b.rows){close(r.n,r.ada+r.sec,'stock reconciliation');close(prev+r.entrants-r.removed,r.n,'flow reconciliation');close(r.newCost-r.reference,r.impact,'annual difference');assert.ok(r.ada>=-1e-8&&r.sec>=-1e-8&&r.secPY>=0&&r.secPY<=r.sec+1e-6);checks++;prev=r.n;}
 close(b.total,b.rows.reduce((s,r)=>s+r.impact,0),'annual sum');close(b.total,b.newCost-b.reference,'scenario sum');close(decomposition(p).reduce((s,x)=>s+x.value,0),b.total,'waterfall sum');
 const charge=random()*100000,d=random()*3000,m=d+random()*4000,pr=random();const py=payer(charge,{deductible:d,maximum:m,privateRate:pr});close(py.public+py.private+py.patient,charge,'payer reconciliation');assert.ok(Object.values(py).every(v=>v>=0));checks++;
}
const cap=budget({...defaults,growth:0,shares:[1,1,1],back:1});close(cap.rows[1].sec,0,'returners protected from immediate reswitch');close(cap.rows[1].ada,120,'capacity-constrained target');
for(const charge of [0,100,650,1000,10000]){const p=payer(charge,{deductible:650,maximum:900});close(p.public+p.patient,charge,'public/patient example');assert.ok(p.patient<=900);checks++;}
close(payer(10000,{deductible:650,maximum:900,privateRate:.8}).patient,180,'residual private coverage');
const mc=monteCarlo(defaults,80,20261019);eq(mc,monteCarlo(defaults,80,20261019));assert.notDeepEqual(mc.values,monteCarlo(defaults,80,7).values);checks++;assert.ok(mc.p5<=mc.median&&mc.median<=mc.p95&&mc.positive>=0&&mc.positive<=1);checks++;
const report={checks,status:'PASS',historicalTopDown:population(historical).at(-1),syntheticUnion:data.length,caseDefinitionCounts:['broad','repeated','strict'].map(d=>({definition:d,count:cohort(data,d).selected.length})),retainedSynthetic:kept.selected.length,fixedStocks:fixed.rows.map(r=>r.sec),fixedNewSwitches:fixed.rows.map(r=>r.newSwitch),costs,q2Costs:q2,main:budget(),fixedNoGrowth:fixed,monteCarlo80:{median:mc.median,p5:mc.p5,p95:mc.p95,positive:mc.positive}};
writeFileSync(new URL('../data/numerical-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({status:'PASS',checks,topDown:report.historicalTopDown,union:data.length,retained:kept.selected.length,mainThreeYear:report.main.total,fixedThreeYear:fixed.total},null,2));
