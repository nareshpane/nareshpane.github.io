import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {chapters,duration,locateScene,sceneArt} from '../js/storyboard.js';
import {prevalenceEstimate,initialCriteria} from '../js/education.js';
import {budget,defaults,annualCosts,payer} from '../js/model.js';
import {policyIllustration,budgetView} from '../js/budget-view.js';
import {sources} from '../js/assumptions.js';
let checks=0;const equal=(a,b)=>{assert.deepEqual(a,b);checks++;};
equal(createHash('sha256').update(readFileSync(new URL('../js/model.js',import.meta.url))).digest('hex').toUpperCase(),'4AD3930F3EEB7BC11177BD47C97731935F0718EAEFFACC3A64C7D7886DDA91C0');
equal(chapters.length,13);equal(duration,51);equal(chapters[0][1],3.5);equal(locateScene(0).index,0);equal(locateScene(51).index,12);equal(locateScene(51).progress,1);
let start=0;chapters.forEach((c,i)=>{equal(locateScene(start).index,i);equal(locateScene(start+c[1]-.001).index,i);start+=c[1];});
equal(prevalenceEstimate(1000,5800000),58000);equal(prevalenceEstimate(0,5800000),0);equal(prevalenceEstimate(100000,5800000),5800000);
const p={age:35,stage:'II',antibiotic:true,an:3,areas:2,sinus:1,dermatologist:true};equal(Object.values(initialCriteria(p)).every(Boolean),true);
for(const change of [{age:17},{stage:'I'},{antibiotic:false},{an:2},{areas:1},{sinus:0},{dermatologist:false}])equal(Object.values(initialCriteria({...p,...change})).every(Boolean),false);
const b=budget(),round=n=>Math.round(n/1000);equal([round(b.reference),round(b.newCost),round(b.total)],[9758,9886,128]);equal(b.rows.map(r=>Math.round(r.sec)),[36,53,73]);
assert.ok(Math.abs(b.total-128236.64828571305)<1e-8);checks++;
equal(annualCosts(defaults).secFirst.total,28242.88);
const full=payer(28242.88,{deductible:0,maximum:0});equal(full.public,28242.88);equal(full.patient,0);
for(const c of [0,400,650,1000,20000]){const v=payer(c,{deductible:650,maximum:900,copay:.25,privateRate:.5});assert.ok(Math.abs(v.public+v.private+v.patient-c)<1e-8);checks++;}
for(let i=0;i<13;i++){const s=sceneArt(i,b,.65);assert.ok(s.startsWith('<svg')&&s.includes('1120 530')&&!s.includes('NaN')&&!s.includes('undefined'));checks++;}
const changed=budget({...defaults,n:240});assert.notEqual(sceneArt(12,b,.65,'live'),sceneArt(12,changed,.65,'live'));checks++;
const html=readFileSync(new URL('../2026-10-19-budget-impact-analysis-medication.html',import.meta.url),'utf8');
equal((html.match(/<h1>/g)||[]).length,1);assert.ok(html.includes('<h1>Budget Impact Analysis: A Health Economics Study for Prescription Drugs</h1>'));checks++;
assert.ok(html.indexOf('id="scene-nav"')<html.indexOf('id="scene-art"')&&html.indexOf('id="scene-caption"')>html.indexOf('id="replay"'));checks++;
assert.ok(!html.includes('mechanism-toggle')&&!html.includes('IL-17A signal'));checks++;
for(const match of html.matchAll(/data-source="([^"]+)"/g)){assert.ok(match[1] in sources,'Missing source '+match[1]);checks++;}
const geography=readFileSync(new URL('../figures/canada-provinces.svg',import.meta.url),'utf8');equal([...geography.matchAll(/<g id="([A-Z]+)"/g)].map(m=>m[1]).sort(),['AB','BC','MB','NB','NL','NS','NT','NU','ON','PE','QC','SK','YT']);
equal(policyIllustration.rows.map(r=>r.impact),[800000,1000000,1400000]);
equal(policyIllustration.rows.reduce((sum,r)=>sum+r.impact,0),3200000);
equal(policyIllustration.newCost-policyIllustration.reference,policyIllustration.total);
for(const row of policyIllustration.rows)equal(row.newCost-row.reference,row.impact);
equal(budgetView(b).total,3200000);equal(budgetView(b,'live'),b);
assert.ok(sceneArt(10,b,.65).includes('3,200')&&sceneArt(12,b,.65).includes('12,800'));checks++;
assert.ok(sceneArt(11,b,.65).includes('price renegotiation'));checks++;
assert.ok(html.indexOf('id="broad-context"')>html.indexOf('</header>')&&html.indexOf('id="broad-context"')<html.indexOf('id="journey"'));checks++;
assert.ok(html.indexOf('id="historical-context"')>html.indexOf('id="limitations"')&&html.indexOf('id="historical-context"')<html.indexOf('id="references"'));checks++;
const report={status:'PASS',checks,engineByteIdentical:true,videoDuration:duration,videoScenes:13,aggregateUnit:'CAD $000; zero decimals; underlying CAD retained',baselineThreeYearCAD:b.total};writeFileSync(new URL('../data/revision-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(report);
