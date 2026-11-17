/* Development QA only. No browser/test dependency is loaded by the website.
 * Supply PLAYWRIGHT_MODULE if playwright is outside normal Node resolution.
 * python -m http.server 8000 from repository root, then node scripts/test-browser.mjs
 */
import {createRequire} from 'node:module';
import {mkdirSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
import {chapters} from '../js/storyboard.js';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.BIA_URL||'http://127.0.0.1:8000/2026-10-19-budget-impact-analysis-medication.html';
const output=process.env.BIA_QA_DIR?pathToFileURL(process.env.BIA_QA_DIR.replaceAll('\\','/')+'/'):new URL('../figures/qa/',import.meta.url);mkdirSync(output,{recursive:true});
const browser=await chromium.launch({headless:true,...(process.env.BIA_BROWSER_PATH?{executablePath:process.env.BIA_BROWSER_PATH}:{})});
const report=[];
try{
 for(const [name,width,height]of [['desktop',1440,1000],['laptop',1280,900],['tablet',768,1180],['mobile',390,844]]){
  const context=await browser.newContext({viewport:{width,height},reducedMotion:name==='desktop'?'no-preference':'reduce',acceptDownloads:true});const page=await context.newPage(),errors=[],external=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.route('**/*',route=>{if(new URL(route.request().url()).origin!==new URL(base).origin){external.push(route.request().url());return route.abort();}return route.continue();});
  await page.goto(base);await page.locator('#budget-cards strong').first().waitFor();
  assert.equal(await page.locator('[data-scene]').count(),14);
  assert.equal(await page.locator('#seek').getAttribute('max'),'56');
  assert.ok(await page.evaluate(()=>document.getElementById('scene-nav').compareDocumentPosition(document.getElementById('scene-art'))&Node.DOCUMENT_POSITION_FOLLOWING));
  assert.ok(await page.evaluate(()=>document.getElementById('replay').compareDocumentPosition(document.getElementById('scene-caption'))&Node.DOCUMENT_POSITION_FOLLOWING));
  assert.equal(await page.locator('.schedule-card').count(),2);
  assert.equal(await page.locator('.schedule-equations').filter({hasText:'$21,840/patient'}).count(),2);
  assert.ok(await page.evaluate(()=>{const seek=document.querySelector('.video-timeline').getBoundingClientRect(),controls=document.querySelector('.video-controls').getBoundingClientRect();return seek.top>=controls.bottom&&seek.width>controls.width*.95;}));
  await page.locator('#broad-context').screenshot({path:new URL(`${name}-context.png`,output).pathname.replace(/^\/([A-Za-z]:)/,'$1')});
  await page.locator('.schedule-primer').screenshot({path:new URL(`${name}-schedule.png`,output).pathname.replace(/^\/([A-Za-z]:)/,'$1'),style:'nav.nav{position:static!important}'});
  assert.equal(await page.locator('#budget-cards .metric').count(),6);assert.equal(await page.locator('#dose-calendar .week').count(),52);
  assert.match(await page.locator('#cohort-metrics').innerText(),/379/);assert.match(await page.locator('#population-result').innerText(),/121.74/);
  await page.screenshot({path:new URL(`${name}-top.png`,output).pathname.replace(/^\/([A-Za-z]:)/,'$1')});
  let sceneStart=0;
  for(let scene=0;scene<chapters.length;scene++){
   await page.locator(`[data-scene="${scene}"]`).click();
   await page.locator('#seek').fill(String(Number((sceneStart+chapters[scene][1]*.85).toFixed(1))));sceneStart+=chapters[scene][1];
   if(name==='desktop')await page.locator('#scene-art').screenshot({path:new URL(`browser-scene-${scene+1}.png`,output).pathname.replace(/^\/([A-Za-z]:)/,'$1')});
   else if([0,2,9,12,13].includes(scene))await page.locator('.video-canvas').screenshot({path:new URL(`${name}-scene-${scene+1}.png`,output).pathname.replace(/^\/([A-Za-z]:)/,'$1')});
  }
  assert.equal(await page.locator('h1').innerText(),'Budget Impact Analysis: A Health Economics Study for Prescription Drugs');
  assert.equal(await page.locator('.lead').innerText(),'An evaluation of the financial and demographic implications of listing new prescription drugs under a provincial publicly funded drug program.');
  assert.ok(await page.evaluate(()=>document.getElementById('historical-context').compareDocumentPosition(document.getElementById('references'))&Node.DOCUMENT_POSITION_FOLLOWING));
  await page.locator('[data-scene="13"]').click();assert.match(await page.locator('#scene-art').textContent(),/New 12,800/);
  if(name==='desktop'){
   await page.locator('#seek').scrollIntoViewIfNeeded();const track=await page.locator('#seek').boundingBox();
   await page.mouse.move(track.x+track.width*.2,track.y+track.height/2);await page.mouse.down();await page.mouse.move(track.x+track.width*.6,track.y+track.height/2,{steps:8});await page.mouse.up();
   assert.ok(Number(await page.locator('#seek').inputValue())>30&&Number(await page.locator('#seek').inputValue())<36,'pointer scrubbing updates timeline');
   const timed=async rate=>{if(await page.locator('#pause').isEnabled())await page.locator('#pause').click();await page.locator('#seek').fill('0');await page.locator('#speed').selectOption(rate);await page.locator('#play').click();await page.waitForTimeout(900);await page.locator('#pause').click();return Number(await page.locator('#seek').inputValue());};
   const slow=await timed('.5'),normal=await timed('1');assert.ok(normal/slow>1.6&&normal/slow<2.5,'1x runs about twice as fast as 0.5x');
  }
  for(const [time,index]of [[2.6,0],[7.7,2],[11.6,3],[18,4],[20.9,5],[27,6],[30.3,7],[36.9,9],[41.3,10],[46.7,11],[49.5,12],[55,13]]){
   await page.locator('#seek').fill(String(time));assert.match(await page.locator('#scene-title').innerText(),new RegExp(`^${index+1} / 14`));
   assert.match(await page.locator('#seek-label').innerText(),new RegExp(`^${time.toFixed(1)} / 56 s`));
  }
  await page.locator('[data-scene="12"]').click();assert.match(await page.locator('#scene-art').textContent(),/From Administrative Data to Policy Advice/);
  if(name==='desktop'){
   await page.locator('#speed').selectOption('2');await page.locator('#seek').fill('47.8');await page.locator('#play').click();await page.waitForTimeout(600);await page.locator('#pause').click();assert.match(await page.locator('#scene-title').innerText(),/^13 \/ 14/);
   const paused=await page.locator('#seek').inputValue();await page.waitForTimeout(150);assert.equal(await page.locator('#seek').inputValue(),paused);
   await page.locator('#seek').fill('55.7');await page.locator('#play').click();await page.waitForTimeout(400);assert.equal(await page.locator('#seek').inputValue(),'56');assert.ok(await page.locator('#play').isEnabled());
  }
  assert.match(await page.locator('#budget-cards').innerText(),/3,200/);await page.locator('#chart-budget-view').selectOption('live');assert.equal(await page.locator('#video-budget-view').inputValue(),'live');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'no document overflow');
  await page.locator('#preset').selectOption('fixed');assert.match(await page.locator('#switch-summary').innerText(),/36 → 48 → 60/);assert.match(await page.locator('#switch-summary').innerText(),/36 → 12 → 12/);
  const before=await page.locator('#budget-cards').innerText();await page.locator('#input-secPrice').fill('0');assert.notEqual(await page.locator('#budget-cards').innerText(),before);assert.match(await page.locator('#budget-cards').innerText(),/saving/i);
  await page.locator('#reset').click();assert.equal(await page.locator('#input-secPrice').inputValue(),'882.59');
  await page.locator('#definition').selectOption('strict');assert.match(await page.locator('#cohort-metrics').innerText(),/270 meet/);
  await page.locator('#use-cohort').click();assert.equal(Number(await page.locator('#input-n').inputValue()),Number((await page.locator('#cohort-metrics strong').innerText()).replaceAll(',','')));
  await page.locator('#reset').click();await page.locator('#frequency').selectOption('2');assert.match(await page.locator('#cost-cards').innerText(),/49,425.04/);
  await page.locator('#calendar-drug').selectOption('S2');assert.equal(await page.locator('#dose-calendar .dose').count(),28);
  await page.locator('#calendar-year').selectOption('1');assert.equal(await page.locator('#dose-calendar .dose').count(),26);
  await page.locator('#reset').click();await page.locator('#run-mc').click();await page.locator('#mc-status').filter({hasText:'draws complete'}).waitFor();const mc=await page.locator('#mc-metrics').innerText();await page.locator('#run-mc').click();await page.locator('#mc-status').filter({hasText:'draws complete'}).waitFor();assert.equal(await page.locator('#mc-metrics').innerText(),mc);assert.match(await page.locator('#mc-check').innerText(),/PASS/);
  const download=page.waitForEvent('download');await page.locator('#export').click();assert.equal((await download).suggestedFilename(),'bc-secukinumab-budget-impact.csv');
  await page.locator('#seek').fill('55');assert.match(await page.locator('#scene-title').innerText(),/14 \/ 14/);await page.locator('#replay').click();await page.locator('#pause').click();assert.match(await page.locator('#scene-title').innerText(),/1 \/ 14/);
  await page.locator('#budget').scrollIntoViewIfNeeded();await page.screenshot({path:new URL(`${name}-budget.png`,output).pathname.replace(/^\/([A-Za-z]:)/,'$1')});
  await page.locator('#reset').click();await page.locator('a.skip').focus();await page.keyboard.press('Tab');assert.ok(await page.evaluate(()=>document.activeElement!==document.body));
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);report.push({name,width,height,status:'PASS',consoleErrors:errors,externalRequests:external});await context.close();
 }
 writeFileSync(new URL('../data/browser-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(report);
}finally{await browser.close();}
