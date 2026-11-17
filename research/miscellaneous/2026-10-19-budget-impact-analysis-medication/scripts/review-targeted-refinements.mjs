/* Local, development-only measurements and screenshot review. */
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {writeFileSync,mkdirSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url),{chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const phase=process.env.BIA_REVIEW_PHASE||'after';
const out=new URL('../figures/qa/targeted-refinements/',import.meta.url);mkdirSync(out,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:process.env.BIA_BROWSER_PATH});
const reports=[];
try{
 for(const [name,width,height]of [['desktop',1440,1000],['mobile',390,844]]){
  const context=await browser.newContext({viewport:{width,height},reducedMotion:'reduce'});
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto(process.env.BIA_URL||'http://127.0.0.1:8000/2026-10-19-budget-impact-analysis-medication.html');await page.locator('#budget-cards strong').first().waitFor();
  const geometry=await page.evaluate(()=>{const rect=e=>{const r=e.getBoundingClientRect();return {width:r.width,height:r.height};};return {context:rect(document.querySelector('#broad-context')),cards:[...document.querySelectorAll('.severity-grid .panel')].map(rect),severityIllustrations:[...document.querySelectorAll('.severity-grid .chart')].map(rect),venn:rect(document.querySelector('#prevalence-overlap')),vennShape:rect(document.querySelector('#prevalence-overlap .chart')),metadata:rect(document.querySelector('.provenance')),viewport:innerWidth,documentWidth:document.documentElement.scrollWidth};});
  await page.screenshot({path:fileURLToPath(new URL(`${phase}-${name}-top.png`,out))});
  for(const [label,selector]of [['context','#broad-context'],['severity','.severity-grid'],['venn','#prevalence-overlap'],['metadata','.provenance'],['national','#national'],['special-authority','#special-authority'],['databases','#databases'],['glossary','#terminology'],['approval','#approval-story']]){
   if(!await page.locator(selector).count())continue;
   if(label==='glossary')await page.locator(selector).evaluate(e=>e.open=true);
   await page.locator(selector).screenshot({path:fileURLToPath(new URL(`${phase}-${name}-${label}.png`,out)),style:'nav.nav{position:static!important}'});
  }
  if(phase==='after'){for(const scene of [3,5,6]){await page.locator(`[data-scene="${scene}"]`).click();await page.locator('.video-canvas').screenshot({path:fileURLToPath(new URL(`${phase}-${name}-scene-${scene+1}.png`,out))});}}
  const functionalChecks=[];if(phase==='after'){await page.locator('#sa-age').fill('17');assert.match(await page.locator('#sa-result').innerText(),/not satisfied/);await page.locator('#sa-age').fill('35');assert.match(await page.locator('#sa-result').innerText(),/^All selected/);functionalChecks.push('Special Authority age response and restoration');await page.locator('#prevalence-rate').fill('2000');assert.match(await page.locator('#prevalence-equation').innerText(),/116,000/);await page.locator('#prevalence-rate').fill('1000');assert.match(await page.locator('#prevalence-equation').innerText(),/58,000/);functionalChecks.push('Prevalence input response and restoration');}
  reports.push({name,geometry,consoleErrors:errors,functionalChecks});await context.close();
 }
 writeFileSync(new URL(`../data/targeted-${phase}-measurements.json`,import.meta.url),JSON.stringify(reports,null,2)+'\n');console.log(JSON.stringify(reports,null,2));
}finally{await browser.close();}
