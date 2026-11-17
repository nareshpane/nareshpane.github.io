/* Development-only browser review. Uses an existing Playwright installation. */
import {createRequire} from 'node:module';
import {writeFileSync,mkdirSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({headless:true,executablePath:process.env.BIA_BROWSER_PATH});
const base=process.env.BIA_URL||'http://127.0.0.1:8000/2026-10-19-budget-impact-analysis-medication.html';
const out=new URL('../figures/qa/illustrations/',import.meta.url);mkdirSync(out,{recursive:true});
const report=[];
try{
 for(const [name,width,height] of [['desktop',1440,1000],['tablet',820,1180],['mobile',390,844]]){
  const context=await browser.newContext({viewport:{width,height},reducedMotion:'reduce'});
  const page=await context.newPage(),errors=[],failed=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  page.on('response',r=>{if(r.status()>=400)failed.push([r.url(),r.status()]);});
  await page.goto(base);await page.locator('#budget-cards strong').first().waitFor();
  await page.evaluate(async()=>{const im=new Image();im.src='figures/illustrations/professional-cast.png';await im.decode();});
  for(let i=0;i<13;i++){
   await page.locator(`[data-scene="${i}"]`).click();
   // Reduced-motion presentation uses a stable mid-scene frame.
   await page.locator('.video-canvas').screenshot({path:fileURLToPath(new URL(`${name}-scene-${i+1}.png`,out))});
   if(name==='mobile'){
    await page.locator('.video-canvas').evaluate(e=>{e.scrollLeft=e.scrollWidth-e.clientWidth;});
    await page.locator('.video-canvas').screenshot({path:fileURLToPath(new URL(`${name}-scene-${i+1}-right.png`,out))});
    await page.locator('.video-canvas').evaluate(e=>{if(e.scrollLeft===0)throw new Error('Diagram did not scroll');e.scrollLeft=0;});
   }
  }
  for(const id of ['severity-clinician','clinical-tree','prevalence-methods','database-grid','sa-illustration','pharmacy-art','dose-equation','limitations','approval-story','care-art','patient-icons','linkage-art','broad-context','historical-context','budget']){
   if(await page.locator('#'+id).count())await page.locator('#'+id).screenshot({path:fileURLToPath(new URL(`${name}-${id}.png`,out))});
  }
  const overflow=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}));
  report.push({name,errors,failed,overflow});await context.close();
 }
 writeFileSync(new URL('../data/illustration-browser-review.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}finally{await browser.close();}
