/* Run with the existing Playwright package in NODE_PATH and the repository served on :8765. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const S=require('./story.js'),M=require('./math.js');
const url=process.env.HYPERGRAPH_TEST_URL||'http://127.0.0.1:8765/research/graph-theory/2026-04-21-hypergraphs-visualization.html';
const OUT=path.join(__dirname,'qa');fs.mkdirSync(OUT,{recursive:true});
(async()=>{
  const browser=await chromium.launch({headless:true,channel:'msedge'});
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],checks=[],layout=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);
  const seek=async t=>{await page.locator('#timeline').fill(String(t));await page.locator('#timeline').dispatchEvent('input');};
  const click=async id=>page.locator('#'+id).click();
  assert.equal(await page.locator('.animation-heading').innerText(),'Animation Video for Hypergraphs');
  assert.equal(await page.locator('#timeline').getAttribute('max'),'18');
  assert.deepEqual(await page.locator('#intro-speed option').evaluateAll(es=>es.map(e=>e.value)),['0.5','1','1.5','2']);
  await page.waitForTimeout(150);assert.equal(await page.locator('#timeline').inputValue(),'0');
  const times=[0,1.5,3.5,5.5,8,10.5,11.2,11.8,12.5,13,13.7,14.4,15,15.5,18];
  for(const t of [...times,...times.toReversed()]){
    await seek(t);const f=S.at(t,true);
    const actual=await page.evaluate(()=>{
      const q=s=>[...document.querySelectorAll(s)];
      return {stage:+document.querySelector('#story').dataset.stage,edges:q('#intro-hyper .edge-node').filter(e=>+e.style.opacity>0).length,
        spokes:q('#intro-hyper .incidence-link').filter(e=>+e.style.opacity>0).length,
        pairs:q('#intro-graph [data-pair]').filter(e=>+e.style.opacity>0).map(e=>e.dataset.pair),
        matrix:!document.querySelector('#intro-matrix-panel').hidden,counter:!document.querySelector('#intro-counterexample').hidden,
        active:q('#intro-matrix thead .active').map(e=>+e.dataset.column)};
    });
    assert.deepEqual(actual,{stage:f.stageIndex+1,edges:f.h.edges.length,spokes:f.total,pairs:f.pairs.map(e=>e.join(',')),matrix:!!f.stage.matrix,counter:!!f.stage.counterexample,active:f.stage.matrix&&f.active!==null?[f.active]:[]},'Synchronized state at '+t);
  }
  assert.deepEqual(await page.locator('#intro-matrix td').allTextContents(),M.incidence(M.example()).flat().map(String));
  for(let j=0;j<4;j++){
    const colours=await page.evaluate(j=>['#intro-hyper .edge-node[data-edge="'+j+'"] rect','#builder-diagram .edge-node[data-edge="'+j+'"] rect'].map(s=>document.querySelector(s).getAttribute('stroke')).concat(document.querySelector('#intro-matrix thead [data-column="'+j+'"]').style.getPropertyValue('--edge-colour'),document.querySelectorAll('#edge-list button')[j].style.getPropertyValue('--edge-colour')),j);
    assert.ok(colours.every(c=>c===S.colours[j]));
  }
  const positions=await page.evaluate(()=>['intro-graph','intro-hyper','builder-diagram'].map(id=>[...document.querySelectorAll('#'+id+' .vertex-node circle')].map(c=>[c.getAttribute('cx'),c.getAttribute('cy')])));
  assert.deepEqual(positions[0],positions[1]);assert.deepEqual(positions[1],positions[2]);
  await click('intro-restart');for(let j=1;j<8;j++){await click('intro-next');assert.equal(Number(await page.locator('#timeline').inputValue()),S.stages[j].start);}
  for(let j=6;j>=0;j--){await click('intro-prev');assert.equal(Number(await page.locator('#timeline').inputValue()),S.stages[j].start);}
  checks.push('Every stage and column scan synchronized in both directions; exact 24 matrix entries, stable shared positions, edge colours, previous/next stages');
  await page.locator('#membership-flow button').focus();await page.keyboard.press('Enter');
  assert.equal(await page.locator('#matrix td.focus').count(),1);
  assert.equal(await page.locator('#matrix td.focus button').getAttribute('data-index'),'2,2');
  assert.equal(await page.locator('#incidence-graph .incidence-link[stroke-width="5"]').count(),1);
  assert.equal(await page.locator('#incidence-graph .incidence-link[stroke-width="5"]').getAttribute('data-vertex'),'C');
  assert.equal(await page.evaluate(()=>document.activeElement.parentElement.id),'membership-flow');
  assert.equal(await page.locator('#incidence-graph .vertex-node circle[fill="#e6efb5"]').count(),1);
  await click('reset');await page.locator('#vertex-select [data-v=A]').click();assert.ok((await page.locator('#selection').innerText()).includes('cardinality 1'));await click('add-edge');await seek(18);
  assert.equal(await page.locator('#story').getAttribute('data-edges'),'4');
  assert.ok((await page.locator('#builder-stats').innerText()).includes('Size 5'));await click('reset');
  checks.push('C in e3 highlights exactly its membership link and matrix entry; keyboard focus retained; builder and animation state independent');

  // Measure one real, unaccelerated run from user activation to the final frame.
  await click('intro-restart');await page.selectOption('#intro-speed','1');
  await page.evaluate(()=>{document.querySelector('#intro-play').addEventListener('click',()=>window.started=performance.now(),{once:true,capture:true});window.seenStages=new Set();window.endTime=null;window.stageObserver=new MutationObserver(()=>{const s=document.querySelector('#story');window.seenStages.add(+s.dataset.stage);if(+s.dataset.time===18&&window.endTime===null)window.endTime=performance.now();});window.stageObserver.observe(document.querySelector('#story'),{attributes:true,attributeFilter:['data-time']});});
  await click('intro-play');
  await page.waitForFunction(()=>document.querySelector('#story').dataset.time==='18.000',{},{timeout:23000});
  const timing=await page.evaluate(()=>({elapsedSeconds:(window.endTime-window.started)/1000,stages:[...window.seenStages]}));
  assert.ok(timing.elapsedSeconds>=18&&timing.elapsedSeconds<20,JSON.stringify(timing));assert.deepEqual(timing.stages,[1,2,3,4,5,6,7,8]);
  checks.push('Real 1x playback completes all eight stages in '+timing.elapsedSeconds.toFixed(3)+' seconds from click to final frame');

  // Virtual animation time makes all four rates reproducible without long real waits.
  const clockPage=await browser.newPage({viewport:{width:1440,height:1000}});
  await clockPage.clock.install();await clockPage.goto(url);await clockPage.clock.pauseAt(new Date(Date.now()+10000));
  for(const rate of [.5,1,1.5,2]){
    await clockPage.locator('#intro-restart').click();await clockPage.selectOption('#intro-speed',String(rate));await clockPage.locator('#intro-play').click();
    await clockPage.clock.runFor(1000);
    const t=Number(await clockPage.locator('#timeline').inputValue());assert.ok(Math.abs(t-rate)<.08,`${rate}x after one second: ${t}`);
    await clockPage.locator('#intro-play').click();await clockPage.clock.runFor(500);assert.equal(Number(await clockPage.locator('#timeline').inputValue()),t);
  }
  await clockPage.close();
  await click('intro-restart');await click('intro-play');await page.locator('#random').scrollIntoViewIfNeeded();await page.waitForTimeout(100);assert.equal(await page.locator('#intro-play').getAttribute('aria-pressed'),'false');
  await click('intro-play');await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'));});assert.equal(await page.locator('#intro-play').getAttribute('aria-pressed'),'false');await page.evaluate(()=>delete document.hidden);
  await page.emulateMedia({reducedMotion:'reduce'});await seek(1.5);assert.equal(await page.locator('#intro-hyper .edge-node[data-edge="0"]').evaluate(e=>e.style.opacity),'1');await page.emulateMedia({reducedMotion:'no-preference'});
  checks.push('Four playback rates and pause verified with virtual clock; offscreen pause; simulated hidden-document event; reduced-motion reveals complete step');

  await seek(18);
  for(const width of [1920,1440,1366,1024,768,390]){
    await page.setViewportSize({width,height:1000});
    const metrics=await page.evaluate(()=>{
      const q=s=>[...document.querySelectorAll(s)];
      const clipped=q('.lesson p,.lesson figcaption,.formula,.story-equation').filter(e=>e.getClientRects().length&&e.scrollWidth>e.clientWidth+2).map(e=>e.textContent.slice(0,90));
      const svgClips=q('svg text').filter(e=>e.getClientRects().length).filter(e=>{const b=e.getBBox(),v=e.ownerSVGElement.viewBox.baseVal;return b.x<v.x-1||b.y<v.y-1||b.x+b.width>v.x+v.width+1||b.y+b.height>v.y+v.height+1;}).map(e=>e.textContent);
      return {viewport:innerWidth,document:document.documentElement.scrollWidth,clipped,svgClips,paragraphMaxWidth:getComputedStyle(document.querySelector('#definitions>p')).maxWidth};
    });
    assert.ok(metrics.document<=width,'Page overflow '+width);assert.deepEqual(metrics.clipped,[],'Text clipping '+width);assert.deepEqual(metrics.svgClips,[],'SVG label clipping '+width);assert.equal(metrics.paragraphMaxWidth,'none');layout.push(metrics);
    if([1440,768,390].includes(width)){
      await seek(11.8);await page.locator('#story').screenshot({path:path.join(OUT,`refinement-story-${width}.png`)});
      await page.locator('#incidence').screenshot({path:path.join(OUT,`refinement-incidence-${width}.png`)});await seek(18);
    }
  }
  await page.setViewportSize({width:1440,height:1000});await seek(18);await page.locator('#intro-counterexample').screenshot({path:path.join(OUT,'refinement-counterexample.png')});await page.locator('#intro-examples').screenshot({path:path.join(OUT,'refinement-examples.png')});
  assert.deepEqual(errors,[]);checks.push('Six requested viewports: no document overflow, paragraph/formula clipping, or SVG label clipping');
  const report={status:'PASS',checks,timing,layout,errors,limitations:['Installed Edge only; hidden-tab handler tested by simulated visibility event.']};
  fs.writeFileSync(path.join(OUT,'refinement-animation-audit.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
