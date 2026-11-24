'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.CPI_PLAYWRIGHT || 'playwright-core');
const root = path.resolve(__dirname,'../..');
const project = path.basename(path.resolve(__dirname,'..'));
const temporary = process.env.CPI_SCREENSHOTS || fs.mkdtempSync(path.join(os.tmpdir(),'cpi-browser-'));
fs.mkdirSync(temporary,{recursive:true});
const mime={'.html':'text/html; charset=utf-8','.js':'application/javascript','.css':'text/css',
  '.svg':'image/svg+xml','.woff2':'font/woff2','.csv':'text/csv','.json':'application/json'};
const server=http.createServer((request,response)=>{
  const name=path.resolve(root,'.'+decodeURIComponent(new URL(request.url,'http://127.0.0.1').pathname));
  if(!name.startsWith(root+path.sep)||!fs.existsSync(name)||!fs.statSync(name).isFile()){
    response.writeHead(404);response.end();return;
  }
  response.writeHead(200,{'Content-Type':mime[path.extname(name)]||'application/octet-stream'});
  fs.createReadStream(name).pipe(response);
});
const viewports=[['wide',1920,1080],['desktop',1536,864],['desktop-1440',1440,900],['laptop',1366,768],['tablet',1024,768],['small-phone',320,740],['mobile',390,844]];
async function checkOpening(page, width) {
  await page.evaluate(()=>document.fonts.ready);
  const header=await page.evaluate(()=>{
    const title=document.querySelector('.hero h1'), card=document.querySelector('.metadata');
    const lineRects=el=>{
      const range=document.createRange();
      range.selectNodeContents(el);
      return [...range.getClientRects()];
    };
    return {
      titleLines:new Set([...title.children].flatMap(lineRects).filter(rect=>rect.width>0).map(rect=>Math.round(rect.top))).size,
      headRight:title.getBoundingClientRect().right, cardLeft:card.getBoundingClientRect().left,
      cardRight:card.getBoundingClientRect().right, cardWidth:card.getBoundingClientRect().width,
      rows:[...card.children].map(row=>({
        text:row.textContent,
        fits:[...row.children].every(cell=>cell.scrollWidth<=cell.clientWidth+1),
        valueLines:lineRects(row.lastElementChild).length
      }))
    };
  });
  assert.equal(header.rows.length,4);
  assert.deepEqual(header.rows.map(r=>r.text),[
    '🗓️Created:November 24, 2026','🧰Harness:Codex on Windows','📑Model:GPT-6.1 Sol','🧠Reasoning:High'
  ]);
  assert(header.rows.every(row=>row.fits&&row.valueLines===1),width+' metadata '+JSON.stringify(header.rows));
  assert(header.cardRight<=width);
  if(width>=1366){
    assert.equal(header.titleLines,2,width+' title lines');
    assert(header.headRight<header.cardLeft);
    assert(header.cardWidth>=315);
  }
  const geometry=await page.evaluate(()=>{
    const seek=document.querySelector('#film-seek');
    const duration=+seek.max, count=10;
    const rect=el=>{
      const {top,bottom,height}=el.getBoundingClientRect();
      return {top,bottom,height};
    };
    return Array.from({length:count},(_,stage)=>{
      seek.value=stage*duration/count;
      seek.dispatchEvent(new Event('input',{bubbles:true}));
      const panels=[...document.querySelectorAll('.map-pair .country')].map(panel=>({
        height:rect(panel).height,
        heading:rect(panel.querySelector('h3')),
        map:rect(panel.querySelector('svg')),
        badge:rect(panel.querySelector('.film-label')),
        caption:rect(panel.querySelector('.caption')),
        label:panel.querySelector('.film-label span:last-child').textContent,
        markerCount:panel.querySelectorAll('svg g.map-feature title').length,
        lineCount:panel.querySelectorAll('svg g.map-feature path[stroke-dasharray]').length,
        svgLabels:panel.querySelectorAll('svg text').length
      }));
      return {panels, step:document.querySelector('#film-status').textContent,
        height:rect(document.querySelector('.film')).height};
    });
  });
  const near=(a,b)=>Math.abs(a-b)<.6;
  geometry.forEach(({height,panels,step},i)=>{
    const [ca,us]=panels;
    assert.match(step,new RegExp('^Step '+(i+1)+' of 10'));
    assert.equal(ca.label,us.label);
    assert(near(height,geometry[0].height),width+' film shifts at stage '+(i+1));
    assert(near(ca.height,us.height)&&near(ca.caption.top,us.caption.top)&&near(ca.badge.top,us.badge.top),
      width+' panel alignment at stage '+(i+1)+' '+JSON.stringify(panels));
    for(const panel of panels){
      assert.equal(panel.svgLabels,0);
      assert(panel.markerCount>0);
      assert.equal(panel.lineCount>0,i>=2&&i<=7);
      assert(panel.heading.bottom<=panel.map.top&&panel.map.bottom<=panel.badge.top+.2&&panel.badge.bottom<=panel.caption.top+.2);
    }
  });
}
async function checkGeography(page) {
  const section=page.locator('#s18-geography');
  assert.equal(await page.locator('.intro > p').count(),3);
  assert.equal(await page.locator('#s18-geography + #s18 .section-number').textContent(),'SECTION 19');
  assert.equal(await section.locator('h3[id^=s18-g]').count(),8);
  assert.equal(await section.locator('#geo-comparison tbody tr').count(),9);
  assert.equal(await section.locator('#geo-comparison td a').count(),0);
  const maps=page.locator('#lab-geo-maps');
  const ca=maps.locator('[data-country=CA]');
  const us=maps.locator('[data-country=US]');
  assert.equal(await ca.locator('.geo-boundaries [data-feature]').count(),13);
  assert.equal(await us.locator('.geo-boundaries [data-feature]').count(),51);
  for(const layer of ['strata','publication','weights','rates','locations','admin'])
    await ca.locator('[name=CAlayer]').selectOption(layer);
  await ca.locator('[name=CAstate]').selectOption('NU');
  assert.match(await ca.locator('[data-detail]').textContent(),/Iqaluit only/);
  await ca.locator('[name=CAlayer]').selectOption('rates');
  const northern=await ca.locator('.geo-boundaries [data-feature=NU] path').getAttribute('fill');
  const ontario=await ca.locator('.geo-boundaries [data-feature=ON] path').getAttribute('fill');
  assert.notEqual(northern,ontario);
  await ca.locator('[name=CAstate]').selectOption('ON');
  await us.locator('[name=USlayer]').selectOption('psus');
  assert.equal(await us.locator('.geo-markers [data-feature]').count(),75);
  await us.locator('[name=USlayer]').selectOption('self');
  assert.equal(await us.locator('.geo-markers [data-feature]').count(),23);
  await us.locator('[name=USlayer]').selectOption('nonself');
  assert.equal(await us.locator('.geo-markers [data-feature]').count(),52);
  await us.locator('[name=USlayer]').selectOption('psus');
  await us.locator('.geo-markers [data-feature=S12A]').press('Enter');
  assert.match(await us.locator('[data-detail]').textContent(),/NJ:.*NY:.*PA: Pike/);
  assert.equal(await us.locator('.geo-counties path').count(),25);
  await us.locator('[name=USlayer]').selectOption('index');
  await us.locator('[name=psu]').selectOption('N12C');
  assert.match(await us.locator('[data-detail]').textContent(),/N120 \(4 sample PSUs\)/);
  for(const layer of ['regions','divisions','rates','admin'])
    await us.locator('[name=USlayer]').selectOption(layer);
  await us.locator('[name=USstate]').selectOption('DC');
  assert.match(await us.locator('[data-detail]').textContent(),/South region.*South Atlantic division/);
  assert.match(await us.locator('[data-detail]').textContent(),/no statewide CPI-U/);
  await us.locator('[name=USstate]').selectOption('NY');
  await us.locator('[name=USlayer]').selectOption('psus');

  const empirical=page.locator('#lab-geo-empirical');
  await empirical.locator('[data-episode="2022-06"]').click();
  assert.match(await empirical.locator('[data-empirical=CA] [data-points]').textContent(),/Prince Edward Island63\.4%/);
  assert.match(await empirical.locator('[data-empirical=US] [data-points]').textContent(),/Midwest64\.4%/);
  await empirical.locator('[name=CAgroup]').selectOption('provinces');
  assert.equal(await empirical.locator('[data-empirical=CA] .geo-dot-row').count(),10);
  await empirical.locator('[name=USgroup]').selectOption('divisions');
  assert.equal(await empirical.locator('[data-empirical=US] .geo-dot-row').count(),9);
  await empirical.locator('[name=USgroup]').selectOption('metros');
  assert.equal(await empirical.locator('[data-empirical=US] .geo-dot-row').count(),7);
  await empirical.locator('[name=CAgroup]').selectOption('all');
  await empirical.locator('[name=USgroup]').selectOption('all');
  await empirical.locator('[data-episode="2026-10"]').click();
  await empirical.locator('[name=period]').selectOption('1');
  assert.match(await empirical.locator('[data-empirical=US] [data-national]').textContent(),/Unavailable/);
  await empirical.locator('[name=period]').selectOption('0');
  await empirical.locator('[data-empirical=CA] [data-series=BC]').click();
  assert.equal(await ca.locator('[name=CAstate]').inputValue(),'BC');

  const weights=page.locator('#lab-geo-weights');
  const wtUS=weights.locator('[data-weight=US]');
  assert.equal(await wtUS.locator('[name=adjust]').isEnabled(),false);
  await wtUS.locator('[name=mode]').selectOption('custom');
  assert.equal(await wtUS.locator('[name=adjust]').isEnabled(),true);
  await wtUS.locator('[name=adjust]').evaluate(el=>{el.value=0;el.dispatchEvent(new Event('input',{bubbles:true}));});
  assert.match(await wtUS.locator('[data-weight-result]').textContent(),/changes the national approximation by 0\.000 pp/);
  assert(await wtUS.locator('.geo-shocked[data-map-area="0100"]').count()>1);
  await wtUS.locator('[name=spill]').evaluate(el=>{el.value=50;el.dispatchEvent(new Event('input',{bubbles:true}));});
  await wtUS.locator('[name=shock]').evaluate(el=>{el.value=0;el.dispatchEvent(new Event('input',{bubbles:true}));});
  assert.match(await wtUS.locator('[data-weight-result]').textContent(),/0\.000 ppShock effect/);
  for(const mode of ['equal','population','official'])await wtUS.locator('[name=mode]').selectOption(mode);
  await wtUS.locator('[data-weight-reset]').click();
  await weights.locator('[data-weight=CA] [data-weight-reset]').click();
  const result=await page.evaluate(()=>{
    const {D,computeWeights,contributionRows}=CPIGeography;
    const near=(a,b)=>Math.abs(a-b)<1e-9;
    return ['CA','US'].every(country=>{
      const area=D.areas[country][0];
      return ['official','equal','population','custom'].every(mode=>{
        const config={area,adjust:37,shock:4,spill:0};
        const w=computeWeights(country,mode,area,config.adjust);
        const rows=contributionRows(country,config,mode);
        const zero=contributionRows(country,{...config,shock:0},mode);
        const target=rows.find(r=>r.code===area);
        const aggregate=rows.reduce((s,r)=>s+r.result,0);
        const baseline=rows.reduce((s,r)=>s+r.contribution,0);
        return near(Study.sum(w),1)&&near(Study.sum(w.map(x=>x*2.5)),2.5)&&
          zero.every(r=>r.effect===0)&&near(target.effect,4*target.weight)&&
          near(aggregate,baseline+target.effect)&&
          contributionRows(country,{...config,adjust:0},'custom').find(r=>r.code===area).effect===0;
      });
    });
  });
  assert(result);

  const spatial=page.locator('#lab-geo-spatial');
  const truth=await spatial.getAttribute('data-population-truth');
  await spatial.locator('[name=design]').selectOption('optimal');
  await spatial.locator('[name=comparison]').selectOption('spend');
  await spatial.locator('[name=resource]').selectOption('budget');
  const shot=await page.evaluate(()=>{
    const cfg={...CPISpatial.defaults,resource:'budget'};
    const pop=CPISpatial.makePopulation(cfg,99);
    const a=CPISpatial.draw(pop,cfg,'strat',CPISampling.rng(100));
    const b=CPISpatial.draw(pop,cfg,'strat',CPISampling.rng(100));
    return a.estimate===b.estimate&&a.cost<=cfg.quotes&&a.count>0&&
      CPISpatial.draw(pop,{...cfg,resource:'quotes'},'optimal',CPISampling.rng(100)).count===cfg.quotes;
  });
  assert(shot);
  await spatial.locator('[data-redraw]').click();
  assert.equal(await spatial.getAttribute('data-population-truth'),truth);
  assert.equal(await spatial.getAttribute('data-draw'),'1');
  await spatial.locator('[name=regions]').evaluate(el=>{el.value=16;el.dispatchEvent(new Event('input',{bubbles:true}));});
  assert.equal(await spatial.getAttribute('data-population-truth'),truth);
  await spatial.locator('[data-regenerate]').click();
  assert.notEqual(await spatial.getAttribute('data-population-truth'),truth);
  await spatial.locator('[data-sim-reset]').click();
  assert.equal(await spatial.getAttribute('data-population-truth'),truth);
  for(const field of await spatial.locator('input[type=range][name]').all()){
    await field.evaluate(el=>{
      const start=el.value;
      el.value=el.max;el.dispatchEvent(new Event('input',{bubbles:true}));
      el.value=start;el.dispatchEvent(new Event('input',{bubbles:true}));
    });
  }
  assert.equal(await spatial.getAttribute('data-population-truth'),truth);
  const matrix=page.locator('#lab-geo-matrix');
  await matrix.locator('[data-cell="22"]').click();
  assert.match(await matrix.locator('[data-cell-detail]').textContent(),/no matched quote/);
  await matrix.locator('[data-cell="00"]').click();
  assert.equal(await matrix.locator('[data-cell-detail] tbody tr').count(),3);
  await page.locator('#lab-geo-architecture .us [data-step="3"]').click();
  assert.match(await page.locator('#lab-geo-architecture .us .geo-readout').textContent(),/75 design PSUs feed 32/);
}
async function run(){
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const url='http://127.0.0.1:'+server.address().port+'/'+project+'.html';
  let browser;
  try {
    browser=await chromium.launch({executablePath:process.env.CPI_EDGE||
      'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
    const context=await browser.newContext({permissions:['clipboard-read','clipboard-write'],reducedMotion:'reduce'});
    const page=await context.newPage();
    const errors=[],failed=[],bad=[];
    page.on('pageerror',error=>errors.push(error.stack || error.message));
    page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
    page.on('requestfailed',request=>failed.push(request.url()+' '+request.failure()?.errorText));
    page.on('response',response=>{if(response.status()>=400)bad.push(response.status()+' '+response.url());});
    await page.setViewportSize({width:1440,height:1000});
    await page.goto(url);
    await page.locator('.katex').first().waitFor();
    const labs=page.locator('[data-lab]');
    for(const el of await labs.all()){
      await el.scrollIntoViewIfNeeded();
      await el.locator('h4').first().waitFor();
    }
    assert.equal(await page.locator('[data-ready=true]').count(),await labs.count());
    assert.equal(await page.locator('.katex-error').count(),0);
    assert(await page.evaluate(()=>CPI_CITIES.CA.find(x=>x[0]==='Yellowknife')[2]<CPI_CITIES.CA.find(x=>x[0]==='Vancouver')[2]));
    assert.equal(await page.locator('section[id^=s]').count(),19);
    await checkGeography(page);
    assert.equal(await page.locator('#stage-comparison tbody tr').count(),10);
    assert.equal(await page.locator('#method-comparison tbody tr').count(),20);
    for(const id of ['stage-comparison','method-comparison']){
      assert.equal(await page.locator('#'+id+' td a, #'+id+' td sup').count(),0);
      assert.equal(await page.locator('#'+id+' + .table-sources a[href="#ca-paper"]').count(),1);
      assert.equal(await page.locator('#'+id+' + .table-sources a[href="#bls-hom"]').count(),1);
    }
    assert.equal(await page.locator('.mast').count(),0);
    assert.equal(await page.locator('.hero .kicker').textContent(),'Comparative statistical study');
    assert.equal(await page.locator('#scope-hierarchy svg').count(),2);
    const sampling=page.locator('#lab-sampling');
    assert.equal(await sampling.locator('[name=design]').inputValue(),'Stratified');
    assert.equal(await sampling.locator('[data-cols="3"] input').count(),3);
    assert.equal(await sampling.locator('[data-cols="4"] input').count(),4);
    const basket=page.locator('#lab-basket');
    assert.equal(await basket.locator('input[name=w0]').isEnabled(),false);
    assert.match(await basket.textContent(),/U.S. CPI-U/);
    await basket.locator('[name=period]').selectOption('12');
    assert.match(await basket.locator('[data-out]').textContent(),/Unavailable/);
    await basket.locator('[name=month]').selectOption('2026-09');
    assert.doesNotMatch(await basket.locator('[data-out]').textContent(),/Unavailable/);
    await basket.locator('[name=mode]').selectOption('w');
    assert.equal(await basket.locator('input[name=w0]').isEnabled(),true);
    await basket.locator('[name=mode]').selectOption('q');
    const iso=page.locator('#lab-simulation');
    const signature=await iso.locator('[data-summary]').getAttribute('data-population');
    await iso.locator('[name=design]').selectOption('PPS');
    await iso.locator('[name=n]').evaluate(x=>{x.value=80;x.dispatchEvent(new Event('input',{bubbles:true}));});
    await iso.locator('[data-draw]').click();
    assert.equal(await iso.locator('[data-summary]').getAttribute('data-population'),signature);
    await iso.locator('[data-pop-settings] summary').click();
    await iso.locator('[name=seed]').fill('9');
    assert.equal(await iso.locator('[data-summary]').getAttribute('data-population'),signature);
    await iso.locator('[data-generate]').click();
    assert.notEqual(await iso.locator('[data-summary]').getAttribute('data-population'),signature);
    await iso.locator('[data-reset]').click();
    await iso.locator('[data-generate]').click();
    assert.equal(await iso.locator('[data-summary]').getAttribute('data-population'),signature);
    const tree=page.locator('#lab-trees');
    for(const node of await tree.locator('.tree-node').all()){
      await node.focus(); await node.press('Enter');
      assert.equal(await node.getAttribute('aria-pressed'),'true');
    }
    assert(await tree.locator('.tree-edge.is-active').count()>0);
    const calendar=page.locator('#lab-calendar');
    assert.equal(await calendar.locator('.calendar span').count(),56);
    assert.equal(await calendar.locator('[data-out] tbody tr').count(),3);
    const variance=page.locator('#lab-variance svg');
    assert.equal(await variance.locator('.plot-bar').count(),4);
    assert(await variance.locator('.plot-bar').first().evaluate(x=>x.width.baseVal.value<=18));
    const yTicks=await variance.locator('.grid + .axis').allTextContents();
    assert.equal(new Set(yTicks).size,yTicks.length);
    assert(await page.evaluate(()=>[...document.querySelectorAll('a[href^="#"]')].every(a=>document.getElementById(a.hash.slice(1)))));
    const counters={sliders:0,selectors:0,checkboxes:0,buttons:0,charts:0,nav:0,downloads:0};
    for(const el of await labs.all()){
      if((await el.getAttribute('id'))?.startsWith('lab-geo-'))continue;
      for(const content of await el.locator('details').all())await content.evaluate(e=>e.open=true);
      for(const field of await el.locator('select').all()){
        const options=await field.locator('option').evaluateAll(x=>x.map(o=>o.value));
        for(const option of options)await field.selectOption(option);
        await field.selectOption(options[0]);counters.selectors++;
      }
      for(const field of await el.locator('input[type=range]').all()){
        await field.evaluate(x=>{
          const old=x.value;
          x.value=x.value===x.max?x.min:x.max;
          x.dispatchEvent(new Event('input',{bubbles:true}));
          if(x.name){x.value=old;x.dispatchEvent(new Event('input',{bubbles:true}));}
        });
        counters.sliders++;
      }
      for(const field of await el.locator('input[type=checkbox]').all()){
        await field.click();await field.click();counters.checkboxes++;
      }
      for(const field of await el.locator('input[type=number],input[type=month]').all()){
        await field.evaluate(x=>{
          const old=x.value;
          x.value=x.type==='month'?'2020-01':String(Math.min(+x.max,Math.max(+x.min,+old+.1)));
          x.dispatchEvent(new Event('input',{bubbles:true}));
          x.value=old;x.dispatchEvent(new Event('input',{bubbles:true}));
        });
      }
      for(const button of await el.locator('button').all()){
        await button.click();
        if(await button.getAttribute('aria-pressed')==='true'&&/Pause/.test(await button.textContent()))await button.click();
        counters.buttons++;
      }
      for(const chart of await el.locator('.chart').all()){
        if(await chart.isVisible()){await chart.hover();counters.charts++;}
      }
      for(const content of await el.locator('details').all())await content.evaluate(e=>e.open=false);
    }
    const simulation=page.locator('#lab-simulation');
    await page.locator('#lab-budget').evaluate(el=>{
      const values={budget:50,c0:8,c1:8,c2:10};
      for(const [key,value] of Object.entries(values)){
        const field=el.querySelector('[name='+key+']');
        field.value=value;
        field.dispatchEvent(new Event('input',{bubbles:true}));
      }
    });
    assert.match(await page.locator('#lab-budget [data-out]').textContent(),/requires 52 budget units/);
    await simulation.locator('button[data-run]').click();
    await page.waitForFunction(()=>document.querySelector('#lab-simulation [data-state]').textContent.includes('Finished'));
    assert.match(await simulation.locator('[data-state]').textContent(),/200 repetitions/);
    const film=page.locator('#film-controls');
    const timeline=film.locator('#film-seek');
    await film.locator('[name=speed]').selectOption('2');
    await film.locator('[data-play]').click();
    await page.waitForFunction(()=>document.querySelector('#film-seek').valueAsNumber>100);
    await film.locator('[data-play]').click();
    const paused=await timeline.inputValue();
    await film.locator('[name=speed]').selectOption('1');
    assert.equal(await timeline.inputValue(),paused);
    await timeline.focus();
    await timeline.press('End');
    assert.match(await film.locator('.film-status').textContent(),/Step 10 of 10/);
    assert.equal(await film.locator('#film-elapsed').textContent(),'00:52');
    const bar=await timeline.boundingBox();
    await page.mouse.move(bar.x+bar.width-5,bar.y+bar.height/2);
    await page.mouse.down();
    await page.mouse.move(bar.x+bar.width*.35,bar.y+bar.height/2,{steps:10});
    await page.mouse.up();
    assert.match(await film.locator('.film-status').textContent(),/Step 4 of 10/);
    await film.locator('[data-next]').click();
    assert.match(await film.locator('.film-status').textContent(),/Step 5 of 10/);
    await film.locator('[data-replay]').click();
    assert.match(await film.locator('.film-status').textContent(),/Step 1 of 10/);
    assert.equal(await film.locator('[data-frame],details').count(),0);
    for(const el of await page.locator('.code').all()){
      await el.locator('summary').click();
      await el.getByRole('button',{name:'Copy Code'}).click();
      assert.equal(await el.getByRole('status').textContent(),'Code copied.');
      assert.match(await page.evaluate(()=>navigator.clipboard.readText()),/const|def/);
      await el.locator('summary').click();
    }
    for(const link of await page.locator('.toc nav a').all()){
      await link.focus();
      await link.press('Enter');
      assert.equal(await page.evaluate(()=>location.hash),await link.getAttribute('href'));
      counters.nav++;
    }
    const sectionLink=page.locator('.toc a[href="#s11-2"]');
    await sectionLink.press('Enter');
    await page.waitForFunction(()=>document.querySelector('.toc a[href="#s11-2"]').matches('[aria-current=location]'));
    assert(await page.locator('.toc').evaluate(toc=>toc.scrollHeight>toc.clientHeight));
    await page.locator('.toc a[href="#s18-g6"]').press('Enter');
    await page.waitForFunction(()=>document.querySelector('.toc a[href="#s18-g6"]').matches('[aria-current=location]'));
    for(const link of await page.locator('a[download]').all()){
      const href=await link.getAttribute('href');
      const response=await context.request.head(new URL(href,url).href);
      assert.equal(response.status(),200,href);
      counters.downloads++;
    }
    const downloadPromise=page.waitForEvent('download');
    await page.locator('a[download][href$="inflation-rates.csv"]').click();
    const download=await downloadPromise;
    assert.equal(download.suggestedFilename(),'inflation-rates.csv');
    assert.equal(await download.failure(),null);
    for(const [label,width,height] of viewports){
      await page.setViewportSize({width,height});
      await page.waitForFunction(w=>innerWidth===w,width);
      await checkOpening(page,width);
      const geometry=await page.locator('#lab-geo-maps').evaluate(el=>{
        const panels=[...el.querySelectorAll('.geo-card')].map(e=>e.getBoundingClientRect());
        const maps=[...el.querySelectorAll('.geo-map')].map(e=>e.getBoundingClientRect());
        return {panelWidths:panels.map(p=>p.width), mapTops:maps.map(p=>p.top), mapWidths:maps.map(p=>p.width)};
      });
      assert(Math.abs(geometry.panelWidths[0]-geometry.panelWidths[1])<1,label+' equal-width map panels');
      if(width>=1024)assert(Math.abs(geometry.mapTops[0]-geometry.mapTops[1])<1,label+' aligned map frames');
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth-innerWidth);
      const offenders=await page.evaluate(()=>[...document.querySelectorAll('main *')].filter(x=>{
        const r=x.getBoundingClientRect();return r.width>0&&(r.right>innerWidth+2||r.left< -2);
      }).slice(0,16).map(x=>(x.closest('[data-lab],section')?.id||'')+' '+x.tagName+'.'+x.className));
      assert(overflow<=1,label+' overflow '+overflow+' '+offenders.join(','));
      if(width>=1024){
        const labels=await page.locator('#lab-weights .weight-bars .barrow > span:first-child').all();
        assert.equal(labels.length,16);
        for(const span of labels)assert(await span.evaluate(e=>e.scrollWidth<=e.clientWidth+1));
      }
      await page.evaluate(()=>scrollTo(0,0));
      await page.locator('#film-controls [data-replay]').evaluate(el=>el.click());
      await page.screenshot({path:path.join(temporary,label+'-opening.png')});
      if(label==='desktop'){
        for(const id of ['lab-geo-maps','lab-geo-architecture','lab-geo-empirical','lab-geo-weights','lab-geo-spatial']){
          await page.locator('#'+id).evaluate(e=>e.scrollIntoView({block:'start'}));
          await page.screenshot({path:path.join(temporary,id+'-desktop.png')});
        }
        await page.locator('.hero').screenshot({path:path.join(temporary,'revised-header.png')});
        for(const stage of [0,5]){
          await timeline.evaluate((el,index)=>{el.value=index*5200;el.dispatchEvent(new Event('input'));},stage);
          await page.locator('.film').screenshot({path:path.join(temporary,'revised-film-'+(stage+1)+'.png')});
        }
        for(const table of ['stage-comparison','method-comparison']){
          await page.locator('#'+table).screenshot({path:path.join(temporary,table+'.png')});
          await page.locator('#'+table+' + .table-sources').screenshot({path:path.join(temporary,table+'-sources.png')});
        }
      }
    }
    await page.locator('#contents summary').click();
    await page.locator('#contents nav a[href="#s16-1"]').click();
    assert.equal(await page.locator('#contents').getAttribute('open'),null);
    await page.locator('#lab-empirical').scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(temporary,'mobile-empirical.png')});
    await page.locator('#s11-2').scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(temporary,'mobile-math.png')});
    await page.locator('#lab-geo-maps').evaluate(el=>el.scrollIntoView({block:'start'}));
    await page.screenshot({path:path.join(temporary,'mobile-geography.png')});
    await page.setViewportSize({width:1440,height:1000});
    for(const id of ['scope-hierarchy','s3','lab-trees','lab-basket','lab-sampling','lab-budget','lab-calendar','lab-weights','lab-housing','lab-variance','lab-uncertainty','lab-simulation']){
      await page.locator('#'+id).evaluate(e=>e.scrollIntoView({block:'start'}));
      await page.screenshot({path:path.join(temporary,id+'.png')});
    }
    assert.deepEqual(errors,[]);
    assert.deepEqual(failed,[]);
    assert.deepEqual(bad,[]);
    const file=await context.newPage();
    const fileErrors=[];
    file.on('pageerror',error=>fileErrors.push(error.message));
    await file.goto(pathToFileURL(path.join(root,project+'.html')).href);
    const offlineLab=file.locator('#lab-simulation');
    await offlineLab.scrollIntoViewIfNeeded();
    await offlineLab.locator('button[data-run]').click();
    await file.waitForFunction(()=>document.querySelector('#lab-simulation [data-state]').textContent.includes('Finished'));
    await file.locator('#lab-geo-maps').scrollIntoViewIfNeeded();
    await file.locator('#lab-geo-maps .geo-map').first().waitFor();
    assert.equal(await file.locator('#lab-geo-maps .geo-map').count(),2);
    assert.deepEqual(fileErrors,[]);
    console.log(JSON.stringify({viewports,labs:await labs.count(),...counters,errors,failed,bad,fileErrors,screenshots:temporary},null,2));
  } finally {
    if(browser)await browser.close();
    server.close();
  }
}
run().catch(error=>{console.error(error);process.exitCode=1;server.close();});

