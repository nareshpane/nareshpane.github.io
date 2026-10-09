/* Real Chromium UI validation via built-in DevTools protocol, no packages.
 * Node 22+ and installed Chrome/Edge. Run after start_preview.ps1.
 * Test-only browser profiles and screenshots go to ignored .qa/.
 */
'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const {spawn} = require('node:child_process');
const assert = require('node:assert/strict');
const BASE = __dirname;
const URL = 'http://localhost:8000/research/harmonized-system/canada-and-provinces-trade-by-hs.html';
const wait = ms => new Promise(resolve=>setTimeout(resolve,ms));
const read = async name => JSON.parse(await fs.readFile(path.join(BASE,name),'utf8'));
const checks = [];
let browser, socket;
async function main() {
  assert(typeof WebSocket === 'function','Node 22+ is required');
  const candidates = [process.env.TRADE_TEST_BROWSER,'C:/Program Files/Google/Chrome/Application/chrome.exe','C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].filter(Boolean);
  let executable;
  for (const file of candidates) if (await fs.access(file).then(()=>true).catch(()=>false)) {executable=file;break;}
  assert(executable,'Chrome/Edge is required; alternatively set TRADE_TEST_BROWSER');
  const qa = path.join(BASE,'.qa');
  await fs.mkdir(qa,{recursive:true});
  const profile = await fs.mkdtemp(path.join(qa,'chromium-'));
  browser = spawn(executable,['--headless=new','--remote-debugging-port=0','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-extensions','--disable-component-update','--user-data-dir='+profile,'about:blank'],{stdio:'ignore',windowsHide:true});
  let connection;
  for (let i=0;i<100;i++) {
    connection = await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8').catch(()=>null);
    if (connection) break;
    await wait(100);
  }
  assert(connection,'Headless browser debugging port unavailable');
  const [port, endpoint] = connection.trim().split(/\r?\n/);
  socket = new WebSocket('ws://127.0.0.1:'+port+endpoint);
  await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
  let serial=0;
  const pending = new Map(), errors=[];
  socket.addEventListener('message',event=>{
    const r=JSON.parse(event.data);
    if (r.id && pending.has(r.id)) {
      const callback=pending.get(r.id);pending.delete(r.id);clearTimeout(callback.timeout);
      r.error?callback.reject(new Error(JSON.stringify(r.error))):callback.resolve(r.result);
    } else if(r.method === 'Runtime.exceptionThrown') errors.push(r.params.exceptionDetails.exception?.description || r.params.exceptionDetails.text);
    else if(r.method === 'Network.responseReceived' && r.params.response.status>=400 && !r.params.response.url.endsWith('/favicon.ico')) errors.push('HTTP '+r.params.response.status+' '+r.params.response.url);
  });
  function command(method,params={},sessionId) {
    return new Promise((resolve,reject)=>{
      const id=++serial;
      const timeout=setTimeout(()=>{pending.delete(id);reject(new Error('CDP timeout: '+method));},15000);
      pending.set(id,{resolve,reject,timeout});socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));
    });
  }
  const target = await command('Target.createTarget',{url:'about:blank'});
  const {sessionId} = await command('Target.attachToTarget',{targetId:target.targetId,flatten:true});
  const c=(method,params)=>command(method,params,sessionId);
  for (const method of ['Page.enable','Runtime.enable','Network.enable']) await c(method);
  const evaluate = async expression => {
    const result=await c('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
    if(result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  };
  async function until(expression,message) {
    for(let i=0;i<100;i++) {if(await evaluate(expression))return;await wait(50);}
    throw new Error('Timed out: '+message+'; browser errors: '+JSON.stringify(errors));
  }
  const screenshot = async name => {
    const shot=await c('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    await fs.writeFile(path.join(qa,name+'.png'),Buffer.from(shot.data,'base64'));
  };
  const key=async (name,code)=>{await c('Input.dispatchKeyEvent',{type:'keyDown',key:name,code:name,windowsVirtualKeyCode:code});await c('Input.dispatchKeyEvent',{type:'keyUp',key:name,code:name,windowsVirtualKeyCode:code});};
  await c('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await c('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'no-preference'}]});
  await c('Page.navigate',{url:URL});
  await c('Page.bringToFront');
  await until("document.getElementById('geography-name')?.textContent==='Canada' && !document.getElementById('results').hidden",'Canada default');
  const summary=await read('geography-summary-2025.json');
  assert.equal(await evaluate("document.querySelectorAll('.hs4-row').length"),20);
  assert.equal(await evaluate("document.getElementById('kpi-total').title"),'$720,769,312,708 CAD');
  checks.push('Canada default, annual value and Top 20');
  await until("document.querySelectorAll('[data-oa-ranked-heading]').length===3",'Animation data');
  await evaluate("document.getElementById('origin-exposure').scrollIntoView({block:'start',behavior:'instant'})");
  await until("document.getElementById('origin-exposure').dataset.playback==='playing'",'Visible autoplay');
  await evaluate("document.querySelector('[data-oa-pause]').click()");
  const paused=await evaluate("document.getElementById('origin-exposure').dataset.animationTime");
  await wait(250);
  assert.equal(await evaluate("document.getElementById('origin-exposure').dataset.animationTime"),paused);
  await evaluate("document.querySelector('[data-oa-pause]').click()");
  await until("Number(document.getElementById('origin-exposure').dataset.animationTime)>"+Number(paused),'Resume advances timeline');
  await evaluate("document.querySelector('[data-oa-replay]').click()");
  assert(Number(await evaluate("document.getElementById('origin-exposure').dataset.animationTime"))<1);
  await evaluate("document.querySelector('[data-oa-speed]').value='2';document.querySelector('[data-oa-speed]').dispatchEvent(new Event('change'))");
  const beforeSpeed=Number(await evaluate("document.getElementById('origin-exposure').dataset.animationTime"));
  await wait(450);
  const advance=Number(await evaluate("document.getElementById('origin-exposure').dataset.animationTime"))-beforeSpeed;
  assert(advance>.5 && advance<1.5,'Playback speed updates timeline');
  const seek=async t=>evaluate("document.querySelector('[data-oa-seek]').value="+t+";document.querySelector('[data-oa-seek]').dispatchEvent(new Event('input'))");
  for(const [time,phase] of [[2,'origins'],[8,'production'],[16,'logistics'],[22,'trade'],[29.5,'border'],[35.8,'hierarchy'],[40,'final']]) {
    await seek(time);
    assert.equal(await evaluate("document.getElementById('origin-exposure').dataset.phase"),phase);
    assert.equal(await evaluate("document.querySelectorAll('.oa-stage [id]').length===new Set([...document.querySelectorAll('.oa-stage [id]')].map(e=>e.id)).size"),true);
    if(['logistics','trade','border'].includes(phase)) {
      assert.equal(await evaluate("document.querySelectorAll('[data-oa-global-route]').length"),6);
      assert(Number(await evaluate("document.getElementById('oa-world').style.opacity"))>.9);
      assert.equal(await evaluate("document.getElementById('oa-checkpoint').style.opacity"),'0');
    }
    if(phase==='logistics') assert.equal(await evaluate("[...document.querySelectorAll('[data-oa-parcel]')].every(e=>e.dataset.loaded==='true' && e.parentElement.id===e.dataset.freight)"),true,'Cargo stays attached');
    if(time===16 || time===22 || time===29.5 || time===35.8 || time===40) {
      await evaluate("document.getElementById('origin-exposure').scrollIntoView({block:'start',behavior:'instant'})");
      await screenshot('animation-'+phase+'-desktop');
    }
  }
  assert.equal(await evaluate("document.querySelector('.oa-headline').textContent"),'From Geography to HS4 and HS6 Exports');
  assert.equal(await evaluate("document.querySelector('.oa-stage').textContent.includes('Section 338')"),false);
  await evaluate("document.querySelector('[data-oa-motion]').click()");
  assert.equal(await evaluate("document.getElementById('origin-exposure').dataset.playback"),'static');
  await evaluate("document.querySelector('[data-oa-motion]').click()");
  await evaluate("document.querySelector('[data-oa-replay]').click()");
  assert(Number(await evaluate("document.getElementById('origin-exposure').dataset.animationTime"))<1);
  assert.equal(await evaluate("[...document.querySelectorAll('[data-oa-parcel]')].every(e=>e.dataset.loaded==='false')"),true,'Replay resets cargo after completion');
  await seek(40);
  checks.push('Seven animation phases, six global destinations, pause/resume, replay before/after completion, speed, still mode, unique SVG ids and attached cargo');
  for(const g of summary.geographies) {
    await evaluate("document.getElementById('geography-search').focus();document.getElementById('geography-search').value="+JSON.stringify(g.name)+";document.getElementById('geography-search').dispatchEvent(new Event('input'))");
    await key('ArrowDown',40);await key('Enter',13);
    await until("document.getElementById('geography-name').textContent==="+JSON.stringify(g.name)+" && !document.getElementById('results').hidden",g.name+' selection');
    assert.equal(await evaluate("document.getElementById('kpi-total').title"),'$'+g.total.toLocaleString('en-CA')+' CAD');
    assert.equal(await evaluate("document.getElementById('kpi-hs4').textContent"),g.hs4Count.toLocaleString('en-CA'));
    assert.equal(await evaluate("document.querySelector('.hs4-label').dataset.heading"),g.largest[0]);
    assert((await evaluate("document.getElementById('detail-title').textContent")).startsWith(g.name+' ·'));
  }
  checks.push('All 14 geographies: keyboard selection, totals, counts, rankings and synchronized details');
  await evaluate("document.getElementById('clear-search').click()");
  await until("document.getElementById('geography-name').textContent==='Canada' && document.getElementById('geography-search').value===''","Clear and reset");
  assert.equal(await evaluate("document.querySelectorAll('#suggestions [role=option]').length"),14);
  await key('ArrowDown',40);
  assert.equal(await evaluate("document.getElementById('geography-search').getAttribute('aria-activedescendant')"),'geo-option-0');
  await key('Escape',27);
  assert.equal(await evaluate("document.getElementById('suggestions').hidden"),true);
  await key('ArrowUp',38);
  assert.equal(await evaluate("document.getElementById('geography-search').getAttribute('aria-activedescendant')"),'geo-option-13');
  await key('Escape',27);
  await evaluate("document.getElementById('geography-search').value='zzzz';document.getElementById('geography-search').dispatchEvent(new Event('input'))");
  assert.equal(await evaluate("document.querySelectorAll('#suggestions [role=option]').length"),0);
  await evaluate("document.getElementById('clear-search').click()");
  await until("!document.getElementById('results').hidden",'Reset loaded');
  await evaluate("document.getElementById('geography-search').blur();document.body.click()");
  assert.equal(await evaluate("document.querySelector('.hs4-row.selected').dataset.hs4"),summary.geographies.find(g=>g.id==='CANADA').largest[0]);
  checks.push('Autocomplete arrows/Enter, Escape, no-match query, clear/reset');
  const renderTimes={};
  for(const n of [10,20,50,'all']) {
    renderTimes[n]=Math.round(await evaluate("(()=>{const t=performance.now();document.querySelector('[name=top-limit][value=\""+n+"\"]').click();return performance.now()-t})()"));
    assert.equal(await evaluate("document.querySelectorAll('.hs4-row').length"),n==='all'?1216:n);
  }
  checks.push('Top 10/20/50/All: '+JSON.stringify(renderTimes)+' ms DOM render');
  const canada=await read('trade-CANADA-2025.json');
  const chart=await evaluate("[...document.querySelectorAll('.hs4-row')].map(r=>({code:r.dataset.hs4,width:parseFloat(r.querySelector('.hs4-stack').style.width),children:[...r.querySelectorAll('.hs6-segment')].map(s=>({code:s.dataset.child,width:parseFloat(s.style.width),color:s.style.backgroundColor}))}))");
  for(let i=0;i<canada.headings.length;i++) {
    const h=canada.headings[i],row=chart[i];
    assert.equal(row.code,h[0]);assert.equal(row.children.length,h[3].length);
    assert(Math.abs(row.width-h[1]/canada.headings[0][1]*100)<1e-4);
    assert(Math.abs(row.children.reduce((s,c)=>s+c.width,0)-100)<1e-3);
    for(let j=0;j<h[3].length;j++) {
      assert.equal(row.children[j].code,h[3][j][0]);
      assert(Math.abs(row.children[j].width-h[3][j][1]/h[1]*100)<1e-4);
      if(j>0) assert(Number(row.children[j].color.match(/\d+/g)[0])>=Number(row.children[j-1].color.match(/\d+/g)[0]));
    }
  }
  checks.push('All 1,216 Canada bars: complete HS6 segments, proportional widths and per-heading gradient');
  await evaluate("document.querySelector('[name=top-limit][value=\"20\"]').click();document.querySelector('[data-geography=AB]').click()");
  await until("document.getElementById('geography-name').textContent==='Alberta' && !document.getElementById('results').hidden",'Alberta detail');
  await evaluate("document.querySelector('.hs4-label[data-heading=\"2711\"]').click()");
  assert.equal(await evaluate("document.querySelectorAll('#hs6-rows tr').length"),6);
  assert.equal(await evaluate("document.getElementById('detail-rank').textContent"),'#2 of '+summary.geographies.find(g=>g.id==='AB').hs4Count);
  await evaluate("document.querySelector('#detail-stack [data-child=\"271121\"]').focus()");
  assert.equal(await evaluate("document.getElementById('tooltip').hidden"),false);
  assert((await evaluate("document.getElementById('tooltip').textContent")).includes('$8,811,519,511 CAD'));
  assert((await evaluate("document.getElementById('tooltip').textContent")).includes('74.9%'));
  await evaluate("document.querySelector('#hs6-rows [data-child=\"271112\"]').click()");
  assert((await evaluate("document.getElementById('tooltip').textContent")).includes('$1,800,360,301 CAD'));
  await evaluate("document.querySelector('.hs4-label[data-heading=\"2711\"]').focus()");
  assert((await evaluate("document.getElementById('tooltip').textContent")).includes('6.6%'));
  await evaluate("document.getElementById('comparison').open=true");
  assert.equal(await evaluate("document.querySelectorAll('#comparison-rows tr').length"),14);
  assert((await evaluate("document.getElementById('comparison-rows').textContent")).includes('No positive recorded exports'));
  checks.push('HS4 selection, full HS6 drilldown, exact keyboard/touch tooltips and absent comparison values');
  await evaluate("document.querySelector('[data-compare=BC]').click()");
  await until("document.getElementById('geography-name').textContent==='British Columbia' && !document.getElementById('results').hidden",'Comparison geography');
  assert((await evaluate("document.getElementById('detail-title').textContent")).includes('2711'));
  await evaluate("document.querySelector('[data-geography=YT]').click()");
  await until("document.getElementById('geography-name').textContent==='Yukon' && !document.getElementById('results').hidden",'Small origin');
  assert((await evaluate("document.getElementById('detail-title').textContent")).includes('Yukon · HS4'));
  assert.equal(await evaluate("document.querySelector('.hs4-row.selected').dataset.hs4"),await evaluate("document.getElementById('detail-title').textContent.split('HS4 ')[1]"));
  checks.push('Comparison synchronization; changing geography updates selected HS4');
  await evaluate("document.querySelector('[data-geography=CANADA]').click()");
  await until("document.getElementById('geography-name').textContent==='Canada' && !document.getElementById('results').hidden",'Canada screenshot');
  await evaluate("document.getElementById('explorer').scrollIntoView({block:'start',behavior:'instant'})");
  await wait(200);await screenshot('explorer-desktop');
  await evaluate("document.getElementById('bar-title').scrollIntoView({block:'start',behavior:'instant'})");
  await wait(200);await screenshot('ranking-desktop');

  // Focused refinements: metadata, display-only formatting and true country data.
  await evaluate("document.querySelector('[data-geography=AB]').click()");
  await until("document.getElementById('geography-name').textContent==='Alberta' && !document.getElementById('results').hidden",'Destination origin Alberta');
  await evaluate("document.querySelector('.hs4-label[data-heading=\"2711\"]').click()");
  await until("!document.getElementById('destination-content').hidden && document.querySelectorAll('.destination-row').length>0",'Alberta 2711 destinations');
  const destAB=await read('destinations-AB-2025.json'), names=(await read('destination-countries-2025.json')).countries;
  const abTrade=await read('trade-AB-2025.json'), abHS4=abTrade.headings.find(h=>h[0]==='2711');
  const first=destAB.headings['2711'][0];
  assert.equal(await evaluate("document.querySelector('.destination-row').dataset.destination"),first[0]);
  assert.equal(await evaluate("document.querySelector('.destination-name').textContent"),names[first[0]]);
  assert.equal(await evaluate("document.querySelector('.destination-share').textContent"),(first[1]/abHS4[1]*100).toFixed(1)+'%');
  assert.equal(await evaluate("document.querySelector('.destination-fill').style.width"),'100%');
  await evaluate("document.querySelector('.destination-row').focus()");
  const countryTip=await evaluate("document.getElementById('tooltip').textContent");
  assert(countryTip.includes('Alberta')&&countryTip.includes('HS4 2711')&&countryTip.includes(names[first[0]])&&countryTip.includes('$'+first[1].toLocaleString('en-CA')+' CAD')&&countryTip.includes('Destination rank 1'));
  const selectedColors=await evaluate("[...document.querySelectorAll('#detail-stack .hs6-segment')].slice(0,4).map(e=>e.style.backgroundColor)");
  assert.deepEqual(selectedColors.slice(0,3),['rgb(174, 64, 27)','rgb(230, 126, 83)','rgb(242, 173, 136)']);
  const formatSamples=await evaluate("[0,.03,8.34,17.56,100].map(TradeDisplay.percent)");
  assert.deepEqual(formatSamples,['0.0%','0.0%','8.3%','17.6%','100.0%']);
  await evaluate("document.querySelector('.hs4-label[data-heading=\"3901\"]').click()");
  await until("document.getElementById('destination-subtitle').textContent.includes('HS4 3901') && document.querySelectorAll('.destination-row').length===10",'Heading with 46 destinations');
  const many=destAB.headings['3901'], manyTotal=abTrade.headings.find(h=>h[0]==='3901')[1];
  for(const n of [10,20,'all']) {
    await evaluate("document.querySelector('[name=destination-limit][value=\""+n+"\"]').click()");
    const length=n==='all'?many.length:Math.min(n,many.length);
    assert.equal(await evaluate("document.querySelectorAll('.destination-row').length"),length);
    const other=many.slice(length).reduce((s,r)=>s+r[1],0);
    const displayed=await evaluate("[...document.querySelectorAll('.destination-row')].map(e=>({code:e.dataset.destination,share:e.querySelector('.destination-share').textContent,width:parseFloat(e.querySelector('.destination-fill').style.width)}))");
    for(const [i,r] of displayed.entries()) {
      assert.equal(r.code,many[i][0]);
      assert.equal(r.share,(many[i][1]/manyTotal*100).toFixed(1)+'%');
      assert(Math.abs(r.width-many[i][1]/many[0][1]*100)<1e-4);
    }
    assert.equal(await evaluate("document.getElementById('destination-other').hidden"),other===0);
    if(other) {
      const label=await evaluate("document.getElementById('destination-other').textContent");
      assert(label.includes('$'+other.toLocaleString('en-CA')+' CAD'));
      assert(label.includes((other/manyTotal*100).toFixed(1)+'%'));
    }
  }
  checks.push('Destination Top 10/20/All, unranked Other, full-HS4 denominator, exact country tooltip, distinct top-three HS6 shades and one-decimal percentages');
  await evaluate("document.querySelector('[name=destination-limit][value=\"10\"]').click();document.getElementById('destinations').scrollIntoView({block:'start',behavior:'instant'})");
  await wait(300);await screenshot('destinations-desktop');
  await evaluate("document.querySelector('.hs4-label[data-heading=\"3901\"]').click()");
  await until("document.getElementById('destination-subtitle').textContent.includes('HS4 3901') && document.querySelectorAll('.destination-row').length>0",'Different heading destinations');
  assert.equal(await evaluate("document.querySelector('.destination-row').dataset.destination"),destAB.headings['3901'][0][0]);
  await evaluate("document.querySelector('[data-geography=BC]').click()");
  await until("document.getElementById('geography-name').textContent==='British Columbia' && document.getElementById('destination-subtitle').textContent.includes('HS4 3901') && document.querySelectorAll('.destination-row').length>0",'Province change preserves eligible HS4');
  const destBC=await read('destinations-BC-2025.json');
  assert.equal(await evaluate("document.querySelector('.destination-row').dataset.destination"),destBC.headings['3901'][0][0]);
  await evaluate("document.querySelector('[name=top-limit][value=\"50\"]').click();document.getElementById('clear-heading').click()");
  assert.equal(await evaluate("document.getElementById('geography-name').textContent"),'British Columbia');
  assert.equal(await evaluate("document.querySelector('[name=top-limit]:checked').value"),'50');
  assert.equal(await evaluate("document.querySelectorAll('.hs4-row.selected,.hs6-segment.active,#hs6-rows tr,.destination-row,#comparison-rows tr').length"),0);
  assert.equal(await evaluate("document.getElementById('detail-body').hidden && !document.getElementById('detail-empty').hidden && document.getElementById('destination-content').hidden"),true);
  assert.equal(await evaluate("document.getElementById('destination-subtitle').textContent"),'Select an HS4 category in Section 02 to see its export destinations.');
  await evaluate("document.querySelector('.hs4-label[data-heading=\"2711\"]').click();document.getElementById('clear-heading').click()");
  await wait(300);
  assert.equal(await evaluate("document.querySelectorAll('.destination-row').length"),0,'Clear cannot be overwritten by an earlier async response');
  await evaluate("document.querySelector('[data-geography=ON]').click()");
  await until("document.getElementById('geography-name').textContent==='Ontario' && !document.getElementById('results').hidden",'Empty state remains on origin change');
  assert.equal(await evaluate("document.getElementById('detail-body').hidden && document.querySelectorAll('.destination-row').length===0"),true);
  await evaluate("document.querySelector('.hs4-label').click()");
  await until("document.querySelectorAll('.destination-row').length>0",'Select after Clear');
  checks.push('HS4/province synchronization, Clear preserves origin and HS4 Top N, all dependent sections empty, asynchronous stale-response protection and selection after Clear');
  await evaluate("window.scrollTo(0,0)"); await wait(200);await screenshot('header-desktop');
  const metaRows=await evaluate("[...document.querySelectorAll('.provenance-item')].map(e=>e.textContent.trim())");
  assert.equal(metaRows.length,4);
  for(const [i,s] of ['Created: October 8, 2026','Harness: Codex on Windows PowerShell','Model: GPT-6.1 Sol','Reasoning: High'].entries()) assert(metaRows[i].includes(s));
  assert.equal(await evaluate("document.querySelectorAll('.provenance-item:nth-child(3) svg').length"),1);
  assert.equal(await evaluate("document.querySelector('.selection-instruction strong').textContent"),'Select a HS4 Category to See Further Info in Later Section');
  assert.equal(await evaluate("getComputedStyle(document.querySelector('.selection-instruction')).backgroundColor"),'rgb(230, 242, 251)');
  assert.equal(await evaluate("getComputedStyle(document.getElementById('clear-heading')).backgroundColor"),'rgb(216, 238, 233)');
  checks.push('Four exact metadata rows, compact model SVG, light-blue instruction and pale-teal Clear button');
  await evaluate("document.querySelector('[name=top-limit][value=\"20\"]').click();document.querySelector('[data-geography=CANADA]').click()");
  await until("document.getElementById('geography-name').textContent==='Canada' && !document.getElementById('results').hidden",'Reset before responsive checks');
  for(const width of [768,390,320]) {
    await c('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:width<650});
    await wait(100);
    assert.equal(await evaluate("document.documentElement.scrollWidth<=innerWidth"),true,'No overflow at '+width+'px');
    assert(await evaluate("document.querySelector('.hs4-track').getBoundingClientRect().width>60"),'Readable tracks at '+width+'px');
    if(width===390) {await evaluate("document.getElementById('bar-title').scrollIntoView({block:'start',behavior:'instant'})");await screenshot('ranking-mobile');await evaluate("document.getElementById('destinations').scrollIntoView({block:'start',behavior:'instant'})");await screenshot('destinations-mobile');await c('Emulation.setTouchEmulationEnabled',{enabled:true,maxTouchPoints:1});const touchBox=await evaluate("(()=>{const r=document.querySelector('.destination-row').getBoundingClientRect();return {x:r.left+60,y:r.top+r.height/2}})()");await c('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:touchBox.x,y:touchBox.y}]});await c('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await wait(100);assert.equal(await evaluate("document.getElementById('tooltip').hidden"),false,'Touch country tooltip');assert.equal(await evaluate("(()=>{const r=document.getElementById('tooltip').getBoundingClientRect();return r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=innerHeight})()"),true,'Tooltip fits mobile screen');await screenshot('destination-tooltip-mobile');await evaluate("document.getElementById('origin-exposure').scrollIntoView({block:'start',behavior:'instant'})");await seek(35.8);await screenshot('animation-hierarchy-mobile');await seek(40);await screenshot('animation-final-mobile');await seek(22);await screenshot('animation-world-mobile');await seek(29.5);await screenshot('animation-classification-mobile');}
  }
  await c('Emulation.setDeviceMetricsOverride',{width:390,height:900,deviceScaleFactor:1,mobile:true});
  await c('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
  await c('Page.reload');
  await until("document.getElementById('geography-name')?.textContent==='Canada' && document.getElementById('origin-exposure').dataset.playback==='static'",'Reduced motion');
  assert.equal(await evaluate("document.getElementById('origin-exposure').dataset.animationTime"),'40.00');
  checks.push('Desktop/tablet/390px/320px layout, resize, actual mobile country touch tooltip, screenshots and reduced-motion static frame');
  assert.deepEqual(errors,[],'Browser runtime/network errors');
  const report = {tested_at:new Date().toISOString(),url:URL,browser:executable,checks,errors,render_ms:renderTimes};
  await fs.writeFile(path.join(BASE,'ui-validation.json'),JSON.stringify(report,null,2)+'\n','utf8');
  console.log(JSON.stringify(report,null,2));
  await command('Browser.close');
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>{socket?.close();if(browser && browser.exitCode===null)browser.kill();});
