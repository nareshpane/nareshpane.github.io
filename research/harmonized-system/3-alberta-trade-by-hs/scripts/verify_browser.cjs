/* Optional Chrome DevTools smoke test, using Node 24 built-ins only.
 * Start localhost:8000 and an isolated headless Chrome with debugging port 9223.
 * Screenshots are saved outside the repository in the OS temporary directory.
 * Run: node research/harmonized-system/3-alberta-trade-by-hs/scripts/verify_browser.cjs */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const URL = 'http://localhost:8000/research/harmonized-system/alberta-trade-by-hs.html';
const pause = ms => new Promise(r => setTimeout(r,ms));
const shots = path.join(os.tmpdir(),'alberta-atlas-review');fs.mkdirSync(shots,{recursive:true});
async function main() {
  const targets = await (await fetch('http://localhost:9223/json')).json();
  const target = targets.find(t => t.type === 'page' && (t.url === 'about:blank' || t.url.startsWith(URL)));
  assert(target,'No isolated test tab');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r,j) => {ws.onopen=r;ws.onerror=j;});
  let serial = 0;const pending = new Map(), errors = [], failed = [];
  ws.onmessage = event => {
    const m = JSON.parse(event.data);
    if (m.id) {const p = pending.get(m.id);pending.delete(m.id);if(m.error)p.reject(m.error);else p.resolve(m.result);}
    else if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.text+': '+(m.params.exceptionDetails.exception?.description || ''));
    else if (m.method === 'Network.responseReceived' && m.params.response.status >= 400) failed.push(m.params.response.url);
  };
  const send = (method,params={}) => new Promise((resolve,reject) => {const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  };
  async function wait(expression) {
    for(let i=0;i<100;i++) {if(await evaluate(expression))return;await pause(100);}
    console.error('Timeout scroll metrics:',await evaluate(`['destination-bars','market-mixes'].map(id=>{const c=document.getElementById(id),r=c.querySelector('.selected');return {id,top:c.scrollTop,height:c.clientHeight,full:c.scrollHeight,box:c.getBoundingClientRect().toJSON(),row:r?.getBoundingClientRect().toJSON()};})`));
    throw new Error('Timed out: '+expression);
  }
  const fill = async (id,value) => evaluate(`(()=>{const n=document.getElementById(${JSON.stringify(id)});n.focus();n.value=${JSON.stringify(value)};n.dispatchEvent(new Event('input',{bubbles:true}));})()`);
  const key = async (id,key) => evaluate(`document.getElementById(${JSON.stringify(id)}).dispatchEvent(new KeyboardEvent('keydown',{key:${JSON.stringify(key)},bubbles:true}))`);
  const click = async selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const lens = async code => {await click(`[data-country="${code}"]`);await wait(`location.hash.includes('country=${code}') && document.getElementById('application').getAttribute('aria-busy')==='false'`);};
  await send('Runtime.enable');await send('Page.enable');await send('Network.enable');
  await send('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'no-preference'}]});
  await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:URL});await wait(`!document.getElementById('application')?.hidden && document.querySelectorAll('.tile').length===954`);
  assert.deepEqual(await evaluate(`({destinations:document.getElementById('destination-limit').value,mix:document.getElementById('mix-limit').value,scale:document.querySelector('input[name="scale"]:checked').value,exclude:document.getElementById('exclude-us').checked,hash:location.hash,rows:document.querySelectorAll('.destination-row').length,mixes:document.querySelectorAll('.mix-row').length})`),{destinations:'all',mix:'all',scale:'linear',exclude:false,hash:'',rows:197,mixes:197});
  assert(await evaluate(`document.getElementById('clear-market').hidden && document.getElementById('selected-product-summary').hidden`),'No reset controls without selections');
  assert.deepEqual(await evaluate(`['.destination-row .inline-track','.mix-bar','.product-row .inline-track'].map(s=>getComputedStyle(document.querySelector(s)).height)`),['16px','24px','8px']);
  for(const code of ['US','CN','JP','MX','NL']) {
    await lens(code);
    assert.equal(await evaluate(`document.querySelector('.market-summary h3').textContent.startsWith('Alberta → ')`),true);
    assert.equal(await evaluate(`document.querySelectorAll('.tile').length===Number(document.querySelectorAll('.market-stats dd')[3].textContent)`),true);
    const heading = {US:'2709',CN:'2709',JP:'1205',MX:'3901',NL:'1001'}[code];
    await fill('product-search',heading);await key('product-search','Enter');
    assert.equal(await evaluate(`document.getElementById('drill-title').textContent`),'Inside HS4 '+heading);
    const child = await evaluate(`document.querySelector('#hs6-list [data-code]').dataset.code`);
    await fill('product-search',child);await key('product-search','Enter');
    assert.equal(await evaluate(`location.hash.includes('hs6=${child}')`),true);
  }
  await fill('country-search','united');await key('country-search','ArrowUp');
  assert(await evaluate(`document.getElementById('country-search').getAttribute('aria-activedescendant')==='country-suggestions-'+(document.querySelectorAll('#country-suggestions li').length-1)`));
  await key('country-search','Escape');assert(await evaluate(`document.getElementById('country-suggestions').hidden`));
  for(const [query,code] of [['jap','JP'],['chin','CN'],['nether','NL'],['angui','AI']]) {
    await fill('country-search',query);await key('country-search','ArrowDown');
    const before = await evaluate('scrollY');await key('country-search','Enter');
    await wait(`location.hash.includes('country=${code}') && document.getElementById('application').getAttribute('aria-busy')==='false'`);
    await wait(`['destination-bars','market-mixes'].every(id=>{const c=document.getElementById(id),r=c.querySelector('[data-country="${code}"]'),b=c.getBoundingClientRect(),p=r.getBoundingClientRect();return p.top>=b.top-1&&p.bottom<=b.bottom+1;})`);
    assert.equal(await evaluate('scrollY'),before,'Country reveal must not scroll the document');
    for(const id of ['destination-bars','market-mixes']) assert(await evaluate(`(()=>{const c=document.getElementById('${id}'),r=c.querySelector('[data-country="${code}"]'),b=c.getBoundingClientRect(),p=r.getBoundingClientRect();return p.top>=b.top-1&&p.bottom<=b.bottom+1&&r.classList.contains('selected');})()`),id+' '+code);
    assert.equal(await evaluate(`document.getElementById('curve-selected').getAttribute('visibility')`),'visible');
  }
  assert.equal(await evaluate(`document.querySelector('.tile').getAttribute('aria-label').includes('$18 CAD')`),true);
  await click('#clear-market');await wait(`!location.hash.includes('country=') && document.querySelectorAll('.tile').length===954`);
  for(const code of ['2709','2711','1205','3901','1001']) {
    await fill('product-search',code);await key('product-search','ArrowDown');await key('product-search','Enter');
    assert.equal(await evaluate(`document.getElementById('drill-title').textContent`),'Inside HS4 '+code);
    const children = await evaluate(`Array.from(document.querySelectorAll('#hs6-list [data-code]')).slice(0,3).map(n=>n.dataset.code)`);
    assert(children.length>0);
    for(const child of children) {await click(`#hs6-list [data-code="${child}"]`);assert.equal(await evaluate(`location.hash.includes('hs6=${child}')`),true);}
    await wait(`document.getElementById('market-mixes').getAttribute('aria-busy')==='false'`);
    const expected = JSON.parse(fs.readFileSync(path.resolve(__dirname,'../data/summary.json'),'utf8')).countries.map(c => {
      const data = JSON.parse(fs.readFileSync(path.resolve(__dirname,'../data/countries/'+c.code+'.json'),'utf8'));
      return {country:c.code,value:data.products.filter(p => p[0].startsWith(code)).reduce((s,p) => s+p[1],0),total:c.total};
    });
    assert(await evaluate(`(${JSON.stringify(expected)}).every(e=>{const row=document.querySelector('.mix-row[data-country="'+e.country+'"]'),selected=row.querySelectorAll('.mix-segment[data-hs4="${code}"]');return selected.length===(e.value>0?1:0)&&(!e.value||(Number(selected[0].dataset.value)===e.value&&selected[0].classList.contains('selected-product')&&!!row.querySelector('.trace-readout')))&&Array.from(row.querySelectorAll('.mix-segment')).reduce((s,p)=>s+Number(p.dataset.value),0)===e.total;})`),'Trace correctness '+code);
  }
  for(const [c,code] of [['JP','1205'],['US','2709'],['MX','1205']]) {
    await lens(c);await fill('product-search',code);await key('product-search','Enter');
    assert(await evaluate(`!document.getElementById('clear-market').hidden && !document.getElementById('clear-market').disabled && document.getElementById('clear-market').textContent==='Clear market' && !document.getElementById('selected-product-summary').hidden && document.getElementById('clear-product').closest('.chart-heading')?.querySelector('h3').textContent==='Every positive HS4, by value' && !document.getElementById('clear-product').hidden && document.getElementById('clear-market').closest('.landscape')!==null && !document.getElementById('clear-product').disabled`),'Market and product resets visible beside their state');
    assert(await evaluate(`document.querySelector('.mix-row[data-country="${c}"]').classList.contains('selected')&&document.querySelector('.mix-row[data-country="${c}"] .mix-segment[data-hs4="${code}"]').classList.contains('selected-product')`));
    assert.deepEqual(await evaluate(`Array.from(document.querySelectorAll('.trace-readout')).map(n=>n.closest('.mix-row').dataset.country)`),[c],'Detailed trace only under selected market');
    await click('#clear-market');assert(await evaluate(`!location.hash.includes('country=')&&location.hash.includes('hs4=${code}')`));
    assert(await evaluate(`document.getElementById('clear-market').hidden && !document.getElementById('selected-product-summary').hidden`),'Clearing market retains product reset');
    await lens(c);await click('#clear-product');assert(await evaluate(`location.hash.includes('country=${c}')&&!location.hash.includes('hs4=')&&document.getElementById('trace-controls').hidden`));
    assert(await evaluate(`!document.getElementById('clear-market').hidden && document.getElementById('selected-product-summary').hidden`),'Clearing product retains market reset');
  }
  await click('#clear-market');
  for(const query of ['wheat','petroleum','plastics']) {
    await fill('product-search',query);assert(await evaluate(`document.querySelectorAll('#product-suggestions li').length>0`));
    assert(await evaluate(`document.querySelectorAll('.tile.match').length>0`));
    await key('product-search','Escape');assert.equal(await evaluate(`document.getElementById('product-suggestions').hidden`),true);
  }
  await click('#clear-product');assert.equal(await evaluate(`document.getElementById('drill-body').hidden`),true);
  await lens('JP');await fill('product-search','1205');await key('product-search','Enter');
  await lens('CN');await evaluate('history.back()');await wait(`location.hash.includes('country=JP') && document.getElementById('application').getAttribute('aria-busy')==='false'`);
  await evaluate('history.forward()');await wait(`location.hash.includes('country=CN') && document.getElementById('application').getAttribute('aria-busy')==='false'`);
  await send('Page.navigate',{url:URL+'#country=JP&hs4=1205&hs6=120510'});
  await wait(`document.getElementById('product-scope').textContent==='Alberta → Japan' && document.getElementById('drill-title').textContent==='Inside HS4 1205'`);
  assert(await evaluate(`document.querySelector('#hs6-list [data-code="120510"]').getAttribute('aria-pressed')==='true'`));
  await evaluate(`document.getElementById('destination-limit').value='all';document.getElementById('destination-limit').dispatchEvent(new Event('change'));`);
  assert.equal(await evaluate(`document.querySelectorAll('.destination-row').length`),197);
  await click('input[name="scale"][value="log"]');assert(await evaluate(`document.getElementById('scale-note').textContent.startsWith('LOG SCALE')`));
  await evaluate(`document.getElementById('mix-limit').value='25';document.getElementById('mix-limit').dispatchEvent(new Event('change'));`);
  assert.equal(await evaluate(`document.querySelectorAll('.mix-row').length`),25);
  await click('#exclude-us');assert(!(await evaluate(`Array.from(document.querySelectorAll('.mix-country')).some(n=>n.textContent==='United States of America')`)));
  await evaluate(`document.getElementById('mix-limit').value='all';document.getElementById('mix-limit').dispatchEvent(new Event('change'));document.getElementById('exclude-us').checked=false;document.getElementById('exclude-us').dispatchEvent(new Event('change'));`);
  for(const width of [1440,1024,768,390]) {
    await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width===390});await pause(500);
    const metrics = await evaluate(`({viewport:innerWidth,doc:document.documentElement.scrollWidth,body:document.body.scrollWidth,tiles:document.querySelectorAll('.tile').length})`);
    assert(metrics.doc <= width+1,JSON.stringify(metrics));assert.equal(metrics.tiles,Number(await evaluate(`document.querySelectorAll('.market-stats dd')[3].textContent`)));
    const png = await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});fs.writeFileSync(path.join(shots,`atlas-${width}.png`),Buffer.from(png.data,'base64'));
    const clip = await evaluate(`(()=>{const r=document.querySelector('[aria-labelledby="mix-title"]').getBoundingClientRect();return {x:0,y:r.top+scrollY,width:innerWidth,height:Math.min(r.height,1150),scale:1};})()`);
    const mixShot = await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip});fs.writeFileSync(path.join(shots,`mix-trace-${width}.png`),Buffer.from(mixShot.data,'base64'));
    console.log('Layout',width,metrics);
  }
  await send('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
  assert.equal(await evaluate(`matchMedia('(prefers-reduced-motion: reduce)').matches`),true);
  assert.equal(await evaluate(`getComputedStyle(document.querySelector('.tile')).transitionDuration`),'0s');
  await click('#clear-market');await click('#clear-product');
  await evaluate(`document.getElementById('destination-limit').value='all';document.getElementById('destination-limit').dispatchEvent(new Event('change'));document.querySelector('input[name="scale"][value="linear"]').click();document.getElementById('mix-limit').value='all';document.getElementById('mix-limit').dispatchEvent(new Event('change'));document.getElementById('exclude-us').checked=false;document.getElementById('exclude-us').dispatchEvent(new Event('change'));`);
  for(const width of [1440,1024,768,390]) {
    await send('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width===390});await pause(200);
    assert(await evaluate(`document.documentElement.scrollWidth<=${width}+1`));
    assert.equal(await evaluate(`document.querySelectorAll('.tile').length`),954);
    assert.equal(await evaluate(`document.querySelectorAll('.mix-row').length`),197);
    assert.equal(await evaluate(`document.getElementById('destination-bars').clientHeight`),width<=480 ? 480 : 700);
    assert.equal(await evaluate(`document.getElementById('market-mixes').clientHeight`),width<=480 ? 560 : 750);
    for(const [name,selector] of [['top','.hero'],['products','.atlas-card']]) {
      const clip = await evaluate(`(()=>{const r=document.querySelector('${selector}').getBoundingClientRect();return {x:0,y:Math.max(0,r.top+scrollY-20),width:innerWidth,height:Math.min(${name === 'top' ? 1050 : 1150},document.documentElement.scrollHeight-(r.top+scrollY-20)),scale:1};})()`);
      const png = await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip});
      fs.writeFileSync(path.join(shots,`all-${name}-${width}.png`),Buffer.from(png.data,'base64'));
    }
  }
  await send('Page.navigate',{url:URL+'#destinations=10&mix=25&scale=log&excludeUS=1'});await wait(`document.querySelectorAll('.mix-row').length===25`);
  assert.equal(await evaluate(`document.getElementById('destination-limit').value`),'10');
  assert.equal(await evaluate(`document.querySelector('input[name="scale"]:checked').value`),'log');
  assert.equal(await evaluate(`document.getElementById('exclude-us').checked`),true);
  assert.deepEqual(errors,[]);assert.deepEqual(failed,[]);
  console.log('PASS: All defaults/197 rows; internal country reveal without document scrolling; exact selected-HS4 traces/Other; simultaneous and independent selections; saved display state, keyboard, four widths and reduced motion; no runtime/HTTP errors. Screenshots:',shots);
  ws.close();
}
main().catch(e => {console.error(e);process.exit(1);});
