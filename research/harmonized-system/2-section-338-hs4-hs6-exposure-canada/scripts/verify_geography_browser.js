/* Real Chromium checks using Node 22+ and built-in CDP WebSocket, no packages.
 * Opens an isolated headless browser and a temporary localhost server.
 * Optional: BROWSER_EXE points to an installed Chromium browser. */
'use strict';
const fs = require('node:fs'), path = require('node:path'), os = require('node:os');
const http = require('node:http'), {spawn} = require('node:child_process');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname,'../../../..');
const base = path.resolve(__dirname,'..');
const candidates = [process.env.BROWSER_EXE,'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe','C:/Program Files/Google/Chrome/Application/chrome.exe'].filter(Boolean);
const browserExe = candidates.find(file => fs.existsSync(file));
assert(browserExe,'Set BROWSER_EXE to an installed Chromium browser');
const preview = http.createServer((request,response) => {
  const file = path.resolve(root,'.' + decodeURIComponent(new URL(request.url,'http://localhost').pathname));
  if (!file.startsWith(root + path.sep)) { response.writeHead(403); response.end(); return; }
  fs.readFile(file,(error,content) => {
    if (error) { response.writeHead(404); response.end(); return; }
    response.setHeader('Content-Type',({'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json','.svg':'image/svg+xml'})[path.extname(file)] || 'application/octet-stream');
    response.end(content);
  });
});
const pause = ms => new Promise(resolve => setTimeout(resolve,ms));
(async () => {
  await new Promise(resolve => preview.listen(0,'127.0.0.1',resolve));
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(),'hs-geography-qa-'));
  const browser = spawn(browserExe,['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=0','--user-data-dir='+temporary,'about:blank'],{windowsHide:true,stdio:'ignore'});
  let socket;
  try {
    const portFile = path.join(temporary,'DevToolsActivePort');
    for (let i = 0; i < 100 && !fs.existsSync(portFile); i++) await pause(100);
    assert(fs.existsSync(portFile),'Browser debugger did not start');
    const port = fs.readFileSync(portFile,'utf8').split('\n')[0];
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    socket = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
    await new Promise((resolve,reject) => { socket.addEventListener('open',resolve,{once:true}); socket.addEventListener('error',reject,{once:true}); });
    let sequence = 0; const pending = new Map(), errors = [];
    socket.addEventListener('message',event => {
      const message = JSON.parse(event.data);
      if (message.id) { const job = pending.get(message.id); pending.delete(message.id); if (message.error) job.reject(message.error); else job.resolve(message.result); }
      if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text + ': ' + JSON.stringify(message.params.exceptionDetails.exception));
      if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') errors.push(JSON.stringify(message.params.args));
    });
    const call = (method,params = {}) => new Promise((resolve,reject) => { const id = ++sequence; pending.set(id,{resolve,reject}); socket.send(JSON.stringify({id,method,params})); });
    const evaluate = async expression => {
      const result = await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
      if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    await call('Runtime.enable'); await call('Page.enable');
    await call('Emulation.setFocusEmulationEnabled',{enabled:true});
    await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await call('Page.navigate',{url:`http://127.0.0.1:${preview.address().port}/research/harmonized-system/section-338-hs4-hs6-exposure-canada.html`});
    for (let i = 0; i < 150; i++) { if (await evaluate('!!document.getElementById("geo-search") && !document.getElementById("geo-search").disabled')) break; await pause(100); }
    assert(await evaluate('!document.getElementById("geo-search").disabled'),'Geography data did not initialize');
    assert.equal(await evaluate('document.getElementById("geo-search").value'),'Canada');
    const layout = await evaluate(`(()=>{
      const links=[...document.querySelectorAll('.hero-explorer-links a')];
      const properties=['fontSize','fontWeight','color','backgroundColor','border','borderRadius','padding'];
      return {links:links.map(link=>({text:link.textContent,href:link.getAttribute('href'),style:properties.map(p=>getComputedStyle(link)[p]),top:link.getBoundingClientRect().top})),
        animationLabel:document.querySelector('.animation-label').textContent,
        productStory:document.getElementById('intro-visual').closest('section').previousElementSibling.querySelector('h2').textContent,
        storyCount:document.querySelectorAll('#intro-visual').length,
        staticAnimations:[...document.getElementById('geography-intro-visual').querySelectorAll('*')].map(e=>getComputedStyle(e).animationName)};
    })()`);
    assert.deepEqual(layout.links.map(link=>link.href),['#geography-explorer','#product-explorer']);
    assert.deepEqual(layout.links[0].style,layout.links[1].style);
    assert.equal(layout.links[0].top,layout.links[1].top);
    assert.equal(layout.animationLabel,'Animation Video');
    assert.equal(layout.productStory,'Section 2: Where does this product come from?');
    assert.equal(layout.storyCount,1);
    assert(layout.staticAnimations.every(name=>name==='none'),'New schematic must remain static');
    const data = Object.fromEntries(['metadata.json','exports-hs4-2025-us.json','exports-hs6-2025-us.json','search-index.json','section338-hs6.json'].map(name => [name,JSON.parse(fs.readFileSync(path.join(base,'data',name),'utf8'))]));
    const api = require('../js/geography-explorer.js');
    const expected = api.createSummaries({meta:data['metadata.json'],origins:data['metadata.json'].origins,hs4:data['exports-hs4-2025-us.json'],hs6:data['exports-hs6-2025-us.json'],scope:data['section338-hs6.json'],products:data['search-index.json'].map(([code,description]) => ({code,description}))});
    const beforeProduct = await evaluate('document.getElementById("selected-code").textContent');
    for (const [code,result] of expected) {
      const count = await evaluate(`document.getElementById('geo-search').focus(); document.getElementById('geo-search').dispatchEvent(new FocusEvent('focus')); document.querySelectorAll('#geo-options [role=option]').length`);
      assert.equal(count,14);
      await evaluate(`document.getElementById('geo-option-${code}').dispatchEvent(new PointerEvent('pointerdown',{bubbles:true}));`);
      const actual = await evaluate(`({name:document.getElementById('geo-search').value,exposed:document.getElementById('geo-exposed').textContent,total:document.getElementById('geo-total').textContent,intensity:document.getElementById('geo-intensity').textContent,codes:[...document.querySelectorAll('.geo-row')].map(row=>row.dataset.hs4)})`);
      assert.equal(actual.name,result.name); assert.equal(actual.total,api.money(result.total)); assert.equal(actual.exposed,api.money(result.exposed)); assert.equal(actual.intensity,api.percent(result.intensity));
      assert.deepEqual(actual.codes,result.ranking.map(row => row.code));
      assert.equal(await evaluate('document.getElementById("selected-code").textContent'),beforeProduct);
    }
    await evaluate(`document.getElementById('geo-clear').click()`);
    assert.equal(await evaluate('document.getElementById("geo-search").value'),'Canada');
    await evaluate(`const input=document.getElementById('geo-search'); input.value='british'; input.dispatchEvent(new Event('input')); input.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowDown',bubbles:true})); input.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));`);
    assert.equal(await evaluate('document.getElementById("geo-search").value'),'British Columbia');
    await evaluate(`document.getElementById('geo-clear').click(); document.getElementById('geography-explorer').scrollIntoView({behavior:'instant'});`);
    await pause(100);
    await evaluate(`document.querySelector('.geo-row').focus()`);
    await pause(700); // Root smooth scrolling must settle before checking visibility.
    const first = expected.get('CANADA').ranking[0];
    const tooltipText = await evaluate('document.getElementById("geo-tooltip").textContent');
    for (const text of [first.code,first.description,first.total.toLocaleString('en-CA'),first.exposed.toLocaleString('en-CA'),'Canada','2025',api.percent(first.intensity)]) assert(tooltipText.includes(text),`Tooltip missing ${text}: ${tooltipText}; focus=${await evaluate('document.activeElement.outerHTML.slice(0,500)')}; errors=${JSON.stringify(errors)}`);
    assert.equal(await evaluate('document.getElementById("geo-tooltip").hidden'),false);
    const visuals = await evaluate(`[...document.querySelectorAll('.geo-row')].map(row=>({width:parseFloat(row.querySelector('.geo-row-fill').style.width),color:row.querySelector('.geo-row-fill').style.backgroundColor}))`);
    expected.get('CANADA').ranking.forEach((row,i) => { assert(Math.abs(visuals[i].width-row.exposed/first.exposed*100)<1e-4); assert.equal(visuals[i].color.replaceAll(' ',''),api.intensityColor(row.intensity)); });
    assert(await evaluate(`const scroller=document.getElementById('geo-chart-scroll'); scroller.scrollTop=400; scroller.scrollTop>0 && scroller.scrollHeight>scroller.clientHeight`));
    for (const id of ['geography-explorer','product-explorer']) {
      await evaluate(`document.querySelector('.explorer-navigation a[href="#${id}"]').click()`);
      assert.equal(await evaluate('location.hash'),'#'+id);
      await evaluate(`document.querySelector('.hero-explorer-links a[href="#${id}"]').click()`);
      assert.equal(await evaluate('location.hash'),'#'+id);
    }
    await evaluate(`document.getElementById('product-search').value='9403'; document.getElementById('product-search').dispatchEvent(new Event('input')); document.getElementById('product-search').dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));`);
    assert.equal(await evaluate('document.getElementById("selected-code").textContent'),'9403');
    await evaluate(`document.getElementById('exposure-geography').value='AB'; document.getElementById('exposure-geography').dispatchEvent(new Event('change'));`);
    assert.equal(await evaluate('document.getElementById("geo-search").value'),'Canada');
    await evaluate(`document.getElementById('intro-visual').scrollIntoView({behavior:'instant'});`);
    await pause(100);
    assert(await evaluate('document.getElementById("intro-visual").classList.contains("played")'),'Relocated schematic animation did not initialize');
    assert(await evaluate('document.getElementById("origin-exposure").querySelector(".oa-stage").children.length > 0'),'Introductory animation did not render');
    for (const width of [1440,1024,768,390,320]) {
      await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<=390});
      const dimensions = await evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})');
      assert(dimensions.scroll <= dimensions.width,`Horizontal overflow at ${width}: ${JSON.stringify(dimensions)}`);
      await evaluate(`document.getElementById('geography-explorer').scrollIntoView({behavior:'instant'});`);
      if (width === 1440 || width === 390) {
        const screenshot = await call('Page.captureScreenshot',{format:'png'});
        const destination = path.join(temporary,`geography-${width}.png`); fs.writeFileSync(destination,Buffer.from(screenshot.data,'base64'));
        console.log('Screenshot: '+destination);
        await evaluate(`document.querySelector('.geo-chart-card').scrollIntoView({behavior:'instant'}); document.getElementById('geo-chart-scroll').scrollTop=0;`);
        const chart = await call('Page.captureScreenshot',{format:'png'});
        const chartDestination = path.join(temporary,`ranking-${width}.png`); fs.writeFileSync(chartDestination,Buffer.from(chart.data,'base64'));
        console.log('Screenshot: '+chartDestination);
        for (const [label,selector] of [['introduction','.hero'],['geography-schematic','.geography-story'],['product-schematic','#explorer']]) {
          await evaluate(`document.querySelector('${selector}').scrollIntoView({behavior:'instant'});`);
          const image = await call('Page.captureScreenshot',{format:'png'});
          const imageDestination = path.join(temporary,`${label}-${width}.png`); fs.writeFileSync(imageDestination,Buffer.from(image.data,'base64'));
          console.log('Screenshot: '+imageDestination);
        }
      }
    }
    assert.deepEqual(errors,[],'Browser JavaScript errors');
    const duplicateIds = await evaluate(`(()=>{const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);return ids.filter((id,i)=>ids.indexOf(id)!==i)})()`);
    assert.deepEqual(duplicateIds,[]);
    console.log('Real browser: all 14 selections, summary cards, full ranking, keyboard search, Clear, focused tooltips, value widths, intensity colours, anchors, scrolling, independent Product Explorer, animation, unique IDs and 320–1440px overflow: PASS');
    await call('Browser.close');
  } finally { if (socket) socket.close(); browser.kill(); preview.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
