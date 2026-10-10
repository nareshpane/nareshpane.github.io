/* Run from repository root with Node 22+. --browser adds installed Chromium
 * checks by extending the existing preview driver in memory, not on disk.
 * Optional --baseline DIR verifies this addition against saved pre-edit files. */
'use strict';
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const base = path.resolve(__dirname,'..');
const load = file => JSON.parse(fs.readFileSync(path.join(base,'data',file),'utf8'));
const [meta,hs4,hs6,index,scope] = ['metadata.json','exports-hs4-2025-us.json','exports-hs6-2025-us.json','search-index.json','section338-hs6.json'].map(load);
const geographic = require('../js/geography-explorer.js'), comparison = require('../js/exposure-comparison.js');
const cache = geographic.createSummaries({meta,hs4,hs6,products:index.map(([code,description]) => ({code,description})),origins:meta.origins,scope});
const unchangedCache = JSON.stringify([...cache]);
const model = comparison.createComparison(cache), policy = new Set(scope.hs6);
assert.equal(model.rows.length,14); assert.equal(JSON.stringify([...cache]),unchangedCache);
for (const [i,row] of model.rows.entries()) {
  if (i) assert(model.rows[i-1].exposed >= row.exposed);
  assert.equal(row.exposed,cache.get(row.geography).exposed);
  assert.equal(row.components.reduce((a,b) => a+b.exposed,0),row.exposed);
  assert.equal(row.topThree.length,Math.min(3,cache.get(row.geography).ranking.length));
  const origin = meta.origins.findIndex(([code]) => code === row.geography);
  const independent = new Map();
  for (const [code,values] of Object.entries(hs6)) if (policy.has(code)) {
    const dollars = origin < 0 ? values.reduce((a,b) => a+b,0) : values[origin];
    independent.set(code.slice(0,4),(independent.get(code.slice(0,4)) || 0)+dollars);
  }
  const sectors = [...independent].filter(([,value]) => value>0).sort((a,b) => b[1]-a[1] || a[0].localeCompare(b[0]));
  assert.equal(sectors.reduce((total,[,value]) => total+value,0),row.exposed);
  assert.deepEqual(row.components.filter(c => c.rank<5).map(c=>[c.code,c.exposed]),sectors.slice(0,4));
  const remainder = row.components.find(c=>c.rank===5);
  assert.equal(remainder ? remainder.exposed : 0,sectors.slice(4).reduce((total,[,value])=>total+value,0));
  assert(Math.abs(row.components.reduce((a,b)=>a+b.share,0)-100)<1e-10);
  const text = comparison.tooltipText(row,geographic);
  for (const sector of row.topThree) {
    assert.equal(sector.share,sector.exposed/row.exposed*100);
    assert(text.includes(sector.description) && text.includes(sector.code));
    assert(text.includes(geographic.percent(sector.share)) && text.includes(geographic.money(sector.exposed)));
  }
}
assert.equal(model.maximum,Math.max(...model.rows.map(row=>row.exposed)));
assert.equal(new Set(comparison.colors.slice(0,4)).size,4);
const missing = new Map(cache); missing.set('AB',{...cache.get('AB'),exposed:null});
const unavailable = comparison.createComparison(missing).rows.find(row=>row.geography==='AB');
assert.equal(unavailable.available,false); assert.equal(unavailable.exposed,null);
assert(comparison.tooltipText(unavailable,geographic).includes('Data unavailable'));
const zero = new Map(cache); zero.set('AB',{...cache.get('AB'),exposed:0,ranking:[]});
assert.equal(comparison.createComparison(zero).rows.find(row=>row.geography==='AB').available,true);
assert.throws(()=>comparison.createComparison(new Map(cache).set('AB',{...cache.get('AB'),exposed:1})),/reconcile/);
const page = path.join(base,'..','section-338-hs4-hs6-exposure-canada.html'), html = fs.readFileSync(page,'utf8');
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(ids.length,new Set(ids).size);
for (const [,value] of html.matchAll(/(?:aria-controls|aria-labelledby|aria-describedby|for)="([^"]+)"/g)) for (const id of value.split(' ')) assert(ids.includes(id));
for (const [,url] of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
  if (/^(?:[a-z]+:|\/\/)/i.test(url)) continue;
  const [file,fragment] = url.split('#');
  if (file) assert(fs.existsSync(path.resolve(path.dirname(page),file)),url);
  else if (fragment) assert(ids.includes(fragment),url);
}
assert(html.indexOf('id="exposure-across-canada"') > html.indexOf('id="policy-content"'));
assert(html.includes('</div></section>\n<section id="exposure-across-canada"'));
const baselineFlag = process.argv.indexOf('--baseline');
if (baselineFlag >= 0) {
  const baseline = process.argv[baselineFlag+1];
  const stripped = html.replace('<link rel="stylesheet" href="2-section-338-hs4-hs6-exposure-canada/css/exposure-comparison.css">\n','')
    .replace('<script src="2-section-338-hs4-hs6-exposure-canada/js/exposure-comparison.js" defer></script>\n','')
    .replace(/<section id="exposure-across-canada"[\s\S]*?<\/section>\n/,'');
  assert.equal(stripped,fs.readFileSync(path.join(baseline,'page.html'),'utf8'),'Existing page content changed');
  const source = fs.readFileSync(path.join(base,'js/geography-explorer.js'),'utf8')
    .replace('    if (window.ExposureComparison) window.ExposureComparison.initialize(cache);\n','');
  assert.equal(source,fs.readFileSync(path.join(baseline,'geography-explorer.js'),'utf8'),'Section 1 logic changed beyond the additive cache handoff');
  const crypto = require('node:crypto');
  for (const [file,hash] of Object.entries(JSON.parse(fs.readFileSync(path.join(baseline,'hashes.json'),'utf8')))) {
    if (file.replaceAll('\\','/').endsWith('/js/geography-explorer.js')) continue;
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'),hash,'Existing asset changed: '+file);
  }
  console.log('Sections 1 and 2, introduction, animation, schematics, navigation, footer, existing styles/scripts/data: preserved exactly; single additive cache handoff: PASS');
}
console.log('Fourteen geographies, independent HS6 totals, top four/remainder, shared scale, ranking, top-three shares/descriptions, missing vs zero, unchanged cache, HTML IDs/assets: PASS');
console.table(model.rows.map(row=>({geography:row.name,exposed:row.exposed,leading:row.components.filter(c=>c.rank<5).map(c=>c.code).join(', ')})));

if (process.argv.includes('--browser')) {
  const Module = require('node:module');
  const filename = path.join(__dirname,'verify_geography_browser.js');
  let source = fs.readFileSync(filename,'utf8');
  const insert = (marker,extra) => { assert(source.includes(marker)); source = source.replace(marker,extra+'\n'+marker); };
  const body = fn => { const text=fn.toString(); return text.slice(text.indexOf('{')+1,text.lastIndexOf('}')); };
  insert("    const beforeProduct = await evaluate", body(async function () {
    const sectionThreeModel = require('../js/exposure-comparison.js').createComparison(expected);
    const sectionThreeApi = require('../js/exposure-comparison.js');
    const sectionThreeBefore = await evaluate('document.getElementById("comparison-bars").innerHTML');
    assert(await evaluate('document.getElementById("exposure-across-canada").previousElementSibling.id === "explorer"'));
    const chartRows = await evaluate(`[...document.querySelectorAll('.comparison-row')].map(row=>({code:row.dataset.geography,total:row.querySelector('.comparison-total').textContent,width:parseFloat(row.querySelector('.comparison-track').style.getPropertyValue('--comparison-width')),components:[...row.querySelectorAll('.comparison-segment')].map(c=>({code:c.dataset.hs4,value:Number(c.dataset.value),rank:Number(c.dataset.rank),share:parseFloat(c.style.width),color:getComputedStyle(c).backgroundColor,minWidth:getComputedStyle(c).minWidth}))}))`);
    assert.equal(chartRows.length,14);
    chartRows.forEach((actual,i)=>{
      const row=sectionThreeModel.rows[i]; assert.equal(actual.code,row.geography); assert.equal(actual.total,api.money(row.exposed));
      assert(Math.abs(actual.width-row.exposed/sectionThreeModel.maximum*100)<1e-4);
      assert.equal(actual.components.reduce((a,c)=>a+c.value,0),row.exposed);
      actual.components.forEach((actualComponent,j)=>{
        const component=row.components[j]; assert.equal(actualComponent.code,component.code || 'other'); assert.equal(actualComponent.value,component.exposed); assert.equal(actualComponent.rank,component.rank);
        assert(Math.abs(actualComponent.share-component.share)<1e-4); assert.equal(actualComponent.minWidth,'0px');
        const hex=sectionThreeApi.colors[component.rank-1]; const rgb=[1,3,5].map(index=>parseInt(hex.slice(index,index+2),16));
        assert.equal(actualComponent.color,'rgb('+rgb.join(', ')+')');
      });
    });
    for (const row of sectionThreeModel.rows) {
      await evaluate(`document.querySelector('.comparison-row[data-geography="${row.geography}"]').scrollIntoView({block:'center',behavior:'instant'}); document.querySelector('.comparison-row[data-geography="${row.geography}"]').focus({preventScroll:true});`);
      await pause(30);
      assert.equal(await evaluate('document.getElementById("comparison-tooltip").textContent'),sectionThreeApi.tooltipText(row,api));
      assert.equal(await evaluate('document.getElementById("comparison-tooltip").hidden'),false);
      assert(await evaluate(`(()=>{const r=document.getElementById('comparison-tooltip').getBoundingClientRect();return r.left>=0&&r.top>=0&&r.right<=innerWidth&&r.bottom<=innerHeight})()`));
      await evaluate(`document.activeElement.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))`);
      assert.equal(await evaluate('document.getElementById("comparison-tooltip").hidden'),true);
      const hoverBox=await evaluate(`(()=>{const r=document.querySelector('.comparison-row[data-geography="${row.geography}"]').getBoundingClientRect();return {x:r.left+40,y:r.top+r.height/2}})()`);
      await call('Input.dispatchMouseEvent',{type:'mouseMoved',...hoverBox});
      assert.equal(await evaluate('document.getElementById("comparison-tooltip").hidden'),false);
      assert.equal(await evaluate('document.getElementById("comparison-tooltip").textContent'),sectionThreeApi.tooltipText(row,api));
      await evaluate(`document.activeElement.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))`);
    }
  }));
  insert("    assert.deepEqual(errors,[],'Browser JavaScript errors');", body(async function () {
    assert.equal(await evaluate('document.getElementById("comparison-bars").innerHTML'),sectionThreeBefore,'Section 3 changed with Section 1 or 2 selectors');
    for (const width of [1440,1024,768,390,320]) {
      await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<=390});
      await evaluate(`document.querySelector('.comparison-chart').scrollIntoView({behavior:'instant'});`);
      assert(await evaluate('document.documentElement.scrollWidth <= innerWidth'),'Page overflow');
      assert(await evaluate(`[...document.querySelectorAll('.comparison-total')].every(e=>{const r=e.getBoundingClientRect();return r.left>=0 && r.right<=innerWidth})`),'Total labels overflow');
      if (width===1440 || width===390) {
        const screenshot=await call('Page.captureScreenshot',{format:'png'}); const destination=path.join(temporary,'comparison-'+width+'.png'); fs.writeFileSync(destination,Buffer.from(screenshot.data,'base64')); console.log('Section 3 screenshot: '+destination);
      }
    }
    await call('Emulation.setTouchEmulationEnabled',{enabled:true});
    await evaluate(`document.querySelector('.comparison-row').scrollIntoView({block:'center',behavior:'instant'}); document.activeElement.blur();`);
    const touchBox=await evaluate(`(()=>{const r=document.querySelector('.comparison-row').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})()`);
    await call('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[touchBox]});
    await call('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]}); await pause(150);
    assert.equal(await evaluate('document.getElementById("comparison-tooltip").hidden'),false);
    assert.equal(await evaluate('getComputedStyle(document.querySelector(".comparison-touch-note")).display'),'inline');
    console.log('Section 3 browser: 14 ranked stacks, shared scale, exact segments/colours, all focused tooltips, full descriptions/shares, Escape, mobile tap, viewport containment and selector independence: PASS');
  }));
  const runner = new Module(filename,module); runner.filename=filename; runner.paths=Module._nodeModulePaths(path.dirname(filename)); runner._compile(source,filename);
}
