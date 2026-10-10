/* Independent arithmetic, production-model comparisons and preservation checks. */
'use strict';
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const base = path.resolve(__dirname,'..');
const read = name => JSON.parse(fs.readFileSync(path.join(base,'data',name),'utf8'));
const [meta,hs4,hs6,index,scope] = ['metadata.json','exports-hs4-2025-us.json','exports-hs6-2025-us.json','search-index.json','section338-hs6.json'].map(read);
const products = index.map(([code,description]) => ({code,description}));
const data = {meta,hs4,hs6,products,origins:meta.origins,scope};
const api = require('../js/geography-explorer.js');
const cache = api.createSummaries(data), policy = new Set(scope.hs6);
const existing = require('../js/exposure-ribbon.js').createModel(data);
assert.equal(cache.size,14);
const audit = [];
for (const [geography,result] of cache) {
  const i = meta.origins.findIndex(([code]) => code === geography);
  const valueFor = values => geography === 'CANADA' ? values.reduce((a,b) => a+b,0) : values[i];
  const total = Object.values(hs6).reduce((a,v) => a + valueFor(v),0);
  const exposed = Object.entries(hs6).reduce((a,[code,v]) => a + (policy.has(code) ? valueFor(v) : 0),0);
  assert.equal(result.total,total); assert.equal(result.exposed,exposed);
  assert.equal(result.intensity,exposed / total * 100);
  let count = 0;
  for (const code of Object.keys(hs4)) {
    const expected = existing.calculate(code,geography);
    const actual = result.ranking.find(row => row.code === code);
    if (!expected.exposed) { assert.equal(actual,undefined); continue; }
    count++; assert.equal(actual.exposed,expected.exposed); assert.equal(actual.total,expected.total);
    assert.equal(actual.intensity,expected.intensity);
    assert(actual.exposed <= actual.total && actual.intensity >= 0 && actual.intensity <= 100);
  }
  assert.equal(result.ranking.length,count);
  result.ranking.forEach((row,j) => { if (j) assert(result.ranking[j-1].exposed >= row.exposed); });
  audit.push({geography,total,exposed,intensity:result.intensity,sectors:count});
}
assert.equal(cache.get('CANADA').total,meta.validation.annual_value_total);
assert.equal(cache.get('CANADA').exposed,scope.matched_exports);
assert.equal(api.money(null),'Data unavailable'); assert.equal(api.percent(null),'Data unavailable');
assert.equal(api.money(1234567890),'CAD 1.2B'); assert.equal(api.money(123456789),'CAD 123.5M');
assert.equal(api.percent(38.64),'38.6%');
assert.notEqual(api.intensityColor(10),api.intensityColor(90));
assert.throws(() => api.createSummaries({...data,scope:null}),/unavailable/);
const missing = structuredClone(hs6); missing[Object.keys(missing)[0]][0] = null;
assert.throws(() => api.createSummaries({...data,hs6:missing}),/Unavailable/);
assert.deepEqual(api.createSummaries({...data,scope:{...scope,hs6:[...scope.hs6,...scope.hs6]}}),cache);
const page = path.relative(process.cwd(),path.join(base,'..','section-338-hs4-hs6-exposure-canada.html')).replace(/\\/g,'/');
const before = execFileSync('git',['show','HEAD:'+page],{encoding:'utf8'});
const after = fs.readFileSync(page,'utf8');
const story = text => text.match(/<section class="trade-story card" aria-labelledby="story-title">[\s\S]*?<\/section>/)[0];
assert.equal(story(after),story(before),'Relocated product schematic changed');
assert.equal(after.split('id="intro-visual"').length - 1,1,'Product schematic duplicated');
const product = text => text.slice(text.indexOf('<section id="explorer"'),text.indexOf('<section class="card agreements"'))
  .replace('<span id="product-explorer" class="geo-anchor" aria-hidden="true"></span>\n','')
  .replace(/<section class="trade-story card" aria-labelledby="story-title">[\s\S]*?<\/section>\n/,'')
  .replace('Section 2: Where does this product come from?','Where does this product come from?');
assert.equal(product(after),product(before),'Product Explorer markup changed beyond its heading, anchor and schematic placement');
const opening = text => text.slice(text.indexOf('<section class="origin-exposure"'),text.indexOf('<!-- End conceptual opening illustration. -->'));
assert.equal(opening(after),opening(before),'Introductory animation changed');
assert(after.indexOf('id="geography-intro-visual"') < after.indexOf('class="explorer-navigation"'));
assert(after.indexOf('id="intro-visual"') > after.indexOf('id="explorer-title"'));
assert(after.indexOf('id="intro-visual"') < after.indexOf('id="product-search"'));
assert(after.includes('Section 1: Canada and Provinces Exposure Explorer'));
assert(after.includes('Section 2: Where does this product come from?'));
assert.equal(after.split('Explore a product ↓').length - 1,1);
assert(after.includes('href="#geography-explorer">Explore a Geography ↓'));
assert(after.includes('href="#product-explorer">Explore a product ↓'));
const ids = new Set([...after.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]));
for (const [,id] of after.matchAll(/url\(#([^)]+)\)/g)) assert(ids.has(id),'Broken SVG reference: ' + id);
const appPath = path.relative(process.cwd(),path.join(base,'js','app.js')).replace(/\\/g,'/');
const originalApp = execFileSync('git',['show','HEAD:'+appPath],{encoding:'utf8'});
const currentApp = fs.readFileSync(appPath,'utf8').replace('      if (window.GeographyExposure) window.GeographyExposure.initialize({hs4,hs6,products,origins,scope,meta});\n','');
assert.equal(currentApp.replace(/\r\n/g,'\n'),originalApp.replace(/\r\n/g,'\n'),'Existing product logic changed');
console.log('All 14 geographic totals, all 16,450 sector/geography combinations, ranking, missing-data handling, duplicated scope, formatting and original explorer preservation: PASS');
console.table(audit);
// Direct recalculation examples: filter HS6 children independently of the model.
for (const [geography,code] of [['AB','8414'],['ON','8537'],['BC','9403']]) {
  const i = meta.origins.findIndex(([origin]) => origin === geography);
  const children = Object.entries(hs6).filter(([child]) => child.startsWith(code));
  const total = children.reduce((a,[,v]) => a+v[i],0);
  const exposed = children.reduce((a,[child,v]) => a+(policy.has(child) ? v[i] : 0),0);
  const row = cache.get(geography).ranking.find(row => row.code === code);
  assert(Math.abs(row.intensity - 100*exposed/total) < 1e-12);
  console.log(`${geography} HS4 ${code}: ${exposed} / ${total} × 100 = ${(100*exposed/total).toFixed(1)}%`);
}
