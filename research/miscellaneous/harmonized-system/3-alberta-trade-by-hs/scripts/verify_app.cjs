/* Independent checks against production aggregation, search and area geometry.
 * Run: node research/harmonized-system/3-alberta-trade-by-hs/scripts/verify_app.cjs */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const base = path.resolve(__dirname,'..');
const read = n => JSON.parse(fs.readFileSync(path.join(base,'data',n),'utf8'));
const source = fs.readFileSync(path.join(base,'js/app.js'),'utf8');
const hook = `globalThis.api = {aggregate,composition,treemapLayout,searchCountries,searchProducts,money,pct,
  load(s,d,p){summary=s;dictionary=d;current=aggregate(p);}, get(){return current;}};`;
const context = {console};vm.createContext(context);
vm.runInContext(source.replace('  init();',hook),context);
const api = context.api, summary = read('summary.json'), dictionary = read('product-index.json');
api.load(summary,dictionary,read('alberta-products.json'));
for (const [query,code] of [['united st','US'],['chi','CN'],['jap','JP'],['mex','MX'],['nether','NL'],['US','US'],['angui','AI']]) assert.equal(api.searchCountries(query)[0].code,code,query);
for (const code of ['2709','2711','1205','3901','1001','010129']) assert.equal(api.searchProducts(code)[0].code,code);
for (const query of ['wheat','petroleum','plastics','rape seeds']) assert(api.searchProducts(query).length,query);
assert.equal(api.searchProducts('nonexistent-unicorn-product').length,0);
let countryPairs = 0;
for (const c of summary.countries) {
  const raw = read('countries/'+c.code+'.json');api.load(summary,dictionary,raw);
  const scope = api.get();countryPairs += raw.products.length;
  assert.equal(scope.total,c.total,c.code);
  assert.equal(scope.four.length,c.hs4Count);
  assert.equal(scope.six.length,c.hs6Count);
  assert.equal(scope.four.slice(0,4).reduce((s,p) => s+p.value,0)/c.total*100,c.top4Share);
  scope.ranks = new Map(scope.four.map((p,i) => [p.code,i+1]));
  for (const code of ['', '2709','2711','1205','1001','3901','0203']) {
    const entries = api.composition(c,code,scope);
    assert.equal(entries.reduce((s,p) => s+p.value,0),c.total);
    assert(Math.abs(entries.reduce((s,p) => s+p.value/c.total*100,0)-100)<1e-10);
    assert(entries.every(p => p.value>0));
    const trace = entries.filter(p => p.selected);
    assert.equal(trace.length,code && scope.map.has(code) ? 1 : 0);
    if (trace.length) {
      assert.equal(trace[0].value,scope.map.get(code).value);
      assert.equal(trace[0].rank,scope.ranks.get(code));
      assert.equal(!!trace[0].extracted,!c.top4.some(p => p[0]===code));
    }
    const other = entries.find(p => p.code === 'other');
    if (other) {
      const shown = new Set(entries.filter(p => p.code!=='other').map(p => p.code));
      assert.equal(other.value,scope.four.filter(p => !shown.has(p.code)).reduce((s,p) => s+p.value,0));
      assert.equal(other.count,scope.four.filter(p => !shown.has(p.code)).length);
    }
  }
  for (const heading of scope.four) assert.equal(heading.children.reduce((s,p) => s+p.value,0),heading.value);
  assert(scope.six.every(p => typeof p.code === 'string' && p.code.length === 6 && p.value > 0));
  for (const [width,height] of [[740,410],[580,410],[650,380],[326,330]]) {
    const tiles = api.treemapLayout(scope.four,width,height);
    assert.equal(tiles.length,c.hs4Count);
    const area = tiles.reduce((s,p) => s+p.w*p.h,0);
    assert(Math.abs(area-width*height)<.0001);
    for (const p of tiles) {
      assert(p.w>0 && p.h>0 && p.x>=0 && p.y>=0);
      assert(p.x+p.w <= width+.00001 && p.y+p.h <= height+.00001);
      assert(Math.abs(p.w*p.h/(width*height)-p.value/c.total)<1e-10);
    }
  }
  if (['US','CN','JP','MX','NL','AI'].includes(c.code)) {
    for (const p of scope.four.slice(0,3)) {
      assert.equal(api.searchProducts(p.code)[0].code,p.code);
      for (const child of p.children.slice(0,3)) assert.equal(api.searchProducts(child.code)[0].code,child.code);
    }
  }
}
assert.equal(api.money(18),'$18');assert.equal(api.pct(.000001),'<0.01%');
// Pale treemap tiles use dark labels; HS6 ribbon labels remain white.
const css = fs.readFileSync(path.join(base,'css/style.css'),'utf8');
const luminance = color => {
  const rgb = [1,3,5].map(i => parseInt(color.slice(i,i+2),16)/255).map(x => x<=.04045 ? x/12.92 : ((x+.055)/1.055)**2.4);
  return .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2];
};
const colors = source.match(/const colors = \[([^\]]+)\]/)[1].match(/#[0-9a-f]{6}/g);
assert(/\.tile \{[^}]*color:var\(--ink\)/.test(css));
for (const color of colors) assert((luminance(color)+.05)/(luminance('#223632')+.05)>=4.5,color+' with dark treemap labels');
const ribbonColors = [...css.matchAll(/\n\.rank-\d \{\s*background:(#[0-9a-f]{6})/g)].map(m => m[1]);
assert.equal(ribbonColors.length,5);
for (const color of ribbonColors) assert(1.05/(luminance(color)+.05)>=4.5,color+' with white ribbon labels');
const lightColors = [...css.matchAll(/\.market-mixes \.rank-\d,\.mix-key \.rank-\d \{ background:(#[0-9a-f]{6}); \}/g)].map(m => m[1]);
assert.equal(lightColors.length,5);
for(const color of [...lightColors,'#f2d5cc']) assert((luminance(color)+.05)/(luminance('#223632')+.05)>=4.5,color+' with dark composition labels');
console.log(`PASS: ${summary.countries.length} markets, ${countryPairs} positive destination-HS6 pairs; exact aggregation, counts, concentration, searches, leading zeros, every tile's area/bounds at four aspect ratios, and label contrast.`);
console.log('PASS: every country composition totals 100%; traced headings appear exactly once at their actual value/rank; Other values/counts exclude extracted products.');
