/* Validate production HS4 exposure functions against an independent aggregation
 * of the existing JSON. No raw files, dependencies or browser install needed. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {createModel,groupSegments} = require('../js/exposure-ribbon.js');
const base = path.resolve(__dirname,'..');
const read = name => JSON.parse(fs.readFileSync(path.join(base,'data',name),'utf8'));
const metadata = read('metadata.json');
const data = {hs4:read('exports-hs4-2025-us.json'),hs6:read('exports-hs6-2025-us.json'),
  products:read('search-index.json').map(([code,description]) => ({code,description})),
  origins:metadata.origins,scope:read('section338-hs6.json')};
const model = createModel(data), scope = new Set(data.scope.hs6);
const appSource = fs.readFileSync(path.join(base,'js/app.js'),'utf8');
const displayedPercent = vm.runInNewContext(appSource.match(/const percent = ([^\n]+);/)[1]);
const expected = new Map();
const sum = values => values.reduce((a,b) => a + b,0);
for (const [code,values] of Object.entries(data.hs6)) {
  const heading = code.slice(0,4);
  if (!expected.has(heading)) expected.set(heading,{total:Array(13).fill(0),exposed:Array(13).fill(0),unexposed:Array(13).fill(0)});
  const group = expected.get(heading);
  values.forEach((value,i) => { group.total[i] += value; group[scope.has(code) ? 'exposed' : 'unexposed'][i] += value; });
}
let combinations = 0, noExports = 0, groupedRibbons = 0;
for (const heading of model.headings) {
  const group = expected.get(heading);
  assert.deepEqual(group.total,data.hs4[heading]);
  const canada = model.calculate(heading,'CANADA');
  let provinceExposed = 0;
  for (const [geography,index] of [['CANADA',null],...data.origins.map(([code],i) => [code,i])]) {
    const r = model.calculate(heading,geography), value = key => index === null ? sum(group[key]) : group[key][index];
    assert.equal(r.total,value('total'));
    assert.equal(r.exposed,value('exposed'));
    assert.equal(r.unexposed,value('unexposed'));
    assert.equal(r.exposed + r.unexposed,r.total);
    assert.equal(r.intensity,r.total ? value('exposed') / value('total') * 100 : null);
    if (index !== null) provinceExposed += r.exposed;
    if (!r.total) noExports++;
    const segments = groupSegments(r);
    assert.equal(sum(segments.map(segment => segment.value)),r.total);
    assert.equal(sum(segments.filter(segment => segment.exposed).map(segment => segment.value)),r.exposed);
    assert(segments.length <= 14);
    assert.deepEqual(segments.flatMap(segment => segment.children.map(child => child.code)).sort(),r.children.filter(child => child.value > 0).map(child => child.code).sort());
    for (const segment of segments) assert(segment.children.every(child => child.exposed === segment.exposed));
    if (segments.some(segment => segment.key.startsWith('other-'))) groupedRibbons++;
    combinations++;
  }
  assert.equal(canada.exposed,provinceExposed);
}
const countryResults = model.headings.map(heading => model.calculate(heading));
assert.equal(sum(countryResults.map(r => r.exposed)),data.scope.matched_exports);
const zero = countryResults.find(r => r.total > 0 && r.exposed === 0);
const full = countryResults.find(r => r.total > 0 && r.exposed === r.total);
assert(zero && full,'Natural 0% and 100% examples exist');
const single = countryResults.find(r => r.children.length === 1);
assert.equal(groupSegments(single).length,1);
const largest = countryResults.reduce((a,b) => a.childCount > b.childCount ? a : b);
assert(groupSegments(largest).length <= 14);
assert.throws(() => model.calculate('8414','UNKNOWN'),/Unknown geography/);
assert.throws(() => createModel({...data,hs4:{...data.hs4,'8414':data.hs4['8414'].map((v,i) => v + (i === 0 ? 1 : 0))}}),/does not equal all HS6/);
const missing = createModel({...data,products:data.products.filter(p => p.code !== '841490')});
assert.equal(missing.calculate('8414').children.find(p => p.code === '841490').description,'Description unavailable in supplied lookup');
function report(heading) {
  const r = model.calculate(heading);
  return {hs4:heading,total_cad:r.total,exposed_cad:r.exposed,unexposed_cad:r.unexposed,
    intensity_percent:r.intensity,displayed_intensity:displayedPercent(r.intensity),
    hs6_children:r.childCount,matched_hs6_children:r.exposedChildCount};
}
console.log(JSON.stringify({checked_combinations:combinations,no_export_combinations:noExports,
  grouped_ribbons:groupedRibbons,examples:['8414','8537','9403',zero.heading,full.heading].map(report),
  largest_heading:{hs4:largest.heading,children:largest.childCount,visible_segments:groupSegments(largest).length}},null,2));
console.log('HS4 totals, matched numerator, complement, 0–100 bounds, Canada/origin equality, grouping conservation, zero exports, one/many children, missing descriptions and mismatch rejection: PASS');
