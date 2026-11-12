/* Execute the production search/formatting functions against real output data.
 * No dependencies, fake chart rendering or alternate search implementation.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const base = path.resolve(__dirname,'..');
const read = name => JSON.parse(fs.readFileSync(path.join(base,'data',name),'utf8'));
const source = fs.readFileSync(path.join(base,'js/app.js'),'utf8');
const hook = `
  globalThis.testAPI = {searchProducts, money, percent, share, highlighted,
    initialize(data) {
      products = data.map(([code,description,source]) => ({code,description,source,search:normalizedText(description)}));
    }
  };
`;
assert.equal(source.split('  init();').length,2,'Exactly one initialization hook');
const context = {window:{matchMedia:() => ({matches:true})}, document:{}, console};
vm.createContext(context);
vm.runInContext(source.replace('  init();',hook),context);
const api = context.testAPI;
api.initialize(read('search-index.json'));
for (const code of ['8414','841490','9403','8537','060110']) {
  assert.equal(api.searchProducts(code)[0].code,code);
}
assert.equal(api.searchProducts('0601.10')[0].code,'060110');
assert(api.searchProducts('841').every(p => p.code.startsWith('841')));
assert.equal(api.searchProducts('8414')[0].code.length,4);
assert.equal(api.searchProducts('841490')[0].code.length,6);
for (const query of ['pump','pumps','furniture','electrical','vacuum pump','electrical panel','plastics','vacuum pum']) {
  const results = api.searchProducts(query);
  assert(results.length > 0 && results.length <= 10,query);
  console.log(query + ': ' + results.map(p => p.code).join(', '));
}
assert(api.searchProducts('electrical panel').some(p => p.code === '8537'));
assert(api.searchProducts('furniture').some(p => p.code === '9403'));
assert.equal(api.searchProducts('no-such-commodity-zzzz').length,0);
assert.equal(api.searchProducts('').length,0);
assert.equal(api.share(0,0),null);
assert.equal(api.share(25,100),25);
assert.equal(api.percent(null),'n/a');
assert.equal(api.money(152042164486),'$152.0B');
assert.equal(api.money(843200000),'$843.2M');
assert.equal(api.money(4700000),'$4.7M');
assert(api.highlighted('<script>pump</script>','pump').includes('&lt;script&gt;'));
// Composition labels use white text. Check the actual palette's WCAG contrast.
const palette = source.match(/const colors = \[([^\]]+)\]/)[1].match(/#[0-9a-f]{6}/g);
for (const color of palette) {
  const linear = [1,3,5].map(i => parseInt(color.slice(i,i+2),16) / 255).map(n => n <= .04045 ? n / 12.92 : ((n + .055) / 1.055) ** 2.4);
  const luminance = linear[0] * .2126 + linear[1] * .7152 + linear[2] * .0722;
  assert(1.05 / (luminance + .05) >= 4.5,`Composition label contrast: ${color}`);
}
console.log('Production search ranking, prefixes, multiword/partial matching, leading zeroes, formatting and safe highlighting: PASS');
