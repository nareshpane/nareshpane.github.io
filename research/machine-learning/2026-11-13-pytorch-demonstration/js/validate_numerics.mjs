// Independent browser-solver parity against canonical Python arrays and stability edge cases.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {riverSimulator} from './demonstrations.js';
const data=JSON.parse(readFileSync(new URL('../outputs/seed-42/river.json',import.meta.url),'utf8'));
const actual=riverSimulator({location:data.truth[0],intensity:data.truth[1]});
assert.equal(actual.field.length,data.field.length);
let maximum=0;
for(let i=0;i<actual.field.length;i++)for(let j=0;j<61;j++)maximum=Math.max(maximum,Math.abs(actual.field[i][j]-data.field[i][j]));
assert(maximum<1e-12,`JS/Python field maximum error ${maximum}`);
for(const velocity of [.1,.35,.7])for(const dispersion of [.002,.005,.015]){
 const value=riverSimulator({velocity,dispersion});
 assert(value.combined<=.75+1e-12);
 assert(value.field.every(row=>row.every(x=>Number.isFinite(x)&&x>=-1e-12)));
 assert(value.field.every(row=>row[0]===0&&row.at(-1)===0));
}
console.log(JSON.stringify({status:'passed',river_js_python_max_error:maximum,stable_control_combinations:9}));
