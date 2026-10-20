/* Lightweight presentation smoke test against a minimal DOM stub.
 * Catches initialization/handler errors; does NOT test browser layout or browser APIs.
 */
import {readFileSync,writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
const elements=[],byID=new Map(),listeners=new Map();
class Element{
 constructor(tag,attrs={}){this.tagName=tag.toUpperCase();this.attrs=attrs;this.id=attrs.id||'';this.value=attrs.value||'';this.min=attrs.min||'';this.max=attrs.max||'';this.checked='checked'in attrs;this.dataset={};for(const[k,v]of Object.entries(attrs))if(k.startsWith('data-'))this.dataset[k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=v;this.style={};this._html='';this.textContent='';this.hidden=false;this.classList={add(){},remove(){},toggle(){return false;}};elements.push(this);if(this.id)byID.set(this.id,this);}
 set innerHTML(s){this._html=s;parse(s);}get innerHTML(){return this._html;}
 addEventListener(type,fn){this['on'+type]=fn;}setAttribute(k,v){this.attrs[k]=String(v);}hasAttribute(k){return k in this.attrs;}replaceChildren(){this._html='';}scrollIntoView(){}getBoundingClientRect(){return {left:10,bottom:50};}click(){this.onclick?.({target:this});}
}
function parse(html){const result=[];for(const m of html.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)){const attrs={};for(const a of m[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g))attrs[a[1]]=a[2]??'';const el=new Element(m[1],attrs);result.push(el);}for(const match of html.matchAll(/<select\b([^>]*)>([\s\S]*?)<\/select>/g)){const id=/id="([^"]+)"/.exec(match[1])?.[1];if(!id)continue;const options=[...match[2].matchAll(/<option([^>]*)>([^<]*)/g)],chosen=options.find(x=>x[1].includes('selected'))||options[0];byID.get(id).value=/value="([^"]*)"/.exec(chosen[1])?.[1]||chosen[2];}return result;}
parse(readFileSync(new URL('../2026-10-19-budget-impact-analysis-medication.html',import.meta.url),'utf8'));
globalThis.document={getElementById(id){const e=byID.get(id);assert.ok(e,`Missing DOM element ${id}`);return e;},querySelectorAll(selector){const a=/\[([^\]]+)\]/.exec(selector)?.[1];return a?elements.filter(e=>a in e.attrs&&(!selector.includes(':checked')||e.checked)):[];},addEventListener(t,fn){if(!listeners.has(t))listeners.set(t,[]);listeners.get(t).push(fn);},createElement(t){return new Element(t);},hidden:false};
globalThis.matchMedia=()=>({matches:true});globalThis.requestAnimationFrame=cb=>setTimeout(()=>cb(performance.now()),0);globalThis.innerWidth=1440;globalThis.innerHeight=1000;
let exportedBlob;const nativeObjectURL=URL.createObjectURL;URL.createObjectURL=blob=>{exportedBlob=blob;return nativeObjectURL(blob);};
await import('../js/app.js');let checks=0;
const get=id=>document.getElementById(id),has=(id,s)=>{assert.ok((get(id).innerHTML+get(id).textContent).includes(s),`${id} missing ${s}`);checks++;};
has('budget-cards','<strong>3,200</strong>');get('chart-budget-view').value='live';get('chart-budget-view').onchange();assert.equal(get('video-budget-view').value,'live');checks++;has('budget-cards','<strong>128</strong>');has('cohort-metrics','379');has('population-result','121.74');has('scene-title','1 / 13');
get('preset').value='fixed';get('preset').onchange();has('switch-summary','36 → 48 → 60');has('switch-summary','36 → 12 → 12');has('budget-cards','<strong>93</strong>');
get('input-secPrice').value='0';for(const fn of listeners.get('input'))fn({target:get('input-secPrice')});has('budget-cards','Illustrative saving');has('evidence-secPrice','hypothetical edited assumption');
get('reset').onclick();has('budget-cards','<strong>128</strong>');assert.equal(get('input-secPrice').value,882.59);checks++;
get('definition').value='strict';get('definition').onchange();has('cohort-metrics','270');get('use-cohort').onclick();assert.equal(Number(get('input-n').value),Number(/<strong>(\d+)/.exec(get('cohort-metrics').innerHTML)[1]));checks++;
get('reset').onclick();get('frequency').value='2';get('frequency').onchange();has('cost-cards','49,425.04');get('reset').onclick();
get('seek').value='50';get('seek').oninput();has('scene-title','13 / 13');has('scene-art','New 9,886');
get('input-secPrice').value='0';for(const fn of listeners.get('input'))fn({target:get('input-secPrice')});has('scene-art','illustrative saving');
get('reset').onclick();get('maximum').value='100';get('deductible').value='650';get('deductible').oninput();has('payer-reconcile','raised to deductible');
await get('run-mc').onclick();has('mc-status','500 draws complete');has('mc-check','PASS');const first=get('mc-metrics').innerHTML;await get('run-mc').onclick();assert.equal(get('mc-metrics').innerHTML,first);checks++;get('export').onclick();checks++;
const exportedCSV=await exportedBlob.text();assert.ok(exportedCSV.includes('128236.64828571305'));assert.ok(exportedCSV.includes('"145.2"'));assert.ok(exportedCSV.includes('patient_example_plan'));checks+=3;
get('payer-plan').value='full';get('payer-plan').onchange();has('payer-table','$0.00');get('payer-plan').value='other';get('payer-plan').onchange();has('coverage-note','hypothetical non-public');
get('prevalence-rate').value='1000';get('prevalence-denominator').value='5800000';get('prevalence-rate').oninput();has('prevalence-equation','58,000');get('prevalence-scope').value='adult';get('prevalence-scope').onchange();has('prevalence-explanation','adults');
get('sa-age').value='17';get('sa-age').oninput();has('sa-result','not satisfied');get('sa-age').value='35';get('sa-age').oninput();has('sa-result','satisfied');
// Exercise the actual video controller with a deterministic simulated animation clock.
let clock=0;const frames=[],visited=new Set(),oldPerformance=globalThis.performance;
Object.defineProperty(globalThis,'performance',{value:{now:()=>clock},configurable:true});
globalThis.requestAnimationFrame=cb=>frames.push(cb);
const advance=ms=>{clock+=ms;const pending=frames.splice(0);pending.forEach(fn=>fn(clock));visited.add(get('scene-title').textContent.split(' / ')[0]);};
get('seek').value=0;get('seek').oninput();get('speed').value='.5';get('play').onclick();advance(1000);assert.equal(Number(get('seek').value),.5);checks++;get('pause').onclick();get('seek').value=0;get('seek').oninput();get('speed').value='2';get('play').onclick();advance(1000);assert.equal(Number(get('seek').value),2);checks++;
get('pause').onclick();advance(1000);assert.equal(Number(get('seek').value),2);checks++;
get('play').onclick();advance(1000);assert.equal(Number(get('seek').value),4);checks++;
get('seek').value=50;get('seek').oninput();advance(1000);assert.equal(Number(get('seek').value),51);has('scene-title','13 / 13');assert.equal(get('play').disabled,false);checks+=2;
get('replay').onclick();has('scene-title','1 / 13');get('pause').onclick();
const tabs=document.querySelectorAll('[data-scene]');tabs[8].onclick();assert.equal(Number(get('seek').value),31.5);has('scene-title','9 / 13');checks++;
get('seek').value=0;get('seek').oninput();get('speed').value='1';visited.clear();get('play').onclick();for(let i=0;i<51;i++)advance(1000);assert.equal(visited.size,13);assert.equal(Number(get('seek').value),51);checks+=2;
const frozen=get('scene-art').innerHTML;advance(1000);assert.equal(get('scene-art').innerHTML,frozen);checks++;
Object.defineProperty(globalThis,'performance',{value:oldPerformance,configurable:true});
const report={status:'PASS',checks,videoScenesVisited:visited.size,simulatedPlaybackSeconds:51,scope:'DOM-stub initialization and handlers plus deterministic simulated-clock video tests (speed, pause/resume, replay, scrub, scene jump, all scenes, final hold). Not browser rendering, keyboard or real download verification.'};writeFileSync(new URL('../data/presentation-validation.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(report);
