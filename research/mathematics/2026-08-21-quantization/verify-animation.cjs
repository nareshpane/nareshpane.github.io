/* Execute the actual film/controller against a small DOM adapter.
   Browser screenshots separately validate fonts, layout and external SVG rendering. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const M=require('./quantization-math.js');
const source=fs.readFileSync(__dirname+'/intro-story.js','utf8');
const asset=fs.readFileSync(__dirname+'/university-ensemble.svg','utf8');
let checks=0;
function check(test,message){assert.ok(test,message);checks++;}
class Element{
  constructor(tag='div'){this.tag=tag;this.attrs={};this.style={};this.children=[];this.textContent='';this.value='';this.dataset={};}
  setAttribute(k,v){this.attrs[k]=String(v);}
  getAttribute(k){return this.attrs[k]??null;}
  appendChild(n){this.children.push(n);n.parentElement=this;return n;}
  append(...nodes){nodes.forEach(n=>this.appendChild(n));}
  remove(){this.parentElement.children=this.parentElement.children.filter(n=>n!==this);}
  cloneNode(){return new Element(this.tag);}
  getBoundingClientRect(){return {width:600,height:120};}
  querySelectorAll(){return this.children.filter(n=>n.attrs['data-story-scene']!==undefined);}
  serialize(){return JSON.stringify([this.tag,this.attrs,this.style,this.textContent,this.children.map(n=>n.serialize())]);}
}
for(const small of [false,true])for(const reduced of [false,true]){
  const ids=Object.fromEntries(['intro-svg','intro-chapter','intro-text','intro-timeline','intro-time','intro-play','intro-restart','intro-speed','intro-animation'].map(id=>[id,new Element()]));
  ids['intro-speed'].value='1';const caption=new Element(),parent=new Element();parent.append(caption);
  const starts=[0,6,15,24,33,42,52],buttons=starts.map(start=>{const n=new Element();n.dataset.time=String(start);return n;});
  let now=0,next=0,callbacks=new Map();
  const document={getElementById:id=>ids[id],createElementNS:(_,tag)=>new Element(tag),createElement:tag=>new Element(tag),querySelector:()=>caption,querySelectorAll:()=>buttons,addEventListener(){}};
  vm.runInNewContext(source,{document,window:{QuantizationMath:M},matchMedia:q=>({matches:q.includes('reduced')?reduced:small,addEventListener(){}}),ResizeObserver:class{observe(){}},IntersectionObserver:class{observe(){}},performance:{now:()=>now},requestAnimationFrame:cb=>{callbacks.set(++next,cb);return next;},cancelAnimationFrame:id=>callbacks.delete(id)});
  const svg=ids['intro-svg'],range=ids['intro-timeline'];
  const seek=t=>{range.value=t;range.oninput();};
  const scene=i=>svg.children.find(n=>n.attrs['data-story-scene']===String(i));
  const active=()=>svg.children.filter(n=>n.style.display!=='none');
  const advance=ms=>{now+=ms;const pending=[...callbacks.values()];callbacks.clear();pending.forEach(cb=>cb(now));};
  check(svg.attrs.viewBox===(small?'0 0 640 650':'0 0 960 540'),'responsive frame');
  // Arbitrary scrubs, including every transition, must be repeatable in either direction.
  const times=[0,.25,5.7,6,12,14.8,15,20,23.7,24,29,32.8,33,38,41.8,42,47,48,51.8,52,57,60];
  const snapshots=new Map();
  for(const t of times){seek(t);const i=starts.findLastIndex(s=>t>=s);snapshots.set(t,scene(i).serialize());check(svg.attrs['data-chapter']===String(i),'correct seek chapter');check(active().length<=(reduced?1:2),'no scene remnants');}
  for(const t of [...times].reverse()){seek(t);const i=starts.findLastIndex(s=>t>=s);check(scene(i).serialize()===snapshots.get(t),'history-independent geometry at '+t);}
  starts.forEach((t,i)=>{buttons[i].onclick();check(active().length===1&&active()[0]===scene(i)&&scene(i).attrs.opacity==='1','exact chapter boundary');});
  // Play through the full 60 s in 50 ms steps, matching an ordinary frame clock.
  ids['intro-restart'].onclick();ids['intro-play'].onclick();
  for(let i=0;i<1200;i++)advance(50);
  check(svg.attrs['data-time']==='60.000'&&ids['intro-play'].attrs['aria-pressed']==='false','completion stops at 60');
  check(callbacks.size===0,'no outstanding animation loop');
  for(const speed of [.5,1,1.5]){
    ids['intro-restart'].onclick();ids['intro-speed'].value=String(speed);ids['intro-play'].onclick();advance(2000);
    check(Math.abs(+range.value-2*speed)<1e-9,'speed scales every visual through the shared clock');
    ids['intro-play'].onclick();const frozen=svg.serialize();advance(3000);check(svg.serialize()===frozen,'pause freezes all geometry');
  }
  // Every reusable character fragment must resolve inside the local SVG asset.
  function visit(n){if(n.tag==='use'){const id=n.attrs.href.split('#')[1];check(asset.includes('id="'+id+'"'),'resolved cast fragment '+id);}n.children.forEach(visit);}
  visit(svg);
  ids['intro-restart'].onclick();check(svg.attrs['data-time']==='0.000'&&active()[0]===scene(0),'restart restores campus');
}
check(Math.abs(2.4-M.round(2.4))>.399999,'coarse error');
check(Math.abs(2.4-.25*M.round(2.4/.25))<.100001,'fine selected error');
const e=M.affine(.73,-1.24,2.61,4,'asymmetric');
check(e.q===8&&Math.abs(e.xhat-.77)<1e-12&&e.q.toString(2)==='1000','affine example');
console.log(`Passed ${checks} animation checks: desktop/mobile, reduced motion, deterministic seeking, transitions, complete playback, speed, pause, restart and SVG fragments.`);
