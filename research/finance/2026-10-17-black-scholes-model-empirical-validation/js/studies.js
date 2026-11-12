'use strict';
// Charts use fixed SVG viewBoxes, so hidden initialization has no zero-size measurement.
// Native disclosure events refresh progress/typesetting only; event handlers are not rebound.
function expandResearchTarget(target){
 if(!target)return false;let changed=false;
 const main=target.matches?.('section.empirical-chapter')?target.querySelector('details.study-disclosure'):null;
 if(main&&!main.open){main.open=true;main.querySelector(':scope > summary')?.setAttribute('aria-expanded','true');changed=true;}
 for(let node=target;node;node=node.parentElement){if(node.tagName==='DETAILS'&&!node.open){node.open=true;node.querySelector(':scope > summary')?.setAttribute('aria-expanded','true');changed=true;}}
 return changed;
}
function refreshStudy(details){
 requestAnimationFrame(()=>{scrollUpdate();
  if(window.MathJax?.typesetPromise){Promise.resolve(window.MathJax.startup?.promise).then(()=>window.MathJax.typesetPromise([details])).catch(()=>{});}
 });
}
document.querySelectorAll('details.study-disclosure').forEach(d=>{
 const summary=d.querySelector(':scope > summary');const sync=()=>summary?.setAttribute('aria-expanded',String(Boolean(d.open)));sync();
 d.addEventListener('toggle',()=>{sync();if(d.open)refreshStudy(d);else requestAnimationFrame(scrollUpdate);});
});
function revealStudyHash(){
 let id;try{id=decodeURIComponent(window.location?.hash?.slice(1)||'');}catch{return;}
 if(!id)return;const target=document.getElementById(id);if(!target)return;
 expandResearchTarget(target);requestAnimationFrame(()=>{target.scrollIntoView?.({block:'start',behavior:'auto'});scrollUpdate();});
}
document.addEventListener('click',e=>{
 const a=e.target.closest?.('a[href^="#"]');if(!a)return;
 let id;try{id=decodeURIComponent(a.getAttribute('href').slice(1));}catch{return;}
 const target=document.getElementById(id);if(!target||!expandResearchTarget(target))return;
 e.preventDefault();window.history?.pushState(null,'','#'+encodeURIComponent(id));
 requestAnimationFrame(()=>{target.scrollIntoView?.({block:'start',behavior:'auto'});scrollUpdate();});
});
window.addEventListener('hashchange',revealStudyHash);revealStudyHash();

