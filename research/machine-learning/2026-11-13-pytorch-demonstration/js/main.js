import{initNavigation}from'./navigation.js';
import{initCodeControls}from'./code-controls.js';
import{initFoundations}from'./foundations.js';
import{initDemonstrations}from'./demonstrations.js';
import{initComparison}from'./comparison.js';
initNavigation();initCodeControls();initFoundations();initDemonstrations();initComparison();
window.MathJax?.startup?.promise?.then(()=>{if(location.hash){const target=document.getElementById(location.hash.slice(1));target?.scrollIntoView({block:'start'})}dispatchEvent(new Event('resize'))});
setTimeout(()=>{if(!document.documentElement.classList.contains('math-ready')){document.querySelector('#math-status').textContent='Native mathematical notation is displayed while the optional MathJax rendering is unavailable.'}},8000);
