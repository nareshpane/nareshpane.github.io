import{initNavigation}from'./navigation.js';
import{initCodeControls}from'./code-controls.js';
import{initDemonstrations}from'./demonstrations.js';
import{initFoundations}from'./foundations.js';
initNavigation();initCodeControls();initDemonstrations();initFoundations();
setTimeout(()=>{if(!window.MathJax?.startup?.document){document.documentElement.classList.add('math-unavailable');const n=document.querySelector('#math-status');if(n)n.textContent='Math rendering is unavailable. Equation panels show text alternatives; the surrounding explanations remain readable.'}},8000);
