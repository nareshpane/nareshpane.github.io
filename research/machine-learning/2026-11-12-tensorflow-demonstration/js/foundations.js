/* Exact numerical walkthroughs. These are neither neural training nor timings. */
import{chart,playback}from'./demonstrations.js';
const $=id=>document.getElementById(id);
const number=x=>Math.abs(x)>0&&(Math.abs(x)<1e-5||Math.abs(x)>=1e5)?x.toExponential(5):Number(x.toPrecision(8)).toString();

export function scalarGraphValues(){
 const x=2,w=1,b=0,target=3,product=w*x,prediction=product+b,error=prediction-target,loss=.5*error**2,dw=error*x,db=error,eta=.1,newW=w-eta*dw,newB=b-eta*db,newPrediction=newW*x+newB;
 return{x,w,b,target,product,prediction,error,loss,dw,db,eta,newW,newB,newPrediction,newLoss:.5*(newPrediction-target)**2};
}
export function networkValues(){
 const x=[1,2],W1=[[1,-1],[2,.5]],b1=[0,1],W2=[[1,.25],[0,.5]],b2=[0,-1];
 const affine=(W,a,b)=>W.map((row,i)=>row.reduce((v,w,j)=>v+w*a[j],b[i]));
 const z1=affine(W1,x,b1),a1=z1.map(v=>Math.max(0,v)),z2=affine(W2,a1,b2),a2=z2.map(v=>Math.max(0,v));
 return{x,W1,b1,z1,a1,W2,b2,z2,a2,output:.5*a2[0]+.5*a2[1]};
}
export function quadraticTrace(eta,steps=25){
 if(!Number.isFinite(eta)||steps<0)throw Error('Finite learning rate and nonnegative step count required.');
 const values=[3];for(let i=0;i<steps;i++)values.push(values[i]-eta*values[i]);
 return values.map((theta,step)=>({step,theta,loss:.5*theta**2}));
}

function staged(prefix,selector,stages){
 let index=0,stop=()=>{};const container=$(prefix+'-walkthrough'),play=$(prefix+'-play'),step=$(prefix+'-step');
 function update(){
  const stage=stages[index];$(prefix+'-stage-label').textContent=`Step ${index+1} / ${stages.length} · ${stage.label}`;$(prefix+'-explanation').textContent=stage.explanation;$(prefix+'-readout').textContent=stage.values;
  const art=container.querySelector('.walkthrough-art');
  container.querySelectorAll(selector).forEach(g=>{const n=Number(g.getAttribute(prefix==='graph'?'data-graph-stage':'data-network-stage'));g.classList.toggle('is-active',n===index);g.classList.toggle('is-past',n<index);if(n===index&&art.scrollWidth>art.clientWidth){const box=g.getBoundingClientRect(),frame=art.getBoundingClientRect();art.scrollLeft+=box.left+box.width/2-frame.left-frame.width/2}});
  step.disabled=index===stages.length-1;play.disabled=index===stages.length-1;
 }
 function advance(){if(index>=stages.length-1)return false;index++;update();return index<stages.length-1}
 stop=playback(play,container,advance,()=>{index=0;update()},$(prefix+'-reset'),1000);
 step.addEventListener('click',()=>{stop();advance()});update();
}
function initGraph(){
 const v=scalarGraphValues();staged('graph','[data-graph-stage]',[
  {label:'Inputs and parameters',explanation:'The input and observed target are fixed. Weight and bias are the two adjustable values.',values:`x=${v.x}; y=${v.target}; w=${v.w}; b=${v.b}.`},
  {label:'Multiply',explanation:'The weight scales the input. Locally, changing w changes this product by x times that change.',values:`w × x = ${v.w} × ${v.x} = ${v.product}; ∂(wx)/∂w = ${v.x}.`},
  {label:'Add the bias',explanation:'Add b to get the prediction. A unit change in b changes the prediction by one unit.',values:`a = wx + b = ${v.product} + ${v.b} = ${v.prediction}; ∂a/∂b = 1.`},
  {label:'Compare with the target',explanation:'Subtract the known target. This residual is negative because the prediction is too small.',values:`a − y = ${v.prediction} − ${v.target} = ${v.error}.`},
  {label:'Evaluate the loss',explanation:'Square the residual and divide by two. The derivative with respect to a is a−y.',values:`L = ½(${v.error})² = ${v.loss}; ∂L/∂a = ${v.error}.`},
  {label:'Propagate derivatives backward',explanation:'Multiply local sensitivities along each path from the loss to a parameter. Sum contributions if a parameter has several paths.',values:`∂L/∂w = (${v.error}) × ${v.x} = ${v.dw}; ∂L/∂b = (${v.error}) × 1 = ${v.db}.`},
  {label:'Update and recompute',explanation:'Use both gradients from the original parameters. The updated forward calculation gives a smaller loss on this one observation.',values:`η=${v.eta}: w=${v.newW}, b=${v.newB}; new a=${v.newPrediction}; new L=${v.newLoss}.`}
 ]);
}
function initNetwork(){
 const v=networkValues(),vector=a=>'['+a.map(number).join(', ')+']';
 staged('network','[data-network-stage]',[
  {label:'Input vector',explanation:'One observation has two features. This is a column vector mathematically, or one row in a Keras batch.',values:`x=${vector(v.x)}; mathematical shape 2×1; batch-of-one shape 1×2.`},
  {label:'First affine transformation',explanation:'Multiply by W1, then add b1. The two dot products use the same incoming vector.',values:`W1x=[−1,3]; b1=${vector(v.b1)}; z1=${vector(v.z1)}.`},
  {label:'First activation',explanation:'ReLU keeps positive entries and replaces a negative entry by zero. This is the first nonlinear stage.',values:`ReLU(${vector(v.z1)}) = ${vector(v.a1)}.`},
  {label:'Second affine transformation',explanation:'Apply a different learned matrix and bias to the first hidden representation, not to the original input.',values:`W2a1=[1,2]; b2=${vector(v.b2)}; z2=${vector(v.z2)}.`},
  {label:'Second activation',explanation:'Both entries are positive here, so ReLU leaves them unchanged. In other regions of input space this stage can change its active pattern.',values:`ReLU(${vector(v.z2)}) = ${vector(v.a2)}.`},
  {label:'Scalar output',explanation:'The head combines the final two features. This specified example returns a number, not a class probability.',values:`0.5 × ${v.a2[0]} + 0.5 × ${v.a2[1]} = ${v.output}.`}
 ]);
}
function initOptimization(){
 const rate=$('optimization-rate'),play=$('optimization-play'),step=$('optimization-step');let values=[{step:0,theta:3,loss:4.5}],stop=()=>{};
 function update(){
  const eta=+rate.value,last=values.at(-1),inside=values.filter(v=>Math.abs(v.theta)<=4);
  chart($('optimization-landscape'),{series:[{points:Array.from({length:161},(_,i)=>{const theta=-4+i/20;return[theta,.5*theta*theta]})},{points:inside.map(v=>[v.theta,v.loss]),color:'#ae6c52',dots:true,width:1.3},{points:Math.abs(last.theta)<=4?[[last.theta,last.loss]]:[],color:'#417ba6',dots:true}],xRange:[-4,4],yRange:[0,8],xLabel:'Parameter θ',yLabel:'L(θ) = ½θ²',vertical:0,height:290});
  const logs=values.map(v=>[v.step,Math.log10(Math.max(v.loss,1e-12))]);
  chart($('optimization-history'),{series:[{points:logs,color:eta>=2?'#ae6c52':'#417ba6',dots:true}],xRange:[0,25],yRange:[Math.min(-1,...logs.map(p=>p[1]))-.5,Math.max(1,...logs.map(p=>p[1]))+.5],xLabel:'Gradient-descent updates',yLabel:'log₁₀(loss), display floor −12',height:290});
  let regime=eta<1?'Monotone contraction':eta===1?'Exact minimum after one update':eta<2?'Oscillation with contraction':eta===2?'Undamped oscillation':'Divergence for this quadratic';
  $('optimization-rate-value').textContent=eta.toFixed(2);$('optimization-readout').textContent=`Step ${last.step}/25; θ=${number(last.theta)}; gradient=${number(last.theta)}; loss=${number(last.loss)}; error multiplier 1−η=${number(1-eta)}. ${regime}.${Math.abs(last.theta)>4?' Current θ is outside the left plot; the right plot still reports its actual loss.':''}`;
  const finished=last.step>=25||last.theta===0;step.disabled=finished;play.disabled=finished;
 }
 function advance(){const last=values.at(-1);if(last.step>=25||last.theta===0)return false;const theta=last.theta-(+rate.value)*last.theta;if(!Number.isFinite(theta)||Math.abs(theta)>1e6)return false;values.push({step:last.step+1,theta,loss:.5*theta**2});update();return values.at(-1).step<25&&theta!==0}
 function reset(){values=[{step:0,theta:3,loss:4.5}];update()}
 stop=playback(play,$('optimization-lab'),advance,reset,$('optimization-reset'),650);step.addEventListener('click',()=>{stop();advance()});rate.addEventListener('input',()=>{stop();reset()});update();return update;
}
export function initFoundations(){
 initGraph();initNetwork();const redraw=initOptimization();let timer;addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(redraw,150)});
}
