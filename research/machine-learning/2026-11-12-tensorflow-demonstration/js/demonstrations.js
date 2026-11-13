/* Every plotted value is calculated here or read from a provenance-labelled asset.
   No TensorFlow model is trained or impersonated by these static laboratories. */
const base=new URL('../data/',import.meta.url),$=id=>document.getElementById(id);
const ns='http://www.w3.org/2000/svg';
const sum=a=>a.reduce((x,y)=>x+y,0),mean=a=>sum(a)/a.length;
const fmt=(x,n=3)=>Number(x).toLocaleString('en-US',{maximumFractionDigits:n});
function svgElement(tag,attributes={},text){const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attributes))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e}
function add(svg,tag,attributes,text){const e=svgElement(tag,attributes,text);svg.append(e);return e}
export function chart(svg,{series=[],xRange,yRange,xLabel='',yLabel='',band=null,vertical=null,bars=false,height=280}){
 if(!svg)return;
 const W=Math.max(320,Math.round(svg.getBoundingClientRect().width)||700),H=height,left=W<450?55:67,right=20,top=34,bottom=46;
 svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.replaceChildren();
 const all=series.flatMap(s=>s.points).concat(band?band.lower.concat(band.upper):[]).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));
 if(!all.length)return;
 let [xmin,xmax]=xRange||[Math.min(...all.map(p=>p[0])),Math.max(...all.map(p=>p[0]))];
 let [ymin,ymax]=yRange||[Math.min(...all.map(p=>p[1])),Math.max(...all.map(p=>p[1]))];
 if(xmax===xmin)xmax=xmin+1;if(ymax===ymin)ymax=ymin+1;
 if(!yRange){const margin=(ymax-ymin)*.1;ymin-=margin;ymax+=margin}
 const x=v=>left+(v-xmin)/(xmax-xmin)*(W-left-right),y=v=>H-bottom-(v-ymin)/(ymax-ymin)*(H-top-bottom);
 for(let i=0;i<5;i++){const value=ymin+(ymax-ymin)*i/4,yy=y(value);add(svg,'line',{x1:left,y1:yy,x2:W-right,y2:yy,class:'grid'});add(svg,'text',{x:left-8,y:yy+4,'text-anchor':'end'},fmt(value,Math.abs(ymax-ymin)<.01?6:Math.abs(ymax-ymin)<30?2:0))}
 for(let i=0;i<5;i++){const value=xmin+(xmax-xmin)*i/4,xx=x(value);add(svg,'text',{x:xx,y:H-bottom+22,'text-anchor':'middle'},fmt(value,xmax-xmin<30?2:0))}
 add(svg,'line',{x1:left,y1:H-bottom,x2:W-right,y2:H-bottom,class:'axis'});
 add(svg,'text',{x:left,y:18},yLabel);add(svg,'text',{x:(W+left-right)/2,y:H-6,'text-anchor':'middle'},xLabel);
 if(band){add(svg,'polygon',{points:band.upper.concat([...band.lower].reverse()).map(p=>`${x(p[0])},${y(p[1])}`).join(' '),fill:'#dbe8d1','fill-opacity':'.8'})}
 if(vertical!==null)add(svg,'line',{x1:x(vertical),x2:x(vertical),y1:top,y2:H-bottom,stroke:'#73836e','stroke-dasharray':'4 4'});
 for(const s of series){const points=s.points.filter(p=>Number.isFinite(p[1]));if(s.bars||bars){for(const p of points)add(svg,'line',{x1:x(p[0]),x2:x(p[0]),y1:y(Math.max(0,ymin)),y2:y(p[1]),stroke:s.color||'#4b7d57','stroke-width':Math.max(3,(W-left-right)/(xmax-xmin+1)*.45)})}
  else add(svg,'polyline',{points:points.map(p=>`${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join(' '),fill:'none',stroke:s.color||'#437c59','stroke-width':s.width||2.3,...(s.dash?{'stroke-dasharray':s.dash}:{}),'stroke-linejoin':'round'});
  if(s.dots)for(const p of points)add(svg,'circle',{cx:x(p[0]),cy:y(p[1]),r:4,fill:s.color||'#437c59'});
 }
 return{x,y};
}
export function playback(button,container,advance,reset,resetButton,delay=450){
 let timer=null;
 function stop(){if(timer!==null)clearInterval(timer);timer=null;button.textContent=button.dataset.label;button.setAttribute('aria-pressed','false')}
 button.dataset.label=button.textContent;button.setAttribute('aria-pressed','false');
 button.addEventListener('click',()=>{if(timer!==null){stop();return}button.textContent='Pause';button.setAttribute('aria-pressed','true');timer=setInterval(()=>{if(advance()===false)stop()},delay)});
 resetButton?.addEventListener('click',()=>{stop();reset()});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});
 new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stop()},{threshold:0}).observe(container);
 return stop;
}
function initShapes(){function update(){const b=Number($('shape-batch').value),count=b*64*64*3;$('shape-output').textContent=`(${b}, 64, 64, 3) → (${b}, 12288); ${fmt(count,0)} float32 values, ${fmt(count*4,0)} bytes. Channel offset (3,) broadcasts; row offset needs (1, 64, 1, 1).`}$('shape-batch').addEventListener('change',update);update()}
function initGradient(){const input=$('learning-rate');function update(){const eta=+input.value,w=1+2*eta,b=eta,p=2*w+b,l=.5*(p-3)**2;$('learning-rate-value').textContent=eta.toFixed(2);$('gradient-readout').textContent=`∂L/∂w = −2; ∂L/∂b = −1. New w = ${fmt(w)}, b = ${fmt(b)}; prediction = ${fmt(p)}; loss = ${fmt(l,5)} (initial 0.5).`;
 chart($('gradient-plot'),{series:[{points:Array.from({length:121},(_,i)=>[i/200,.5*(-1+5*i/200)**2])},{points:[[eta,l]],dots:true,color:'#b16a50'}],xRange:[0,.6],yRange:[0,2.1],xLabel:'Learning rate η',yLabel:'Loss after one simultaneous update',height:240})}
 input.addEventListener('input',update);$('gradient-reset').addEventListener('click',()=>{input.value='.1';update()});update();return update}
function initKernel(){const matrix=$('kernel-matrix');for(let i=1;i<=25;i++){const c=document.createElement('span');c.textContent=i;matrix.append(c)}
 function update(){const pos=+$('kernel-position').value,r=Math.floor(pos/3),c=pos%3,kind=$('kernel-kind').value,k=kind==='mean'?Array(9).fill(1/9):[1,0,-1,1,0,-1,1,0,-1];let result=0;
 [...matrix.children].forEach((cell,i)=>{const row=Math.floor(i/5),col=i%5,selected=row>=r&&row<r+3&&col>=c&&col<c+3;cell.classList.toggle('selected',selected);if(selected)result+=(i+1)*k[(row-r)*3+col-c]});
 $('kernel-position-value').textContent=`${pos+1} / 9`;$('kernel-output').textContent=`Top-left (row ${r}, column ${c}); kernel ${kind==='mean'?'all entries 1/9':'[[1,0,−1],[1,0,−1],[1,0,−1]]'}; cross-correlation = ${fmt(result,6)}.`}
 $('kernel-position').addEventListener('input',update);$('kernel-kind').addEventListener('change',update);
 playback($('kernel-play'),matrix.closest('.lab'),()=>{const p=+$('kernel-position').value;if(p===8)return false;$('kernel-position').value=p+1;update();return p+1<8},()=>{$('kernel-position').value=0;update()},$('kernel-reset'),600);update()}
function drawPixels(canvas,pixels,w,h){const context=canvas.getContext('2d');canvas.width=w;canvas.height=h;const image=context.createImageData(w,h);for(let i=0;i<pixels.length;i++){image.data[4*i]=pixels[i][0];image.data[4*i+1]=pixels[i][1];image.data[4*i+2]=pixels[i][2];image.data[4*i+3]=255}context.putImageData(image,0,0)}
function initVision(data){
 data.scenes.forEach((scene,i)=>{const o=document.createElement('option');o.value=i;o.textContent=scene.label;$('vision-scene').append(o)});
 function update(){const scene=data.scenes[+$('vision-scene').value],threshold=+$('vision-threshold').value,alpha=+$('vision-alpha').value,view=$('vision-view').value,channel=$('vision-channel').value;let tp=0,fp=0,fn=0,tn=0;
  const input=[],truth=[],result=[];
  for(let i=0;i<scene.mask.length;i++){const rgb=scene.rgb[i],yes=scene.mask[i]===1,predicted=scene.score[i]>=threshold;tp+=yes&&predicted;fp+=!yes&&predicted;fn+=yes&&!predicted;tn+=!yes&&!predicted;
   input.push(channel==='rgb'?rgb:[rgb[+channel],rgb[+channel],rgb[+channel]]);truth.push(yes?[71,122,68]:[240,240,226]);
   if(view==='prediction')result.push(predicted?[71,122,68]:[240,240,226]);
   else if(view==='overlay')result.push(predicted?rgb.map((v,j)=>v*(1-alpha)+[58,190,87][j]*alpha):rgb);
   else if(view==='errors')result.push(predicted&&!yes?[190,83,65]:yes&&!predicted?[63,111,179]:yes?[80,133,70]:[240,240,226]);
   else{const g=Math.round(255*scene.score[i]);result.push([g,g,g])}
  }
  drawPixels($('vision-rgb'),input,scene.width,scene.height);drawPixels($('vision-truth'),truth,scene.width,scene.height);drawPixels($('vision-result'),result,scene.width,scene.height);
  const precision=tp+fp?tp/(tp+fp):0,recall=tp+fn?tp/(tp+fn):0,iou=tp+fp+fn?tp/(tp+fp+fn):1,dice=2*tp+fp+fn?2*tp/(2*tp+fp+fn):1;
  $('vision-threshold-value').textContent=threshold.toFixed(2);$('vision-alpha').disabled=view!=='overlay';
  $('vision-readout').textContent=`TP ${tp} · FP ${fp} · FN ${fn} · TN ${tn} | accuracy ${fmt((tp+tn)/scene.mask.length)} · precision ${fmt(precision)} · recall ${fmt(recall)} · IoU ${fmt(iou)} · Dice ${fmt(dice)} | true cover ${fmt(100*(tp+fn)/scene.mask.length,1)}%; marked cover ${fmt(100*(tp+fp)/scene.mask.length,1)}%.`;
 }
 ['vision-scene','vision-channel','vision-view','vision-threshold','vision-alpha'].forEach(id=>$(id).addEventListener('input',update));update();
}
const tokenize=text=>text.toLowerCase().replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g,'').trim().split(/\s+/u).filter(Boolean);
function initNlp(data){const index=new Map(data.vocabulary.map((w,i)=>[w,i]));data.examples.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=e.text.slice(0,72)+(e.text.length>72?'…':'');$('nlp-preset').append(o)});
 function update(){const text=$('nlp-input').value,tokens=tokenize(text),ids=tokens.map(t=>index.get(t)??1),encoded=ids.slice(0,data.sequence_length);while(encoded.length<data.sequence_length)encoded.push(0);
  $('nlp-tokens').replaceChildren();tokens.slice(0,48).forEach((token,i)=>{const c=document.createElement('span'),label=document.createElement('span'),id=document.createElement('small');label.textContent=token;id.textContent=`ID ${ids[i]}`;c.append(label,id);$('nlp-tokens').append(c)});
  $('nlp-encoding').textContent=`${tokens.length} tokens; vocabulary ${data.vocabulary.length} entries; encoded to length ${data.sequence_length}: [${encoded.join(', ')}].${ids.length>48?' Sequence truncated for display.':''}`;
  const exact=data.examples.find(e=>e.text===text),scores=$('nlp-probabilities');scores.replaceChildren();
  if(exact){exact.probabilities.forEach((p,i)=>{const row=document.createElement('div');row.className='score-row';const name=document.createElement('span');name.textContent=data.classes[i];const track=document.createElement('div');track.className='score-track';const fill=document.createElement('div');fill.className='score-fill';fill.style.width=`${p*100}%`;track.append(fill);const value=document.createElement('span');value.textContent=`${fmt(p*100,1)}%`;row.append(name,track,value);scores.append(row)});$('nlp-result').textContent=`Executed naïve Bayes prediction: ${exact.predicted_label}. Intended class: ${exact.true_label??'ambiguous / unspecified'}. Scores are for this exact stored sentence.`}
  else $('nlp-result').textContent=tokens.length?'Tokenization-only mode. This edited message has no stored fitted prediction. Choose a preset to inspect genuine baseline probabilities.':'Enter a message or select a preset. Empty input has no meaningful classification.';
 }
 function preset(){$('nlp-input').value=data.examples[+$('nlp-preset').value].text;update()}
 $('nlp-use').addEventListener('click',preset);$('nlp-preset').addEventListener('change',preset);$('nlp-input').addEventListener('input',update);preset();
 const tbody=$('nlp-errors').querySelector('tbody');tbody.replaceChildren();
 for(const e of data.examples){const row=document.createElement('tr');for(const value of[e.text,e.true_label??'Ambiguous',e.predicted_label,`${fmt(100*Math.max(...e.probabilities),1)}%`]){const td=document.createElement('td');td.textContent=value;row.append(td)}tbody.append(row)}
}
function acf(values,maxLag=28){const m=mean(values),v=values.map(x=>x-m),den=sum(v.map(x=>x*x));if(den===0)return null;return Array.from({length:maxLag},(_,i)=>{const lag=i+1;let n=0;for(let j=lag;j<v.length;j++)n+=v[j]*v[j-lag];return[lag,n/den]})}
function initForecast(data){
 const labels={trend:'Trend',weekly:'Weekly',annual:'Annual',weather:'Weather',noise:'Noise',shocks:'Shocks'};
 function decomposition(){const keys=[...document.querySelectorAll('#component-controls input:checked')].map(i=>i.value),values=data.demand.map((_,i)=>sum(keys.map(k=>data.components[k][i])));chart($('demand-plot'),{series:[{points:values.map((v,i)=>[i,v])}],xLabel:'Day index (six simulated years)',yLabel:'Selected component sum · MWh/day',height:300});const correlations=acf(values);if(correlations)chart($('acf-plot'),{series:[{points:correlations,bars:true}],xRange:[0,29],yRange:[-1,1],xLabel:'Lag in days',yLabel:'Sample autocorrelation',height:220});else{$('acf-plot').replaceChildren();add($('acf-plot'),'text',{x:25,y:50},'ACF undefined: selected series is constant.')};$('demand-plot').setAttribute('aria-label',`Simulated sum of ${keys.map(k=>labels[k]).join(', ')||'no components'} in megawatt-hours per day`)}
 document.querySelectorAll('#component-controls input').forEach(i=>i.addEventListener('change',decomposition));
 const origin=$('forecast-origin');origin.max=data.examples.length-1;
 function update(){const e=data.examples[+origin.value],h=+$('forecast-horizon').value,L=e.history.length,actual=e.history.map((v,i)=>[i-L,v]).concat(e.actual.slice(0,h).map((v,i)=>[i,v]));const series=[{points:actual,color:'#437c59'},{points:e.seasonal_naive.slice(0,h).map((v,i)=>[i,v]),color:'#ad624e',dots:true}];
 if($('forecast-last').checked)series.push({points:e.last_value.slice(0,h).map((v,i)=>[i,v]),color:'#507caa',dash:'4 3'});
 const band=$('forecast-interval').checked?{lower:e.lower.slice(0,h).map((v,i)=>[i,v]),upper:e.upper.slice(0,h).map((v,i)=>[i,v])}:null;
 chart($('forecast-plot'),{series,band,xRange:[-L,7],vertical:0,xLabel:'Days from origin (0 = first future target)',yLabel:'Simulated demand · MWh/day',height:300});
 chart($('horizon-plot'),{series:[{points:data.metrics.seasonal_naive.by_horizon_mae.map((v,i)=>[i+1,v]),color:'#ad624e',dots:true},{points:data.metrics.last_value.by_horizon_mae.map((v,i)=>[i+1,v]),color:'#507caa',dots:true}],xRange:[1,7],xLabel:'Forecast horizon in days',yLabel:'All test origins: MAE · MWh/day',height:250});
 chart($('forecast-residual-plot'),{series:[{points:data.residual_autocorrelation_horizon1.slice(1).map((v,i)=>[i+1,v]),bars:true,color:'#ad624e'}],xRange:[0,13],yRange:[-1,1],xLabel:'Lag in daily forecast origins',yLabel:'Horizon-one residual ACF',height:220});
 const date=new Date(Date.parse(`${data.start_date}T00:00:00Z`)+e.origin*86400000).toISOString().slice(0,10);$('forecast-origin-value').textContent=`${+origin.value+1} / ${data.examples.length}`;
 const mae=mean(e.actual.slice(0,h).map((v,i)=>Math.abs(v-e.seasonal_naive[i]))),coverage=mean(e.actual.slice(0,h).map((v,i)=>Number(v>=e.lower[i]&&v<=e.upper[i])));
 $('forecast-readout').textContent=`Origin o=${e.origin} (${date}); history indices ${e.origin-L}…${e.origin-1}; targets ${e.origin}…${e.origin+h-1}. This displayed path: seasonal-naïve MAE ${fmt(mae,2)} MWh/day; interval coverage ${fmt(coverage*100,1)}%. Aggregate scores use all ${data.n_test_origins} daily test origins, not just ${data.examples.length} gallery paths.`;
 }
 ['forecast-origin','forecast-horizon','forecast-last','forecast-interval'].forEach(id=>$(id).addEventListener('input',update));
 const tbody=$('forecast-metrics').querySelector('tbody');tbody.replaceChildren();for(const[name,label]of[['last_value','Last value'],['seasonal_naive','Seasonal naïve']]){const m=data.metrics[name],row=document.createElement('tr');for(const value of[label,fmt(m.mae,3),fmt(m.rmse,3),`${fmt(m.coverage*100,2)}%`]){const td=document.createElement('td');td.textContent=value;row.append(td)}tbody.append(row)}
 playback($('forecast-play'),$('forecast-lab'),()=>{const next=+origin.value+1;if(next>=data.examples.length)return false;origin.value=next;update();return next<data.examples.length-1},()=>{origin.value=0;update()},$('forecast-reset'),650);
 decomposition();update();return()=>{decomposition();update()}
}
function finiteDifference(kappa,nx){const dx=1/(nx-1),steps=Math.ceil(kappa/(.4*dx*dx)),dt=1/steps,r=kappa*dt/(dx*dx),fields=[Array.from({length:nx},(_,i)=>i===0||i===nx-1?0:Math.sin(Math.PI*i/(nx-1)))];for(let j=0;j<steps;j++){const previous=fields[j],next=Array(nx).fill(0);for(let i=1;i<nx-1;i++)next[i]=(1-2*r)*previous[i]+r*(previous[i-1]+previous[i+1]);fields.push(next)}return{fields,steps,dt,r}}
function initHeat(data){let cached='',solver=null;const rod=$('heat-rod');for(let i=0;i<51;i++)rod.append(document.createElement('span'));
 function update(){const t=+$('heat-time').value,k=+$('heat-kappa').value,nx=+$('heat-resolution').value,location=+$('heat-location').value,key=`${k}/${nx}`;if(key!==cached){cached=key;solver=finiteDifference(k,nx)}const x=Array.from({length:nx},(_,i)=>i/(nx-1)),exact=x.map(x=>Math.exp(-k*Math.PI**2*t)*Math.sin(Math.PI*x)),step=t*solver.steps,lo=Math.min(solver.steps,Math.floor(step)),hi=Math.min(solver.steps,lo+1),a=step-lo,fd=solver.fields[lo].map((v,i)=>v*(1-a)+solver.fields[hi][i]*a),error=fd.map((v,i)=>Math.abs(v-exact[i]));
 [...rod.children].forEach((c,i)=>{const u=Math.exp(-k*Math.PI**2*t)*Math.sin(Math.PI*i/50);c.style.backgroundColor=`rgb(${Math.round(220+(194-220)*u)},${Math.round(235+(69-235)*u)},${Math.round(244+(48-244)*u)})`});
 const series=[{points:x.map((v,i)=>[v,exact[i]])}];if($('heat-show-fd').checked)series.push({points:x.map((v,i)=>[v,fd[i]]),color:'#ad624e',dash:'5 3'});
 chart($('heat-profile'),{series,xRange:[0,1],yRange:[0,1.05],xLabel:'Dimensionless position x',yLabel:'Dimensionless temperature u',vertical:location,height:300});
 chart($('heat-error'),{series:[{points:x.map((v,i)=>[v,error[i]]),color:'#ad624e'}],xRange:[0,1],yRange:[0,Math.max(1e-6,Math.max(...error)*1.15)],xLabel:'Dimensionless position x',yLabel:'Absolute FTCS error',height:220});
 const value=Math.exp(-k*Math.PI**2*t)*Math.sin(Math.PI*location);$('heat-time-value').textContent=t.toFixed(2);$('heat-kappa-value').textContent=k.toFixed(2);$('heat-readout').textContent=`t=${t.toFixed(2)}, κ=${k.toFixed(2)}, x=${location.toFixed(2)} → exact u=${fmt(value,6)}. Stable FTCS: ${nx} points, Δt=${fmt(solver.dt,7)}, r=${fmt(solver.r,6)}, max absolute profile error=${fmt(Math.max(...error),8)}. Bundled κ=0.1 reference grid global relative L²=${fmt(data.metrics.relative_l2,8)}.`;
 }
 ['heat-time','heat-kappa','heat-resolution','heat-location','heat-show-fd'].forEach(id=>$(id).addEventListener('input',update));
 playback($('heat-play'),$('heat-lab'),()=>{const t=+$('heat-time').value;if(t>=1)return false;$('heat-time').value=Math.min(1,t+.01).toFixed(2);update();return+$('heat-time').value<1},()=>{$('heat-time').value=0;update()},$('heat-reset'),90);
 function stability(){const r=+$('stability-r').value,g=1-4*r*Math.sin(39*Math.PI/80)**2,points=[];for(let j=0;j<=80;j++){const logAmp=-6+j*Math.log10(Math.abs(g));if(logAmp>6)break;points.push([j,Math.max(-18,logAmp)])}$('stability-r-value').textContent=r.toFixed(2);chart($('stability-plot'),{series:[{points,color:r>.5?'#ad624e':'#437c59'}],xRange:[0,80],yRange:[-18,6],xLabel:'Time steps',yLabel:'log₁₀(error-mode amplitude)',height:240});$('stability-readout').textContent=`r=${r.toFixed(2)}; mode multiplier G=${fmt(g,6)}; |G|=${fmt(Math.abs(g),6)}. ${r<=.5?'All modes satisfy the standard stable FTCS bound.':'The scheme violates r≤0.5; high-frequency error grows.'} Plot floor = 10⁻¹⁸; upper stop = 10⁶.`}
 $('stability-r').addEventListener('input',stability);update();stability();return()=>{update();stability()}
}
async function load(name,statusId,initialize){try{const response=await fetch(new URL(name,base));if(!response.ok)throw Error(`HTTP ${response.status}`);const data=await response.json();return initialize(data)}catch(error){const status=$(statusId);if(status)status.textContent=`Laboratory data unavailable (${error.message}). Serve this page over HTTP or GitHub Pages; file: access may block JSON. The article and downloadable source remain available.`;return null}}
export async function initDemonstrations(){
 const redraw=[initGradient()];initShapes();initKernel();
 const results=await Promise.allSettled([load('vision-demo.json','vision-status',initVision),load('nlp-demo.json','nlp-status',initNlp),load('forecasting-demo.json','forecast-status',initForecast),load('scientific-demo.json','heat-status',initHeat)]);
 results.forEach(r=>{if(r.status==='fulfilled'&&typeof r.value==='function')redraw.push(r.value)});
 let timer;addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(()=>redraw.forEach(f=>f()),150)});
}
