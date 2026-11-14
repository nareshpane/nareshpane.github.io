import{$,fmt,svg,chart,readouts,playback,colors,linePath}from'./utils.js';
import{riverSimulator,sensorAt,observationLoss}from'./river-model.js';

export function initRiverLab(d){const lab=$('river-lab');let simulation=null,activePhi=null,lastKey='',player;
 const times=d.observation_indices.map(i=>d.t[i]);
 const parameters=()=>({location:activePhi?.[0]??+$('river-location').value,intensity:activePhi?.[1]??+$('river-intensity').value,velocity:+$('river-velocity').value,dispersion:+$('river-dispersion').value});
 function calculate(){const p=parameters(),key=JSON.stringify(p);if(key!==lastKey){simulation=riverSimulator(p);lastKey=key}for(const name of ['location','intensity','velocity','dispersion'])$('river-'+name+'-value').textContent=fmt(p[name],name==='dispersion'?4:3)}
 function scene(profile,phi){const W=Math.max(280,Math.min(500,$('river-scene').clientWidth)),H=128,left=20,length=W-40,center=u=>56+6*Math.sin(3*Math.PI*u),points=simulation.x.map(x=>[left+x*length,center(x)]),path=linePath(points),norm=Math.max(3,...simulation.field.at(-1));
  let body='<defs><marker id="river-flow-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10" fill="#739bb6"/></marker><linearGradient id="river-plume-gradient" x1="0" x2="1">';
  profile.forEach((value,j)=>body+=`<stop offset="${j/(profile.length-1)}" stop-color="#b77960" stop-opacity="${Math.min(.82,Math.max(0,value/norm))}"/>`);
  body+='</linearGradient></defs>';
  body+=`<path d="${path}" fill="none" stroke="#deeff9" stroke-width="29" stroke-linecap="round"/><path d="${path}" fill="none" stroke="url(#river-plume-gradient)" stroke-width="22" stroke-linecap="round"/>`;
  for(const u of [.12,.54,.92]){const x=left+u*length,y=center(u);body+=`<path d="M${x-12} ${y}h24" stroke="#739bb6" stroke-width="1.3" marker-end="url(#river-flow-arrow)"/>`}
  d.sensors.forEach((index,k)=>{const u=index/60,x=left+u*length,y=center(u);body+=`<path d="M${x} 27V${y+13}" stroke="${colors[k]}" stroke-dasharray="3 3"/><circle cx="${x}" cy="23" r="5" fill="${colors[k]}"/><text x="${x}" y="12" text-anchor="middle" style="font-size:11px">S${k+1} · ${fmt(u,2)}</text>`});
  const sourceX=left+phi.location*length;body+=`<path d="M${sourceX} 91V${center(phi.location)+10}" stroke="#a46a51" stroke-width="1.5"/><path d="M${sourceX} 88l5 5-5 5-5-5Z" fill="#b57a60"/><text x="${Math.max(75,Math.min(W-75,sourceX))}" y="116" text-anchor="middle" style="font-size:11px">ξ=${fmt(phi.location,3)} · q=${fmt(phi.intensity,3)}</text><text x="${W-15}" y="84" text-anchor="end" style="font-size:10px">Flow → · 1D domain</text>`;
  $('river-scene').innerHTML=svg(body,W,H,'Continuous color ribbon interpolates the computed concentration, with matched station colors');
 }
 function render(frame){const inference=$('river-mode').value==='inference';let record=null;
  if(inference){record=d.pytorch.history[Math.min(frame,d.pytorch.history.length-1)];activePhi=record.slice(0,2);$('river-location').value=activePhi[0];$('river-intensity').value=activePhi[1]}
  calculate();const fraction=inference?1:frame/40,index=Math.round(fraction*(simulation.t.length-1)),time=simulation.t[index],phi=parameters(),profile=simulation.field[index],loss=observationLoss(simulation,d);
  scene(profile,phi);
  $('river-time').value=inference?1.2:frame/40*1.2;$('river-time').disabled=inference;$('river-time-value').textContent=fmt(time,3);
  chart($('river-profile-plot'),[{name:`Live profile · t=${fmt(time,3)}`,points:simulation.x.map((x,j)=>[x,profile[j]])}],{xlabel:'x (dimensionless)',ylabel:'Concentration c',ymin:0,ymax:Math.max(3,...profile)*1.1,height:160,compact:true,legend:false,cursorX:phi.location});
  const selected=$('river-sensor').value,lines=[];
  d.sensors.forEach((station,k)=>{if(selected!=='all'&&+selected!==k)return;lines.push({name:`S${k+1} predicted`,points:simulation.t.filter((_,i)=>i%5===0||i===simulation.t.length-1).map(t=>[t,sensorAt(simulation,t,station)]),color:colors[k]},{name:`S${k+1} observed`,points:times.map((t,i)=>[t,d.observations[i][k]]),dots:true,color:colors[k]})});
  chart($('river-sensor-plot'),lines,{xlabel:'Time (dimensionless)',ylabel:'Sensor c',ymin:0,height:160,compact:true,legend:false,cursorX:time});
  const diagnostic=[{name:'Stored intensity-profiled minimum',points:d.profile.map(row=>[row[0],row[2]]),color:colors[2]}];
  if(record)diagnostic.push({name:'Recorded fit iterate',points:[[record[0],record[2]]],dots:true,color:colors[1]});
  chart($('river-loss-plot'),diagnostic,{xlabel:'Source position ξ (original physics)',ylabel:'Profile objective',ymin:0,height:160,compact:true,legend:false,cursorX:phi.location});
  const originalPhysics=Math.abs(phi.velocity-.35)<1e-12&&Math.abs(phi.dispersion-.005)<1e-12;
  $('river-reference-note').textContent=originalPhysics?'Recorded profile · v=.35, D=.005; q optimized separately at each ξ.':'Fixed reference only · current v/D differ. This recorded profile is not recomputed.';
  readouts($('river-readouts'),[['Displayed t',fmt(time,3)],['Combined A + 2B',fmt(simulation.combined,4)],['Current fitting loss',fmt(loss,4)],['Source (ξ, q)',`(${fmt(phi.location,3)}, ${fmt(phi.intensity,3)})`]]);
  $('river-position').textContent=inference?`Fit iterate ${frame+1} / ${d.pytorch.history.length}`:`Transport ${frame} / 40`;
  $('river-stage').textContent=inference?'Recorded PyTorch/L-BFGS parameters; fields are recomputed by the browser. The dot is recorded fitting loss, not a confidence interval.':'Live JavaScript transport. All live curves share the sliders; observations and the source-profile reference stay recorded. No Python optimization is rerun.';
  lab.dataset.mode=inference?'recorded-inference':'browser-transport';lab.dataset.loss=loss;lab.dataset.combined=simulation.combined;lab.dataset.source=JSON.stringify([phi.location,phi.intensity]);
 }
 player=playback(lab,{count:()=>$('river-mode').value==='inference'?d.pytorch.history.length:41,render,interval:450});
 for(const key of ['location','intensity','velocity','dispersion'])$('river-'+key).addEventListener('input',()=>{$('river-mode').value='transport';activePhi=null;player.set(40)});
 $('river-mode').addEventListener('input',()=>{if($('river-mode').value==='inference'){$('river-velocity').value=.35;$('river-dispersion').value=.005}player.set(0)});
 $('river-time').addEventListener('input',()=>{$('river-mode').value='transport';player.set(Math.round(+$('river-time').value/1.2*40))});
 $('river-sensor').addEventListener('input',()=>player.render());let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>player.render(),100)});player.set(40);
}
