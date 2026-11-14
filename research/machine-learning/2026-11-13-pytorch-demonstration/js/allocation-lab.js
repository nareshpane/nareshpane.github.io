import{$,fmt,svg,chart,readouts,playback,colors}from'./utils.js';
import{softmax,dispatch,browserPolicy,empiricalCvar}from'./allocation-model.js';

export function initAllocationLab(d){const lab=$('allocation-lab');$('allocation-scenario').max=d.test_scenarios.length-1;let localHistory=[],player;
 const capacity=()=>d.capacity.map((v,j)=>j<+$('allocation-stations').value?Math.max(1,Math.round(v*+$('allocation-capacity').value)):0);
 const rebuild=()=>{localHistory=browserPolicy(d,capacity(),+$('allocation-risk').value,+$('allocation-fairness').value)};
 const count=()=>$('allocation-strategy').value==='browser'?localHistory.length:$('allocation-strategy').value==='pytorch_relaxed'?d.history.length:1;
 function city(demand,result,cap){const point=p=>[44+p[0]*532,12+p[1]*245],positions=d.districts.map(point),stationPositions=d.stations.map(point);let body='<defs>';
  for(let j=0;j<3;j++)body+=`<marker id="dispatch-flow-arrow-${j}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10" fill="${colors[j]}"/></marker>`;
  body+='</defs>';
  for(const x of [90,195,300,405,510])body+=`<path d="M${x} 25V252" stroke="#eef1f3" stroke-width="6"/>`;
  for(const y of [45,115,185,245])body+=`<path d="M23 ${y}H580" stroke="#eef1f3" stroke-width="6"/>`;
  for(let i=0;i<8;i++)for(let j=0;j<3;j++)if(result.flow[i][j]){const[a,b]=stationPositions[j],[x,y]=positions[i],length=Math.hypot(x-a,y-b),radius=12+Math.sqrt(demand[i])*1.8,start=23/length,end=(radius+3)/length;
   body+=`<path d="M${a+(x-a)*start} ${b+(y-b)*start}L${x-(x-a)*end} ${y-(y-b)*end}" fill="none" stroke="${colors[j]}" stroke-width="${1+Math.sqrt(result.flow[i][j])*.75}" opacity=".75" marker-end="url(#dispatch-flow-arrow-${j})"/>`}
  for(let i=0;i<8;i++){const[x,y]=positions[i],radius=12+Math.sqrt(demand[i])*1.8;body+=`<circle cx="${x}" cy="${y}" r="${radius}" fill="${result.unserved[i]?'#fff0eb':'#edf6fc'}" stroke="${result.unserved[i]?'#b77967':'#91b2c9'}"/><text x="${x}" y="${y-2}" text-anchor="middle" style="font-size:12px;font-weight:600">D${i+1}</text><text x="${x}" y="${y+11}" text-anchor="middle" style="font-size:11.5px">d=${demand[i]}</text>`;
   if(result.unserved[i])body+=`<text x="${x}" y="${y+radius+12}" text-anchor="middle" style="font-size:11.5px;fill:#9d5147">${result.unserved[i]} unserved</text>`}
  for(let j=0;j<3;j++){const[x,y]=stationPositions[j],full=cap[j]>0&&result.loads[j]===cap[j],fraction=cap[j]?result.loads[j]/cap[j]:0,label=cap[j]?`S${j+1} · ${result.loads[j]} / ${cap[j]}${full?' · full':''}`:`S${j+1} · unavailable`;
   body+=`<rect x="${x-54}" y="${y-16}" width="108" height="34" rx="6" fill="${cap[j]?'#fff7f0':'#f1f3f5'}" stroke="${colors[j]}"/><text x="${x}" y="${y-1}" text-anchor="middle" style="font-size:12px;font-weight:600">${label}</text><rect x="${x-42}" y="${y+7}" width="84" height="4" rx="2" fill="#dde6ec"/><rect x="${x-42}" y="${y+7}" width="${84*fraction}" height="4" rx="2" fill="${colors[j]}"/>`}
  $('allocation-diagram').innerHTML=svg(body,620,270,'Eight synthetic districts and three stations with integer flows, load bars, and explicit shortages');
 }
 function history(strategy,frame,objective,iteration){if(strategy==='browser'||strategy==='pytorch_relaxed'){
   const rows=strategy==='browser'?localHistory:d.history;
   chart($('allocation-history-plot'),[{name:strategy==='browser'?'Browser gradient-descent objective':'Recorded PyTorch objective',points:rows.map(row=>[row.step,row.objective]),color:colors[0]},{name:'Current iterate',points:[[iteration,objective]],dots:true,color:colors[1]}],{xlabel:'Optimization step',ylabel:'Relaxed objective',height:160,compact:true,legend:false,cursorX:iteration});
   $('allocation-history-note').textContent=strategy==='browser'?'Live JS calculation · 48 stored train scenarios; no fresh PyTorch fit.':'Recorded original fit · capacities 12/10/11, λ=.7, μ=.25; displayed dispatch is evaluated separately.';
  }else if(strategy==='scipy_relaxed'){
   chart($('allocation-history-plot'),[{name:'Recorded SciPy final objective only',points:[[d.scipy.iterations,d.scipy.objective]],dots:true}],{xlabel:'Optimizer iteration',ylabel:'Relaxed objective',height:160,compact:true,legend:false,markers:true});
   $('allocation-history-note').textContent='Only the final SciPy objective is stored; no iteration path is invented.';
  }else{
   $('allocation-history-plot').innerHTML='<div class="history-empty"><div class="pipeline"><span>Preference rule</span><b>→</b><span>Feasible dispatch</span></div><span>This direct rule has no optimization trajectory.</span></div>';
   $('allocation-history-note').textContent='Choose PyTorch history or the browser optimizer to inspect real saved/calculated iterations.';
  }
 }
 function render(frame){const strategy=$('allocation-strategy').value,scenario=+$('allocation-scenario').value,cap=capacity(),demand=[...d.test_scenarios[scenario]],extra=Math.round(+$('allocation-concentration').value*8);demand[7]+=extra;
  let policy=d.policies[strategy],objective=null,iteration=null;
  if(strategy==='browser'){const row=localHistory[Math.min(frame,localHistory.length-1)];policy=row.policy;objective=row.objective;iteration=row.step}
  if(strategy==='pytorch_relaxed'){const row=d.history[Math.min(frame,d.history.length-1)];policy=Array.from({length:8},(_,i)=>softmax(row.vector.slice(i*3,i*3+3)));objective=row.objective;iteration=row.step}
  const result=dispatch(demand,policy,cap,d.cost),totalCapacity=cap.reduce((s,v)=>s+v,0),utilization=result.served/Math.max(1,totalCapacity);
  city(demand,result,cap);
  readouts($('allocation-readouts'),[['Total demand',result.total],['Served demand',result.served],['Unserved demand',result.lost],['Available capacity',totalCapacity],['Cost / demand',fmt(result.cost,3)],['Serving cost',fmt(result.serviceCost,2)],['Unserved penalty',fmt(result.shortageCost,2)],['Total proxy cost',fmt(result.totalCost,2)],['District penalized SD',fmt(result.disparity,3)],['Capacity utilization',fmt(100*utilization,1)+'%'],['Capacity violations',result.violations]]);
  $('allocation-scenario-value').textContent=`#${scenario+1} / ${d.test_scenarios.length}`;$('allocation-capacity-value').textContent=fmt(+$('allocation-capacity').value,2);$('allocation-risk-value').textContent=fmt(+$('allocation-risk').value,2);$('allocation-fairness-value').textContent=fmt(+$('allocation-fairness').value,2);$('allocation-concentration-value').textContent=`+${extra} units`;
  const samples=d.test_scenarios.map(row=>dispatch(row,policy,cap,d.cost).cost).sort((a,b)=>a-b),mean=samples.reduce((s,v)=>s+v,0)/samples.length,tail=empiricalCvar(samples);
  chart($('allocation-cost-plot'),[{name:'Selected policy · realized integer cost',points:samples.map((v,i)=>[(i+1)/samples.length,v])}],{xlabel:'Held-out quantile',ylabel:'Cost / demand',ymin:0,height:160,compact:true,legend:false});
  $('allocation-distribution-note').textContent=`160 original held-out scenarios · mean ${fmt(mean,3)}, CVaR₀.₉ ${fmt(tail,3)}. Current capacity; no extra D8 demand in this distribution.`;
  history(strategy,frame,objective,iteration);
  $('allocation-position').textContent=iteration===null?`Direct policy · scenario ${scenario+1}`:`${strategy==='browser'?'Browser':'PyTorch'} step ${iteration} · scenario ${scenario+1}`;
  $('allocation-stage').textContent=strategy==='browser'?`Browser-calculated optimization on 48 stored training scenarios. Realized scenario ${scenario+1}${extra?' has added D8 stress demand':''}; its hard-capacity dispatch is independent of the smooth training penalties.`:strategy==='pytorch_relaxed'?`Recorded PyTorch parameters; integer assignments are calculated now for scenario ${scenario+1}. Capacity/demand edits are browser stress tests, not retraining.`:'Stored or direct policy preferences; integer dispatch is evaluated now. Risk/fairness sliders switch to the explicitly disclosed browser optimizer.';
  Object.assign(lab.dataset,{strategy,step:iteration??'',demand:String(result.total),served:String(result.served),unserved:String(result.lost),penalty:String(result.shortageCost),capacity:String(totalCapacity),cost:String(result.cost),violations:String(result.violations)});
 }
 rebuild();player=playback(lab,{count,render,interval:300});
 $('allocation-strategy').addEventListener('input',()=>{const strategy=$('allocation-strategy').value;if(strategy==='browser')rebuild();else if(strategy==='pytorch_relaxed'||strategy==='scipy_relaxed'){$('allocation-risk').value=d.config.risk;$('allocation-fairness').value=d.config.fairness}player.set(count()-1)});
 for(const id of ['allocation-risk','allocation-fairness'])$(id).addEventListener('input',()=>{$('allocation-strategy').value='browser';rebuild();player.set(count()-1)});
 for(const id of ['allocation-capacity','allocation-stations'])$(id).addEventListener('input',()=>{if($('allocation-strategy').value==='browser'){rebuild();player.set(count()-1)}else{player.stop();player.render()}});
 for(const id of ['allocation-scenario','allocation-concentration'])$(id).addEventListener('input',()=>{player.stop();player.render()});player.set(count()-1);
}
