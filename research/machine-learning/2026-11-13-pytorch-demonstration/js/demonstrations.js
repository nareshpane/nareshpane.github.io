import{initRiverLab}from'./river-lab.js';
import{initAllocationLab}from'./allocation-lab.js';
export{riverSimulator}from'./river-model.js';
import{$,fmt,svg,chart,readouts,playback,load,colors}from'./utils.js';
function survivalLab(d){const lab=$('survival-lab');
 function render(){const profile=+$('survival-profile').value,model=$('survival-model').value,cause=+$('survival-cause').value,horizon=+$('survival-horizon').value;
  const p=d.profiles[model][profile],s=[1],f1=[0],f2=[0];for(let i=0;i<p.length;i++){f1.push(f1.at(-1)+s.at(-1)*p[i][1]);f2.push(f2.at(-1)+s.at(-1)*p[i][2]);s.push(s.at(-1)*p[i][0])}
  let lines=[{name:'Event-free S(t)',points:s.map((v,i)=>[i,v])}];if(cause!==2)lines.push({name:'Cause 1 incidence',points:f1.map((v,i)=>[i,v]),color:colors[1]});if(cause!==1)lines.push({name:'Cause 2 incidence',points:f2.map((v,i)=>[i,v]),color:colors[2]});
  chart($('survival-plot'),lines,{xlabel:'Time bin (synthetic)',ylabel:'Probability',ymin:0,ymax:1});
  $('survival-horizon-value').textContent=`bin ${horizon}`;
  readouts($('survival-readouts'),[[`S(${horizon})`,fmt(s[horizon])],[`F₁(${horizon})`,fmt(f1[horizon])],[`F₂(${horizon})`,fmt(f2[horizon])],['Probability sum',fmt(s[horizon]+f1[horizon]+f2[horizon],8)]]);
  const calibration=d[model].calibration;
  chart($('survival-calibration'),[{name:'Synthetic latent audit bins',points:calibration.map(v=>[v.predicted,v.latent_frequency]),dots:true},{name:'Perfect calibration reference',points:[[0,0],[1,1]],color:'#aebcc6',dashed:true}],{xlabel:'Mean predicted cause-1 risk at bin 8',ylabel:'Latent cause-1 frequency',ymin:0,ymax:1});
 }
 let body='<text x="20" y="20">Held-out observed timelines · ● cause 1 · ◆ cause 2 · │ end-of-bin censor</text>';
 d.timelines.forEach((v,i)=>{const y=42+i*17,x=120+v.time/12*570;body+=`<text x="22" y="${y+4}">Subject ${i+1}</text><line x1="120" y1="${y}" x2="${x}" y2="${y}" stroke="#c2d4e2" stroke-width="2"/>`;body+=v.cause===1?`<circle cx="${x}" cy="${y}" r="4" fill="#b57a60"/>`:v.cause===2?`<path d="M${x} ${y-5}l5 5-5 5-5-5Z" fill="#8d78a5"/>`:`<path d="M${x} ${y-6}v12" stroke="#477da5" stroke-width="2"/>`});
 $('survival-timelines').innerHTML=svg(body,760,300,'Fourteen synthetic held-out event and censoring timelines');
 lab.querySelectorAll('input,select').forEach(x=>x.addEventListener('input',render));render();
}
function financialLab(d){const lab=$('financial-lab'),g=d.gallery,n=g.x.length;
 $('financial-account').innerHTML=g.x.map((_,i)=>`<option value="${i}">Account ${i}</option>`).join('');$('financial-time').max=g.src.length;
 const position=i=>[62+(i%8)*90,55+Math.floor(i/8)*58];
 const descriptions=['Observe the time-stamped candidate events. Every model score below uses only the preceding three-day graph.',
  'The historical window ends before this day begins. A slider reveals candidates in arrival order without adding them to that historical snapshot.',
  'Read each account’s five standardized historical/account features. Scaling was fitted on training days only.',
  'Construct incoming messages x_u for historical edges u→v. This transparent layer uses a neighbor mean.',
  'Sum incoming messages with index_add and divide by incoming degree, clamped to one for empty neighborhoods.',
  'Combine a learned self transformation and neighbor transformation, then apply tanh. Stored values are from the fitted PyTorch model.',
  'Concatenate source embedding, destination embedding, log amount, and same-institution indicator. The edge head returns a logit and sigmoid score.',
  'Apply the chosen illustrative threshold. Highlighted synthetic latent-regime predictions are not findings of wrongdoing.'];
 function render(stage){const selected=+$('financial-account').value,time=+$('financial-time').value,threshold=+$('financial-threshold').value;
  $('financial-time-value').textContent=`${time} of ${g.src.length} candidates shown`;$('financial-threshold-value').textContent=fmt(threshold,2);
  let b='<defs><marker id="finance-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10" fill="#b2c4d1"/></marker></defs>';
  let incoming=[],outgoing=[];
  for(let i=0;i<g.hs.length;i++)if(g.hd[i]===selected||g.hs[i]===selected){if(g.hd[i]===selected)incoming.push(g.hs[i]);else outgoing.push(g.hd[i]);if(stage>=3&&stage<=5){const a=position(g.hs[i]),z=position(g.hd[i]);b+=`<path d="M${a}L${z}" stroke="#c5ad9d" stroke-width="1.5" fill="none" marker-end="url(#finance-arrow)"/>`}}
  for(let i=0;i<Math.min(time,g.src.length);i++){if(i<time-35&&g.src[i]!==selected&&g.dst[i]!==selected)continue;const a=position(g.src[i]),z=position(g.dst[i]),risk=stage>=7&&g.probability[i]>=threshold;
   b+=`<path d="M${a}L${z}" fill="none" stroke="${risk?'#b96f69':'#c1d2df'}" stroke-width="${risk?2.2:1}" opacity=".8" marker-end="url(#finance-arrow)"/>`}
  for(let i=0;i<n;i++){const[x,y]=position(i);b+=`<circle cx="${x}" cy="${y}" r="${i===selected?19:14}" fill="${i===selected?'#fff0e4':stage>=5?'#eee6f6':'#e9f3fb'}" stroke="${i===selected?'#b57a60':'#91aec2'}" stroke-width="${i===selected?2:1}"/><text x="${x}" y="${y+4}" text-anchor="middle" style="font-size:11px">${i}</text>`}
  $('financial-diagram').innerHTML=svg(b,760,405,'Directed synthetic transactions and selected account historical messages');
  $('financial-stage').textContent=`${stage+1}/8 · ${descriptions[stage]}`;
  const index=Math.max(0,Math.min(g.src.length-1,time-1)),score=g.probability[index];
  readouts($('financial-readouts'),[['Candidate',`${g.src[index]} → ${g.dst[index]}`],['Arrival day',fmt(g.candidate_times[index],3)],['Stored GNN score',fmt(score)],['Threshold status',score>=threshold?'Highlighted':'Below threshold']]);
  $('financial-account-note').textContent=`Account ${selected}: incoming historical edges ${incoming.length} from [${[...new Set(incoming)].join(', ')}]; outgoing historical edges ${outgoing.length} to [${[...new Set(outgoing)].join(', ')}]. x=[${g.x[selected].map(v=>fmt(v,3)).join(', ')}]. Mean incoming x=[${g.aggregate[selected].map(v=>fmt(v,3)).join(', ')}]. First three embedding values=[${g.embedding[selected].slice(0,3).map(v=>fmt(v,3)).join(', ')}]. Historical cutoff ${fmt(g.history_max_time,3)} < candidate day ${g.day}.`;
 }
 const player=playback(lab,{count:8,render});lab.querySelectorAll('input:not([data-frame]),select').forEach(x=>x.addEventListener('input',()=>player.render()));
}
export async function initDemonstrations(){const jobs=[['survival-demo.json','survival-lab',survivalLab],['financial-network-demo.json','financial-lab',financialLab],['river-demo.json','river-lab',initRiverLab],['allocation-demo.json','allocation-lab',initAllocationLab]];await Promise.all(jobs.map(async([file,id,init])=>{try{init(await load(file));$(id).dataset.loaded='true'}catch(error){$(id).insertAdjacentHTML('beforeend',`<p class="lab-error" role="status">Could not load ${file}. Use a local HTTP server; see README.md.</p>`);console.error(error)}}))}
