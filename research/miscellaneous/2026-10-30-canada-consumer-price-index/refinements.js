/* Additional explanations share the existing data, numerical functions and chart engine. */
'use strict';
(() => {
 const U=window.CPI_UI,R=window.CPI_REFINEMENT_DATA;
 if(!U||!R){console.error('CPI refinement dependencies are missing');return;}
 const {$,D,V,esc,fmt,indexText,signed,pct,pctSigned,money,sum,rate,table,options,svg,bars,lineChart,icon,series,productId,monthlyPoints,miniUniverse,makePlayer}=U;
 const math=tex=>katex.renderToString(tex,{throwOnError:true,output:'htmlAndMathml'});
 const source='<a href="https://doi.org/10.25318/1810000401-eng">Table 18-10-0004-01</a>';
 const nationalWeight=(label,kind='reference')=>{
  const exact=R.national_weights[label];
  if(exact)return exact[kind];
  const key=Object.keys(R.national_weights).find(k=>k.replace(/ \(\d{4,6}=100\)$/,'')===label.replace(/ \(\d{4,6}=100\)$/,''));
  return key?R.national_weights[key][kind]:undefined;
 };
 function facts(label){
  const s=series(label),w=nationalWeight(label),l=nationalWeight(label,'link'),c=s?.values['2026-09'],p=s?.values['2025-09'];
  return `<div class="class-facts"><span><b>Code</b> ${esc(productId(label)??'Not in public codelist')}</span><span><b>2025-price weight</b> ${w?fmt(w.value,2)+'%':'Not available in extracted weight table'}</span><span><b>April 2026 weight</b> ${l?fmt(l.value,2)+'%':'Not available'}</span><span><b>September index</b> ${fmt(c,1)}</span><span><b>Calculated YoY</b> ${c&&p?pctSigned(rate(c,p)):'Unavailable'}</span></div><p class="source">${source}; ${s?esc(s.vector)+'; '+esc(s.index_reference??s.unit)+'. ':''}<a href="https://doi.org/10.25318/1810000701-eng">Table 18-10-0007-01</a>, basket year 2025; reference- and link-price weights are distinguished above.</p>`;
 }
 function classification(){
  const icons=['apple','house','furniture','shirt','car','medicine','book','wine'];
  const descriptions=['Groceries and restaurant meals enter the food branch; their price changes receive detailed expenditure weights.','Rent, owned accommodation and utilities require comparable housing services and specialized treatment of several costs.','Household goods and services include furnishing, operation, communication and other in-scope expenses.','Clothing and footwear prices require matched specifications, quality treatment and attention to seasonal patterns.','Transportation combines private and public travel; gasoline is also a member of the special energy aggregate.','Health and personal care contain both goods and services, with different collection procedures where needed.','Recreation, education and reading include products and services such as equipment, travel, books and tuition.','Alcoholic beverages, tobacco and recreational cannabis form their own major consumption component.'];
  function branch(id,depth=0){
   const item=D.products[id],kids=Object.keys(D.products).filter(k=>D.products[k].parent===id);
   if(!kids.length)return `<div class="leaf">${esc(item.label)} <span class="class-code">code ${id}</span></div>`;
   const top=depth===1,mi=D.majors.indexOf(item.label),w=nationalWeight(item.label);
   if(depth===0)return `<details open class="classification-root" data-product="${id}"><summary>${esc(item.label)} · published product hierarchy</summary>${kids.map(k=>branch(k,depth+1)).join('')}</details>`;
   return `<details class="${top?'classification-category':'class-node'}" data-product="${id}"><summary>${top?icon(icons[mi]??'book',item.label):''}<span>${esc(item.label)}${top?'':` <small class="class-code">· ${id}</small>`}</span>${top?`<span class="class-weight">${w?pct(w.value)+' · 2025 prices':''}</span>`:''}</summary><div class="class-children">${kids.map(k=>branch(k,depth+1)).join('')}</div></details>`;
  }
  $('classification').innerHTML=branch(productId('All-items'));
  function detail(id){const label=D.products[id].label,i=D.majors.indexOf(label);$('classification-detail').innerHTML=`<h4>${esc(label)}</h4><p>${i>=0?descriptions[i]:'This is a published product grouping. Its child classes refine its scope; special aggregates may combine members from several branches.'}</p>${facts(label)}`;}
  $('classification').addEventListener('click',e=>{const summary=e.target.closest('summary');if(summary){const id=summary.closest('details').dataset.product;if(id)detail(id);}});
  detail(productId('All-items'));
  const food=series('Food').values;$('food-branch-summary').textContent=`Its 2025-price basket share is ${pct(nationalWeight('Food').value)}; September 2026 annual price growth is ${pctSigned(rate(food['2026-09'],food['2025-09']))}.`;
 }

 function setExplorer(){
  const paths=['2','3','4','5','6','7'].filter(k=>D.products[k]);
  const views={
   hierarchy:paths.map(k=>D.products[k].label),
   partition:['All-items','Goods','Services','Durable goods','Semi-durable goods','Non-durable goods'],
   overlap:['All-items','All-items excluding gasoline','All-items excluding energy','All-items excluding food and energy','Energy','Food','Services','Gasoline']
  };
  const definitions={
   'All-items':'The complete in-scope basket. In the diagrams, the outer rectangle represents its membership rather than its dollar cost.',
   'Goods':'Goods and services partition the in-scope consumption basket. Durable, semi-durable and non-durable goods subdivide the goods set; these are an alternative analytical classification, not three additional major components.',
   'Services':'Services complement goods in the analytical goods/services classification. Service transactions occur in several major components, including food, shelter, transport, health and recreation.',
   'Durable goods':'Goods expected to provide consumption services over an extended period, such as selected vehicles and furniture; a subset of goods.',
   'Semi-durable goods':'An analytical subset of goods that includes selected items with an intermediate useful life. Use the official classification rather than assigning arbitrary item lifetimes here.',
   'Non-durable goods':'Goods consumed relatively quickly or with short useful lives, including food goods and fuels. This is a subset of goods, not a peer of all-items.',
   'Energy':'A special aggregate containing electricity; natural gas; fuel oil and other fuels; gasoline; and fuel, parts and accessories for recreational vehicles. It crosses shelter, transportation and recreation branches.',
   'All-items excluding gasoline':'The complement of gasoline within all-items. Other energy products remain included, so this set is broader than all-items excluding energy.',
   'All-items excluding energy':'The complement of the defined energy aggregate. It overlaps food, services and the other exclusion measures; it is not a mutually exclusive sibling of them.',
   'All-items excluding food and energy':'All-items minus the union of food and energy. Its complement is Food ∪ Energy; it overlaps the other exclusion baskets.',
   'Food':'The food branch contains purchases from stores and restaurants. It is distinct from the energy special aggregate, and includes both goods and food-service transactions.',
   'Gasoline':'Gasoline belongs to the transportation branch and the energy special aggregate. Its removal does not remove electricity, natural gas or the other energy products.'
  };
  function panel(label){
   const relation=$('set-view').value;
   const extra=relation==='hierarchy'?math('A\\subseteq B')+' means that each highlighted lower product group is contained in its published ancestors.':relation==='partition'?'Let G denote goods, S services and U all-items: '+math('G\\cup S=U,\\quad G\\cap S=\\varnothing'):'Exclusion sets overlap. For example, '+math('(U\\setminus E)\\subseteq(U\\setminus G)')+' when gasoline G is a subset of energy E; U denotes all-items.';
   $('set-description').innerHTML=`<h4>${esc(label)}</h4><p>${definitions[label]??'A published product group in the authentic hierarchy. All displayed containment edges follow the SDMX parent codes.'}</p><p>${extra}</p>${facts(label)}`;
  }
  function nested(){
   const selected=$('set-selection').value;
   const node=i=>i>=paths.length?'':`<div class="set-box ${D.products[paths[i]].label===selected?'active':''}"><button type="button" data-select-set="${esc(D.products[paths[i]].label)}" aria-pressed="${D.products[paths[i]].label===selected}">${esc(D.products[paths[i]].label)} · code ${paths[i]}</button>${node(i+1)}</div>`;
   $('set-diagram').innerHTML=node(0);
  }
  function partition(){
   const selected=$('set-selection').value;
   const b=label=>`<button type="button" data-select-set="${esc(label)}" aria-pressed="${selected===label}">${esc(label)}</button>`;
   $('set-diagram').innerHTML=`<div class="set-box">${b('All-items')}<div class="two"><div class="set-box">${b('Goods')}<div class="set-chip-row">${['Durable goods','Semi-durable goods','Non-durable goods'].map(b).join('')}</div></div><div class="set-box">${b('Services')}<p class="hint">Examples: rent, transit, dental care and restaurant service.</p></div></div><p class="hint">Schematic partition; rectangle sizes are not expenditure weights.</p></div>`;
  }
  const prototypes=[
   {family:'Food',label:'Groceries',kind:'goods',energy:false,gas:false,color:U.COLORS[0]},
   {family:'Food',label:'Restaurant meals',kind:'services',energy:false,gas:false,color:U.COLORS[0]},
   {family:'Shelter',label:'Electricity, gas & fuel',kind:'goods',energy:true,gas:false,color:U.COLORS[1]},
   {family:'Shelter',label:'Rent service',kind:'services',energy:false,gas:false,color:U.COLORS[1]},
   {family:'Transportation',label:'Gasoline',kind:'goods',energy:true,gas:true,color:U.COLORS[4]},
   {family:'Transportation',label:'Transit service',kind:'services',energy:false,gas:false,color:U.COLORS[4]},
   {family:'Recreation',label:'Recreation vehicle fuel',kind:'goods',energy:true,gas:false,color:U.COLORS[6]},
   {family:'Recreation',label:'Concert admission',kind:'services',energy:false,gas:false,color:U.COLORS[6]},
   {family:'Other branches',label:'Furniture',kind:'goods',energy:false,gas:false,color:U.COLORS[2]},
   {family:'Other branches',label:'Dental service',kind:'services',energy:false,gas:false,color:U.COLORS[5]}
  ];
  function member(p,label){if(label==='All-items')return true;if(label==='Food')return p.family==='Food';if(label==='Services')return p.kind==='services';if(label==='Gasoline')return p.gas;if(label==='Energy')return p.energy;if(label==='All-items excluding gasoline')return !p.gas;if(label==='All-items excluding energy')return !p.energy;return p.family!=='Food'&&!p.energy;}
  function overlap(){const selected=$('set-selection').value;let s='<rect x="6" y="6" width="788" height="332" rx="14" fill="#fffaf0" stroke="#b69c80"/><text x="21" y="29">All-items U · representative membership examples</text><ellipse cx="400" cy="123" rx="229" ry="75" fill="#f6dfa6" fill-opacity=".5" stroke="#bf9842" stroke-dasharray="5 4"/><text x="400" y="62" text-anchor="middle">Energy spans several branches</text>';
   prototypes.forEach((p,i)=>{const col=Math.floor(i/2),row=i%2,x=20+col*155,y=87+row*139,on=member(p,selected),select=p.energy?'Energy':p.kind==='services'?'Services':p.family==='Food'?'Food':'All-items';s+=`<g class="set-node ${on?'set-node-active':'set-muted'}" role="button" tabindex="0" data-select-set="${select}" aria-label="${esc(p.label)}; ${esc(p.family)}; ${on?'included':'excluded'}"><rect x="${x}" y="${y}" width="140" height="101" rx="10" fill="${p.color}" fill-opacity=".28" stroke="${p.color}"/><text x="${x+70}" y="${y+23}" text-anchor="middle" font-size="11">${esc(p.family)}</text><text x="${x+70}" y="${y+47}" text-anchor="middle" font-size="11">${esc(p.label)}</text><text x="${x+70}" y="${y+72}" text-anchor="middle" font-size="10">${p.kind}${p.energy?' · energy':''}</text><text x="${x+70}" y="${y+89}" text-anchor="middle" font-size="10">${on?'Included':'Excluded'}</text></g>`;});
   $('set-diagram').innerHTML=svg(s,800,350,'Special aggregates overlap major components; selected '+selected)+'<p class="hint">The examples show membership, not a complete microdata universe. The energy outline intersects three component columns. No area or overlap size is a measured weight.</p>';
  }
  function render(){const v=$('set-view').value;if(v==='hierarchy')nested();else if(v==='partition')partition();else overlap();panel($('set-selection').value);}
  function changeView(){options('set-selection',views[$('set-view').value]);render();}
  $('set-view').addEventListener('change',changeView);$('set-selection').addEventListener('change',render);
  const choose=e=>{const b=e.target.closest('[data-select-set]');if(!b)return;const value=b.dataset.selectSet;if(!views[$('set-view').value].includes(value))return;$('set-selection').value=value;render();};
  $('set-diagram').addEventListener('click',choose);$('set-diagram').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(e);}});changeView();
 }

 function miniCalculations(){
  function weighted(){
   U.miniState.milkSecondPrice=+$('weighted-milk-price').value;
   const g=miniUniverse()[0],r=g.p1.map((v,i)=>v/g.p0[i]),a=+$('elementary-weight').value/100,mean=window.CPI_APP.elementary(g.p0,g.p1),weighted=Math.exp(a*Math.log(r[0])+(1-a)*Math.log(r[1]));
   const min=.55,max=1.65,w=800,h=190,x=v=>60+(Math.log(v)-Math.log(min))/(Math.log(max)-Math.log(min))*680;
   let s='<text x="400" y="23" text-anchor="middle">Relatives on a logarithmic horizontal scale</text><line x1="60" x2="740" y1="91" y2="91" stroke="#97a68e"/>';
   [.6,.8,1,1.2,1.6].forEach(v=>s+=`<path d="M${x(v)} 83v17" stroke="#7b9281"/><text x="${x(v)}" y="126" text-anchor="middle">${fmt(v,1)}</text>`);
   [{name:'Offer A',v:r[0],y:65,c:'#dc884c'},{name:'Offer B',v:r[1],y:91,c:'#b36c38'},{name:'Weighted G',v:weighted,y:153,c:'#39775d'}].forEach(p=>s+=`<circle cx="${x(p.v)}" cy="${p.y}" r="7" fill="${p.c}"/><text x="${x(p.v)}" y="${p.y-13}" text-anchor="middle">${p.name}: ${fmt(p.v)}</text>`);
   $('relative-distribution').innerHTML=svg(s,w,h,'Matched milk price relatives and explicitly weighted geometric mean');
   $('weighted-elementary-result').innerHTML=`<p><b>Inputs:</b> Offer A ${money(g.p0[0])} → ${money(g.p1[0])}; Offer B ${money(g.p0[1])} → ${money(g.p1[1])}. Relatives ${fmt(r[0])} and ${fmt(r[1])}.</p><p><b>Substitution:</b> ${math(`G_w=\\exp(${fmt(a,2)}\\log ${fmt(r[0])}+${fmt(1-a,2)}\\log ${fmt(r[1])})`)} = <b>${fmt(weighted)}</b>.</p><p>Equal-weight Jevons = ${fmt(mean.jevons)}; Carli = ${fmt(mean.carli)}; Dutot = ${fmt(mean.dutot)}. Explicit weights: ${pct(a*100)} / ${pct((1-a)*100)}. The continuous upper-level walkthrough uses the <em>equal-weight</em> elementary Jevons factor ${fmt(mean.jevons)}; the weighted result is a separate exceptional-formula illustration.</p>`;
   redrawMini();$('trace-select').dispatchEvent(new Event('change'));U.players['synthesis-player']?.refresh();
  }
  function reprice(){
   const groups=miniUniverse(),f=+$('mini-price-progress').value,cost=sum(groups.map(g=>g.expenditure0*(1+f*(g.index-1))));
   $('mini-weight-table').innerHTML=table(['Product-group proxy','Reference price scale','Later price proxy','Fixed qᵇ','Earlier E, $','Later E, $','Earlier share','Later dollar share'],groups.map(g=>{const relative=1+f*(g.index-1),pt=g.p0[0]*relative,ex=g.quantity*pt;return [esc(g.name),fmt(g.p0[0],2),fmt(pt,2),fmt(g.quantity),fmt(g.expenditure0,2),fmt(ex,2),fmt(g.share*100,1)+'%',fmt(ex/cost*100,1)+'%'];}),'Synthetic quantities in product-specific units. A price proxy rescales the geometric price index to the first offer’s reference-price unit; it is not a newly observed shelf price.');
   bars('mini-weight-chart',groups.map(g=>({label:g.name,value:100*g.expenditure0*(1+f*(g.index-1))/cost,color:g.color})),{suffix:'%',precision:1,label:'Price-updated synthetic expenditure shares'});
   $('mini-weight-result').innerHTML=`<b>Fixed quantities → repriced expenditure:</b> ${groups.map(g=>fmt(g.expenditure0*(1+f*(g.index-1)),2)).join(' + ')} = $${fmt(cost,2)}. Original cost = $100.00. Lowe factor = ${fmt(cost/100)}; price change = <b>${pctSigned(cost-100)}</b>. The quantity vector stays [${groups.map(g=>fmt(g.quantity)).join(', ')}] while dollar shares change.`;
  }
  function trace(){const groups=miniUniverse(),k=+$('continuous-branch').value,g=groups[k],J=g.index,factor=sum(groups.map(x=>x.share*x.index)),level=120*factor;
   $('continuous-flow').innerHTML=[['Matched offers',g.p1.map((p,i)=>`${fmt(p,2)} / ${fmt(g.p0[i],2)}`).join(' · ')],['Elementary factor',fmt(J)],['Weighted term',`${fmt(g.share,2)} × ${fmt(J)}`],['Index level',`${indexText(level)} (120 × P)`],['Monthly inflation',pctSigned(100*(factor-1))]].map(([a,b])=>`<div class="flow-cell"><span>${a}</span><b>${b}</b><small>${esc(g.branch)}</small></div>`).join('');
   $('continuous-inputs').innerHTML=table(['Group','Offer prices, earlier → later','Price relatives','Elementary J','Reference share','Term sᵢJᵢ'],groups.map(x=>[esc(x.name),x.p0.map((p,i)=>money(p)+' → '+money(x.p1[i])).join('<br>'),x.p1.map((p,i)=>fmt(p/x.p0[i])).join(', '),fmt(x.index),fmt(x.share*100,0)+'%',fmt(x.share*x.index)]));
   $('continuous-trace').innerHTML=`<p><b>1 · Inputs:</b> ${esc(g.name)} has relatives ${g.p1.map((p,i)=>fmt(p/g.p0[i])).join(' and ')}.</p><p><b>2 · Elementary calculation:</b> ${math(`J=\\exp\\!\\left(\\tfrac12(${g.p1.map((p,i)=>`\\log ${fmt(p/g.p0[i])}`).join('+')} )\\right)`)} = ${fmt(J)}.</p><p><b>3 · Upper-level aggregation:</b> P = ${groups.map(x=>`${fmt(x.share,2)} × ${fmt(x.index)}`).join(' + ')} = <b>${fmt(factor)}</b>. Shares add to 1.</p><p><b>4 · Link to a hypothetical level:</b> 120.0000 × ${fmt(factor)} = ${fmt(level)}.</p><p><b>5 · Interpretation:</b> 100 × (${fmt(level)} / 120 − 1) = <b>${pctSigned(100*(factor-1))}</b>. The selected branch contributes ${signed(g.share*(g.index-1)*100,2)} percentage points in this fixed synthetic basket. No official microdata reconstruction is claimed.</p>`;
  }
  function redrawMini(){reprice();trace();}
  $('mini-price-progress').addEventListener('input',reprice);$('continuous-branch').addEventListener('change',trace);$('weighted-milk-price').addEventListener('input',weighted);$('elementary-weight').addEventListener('input',weighted);
  $('relative-distribution').classList.add('wide-chart');weighted();
 }

 function samplingFrame(){
  const products=['Milk','Bread','Gasoline','Medicine','Soap'],base=[4,3,1.6,12,5],changes=[.10,.05,.20,.02,-.01];
  const population=[];for(let a=0;a<4;a++)for(let o=0;o<3;o++)for(let p=0;p<5;p++){const p0=base[p]*(1+.025*a+.015*o),relative=1+changes[p]+a*.008+o*.004;population.push({id:`${a}-${o}-${p}`,a,o,p,label:products[p],p0,p1:p0*relative,relative});}
  const selected=new Set(population.map(p=>p.id));
  const scope=p=>($('sample-area').value==='all'||p.a===+$('sample-area').value)&&($('sample-outlet').value==='all'||p.o===+$('sample-outlet').value);
  const geoScope=p=>$('sample-area').value==='all'||p.a===+$('sample-area').value;
  const gm=rows=>rows.length?Math.exp(sum(rows.map(p=>Math.log(p.relative)))/rows.length):NaN;
  function samplingSteps(){
   const areas=population.filter(geoScope),outlets=population.filter(scope),picked=outlets.filter(p=>selected.has(p.id));
   const chain=(active,rows)=>`<div class="sample-chain"><span>Canada · 4 artificial areas</span> → <span>${new Set(rows.map(p=>p.a)).size} areas</span> → <span>${new Set(rows.map(p=>p.a+'-'+p.o)).size} outlets</span> → <span>${rows.length} offers</span> → <b>${rows.length*2} monthly prices</b></div><p>Current stage: ${active}. Candidate price relatives have geometric mean ${fmt(gm(rows))}. Selection counts are synthetic; no official probabilities are assigned.</p>`;
   return [
    {title:'Geography',body:chain('choose collection areas',areas)+'<p>Geography provides coverage and collection strata. Published geographical indices need not coincide with each collection area.</p>'},
    {title:'Retail outlets',body:chain('choose outlets within those areas',outlets)+'<p>Outlet frames identify eligible sellers and what they sell. Administrative, scanner and online sources can change the collection route.</p>'},
    {title:'Product offers',body:chain('retain specified matched offers',picked)+'<p>Track the specified offer’s two prices. Only selected, in-scope offers enter the equal-offer geometric mean in this teaching experiment.</p>'}
   ];
  }
  makePlayer('sampling-player',samplingSteps);
  // Build native buttons once so selecting an offer preserves keyboard focus.
  $('sample-frame').innerHTML=Array.from({length:4},(_,a)=>`<div class="sample-area-card"><h4>Area ${String.fromCharCode(65+a)} · 3 outlets</h4>${Array.from({length:3},(_,o)=>`<div class="sample-outlet-row"><span>Outlet ${o+1}</span><div class="sample-offers">${population.filter(p=>p.a===a&&p.o===o).map(p=>`<button type="button" class="sample-offer" data-offer="${p.id}"><span class="offer-marker" aria-hidden="true"></span><span>${p.label}</span><strong>${fmt(p.relative,3)}</strong></button>`).join('')}</div></div>`).join('')}</div>`).join('');
  const byId=new Map(population.map(p=>[p.id,p]));
  function update(){
   $('sample-frame').querySelectorAll('[data-offer]').forEach(b=>{const p=byId.get(b.dataset.offer),eligible=scope(p),on=eligible&&selected.has(p.id);b.disabled=!eligible;b.setAttribute('aria-pressed',String(on));b.querySelector('.offer-marker').textContent=!eligible?'—':on?'✓':'○';b.setAttribute('aria-label',`${p.label}, Area ${String.fromCharCode(65+p.a)}, Outlet ${p.o+1}, relative ${fmt(p.relative,3)}; ${!eligible?'outside filters':on?'included':'deselected'}`);b.title=`${p.label}: ${money(p.p0)} → ${money(p.p1)}; price relative ${fmt(p.relative,3)}; ${!eligible?'filtered out':on?'select to exclude':'select to include'}`;});
   const rows=population.filter(p=>scope(p)&&selected.has(p.id)),frame=population.filter(scope),logsum=sum(rows.map(p=>Math.log(p.relative))),estimate=gm(rows),truth=gm(population);
   $('sample-frame-status').innerHTML=`<b>Selected observations: ${rows.length} / 60</b><span>${frame.length} eligible under the filters · ${selected.size} saved selections in the full grid</span>`;
   const comparisonRow=(name,value)=>`<div class="sample-comparison-row"><span>${name}</span><b>${Number.isFinite(value)?fmt(value*100,2):'Unavailable'}</b><div class="sample-index-track" aria-hidden="true"><i style="width:${Number.isFinite(value)?value*100/130*100:0}%"></i></div></div>`;
   $('sample-frame-comparison').innerHTML='<h4>Complete population versus selected offers</h4>'+comparisonRow('Complete population index',truth)+comparisonRow('Selected-offer index',estimate)+'<p class="hint">Earlier prices = 100. Bar scale: 0–130 index points; the thin marker indicates 100. Equal offer weights are used, not expenditure weights.</p>';
   $('sample-frame-result').innerHTML=rows.length?`<p><b>Selected-offer growth:</b> ${pctSigned((estimate-1)*100)} · <b>Complete-population growth:</b> ${pctSigned((truth-1)*100)}.</p><p><b>Geometric calculation:</b> G = exp[(1/n) Σ log rᵢ]. For the ${rows.length} included offers, Σ log rᵢ = ${fmt(logsum)}; mean log = ${fmt(logsum/rows.length)}; G = <b>${fmt(estimate)}</b>. The index is 100G = ${fmt(100*estimate)}.</p><p><b>Subset minus complete population:</b> ${signed((estimate-truth)*100,2)} percentage points. Each selected offer has weight 1/${rows.length}. This user-chosen subset is not a representative probability sample or an official CPI sampling-error estimate.</p>`:'<p>No offers selected. The geometric mean and selected-offer index are <b>unavailable</b>; neither is zero.</p>';
   $('sample-frame-table').innerHTML=table(['Area / outlet / offer','Earlier price','Later price','Relative','Log relative'],rows.map(p=>[`Area ${String.fromCharCode(65+p.a)} / ${p.o+1} / ${p.label}`,fmt(p.p0,2),fmt(p.p1,2),fmt(p.relative),fmt(Math.log(p.relative))]));
   U.players['sampling-player'].refresh();
  }
  $('sample-frame').addEventListener('click',e=>{const b=e.target.closest('[data-offer]');if(!b)return;const id=b.dataset.offer;if(selected.has(id))selected.delete(id);else selected.add(id);update();});
  ['sample-area','sample-outlet'].forEach(id=>$(id).addEventListener('change',update));$('sample-frame-reset').addEventListener('click',()=>{population.forEach(p=>selected.add(p.id));$('sample-area').value='all';$('sample-outlet').value='all';update();});update();
 }

 function quality(){
  document.querySelectorAll('.laptop-icon').forEach((el,i)=>el.innerHTML=svg(`<rect x="35" y="10" width="100" height="66" rx="5" fill="${i?'#b4d7db':'#d8ccdf'}" stroke="#455e64" stroke-width="2"/><rect x="43" y="17" width="84" height="51" fill="#f6faf2"/><path d="M35 76L10 89h150l-25-13" fill="#c8d7d4" stroke="#455e64" stroke-width="2"/><path d="M69 82h32" stroke="#455e64" stroke-width="2"/>`,170,100,'Laptop model illustration'));
  function update(){const q=+$('quality-value').value,adjusted=1000-q;$('quality-result').innerHTML=`<p><b>Observed:</b> ${money(1000)} − ${money(900)} = $100; unadjusted increase = ${pct(rate(1000,900))}.</p><p><b>Assumed quality adjustment:</b> replacement price ${money(1000)} − assumed improvement ${money(q)} = <b>${money(adjusted)}</b>.</p><p><b>Comparable-price calculation:</b> 100 × (${fmt(adjusted,2)} / 900 − 1) = <b>${pctSigned(rate(adjusted,900))}</b>. This hypothetical adjustment separates an assumed feature-value difference from the observed price gap.</p>`;}
  $('quality-value').addEventListener('input',update);update();
 }

 const PRIMARY=['2','3','5','7','9','11','14','18','20','23','26','29','30','31'];
 const PROVINCES=['3','5','7','9','11','14','18','20','23','26'],CITIES=['29','30','31'];
 const yoy=g=>{const s=D.series[g+'.2'];return rate(s.values['2026-09'],s.values['2025-09']);};
 const geoName=g=>D.geographies[g].label;
 const geoRateColor=value=>{const t=Math.min(1,Math.max(0,value/7));return `hsl(${110-100*t} ${38+10*t}% ${88-12*t}%)`;};
 function geographicComparisons(){
  const tilePositions={'29':[1,1],'30':[3,1],'31':[6,1],'2':[9,1],'26':[1,2],'23':[2,2],'20':[3,2],'18':[4,2],'14':[5,2],'11':[6,2],'3':[9,2],'9':[7,3],'5':[8,3],'7':[9,3]};
  $('geography-tiles').querySelectorAll('[data-geo]').forEach(b=>{const [col,row]=tilePositions[b.dataset.geo];b.style.setProperty('--geo-column',col);b.style.setProperty('--geo-row',row);});
  $('geography-tiles').insertAdjacentHTML('afterend','<p class="hint geo-cartogram-note">Schematic geographic tiles: provinces run broadly west to east, with the three northern capital cities above. Canada is the national comparator at upper right. Tile locations are approximate; sizes represent neither land area nor expenditure. Narrow screens use a compact selection list.</p>');
  $('geography-tiles').querySelectorAll('[data-geo]').forEach(b=>b.style.setProperty('--tint',geoRateColor(yoy(b.dataset.geo))));
  const ranked=PROVINCES.map(g=>({g,value:yoy(g)})).sort((a,b)=>b.value-a.value);
  bars('province-ranking',ranked.map(r=>({label:geoName(r.g),value:r.value,color:geoRateColor(r.value)})),{suffix:'%',precision:1,maxValue:7,label:'Ten provinces ranked by independently calculated annual inflation'});
  bars('territorial-ranking',CITIES.map(g=>({label:geoName(g).split(',')[0]+' · city',value:yoy(g),color:geoRateColor(yoy(g))})),{suffix:'%',precision:1,maxValue:7,label:'Northern capital-city annual inflation, separately shown from provinces'});
  $('territorial-ranking').insertAdjacentHTML('beforebegin','<h4>Northern capital-city comparison · not provincial or territorial totals</h4>');
  const alias={'Whitehorse, Yukon':'Whitehorse','Yellowknife, Northwest Territories':'Yellowknife','Iqaluit, Nunavut':'Iqaluit'};
  $('geo-verification').innerHTML=table(['Geography','Sep 2025','Sep 2026','Calculated YoY','Published YoY','Difference, pp','Result','Reference / flags / vector'],PRIMARY.map(g=>{const s=D.series[g+'.2'],v=s.values,calc=yoy(g),official=V.rows.find(r=>r.series==='All-items'&&r.metric==='YoY'&&r.geography===(alias[s.geography]??s.geography)&&r.official!==null),flags=s.flags['2026-09'];return [esc(s.geography),fmt(v['2025-09'],1),fmt(v['2026-09'],1),fmt(calc)+'%',official?fmt(official.official,1)+'%':'Unavailable',official?signed(calc-official.official):'—',official&&Math.abs(calc-official.official)<=.05?'Within rounding tolerance':'No comparator',`${esc(s.index_reference??s.unit)}<br>${flags?esc(JSON.stringify(flags)):'No flag recorded'}<br>${esc(s.vector)}`];}),'Source observations and their rounding differences; no source values are replaced with user-supplied expected rates.');
  let selected=['2','14','26'];
  $('geo-multi').insertAdjacentHTML('beforeend',PRIMARY.map(g=>`<label><input type="checkbox" value="${g}" ${selected.includes(g)?'checked':''}> ${esc(geoName(g))}</label>`).join(''));
  const seriesColors=['#2c775b','#b76c32','#467da5','#8e70ad'];
  function compare(){const from=$('geo-history-start').value,lines=selected.map((g,i)=>({name:geoName(g),points:monthlyPoints(D.series[g+'.2'],'yoy',from),color:seriesColors[i]}));lineChart('geo-comparison-chart',lines,{label:'Comparison of selected geographical annual inflation histories',suffix:'%',detail:'geo-comparison-detail'});const maps=lines.map(l=>Object.fromEntries(l.points.map(p=>[p.x,p.y]))),months=[...new Set(lines.flatMap(l=>l.points.map(p=>p.x)))].sort();$('geo-comparison-table').innerHTML=table(['Month',...selected.map(g=>esc(geoName(g))+' YoY, %')],months.map(t=>[t,...maps.map(m=>fmt(m[t],1))]));$('geo-multi-status').textContent=`${selected.length} selected geographies. Curve colours identify geographies; the tile colours above encode inflation rates.`;}
  $('geo-multi').addEventListener('change',e=>{const chosen=[...$('geo-multi').querySelectorAll('input:checked')].map(c=>c.value);if(chosen.length<2||chosen.length>4){e.target.checked=!e.target.checked;$('geo-multi-status').textContent='Choose between two and four series; the previous selection is retained.';return;}selected=chosen;compare();});$('geo-history-start').addEventListener('change',compare);compare();
  const opts=PRIMARY.map(g=>({value:g,label:geoName(g)}));options('component-geo-a',opts,'26');options('component-geo-b',opts,'23');
  function components(){const a=$('component-geo-a').value,b=$('component-geo-b').value,labels=['Food','Shelter','Transportation'],colors=[U.COLORS[0],U.COLORS[1],U.COLORS[4]],rows=[],plot=[];labels.forEach((l,i)=>[a,b].forEach((g,j)=>{const s=series(l,g),now=s?.values['2026-09'],old=s?.values['2025-09'],value=now&&old?rate(now,old):NaN;rows.push([esc(l),esc(geoName(g)),fmt(old,1),fmt(now,1),pct(value),s?esc(s.vector):'Unavailable']);if(Number.isFinite(value))plot.push({label:l+' · '+geoName(g).split(',')[0],value,color:colors[i]+(j?'ff':'a5')});}));if(plot.length)bars('regional-components-chart',plot,{suffix:'%',precision:1,label:'Matching food, shelter and transportation annual inflation in two geographies'});else $('regional-components-chart').innerHTML='<p class="hint">Matching component series are unavailable for these geographies.</p>';$('regional-components-table').innerHTML=table(['Component','Geography','Sep 2025 index','Sep 2026 index','Calculated YoY','Vector'],rows,'The first geography uses lighter bars; the second uses fuller colour. Food is orange, shelter violet and transportation blue. Unavailable component series are not replaced.');}
  ['component-geo-a','component-geo-b'].forEach(id=>$(id).addEventListener('change',components));components();
  const regionalNote=document.createElement('p');regionalNote.className='source';$('geo-stats').after(regionalNote);
  function weightNote(){const g=$('geo-select').value,name=geoName(g),w=R.regional_weights[name];regionalNote.innerHTML=w?.link?`Official geographic expenditure share at April 2026 link-month prices: <b>${pct(w.link.value)}</b> of Canada’s in-scope basket. This is a geographic distribution of all-items expenditure, distinct from a product share within ${esc(name)}. Source: <a href="https://doi.org/10.25318/1810000701-eng">Table 18-10-0007-01</a>, ${esc(w.link.vector)}.`:'A matching national geographic expenditure share is not available in the extracted weight rows.';}
  $('geo-select').addEventListener('change',weightNote);$('geography-tiles').addEventListener('click',weightNote);weightNote();
 }
 function regionalModel(){let weights=[.4,.35,.25];const changes=[[2,3,8],[1,4,2],[3,2,-1]],within=[.2,.5,.3];
  function render(){const factors=changes.map(row=>sum(row.map((p,i)=>within[i]*(1+p/100)))),national=sum(factors.map((p,g)=>weights[g]*p));$('region-model-table').innerHTML=table(['Region','National weight, %','Food change, %','Shelter change, %','Transport change, %','Regional factor','Regional inflation, %','National weighted term'],changes.map((row,i)=>['Region '+String.fromCharCode(65+i),fmt(weights[i]*100,2),...row.map(v=>fmt(v,1)),fmt(factors[i]),fmt((factors[i]-1)*100,1),fmt(weights[i]*factors[i])]));const g=+$('region-model-select').value;$('region-model-result').innerHTML=`<p><b>Within selected region:</b> P = ${changes[g].map((p,i)=>`${fmt(within[i],2)} × ${fmt(1+p/100,4)}`).join(' + ')} = ${fmt(factors[g])}.</p><p><b>Across regions:</b> ${weights.map((w,i)=>`${fmt(w)} × ${fmt(factors[i])}`).join(' + ')} = <b>${fmt(national)}</b>. National change = ${pctSigned(100*(national-1))}.</p><p>The unweighted mean of these three regional rates would be ${pct(sum(factors.map(p=>(p-1)*100))/3)}. It coincides only under special weights or movements. All regions and inputs here are synthetic.</p>`;}
  function select() {const g=+$('region-model-select').value;$('region-model-weight').value=weights[g]*100;$('region-model-shock').value=changes[g][2];render();}
  function input(e){const g=+$('region-model-select').value;if(e.target.id==='region-model-weight'){const w=+$('region-model-weight').value/100,oldOthers=1-weights[g];weights=weights.map((v,i)=>i===g?w:(oldOthers>1e-12?v/oldOthers:1/(weights.length-1))*(1-w));}else changes[g][2]=+$('region-model-shock').value;render();}
  $('region-model-select').addEventListener('change',select);['region-model-weight','region-model-shock'].forEach(id=>$(id).addEventListener('input',input));select();
 }
 function officialWalkthrough(){
  function update(){const which=$('official-walkthrough').value,ns=series('All-items').values,sa=D.seasonal['All-items'].values,on=series('All-items','14').values;const a=which==='yoy'?[ns['2025-09'],ns['2026-09'],3.1,'September 2025','September 2026']:which==='mom'?[ns['2026-08'],ns['2026-09'],.1,'August 2026','September 2026']:which==='sa'?[sa['2026-08'],sa['2026-09'],.4,'August 2026 · SA','September 2026 · SA']:[on['2025-09'],on['2026-09'],2.5,'Ontario · September 2025','Ontario · September 2026'];const calc=rate(a[1],a[0]);$('official-walkthrough-steps').innerHTML=[['1 · Earlier observation',fmt(a[0],1),a[3]],['2 · Later observation',fmt(a[1],1),a[4]],['3 · Substitute',math(`100\\left(\\frac{${fmt(a[1],1)}}{${fmt(a[0],1)}}-1\\right)`),'Rate from publicly rounded index values'],['4 · Calculate',pct(calc),fmt(calc)+'% to four decimals'],['5 · Interpret',fmt(a[2],1)+'%','Published rate; '+(Math.abs(calc-a[2])<=.05?'within rounding tolerance':'check discrepancy')]].map(([title,value,note])=>`<div class="calculation-step"><span>${title}</span><b>${value}</b><p>${note}</p></div>`).join('');}
  $('official-walkthrough').addEventListener('change',update);update();
 }
 classification();setExplorer();miniCalculations();samplingFrame();quality();geographicComparisons();regionalModel();officialWalkthrough();
 window.CPI_REFINEMENTS={version:'1.1',illustrations:document.querySelectorAll('.editorial-illustration img').length,originalIllustrations:R.illustrations.length};
})();
