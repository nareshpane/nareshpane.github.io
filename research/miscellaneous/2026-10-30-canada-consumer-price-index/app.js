/* Static CPI monograph. Official data are preprocessed locally; no runtime APIs.
 * Numerical functions are shared between initial rendering and user updates.
 */
'use strict';
(() => {
const D=window.CPI_DATA,V=window.CPI_VALIDATION;
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,n=4)=>Number.isFinite(x)?(Math.abs(x)<.5*10**(-n)?0:x).toLocaleString('en-CA',{minimumFractionDigits:n,maximumFractionDigits:n}):'Unavailable';
const signed=(x,n=4)=>Number.isFinite(x)?(Math.abs(x)<.5*10**(-n)?'':x>0?'+':'')+fmt(x,n):'Unavailable';
// Display only: ordinary results, index levels and plots use 1 decimal.
// Source weights and currency retain 2; ratios, derivations and tiny audit differences retain 4+.
const indexText=x=>fmt(x,1);
// These helpers never feed rounded strings back into a calculation.
const pct=(x,n=1)=>Number.isFinite(x)?fmt(x,n)+'%':'Unavailable';
const pctSigned=(x,n=1)=>Number.isFinite(x)?signed(x,n)+'%':'Unavailable';
const money=x=>'$'+fmt(x,2),sum=a=>a.reduce((x,y)=>x+y,0),rate=(a,b)=>100*(a/b-1);
const COLORS=['#e58c4e','#9c85c6','#ddb95a','#d27c94','#6cafd2','#75b494','#7ebbc1','#b29d72'];
const TINTS=['#fae3cd','#eae0f6','#f5edce','#f6e0e6','#dfedf8','#deefdf','#def1ee','#eee6d9'];
const SHORT=['Food','Shelter','Household','Clothing','Transport','Health','Recreation','Alcohol, tobacco & cannabis'];
const productId=label=>Object.keys(D.products).find(k=>D.products[k].label===label);
const series=(label,geo='2')=>D.series[geo+'.'+productId(label)];
const national=series('All-items').values;
const MINI_STATE={milkSecondPrice:5.25};
const PLAYERS={};
const weight=(year,kind='reference')=>D.majors.map(label=>D.weights.find(w=>w.year===year&&w.label===label&&w.price_period===('Weight at basket '+(kind==='link'?'link month':'reference period')+' prices'))?.value??NaN);
const normalize=a=>a.map(x=>x/sum(a));
function householdBasket(mode='official'){
 const raw=mode==='official'?weight('2025','link'):[22,35,10,5,12,7,6,3];
 return {raw,shares:normalize(raw),publishedTotal:sum(raw)};
}
function rebalanceShares(current,k,share,baseline=current){
 const other=sum(current.filter((_,i)=>i!==k)),fallback=sum(baseline.filter((_,i)=>i!==k));
 return current.map((v,i)=>i===k?share:(other>1e-12?v/other:baseline[i]/fallback)*(1-share));
}
const attach=(ids,fn,event='input')=>ids.forEach(id=>$(id).addEventListener(event,fn));
const options=(id,items,value)=>{$(id).innerHTML=items.map(x=>`<option value="${esc(typeof x==='string'?x:x.value)}">${esc(typeof x==='string'?x:x.label)}</option>`).join('');if(value!==undefined)$(id).value=value;};
function table(headers,rows,caption=''){return `<table>${caption?`<caption>${caption}</caption>`:''}<thead><tr>${headers.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>'<tr>'+r.map((c,i)=>i===0?`<th scope="row">${c}</th>`:`<td>${c}</td>`).join('')+'</tr>').join('')}</tbody></table>`;}
function stat(label,value,note,color=COLORS[0]){return `<div class="stat" style="--color:${color}"><div class="stat-label">${label}</div><div class="stat-value">${value}</div><div class="stat-note">${note}</div></div>`;}
function headlineStat(label,value,note,kind){
 const paths=[
  '<path d="M5 13h24l-3 15H8zM10 13l5-8M24 13l-5-8M12 17v7M18 17v7M24 17v7"/>',
  '<rect x="4" y="7" width="25" height="23" rx="4"/><path d="M4 14h25M11 4v7M23 4v7M9 25l6-6 5 3 6-7M21 15h5v5"/>',
  '<rect x="2" y="8" width="14" height="20" rx="3"/><rect x="19" y="8" width="14" height="20" rx="3"/><path d="M2 14h14M19 14h14M7 5v6M25 5v6M6 20h5M23 20h5"/>',
  '<rect x="4" y="7" width="26" height="23" rx="4"/><path d="M4 14h26M11 4v7M23 4v7M7 23c4-12 6 9 10 0s7-5 10 0"/>'
 ];
 return `<div class="stat headline-card"><div class="stat-heading"><svg viewBox="0 0 36 36" aria-hidden="true"><rect x="1" y="1" width="34" height="34" rx="10" fill="${TINTS[kind]}"/><g fill="none" stroke="${['#6e72a0','#b96c33','#438d70','#4b88a5'][kind]}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${paths[kind]}</g></svg><div class="stat-label">${label}</div></div><div class="stat-value">${value}</div><div class="stat-note">${note}</div></div>`;
}
function svg(content,width=800,height=300,label='Calculation diagram'){return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(label)}">${content}</svg>`;}
function bars(id,items,{suffix='',precision=1,label='Comparison',height,maxValue}={}){
 const mobile=innerWidth<=600,w=mobile?Math.max(300,$(id).clientWidth):800,left=mobile?115:205,right=mobile?90:95,h=height??Math.max(150,items.length*43+40),top=15,bottom=25;
 const lo=Math.min(0,...items.map(v=>v.value)),hi=Math.max(0.0001,maxValue??0,...items.map(v=>v.value)),span=hi-lo,x=v=>left+(v-lo)/span*(w-left-right),step=(h-top-bottom)/items.length;
 let s=`<line x1="${x(0)}" x2="${x(0)}" y1="5" y2="${h-bottom+5}" stroke="#789489"/>`;
 items.forEach((r,i)=>{const y=top+i*step,xx=x(r.value),label=mobile&&r.label.length>18?r.label.slice(0,17)+'…':r.label,value=suffix===' dollars'?'$'+fmt(r.value,precision):fmt(r.value,precision)+suffix;s+=`<text x="${left-12}" y="${y+17}" text-anchor="end">${esc(label)}</text><rect x="${Math.min(xx,x(0))}" y="${y}" width="${Math.max(1,Math.abs(xx-x(0)))}" height="26" rx="4" fill="${r.color??COLORS[i%8]}"><title>${esc(r.label)}: ${esc(value)}</title></rect><text x="${xx+7}" y="${y+18}">${esc(value)}</text>`;});
 $(id).innerHTML=svg(s,w,h,label+'; '+items.map(r=>r.label+' '+fmt(r.value,precision)+suffix).join('; '));
}
function lineChart(id,lines,{label='Time series',suffix='',detail,fill=false,minZero=false}={}){
 const node=$(id),mobile=innerWidth<=600,width=mobile?Math.max(300,node.clientWidth):800,height=mobile?275:310,L=mobile?45:63,R=mobile?13:22,T=25,B=47;
 const all=lines.flatMap(l=>l.points).filter(p=>Number.isFinite(p.y));
 if(!all.length){node.innerHTML='<p class="hint">No observations available for this selection.</p>';return;}
 const keys=[...new Set(all.map(p=>p.x))].sort(),vals=all.map(p=>p.y),lo0=Math.min(...vals),hi0=Math.max(...vals),pad=Math.max((hi0-lo0)*.12,.15);
 const lo=minZero?Math.min(0,lo0-pad):lo0-pad,hi=hi0+pad,x=k=>L+keys.indexOf(k)/Math.max(1,keys.length-1)*(width-L-R),y=v=>height-B-(v-lo)/(hi-lo)*(height-T-B);
 let s='';for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,yy=y(v);s+=`<line class="gridline" x1="${L}" x2="${width-R}" y1="${yy}" y2="${yy}"/><text class="axis" x="${L-9}" y="${yy+4}" text-anchor="end">${fmt(v,1)}</text>`;}
 [...new Set([0,Math.floor((keys.length-1)/3),Math.floor(2*(keys.length-1)/3),keys.length-1])].forEach(i=>s+=`<text class="axis" x="${x(keys[i])}" y="${height-20}" text-anchor="middle">${esc(keys[i])}</text>`);
 lines.forEach((l,li)=>{const points=l.points.filter(p=>Number.isFinite(p.y));if(!points.length)return;const path=points.map((p,i)=>`${i?'L':'M'}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`).join(' ');
 if(fill&&li===0)s+=`<path d="${path} L${x(points.at(-1).x)},${height-B} L${x(points[0].x)},${height-B}Z" fill="${l.color}" opacity=".12"/>`;
 s+=`<path d="${path}" stroke="${l.color}" stroke-width="2.6" fill="none" stroke-linejoin="round"/>`;
 points.forEach(p=>s+=`<circle class="chart-point" cx="${x(p.x)}" cy="${y(p.y)}" r="${keys.length<30?4:2.2}" fill="${l.color}" tabindex="${keys.length>150?'-1':'0'}" data-x="${esc(p.x)}" data-y="${p.y}" data-name="${esc(l.name)}"><title>${esc(l.name)} · ${esc(p.x)}: ${fmt(p.y,1)}${esc(suffix)}</title></circle>`);
 });
 node.innerHTML=svg(s,width,height,label)+`<div class="hint">${lines.map(l=>`<span style="display:inline-block;margin-right:18px"><span style="color:${l.color}" aria-hidden="true">●</span> ${esc(l.name)}</span>`).join('')} ${suffix?'· '+esc(suffix):''}</div>`;
 if(detail){const show=p=>$(detail).textContent=`${p.dataset.name} · ${p.dataset.x}: ${fmt(+p.dataset.y,1)}${suffix}. Source values are publicly rounded.`;node.querySelectorAll('.chart-point').forEach(p=>{p.addEventListener('pointerenter',()=>show(p));p.addEventListener('focus',()=>show(p));});const last=node.querySelector('.chart-point:last-of-type');if(last)show(last);}
}
const ICON_PATHS={
 milk:'<path fill="#f7fbef" d="M22 19h19l5 12v27H17V31z"/><path fill="#b2d8e6" d="M17 31h29v21H17z"/><path fill="#e7ae6c" d="M22 19v-7h19v7l-5 12H17"/><path d="M31 12v7l5 12M23 39h16M26 45h10"/>',
 bread:'<path fill="#d99a52" d="M14 32c-8-10 0-21 11-18 5-8 17-6 20 2 12 0 17 13 8 19v22H14z"/><path fill="#f7d9a0" d="M20 34h27v23H20z"/><path d="M25 24l-5 8M36 21l-5 10M46 25l-5 9"/>',
 apple:'<path fill="#e89172" d="M32 24c-21-15-29 16-13 30 9 8 9 2 13 3 6 3 13 3 20-11 10-21-3-31-20-22z"/><path fill="#91b67d" d="M32 21c3-12 13-12 15-12-2 10-7 13-15 12"/><path d="M31 26V15M21 32c-4 3-5 8-2 12"/>',
 eggs:'<path fill="#e4c19b" d="M8 39h48l-5 18H13z"/><ellipse fill="#fff8e7" cx="19" cy="31" rx="9" ry="13"/><ellipse fill="#f3dfb6" cx="33" cy="31" rx="9" ry="13"/><ellipse fill="#fff8e7" cx="47" cy="31" rx="9" ry="13"/><path d="M9 41h46"/>',
 banana:'<path fill="#f0ca59" d="M12 21c5 27 27 37 43 9-6 31-40 35-48 4z"/><path fill="#e9ba40" d="M17 14c3 23 22 34 36 10-4 27-31 33-42 9z"/><path d="M16 15l-2-5M51 25l4-7"/>',
 meat:'<path fill="#dd8e8c" d="M9 32c3-13 18-18 30-14s21 14 16 27-31 13-42 5c-7-4-6-11-4-18z"/><path fill="#fff0db" d="M17 31c3-6 12-8 18-5s5 13-2 16-20 0-16-11z"/><path d="M39 34l10 5M15 49l29-1"/>',
 house:'<path fill="#c6b7df" d="M12 29l20-18 20 18v29H12z"/><path fill="#f6edda" d="M27 39h12v19H27z"/><path fill="#c4e2e6" d="M17 31h9v10h-9zM40 31h8v10h-8z"/><path d="M7 30L32 7l25 23M43 16V8h7v15"/>',
 apartment:'<path fill="#b5a8d4" d="M13 9h38v50H13z"/><path fill="#d8e8ed" d="M20 17h8v8h-8zM36 17h8v8h-8zM20 32h8v8h-8zM36 32h8v8h-8z"/><path fill="#fae4b7" d="M27 47h11v12H27z"/><path d="M8 59h48"/>',
 rent:'<path fill="#fff7df" d="M16 8h32v50l-4-3-5 3-5-3-5 3-5-3-4 3-4-3z"/><path d="M23 20h18M23 28h18M23 36h9"/><path fill="#c7b2e1" d="M36 38l9 3v9h-9z"/>',
 electricity:'<path fill="#f3cc65" d="M36 5L15 35h15l-3 23 22-33H34z"/>',
 furniture:'<path fill="#dcb969" d="M16 28V16c0-8 32-8 32 0v12"/><path fill="#efd69a" d="M8 27h10v16h28V27h10v27H8z"/><path d="M13 54v5M51 54v5M20 31h24"/>',
 cleaning:'<path fill="#9ecbb3" d="M20 26h25l4 31H16z"/><path fill="#e2c26e" d="M22 12h20v8H22zM32 12h17l6 6-4 5-12-4H29z"/><path d="M26 20v6M23 37h18M25 45h14"/>',
 shirt:'<path fill="#dba0af" d="M20 11l12 5 12-5 14 15-10 10-6-5v28H22V31l-6 5L6 26z"/><path d="M24 13c0 10 16 10 16 0"/>',
 jacket:'<path fill="#c38aa1" d="M23 8h18l14 15-8 12-5-4v28H22V31l-5 4-8-12z"/><path fill="#f8e3d0" d="M23 8l9 14 9-14"/><path d="M32 22v37M24 42h5M36 42h5"/>',
 shoe:'<path fill="#d694a8" d="M9 32l13-10 11 13 19 6 5 9H8z"/><path fill="#fff3da" d="M8 50h49v7H8z"/><path d="M27 33l-6 5M34 37l-6 5M41 40l-6 5"/>',
 car:'<path fill="#83b7d7" d="M10 31l8-16h27l10 16v19H9z"/><path fill="#e2f1eb" d="M22 20h19l6 12H16z"/><path d="M31 20v12M9 37h46"/><circle fill="#344957" cx="18" cy="50" r="7"/><circle fill="#344957" cx="46" cy="50" r="7"/>',
 gasoline:'<path fill="#7eb0d2" d="M11 10h28v47H11z"/><path fill="#edf6e9" d="M17 17h16v14H17z"/><path d="M39 26c19 0 0 27 13 27 8 0 3-28 3-28l-8-9M18 41h14M8 58h35"/>',
 transit:'<path fill="#90bed2" d="M13 8h38v45H13z"/><path fill="#e1f0de" d="M19 15h26v21H19z"/><path d="M32 15v21M18 45h5M40 45h5M20 54l-4 6M44 54l4 6"/>',
 medicine:'<path fill="#9bc5a5" d="M20 18h25v40H20z"/><path fill="#f2deb6" d="M19 9h27v10H19z"/><path fill="#fff9e6" d="M20 29h25v21H20z"/><path d="M32 33v13M26 39h12"/>',
 tooth:'<path fill="#f8f8eb" d="M16 12c9-8 14 2 17 0 21-13 23 9 17 21-5 9-4 24-10 24-6 0-2-20-8-20s-3 20-9 20c-5 0-4-14-8-23-4-9-6-17 1-22z"/><path d="M22 18l11 3 9-4"/>',
 book:'<path fill="#a0c9cb" d="M8 14c10-4 17-2 24 3 8-5 16-7 24-3v40c-10-4-17-2-24 3-8-5-16-7-24-3z"/><path d="M32 17v40M14 24l11 3M39 27l10-3M14 34l11 3M39 37l10-3"/>',
 concert:'<path fill="#90bfc8" d="M12 31h7v20h-7zM27 22h8v29h-8zM43 13h8v38h-8z"/><path d="M9 58h46M47 13V7l10 4"/>',
 travel:'<path fill="#9ac7ca" d="M9 23h46v34H9z"/><path d="M25 23v-9h15v9M18 23v34M46 23v34M28 37h9M15 58v3M49 58v3"/>',
 wine:'<path fill="#c9b48a" d="M20 8h24v15c0 15-24 15-24 0z"/><path fill="#d09590" d="M20 20h24v4c0 13-24 13-24 0z"/><path d="M32 35v21M23 57h18"/>',
 leaf:'<path fill="#b6bd8f" d="M32 57c-3-20-25-13-20-35 9 3 15 10 20 18-6-22 2-28 7-33 9 15 6 25-2 35 7-13 14-15 21-14-1 17-16 17-26 29z"/><path d="M32 40v18"/>',
 ticket:'<path fill="#dbc28b" d="M8 18h48v12c-12 1-12 11 0 12v12H8V42c12-1 12-11 0-12z"/><path d="M39 21v5M39 31v5M39 41v5M39 50v3M18 23h11"/>'
};
function icon(type,label='',size=64){return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="${size}" height="${size}" role="img" aria-label="${esc(label||type)}"><g fill="none" stroke="#344b50" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${ICON_PATHS[type]??ICON_PATHS.house}</g></svg>`;}
function heroBasket(){let s='<ellipse cx="116" cy="145" rx="96" ry="10" fill="#d6dfcb"/>';['milk','bread','house','apple','car'].forEach((k,i)=>s+=`<g transform="translate(${7+i*37},${i%2?25:13}) scale(.9)"><g fill="none" stroke="#344b50" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${ICON_PATHS[k]}</g></g>`);s+='<path d="M25 77h184l-18 61H44z" fill="#efbb77" stroke="#526b5d" stroke-width="2"/><path d="M61 76l24-48M175 76l-24-48M40 96h155M46 117h144M70 80l7 55M115 80v55M162 80l-7 55" fill="none" stroke="#99774e" stroke-width="2"/><rect x="20" y="73" width="195" height="10" rx="5" fill="#d68d4f"/>';return svg(s,235,165,'Original illustration of products and services in a household basket');}
function inputTable(id,rows,fields,fn){$(id).querySelector('tbody').innerHTML=rows.map((r,i)=>`<tr><th scope="row">${esc(r.label)}</th>${fields.map(f=>`<td><input type="number" min="${f.min??.01}" max="${f.max??10000}" step="${f.step??.01}" value="${r[f.key]}" data-i="${i}" data-key="${f.key}" aria-label="${esc(r.label+' '+f.label)}"></td>`).join('')}${id==='elementary-inputs'?'<td class="relative"></td><td class="log"></td>':''}</tr>`).join('');$(id).addEventListener('input',e=>{const el=e.target;if(!el.matches('input'))return;el.setAttribute('aria-invalid',String(!el.validity.valid));if(!el.validity.valid||el.value==='')return;rows[+el.dataset.i][el.dataset.key]=+el.value;fn();});}
function renderMath(){document.querySelectorAll('.math').forEach(el=>{try{katex.render(el.textContent,el,{displayMode:el.classList.contains('display'),throwOnError:true,output:'htmlAndMathml',strict:'warn'});}catch(e){el.classList.add('math-error');console.error(e);}});}

// Each animation has an explicit stage, a synchronized numerical explanation,
// keyboard-accessible seeking, pause and restart. Nothing autoplays.
function makePlayer(id,steps){
 const el=$(id),range=el.querySelector('.timeline'),button=el.querySelector('.play');let playing=false,last=0,value=0,raf;
 const getSteps=()=>typeof steps==='function'?steps():steps;
 if(id==='synthesis-player')el.classList.add('synthesis-player');
 const stagePaths=[
 '<path d="M7 26L24 10l17 16M12 24v18h24V24M20 42V29h9v13"/><circle cx="38" cy="10" r="6"/>',
 '<path d="M7 21h34l-5 21H12zM15 21l6-14M33 21L27 7M17 25v13M25 25v13M33 25v13"/>',
 '<path d="M24 6v35M10 41h28M9 15h30M12 15L5 31h14zM36 15l-7 16h14z"/>',
 '<path d="M6 17h36M6 34h36M10 17v26M38 17v26M12 7h10v10H12zM28 23h9v11h-9z"/>',
 '<path d="M10 5h27v37l-5-3-5 3-5-3-5 3-7-3zM16 14h15M16 21h10M20 29l5 5 11-13"/>',
 '<path d="M8 37V12h31M11 32l9-8 8 3 11-17M31 10h8v8M9 43h30"/>',
 '<circle cx="9" cy="10" r="5"/><circle cx="24" cy="10" r="5"/><circle cx="39" cy="10" r="5"/><circle cx="24" cy="37" r="7"/><path d="M9 15l12 16M24 15v15M39 15L27 31"/>',
 '<path d="M20 15l4-4c13-10 24 7 12 16l-6 4M28 33l-4 4C11 47 0 30 12 21l6-4M17 31l14-14"/>',
 '<path d="M9 7v35h32M15 34l7-13 8 5 10-14M15 12h9"/>',
 '<circle cx="14" cy="13" r="6"/><circle cx="35" cy="35" r="6"/><path d="M38 6L10 43M25 8h12v12"/>'
 ];
 function stageIcon(i){return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" aria-hidden="true"><rect x="1" y="1" width="46" height="46" rx="11" fill="${TINTS[i%8]}"/><g stroke="#42645f" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${stagePaths[i%10]}</g></svg>`;}
 const stageContainer=el.querySelector('.stages');
 stageContainer.innerHTML=getSteps().map((s,j)=>`<button type="button" class="stage" data-stage="${j}" ${id==='synthesis-player'&&j>=5?`style="grid-column:${10-j};grid-row:2"`:''}><span class="stage-num">${String(j+1).padStart(2,'0')}</span>${id==='synthesis-player'?`<span class="stage-symbol">${stageIcon(j)}</span>`:''}<span class="stage-title">${esc(s.title)}</span></button>`).join('');
 function paint(){const a=getSteps(),n=a.length,i=Math.min(n-1,Math.floor(value+1e-7));range.max=n-1;range.value=value;stageContainer.querySelectorAll('[data-stage]').forEach((s,j)=>{s.classList.toggle('current',j===i);s.classList.toggle('done',j<i);if(j===i)s.setAttribute('aria-current','step');else s.removeAttribute('aria-current');});el.querySelector('.stage-detail').innerHTML=a[i].body;el.querySelector('.stage-number').textContent=`${i+1} / ${n}`;button.textContent=playing?'Pause':'Play';button.setAttribute('aria-pressed',String(playing));}
 function frame(t){if(!playing)return;if(last)value+=(t-last)/1600;last=t;const max=getSteps().length-1;if(value>=max){value=max;playing=false;last=0;}paint();if(playing)raf=requestAnimationFrame(frame);}
 button.addEventListener('click',()=>{playing=!playing;last=0;if(value>=getSteps().length-1&&playing)value=0;paint();if(playing)raf=requestAnimationFrame(frame);else cancelAnimationFrame(raf);});
 stageContainer.addEventListener('click',e=>{const b=e.target.closest('[data-stage]');if(!b)return;value=+b.dataset.stage;playing=false;last=0;cancelAnimationFrame(raf);paint();});
 el.querySelector('.restart').addEventListener('click',()=>{playing=false;last=0;value=0;cancelAnimationFrame(raf);paint();});range.addEventListener('input',()=>{value=+range.value;last=0;paint();});paint();PLAYERS[id]={refresh:paint};return PLAYERS[id];
}

function setupBasket(){
 const r=[{label:'Milk · cartons',q:2,p0:4,p1:4.4},{label:'Bread · loaves',q:3,p0:3,p1:3.3},{label:'Apples · kg',q:2,p0:5,p1:5.5}];
 function update(){const c0=sum(r.map(x=>x.q*x.p0)),c1=sum(r.map(x=>x.q*x.p1));$('receipt').innerHTML='<h4>THE SAME SHOPPING LIST</h4>'+r.map(x=>`<div class="receipt-row"><span>${x.q} × ${esc(x.label.split(' · ')[0])}</span><span>${money(x.q*x.p0)} → ${money(x.q*x.p1)}</span></div>`).join('')+`<div class="receipt-row receipt-total"><span>${money(c0)}</span><span>${money(c1)}</span></div><p><strong>${pctSigned(rate(c1,c0))}</strong><br>(${money(c1)} − ${money(c0)}) ÷ ${money(c0)} × 100</p>`;bars('basket-chart',[{label:'Month 1 cost',value:c0,color:'#91b58b'},{label:'Month 2 cost',value:c1,color:'#df9b5e'}],{suffix:' dollars',precision:2,height:130,label:'Fixed quantity basket costs'});}
 inputTable('basket-inputs',r,[{key:'q',label:'quantity',min:.1,step:.1},{key:'p0',label:'first month price'},{key:'p1',label:'second month price'}],update);update();
 $('basket-gallery').innerHTML=[['milk','Milk'],['bread','Bread'],['eggs','Eggs'],['apple','Apples'],['house','Housing'],['electricity','Electricity'],['car','Transport'],['medicine','Medicine'],['book','Books'],['travel','Travel']].map(([k,l])=>`<figure>${icon(k,l)}<figcaption>${l}</figcaption></figure>`).join('');
 function normal(){const c=+$('normal-cost').value,index=c/27*100;$('normal-output').textContent=money(c);const x=80+(index-50)/150*640;$('normal-chart').innerHTML=svg(`<line x1="80" x2="720" y1="55" y2="55" stroke="#698e77" stroke-width="2"/>${[50,100,150,200].map(v=>`<path d="M${80+(v-50)/150*640},48v14" stroke="#698e77"/><text x="${80+(v-50)/150*640}" y="85" text-anchor="middle">${v}</text>`).join('')}<circle cx="${x}" cy="55" r="10" fill="#dc884e"/><text x="${x}" y="25" text-anchor="middle">${indexText(index)}</text>`,800,110,'Basket dollar cost normalized to an index');$('normal-explain').textContent=`$27 reference cost → 100; ${money(c)} current cost → index ${indexText(index)}. Cumulative price change: ${pctSigned(index-100)}.`;}
 $('normal-cost').addEventListener('input',normal);normal();
}
function setupClassification(){
 const examples=[['milk','bread','apple','eggs','banana','meat'],['apartment','rent','electricity'],['furniture','cleaning','house'],['shirt','jacket','shoe'],['car','gasoline','transit'],['medicine','tooth','leaf'],['book','concert','travel'],['wine','leaf','ticket']];
 const descriptions=['Groceries and meals from restaurants.','Rent, owned accommodation and utilities.','Furniture, communications and household services.','Garments, shoes and clothing services.','Vehicles, fuel and transportation services.','Medicines, dental and personal services.','Learning, travel, books and recreation.','Alcohol, tobacco and recreational cannabis.'];
 $('component-panels').innerHTML=D.majors.map((m,i)=>`<div class="component-card" style="--color:${COLORS[i]};--tint:${TINTS[i]}"><div class="icons">${examples[i].map(k=>icon(k)).join('')}</div><h3>${esc(m)}</h3><p>${descriptions[i]}</p></div>`).join('');
 function branch(id,depth=0){const p=D.products[id],kids=Object.keys(D.products).filter(k=>D.products[k].parent===id);return kids.length?`<details ${depth===0?'open':''}><summary>${esc(p.label)} <small>· code ${id}</small></summary>${kids.map(k=>branch(k,depth+1)).join('')}</details>`:`<div class="leaf">${esc(p.label)} · ${id}</div>`;}
 $('classification').innerHTML=branch(productId('All-items'));
}
function setupWeights(){
 let selectedTile=0,baseline=householdBasket(),weights=[...baseline.shares],weightsChanged=false;
 options('shock-category',D.majors.map((l,i)=>({value:String(i),label:l})),'0');
 function tileDetail(){const kind=$('weight-period').value,w=weight('2025',kind);$('weight-tile-detail').textContent=`${D.majors[selectedTile]}: published weight ${pct(w[selectedTile],2)} at ${kind==='link'?'April 2026 link-month':'2025 reference-period'} prices. Source: Statistics Canada Table 18-10-0007-01.`;$('weight-treemap').querySelectorAll('[data-weight-component]').forEach(e=>e.setAttribute('aria-pressed',String(+e.dataset.weightComponent===selectedTile)));}
 function show(){const kind=$('weight-period').value,w=weight('2025',kind),ref=weight('2025'),link=weight('2025','link'),groups=[[0,1,2,3],[4,5,6,7]],labels=[...SHORT.slice(0,7),'Alcohol'];labels[3]='Clothes';
  $('weight-treemap').style.gridTemplateRows=groups.map(ids=>sum(ids.map(i=>w[i]))+'fr').join(' ');
  $('weight-treemap').innerHTML=groups.map(ids=>`<div class="treemap-row" style="grid-template-columns:${ids.map(i=>w[i]+'fr').join(' ')}">${ids.map(i=>`<button type="button" data-weight-component="${i}" class="tree-tile ${w[i]<6?'small-tile':''}" style="--tile-tint:${TINTS[i]}" title="${esc(D.majors[i])}: ${pct(w[i],2)}; ${kind==='link'?'April 2026':'2025'} prices; Table 18-10-0007-01" aria-label="${esc(D.majors[i])}, ${pct(w[i],2)}, select for source details"><span>${esc(labels[i])}</span><strong>${pct(w[i])}</strong></button>`).join('')}</div>`).join('');
  $('weight-stack').innerHTML=w.map((v,i)=>`<span style="width:${v/sum(w)*100}%;background:${COLORS[i]}" title="${esc(D.majors[i])}: ${pct(v)}; published ${pct(v,2)}"></span>`).join('');
  $('weight-stack').setAttribute('aria-label',w.map((v,i)=>SHORT[i]+' '+pct(v)).join('; '));
  $('weight-table').innerHTML=table(['Component','2025 prices, %','April 2026 prices, %','Change, pp'],D.majors.map((l,i)=>[esc(l),fmt(ref[i],2),fmt(link[i],2),`<span class="weight-change ${link[i]>=ref[i]?'positive':'negative'}">${signed(link[i]-ref[i],2)}</span>`]),'Published source precision retained. Rounded column totals: '+pct(sum(ref),2)+' and '+pct(sum(link),2)+'.');tileDetail();
 }
 $('weight-treemap').addEventListener('click',e=>{const b=e.target.closest('[data-weight-component]');if(b){selectedTile=+b.dataset.weightComponent;tileDetail();}});
 function shock(){const k=+$('shock-category').value,g=+$('shock-size').value,official=$('shock-mode').value==='official',effect=weights[k]*g,factor=1+effect/100;
  $('shock-weight-value').textContent=pct(weights[k]*100);$('shock-size-value').textContent=pctSigned(g);$('shock-weight').setAttribute('aria-valuetext',pct(weights[k]*100));$('shock-size').setAttribute('aria-valuetext',pctSigned(g));
  $('shock-starting').textContent=official?'Starting weights: Statistics Canada, 2025 expenditure basket, price-updated to April 2026.':'Starting weights: fictional household — food 22%, shelter 35%, and six other illustrative expenditure shares.';
  $('shock-status').textContent=(official?(weightsChanged?'Hypothetical scenario based on official starting weights.':'Hypothetical price shock using official starting weights.'):'Synthetic household scenario.')+' Updated share total: '+pct(sum(weights)*100)+'.';
  const panel=(title,share,change,contribution)=>`<div><h4>${title}</h4><p>${esc(D.majors[k])}</p><dl><dt>Expenditure share</dt><dd>${pct(share*100)}</dd><dt>Price change</dt><dd>${pctSigned(change)}</dd><dt>Model contribution</dt><dd>${signed(contribution,2)} pp</dd></dl></div>`;
  $('shock-comparison').innerHTML=panel('Original basket',baseline.shares[k],0,0)+panel('Modified scenario',weights[k],g,effect);
  $('shock-dollar-example').innerHTML=`On a <b>normalized $100 basket</b>, ${esc(SHORT[k].toLowerCase())} represents $${fmt(weights[k]*100,2)}: $${fmt(weights[k]*100,2)} ÷ $100 → <b>${pct(weights[k]*100)}</b>. This rescales shares; it is not an observed household budget.`;
  bars('shock-bars',weights.map((v,i)=>({label:SHORT[i],value:v*100,color:COLORS[i]})),{suffix:'%',precision:1,label:'Scenario expenditure shares; total 100 percent'});
  $('shock-result').innerHTML=`<b>One-period fixed-share model:</b> P = Σ wⱼ(1 + gⱼ/100), with only ${esc(SHORT[k].toLowerCase())} changing. Thus 100(P − 1) = wₖgₖ = ${fmt(weights[k])} × ${fmt(g,1)} = <b>${signed(effect,2)} pp</b>. Model aggregate change = <b>${pctSigned(effect)}</b>; factor = ${fmt(factor)}. Rounded display values may not reproduce the last decimal; calculations retain full precision.`;
 }
 function selected(){const k=+$('shock-category').value;$('shock-weight').value=weights[k]*100;shock();}
 function reset(){baseline=householdBasket($('shock-mode').value);weights=[...baseline.shares];weightsChanged=false;$('shock-category').value='0';$('shock-size').value='5';selected();}
 $('shock-mode').addEventListener('change',reset);$('shock-reset').addEventListener('click',reset);$('shock-category').addEventListener('change',selected);
 $('shock-weight').addEventListener('input',()=>{const k=+$('shock-category').value;weights=rebalanceShares(weights,k,+$('shock-weight').value/100,baseline.shares);weightsChanged=true;shock();});
 $('shock-size').addEventListener('input',shock);$('weight-period').addEventListener('change',show);show();reset();
}
function setupCollection(){
 makePlayer('collection-player',[
 {title:'Retailer',body:'Choose an outlet and a specified product offer: <b>500 g cereal, $5.00</b>. This miniature observation is illustrative.'},
 {title:'Product observation',body:'Record price, package size, product attributes and transaction conditions. Unit price: <b>$5 ÷ 0.5 kg = $10/kg</b>.'},
 {title:'Price validation',body:'Next package: <b>450 g, $5.00</b>. Compare consistent units: $5 ÷ 0.45 kg = $11.11/kg. Review the 11.1% movement and any quality change.'},
 {title:'Elementary index',body:'For one comparable offer, relative = <b>11.1111 ÷ 10 = 1.1111</b>. Multiple offers usually require an elementary formula, often Jevons.'},
 {title:'Higher-level index',body:'If this group has a synthetic 2% reference share and everything else is unchanged, factor = <b>0.02 × 1.1111 + 0.98 × 1 = 1.002222</b>: a 0.2% increase.'}
 ]);
 function pack(){const grams=+$('package-size').value,r=500/grams;$('package-illustration').innerHTML=svg(`<rect x="65" y="20" width="100" height="130" rx="8" fill="#b8d6a4"/><rect x="235" y="${150-grams/500*130}" width="100" height="${grams/500*130}" rx="8" fill="#efbe8b"/><text x="115" y="80" text-anchor="middle">500 g</text><text x="285" y="80" text-anchor="middle">${grams} g</text><text x="115" y="175" text-anchor="middle">$5.00</text><text x="285" y="175" text-anchor="middle">$5.00</text>`,400,195,'Package size at unchanged dollar price');$('package-result').textContent=`$5 / 0.500 kg = $10.00/kg; $5 / ${fmt(grams/1000,3)} kg = $${fmt(5/(grams/1000),2)}/kg. Relative ${fmt(r)} → ${pctSigned(100*(r-1))} pure unit-price change.`;}
 $('package-size').addEventListener('input',pack);pack();
}
function elementary(p0,p1){const r=p1.map((x,i)=>x/p0[i]);return {r,jevons:Math.exp(sum(r.map(Math.log))/r.length),carli:sum(r)/r.length,dutot:sum(p1)/sum(p0)};}
function miniUniverse(){
 return [
  {name:'Milk',branch:'Food proxy',p0:[4,5],p1:[4.4,MINI_STATE.milkSecondPrice],share:.2,color:COLORS[0],unit:'carton'},
  {name:'Rental service',branch:'Shelter proxy',p0:[1000,1200],p1:[1020,1224],share:.5,color:COLORS[1],unit:'month of service'},
  {name:'Gasoline',branch:'Transportation proxy',p0:[1.6,1.8],p1:[1.92,1.98],share:.3,color:COLORS[4],unit:'litre'}
 ].map(g=>({...g,index:elementary(g.p0,g.p1).jevons,expenditure0:100*g.share,quantity:100*g.share/g.p0[0]}));
}
function setupElementary(){
 const r=[{label:'Offer A',p0:10,p1:11},{label:'Offer B',p0:20,p1:18},{label:'Offer C',p0:5,p1:6}];
 function update(){const e=elementary(r.map(x=>x.p0),r.map(x=>x.p1));$('elementary-inputs').querySelectorAll('tbody tr').forEach((tr,i)=>{tr.querySelector('.relative').textContent=fmt(e.r[i]);tr.querySelector('.log').textContent=fmt(Math.log(e.r[i]));});bars('elementary-chart',[{label:'Jevons × 100',value:e.jevons*100,color:'#6c9d72'},{label:'Carli × 100',value:e.carli*100,color:'#d99862'},{label:'Dutot × 100',value:e.dutot*100,color:'#8c9cc5'}],{label:'Elementary price index formula comparison'});$('elementary-result').textContent=`Product of relatives = ${fmt(e.r.reduce((a,b)=>a*b,1))}; mean log = ${fmt(sum(e.r.map(Math.log))/e.r.length)}; exp(mean log) = ${fmt(e.jevons)}. Jevons growth ${pctSigned((e.jevons-1)*100)}; arithmetic relative ${fmt(e.carli)}; price-sum ratio ${fmt(e.dutot)}.`;}
 inputTable('elementary-inputs',r,[{key:'p0',label:'earlier price'},{key:'p1',label:'later price'}],update);
 $('reciprocal').addEventListener('click',()=>{r[0].p1=r[0].p0*2;r[1].p1=r[1].p0/2;r[2].p1=r[2].p0;const inputs=$('elementary-inputs').querySelectorAll('input[data-key=p1]');inputs.forEach((e,i)=>e.value=r[i].p1);update();$('elementary-result').textContent+=' The first two offers have reciprocal changes; the third is unchanged. Jevons stays at 1.';});update();
}
function indices(rows){const cost=(p,q)=>sum(rows.map(r=>r[p]*r[q])),L=cost('pt','q0')/cost('p0','q0'),P=cost('pt','qt')/cost('p0','qt'),Lo=cost('pt','qb')/cost('p0','qb');return {L,P,F:Math.sqrt(L*P),Lo,cost};}
function setupFormulas(){const r=[{label:'A · rice',p0:2,pt:3,q0:10,qt:6,qb:8},{label:'B · beans',p0:4,pt:4.2,q0:5,qt:8,qb:6}];
 function update(){const a=indices(r),names=[['Laspeyres','L'],['Paasche','P'],['Fisher','F'],['Lowe','Lo']];$('formula-results').innerHTML=names.map(([n,k])=>`<div class="formula-result">${n}<b>${indexText(a[k]*100)}</b>${pctSigned(100*(a[k]-1))}</div>`).join('');bars('formula-chart',names.map(([n,k],i)=>({label:n,value:100*a[k],color:COLORS[i]})),{label:'Fixed basket formula values'});$('formula-work').innerHTML=`<p>Earlier quantities: [${r.map(x=>fmt(x.q0,1)).join(', ')}]; later quantities: [${r.map(x=>fmt(x.qt,1)).join(', ')}]; separate basket: [${r.map(x=>fmt(x.qb,1)).join(', ')}].</p><p>L = (${r.map(x=>`${x.pt} × ${x.q0}`).join(' + ')}) / (${r.map(x=>`${x.p0} × ${x.q0}`).join(' + ')}) = <b>${fmt(a.cost('pt','q0'))} / ${fmt(a.cost('p0','q0'))} = ${fmt(a.L)}</b>.</p><p>P = ${fmt(a.cost('pt','qt'))} / ${fmt(a.cost('p0','qt'))} = ${fmt(a.P)}. Fisher = √(${fmt(a.L)} × ${fmt(a.P)}) = ${fmt(a.F)}.</p><p>Lowe = ${fmt(a.cost('pt','qb'))} / ${fmt(a.cost('p0','qb'))} = <b>${fmt(a.Lo)}</b>. Ratios become indices after multiplication by 100.</p>`;}
 inputTable('formula-inputs',r,[{key:'p0',label:'price at 0'},{key:'pt',label:'price at t'},{key:'q0',label:'quantity at 0',min:.1},{key:'qt',label:'quantity at t',min:.1},{key:'qb',label:'quantity at b',min:.1}],update);update();
}
function setupTrace(){
 options('trace-select',[{value:'milk',label:'Milk offers'},{value:'rent',label:'Rental-service offers'},{value:'gasoline',label:'Gasoline offers'}],'milk');
 const picked=()=>({milk:0,rent:1,gasoline:2}[$('trace-select').value]);
 const aggregate=groups=>sum(groups.map(g=>g.share*g.index));
 function update(){const groups=miniUniverse(),g=groups[picked()],all=aggregate(groups);
  $('trace-diagram').innerHTML=[`${esc(g.name)} offers`,`${esc(g.name)} elementary<br><b>${fmt(g.index)}</b>`,`${esc(g.branch)}<br><b>${fmt(g.index)}</b>`,`All-items proxy<br><b>${fmt(all)}</b>`].map((s,i)=>`<div class="trace-node ${i<3?'selected':''}">${s}</div>`).join('<span class="trace-arrow" aria-hidden="true">→</span>');
  $('trace-work').innerHTML=`<p>Same miniature universe: ${groups.map(x=>`${esc(x.name)} J = ${fmt(x.index)}`).join('; ')}.</p><p>Upper factor = ${groups.map(x=>`${fmt(x.share,2)} × ${fmt(x.index)}`).join(' + ')} = <b>${fmt(all)}</b>. A hypothetical 120.0 index becomes ${indexText(120*all)}; growth = ${pctSigned((all-1)*100)}.</p><p>Selected ${esc(g.name)} reference share = ${pct(g.share*100)}; contribution = ${pct(g.share*100)} × ${pct((g.index-1)*100)} / 100 = <b>${fmt(g.share*(g.index-1)*100,2)} pp</b>. Change the milk price in Chapter 6 to update this diagram and the detailed trace below.</p>`;
  PLAYERS['aggregation-player']?.refresh();
 }
 $('trace-select').addEventListener('change',update);update();
 makePlayer('aggregation-player',()=>{const groups=miniUniverse(),all=aggregate(groups);return [
  {title:'Comparable offers',body:groups.map(g=>`${esc(g.name)}: ${g.p0.map((p,i)=>money(p)+' → '+money(g.p1[i])).join('; ')}.`).join('<br>')},
  {title:'Elementary means',body:groups.map(g=>`${esc(g.name)}: exp(mean log relative) = <b>${fmt(g.index)}</b>.`).join('<br>')},
  {title:'Basic and intermediate',body:'In this synthetic hierarchy the elementary groups also act as basic classes. Single-child intermediate nodes pass their factor upward; no extra geometric averaging is introduced.'},
  {title:'Major components',body:groups.map(g=>`${esc(g.branch)}: factor ${fmt(g.index)}, reference share ${fmt(g.share*100,0)}%.`).join('<br>')+' Each branch contains one representative elementary group in this deliberately small example.'},
  {title:'All-items',body:`${groups.map(g=>`${fmt(g.share,2)} × ${fmt(g.index)}`).join(' + ')} = <b>${fmt(all)}</b>. 120 × this factor = ${indexText(120*all)}; inflation = ${pct((all-1)*100)}. Synthetic result only.`}
 ];});
}
function setupLink(){
 function weightsChange(){const y=$('basket-year').value,w=weight(y),n=weight('2025');bars('basket-change',w.map((v,i)=>({label:SHORT[i]+' · '+y+' → 2025',value:n[i]-v,color:COLORS[i]})),{suffix:' pp',precision:2,label:'Change in official reference-price expenditure shares'});}
 $('basket-year').addEventListener('change',weightsChange);weightsChange();
 const p=makePlayer('chain-player',()=>{const n=$('chain-basket').value==='new',wa=n?.3:.7,wb=1-wa,end=120*(wa*1.1+wb*.98);return [
 {title:'Earlier basket',body:'Old quantities imply 70% A and 30% B at the link. The published index climbs from <b>100 to 120</b> before the update.'},
 {title:'Common link',body:`At link prices both baskets are normalized to a factor of <b>1.0000</b>. Published level: <b>120.0</b>. Selected post-link shares: ${fmt(wa*100,0)}% A and ${fmt(wb*100,0)}% B.`},
 {title:'Reprice the basket',body:`After the link A moves ×1.10 and B ×0.98. Weighted factor = ${fmt(wa,2)} × 1.10 + ${fmt(wb,2)} × 0.98 = <b>${fmt(wa*1.1+wb*.98)}</b>.`},
 {title:'Connect growth',body:`Published linked index = 120 × ${fmt(wa*1.1+wb*.98)} = <b>${indexText(end)}</b>. Growth after the link is ${pctSigned(rate(end,120))}; the update itself caused no jump.`}
 ];});$('chain-basket').addEventListener('change',p.refresh);
 makePlayer('disinflation-player',[
 {title:'Starting level',body:'Synthetic price level <b>100.0</b>. Start with a reference basket costing $100.'},
 {title:'Inflation 6%',body:'100 × 1.06 = <b>106.0</b>. Prices increased by 6%.'},
 {title:'Inflation 3%',body:'106 × 1.03 = <b>109.2</b>. Inflation fell from 6% to 3%, but the price level increased again.'},
 {title:'Inflation 1%',body:'109.18 × 1.01 = <b>110.3</b>. Inflation slowed further; prices are still rising. Cumulative increase: 10.3%.'}
 ]);
}
function monthlyPoints(s,metric='level',from='2019'){return Object.entries(s.values).sort(([a],[b])=>a.localeCompare(b)).filter(([t])=>t>=from).map(([t,v])=>{if(metric==='level')return {x:t,y:v};const [y,m]=t.split('-').map(Number),prev=metric==='yoy'?`${y-1}-${String(m).padStart(2,'0')}`:m===1?`${y-1}-12`:`${y}-${String(m-1).padStart(2,'0')}`;return {x:t,y:s.values[prev]?rate(v,s.values[prev]):NaN};}).filter(p=>Number.isFinite(p.y));}
function setupHistory(){
 const choices=['All-items',...D.majors,'All-items excluding gasoline','Energy','Goods','Services'];options('history-series',choices,'All-items');
 function update(){const label=$('history-series').value,metric=$('history-metric').value,pts=monthlyPoints(series(label),metric,$('history-start').value);lineChart('history-chart',[{name:label,points:pts,color:'#327b66'}],{label:'Official monthly Canadian CPI '+metric,suffix:metric==='level'?' index':'%',detail:'history-detail',fill:true});$('history-table').innerHTML=table(['Month',metric==='level'?'Index level':'Rate, %'],pts.map(p=>[p.x,fmt(p.y,1)]));}
 attach(['history-series','history-metric','history-start'],update,'change');update();
}
function setupSeason(){
 function synth(){const a=+$('season-amplitude').value,pts=Array.from({length:36},(_,i)=>({x:`Y${Math.floor(i/12)+1}-${String(i%12+1).padStart(2,'0')}`,trend:100+.45*i,season:a*Math.sin(2*Math.PI*(i%12)/12),resid:.5*Math.sin(i*2.3)}));lineChart('synthetic-season',[{name:'Observed Y',color:'#d7874b',points:pts.map(p=>({x:p.x,y:p.trend+p.season+p.resid}))},{name:'Known adjusted Y − S',color:'#327e68',points:pts.map(p=>({x:p.x,y:p.trend+p.resid}))},{name:'Supplied trend T',color:'#8e87b8',points:pts.map(p=>({x:p.x,y:p.trend}))}],{label:'Known synthetic seasonal decomposition'});const p=pts[15];$('season-result').textContent=`Example ${p.x}: T = ${fmt(p.trend,1)}, S = ${fmt(p.season,1)}, R = ${fmt(p.resid,1)}. Y = ${fmt(p.trend+p.season+p.resid,1)}; Y − S = ${fmt(p.trend+p.resid,1)}. The known seasonal amplitude is ${fmt(a,1)} points.`;}
 $('season-amplitude').addEventListener('input',synth);synth();options('season-series',['All-items',...D.majors],'All-items');
 function actual(){const label=$('season-series').value,sa=D.seasonal[label],nsa=series(label),months=Object.keys(sa.values).filter(t=>t>='2025-01'&&t<='2026-09').sort();lineChart('actual-season',[{name:'Unadjusted',color:'#d98548',points:months.map(t=>({x:t,y:nsa.values[t]}))},{name:'Official adjusted',color:'#397fac',points:months.map(t=>({x:t,y:sa.values[t]}))}],{label:'Actual adjusted versus unadjusted '+label});$('season-actual-table').innerHTML=table(['Month','Unadjusted index','Adjusted index','NSA MoM, %','SA MoM, %'],['2026-07','2026-08','2026-09'].map(t=>{const m=+t.slice(-2),pr=`2026-${String(m-1).padStart(2,'0')}`;return [t,fmt(nsa.values[t],1),fmt(sa.values[t],1),fmt(rate(nsa.values[t],nsa.values[pr]),1),fmt(rate(sa.values[t],sa.values[pr]),1)];}));}
 $('season-series').addEventListener('change',actual);actual();
}
function setupWaterfall(){
 function update(){const gs=+$('gas-share').value/100,g=+$('gas-shock').value,shares=[gs,...[.15,.2,.3,.3].map(s=>s/.95*(1-gs))],changes=[g,1,2,3,-1],labels=['Gasoline','Other transport','Food','Shelter','Other'],c=shares.map((s,i)=>s*changes[i]),total=sum(c),w=800,h=340,L=60,R=30,T=35,B=65,vals=[0,total];let acc=0;c.forEach(v=>{acc+=v;vals.push(acc);});const low=Math.min(...vals,0)-.4,high=Math.max(...vals,0)+.6,y=v=>h-B-(v-low)/(high-low)*(h-T-B),bw=75,step=(w-L-R)/6;let content='';for(let i=0;i<5;i++){const v=low+(high-low)*i/4;content+=`<line class="gridline" x1="${L}" x2="${w-R}" y1="${y(v)}" y2="${y(v)}"/><text class="axis" x="${L-8}" y="${y(v)+4}" text-anchor="end">${fmt(v,1)}</text>`;}acc=0;c.concat(total).forEach((v,i)=>{const from=i===5?0:acc,to=from+v,x=L+i*step+14;content+=`<rect x="${x}" y="${Math.min(y(from),y(to))}" width="${bw}" height="${Math.max(1,Math.abs(y(to)-y(from)))}" fill="${i===5?'#344f60':v>=0?COLORS[i]:'#9d87bb'}" rx="3"/><text x="${x+bw/2}" y="${Math.min(y(from),y(to))-10}" text-anchor="middle">${signed(v,2)}</text><text x="${x+bw/2}" y="${h-38}" text-anchor="middle">${labels[i]??'Total'}</text>`;if(i<4)content+=`<line x1="${x+bw}" x2="${x+step}" y1="${y(to)}" y2="${y(to)}" stroke="#8da299" stroke-dasharray="3 3"/>`;if(i<5)acc=to;});$('waterfall').innerHTML=svg(content,w,h,'Synthetic percentage point contributions; '+c.map((v,i)=>labels[i]+' '+fmt(v,2)+' pp').join('; '));$('waterfall-result').textContent=`Gasoline: ${pct(shares[0]*100)} share × ${pctSigned(g)} change = ${signed(c[0],2)} pp. All contributions: ${c.map(v=>signed(v,2)).join(' ')} = ${signed(total,2)} percentage points. Aggregate factor: ${fmt(1+total/100)}.`;}
 attach(['gas-share','gas-shock'],update);update();
}
function weightedCore(items){const a=items.map(x=>({...x})).sort((x,y)=>x.x-y.x);let cum=0,trim=0,median=null;a.forEach(r=>{r.start=cum;r.end=cum+r.w;r.retained=Math.max(0,Math.min(.8,r.end)-Math.max(.2,r.start));trim+=r.retained*r.x;if(median===null&&r.end>=.5-1e-12)median=r.x;cum=r.end;});return {items:a,mean:sum(a.map(r=>r.w*r.x)),median,trim:trim/.6};}
function setupCore(){
 function update(){const c=weightedCore([[-4,.08],[-1,.17],[.1,.12],[.3,.18],[.5,.13],[.8,.14],[2,.1],[+$('core-outlier').value,.08]].map(([x,w],i)=>({x,w,label:String.fromCharCode(65+i)}))),w=800,h=260,L=40,R=25,ww=w-L-R;let s='';c.items.forEach((r,i)=>{const x=L+r.start*ww,width=r.w*ww;s+=`<rect x="${x}" y="45" width="${width}" height="105" fill="${COLORS[i]}" opacity=".65"/><text x="${x+width/2}" y="76" text-anchor="middle">${r.label}</text><text x="${x+width/2}" y="104" text-anchor="middle">${fmt(r.x,1)}%</text><text x="${x+width/2}" y="132" text-anchor="middle">${fmt(r.w*100,0)}% wt</text>`;});s+=`<rect x="${L}" y="43" width="${.2*ww}" height="109" fill="#df8b46" opacity=".35"/><rect x="${L+.8*ww}" y="43" width="${.2*ww}" height="109" fill="#df8b46" opacity=".35"/>`;[0,.2,.5,.8,1].forEach(v=>s+=`<path d="M${L+v*ww} 35v130" stroke="${v===.5?'#235f4a':'#597978'}" stroke-dasharray="${v===.5?'0':'4 4'}"/><text x="${L+v*ww}" y="185" text-anchor="middle">${fmt(v*100,0)}%</text>`);s+='<text x="400" y="220" text-anchor="middle">Cumulative expenditure weight (sorted by monthly price change)</text>';$('core-distribution').innerHTML=svg(s,w,h,'Synthetic weighted trim and median distribution');$('core-result').innerHTML=`Weighted mean = <b>${pctSigned(c.mean)}</b>; weighted median = <b>${pctSigned(c.median)}</b>; trimmed middle-60% mean = <b>${pctSigned(c.trim)}</b>. Tail weight removed = 20% + 20%. Retained weight = ${pct(sum(c.items.map(r=>r.retained))*100)}.`;$('core-mini-table').innerHTML=table(['Category','Monthly change, %','Weight, %','Retained weight, %'],c.items.map(r=>[r.label,fmt(r.x,1),fmt(r.w*100,1),fmt(r.retained*100,2)]));}
 $('core-outlier').addEventListener('input',update);update();$('core-official').innerHTML=['CPI-trim','CPI-median','CPI-common'].map((l,i)=>stat(l,fmt(D.core[l],1)+'%','September 2026 annual rate · release Table 4',COLORS[i])).join('');$('special-core').innerHTML=table(['Public alternative aggregate','September index','Calculated YoY, %'],['All-items excluding gasoline','All-items excluding energy','All-items excluding food and energy'].map(l=>{const v=series(l).values;return [esc(l),fmt(v['2026-09'],1),fmt(rate(v['2026-09'],v['2025-09']),1)];}));
}
function setupGeography(){
 const geos=Object.keys(D.geographies).filter(g=>D.series[g+'.2']).map(g=>({value:g,label:D.geographies[g].label}));options('geo-select',geos,'2');const tiles=[['2','CA'],['3','NL'],['5','PE'],['7','NS'],['9','NB'],['11','QC'],['14','ON'],['18','MB'],['20','SK'],['23','AB'],['26','BC'],['29','Whitehorse'],['30','Yellowknife'],['31','Iqaluit']];
 $('geography-tiles').innerHTML=tiles.map(([g,l])=>{const v=D.series[g+'.2'].values,r=rate(v['2026-09'],v['2025-09']);return `<button type="button" class="geo-tile" data-geo="${g}" aria-pressed="${g==='2'}" title="${esc(D.geographies[g].label)}" style="--tint:${r>4.5?'#f6e1c9':r>3?'#e8e3f4':'#e0eee6'}">${l}<strong>${fmt(r,1)}%</strong><span>YoY</span></button>`;}).join('');
 function update(){const g=$('geo-select').value,s=D.series[g+'.2'],v=s.values,latest=v['2026-09'];$('geography-tiles').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.geo===g)));$('geo-stats').innerHTML=stat('September index',fmt(latest,1),esc(s.index_reference??s.unit),COLORS[1])+stat('Unadjusted MoM',signed(rate(latest,v['2026-08']),1)+'%',esc(s.geography),COLORS[4])+stat('Year-over-year',signed(rate(latest,v['2025-09']),1)+'%',esc(s.geography),COLORS[0]);lineChart('geo-chart',[{name:s.geography+' · YoY',color:'#8b79b5',points:monthlyPoints(s,'yoy','2019')}],{label:'Regional CPI annual inflation',suffix:'%'});$('geo-components').innerHTML=table(['Available major component','Sep 2026 index','Calculated MoM, %','Calculated YoY, %'],D.majors.map(l=>{const ss=series(l,g),v=ss?.values??{},a=v['2026-09'];return [esc(l),fmt(a,1),fmt(a&&v['2026-08']?rate(a,v['2026-08']):NaN,1),fmt(a&&v['2025-09']?rate(a,v['2025-09']):NaN,1)];}),'Geography: '+esc(s.geography)+'. Values are within-geography changes; missing components remain unavailable.');}
 $('geo-select').addEventListener('change',update);$('geography-tiles').addEventListener('click',e=>{const b=e.target.closest('[data-geo]');if(b){$('geo-select').value=b.dataset.geo;update();}});update();
}
function setupMoney(){const months=Object.keys(national).sort();options('money-base',months,'2002-01');options('money-target',months,'2026-09');
 function update(){const input=$('money-value');if(!input.validity.valid||input.value==='')return;const v=+input.value,b=$('money-base').value,t=$('money-target').value,ib=national[b],it=national[t],equivalent=v*it/ib,real=v*ib/it;$('money-result').innerHTML=`<p>${esc($('money-context').value)} comparison: CPI(${b}) = ${fmt(ib,1)}; CPI(${t}) = ${fmt(it,1)}.</p><p><b>${money(v)}</b> in ${b} requires <b>${money(equivalent)}</b> in ${t} for CPI-equivalent purchasing power: ${money(v)} × ${fmt(it/ib)}.</p><p>If the nominal amount stays at ${money(v)} in ${t}, it buys <b>${money(real)}</b> in ${b} purchasing-power dollars. Period inflation: ${pctSigned(rate(it,ib))}.</p>`;bars('money-chart',[{label:`Amount at ${b}`,value:v,color:'#dfad65'},{label:`Equivalent at ${t}`,value:equivalent,color:'#83b49d'}],{suffix:' dollars',precision:2,label:'Purchasing power conversion'});}
 attach(['money-context','money-base','money-target'],update,'change');$('money-value').addEventListener('input',update);update();
}
function setupAudit(){
 const all=V.rows;$('audit-stats').innerHTML=stat('Independent release comparisons',String(V.independent_comparisons),V.summary.Matched+' matched · '+V.summary['Within rounding tolerance']+' within rounding tolerance',COLORS[5])+stat('Comparison failures',String(V.failures.length),'Index levels and growth-rate checks',COLORS[4])+stat('Source observations scanned','1,154,918','26 data XML files · 499 extracted series',COLORS[1]);
 const rates=all.filter(r=>r.category==='A'&&r.metric==='YoY'&&r.geography==='Canada'&&D.majors.includes(r.series));
 // The paired endpoints expose small differences hidden by one-decimal display.
 const w=800,h=390,L=225,R=60,T=35,scale=x=>L+x/9*(w-L-R);let s='';for(let i=0;i<=9;i+=3)s+=`<line class="gridline" x1="${scale(i)}" x2="${scale(i)}" y1="20" y2="340"/><text x="${scale(i)}" y="365" text-anchor="middle">${i}%</text>`;rates.forEach((r,i)=>{const y=T+i*38;s+=`<text x="${L-12}" y="${y+4}" text-anchor="end">${esc(SHORT[D.majors.indexOf(r.series)])}</text><line x1="${scale(r.official)}" x2="${scale(r.calculated)}" y1="${y}" y2="${y}" stroke="${COLORS[i]}" stroke-width="3"/><circle cx="${scale(r.official)}" cy="${y}" r="5" fill="white" stroke="#435a65"/><circle cx="${scale(r.calculated)}" cy="${y}" r="3" fill="${COLORS[i]}"/><text x="${scale(Math.max(r.official,r.calculated))+10}" y="${y+4}">${fmt(r.calculated)} / ${fmt(r.official,1)}</text>`;});$('replication-chart').innerHTML=svg(s,w,h,'Calculated component annual inflation versus published rates')+'<p class="hint">Filled dot: ratio calculated from public indices. Open dot: official one-decimal rate. Label: calculated / official. Tiny separations are expected from display rounding.</p>';
 function update(){const term=$('audit-search').value.trim().toLowerCase(),level=$('audit-level').value,metric=$('audit-metric').value,rows=all.filter(r=>(level==='all'||r.category===level)&&(metric==='all'||r.metric===metric)&&(!term||[r.series,r.geography,r.result].join(' ').toLowerCase().includes(term)));$('audit-count').textContent=`${rows.length} of ${all.length} audit rows. Calculations without a release comparator remain explicitly labelled. Horizontally scroll the contained table for all columns.`;
 $('audit-table').innerHTML=table(['Series / metric','Geography','Reference month','Official','Calculated / extracted','Difference','Tolerance','Level','Result','Source'],rows.map(r=>[esc(r.series)+'<br><small>'+esc(r.metric)+'</small>',esc(r.geography),r.reference_period,r.official===null?'—':fmt(r.official,1),r.calculated===null?'—':fmt(r.calculated),r.difference===null?'—':signed(r.difference),['C','D'].includes(r.category)?'—':r.tolerance<1e-7?'1e−8':fmt(r.tolerance,2),'Level '+r.category,`<span class="badge ${r.category==='C'?'approx':r.category==='D'?'unavailable':''}">${esc(r.result)}</span>`,`<a href="${auditSource(r)}">${esc(r.source)}</a>`]));}
 function auditSource(r){return r.source.includes('Bank of Canada')?'https://www.bankofcanada.ca/rates/price-indexes/cpi/':r.source.includes('0007')?'https://doi.org/10.25318/1810000701-eng':r.source.includes('0006')?'https://doi.org/10.25318/1810000601-eng':r.source.includes('Unpublished')?'#s16':'https://doi.org/10.25318/1810000401-eng';}
 attach(['audit-search'],update);attach(['audit-level','audit-metric'],update,'change');update();
 const a=V.aggregation;$('aggregation-work').innerHTML=`<p>April headline anchor = <b>${fmt(a.link_index,1)}</b>; normalized eight-component factor = <b>${fmt(a.factor,8)}</b>.</p><p>Estimated September index = ${fmt(a.link_index,1)} × ${fmt(a.factor,8)} = <b>${fmt(a.estimated_index)}</b>. Published = <b>170.0</b>; difference = <b>${signed(a.index_difference)} index points</b>.</p><p>Cross-basket annual approximation = <b>${fmt(a.estimated_yoy)}%</b>, versus official 3.1%. This is still Level C.</p><p>For comparison, directly weighting the component <em>levels</em> would give ${fmt(a.naive_weighted_levels)}, a different and unjustified result. The component reference scales are not link-month relatives.</p>`;
 $('aggregation-inputs').innerHTML=table(['Component','April weight, %','Normalized share','April index','September index','Relative','Weighted factor'],D.majors.map((l,i)=>{const v=series(l).values,r=v['2026-09']/v['2026-04'];return [esc(l),fmt(weight('2025','link')[i],2),fmt(a.weights_2025[i],6),fmt(v['2026-04'],1),fmt(v['2026-09'],1),fmt(r,6),fmt(a.weights_2025[i]*r,6)];}));
 const rounds=[['NSA YoY',170,164.9],['NSA MoM',170,169.8],['SA MoM',170,169.4]];$('rounding-work').innerHTML=table(['Comparison','Rate from displayed indices, %','Possible input-rounding interval, %'],rounds.map(([l,a,b])=>[l,fmt(rate(a,b)),`[${fmt(rate(a-.05,b+.05))}, ${fmt(rate(a+.05,b-.05))})`]))+'<p>The comparison test allows ±0.05 percentage points around the official reported rate. It separately displays the wider input-rounding intervals; those intervals are not used to override a failed test.</p>';
}
function seeded(seed){return ()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};}
function sampleWithoutReplacement(pop,n,rng){const a=pop.slice();for(let i=0;i<n;i++){const j=i+Math.floor(rng()*(a.length-i));[a[i],a[j]]=[a[j],a[i]];}return a.slice(0,n);}
function setupSampling(){let draw=1;const rng=seeded(38219),pop=Array.from({length:240},()=>Math.exp(.025+(rng()-.5)*.32)),truth=(Math.exp(sum(pop.map(Math.log))/pop.length)-1)*100;
 function update(){const n=+$('sample-size').value,rng=seeded(draw*1013+n),est=Array.from({length:60},()=>{const sample=sampleWithoutReplacement(pop,n,rng);return (Math.exp(sum(sample.map(Math.log))/n)-1)*100;}),w=800,h=250,L=50,R=30,lo=Math.min(truth,...est)-.4,hi=Math.max(truth,...est)+.4,x=v=>L+(v-lo)/(hi-lo)*(w-L-R);let s=`<line x1="${x(truth)}" x2="${x(truth)}" y1="35" y2="190" stroke="#257359" stroke-width="2"/><text x="${x(truth)}" y="22" text-anchor="middle">Population ${pct(truth)}</text>`;est.forEach((v,i)=>s+=`<circle cx="${x(v)}" cy="${75+(i%6)*18}" r="${i===0?7:4}" fill="${i===0?'#da8143':'#91adbc'}" opacity="${i===0?1:.65}"><title>Sample ${i+1}: ${pct(v)}</title></circle>`);[0,.25,.5,.75,1].forEach(f=>{const v=lo+f*(hi-lo);s+=`<text class="axis" x="${x(v)}" y="220" text-anchor="middle">${pct(v)}</text>`;});$('sample-chart').innerHTML=svg(s,w,h,'60 synthetic sampled geometric means compared with complete synthetic population');$('sample-result').textContent=`Sample size ${n} of 240; highlighted estimate ${pct(est[0])}; population geometric-mean change ${pct(truth)}; difference ${signed(est[0]-truth,2)} pp. Across 60 draws, mean absolute deviation = ${fmt(sum(est.map(v=>Math.abs(v-truth)))/est.length,2)} pp. Draw seed ${draw*1013+n}.`;}
 $('sample-size').addEventListener('input',update);$('sample-again').addEventListener('click',()=>{draw++;update();});update();
}
function setupSynthesis(){makePlayer('synthesis-player',()=>{const groups=miniUniverse(),factor=sum(groups.map(g=>g.share*g.index));return [
 {title:'Define scope',body:'Define whose consumption and which transactions the CPI measures. A balance-sheet asset purchase is not automatically consumption.'},
 {title:'Identify basket',body:'Organize products into representative classes: <b>8 major components, 220 basic aggregates and 491 elementary product classes</b> in the 2026 update.'},
 {title:'Determine weights',body:'Synthetic miniature: milk $20 + rental service $50 + gasoline $30 = <b>$100</b>. Normalize to shares <b>0.20, 0.50 and 0.30</b>. These teaching shares are distinct from official 2025 reference-price and April 2026 link-price weights.'},
 {title:'Select offers',body:'Choose specified product offers and outlets across geographic strata; preserve characteristics needed for like-for-like price comparisons.'},
 {title:'Validate prices',body:'Record comparable prices, quantities and taxes. Apply appropriate quality treatment and imputation; do not confuse a smaller package with an unchanged unit price.'},
 {title:'Elementary change',body:`Synthetic milk offers: $4 → $4.40 and $5 → $${fmt(MINI_STATE.milkSecondPrice,2)}. <b>J = √(1.10 × ${fmt(MINI_STATE.milkSecondPrice/5)}) = ${fmt(groups[0].index)}</b>. This is the equal-offer elementary factor, not an expenditure-weighted upper aggregate.`},
 {title:'Weighted aggregation',body:`Synthetic Lowe factor: ${groups.map(g=>`${fmt(g.share,2)} × ${fmt(g.index)}`).join(' + ')} = <b>${fmt(factor)}</b>. From a hypothetical earlier level 120, the later level is <b>${indexText(120*factor)}</b>. The miniature growth is ${pct(100*(factor-1))}.`},
 {title:'Update and link',body:'At link month ℓ, new-basket factor = 1. Multiply subsequent new-basket factors by <b>Iℓ</b>; continuity does not require identical dollar basket costs.'},
 {title:'Publish index',body:'September 2026 headline CPI = <b>170.0 (2002 = 100)</b>. Published precision differs from internal calculation precision.'},
 {title:'Calculate inflation',body:'170.0 / 164.9 − 1 → <b>3.0928%, reported 3.1%</b>. Adjacent-month unadjusted rate → 0.1178%, reported 0.1%. These are Level A arithmetic checks, not microdata reconstruction.'}
 ];});}
function navigation(){const links=[...document.querySelectorAll('.contents nav a[data-section]')],chapters=[...document.querySelectorAll('.chapter')];function mark(id){links.forEach(a=>{const on=a.dataset.section===id;a.classList.toggle('active',on);if(on)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});$('chapter-jump').value=id;$('reading-progress').textContent=(+id.slice(1))+' / 18';}
 const observer=new IntersectionObserver(entries=>{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);if(visible.length)mark(visible[0].target.id);},{rootMargin:'-10% 0px -60% 0px',threshold:0});chapters.forEach(s=>observer.observe(s));$('chapter-jump').addEventListener('change',e=>$(e.target.value).scrollIntoView({behavior:'auto'}));mark('s01');}
function init(){
 if(!D||!V){document.body.insertAdjacentHTML('afterbegin','<p class="noscript">The local data assets did not load. Check the dedicated asset folder and relative paths.</p>');return;}
 $('headline-stats').innerHTML=headlineStat('All-items CPI',fmt(national['2026-09'],1),'September 2026 · 2002 = 100',0)+headlineStat('Year-over-year inflation',pctSigned(rate(national['2026-09'],national['2025-09'])),'Versus September 2025',1)+headlineStat('Monthly · unadjusted',pctSigned(rate(national['2026-09'],national['2026-08'])),'Versus August 2026',2)+headlineStat('Monthly · seasonally adjusted',pctSigned(rate(D.seasonal['All-items'].values['2026-09'],D.seasonal['All-items'].values['2026-08'])),'Published seasonal adjustment',3);
 renderMath();setupBasket();setupClassification();setupWeights();setupCollection();setupElementary();setupFormulas();setupTrace();setupLink();setupHistory();setupSeason();setupWaterfall();setupCore();setupGeography();setupMoney();setupAudit();setupSampling();setupSynthesis();navigation();
 ['waterfall','core-distribution','replication-chart','normal-chart','sample-chart'].forEach(id=>$(id).classList.add('wide-chart'));
 window.CPI_APP={rate,elementary,indices,weightedCore,normalize,householdBasket,rebalanceShares,pct,pctSigned,sampleWithoutReplacement,version:'1.1'};
 window.CPI_UI={D,V,$,esc,fmt,indexText,signed,pct,pctSigned,money,sum,rate,COLORS,TINTS,SHORT,productId,series,national,weight,normalize,attach,options,table,stat,svg,bars,lineChart,icon,makePlayer,monthlyPoints,seeded,sampleWithoutReplacement,miniUniverse,miniState:MINI_STATE,players:PLAYERS};
}
init();
})();
