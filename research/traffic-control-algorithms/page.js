/* Rendering and explanatory sequencing. Models live in simulation.js. */
(() => {
  'use strict';
  const M = window.TrafficModels;
  const $ = id => document.getElementById(id);
  const colors = {ink: '#233431', muted: '#59625d', road: '#737d79', line: '#eee8d8', cream: '#f3efe6', green: '#287352', red: '#a3483e', blue: '#286787', gold: '#a87524', amber: '#e7ba58', peach: '#e9c4ab'};
  const palette = ['#286787', '#b76646', '#e6b65a', '#46745e', '#f3e6cc', '#746486'];
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  let globalPaused = reduced.matches;
  const jobs = [];
  const observer = new IntersectionObserver(entries => entries.forEach(e => jobs.filter(j => j.element === e.target).forEach(j => { j.visible = e.isIntersecting; })), {rootMargin: '100px'});
  function job(element, update, playing = true) {
    const item = {element: $(element), update, playing: playing && !reduced.matches, visible: false}; jobs.push(item); observer.observe(item.element); return item;
  }
  function canvasHeight(id, height) { if ($(id).height !== height) $(id).height = height; }
  function setText(id, value) { const e = $(id); if (e.textContent !== String(value)) e.textContent = value; }
  function range(id, out, suffix = '') { setText(out, $(id).value + suffix); return Number($(id).value); }
  function bind(ids, fn) { ids.forEach(id => $(id).addEventListener('input', fn)); fn(); }
  function resume(item) { globalPaused = false; item.playing = true; updateMotion(); syncButtons(); }
  function toggle(item, button, noun = '') { item.playing = globalPaused ? true : !item.playing; if (item.playing) { globalPaused = false; updateMotion(); } button.textContent = `${item.playing ? 'Pause' : 'Play'}${noun ? ' ' + noun : ''}`; syncButtons(); }
  function updateMotion() { setText('all-motion', globalPaused ? 'Resume all motion' : 'Pause all motion'); }
  $('all-motion').onclick = () => { globalPaused = !globalPaused; if (!globalPaused) jobs.filter(j => j.autostart).forEach(j => { j.playing = true; }); updateMotion(); syncButtons(); };
  reduced.addEventListener('change', e => { if (e.matches) { globalPaused = true; updateMotion(); } });
  if (reduced.matches) setText('motion-note', 'Reduced motion: animations start paused. Step buttons and time sliders work without autoplay.');
  updateMotion();
  function svgText(x, y, text, size = 18, fill = colors.ink, anchor = 'start') { return `<text x="${x}" y="${y}" font-family="Arial,sans-serif" font-size="${size}" fill="${fill}" text-anchor="${anchor}">${text}</text>`; }
  function rect(x, y, w, h, fill, rx = 5, extra = '') { return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" ${extra}/>`; }
  function line(x1, y1, x2, y2, stroke = colors.ink, width = 2, extra = '') { return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${width}" ${extra}/>`; }
  const angles = {N: 0, S: 180, E: 90, W: 270};
  function svgRoad() {
    let s = rect(0, 0, 600, 600, '#ede8dc');
    [[20, 25, 170, 145], [413, 40, 152, 120], [32, 425, 154, 145], [423, 431, 145, 130]].forEach(([x,y,w,h]) => { s += rect(x + 5, y + 6, w, h, '#ddd5c5', 12) + rect(x, y, w, h, '#e4d8c2', 12); });
    s += rect(215, 0, 170, 600, '#d2d0c2', 0) + rect(0, 215, 600, 170, '#d2d0c2', 0) + rect(230, 0, 140, 600, colors.road, 0) + rect(0, 230, 600, 140, colors.road, 0);
    for (const a of [0,90,180,270]) {
      s += `<g transform="rotate(${a} 300 300)">` + line(300, 0, 300, 185, '#e5c873', 3) + line(263, 0, 263, 174, '#dcded4', 2, 'stroke-dasharray="12 12"') + line(337, 0, 337, 174, '#dcded4', 2, 'stroke-dasharray="12 12"') + line(235, 199, 298, 199, '#fff9e8', 5);
      for (let x = 234; x < 366; x += 13) s += rect(x, 209, 8, 13, '#f7f3e5', 0);
      s += '</g>';
    }
    [[190,190],[410,190],[190,410],[410,410]].forEach(([x,y]) => { s += `<circle cx="${x+3}" cy="${y+3}" r="16" fill="#c5c5ab"/><circle cx="${x}" cy="${y}" r="16" fill="#92a783"/><circle cx="${x-4}" cy="${y-4}" r="10" fill="#a5b794"/>`; });
    return s + svgText(320,30,'N',20) + svgText(320,583,'S',20) + svgText(579,285,'E',20) + svgText(12,285,'W',20);
  }
  function movementPath(name, color, dashed = false) {
    const [origin, kind] = name.split('_');
    const path = kind === 'T' ? 'M278 70 L278 528' : kind === 'L' ? 'M294 70 L294 178 C294 270 327 312 426 322 L529 322' : 'M247 70 L247 178 Q247 278 165 278 L70 278';
    return `<g transform="rotate(${angles[origin]} 300 300)"><path d="${path}" fill="none" stroke="${color}" stroke-width="7" ${dashed ? 'stroke-dasharray="10 7"' : ''}/><path d="${kind === 'T' ? 'M270 516 L278 532 L286 516' : kind === 'L' ? 'M517 314 L533 322 L517 330' : 'M82 270 L66 278 L82 286'}" fill="none" stroke="${color}" stroke-width="6"/></g>`;
  }
  function pedestrianPaths(axis = 'both') {
    let s = '';
    if (axis === 'both' || axis === 'EW') [215,385].forEach(y => { s += line(216,y,384,y,'#193f67',6,'stroke-dasharray="7 5"'); });
    if (axis === 'both' || axis === 'NS') [215,385].forEach(x => { s += line(x,216,x,384,'#193f67',6,'stroke-dasharray="7 5"'); });
    return s;
  }
  function drawMovement() {
    let s = svgRoad();
    document.querySelectorAll('#movement-controls input:checked').forEach(input => {
      if (input.value === 'P') s += pedestrianPaths();
      else ['N','S','E','W'].forEach((o,i) => { s += movementPath(`${o}_${input.value}`, [colors.blue, '#f4cd77', '#b94335', '#e8eee0'][i], input.value === 'L'); });
    });
    $('movement-svg').innerHTML = s;
  }
  document.querySelectorAll('#movement-controls input').forEach(e => e.onchange = drawMovement); drawMovement();
  function canvasText(c, text, x, y, size = 16, color = colors.ink, align = 'left') { c.fillStyle = color; c.font = `${size}px Arial`; c.textAlign = align; c.fillText(text, x, y); }
  function canvasRect(c, x, y, w, h, color, radius = 0) { c.fillStyle = color; c.beginPath(); c.roundRect(x, y, w, h, radius); c.fill(); }
  function car(c, x, y, angle, color, bus = false, scale = 1) {
    c.save(); c.translate(x,y); c.rotate(angle); c.scale(scale,scale);
    canvasRect(c,-9,-(bus ? 14 : 12),18,bus ? 28 : 24,'#00000020',4);
    canvasRect(c,-8,-(bus ? 14 : 12),16,bus ? 28 : 24,color,4);
    canvasRect(c,-6,-7,12,6,'#d7e9e9',2); canvasRect(c,-6,5,12,4,'#344a4d',1);
    if (bus) { canvasRect(c,-6,-14,12,4,'#d7e9e9',1); }
    canvasRect(c,-7,10,4,2,'#f3e7aa',1); canvasRect(c,3,10,4,2,'#f3e7aa',1); c.restore();
  }
  function person(c,x,y,color = colors.ink) { c.fillStyle = color; c.beginPath(); c.arc(x,y-3,3,0,Math.PI*2); c.fill(); canvasRect(c,x-2,y,4,8,color,2); }
  function heroBackground(c) {
    canvasRect(c,0,0,600,600,'#ede8dc');
    [[20,30,164,129],[420,23,154,145],[27,428,150,142],[421,423,153,151]].forEach(([x,y,w,h],i) => {
      canvasRect(c,x+5,y+7,w,h,'#d9d1bf',13); canvasRect(c,x,y,w,h,i%2 ? '#e3cbb8':'#dfd4bd',13);
      canvasRect(c,x+12,y+12,w-24,h-24,'#eae0ce',8);
      for (let dx = 23; dx < w-15; dx+=27) canvasRect(c,x+dx,y+20,14,7,'#b7c7c0',2);
      canvasText(c,['LIBRARY','CAFÉ','GARDEN COURT','WORKSHOP'][i],x+w/2,y+h/2,10,'#736c5c','center');
    });
    canvasRect(c,214,0,172,600,'#d0cdbf'); canvasRect(c,0,214,600,172,'#d0cdbf');
    canvasRect(c,230,0,140,600,colors.road); canvasRect(c,0,230,600,140,colors.road);
    for (const angle of [0, Math.PI/2,Math.PI,Math.PI*1.5]) {
      c.save(); c.translate(300,300); c.rotate(angle); c.translate(-300,-300);
      c.strokeStyle = '#e7cb77'; c.lineWidth = 2; c.beginPath(); c.moveTo(299,0); c.lineTo(299,185); c.moveTo(303,0); c.lineTo(303,185); c.stroke();
      c.strokeStyle='#d9ded7'; c.setLineDash([11,13]); c.beginPath(); c.moveTo(255,0); c.lineTo(255,175); c.moveTo(347,0); c.lineTo(347,175); c.stroke(); c.setLineDash([]);
      canvasRect(c,234,199,63,4,'#fff9e8');
      for(let x=236;x<364;x+=13) canvasRect(c,x,211,7,12,'#f4f1e1');
      c.strokeStyle='#e6c175';c.lineWidth=1.5;c.strokeRect(267,145,22,30);
      c.restore();
    }
    [[193,190],[409,190],[190,410],[410,411],[100,192],[505,405],[190,85],[411,516]].forEach(([x,y])=>{
      c.fillStyle='#00000012';c.beginPath();c.arc(x+3,y+4,17,0,7);c.fill();c.fillStyle='#91a17e';c.beginPath();c.arc(x,y,16,0,7);c.fill();c.fillStyle='#a6b591';c.beginPath();c.arc(x-4,y-4,10,0,7);c.fill();
    });
    canvasText(c,'N',320,24,16);canvasText(c,'S',320,586,16);canvasText(c,'W',14,285,16);canvasText(c,'E',575,285,16);
  }
  const hero = new M.Hero(), hc = $('hero-canvas').getContext('2d'), backdrop = document.createElement('canvas'); $('hero-canvas').width=1200; $('hero-canvas').height=1200; hc.scale(2,2); backdrop.width=1200;backdrop.height=1200;const bgc=backdrop.getContext('2d');bgc.scale(2,2);heroBackground(bgc);
  function drawNetworkTransition(c,t) {
    const alpha = M.clamp((t-61)/2,0,1); c.save(); c.globalAlpha=alpha;
    canvasRect(c,0,0,600,600,'#eee8da');
    const positions=[130,300,470];
    positions.forEach(p=> {canvasRect(c,55,p-12,490,24,'#a4aaa0');canvasRect(c,p-12,55,24,490,'#a4aaa0');});
    for(let row=0;row<3;row++)for(let col=0;col<3;col++){
      const x=positions[col],y=positions[row];canvasRect(c,x-19,y-19,38,38,'#f9f4e9',8);
      const on = M.mod(t*1.4-row-col,4)<2;c.fillStyle=on?colors.green:colors.red;c.beginPath();c.arc(x,y,7,0,7);c.fill();
      canvasText(c,on?'NS':'EW',x,y+38,12,colors.ink,'center');
      if(col<2){const p=M.mod(t*.5+row,1);c.fillStyle=colors.gold;c.beginPath();c.arc(x+170*p,y-20,4,0,7);c.fill();}
    }
    canvasRect(c,89,242,422,110,'#fff9ee',14);canvasText(c,'ONE INTERSECTION',300,270,18,colors.ink,'center');canvasText(c,'↓   CORRIDOR   ↓',300,301,18,colors.green,'center');canvasText(c,'CITY NETWORK',300,332,21,colors.ink,'center');c.restore();
  }
  function renderHero() {
    const t=hero.time, signal=M.heroSignal(t), stage=Math.max(0,hero.stage), st=M.story[stage]; hc.drawImage(backdrop,0,0,600,600);
    if(stage===1||stage===2) {
      hc.save(); hc.lineWidth=9;hc.globalAlpha=.55;hc.strokeStyle=colors.amber;hc.beginPath();hc.moveTo(278,55);hc.lineTo(278,545);hc.stroke();hc.strokeStyle=stage===1?'#ffcfb2':'#b8e4bc';hc.beginPath();if(stage===1){hc.moveTo(55,322);hc.lineTo(545,322);}else{hc.moveTo(322,545);hc.lineTo(322,55);}hc.stroke();hc.restore();
      if(stage===1){hc.strokeStyle='#a13e30';hc.lineWidth=3;hc.beginPath();hc.arc(278,322,22,0,7);hc.stroke();canvasText(hc,'CONFLICT',300,410,15,colors.ink,'center');}
    }
    hero.lanes.forEach((lane,i)=>{
      const ang=[0,Math.PI,Math.PI/2,-Math.PI/2][i];
      lane.forEach(v=>{if(v.p < -20)return; const dx=-22,dy=v.p-300,x=300+dx*Math.cos(ang)-dy*Math.sin(ang),y=300+dx*Math.sin(ang)+dy*Math.cos(ang);car(hc,x,y,ang,palette[v.id%palette.length],v.id%17===0);});
      hc.save();hc.translate(300,300);hc.rotate(ang);hc.translate(-300,-300);canvasRect(hc,199,155,22,48,'#34453f',5);
      const state=(i<2?0:1)===signal.phase?signal.state:'ALL RED';
      [colors.red,colors.amber,colors.green].forEach((color,k)=>{hc.fillStyle=state===(['ALL RED','YELLOW','GREEN'][k])?color:'#59645b';hc.beginPath();hc.arc(210,164+k*14,5,0,7);hc.fill();});hc.restore();
    });
    const pedestrian = t>=53&&t<61;
    if(t<61){
      const progress=pedestrian?M.clamp((t-53)/7,0,1):0;person(hc,205+190*progress,216);person(hc,201+190*progress,208,colors.blue);
      if(t>=40){canvasRect(hc,386,167,124,28,'#fff7e6',5);canvasText(hc,t<53?'CALL WAITING':t<56?'WALK':'CLEARANCE',448,186,12,colors.ink,'center');}
    }
    canvasRect(hc,13,543,183,39,'#fff9ed',8);canvasText(hc,signal.state==='GREEN'?(signal.phase===0?'N/S THROUGH · GREEN':'E/W THROUGH · GREEN'):signal.state,24,568,12,colors.ink);
    if(t>=61)drawNetworkTransition(hc,t);
    setText('hero-time',`${t.toFixed(1)} s`);setText('hero-state',signal.state);$('hero-state').dataset.state=signal.state;
    setText('hero-phase',t>=53&&t<61?'Pedestrian service':signal.phase>=0?(signal.phase===0?'N/S through':'E/W through'):t>=61?'Network view':t<16?'Arrival demonstration':'Clearance');
    setText('hero-green',signal.state==='GREEN'?`${(t-signal.start).toFixed(1)} s`:'—');hero.queues.forEach((q,i)=>setText(`hero-q${i}`,`${q} vehicles`));
    setText('hero-ped',t<40?'None':t<53?'Waiting':t<56?'WALK':t<61?'Clearance':'Served');setText('hero-next',t<16?'N/S through':t<28?'E/W through':t<35?'E/W through':t<53?'Pedestrian service':'N/S through');
    setText('hero-stage',`${String(stage+1).padStart(2,'0')} / 09 · ${st[2]}`);setText('hero-heading',st[3]);setText('hero-caption',st[4]);$('hero-progress').style.width=`${t/68*100}%`;
    [...$('hero-strip').children].forEach((e,i)=>e.classList.toggle('active',i===(signal.state==='GREEN'?(signal.phase===1?3:0):signal.state==='YELLOW'?1:2)));
  }
  const heroJob=job('hero-demo',dt=>{hero.advance(dt*Number($('hero-speed').value));renderHero();});heroJob.autostart=true;
  $('hero-play').onclick=()=>toggle(heroJob,$('hero-play'));$('hero-restart').onclick=()=>{hero.reset();renderHero();};$('hero-step').onclick=()=>{heroJob.playing=false;hero.advance(1);renderHero();syncButtons();};$('hero-speed').oninput=()=>range('hero-speed','hero-speed-out','×');renderHero();
  // Conflict matrix: each cell is a real keyboard-focusable button.
  let selected=[0,2];
  $('conflict-matrix').innerHTML='<thead><tr><th scope="col">C</th>'+M.movements.map(x=>`<th scope="col">${x}</th>`).join('')+'</tr></thead><tbody>'+M.movements.map((name,i)=>`<tr><th scope="row">${name}</th>`+M.movements.map((other,j)=>`<td><button type="button" class="${M.conflicts[i][j]?'conflict':'compatible'}" data-i="${i}" data-j="${j}" aria-label="${name} and ${other}: ${M.conflicts[i][j]?'conflict':'compatible'}">${M.conflicts[i][j]}</button></td>`).join('')+'</tr>').join('')+'</tbody>';
  function renderConflict(i,j) {
    selected=[i,j];const conflict=M.conflicts[i][j];$('conflict-paths').innerHTML=svgRoad()+movementPath(M.movements[i], '#f2cf70')+movementPath(M.movements[j], '#b5d7f2',true);
    setText('conflict-detail',`${M.movements[i]} + ${M.movements[j]}: ${i===j?'the same movement (diagonal 0).':conflict?'C = 1. These movements are excluded from simultaneous protected service.':'C = 0. Compatible under the stated lane and turn assumptions.'} Solid gold = first path; dashed blue = second.`);
    $('conflict-paths').setAttribute('aria-label',`Selected paths ${M.movements[i]} and ${M.movements[j]}. ${conflict?'Conflict':'Compatible'}.`);
    document.querySelectorAll('#conflict-matrix button').forEach(b=>b.classList.toggle('selected',Number(b.dataset.i)===i&&Number(b.dataset.j)===j));
    const pos=M.movements.map((_,k)=>[260+135*Math.cos(k*Math.PI/4-Math.PI/2),165+126*Math.sin(k*Math.PI/4-Math.PI/2)]);let s='';
    M.movements.forEach((_,a)=>M.movements.forEach((__,b)=>{if(a<b&&M.conflicts[a][b])s+=line(...pos[a],...pos[b],(a===i&&b===j)||(b===i&&a===j)?colors.red:'#ccc4b7',(a===i&&b===j)||(b===i&&a===j)?4:1.2);}));
    pos.forEach(([x,y],k)=>{s+=`<circle cx="${x}" cy="${y}" r="28" fill="${selected.includes(k)?'#f1d995':'#fdfaf2'}" stroke="#a69b88"/>`+svgText(x,y+5,M.movements[k],window.innerWidth<600?22:13,colors.ink,'middle');});$('conflict-graph').innerHTML=s;
  }
  document.querySelectorAll('#conflict-matrix button').forEach(b=>{const pick=()=>renderConflict(Number(b.dataset.i),Number(b.dataset.j));b.onmouseenter=pick;b.onfocus=pick;b.onclick=pick;});renderConflict(0,2);
  const fsmStates=[['NS_GREEN','N/S has permission; E/W waits.','green'],['NS_YELLOW','N/S permission is ending; E/W still waits.','yellow'],['ALL_RED_1','Clear N/S traffic before admitting E/W.','clearance'],['EW_GREEN','E/W has permission; N/S waits.','green'],['EW_YELLOW','E/W permission is ending; N/S still waits.','yellow'],['ALL_RED_2','Clear E/W traffic, then return to NS_GREEN.','clearance']];let fsmIndex=0;
  $('fsm').innerHTML=fsmStates.map((s,i)=>`<button type="button" data-state="${i}">${s[0]} →<span>${s[2]}</span></button>`).join('');
  function renderFSM(){document.querySelectorAll('#fsm button').forEach((b,i)=>{b.classList.toggle('active',i===fsmIndex);b.setAttribute('aria-pressed',i===fsmIndex);});setText('fsm-detail',fsmStates[fsmIndex][1]);}
  document.querySelectorAll('#fsm button').forEach(b=>b.onclick=()=>{fsmIndex=Number(b.dataset.state);renderFSM();});$('fsm-next').onclick=()=>{fsmJob.playing=false;fsmIndex=(fsmIndex+1)%6;renderFSM();syncButtons();};let fsmClock=0;const fsmJob=job('fsm',dt=>{fsmClock+=dt;if(fsmClock>=3){fsmClock=0;fsmIndex=(fsmIndex+1)%6;renderFSM();}});fsmJob.autostart=true;$('fsm-play').onclick=()=>toggle(fsmJob,$('fsm-play'),'state machine');renderFSM();
  function clearance(){const speed=range('clear-speed','clear-speed-out',' km/h')/3.6,d=range('clear-distance','clear-distance-out',' m'),stop=speed+speed*speed/6;let s=rect(30,105,790,80,'#929b92')+line(630,105,630,185,'#fff9e9',6)+svgText(635,85,'STOP LINE',16)+line(640,110,780,110,colors.red,3,'stroke-dasharray="8 6"');const x=630-d*5; s+=rect(x-35,125,35,24,colors.blue)+rect(x-27,128,12,18,'#c8dedf',2)+line(x,205,630,205,colors.blue,2)+svgText((x+630)/2,230,`${d} m to line`,17,colors.blue,'middle');s+=line(630-stop*5,75,630,75,colors.gold,4)+svgText(60,37,`Illustrative stopping distance: ${stop.toFixed(0)} m`,19)+svgText(665,152,'conflict area',14,'#fff9e9');$('clearance-svg').innerHTML=s;setText('clear-result',`${d>=stop?'Enough modeled distance to stop comfortably.':'Inside the modeled comfortable stopping distance.'} At ${Math.round(speed*3.6)} km/h: d ≈ ${stop.toFixed(1)} m; illustrative Y ≈ ${(1+speed/6).toFixed(1)} s. This comparison alone does not determine a dilemma zone or a field timing.`);}
  bind(['clear-speed','clear-distance'],clearance);
  function plotSeries(id, series, labels, maxX, caption='queue (vehicles)', width=850,height=260) {
    const left=60,right=width-24,top=38,bottom=height-44,maxY=Math.max(10,...series.flat().map(p=>p[1]))*1.1;
    let s='';for(let i=0;i<5;i++){const y=bottom-(bottom-top)*i/4;s+=line(left,y,right,y,'#ded7c9',1)+svgText(left-9,y+5,Math.round(maxY*i/4),12,colors.muted,'end');}
    s+=svgText(left,20,caption,14)+svgText(right,bottom+32,`${maxX} s`,13,colors.muted,'end')+svgText(left,bottom+32,'0',13);
    series.forEach((data,i)=>{s+=`<polyline points="${data.map(([x,y])=>`${left+x/maxX*(right-left)},${bottom-y/maxY*(bottom-top)}`).join(' ')}" fill="none" stroke="${[colors.blue,colors.red,colors.green][i]}" stroke-width="3" ${i===1?'stroke-dasharray="7 4"':''}/>`+svgText(right-165,22+i*20,labels[i],13,[colors.blue,colors.red,colors.green][i]);});$(id).innerHTML=s;
  }
  function fixed(){const g=range('split','split-out',' s'),data=M.fixedQueues(g);$('fixed-timeline').innerHTML=[[g,'N/S','green'],[4,'Y','yellow'],[2,'R','red'],[56-g,'E/W','green'],[4,'Y','yellow'],[2,'R','red']].map(([n,l,c])=>`<div class="${c}" style="flex:${n}" title="${l}: ${n} seconds">${l}${n>10?' '+n+' s':''}</div>`).join('');plotSeries('fixed-plot',[data.map(p=>[p[0],p[1]]),data.map(p=>[p[0],p[2]])],['N/S · solid','E/W · dashed'],272);const end=data[data.length-1];setText('fixed-result',`After four cycles: N/S queue ≈ ${end[1].toFixed(1)}, E/W queue ≈ ${end[2].toFixed(1)} vehicles. Cycle stays 68 s; E/W green is ${56-g} s. Queue growth changes because the service allocation changed.`);}
  bind(['split'],fixed);
  function flow(){const arrival=range('flow-arrival','flow-arrival-out',' veh/h'),g=range('flow-green','flow-green-out',' s'),minute=range('flow-time','flow-time-out',' min'),mu=1800*g/60,q=Math.max(0,8+(arrival-mu)*minute/60);let s='';const max=Math.max(arrival,mu)*10/60+8;const px=x=>60+x/10*500,py=y=>225-y/max*180;s+=line(60,225,560,225,'#b8b09f')+line(60,45,60,225,'#b8b09f')+line(60,py(8),560,py(8+arrival*10/60),colors.blue,3)+line(60,225,560,py(mu*10/60),colors.green,3,'stroke-dasharray="8 5"')+line(px(minute),40,px(minute),225,colors.gold,2)+svgText(65,25,'Cumulative vehicles / potential service',16)+svgText(60,254,'0',13)+svgText(560,254,'10 min',13,colors.ink,'end');s+=svgText(610,50,'QUEUE NOW',16)+svgText(610,85,`${Math.round(q)} vehicles`,22)+svgText(610,225,'λ · blue solid',14,colors.blue)+svgText(610,249,'μ · green dashed',14,colors.green);for(let i=0;i<Math.min(36,Math.round(q));i++)s+=rect(610+i%9*22,110+Math.floor(i/9)*23,16,13,palette[i%6],3);if(q>36)s+=svgText(610,210,`+ ${Math.round(q)-36} more`,13);$('flow-plot').innerHTML=s;setText('flow-result',`μ ≈ ${mu} veh/h. λ ${arrival<mu?'<':arrival>mu?'>':'='} μ: ${arrival<mu?'spare average capacity can drain the initial queue':arrival>mu?'demand exceeds service; the queue grows':'no spare average capacity; the initial queue persists'}.`);}
  let flowTime=Number($('flow-time').value); const flowJob=job('flow-plot',dt=>{flowTime=M.mod(flowTime+dt*.7,10);$('flow-time').value=flowTime.toFixed(1);flow();},false); $('flow-time').addEventListener('input',()=>{flowTime=Number($('flow-time').value);});$('flow-run').onclick=()=>toggle(flowJob,$('flow-run'),'queue');bind(['flow-arrival','flow-green','flow-time'],flow);
  function queue(){const vals=['queue-q','queue-a','queue-d'].map(id=>M.clamp(Math.round(Number($(id).value)||0),0,50)),[q,a,d]=vals,left=Math.max(0,q+a-d);setText('queue-equation',`max(0, ${q} + ${a} − ${d}) = ${left} vehicles; actual departures = ${Math.min(q+a,d)}.`);let s=svgText(22,35,`${left} remain`,20);for(let i=0;i<Math.min(40,left);i++)s+=rect(24+(i%20)*39,58+Math.floor(i/20)*34,28,19,palette[i%6],4);if(left>40)s+=svgText(810,125,`+${left-40}`,14,colors.ink,'end');$('queue-svg').innerHTML=s;}
  bind(['queue-q','queue-a','queue-d'],queue);
  let detectorTime=0;function detector(){const x=80+detectorTime*100,hit=x>=365&&x<=485;let s=rect(20,50,810,75,'#929b92')+rect(385,59,80,57,hit?'#e7c86b':'#a9bfa6',0,'stroke="#335e46" stroke-width="3"')+line(425,126,425,153,colors.gold,3)+rect(390,147,170,27,'#e9ddbe')+svgText(475,166,hit?'DETECTOR ACTIVE':'DETECTOR ZONE',14,colors.ink,'middle')+rect(x-20,72,40,24,colors.blue)+rect(x-8,75,12,18,'#d4e4e4',2)+svgText(25,29,'A vehicle creates an observation, then a controller call.',17);$('detector-svg').innerHTML=s;setText('detector-result',hit?'Vehicle detected → call registered':detectorTime>4.05?'Call retained for service':'No vehicle in detector');}
  const detectorJob=job('detector-svg',dt=>{detectorTime=Math.min(7,detectorTime+dt);detector();if(detectorTime>=7)detectorJob.playing=false;},false);$('detector-run').onclick=()=>{if(reduced.matches){detectorTime=detectorTime<3.5?3.5:7;detector();}else{detectorTime=0;resume(detectorJob);}};detector();
  const comparisons=[M.comparison('fixed'),M.comparison('actuated')];
  function actuated(){const t=range('actuated-time','actuated-time-out',' s');plotSeries('actuated-plot',comparisons.map(a=>a.slice(0,t+1).map(p=>[p.t,p.q[0]+p.q[1]])),['Fixed · solid','Actuated · dashed'],180,'total queued (vehicles)',850,330);['fixed-comparison','actuated-comparison'].forEach((id,i)=>{const p=comparisons[i][t];setText(id,`${i?'Actuated':'Fixed-time'}: ${p.q[0]+p.q[1]} queued; ${p.served} departed. ${p.phase===0?'N/S':'E/W'} ${p.state.toLowerCase()}. Accumulated queue area: ${p.wait} veh·s.`);});}
  let compareFraction=0;const compareJob=job('actuated-plot',dt=>{compareFraction+=dt*8;if(compareFraction>=1){const step=Math.floor(compareFraction);compareFraction-=step;$('actuated-time').value=(Number($('actuated-time').value)+step)%181;actuated();}},false);$('actuated-run').onclick=()=>toggle(compareJob,$('actuated-run'),'comparison');bind(['actuated-time'],actuated);
  function scores(){const q=range('weight-q','weight-q-out'),w=range('weight-w','weight-w-out'),p=range('weight-p','weight-p-out'),a=14*q+2*w,b=6*q+9*w+p,fair=$('fairness').checked;setText('score-ns',a.toFixed(1));setText('score-ew',b.toFixed(1));setText('score-result',`${fair||b>a?'E/W + compatible crossing':a===b?'Tie → N/S by the stated tie-break':'N/S through'} is the next candidate. ${fair?'The 90 s wait exceeds the 80 s guard, overriding the score.':'Chosen by score; ties prefer N/S.'} Safe transitions still apply.`);}
  bind(['weight-q','weight-w','weight-p','fairness'],scores);
  function pressure(){const up=range('pressure-up','pressure-up-out',' vehicles'),down=range('pressure-down','pressure-down-out',' vehicles');let s='';[[up,down,'A'],[10,9,'B']].forEach(([a,b,name],row)=>{const y=45+row*133;s+=svgText(25,y+18,`Route ${name}`,18)+rect(160,y,290,65,'#e4dfd2')+rect(510,y,290,65,'#e4dfd2')+line(463,y+32,498,y+32,colors.ink,4)+svgText(475,y+24,'→',25,colors.ink,'middle');[a,b].forEach((count,k)=>{for(let i=0;i<count;i++)s+=rect((k?520:170)+(i%10)*27,y+12+Math.floor(i/10)*25,20,15,k?colors.red:colors.blue,3);});s+=svgText(160,y+93,`${a} upstream − ${b} downstream = ${a-b}`,18);});$('pressure-svg').innerHTML=s;setText('pressure-result',down===20?'Route A has no receiving storage: hold it, regardless of raw pressure. Route B remains feasible.':`${up-down>1?'Route A':up-down<1?'Route B':'Tie'} has the greater simplified pressure (${up-down} versus 1). A release also requires upstream demand and safe permission.`);}
  bind(['pressure-up','pressure-down'],pressure);
  function people(index){document.querySelectorAll('#phase-plan button').forEach((b,i)=>b.setAttribute('aria-pressed',i===index));let s=svgRoad();const origins=index<2?['N','S']:['E','W'];origins.forEach(o=>{s+=movementPath(`${o}_${index%2?'T':'L'}`,'#f1d383');});if(index%2)s+=pedestrianPaths(index===1?'NS':'EW');$('people-svg').innerHTML=s;setText('people-title',[ 'Protected N/S left turns','N/S through + parallel crossings','Protected E/W left turns','E/W through + parallel crossings'][index]);setText('people-detail',index%2?'Dashed blue paths cross the side streets parallel to the active through traffic. Turning vehicles across those crossings are held in this simplified plan. WALK and clearance must fit the available service.':'Opposing protected lefts use dedicated turn paths. Through traffic and pedestrians wait. The geometry is assumed to support concurrent opposing turns.');}
  document.querySelectorAll('#phase-plan button').forEach(b=>b.onclick=()=>people(Number(b.dataset.phase)));people(0);
  let twoTime=0;
  function two(){let s='';[36,0].forEach((offset,row)=>{const y=82+row*164,green=M.mod(twoTime-offset,80)<22,greenA=twoTime<22;s+=svgText(25,y-39,row?'Poor offset · 0 s':'Progression offset · 36 s',18)+rect(35,y,830,56,'#939d94',0)+line(130,y,130,y+56,'#fff7df',4)+line(710,y,710,y+56,'#fff7df',4);s+=rect(107,y-28,70,23,greenA?'#d3e6d4':'#ebc9bf')+svgText(142,y-11,`A · ${greenA?'GO':'STOP'}`,12,colors.ink,'middle')+rect(680,y-28,85,23,green?'#d3e6d4':'#ebc9bf')+svgText(722,y-11,`B · ${green?'GO':'STOP'}`,12,colors.ink,'middle');for(let i=0;i<8;i++){const distance=(twoTime-i*1.25)*400/36;let x=130+distance/400*580;if(row===1)x=Math.min(x,697-i*20);if(x>39&&x<850)s+=rect(x-7,y+18,15,21,palette[i%6],3);}s+=svgText(410,y+82,'400 m · 40 km/h · about 36 s',15,colors.muted,'middle');});$('two-svg').innerHTML=s;$('two-time').value=twoTime;setText('two-time-out',`${twoTime.toFixed(1)} s`);}
  const twoJob=job('two-svg',dt=>{twoTime=M.mod(twoTime+dt*2,65);two();});twoJob.autostart=true;$('two-run').onclick=()=>toggle(twoJob,$('two-run'),'platoons');$('two-reset').onclick=()=>{twoTime=0;two();};$('two-time').oninput=()=>{twoTime=Number($('two-time').value);twoJob.playing=false;two();syncButtons();};two();
  let corridor=new M.Corridor();const wc=$('wave-canvas').getContext('2d');
  function drawWave(){if(window.innerWidth<600){drawWaveMobile();return;}canvasHeight('wave-canvas',530);const c=wc,t=corridor.time;canvasRect(c,0,0,900,530,'#f4efe4');canvasText(c,'FIVE SIGNALS · EASTBOUND →',28,29,15);canvasRect(c,65,74,765,38,'#90998f');
    for(let i=0;i<5;i++){const x=78+i*182;canvasRect(c,x-12,49,24,87,'#929a91');canvasRect(c,x-16,86,32,3,'#fff9e9');const green=corridor.green(i);canvasRect(c,x-29,44,58,21,green?'#d3e8d4':'#ebc7bc',5);canvasText(c,green?'GO':'STOP',x,59,11,colors.ink,'center');canvasText(c,`${i+1}`,x,150,14,colors.ink,'center');}
    corridor.cars.forEach((v,i)=>{const x=78+v.x/400*182;if(x>28&&x<867)canvasRect(c,x,96,3.4,8,palette[i%6],1);});
    const left=85,right=860,top=200,bottom=465,horizon=240;const x=time=>left+time/horizon*(right-left),y=i=>bottom-i*(bottom-top)/4;
    for(let sec=0;sec<=240;sec+=40){c.strokeStyle='#d9d3c6';c.lineWidth=1;c.beginPath();c.moveTo(x(sec),top-10);c.lineTo(x(sec),bottom+5);c.stroke();canvasText(c,String(sec),x(sec),490,12,colors.muted,'center');}
    for(let i=0;i<5;i++){canvasRect(c,left,y(i)-8,right-left,16,'#e7c8bd',3);for(let start=-corridor.cycle*4;start<240;start+=corridor.cycle){const a=Math.max(0,start+i*corridor.offset),b=Math.min(240,start+i*corridor.offset+28);if(b>a)canvasRect(c,x(a),y(i)-8,x(b)-x(a),16,'#90b797',1);}canvasText(c,`${i*400} m`,left-12,y(i)+4,12,colors.ink,'right');}
    const travel=400/corridor.speed,band=corridor.bandwidth();
    if(band.width>.1){c.fillStyle='#e8bc5550';c.beginPath();c.moveTo(x(band.start),y(0));c.lineTo(x(band.start+band.width),y(0));c.lineTo(x(band.start+band.width+4*travel),y(4));c.lineTo(x(band.start+4*travel),y(4));c.closePath();c.fill();}
    [1,5,9,13,17,21].forEach(depart=>{c.strokeStyle=colors.blue;c.lineWidth=1.4;c.beginPath();c.moveTo(x(depart),y(0));c.lineTo(x(depart+4*travel),y(4));c.stroke();});
    c.strokeStyle=colors.gold;c.lineWidth=2;c.beginPath();c.moveTo(x(t),top-12);c.lineTo(x(t),bottom+10);c.stroke();canvasText(c,'Distance along corridor ↑',25,180,14);canvasText(c,'Time (seconds) →',right,516,14,colors.ink,'right');
    setText('wave-result',`Travel time per link ≈ ${travel.toFixed(0)} s. Uninterrupted departure bandwidth ≈ ${band.width.toFixed(0)} s at this speed and offset. ${band.width<1?'No continuous departure window survives all five signals.':'Gold shows the longest feasible departure window in the first green.'}`);$('wave-time').value=t;setText('wave-time-out',`${t.toFixed(0)} s`);
  }
  function drawWaveMobile(){
    const c=wc,t=corridor.time;canvasHeight('wave-canvas',950);
    canvasRect(c,0,0,900,950,'#f4efe4');canvasText(c,'FIVE SIGNALS · EASTBOUND →',35,52,31);
    canvasRect(c,70,115,750,64,'#90998f');
    for(let i=0;i<5;i++){const x=105+i*165;canvasRect(c,x-18,93,36,110,'#929a91');canvasRect(c,x-40,90,80,34,corridor.green(i)?'#d3e8d4':'#ebc7bc',6);canvasText(c,corridor.green(i)?'GO':'STOP',x,116,26,colors.ink,'center');canvasText(c,String(i+1),x,240,30,colors.ink,'center');}
    corridor.cars.forEach((v,i)=>{const x=105+v.x/400*165;if(x>35&&x<855)canvasRect(c,x,158,3.2,16,palette[i%6],1);});
    const left=155,right=845,top=360,bottom=805,xx=t=>left+t/240*(right-left),yy=i=>bottom-i*(bottom-top)/4;
    canvasText(c,'Distance (m) ↑',35,304,32);
    for(let i=0;i<5;i++){canvasRect(c,left,yy(i)-14,right-left,28,'#e7c8bd',3);for(let start=-corridor.cycle*4;start<240;start+=corridor.cycle){const a=Math.max(0,start+i*corridor.offset),b=Math.min(240,start+i*corridor.offset+28);if(b>a)canvasRect(c,xx(a),yy(i)-14,xx(b)-xx(a),28,'#90b797',1);}canvasText(c,String(i*400),left-18,yy(i)+10,29,colors.ink,'right');}
    const travel=400/corridor.speed,band=corridor.bandwidth();if(band.width>.1){c.fillStyle='#e8bc5550';c.beginPath();c.moveTo(xx(band.start),yy(0));c.lineTo(xx(band.start+band.width),yy(0));c.lineTo(xx(band.start+band.width+4*travel),yy(4));c.lineTo(xx(band.start+4*travel),yy(4));c.closePath();c.fill();}
    [1,7,13,19].forEach(d=>{c.strokeStyle=colors.blue;c.lineWidth=3;c.beginPath();c.moveTo(xx(d),yy(0));c.lineTo(xx(d+4*travel),yy(4));c.stroke();});
    [0,80,160,240].forEach(sec=>{canvasText(c,String(sec),xx(sec),864,29,colors.ink,'center');});canvasText(c,'Time (seconds) →',right,916,32,colors.ink,'right');
    c.strokeStyle=colors.gold;c.lineWidth=3;c.beginPath();c.moveTo(xx(t),top-22);c.lineTo(xx(t),bottom+20);c.stroke();
    setText('wave-result',`Travel time per link ≈ ${travel.toFixed(0)} s. Uninterrupted departure bandwidth ≈ ${band.width.toFixed(0)} s. ${band.width<1?'No uninterrupted window at this speed and offset.':'Gold marks the longest feasible departure window.'}`);$('wave-time').value=t;setText('wave-time-out',`${t.toFixed(0)} s`);
  }
  function configureWave(){corridor.configure(range('wave-cycle','wave-cycle-out',' s'),range('wave-offset','wave-offset-out',' s'),range('wave-speed','wave-speed-out',' km/h'));drawWave();}
  const waveJob=job('wave-canvas',dt=>{let remaining=dt*3;while(remaining>0){const step=Math.min(remaining,1/30);corridor.advance(step);remaining-=step;}if(corridor.time>240)corridor.reset();drawWave();});waveJob.autostart=true;
  bind(['wave-cycle','wave-offset','wave-speed'],configureWave);$('wave-play').onclick=()=>toggle(waveJob,$('wave-play'),'corridor');$('wave-reset').onclick=()=>{corridor.reset();drawWave();};$('wave-time').oninput=()=>{waveJob.playing=false;corridor.seek(Number($('wave-time').value));drawWave();syncButtons();};
  $('tradeoff-svg').innerHTML=rect(30,85,790,76,'#969e94')+line(80,111,770,111,colors.blue,4)+line(770,139,80,139,colors.red,4)+svgText(760,116,'→',29,colors.blue)+svgText(62,145,'←',29,colors.red)+rect(181,65,6,116,'#fff7e9')+rect(641,65,6,116,'#fff7e9')+svgText(185,43,'A',23)+svgText(645,43,'B',23)+svgText(310,55,'Eastbound asks for +36 s',19,colors.blue)+svgText(310,204,'Westbound asks for −36 s ≡ +44 s',19,colors.red)+rect(255,99,30,16,colors.blue)+rect(302,99,30,16,colors.blue)+rect(530,134,30,16,colors.red)+rect(580,134,30,16,colors.red);
  let networkIndex=0;
  function network(){let s='';for(let r=0;r<4;r++)for(let col=0;col<4;col++){const x=95+col*137,y=70+r*108;if(col<3)s+=line(x,y,x+137,y,'#adb3a7',13);if(r<3)s+=line(x,y,x,y+108,'#adb3a7',13);}
    const row=Math.floor(networkIndex/3),col=networkIndex%3,x=95+col*137,y=70+row*108;s+=line(x,y,x+137,y,colors.gold,13)+svgText(x+68,y-20,'xₑ(t)',18,colors.ink,'middle');for(let r=0;r<4;r++)for(let col=0;col<4;col++){const xx=95+col*137,yy=70+r*108;s+=rect(xx-19,yy-19,38,38,'#fff9ec',8)+svgText(xx,yy+5,`${r*4+col+1}`,14,colors.ink,'middle');}s+=svgText(x,y+40,'uᵥ(t)',15,colors.blue,'middle');$('network-svg').innerHTML=s;setText('network-detail',`Link e connects intersection ${row*4+col+1} to ${row*4+col+2}. Its vehicle state xₑ is influenced by the upstream decision uᵥ and by available receiving space.`);}
  $('network-select').onclick=()=>{networkIndex=(networkIndex+1)%12;network();};network();
  function architecture(mode){document.querySelectorAll('#architecture-controls button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.mode===mode));let s='';const points=Array.from({length:6},(_,i)=>[100+i*130,235]);
    if(mode==='central'){s+=rect(315,30,220,60,'#e3ddcc',10)+svgText(425,67,'Management system',18,colors.ink,'middle');points.forEach(([x,y])=>{s+=line(425,90,x,y-20,colors.blue,2);});}
    if(mode==='local'){for(let i=0;i<5;i++)s+=line(points[i][0]+24,235,points[i+1][0]-24,235,colors.blue,3);s+=svgText(425,75,'Local observations + neighbor messages',20,colors.ink,'middle');}
    if(mode==='hybrid'){[230,620].forEach((x,g)=>{s+=rect(x-90,40,180,55,'#e3ddcc',10)+svgText(x,72,`Region ${g+1}`,18,colors.ink,'middle');points.slice(g*3,g*3+3).forEach(([px,py])=>{s+=line(x,95,px,py-20,colors.blue,2);});});s+=line(320,66,530,66,colors.gold,3,'stroke-dasharray="7 5"');}
    points.forEach(([x,y],i)=>{s+=rect(x-25,y-25,50,50,'#d7e4d4',10)+svgText(x,y+6,`${i+1}`,19,colors.ink,'middle');});$('architecture-svg').innerHTML=s;setText('architecture-detail',{central:'Centralized: observations travel to a shared decision layer; coordinated settings return to local controllers. Local safety logic still governs indications.',local:'Distributed: each controller uses local state and selected neighbor information. No all-to-all communication is implied; local coordination can miss distant effects.',hybrid:'Hybrid: regions coordinate corridors while each intersection handles immediate calls. Hierarchy reduces the size of each decision problem but requires consistent interfaces.'}[mode]);}
  document.querySelectorAll('#architecture-controls button').forEach(b=>b.onclick=()=>architecture(b.dataset.mode));architecture('central');
  let mpc={q:[15,7],phase:0,time:0};
  function renderMPC(){const plan=M.mpcPlan(mpc.q,mpc.phase,mpc.time);$('mpc-horizon').innerHTML=plan.steps.map((s,k)=>`<div><strong>t + ${k}</strong>${s.phase===0?'N/S':'E/W'}<br>${s.change?'clear → go':'hold green'}<br>Q: ${s.q.join(' / ')}</div>`).join('');setText('mpc-result',`Observed queues: ${mpc.q.join(' / ')}. Step ${mpc.time}. Best predicted sum of end-of-block queue totals: ${plan.cost} vehicle-blocks. Only the first block is executed.`);}
  $('mpc-next').onclick=()=>{const first=M.mpcPlan(mpc.q,mpc.phase,mpc.time).steps[0];mpc={q:first.q,phase:first.phase,time:mpc.time+1};renderMPC();};$('mpc-reset').onclick=()=>{mpc={q:[15,7],phase:0,time:0};renderMPC();};renderMPC();
  let spill={gated:false,down:8,blocked:0,up:5,step:0};
  function spillback(){let s=rect(20,123,810,84,'#8f998f')+rect(345,20,95,292,'#8f998f')+rect(20,123,810,84,'#8f998f')+line(315,123,315,207,'#fff7e9',4)+svgText(50,50,'UPSTREAM',17)+svgText(500,50,'RECEIVING LINK · 8 spaces',17);
    for(let i=0;i<8;i++)s+=rect(495+i*38,145,30,30,i<spill.down?colors.red:'#cdd0bd',4);
    for(let i=0;i<Math.min(7,spill.up);i++)s+=rect(275-i*37,151,28,20,colors.blue,4);
    for(let i=0;i<Math.min(3,spill.blocked);i++)s+=rect(354+i*30,151,27,24,colors.gold,4);
    s+=svgText(50,246,`${spill.up} waiting upstream`,17)+svgText(491,246,`${spill.down}/8 stored; ${spill.blocked} in conflict area`,17)+svgText(390,91,spill.blocked?'BLOCKED':'CROSSING OPEN',15,spill.blocked?colors.red:colors.green,'middle')+rect(376,267,24,32,spill.blocked?colors.red:colors.green,3);$('spillback-svg').innerHTML=s;
    $('spillback-toggle').setAttribute('aria-pressed',spill.gated);setText('spillback-toggle',spill.gated?'Disable downstream gating':'Enable downstream gating');setText('spillback-result',spill.blocked?`Spillback blocks perpendicular traffic. ${spill.gated?'Gating prevents new entries; existing blockage must still drain.':'More green has admitted vehicles with nowhere to go.'}`:spill.gated?'Gating holds vehicles upstream when storage is full. The crossing stays available to the other direction.':'Receiving road starts full. Advance arrivals to see excess entries obstruct the crossing.');}
  $('spillback-next').onclick=()=>{spill.step++;if(spill.step%2===0&&spill.down>0){spill.down--;if(spill.blocked>0){spill.blocked--;spill.down++;}}spill.up+=2;for(let i=0;i<2;i++){if(spill.up===0)break;if(spill.down<8){spill.down++;spill.up--;}else if(!spill.gated&&spill.blocked<3){spill.blocked++;spill.up--;}}spillback();};$('spillback-toggle').onclick=()=>{spill.gated=!spill.gated;spillback();};$('spillback-reset').onclick=()=>{spill={gated:spill.gated,down:8,blocked:0,up:5,step:0};spillback();};spillback();
  let city=new M.City(),cityFraction=0;const lc=$('lab-canvas').getContext('2d');
  function drawCity(){const c=lc,narrow=window.innerWidth<600;canvasHeight('lab-canvas',narrow?670:580);canvasRect(c,0,0,900,narrow?670:580,'#eee9dc');
    for(let r=0;r<3;r++)for(let col=0;col<3;col++){const x=210+col*250,y=130+r*180;canvasRect(c,x-112,y-98,77,60,(r+col)%2?'#e2cbb8':'#dcd6c0',8);c.fillStyle='#9aab88';c.beginPath();c.arc(x+53,y-65,13,0,7);c.fill();}
    for(let k=0;k<3;k++){canvasRect(c,35,110+k*180,820,40,'#929b91');canvasRect(c,190+k*250,30,40,520,'#929b91');}
    city.nodes.forEach((n,i)=>{
      const row=Math.floor(i/3),col=i%3,x=210+col*250,y=130+row*180;
      canvasRect(c,x-21,y-21,42,42,'#e5dfca',6);
      const fill=n.state==='GREEN'?colors.green:n.state==='YELLOW'?colors.gold:colors.red;
      canvasRect(c,x-(narrow?31:20),y-(narrow?26:16),narrow?62:40,narrow?52:32,fill,5);canvasText(c,n.state==='GREEN'?(n.phase===0?'↓':'→'):n.state==='YELLOW'?'Y':'R',x,y+(narrow?11:7),narrow?34:21,'#fffdf1','center');
      n.q.forEach((q,d)=>{for(let j=0;j<Math.min(q.length,10);j++){const px=d===0?x-7:x-36-j*11,py=d===0?y-36-j*9:y+5;canvasRect(c,px,py,d===0?7:8,d===0?7:7,palette[j%6],1);}canvasText(c,String(q.length),d===0?x-32:x-78,d===0?y-49:y+35,narrow?30:14,colors.ink,'center');});
      if(!narrow)canvasText(c,`${row+1}.${col+1}`,x+33,y-25,12,colors.muted);
    });
    city.transit.forEach(v=>{const from=[210+v.from%3*250,130+Math.floor(v.from/3)*180],to=[210+v.to%3*250,130+Math.floor(v.to/3)*180],f=M.clamp((city.time-(v.at-4))/4,0,1);const x=from[0]+(to[0]-from[0])*f,y=from[1]+(to[1]-from[1])*f;canvasRect(c,x+(v.d===0?-7:0),y+(v.d===1?5:0),6,6,colors.blue,1);});
    canvasText(c,'SOUTHBOUND ↓   ·   EASTBOUND →',35,narrow?35:24,narrow?29:15);if(narrow){canvasText(c,'Arrows: GO · Y: yellow · R: all-red',35,596,29);canvasText(c,'Numbers = queued vehicles',35,643,29);}else canvasText(c,'GO: arrows     Y: yellow     R: all-red     numbers: queued vehicles',35,574,13);
    const m=city.metrics;setText('lab-time',`${city.time} s`);setText('lab-queue',m.average.toFixed(1));setText('lab-wait',m.wait===null?'—':`${Math.round(m.wait)} s`);setText('lab-stops',m.stops);setText('lab-throughput',m.throughput);setText('lab-result',`${m.queued} queued inside · ${m.transit} traveling between nodes · ${m.backlog} waiting outside · ${m.generated} generated. ${m.backlog>0?'External demand is backing up: internal metrics alone understate congestion.':'Every generated vehicle is accounted for in queues, travel, external backlog or exits.'}`);
  }
  function configureCity(){const demand=range('lab-demand','lab-demand-out','×'),balance=range('lab-balance','lab-balance-out','%')/100,cycle=range('lab-cycle','lab-cycle-out',' s'),offset=range('lab-offset','lab-offset-out',' s'),strategy=$('lab-strategy').value;range('lab-speed','lab-speed-out','×');$('lab-cycle').disabled=strategy!=='fixed';$('lab-offset').disabled=strategy!=='fixed';city=new M.City({demand,balance,cycle,offset,strategy});cityFraction=0;drawCity();}
  const cityJob=job('lab-canvas',dt=>{cityFraction+=dt*Number($('lab-speed').value);if(cityFraction>=1){const n=Math.floor(cityFraction);cityFraction-=n;city.step(n);drawCity();}});cityJob.autostart=true;
  bind(['lab-demand','lab-balance','lab-cycle','lab-offset','lab-strategy'],configureCity);$('lab-speed').oninput=()=>range('lab-speed','lab-speed-out','×');$('lab-play').onclick=()=>toggle(cityJob,$('lab-play'),'city');$('lab-reset').onclick=configureCity;$('lab-step').onclick=()=>{cityJob.playing=false;city.step(10);drawCity();syncButtons();};
  let summaryIndex=0;const summaryText=['The indication tells a road user whether and how to proceed.','A movement is the actual path a user takes through shared space.','A feasible phase groups movements that can receive permission together.','The controller observes calls and queues, times intervals, and chooses legal transitions.','A corridor relates timing references to platoon travel between intersections.','Network optimization considers how local discharge changes queues and available storage elsewhere.'];
  function summary(){[...$('summary-ladder').children].forEach((e,i)=>e.classList.toggle('active',i===summaryIndex));setText('summary-detail',summaryText[summaryIndex]);}$('summary-next').onclick=()=>{summaryIndex=(summaryIndex+1)%6;summary();};summary();
  function syncButtons(){[['hero-play',heroJob,''],['fsm-play',fsmJob,'state machine'],['two-run',twoJob,'platoons'],['wave-play',waveJob,'corridor'],['lab-play',cityJob,'city'],['flow-run',flowJob,'queue'],['actuated-run',compareJob,'comparison']].forEach(([id,item,noun])=>setText(id,`${item.playing&&!globalPaused?'Pause':'Play'}${noun?' '+noun:''}`));}
  syncButtons();
  window.addEventListener('resize',()=>{drawWave();drawCity();renderConflict(...selected);});
  let previous=0;
  function frame(now){const dt=previous?Math.min(.1,(now-previous)/1000):0;previous=now;if(!document.hidden&&!globalPaused)jobs.forEach(j=>{if(j.visible&&j.playing)j.update(dt);});window.requestAnimationFrame(frame);}
  document.addEventListener('visibilitychange',()=>{previous=0;});window.requestAnimationFrame(frame);
  // Small read-only inspection surface for reproducibility and browser QA.
  window.TrafficPage={get hero(){return hero;},get city(){return city;},get corridor(){return corridor;},renderHero,drawCity,drawWave,get jobs(){return jobs;}};
})();
