/* Original university film: one deterministic clock, articulated SVG cast, seven scenes. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id), svg = $('intro-svg');
  if (!svg) return;
  const NS = 'http://www.w3.org/2000/svg', M = window.QuantizationMath;
  const asset = '2026-08-21-quantization/university-ensemble.svg#';
  const C = {ink:'#304d56', muted:'#687f83', blue:'#456e9b', green:'#527d61', red:'#b84545', paleBlue:'#eaf3fa', mint:'#e2f1e3', peach:'#f6dfd1', lavender:'#e7e0ef', wall:'#fffdf8', line:'#b8cbc4'};
  const motion = matchMedia('(prefers-reduced-motion: reduce)'), compact = matchMedia('(max-width:600px)');
  const starts = [0,6,15,24,33,42,52], ends = [6,15,24,33,42,52,60];
  const measurements = [.7,1.2,1.8,2.4,2.9,3.3], weights = [.73,-1.24,.18,2.61];
  const example = M.affine(.73, -1.24, 2.61, 4, 'asymmetric');
  // Captions follow the visual reveals, rather than describing a later frame.
  const cues = [
    [0,'A question on campus','Can many possible numbers be represented using only a few choices? Two students take their measurements to a university class.'],
    [6,'Six measurements enter the classroom','The professor asks for whole numbers only. Watch six distinct inputs map to their nearest integers.'],
    [10.6,'Different inputs can share a result','Distinct inputs share the results 1, 2 and 3. Quantization keeps a representative instead of the exact input.'],
    [15,'A closer look at 2.4','Blue marks input 2.4; green marks its reconstruction, 2. The gap makes the approximation visible.'],
    [18,'Error is a numerical difference','The red gap is the signed error: e = x − Q(x) = 2.4 − 2 = 0.4.'],
    [21,'The step sets the spacing','A quantization step is the spacing between adjacent levels. Here, Δ = 1.'],
    [24,'A study group changes the step','Both grids round the same input, 2.4, to the nearest level. Only their spacing changes.'],
    [28,'Finer steps, a closer answer here','Steps of 1 and 0.25 give 2 and 2.5. For this input, absolute error falls from 0.4 to 0.1.'],
    [33,'The numerical idea reaches the computer lab','Networks contain learned parameters called weights. These four labels belong to a teaching schematic, not an LLM architecture.'],
    [36.8,'Store a code; reconstruct an approximation','This 4-bit affine scheme maps 0.73 to code 1000, then reconstructs 0.77. The signed error is −0.04.'],
    [40,'The scheme matters','Decoding uses a scale and zero-point. Four bits identify 16 possible codes.'],
    [42,'The same N weights, two storage budgets','For N weights, 16-bit codes need 16N raw bits; 4-bit codes need 4N. Each tile represents N bits.'],
    [46,'One quarter of the raw weight bits','Same weights, one quarter of the raw code storage: 4N / 16N = ¼.'],
    [48,'Raw code size is only part of the picture','Scales, codebooks, activations and caches add overhead. Raw-bit savings do not imply proportionate runtime-memory savings or inference speedups.'],
    [52,'One idea connects both settings','Measurements map to discrete levels; weights map to compact codes. Both reconstruct approximations.'],
    [55,'Choose the tradeoff deliberately','Quantization uses carefully chosen approximations to reduce representation costs. Model-quality effects depend on the method, model and task.'],
    [58,'Start with the six numbers below','Continue below with six numbers and the mathematics of approximation.']
  ].map(([start,title,text])=>({start,title,text}));
  const clamp=x=>Math.max(0,Math.min(1,x)), ease=x=>{x=clamp(x);return x*x*(3-2*x);};
  const at=(u,start,duration=.65)=>motion.matches?Number(u>=start):ease((u-start)/duration);
  function el(tag,attrs={},parent=svg){const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));parent.appendChild(n);return n;}
  function text(p,x,y,value,size=24,color=C.ink,attrs={}){const n=el('text',{x,y,'font-size':size,fill:color,...attrs},p);n.textContent=value;return n;}
  const center=(p,x,y,v,s=24,c=C.ink,a={})=>text(p,x,y,v,s,c,{'text-anchor':'middle',...a});
  const box=(p,x,y,w,h,fill=C.wall,a={})=>el('rect',{x,y,width:w,height:h,rx:12,fill,...a},p);
  const line=(p,x1,y1,x2,y2,stroke=C.line,a={})=>el('line',{x1,y1,x2,y2,stroke,'stroke-width':2,'stroke-linecap':'round',...a},p);
  const path=(p,d,fill='none',stroke=C.ink,a={})=>el('path',{d,fill,stroke,'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round',...a},p);
  const show=(n,p)=>n.setAttribute('opacity',clamp(p));
  function reveal(n,p,dy=9){show(n,p);n.setAttribute('transform',`translate(0 ${dy*(1-p)})`);}
  function bracket(p,x1,x2,y,color=C.red){return path(p,`M${x1} ${y+7}V${y}H${x2}V${y+7}`,'none',color,{'stroke-width':3});}
  function chip(p,x,y,w,v,fill,color=C.ink,size=22){const g=el('g',{},p);box(g,x,y,w,38,fill,{rx:19});center(g,x+w/2,y+26,v,size,color);return g;}
  let seconds=0,playing=false,lastTime=0,frame=0,cueIndex=-1,scenes=[],updates=[],small=false,W=960,H=540;

  // Arms, legs, gaze and expression are independent parts, all driven by story time.
  function actor(p,id,x,y,s=1,seated=false){
    const g=el('g',{'data-character':id,transform:`translate(${x} ${y}) scale(${s})`},p);
    const skin={professor:'#e3b695',ochre:'#efc5a5',lavender:'#b9805e'}[id];
    const shirt={professor:'#8db8a0',ochre:'#dfb681',lavender:'#b7abc9'}[id];
    el('ellipse',{cx:69,cy:232,rx:49,ry:7,fill:'#dce6df'},g);
    if(seated){box(g,28,100,79,69,'#58777b',{rx:16});line(g,61,166,61,224,C.ink,{'stroke-width':7});line(g,33,227,92,227,C.ink,{'stroke-width':6});}
    const legs=path(g,'','none','#526b70',{'stroke-width':19});
    const shoes=path(g,'','none',C.ink,{'stroke-width':9});
    el('use',{href:asset+'torso-'+id,width:140,height:240},g);
    const left=path(g,'','none',shirt,{'stroke-width':16}),right=path(g,'','none',shirt,{'stroke-width':16});
    const hands=[el('circle',{r:7,fill:skin},g),el('circle',{r:7,fill:skin},g)];
    const head=el('g',{},g);el('use',{href:asset+'head-'+id,width:140,height:240},head);
    const brows=path(head,'M51 37L59 36M79 36L87 37','none',C.ink,{'stroke-width':1.8});
    const eyes=[el('ellipse',{cx:56,cy:46,rx:2,ry:2,fill:C.ink},head),el('ellipse',{cx:82,cy:46,rx:2,ry:2,fill:C.ink},head)];
    const mouth=path(head,'M61 63Q69 69 78 62','none',id==='lavender'?'#684939':'#99694f',{'stroke-width':2});
    return (u,{gesture=0,think=0,walk=0,nod=0,dx=0,dy=0}={})=>{
      const decorative=motion.matches?0:1, beat=Math.sin(u*5)*walk*decorative;
      g.setAttribute('transform',`translate(${x+dx} ${y+dy-Math.abs(beat)*2}) scale(${s})`);
      const feet=seated?[[48,185,56,218],[91,181,110,213]]:[[51+beat*8,188,47+beat*12,224],[87-beat*8,188,91-beat*12,224]];
      legs.setAttribute('d',`M52 149L${feet[0][0]} ${feet[0][1]}L${feet[0][2]} ${feet[0][3]}M86 149L${feet[1][0]} ${feet[1][1]}L${feet[1][2]} ${feet[1][3]}`);
      shoes.setAttribute('d',`M${feet[0][2]-12} ${feet[0][3]+3}h18M${feet[1][2]-2} ${feet[1][3]+3}h20`);
      const lx=seated?61:45,ly=seated?131:144;
      left.setAttribute('d',`M44 87Q${20+think*16} ${116-think*11} ${lx+think*9} ${ly-think*67}`);
      const rx=98+gesture*36,ry=(seated?132:143)-gesture*75;
      right.setAttribute('d',`M94 87Q${116+gesture*5} ${113-gesture*30} ${rx} ${ry}`);
      hands[0].setAttribute('cx',lx+think*9);hands[0].setAttribute('cy',ly-think*67);
      hands[1].setAttribute('cx',rx);hands[1].setAttribute('cy',ry);
      const angle=gesture*-5+think*5+nod*decorative;
      head.setAttribute('transform',`rotate(${angle} 69 77)`);
      const blink=decorative && (u%4.1>3.94);eyes.forEach(e=>e.setAttribute('ry',blink?.4:2));
      // Brows and smile turn a question into recognition as an event resolves.
      brows.setAttribute('d',think>.4?'M51 36L59 33M79 33L87 36':'M51 37L59 36M79 36L87 37');
      mouth.setAttribute('d',think>.6?'M63 64Q69 62 76 64':'M61 63Q69 70 78 62');
    };
  }
  function plant(p,x,y,s=1){const g=el('g',{transform:`translate(${x} ${y}) scale(${s})`},p);path(g,'M0 0Q5-35-3-83','none','#71937d');[-25,-43,-61,-77].forEach((v,i)=>{path(g,`M1 ${v}Q-27 ${v-22}-21 ${v-6}Q-10 ${v+3}1 ${v}`,'#b4d4b4','#71937d');path(g,`M2 ${v-8}Q29 ${v-30}23 ${v-12}Q15 ${v}2 ${v-8}`,'#cde3c7','#71937d');});path(g,'M-20 0H23L17 33H-14Z','#e5c2ab','#bc9b87');return g;}
  function room(p,name){
    box(p,0,0,W,H,C.wall,{rx:0});path(p,`M0 ${H-101}Q${W*.45} ${H-119} ${W} ${H-96}V${H}H0Z`,'#edf3eb','none');
    text(p,28,31,'UNIVERSITY / '+name,small?19:16,C.muted,{'letter-spacing':2});
    if(!small){box(p,35,66,130,117,'#eef6f3',{rx:2,stroke:'#b7cdc4'});line(p,100,66,100,183);line(p,35,123,165,123);plant(p,175,420,.85);}
  }
  function board(p,title,{laptop=false}={}){
    const b={x:small?28:242,y:66,w:small?584:682,h:small?350:325};
    if(laptop){box(p,b.x-8,b.y-8,b.w+16,b.h+16,'#60828a',{rx:15});path(p,`M${b.x-25} ${b.y+b.h+10}H${b.x+b.w+25}l-20 15H${b.x-5}Z`,'#b6c9c9','#6e8e93');}
    else{line(p,b.x+64,b.y+b.h,b.x+38,H-71,C.line,{'stroke-width':7});line(p,b.x+b.w-64,b.y+b.h,b.x+b.w-38,H-71,C.line,{'stroke-width':7});box(p,b.x-7,b.y-7,b.w+14,b.h+14,'#a9c3b5');}
    box(p,b.x,b.y,b.w,b.h,'#f9fcf9',{rx:7});
    text(p,b.x+22,b.y+38,title,small?27:28,C.ink,{'font-weight':600});
    line(p,b.x+20,b.y+54,b.x+b.w-20,b.y+54,'#e0e9df');return b;
  }
  function audience(p){
    const y=small?448:352,s=small?.78:.73;
    const a=actor(p,'ochre',small?260:650,y,s,true),b=actor(p,'lavender',small?443:806,y,s,true);
    // Layered writing desks echo the tiered lecture-hall reference.
    box(p,small?247:632,small?564:463,small?357:302,30,'#bdcec5',{rx:3,stroke:'#819c91'});
    line(p,small?269:654,small?594:493,small?269:654,H,C.line,{'stroke-width':5});
    line(p,W-45,small?594:493,W-45,H,C.line,{'stroke-width':5});
    return [a,b];
  }
  function group(p){
    const y=small?439:345;
    const a=actor(p,'ochre',small?58:232,y,small?.78:.78,true),b=actor(p,'lavender',small?439:786,y,small?.78:.78,true);
    const c=actor(p,'professor',small?257:63,small?466:240,small?.72:1.14);
    const cx=small?330:577,ty=small?568:470,rx=small?236:325;
    line(p,cx,ty,cx,H-9,'#72958d',{'stroke-width':11});el('ellipse',{cx,cy:ty,rx,ry:small?30:25,fill:'#b6d5c9',stroke:'#5f897f','stroke-width':2},p);
    path(p,`M${cx-105} ${ty-8}l50-8 30 19-52 5Z`,'#fffdf8','#c6d6cf');
    box(p,cx+64,ty-23,25,25,'#edd0bd',{rx:3});path(p,`M${cx+89} ${ty-19}q17 0 0 13`,'none','#b18e7a');
    return [a,b,c];
  }
  function build(){
    small=compact.matches;W=small?640:960;H=small?650:540;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
    svg.querySelectorAll('[data-story-scene]').forEach(n=>n.remove());scenes=[];updates=[];
    const make=()=>{const n=el('g',{'data-story-scene':scenes.length,'aria-hidden':'true'},svg);scenes.push(n);return n;};
    // 0–6: A short campus establishing shot, with the question already in view.
    {
      const g=make();box(g,0,0,W,H,C.wall,{rx:0});
      path(g,`M25 ${H-140}Q${W/2} ${H-230} ${W-20} ${H-130}V${H}H20Z`,'#e5f2e3','none');
      const building=el('g',{},g),bx=small?118:228,bw=small?404:505,by=small?200:180;
      box(building,bx-72,by+61,bw+144,128,'#ebefeb',{rx:0,stroke:'#afc3bc'});
      path(building,`M${bx-82} ${by+61}h${bw+164}l-24-15H${bx-61}Z`,'#d79c89','none');
      box(building,bx,by,bw,191,'#f4f3eb',{rx:0,stroke:'#9fb5ac'});
      path(building,`M${bx-17} ${by}L${bx+bw/2} ${by-75}L${bx+bw+17} ${by}Z`,'#f2d4c8','#b79480');
      line(building,bx-8,by,bx+bw+8,by,'#99b2a8',{'stroke-width':6});
      el('circle',{cx:bx+bw/2,cy:by-28,r:17,fill:'#fffdf8',stroke:'#bac9bf','stroke-width':2},building);
      for(let i=0;i<5;i++){const x=bx+26+i*(bw-85)/4;box(building,x,by+22,17,142,'#d2ddd5',{rx:0});line(building,x-3,by+164,x+21,by+164,'#a6bbb0',{'stroke-width':5});}
      for(let i=0;i<3;i++)box(building,bx+bw/2-51+i*36,by+58,22,53,'#78969a',{rx:10});
      box(building,bx+bw/2-32,by+128,64,59,'#45636b',{rx:2});
      for(let i=0;i<3;i++)box(building,bx-15-i*13,by+190+i*8,bw+30+i*26,8,'#d5ddd4',{rx:0});
      const trees=[small?46:131,small?587:848].map((x,i)=>{const tr=el('g',{},g);path(tr,`M${x} ${by+213}L${x+5} ${by+91}`,'none','#b08872',{'stroke-width':10});[[-24,0,35],[15,-23,42],[37,9,33],[-7,22,40]].forEach(([dx,dy,r])=>el('circle',{cx:x+dx,cy:by+78+dy,r,fill:i?'#a8cbb1':'#b7d5bb'},tr));return tr;});
      const question=el('g',{},g);center(question,W/2,small?63:49,'Many possible numbers.',small?34:36,C.ink,{'font-weight':600});center(question,W/2,small?105:91,'Only a few choices?',small?34:36,C.ink,{'font-weight':600});
      const a=actor(g,'ochre',small?148:330,small?413:314,.9),b=actor(g,'lavender',small?372:552,small?413:314,.9);
      const bubbles=[chip(g,small?149:335,small?371:286,96,'2.4',C.paleBlue,C.blue,29),chip(g,small?367:558,small?371:286,96,'2 ?',C.mint,C.green,29)];
      updates.push(u=>{const p=at(u,0,4.8);a(u,{walk:1-p,gesture:at(u,1,1)*.7,dx:-42*(1-p)});b(u,{walk:1-p,think:1-at(u,4),dx:-32*(1-p)});reveal(bubbles[0],at(u,.6));reveal(bubbles[1],at(u,2.2));trees.forEach((tr,i)=>tr.setAttribute('transform',`rotate(${motion.matches?0:.5*Math.sin(u*.8+i)} ${i?W-100:100} ${by+210})`));building.setAttribute('transform',`translate(0 ${-3*p})`);});
    }
    // 6–15: The whole class sees the many-to-one mapping develop briskly.
    {
      const g=make();room(g,'MATHEMATICS');const b=board(g,'Report only whole numbers');
      const p=actor(g,'professor',small?54:47,small?425:219,small?.87:1.2),[a,c]=audience(g);
      const rows=measurements.map((x,i)=>{const col=i%3,row=Math.floor(i/3),lx=b.x+24+col*(b.w-40)/3,y=b.y+109+row*98;
        const input=text(g,lx,y,x.toFixed(1),small?34:36,C.blue);const arrow=text(g,lx+72,y,'→',29,C.muted);
        const out=text(g,lx+128,y,M.round(x),36,C.green,{'font-weight':700});
        const route=line(g,lx+7,y+15,lx+144,y+15,'#abc1b2',{'stroke-width':3});return {input,arrow,out,route};});
      const conclusion=chip(g,b.x+21,b.y+b.h-53,b.w-42,'6 inputs → 3 distinct results',C.mint,C.green,small?25:27);
      const recognition=chip(g,small?373:758,small?423:342,small?218:163,'Same answer!',C.peach,C.ink,small?23:19);
      updates.push(u=>{rows.forEach((r,i)=>{const t=.45+i*.79;show(r.input,at(u,.15+i*.13,.4));show(r.arrow,at(u,t,.35));reveal(r.out,at(u,t+.3,.5));r.route.setAttribute('stroke-dasharray','140');r.route.setAttribute('stroke-dashoffset',140*(1-at(u,t+.2,.6)));});reveal(conclusion,at(u,6.4));reveal(recognition,at(u,7.4));p(u,{gesture:.4+.55*at(u,.4,1),nod:Math.sin(u*1.3)*1.3});a(u,{think:1-at(u,5.5),gesture:.55*at(u,6.7)});c(u,{gesture:.8*at(u,7.2),nod:Math.sin(u*3)*at(u,6)*2});});
    }
    // 15–24: A student traces the signed error, then the adjacent-level spacing.
    {
      const g=make();room(g,'AT THE PRESENTATION BOARD');const b=board(g,'Where did 2.4 go?');
      const X=v=>b.x+55+(v-1)/2*(b.w-110),y=b.y+144;
      line(g,X(1),y,X(3),y,'#91aaa1',{'stroke-width':3});[1,2,3].forEach(v=>{line(g,X(v),y-10,X(v),y+10,C.green,{'stroke-width':3});center(g,X(v),y+40,v,28,C.green);});
      const input=el('g',{},g);el('circle',{cx:X(2.4),cy:y,r:9,fill:C.blue},input);center(input,X(2.4),y-70,'x = 2.4',28,C.blue);
      const recon=el('g',{},g);box(recon,X(2)-9,y-9,18,18,C.green,{rx:3});center(recon,X(2)-20,y+78,'Q(x) = 2',26,C.green);
      const traveler=el('circle',{cx:X(2.4),cy:y,r:6,fill:C.blue},g);
      const error=el('g',{},g);const gap=bracket(error,X(2),X(2.4),y-26);center(error,(X(2)+X(2.4))/2,y-38,'0.4',25,C.red);
      const eq=center(g,b.x+b.w/2,b.y+b.h-23,'e = 2.4 − 2 = 0.4',30,C.red,{'font-weight':600});
      const delta=el('g',{},g);bracket(delta,X(1),X(2),y+91,C.green);center(delta,(X(1)+X(2))/2,y+128,'step Δ = 1',25,C.green);
      const a=actor(g,'ochre',small?54:52,small?428:217,small?.87:1.23),p=actor(g,'professor',small?423:786,small?446:356,.75);
      updates.push(u=>{show(input,1);show(recon,at(u,.65));traveler.setAttribute('cx',X(2.4)-at(u,1,1.1)*(X(2.4)-X(2)));show(traveler,1-at(u,2.1,.25));show(error,at(u,2.5));gap.setAttribute('stroke-dasharray','160');gap.setAttribute('stroke-dashoffset',160*(1-at(u,2.5,.9)));reveal(eq,at(u,3.4));reveal(delta,at(u,6.2));a(u,{gesture:.2+.8*at(u,2.3),think:.65*(1-at(u,3.4))});p(u,{gesture:.75*at(u,6),nod:at(u,3)*Math.sin(u*2)*2});});
    }
    // 24–33: Identical input on nested grids; the finer selected approximation is closer.
    {
      const g=make();room(g,'STUDENT STUDY GROUP');const b=board(g,'Same input: x = 2.4',{laptop:true});
      const X=v=>b.x+58+(v-1)/2*(b.w-116),ys=[b.y+153,b.y+258];
      const fine=el('g',{},g),coarse=el('g',{},g);
      [coarse,fine].forEach((n,i)=>{const delta=i?.25:1,y=ys[i];text(n,b.x+22,y-44,i?'FINE   Δ = 0.25':'COARSE   Δ = 1',small?22:23,C.muted);line(n,X(1),y,X(3),y);for(let v=1;v<=3.001;v+=delta){line(n,X(v),y-8,X(v),y+8,C.green);el('circle',{cx:X(v),cy:y,r:i?3.5:5,fill:C.green},n);} [1,2,3].forEach(v=>center(n,X(v),y+31,v,22,C.muted));});
      const input=line(g,X(2.4),ys[0]-18,X(2.4),ys[1]+13,C.blue,{'stroke-width':2,'stroke-dasharray':'5 5'});
      const dots=ys.map(y=>el('circle',{cx:X(2.4),cy:y,r:8,fill:C.blue},g));
      const approximations=ys.map((y,i)=>{const n=el('g',{},g),q=i?2.5:2;box(n,X(q)-7,y-7,14,14,C.green,{rx:2});line(n,X(q),y-16,X(2.4),y-16,C.red,{'stroke-width':4});center(n,b.x+b.w-77,y-28,i?'|e| = 0.1':'|e| = 0.4',23,C.red);return n;});
      const conclusion=chip(g,b.x+40,b.y+b.h-42,b.w-80,'2.4 → 2       2.4 → 2.5',C.mint,C.green,small?27:28);
      const [a,c,p]=group(g);const selection=chip(g,small?208:444,small?421:394,small?205:247,'Try a finer step',C.peach,C.ink,small?23:22);
      updates.push(u=>{show(coarse,1);show(fine,at(u,1.5));show(input,at(u,.2));dots.forEach((n,i)=>show(n,at(u,i?2:.3)));approximations.forEach((n,i)=>reveal(n,at(u,i?4:1)));reveal(conclusion,at(u,4.6));reveal(selection,at(u,2.5)*(1-at(u,7.2)));a(u,{gesture:.8*at(u,1),think:.5*(1-at(u,3))});c(u,{gesture:.95*at(u,3.2),nod:Math.sin(u*2)*at(u,6)*2});p(u,{gesture:.6*at(u,5)});});
    }
    // 33–42: A schematic network becomes a code and an explicitly decoded weight.
    {
      const g=make();room(g,'COMPUTER LAB');const b=board(g,'Networks learn numerical weights');
      const xs=[b.x+73,b.x+b.w/2,b.x+b.w-73],ys=[[b.y+112,b.y+202],[b.y+99,b.y+158,b.y+217],[b.y+112,b.y+202]],edges=[];
      for(let layer=0;layer<2;layer++)ys[layer].forEach(y=>ys[layer+1].forEach(z=>edges.push(line(g,xs[layer],y,xs[layer+1],z,'#b1c7c0',{'stroke-width':2}))));
      xs.forEach((x,i)=>ys[i].forEach(y=>el('circle',{cx:x,cy:y,r:14,fill:i===1?C.mint:C.paleBlue,stroke:'#8cae9d','stroke-width':2},g)));
      const weightLabels=weights.map((w,i)=>chip(g,b.x+100+(i%2)*(b.w-220)/2,b.y+82+Math.floor(i/2)*88,106,w.toFixed(2),C.paleBlue,C.blue,25));
      const conversion=el('g',{},g);box(conversion,b.x+16,b.y+65,b.w-32,190,'#f9fcf9');
      const values=[['weight','0.73',C.paleBlue,C.blue],['4-bit code',example.q.toString(2).padStart(4,'0'),C.lavender,C.ink],['reconstruct',example.xhat.toFixed(2),C.mint,C.green]];
      values.forEach(([label,v,fill,color],i)=>{const cw=(b.w-80)/3,x=b.x+20+i*(cw+20);box(conversion,x,b.y+83,cw,116,fill);center(conversion,x+cw/2,b.y+115,label,small?21:22,C.muted);center(conversion,x+cw/2,b.y+171,v,small?33:37,color,{'font-weight':600});if(i<2)center(conversion,x+cw+10,b.y+158,'→',25,C.muted);});
      const error=center(conversion,b.x+b.w/2,b.y+238,'e = 0.73 − 0.77 = −0.04',small?26:29,C.red);
      const formula=center(g,b.x+b.w/2,b.y+b.h-23,'scale × (8 − 5) = 0.77',small?25:28,C.green);
      const [a,c,p]=group(g);
      updates.push(u=>{const phase=at(u,3.8);edges.forEach((n,i)=>{n.setAttribute('stroke',i===(motion.matches?0:Math.floor(u*1.6)%edges.length)?C.blue:'#b1c7c0');show(n,1-phase);});weightLabels.forEach((n,i)=>show(n,at(u,.3+i*.65)*(1-phase)));reveal(conversion,phase);show(error,at(u,5.9));reveal(formula,at(u,7.1));a(u,{think:.6*(1-at(u,4)),gesture:.6*at(u,6)});c(u,{gesture:.95*at(u,.8),nod:Math.sin(u)*2});p(u,{gesture:.65*at(u,4)});});
    }
    // 42–52: Equal-N memory blocks, then a precise boundary on the claim.
    {
      const g=make();room(g,'THE WEIGHT STORAGE BUDGET');const b=board(g,'Same N weights. Count the raw bits.');
      const tw=small?35:39,gap=7,left=b.x+40,right=b.x+b.w*.59,top=b.y+113;
      const tiles=[[],[]];[16,4].forEach((count,j)=>{const x=j?right:left;center(g,x+2*tw+gap*1.5,b.y+89,j?'4-bit':'16-bit',29,j?C.green:C.blue,{'font-weight':600});for(let i=0;i<count;i++)tiles[j].push(box(g,x+i%4*(tw+gap),top+Math.floor(i/4)*(tw+gap),tw,tw,j?C.mint:C.lavender,{rx:5,stroke:j?'#88ac92':'#aaa0b9'}));});
      const counts=[center(g,left+2*tw+gap*1.5,b.y+b.h-(small?24:9),'16N bits',28,C.blue),center(g,right+2*tw+gap*1.5,top+94,'4N bits',28,C.green)];
      const quarter=chip(g,right-6,top+115,tw*4+gap*3+12,'¼ of the raw bits',C.peach,C.ink,small?22:23);
      const foot=chip(g,b.x+26,b.y+b.h+24,b.w-52,'Each tile = N bits',C.paleBlue,C.blue,24);
      const caveat=el('g',{},g);box(caveat,b.x+12,b.y+204,b.w-24,95,C.peach);center(caveat,b.x+b.w/2,b.y+242,'Total runtime memory ≠ raw weight bits',small?24:26,C.ink);center(caveat,b.x+b.w/2,b.y+278,'Inference speed is a separate question',small?24:26,C.ink);
      const p=actor(g,'professor',small?53:44,small?433:226,small?.83:1.18),a=actor(g,'ochre',small?291:681,small?463:363,.69),c=actor(g,'lavender',small?472:823,small?463:363,.69);
      updates.push(u=>{tiles.forEach((arr,j)=>arr.forEach((n,i)=>{reveal(n,at(u,.25+i*.105+(j?1.4:0),.4),j?12:4);show(n,at(u,.25+i*.105+(j?1.4:0),.4)*(1-.82*at(u,6)));}));counts.forEach(n=>show(n,at(u,2.1)*(1-at(u,6))));reveal(quarter,at(u,4));show(quarter,at(u,4)*(1-at(u,6)));reveal(caveat,at(u,6));show(foot,1-at(u,6));p(u,{gesture:.85*at(u,.4),nod:Math.sin(u)*1.5});a(u,{gesture:.7*at(u,4),think:.7*at(u,6.3)});c(u,{gesture:.75*at(u,5),think:.6*at(u,6.5)});});
    }
    // 52–60: A shared board synthesizes both settings and hands off to the article.
    {
      const g=make();room(g,'ONE CONNECTING IDEA');const b=board(g,'Quantization: choose an approximation');
      const cards=[el('g',{},g),el('g',{},g)],cw=(b.w-54)/2;
      [['Measurements','2.4 → 2',C.paleBlue,C.blue],['Model weights','0.73 → 1000',C.mint,C.green]].forEach(([title,v,fill,color],i)=>{const x=b.x+18+i*(cw+18);box(cards[i],x,b.y+78,cw,112,fill);center(cards[i],x+cw/2,b.y+114,title,small?24:26,color);center(cards[i],x+cw/2,b.y+166,v,small?31:35,color,{'font-weight':600});});
      const trade=el('g',{},g);center(trade,b.x+b.w/2,b.y+242,'Fewer bits ↔ Approximation error',small?29:31,C.red,{'font-weight':600});
      const qualification=center(g,b.x+b.w/2,b.y+b.h-(small?23:40),'Assess the method, model and task.',small?25:27,C.muted);
      const p=actor(g,'professor',small?54:47,small?430:226,small?.84:1.17),a=actor(g,'ochre',small?285:677,small?446:354,.76),c=actor(g,'lavender',small?467:821,small?446:354,.76);
      const next=chip(g,small?154:295,small?592:467,small?344:350,'Next: start with six numbers ↓',C.peach,C.ink,small?24:24);
      updates.push(u=>{cards.forEach((n,i)=>reveal(n,at(u,.2+i*1.2)));reveal(trade,at(u,3));reveal(qualification,at(u,4.7));reveal(next,at(u,6));p(u,{gesture:.65*at(u,.6)});a(u,{gesture:.65*at(u,2.1),nod:Math.sin(u*2)*at(u,5)*1.6});c(u,{gesture:.8*at(u,3.2),nod:Math.sin(u*2)*at(u,5)*1.6});});
    }
  }
  function render(t){
    seconds=t>=60-1e-9?60:Math.max(0,t);const index=starts.findLastIndex(s=>seconds>=s);
    // Crossfade into the incoming composition during the last 450 ms of a scene.
    // Exact chapter boundaries therefore show the requested chapter, even while paused.
    const blend=motion.matches||index===6?0:at(seconds,ends[index]-.45,.45);
    scenes.forEach((scene,i)=>{
      const active=i===index,incoming=i===index+1&&blend>0;
      scene.style.display=active||incoming?'':'none';show(scene,active?1-blend:incoming?blend:0);
      if(active||incoming){const u=Math.max(0,Math.min(ends[i]-starts[i],seconds-starts[i]));updates[i](u);const push=motion.matches?0:.009*clamp(u/(ends[i]-starts[i]));scene.setAttribute('transform',`translate(${-W*push/2} ${-H*push/2}) scale(${1+push})`);}
    });
    const cue=cues.findLastIndex(c=>seconds>=c.start);
    if(cue!==cueIndex){$('intro-chapter').textContent=cues[cue].title;$('intro-text').textContent=cues[cue].text;cueIndex=cue;}
    svg.setAttribute('data-chapter',index);svg.setAttribute('data-time',seconds.toFixed(3));
    document.querySelectorAll('#intro-animation .chapter-links [data-time]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.time===starts[index])));
    $('intro-timeline').value=seconds;$('intro-timeline').setAttribute('aria-valuetext',`${seconds.toFixed(1)} seconds: ${cues[cue].title}`);
    $('intro-time').textContent=(seconds>=60?'1:00':'0:'+String(Math.floor(seconds)).padStart(2,'0'))+' / 1:00';
  }
  function stop(){playing=false;cancelAnimationFrame(frame);$('intro-play').textContent='Play';$('intro-play').setAttribute('aria-pressed','false');}
  function tick(now){if(!playing)return;render(seconds+(now-lastTime)/1000*+$('intro-speed').value);lastTime=now;if(seconds>=60)stop();else frame=requestAnimationFrame(tick);}
  $('intro-play').onclick=()=>{if(playing){stop();return;}if(seconds>=60)render(0);playing=true;lastTime=performance.now();$('intro-play').textContent='Pause';$('intro-play').setAttribute('aria-pressed','true');frame=requestAnimationFrame(tick);};
  $('intro-restart').onclick=()=>{stop();render(0);};
  $('intro-timeline').oninput=()=>{render(+$('intro-timeline').value);lastTime=performance.now();};
  $('intro-speed').onchange=()=>{lastTime=performance.now();};
  document.querySelectorAll('#intro-animation .chapter-links [data-time]').forEach(b=>b.onclick=()=>{stop();render(+b.dataset.time);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stop();}).observe($('intro-animation'));
  motion.addEventListener('change',()=>{stop();render(seconds);});compact.addEventListener('change',()=>{build();render(seconds);sizeExplanation();});
  function sizeExplanation(){
    const box=document.querySelector('#intro-animation .animation-explanation'),probe=box.cloneNode(false);
    probe.style.cssText=`position:absolute;visibility:hidden;pointer-events:none;width:${box.getBoundingClientRect().width}px;height:auto;`;
    const title=document.createElement('b'),body=document.createElement('p');probe.append(title,body);box.parentElement.appendChild(probe);
    let height=0;cues.forEach(c=>{title.textContent=c.title;body.textContent=c.text;height=Math.max(height,probe.getBoundingClientRect().height);});probe.remove();box.style.height=Math.ceil(height)+'px';
  }
  let captionWidth=0;new ResizeObserver(entries=>{const width=entries[0].contentRect.width;if(Math.abs(width-captionWidth)>1){captionWidth=width;sizeExplanation();}}).observe(document.querySelector('#intro-animation .animation-explanation'));
  if(document.fonts)document.fonts.ready.then(sizeExplanation);
  build();render(0);sizeExplanation();
})();
