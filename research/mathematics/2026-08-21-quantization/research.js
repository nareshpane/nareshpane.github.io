/* Research-page navigation and scientific figures. No network dependencies. */
(() => {
  'use strict';
  const M=window.QuantizationMath, $=id=>document.getElementById(id);
  const NS='http://www.w3.org/2000/svg';
  const C={input:'#315f9a', quant:'#467950',error:'#b84545',grid:'#dce4e5',text:'#43565d'};
  const f=(x,n=4)=>Number(x).toFixed(n);
  function el(tag,attrs={},parent) {
    const e=document.createElementNS(NS,tag);
    Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));
    if(parent)parent.appendChild(e); return e;
  }
  function text(parent,x,y,value,attrs={}) {const e=el('text',{x,y,fill:C.text,'font-size':13,...attrs},parent);e.textContent=value;return e;}
  function line(parent,x1,y1,x2,y2,attrs={}) {return el('line',{x1,y1,x2,y2,stroke:C.grid,...attrs},parent);}
  function path(parent,pts,attrs={}) {return el('path',{d:pts.map((p,i)=>(i?'L':'M')+p.join(',')).join(' '),fill:'none',stroke:C.input,'stroke-width':2,...attrs},parent);}
  function bind(ids,fn,type='input') {ids.forEach(id=>$(id).addEventListener(type,fn));}
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');

  // Explicit anchors exist in the HTML; navigation reflects the actual hierarchy.
  const headings=Array.from(document.querySelectorAll('main h2,main h3'));
  let group;
  headings.forEach(h=>{
    if(h.tagName==='H2') {
      group=document.createElement('details'); group.open=false;
      const summary=document.createElement('summary');
      const a=document.createElement('a'); a.href='#'+h.id;a.textContent=h.textContent;
      summary.appendChild(a);group.appendChild(summary);$('toc').appendChild(group);
    } else if(group) {
      const a=document.createElement('a');a.href='#'+h.id;a.textContent=h.textContent;a.className='toc-sub';group.appendChild(a);
    }
  });
  const links=Array.from($('toc').querySelectorAll('a'));
  function active(h,fromClick=false) {
    const current=links.find(a=>a.hash==='#'+h.id);
    const changed=current && !current.hasAttribute('aria-current');
    links.forEach(a=>a===current?a.setAttribute('aria-current','location'):a.removeAttribute('aria-current'));
    // Expand only on deliberate navigation, preserving Collapse All while reading.
    if(fromClick && current) current.closest('details').open=true;
    if(changed && current && innerWidth>900){
      const box=$('page-nav'),d=current.closest('details'),shown=d.open?current:d.querySelector('summary');
      const r=shown.getBoundingClientRect(),b=box.getBoundingClientRect();
      if(r.top<b.top+85 || r.bottom>b.bottom-10)box.scrollTop+=r.top-b.top-100;
    }
  }
  let clickTarget=null;
  $('toc').addEventListener('click',event=>{
    const a=event.target.closest('a');if(!a)return;
    event.preventDefault(); const h=$(decodeURIComponent(a.hash.slice(1)));
    clickTarget=h; active(h,true);window.history.replaceState(null,'',a.hash);
    if(innerWidth<=900){$('page-nav').classList.remove('is-open');$('nav-toggle').setAttribute('aria-expanded','false');}
    h.scrollIntoView({behavior:reduced.matches?'instant':'smooth',block:'start'});
  });
  function spy() {
    if(clickTarget){const r=clickTarget.getBoundingClientRect();if(Math.abs(r.top-(innerWidth<=900?75:26))<6)clickTarget=null;else return;}
    const offset=innerWidth<=900?100:60;
    let h=headings[0];for(const item of headings){if(item.getBoundingClientRect().top<=offset)h=item;else break;}
    if(innerHeight+scrollY>=document.documentElement.scrollHeight-4)h=headings.at(-1);
    active(h);
  }
  // User interruption releases click targeting immediately.
  ['wheel','touchstart','keydown'].forEach(type=>window.addEventListener(type,()=>{clickTarget=null;},{passive:true}));
  let spyFrame=0;
  window.addEventListener('scroll',()=>{if(!spyFrame)spyFrame=requestAnimationFrame(()=>{spyFrame=0;spy();});},{passive:true});
  const headingObserver=new IntersectionObserver(spy,{rootMargin:'-70px 0px -65% 0px'});headings.forEach(h=>headingObserver.observe(h));
  window.addEventListener('scrollend',()=>{clickTarget=null;spy();});
  $('nav-expand').onclick=()=>{$('toc').querySelectorAll('details').forEach(d=>d.open=true);};
  $('nav-collapse').onclick=()=>{$('toc').querySelectorAll('details').forEach(d=>d.open=false);};
  $('nav-toggle').onclick=()=>{const open=$('page-nav').classList.toggle('is-open');$('nav-toggle').setAttribute('aria-expanded',String(open));};
  spy();

  // Draggable legacy number line; its range control remains the keyboard alternative.
  let draggingNumber=false;
  const num=$('numline-svg');
  function numberPointer(e){const r=num.getBoundingClientRect();$('num-val').value=M.clamp(((e.clientX-r.left)/r.width*760-40)/680*10,0,10);$('num-val').dispatchEvent(new Event('input'));}
  num.addEventListener('pointerdown',e=>{draggingNumber=true;num.setPointerCapture(e.pointerId);numberPointer(e);});
  num.addEventListener('pointermove',e=>{if(draggingNumber)numberPointer(e);});
  num.addEventListener('pointerup',()=>draggingNumber=false);num.addEventListener('pointercancel',()=>draggingNumber=false);
  $('num-val-out').min='0';$('num-val-out').max='10';

  function drawStair(){
    const svg=$('stair-svg');svg.replaceChildren();
    const x=Number($('stair-x').value),d=Number($('stair-delta').value),mode=$('stair-mode').value;
    const q=v=>M.staircase(v,d,mode),qx=q(x);
    const panels=[{x0:55,x1:415,name:'Reconstruction Q(x)',value:q,color:C.quant},{x0:515,x1:875,name:'Signed error e(x)',value:v=>v-q(v),color:C.error}];
    panels.forEach((p,k)=>{
      const px=v=>p.x0+(v+1.4)/2.8*(p.x1-p.x0),py=v=>153-v*86;
      text(svg,p.x0,26,p.name,{'font-weight':700});
      if(k===1)el('rect',{x:p.x0,y:py(d/2),width:p.x1-p.x0,height:d*86,fill:'#fae8e3'},svg);
      [-1,0,1].forEach(v=>{line(svg,p.x0,py(v),p.x1,py(v));text(svg,p.x0-9,py(v)+4,v,{'text-anchor':'end'});text(svg,px(v),289,v,{'text-anchor':'middle'});});
      line(svg,p.x0,274,p.x1,274,{stroke:'#a5b5b7'});text(svg,p.x1,314,'Input x',{'text-anchor':'end'});
      if(k===0)path(svg,[[px(-1.4),py(-1.4)],[px(1.4),py(1.4)]],{stroke:'#a8b5bb','stroke-dasharray':'5 4'});
      // Exact linear segments separated at analytic decision boundaries.
      const levels=[];for(let v=-1.5;v<=1.5;v+=d/2){const c=q(v);if(!levels.includes(c))levels.push(c);}
      levels.sort((a,b)=>a-b);
      levels.slice(1).forEach((c,i)=>{const boundary=(c+levels[i])/2;line(svg,px(boundary),45,px(boundary),270,{'stroke-dasharray':'3 5'});});
      levels.forEach((c,i)=>{const lo=i===0?-1.4:(levels[i-1]+c)/2,hi=i===levels.length-1?1.4:(c+levels[i+1])/2;
        path(svg,[[px(lo),py(k===0?c:lo-c)],[px(hi),py(k===0?c:hi-c)]],{stroke:p.color,'stroke-width':2.5});
        if(k===0){el('circle',{cx:px(lo),cy:py(c),r:2.5,fill:p.color},svg);el('circle',{cx:px(hi),cy:py(c),r:2.5,fill:i===levels.length-1?p.color:'white',stroke:p.color},svg);}
      });
      line(svg,px(x),45,px(x),270,{stroke:C.input,'stroke-dasharray':'4 3'});
      if(k===0){el('circle',{cx:px(x),cy:py(x),r:5,fill:C.input},svg);line(svg,px(x),py(x),px(x),py(qx),{stroke:C.error,'stroke-width':4});}
      el(k===0?'rect':'circle',k===0?{x:px(x)-5,y:py(qx)-5,width:10,height:10,fill:C.quant}:{cx:px(x),cy:py(x-qx),r:5,fill:C.error},svg);
    });
    $('stair-readout').textContent='x = '+f(x)+'   Q(x) = '+f(qx)+'   e = '+f(x-qx)+'   Δ/2 = '+f(d/2)+'   '+(Math.abs(x-qx)>d/2+1e-10?'Overload: bound exceeded.':'Within the granular error bound.');
  }
  bind(['stair-x','stair-delta','stair-mode'],drawStair);$('stair-reset').onclick=()=>{$('stair-x').value=.37;$('stair-delta').value=.25;$('stair-mode').value='rise';drawStair();};drawStair();
  function drawComparison(){
    const svg=$('compare-svg');svg.replaceChildren();const x=Number($('compare-x').value),px=v=>80+(v+1)*245;
    text(svg,80,23,'Input / reconstruction, common range [−1,1]');text(svg,650,23,'Q(x)           signed error',{'font-family':'monospace'});
    [1,2,3,4,6,8].forEach((b,i)=>{const y=53+i*40,r=M.uniform(x,2**b);text(svg,13,y+5,b+' bit'+(b===1?'':'s'));line(svg,px(-1),y,px(1),y,{stroke:'#b2c2ca'});
      for(let k=0;k<Math.min(2**b,32);k++){const v=-1+(k+.5)*2/Math.min(2**b,32);line(svg,px(v),y-4,px(v),y+4);}
      line(svg,px(x),y-10,px(r.value),y-10,{stroke:C.error,'stroke-width':3});el('circle',{cx:px(x),cy:y,r:5,fill:C.input},svg);el('rect',{x:px(r.value)-4,y:y-4,width:8,height:8,fill:C.quant},svg);
      text(svg,650,y+5,f(r.value)+'    '+f(r.error),{'font-family':'monospace'});text(svg,590,y+5,2**b+' levels',{'font-size':11});
    });
    text(svg,80,305,'−1');text(svg,570,305,'+1',{'text-anchor':'end'});
  }
  $('compare-x').oninput=drawComparison;$('compare-reset').onclick=()=>{$('compare-x').value=.37;drawComparison();};drawComparison();
  [-1.4,-1,-.75,0,.37,1,1.4].forEach(x=>{const r=M.uniform(x);const tr=document.createElement('tr');[f(x),r.index,r.index.toString(2).padStart(3,'0'),f(r.value),f(r.error)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td);});$('worked-rows').appendChild(tr);});

  let data,levels,baseline,history,iteration=0,lloydTimer=null;
  function stopLloyd(){clearInterval(lloydTimer);lloydTimer=null;$('lloyd-run').textContent='Run iterations';$('lloyd-run').setAttribute('aria-pressed','false');}
  function resetLloyd(){stopLloyd();const L=Number($('density-levels').value);data=M.distribution($('density-kind').value);baseline=Array.from({length:L},(_,i)=>-1+(i+.5)*2/L);levels=$('density-init').value==='biased'?baseline.map(c=>-.85+(c+1)*.45):baseline.slice();iteration=0;history=[M.distortion(data,levels)];$('density-cell').replaceChildren();levels.forEach((_,i)=>{const o=document.createElement('option');o.value=i;o.textContent='Cell '+i;$('density-cell').appendChild(o);});drawDensity();}
  function stepLloyd(){levels=M.lloyd(data,levels);iteration++;history.push(M.distortion(data,levels));drawDensity();if(iteration>=60 || Math.abs(history.at(-1)-history.at(-2))<1e-11)stopLloyd();}
  function drawDensity(){
    const svg=$('density-svg');svg.replaceChildren();const px=x=>65+(x+1)*385,top=190,bot=340;
    const peak=Math.max(...data.map(p=>p.w*data.length/2));const selected=Number($('density-cell').value);
    const bounds=[-1,...levels.slice(1).map((c,i)=>(c+levels[i])/2),1],lo=bounds[selected],hi=bounds[selected+1];
    el('rect',{x:px(lo),y:40,width:px(hi)-px(lo),height:bot-40,fill:'#e2efda'},svg);
    text(svg,65,23,'Normalized input density fX(x)');text(svg,65,220,'Squared-error density (x − Q(x))² fX(x)');
    const errors=data.map(p=>(p.x-levels[M.nearest(p.x,levels)])**2*p.w*data.length/2),errorPeak=Math.max(...errors,1e-8);
    let densityPts=[],errorPts=[];
    data.forEach((p,i)=>{if(i%4===0){densityPts.push([px(p.x),top-p.w*data.length/2/peak*135]);errorPts.push([px(p.x),bot-errors[i]/errorPeak*95]);}});
    path(svg,densityPts,{stroke:C.input,'stroke-width':2.5});path(svg,errorPts,{stroke:C.error});
    line(svg,65,top,835,top);line(svg,65,bot,835,bot);
    bounds.slice(1,-1).forEach(t=>line(svg,px(t),42,px(t),bot,{'stroke-dasharray':'4 4',stroke:'#78917c'}));
    baseline.forEach(c=>line(svg,px(c),top-10,px(c),top+11,{stroke:'#b49d7c','stroke-dasharray':'2 2','stroke-width':2}));
    levels.forEach((c,i)=>{el('rect',{x:px(c)-5,y:top-5,width:10,height:10,fill:C.quant},svg);text(svg,px(c),top+24,'c'+i,{'text-anchor':'middle','font-size':11});});
    [-1,-.5,0,.5,1].forEach(v=>{text(svg,px(v),365,v,{'text-anchor':'middle'});});text(svg,835,385,'Input x (dimensionless)',{'text-anchor':'end'});
    [0,peak/2,peak].forEach(v=>text(svg,54,top-v/peak*135+4,f(v,2),{'text-anchor':'end','font-size':11}));
    [0,errorPeak].forEach(v=>text(svg,54,bot-v/errorPeak*95+4,f(v,3),{'text-anchor':'end','font-size':11}));
    const D=M.distortion(data,levels),D0=M.distortion(data,baseline),cellD=data.reduce((sum,p)=>M.nearest(p.x,levels)===selected?sum+p.w*(p.x-levels[selected])**2:sum,0);
    $('density-readout').textContent='Iteration '+iteration+' · D = '+f(D,6)+' · uniform baseline = '+f(D0,6)+' · selected cell D = '+f(cellD,6)+' · Δ²/12 baseline = '+f((2/levels.length)**2/12,6)+' · levels ['+levels.map(c=>f(c)).join(', ')+'] · cell ['+f(lo)+', '+f(hi)+']';
    const hist=$('distortion-svg');hist.replaceChildren();const max=Math.max(...history,1e-8),hx=i=>65+i/Math.max(10,history.length-1)*770,hy=v=>120-v/max*85;
    text(hist,65,22,'Distortion D (dimensionless²)');line(hist,65,120,835,120);text(hist,835,155,'Iteration',{'text-anchor':'end'});text(hist,54,hy(max)+4,f(max,4),{'text-anchor':'end','font-size':11});text(hist,54,124,'0',{'text-anchor':'end'});
    path(hist,history.map((v,i)=>[hx(i),hy(v)]),{stroke:C.quant});history.forEach((v,i)=>el('circle',{cx:hx(i),cy:hy(v),r:3,fill:C.quant},hist));[0,Math.max(10,history.length-1)].forEach(i=>text(hist,hx(i),139,i,{'text-anchor':'middle','font-size':11}));
  }
  bind(['density-kind','density-levels','density-init'],resetLloyd);$('density-cell').oninput=drawDensity;$('lloyd-reset').onclick=resetLloyd;$('lloyd-step').onclick=()=>{stopLloyd();stepLloyd();};
  $('lloyd-run').onclick=()=>{if(lloydTimer){stopLloyd();return;}lloydTimer=setInterval(stepLloyd,reduced.matches?1200:650);$('lloyd-run').textContent='Pause iterations';$('lloyd-run').setAttribute('aria-pressed','true');};
  new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stopLloyd();}).observe($('viz-lloyd'));document.addEventListener('visibilitychange',()=>{if(document.hidden)stopLloyd();});resetLloyd();

  // Exact Voronoi cells: clip the box against each pairwise bisector half-plane.
  const rng=M.seeded(6331),cloud=Array.from({length:180},(_,i)=>{const cluster=i%3,centers=[[-.5,-.3],[.35,.5],[.45,-.5]],u=()=>rng()+rng()+rng()-1.5;return [M.clamp(centers[cluster][0]+u()*.39,-.96,.96),M.clamp(centers[cluster][1]+u()*.35,-.96,.96)];});
  let codebook,vqIteration=0,drag=-1;
  const colors=['#537ba0','#8b729f','#b48154','#5e927d'];
  const dist=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2;
  function assign(p){let k=0;codebook.forEach((c,i)=>{if(dist(p,c)<dist(p,codebook[k]))k=i;});return k;}
  function cell(k){let poly=[[-1,-1],[1,-1],[1,1],[-1,1]];const c=codebook[k];codebook.forEach((b,j)=>{if(j===k)return;const n=[b[0]-c[0],b[1]-c[1]],offset=(b[0]**2+b[1]**2-c[0]**2-c[1]**2)/2;
      // Coincident centers: lower index owns their shared area, consistent with assign.
      if(Math.abs(n[0])+Math.abs(n[1])<1e-12){if(j<k)poly=[];return;}
      const next=[];poly.forEach((p,i)=>{const q=poly[(i+1)%poly.length],a=p[0]*n[0]+p[1]*n[1]-offset,bv=q[0]*n[0]+q[1]*n[1]-offset;if(a<=1e-12)next.push(p);if((a<0&&bv>0)||(a>0&&bv<0)){const t=a/(a-bv);next.push([p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])]);}});poly=next;});return poly;}
  const vx=x=>245+(x+1)*180,vy=y=>390-(y+1)*180;
  function drawVQ(){const svg=$('vq-svg'),focused=document.activeElement?.dataset?.codeword;svg.replaceChildren();
    codebook.forEach((c,k)=>{const poly=cell(k);if(poly.length)el('polygon',{points:poly.map(p=>vx(p[0])+','+vy(p[1])).join(' '),fill:colors[k],'fill-opacity':.1,stroke:colors[k],'stroke-opacity':.5},svg);});
    for(let t=-1;t<=1;t+=.5){line(svg,vx(t),30,vx(t),390);line(svg,245,vy(t),605,vy(t));text(svg,vx(t),412,t,{'text-anchor':'middle'});text(svg,232,vy(t)+4,t,{'text-anchor':'end'});}
    text(svg,245,18,'Coordinate x₂');text(svg,620,412,'Coordinate x₁');text(svg,30,55,'180 source points',{'font-size':14});text(svg,30,79,'4 vector codewords');text(svg,30,103,'2 bits per vector');text(svg,650,55,'Codeword index');
    cloud.forEach(p=>el('circle',{cx:vx(p[0]),cy:vy(p[1]),r:3,fill:colors[assign(p)],'fill-opacity':.85},svg));
    codebook.forEach((c,k)=>{const g=el('g',{'data-codeword':k,tabindex:0,role:'button','aria-label':'Codeword '+k+', x1 '+f(c[0],2)+', x2 '+f(c[1],2)+'. Use arrow keys to move.'},svg);el('circle',{cx:vx(c[0]),cy:vy(c[1]),r:13,fill:'white',stroke:colors[k],'stroke-width':3},g);text(g,vx(c[0]),vy(c[1])+4,k,{'text-anchor':'middle','font-weight':700});text(svg,650,95+k*35,k+': ('+f(c[0],2)+', '+f(c[1],2)+')',{fill:colors[k]});});
    const D=cloud.reduce((s,p)=>s+dist(p,codebook[assign(p)]),0)/cloud.length;$('vq-readout').textContent='Centroid steps: '+vqIteration+' · empirical distortion = '+f(D,6)+' · 4 codewords × 2 coordinates';
    if(focused!==undefined)svg.querySelector('[data-codeword="'+focused+'"]').focus({preventScroll:true});
  }
  function resetVQ(){codebook=[[-.75,-.65],[-.45,.7],[.7,.65],[.7,-.7]];vqIteration=0;drawVQ();}
  $('vq-step').onclick=()=>{const sums=codebook.map(()=>[0,0,0]);cloud.forEach(p=>{const k=assign(p);sums[k][0]+=p[0];sums[k][1]+=p[1];sums[k][2]++;});codebook=codebook.map((c,k)=>sums[k][2]?[sums[k][0]/sums[k][2],sums[k][1]/sums[k][2]]:c);vqIteration++;drawVQ();};$('vq-reset').onclick=resetVQ;
  const vsvg=$('vq-svg');vsvg.addEventListener('keydown',e=>{const g=e.target.closest('[data-codeword]');if(!g||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();const k=Number(g.dataset.codeword),v=codebook[k];v[0]=M.clamp(v[0]+(e.key==='ArrowLeft'?-.02:e.key==='ArrowRight'?.02:0),-1,1);v[1]=M.clamp(v[1]+(e.key==='ArrowDown'?-.02:e.key==='ArrowUp'?.02:0),-1,1);drawVQ();});
  vsvg.addEventListener('pointerdown',e=>{const g=e.target.closest('[data-codeword]');if(!g)return;drag=Number(g.dataset.codeword);e.preventDefault();vsvg.setPointerCapture(e.pointerId);});
  vsvg.addEventListener('pointermove',e=>{if(drag<0)return;const p=new DOMPoint(e.clientX,e.clientY).matrixTransform(vsvg.getScreenCTM().inverse());codebook[drag]=[M.clamp((p.x-245)/180-1,-1,1),M.clamp((390-p.y)/180-1,-1,1)];drawVQ();});
  vsvg.addEventListener('pointerup',()=>drag=-1);vsvg.addEventListener('pointercancel',()=>drag=-1);resetVQ();
})();
