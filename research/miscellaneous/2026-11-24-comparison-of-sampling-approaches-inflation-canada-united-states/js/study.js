'use strict';
(() => {
  const {$,$$,labs,asset,escape,select}=Study;
  const reduce=matchMedia('(prefers-reduced-motion: reduce)');
  const frames=[
    ['Households and transactions','Households buy goods and services across diverse Canadian markets.','CPI-U follows the prices paid by urban consumers across diverse markets.','⌂','Households'],
    ['Geographic frames','Provincial and territorial collection is supplemented by central sources.','Urban primary sampling areas include certainty areas and sampled areas.','◎','Area frames'],
    ['Outlet and item selection','Market information guides outlet and representative product choices.','Probability selection connects areas, outlets and item offers in applicable strata.','▤','Outlets and products'],
    ['Collection channels','Scheduled quotes, scanner files and administrative records feed the CPI.','Survey prices, rental observations and secondary data feed the CPI-U.','▥','Price records'],
    ['Repeated observations','Compare matched product offers across months.','Reprice sampled items and return to rental survey panels on schedule.','→','pₜ₋₁ → pₜ'],
    ['Quality and missingness','Adjust comparable quantities and quality or borrow donor movement.','Apply component-specific replacement, quality and imputation rules.','≈','Quality-adjusted relative'],
    ['Elementary price changes','Combine relatives with the elementary formula appropriate to the source.','Combine area-item prices, commonly using a weighted geometric mean.','Σ','Elementary indexes'],
    ['National aggregation','Apply expenditure weights and link successive Lowe baskets.','Aggregate area-item indexes with modified Laspeyres-type CPI-U weights.','↗','Monthly national index'],
    ['Statistical uncertainty','Frame gaps require source checks, sensitivity and component evidence.','BLS publishes sampling standard errors for selected CPI changes.','±','Reliability and uncertainty'],
    ['Compare the evidence','Where do broad data sources, matching and coverage improve measurement?','When do explicit probability designs and replication improve inference?','?','Explore the study']
  ];
  const duration=52000, step=duration/frames.length;
  const controls=$('#film-controls');
  let elapsed=0,last=0,running=false, raf=0, stage=-1;
  controls.insertAdjacentHTML('beforeend','<div class="film-seek"><output id="film-elapsed">00:00</output>'+
    '<input id="film-seek" type="range" min="0" max="'+duration+'" step="100" value="0" aria-label="Seek animation">'+
    '<span id="film-duration">00:52</span></div><div class="controls"><button type="button" data-play>Play</button>'+
    '<button type="button" data-replay>Replay</button><button type="button" data-next>Next step</button>'+
    select('Speed','speed',[[.5,'0.5×'],[1,'1×'],[1.5,'1.5×'],[2,'2×']])+'</div>');
  const speed=$('select',controls), play=$('[data-play]',controls), timeline=$('#film-seek');
  const timeLabel = ms => '00:'+String(Math.floor(ms/1000)).padStart(2,'0');
  speed.value='1';
  function drawMap(country,index) {
    const overlay=$('#map-'+country.toLowerCase());
    const points=CPI_CITIES[country];
    const focus=points.filter((_,i)=>index<2 || (i+index)%3!==0);
    const paths=index>=2&&index<=7?focus.map(([,x,y])=>'<path d="M'+x+' '+y+' Q'+((x+202)/2)+' '+(Math.min(y,151)-17)+' 202 151" stroke="currentColor" stroke-dasharray="3 3" fill="none" opacity=".48"/>').join(''):'';
    const icons={
      0:'M1 7L7 1l6 6M3 6v7h8V6M6 13V9h2v4',
      2:'M2 5h10l-1-4H3L2 5m1 0v8h8V5M5 13V8h4v5',
      3:'M3 1h8v12H3zm2 3h4M5 7h4m-4 3h3',
      4:'M2 5h10m-7-3L2 5l3 3m4-2 3 3-3 3M2 9h10'
    };
    const dots=points.map(([name,x,y],i)=>{
      const alpha=focus.includes(points[i])?1:.27, r=index===1?4.2:3.1;
      return icons[index]&&i%2===0 ?
        '<g transform="translate('+(x-6)+','+(y-6)+')" opacity="'+alpha+'"><circle cx="6" cy="6" r="8" fill="white" opacity=".87"/>'+
        '<path d="'+icons[index]+'" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/><title>'+escape(name)+' · illustrative location</title></g>' :
        '<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="currentColor" stroke="white" stroke-width="1" opacity="'+alpha+'"><title>'+escape(name)+' · illustrative location</title></circle>';
    }).join('');
    overlay.innerHTML='<g class="map-overlay"><g class="map-feature">'+paths+dots+'</g></g>';
  }
  function update() {
    const index=Math.min(frames.length-1,Math.floor(elapsed/step));
    timeline.value=elapsed;
    $('#film-elapsed').value=timeLabel(elapsed);
    timeline.setAttribute('aria-valuetext',timeLabel(elapsed)+' of '+timeLabel(duration)+' · '+frames[index][0]);
    if(index===stage)return;
    stage=index;
    const frame=frames[index];
    $('#film-status').textContent='Step '+(index+1)+' of '+frames.length+' · '+frame[0];
    $('#film-ca').textContent=frame[1]; $('#film-us').textContent=frame[2];
    for(const country of ['ca','us']){
      const [icon,label]=$$('#film-label-'+country+' span');
      icon.textContent=frame[3];
      label.textContent=frame[4];
      drawMap(country.toUpperCase(),index);
    }
  }
  function stop(){
    running=false;
    cancelAnimationFrame(raf);
    play.textContent='Play';
  }
  function tick(now){
    if(!running)return;
    elapsed=Math.min(duration,elapsed+Math.max(0,now-last)*(+speed.value));
    last=now;update();
    if(elapsed>=duration)stop();
    else raf=requestAnimationFrame(tick);
  }
  function start(){
    if(elapsed>=duration)elapsed=0;
    running=true;last=performance.now();
    play.textContent='Pause';raf=requestAnimationFrame(tick);
  }
  const seek=index=>{elapsed=Math.min(frames.length-1,Math.max(0,index))*step;update();};
  play.onclick=()=>running?stop():start();
  $('[data-replay]',controls).onclick=()=>{stop();seek(0);if(!reduce.matches)start();};
  $('[data-next]',controls).onclick=()=>{stop();seek(stage===frames.length-1?0:stage+1);};
  timeline.addEventListener('input',()=>{
    elapsed=timeline.valueAsNumber;
    last=performance.now();
    update();
  });
  reduce.addEventListener('change',()=>{if(reduce.matches)stop();});
  update();

  $$('[data-scope]').forEach(el=>{el.innerHTML=CPIArchitecture.scopeDiagram(el.dataset.scope);});
  $$('[data-method-icon]').forEach(el=>{el.innerHTML=CPIArchitecture.icon(el.dataset.methodIcon);});

  const toc=$('#contents'), small=matchMedia('(max-width:900px)');
  const navLinks=$$('nav a',toc);
  const entries=navLinks.map(a=>[a,document.getElementById(a.hash.slice(1))]).filter(([,el])=>el);
  const byTarget=new Map(entries.map(([a,el])=>[el,a]));
  let active, queued=false;
  function track(){
    queued=false;
    let current=entries[0][1];
    for(const [,target] of entries){
      if(target.getBoundingClientRect().top>130)break;
      current=target;
    }
    const anchor=byTarget.get(current);
    if(anchor===active)return;
    active?.removeAttribute('aria-current');active=anchor;
    active.setAttribute('aria-current','location');
    if(toc.open&&!small.matches){
      const a=anchor.getBoundingClientRect(), b=toc.getBoundingClientRect();
      if(a.top<b.top||a.bottom>b.bottom)toc.scrollTop+=a.top-b.top-b.height*.45;
    }
  }
  function queueTrack(){if(!queued){queued=true;requestAnimationFrame(track);}}
  function setTOC(){toc.open=!small.matches;queueTrack();}
  setTOC();small.addEventListener('change',setTOC);
  navLinks.forEach(link=>link.addEventListener('click',()=>{if(small.matches)toc.open=false;}));
  window.addEventListener('scroll',queueTrack,{passive:true});
  window.addEventListener('resize',queueTrack,{passive:true});

  function formulas() {
    for(const node of $$('.tex')){
      katex.render(node.textContent,node,{displayMode:true,throwOnError:false,strict:'ignore'});
    }
    const walker=document.createTreeWalker($('#study'),NodeFilter.SHOW_TEXT,{
      acceptNode(node){return node.data.includes('\\(')&&!node.parentElement.closest('.tex,code,pre,svg')?
        NodeFilter.FILTER_ACCEPT:NodeFilter.FILTER_REJECT;}
    });
    const nodes=[];while(walker.nextNode())nodes.push(walker.currentNode);
    for(const node of nodes){
      const frag=document.createDocumentFragment();
      node.data.split(/(\\\([^]*?\\\))/g).forEach((content,i)=>{
        if(i%2===0){frag.append(content);return;}
        const span=document.createElement('span');
        katex.render(content.slice(2,-2),span,{throwOnError:false,strict:'ignore'});
        frag.append(span);
      });
      node.replaceWith(frag);
    }
  }
  function fitEquations(root=document) {
    for(const box of $$('.equation',root)){
      const math=$('.katex-html',box);
      if(!math) continue;
      box.style.fontSize='';
      const style=getComputedStyle(box);
      const width=box.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight);
      const natural=math.getBoundingClientRect().width;
      if(natural>width+.5 && width>0)box.style.fontSize=parseFloat(style.fontSize)*width/natural+'px';
    }
  }
  if(globalThis.katex){
    formulas();
    document.fonts.ready.then(()=>fitEquations());
    window.addEventListener('resize',()=>fitEquations(),{passive:true});
    document.addEventListener('toggle',event=>{
      if(event.target instanceof HTMLDetailsElement && event.target.open)fitEquations(event.target);
    },true);
  }
  const snippets=[
    ['empirical','Python','scripts/prepare_data.py',[
      'def pct(new, old):',
      '    return 100 * (new / old - 1) if new is not None and old not in (None, 0) else None',
      '',
      'def lagged(values, lag):',
      '    return [None if i < lag else pct(value, values[i - lag]) for i, value in enumerate(values)]'
    ].join('\n')],
    ['simulation','JavaScript','js/sampling-engine.js',[
      'const errors=samples.map(s=>s.estimate-pop.truth);',
      'const bias=mean(errors), squared=errors.map(e=>e*e), v=mean(errors.map(e=>(e-bias)**2));',
      'const coverage=mean(samples.map(s=>Math.abs(s.estimate-pop.truth)<=1.96*s.se?1:0));'
    ].join('\n')]
  ];
  function highlight(text){
    const matcher=/(#[^\n]*|"[^"\n]*"|'[^'\n]*'|\b(?:const|def|return|None|if|else|in|for|not|Math)\b)/g;
    return text.split(matcher).map((part,i)=>i%2?'<span class="syntax-'+
      (part.startsWith('#')?'comment':/^['"]/.test(part)?'string':'key')+'">'+escape(part)+'</span>':escape(part)).join('');
  }
  function codePanel([id,lang,path,source]){
    const root=$('[data-code='+id+']');
    root.innerHTML='<details class="code"><summary><span aria-hidden="true">⟨/⟩</span> '+lang+' · Inspect calculation</summary><div><pre><code>'+
      highlight(source)+'</code></pre><div class="controls"><button type="button">Copy Code</button><a download href="'+asset(path)+
      '">Download complete script</a><span role="status"></span></div></div></details>';
    $('button',root).onclick=async()=>{
      const status=$('[role=status]',root);
      try{await navigator.clipboard.writeText(source);status.textContent='Code copied.';}
      catch{
        const field=document.createElement('textarea');field.value=source;root.append(field);field.select();
        const copied=document.execCommand('copy');field.remove();status.textContent=copied?'Code copied.':'Select the code above to copy it.';
      }
    };
  }
  snippets.forEach(codePanel);
  function activate(el){
    labs[el.dataset.lab](el);
    if(globalThis.katex) {
      for(const node of $$('.tex',el))katex.render(node.textContent,node,{displayMode:true,throwOnError:false,strict:'ignore'});
      fitEquations(el);
    }
    el.dataset.ready='true';
  }
  const labsOnPage=$$('[data-lab]');
  if('IntersectionObserver' in window){
    const observer=new IntersectionObserver(records=>{
      records.forEach(({target,isIntersecting})=>{
        if(isIntersecting){observer.unobserve(target);activate(target);}
      });
    },{rootMargin:'550px 0px'});
    labsOnPage.forEach(el=>observer.observe(el));
  }else labsOnPage.forEach(activate);
  queueTrack();
})();

