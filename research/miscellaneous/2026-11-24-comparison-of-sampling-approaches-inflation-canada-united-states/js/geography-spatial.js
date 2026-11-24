'use strict';
const CPISpatial = (() => {
  const {$, labs, panel, select, range, listen, picks, controls, metrics, table, fmt, sum, mean, norm, histogram} = Study;
  const {rng, sampleVariance} = CPISampling;
  const designs={metro:'Concentrate in large markets',equal:'Equal allocation',pop:'Population proportional',
    spend:'Expenditure proportional',strat:'Stratified random areas',optimal:'Cost-adjusted optimal'};
  const studyColors=['#a64936','#285f91'];
  const defaults={regions:12,quotes:132,areas:7,heterogeneity:2.2,correlation:.4,urban:1.5,remote:2.5,shock:5};
  const distances=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
  function spatialEffects(points,correlation,random) {
    const length=.04+correlation*.62;
    const factor=points.map((_,i)=>Array(i+1).fill(0));
    for(let i=0;i<points.length;i++){
      for(let j=0;j<=i;j++){
        const cov=i===j?1:correlation*Math.exp(-distances(points[i],points[j])/length);
        let partial=0;
        for(let k=0;k<j;k++)partial+=factor[i][k]*factor[j][k];
        factor[i][j]=i===j?Math.sqrt(Math.max(1e-12,cov-partial)):(cov-partial)/factor[j][j];
      }
    }
    const noise=points.map(()=>random.normal());
    return factor.map(row=>sum(row.map((v,i)=>v*noise[i])));
  }
  function makePopulation(config,seed) {
    const random=rng(seed);
    const {regions,heterogeneity,correlation,urban,shock}=config;
    const columns=Math.ceil(Math.sqrt(regions*1.55));
    const rows=Math.ceil(regions/columns);
    const points=Array.from({length:regions},(_,i)=>({
      id:i,x:(i%columns+.55+(random()-.5)*.34)/columns,
      y:(Math.floor(i/columns)+.55+(random()-.5)*.32)/rows
    }));
    const regional=spatialEffects(points,correlation,random);
    const shockArea=regions-1;
    const result=points.map((p,i)=>{
      const density=.12+.78*random();
      const remote=Math.max(.04,Math.min(.97,(1-density)*.65+.35*p.x));
      const population=Math.exp(1.5+2.6*density+random.normal()*.42);
      const spend=population*Math.exp(.8*density+random.normal()*.22);
      const outlets=Array.from({length:14},(_,j)=>{
        const city=j<Math.round(14*density);
        const cluster=.85*random.normal();
        return {
          x:p.x+(random()-.5)*(.55/columns),y:p.y+(random()-.5)*(.55/rows),city,
          prices:Array.from({length:24},()=>2+heterogeneity*regional[i]+urban*(city?.5:-.5)+
            (i===shockArea?shock:0)+cluster+1.8*random.normal())
        };
      });
      const prices=outlets.flatMap(o=>o.prices);
      return {...p,density,remote,population,spend,outlets,prices,
        truth:mean(prices),deviation:Math.sqrt(sampleVariance(prices))};
    });
    const weights=norm(result.map(r=>r.spend));
    result.forEach((r,i)=>{r.weight=weights[i];});
    return {regions:result,truth:sum(result.map(r=>r.truth*r.weight)),seed,shockArea};
  }
  function spreadAreas(regions,k) {
    const selected=[regions.reduce((a,b)=>a.spend>b.spend?a:b)];
    while(selected.length<k){
      const distance=r=>Math.min(...selected.map(s=>distances(r,s)));
      selected.push(regions.filter(r=>!selected.includes(r)).reduce((a,b)=>distance(a)>distance(b)?a:b));
    }
    return selected;
  }
  function chooseAreas(regions,k,design,random) {
    if(design==='metro')return {selected:[...regions].sort((a,b)=>b.density-a.density).slice(0,k)};
    if(design==='strat'){
      const ordered=[...regions].sort((a,b)=>a.x+a.y*.55-b.x-b.y*.55);
      const strata=Array.from({length:k},(_,i)=>ordered.slice(Math.floor(i*regions.length/k),Math.floor((i+1)*regions.length/k)));
      const selected=strata.map(s=>s[Math.floor(random()*s.length)]);
      return {selected,strata};
    }
    return {selected:spreadAreas(regions,k)};
  }
  function allocate(selected,n,mode,design,remote) {
    const parts=selected.map(r=>{
      const unit=1+r.remote*remote;
      const score=design==='pop'?r.population:design==='spend'?r.spend:
        design==='optimal'?r.weight*r.deviation/Math.sqrt(unit):1;
      return {r,unit,setup:2+2*r.remote,score,n:0};
    });
    let cost=0,count=0;
    function add(p) {cost+=p.unit+(p.n===0?p.setup:0);p.n++;count++;}
    for (const p of parts) if(mode==='quotes'||cost+p.setup+p.unit<=n)add(p);
    while(count<n) {
      const feasible=parts.filter(p=>p.n<p.r.prices.length && (mode==='quotes'||
        cost+p.unit+(p.n===0?p.setup:0)<=n));
      if(!feasible.length)break;
      const priority=p=>p.score/(p.n+1);
      add(feasible.reduce((a,b)=>priority(a)>=priority(b)?a:b));
    }
    return {parts:parts.filter(p=>p.n),cost,count};
  }
  function sampleQuotes(region,n,random) {
    const indices=new Map(), quoteIndices=[];
    let total=0;
    for(let j=0;j<n;j++){
      const k=j+Math.floor(random()*(region.prices.length-j));
      const id=indices.get(k)??k;
      indices.set(k,indices.get(j)??j);
      total+=region.prices[id];quoteIndices.push(id);
    }
    return {estimate:total/n,quoteIndices};
  }
  function draw(pop,config,design,random) {
    const k=Math.min(config.areas,pop.regions.length);
    const {selected,strata}=chooseAreas(pop.regions,k,design,random);
    const {parts,cost,count}=allocate(selected,config.quotes,config.resource,design,config.remote);
    const samples=parts.map(p=>({...p,...sampleQuotes(p.r,p.n,random)}));
    const donor=mean(samples.map(s=>s.estimate));
    // With all macrostrata represented, equal-probability area selection and this
    // expansion give a design-unbiased expenditure total for the stratified method.
    const allGroups=strata && samples.length===strata.length;
    const estimate=allGroups?sum(samples.map(s=>{
      const h=strata.find(g=>g.includes(s.r));
      return h.length*s.r.weight*s.estimate;
    })):sum(pop.regions.map(r=>r.weight*(samples.find(s=>s.r===r)?.estimate??donor)));
    return {estimate,cost,count,samples,error:estimate-pop.truth,
      estimator:allGroups?'Stratified inverse-probability expansion':'Expenditure-weighted means with pooled donor for unsampled areas'};
  }
  function experiment(pop,config,runs=90) {
    return Object.keys(designs).map((design,j)=>{
      const random=rng(pop.seed+919*j+2017);
      const repeated=Array.from({length:runs},()=>draw(pop,config,design,random));
      const errors=repeated.map(x=>x.error);
      return {design,errors,bias:mean(errors),variance:sampleVariance(errors),
        rmse:Math.sqrt(mean(errors.map(e=>e**2))),cost:mean(repeated.map(s=>s.cost)),
        count:mean(repeated.map(s=>s.count))};
    });
  }
  function populationMap(pop,drawn) {
    return '<svg class="geo-map geo-sim-map" viewBox="0 0 460 260" role="img" aria-label="Synthetic regions and outlets, with sampled outlets highlighted">'+
      pop.regions.map(r=>{
        const draw=drawn.samples.find(s=>s.r===r);
        const selected=new Set(draw?.quoteIndices.map(q=>Math.floor(q/24))||[]);
        return '<g><circle cx="'+(r.x*460)+'" cy="'+(r.y*250)+'" r="21" fill="'+
          (r.id===pop.shockArea?'#f1bdac':draw?'#c1d9e8':'#e4ebef')+'" stroke="'+(draw?'#285f91':'#a8b8bf')+'"/>'+
          r.outlets.map((outlet,i)=>'<circle cx="'+(outlet.x*460)+'" cy="'+(outlet.y*250)+'" r="'+(selected.has(i)?2.5:1.3)+
            '" fill="'+(selected.has(i)?'#245987':outlet.city?'#8097a4':'#aebdc5')+'"/>').join('')+
          '<text x="'+(r.x*460)+'" y="'+(r.y*250+30)+'" text-anchor="middle">'+(r.id+1)+'</text><title>'+
          'Region '+(r.id+1)+': '+(draw?.n||0)+' sampled quotes; '+fmt(100*r.weight,1)+'% expenditure share</title></g>';
      }).join('')+'</svg>';
  }
  labs.geoSpatial=el=>{
    el.innerHTML=panel('Synthetic finite population','Does collecting more prices in more places improve inflation measurement?',
      controls('Population and sample size',4,range('Geographical regions','regions',defaults.regions,4,20)+
        range('Quote count or budget','quotes',defaults.quotes,40,300)+range('Directly sampled areas','areas',defaults.areas,1,20)+
        range('Geographical heterogeneity','heterogeneity',defaults.heterogeneity,0,6,.1))+
      controls('Spatial differences and collection',4,range('Spatial correlation','correlation',defaults.correlation,0,.95,.05)+
        range('Urban–rural difference (pp)','urban',defaults.urban,0,8,.1)+range('Remote collection cost','remote',defaults.remote,0,5,.1)+
        range('Localized shock (pp)','shock',defaults.shock,0,18,.5),'secondary')+
      '<div class="controls geo-sim-controls">'+select('Sampling strategy','design',Object.entries(designs))+
      select('Comparator','comparison',Object.entries(designs))+select('Match resources by','resource',[
        ['quotes','Equal quote count'],['budget','Equal collection budget']])+'</div>'+
      '<div class="geo-episodes"><button type="button" data-redraw>Draw again</button>'+
      '<button type="button" data-regenerate>Generate population</button><button type="button" data-sim-reset>Reset laboratory</button></div>'+
      '<p class="geo-source-line" data-population aria-live="polite"></p><div class="geo-pair">'+
      '<section class="geo-card us"><h5>Population and current draw</h5><div data-pop-map></div><p class="geo-source-line">Regions are numbered. Dark blue points are sampled outlets. Peach marks the shocked region.</p><div data-draw></div></section>'+
      '<section class="geo-card ca"><h5>Repeated-sample errors</h5><div data-distribution></div><p class="geo-source-line">Both distributions use 90 draws on the same synthetic population.</p></section></div>'+
      '<div data-comparison></div>',
      'Population controls take effect when you generate. Remote cost, sample size and geographical spread change collection on the current population.');
    $('[name=design]',el).value='strat';
    $('[name=comparison]',el).value='metro';
    const popKeys=['regions','heterogeneity','correlation','urban','shock'];
    let populationConfig={...defaults}, seed=60261, drawNo=0;
    let population=makePopulation(populationConfig,seed), cacheKey='', results, config, currentDraw;
    function render(p) {
      config={...p,areas:Math.min(p.areas,population.regions.length)};
      const field=$('[name=areas]',el);
      field.max=population.regions.length;
      field.value=config.areas;
      field.previousElementSibling.value=config.areas;
      const key=[population.seed,config.quotes,config.areas,config.remote,config.resource].join(',');
      if(key!==cacheKey){
        results=experiment(population,config);
        cacheKey=key;
      }
      currentDraw=draw(population,config,config.design,rng(population.seed+4567+drawNo*2089));
      const pending=popKeys.some(k=>p[k]!==populationConfig[k]);
      el.dataset.populationSeed=population.seed;
      el.dataset.populationTruth=population.truth;
      el.dataset.draw=drawNo;
      $('[data-population]',el).textContent='Population seed '+population.seed+' · '+population.regions.length+
        ' areas · draw '+(drawNo+1)+'. '+(pending?'Population changes are ready to generate.':'');
      $('[data-pop-map]',el).innerHTML=populationMap(population,currentDraw);
      const chosen=results.find(r=>r.design===config.design);
      $('[data-draw]',el).innerHTML=metrics([['Known target',fmt(population.truth,2)+'%'],
        ['Current estimate',fmt(currentDraw.estimate,2)+'%'],['Mean error',fmt(chosen.bias,2)+' pp'],
        ['Error variance',fmt(chosen.variance,2)+' pp²'],['RMSE',fmt(chosen.rmse,2)+' pp'],
        ['Current cost',fmt(currentDraw.cost,1)+' units'],['Collected quotes',currentDraw.count],
        ['Direct areas',currentDraw.samples.length]])+'<p class="geo-source-line">'+currentDraw.estimator+'.</p>';
      const comparison=results.find(r=>r.design===config.comparison);
      const lists=[chosen,...(chosen===comparison?[]:[comparison])].map((r,i)=>({label:designs[r.design],values:r.errors,color:studyColors[i]}));
      histogram($('[data-distribution]',el),lists,{x:'Estimate minus known target (pp)'});
      $('[data-comparison]',el).innerHTML='<div class="table-wrap">'+table(
        ['Sampling strategy','Mean quotes','Mean cost','Bias (pp)','Variance (pp²)','RMSE (pp)'],
        results.map(r=>[designs[r.design],fmt(r.count,1),fmt(r.cost,1),fmt(r.bias,2),fmt(r.variance,2),fmt(r.rmse,2)]))+'</div>';
    }
    const update=listen(el,render);
    el.addEventListener('click',event=>{
      const b=event.target.closest('button');
      if(!b)return;
      if(b.hasAttribute('data-redraw'))drawNo++;
      else if(b.hasAttribute('data-regenerate')){
        const p=picks(el);
        populationConfig=Object.fromEntries(popKeys.map(k=>[k,p[k]]));
        population=makePopulation(populationConfig,++seed);drawNo=0;
      } else if(b.hasAttribute('data-sim-reset')){
        for(const [k,v] of Object.entries(defaults))$('[name="'+k+'"]',el).value=v;
        $('[name=design]',el).value='strat';$('[name=comparison]',el).value='metro';
        $('[name=resource]',el).value='quotes';
        seed=60261;drawNo=0;populationConfig={...defaults};population=makePopulation(populationConfig,seed);
        $('[name=areas]',el).max=defaults.regions;$('[name=areas]',el).value=defaults.areas;
      } else return;
      update();
    });
  };
  return {defaults,makePopulation,draw,experiment};
})();
