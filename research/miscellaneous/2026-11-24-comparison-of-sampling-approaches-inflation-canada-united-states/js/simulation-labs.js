'use strict';
(() => {
  const {$,$$,labs,panel,table,select,range,input,controls,equation,listen,picks,plot,histogram,colors,fmt,metrics,asset,escape} = Study;
  const synthetic='Synthetic statistical simulation';
  const E=CPISampling;
  const seed=20261124;
  const designs=E.designs;
  const scenarioNames=Object.keys(CPI_SIMULATED.scenarios);
  const populationMap=(element,pop,sample=null)=>{
    const picked=new Set(sample?.selected.map(r=>r.id) || []);
    const marks=[
      (x,y,r)=>'<circle cx="'+x+'" cy="'+y+'" r="'+r+'"/>',
      (x,y,r)=>'<rect x="'+(x-r)+'" y="'+(y-r)+'" width="'+(2*r)+'" height="'+(2*r)+'" rx=".6"/>',
      (x,y,r)=>'<path d="M'+x+' '+(y-r)+'l'+r+' '+r+'-'+r+' '+r+'-'+r+' -'+r+'z"/>',
      (x,y,r)=>'<path d="M'+x+' '+(y-r)+'l'+r+' '+(2*r)+'h-'+(2*r)+'z"/>'
    ];
    let shape='';
    for(let k=0;k<8;k++){
      const x=13+k%4*158,y=14+Math.floor(k/4)*71;
      shape+='<rect x="'+x+'" y="'+y+'" width="146" height="62" rx="6" fill="#edf3f1"/>'+
        '<text x="'+(x+7)+'" y="'+(y+13)+'" fill="#49615d" font-size="12">Region '+(k+1)+'</text>';
      pop.frames[k].forEach((row,i)=>{
        const cx=x+9+(i%15)*9.1,cy=y+23+Math.floor(i/15)*7;
        const active=sample?picked.has(row.id):row.accessible;
        const color=colors[active?row.type:7];
        shape+='<g fill="'+color+'" opacity="'+(active?1:.4)+'">'+marks[row.type](cx,cy,active?3.25:2.1)+
          '<title>Outlet '+row.id+' · '+E.types[row.type]+' · '+(row.urban?'urban':'rural')+
          ' · Change '+fmt(row.truth,2)+'% · spend '+fmt(row.weight,2)+'</title></g>';
      });
    }
    const title=sample?'Eight synthetic regions with selected outlets highlighted':'Eight synthetic regions with eligible outlets highlighted';
    element.innerHTML='<svg viewBox="0 0 650 155" width="100%" role="img" aria-label="'+title+'">'+shape+
      '</svg><div class="plot-legend">'+E.types.map((t,i)=>'<span style="--key:'+colors[i]+'">'+['○','□','◇','△'][i]+' '+t+'</span>').join('')+
      '<span style="--key:'+colors[7]+'">'+(sample?'Unselected':'Outside frame')+'</span></div>';
  };
  const descriptions={
    SRS:['Simple random sampling','Select equally likely subsets from the accessible outlet frame.'],
    Stratified:['Stratified random sampling','Divide the frame into regions and randomly select outlets within each.'],
    PPS:['Probability proportional to size','Draw with replacement using expenditure-proportional draw probabilities.'],
    'Multistage PPS':['Multistage sampling','Draw regions with replacement using expenditure shares, then sample outlets within each region.'],
    Judgmental:['Judgmental selection','Prioritize outlets using a size-related score. The illustration provides no design inclusion probabilities.'],
    'Cut-off':['Cut-off selection','Collect the largest outlets by expenditure in the accessible frame.'],
    Calibrated:['Calibrated judgmental selection','Reweight observed chain and independent outlets to the corresponding expenditure totals.'],
    Hybrid:['Hybrid selection','Combine merchant scanner records with a targeted supplemental sample.']
  };
  function targetDisplay(el,results){
    const cases=[[0,.25],[0,1.0],[.75,.25],[.75,1.0]];
    const labels=['Low bias, low variance','Low bias, high variance','High bias, low variance','High bias, high variance'];
    const base=results[0], src=base.errors;
    const width=460, scale=28;
    const sd=Math.sqrt(base.variance)||1;
    const scaled=cases.map(([bias,spread],j)=>{
      const cx=55+j*115, cy=52;
      const points=src.slice(0,22).map((e,i)=>{
        const dx=bias+spread*(e-base.bias)/sd*.5;
        const dy=spread*(src[(i+7)%src.length]-base.bias)/sd*.6;
        return '<circle cx="'+(cx+dx*scale)+'" cy="'+(cy+dy*scale)+'" r="2" fill="'+colors[j]+'" opacity=".73"/>';
      }).join('');
      return '<g>'+[11,23,35].map(r=>'<circle cx="'+cx+'" cy="'+cy+'" r="'+r+'" fill="none" stroke="#cbdad9"/>').join('')+
        '<path d="M'+(cx-5)+','+cy+'h10m-5,-5v10" stroke="#546d68"/>'+points+
        '<text x="'+cx+'" y="105" text-anchor="middle" font-size="9.3" fill="#44575b">'+labels[j]+'</text></g>';
    }).join('');
    el.innerHTML='<svg viewBox="0 0 '+width+' 112" width="100%" role="img" aria-label="Four target plots contrasting high and low bias and variance">'+scaled+
      '</svg><p class="source">Centered draws from the '+base.design+' experiment, rescaled and shifted to isolate spread and bias. The center cross is the true target.</p>';
  }
  labs.sampling=el=>{
    const options=designs.filter(d=>d!=='Hybrid').map(d=>[d,descriptions[d][0]]);
    const defaults={N:160,n:36,association:.35,geography:.8,judgment:.55,rho:.25,coverage:1,design:'Stratified'};
    el.innerHTML=panel(synthetic,'Randomization laboratory','<div class="design-picker">'+
      select('Sampling design','design',options)+'<p class="note" data-description></p></div>'+
      controls('Primary sampling parameters',3,range('Outlets','N',defaults.N,48,320,8)+range('Sample size','n',defaults.n,16,120,4)+
      range('Size–price association','association',defaults.association,-1,1,.05))+
      controls('Population and selection parameters',4,range('Geographic heterogeneity','geography',defaults.geography,0,3,.05)+
      range('Judgmental preference','judgment',defaults.judgment,0,1,.05)+range('Cluster dependence','rho',defaults.rho,0,.95,.05)+
      range('Small-outlet coverage','coverage',defaults.coverage,0,1,.05),'secondary')+
      '<div class="controls control-actions"><button type="button" data-draw>Draw again</button><button type="button" data-play aria-pressed="false">Animate draws</button>'+
      '<button type="button" data-reset>Reset parameters</button></div><div data-flow></div><div data-out></div><div data-map></div><div data-unit></div><div data-dist></div>',
      'The target uses outlet expenditure shares. The flow counts distinct outlets at each stage. Probabilities apply to the accessible frame under the stated design.');
    $('[name=design]',el).value=defaults.design;
    let iteration=0, timer;
    const update=listen(el,p=>{
      const config={...p,n:Math.min(p.n,p.N),seed,runs:100};
      const {pop,results}=E.run(config,[p.design]);
      const draw=E.draw(pop,p.design,config.n,seed+1+iteration,p.judgment);
      const expansion=draw.weights.filter(Number.isFinite);
      const distinct=new Set(draw.selected).size;
      $('[data-description]',el).textContent=descriptions[p.design][1];
      $('[data-flow]',el).innerHTML='<div class="flow"><span>Population U · '+pop.rows.length+'</span><i>→</i><span>Frame · '+pop.frame.length+
        '</span><i>→</i><span>Distinct sample s · '+distinct+'</span><i>→</i><span>Weighted estimate</span></div>';
      $('[data-out]',el).innerHTML=metrics([['Population change',fmt(pop.truth,2)+'%'],['Sample estimate',fmt(draw.estimate,2)+'%'],
        ['Error',fmt(draw.estimate-pop.truth,2)+' pp'],['Expansion weights',expansion.length?fmt(Math.min(...expansion),2)+'–'+fmt(Math.max(...expansion),2):'Unavailable']]);
      populationMap($('[data-map]',el),pop,draw);
      const item=draw.selected[0], weight=draw.weights[0], fixed=['SRS','Stratified'].includes(p.design);
      const probability=fixed?'inclusion probability πᵢ':p.design==='PPS'?'single-draw probability':p.design==='Multistage PPS'?'per-draw joint probability':'inclusion probability';
      $('[data-unit]',el).innerHTML='<p class="note"><strong>First observed outlet</strong> '+E.types[item.type]+' · price change yᵢ '+fmt(item.observed,2)+
        '% · expenditure share wᵢ '+fmt(100*item.share,2)+'% · '+probability+' '+(Number.isFinite(weight)?fmt(1/weight,4):'unavailable')+
        ' · '+(fixed?'design weight dᵢ ':'expansion factor ')+(Number.isFinite(weight)?fmt(weight,2):'unavailable')+'</p>';
      histogram($('[data-dist]',el),[{label:p.design+' simulated errors',values:results[0].errors}]);
    });
    const play=$('[data-play]',el);
    function stop(){
      clearInterval(timer);timer=null;
      play.setAttribute('aria-pressed','false');play.textContent='Animate draws';
    }
    $('[data-draw]',el).onclick=()=>{iteration++;update();};
    play.onclick=()=>{
      if(timer){stop();return;}
      timer=setInterval(()=>{iteration++;update();},1000);
      play.setAttribute('aria-pressed','true');play.textContent='Pause draws';
    };
    $('[data-reset]',el).onclick=()=>{
      stop();iteration=0;
      Object.entries(defaults).forEach(([key,value])=>{$('[name='+key+']',el).value=value;});
      update();
    };
  };
  labs.uncertainty=el=>{
    el.innerHTML=panel(synthetic,'Change the assumptions behind an interval',
      controls('Primary sampling parameters',3,range('Sample observations','n',100,20,400,5)+range('Number of clusters','clusters',20,4,40)+
      range('Intracluster correlation','rho',.2,0,.9,.02))+
      controls('Additional uncertainty parameters',4,range('Price-change SD (pp)','vol',3,.5,12,.1)+range('Weight coefficient of variation','cv',.4,0,2,.05)+
      range('Nonresponse (%)','missing',12,0,75)+range('Selection shift (pp)','shift',.7,-4,4,.1),'secondary')+
      '<div class="eq-strip">'+equation(String.raw`DEFF_c\approx 1+(\overline m-1)\rho`)+
      equation(String.raw`DEFF_w\approx 1+CV(w)^2`)+equation(String.raw`SE\approx\sigma\sqrt{DEFF_c DEFF_w/n_{\rm resp}}`,'green')+
      equation(String.raw`MSE=\operatorname{Var}+\operatorname{Bias}^2`,'rose')+'</div><div data-out></div><div data-chart></div>',
      'The cluster approximation assumes equal responding cluster sizes and common within-cluster correlation. The weight effect assumes independent observations with similar variances. Multiplying these effects is an illustrative approximation. The synthetic shift leaves variance fixed and illustrates how selection can change coverage. Neither curve represents an agency CPI.');
    listen(el,p=>{
      const effective=p.n*(1-p.missing/100), groups=Math.min(effective,p.clusters), m=effective/groups;
      const deff=1+(m-1)*p.rho, kish=1+p.cv**2;
      const se=p.vol*Math.sqrt(deff*kish/effective);
      const random=E.rng(seed), R=400;
      const design=Array.from({length:R},()=>se*random.normal());
      const model=design.map(e=>e+p.shift);
      const coverage=xs=>Study.mean(xs.map(x=>Math.abs(x)<=1.96*se?1:0));
      const nominal='95.0%';
      $('[data-out]',el).innerHTML=metrics([['Cluster design effect',fmt(deff,2)],['Weight effect',fmt(kish,2)],['Design SD',fmt(se,3)+' pp'],
        ['95% interval half-width',fmt(1.96*se,3)+' pp'],['Model MSE',fmt(se*se+p.shift**2,3)+' pp²']])+
        table(['Coverage of target','Nominal','Empirical'],[
          ['Centered design',nominal,fmt(100*coverage(design),1)+'%'],['Shifted model',nominal,fmt(100*coverage(model),1)+'%']
        ])+'<p class="note">Model MSE = '+fmt(se*se,3)+' pp² variance + '+fmt(p.shift**2,3)+' pp² squared bias.</p>';
      const references=[{value:0,label:'Target and centered expectation',color:'#293c44'}];
      if(p.shift!==0)references.push({value:p.shift,label:'Shifted expectation',color:colors[3],dash:'6 3'});
      histogram($('[data-chart]',el),[{label:'Design, centered',values:design,color:colors[2]},{label:'Model, shifted',values:model,color:colors[3]}],{references});
    });
  };
  labs.simulation=el=>{
    el.innerHTML=panel(synthetic,'Compare estimators on a fixed population',
      '<div class="controls">'+select('Estimator for the sample and distribution','design',designs.map(d=>[d,descriptions[d][0]]))+
      select('Missing quote','imputation',[['carry','Carry forward'],['donor','Donor mean by outlet type']])+'</div>'+
      controls('Sampling experiment',3,range('Sample observations','n',56,16,120,8)+range('Replications','runs',200,50,900,50)+
        range('Draw seed offset','sampleOffset',0,0,100))+
      '<details class="derivation" data-pop-settings><summary>Population generation settings</summary><div><div class="controls">'+
      select('Stress scenario','scenario',scenarioNames)+'<label>Population seed '+input('Population seed','seed',seed,0,2147483647,1)+'</label></div>'+
      controls('Population structure',3,range('Outlets','N',160,64,320,8)+range('Region dependence','rho',.25,0,.95,.05)+
      range('Size–price association','association',.25,-1,1,.05))+
      controls('Population disturbances',3,range('Geographic heterogeneity','geography',.7,0,2,.1)+range('Nonresponse probability','loss',0,0,.55,.05),'secondary')+
      '</div></details><div class="controls control-actions"><button type="button" data-generate>Generate population</button>'+
      '<button type="button" data-draw>Draw sample</button><button type="button" data-run>Repeat experiment</button>'+
      '<button type="button" data-reset>Reset seeds</button></div><p class="note" data-state role="status"></p><div data-summary></div>'+
      '<div class="pairs experiment-pair"><div class="panel green"><h4>Fixed finite population</h4><div data-portfolio></div></div>'+
      '<div class="panel gray"><h4>Selected observations</h4><div data-map></div></div>'+
      '<div class="panel lav"><h4>Sampling distribution</h4><div data-dist></div></div>'+
      '<div class="panel"><h4>Estimator performance</h4><div data-bias></div></div></div>'+
      '<details class="derivation"><summary>Coverage, Monte Carlo uncertainty and cost</summary><div>'+
      '<h4>Coverage and collection cost</h4><div data-cost></div><h4>Cost and accuracy frontier</h4><div data-frontier></div>'+
      '<h4>Bias and dispersion</h4><div data-target></div></div></details>',
      'Generating replaces the synthetic population. Sampling controls and imputation operate on the current population. Repeating uses the selected seed offset and compares all designs against the same fixed target. Precomputed Python simulations remain available in the stress matrix.');
    const status=$('[data-state]',el), runButton=$('[data-run]',el), design=$('[name=design]',el);
    design.value='Stratified';
    let fixed, applied, currentSample, currentRuns, drawIndex=0, worker, fallbackTimer, task=0;
    const values=()=>picks(el);
    const populationSettings=p=>Object.fromEntries(['N','seed','rho','association','geography','loss','scenario'].map(key=>[key,p[key]]));
    const config=p=>({...p,n:Math.min(p.n,fixed.rows.length),seed:applied.seed+p.sampleOffset,judgment:.6});
    const matches=p=>currentRuns&&['n','runs','imputation','sampleOffset'].every(key=>p[key]===currentRuns.settings[key]);
    function renderSample(p){
      const state=E.withImputation(fixed,p.imputation);
      const c=config(p), distinctRun=applied.seed+44000+drawIndex+c.sampleOffset;
      currentSample=E.draw(state,p.design,c.n,distinctRun,c.judgment);
      populationMap($('[data-map]',el),fixed,currentSample);
      $('[data-summary]',el).innerHTML=metrics([['Population seed and size',applied.seed+' · '+fixed.rows.length],['Fixed target',fmt(fixed.truth,3)+'%'],
        ['Current draw estimate',fmt(currentSample.estimate,3)+'%'],['Current error',fmt(currentSample.estimate-fixed.truth,3)+' pp']]);
      $('[data-summary]',el).dataset.population=String(applied.seed)+':'+fixed.rows.length+':'+fixed.truth;
    }
    function renderResults(p) {
      const available=matches(p);
      if(!available){
        for(const name of ['dist','bias','cost','frontier','target']){
          const slot=$('[data-'+name+']',el);
          slot.innerHTML='<p class="note">Repeat the experiment for these sampling settings.</p>';
          Study.clearPlot(slot);
        }
        return;
      }
      const {results}=currentRuns.result;
      const selected=results.find(x=>x.design===p.design);
      histogram($('[data-dist]',el),[{label:descriptions[p.design][0],values:selected.errors,color:colors[3]}]);
      $('[data-bias]',el).innerHTML=table(['Design','Bias (pp)','Variance (pp²)','RMSE (pp)'],results.map(r=>
        [r.design,fmt(r.bias,3),fmt(r.variance,3),fmt(r.rmse,3)]));
      $('[data-cost]',el).innerHTML=table(['Design','Coverage','Bias MCSE (pp)','Cost units'],results.map(r=>
        [r.design,fmt(r.coverage*100,1)+'%',fmt(r.biasMCSE,3),fmt(r.cost,1)]));
      targetDisplay($('[data-target]',el),[selected]);
      const costs=[...new Set(results.map(r=>r.cost))].sort((a,b)=>a-b);
      const onFrontier=results.filter(a=>!results.some(b=>b.cost<=a.cost&&b.rmse<=a.rmse&&(b.cost<a.cost||b.rmse<a.rmse)));
      plot($('[data-frontier]',el),costs,[
        ...results.map(r=>({label:r.design,values:costs.map(c=>c===r.cost?r.rmse:null),scatter:true})),
        {label:'Nondominated frontier',values:costs.map(c=>onFrontier.find(r=>r.cost===c)?.rmse??null),color:'#233b40'}
      ],{title:'Hypothetical collection cost and RMSE',numericX:true,x:'Cost units',y:'RMSE (pp)',precision:2,zero:true});
    }
    function halt(){
      task++;
      if(worker){worker.terminate();worker=null;}
      clearTimeout(fallbackTimer);
      runButton.disabled=false;
    }
    function generate(p){
      halt();
      applied=populationSettings(p);
      fixed=E.population(applied);
      currentRuns=null;
      drawIndex=0;
      populationMap($('[data-portfolio]',el),fixed);
      renderSample(p);
      renderResults(p);
      status.textContent='Population generated · '+applied.scenario;
    }
    function repeat(p){
      halt();
      const request=task;
      const payload={population:E.withImputation(fixed,p.imputation),config:config(p)};
      runButton.disabled=true;
      status.textContent='Running experiment on the current population…';
      function done(result,error){
        if(request!==task)return;
        runButton.disabled=false;
        if(error){status.textContent=error;return;}
        currentRuns={settings:p,result};
        const now=values();
        renderResults(now);
        status.textContent=matches(now)?'Finished · '+p.runs+' repetitions per design':'Sampling settings changed · repeat to refresh the results.';
      }
      function fallback(){
        fallbackTimer=setTimeout(()=>{
          try{done(E.runPopulation(payload.population,payload.config));}catch(error){done(null,error.message);}
        },30);
      }
      if(location.protocol==='file:'){fallback();return;}
      try {
        worker=new Worker(asset('js/simulation-worker.js'));
        worker.onmessage=e=>{
          if(request!==task)return;
          worker.terminate();worker=null;
          done(e.data.result,e.data.error);
        };
        worker.onerror=e=>{
          e.preventDefault();
          if(request!==task)return;
          worker.terminate();worker=null;fallback();
        };
        worker.postMessage(payload);
      } catch {fallback();}
    }
    generate(values());
    const defaultSettings=values();
    currentRuns={settings:defaultSettings,result:E.runPopulation(fixed,config(defaultSettings))};
    renderResults(defaultSettings);
    status.textContent='Baseline · '+defaultSettings.runs+' repetitions per design';
    listen(el,p=>{
      renderSample(p);
      renderResults(p);
      if(Object.entries(populationSettings(p)).some(([key,value])=>value!==applied[key])){
        status.textContent='Population settings edited · generate to apply changes.';
      }else if(!matches(p))status.textContent='Sampling settings edited · repeat to refresh the results.';
      else if(!runButton.disabled)status.textContent='Current population · '+currentRuns.settings.runs+' repetitions per design';
    });
    $('[data-generate]',el).onclick=()=>generate(values());
    $('[data-draw]',el).onclick=()=>{drawIndex++;renderSample(values());};
    runButton.onclick=()=>repeat(values());
    $('[data-reset]',el).onclick=()=>{
      $('[name=seed]',el).value=seed;
      const offset=$('[name=sampleOffset]',el);
      offset.value=0;drawIndex=0;
      offset.dispatchEvent(new Event('input',{bubbles:true}));
      status.textContent='Seeds reset for subsequent draws and population generation. Current population retained.';
    };
  };
  labs.stress=el=>{
    const bundle=CPI_SIMULATED, designNames=[...new Set(bundle.rows.map(r=>r.design))];
    el.innerHTML=panel(synthetic,'Compare performance across fixed stress scenarios','<div class="controls">'+
      select('Metric','metric',[['rmse','RMSE'],['bias','Bias'],['variance','Variance'],['coverage','Coverage'],['width','Interval width'],['cost','Cost']])+
      select('Design','design',designNames)+'</div><div data-heat></div><div data-detail></div>',
      'Python: '+bundle.runs.toLocaleString()+' repetitions per cell, 56 nominal draws, seed '+bundle.seed+'. Darker shade means larger displayed metric magnitude, or lower coverage.');
    listen(el,p=>{
      const metric=p.metric, all=bundle.rows.map(r=>metric==='coverage'?1-r[metric]:Math.abs(r[metric]));
      const max=Math.max(...all)||1;
      const headings=designNames.map(d=>'<th scope="col">'+d.replace('Multistage PPS','Multi-stage')+'</th>').join('');
      $('[data-heat]',el).innerHTML='<table class="heat"><thead><tr><th scope="col">Scenario</th>'+headings+'</tr></thead><tbody>'+
        scenarioNames.map(s=>'<tr><th scope="row">'+s+'</th>'+designNames.map(d=>{
          const row=bundle.rows.find(r=>r.scenario===s&&r.design===d), val=row[metric], alpha=.12+.64*(metric==='coverage'?1-val:Math.abs(val))/max;
          return '<td style="background:rgba(164,104,137,'+alpha.toFixed(3)+')" title="'+escape(s+' · '+d)+'">'+fmt(metric==='coverage'?100*val:val,metric==='coverage'?0:2)+(metric==='coverage'?'%':'')+'</td>';
        }).join('')+'</tr>').join('')+'</tbody></table>';
      const selected=bundle.rows.filter(r=>r.design===p.design);
      $('[data-detail]',el).innerHTML='<h4>'+p.design+' Monte Carlo diagnostics</h4>'+
        table(['Scenario','Bias MCSE (pp)','Coverage MCSE','95% error range (pp)'],selected.map(r=>[r.scenario,fmt(r.bias_mcse,3),fmt(100*r.coverage_mcse,1)+'%',fmt(r.q025,2)+' to '+fmt(r.q975,2)]));
    });
  };
})();

