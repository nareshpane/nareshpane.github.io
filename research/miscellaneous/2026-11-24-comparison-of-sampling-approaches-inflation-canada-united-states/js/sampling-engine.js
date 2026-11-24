'use strict';
globalThis.CPISampling = (() => {
  const designs = ['SRS','Stratified','PPS','Multistage PPS','Judgmental','Cut-off','Calibrated','Hybrid'];
  const types = ['Chain market','Independent','Discounters','Online'];
  const sum = values => values.reduce((a,b)=>a+b,0);
  const mean = values => sum(values)/values.length;
  function sampleVariance(values) {
    if (values.length<2) return 0;
    const average=mean(values);
    return sum(values.map(x=>(x-average)**2))/(values.length-1);
  }
  function rng(seed) {
    let state=seed>>>0;
    const next = () => {
      state=(state+0x6D2B79F5)>>>0;
      let t=Math.imul(state^(state>>>15),1|state);
      t^=t+Math.imul(t^(t>>>7),61|t);
      return ((t^(t>>>14))>>>0)/4294967296;
    };
    next.normal = () => Math.sqrt(-2*Math.log(Math.max(1e-12,next())))*Math.cos(2*Math.PI*next());
    return next;
  }
  function choose(random, weights, values) {
    let u=random()*sum(weights);
    for(let i=0;i<weights.length;i++) {
      u-=weights[i];
      if(u<0) return values[i];
    }
    return values.at(-1);
  }
  function srs(random, rows, n) {
    const copy=rows.slice();
    for(let i=0;i<n;i++) {
      const j=i+Math.floor(random()*(copy.length-i));
      [copy[i],copy[j]]=[copy[j],copy[i]];
    }
    return copy.slice(0,n);
  }
  function population(config) {
    const {N, seed, rho=0.25, association=.35, geography=1, loss=0, scenario='Baseline', coverage=1, imputation='carry'}=config;
    const random=rng(seed), regional=Array.from({length:8},()=>random.normal()), key=scenario==='Baseline'?'':scenario[0];
    const rows=Array.from({length:N},(_,id)=>{
      const region=id%8, urban=random()<.76, type=Math.floor(random()*types.length);
      const weight=Math.exp((type===0?1.25:0)+ (urban?.35:0)+random.normal()*.72);
      const chain=type===0 || type===2;
      const quality=.7+random()*.8;
      const noise=2*(Math.sqrt(rho)*regional[region]+Math.sqrt(1-rho)*random.normal());
      let truth=1.5+noise+geography*regional[region]+association*Math.log(weight)*3;
      if(key==='A') truth+=3+3*(type===0);
      if(key==='B') truth+=3*Math.log(weight);
      if(key==='C' && region===6) truth+=15;
      if(key==='F') truth+=2*Math.log(weight);
      if(key==='G' && !chain) truth+=7;
      if(key==='H' && [0,3,5].includes(region)) truth+=5;
      const refused=random()<Math.min(.95,loss+(key==='D' && truth>3?.73:0)+(key==='H' && [0,3,5].includes(region)?.8:0));
      const adjustment=(key==='E' && random()<.35) ? 6.5*quality:0;
      const observed=refused ? 0 : truth+adjustment;
      const accessible=weight>1.3 || random()<coverage;
      return {id,region,urban,type,chain,weight,truth,observed,refused,accessible};
    });
    const trueWeights=rows.map(r=>r.weight*(key==='F'?Math.max(.3,1+.12*r.truth):1));
    const truth=sum(rows.map((r,i)=>r.truth*trueWeights[i]))/sum(trueWeights);
    return withImputation({rows,truth},imputation);
  }
  function withImputation(pop, method) {
    const rows=pop.rows.map(row=>({...row, observed:row.refused ? 0 : row.observed}));
    const total=sum(rows.map(r=>r.weight));
    if(method==='donor'){
      const donors=rows.filter(r=>!r.refused && r.accessible);
      const means=types.map((_,i)=>{
        const typed=donors.filter(r=>r.type===i);
        const usable=typed.length?typed:donors;
        return usable.length?mean(usable.map(r=>r.observed)):0;
      });
      rows.filter(r=>r.refused).forEach(row=>{row.observed=means[row.type];});
    }
    rows.forEach(row=>{row.share=row.weight/total;row.contribution=row.share*row.observed;});
    const frames=Array.from({length:8},(_,region)=>rows.filter(r=>r.region===region));
    const frame=rows.filter(r=>r.accessible);
    const groups=frames.map(r=>r.filter(x=>x.accessible)).filter(r=>r.length);
    return {rows,frames,frame,groups,truth:pop.truth};
  }
  function draw(pop, design, size, seed, judgment=.6) {
    const random=typeof seed==='function'?seed:rng(seed), {frame,groups}=pop, N=frame.length;
    const n=Math.min(size,N), selected=[], weights=[];
    const record = (row, weight) => {selected.push(row);weights.push(weight);};
    const weighted = sample => sum(sample.map(r=>r.weight*r.observed))/sum(sample.map(r=>r.weight));
    const work = sample => sampleVariance(sample.map(r=>r.observed))/sample.length;
    const targeted = frame => frame.map(r=>[r,(1-judgment)*random()+judgment*Math.log1p(r.weight)+random()*.2])
      .sort((a,b)=>b[1]-a[1]).map(r=>r[0]);
    let estimate, samplingVariance, cost=n;
    if(design==='SRS') {
      srs(random,frame,n).forEach(r=>record(r,N/n));
      const z=selected.map(r=>N*r.contribution);
      estimate=mean(z);
      samplingVariance=(1-n/N)*sampleVariance(z)/n;
    } else if(design==='Stratified') {
      estimate=0; samplingVariance=0;
      const allocations=groups.map(group=>Math.min(group.length,Math.floor(n/groups.length)));
      let left=n-sum(allocations);
      while(left>0) {
        groups.forEach((group,j)=>{
          if(left && allocations[j]<group.length) {allocations[j]++;left--;}
        });
      }
      groups.forEach((group,j)=>{
        const m=allocations[j];
        if(!m) return;
        const sample=srs(random,group,m), z=sample.map(r=>group.length*r.contribution);
        sample.forEach(r=>record(r,group.length/m));
        estimate+=mean(z);
        samplingVariance+=(1-m/group.length)*sampleVariance(z)/m;
      });
    } else if(design==='Multistage PPS' || design==='PPS') {
      const probabilities=groups.map(group=>sum(group.map(r=>r.share)));
      const frameMass=sum(probabilities);
      const effectiveDraws=design==='PPS'?n:Math.floor(n/2);
      const draws=Array.from({length:effectiveDraws},()=>{
        if(design==='PPS') {
          const row=choose(random,frame.map(r=>r.share),frame);
          record(row,frameMass/row.share);
          return frameMass*row.observed;
        }
        const group=choose(random,probabilities,groups);
        const p=probabilities[groups.indexOf(group)]/frameMass;
        const sample=srs(random,group,Math.min(2,group.length));
        sample.forEach(r=>record(r,group.length/(sample.length*p)));
        return group.length*mean(sample.map(r=>r.contribution))/p;
      });
      cost=design==='PPS'?n:2.7*effectiveDraws;
      estimate=mean(draws);
      samplingVariance=sampleVariance(draws)/effectiveDraws;
    } else if(design==='Hybrid') {
      const feed=frame.filter(r=>r.chain && r.weight>2.5), used=new Set(feed);
      const rest=frame.filter(r=>!used.has(r)), supplemental=targeted(rest).slice(0,Math.floor(n/2));
      const covered=sum(feed.map(r=>r.share));
      feed.forEach(r=>record(r,1));
      supplemental.forEach(r=>record(r,NaN));
      estimate=sum(feed.map(r=>r.contribution))+(supplemental.length?(1-covered)*weighted(supplemental):0);
      samplingVariance=supplemental.length?(1-covered)**2*work(supplemental):0;
      cost=9+supplemental.length+feed.length*.12;
    } else {
      const sample=(design==='Cut-off'?frame.slice().sort((a,b)=>b.weight-a.weight):targeted(frame)).slice(0,n);
      sample.forEach(r=>record(r,NaN));
      if(design==='Calibrated') {
        estimate=0; samplingVariance=0;
        for(const chain of [true,false]) {
          const mass=sum(pop.rows.filter(r=>r.chain===chain).map(r=>r.share));
          const domain=sample.filter(r=>r.chain===chain);
          const usable=domain.length?domain:sample;
          estimate+=mass*weighted(usable);
          samplingVariance+=mass**2*work(usable);
        }
      } else {
        estimate=weighted(sample);
        samplingVariance=work(sample);
      }
    }
    return {estimate,se:Math.sqrt(samplingVariance),selected,weights,cost};
  }
  function run(config, selectedDesigns=designs) {
    return runPopulation(population(config),config,selectedDesigns);
  }
  function runPopulation(pop, config, selectedDesigns=designs) {
    const results=selectedDesigns.map((design,index)=>{
      const random=rng(config.seed+1357+index*991);
      const samples=Array.from({length:config.runs},()=>draw(pop,design,config.n,random,config.judgment));
      const errors=samples.map(s=>s.estimate-pop.truth);
      const bias=mean(errors), squared=errors.map(e=>e*e), v=mean(errors.map(e=>(e-bias)**2));
      const coverage=mean(samples.map(s=>Math.abs(s.estimate-pop.truth)<=1.96*s.se?1:0));
      return {design,bias,variance:v,rmse:Math.sqrt(mean(squared)),coverage,
        width:mean(samples.map(s=>3.92*s.se)),cost:mean(samples.map(s=>s.cost)),errors,
        biasMCSE:Math.sqrt(sampleVariance(errors)/errors.length),coverageMCSE:Math.sqrt(coverage*(1-coverage)/errors.length),
        sample:samples[0]};
    });
    return {pop,results};
  }
  return {designs,types,rng,population,withImputation,draw,run,runPopulation,sampleVariance};
})();

