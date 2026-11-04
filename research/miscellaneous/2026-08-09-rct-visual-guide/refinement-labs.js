/* Extend the preserved labs. Assignment seeds and original reroll buttons stay shared. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id),M=window.RCTMath;
  let history=[];
  function balanceAt(n,seedValue){
    const ids=Array.from({length:n},(_,i)=>i),next=random(seedValue);
    for(let i=n-1;i>0;i--){const j=Math.floor(next()*(i+1));[ids[i],ids[j]]=[ids[j],ids[i]];}
    const treated=new Set(ids.slice(0,n/2));
    return [['Age, years','age'],['Baseline migraine days','days'],['Prior preventive use','prior']].map(([name,key])=>{const t=[],c=[];baseline.slice(0,n).forEach((v,i)=>(treated.has(i)?t:c).push(v[key]));const mt=mean(t),mc=mean(c),sp=Math.sqrt((sd(t)**2+sd(c)**2)/2);return{name,key,t:mt,c:mc,smd:(mt-mc)/sp};});
  }
  function distribution(){const bins=Array(25).fill(0),limit=Math.max(.6,Math.ceil(Math.max(...history.map(Math.abs),0)*5)/5),w=680/25;
    history.forEach(v=>bins[Math.min(24,Math.max(0,Math.floor((v+limit)/(2*limit)*25)))]++);
    const max=Math.max(1,...bins),x=v=>60+(v+limit)/(2*limit)*680;
    $('balanceDistribution').innerHTML=`<title>Age SMD over ${history.length} synthetic randomizations</title><text x="60" y="24" font-size="17">Age imbalance across randomizations · frequency</text>${bins.map((n,i)=>`<rect x="${60+i*w}" y="${190-n/max*140}" width="${w-2}" height="${n/max*140}" fill="#a9cbb5"/>`).join('')}<line x1="60" x2="740" y1="190" y2="190" stroke="#76877c"/><line x1="400" x2="400" y1="37" y2="190" stroke="#406c59" stroke-dasharray="4 4"/>${[-limit,0,limit].map(v=>`<text x="${x(v)}" y="216" font-size="15" text-anchor="middle">${v.toFixed(1)}</text>`).join('')}<text x="400" y="237" font-size="14" text-anchor="middle">Age SMD (treatment − control)</text>`;
    const m=mean(history),rms=Math.sqrt(mean(history.map(v=>v*v)));
    $('balanceDistributionStatus').textContent=`${history.length} allocations at N=${$('balanceN').value}; mean age SMD ${m.toFixed(3)}, RMS imbalance ${rms.toFixed(3)}. Complete 1:1 randomization of fixed synthetic migraine covariates. Changing N starts a fresh distribution; small N generally produces wider imbalance, but a single reroll need not.`;
  }
  drawBalance=function(){const n=+$('balanceN').value,r=balanceAt(n,balanceSeed);history.push(r[0].smd);$('balanceNOut').textContent=n;
    $('balanceRows').innerHTML=r.map(v=>`<tr><th scope="row">${v.name}</th><td>${(v.t*(v.key==='prior'?100:1)).toFixed(1)}${v.key==='prior'?'%':''}</td><td>${(v.c*(v.key==='prior'?100:1)).toFixed(1)}${v.key==='prior'?'%':''}</td><td>${v.smd.toFixed(3)}</td></tr>`).join('');
    const limit=Math.max(.5,Math.ceil(Math.max(...r.map(v=>Math.abs(v.smd)))*5)/5),x=v=>310+(v+limit)/(2*limit)*330;
    $('smdPlot').innerHTML=`<title>Baseline SMD for the current assignment</title><line x1="${x(0)}" x2="${x(0)}" y1="15" y2="145" stroke="#698878" stroke-dasharray="4 4"/>${r.map((v,i)=>{const y=35+i*43;return `<text x="12" y="${y+5}" font-size="15">${v.name}</text><line x1="310" x2="640" y1="${y}" y2="${y}" stroke="#dbe2dd"/><circle cx="${x(v.smd)}" cy="${y}" r="6" fill="#487c9c"/><text x="670" y="${y+5}" font-size="14">${v.smd.toFixed(3)}</text>`;}).join('')}${[-limit,0,limit].map(v=>`<text x="${x(v)}" y="168" font-size="13" text-anchor="middle">${v.toFixed(1)}</text>`).join('')}`;
    $('balanceSeed').textContent=`seed ${balanceSeed} · ${n/2} per arm`;distribution();
  };
  $('balanceN').addEventListener('input',()=>{history=[];drawBalance();});
  $('balanceRepeat').addEventListener('click',()=>{const n=+$('balanceN').value;for(let i=0;i<199;i++){balanceSeed=(balanceSeed+3571)>>>0;history.push(balanceAt(n,balanceSeed)[0].smd);}balanceSeed=(balanceSeed+3571)>>>0;drawBalance();});
  $('balanceReset').addEventListener('click',()=>{$('balanceN').value=540;balanceSeed=81421;history=[];drawBalance();});drawBalance();

  drawSampling=function(){const n=+$('coverageN').value,level=+$('coverageLevel').value,z=M.quantile((1+level)/2),se=.055*Math.sqrt(270/n),truth=.12,next=random(sampleSeed);
    const intervals=Array.from({length:60},()=>{const est=truth+normal(next)*se;return{est,lo:est-z*se,hi:est+z*se};});
    const lo=Math.floor(Math.min(-.05,...intervals.map(v=>v.lo))*20)/20,hi=Math.ceil(Math.max(.3,...intervals.map(v=>v.hi))*20)/20,x=v=>80+(v-lo)/(hi-lo)*680;let count=0;
    $('samplingPlot').innerHTML=`<title>60 synthetic ${(level*100).toFixed(0)}% intervals at n=${n} per arm</title><line x1="${x(truth)}" x2="${x(truth)}" y1="10" y2="510" stroke="#397e9e" stroke-width="2"/>${intervals.map((v,i)=>{const hit=v.lo<=truth&&truth<=v.hi;if(hit)count++;const y=25+i*8.1,color=hit?'#39795b':'#b15c54';return `<line x1="${x(v.lo)}" x2="${x(v.hi)}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="3"/><circle cx="${x(v.est)}" cy="${y}" r="2.5" fill="${color}"/>`;}).join('')}<text x="80" y="532" font-size="14">${(lo*100).toFixed(0)} pp</text><text x="${x(truth)}" y="532" text-anchor="middle" font-size="14">Truth 12 pp</text><text x="760" y="532" text-anchor="end" font-size="14">${(hi*100).toFixed(0)} pp</text>`;
    $('coverageNOut').textContent=n;$('samplingStatus').textContent=`${count}/60 intervals cover the fixed 12 pp truth; nominal confidence ${level*100}% (z=${z.toFixed(3)}), n=${n} per arm, SE ${(se*100).toFixed(2)} pp, seed ${sampleSeed}. Same random draws when changing confidence: higher levels widen intervals. In this known-SE normal model, larger n narrows intervals without changing their nominal coverage. Axis adapts.`;
  };
  ['coverageLevel','coverageN'].forEach(id=>$(id).addEventListener('input',drawSampling));
  $('coverageReset').addEventListener('click',()=>{$('coverageN').value=270;$('coverageLevel').value='.95';sampleSeed=1212;drawSampling();});drawSampling();

  // Remove original function listeners before replacing them, so one event draws once.
  ['powerN','powerEffect','powerBaseline'].forEach(id=>$(id).removeEventListener('input',drawPower));
  drawPower=function(){const n=+$('powerN').value,p0=+$('powerBaseline').value/100,alpha=+$('powerAlpha').value/100,e=$('powerEffect');e.max=((p0-.005)*100).toFixed(1);if(+e.value>+e.max)e.value=e.max;const d=+e.value/100,p=M.planning(n,d,p0,alpha),x=v=>55+(v-100)/5900*690;
    const pts=[];for(let nn=100;nn<=6000;nn+=100)pts.push(`${x(nn)},${210-M.planning(nn,d,p0,alpha).power*170}`);
    $('powerPlot').innerHTML=`<title>Power by per-arm sample size at alpha ${alpha}</title><line x1="55" x2="745" y1="210" y2="210" stroke="#89978e"/><line x1="55" x2="745" y1="74" y2="74" stroke="#b79558" stroke-dasharray="4 4"/><text x="58" y="65" font-size="14">80% power</text><text x="9" y="43" font-size="14">100%</text><text x="16" y="210" font-size="14">0%</text><polyline points="${pts.join(' ')}" fill="none" stroke="#497f9e" stroke-width="4"/><circle cx="${x(n)}" cy="${210-p.power*170}" r="7" fill="#a88049"/><text x="55" y="238" font-size="14">100</text><text x="745" y="238" text-anchor="end" font-size="14">6,000 per arm</text>`;
    const xmin=Math.min(-4,p.mu-4*p.sigma),xmax=4,xx=v=>60+(v-xmin)/(xmax-xmin)*680,yy=v=>232-v*425,pdf=(z,mu=0,sigma=1)=>Math.exp(-.5*((z-mu)/sigma)**2)/(sigma*Math.sqrt(2*Math.PI));
    function curve(mu,sigma){return Array.from({length:501},(_,i)=>{const z=xmin+i/500*(xmax-xmin);return `${xx(z)},${yy(pdf(z,mu,sigma))}`;}).join(' ');}
    function area(a,b,mu,sigma,color){const pts=Array.from({length:151},(_,i)=>{const z=a+(b-a)*i/150;return `${xx(z)},${yy(pdf(z,mu,sigma))}`;});return `<polygon points="${xx(a)},232 ${pts.join(' ')} ${xx(b)},232" fill="${color}"/>`;}
    $('hypothesisPlot').innerHTML=`<title>Null and alternative densities; alpha ${alpha}, beta ${p.beta.toFixed(3)}, power ${p.power.toFixed(3)}</title><rect x="60" y="38" width="${xx(-p.critical)-60}" height="194" fill="#fff0e7"/><rect x="${xx(p.critical)}" y="38" width="${740-xx(p.critical)}" height="194" fill="#fff0e7"/>${area(xmin,-p.critical,p.mu,p.sigma,'#b4d9bf')}${area(p.critical,xmax,p.mu,p.sigma,'#b4d9bf')}<polyline points="${curve(0,1)}" fill="none" stroke="#477f9d" stroke-width="2.5"/><polyline points="${curve(p.mu,p.sigma)}" fill="none" stroke="#468160" stroke-width="2.5"/>${[-p.critical,p.critical].map(v=>`<line x1="${xx(v)}" x2="${xx(v)}" y1="38" y2="232" stroke="#ac7f4b" stroke-dasharray="4 4"/><text x="${xx(v)}" y="255" font-size="13" text-anchor="middle">${v.toFixed(2)}</text>`).join('')}<line x1="60" x2="740" y1="232" y2="232" stroke="#87988b"/><text x="60" y="25" font-size="15" fill="#468160">Alternative · mean ${p.mu.toFixed(2)}</text><text x="740" y="25" text-anchor="end" font-size="15" fill="#477f9d">Null · mean 0</text><text x="${xx(0)}" y="279" text-anchor="middle" font-size="14">0</text><text x="60" y="292" font-size="13">Z · axis expands to retain both densities · shaded green area = power</text>`;
    $('powerNOut').textContent=n;$('powerBaselineOut').textContent=(p0*100).toFixed(0)+'%';$('powerEffectOut').textContent=(d*100).toFixed(1)+' pp';$('powerAlphaOut').textContent=(alpha*100).toFixed(0)+'%';
    $('matrixAlpha').textContent=(alpha*100).toFixed(0)+'%';$('matrixBeta').textContent=(p.beta*100).toFixed(1)+'%';$('matrixPower').textContent=(p.power*100).toFixed(1)+'%';
    $('powerStatus').textContent=`Synthetic fixed-sample planning: ${n} per arm, risks ${(p0*100).toFixed(1)}% and ${(p.p1*100).toFixed(1)}%; α ${(alpha*100).toFixed(0)}%, power ${(p.power*100).toFixed(1)}%, β ${(p.beta*100).toFixed(1)}%. Two-sided normal approximation; no sequential or continuity correction.`;
  };
  ['powerN','powerEffect','powerBaseline','powerAlpha'].forEach(id=>$(id).addEventListener('input',drawPower));
  $('powerReset').addEventListener('click',()=>{$('powerN').value=2300;$('powerBaseline').value=8;$('powerEffect').max=7.5;$('powerEffect').value=3.6;$('powerAlpha').value=5;drawPower();});drawPower();
})();
