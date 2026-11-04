"use strict";
// Only aggregate results enter the page. All simulations below are labeled synthetic.
(() => {
  const $=id=>document.getElementById(id), reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
  const nav=$('readerNav'), groups=[...nav.querySelectorAll('.nav-group')];
  $('navExpand').addEventListener('click',()=>groups.forEach(g=>g.open=true));
  $('navCollapse').addEventListener('click',()=>groups.forEach(g=>g.open=false));
  const toggle=$('navToggle');
  function closeNav(){nav.classList.remove('mobile-open');toggle.setAttribute('aria-expanded','false');}
  toggle.addEventListener('click',()=>{const open=nav.classList.toggle('mobile-open');toggle.setAttribute('aria-expanded',String(open));});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&nav.classList.contains('mobile-open')){closeNav();toggle.focus();}});
  nav.addEventListener('click',e=>{const a=e.target.closest('a');if(a){closeNav();const target=document.getElementById(a.hash.slice(1));if(target){target.setAttribute('tabindex','-1');target.focus({preventScroll:true});}}});
  const entries=[...nav.querySelectorAll('a[href^="#"]')].map(link=>({link,node:document.getElementById(link.hash.slice(1))})).filter(x=>x.node);
  let active='',scheduled=false;
  function updateNav(){scheduled=false;let chosen=entries[0];for(const e of entries){if(e.node.getBoundingClientRect().top<=135)chosen=e;else break;}if(!chosen||chosen.node.id===active)return;active=chosen.node.id;
    entries.forEach(e=>{if(e===chosen)e.link.setAttribute('aria-current','true');else e.link.removeAttribute('aria-current');});
    const group=chosen.link.closest('details');groups.forEach(g=>g.classList.toggle('active-parent',g===group));if(group)group.open=true;
    const r=chosen.link.getBoundingClientRect(),nr=nav.getBoundingClientRect(),headerHeight=nav.querySelector('.nav-header').offsetHeight;if(r.top<nr.top+headerHeight+8)nav.scrollTop-=nr.top+headerHeight+8-r.top;else if(r.bottom>nr.bottom-24)nav.scrollTop+=r.bottom-nr.bottom+24;
  }
  function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(updateNav);}}
  if('IntersectionObserver' in window){const io=new IntersectionObserver(schedule,{rootMargin:'-15% 0px -65% 0px',threshold:0});entries.forEach(e=>io.observe(e.node));}
  addEventListener('scroll',schedule,{passive:true});addEventListener('resize',schedule);schedule();
  // Finite animation, user-triggered, with no redundant playback controls.
  $('reroll').addEventListener('click',()=>{if(!reduced)$('assignmentGrid').animate([{opacity:.25,transform:'translateY(3px)'},{opacity:1,transform:'translateY(0)'}],{duration:450,easing:'ease-out'});});
  const fmt=(x,n=3)=>Number(x).toFixed(n);
  function toy(){let a=+$('toyTreat').value,c=+$('toyControl').value,p1=a/100,p0=c/100,rd=p1-p0,rr=p1/p0;
    $('toyTreatOut').textContent=a;$('toyControlOut').textContent=c;
    $('toyTable').innerHTML=`<table><caption>Synthetic counts; 100 assigned per arm</caption><thead><tr><th scope="col">Arm</th><th scope="col">Event</th><th scope="col">Non-event</th></tr></thead><tbody><tr><th scope="row">Treatment</th><td>${a}</td><td>${100-a}</td></tr><tr><th scope="row">Control</th><td>${c}</td><td>${100-c}</td></tr></tbody></table>`;
    const effect=window.RCTMath.effects(a,c);
    $('toyIcons').innerHTML=[[a,'Treatment','#4c86a9'],[c,'Control','#b48164']].map(([events,label,color])=>`<div class="icon-arm"><h4>${label}: ${events}/100 events</h4><svg viewBox="0 0 250 270" role="img" aria-label="${label}: ${events} events and ${100-events} non-events; each person is one hypothetical participant"><title>${label}: ${events} events per 100</title>${Array.from({length:100},(_,i)=>`<g transform="translate(${12+(i%10)*24} ${8+Math.floor(i/10)*25})" fill="${i<events?color:'#e3e8e4'}"><circle cx="5" cy="4" r="3.3"/><path d="M1 9H9L11 18H8V23H6V17H4V23H2V18H-1Z"/></g>`).join('')}</svg><p class="small">Colored = adverse event; pale = no event.</p></div>`).join('');
    $('toyBars').innerHTML=`<title>Synthetic treatment risk ${a}% and control risk ${c}% on a common scale</title>${[[a,'Treatment',45,'#4c86a9'],[c,'Control',95,'#b48164']].map(([v,label,y,color])=>`<text x="10" y="${y+5}" font-size="16">${label}</text><rect x="125" y="${y-18}" width="570" height="28" rx="4" fill="#edf0eb"/><rect x="125" y="${y-18}" width="${v/100*570}" height="28" rx="4" fill="${color}"/><text x="712" y="${y+4}" font-size="16">${v}%</text>`).join('')}<text x="125" y="138" font-size="14">0%</text><text x="695" y="138" text-anchor="end" font-size="14">100% · shared risk scale</text>`;
    $('toyEquations').innerHTML=`<p>p̂₁ = ${a}/100 = ${fmt(p1,2)}; p̂₀ = ${c}/100 = ${fmt(p0,2)}</p><p>RD = ${fmt(p1,2)} − ${fmt(p0,2)} = ${fmt(rd*100,1)} pp; RR = ${a}/${c} = ${fmt(rr)}</p><p>SE(log RR) = √(1/${a} + 1/${c} − 2/100) = ${fmt(Math.sqrt(1/a+1/c-.02))}</p><p>95% Katz CI = exp[log(${fmt(rr)}) ± 1.960 × SE] = [${fmt(effect.lo)}, ${fmt(effect.hi)}]</p>`;
    $('toyResult').textContent=`RD ${fmt(rd*100,1)} percentage points; ARR ${fmt(-rd*100,1)} pp; RR ${fmt(rr)}; RRR ${fmt((1-rr)*100,1)}%; OR ${fmt((a/(100-a))/(c/(100-c)))}. ${rd<0?'Reciprocal ARR '+fmt(-1/rd,1)+' participants.':rd>0?'Reciprocal excess risk '+fmt(1/rd,1)+' participants.':'No risk difference; reciprocal is infinite.'} Katz intervals are approximate and unreliable with very few events.`;
  }
  ['toyTreat','toyControl'].forEach(id=>$(id).addEventListener('input',toy));$('toyReset').addEventListener('click',()=>{$('toyTreat').value=12;$('toyControl').value=20;toy();});toy();
  function missing(){const t=+$('missingTreat').value,c=+$('missingControl').value;const a=123+Math.round(129*t/100),b=206+Math.round(82*c/100);const rr=a/b,se=Math.sqrt(1/a+1/b-2/2278),lo=Math.exp(Math.log(rr)-1.96*se),hi=Math.exp(Math.log(rr)+1.96*se);
    $('missingTreatOut').textContent=t+'%';$('missingControlOut').textContent=c+'%';const x=v=>80+v/1.7*680;
    $('sensitivityPlot').innerHTML=`<title>Hypothetical sensitivity RR ${fmt(rr)} with approximate interval ${fmt(lo)} to ${fmt(hi)}</title><line x1="80" y1="100" x2="760" y2="100" stroke="#cfc6b7"/><line x1="${x(1)}" y1="35" x2="${x(1)}" y2="115" stroke="#777" stroke-dasharray="5 4"/><line x1="${x(lo)}" y1="70" x2="${x(hi)}" y2="70" stroke="#78608c" stroke-width="5"/><circle cx="${x(rr)}" cy="70" r="7" fill="#78608c"/>${[0,.5,1,1.5].map(v=>`<text x="${x(v)}" y="130" text-anchor="middle" font-size="16">${v}</text>`).join('')}<text x="80" y="165" font-size="17">Hypothetical RR · linear scale · 1 = equal risks</text>`;
    $('missingResult').textContent=`Assumed counts, after rounding to whole events: ${a}/2,278 synbiotic; ${b}/2,278 placebo. RR ${fmt(rr)}, approximate 95% interval [${fmt(lo)}, ${fmt(hi)}]. This is a scenario, not an empirical estimate of missing outcomes.`;
  }
  ['missingTreat','missingControl'].forEach(id=>$(id).addEventListener('input',missing));$('missingReset').addEventListener('click',()=>{$('missingTreat').value=0;$('missingControl').value=0;missing();});missing();
  // Static page outputs remain readable if aggregate script or MathJax is unavailable.
  if(!window.RCT_RESULTS){$('aggregateStatus').textContent='Aggregate script unavailable. Static tables, formulas and SVG figures remain the verified results.';}
})();
