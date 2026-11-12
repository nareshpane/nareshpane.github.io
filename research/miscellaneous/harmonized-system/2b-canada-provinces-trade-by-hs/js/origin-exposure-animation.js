/* Seven-sequence 40-second introduction, adapted from Study 02.
 * Canadian origin scenes and transport artwork are retained; middle scenes use
 * the reference study’s validated world geometry and schematic global routes.
 * Last scenes use Alberta's observed 2025 data; no tariff matching is performed.
 * Deterministic render(time) resets every object and cargo attachment on Replay.
 */
(() => {
  'use strict';
  const root = document.getElementById('origin-exposure');
  if (!root) return;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const narrow = matchMedia('(max-width: 650px)');
  const duration = 40;
  const dataFile = '2b-canada-provinces-trade-by-hs/animation-data-2025.json';
  let animationData = null, dataUnavailable = false, speed = 1, still = false;
  const orange = (rank,count) => window.TradeDisplay.shade(rank+1,count);
  const dollars = v => v>=1e9?'$'+(v/1e9).toFixed(1)+'B':'$'+Math.round(v/1e6)+'M';
  const mapFile = '2b-canada-provinces-trade-by-hs/animation-assets/animation-maps.svg';
  const provinces = [
    ['BC','British Columbia',108.5,253.6,'forest'],['AB','Alberta',154.9,271.3,'energy'],
    ['SK','Saskatchewan',196.9,290.3,'grain'],['MB','Manitoba',240,285.2,'gear'],
    ['ON','Ontario',314.2,316.6,'car'],['QC','Quebec',385.2,276.8,'plane'],
    ['NB','New Brunswick',434.8,320.5,'forest'],['NS','Nova Scotia',461.9,322.1,'fish'],
    ['PE','Prince Edward Island',455.3,310.6,'grain'],['NL','Newfoundland and Labrador',448.3,242.7,'energy'],
    ['YT','Yukon',102.4,155.6,'mineral'],['NT','Northwest Territories',157.5,176.2,'mineral'],
    ['NU','Nunavut',253.8,158.5,'mineral']
  ];
  const colors = ['#c2d3bd','#dbc6a9','#d8d8b1','#bcd1c9','#b9ced2','#c9c7d8'];
  // Short, gently sloped driving lanes on either side of a schematic crossing.
  // The longer curved lines still describe conceptual Canada-to-U.S. corridors.
  const truckLegs = {
    0: [[156,303],[106,311],[124,330],[174,338]],
    1: [[160,323],[214,332],[214,352],[268,361]],
    3: [[353,325],[405,334],[392,365],[424.7,372.2]]
  };


  const worldMapFile = '2b-canada-provinces-trade-by-hs/animation-assets/world-map.svg';
  const rx=[1,.9986,.9954,.99,.9822,.973,.96,.9427,.9216,.8962,.8679,.835,.7986,.7597,.7186,.6732,.6213,.5722,.5322];
  const ry=[0,.062,.124,.186,.248,.31,.372,.434,.4958,.5571,.6176,.6769,.7346,.7903,.8435,.8936,.9394,.9761,1];
  function worldPoint(lon,lat) {
    const pos=Math.min(Math.abs(lat),89.999)/5,i=Math.floor(pos),u=pos-i;
    return [560+lon/180*(rx[i]+(rx[i+1]-rx[i])*u)*518,240-Math.sign(lat)*(ry[i]+(ry[i+1]-ry[i])*u)*200];
  }
  const exportOrigin=worldPoint(-114.5,54.8), pacificPort=worldPoint(-126,48), atlanticPort=worldPoint(-61,45);
  const partners=[
    ['US','United States',-98,38,360,155,'land'],
    ['GB','United Kingdom',-2,54,575,74,'atlantic'],
    ['JP','Japan',139,37,1022,130,'pacific'],
    ['BR','Brazil',-51,-12,473,298,'south'],
    ['IN','India',78,22,786,233,'indian'],
    ['ZA','South Africa',24,-29,670,344,'africa']
  ];
  // Ocean routes depart coastal points; Pacific geometry wraps at the map seam.
  const globalRoutes=partners.map(([code,name,lon,lat,labelX,labelY,mode],i)=>{
    const end=worldPoint(lon,lat);
    let legs;
    if(mode==='land') legs=[[...exportOrigin,300,112,...end]];
    else if(mode==='pacific') legs=[[...pacificPort,175,112,...worldPoint(-179.5,40)],[...worldPoint(179.5,40),995,140,...end]];
    else if(mode==='atlantic') legs=[[...atlanticPort,480,115,...worldPoint(-5,50)]];
    else if(mode==='south') legs=[[...atlanticPort,450,210,...worldPoint(-43.2,-22.9)]];
    else if(mode==='africa') legs=[[...atlanticPort,450,285,...worldPoint(18.3,-34.5)]];
    else legs=[[...atlanticPort,425,295,...worldPoint(-5,-42)],[...worldPoint(-5,-42),680,395,...worldPoint(72,17)]];
    return {code,name,end,labelX,labelY,mode,legs,start:19.2+i*.23};
  });
  function curvePoint(p,u) {const [sx,sy,cx,cy,ex,ey]=p;return [(1-u)**2*sx+2*(1-u)*u*cx+u*u*ex,(1-u)**2*sy+2*(1-u)*u*cy+u*u*ey];}

  const cargo = [['841490',true],['940360',false],['841480',true],['853710',false],['841410',true],['060110',false]];
  const svg = root.querySelector('.oa-stage');
  const controls = root.querySelector('.oa-controls');
  const pause = root.querySelector('[data-oa-pause]');
  const replay = root.querySelector('[data-oa-replay]');
  const heading = root.querySelector('.oa-heading');
  const headline = root.querySelector('.oa-headline');
  const subline = root.querySelector('.oa-subline');
  const kicker = root.querySelector('.oa-kicker');
  const progress = root.querySelector('.oa-progress-fill');
  const note = root.querySelector('.oa-static-note');
  const clamp = x => Math.max(0, Math.min(1, x));
  const ease = x => { x = clamp(x); return x*x*(3-2*x); };
  const ramp = (t,a,b) => ease((t-a)/(b-a));
  const windowAlpha = (t,a,b) => ramp(t,a,a+.65)*(1-ramp(t,b-.65,b));
  const mix = (a,b,u) => a+(b-a)*u;
  const opacity = (el,v) => { el.style.opacity = v.toFixed(3); };
  const move = (el,x,y,s=1,angle=0) => el.setAttribute('transform',`translate(${x.toFixed(2)} ${y.toFixed(2)}) scale(${s}) rotate(${angle})`);
  const get = id => root.querySelector(`#oa-${id}`);
  const use = (id,x=0,y=0,s=1,extra='') => `<use href="#oa-${id}" transform="translate(${x} ${y}) scale(${s})" ${extra}/>`;
  const text = (x,y,label,cls='oa-label',extra='') => `<text x="${x}" y="${y}" class="${cls}" ${extra}>${label}</text>`;
  const line = d => `<path d="${d}" class="oa-line"/>`;
  const panel = (x,y,w,h,fill='#fffdf7') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${fill}" stroke="#d4dacb"/>`;
  const tile = (code,matched=false) => `<rect x="-44" y="-23" width="88" height="46" rx="7" fill="${matched?'#f1d8c7':'#e5ece1'}" stroke="${matched?'#b75c38':'#86a68e'}" stroke-width="1.5"/><path d="M-35-14H-27M-35-10H-27" stroke="${matched?'#b75c38':'#6d927e'}"/>${text(0,9,code,'oa-code','text-anchor="middle"')}`;
  const defs = `<defs>
    <pattern id="oa-ties" width="15" height="10" patternUnits="userSpaceOnUse"><path d="M7 0V10" stroke="#b8b4a6" stroke-width="2"/></pattern>
    <symbol id="oa-forest" viewBox="-18 -20 36 40"><path d="M-9 8V17M7 5V17" stroke="#547763" stroke-width="2"/><path d="M-9-17L-18 8H0ZM7-20L-3 5H17Z" fill="#86a68c" stroke="#547763"/></symbol>
    <symbol id="oa-energy" viewBox="-18 -20 36 40"><path d="M-11 15L-5-16H5L11 15M-8 0H8M-5-12L8 0L-10 10H11" fill="none" stroke="#9b754d" stroke-width="2"/><path d="M13-8Q22 2 14 6Q8 2 13-8" fill="#bf9060"/></symbol>
    <symbol id="oa-grain" viewBox="-18 -20 36 40"><path d="M0 18V-18M0-9L-9-16M0-2L9-9M0 4L-9-3M0 10L9 3M-12 18H12" fill="none" stroke="#b5833e" stroke-width="3" stroke-linecap="round"/></symbol>
    <symbol id="oa-gear" viewBox="-18 -20 36 40"><path d="M-6-16H6L8-10L14-8L18-2V6L11 9L7 16H-7L-11 9L-18 6V-2L-14-8L-8-10Z" fill="#bacdc7" stroke="#54786b"/><circle r="7" fill="#f8f5ec" stroke="#54786b" stroke-width="2"/></symbol>
    <symbol id="oa-car" viewBox="-18 -20 36 40"><path d="M-17 4L-12-7H9L15 1L18 3V12H-18V4Z" fill="#9ab7be" stroke="#54786b"/><path d="M-9-4H6L10 1H-11Z" fill="#eef3ef"/><circle cx="-11" cy="12" r="4" fill="#47685a"/><circle cx="11" cy="12" r="4" fill="#47685a"/></symbol>
    <symbol id="oa-plane" viewBox="-18 -20 36 40"><path d="M-3-19H3L5-3L18 5V10L5 5L4 14L10 18H-10L-4 14L-5 5L-18 10V5L-5-3Z" fill="#b8b5cd" stroke="#696a83"/></symbol>
    <symbol id="oa-fish" viewBox="-18 -20 36 40"><path d="M-13 0Q0-17 14 0Q0 17-13 0L-18-9V9Z" fill="#a2bdc4" stroke="#547c85"/><circle cx="8" cy="-2" r="1.5" fill="#28584f"/></symbol>
    <symbol id="oa-mineral" viewBox="-18 -20 36 40"><path d="M-17-5L-7-16H7L17-5L0 17ZM-17-5H17M-7-16L-5-5L0 17L5-5L7-16" fill="#c9c2d2" stroke="#84758d" stroke-width="1.5"/></symbol>
    <symbol id="oa-box" overflow="visible"><rect x="-10" y="-10" width="20" height="20" rx="3" fill="#e4c092" stroke="#957348"/><path d="M-10-3H10M0-10V10" stroke="#957348"/></symbol>
    <symbol id="oa-container" overflow="visible"><rect width="48" height="24" rx="3" fill="var(--oa-cargo,#bdcfc2)" stroke="#54786b" stroke-width="1.5"/><path d="M8 4V20M16 4V20M24 4V20M32 4V20M40 4V20" stroke="#54786b" opacity=".5"/></symbol>
    <symbol id="oa-truck-body" overflow="visible">${use('container',0,-28,1.3)}<path d="M62-19H78L91-4V5H62Z" fill="#89aa98" stroke="#28584f" stroke-width="1.5"/><path d="M68-15H77L85-6H68Z" fill="#edf4f0"/><path d="M0 5H65" stroke="#54786b" stroke-width="3"/></symbol>
    <symbol id="oa-wheel" overflow="visible"><circle r="6" fill="#486859"/><circle r="3.5" fill="#dfdfcd"/><path d="M-3 0H3M0-3V3" stroke="#486859"/></symbol>
    <symbol id="oa-train" overflow="visible"><path d="M0-32H35L47-12V5H0Z" fill="#557f77" stroke="#28584f" stroke-width="1.5"/><rect x="24" y="-26" width="10" height="12" fill="#dce9e4"/>${[55,109,163].map((x,i)=>`${use('container',x,-25,1,`style="--oa-cargo:${colors[i]}"`)}<path d="M${x-7} 5H${x+48}" stroke="#54786b" stroke-width="3"/>${use('wheel',x+8,8,.8)}${use('wheel',x+38,8,.8)}`).join('')}${use('wheel',12,8)}${use('wheel',35,8)}</symbol>
    <symbol id="oa-ship" overflow="visible"><path d="M0 4H132L118 25H21Z" fill="#608890" stroke="#28584f" stroke-width="1.5"/><path d="M10 14H125" stroke="#dce7df" stroke-width="2"/><rect x="10" y="-24" width="19" height="28" rx="2" fill="#e8ecdf" stroke="#54786b"/><path d="M19-24V-35" stroke="#54786b"/><rect x="13" y="-20" width="12" height="6" fill="#9ab7be"/>${use('container',36,-20,.8)}${use('container',77,-20,.8, 'style="--oa-cargo:#dbc6a9"')}${use('container',58,-40,.8, 'style="--oa-cargo:#b9ced2"')}<path d="M-5 32Q20 27 43 32T91 32T139 32" fill="none" stroke="#a3bcc2" stroke-width="2"/></symbol>
    <symbol id="oa-document" overflow="visible"><path d="M0 0H25L35 10V44H0Z" fill="#fffdf7" stroke="#90a99b"/><path d="M25 0V10H35M6 18H28M6 25H28M6 32H20" fill="none" stroke="#90a99b"/><circle cx="27" cy="35" r="8" fill="#efccba" stroke="#b75c38"/><path d="M23 35L26 38L31 32" fill="none" stroke="#b75c38"/></symbol>
  </defs>`;
  let refs, routes, elapsed = 0, last = null, frame = 0, lastPaint = 0;
  let userPaused = false, visible = true, finished = false, phase = '';
  function truck(id) {
    return `<g id="oa-${id}">${use('truck-body')}<g data-oa-wheel transform="translate(13 9)">${use('wheel')}</g><g data-oa-wheel transform="translate(31 9)">${use('wheel')}</g><g data-oa-wheel transform="translate(77 9)">${use('wheel')}</g></g>`;
  }
  function layout() {
    const m = narrow.matches;
    svg.setAttribute('viewBox',m?'0 0 560 630':'0 0 1120 490');
    const map = `<g id="oa-geography"><use href="${mapFile}#US" id="oa-us-land" fill="#e3dccb" stroke="#b6af99" stroke-width="1"/>${provinces.map(([abbr,name],i)=>`<use href="${mapFile}#${abbr}" class="oa-province" data-oa-origin="${abbr}" fill="${colors[i%colors.length]}"><title>${name}</title></use>`).join('')}<path id="oa-border-line" class="oa-border" d="M90.4 306.2L136.1 323.4L195.7 336.8L256.6 340.6L288 348.2L328.7 361.4L346.3 396.7L368.1 381.5L388.7 355.2L410.7 347L418.3 323.2L429.9 317.7L438.2 334.7"/>
      <g id="oa-origin-labels">${provinces.map(([abbr,,x,y],i)=>text(i===6?490:i===7?526:i===8?512:x, i===6?380:i===7?344:i===8?303:y,abbr,'oa-origin-label')).join('')}</g>
      <g id="oa-production-icons">${provinces.map(([abbr,,x,y,icon],i)=>{ const px=i===6?490:i===7?526:i===8?512:x, py=i===6?346:i===7?310:i===8?269:y+22; return `<g data-oa-production="${abbr}">${line(`M${x} ${y+4}L${px} ${py}`)}<circle cx="${px}" cy="${py}" r="19" fill="#fffdf7" stroke="#d4dacb"/>${use(icon,px-14,py-16,1,'width="28" height="32"')}</g>`;}).join('')}</g>
      <g id="oa-trade-routes">${Object.entries(truckLegs).map(([i,[a,b,c,d]])=>{
        const x=(b[0]+c[0])/2,y=(b[1]+c[1])/2;
        return `<g data-oa-truck-lane="${i}"><path d="M${a}L${b}M${c}L${d}" fill="none" stroke="#d6d8c8" stroke-width="6" stroke-linecap="round"/><path d="M${a}L${b}M${c}L${d}" fill="none" stroke="#f8f5ec" stroke-width="1" stroke-dasharray="4 5"/><path d="M${b}L${c}" class="oa-line" stroke-dasharray="2 3"/><rect x="${x-7}" y="${y-9}" width="14" height="18" rx="3" fill="#fffdf7" stroke="#b75c38"/><path d="M${x-4} ${y-3}H${x+4}M${x-4} ${y+3}H${x+4}" stroke="#b75c38" stroke-width="1.5"/></g>`;
      }).join('')}${[
        ['M108.5 253.6Q95 289 89.1 319.9','truck',108.5,253.6,96,285,89.1,319.9],
        ['M196.9 290.3Q224 340 228.6 416','truck',196.9,290.3,224,340,228.6,416],
        ['M314.2 316.6Q335 350 344.8 406.8','train',314.2,316.6,335,350,344.8,406.8],
        ['M385.2 276.8Q405 324 424.7 372.2','truck',385.2,276.8,405,324,424.7,372.2],
        ['M460.2 327.4Q480 374 423.3 368.6','ship',460.2,327.4,480,374,423.3,368.6]
      ].map(([d,mode,...pts],i)=>`<path id="oa-route-${i}" class="oa-route" d="${d}" pathLength="1" stroke-dasharray="1"/><g id="oa-carrier-${i}" data-mode="${mode}" data-points="${pts}">${mode==='truck'?`<g transform="translate(-45 0)">${truck('crossing-'+i)}</g>`:mode==='train'?`<g transform="translate(210 0) scale(-1 1)">${use(mode)}</g>`:use(mode)}</g><circle cx="${pts[4]}" cy="${pts[5]}" r="3" fill="#b5833e"/>`).join('')}</g>
      <g id="oa-country-labels">${text(260,228,'CANADA','oa-country')}${text(265,469,'UNITED STATES','oa-country')}</g>
    </g>`;
    const cardX=m?35:760, cardY=m?440:80, cardW=m?490:305;
    const productionCards = `<g id="oa-production-cards">${panel(cardX,cardY,cardW,m?104:250)}${text(cardX+20,cardY+30,'MANY PRODUCTS','oa-micro')}${(m?['grain','gear','mineral','fish']:['forest','grain','car','mineral']).map((icon,i)=>`${use(icon,cardX+(m?18+i*122:20),cardY+(m?43:48+i*48),1,'width="30" height="34"')}${m?'':text(cardX+67,cardY+71+i*48,['Forestry &amp; marine','Agriculture &amp; energy','Manufacturing','Minerals'][i],'oa-small')}`).join('')}${text(m?280:912,m?585:375,'13 ORIGINS → ONE CANADA','oa-label','text-anchor="middle"')}${text(m?280:912,m?611:403,'Production symbols are examples','oa-small','text-anchor="middle"')}</g>`;
    const railY=m?604:460, roadY=m?536:389;
    const logistics = `<g id="oa-logistics" class="oa-scene"><path d="M0 ${roadY+15}H${m?390:825}" stroke="#d6d8c8" stroke-width="22"/><path d="M0 ${roadY+15}H${m?390:825}" stroke="#f8f5ec" stroke-dasharray="14 18"/><path d="M0 ${railY+14}H${m?560:1120}" stroke="#98ab99" stroke-width="3"/><rect y="${railY+7}" width="${m?560:1120}" height="10" fill="url(#oa-ties)"/><path d="M${m?420:890} ${roadY-10}V${roadY-74}H${m?500:1005}V${roadY-10}M${m?432:902} ${roadY-74}V${roadY-105}H${m?502:1020}" fill="none" stroke="#769a8c" stroke-width="4"/>${use('container',m?422:902,roadY-28)}${use('container',m?472:952,roadY-28,1,'style="--oa-cargo:#dbc6a9"')}${truck('road-0')}${truck('road-1')}${truck('road-2')}<g id="oa-inland-train"><g transform="translate(210 0) scale(-1 1)">${use('train')}</g></g><g id="oa-coastal-ship">${use('ship')}</g>${provinces.map(([, ,x,y],i)=>`<g data-oa-parcel="${i}">${use('box')}</g>`).join('')}</g>`;
    const tradeNote = `<g id="oa-trade-note" class="oa-scene">${panel(m?35:790,m?505:116,m?490:278,m?100:190)}${text(m?55:810,m?539:152,'CANADA → UNITED STATES','oa-micro')}${text(m?55:810,m?568:190,'2025 domestic exports','oa-label')}${text(m?55:810,m?594:224,'Conceptual corridors','oa-small')}${m?'':text(810,265,'One illustrated destination.','oa-small')}</g>`;
    const bx=m?226:485, by=m?180:110;
    const border = `<g id="oa-checkpoint" class="oa-scene">${text(m?35:65,55,'CANADA','oa-country')}${text(m?525:1040,55,'UNITED STATES','oa-country','text-anchor="end"')}${[0,1,2].map(i=>`<path d="M${m?20:35} ${m?188+i*65:173+i*91}H${m?325:735}" stroke="#e6e5d9" stroke-width="38"/><path d="M${m?20:35} ${m?188+i*65:173+i*91}H${m?325:735}" stroke="#bdc6b6" stroke-dasharray="10 12"/>`).join('')}
      <rect x="${bx}" y="${by}" width="${m?79:155}" height="${m?187:267}" rx="10" fill="#deeadf" stroke="#86a68e"/><path d="M${bx-10} ${by}H${bx+(m?89:165)}" stroke="#54786b" stroke-width="6"/>${text(bx+(m?39:77),by-24,'U.S. BORDER','oa-label','text-anchor="middle"')}${use('document',bx+(m?21:60),by+16,m?.9:1.1)}
      <g id="oa-scanner"><rect x="${bx+8}" y="${by+67}" width="${m?63:139}" height="30" rx="4" fill="#b75c38" opacity=".16"/><path d="M${bx+8} ${by+67}V${by+97}M${bx+(m?71:147)} ${by+67}V${by+97}" stroke="#b75c38" stroke-width="2"/></g><g id="oa-gate" transform="translate(${bx+12} ${by+(m?171:246)})"><path d="M0 0H${m?54:127}" stroke="#b75c38" stroke-width="5"/><path d="M0 0H${m?54:127}" stroke="#f4dfc9" stroke-width="5" stroke-dasharray="8 10"/></g>
      ${panel(m?364:795,m?105:111,m?177:281,m?160:116,'#f4e1d2')}${text(m?379:815,m?133:143,'HS4 8414','oa-micro oa-policy-text')}${text(m?379:815,m?153:165,'HEADING','oa-micro oa-policy-text')}${panel(m?364:795,m?363:313,m?177:281,m?165:116,'#e7eee1')}${text(m?379:815,m?394:347,'OTHER HS4','oa-micro')}
      <path d="M${bx+(m?80:156)} ${by+90}Q${m?330:720} ${by+90} ${m?360:786} ${m?205:200}" fill="none" stroke="#b75c38" stroke-width="2"/><path d="M${bx+(m?80:156)} ${by+90}Q${m?334:724} ${m?428:378} ${m?360:786} ${m?428:378}" fill="none" stroke="#86a68e" stroke-width="2"/>
      ${truck('inspection-0')}${truck('inspection-1')}${truck('inspection-2')}${cargo.map(([c,match],i)=>`<g id="oa-screen-tile-${i}">${tile(c)}<rect class="oa-stamp" x="-44" y="-23" width="88" height="46" rx="7" fill="${match?'#d77b4a':'#86a68e'}" opacity="0"/></g>`).join('')}${text(m?280:560,m?558:473,'HS codes classify goods · schematic product grouping','oa-small','text-anchor="middle"')}</g>`;

    const world = '<g id="oa-world" class="oa-scene"><g id="oa-world-position" transform="'+(m?'translate(0 42) scale(.5)':'translate(58 0) scale(.9)')+'"><use href="'+worldMapFile+'#world-land" fill="#dae4d4" stroke="#91ab96" stroke-width=".9"/>'+text(285,83,'CANADA','oa-label','text-anchor="middle"')+'<circle cx="'+exportOrigin[0]+'" cy="'+exportOrigin[1]+'" r="10" fill="#b75c38" stroke="#fff5e9" stroke-width="3"/>'+globalRoutes.map((r,i)=>{
      const d=r.legs.map(p=>'M'+p[0]+' '+p[1]+'Q'+p[2]+' '+p[3]+' '+p[4]+' '+p[5]).join('');
      return '<g data-oa-global-route="'+r.code+'"><path id="oa-global-path-'+i+'" d="'+d+'" class="oa-route" pathLength="1" stroke-dasharray="1"/><circle cx="'+r.end[0]+'" cy="'+r.end[1]+'" r="5" fill="#cc7a45"/><path d="M'+r.end.join(' ')+'L'+r.labelX+' '+(r.labelY-5)+'" class="oa-line"/>'+text(r.labelX,r.labelY,r.name,'oa-world-label','text-anchor="middle"')+'<g id="oa-global-carrier-'+i+'">'+(r.mode==='land'?truck('world-land-truck'):use('ship',-66,0))+'</g></g>';
    }).join('')+'<path d="M'+exportOrigin.join(' ')+'L'+pacificPort.join(' ')+'M'+exportOrigin.join(' ')+'L'+atlanticPort.join(' ')+'" class="oa-line" stroke-dasharray="3 4"/></g>'+text(m?280:560,m?376:367,'Canada → international destinations','oa-label','text-anchor="middle" data-oa-global-caption')+(m?text(280,408,'Americas · Europe · Asia · Africa','oa-label','text-anchor="middle" data-oa-global-caption'):'')+text(m?280:560,m?439:389,'Illustrative routes, not shipment volumes or product-mode assignments','oa-small','text-anchor="middle" data-oa-global-caption')+'<g id="oa-global-classification">'+panel(m?25:150,m?467:405,m?510:820,m?125:70,'#fff4e3')+text(m?45:170,m?496:432,'HS6 PRODUCT CODES','oa-micro')+text(m?45:170,m?526:461,'841410 · 841480 · 841490','oa-code')+line(m?'M300 538H330':'M535 442H645')+text(m?355:678,m?558:432,'HS4 HEADING','oa-micro')+text(m?355:678,m?584:461,'8414','oa-code')+'</g></g>';
    // Geography first: validated Canada boundaries, AB highlighted, then actual
    // ranked exports. Late-scene assets are enclosed in their own fading groups.
    const rows=animationData?.headings || [];
    const hx=m?48:615, hy=m?344:124, maxWidth=m?462:440;
    const headingBars=rows.map((h,i)=>{
      let offset=0;
      const width=maxWidth*h[1]/rows[0][1], y=hy+i*(m?67:87);
      return '<g data-oa-ranked-heading>'+text(hx,y-13,'HS4 '+h[0],'oa-code')+text(m?510:1060,y-13,dollars(h[1]),'oa-small','text-anchor="end"')+
        '<rect x="'+hx+'" y="'+y+'" width="'+width+'" height="27" rx="3" fill="#ae401b"/>'+h[3].map((c,j)=>{
          const x=hx+offset, w=width*c[1]/h[1];offset+=w;
          return '<rect data-oa-ranked-segment x="'+x+'" y="'+y+'" width="'+w+'" height="27" fill="'+orange(j,h[3].length)+'"/>';
        }).join('')+'</g>';
    }).join('');
    const hierarchy = '<g id="oa-hierarchy" class="oa-scene"><g id="oa-selection-map" transform="translate('+(m?88:28)+' '+(m?-9:5)+') scale('+(m?.64:.9)+')">'+provinces.map(([abbr])=>'<use href="'+mapFile+'#'+abbr+'" class="oa-province" fill="'+(abbr==='AB'?'#bf5b2f':'#ead8b8')+'"/>').join('')+text(155,278,'AB','oa-origin-label','text-anchor="middle"')+'</g>'+text(m?280:280,m?290:425,'SELECT ALBERTA','oa-label','text-anchor="middle"')+'<g id="oa-ribbon">'+text(hx,m?313:65,'ALBERTA · 2025 DOMESTIC EXPORTS','oa-micro')+headingBars+text(m?280:840,m?558:410,'HS4 totals → ranked HS6 segments','oa-label','text-anchor="middle"')+text(m?280:840,m?590:444,'Observed values · all destinations','oa-small','text-anchor="middle"')+'</g></g>';
    const focus=rows[1], fx=m?30:100, fy=m?344:274, fw=m?500:920;
    let finalOffset=0;
    const focusSegments=focus?focus[3].map((c,j)=>{
      const x=fx+finalOffset,w=fw*c[1]/focus[1];finalOffset+=w;
      return '<g data-oa-final-segment><rect x="'+x+'" y="'+fy+'" width="'+w+'" height="38" fill="'+orange(j,focus[3].length)+'"/>'+ (w>(m?98:84)?text(x+w/2,fy+25,c[0],'oa-micro','text-anchor="middle" style="fill:'+(j===0?'#fff8e8':'#613c29')+'"'):'')+'</g>';
    }).join(''):'';

    const destinationBranch = (animationData?.destinations || []).map(([code,value],i)=>{
      const x=m?36+i*166:170+i*280,y=m?429:350;
      const label=code==='US'?'United States':animationData.countryNames[code];
      return '<path d="M'+(m?280:560)+' '+(fy+38)+'Q'+(m?280:560)+' '+(y-10)+' '+(x+(m?76:110))+' '+y+'" class="oa-line"/>'+panel(x,y,m?154:220,m?43:42,'#fff0df')+text(x+(m?77:110),y+26,label,'oa-small','text-anchor="middle"');
    }).join('');
    const final = '<g id="oa-final" class="oa-scene">'+panel(m?30:100,m?46:68,m?500:310,m?104:128,'#fff0d0')+text(m?52:126,m?78:106,'SELECT GEOGRAPHY','oa-micro')+text(m?52:126,m?121:155,'Alberta','oa-country')+'<g id="oa-final-process">'+(m?'':line('M430 130H530'))+panel(m?30:560,m?173:68,m?500:460,m?109:128,'#ffe3b7')+text(m?52:585,m?205:106,focus?'RANK 2 · HS4 '+focus[0]:'HS4 EXPORT RANKING','oa-micro')+text(m?52:585,m?252:155,focus?dollars(focus[1])+' CAD':'Select a geography','oa-country')+'</g><g id="oa-final-ribbon">'+text(fx,fy-20,focus?'HS4 '+focus[0]+' · '+focus[3].length+' HS6 PRODUCTS':'GEOGRAPHY → HS4 → HS6','oa-label')+(focus?'<rect x="'+fx+'" y="'+fy+'" width="'+fw+'" height="38" fill="#ae401b"/>'+focusSegments:'')+destinationBranch+text(m?280:560,m?491:416,'Select a geography. Rank its HS4 exports.','oa-label','text-anchor="middle"')+text(m?280:560,m?530:452,'Explore the HS6 products behind each heading.','oa-label','text-anchor="middle"')+'</g></g>';
    const loading=rows.length?'':'<g id="oa-data-note">'+text(m?280:840,m?425:210,dataUnavailable?'Observed export data unavailable':'Loading observed Alberta exports…','oa-small','text-anchor="middle"')+'</g>';
    svg.innerHTML=defs+world+`<g id="oa-map-scene" class="oa-scene">${map}${productionCards}</g>`+logistics+tradeNote+border+hierarchy+loading+final;
    refs=Object.fromEntries(['map-scene','geography','us-land','border-line','origin-labels','production-icons','production-cards','trade-routes','country-labels','logistics','trade-note','checkpoint','scanner','gate','hierarchy','final','ribbon','final-ribbon','final-process'].map(id=>[id,get(id)]));
    refs.world=get('world'); refs.worldClassification=get('global-classification');
    refs.globalRoutes=globalRoutes.map((r,i)=>({...r,group:root.querySelector('[data-oa-global-route="'+r.code+'"]'),path:get('global-path-'+i),carrier:get('global-carrier-'+i)}));
    refs.origins=[...root.querySelectorAll('[data-oa-origin]')];
    refs.products=[...root.querySelectorAll('[data-oa-production]')];
    refs.parcels=[...root.querySelectorAll('[data-oa-parcel]')];
    refs.children=[...root.querySelectorAll('[data-oa-ranked-heading]')];
    refs.branches=[...root.querySelectorAll('[data-oa-ranked-segment]')];
    routes=[...root.querySelectorAll('[data-points]')].map((el,i)=>({el,path:get(`route-${i}`),mode:el.dataset.mode,p:el.dataset.points.split(',').map(Number)}));
    render(reduced.matches||still?40:elapsed);
  }
  const phases=[
    [0,'origins','Thirteen origins. One Canada.','Provincial and territorial origins form the Canadian total.','01 / 07 · Origin'],
    [6,'production','Different places. Different products.','Illustrative production, across a connected country.','02 / 07 · Production'],
    [12,'logistics','Products enter the network.','Inland freight connects Canadian production with ports and land corridors.','03 / 07 · Logistics'],
    [19,'trade','Canadian goods travel abroad.','Canada connects with destinations across the Americas, Europe, Asia and Africa.','04 / 07 · Canada → World'],
    [25,'border','Across destinations, products have a hierarchy.','HS6 product codes group goods into HS4 headings.','05 / 07 · Product classification'],
    [32,'hierarchy','One Geography: Different Levels of Exposure','Alberta’s export values and composition differ across HS4 headings.','06 / 07 · Geography → HS4'],
    [37,'final','From Geography to HS4 and HS6 Exports','Select a geography. Rank its headings. Explore their products.','07 / 07 · HS4 → HS6']
  ];
  function render(t) {
    const m=narrow.matches;
    root.dataset.animationTime=t.toFixed(2);
    const copy=[...phases].reverse().find(p=>t>=p[0]);
    if(phase!==copy[1]) {phase=copy[1];root.dataset.phase=phase;headline.textContent=copy[2];subline.textContent=copy[3];kicker.textContent=copy[4];}
    opacity(heading,Math.min(1,...phases.slice(1).map(p=>ease(Math.abs(t-p[0])/.35))));
    opacity(refs['map-scene'],ramp(t,0,.8)*(1-ramp(t,11.4,12.2)));
    opacity(refs.world,windowAlpha(t,11.7,32.2));
    opacity(refs.worldClassification,ramp(t,25.1,26.3));
    refs.globalRoutes.forEach(r=>{
      const reveal=ramp(t,12.3,14.8); opacity(r.group,reveal);
      r.path.style.strokeDashoffset=(1-ramp(t,12.8,16.8)).toFixed(3);
      const u=ramp(t,r.start,r.start+6.4), legIndex=Math.min(r.legs.length-1,Math.floor(u*r.legs.length)), p=r.legs[legIndex], local=u===1?1:u*r.legs.length-legIndex;
      const [x,y]=curvePoint(p,local), facing=mix(p[2]-p[0],p[4]-p[2],local)<0?-1:1;
      move(r.carrier,x,y,r.mode==='land'?.3:.3);
      r.carrier.setAttribute('transform',r.carrier.getAttribute('transform')+' scale('+facing+' 1)');
      opacity(r.carrier,ramp(t,r.start,r.start+.45));
    });
    const camera=ramp(t,18.2,20);
    const logisticsSpace=ramp(t,10.9,11.7)*(1-camera);
    const mapX=(m?mix(-26,48,camera):mix(132,230,camera))+(m?12:72)*logisticsSpace;
    const mapY=(m?mix(0,5,camera):mix(-12,2,camera))+(m?0:12)*logisticsSpace;
    const mapScale=(m?mix(1,.82,camera):mix(1.12,.80,camera))-(m?.04:.28)*logisticsSpace;
    move(refs.geography,mapX,mapY,mapScale);
    refs.origins.forEach((el,i)=>opacity(el,.22+.78*ramp(t,.35+i*.24,1.3+i*.24)));
    opacity(refs['us-land'],ramp(t,18.3,20));
    opacity(refs['border-line'],ramp(t,19.1,20.4));
    opacity(refs['country-labels'],ramp(t,18.4,20));
    opacity(refs['origin-labels'],1-ramp(t,17.8,19));
    opacity(refs['production-icons'],windowAlpha(t,5.5,13.8));
    refs.products.forEach((el,i)=>opacity(el,ramp(t,5.8+i*.18,6.5+i*.18)));
    opacity(refs['production-cards'],1-ramp(t,11.6,12.4));
    opacity(refs.logistics,windowAlpha(t,11.7,19.2));
    refs.logistics.setAttribute('transform',m?'translate(0 58) scale(.88)':'translate(80 150) scale(.65)');
    // The old North-American destination and border groups are kept out of the timeline.
    const roadY=m?536:389,railY=m?604:460;
    const fleet=[0,1,2].map(i=>{
      const el=get(`road-${i}`),scale=m?.67:.9;
      const x=(m?30+i*135:110+i*220)+(m?385:580)*ramp(t,14.9,18.3);
      move(el,x,roadY,scale);opacity(el,windowAlpha(t,12,18.5));
      return {el,x,y:roadY,scale};
    });
    const rail={el:get('inland-train'),x:mix(m?20:95,m?345:930,ramp(t,15.1,18.1)),y:railY,scale:m?.74:1};
    move(rail.el,rail.x,rail.y,rail.scale);
    move(get('coastal-ship'),mix(m?460:975,m?394:882,ramp(t,15.9,19)),m?552:402,m?.7:1);
    get('coastal-ship').setAttribute('transform',get('coastal-ship').getAttribute('transform')+' scale(-1 1)');
    opacity(get('coastal-ship'),windowAlpha(t,15.5,19.2));
    refs.parcels.forEach((el,i)=>{
      const a=10.4+i*.04,u=ramp(t,a,a+1.1),[,,x,y]=provinces[i];
      const carrier=i<9?fleet[i%3]:rail;
      const slotX=i<9?12+Math.floor(i/3)*20:[17,37,69,125][i-9],slotY=-15,cargoScale=i<9?.8:.85;
      const targetX=carrier.x+slotX*carrier.scale,targetY=carrier.y+slotY*carrier.scale;
      const loaded=u>=1,parent=loaded?carrier.el:refs.logistics;
      if(el.parentNode!==parent) parent.appendChild(el);
      if(loaded) move(el,slotX,slotY,cargoScale);
      else move(el,mix(mapX+x*mapScale,targetX,u),mix(mapY+y*mapScale,targetY,u),mix(1,cargoScale*carrier.scale,u));
      opacity(el,ramp(t,a-.25,a));
      el.dataset.loaded=String(loaded);el.dataset.freight=carrier.el.id;
    });
    opacity(refs['trade-note'],0);
    opacity(refs['trade-routes'],0);
    routes.forEach(({el,path,mode,p},i)=>{
      const start=19.5+i*.3,u=ramp(t,start,23.9+i*.08),[sx,sy,cx,cy,ex,ey]=p;
      path.style.strokeDashoffset=(1-ramp(t,start,start+1.1)).toFixed(3);
      if(mode==='truck') {
        const [a,b,c,d]=truckLegs[i],incoming=u<.5,from=incoming?a:c,to=incoming?b:d;
        const travel=incoming?ramp(u,0,.38):ramp(u,.62,1);
        const facing=to[0]<from[0]?-1:1;
        const tilt=Math.atan2(facing*(to[1]-from[1]),Math.abs(to[0]-from[0]))*180/Math.PI;
        move(el,mix(from[0],to[0],travel),mix(from[1],to[1],travel),.42,Math.max(-12,Math.min(12,tilt)));
        el.setAttribute('transform',el.getAttribute('transform')+` scale(${facing} 1)`);
        // Inspection masks the short handoff between the two driving lanes.
        opacity(el,windowAlpha(t,start,24.6)*((1-ramp(u,.38,.46))+ramp(u,.54,.62)));
      }else{
        const x=(1-u)**2*sx+2*(1-u)*u*cx+u*u*ex,y=(1-u)**2*sy+2*(1-u)*u*cy+u*u*ey;
        const facing=mix(cx-sx,ex-cx,u)<0?-1:1;
        move(el,x,y,mode==='train'?.23:mode==='ship'?.37:.42);
        el.setAttribute('transform',el.getAttribute('transform')+` scale(${facing} 1)`);
        opacity(el,windowAlpha(t,start,24.6));
      }
    });
    opacity(refs.checkpoint,0);
    const bx=m?226:485,by=m?180:110;
    [0,1,2].forEach(i=>{move(get(`inspection-${i}`),mix(m?-95:-100,bx-90,ramp(t,25.1+i*.3,26.5+i*.3)),m?180+i*65:165+i*91,m?.6:.8);opacity(get(`inspection-${i}`),1-ramp(t,27.1,28.2));});
    let screening=0;
    cargo.forEach(([,matched],i)=>{
      const start=26+i*.46,u=ramp(t,start,start+.65),v=ramp(t,start+.7,start+1.5);
      const x=mix(m?135:330,bx+(m?40:77),u),y=mix(m?235:210,by+82,u);
      const slot=cargo.slice(0,i).filter(c=>c[1]===matched).length;
      // Fill the far position first, so later tiles do not cross settled ones.
      const targetX=m?(405+(slot===0?1:0)*79):(844+(2-slot)*89);
      const targetY=matched?(m?187+Math.floor(slot/2)*49:204):(m?437+Math.floor(slot/2)*49:390);
      const product=get(`screen-tile-${i}`),classification=ramp(t,start+.5,start+.75);
      // Reach the lane's product row before entering its labelled panel.
      move(product,mix(x,targetX,v),mix(y,targetY,ramp(t,start+.65,start+1.1)),m?.77:.95);
      opacity(product,ramp(t,start-.3,start));
      product.firstElementChild.setAttribute('fill',matched&&classification>.5?'#f1d8c7':'#e5ece1');
      product.firstElementChild.setAttribute('stroke',matched&&classification>.5?'#b75c38':'#86a68e');
      opacity(product.querySelector('.oa-stamp'),.14*classification);
      screening=Math.max(screening,windowAlpha(t,start+.15,start+.8));
    });
    opacity(refs.scanner,.4+.6*screening);
    refs.gate.setAttribute('transform',`translate(${bx+12} ${by+(m?171:246)}) rotate(${-63*ramp(t,26.5,27)*(1-ramp(t,30,30.7))})`);
    opacity(refs.hierarchy,windowAlpha(t,31.8,37.5));
    refs.children.forEach((el,i)=>opacity(el,ramp(t,33.1+i*.2,33.8+i*.2)));
    opacity(get('selection-map'),1-.3*ramp(t,34.2,35.3));
    refs.branches.forEach(el=>opacity(el,ramp(t,34.5,35.5)));
    root.querySelectorAll('[data-oa-final-segment]').forEach(el=>opacity(el,ramp(t,38.4,39.4)));
    opacity(refs.ribbon,ramp(t,33,34));
    opacity(refs.final,ramp(t,37,38));
    opacity(refs['final-process'],ramp(t,37.4,38.4));
    opacity(refs['final-ribbon'],ramp(t,38,39));
    // Wheel motion derives from the same timeline, so Pause also stops wheels.
    root.querySelectorAll('[data-oa-wheel]').forEach(w=>w.firstElementChild.setAttribute('transform',`rotate(${t*240})`));
    root.querySelector('[data-oa-seek]').value=t;
    progress.style.transform=`scaleX(${clamp(t/duration)})`;
  }
  function tick(now) {
    if(last!==null) elapsed=Math.min(duration,elapsed+Math.min((now-last)/1000,.12)*speed);
    last=now;
    if(now-lastPaint>=30 || elapsed===duration) {render(elapsed);lastPaint=now;}
    if(elapsed>=duration) {finished=true;frame=0;last=null;reconcile();return;}
    frame=requestAnimationFrame(tick);
  }
  function reconcile() {
    const run=!reduced.matches&&!still&&!userPaused&&!finished&&visible&&!document.hidden;
    if(run&&!frame) {last=null;frame=requestAnimationFrame(tick);}
    else if(!run&&frame) {cancelAnimationFrame(frame);frame=0;last=null;}
    pause.disabled=finished;
    pause.textContent=finished?'Complete':userPaused?'Resume':'Pause';
    pause.setAttribute('aria-label',finished?'Animation complete':`${userPaused?'Resume':'Pause'} geography and export composition animation`);
    pause.setAttribute('aria-pressed',String(userPaused));
    root.dataset.playback=reduced.matches||still?'static':finished?'complete':userPaused?'paused':run?'playing':'suspended';
  }
  function staticFrame() {render(40);controls.hidden=reduced.matches;note.hidden=false;}
  pause.addEventListener('click',()=>{userPaused=!userPaused;reconcile();});
  replay.addEventListener('click',()=>{
    if(frame) cancelAnimationFrame(frame);
    frame=0;elapsed=0;last=null;lastPaint=0;userPaused=false;finished=false;
    render(reduced.matches||still?40:0);reconcile();
  });
  document.addEventListener('visibilitychange',reconcile);
  if('IntersectionObserver' in window) new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;reconcile();},{threshold:.08}).observe(root);
  narrow.addEventListener('change',layout);
  reduced.addEventListener('change',()=>{
    if(reduced.matches||still) staticFrame();
    else {controls.hidden=false;note.hidden=true;elapsed=0;userPaused=false;finished=false;render(0);}
    reconcile();
  });
  root.querySelector('[data-oa-speed]').addEventListener('change',event=>{speed=Number(event.target.value);});
  root.querySelector('[data-oa-seek]').addEventListener('input',event=>{
    elapsed=Math.max(0,Math.min(duration,Number(event.target.value)));
    userPaused=true;finished=elapsed>=duration;render(reduced.matches||still?40:elapsed);reconcile();
  });
  root.querySelector('[data-oa-motion]').addEventListener('change',event=>{
    still=event.target.checked;
    if(still) staticFrame(); else {note.hidden=true;render(elapsed);}
    reconcile();
  });
  fetch(dataFile).then(response=>{if(!response.ok)throw new Error('Animation data HTTP '+response.status);return response.json();}).then(data=>{
    if(data.geography!=='Alberta'||data.headings.length!==3)throw new Error('Invalid animation dataset');
    animationData=data;layout();
  }).catch(error=>{dataUnavailable=true;layout();console.error(error);});
  layout();
  root.querySelector('.oa-fallback').hidden=true;
  svg.removeAttribute('hidden');
  if(reduced.matches) staticFrame(); else {controls.hidden=false;note.hidden=true;render(0);}
  reconcile();
})();
