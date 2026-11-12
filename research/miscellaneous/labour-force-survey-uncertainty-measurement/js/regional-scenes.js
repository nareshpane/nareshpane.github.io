/* Regional variants extend the existing scene graph and use its single clock.
 * Coordinates use the existing Natural Earth Lambert projection, rounded anchors.
 * Landscapes are conceptual settings, not actual PSU boundaries or respondent locations. */
(() => {
  'use strict';
  const A=window.LFSArt,T=A.text;
  const regions=[
    {id:'toronto',geo:'ON',point:[363.54,375.59],label:'Toronto · high-density urban housing',box:[245,267,165,137]},
    {id:'north-ontario',geo:'ON',point:[321.44,324.58],label:'Northern Ontario · sparsely settled areas',box:[245,267,165,137]},
    {id:'vancouver',geo:'BC',point:[92.13,304.01],label:'Vancouver · coastal urban density',box:[50,198,115,132]},
    {id:'kelowna',geo:'BC',point:[113.22,306.6],label:'Kelowna · interior urban and suburban housing',box:[50,198,115,132]},
    {id:'regina',geo:'SK',point:[199.75,323.7],label:'Regina · urban residential areas',box:[164,230,65,110]},
    {id:'rural-saskatchewan',geo:'SK',point:[193.22,312.94],label:'Rural Saskatchewan · dispersed dwellings',box:[164,230,65,110]},
    {id:'edmonton',geo:'AB',point:[156.32,285.58],label:'Edmonton · mixed-density urban neighbourhoods',box:[121,220,67,113]},
    {id:'pei',geo:'PE',point:[454.8,312],label:'PEI · direct dwelling selection within strata',box:[425,292,65,37]},
    {id:'yukon',geo:'YT',point:[102.4,155.6],label:'Yukon · eight interviews, three months apart',box:[48,113,129,73]}
  ];
  function road(d,w=32){return `<path d="${d}" fill="none" stroke="#c6cec5" stroke-width="${w+8}"/><path d="${d}" fill="none" stroke="#f9f5ea" stroke-width="${w+12}"/><path d="${d}" fill="none" stroke="#c6cec5" stroke-width="${w}"/><path d="${d}" fill="none" stroke="#fffaf1" stroke-width="2" stroke-dasharray="12 16"/>`;}
  const trees=(points,pine=false)=>points.map(([x,y,s=.7])=>A.tree(x,y,s,pine)).join('');
  // Taller, narrower towers keep the established window/dwelling illustration.
  function tower(x,y,width,height){return `<g class="urban-tower" transform="translate(${x} ${y}) scale(${width/96} ${height/205})">${A.building(0,0,'tower')}</g>`;}
  function streetDetails(y){return `<g class="urban-street-details" stroke="#faf6e9" fill="none"><path d="M462 ${y-21}v42m10-42v42m43-42v42m10-42v42" stroke-width="5"/><path d="M0 ${y-30}H1000M0 ${y+30}H1000" stroke-width="3"/></g><g stroke="#6f8d81" stroke-width="2" fill="none"><path d="M26 ${y-37}v-37h13M968 ${y+34}v-37h-13"/><rect x="127" y="${y-19}" width="34" height="13" rx="4" fill="#8ca59d"/><rect x="736" y="${y+7}" width="30" height="13" rx="4" fill="#c5a681"/></g>`;}
  function landscape(id){
    let s='<rect width="1000" height="540" rx="12" fill="#edf2e7"/>';
    if(id==='toronto'){
      s+='<rect width="1000" height="150" fill="#e2eeee"/><path d="M60 150V63H112V150M135 150V32H180V150M210 150V85H260V150M510 150V55H570V150M630 150V18H680V150M725 150V65H790V150" fill="#c2d1cc"/>';
      s+='<path d="M400 146L413 38L426 146M412 38V8" stroke="#809993" stroke-width="4" fill="none"/><ellipse cx="413" cy="55" rx="15" ry="5" fill="#809993"/>';
      s+=road('M0 326H1000M485 150V540',36);
      [[55,290,88,245],[195,290,86.4,184.5],[318,290,74,228],[402,290,50,160],[570,290,89,261],[688,290,65,210],[774,290,62,174],[859,290,77,233]].forEach(([x,y,w,h])=>s+=tower(x,y,w,h));
      [[55,510,70,143],[151,510,66,142],[246,510,72,140],[605,510,82,145],[721,510,64,140],[814,510,61,143],[909,510,58,134]].forEach(([x,y,w,h])=>s+=tower(x,y,w,h));
      s+=streetDetails(326);
      s+='<rect x="353" y="382" width="93" height="130" rx="14" fill="#c4dab2"/>'+trees([[395,441,.7],[935,510,.65],[535,282,.6]]);
      s+=A.human(442,270,0,.4)+A.human(536,371,2,.4)+A.human(951,406,1,.45);
      s+='<rect x="203.1" y="138.3" width="15.3" height="13.5" rx="2" fill="#efbd70" stroke="#b66d3e"/><path d="M40 62H438V305H40Z" fill="none" stroke="#7ea691" stroke-dasharray="8 5" stroke-width="2"/>';
    }else if(id==='north-ontario'){
      s+='<path d="M0 60L150 19L360 74L550 30L790 69L1000 15V200H0Z" fill="#c5d8ce"/><path d="M0 139Q300 89 500 140T1000 123V540H0Z" fill="#f9fbf6"/>';
      s+=trees([[25,175,1],[100,195,.9],[179,170,1.1],[269,195,.9],[355,172,1],[444,184,.8],[542,175,1.1],[635,183,.9],[730,178,1],[825,168,.85],[925,175,1.2],[971,195,.8]],true);
      s+=road('M0 386Q240 330 445 395T1000 368',12);
      [[135,332,.7],[815,330,.65]].forEach(([x,y,z])=>{s+=A.building(x,y,'house',z);s+=`<path d="M${x-8} ${y-64*z}L${x+50*z} ${y-108*z}L${x+106*z} ${y-64*z}" stroke="#fffdf5" stroke-width="7" fill="none"/>`;});
      s+='<path d="M183 332L199 356M844 330L865 372" stroke="#dee5de" stroke-width="5"/>';
      s+=trees([[320,505,1.1],[690,478,1],[905,498,.9],[49,496,1.3]],true);
    }else if(id==='vancouver'){
      s+='<rect width="1000" height="160" fill="#dcebea"/><path d="M0 154L135 35L229 130L353 22L478 130L574 41L700 141L830 33L1000 142V220H0Z" fill="#bacfc8"/><path d="M110 57L135 35L166 69L142 62ZM325 49L353 22L386 54L357 45Z" fill="#f4f8ee"/><path d="M0 180H1000V285H0Z" fill="#bad7d9"/><path d="M0 222Q180 204 370 228T760 219T1000 225" fill="none" stroke="#d9ece7" stroke-width="6"/>';
      s+=road('M0 340H1000M520 290V540',30);
      [[45,311,83,238],[157,311,74,286],[265,311,65,197],[362,311,72,251],[593,311,79,238],[702,311,63,194],[793,311,68,272],[891,311,66,217],[58,525,72,143],[165,525,65,148],[265,525,67,137],[365,525,58,145],[601,525,73,148],[704,525,57,137]].forEach(([x,y,w,h])=>s+=tower(x,y,w,h));
      s+=streetDetails(340);
      s+='<rect x="790" y="398" width="155" height="120" rx="15" fill="#c5dcb8"/>'+trees([[837,470,.8],[920,445,.6],[576,480,.8]]);
      s+=A.human(421,421,4,.45)+A.human(554,372,3,.43);
    }else if(id==='kelowna'){
      s+='<path d="M0 120Q125 5 300 104T660 105T1000 75V260H0Z" fill="#d1dec3"/><path d="M0 160Q255 110 490 166L320 280H0Z" fill="#c2dedd"/><path d="M20 200Q175 167 335 193" fill="none" stroke="#e7f0e4" stroke-width="6"/>';
      s+='<path d="M35 253H363V341H35ZM658 220H972V339H658ZM28 406H376V525H28ZM639 410H978V525H639Z" fill="#dbe8ce"/>';
      s+=road('M0 368H1000M565 540V272Q565 248 542 248H477',19)+'<circle cx="474" cy="248" r="32" fill="#f8f5e9"/><circle cx="474" cy="248" r="26" fill="#c6cec5"/>';
      [[76,332,'house',.8],[259,332,'townhouse',.74],[686,332,'midrise',.58],[867,332,'house',.75],[62,502,'duplex',.76],[266,502,'house',.8],[658,502,'townhouse',.76],[865,502,'house',.75]].forEach(([x,y,k,z])=>s+=A.building(x,y,k,z));
      s+=trees([[187,303,.55],[382,500,.7],[789,317,.6],[800,490,.65],[619,283,.7]]);
      s+='<rect x="400" y="429" width="110" height="90" rx="15" fill="#c6ddb5"/>'+trees([[445,463,.55],[45,290,.65],[489,314,.65],[619,490,.65],[960,487,.75]])+A.human(522,321,0,.4);
    }else if(id==='regina'){
      s+='<path d="M0 70H1000V150H0Z" fill="#e2e6cc"/><path d="M60 160V108H120V160M170 160V62H210V160M255 160V92H318V160M405 160V80H465V160M522 160V35H560V160M610 160V73H674V160" fill="#c1cdbb"/>';
      s+=road('M0 344H1000M490 157V540',29);
      [[65,307,'midrise',1],[231,307,'midrise',.8],[344,307,'house',.9],[590,308,'midrise',1],[780,308,'townhouse',1],[65,503,'duplex',1],[230,503,'house',1],[590,503,'house',.95],[807,503,'duplex',.9]].forEach(([x,y,k,z])=>s+=A.building(x,y,k,z));
      s+='<rect x="345" y="411" width="104" height="97" rx="13" fill="#c8dcb4"/>'+trees([[392,458,.65],[963,479,.8],[552,305,.6],[26,302,.6]])+A.human(760,381,1,.44);
    }else if(id==='rural-saskatchewan'||id==='pei'||id==='yukon'){
      if(id==='yukon'){s=A.landscape('rural');s+=trees([[30,182,1],[195,177,1.2],[484,174,1.1],[804,180,1.2]],true);}
      else{s+='<path d="M0 90H1000V540H0Z" fill="#e9e3bc"/><path d="M0 165H1000M0 250H1000M0 475H1000M270 90L190 540M660 90L730 540" stroke="#d2ce9d" stroke-width="15"/>';s+=road('M0 367Q480 295 1000 403',id==='rural-saskatchewan'?10:15);s+=A.building(90,310,'house',.65)+A.building(665,340,'house',.68)+(id==='pei'?A.building(390,491,'duplex',.72):'');s+='<path d="M820 276L865 240L910 276V334H820Z" fill="#bd8060" stroke="#8c7057"/><rect x="835" y="286" width="55" height="48" fill="#e2c9a5"/>'+trees([[235,303,.6],[608,324,.7],[508,490,.7],[948,335,.8]]);}
    }else if(id==='edmonton'){
      s+='<path d="M0 94Q240 43 490 110T1000 90V190H0Z" fill="#cfdec8"/><path d="M300 146V21H347V146M370 146V43H421V146M450 146V69H491V146M537 146V12H586V146M610 146V57H660V146" fill="#bbcdd0"/>';
      s+=road('M0 335H1000M485 140V540',43);
      s+='<path d="M32 396H434V530H32ZM565 396H978V530H565Z" fill="#dce8d0"/>';
      [[62,286,'tower',.8],[210,286,'midrise',1],[337,286,'tower',.72],[592,286,'midrise',.95],[758,286,'townhouse',1],[72,512,'house',.8],[284,512,'duplex',.8],[610,512,'townhouse',.8],[862,512,'house',.8]].forEach(([x,y,k,z])=>s+=A.building(x,y,k,z));
      s+=trees([[223,479,.7],[788,480,.65],[958,518,.65]]);
      s+='<rect x="843" y="85" width="110" height="70" rx="14" fill="#c0d8b0"/>'+trees([[896,151,.5],[411,500,.7],[561,472,.65],[18,472,.7]])+A.human(555,389,2,.43)+A.human(426,277,4,.4);
    }
    return s;
  }
  function build(camera){
    camera.insertAdjacentHTML('beforeend',`<g id="film-regions">${regions.map(r=>`<g id="film-region-${r.id}" transform="translate(${r.point[0]-15} ${r.point[1]-8.1}) scale(.03)" opacity="0"><title>${r.label}</title>${landscape(r.id)}</g>`).join('')}</g>`);
  }
  function state(t,ramp,mix){
    const index=Math.max(0,Math.min(8,Math.floor((t-5)/5))),r=regions[index],p=t-(5+index*5),previous=index?regions[index-1].box:[0,0,570,440];
    let box;
    if(t<5)box=[0,0,570,440].map((v,j)=>mix(v,regions[0].box[j],ramp(t,3.5,5)));
    else if(p<.6)box=previous.map((v,j)=>mix(v,r.box[j],ramp(p,0,.6)));
    else{const close=[r.point[0]-18,r.point[1]-10,36,20.2],u=ramp(p,.6,1.6)*(1-ramp(p,4,4.7));box=r.box.map((v,j)=>j>1?Math.exp(mix(Math.log(v),Math.log(close[j]),u)):mix(v,close[j],u));}
    for(let i=0;i<regions.length;i++)document.getElementById('film-region-'+regions[i].id).setAttribute('opacity',t>=5&&i===index?(ramp(p,.7,1.5)*(1-ramp(p,4,4.65))).toFixed(4):0);
    const fade=t<5?0:ramp(p,.8,1.5)*(1-ramp(p,4,4.6));
    document.getElementById('film-map').setAttribute('opacity',1-fade);
    document.getElementById('film-map-labels').setAttribute('opacity',1-ramp(p,.4,1.0));
    return {box,geo:t<5?'Canada':r.geo,region:t<5?'Canada':r.id};
  }
  window.LFSRegions={regions,landscape,build,state};
})();
