/* A single 130.5-second clock owns camera, selection, people and records.
 * SVG nodes are built once; render(t) restores every animated property.
 * All sampling boundaries/records are conceptual, not production geography. */
(() => {
  'use strict';
  const root=document.getElementById('lfs-journey');if(!root)return;
  const A=window.LFSArt,svg=document.getElementById('journey-scene');
  const $=id=>document.getElementById(id),T=A.text;
  const reduced=matchMedia('(prefers-reduced-motion: reduce)'),narrow=matchMedia('(max-width:600px)');
  const openingTrim=2.5;
  const scenes=[
    [0,'Canada: a geography of households','The ten provinces contribute to national monthly totals. Territorial results are separate.'],
    [5,'Toronto · high-density urban housing','Many apartment dwellings occupy a compact metropolitan area. A province contains several settlement environments, not just the one shown here.'],
    [10,'Northern Ontario · sparsely settled areas','Forests, distance and small roads change collection logistics. This illustrative winter setting does not imply universal frame coverage or year-round snow.'],
    [15,'Vancouver · coastal urban density','Coastal residential towers, parks and metropolitan streets illustrate one environment within British Columbia.'],
    [20,'Kelowna · interior urban and suburban housing','An Interior BC city combines mid-rise apartments, townhouses and detached homes. This setting is distinct from the coastal metropolis.'],
    [25,'Regina · urban residential areas','Urban apartments, residential streets and parks also form part of Saskatchewan’s dwelling frame.'],
    [30,'Rural Saskatchewan · dispersed dwellings','Prairie farmhouses and small communities illustrate longer distances between dwellings within the same province.'],
    [35,'Edmonton · mixed-density urban neighbourhoods','A substantial metropolitan setting combines downtown towers, wider roads, apartments and surrounding residential communities.'],
    [40,'Prince Edward Island: one stage','Dwellings are selected directly from address lists within strata; there is no first-stage geographic PSU draw.'],
    [45,'Yukon: a different rotation','Dwellings are interviewed every three months for two years: eight interviews, with about one-eighth entering each quarter.'],
    ...[
    [25,'Ontario: from strata to a neighbourhood','Domains and strata organize the frame. The outlined neighbourhood is an illustrative geographic PSU.'],
    [33,'A building contains dwellings','A selected PSU contains many dwellings. Illuminated windows illustrate selected dwellings, not whole selected buildings.'],
    [41,'Six panels, one monthly handover','Each selected dwelling remains in the provincial sample for six consecutive months. The outgoing panel completes interview 6/6; five panels stay as a new panel enters. Household composition can change within a retained dwelling.'],
    [53,'Inside a selected dwelling','Usual residents form a household. Labour force information concerns eligible civilian members aged 15+; household composition varies.'],
    [61,'Questions become records','Visits and telephone interviews are computer assisted; eligible households can complete an electronic questionnaire. A knowledgeable respondent may report by proxy.'],
    [72,'Missing values need treatment','Carry-forward uses an applicable prior value; a compatible donor supplies another record; a logical rule can determine status. Imputed values are estimates.'],
    [84,'Weights, diagnostics and uncertainty','Final person weights, calibration and design-aware variance estimation support totals, rates and quality review.'],
    [96,'From official release to public discussion','Statistics Canada publishes monthly labour market indicators. Media, researchers, policymakers and the public interpret and discuss the release.'],
    [104,'How reliable are these estimates?','Continue into the article: sampling variance measures one source of uncertainty; nonsampling errors require separate scrutiny.']
    ].map(([t,title,caption])=>[t+25,title,caption])
  ].map(([t,title,caption])=>[t===0?0:t-openingTrim,title,caption]);
  const duration=133-openingTrim,clamp=x=>Math.max(0,Math.min(1,x)),ease=x=>{x=clamp(x);return x*x*x*(x*(x*6-15)+10);};
  const ramp=(t,a,b)=>ease((t-a)/(b-a)),mix=(a,b,u)=>a+(b-a)*u;
  const alpha=(el,v)=>el.setAttribute('opacity',clamp(v).toFixed(4));
  const move=(el,x,y,s=1)=>el.setAttribute('transform',`translate(${x.toFixed(3)} ${y.toFixed(3)}) scale(${s.toFixed(5)})`);
  const geo=[['BC',108.5,253.6],['AB',154.9,271.3],['SK',196.9,290.3],['MB',240,285.2],['ON',314.2,316.6],['QC',385.2,276.8],['NB',434.8,320.5],['NS',461.9,322.1],['PE',455.3,310.6],['NL',448.3,242.7],['YT',102.4,155.6],['NT',157.5,176.2],['NU',253.8,158.5]];
  const map=geo.map(([c,x,y],i)=>`<path id="film-geo-${c}" d="${window.LFS_MAP[c]}" fill="${['#c7d9be','#ddd0b4','#c6d7cf','#d6d0dd'][i%4]}" stroke="#819989" stroke-width=".7"/>`).join('');
  const labels=geo.filter(g=>!['NB','NS','PE'].includes(g[0])).map(([c,x,y])=>T(x,y,c,10,'text-anchor="middle"')).join('');
  const record=(id,label='record')=>`<g id="${id}"><rect width="95" height="115" rx="8" fill="#fffaed" stroke="#a4b8a1" stroke-width="2"/><path d="M14 27H80M14 44H80M14 61H68M14 78H75" stroke="#a1b59d" stroke-width="4"/>${T(12,102,label,14)}</g>`;
  const arrow='<defs><marker id="film-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="none" stroke="#b96643" stroke-width="1.6"/></marker></defs>';
  let art=`<title id="journey-scene-title">A dwelling's journey through the Labour Force Survey</title><desc id="journey-scene-desc">A continuous camera travels from Canada to a conceptual Ontario neighbourhood, selected dwellings, household interviews, imputation, analytical review and publication. Use the bottom controls to explore still scenes.</desc>${arrow}<svg id="film-camera" x="0" y="0" width="1000" height="560" viewBox="0 0 570 440"><g id="film-map">${map}<g id="film-map-labels">${labels}</g><g id="film-domains"><path d="M282 303L314 303L329 345L295 339Z" fill="#eec68b66" stroke="#b38c50"/><path d="M329 345L353 344L369 373L344 390Z" fill="#92bfa366" stroke="#699780"/><path d="M299 308L324 321M304 324L329 334M331 356L357 357M339 366L361 369" stroke="#fff7e7" stroke-width="1"/></g><circle id="film-focus" cx="350" cy="388" r="4" fill="none" stroke="#b96643" stroke-width="1"/></g>
    <g transform="translate(320 372) scale(.06)" id="film-neighbourhood">${A.landscape('urban','film-city')}<path id="film-psu" d="M555 150H935V305H555Z" fill="none" stroke="#26766b" stroke-width="5" stroke-dasharray="14 8"/><g id="film-unit-labels">${T(560,134,'Selected geographic PSU',22)}${T(755,316,'Dwelling ≠ building',22)}</g>
    <g id="film-rotation"><rect width="1000" height="540" rx="14" fill="#f6f0df"/>${T(55,58,'Selected dwelling panels',28)}${T(55,97,'5/6 retained · 1/6 replaced',25)}${Array.from({length:7},(_,i)=>`<g id="film-panel-${i}"><rect width="130" height="230" rx="14" fill="${i===6?'#bf7650':A.palette[i]}"/>${T(16,34,i===6?'New panel':`Panel ${String.fromCharCode(65+i)}`,20,'data-light="true"')}${A.building(20,143,i%2?'duplex':'house',.85)}${A.building(34,203,i%2?'midrise':'house',.55)}</g>`).join('')}${T(55,455,'One symbol = an illustrative dwelling',22)}${T(55,490,'Panels are sample groups, not population fractions',20)}<text id="film-month" x="760" y="58" font-size="24"></text></g>
    <g id="film-interior"><rect width="1000" height="540" rx="14" fill="#f5e7cf"/><rect y="376" width="1000" height="164" fill="#e0c4a0"/><path d="M0 405H1000M0 459H1000M150 377L110 540M610 377L630 540" stroke="#c6a883" stroke-width="2"/><rect x="74" y="80" width="170" height="183" rx="8" fill="#d9e9e7" stroke="#fff6e6" stroke-width="12"/><path d="M158 80V262M75 170H242" stroke="#fff6e6" stroke-width="6"/>${A.tree(189,245,.8)}<rect x="570" y="65" width="270" height="145" rx="6" fill="#fff8e9" stroke="#ccb394"/>${T(595,100,'Household roster',23)}${T(595,140,'Usual residents',22)}${T(595,179,'Eligible members: 15+',22)}<rect x="80" y="331" width="380" height="84" rx="18" fill="#a8b6a0"/><rect x="85" y="304" width="365" height="57" rx="16" fill="#bdc8af"/>
    <g id="film-family">${A.human(145,273,0,1.15)}${A.human(238,283,1,1.03)}${A.human(318,303,3,.8)}${A.human(390,325,2,.58)}</g><g id="film-visitor">${A.human(650,272,4,1.3,'thoughtful')}${A.monitor(684,321,2)}<rect x="639" y="395" width="180" height="12" rx="5" fill="#a58862"/></g><g id="film-bubble"><path d="M490 215H935V269H604L575 290L580 269H490Z" fill="#fffaf0" stroke="#c5ad8c"/><text id="film-question" x="511" y="250" font-size="25"></text></g><g id="film-methods">${T(70,475,'Visit · telephone · electronic questionnaire',25)}${T(70,512,'One respondent may report by proxy',22)}</g></g>
    <g id="film-household-types"><rect width="1000" height="540" fill="#fbf3e5"/>${T(36,55,'Different households, the same eligibility rules',27)}${[[1,'Living alone'],[2,'Couple'],[3,'One child'],[5,'Several children'],[4,'Multigenerational'],[3,'Shared apartment']].map(([n,label],i)=>{const x=35+i%3*325,y=105+Math.floor(i/3)*210;return `<g transform="translate(${x} ${y})"><rect width="285" height="185" rx="12" fill="${i%2?'#e5ece0':'#f0e1cc'}" stroke="#cab997"/>${T(16,30,label,21)}${Array.from({length:n},(_,j)=>A.human(38+j*48,64,(i+j)%6,j>=2&&i!==4&&i!==5?.67:.95,i===4?'thoughtful':'calm')).join('')}</g>`;}).join('')}</g>
    <g id="film-processing"><rect width="1000" height="540" rx="14" fill="#edf2e9"/>${T(40,49,'Editing and imputation',28)}<rect x="32" y="85" width="460" height="340" rx="12" fill="#fff9ed" stroke="#bac6ad"/>${T(54,123,'Record',23)}${T(201,123,'Field',23)}${T(356,123,'Value',23)}<path d="M46 143H475M46 217H475M46 291H475M46 365H475" stroke="#d4dec9"/>${T(54,188,'A · current',21)}${T(201,188,'usual hours',21)}<text id="film-value-0" x="356" y="188" font-size="24">?</text>${T(54,261,'B · recipient',21)}${T(201,261,'usual hours',21)}<text id="film-value-1" x="356" y="261" font-size="24">?</text>${T(54,337,'C · jobholder',21)}${T(201,337,'status',21)}<text id="film-value-2" x="356" y="337" font-size="20">?</text>
    <g id="film-prior"><rect x="555" y="113" width="405" height="71" rx="9" fill="#dce9e0"/>${T(574,142,'Previous month: 35 hours',23)}${T(574,169,'Carry-forward · if applicable',18)}<path d="M555 153H463" stroke="#b96643" stroke-width="3" marker-end="url(#film-arrow)"/></g><g id="film-donor"><rect x="555" y="221" width="405" height="71" rx="9" fill="#e2deeb"/>${T(574,250,'Compatible donor: 30 hours',23)}${T(574,278,'Donor draw · not the unknown truth',18)}<path d="M555 261H463" stroke="#b96643" stroke-width="3" marker-end="url(#film-arrow)"/></g><g id="film-rule"><rect x="555" y="330" width="405" height="81" rx="9" fill="#f0e0ca"/>${T(574,359,'Job + 0 hours + vacation',23)}${T(574,389,'Logical rule → employed, absent',21)}<path d="M555 366H463" stroke="#b96643" stroke-width="3" marker-end="url(#film-arrow)"/></g>${T(40,482,'Observed / edited / imputed flags remain attached',24)}${T(40,519,'A completed field can still contain error',22)}</g>
    <g id="film-analysis"><rect width="1000" height="540" rx="14" fill="#edf1e8"/><rect y="405" width="1000" height="135" fill="#dfe2d2"/><rect x="230" y="55" width="530" height="174" rx="10" fill="#fff9eb" stroke="#c6ceb7"/>${T(258,93,'Population controls · variance · quality',24)}${T(264,136,'T̂ = Σ wᵢyᵢ',27)}${T(474,136,'estimate ± 1.96 SE',24)}<path d="M272 177H425M272 169V185M425 169V185" stroke="#507a9b" stroke-width="3"/><circle cx="350" cy="177" r="5" fill="#b96643"/>${T(475,183,'Diagnostics before release',21)}${[0,1,2].map((i)=>{const x=50+i*330;return `<g id="film-analyst-${i}">${A.human(x+45,302,i,1.2,i===1?'surprise':'thoughtful')}${A.monitor(x+128,291,i)}<rect x="${x}" y="390" width="272" height="12" rx="4" fill="#ae9675"/><path d="M${x+20} 402V484M${x+250} 402V484" stroke="#ae9675" stroke-width="8"/></g>`;}).join('')}<g id="film-discussion">${A.human(790,140,4,1.05,'thoughtful')}${A.human(900,154,5,.95)}</g>${T(48,523,'Final weights ≠ household size · design-aware uncertainty',22)}</g>
    <g id="film-release"><rect width="1000" height="540" rx="14" fill="#f7f0e2"/><rect x="300" y="45" width="400" height="110" rx="12" fill="#d5e4d6" stroke="#8ea58f"/><path d="M328 70L337 78L345 63L348 81L362 76L356 90L369 96L350 101L350 111H341V101L325 96L335 89Z" fill="#b96643"/>${T(384,88,'Statistics Canada',26)}${T(329,128,'Monthly labour market release',23)}${[['News websites','web'],['Television','tv'],['Radio','radio'],['Newspapers','paper'],['Social media','social'],['Research / policy','book']].map(([s,type],i)=>{const x=42+(i%3)*325,y=200+Math.floor(i/3)*137;let icon=type==='tv'?'<rect width="65" height="43" rx="6" fill="#507a9b"/><rect x="6" y="6" width="53" height="29" fill="#e9f3eb"/><path d="M24-10L33 0L42-10M22 48H45" stroke="#507a9b" stroke-width="3"/>':type==='radio'?'<rect width="65" height="40" rx="6" fill="#b89a6b"/><circle cx="18" cy="20" r="11" fill="#f7edda"/><path d="M35 12H57M35 22H57M45 0L60-25" stroke="#876b4c" stroke-width="3"/>':type==='social'?'<path d="M0 0H63V39H25L10 52V39H0Z" fill="#87729e"/><path d="M12 13H50M12 25H40" stroke="#fff9ee" stroke-width="3"/>':type==='book'?'<path d="M0 0Q25-8 34 0Q48-8 65 0V45Q47 37 34 45Q15 37 0 45Z" fill="#d1d7b9" stroke="#829771"/><path d="M34 0V44M8 10H24M43 10H58M8 22H24M43 22H58" stroke="#829771"/>':'<rect width="64" height="48" rx="5" fill="#fff9ed" stroke="#90a58d"/><rect x="8" y="9" width="20" height="21" fill="#c4d8c7"/><path d="M35 10H56M35 21H56M8 37H56" stroke="#90a58d" stroke-width="3"/>';return `<g id="film-channel-${i}"><path d="M500 155Q500 ${y-20} ${x+130} ${y}" fill="none" stroke="#afbea4" stroke-width="2"/><rect x="${x}" y="${y}" width="270" height="111" rx="12" fill="#fffbef" stroke="#c7ceb6"/><g transform="translate(${x+102} ${y+18})">${icon}</g>${T(x+135,y+94,s,21,'text-anchor="middle"')}</g>`;}).join('')}${T(190,514,'Publish → interpret → discuss',27)}</g>
    ${record('film-record','response')}${record('film-output','estimate')}
    </g></svg><g id="film-inset" transform="translate(18 18) scale(.23)"><rect width="570" height="440" rx="18" fill="#fff9ed" stroke="#b5c4aa" stroke-width="6"/>${map.replaceAll('id="film-geo-','id="film-mini-')}<circle cx="350" cy="388" r="9" fill="#b96643"/>${T(280,425,'Ontario → conceptual neighbourhood',22,'text-anchor="middle"')}</g>
    <g id="film-landscape"><rect width="380" height="255" rx="12" fill="#fff9ee" stroke="#c6ceb7"/><g id="film-env-urban" transform="translate(10 20) scale(.36)">${A.landscape('urban')}</g><g id="film-env-suburban" transform="translate(10 20) scale(.36)">${A.landscape('suburban')}</g><g id="film-env-rural" transform="translate(10 20) scale(.36)">${A.landscape('rural')}</g><text id="film-place" x="20" y="237" font-size="22"></text></g><g id="film-question-end"><rect x="165" y="195" width="670" height="150" rx="18" fill="#fff9eddc" stroke="#c7b799"/>${T(500,251,'How reliable are',36,'text-anchor="middle"')}${T(500,301,'these estimates?',36,'text-anchor="middle"')}</g>`;
  // The only scene graph construction. Seek/replay never insert nodes.
  svg.innerHTML=art;
  window.LFSRegions.build($('film-camera'));
  const unitLabels=$('film-unit-labels').querySelectorAll('text');
  unitLabels[0].textContent='Primary Sampling Unit (PSU)';
  unitLabels[1].textContent='Secondary Sampling Unit (SSU): dwelling';
  unitLabels[1].setAttribute('x','555');unitLabels[1].setAttribute('y','310');unitLabels[1].setAttribute('font-size','18');
  unitLabels[1].insertAdjacentHTML('beforebegin','<rect id="film-ssu-label-bg" x="548" y="287" width="392" height="31" rx="4" fill="#edf1e5"/>');
  $('film-city').insertAdjacentHTML('beforeend',`<g id="film-stratum"><rect x="35" y="59" width="930" height="465" rx="10" fill="none" stroke="#8ca477" stroke-width="3"/><rect x="35" y="15" width="930" height="36" rx="7" fill="#e3ecd4"/>${T(55,40,'Ontario → Stratum → PSU → Dwelling (SSU) → Eligible people',22)}</g>`);
  for(let i=0;i<7;i++)$('film-panel-'+i).insertAdjacentHTML('beforeend',T(13,220,i===0?'Interview 6/6':i===6?'Interview 1/6':`Interview ${6-i}/6`,17,'data-light="true"'));
  const rotationTexts=$('film-rotation').querySelectorAll(':scope>text');rotationTexts[1].textContent='1/6 replaced each month · 5/6 retained';
  rotationTexts[2].textContent='Six consecutive months in the provincial sample';
  rotationTexts[3].textContent='Dwelling rotation ≠ changing household composition';
  // Connected rooms occupy fixed positions in the same world. The camera follows records.
  $('film-neighbourhood').insertAdjacentHTML('afterbegin','<path d="M900 320Q1060 320 1170 180M2050 180Q2160 180 2300 320M3100 320Q3220 320 3390 105" stroke="#97b4a0" stroke-width="8" stroke-dasharray="10 12" fill="none"/>');
  $('film-processing').insertAdjacentHTML('beforeend',[0,1].map(i=>`<g id="film-imputed-tile-${i}"><rect x="-7" y="-25" width="74" height="36" rx="7" fill="${i?'#e2deeb':'#dce9e0'}" stroke="#b96643"/>${T(6,0,i?'30*':'35*',25)}</g>`).join(''));
  $('film-interior').insertAdjacentHTML('beforeend',`<g id="film-phone"><path d="M0 0Q-12 12 12 37L21 31L13 21L7 26Q-2 16 1 10L7 5Z" fill="#426c66"/>${T(35,28,'Telephone',21)}</g><g id="film-online">${A.monitor(0,0,2)}${T(109,32,'Self-completed form',21)}</g><g id="film-answer"><rect width="180" height="70" rx="9" fill="#fff9ed" stroke="#91ae98"/>${T(14,27,'Structured response',17)}${T(14,55,'hours = 35',22)}</g>`);
  // Chairs and bent legs distinguish seated analysts from the standing discussion.
  for(let i=0;i<3;i++)$('film-analyst-'+i).insertAdjacentHTML('afterbegin',`<g transform="translate(${65+i*330} 350)"><rect width="65" height="75" rx="10" fill="#b1bfad"/><path d="M16 68L12 118M50 68L55 118" stroke="#8ca08b" stroke-width="6"/></g>`);
  function groupRange(parent,first,last,id){const g=document.createElementNS('http://www.w3.org/2000/svg','g');g.id=id;parent.insertBefore(g,first);while(first&&first!==last){const next=first.nextSibling;g.appendChild(first);first=next;}return g;}
  groupRange($('film-analysis'),$('film-analysis').querySelector('rect[x="230"]'),$('film-analyst-0'),'film-whiteboard');
  groupRange($('film-release'),$('film-release').querySelector('rect[x="300"]'),$('film-channel-0'),'film-publisher');
  const householdCards=[...$('film-household-types').children].filter(e=>e.tagName==='g');
  const routes=$('film-neighbourhood').firstElementChild;
  function responsive(m){
    for(const [id,h]of[['rotation',800],['household-types',760],['processing',880],['analysis',800],['release',730]]){const bg=$('film-'+id).querySelector('rect');bg.setAttribute('width',m?600:1000);bg.setAttribute('height',m?h:540);}
    move($('film-whiteboard'),m?-195:0,0);move($('film-publisher'),m?-200:0,0);
    for(let i=0;i<3;i++){const x=50+i*330,s=m?.85:1;move($('film-analyst-'+i),m?35+i%2*290-x*s:0,m?73.3+Math.floor(i/2)*240:0,s);}
    householdCards.forEach((g,i)=>move(g,m?15+i%2*290:35+i%3*325,m?100+Math.floor(i/2)*205:105+Math.floor(i/3)*210,m?.94:1));
    for(let i=0;i<6;i++){const g=$('film-channel-'+i),oldX=42+i%3*325,oldY=200+Math.floor(i/3)*137,tx=m?15+i%2*290-oldX:0,ty=m?200+Math.floor(i/2)*137-oldY:0;move(g,tx,ty);g.querySelector('path').setAttribute('d',m?`M${300-tx} ${155-ty}Q${300-tx} ${oldY-20} ${oldX+135} ${oldY}`:`M500 155Q500 ${oldY-20} ${oldX+130} ${oldY}`);}
    [el.prior,el.donor,el.rule].forEach((g,i)=>{move(g,m?-520:0,m?360-i*17:0);g.querySelector('path').setAttribute('opacity',m?0:1);});
    const adjustText=(parent,oldY,newY)=>{const t=[...parent.querySelectorAll(':scope>text')].find(e=>+e.dataset.originalY===oldY||+e.getAttribute('y')===oldY);if(t){t.dataset.originalY=oldY;t.setAttribute('y',m?newY:oldY);}};
    adjustText(el.rotation,455,720);adjustText(el.rotation,490,760);
    adjustText(el.processing,482,815);adjustText(el.processing,519,853);adjustText(el.analysis,523,770);adjustText(el.release,514,704);
    const releaseFooter=el.release.querySelector(':scope>text');if(releaseFooter)releaseFooter.setAttribute('x',m?50:190);
    const hhTitle=el['household-types'].querySelector('text');hhTitle.setAttribute('font-size',m?22:27);
    const month=$('film-month');month.setAttribute('x',m?395:760);month.setAttribute('font-size',m?21:24);
    for(const parent of[el.rotation,el.processing,el.analysis])parent.querySelectorAll(':scope>text').forEach(e=>{if(!e.dataset.desktopFont)e.dataset.desktopFont=e.getAttribute('font-size');e.setAttribute('font-size',m?Math.min(+e.dataset.desktopFont,23):e.dataset.desktopFont);});
    const analysisFooter=el.analysis.querySelector(':scope>text');if(analysisFooter)analysisFooter.textContent=m?'Weights · variance · quality review':'Final weights ≠ household size · design-aware uncertainty';
  }
  const ids=['camera','map','map-labels','domains','focus','neighbourhood','city','psu','unit-labels','rotation','interior','household-types','family','visitor','bubble','methods','processing','prior','donor','rule','analysis','discussion','release','record','output','inset','landscape','env-urban','env-suburban','env-rural','question-end'];
  const el=Object.fromEntries(ids.map(id=>[id,$('film-'+id)]));
  const windows=[...$('film-city').querySelectorAll('[data-dwelling]')];
  const cities=[['ON',314.2,316.6,'Ontario · dense urban blocks'],['BC',108.5,253.6,'British Columbia · suburbs'],['SK',196.9,290.3,'Saskatchewan · rural dwellings'],['PE',455.3,310.6,'PEI · direct dwelling selection'],['YT',102.4,155.6,'Yukon · quarterly interviews']];
  let elapsed=0,last=null,frame=0,visible=false,userPaused=false,current=-1,geoCode='Canada';
  const at=t=>Math.max(0,scenes.findLastIndex(s=>t>=s[0]));
  const overview=[0,0,570,440],cityBox=[316,368,68,38];
  // Hold each focus before travelling, so labels describe the geography on screen.
  const on=[244,267,170,137],bc=[28,202,185,104],sk=[132,240,175,99],pe=[426,291,60,34],yt=[48,114,128,72];
  const cameras=[[0,overview],[3,overview],[6,on],[8.2,on],[9.5,bc],[11,bc],[12.5,sk],[14,sk],[15.7,pe],[19,pe],[20.7,yt],[24,yt],[25.8,on],[28,on],[32,[317,363,56,32]],[33,cityBox]];
  function camera(t){let box=[...cityBox];for(let i=0;i<cameras.length-1;i++){const[a,p]=cameras[i],[b,q]=cameras[i+1];if(t>=a&&t<b){const u=ramp(t,a,b);return p.map((v,j)=>j>1?Math.exp(mix(Math.log(v),Math.log(q[j]),u)):mix(v,q[j],u));}}box[0]+=66*(ramp(t,71,74)+ramp(t,83,87)+ramp(t,95,98));return t<cameras[0][0]?overview:box;}
  function render(playbackTime){
    // Skip only the redundant opening hold; preserve all existing scene clocks.
    const realTime=playbackTime<1?playbackTime:playbackTime+openingTrim;
    let t=realTime<50?0:realTime-25;
    const stage=at(playbackTime),m=narrow.matches;
    const portrait=m&&((t>=41&&t<58)||t>=72),height=m?(portrait?800:430):560;
    svg.setAttribute('viewBox',m?`0 0 600 ${height}`:'0 0 1000 560');svg.style.aspectRatio=m?`600 / ${height}`:'1000 / 560';
    el.camera.setAttribute('width',m?'600':'1000');el.camera.setAttribute('height',height);
    responsive(m);
    const box=camera(t);
    if(m){let target=null,u=1;if(t>=40&&t<57){target=t<53?[320,372,36,48]:[319,371,39,46];u=ramp(t,40,42)*(1-ramp(t,56,58));}else if(t>=71){target=t<84?[378,371,38,55]:t<96?[444,371,38,49]:[510,371,38,44];u=ramp(t,71,74);target[0]=box[0]+2;}if(target)for(let j=0;j<4;j++)box[j]=mix(box[j],target[j],u);}
    el.camera.setAttribute('viewBox',box.join(' '));
    // Continuous zoom: the neighbourhood is physically anchored in southern Ontario.
    alpha(el.map,1-ramp(t,29,33));alpha(el['map-labels'],1-ramp(t,25,29));
    alpha(el.domains,ramp(t,25,27)*(1-ramp(t,30,33)));alpha(el.focus,ramp(t,25,28)*(1-ramp(t,32,34)));
    const location=t<6?null:cities[t<9?0:t<12?1:t<15?2:t<20?3:t<25?4:0];geoCode=location?location[0]:'Canada';
    geo.forEach(([c])=>{$('film-geo-'+c).setAttribute('fill',c===geoCode?'#e8b87f':'#cedec6');});
    move(el.landscape,m?210:590,m?190:245,m?.9:1);alpha(el.landscape,ramp(t,6,7)*(1-ramp(t,25,28)));
    alpha(el['env-urban'],1-ramp(t,8.5,9.5));alpha(el['env-suburban'],ramp(t,8.5,9.5)*(1-ramp(t,11.5,12.5)));alpha(el['env-rural'],ramp(t,11.5,12.5));
    if(location)$('film-place').textContent=location[3];
    alpha(el.neighbourhood,ramp(t,28,32));alpha(el.city,1-ramp(t,40,42));
    alpha($('film-stratum'),ramp(t,29,32)*(1-ramp(t,40,42)));
    alpha(el.psu,ramp(t,31,34)*(1-ramp(t,40,42)));alpha(el['unit-labels'],ramp(t,33,34)*(1-ramp(t,40,42)));
    windows.forEach((w,i)=>{const chosen=w.closest('#film-city-b2,#film-city-b3')&&i%3===0,u=ramp(t,34+i%5*.3,37+i%5*.3);w.setAttribute('fill',chosen?`rgb(${Math.round(mix(233,243,u))},${Math.round(mix(243,188,u))},${Math.round(mix(239,99,u))})`:'#e9f3ef');w.setAttribute('opacity',chosen?1:mix(1,.42,u));});
    alpha(el.rotation,ramp(t,40,42)*(1-ramp(t,52,54)));
    const handover=ramp(t,46,50);for(let i=0;i<7;i++){const p=$('film-panel-'+i),slot=i===6?0:i;move(p,i===0?mix(55,-130,handover):i===6?mix(-130,55,handover):55+(m?slot%3:slot)*148,m?150+Math.floor(slot/3)*245:150);alpha(p,i===0?1-handover:i===6?handover:1);}
    $('film-month').textContent=handover>.5?'Month t + 1':'Month t';
    for(let i=1;i<6;i++)$('film-panel-'+i).lastElementChild.textContent=`Interview ${6-i+(handover>.5?1:0)}/6`;
    // Emphasis is derived from the clock, so Pause freezes it too.
    rotationTexts[1].setAttribute('fill',handover>0&&handover<1?'#a65c2f':'#35574e');
    alpha(el['household-types'],ramp(t,52,54)*(1-ramp(t,57,60)));
    const enter=ramp(t,57,61);move(el.interior,mix(795,0,enter),mix(210,0,enter),Math.exp(mix(Math.log(.05),0,enter)));alpha(el.interior,ramp(t,57,60)*(1-ramp(t,71,73)));
    move(el.visitor,mix(220,0,ramp(t,61,64)),0);alpha(el.visitor,ramp(t,61,64));alpha(el.bubble,ramp(t,64,65)*(1-ramp(t,70,72)));alpha(el.methods,ramp(t,61,64));
    const q=Math.floor(clamp((t-64)/7)*3.99);$('film-question').textContent=['Worked last week?','Looking for work?','Available to work?','Hours worked?'][q];
    // Record leaves the interviewer's computer and lands in the processing table.
    const transmit=ramp(t,69,74);move(el.record,mix(695,1166,transmit),mix(319,109,transmit)-65*Math.sin(transmit*Math.PI),mix(.54,.75,transmit));alpha(el.record,ramp(t,68,69)*(1-ramp(t,74,75)));
    move(el.processing,1100,0);alpha(el.processing,ramp(t,70,72)*(1-ramp(t,85,88)));
    [el.prior,el.donor,el.rule].forEach((e,i)=>alpha(e,ramp(t,74+i*2.5,75+i*2.5)));
    ['35*','30*','employed*'].forEach((v,i)=>{$('film-value-'+i).textContent=t>=76+i*2.5?v:'?';});
    for(let i=0;i<2;i++){const u=ramp(t,75+i*2.5,76+i*2.5);move($('film-imputed-tile-'+i),mix(m?110:630,354,u),mix(m?513+i*91:153+i*108,188+i*73,u));alpha($('film-imputed-tile-'+i),ramp(t,74.8+i*2.5,75+i*2.5)*(1-ramp(t,76+i*2.5,76.3+i*2.5)));}
    move($('film-phone'),80,418);alpha($('film-phone'),ramp(t,64,65)*(1-ramp(t,67,68)));
    move($('film-online'),70,412,.55);alpha($('film-online'),ramp(t,67,68));
    move($('film-answer'),780,303);alpha($('film-answer'),ramp(t,68,69)*(1-ramp(t,71,73)));
    const analyse=ramp(t,82,84);move(el.analysis,2200,0);alpha(el.analysis,analyse*(1-ramp(t,97,99)));move(el.discussion,m?-365:0,(m?405:0)+9*Math.sin((t-84)*.7)*ramp(t,87,89),m?.8:1);
    const send=ramp(t,93,98);move(el.output,mix(2980,3780,send),mix(295,64,send)-50*Math.sin(send*Math.PI),mix(.75,.6,send));alpha(el.output,ramp(t,91,93)*(1-ramp(t,98,100)));
    move(el.release,3300,0);alpha(el.release,ramp(t,94,96));
    for(let i=0;i<6;i++)alpha($('film-channel-'+i),ramp(t,98+i*.55,99+i*.55));
    alpha(el.inset,ramp(t,28,32)*(1-ramp(t,40,42)));move(el.inset,18,18,m?.19:.23);
    alpha(el['question-end'],ramp(t,104,106));move(el['question-end'],m?-135:0,m?80:0,m?.84:1);
    // Regional environments are anchored to the map and appear only as the camera approaches.
    alpha($('film-regions'),realTime<50?1:0);alpha(el.landscape,0);
    if(realTime<50){const regional=window.LFSRegions.state(realTime,ramp,mix);el.camera.setAttribute('viewBox',regional.box.join(' '));geoCode=regional.geo;geo.forEach(([c])=>$('film-geo-'+c).setAttribute('fill',c===geoCode?'#e8b87f':'#cedec6'));}
    else if(realTime<51.5){const from=window.LFSRegions.regions.at(-1).box,u=ramp(realTime,50,51.5);el.camera.setAttribute('viewBox',from.map((v,j)=>mix(v,box[j],u)).join(' '));}
    if(stage!==current){current=stage;$('journey-heading').textContent=scenes[stage][1];$('journey-caption').textContent=scenes[stage][2];$('journey-scene-desc').textContent=scenes[stage][1]+'. '+scenes[stage][2];}
    $('journey-progress').value=playbackTime;$('journey-time').textContent=`${Math.floor(playbackTime)} / ${duration} s · ${reduced.matches?'still scenes':playbackTime>=duration?'complete':frame?'playing':'paused'} · ${stage+1}/${scenes.length}`;
    $('journey-play').disabled=!!frame||reduced.matches;$('journey-pause').disabled=!frame;
    $('journey-prev').disabled=stage===0;$('journey-next').disabled=stage===scenes.length-1;
  }
  function stop(){if(frame)cancelAnimationFrame(frame);frame=0;last=null;}
  function tick(now){if(last!==null)elapsed=Math.min(duration,elapsed+(now-last)/1000);last=now;render(elapsed);if(elapsed<duration&&visible&&!document.hidden&&!userPaused&&!reduced.matches)frame=requestAnimationFrame(tick);else{stop();render(elapsed);}}
  function reconcile(){const run=visible&&!document.hidden&&!userPaused&&!reduced.matches&&elapsed<duration;if(run&&!frame){last=null;frame=requestAnimationFrame(tick);}else if(!run)stop();render(elapsed);}
  function pause(){userPaused=true;stop();render(elapsed);}
  function play(){if(reduced.matches)return;if(elapsed>=duration)elapsed=0;userPaused=false;reconcile();}
  function setTime(t){stop();elapsed=Math.max(0,Math.min(duration,t));userPaused=true;render(elapsed);}
  function seek(i){const running=!!frame;stop();i=Math.max(0,Math.min(scenes.length-1,i));elapsed=scenes[i][0]+Math.min(2,(scenes[i+1]?.[0]??duration)-scenes[i][0]-.1);userPaused=!running;render(elapsed);reconcile();}
  $('journey-play').onclick=play;$('journey-pause').onclick=pause;$('journey-replay').onclick=()=>{stop();elapsed=0;userPaused=reduced.matches;render(0);reconcile();};
  $('journey-prev').onclick=()=>seek(current-1);$('journey-next').onclick=()=>seek(current+1);
  $('journey-progress').addEventListener('input',e=>setTime(+e.target.value));
  document.addEventListener('visibilitychange',reconcile);reduced.addEventListener('change',()=>{userPaused=reduced.matches;reconcile();});narrow.addEventListener('change',()=>render(elapsed));
  if('IntersectionObserver'in window)new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;reconcile();},{threshold:.15}).observe(svg);else{visible=true;reconcile();}
  window.LFSJourney={setTime,seek,play,pause,scenes,duration,state:()=>({elapsed,playing:!!frame,stage:current,geo:geoCode,reduced:reduced.matches,visible})};render(0);
  // Small persistent illustrations elsewhere in the article.
  document.querySelectorAll('#overview .card').forEach((card,i)=>card.insertAdjacentHTML('beforeend',A.summary(i)));
  document.querySelectorAll('#population .sampled').forEach(unit=>{const window=unit.querySelector('[data-dwelling]');if(window){window.setAttribute('fill','#f3bb62');window.setAttribute('stroke','#a65c2f');window.setAttribute('stroke-width','3');}});
  document.querySelectorAll('.environment-grid>div>span').forEach((span,i)=>{span.innerHTML=`<svg viewBox="0 0 1000 540" aria-hidden="true">${A.landscape(['urban','suburban','rural'][i])}</svg>`;});
  const guide=$('geographic-guide');
  if(guide){guide.innerHTML='<figure class="weighting-figure"></figure>';}
  window.LFSVisuals.init();
  const links=[...document.querySelectorAll('.toc a')];
  // Track sections without attaching a scrolling listener.
  if('IntersectionObserver'in window){const observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting)links.forEach(a=>a.getAttribute('href')==='#'+e.target.id?a.setAttribute('aria-current','location'):a.removeAttribute('aria-current'));},{rootMargin:'-10% 0px -65% 0px'});links.forEach(a=>{const section=document.querySelector(a.getAttribute('href'));if(section)observer.observe(section);});}
})();
