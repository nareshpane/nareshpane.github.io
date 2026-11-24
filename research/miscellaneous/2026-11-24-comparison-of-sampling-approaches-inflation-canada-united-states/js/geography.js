'use strict';
const CPIGeography = (() => {
  const {$, $$, labs, controls, select, range, listen, panel, metrics, table,
    escape, plot, fmt, mean, sum, norm, valid} = Study;
  const D = CPI_GEO;
  const nations = {CA:'Canada', US:'United States'};
  const labels = {all:'All items', food:'Food', shelter:'Shelter', transport:'Transportation',
    energy:'Energy', gasoline:'Gasoline'};
  const colors = {CA:'#a64936', US:'#285f91'};
  const national = {CA:'CA', US:'0000'};
  const starts = {CA:'ON', US:'0100'};
  const current = {month:D.dates.length-1, item:'all', period:0, CA:'ON', US:'0100'};
  const changes = new EventTarget();
  function share(update) {
    if (Object.keys(update).every(key=>current[key]===update[key])) return;
    Object.assign(current,update);
    changes.dispatchEvent(new Event('change'));
  }
  const choices = entries => Object.entries(entries);
  const pickOptions = (country, areas) => areas.map(code=>[code,D.labels[country][code]]);
  const monthOptions = D.dates.map((date,index)=>[index,date]).reverse();
  const name = (country,code) => D.labels[country][code] || code;
  const rate = (country,area,key=current.item,month=current.month,period=current.period) =>
    D.rates[country][key]?.[area]?.[month]?.[period] ?? null;
  const display = (value, unit='%', decimals=1) => valid(value) ? fmt(value,decimals)+unit : 'Unavailable';
  const periodName = () => current.period ? '12-month' : 'Monthly';
  const scope = country => country==='CA' ? '10 provinces and 3 territorial capitals' : '4 U.S. Census regions';
  const weightDate = country => country==='CA' ?
    '2025 spending, April 2026 prices' : '2024 spending, December 2025 relative importance';
  function sync(el, values) {
    for (const [key,value] of Object.entries(values)) {
      const field = $('[name="'+key+'"]',el);
      if (field && [...field.options].some(o=>o.value===String(value))) field.value=value;
    }
  }

  const arch = {
    CA:[
      ['⌂','Target geography','Private household consumption in all ten provinces and the three northern capitals. Provinces include rural areas.'],
      ['▦','Strata and collection areas','Nineteen index strata. Census subdivisions and, where divided, groups of census tracts guide collection-area selection.'],
      ['▤','Local and central price sources','Outlet–product offers, scanner records, administrative files, web prices and other sources. Alternative sources may follow different geographic pathways.'],
      ['⊞','Stratum × product indexes','Matched price relatives enter elementary cells. Geographic or product donors can fill some gaps.'],
      ['⇢','Provincial and capital CPI','Separate published indexes summarize the provinces and the three northern cities.'],
      ['Σ','National all-items CPI','Price-updated expenditure weights enter upper-level aggregation and basket links.']
    ],
    US:[
      ['⌂','Urban target population','CPI-U covers urban consumers across the 50 states and D.C., with specified population exclusions.'],
      ['▦','CBSAs and geographical strata','Group qualifying areas within the design geography, selecting one representative PSU per stratum. Large metros and the Alaska/Hawaii exceptions enter with certainty.'],
      ['▤','Price collection and other inputs','Commodities and services outlets and items, housing units, and approved alternative inputs enter parallel paths.'],
      ['⊞','Basic area × item indexes','The 75 design PSUs feed 32 basic index areas. Area–item combinations enter basic-index estimation.'],
      ['⇢','Published area indexes','Regions, selected divisions, area size classes and eligible metro or urban-state indexes have distinct publication rules.'],
      ['Σ','U.S. city average CPI-U','Consumer Expenditure Surveys and price-updated area–item weights govern upper-level CPI-U aggregation.']
    ]
  };
  labs.geoArchitecture = el => {
    el.innerHTML=panel('Methodology','From coverage to national aggregation',
      '<div class="geo-pair">'+Object.entries(arch).map(([country,nodes])=>
        '<section class="geo-card '+country.toLowerCase()+'"><h5>'+nations[country]+'</h5><ol class="geo-flow">'+
        nodes.map(([icon,title],i)=>'<li><button type="button" data-step="'+i+'" aria-pressed="'+(i===0)+'"><span aria-hidden="true">'+icon+'</span>'+title+'</button></li>').join('')+
        '</ol><p class="geo-readout" aria-live="polite">'+nodes[0][2]+'</p></section>').join('')+'</div>',
      'Publication and national aggregation are parallel outputs of the elementary indexes. Arrows indicate the flow of information.');
    $$('.geo-card',el).forEach((card,j)=>card.addEventListener('click',event=>{
      const button=event.target.closest('button[data-step]');
      if (!button) return;
      $$('button',card).forEach(b=>b.setAttribute('aria-pressed',b===button));
      $('.geo-readout',card).textContent=Object.values(arch)[j][+button.dataset.step][2];
    }));
  };

  const cells = [
    [[25,[[3.40,3.62],[3.56,3.71],[3.28,3.34]]],[18,[[1.45,1.53],[1.50,1.54],[1.48,1.56]]],[7,[[18,18.5],[17,17]]]],
    [[14,[[3.70,3.75],[3.56,3.63]]],[11,[[1.47,1.59],[1.44,1.58]]],[5,[[21,20.5],[19,20]]]],
    [[8,[[4.10,4.15],[3.97,4.03]]],[12,[[1.72,1.85],[1.70,1.83]]],[0,[]]]
  ];
  labs.geoMatrix = el => {
    const columns=['Bread','Gasoline','Haircut'];
    const rows=['Stratum A · large urban market','Stratum B · mixed market','Stratum C · remote market'];
    el.innerHTML=panel('Synthetic example','Geography × product',
      '<div class="table-wrap"><table class="geo-matrix"><caption>Cell shares of the whole example basket (%)</caption><thead><tr><th scope="col">Geographical stratum</th>'+
      [...columns,'Row total'].map(c=>'<th scope="col">'+c+'</th>').join('')+'</tr></thead><tbody>'+
      cells.map((row,i)=>'<tr><th scope="row">'+rows[i]+'</th>'+row.map(([w],j)=>
        '<td><button type="button" aria-label="'+rows[i]+', '+columns[j]+', '+w+' percent" aria-pressed="false" data-cell="'+i+j+'">'+w+'</button></td>').join('')+
        '<td><b>'+sum(row.map(([w])=>w))+'</b></td></tr>').join('')+'</tbody><tfoot><tr><th scope="row">Column total</th>'+
        columns.map((_,j)=>'<td>'+sum(cells.map(row=>row[j][0]))+'</td>').join('')+'<td>100</td></tr></tfoot></table></div><div class="geo-readout" data-cell-detail aria-live="polite"></div>',
      'Prices, shares and strata are illustrative. The example uses equal quote weights and a geometric mean.');
    function draw(button) {
      const [i,j]=[...button.dataset.cell].map(Number);
      const [weight,quotes]=cells[i][j];
      const index=quotes.length ? 100*Math.exp(mean(quotes.map(([p,q])=>Math.log(q/p)))) : null;
      $$('[data-cell]',el).forEach(b=>b.setAttribute('aria-pressed',b===button));
      $('[data-cell-detail]',el).innerHTML='<strong>'+rows[i]+' · '+columns[j]+'</strong>'+
        (quotes.length ? '<p>Basket share '+weight+'%. Elementary relative '+fmt(index,2)+' (base 100).</p>'+
          table(['Matched outlet','Earlier price','Current price','Relative'],quotes.map(([p,q],k)=>
            [k+1,fmt(p,2),fmt(q,2),fmt(q/p,4)])) : '<p>This cell has zero expenditure in the example and no matched quote.</p>');
    }
    el.addEventListener('click',e=>{const b=e.target.closest('[data-cell]');if(b)draw(b);});
    draw($('[data-cell]',el));
  };

  const layers = {
    CA:{admin:'Administrative boundaries', strata:'Documented stratum counts', publication:'Published CPI areas',
      weights:'Geographical expenditure weights', rates:'Published inflation rates', locations:'Illustrative collection locations'},
    US:{admin:'State and D.C. boundaries', regions:'Census regions', divisions:'Census divisions',
      psus:'2018 primary sampling units', self:'Certainty selections', nonself:'Non-self-representing PSUs',
      index:'32 basic CPI index areas', rates:'Published CPI-U by Census region'}
  };
  const layerNotes = {
    CA:{admin:'The map shows all provinces and territories. City markers identify the three northern CPI geographies.',
      strata:'Colour shows the number of CPI strata in a province. Capital markers each stand for one stratum. Subprovincial polygons were unavailable.',
      publication:'Provincial fills refer to provincial CPI series. In the North, dots identify the three separately published capital series.',
      weights:'Each fill or capital dot represents an all-items expenditure share. Darker marks indicate a larger share.',
      rates:'Colour follows the selected month, item and horizon. Grey indicates an unavailable observation.',
      locations:'Provincial dots are illustrative urban locations. Northern dots identify the published capitals. The dots give no information about actual outlet locations.'},
    US:{admin:'State boundaries provide context. Select a state or D.C. to see which 2018 PSUs intersect it.',
      regions:'Four Census regions group states and D.C. BLS aggregates basic areas into each region.',
      divisions:'Nine Census divisions group states and D.C. and constrain the non-self-representing strata.',
      psus:'Each dot locates a documented PSU approximately. Select it to see the matched constituent county boundaries.',
      self:'21 large metro PSUs and two urban-state certainty exceptions. Anchorage and Honolulu represent their wider urban-state strata.',
      nonself:'52 selected PSUs. Each represents a stratum that can include other CBSAs outside the selected footprint.',
      index:'Markers identify constituent PSUs. Selecting one draws the sampled footprints belonging to its basic index area.',
      rates:'State outlines are reference boundaries. The shared fill is an official Census-region rate for urban consumers.'}
  };
  const regionColors=['#b6d1e7','#a8c8e1','#78a9cf','#5e93bf'];
  const divisionColors=['#bed7ec','#9dc3e2','#aac4df','#709bc8','#7aafd4','#6091bc','#a7d1df','#6eafc7','#93bdd8'];
  const swatchesCA=['#f6e2d7','#e9b69e','#cf856d','#a74f38'];
  const swatchesUS=['#e1eff8','#abcee6','#76a6cc','#386d9c'];
  const missing='#e6e9eb';
  const selector = (code,title,body,active=false) =>
    '<g role="button" tabindex="0" class="geo-feature'+(active?' geo-selected':'')+'" data-feature="'+escape(code)+'" aria-label="'+escape(title)+'" aria-pressed="'+active+'">'+body+'<title>'+escape(title)+'</title></g>';
  const swatch = (value,palette,limit) => !valid(value) ? missing :
    palette[Math.min(palette.length-1,Math.max(0,Math.floor(value/limit*palette.length)))];
  const weightShade = value => swatchesCA[value<.25?0:value<3?1:value<15?2:3];
  const mapKey = entries => '<span class="geo-map-key">'+entries.map(([key,paint])=>
    '<span><i style="background:'+paint+'"></i> '+escape(key)+'</span>').join('')+'</span>';
  function rateShade(value, limit, palette) {
    return value<0 ? swatch(-value,['#e2e8dd','#bfd4c3','#87b29c','#4c8878'],limit) : swatch(value,palette,limit);
  }
  function specs(country) {
    const us = country==='US', palette=us?swatchesUS:swatchesCA;
    const values=Object.keys(D.maps[country]).map(key=>rate(country,us?D.maps.US[key].region:key));
    const limit=Math.max(.1,...values.filter(valid).map(Math.abs));
    return {palette,limit};
  }
  function drawMap(country, layer, selected, selectedPsu) {
    const us=country==='US';
    const {palette,limit}=specs(country);
    const base=us?'#e4eef6':'#f3ddd0';
    const map=D.maps[country];
    const psu=D.psus.find(p=>p.code===selectedPsu);
    function fill(code,a) {
      if (us) {
        if (layer==='regions') return regionColors[D.areas.US.indexOf(a.region)];
        if (layer==='divisions') return divisionColors[D.divisions.indexOf(a.division)];
        if (layer==='rates') return rateShade(rate('US',a.region),limit,palette);
      } else if (!a.capital) {
        if (layer==='strata') return palette[a.strata-1];
        if (layer==='publication') return palette[1];
        if (layer==='weights') return weightShade(D.weights.CA.geography[code]);
        if (layer==='rates') return rateShade(rate('CA',code),limit,palette);
      }
      return base;
    }
    const countries=Object.entries(map).map(([code,a])=>{
      const path='<path fill="'+fill(code,a)+'" d="'+a.path+'"/>';
      const dot=a.point&&code==='DC'?'<circle class="geo-dc" cx="'+a.point[0]+'" cy="'+a.point[1]+'" r="2.1"/>':'';
      return selector(code,a.name,path+dot,code===selected);
    }).join('');
    const group=psu && (layer==='index' ? D.psus.filter(p=>p.basic===psu.basic) : [psu]);
    const outlines=group ? '<g class="geo-counties">'+group.flatMap(p=>p.paths).map(d=>'<path d="'+d+'"/>').join('')+'</g>' : '';
    let markers='';
    if (us && ['psus','self','nonself','index'].includes(layer)) {
      const psus=D.psus.filter(p=>p.point&&(layer!=='self'||p.code.startsWith('S'))&&(layer!=='nonself'||p.code.startsWith('N')));
      markers=psus.map(p=>{
        const [x,y]=p.point, inGroup=group&&group.includes(p);
        const paint=layer==='index'?divisionColors[D.basics.indexOf(p.basic)%divisionColors.length] :
          p.code.startsWith('S')?'#205785':'#6794b4';
        return selector(p.code,p.code+' · '+p.name,
          '<circle class="geo-psu '+(inGroup?'matched':'')+'" cx="'+x+'" cy="'+y+'" r="'+(p.code.startsWith('S')?3.3:2.4)+'" fill="'+paint+'"/>',p===psu);
      }).join('');
    }
    if (!us) {
      markers=Object.entries(map).filter(([,a])=>a.capital||layer==='locations').map(([code,a])=>{
        const [x,y]=a.point;
        let paint='#a64936';
        if (layer==='rates') paint=rateShade(rate('CA',code),limit,palette);
        if (layer==='weights') paint=weightShade(D.weights.CA.geography[code]);
        return selector(code,a.capital ? a.area+' · published CPI geography' : a.name+' · illustrative urban point',
          '<circle class="geo-capital" cx="'+x+'" cy="'+y+'" r="'+(a.capital?4:3.1)+'" fill="'+paint+'"/>',code===selected);
      }).join('');
    }
    const insets=us?'<g class="geo-insets"><path d="M10 170h102v67H10zM347 183h97v54h-97z"/><text x="16" y="232">Alaska</text><text x="353" y="232">Hawaii</text></g>':'';
    return '<svg class="geo-map" viewBox="0 0 460 248" aria-label="'+nations[country]+' · '+layers[country][layer]+'" role="group">'+
      insets+'<g class="geo-boundaries" fill-rule="evenodd">'+countries+'</g>'+outlines+'<g class="geo-markers">'+markers+'</g></svg>';
  }
  function areaSummary(country, code) {
    return '<dl class="geo-stat-list"><div><dt>Basket share</dt><dd>'+display(D.weights[country].geography[code],'%',2)+'</dd></div>'+
      '<div><dt>Monthly</dt><dd>'+display(rate(country,code,current.item,current.month,0))+'</dd></div>'+
      '<div><dt>12-month</dt><dd>'+display(rate(country,code,current.item,current.month,1))+'</dd></div></dl>';
  }
  function details(country,code,layer,psuCode) {
    const a=D.maps[country][code];
    const period=D.dates[current.month]+' · '+labels[current.item]+' · unadjusted';
    if (country==='CA') return '<h6>'+escape(a.name)+(a.capital?' · '+a.area:'')+'</h6>'+
      '<p>'+a.strata+' documented CPI '+(a.strata===1?'stratum':'strata')+' · '+period+'</p>'+areaSummary('CA',code)+
      '<p>'+ (a.capital ? 'The figures cover '+a.area+' only. Communities outside the capital are outside the territorial CPI scope.' :
        a.strata>1 ? 'The provincial CPI combines strata based on 2006 Census economic regions. Local collection areas and central price sources feed those strata.' :
        'All provincial areas are in scope. Selected collection areas and central price sources support the provincial index.')+'</p>';
    const p=D.psus.find(p=>p.code===psuCode);
    if (p) {
      const total=layer==='index' ? D.psus.filter(x=>x.basic===p.basic).length : 1;
      return '<h6>'+escape(p.name)+' · '+p.code+'</h6><p>'+p.type+' · '+name('US',p.region)+' · '+name('US',p.division)+
        ' · basic area '+p.basic+(total>1?' ('+total+' sample PSUs)':'')+'</p><p>'+escape(p.definition)+'</p>'+
        '<p>2010 design population represented: '+p.population.toLocaleString('en-US')+'. Pricing cycle: '+({M:'monthly',E:'even months',O:'odd months'}[p.cycle]||p.cycle)+'.</p>'+
        (p.missing.length ? '<p>Unmatched county boundaries: '+escape(p.missing.join(', '))+'.</p>':'')+
        (D.metros.includes(p.code)?'<p>'+period+'</p>'+areaSummary('US',p.code):'');
    }
    const related=D.psus.filter(p=>p.states.includes(code));
    const published=related.filter(p=>D.publishedLocal?.[p.code]||D.metros.includes(p.code));
    return '<h6>'+escape(a.name)+'</h6><p>'+name('US',a.region)+' region · '+name('US',a.division)+' division</p>'+
      '<p>'+(related.length ? related.length+' documented '+(related.length===1?'PSU intersects':'PSUs intersect')+' this jurisdiction.' :
        'No PSU in the appendix has a listed county in this jurisdiction. In-scope urban consumers can be represented by a sampled PSU elsewhere.')+'</p>'+
      '<p>'+(['AK','HI'].includes(code) ? 'BLS publishes an urban-state CPI. Its scope is urban consumers.' :
        'BLS publishes no statewide CPI-U for this jurisdiction.')+'</p>'+
      (published.length ? '<p>Published local all-items CPI-U: '+published.map(p=>escape(D.publishedLocal?.[p.code]?.name||name('US',p.code))).join('; ')+'.</p>' : '')+
      '<p>'+name('US',a.region)+' region · '+period+'</p>'+areaSummary('US',a.region)+
      (related.length?'<details><summary>Intersecting PSU definitions</summary><ul>'+related.map(p=>'<li>'+p.code+' · '+escape(p.name)+'</li>').join('')+'</ul></details>':'');
  }

  labs.geoMaps = el => {
    const panes=Object.keys(nations);
    const selected={CA:current.CA,US:'NY'};
    let psu=null;
    el.innerHTML=panel('Official geography and observations','Explore the two geographical designs',
      controls('Inflation layer',3,select('Month','month',monthOptions)+select('Category','item',choices(labels))+
        select('Change','period',[[0,'Monthly (NSA)'],[1,'12-month (NSA)']]))+
      '<div class="geo-pair">'+panes.map(country=>'<section class="geo-card '+country.toLowerCase()+'" data-country="'+country+'"><h5>'+
        nations[country]+'</h5><div class="geo-map-controls">'+select('Map layer',country+'layer',choices(layers[country]))+
        select(country==='CA'?'Province or territory':'State or district',country+'state',
          Object.entries(D.maps[country]).sort((a,b)=>a[1].name.localeCompare(b[1].name)).map(([k,a])=>[k,a.name]))+
        (country==='US'?select('Sample PSU','psu',[['','Choose a PSU'],...D.psus.map(p=>[p.code,p.code+' · '+p.name])]):'')+
        '</div><div data-map></div><p class="geo-legend" data-legend></p><div class="geo-readout" data-detail aria-live="polite"></div></section>').join('')+'</div>',
      'Boundary sources: Natural Earth and U.S. Census Bureau. Sample definitions: Statistics Canada and BLS. The two maps have different projection scales.');
    sync(el,{...current,CAstate:selected.CA,USstate:selected.US});
    function draw() {
      sync(el,{month:current.month,item:current.item,period:current.period,CAstate:selected.CA,USstate:selected.US,psu:psu||''});
      panes.forEach(country=>{
        const card=$('[data-country="'+country+'"]',el), layer=$('[name="'+country+'layer"]',el).value;
        const map=$('[data-map]',card);
        const feature=document.activeElement?.dataset.feature;
        const wasFocused=map.contains(document.activeElement)&&feature;
        map.innerHTML=drawMap(country,layer,selected[country],country==='US'?psu:null);
        if(wasFocused) $('[data-feature="'+feature+'"]',map)?.focus({preventScroll:true});
        const palette=country==='CA'?swatchesCA:swatchesUS;
        let key='';
        if(layer==='rates')key=mapKey([['Decline','#4c8878'],['Increase',palette.at(-1)],['Unavailable',missing]]);
        if(layer==='weights')key=mapKey(['<0.25%','0.25–<3%','3–<15%','≥15%'].map((s,i)=>[s,palette[i]]));
        if(layer==='strata')key=mapKey(palette.map((paint,i)=>[String(i+1),paint]));
        if(layer==='regions')key=mapKey(D.areas.US.map((code,i)=>[name('US',code),regionColors[i]]));
        if(layer==='divisions')key=mapKey(D.divisions.map((code,i)=>[name('US',code),divisionColors[i]]));
        if(['psus','self','nonself'].includes(layer))key=mapKey([['Certainty','#205785'],['Non-self-representing','#6794b4']]);
        $('[data-legend]',card).innerHTML=layerNotes[country][layer]+key;
        $('[data-detail]',card).innerHTML=details(country,selected[country],layer,country==='US'?psu:null);
      });
    }
    function state(country,code) {
      selected[country]=code;
      if(country==='US') psu=null;
      const area=country==='US'?D.maps.US[code].region:code;
      share({[country]:area});
      draw();
    }
    function sample(code) {
      psu=code||null;
      if (psu) {
        const p=D.psus.find(x=>x.code===psu);
        const layer=$('[name=USlayer]',el);
        if (!['psus','index','self','nonself'].includes(layer.value) ||
            (layer.value==='self'&&p.code.startsWith('N')) || (layer.value==='nonself'&&p.code.startsWith('S'))) layer.value='psus';
        share({US:p.region});
      }
      draw();
    }
    el.addEventListener('input',event=>{
      const field=event.target;
      if (field.name==='psu') sample(field.value);
      else if (field.name?.endsWith('state')) state(field.name.slice(0,2),field.value);
      else if (['month','item','period'].includes(field.name)) share({[field.name]:field.name==='item'?field.value:+field.value});
      else if (field.name?.endsWith('layer')) {
        if(field.name==='USlayer' && ['admin','regions','divisions','rates'].includes(field.value)) psu=null;
        if(field.name==='USlayer' && field.value==='self' && psu?.startsWith('N')) psu=null;
        if(field.name==='USlayer' && field.value==='nonself' && psu?.startsWith('S')) psu=null;
        draw();
      }
    });
    const activate=event=>{
      const f=event.target.closest('[data-feature]');
      if (!f) return;
      const country=f.closest('[data-country]').dataset.country, code=f.dataset.feature;
      if (country==='US' && code.length===4) sample(code);
      else state(country,code);
    };
    el.addEventListener('click',activate);
    el.addEventListener('keydown',event=>{
      if (event.target.matches('[data-feature]') && ['Enter',' '].includes(event.key)) {
        event.preventDefault();activate(event);
      }
    });
    const previous={CA:current.CA,US:current.US};
    changes.addEventListener('change',()=>{
      if(current.CA!==previous.CA) selected.CA=current.CA;
      if(current.US!==previous.US){
        if(psu && D.psus.find(p=>p.code===psu).region!==current.US)psu=null;
        if(D.maps.US[selected.US].region!==current.US&&!psu)
          selected.US=Object.keys(D.maps.US).find(k=>D.maps.US[k].region===current.US);
      }
      previous.CA=current.CA;previous.US=current.US;
      draw();
    });
    draw();
  };

  const modeNames = {official:'Official expenditures', equal:'Equal regions', population:'Population-based',
    custom:'User-defined shock-area share'};
  const adj = {
    CA:{NL:['QC','NS'],PE:['NB','NS'],NS:['NB','PE','NL'],NB:['QC','PE','NS'],QC:['ON','NL','NB','NU'],
      ON:['QC','MB'],MB:['ON','SK','NU'],SK:['MB','AB','NT'],AB:['SK','BC','NT'],
      BC:['AB','YT','NT'],YT:['BC','NT'],NT:['YT','BC','AB','SK','NU'],NU:['NT','MB','QC']},
    US:{'0100':['0200','0300'],'0200':['0100','0300','0400'],'0300':['0100','0200','0400'],'0400':['0200','0300']}
  };
  const syntheticPopulation = {NL:4,PE:2,NS:6,NB:5,QC:22,ON:36,MB:7,SK:6,AB:13,BC:14,YT:1,NT:1,NU:1};
  function computeWeights(country, mode, selected, assigned) {
    const areas = D.areas[country];
    if (mode==='equal') return areas.map(()=>1/areas.length);
    if (mode==='population') return norm(areas.map(k=>(country==='US'?D.population:syntheticPopulation)[k]));
    const official=areas.map(k=>D.weights[country].geography[k]);
    if (mode==='custom') {
      const rest=sum(official.filter((_,i)=>areas[i]!==selected));
      return official.map((v,i)=>areas[i]===selected?assigned/100:(1-assigned/100)*v/rest);
    }
    return norm(official);
  }
  function contributionRows(country, config, mode) {
    const {area,adjust,shock,spill}=config;
    const w=computeWeights(country,mode,area,adjust);
    return D.areas[country].map((code,i)=>{
      const baseline=rate(country,code,'all'), observed=rate(country,code);
      const delta=code===area?shock:adj[country][area].includes(code)?spill/100*shock:0;
      const product=(D.weights[country].product[current.item]?.[code] ?? NaN)/100;
      const effect=delta===0||w[i]===0 ? 0 : valid(product) ? w[i]*product*delta : null;
      const contribution=w[i]===0 ? 0 : valid(baseline) ? w[i]*baseline : null;
      return {code,weight:w[i],baseline,observed,delta,product,effect,contribution,
        result:valid(contribution)&&valid(effect)?contribution+effect:null};
    });
  }
  const completeSum = values => values.every(valid) ? sum(values) : null;
  const stackColors=['#9d4a3a','#bd6a54','#a86e47','#b18956','#926d67','#65567d','#407573',
    '#468596','#41658b','#64734f','#a05774','#756a53','#777f8a'];
  function shockMap(country) {
    const paths=Object.entries(D.maps[country]).map(([code,a])=>{
      const region=country==='US'?a.region:code;
      const title=name(country,region)+(a.capital?' · reported capital only':'');
      const coverage=a.capital?'<circle cx="'+a.point[0]+'" cy="'+a.point[1]+'" r="4"/>':'';
      const shape='<path d="'+a.path+'"'+(a.capital?' class="geo-capital-outline"':'')+'/>';
      return '<g data-map-area="'+region+'">'+shape+coverage+'<title>'+escape(title)+'</title></g>';
    }).join('');
    return '<svg class="geo-map geo-shock-map" viewBox="0 0 460 248" role="img" aria-label="'+
      nations[country]+' hypothetical shock footprint. The shock-area selector also controls this map.">'+paths+'</svg>';
  }
  function contributions(rows,country) {
    const finite=rows.filter(r=>valid(r.result));
    const max=Math.max(.1,...finite.map(r=>Math.abs(r.result)));
    const positive=finite.filter(r=>r.result>0), negative=finite.filter(r=>r.result<0);
    const scale=Math.max(.001,sum(positive.map(r=>r.result)),-sum(negative.map(r=>r.result)));
    const stack=(list,sign)=>{
      const size=sum(list.map(r=>Math.abs(r.result)));
      return '<div class="geo-stack-row"><span>'+sign+'</span><span class="geo-stack-track">'+
        list.map(r=>'<i style="width:'+(100*Math.abs(r.result)/scale)+'%;background:'+stackColors[rows.indexOf(r)]+'" title="'+
          escape(name(country,r.code)+': '+display(r.result,' pp',3))+'"></i>').join('')+
        '</span><span>'+display(sign==='Positive'?size:-size,' pp',2)+'</span></div>';
    };
    return '<div class="geo-stacks" aria-label="Stacked contributions of reported rows">'+
      stack(positive,'Positive')+stack(negative,'Negative')+'</div>'+
      '<div class="table-wrap"><table class="geo-weight-table"><thead><tr>'+
      ['Geography','Published area share','Model area share','Observed category rate','Scenario shock','Effect on all items','All-items contribution'].map(h=>'<th scope="col">'+h+'</th>').join('')+
      '</tr></thead><tbody>'+rows.map(r=>'<tr><th scope="row"><button type="button" class="geo-row-button" data-shock-area="'+r.code+'">'+escape(name(country,r.code))+'</button></th>'+
        '<td>'+display(D.weights[country].geography[r.code],'%',2)+'</td><td>'+display(100*r.weight,'%',2)+'</td><td>'+display(r.observed)+'</td>'+
        '<td>'+display(r.delta,' pp')+'</td><td>'+display(r.effect,' pp',3)+'</td><td><span class="geo-result">'+display(r.result,' pp',3)+'</span>'+
        '<span class="geo-diverging" aria-hidden="true"><i style="--bar:'+ (valid(r.result)?Math.abs(r.result)/max*50:0)+'%;--left:'+
        (valid(r.result)&&r.result<0?50-Math.abs(r.result)/max*50:50)+'%;--bar-color:'+stackColors[rows.indexOf(r)]+'"></i></span></td></tr>').join('')+
      '</tbody></table></div>';
  }
  labs.geoWeights = el => {
    const horizon=[[0,'Monthly (NSA)'],[1,'12-month (NSA)']];
    el.innerHTML=panel('Official inputs and hypothetical shocks','Geographical weighting laboratory',
      controls('Shared observations',3,select('Month','month',monthOptions)+select('Category affected','item',choices(labels))+select('Change','period',horizon))+
      '<div class="geo-pair">'+Object.keys(nations).map(country=>'<section class="geo-card '+country.toLowerCase()+'" data-weight="'+country+'"><h5>'+
        nations[country]+'</h5><p class="geo-source-line">'+scope(country)+' · '+weightDate(country)+'</p>'+
        '<div class="geo-map-controls">'+select('Shock area','area',pickOptions(country,D.areas[country]))+
        select('Weighting','mode',choices(modeNames))+'</div>'+
        controls('Hypothetical inputs',3,range('Price shock (pp)','shock',6,-15,30,.5)+range('Neighbor spillover (%)','spill',0,0,100,5)+
          range('Shock area share (%)','adjust',20,0,100,.5),'secondary')+
        '<div class="geo-shock-preview"><div>'+shockMap(country)+'</div><p class="geo-source-line">Scenario footprint<br>Dark outline marks the shock area. Shading shows local shock magnitude.</p></div>'+
        '<button type="button" data-weight-reset>Reset scenario</button><div data-weight-result aria-live="polite"></div></section>').join('')+'</div>',
      'The national scenarios are fixed-weight approximations. The spillover uses the displayed scenario links. Published rates and basket shares have their own reference periods.');
    sync(el,current);
    for (const country of Object.keys(nations)) {
      const card=$('[data-weight="'+country+'"]',el);
      sync(card,{area:current[country]});
      function render(config) {
        const rows=contributionRows(country,config,config.mode);
        const effect=completeSum(rows.map(r=>r.effect)), baseline=completeSum(rows.map(r=>r.contribution));
        const selected=rows.find(r=>r.code===config.area);
        const official=rate(country,national[country],'all');
        const local=selected.weight===0?0:valid(selected.product)?selected.weight*selected.product:null;
        $('[name=adjust]',card).disabled=config.mode!=='custom';
        const byArea=new Map(rows.map(r=>[r.code,r]));
        const maxShock=Math.max(.1,Math.abs(config.shock));
        $$('[data-map-area]',card).forEach(group=>{
          const row=byArea.get(group.dataset.mapArea);
          group.classList.toggle('geo-shocked',row===selected);
          group.style.fill=rateShade(row.delta,maxShock,country==='CA'?swatchesCA:swatchesUS);
        });
        const summary=metrics([
          ['Official national all-items change',display(official)],['Fixed-weight baseline',display(baseline)],
          ['Shock effect',display(effect,' pp',3)],['Scenario national estimate',display(valid(baseline)&&valid(effect)?baseline+effect:null)]
        ]);
        const modes=Object.keys(modeNames).map(mode=>{
          const test=contributionRows(country,config,mode);
          return [modeNames[mode],display(completeSum(test.map(r=>r.result)),'%',2),display(completeSum(test.map(r=>r.effect)),' pp',3)];
        });
        const known=sum(rows.filter(r=>valid(r.result)).map(r=>r.weight));
        const selectedMath='A 1 pp local '+labels[current.item].toLowerCase()+' shock in '+escape(name(country,config.area))+
          ' changes the national approximation by '+display(local,' pp',3)+' before spillovers.';
        const notes='<p>'+selectedMath+' Modeled neighbors: '+adj[country][config.area].map(k=>escape(name(country,k))).join(', ')+
          '.</p><p>'+(country==='US'?'Population mode uses the appendix’s 2010 CPI design population represented.':
            'Population mode uses synthetic counts. Canadian northern spillover links are illustrative.')+
          ' Reported rows cover '+display(100*known,'%',1)+' of model area weight.'+
          (known<.99999?' A missing rate or an unknown product share with a nonzero shock prevents a complete estimate.':'')+'</p>';
        $('[data-weight-result]',card).innerHTML=summary+'<div class="geo-readout">'+notes+'</div>'+
          contributions(rows,country)+'<div class="table-wrap">'+table(['Weighting basis','National scenario','Shock effect'],modes)+'</div>';
      }
      const update=listen(card,render);
      card.addEventListener('input',e=>{if(e.target.name==='area')share({[country]:e.target.value});});
      card.addEventListener('click',e=>{
        const row=e.target.closest('[data-shock-area]');
        const mapArea=e.target.closest('[data-map-area]');
        if(row||mapArea){
          share({[country]:row?.dataset.shockArea||mapArea.dataset.mapArea});
          $('[name=area]',card).focus();
        }
        if(e.target.closest('[data-weight-reset]')){
          ['shock','spill','adjust'].forEach(k=>{const f=$('[name="'+k+'"]',card);f.value=f.defaultValue;});
          sync(card,{area:starts[country],mode:'official'});
          share({[country]:starts[country]});
          update();
        }
      });
      changes.addEventListener('change',()=>{sync(card,{area:current[country]});update();});
    }
    el.addEventListener('input',e=>{
      const f=e.target;
      if(['month','period','item'].includes(f.name))share({[f.name]:f.name==='item'?f.value:+f.value});
    });
    changes.addEventListener('change',()=>sync(el,current));
  };

  function dispersion(country, areas, rates) {
    if (areas.length<2 || !rates.every(valid)) return null;
    const source=areas.map(area=>{
      const base=D.weights[country].geography[area];
      return current.item==='all'?base:base*(D.weights[country].product[current.item]?.[area] ?? NaN)/100;
    });
    if (!source.every(valid)||!sum(source)) return null;
    const w=norm(source), avg=sum(w.map((x,i)=>x*rates[i]));
    return Math.sqrt(sum(w.map((x,i)=>x*(rates[i]-avg)**2)));
  }
  function dotRows(country,areas,reference,values,[min,max]) {
    const position=v=>4+92*(v-min)/(max-min||1);
    return '<div class="geo-dots" aria-label="Regional rates with national reference">'+areas.map((code,i)=>{
      const v=values[i];
      return '<div class="geo-dot-row"><button type="button" class="geo-row-button" data-series="'+code+'">'+escape(name(country,code))+
        '</button><span class="geo-dot-track" aria-hidden="true">'+(valid(reference)?'<i class="geo-national-tick" style="left:'+position(reference)+'%"></i>':'')+
        (valid(v)?'<i class="geo-rate-dot" style="left:'+position(v)+'%;background:'+colors[country]+'"></i>':'')+
        '</span><b>'+display(v)+'</b></div>';
    }).join('')+'<div class="geo-dot-row geo-dot-domain"><span></span><span>'+fmt(min,1)+' to '+fmt(max,1)+'%</span><span></span></div></div>';
  }

  labs.geoEmpirical = el => {
    const grouping={CA:{all:'Provinces and capitals',provinces:'Ten provinces',capitals:'Three territorial capitals'},
      US:{all:'Four Census regions',divisions:'Nine Census divisions',metros:'Selected published local indexes'}};
    const panels=Object.keys(nations);
    const trace={CA:current.CA,US:current.US};
    el.innerHTML=panel('Published CPI observations','Regional inflation and dispersion',
      controls('Compare the same reference month',3,select('Month','month',monthOptions)+
        select('Category','item',choices(labels))+select('Change','period',[[0,'Monthly (NSA)'],[1,'12-month (NSA)']]))+
      '<div class="geo-episodes"><button type="button" data-episode="2022-06">June 2022 gasoline</button>'+
      '<button type="button" data-episode="2026-10">October 2026 all items</button></div>'+
      '<div class="geo-pair">'+panels.map(country=>'<section class="geo-card '+country.toLowerCase()+'" data-empirical="'+country+'"><h5>'+
        nations[country]+'</h5><div class="geo-map-controls">'+select('Geographical grouping',country+'group',choices(grouping[country]))+
        '<div data-trace-select></div></div><p class="geo-source-line" data-national></p><div data-points></div><div data-dispersion></div>'+
        '<h6>Selected geographical series and national comparator</h6><div data-history></div></section>').join('')+'</div>',
      'Rates come from published unadjusted index levels. Monthly changes require adjacent calendar months. Missing levels create gaps and comparable yearly observations may still be available for bimonthly indexes.');
    sync(el,current);
    function areasFor(country,group) {
      if(country==='CA') {
        if(group==='provinces')return D.areas.CA.slice(0,10);
        if(group==='capitals')return D.areas.CA.slice(10);
        return D.areas.CA;
      }
      if(group==='divisions')return D.divisions;
      if(group==='metros')return D.metros;
      return D.areas.US;
    }
    function draw() {
      sync(el,current);
      const selections=panels.map(country=>{
        const card=$('[data-empirical="'+country+'"]',el);
        const group=$('[name="'+country+'group"]',card).value;
        const areas=areasFor(country,group);
        return {country,card,group,areas,values:areas.map(k=>rate(country,k)),nationalRate:rate(country,national[country])};
      });
      const all=selections.flatMap(x=>[...x.values,x.nationalRate]).filter(valid);
      const axis=[Math.min(0,...all),Math.max(.1,...all)];
      selections.forEach(({country,card,group,areas,values,nationalRate})=>{
        if(!areas.includes(trace[country]))trace[country]=areas[0];
        const traceSelect=$('[data-trace-select]',card);
        if($('select',traceSelect)?.dataset.group!==group) {
          traceSelect.innerHTML=select('History','trace',pickOptions(country,areas));
          $('select',traceSelect).dataset.group=group;
        }
        sync(card,{trace:trace[country]});
        const measured=values.filter(valid);
        const spread=dispersion(country,areas,values);
        const range=measured.length>1?Math.max(...measured)-Math.min(...measured):null;
        const available=measured.length+'/'+areas.length;
        $('[data-national]',card).textContent=D.dates[current.month]+' · '+periodName()+' · National reference '+display(nationalRate)+
          (valid(nationalRate)?' (dashed line)':'');
        $('[data-points]',card).innerHTML=dotRows(country,areas,nationalRate,values,axis);
        $('[data-dispersion]',card).innerHTML=metrics([['Available rates',available],
          ['Observed range',display(range,' pp')],['Weighted dispersion',display(spread,' pp',2)]])+
          '<p class="geo-source-line">'+(valid(spread)?weightDate(country)+'; '+(current.item==='all'?'area weights.':'category weights from area × component shares.') :
            'Weighted dispersion requires reported rates and compatible weights for the entire selected grouping.')+'</p>';
        const lines=[national[country],trace[country]].map((area,i)=>({
          label:name(country,area),color:i?colors[country]:'#66777a',
          values:D.dates.map((_,month)=>rate(country,area,current.item,month,current.period))
        }));
        const history=$('[data-history]',card);
        history._plotFocus=current.month;
        plot(history,D.dates,lines,{title:name(country,trace[country])+' and national CPI change',y:'Change (%)',
          height:220,scatter:false,precision:2});
      });
    }
    el.addEventListener('input',e=>{
      const f=e.target;
      if(['month','item','period'].includes(f.name))share({[f.name]:f.name==='item'?f.value:+f.value});
      else if(f.name==='trace') {trace[f.closest('[data-empirical]').dataset.empirical]=f.value;draw();}
      else if(f.name?.endsWith('group'))draw();
    });
    el.addEventListener('click',e=>{
      const b=e.target.closest('button');
      if(b?.dataset.episode){
        const june=b.dataset.episode==='2022-06';
        share({month:D.dates.indexOf(b.dataset.episode),item:june?'gasoline':'all',period:june?1:0});
      }
      if(b?.dataset.series){
        const card=b.closest('[data-empirical]');
        const country=card.dataset.empirical;
        trace[country]=b.dataset.series;
        if(D.areas[country].includes(trace[country]))share({[country]:trace[country]});
        draw();
        $('[name=trace]',card).focus();
      }
    });
    const previous={CA:current.CA,US:current.US};
    changes.addEventListener('change',()=>{
      for(const country of panels){
        if(previous[country]!==current[country]){
          trace[country]=current[country];
          previous[country]=current[country];
        }
      }
      draw();
    });
    draw();
  };
  return {D,computeWeights,contributionRows};
})();
