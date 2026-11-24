'use strict';
const CPIArchitecture = (() => {
  const {escape,$,$$,labs,panel} = Study;
  const paths = {
    population:'M4 7a2 2 0 1 0 4 0 2 2 0 0 0-4 0m9 0a2 2 0 1 0 4 0 2 2 0 0 0-4 0M2 17v-2a4 4 0 0 1 8 0v2m1 0v-2a4 4 0 0 1 8 0v2',
    geography:'M2 5l6-2 5 2 6-2v13l-6 2-5-2-6 2zm6-2v13m5-11v13',
    outlet:'M3 8h14l-2-5H5zm1 0v10h12V8m-8 10v-6h4v6',
    box:'M3 6l7-3 7 3v9l-7 3-7-3zm0 0 7 3 7-3m-7 3v9',
    select:'M10 2l8 8-8 8-8-8zm-3 8 2 2 4-4',
    data:'M4 5c0-4 12-4 12 0v10c0 4-12 4-12 0zm0 0c0 4 12 4 12 0M4 10c0 4 12 4 12 0',
    price:'M3 3h7l8 8-7 7-8-8zm4 4h.1',
    house:'M2 10l8-7 8 7m-14-2v10h12V8m-8 10v-5h4v5',
    bread:'M3 16V9c0-5 14-5 14 0v7zm4-8 2 3m2-4 2 3',
    milk:'M6 3h7l3 5v10H5V8zm0 0v5h10M5 12h11',
    fuel:'M3 18V3h9v15M3 9h9m0 3h2v4c0 2 4 2 4 0V8l-2-2',
    transit:'M4 16V5c0-2 12-2 12 0v11zm0-7h12m-10 7v2m8-2v2M7 13h.1m6 0h.1',
    clothing:'M6 3l4 2 4-2 5 5-3 3-2-2v9H6V9l-2 2-3-3z',
    internet:'M2 7a13 13 0 0 1 16 0M5 11a8 8 0 0 1 10 0m-7 4a3 3 0 0 1 4 0m-2 3h.1',
    energy:'M11 2L4 13h5l-1 6 8-11h-5z',
    rotate:'M15 6a6 6 0 1 0 1 7m-1-11v5h-5',
    sum:'M16 3H5l6 7-6 7h11'
  };
  function icon(name, klass='dimension-icon') {
    return '<svg class="'+klass+'" viewBox="0 0 21 21" aria-hidden="true"><path d="'+paths[name]+'" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round"/></svg>';
  }
  const scope = {
    ca:['Private households in scope','Geographic and business frames','Reachable outlets and offers','Selected representative offers','Observed or imputed prices','National CPI'],
    us:['Urban consumer target population','Area, outlet and housing frames','Eligible outlets, items and dwellings','Sampled outlets, items and dwellings','Observed or imputed prices','National CPI-U']
  };
  function scopeDiagram(key) {
    const isCanada=key==='ca', labels=scope[key];
    const boxes=labels.map((label,i)=>{
      const y=12+i*49, w=[358,326,294,264,259,198][i], x=190-w/2;
      const dotCount=[12,9,6,3,0,0][i];
      const dots=Array.from({length:dotCount},(_,j)=>'<circle cx="'+(190+(j-(dotCount-1)/2)*12)+'" cy="'+(y+25)+'" r="2.2" fill="currentColor" opacity="'+(.35+i*.15)+'"/>').join('');
      const symbol=['population','geography','outlet','select','price','sum'][i];
      return '<g><rect x="'+x+'" y="'+y+'" width="'+w+'" height="35" rx="7" fill="'+(i===5?'currentColor':'white')+'" fill-opacity="'+(i===5?.14:.75)+'" stroke="currentColor" stroke-opacity=".36"/>'+
        '<path transform="translate('+(x+7)+' '+(y+7)+')" d="'+paths[symbol]+'" fill="none" stroke="currentColor" stroke-width="1.2"/>'+
        '<text x="190" y="'+(y+(dotCount?15:22))+'" fill="#31444b" text-anchor="middle" font-size="12">'+escape(label)+'</text>'+dots+'</g>';
    }).join('');
    const arrows=Array.from({length:5},(_,i)=>'<path d="M190 '+(47+i*49)+'v12m-3-3 3 3 3-3" fill="none" stroke="currentColor" stroke-opacity=".57"/>').join('');
    const feed=isCanada?'Scanner, administrative and rent streams':'Housing and alternative data streams';
    const bypass='<path d="M375 77V219h-49m4-4-4 4 4 4" stroke="currentColor" stroke-opacity=".58" stroke-dasharray="3 3" fill="none"/>'+
      '<title>'+feed+' converge at the observations stage</title>';
    return '<svg class="hierarchy" style="color:var(--country)" viewBox="0 0 380 301" role="img" aria-label="'+escape(labels.join(' → ')+'; '+feed)+'">'+arrows+bypass+boxes+'</svg><p class="note">'+feed+' can enter the observation or estimation stage through their own pathways.</p>';
  }
  const designs = {
    ca:{
      nodes:[
        ['target',190,25,'frame','population','Household purchases','s2','Private household purchases','Domestic expenditures in scope define the population.','A scope definition has no selection probability.','Excluded purchases and territories outside scope.','ca-scope-chapter'],
        ['geo',66,88,'frame','geography','Geographic frame','geo','Collection area','Areas are usually census subdivisions or groups of census tracts in large subdivisions.','Depends on the component and source.','Thin coverage of small markets.','ca5'],
        ['rent',190,88,'probability','house','Rent via LFS','rent','Rental dwelling','LFS sample provides reported rent characteristics for the rent model.','Defined by the underlying probability survey.','Nonresponse and model dependence.','ca7'],
        ['feeds',314,88,'record','data','Retail feeds','source','Transaction record','Participating retailer feeds and matched records inform selected components.','Feed participation determines coverage.','Retailer coverage and product mix.','ca5'],
        ['outlet',66,151,'judgment','outlet','Outlet selection','outlet','Retail outlet','Market information and industry or product relevance guide selection.','Generally unavailable for judgmental selection.','Outlet coverage and frame changes.','ca5'],
        ['fee',314,151,'record','price','Admin. prices','source','Fee or regulated price','Some goods and services use external records or close-to-complete fee schedules.','Depends on the source and domain.','Timing and scope differences.','ca5'],
        ['offer',66,214,'judgment','box','Matched offer','offer','Representative product offer','Specify a product and match a concrete offer. Certain components use cut-offs or distinct sources.','Generally unavailable for judgmental offers.','Product churn and comparable replacement.','ca5'],
        ['process',190,277,'frame','price','Price processing','process','Price observation or donor movement','Edit, match, quality-adjust and impute where applicable.','Missing data treatment introduces no new sampled unit.','Quality-model and donor assumptions.','ca7'],
        ['index',190,337,'aggregate','sum','Aggregate CPI','index','Linked price indexes','Combine elementary aggregates with expenditure weights and link Lowe basket segments.','Expenditure weights serve aggregation.','Weight vintage and aggregation effects.','ca6']
      ],
      edges:[['target','geo'],['target','rent'],['target','feeds'],['geo','outlet'],['outlet','offer'],['offer','process'],['rent','process'],['target','fee',true],['feeds','process',true],['fee','process',true],['process','index']]
    },
    us:{
      nodes:[
        ['target',190,25,'frame','population','Urban households','s2','Urban consumer expenditures','CPI-U covers expenditures of urban consumers in scope.','A scope definition has no selection probability.','Geographies and residents outside scope.','bls-concepts'],
        ['geo',66,88,'frame','geography','Geographic frame','geo','Core-based statistical area','Stratify according to the documented geographic design.','Design-defined within probability stages.','Frame age and market coverage.','bls-design'],
        ['rent',190,88,'probability','house','Housing sample','rent','Rental dwelling and sample segment','Separate housing design supplies rent and OER observations.','Defined within the applicable housing design.','Nonresponse and rental-to-owner representation.','bls-housing'],
        ['feeds',314,88,'record','data','Alternative data','source','Secondary price or transaction record','Use accepted secondary sources and corporate data for selected items.','Depends on source coverage.','Provider and definition changes.','bls-data'],
        ['certainty',66,151,'certainty','select','Certainty areas','geo','Self-representing pricing area','Include self-representing areas with certainty.','One at the geographic stage.','Within-area sampling and response error remain.','bls-design'],
        ['sample',190,151,'probability','select','Sampled areas','geo','Non-self-representing pricing area','Sample noncertainty areas under the 2018 design.','Defined by the geographic design.','Between-area sampling variation.','bls-design'],
        ['outlet',66,214,'probability','outlet','Outlet and item','offer','Outlet and unique item offer','For applicable components, expenditure-based outlet sampling precedes sales-informed item selection.','Defined conditionally for applicable stages.','Outlet births, item turnover and refused initiation.','bls-design'],
        ['process',190,277,'frame','price','Price processing','process','Quote or imputed movement','Edit, replace, quality-adjust and impute as appropriate.','Missing data treatment introduces no new sampled unit.','Donor and quality assumptions.','bls-calc'],
        ['index',190,337,'aggregate','sum','Aggregate CPI-U','index','Area-item price indexes','Aggregate using expenditure cost weights for CPI-U.','Expenditure weights serve aggregation.','Weight vintage and aggregation effects.','bls-calc']
      ],
      edges:[['target','geo'],['target','rent'],['target','feeds'],['geo','certainty'],['geo','sample'],['certainty','outlet'],['sample','outlet'],['outlet','process'],['rent','process'],['feeds','process',true],['process','index']]
    }
  };
  function treeSvg({nodes,edges}, country) {
    const map=new Map(nodes.map(n=>[n[0],n]));
    const wires=edges.map(([parent,child,alternative])=>{
      const [,x0,y0]=map.get(parent), [,x1,y1]=map.get(child);
      const mid=(y0+y1)/2;
      let path='M'+x0+' '+(y0+15)+'V'+mid+'H'+x1+'V'+(y1-15);
      if(parent==='rent')path='M'+(x0+56)+' '+y0+'H250V'+y1+'H'+(x1+56);
      if(parent==='target'&&child==='fee')path='M'+(x0+56)+' '+y0+'H376V'+y1+'H'+(x1+56);
      if(parent==='feeds'&&child==='process'&&country==='ca')path='M'+(x0+56)+' '+y0+'H374V252H'+x1+'V'+(y1-15);
      return '<path class="tree-edge" data-parent="'+parent+'" data-child="'+child+'" d="'+path+'" marker-end="url(#arr-'+country+')"'+(alternative?' stroke-dasharray="4 3"':'')+'/>';
    }).join('');
    const cards=nodes.map(([id,x,y,type,symbol,label])=>{
      const cx=x-45;
      const mark=type==='probability'?'<circle class="node-symbol" cx="'+cx+'" cy="'+y+'" r="4"/>':
        type==='judgment'?'<path class="node-symbol" d="M'+cx+' '+(y-5)+'l5 5-5 5-5-5z"/>':
        '<path class="node-symbol" transform="translate('+(cx-5)+' '+(y-5)+') scale(.55)" d="'+paths[symbol]+'"/>';
      const inner=type==='certainty'?'<rect x="'+(x-54)+'" y="'+(y-12)+'" width="108" height="24" rx="5" fill="none" stroke="currentColor" stroke-opacity=".35"/>':'';
      const lines=['Household purchases','Urban households'].includes(label)?label.split(' '):[label];
      const caption='<text>'+lines.map((line,i)=>
        '<tspan x="'+(x+8)+'" y="'+(y+4+i*11-(lines.length-1)*5.5)+'">'+escape(line)+'</tspan>').join('')+'</text>';
      return '<g class="tree-node" style="color:var(--country)" data-node="'+id+'" tabindex="0" role="button" aria-pressed="false" aria-label="'+escape(label)+'">'+
        '<rect class="node-box" x="'+(x-57)+'" y="'+(y-15)+'" width="114" height="30" rx="7"/>'+inner+mark+caption+'</g>';
    }).join('');
    return '<svg class="tree-svg" viewBox="0 0 382 357" role="group" aria-label="'+country+' interactive sampling pathways">'+
      '<defs><marker id="arr-'+country+'" markerHeight="6" markerWidth="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3 0 6z" fill="context-stroke"/></marker></defs>'+wires+cards+'</svg>';
  }
  labs.trees=el=>{
    const countries={ca:'Canada',us:'United States'};
    el.innerHTML=panel('Illustrative methodological diagram','Trace a sampling path',
      '<div class="pairs diagram-pair">'+Object.entries(countries).map(([key,name])=>
        '<section class="country '+key+'"><h4>'+name+'</h4>'+treeSvg(designs[key],key)+'<div class="tree-detail" aria-live="polite"></div></section>').join('')+'</div>'+
      '<div class="tree-legend note"><span><i>○</i>Probability</span><span><i>◇</i>Judgmental</span><span><i>▣</i>Certainty</span><span>'+icon('data')+'Records</span><span><i>┄</i>Alternative route</span><span><i>Σ</i>Aggregation</span></div>',
      'Price processing includes matching, quality adjustment and, when required, imputation. Select a node with a pointer, Enter or Space.');
    const panels=Object.fromEntries(Object.keys(countries).map(key=>[key,$('.country.'+key,el)]));
    function show(key,node,paired=true){
      const box=panels[key], {nodes,edges}=designs[key];
      const row=nodes.find(n=>n[0]===node);
      const [, , , , ,label,match,unit,mechanism,probability,risk,ref]=row;
      const related=new Set([node]);
      for(const [parent,child] of edges){
        if(child===node)related.add(parent);
        if(parent===node)related.add(child);
      }
      const upstream=new Set([node]), downstream=new Set([node]);
      function trace(ids, source, target){
        let size;
        do {
          size=ids.size;
          for(const edge of edges)if(ids.has(edge[source]))ids.add(edge[target]);
        }while(size!==ids.size);
      }
      trace(upstream,1,0); trace(downstream,0,1);
      $$('.tree-edge',box).forEach(w=>{
        const {parent,child}=w.dataset;
        w.classList.toggle('is-active',upstream.has(parent)&&upstream.has(child)||downstream.has(parent)&&downstream.has(child));
      });
      $$('.tree-node',box).forEach(n=>{
        n.setAttribute('aria-pressed',n.dataset.node===node);
        n.classList.toggle('is-related',related.has(n.dataset.node));
      });
      $('.tree-detail',box).innerHTML='<h5>'+label+'</h5><dl>'+[
        ['Unit',unit],['Mechanism',mechanism],['Inclusion',probability],['Risk',risk]
      ].map(([term,value])=>'<dt>'+term+'</dt><dd>'+escape(value)+'</dd>').join('')+'</dl><p><a href="#'+ref+'">Methodology reference</a></p>';
      if(paired){
        const other=key==='ca'?'us':'ca';
        const linked=designs[other].nodes.find(n=>n[6]===(match==='outlet'?'offer':match));
        if(linked)show(other,linked[0],false);
      }
    }
    for(const [key,box] of Object.entries(panels)){
      $$('.tree-node',box).forEach(node=>{
        const pick=()=>show(key,node.dataset.node);
        node.addEventListener('click',pick);
        node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick();}});
      });
    }
    show('ca','geo');
  };
  return {icon,scopeDiagram};
})();

