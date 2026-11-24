'use strict';
(() => {
  const {$,$$,labs,panel,table,input,select,range,controls,equation,listen,sum,norm,fmt,pct,mean,metrics,plot,colors,escape} = Study;
  const simple = 'Simplified mathematical demonstration';
  const synthetic = 'Synthetic statistical simulation';
  const names = ['Bread','Milk','Gasoline','Rent','Transit','Clothing','Internet','Electricity'];
  const prices = [3,4,1.5,1250,3,34,63,.17];
  const steps = [.05,.1,.03,35,0,-2,3,.012];
  const quantities = [14,8,70,1,18,1,1,175];
  function makePriceTable(items, values, formula) {
    const heads = formula ? ['Product','Base price','New price','Weight','Usable'] : ['Product','Base price','New price','Quantity','Share %'];
    return table(heads,items.map((name,i)=>[name,
      input(name+' base price','p'+i,values[i],.01,99999,.01),
      input(name+' new price','t'+i,+(values[i]+steps[i]).toFixed(2),.01,99999,.01),
      formula?input(name+' weight','w'+i,i+1,0,100,1):input(name+' quantity','q'+i,quantities[i],0,99999,.1),
      formula?'<input type="checkbox" name="use'+i+'" checked aria-label="'+name+' usable">':
        input(name+' expenditure share','w'+i,+(values[i]*quantities[i]*100/sum(values.map((v,j)=>v*quantities[j]))).toFixed(2),0,100,.01)]),'entry');
  }
  labs.basket = el => {
    const icons=['bread','milk','fuel','house','transit','clothing','internet','energy'];
    el.innerHTML=panel(simple,'Price your household basket',
      '<div class="controls">'+select('Basket weights','mode',[['q','From base prices and quantities'],['w','Entered expenditure shares']])+
      select('Comparison interval','period',[['1','One month'],['12','Twelve months']])+
      select('Benchmark month','month',[['2026-10','October 2026'],['2026-09','September 2026']])+'</div>'+
      makePriceTable(names,prices,false)+'<div data-out></div><div data-distribution></div>',
      'The household figures are hypothetical. Both national comparisons use unadjusted all-items series and the same calendar endpoints. Sources: <a href="#ca-nsa">Statistics Canada 18-10-0004-01</a> and <a href="#us-api">BLS CUUR0000SA0</a>.');
    const priceTable=$('table',el);
    priceTable.classList.add('basket-table');
    const weights=$$('input[name^=w]',priceTable), outputs=[];
    $$('tbody tr',priceTable).forEach((row,i)=>{
      $('th',row).insertAdjacentHTML('afterbegin',CPIArchitecture.icon(icons[i]));
      weights[i].insertAdjacentHTML('beforebegin','<output class="basket-share" aria-label="'+names[i]+' calculated share"></output>');
      outputs.push(weights[i].previousElementSibling);
    });
    listen(el,p=>{
      const assigned=p.mode==='w';
      const shares=norm(names.map((_,i)=>assigned?p['w'+i]:p['p'+i]*p['q'+i]));
      weights.forEach((input,i)=>{
        input.hidden=!assigned;input.disabled=!assigned;
        outputs[i].hidden=assigned;outputs[i].value=fmt(100*shares[i],1)+'%';
        $('[name=q'+i+']',el).disabled=assigned;
      });
      const validWeights=sum(shares)>0;
      const contributions=names.map((_,i)=>shares[i]*100*(p['t'+i]/p['p'+i]-1));
      const growth=sum(contributions), d=+p.period, i=CPI_DATA.months.indexOf(p.month);
      const national=['CA_all','US_all'].map(key=>pct(CPI_DATA.series[key][i],CPI_DATA.series[key][i-d]));
      const asPercent=v=>Number.isFinite(v)?fmt(v)+'%':'Unavailable';
      const interval=d===1?'one month':'twelve months';
      $('[data-out]',el).innerHTML=metrics([
        ['Illustrative household index',validWeights?fmt(100+growth,2):'—'],
        ['Illustrative household inflation',validWeights?fmt(growth,2)+'%':'—'],
        ['Canada CPI · '+p.month,asPercent(national[0]),'ca'],
        ['U.S. CPI-U · '+p.month,asPercent(national[1]),'us']
      ])+'<p class="note">Comparison interval: '+interval+'. '+(national.includes(null)?'A national observation is unavailable for one endpoint.':'Both national series use the chosen month.')+'</p>';
      const stack='<div class="stack" role="img" aria-label="'+escape(names.map((name,j)=>name+' '+fmt(shares[j]*100)+'%').join(', '))+'">'+
        names.map((name,j)=>shares[j]>0?'<span style="width:'+shares[j]*100+'%;background:'+colors[j]+'" title="'+name+' '+fmt(shares[j]*100)+'%"></span>':'').join('')+'</div>';
      $('[data-distribution]',el).innerHTML='<h4>Expenditure distribution and contributions</h4>'+stack+
        table(['Product','Share used','Contribution (pp)'],names.map((name,j)=>[CPIArchitecture.icon(icons[j])+name,
          fmt(shares[j]*100,1)+'%',validWeights?fmt(contributions[j],3):'—']));
    });
  };
  labs.budget=el=>{
    const strata=['Urban','Regional','Remote'];
    const icons=['outlet','house','geography'];
    el.innerHTML=panel(simple,'Allocate a collection budget',
      controls('Total collection budget',1,range('Budget units','budget',160,50,450))+
      controls('Collection costs',3,strata.map((label,i)=>range(label+' cost per observation','c'+i,[1,2,4][i],1,[8,8,10][i])).join(''))+
      controls('Price variability',3,strata.map((label,i)=>range(label+' standard deviation','s'+i,[2,3,5][i],1,9)).join(''),'secondary')+
      '<div class="eq-strip">'+equation(String.raw`\min_{\{n_h\}}\sum_h\frac{W_h^2S_h^2}{n_h}\quad\text{s.t.}\quad\sum_hc_hn_h\le B`,'green')+
      equation(String.raw`n_h\propto\frac{W_hS_h}{\sqrt{c_h}}`,'blue')+'</div><div data-out></div>',
      'Fixed stratum shares are 60%, 28% and 12%. The continuous model assumes independent strata, known variability and constant costs per observation. Bounds of 2 to 120 observations can change the allocation. Integer sample planning requires a further rounding or optimization step.');
    listen(el,p=>{
      const weights=[.6,.28,.12], cost=[p.c0,p.c1,p.c2], s=[p.s0,p.s1,p.s2];
      const at=t=>weights.map((w,i)=>Math.max(2,Math.min(120,t*w*s[i]/Math.sqrt(cost[i]))));
      const spend=allocation=>sum(allocation.map((n,i)=>n*cost[i]));
      if(spend(at(0))>p.budget){
        $('[data-out]',el).textContent='The minimum of two observations per stratum requires '+spend(at(0))+' budget units.';
        return;
      }
      function allocate(budget) {
        let lo=0,hi=1e5;
        for(let i=0;i<60;i++){
          const mid=(lo+hi)/2;
          if(spend(at(mid))>budget)hi=mid;
          else lo=mid;
        }
        return at(lo);
      }
      const objective=ns=>sum(ns.map((v,i)=>(weights[i]*s[i])**2/v));
      const n=allocate(p.budget), variance=objective(n);
      const more=objective(allocate(p.budget+20));
      const spent=spend(n);
      const barRows=strata.map((label,i)=>[label,n[i],n[i]*cost[i]]);
      $('[data-out]',el).innerHTML=table(['Stratum','Continuous n','Budget used','Allocation bounds'],barRows.map(([label,count,used],i)=>
        [CPIArchitecture.icon(icons[i])+label,fmt(count,2),fmt(used,2),count<=2.0001?'Lower bound':count>=119.9999?'Upper bound':'Interior']))+
        '<h4>Allocated observations out of 120 per stratum</h4><div class="weight-bars">'+barRows.map(([label,count],i)=>
          '<div class="barrow"><span>'+label+'</span><span class="bartrack"><span style="width:'+count/1.2+'%;background:'+colors[i]+'"></span></span><span>'+fmt(count,1)+'</span></div>').join('')+'</div>'+
        '<h4>Budget use</h4><div class="stack" role="img" aria-label="Budget used '+fmt(100*spent/p.budget,1)+' percent">'+
        barRows.map(([label,_,used],i)=>'<span title="'+label+' '+fmt(used,2)+' units" style="width:'+100*used/p.budget+'%;background:'+colors[i]+'"></span>').join('')+'</div>'+
        metrics([['Total cost',fmt(spent,2)+' / '+p.budget],['Objective variance',fmt(variance,4)],['Variance with 20 more units',fmt(more,4)]]);
    });
  };
  labs.calendar=el=>{
    const frequencies=[[1,'Daily'],[3,'Every three days'],[7,'Weekly'],[14,'Fortnightly'],[28,'Once per month']];
    const timings=[[1,'From day 1'],[5,'From day 5'],[10,'From day 10'],[15,'From day 15']];
    const routes=[['quote','Survey quote'],['online','Online price'],['scanner','Scanner record']];
    const missing=[['omit','Use observed days'],['carry','Carry previous quote']];
    const symbols={Promotion:'%', 'Seasonal discount':'%', 'Missing quote':'×', Discontinued:'↺', Shortage:'!', 'Fuel fluctuations':'↕'};
    const countries=[
      {key:'ca', title:'Canada-inspired scenario', color:colors[0], frequency:'7', timing:'1'},
      {key:'us', title:'United States-inspired scenario', color:colors[1], frequency:'14', timing:'5'}
    ];
    const section=country=>'<div class="country '+country.key+'"><h4>'+country.title+'</h4><div class="calendar-controls">'+
      select('Frequency',country.key+'freq',frequencies)+select('Timing',country.key+'time',timings)+
      select('Pathway',country.key+'route',routes)+select('Missing-price rule',country.key+'missing',missing)+
      '</div><div data-days="'+country.key+'"></div></div>';
    el.innerHTML=panel(synthetic,'See the prices each collection schedule can miss',
      '<div class="controls">'+select('Product','item',['Groceries','Gasoline','Clothing','Online','Rent'])+
      select('Price event','event',Object.keys(symbols))+select('Month','month',[[0,'Month 1'],[1,'Month 2'],[2,'Month 3']])+
      '<button type="button" data-run aria-pressed="false">Play months</button><button type="button" data-next>Next month</button></div>'+
      controls('Shared price event',2,range('Magnitude (%)','magnitude',25,0,60)+range('Duration (days)','length',5,1,18))+
      '<div class="pairs calendar-pair">'+countries.map(section).join('')+'</div>'+
      '<p class="note">● Scheduled quote · ◇ Online price · ▤ Scanner price · × Unavailable · ≈ Carried forward · % Discount · ! Shortage · ↺ Replacement</p>'+
      '<div data-out></div><div data-chart></div><div class="eq-strip">'+
      equation(String.raw`\mu=\frac{1}{D}\sum_{d=1}^D p_d`)+equation(String.raw`\widehat{\mu}_S=\frac{1}{|S|}\sum_{d\in S}p_d`,'green')+
      equation(String.raw`e_S=100\left(\frac{\widehat{\mu}_S}{\mu}-1\right)`,'rose')+'</div>',
      'Both calendars use one synthetic 28-day price process. Schedules and missing-price rules are hypothetical and illustrate a daily-average estimation problem. Scanner, online and survey tracks differ by observation mechanism. A carried quote stands in for the missing price only in the estimate.');
    countries.forEach(({key,frequency,timing})=>{
      $('[name='+key+'freq]',el).value=frequency;
      $('[name='+key+'time]',el).value=timing;
    });
    const basePrices={Groceries:4,Gasoline:1.55,Clothing:49,Online:25,Rent:825};
    let timer;
    const update=listen(el,p=>{
      const month=+p.month, eventStart=7+2*month, magnitude=p.magnitude/100;
      const days=Array.from({length:28},(_,i)=>{
        const active=i>=eventStart&&i<eventStart+p.length;
        let price=basePrices[p.item]*(1+.012*month);
        if(active && ['Promotion','Seasonal discount'].includes(p.event))price*=1-magnitude;
        if(active && ['Shortage','Discontinued'].includes(p.event))price*=1+magnitude;
        if(active && p.event==='Fuel fluctuations')price*=1+magnitude*Math.sin(i*1.3+month);
        return {price,active};
      });
      const target=mean(days.map(d=>d.price));
      const results=countries.map(({key,title,color})=>{
        const frequency=+p[key+'freq'], start=+p[key+'time']-1, route=p[key+'route'];
        let previous=null,carries=0;
        const observed=[], usable=[];
        const cells=days.map(({price,active},i)=>{
          const scheduled=i>=start&&(i-start)%frequency===0;
          const unavailable=active&&['Missing quote','Discontinued','Shortage'].includes(p.event)&&route!=='scanner';
          const quote=scheduled&&!unavailable?price:null;
          if(quote!==null)previous=quote;
          const carried=scheduled&&quote===null&&p[key+'missing']==='carry'&&previous!==null;
          if(carried)carries++;
          observed.push(quote);
          if(quote!==null||carried)usable.push(carried?previous:quote);
          const mark=scheduled?(carried?'≈':unavailable?'×':route==='scanner'?'▤':route==='online'?'◇':'●'):'';
          return '<span class="'+(scheduled?'observed ':'')+(active?'event ':'')+(scheduled&&unavailable?'missing ':scheduled&&route==='scanner'?'feed':'')+
            '" title="Day '+(i+1)+(active?' · '+p.event:'')+(scheduled?' · '+(quote===null?(carried?'Carried quote':'Unavailable'):fmt(quote,2)):'')+'">'+
            (i+1)+(mark?' '+mark:active?' '+symbols[p.event]:'')+'</span>';
        });
        $('[data-days='+key+']',el).innerHTML='<div class="calendar">'+cells.join('')+'</div>';
        const average=mean(usable);
        return {key,title,color,observed,average,carries,count:observed.filter(v=>v!==null).length};
      });
      $('[data-out]',el).innerHTML=table(['Estimate','Daily average','Relative error (%)','Observed','Carried'],[
        ['Complete period',fmt(target,2),'—',28,0],
        ...results.map(r=>[r.title,fmt(r.average,2),fmt(pct(r.average,target),2),r.count,r.carries])
      ]);
      const chart=[
        {label:'Complete daily prices',color:'#485b63',values:days.map(d=>d.price)},
        {label:'Complete-period average',color:'#70817a',values:days.map(()=>target),dash:'5 4'},
        ...results.flatMap(r=>[
          {label:r.key.toUpperCase()+' observed',color:r.color,values:r.observed,scatter:true},
          {label:r.key.toUpperCase()+' average estimate',color:r.color,values:days.map(()=>r.average),dash:'5 4'}
        ])
      ];
      plot($('[data-chart]',el),days.map((_,i)=>String(i+1)),chart,{title:'Daily prices sampled by two hypothetical collection schedules',x:'Day of synthetic month',y:'Price (units)',precision:2});
    });
    const next=()=>{
      const field=$('[name=month]',el);
      field.value=(+field.value+1)%3;update();
    };
    $('[data-next]',el).onclick=next;
    $('[data-run]',el).onclick=event=>{
      if(timer){clearInterval(timer);timer=null;}
      else timer=setInterval(next,1600);
      event.currentTarget.textContent=timer?'Pause months':'Play months';
      event.currentTarget.setAttribute('aria-pressed',!!timer);
    };
  };
  labs.pipeline=el=>{
    el.innerHTML=panel('Illustrative methodological diagram','Follow a transaction into an elementary index','<div class="tree-buttons">'+
      ['Raw receipts','Product IDs','Unit values','Matched products','Elementary change'].map((label,i)=>'<button type="button" data-stage="'+i+'">'+label+'</button>').join('<span aria-hidden="true">→</span>')+
      '</div><div data-out></div>','Three synthetic receipt lines across two periods. Unit values are computed within the product and time cell.');
    const data=[['January','A·01',2,6],['January','A·01',3,8.4],['February','A·01',5,15.6],['February','B·01',4,12]];
    const unit=(period,key)=>{const rows=data.filter(r=>r[0]===period&&r[1]===key);return sum(rows.map(r=>r[3]))/sum(rows.map(r=>r[2]));};
    const stages=[
      table(['Month','Barcode','Quantity','Dollars'],data),
      table(['Barcode','Normalized ID','Category'],[['A·01','A01','Oats'],['B·01','B01','Oats']]),
      table(['ID','January','February'],[['A01',fmt(unit('January','A·01'),2),fmt(unit('February','A·01'),2)],['B01','—',fmt(unit('February','B·01'),2)]]),
      table(['ID','Matched','Reason'],[['A01','Yes','Present in both periods'],['B01','No','New in February']]),
      metrics([['Matched A01 price change',fmt(pct(unit('February','A·01'),unit('January','A·01')),2)+'%']])
    ];
    function show(index){
      $$('button',el).forEach((b,i)=>b.setAttribute('aria-pressed',i===index));
      $('[data-out]',el).innerHTML=stages[index];
    }
    el.onclick=e=>{if(e.target.dataset.stage!==undefined)show(+e.target.dataset.stage);};
    show(0);
  };
  labs.formulas=el=>{
    el.innerHTML=panel(simple,'Compare elementary formulas','<div class="controls">'+
      ['Extreme rise','Extreme fall','Missing offer','Size replacement','Uneven weights','Reset'].map((n,i)=>'<button type="button" data-case="'+i+'">'+n+'</button>').join('')+
      '</div>'+makePriceTable(names,prices,true)+'<div data-out></div><div data-chart></div>',
      'Unusable offers are omitted from all matched-pair calculations. The size scenario converts an illustrative 450 g replacement to a 500 g equivalent.');
    const get=name=>$('[name='+name+']',el);
    const update=listen(el,p=>{
      const active=names.map((_,i)=>i).filter(i=>p['use'+i]);
      if(!active.length){$('[data-out]',el).textContent='Select at least one usable offer.';$('[data-chart]',el).textContent='';return;}
      const ratios=active.map(i=>p['t'+i]/p['p'+i]), logs=ratios.map(Math.log), shares=norm(active.map(i=>p['w'+i]));
      const indexes=[
        ['Carli',mean(ratios)],['Dutot',sum(active.map(i=>p['t'+i]))/sum(active.map(i=>p['p'+i]))],
        ['Jevons',Math.exp(mean(logs))],['Weighted Jevons',sum(shares)?Math.exp(sum(shares.map((w,i)=>w*logs[i]))):null]];
      $('[data-out]',el).innerHTML=metrics(indexes.map(([name,v])=>[name,v===null?'—':fmt(100*(v-1),3)+'%']))+
        '<p class="note">Matched '+active.length+' · Sum of log relatives '+fmt(sum(logs),5)+'</p>';
      plot($('[data-chart]',el),active.map(i=>names[i]),[{label:'Individual change',values:ratios.map(x=>100*(x-1)),color:colors[2]}],{x:'Matched product',y:'Change (%)',bars:true});
    });
    el.addEventListener('click',e=>{
      const i=+e.target.dataset.case;if(!Number.isFinite(i))return;
      if(i===5)$$('input',el).forEach(field=>{if(field.type==='checkbox')field.checked=true;else field.value=field.defaultValue;});
      if(i===0)get('t0').value=prices[0]*5;
      if(i===1)get('t0').value=prices[0]*.1;
      if(i===2)get('use0').checked=false;
      if(i===3)get('t0').value=(prices[0]*500/450).toFixed(2);
      if(i===4)get('w0').value=50;
      update();
    });
  };
  labs.aggregate=el=>{
    el.innerHTML=panel(simple,'Change the links in an index tree','<div class="form-grid">'+range('Food change (%)','food',4,-10,25,.1)+
      range('Shelter change (%)','shelter',2,-10,25,.1)+range('Other change (%)','other',1,-10,25,.1)+range('Food weight (%)','fw',20,1,60)+
      range('Shelter weight (%)','sw',35,1,55)+range('Starting index','base',147,70,170,.1)+'</div>'+
      '<div class="tree-buttons">'+['Quotes','Elementary','Product classes','Main components','National CPI'].map((x,i)=>'<button type="button" data-node="'+i+'">'+x+'</button>').join('<i>→</i>')+
      '</div><div data-out></div>','Other keeps a raw weight of 45 before normalization. The example represents a single additive link.');
    let node=4;
    const update=listen(el,p=>{
      const w=norm([p.fw,p.sw,45]), rates=[p.food,p.shelter,p.other], total=sum(rates.map((v,i)=>v*w[i]));
      const views=['One illustrative price pair per domain has a base of 100 and a current value of '+rates.map(r=>fmt(100+r,2)).join(', ')+'.',
        'Relative = current ÷ base. Elementary relatives '+rates.map(r=>fmt(1+r/100,4)).join(', ')+'.',
        'This example has one elementary group in each product class. Its relative passes unchanged to the class.',
        'Normalized shares '+w.map(x=>fmt(100*x,1)+'%').join(', ')+' yield contributions '+rates.map((r,i)=>fmt(r*w[i],2)).join(', ')+' pp.',
        'Multiply the prior index '+fmt(p.base,2)+' by '+fmt(1+total/100,5)+' to chain the new link.'];
      $$('button',el).forEach((b,i)=>b.setAttribute('aria-pressed',i===node));
      $('[data-out]',el).innerHTML='<p>'+views[node]+'</p>'+metrics([['National change',fmt(total,2)+'%'],['Linked index',fmt(p.base*(1+total/100),2)]]);
    });
    el.onclick=e=>{if(e.target.dataset.node!==undefined){node=+e.target.dataset.node;update();}};
  };
  labs.quality=el=>{
    const choices=[['Smartphone',780,840,100,125],['Laptop',950,1020,100,130],['Automobile',30000,33100,100,110],
      ['Grocery package',5,5,500,450],['Appliance',420,460,100,125]];
    el.innerHTML=panel(synthetic,'Separate price, package size and quality','<div class="controls">'+
      select('Example','example',choices.map((row,i)=>[i,row[0]]))+'</div><div class="form-grid">'+
      [['Old price','old',780],['New price','now',840],['Old quantity or score','q0',100],['New quantity or score','q1',125],['Donor change (%)','donor',2]].map(([label,name,v])=>'<label>'+label+input(label,name,v,name==='donor'?-50:.01,50000,.01)+'</label>').join('')+
      '</div><div data-out></div><div data-chart></div>','The synthetic log-price regression uses a one-dimensional quality or quantitative characteristic. Its coefficient only applies to this illustration.');
    const x=Array.from({length:12},(_,i)=>.5+.12*i);
    const y=x.map((v,i)=>4.1+.84*v+.13*Math.sin(2.3*i));
    const mx=mean(x),my=mean(y),beta=sum(x.map((v,i)=>(v-mx)*(y[i]-my)))/sum(x.map(v=>(v-mx)**2)), alpha=my-beta*mx;
    const fitted=x.map(v=>alpha+beta*v),residual=y.map((v,i)=>v-fitted[i]);
    let last=0;
    listen(el,p=>{
      if(+p.example!==last){
        last=+p.example;const values=choices[last].slice(1);
        ['old','now','q0','q1'].forEach((name,i)=>{$('[name='+name+']',el).value=values[i];p[name]=values[i];});
      }
      const adjusted=p.now*Math.exp(-beta*(p.q1/p.q0-1));
      $('[data-out]',el).innerHTML=metrics([['Observed price',fmt(pct(p.now,p.old),2)+'%'],
        ['Unit price',fmt(pct(p.now/p.q1,p.old/p.q0),2)+'%'],['Hedonic price',fmt(pct(adjusted,p.old),2)+'%'],['Donor imputation',fmt(p.donor,2)+'%']])+
        '<p class="note">Intercept '+fmt(alpha,3)+' · Slope '+fmt(beta,3)+' · Residual SD '+fmt(Study.sd(residual),3)+'</p>';
    });
    plot($('[data-chart]',el),x.map(v=>fmt(v,2)),[{label:'Synthetic log price',values:y,scatter:true},{label:'Fitted log price',values:fitted}],
      {title:'Synthetic log-price fit',x:'Characteristic score',y:'Log price',precision:2});
  };
  labs.housing=el=>{
    const house='M-13 2L0 -9 13 2m-23-2v19h20V0M-3 19V7h6v12';
    const rentalDiagram='<svg class="housing-schema" viewBox="0 0 360 100" role="img" aria-label="An owner-occupied dwelling matched to a comparable rental dwelling">'+
      '<path d="'+house+'" transform="translate(78 30)" fill="white" stroke="var(--country)" stroke-width="1.6"/>'+
      '<path d="'+house+'" transform="translate(282 30)" fill="white" stroke="var(--country)" stroke-width="1.6"/>'+
      '<path d="M104 36H255m-6-5 6 5-6 5" fill="none" stroke="var(--country)" stroke-width="1.3"/>'+
      '<g font-size="12" fill="#31454c" text-anchor="middle"><text x="180" y="27">Match dwelling quality</text><text x="78" y="76">Owner-occupied</text>'+
      '<text x="282" y="76">Comparable rental</text><text x="180" y="94">Use comparable rent change</text></g></svg>';
    const diagram=(nodes,label)=>'<svg class="housing-schema" viewBox="0 0 360 100" role="img" aria-label="'+label+'">'+
      '<path d="M180 37V56M68 56H292M68 56V68M180 56V68M292 56V68" stroke="var(--country)" stroke-opacity=".64" fill="none"/>'+
      '<path d="M167 20l13-11 13 11m-23-2v19h20V18m-13 19V25h6v12" fill="white" stroke="var(--country)" stroke-width="1.5"/>'+
      nodes.map((text,i)=>'<g><rect x="'+(15+i*112)+'" y="68" width="106" height="25" rx="5" fill="white" stroke="var(--country)" stroke-opacity=".45"/>'+
        '<text x="'+(68+i*112)+'" y="84" text-anchor="middle" font-size="11" fill="#31454c">'+text+'</text></g>').join('')+'</svg>';
    el.innerHTML=panel(simple,'Compare interest exposure and rental equivalence',
      controls('Interest rates and exposure',4,range('Old mortgage rate (%)','old',3,1,12,.1)+
      range('New mortgage rate (%)','now',5,1,12,.1)+range('Renewal exposure (%)','e',24,0,100)+range('Interest share (%)','mw',20,1,50))+
      controls('Rent and other housing costs',4,range('Market rent change (%)','rent',4,-10,20,.1)+range('Replacement cost change (%)','replacement',3,-10,20,.1)+
      range('Property tax change (%)','tax',2,-10,20,.1)+range('OER share (%)','ow',70,10,95),'secondary')+
      '<div class="eq-strip">'+equation(String.raw`\widetilde{i}_t=(1-\lambda)i_0+\lambda i_1`,'peach')+
      equation(String.raw`\Delta MIC=100\left(\frac{\widetilde{i}_t}{i_0}-1\right)`,'peach')+
      equation(String.raw`r_t^{\mathrm{rent}}=\frac{R_t(q)}{R_0(q)}`,'blue')+'</div>'+
      '<div data-out></div><h4>Published historical shelter series</h4><div data-chart></div>',
      'The Canadian simulation holds mortgage principal fixed and renews a fraction λ at the new rate. The U.S. scenario applies the full market rent shock to quality-constant rental equivalence and 68% to continuing rents. Both estimates are illustrative. Historical observations use agency series and remain independent of the controls.');
    const data=CPI_DATA, start=data.months.indexOf('2015-01'), series=data.series;
    const show=['CA_shelter','CA_interest','CA_replacement','US_shelter','US_oer'];
    listen(el,p=>{
      const renewed=((100-p.e)*p.old+p.e*p.now)/100;
      const interest=100*(renewed/p.old-1), other=norm([21,15,24]);
      const ca=p.mw/100*interest+(1-p.mw/100)*sum(other.map((v,i)=>v*[p.replacement,p.tax,p.rent][i]));
      const us=p.ow/100*p.rent+(1-p.ow/100)*p.rent*.68;
      $('[data-out]',el).innerHTML='<div class="pairs"><div class="country ca"><h4>Canada</h4>'+
        diagram(['Mortgage interest','Replacement cost','Taxes and rent'],'Owned accommodation linked to interest, replacement costs, and taxes; rent is a separate shelter component')+
        metrics([['Blended mortgage rate',fmt(renewed,2)+'%'],['Mortgage-interest change',fmt(interest,2)+'%'],['Illustrative shelter change',fmt(ca,2)+'%']])+
        '</div><div class="country us"><h4>United States</h4>'+
        rentalDiagram+
        metrics([['Rental-equivalent change',fmt(p.rent,2)+'%'],['Illustrative shelter change',fmt(us,2)+'%']])+'</div></div>';
    });
    plot($('[data-chart]',el),data.months.slice(start),show.map(key=>({label:key.slice(0,2)+' '+data.metadata[key].label,values:Study.lag(series[key],12).slice(start)})),{x:'Reference month',y:'12-month change (%)'});
  };
})();

