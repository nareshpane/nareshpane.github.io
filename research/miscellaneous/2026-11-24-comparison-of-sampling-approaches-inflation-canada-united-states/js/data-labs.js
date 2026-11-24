'use strict';
(() => {
  const {$,labs,panel,table,select,range,controls,listen,sum,norm,fmt,pct,lag,mean,sd,corr,metrics,plot,histogram,colors,valid} = Study;
  const official = 'Author-generated analysis of official data';
  const demo = 'Simplified mathematical demonstration';
  const {months,series,metadata,weights,uncertainty,response,effects} = CPI_DATA;
  const mapped=['food','shelter','apparel','transport'];
  const countryNames=['Canada','United States'];
  const countrySeries=(a,b)=>[{label:'Canada',values:a,color:colors[0]},{label:'United States',values:b,color:colors[1]}];
  const periods=Object.keys(weights.CA).reverse();
  function dateControls(from,to) {
    return '<label>From <input type="month" name="from" min="'+months[0]+'" max="'+months.at(-1)+'" value="'+from+'"></label>'+
      '<label>Through <input type="month" name="through" min="'+months[0]+'" max="'+months.at(-1)+'" value="'+to+'"></label>';
  }
  function extent(from,to) {
    const start=months.findIndex(m=>m>=from);
    const end=months.findLastIndex(m=>m<=to);
    return [start<0?months.length:start,end+1];
  }
  labs.weights=el=>{
    const short={
      'Household operations, furnishings and equipment':'Household operations',
      'Recreation, education and reading':'Recreation, education, reading',
      'Alcoholic beverages, tobacco products and recreational cannabis':'Alcohol, tobacco and cannabis',
      'Education and communication':'Education and communication'
    };
    el.innerHTML=panel(official,'Compare published expenditure distributions',
      '<div class="controls">'+select('Snapshot year','year',periods)+select('View','view',['Bars','Stack','Treemap','History'])+'</div>'+
      '<div data-vintage></div><div data-out></div>'+
      controls('Hypothetical common price shocks',4,mapped.map((key,i)=>range(weights.mapping[key][0]+' shock (%)','v'+i,[4,3,1,2][i],-20,30,.1)).join(''))+
      '<div data-fig></div>'+
      '<details class="derivation"><summary>Category mapping for the shock experiment</summary>'+table(['Canada','United States','Comparison'],[
        ['Food','Food','Broad food scope, with different internal classification.'],
        ['Shelter','Shelter','BLS shelter is inside its broader Housing group. Canada includes owned-accommodation costs.'],
        ['Clothing','Apparel','Approximate scope match. Footwear and other items differ from the all-group display.'],
        ['Transportation','Transportation','Approximate match with different weights and composition.']
      ])+'</details>',
      'Canada uses link-month weights for the selected expenditure year. The U.S. panel uses December relative importance with the CE vintage applied to the following January index. For example, December 2025 uses 2024 spending. Major group boundaries differ. Unmapped groups receive zero shock.');
    const draw=(country,view,year)=>{
      const vals=weights[country][year], names=weights.major[country], labels=names.map((label,i)=>[label,vals[label],colors[i]]);
      if(view==='History') return table(['Major group',...periods],names.map(label=>[label,...periods.map(p=>fmt(weights[country][p][label],1)+'%')]));
      const legend='<div class="plot-legend">'+labels.map(([name,v,c])=>'<span style="--key:'+c+'">'+name+' '+fmt(v,1)+'%</span>').join('')+'</div>';
      if(view==='Stack') return '<div class="stack" aria-hidden="true">'+labels.map(([name,v,c])=>'<span title="'+name+' '+fmt(v,1)+'%" style="width:'+v+'%;background:'+c+'"></span>').join('')+'</div>'+legend;
      if(view==='Treemap'){
        const columns=[labels.slice(0,2),labels.slice(2,5),labels.slice(5)];
        let x=0;
        const shapes=columns.map(column=>{
          const subtotal=sum(column.map(r=>r[1])), width=subtotal*4;
          let y=0;
          const rects=column.map(([name,v,c])=>{
            const h=v/subtotal*120,tag='<rect x="'+x+'" y="'+y+'" width="'+width+'" height="'+h+'" fill="'+c+'" stroke="white"><title>'+name+' '+fmt(v,1)+'%</title></rect>';
            y+=h;return tag;
          }).join('');
          x+=width;return rects;
        }).join('');
        return '<svg viewBox="0 0 400 120" width="100%" role="img" aria-label="Expenditure share treemap">'+shapes+'</svg>'+legend;
      }
      const scale=50;
      return '<div class="weight-bars">'+labels.map(([name,v])=>'<div class="barrow"><span title="'+name+'" aria-label="'+name+'">'+(short[name]||name)+
        '</span><span class="bartrack"><span style="width:'+100*v/scale+'%"></span></span><span>'+fmt(v,1)+'%</span></div>').join('')+
        '</div><p class="note">Shared bar scale · 0 to '+scale+'% of the basket</p>';
    };
    listen(el,p=>{
      const year=p.year, shocks=mapped.map((k,i)=>p['v'+i]);
      const compare=['CA','US'].map((key,j)=>sum(mapped.map((k,i)=>weights[key][year][weights.mapping[k][j]]/100*shocks[i])));
      const links={2025:'April 2026',2024:'April 2025',2023:'April 2024'};
      $('[data-vintage]',el).innerHTML='<p class="note"><strong>Canada</strong> '+year+' spending, price-updated to '+links[year]+
        ' at the basket link. <strong>United States</strong> December '+year+' relative importance based on '+(+year-1)+
        ' Consumer Expenditure data for the following CPI year. <a href="#ca-weights">Statistics Canada weights</a> · <a href="#us-weights">BLS relative importance</a></p>';
      $('[data-out]',el).innerHTML='<div class="pairs weight-pair">'+['CA','US'].map((c,i)=>'<div class="country '+c.toLowerCase()+'"><h4>'+countryNames[i]+'</h4>'+draw(c,p.view,year)+'</div>').join('')+'</div>';
      const food=['CA','US'].map((key,j)=>{
        const weight=weights[key][year][weights.mapping.food[j]];
        return [countryNames[j],fmt(weight,2)+'%',fmt(shocks[0],1)+'%',fmt(weight*shocks[0]/100,3)+' pp'];
      });
      $('[data-fig]',el).innerHTML='<h4>Food contribution at the current shock</h4>'+table(['Economy','Official food share','Hypothetical change','Contribution'],food)+
        metrics([['Canada shock effect',fmt(compare[0],2)+' pp'],['U.S. shock effect',fmt(compare[1],2)+' pp'],['Canada − U.S.',fmt(compare[0]-compare[1],2)+' pp']]);
    });
  };
  labs.variance=el=>{
    const categories=[...new Set(uncertainty.map(r=>r.item))];
    el.innerHTML=panel(official,'Compare annual median standard errors','<div class="controls">'+
      select('Year','year',[2025,2024,2023])+select('Series','item',categories)+'</div><div data-chart></div><div data-out></div>',
      'Source: BLS Tables 1V, U.S. city average. BLS publishes the annual medians of standard errors for each change interval directly. A standard error reflects sampling variability in the estimated CPI change, in percentage points. Canada’s judgmental components limit directly comparable design-based measures.');
    listen(el,p=>{
      const rows=uncertainty.filter(r=>r.year===+p.year && r.item===p.item);
      plot($('[data-chart]',el),rows.map(r=>String(r.months)),[{label:'Median standard error',values:rows.map(r=>r.se_pp),color:colors[1]}],
        {title:'BLS standard errors for change intervals',bars:true,barWidth:18,barLabels:true,tickPrecision:2,x:'Change interval (months)',y:'Standard error (pp)',precision:2});
      $('[data-out]',el).innerHTML=table(['Interval','Median change (%)','Median SE (pp)'],rows.map(r=>[r.months+(r.months===1?' month':' months'),fmt(r.median_change,2),fmt(r.se_pp,2)]));
    });
  };
  labs.response=el=>{
    const domains=[...new Set(response.map(row=>row.domain))];
    el.innerHTML=panel(official,'Follow BLS collection and estimation rates','<div class="controls">'+
      select('Survey unit','domain',domains)+'</div><div data-chart></div><div data-latest></div>',
      'Source: BLS monthly response-rate workbook, January 2019–October 2026. Housing observations are reported units. The published October 2025 gap remains visible.');
    listen(el,p=>{
      const rows=response.filter(r=>r.domain===p.domain), map=new Map(rows.map(r=>[r.month,r]));
      const dates=months.filter(m=>m>='2019-01'), latest=map.get(dates.at(-1));
      const values=k=>dates.map(d=>map.get(d)?.[k]??null);
      plot($('[data-chart]',el),dates,[{label:'Collected or reported',values:values('observed_pct'),color:colors[1]},
        {label:'Used in estimation',values:values('used_pct'),color:colors[2]}],{y:'Percent of targeted units'});
      $('[data-latest]',el).innerHTML=metrics([['October 2026 targeted',latest.targeted.toLocaleString('en-US')],
        ['Collected or reported',fmt(latest.observed_pct)+'%'],['Used in estimation',fmt(latest.used_pct)+'%']]);
    });
  };
  labs.effects=el=>{
    el.innerHTML=panel('Official data visualization','BLS published seasonally adjusted effects',
      table(['October 2026 category','Change (%)','Effect on all items (pp)'],effects.filter(r=>valid(r.official_effect_pp)).map(r=>
        [r.label,fmt(r.adjusted_monthly_change_pct),fmt(r.official_effect_pp,3)])),
      'Source: BLS October 2026 release, Table 6. Overlapping groups such as food and food at home must not be summed.');
  };
  const nsa=countrySeries(series.CA_all,series.US_all);
  const sa=countrySeries(series.CA_sa,series.US_sa);
  labs.season=el=>{
    el.innerHTML=panel(official,'Inspect seasonal and unadjusted movements','<div class="controls">'+select('Country','country',['CA','US'])+
      select('View','view',[['both','Official raw and adjusted'],['raw','Official unadjusted'],['sa','Official adjusted'],['trend','Illustrative trend'],
        ['factor','Illustrative seasonal component'],['mom','Monthly change'],['yoy','Twelve-month change']])+
      dateControls('2019-01',months.at(-1))+'</div><div data-chart></div>',
      'Sources: Statistics Canada 18-10-0004 and 18-10-0006; BLS CPI-U API. The illustrative calculation uses a centered 13-month geometric mean and averaged log deviations by calendar month.');
    listen(el,p=>{
      const a=series[p.country+'_all'], b=series[p.country+'_sa'];
      const trend=a.map((_,i)=>{
        const window=a.slice(i-6,i+7);
        return window.length===13&&window.every(valid)?Math.exp(mean(window.map(Math.log))):null;
      });
      const factors=Array.from({length:12},(_,i)=>mean(a.map((v,j)=>j%12===i&&valid(v)&&valid(trend[j])?Math.log(v/trend[j]):null).filter(valid)));
      const center=mean(factors);
      const [start,end]=extent(p.from,p.through);
      const s=(label,values,index=0)=>({label,values:values.slice(start,end),color:colors[index]});
      const views={
        both:[s('Official unadjusted',a),s('Official adjusted',b,1)],
        raw:[s('Official unadjusted',a)],sa:[s('Official adjusted',b,1)],
        trend:[s('Illustrative trend',trend,2),s('Official unadjusted',a)],
        factor:[s('Illustrative factor minus one',a.map((v,i)=>valid(v)?100*Math.expm1(factors[i%12]-center):null),2)],
        mom:[s('Unadjusted monthly',lag(a,1)),s('Adjusted monthly',lag(b,1),1)],
        yoy:[s('Unadjusted twelve-month',lag(a,12))]
      };
      plot($('[data-chart]',el),months.slice(start,end),views[p.view],{y:['mom','yoy','factor'].includes(p.view)?'Change (%)':'Index'});
    });
  };
  labs.audit=el=>{
    const i=months.length-1, ca=series.CA_all, us=series.US_all;
    el.innerHTML=panel(official,'October 2026 release arithmetic',
      table(['Measure','Published','From index levels'],[
        ['Canada one month unadjusted','0.2%',fmt(pct(ca[i],ca[i-1]))+'%'],
        ['Canada twelve months unadjusted','3.1%',fmt(pct(ca[i],ca[i-12]))+'%'],
        ['Canada one month adjusted','0.2%',fmt(pct(series.CA_sa[i],series.CA_sa[i-1]))+'%'],
        ['U.S. one month adjusted','0.4%',fmt(pct(series.US_sa[i],series.US_sa[i-1]))+'%'],
        ['U.S. twelve months unadjusted','—',valid(pct(us[i],us[i-12]))?'Unexpected value':'—']
      ]),'November 2026 release tables, displayed at publication precision.');
  };
  labs.empirical=el=>{
    el.innerHTML=panel(official,'Track national headline movements','<div class="controls">'+
      select('Measure','metric',[['rebase','Rebased index'],['cumulative','Cumulative inflation'],['monthly','Monthly unadjusted'],['sa','Monthly adjusted'],
        ['annual','Twelve-month inflation'],['gap','Annual differential'],['monthly_gap','Monthly adjusted differential']])+
      dateControls('2000-01',months.at(-1))+
      '<label><input type="checkbox" name="ca" checked> Canada</label><label><input type="checkbox" name="us" checked> United States</label></div>'+
      '<div class="controls"><button type="button" data-zoom="2000-01">Full period</button><button type="button" data-zoom="2020-01">Since 2020</button>'+
      '<button type="button" data-zoom="2025-08">Shutdown detail</button></div><div data-chart></div>',
      'Sources: Statistics Canada, 18-10-0004 and 18-10-0006, and BLS API, CUUR0000SA0 and CUSR0000SA0. The October 2025 U.S. gap is left open.');
    const update=listen(el,p=>{
      const [start,end]=extent(p.from,p.through);
      let view, y='Change (%)';
      if(p.metric==='gap'||p.metric==='monthly_gap'){
        const [a,b]=(p.metric==='gap'?nsa:sa).map(s=>lag(s.values,p.metric==='gap'?12:1));
        view=[{label:'Canada − United States',values:a.map((v,i)=>valid(v)&&valid(b[i])?v-b[i]:null),color:colors[2]}];
        y='Difference (pp)';
      } else {
        const source=p.metric==='sa'?sa:nsa;
        view=source.map(s=>({...s,values:p.metric==='rebase'?s.values.map(x=>valid(x)?100*x/s.values[0]:null):
          p.metric==='cumulative'?s.values.map(x=>pct(x,s.values[0])):lag(s.values,p.metric==='annual'?12:1)}))
          .filter((_,i)=>i?p.us:p.ca);
        if(p.metric==='rebase')y='Jan 2000 = 100';
      }
      plot($('[data-chart]',el),months.slice(start,end),view.map(s=>({...s,values:s.values.slice(start,end)})),{y});
    });
    el.addEventListener('click',event=>{
      if(event.target.dataset.zoom){$('[name=from]',el).value=event.target.dataset.zoom;$('[name=through]',el).value=months.at(-1);update();}
    });
  };
  labs.components=el=>{
    const keys=[...mapped,'energy','gasoline','rent','core','food_home'];
    el.innerHTML=panel(official,'Compare mapped components','<div class="controls">'+
      select('Component','category',keys.map(k=>[k,metadata['CA_'+k].label]))+
      select('Measure','view',[['annual','Twelve-month inflation'],['monthly','Monthly unadjusted'],['level','Rebased index']])+
      dateControls('2015-01',months.at(-1))+'</div><p data-map class="note"></p><div data-chart></div>',
      'Sources: Statistics Canada table 18-10-0004 and BLS CPI-U API. These crosswalks retain each agency’s native classification.');
    listen(el,p=>{
      const [start,end]=extent(p.from,p.through), ca=series['CA_'+p.category], us=series['US_'+p.category];
      $('[data-map]',el).textContent='Statistics Canada: '+metadata['CA_'+p.category].label+' · BLS: '+metadata['US_'+p.category].label+'.';
      const values=[ca,us].map(vals=>{
        if(p.view!=='level')return lag(vals,p.view==='annual'?12:1);
        const i=ca.findIndex((v,j)=>valid(v)&&valid(us[j]));
        const base=vals[i];
        return vals.map(v=>valid(v)&&valid(base)?100*v/base:null);
      });
      plot($('[data-chart]',el),months.slice(start,end),countrySeries(...values.map(v=>v.slice(start,end))),
        {y:p.view==='level'?'First common month = 100':'Change (%)'});
    });
  };
  labs.counter=el=>{
    el.innerHTML=panel(official,'Cross-apply mapped expenditure baskets','<div class="controls">'+
      select('Weight snapshot','year',periods)+select('Comparison month','month',months.slice(-34).reverse())+
      select('Pricing treatment','version',[['updated','Price-updated weights'],['static','Snapshot weights']])+
      '</div><div data-out></div>','The four mapped categories are a partial basket. U.S. vintage is December of the selected year. Canadian link months are recorded from the annual basket timing. Comparisons outside the active vintage act as counterfactuals.');
    listen(el,p=>{
      const i=months.indexOf(p.month), year=+p.year;
      const canadaLinks={2023:'2024-04',2024:'2025-04',2025:'2026-04'};
      const mixed=['CA','US'].map((c,j)=>{
        const base=months.indexOf(j?year+'-12':canadaLinks[year]);
        const raw=mapped.map(k=>weights[c][p.year][weights.mapping[k][j]]/100);
        const levels=mapped.map(k=>series[c+'_'+k]);
        const update=levels.map((s,k)=>valid(s[i-1])&&valid(s[base])?raw[k]*s[i-1]/s[base]:null);
        const updated=p.version==='updated'&&update.every(valid);
        const current=updated?update.map(v=>v/sum(update)*sum(raw)):raw;
        return {current, normalized:norm(current), rates:levels.map(s=>pct(s[i],s[i-1])), updated};
      });
      const [ca,us]=mixed, common=ca.normalized.map((v,j)=>(v+us.normalized[j])/2);
      const dot=(w,r)=>r.every(valid)?sum(w.map((v,j)=>v*r[j])):null;
      const show=x=>valid(x)?fmt(x,2)+' pp':'—';
      const combos=[
        ['Canadian partial weights',dot(ca.normalized,ca.rates),dot(ca.normalized,us.rates)],
        ['American partial weights',dot(us.normalized,ca.rates),dot(us.normalized,us.rates)],
        ['Common partial weights',dot(common,ca.rates),dot(common,us.rates)]
      ].map(([name,a,b])=>[name,show(a),show(b)]);
      const headline=['CA','US'].map(c=>pct(series[c+'_all'][i],series[c+'_all'][i-1]));
      const parts=ca.current.map((v,j)=>[weights.mapping[mapped[j]][0],show(valid(ca.rates[j])?v*ca.rates[j]:null),show(valid(us.rates[j])?us.current[j]*us.rates[j]:null)]);
      const contribution=mixed.map(x=>dot(x.current,x.rates));
      parts.push(['Partial contribution',...contribution.map(show)]);
      parts.push(['Headline minus partial',...headline.map((v,j)=>show(valid(v)&&valid(contribution[j])?v-contribution[j]:null))]);
      $('[data-out]',el).innerHTML='<p class="note">Partial basket share: Canada '+fmt(100*sum(ca.current))+'%, U.S. '+fmt(100*sum(us.current))+'%. Applied shares: '+mixed.map(x=>x.updated?'within-partition price-updated':'published snapshot').join(' / ')+'.</p>'+
        table(['Weights','Canadian price changes','American price changes'],combos)+
        '<h4>Contribution to headline growth</h4>'+table(['Mapped category','Canada','United States'],parts);
    });
  };
  labs.rolling=el=>{
    el.innerHTML=panel(official,'Inspect dependence in national monthly inflation','<div class="controls">'+
      select('Statistic','view',[['corr','Rolling correlation'],['mean','Rolling mean'],['sd','Rolling standard deviation'],['acf','Autocorrelation'],
        ['cross','Cross-correlation'],['dist','Monthly distributions'],['gap','Monthly differentials']])+
      dateControls(months[0],months.at(-1))+'</div><div class="form-grid">'+range('Rolling calendar months or maximum lag','window',36,6,60)+'</div><div data-chart></div>',
      'Sources: official seasonally adjusted national all-items indexes. ACF and cross-correlation use pairwise valid observations at each lag. Rolling correlations are descriptive.');
    const x=lag(series.CA_sa,1), y=lag(series.US_sa,1);
    listen(el,p=>{
      const [a,b]=extent(p.from,p.through);
      const xx=x.slice(a,b), yy=y.slice(a,b), out=$('[data-chart]',el);
      if(p.view==='dist'){histogram(out,countrySeries(xx,yy),{x:'monthly inflation (%)'});return;}
      if(p.view==='acf'||p.view==='cross'){
        const lags=Array.from({length:p.window+1},(_,i)=>i);
        const acf=(x,y)=>lags.map(k=>corr(x.slice(k),y.slice(0,y.length-k)));
        const s=p.view==='acf'?countrySeries(acf(xx,xx),acf(yy,yy)):[{label:'CA at t versus US at t − k',values:acf(xx,yy)}];
        plot(out,lags,s,{x:'Lag (months)',y:'Correlation',precision:2,zero:true});return;
      }
      let lines, unit='Inflation (%)';
      const window=(s,i)=>s.slice(Math.max(0,i+1-p.window),i+1);
      if(p.view==='gap'){
        lines=[{label:'Canada − United States',values:xx.map((v,i)=>valid(v)&&valid(yy[i])?v-yy[i]:null)}];unit='Difference (pp)';
      } else if(p.view==='corr'){
        lines=[{label:'Canada and United States',values:xx.map((_,i)=>i<p.window-1?null:corr(window(xx,i),window(yy,i)))}];unit='Correlation';
      } else {
        const roll=s=>s.map((_,i)=>i<p.window-1?null:(p.view==='mean'?mean:sd)(window(s,i).filter(valid)));
        lines=countrySeries(roll(xx),roll(yy));
        if(p.view==='sd')unit='Std. deviation (pp)';
      }
      plot(out,months.slice(a,b),lines,{y:unit,precision:2});
    });
  };
  labs.timeline=el=>{
    const releases=[
      ['October 2025','Oct 2025','Nov 2025','Canadian release on November 17, 2025.','BLS could not publish most October 2025 indexes. See the shutdown FAQ.'],
      ['September 2026','Sep 2026','Oct 2026','Canadian release on October 19, 2026.','U.S. release on October 14, 2026.'],
      ['October 2026','Oct 2026','Nov 2026','Canadian release on November 16, 2026.','U.S. release on November 10, 2026.']
    ];
    el.innerHTML=panel('Illustrative methodological diagram with official release dates','Follow observation to publication',
      '<div class="controls">'+select('Reference month','month',releases.map((r,i)=>[i,r[0]]).reverse())+
      select('Stage','stage',['Observation','Review and aggregation','Publication'])+'</div><div data-out></div>',
      'Sources: The Daily, BLS releases and the BLS shutdown FAQ. Review and aggregation are schematic.');
    listen(el,p=>{
      const r=releases[+p.month];
      const text=p.stage==='Observation'?[r[1]+' · scheduled collection and source records.',r[1]+' · '+(+p.month===0?'survey collection suspended.':'scheduled survey and secondary sources.')]:
        p.stage==='Review and aggregation'?['Match and adjust quotes, then calculate and aggregate index links.','Review usable quotes, impute where appropriate and aggregate cost weights.']:r.slice(3);
      $('[data-out]',el).innerHTML='<div class="flow"><span>'+r[1]+' observations</span><i>→</i><span>Review and aggregation</span><i>→</i><span>'+r[2]+' publication</span></div>'+
        '<div class="pairs"><div class="country ca"><h4>Canada</h4><p>'+text[0]+'</p></div><div class="country us"><h4>United States</h4><p>'+text[1]+'</p></div></div>';
    });
  };
})();

