'use strict';
const Study = (() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const asset = file => $('#study').dataset.assets + file;
  const colors = ['#a83f31', '#245b91', '#427469', '#775985', '#ae7039', '#507b8c', '#875469', '#707764'];
  const valid = Number.isFinite;
  const sum = values => values.reduce((total, value) => total + value, 0);
  const mean = values => values.length ? sum(values) / values.length : null;
  const sd = values => values.length > 1 ? Math.sqrt(sum(values.map(x => (x - mean(values)) ** 2)) / (values.length - 1)) : null;
  const fmt = (number, decimals = 1) => valid(number) ? number.toFixed(decimals) : '—';
  const pct = (current, base) => valid(current) && valid(base) && base > 0 ? 100 * (current / base - 1) : null;
  const lag = (values, n) => values.map((value, index) => pct(value, values[index - n]));
  const norm = values => {
    const total = sum(values);
    return total > 0 ? values.map(value => value / total) : values.map(() => 0);
  };
  const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  function corr(xs, ys) {
    const pairs = xs.map((x, i) => [x, ys[i]]).filter(pair => pair.every(valid));
    if (pairs.length < 3) return null;
    const mx = mean(pairs.map(row => row[0])), my = mean(pairs.map(row => row[1]));
    const dx = pairs.map(row => row[0] - mx), dy = pairs.map(row => row[1] - my);
    const scale = Math.sqrt(sum(dx.map(v => v*v)) * sum(dy.map(v => v*v)));
    return scale ? sum(dx.map((v,i) => v*dy[i])) / scale : null;
  }
  const input = (label, name, value, min = 0, max = 99999, step = .1) =>
    '<input type="number" name="' + name + '" value="' + value + '" min="' + min + '" max="' + max + '" step="' + step + '" aria-label="' + escape(label) + '">';
  const select = (label, name, choices) =>
    '<label>' + label + ' <select name="' + name + '">' + choices.map(choice => {
      const [value, text] = Array.isArray(choice) ? choice : [choice, choice];
      return '<option value="' + escape(value) + '">' + escape(text) + '</option>';
    }).join('') + '</select></label>';
  const range = (label, name, value, min, max, step = 1) =>
    '<label><span class="slider-label">' + label + '</span><output>' + value + '</output><input type="range" name="' + name + '" value="' + value + '" min="' + min + '" max="' + max + '" step="' + step + '"></label>';
  const controls = (title, cols, content, tone = '') =>
    '<div class="parameter-group '+tone+'"><h5>'+title+'</h5><div class="form-grid parameter-grid" data-cols="'+cols+'">'+content+'</div></div>';
  const equation = (tex, tone = 'blue') =>
    '<figure class="equation math-'+tone+'"><div class="tex">'+escape(tex)+'</div></figure>';
  const picks = element => Object.fromEntries($$('input,select', element).filter(x => x.name)
    .map(x => [x.name, x.type === 'checkbox' ? x.checked : ['range', 'number'].includes(x.type) ?
      Math.min(+x.max || Infinity, Math.max(+x.min, x.valueAsNumber || 0)) : x.value]));
  const panel = (type, title, body, note = '') => '<p class="type">' + type + '</p><h4>' + title + '</h4>' + body +
    (note ? '<p class="source">' + note + '</p>' : '');
  const metric = (label, val, tone = '') => '<div class="metric '+tone+'"><b>' + val + '</b><small>' + label + '</small></div>';
  const metrics = content => '<div class="metrics">' + content.map(row => metric(...row)).join('') + '</div>';
  const table = (head, rows, klass = '') => '<table class="' + klass + '"><thead><tr>' +
    head.map(column => '<th scope="col">' + column + '</th>').join('') + '</tr></thead><tbody>' +
    rows.map(row => '<tr>' + row.map((val,i) => i ? '<td>' + val + '</td>' : '<th scope="row">' + val + '</th>').join('') + '</tr>').join('') +
    '</tbody></table>';
  function listen(element, handler) {
    function update() {
      $$('input[type=range][name]',element).forEach(x => {
        const decimals = (x.step.split('.')[1] || '').length;
        x.previousElementSibling.value = (+x.value).toFixed(decimals);
      });
      handler(picks(element));
    }
    element.addEventListener('input', update);
    update();
    return update;
  }
  const plotSpecs = new Map();
  const clearPlot = el => plotSpecs.delete(el);
  let resizeFrame = 0;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
      for (const [el, args] of plotSpecs) {
        if(el.isConnected && el.offsetWidth > 0) plot(el, ...args);
      }
    });
  }, {passive:true});
  function plot(el, labels, series, options = {}) {
    plotSpecs.set(el, [labels, series, options]);
    const values = series.flatMap(s => s.values.filter(valid));
    const rangeControl = el._plotFocus ?? Math.max(0,labels.length - 1);
    if (!values.length || !labels.length) {
      el.innerHTML = '<p class="empty">No observations in this view.</p>';
      return;
    }
    const W = Math.max(260, Math.min(950, el.clientWidth || 740));
    const compact = W < 450;
    const H = options.height || (compact ? 198 : 230);
    let lo = Math.min(...values), hi = Math.max(...values);
    if (options.bars || options.zero) {lo = Math.min(lo,0); hi = Math.max(hi,0);}
    const spread = hi-lo || Math.max(Math.abs(lo)/8, .2);
    const ideal = spread / 4;
    const order = 10 ** Math.floor(Math.log10(ideal));
    const interval = [1, 2, 2.5, 5, 10].find(k => k >= ideal/order - 1e-9)*order;
    const firstTick = Math.floor(lo/interval + 1e-8)*interval;
    const lastTick = Math.max(firstTick+interval, Math.ceil(hi/interval - 1e-8)*interval);
    const ticksY = Array.from({length:1+Math.round((lastTick-firstTick)/interval)}, (_,i) => firstTick+i*interval);
    const decimals = Math.min(8, Math.max(options.tickPrecision ?? 0, Math.ceil(-Math.log10(interval)) + (Math.abs(interval/order-2.5)<.01 ? 1 : 0)));
    const tickLabel = v => fmt(Math.abs(v)<interval*1e-8 ? 0 : v,decimals);
    const longest = Math.max(...ticksY.map(v=>tickLabel(v).length));
    const x0 = Math.max(57, 35 + longest*(compact ? 6 : 7));
    const y0 = options.barLabels ? 29 : 17;
    const right = W-16, bottom = H-45, n = labels.length;
    const y = value => bottom-(bottom-y0)*(value-firstTick)/(lastTick-firstTick);
    const binEdges = options.binEdges;
    const position = options.numericX && !binEdges;
    const xMin = binEdges ? binEdges[0] : position ? labels[0] : 0;
    const xMax = binEdges ? binEdges.at(-1) : position ? labels.at(-1) : options.bars ? n : n-1;
    const scaleX = v => x0+(right-x0)*(xMax===xMin ? .5 : (v-xMin)/(xMax-xMin));
    const x = i => scaleX(binEdges ? (binEdges[i]+binEdges[i+1])/2 : position ? labels[i] : options.bars ? i+.5 : i);
    const rules = ticksY.map(value => {
      const pos = y(value);
      return '<path class="grid" d="M'+x0+','+pos+'H'+right+'"/><text class="axis" x="'+(x0-8)+'" y="'+(pos+4)+'" text-anchor="end">'+tickLabel(value)+'</text>';
    }).join('');
    const fractions = compact ? [0,.5,1] : [0,.25,.5,.75,1];
    const indexes = n<=5 ? labels.map((_,i)=>i) : [...new Set(fractions.map(f=>Math.floor((n-1)*f)))];
    const decimalsX = binEdges ? Math.min(8,Math.max(2,Math.ceil(-Math.log10((xMax-xMin)/(fractions.length-1))))) : 1;
    const ticksX = binEdges ? fractions.map(f=>{
      const v=xMin+(xMax-xMin)*f;
      return [scaleX(v),fmt(v,options.xPrecision ?? decimalsX)];
    }) : indexes.map(i=>[x(i),position?fmt(labels[i]):labels[i]]);
    const ticks = ticksX.map(([px,label],i) => '<text class="axis" x="'+px+'" y="'+(bottom+17)+'" text-anchor="'+(options.bars&&!binEdges?'middle':i===0?'start':i===ticksX.length-1?'end':'middle')+'">'+escape(label)+'</text>').join('');
    const zero = firstTick < 0 && lastTick > 0 ? '<path class="reference-line" stroke="#899b9d" d="M'+x0+','+y(0)+'H'+right+'"/>' : '';
    const vertical = (options.references || []).filter(r=>r.value>=xMin&&r.value<=xMax).map(r=>{
      const px=scaleX(r.value);
      return '<path class="reference-line" stroke="'+r.color+'" stroke-dasharray="'+(r.dash||'4 3')+'" d="M'+px+','+y0+'V'+bottom+'"><title>'+escape(r.label)+'</title></path>';
    }).join('');
    const traces = series.map((s,j) => {
      const color = s.color || colors[j % colors.length];
      if (options.bars) {
        const slot = (right-x0)/n;
        const width = binEdges ? slot : Math.min(options.barWidth || 23, slot*.77/series.length);
        return s.values.map((v,i) => {
          if (!valid(v) || v===0) return '';
          const left = binEdges ? scaleX(binEdges[i]) : x(i)+width*(j-series.length/2);
          const top = Math.min(y(v),y(0));
          return '<rect class="plot-bar" x="'+left+'" y="'+top+'" width="'+width+'" height="'+Math.abs(y(0)-y(v))+'" fill="'+color+'" opacity="'+(binEdges&&series.length>1?.45:1)+'"/>'+
            (options.barLabels ? '<text class="bar-value" x="'+(left+width/2)+'" y="'+(top-5)+'" text-anchor="middle">'+fmt(v,options.precision ?? 2)+'</text>' : '');
        }).join('');
      }
      let data = '', active = false;
      s.values.forEach((v,i) => {
        if (!valid(v)) {active = false; return;}
        data += (active?'L':'M')+x(i).toFixed(2)+','+y(v).toFixed(2);
        active = true;
      });
      const dots = n < 25 || options.scatter || s.scatter ? s.values.map((v,i) => valid(v) ? '<circle cx="'+x(i)+'" cy="'+y(v)+'" r="2.6" fill="'+color+'"/>' : '').join('') : '';
      return (options.scatter || s.scatter ? '' : '<path class="line" stroke="'+color+'"'+(s.dash?' stroke-dasharray="'+s.dash+'"':'')+' d="'+data+'"/>')+dots;
    }).join('');
    el.innerHTML = '<figure><svg class="chart" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+escape(options.title || 'Statistical plot')+'">'+
      rules+ticks+zero+traces+vertical+'<path class="focus-line" d=""/><text class="axis-title" x="'+x0+'" y="'+(H-5)+'">'+escape(options.x || 'Reference month')+
      '</text><text class="axis-title" text-anchor="middle" transform="translate(14 '+((bottom+y0)/2)+') rotate(-90)">'+escape(options.y || '')+'</text></svg>'+
      '<div class="plot-legend">'+series.map((s,i)=>'<span style="--key:'+(s.color||colors[i%colors.length])+'">'+escape(s.label)+'</span>').join('')+(options.references||[]).map(r=>'<span style="--key:'+r.color+'">'+escape(r.label)+'</span>').join('')+'</div>'+
      '<label class="note">Inspect observation <input type="range" min="0" max="'+(n-1)+'" value="'+Math.min(rangeControl,n-1)+'" aria-label="Inspect observation"></label>'+
      '<div class="plot-readout" aria-live="polite"></div></figure>';
    const svg = $('svg',el), focus = $('.focus-line',svg), readout = $('.plot-readout',el), scrub = $('input',el);
    function show(i) {
      el._plotFocus = i; scrub.value = i;
      focus.setAttribute('d','M'+x(i)+','+y0+'V'+bottom);
      readout.textContent = (binEdges ? fmt(binEdges[i],3)+' to '+fmt(binEdges[i+1],3) : position ? fmt(labels[i]) : labels[i])+' · '+series.map(s => s.label+' '+fmt(s.values[i],options.precision ?? 1)).join(' · ');
    }
    scrub.addEventListener('input',event => {event.stopPropagation(); show(+scrub.value);});
    svg.addEventListener('pointermove', event => {
      const rect = svg.getBoundingClientRect();
      const ratio = Math.min(1,Math.max(0,((event.clientX-rect.left)/rect.width*W-x0)/(right-x0)));
      const position=x0+(right-x0)*ratio;
      let best=0;
      for(let i=1;i<n;i++)if(Math.abs(x(i)-position)<Math.abs(x(best)-position))best=i;
      show(best);
    });
    show(+scrub.value);
  }
  function histogram(el, lists, options = {}) {
    const all = lists.flatMap(s=>s.values.filter(valid));
    if (!all.length) return;
    const marks = options.references || [{value:0,label:'Zero error',color:'#293c44'}];
    const lo = Math.min(...all,...marks.map(m=>m.value));
    const hi = Math.max(...all,...marks.map(m=>m.value));
    const count = 17, extent = hi-lo || .1;
    const min = lo-extent*.04, width = extent*1.08/count;
    const edges = Array.from({length:count+1},(_,i)=>min+i*width);
    const labels = edges.slice(0,-1).map((edge,i)=>(edge+edges[i+1])/2);
    const series = lists.map(s => {
      const values = Array(count).fill(0);
      s.values.filter(valid).forEach(v => values[Math.min(count-1,Math.max(0,Math.floor((v-min)/width)))]++);
      return {...s,values};
    });
    plot(el,labels,series,{title:'Histogram',bars:true,binEdges:edges,references:marks,x:options.x || 'Estimation error (pp)',y:'Count',precision:0});
  }
  const labs = {};
  return {$,$$,asset,colors,valid,sum,mean,sd,fmt,pct,lag,norm,corr,escape,input,select,range,controls,equation,picks,panel,metric,metrics,table,listen,plot,clearPlot,histogram,labs};
})();

