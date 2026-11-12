/* Dependency-free annual export atlas. Values are integer CAD; HS codes stay strings. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const DATA = '3-alberta-trade-by-hs/data/';
  const colors = ['#99c4b6','#9fbfcf','#b8cdb0','#8fb8bc','#abc7bd','#b5d3cf','#8eafc0'];
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const normalized = value => String(value).normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[^a-z0-9]+/g,' ').trim();
  const money = value => value >= 1e9 ? '$' + (value / 1e9).toFixed(1) + 'B' : value >= 1e6 ? '$' + (value / 1e6).toFixed(value >= 100e6 ? 0 : 1) + 'M' : value >= 1e3 ? '$' + (value / 1e3).toFixed(1) + 'K' : '$' + value.toLocaleString('en-CA');
  const exact = value => '$' + value.toLocaleString('en-CA') + ' CAD';
  const pct = value => value === 0 ? '0%' : value < .01 ? '<0.01%' : value.toFixed(value < 10 ? 2 : 1) + '%';
  const rankSort = (a,b) => b.value - a.value || a.code.localeCompare(b.code);
  let summary, dictionary, globalProducts, current, country = '', hs4 = '', hs6 = '';
  let productLimit = 25, childLimit = 25, requestId = 0, curveIndex = 0;
  let searchMatches = new Set(), productSearch, countrySearch;
  const cache = new Map();
  const marketDetails = new Map();
  const mixRows = new Map();
  let allDetailsTask = null, traceFailed = false;
  const countries = () => summary.countries;
  const market = () => countries().find(c => c.code === country);
  const scopeName = () => market()?.name || 'All destinations';
  const description = code => dictionary[code.length === 4 ? 'hs4' : 'hs6'][code] || 'HS' + code.length + ' ' + code;
  const fetchJSON = async file => {
    const response = await fetch(DATA + file);
    if (!response.ok) throw new Error(`Could not load ${file} (HTTP ${response.status})`);
    return response.json();
  };
  function aggregate(data) {
    const map = new Map();
    const six = data.products.map(([code,value]) => ({code,value})).filter(p => p.value > 0).sort(rankSort);
    for (const p of six) {
      const code = p.code.slice(0,4);
      if (!map.has(code)) map.set(code,{code,value:0,children:[]});
      map.get(code).value += p.value;
      map.get(code).children.push(p);
    }
    const four = [...map.values()].sort(rankSort);
    return {four,six,map,total:six.reduce((s,p) => s + p.value,0)};
  }
  async function loadMarketDetail(code) {
    if (!cache.has(code)) cache.set(code,fetchJSON('countries/'+code+'.json'));
    try {
      const data = await cache.get(code);
      if (!marketDetails.has(code)) {
        const detail = aggregate(data), c = countries().find(c => c.code === code);
        if (detail.total !== c.total) throw new Error('Country detail does not reconcile with summary');
        detail.ranks = new Map(detail.four.map((p,i) => [p.code,i+1]));
        marketDetails.set(code,detail);
      }
      return data;
    } catch (error) {cache.delete(code);throw error;}
  }
  function ensureTraceDetails() {
    if (!hs4 || marketDetails.size === countries().length || allDetailsTask || traceFailed) return;
    traceFailed = false;
    let index = 0;
    // Read the existing annual files once, with at most six requests in flight.
    allDetailsTask = Promise.all(Array.from({length:6},async () => {
      while (index < countries().length) {
        const c = countries()[index++];
        if (marketDetails.has(c.code)) continue;
        try {await loadMarketDetail(c.code);} catch (error) {traceFailed = true;}
      }
    })).finally(() => {allDetailsTask = null;if (hs4) {renderMixes();revealCountry();}});
  }
  function composition(c, selectedCode, detail) {
    const entries = c.top4.map(([code,value],i) => ({code,value,rank:i+1,selected:code === selectedCode}));
    let other = c.total-entries.reduce((s,p) => s+p.value,0);
    let count = c.hs4Count-entries.length;
    if (selectedCode && !entries.some(p => p.selected) && detail?.map.has(selectedCode)) {
      const value = detail.map.get(selectedCode).value;
      entries.push({code:selectedCode,value,rank:detail.ranks.get(selectedCode),selected:true,extracted:true});
      other -= value;count -= 1;
    }
    if (other < 0 || entries.reduce((s,p) => s+p.value,0)+other !== c.total) throw new Error('Composition does not reconcile');
    if (other > 0) entries.push({code:'other',value:other,count,selected:false});
    return entries;
  }
  function revealInside(container, row) {
    if (!row) return;
    const bounds = container.getBoundingClientRect(), rect = row.getBoundingClientRect();
    const delta = rect.top < bounds.top+6 ? rect.top-bounds.top-6 : rect.bottom > bounds.bottom-6 ? rect.bottom-bounds.bottom+6 : 0;
    if (delta) container.scrollTo({top:container.scrollTop+delta,behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
  }
  function revealCountry() {
    for (const id of ['destination-bars','market-mixes']) {
      const container = $(id);
      revealInside(container,container.querySelector(`[data-country="${country}"]`));
    }
  }
  function matchScore(query, name, code = '') {
    const q = normalized(query), n = normalized(name), c = normalized(code);
    if (!q) return Infinity;
    if (n === q || c === q) return 0;
    if (n.startsWith(q) || c.startsWith(q)) return 1;
    if (n.split(' ').some(w => w.startsWith(q))) return 2;
    if (n.includes(q)) return 3;
    const words = q.split(' ');
    if (words.every(w => n.includes(w))) return 4;
    return Infinity;
  }
  function searchCountries(query) {
    return countries().map(c => ({...c,score:matchScore(query,c.name,c.code)}))
      .filter(c => Number.isFinite(c.score)).sort((a,b) => a.score-b.score || a.rank-b.rank).slice(0,10);
  }
  function searchProducts(query, limit = 10) {
    return [...current.four,...current.six].map(p => ({...p,score:matchScore(query,description(p.code),p.code)}))
      .filter(p => Number.isFinite(p.score)).sort((a,b) => a.score-b.score || a.code.length-b.code.length || rankSort(a,b)).slice(0,limit);
  }
  function autocomplete(inputId, listId, search, label, choose, onQuery = () => {}) {
    const input = $(inputId), list = $(listId);
    let results = [], active = -1;
    function close() {list.hidden = true;input.setAttribute('aria-expanded','false');input.removeAttribute('aria-activedescendant');active = -1;}
    function activate(index) {
      active = index;
      [...list.children].forEach((li,i) => li.setAttribute('aria-selected',String(i === index)));
      if (index >= 0) {
        input.setAttribute('aria-activedescendant',`${listId}-${index}`);
        const li = list.children[index];
        if (li.offsetTop < list.scrollTop) list.scrollTop = li.offsetTop;
        else if (li.offsetTop + li.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = li.offsetTop + li.offsetHeight - list.clientHeight;
      }
    }
    function show() {
      onQuery(input.value);
      results = search(input.value); active = -1;
      input.removeAttribute('aria-activedescendant');
      list.innerHTML = results.map((r,i) => `<li role="option" id="${listId}-${i}" aria-selected="false" data-index="${i}">${label(r)}</li>`).join('');
      list.hidden = results.length === 0;
      input.setAttribute('aria-expanded',String(results.length > 0));
      if (input.value.trim()) $('live-status').textContent = results.length ? `${results.length} suggestions. Use arrow keys and Enter to select.` : 'No matching positive exports in this scope.';
    }
    input.addEventListener('input',show);
    input.addEventListener('focus',() => {if (input.value) show();});
    input.addEventListener('keydown',event => {
      if (event.key === 'Escape') {close();return;}
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();if (list.hidden) show();
        if (results.length) activate(active < 0 ? (event.key === 'ArrowDown' ? 0 : results.length-1) : (active + (event.key === 'ArrowDown' ? 1 : -1) + results.length) % results.length);
      } else if (event.key === 'Enter' && !list.hidden && results.length) {
        event.preventDefault();const result = results[active < 0 ? 0 : active];close();choose(result);
      }
    });
    list.addEventListener('pointerdown',event => {
      const li = event.target.closest('[data-index]');
      if (!li) return;
      event.preventDefault();const result = results[Number(li.dataset.index)];close();choose(result);
    });
    input.addEventListener('blur',close);
    return {close,clear() {input.value = '';close();onQuery('');}};
  }
  function showTip(html, target, event) {
    const tip = $('tooltip');tip.innerHTML = html;tip.hidden = false;
    const rect = target.getBoundingClientRect();
    let x = event && Number.isFinite(event.clientX) ? event.clientX + 14 : rect.left + 10;
    let y = event && Number.isFinite(event.clientY) ? event.clientY + 16 : rect.bottom + 8;
    x = Math.min(Math.max(12,x),window.innerWidth-tip.offsetWidth-12);
    if (y + tip.offsetHeight > window.innerHeight-12) y = Math.max(12,(event?.clientY ?? rect.top)-tip.offsetHeight-12);
    tip.style.left = x+'px';tip.style.top = y+'px';
  }
  const hideTip = () => {$('tooltip').hidden = true;};
  const tipLabel = content => content.replace(/<[^>]*>/g,' ').replace(/&amp;/g,'&');
  function attachTip(element, content) {
    const text = () => typeof content === 'function' ? content() : content;
    element.setAttribute('aria-label',tipLabel(text()));
    element.addEventListener('pointerenter',e => showTip(text(),element,e));
    element.addEventListener('pointermove',e => showTip(text(),element,e));
    element.addEventListener('pointerleave',hideTip);
    element.addEventListener('focus',() => {element.setAttribute('aria-describedby','tooltip');showTip(text(),element);});
    element.addEventListener('blur',() => {element.removeAttribute('aria-describedby');hideTip();});
    element.addEventListener('keydown',e => {if (e.key === 'Escape') hideTip();});
  }
  const countryTip = c => `<strong>${escape(c.name)} · Rank ${c.rank}</strong>${exact(c.total)}<br>${pct(c.share)} of Alberta exports<br>${c.hs4Count} active HS4 · ${c.hs6Count} active HS6`;
  const productTip = p => `<strong>HS4 ${p.code}</strong>${escape(description(p.code))}<br>${exact(p.value)}<br>${pct(p.value/current.total*100)} of ${country ? escape(scopeName())+' exports' : 'Alberta exports'}${country ? '<br>'+pct(p.value/summary.overall.total*100)+' of all Alberta exports' : ''}<small>${p.children.length} active HS6 children</small>`;
  const childTip = (p,parent) => `<strong>HS6 ${p.code}</strong>${escape(description(p.code))}<br>${exact(p.value)}<br>${pct(p.value/parent.value*100)} of HS4 ${parent.code}<br>${pct(p.value/current.total*100)} of ${country ? escape(scopeName())+' exports' : 'Alberta exports'}`;
  function renderOverview() {
    const s = summary.overall, largest = countries()[0];
    const cards = [['2025 domestic exports',money(s.total),'Current-dollar CAD'],['Destination markets',countries().length,'Countries & territories'],['Active HS4 headings',s.hs4Count.toLocaleString(),'Positive annual exports'],['Active HS6 subheadings',s.hs6Count.toLocaleString(),'Positive annual exports'],['Largest destination',pct(largest.share),largest.name]];
    $('overview').innerHTML = cards.map(([title,value,note]) => `<div><dt>${title}</dt><dd>${value}</dd><small>${escape(note)}</small></div>`).join('');
  }
  function renderDestinations() {
    const limit = $('destination-limit').value;
    const rows = limit === 'all' ? [...countries()] : countries().slice(0,Number(limit));
    if (country && !rows.some(c => c.code === country)) rows.push(market());
    const log = document.querySelector('input[name="scale"]:checked').value === 'log';
    const maximum = countries()[0].total;
    $('scale-note').textContent = log ? `LOG SCALE · Bar length uses log(1 + CAD value), from $0 to ${money(maximum)}. Labels show actual values.` : `LINEAR SCALE · Bar length is proportional to value, from $0 to ${money(maximum)}. Select a market.`;
    const container = $('destination-bars');
    const previous = new Map([...container.children].map(n => [n.dataset.country,n]));
    for (const c of rows) {
      let button = previous.get(c.code);
      if (!button) {
        button = document.createElement('button');button.type = 'button';button.dataset.country = c.code;button.className = 'destination-row';
        button.innerHTML = `<span class="rank">${c.rank}</span><span class="destination-name"><span>${escape(c.name)}</span><span class="selection-marker" hidden>Selected</span><span class="inline-track"><i></i></span></span><span class="amount">${money(c.total)}</span><span class="share">${pct(c.share)}</span>`;
        button.title = c.name;
        button.addEventListener('click',() => selectCountry(c.code));attachTip(button,countryTip(c));container.append(button);
      }
      button.classList.toggle('selected',c.code === country);button.setAttribute('aria-pressed',String(c.code === country));
      button.querySelector('.selection-marker').hidden = c.code !== country;
      button.setAttribute('aria-label',tipLabel(countryTip(c))+(c.code === country ? ' Selected country.' : ''));
      const length = (log ? Math.log1p(c.total)/Math.log1p(maximum) : c.total/maximum)*100;
      button.querySelector('i').style.width = length+'%';
    }
    const visible = new Set(rows.map(c => c.code));
    [...container.children].forEach(n => {if (!visible.has(n.dataset.country)) n.remove();});
    // Keep value ranking when a formerly out-of-range selection is now in view.
    rows.forEach((c,i) => {const row = container.querySelector(`[data-country="${c.code}"]`);if (container.children[i] !== row) container.insertBefore(row,container.children[i] || null);});
  }
  function curveCoordinates(index) {
    const cumulative = countries().slice(0,index+1).reduce((s,c) => s+c.total,0)/summary.overall.total*100;
    return {x:48+(index+1)/countries().length*454,y:240-cumulative*2.1,cumulative};
  }
  function inspectCurve(index,event) {
    curveIndex = Math.max(0,Math.min(countries().length-1,index));
    const c = countries()[curveIndex], p = curveCoordinates(curveIndex), svg = $('concentration-curve');
    const dot = $('curve-focus');dot.setAttribute('cx',p.x);dot.setAttribute('cy',p.y);dot.setAttribute('r','5');
    $('curve-desc').textContent = `Rank ${c.rank}, ${c.name}, ${exact(c.total)}, ${pct(c.share)} individual share, ${pct(p.cumulative)} cumulative share. Arrow keys move by rank; Enter selects.`;
    showTip(countryTip(c)+`<br>${pct(p.cumulative)} cumulative share`,svg,event);
  }
  function renderCurve() {
    const svg = $('concentration-curve');
    const points = countries().map((_,i) => curveCoordinates(i));
    const path = 'M48,240 '+points.map(p => `L${p.x},${p.y}`).join(' ');
    let html = '<title id="curve-title">Cumulative share of Alberta exports by destination rank</title><desc id="curve-desc">All '+countries().length+' active markets ranked largest to smallest. Arrow keys inspect; Enter selects.</desc>';
    for (const percent of [0,25,50,75,100]) html += `<line class="curve-grid" x1="48" x2="502" y1="${240-percent*2.1}" y2="${240-percent*2.1}"/><text x="40" y="${244-percent*2.1}" text-anchor="end">${percent}%</text>`;
    html += `<path class="curve-area" d="${path} L502,240 Z"/><path class="curve-line" d="${path}"/>`;
    const markers = [...new Set([1,3,10,...Object.values(summary.thresholds)])];
    for (const rank of markers) {const p = points[rank-1];html += `<line x1="${p.x}" x2="${p.x}" y1="${p.y}" y2="240" stroke="#b1c5ad" stroke-dasharray="3 4"/><circle class="curve-marker" cx="${p.x}" cy="${p.y}" r="3.5"/>`;}
    for (const rank of [1,50,100,150,countries().length]) html += `<text class="curve-rank-tick" x="${48+rank/countries().length*454}" y="261" text-anchor="middle">${rank}</text>`;
    html += '<text class="curve-rank-axis" x="275" y="285" text-anchor="middle">Destination rank · largest to smallest</text><g id="curve-selected" hidden><line id="curve-selected-line" y2="240" stroke="#624817" stroke-width="3" stroke-dasharray="6 4"/><circle id="curve-selected-halo" r="15" fill="#fff2dc" opacity=".65"/><path id="curve-selected-mark" fill="#fff8e9" stroke="#624817" stroke-width="3"/><g id="curve-rank-tag" aria-hidden="true"><rect width="80" height="24" rx="5" fill="#fff8e9" stroke="#ccb383"/><text id="curve-selected-label" class="curve-rank-label" x="8" y="17"></text></g></g><circle id="curve-focus" cx="48" cy="240" r="0" fill="#9b6925" stroke="#fff" stroke-width="2"/>';
    svg.innerHTML = html;
    $('thresholds').innerHTML = [1,3,10].map(r => `<span>Top ${r}<strong>${pct(points[r-1].cumulative)}</strong></span>`).join('') + [90,95,99].map(p => `<span>${p}% of exports<strong>${summary.thresholds[p]} markets</strong></span>`).join('');
    $('curve-summary').textContent = `The top ten markets receive ${money(summary.top10Total)}, or ${pct(summary.top10Share)}, of Alberta exports. All ${countries().length} destinations are represented in the curve.`;
    svg.addEventListener('pointermove',e => {const rect = svg.getBoundingClientRect();const x = (e.clientX-rect.left)/rect.width*520;inspectCurve(Math.round((x-48)/454*countries().length)-1,e);});
    svg.addEventListener('pointerleave',hideTip);
    svg.addEventListener('focus',() => inspectCurve(country ? market().rank-1 : curveIndex));
    svg.addEventListener('blur',hideTip);
    svg.addEventListener('click',e => {
      const rect = svg.getBoundingClientRect();
      if (e.detail) inspectCurve(Math.round(((e.clientX-rect.left)/rect.width*520-48)/454*countries().length)-1,e);
      selectCountry(countries()[curveIndex].code);
    });
    svg.addEventListener('keydown',e => {
      if (['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) {
        e.preventDefault();inspectCurve(e.key === 'Home' ? 0 : e.key === 'End' ? countries().length-1 : curveIndex+(e.key === 'ArrowRight' ? 1 : -1));
      } else if (e.key === 'Enter' || e.key === ' ') {e.preventDefault();selectCountry(countries()[curveIndex].code);}
      else if (e.key === 'Escape') hideTip();
    });
  }
  function renderMarket() {
    const c = market();$('clear-market').disabled = !c;$('clear-market').hidden = !c;
    $('country-search').value = c?.name || '';
    $('product-scope').textContent = 'Alberta → '+scopeName();
    $('product-search-label').textContent = c ? "Search this market's HS4 or HS6 products" : "Search Alberta's HS4 or HS6 products";
    if (!c) {$('market-summary').innerHTML = '<h3>Alberta → All destinations</h3><p class="small">The Product Atlas currently combines every destination. Select a market to see its size, product breadth and concentration.</p>';return;}
    const cards = [['Annual exports',money(c.total)],['Share of Alberta',pct(c.share)],['Destination rank','#'+c.rank],['Active HS4',c.hs4Count],['Active HS6',c.hs6Count],['Top-four HS4',pct(c.top4Share)]];
    $('market-summary').innerHTML = `<h3>Alberta → ${escape(c.name)}</h3><dl class="market-stats">${cards.map(([label,value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join('')}</dl><p class="market-largest">Largest heading: <strong>HS4 ${c.top4[0][0]}</strong> · ${escape(description(c.top4[0][0]))} · ${money(c.top4[0][1])}. Top-four concentration measures the product mix within this market.</p>`;
  }
  function renderCountryMark() {
    $('curve-selected').setAttribute('visibility',country ? 'visible' : 'hidden');
    $('curve-selected').removeAttribute('hidden');
    $('curve-selection').hidden = !country;$('curve-state').hidden = !country;
    if (!country) return;
    const c = market(), p = curveCoordinates(c.rank-1);
    $('curve-selected-line').setAttribute('x1',p.x);$('curve-selected-line').setAttribute('x2',p.x);$('curve-selected-line').setAttribute('y1',p.y);
    $('curve-selected-halo').setAttribute('cx',p.x);$('curve-selected-halo').setAttribute('cy',p.y);
    $('curve-selected-mark').setAttribute('d',`M${p.x},${p.y-10}l10,10l-10,10l-10,-10Z`);
    $('curve-rank-tag').setAttribute('transform',`translate(${p.x > 390 ? p.x-95 : p.x+16},${Math.max(5,p.y-32)})`);
    $('curve-selected-label').textContent = 'Rank '+c.rank;
    $('curve-selection').innerHTML = `<span class="sr-only">Selected destination: </span><span>${escape(c.name)} ·</span> <span>Rank <strong>${c.rank} of ${countries().length}</strong></span>`;
  }
  function renderSelectedProduct() {
    const code = hs6 || hs4, box = $('selected-product-summary');
    box.hidden = !code;
    if (!code) {$('focus-content').replaceChildren();return;}
    const products = hs6 ? current.six : current.four;
    const product = products.find(p => p.code === code);
    const rank = products.findIndex(p => p.code === code)+1;
    const heading = `<div class="focus-header"><div class="focus-identity"><p class="focus-code">HS${code.length} ${code}</p><p class="focus-description">${escape(description(code))}</p></div><div class="focus-scope"><span class="small">Scope</span><span class="scope-pill geography-scope">Alberta → ${escape(scopeName())}</span></div></div>`;
    if (!product) {
      $('focus-content').innerHTML = heading+'<p class="focus-empty">No positive exports in this scope.</p>';
      return;
    }
    const metrics = [['Annual exports',money(product.value)],['Share of current scope',pct(product.value/current.total*100)],
                     ['Rank within current scope',`#${rank} of ${products.length}`],
                     [hs6 ? 'HS4 heading' : 'Active HS6 children',hs6 ? hs4 : product.children.length]];
    $('focus-content').innerHTML = heading+`<dl class="focus-metrics">${metrics.map(([label,value],i) => `<div><dt>${label}</dt><dd>${escape(value)}</dd>${i === 0 ? '<small>'+exact(product.value)+'</small>' : ''}</div>`).join('')}</dl>`;
  }
  /* Squarified treemap: no value floor, no discarded tail, no area distortion.
   * Zero padding preserves exact area; hairline boundaries are visual only. */
  function treemapLayout(items,width,height) {
    const total = items.reduce((s,p) => s+p.value,0);
    const pending = items.map(p => ({...p,area:p.value/total*width*height}));
    const result = [];
    let x = 0,y = 0,w = width,h = height,row = [];
    const worst = (r,side) => {
      if (!r.length || side <= 0) return Infinity;
      const sum = r.reduce((s,p) => s+p.area,0), min = Math.min(...r.map(p => p.area)), max = Math.max(...r.map(p => p.area));
      return Math.max(side*side*max/(sum*sum),sum*sum/(side*side*min));
    };
    function commit() {
      const area = row.reduce((s,p) => s+p.area,0);
      if (w >= h) {
        const strip = area/h;let pos = y;
        for (const p of row) {const length = p.area/strip;result.push({...p,x,y:pos,w:strip,h:length});pos += length;}
        x += strip;w = Math.max(0,w-strip);
      } else {
        const strip = area/w;let pos = x;
        for (const p of row) {const length = p.area/strip;result.push({...p,x:pos,y,w:length,h:strip});pos += length;}
        y += strip;h = Math.max(0,h-strip);
      }
      row = [];
    }
    for (const p of pending) {
      if (!row.length || worst([...row,p],Math.min(w,h)) <= worst(row,Math.min(w,h))) row.push(p);
      else {commit();row.push(p);}
    }
    if (row.length) commit();
    return result;
  }
  function tileColor(code) {return colors[Number(code.slice(0,2)) % colors.length];}
  function renderTreemap() {
    if (!current || !$('treemap').clientWidth) return;
    const container = $('treemap'), width = container.clientWidth, height = container.clientHeight;
    const previous = new Map([...container.children].map(n => [n.dataset.hs4,n]));
    for (const p of treemapLayout(current.four,width,height)) {
      let button = previous.get(p.code);
      if (!button) {button = document.createElement('button');button.type = 'button';button.className = 'tile';button.dataset.hs4 = p.code;button.addEventListener('click',() => selectProduct(p.code));}
      if (!button.dataset.tipReady) {
        attachTip(button,() => productTip(current.map.get(p.code)));button.dataset.tipReady = '1';
        button.addEventListener('keydown',e => {
          if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(e.key)) return;
          e.preventDefault();const i = current.four.findIndex(n => n.code === p.code);
          const next = e.key === 'Home' ? 0 : e.key === 'End' ? current.four.length-1 : (i+(['ArrowRight','ArrowDown'].includes(e.key) ? 1 : -1)+current.four.length)%current.four.length;
          container.querySelector(`[data-hs4="${current.four[next].code}"]`).focus({preventScroll:true});
        });
      }
      if (!button.parentNode) container.append(button);
      button.setAttribute('aria-label',tipLabel(productTip(p)));
      Object.assign(button.style,{left:p.x/width*100+'%',top:p.y/height*100+'%',width:p.w/width*100+'%',height:p.h/height*100+'%',background:tileColor(p.code)});
      button.tabIndex = p.code === (hs4 || current.four[0]?.code) ? 0 : -1;
      button.setAttribute('aria-pressed',String(p.code === hs4));
      button.classList.toggle('selected',p.code === hs4);button.classList.toggle('match',searchMatches.has(p.code));
      button.innerHTML = p.w > 80 && p.h > 47 ? `<span class="tile-label"><span class="tile-code">${p.code}</span>${p.h > 125 && p.w > 110 ? `<span class="tile-desc">${escape(description(p.code))}</span>` : ''}${p.h > 85 ? `<span class="tile-value">${money(p.value)} · ${pct(p.value/current.total*100)}</span>` : ''}</span>` : '';
    }
    [...container.children].forEach(n => {if (!current.map.has(n.dataset.hs4)) n.remove();});
    $('atlas-summary').textContent = `${current.four.length} headings · ${current.six.length} subheadings · ${money(current.total)} in this scope.${hs4 ? ' Selected: HS4 '+hs4+' · '+(current.map.has(hs4) ? money(current.map.get(hs4).value) : 'no positive exports in this scope')+'.' : ''}${searchMatches.size ? ' Search highlights '+searchMatches.size+' matching headings.' : ''}`;
  }
  function productRow(p,index,parent) {
    const button = document.createElement('button');button.type = 'button';button.className = 'product-row';button.dataset.code = p.code;
    const selected = p.code.length === 4 ? p.code === hs4 : p.code === hs6;
    button.classList.toggle('selected',selected);button.setAttribute('aria-pressed',String(selected));
    button.classList.toggle('match',p.code.length === 4 && searchMatches.has(p.code));
    const denominator = parent?.value || current.total;
    button.innerHTML = `<span class="row-top"><span class="small">${index+1}</span><strong>${p.code}</strong>${selected ? '<span class="selection-marker">Selected</span>' : ''}<span class="amount">${money(p.value)}</span></span><span class="row-description">${escape(description(p.code))}</span><span class="row-bottom"><span class="inline-track"><i style="width:${p.value/denominator*100}%"></i></span><span>${pct(p.value/denominator*100)}${parent ? ' of heading' : ''}</span></span>`;
    button.addEventListener('click',() => selectProduct(p.code));attachTip(button,parent ? childTip(p,parent) : productTip(p));return button;
  }
  function renderProductList() {
    // Selected/search results stay visible even if their overall rank is deep in the tail.
    const matches = searchMatches.size ? current.four.filter(p => searchMatches.has(p.code)) : current.four;
    let rows = matches.slice(0,productLimit);
    const selected = current.map.get(hs4);
    if (selected && !rows.includes(selected)) rows = [selected,...rows];
    $('product-list').replaceChildren(...rows.map(p => productRow(p,current.four.indexOf(p))));
    $('product-count').textContent = searchMatches.size ? `${matches.length} matching HS4 headings · ${rows.length} shown` : `${current.four.length} headings · ${rows.length} shown`;
    $('more-products').hidden = productLimit >= matches.length;
  }
  function renderDrill() {
    const parent = current.map.get(hs4);$('clear-product').disabled = !hs4 && !$('product-search').value;$('clear-product').hidden = !hs4 && !hs6;
    document.querySelector('.drill-card').classList.toggle('tracing-heading',!!hs4);
    $('drill-title').textContent = hs4 ? 'Inside HS4 '+hs4 : 'Inside an HS4 heading';
    $('drill-scope').textContent = 'Alberta → '+scopeName();
    $('treemap-instruction-copy').textContent = parent
      ? `HS4 ${hs4} is selected. Its HS6 subheadings are shown in the “Inside HS4” section below.`
      : 'Select an HS4 category to reveal its HS6 subheadings in the “Inside HS4” section below.';
    $('drill-body').hidden = !parent;
    if (!parent) {$('drill-description').textContent = hs4 ? `${description(hs4)} · No positive exports in this geographic scope. This HS4 is still traced across other markets below.` : 'Select an HS4 tile, ranked row or HS6 search result to reveal its composition.';return;}
    $('drill-description').textContent = `${description(hs4)} · ${money(parent.value)} · ${pct(parent.value/current.total*100)} of this scope · ${parent.children.length} HS6 children.`;
    const previous = new Map([...$('hs6-ribbon').children].map(n => [n.dataset.code,n]));
    const fragment = document.createDocumentFragment();
    parent.children.forEach((p,i) => {
      let button = previous.get(p.code);
      if (!button || button.dataset.scope !== country) {
        button = document.createElement('button');button.type = 'button';button.dataset.code = p.code;button.dataset.scope = country;
        button.addEventListener('click',() => selectProduct(p.code));attachTip(button,childTip(p,parent));
      }
      button.className = 'rank-'+i%5+(p.code === hs6 ? ' selected' : '');button.setAttribute('aria-pressed',String(p.code === hs6));
      button.style.width = p.value/parent.value*100+'%';button.textContent = p.value/parent.value > .14 ? p.code : '';
      fragment.append(button);
    });
    $('hs6-ribbon').replaceChildren(fragment);
    let rows = parent.children.slice(0,childLimit);
    const selected = parent.children.find(p => p.code === hs6);
    $('selected-hs6').hidden = !selected;
    $('selected-hs6').textContent = selected ? `Selected HS6 ${selected.code} · ${description(selected.code)} · ${exact(selected.value)} · ${pct(selected.value/parent.value*100)} of this heading · ${pct(selected.value/current.total*100)} of ${country ? scopeName()+' exports' : 'Alberta exports'}.` : '';
    if (selected && !rows.includes(selected)) rows = [selected,...rows];
    $('hs6-list').replaceChildren(...rows.map(p => productRow(p,parent.children.indexOf(p),parent)));
    $('more-hs6').hidden = childLimit >= parent.children.length;
  }
  function renderMixes() {
    const limit = $('mix-limit').value;
    let rows = countries().filter(c => !$('exclude-us').checked || c.code !== 'US');
    if (limit !== 'all') rows = rows.slice(0,Number(limit));
    if (country && !rows.some(c => c.code === country) && !(country === 'US' && $('exclude-us').checked)) rows.push(market());
    const container = $('market-mixes'), scrollTop = container.scrollTop;
    $('trace-controls').hidden = !hs4;$('retry-trace').hidden = !hs4 || !traceFailed;
    const pending = hs4 && marketDetails.size !== countries().length;
    $('trace-status').textContent = hs4 ? `Tracing HS4 ${hs4} · ${description(hs4)} across Alberta's export markets.${pending ? (traceFailed ? ' Some country details failed to load; retry to complete the trace.' : ' Loading country product detail…') : ''}` : '';
    container.setAttribute('aria-busy',String(!!pending && !traceFailed));
    const rendered = rows.map(c => {
      const showTrace = !country || c.code === country;
      const key = hs4+(hs4 ? '|'+marketDetails.has(c.code)+'|'+showTrace : '');
      let row = mixRows.get(c.code);
      if (row?.dataset.composition === key) {
        row.classList.toggle('selected',c.code === country);
        const button = row.querySelector('.mix-country');
        button.setAttribute('aria-pressed',String(c.code === country));button.querySelector('.selection-marker').hidden = c.code !== country;
        button.setAttribute('aria-label',tipLabel(countryTip(c))+(c.code === country ? ' Selected country.' : ''));
        return row;
      }
      if (!row) {row = document.createElement('div');mixRows.set(c.code,row);}
      row.replaceChildren();row.dataset.composition = key;
      row.className = 'mix-row'+(c.code === country ? ' selected' : '');row.dataset.country = c.code;
      const button = document.createElement('button');button.type = 'button';button.className = 'mix-country';button.title = c.name;
      button.innerHTML = `${escape(c.name)}<span class="selection-marker"${c.code === country ? '' : ' hidden'}>Selected</span>`;
      button.addEventListener('click',() => selectCountry(c.code));button.setAttribute('aria-pressed',String(c.code === country));attachTip(button,() => countryTip(c)+(c.code === country ? '<small>Selected country</small>' : ''));
      const bar = document.createElement('div');bar.className = 'mix-bar';
      const entries = composition(c,hs4,marketDetails.get(c.code));
      entries.forEach(p => {
        const segment = document.createElement('button');segment.type = 'button';segment.dataset.hs4 = p.code;segment.dataset.value = p.value;segment.dataset.rank = p.rank || '';
        segment.className = 'mix-segment '+(p.code === 'other' ? 'rank-4' : p.extracted ? 'trace-color' : 'rank-'+(p.rank-1))+(p.selected ? ' selected-product' : '');segment.style.width = p.value/c.total*100+'%';
        segment.textContent = p.value/c.total > .13 && p.code !== 'other' ? (p.selected ? '✓ ' : '')+p.code : '';
        segment.setAttribute('aria-pressed',String(p.selected));
        const label = p.code === 'other' ? 'Other HS4 headings' : `HS4 ${p.code} · Rank ${p.rank} within this destination`;
        const tooltip = `<strong>${escape(c.name)} · ${label}</strong>${p.code === 'other' ? `${p.count} HS4 headings represented<br>` : escape(description(p.code))+'<br>'}${exact(p.value)} · ${pct(p.value/c.total*100)} of Alberta exports to this destination${p.code === 'other' ? (hs4 && marketDetails.has(c.code) ? '<small>Excludes the selected HS4 wherever present.</small>' : '') : '<small>'+(p.selected ? 'Currently selected HS4'+(p.extracted ? ' · extracted from Other' : '') : 'Not the selected HS4')+'</small>'}`;
        attachTip(segment,tooltip);
        segment.addEventListener('click',() => p.code === 'other' ? selectCountry(c.code) : setState(c.code,p.code,''));bar.append(segment);
      });
      const share = document.createElement('span');share.className = 'mix-percent';share.textContent = pct(c.top4Share);share.title = 'Top-four HS4 concentration';
      row.append(button,bar,share);
      const traced = entries.find(p => p.selected);
      if (traced && showTrace) {
        // A text marker keeps even subpixel flows visible without enlarging their area.
        const badge = document.createElement('button');badge.type = 'button';badge.className = 'trace-readout';
        badge.innerHTML = `<strong>✓ HS4 ${traced.code}</strong> · Rank #${traced.rank} · ${money(traced.value)} · ${pct(traced.value/c.total*100)}`;
        attachTip(badge,`<strong>${escape(c.name)} · Selected HS4 ${traced.code}</strong>${escape(description(traced.code))}<br>${exact(traced.value)} · ${pct(traced.value/c.total*100)} of Alberta exports to this destination · Rank ${traced.rank}`);
        badge.addEventListener('click',() => setState(c.code,traced.code,''));row.append(badge);
      }
      return row;
    });
    const visible = new Set(rows.map(c => c.code));
    [...container.children].forEach(row => {if (!visible.has(row.dataset.country)) row.remove();});
    rendered.forEach((row,i) => {if (container.children[i] !== row) container.insertBefore(row,container.children[i] || null);});
    // Assigning even the same scrollTop cancels an in-flight native smooth reveal.
    if (container.scrollTop !== scrollTop) container.scrollTop = scrollTop;
    $('mix-summary').textContent = `Right-hand percentages show top-four HS4 concentration. ${rows[0].name}: ${pct(rows[0].top4Share)}. Select a market name or segment to connect it to the atlas.`;
    ensureTraceDetails();
  }
  function renderProducts() {renderSelectedProduct();renderTreemap();renderProductList();renderDrill();}
  function writeHash() {
    const params = new URLSearchParams();if (country) params.set('country',country);if (hs4) params.set('hs4',hs4);if (hs6) params.set('hs6',hs6);
    // Default display modes stay implicit; only intentional nondefault views are shared.
    if ($('destination-limit').value !== 'all') params.set('destinations',$('destination-limit').value);
    if ($('mix-limit').value !== 'all') params.set('mix',$('mix-limit').value);
    if (document.querySelector('input[name="scale"]:checked').value !== 'linear') params.set('scale','log');
    if ($('exclude-us').checked) params.set('excludeUS','1');
    const hash = params.toString();
    if (location.hash.slice(1) !== hash) history.pushState(null,'',location.pathname+location.search+(hash ? '#'+hash : ''));
  }
  async function setState(nextCountry,next4 = hs4,next6 = '',push = true) {
    const token = ++requestId;
    const requested = countries().some(c => c.code === nextCountry) ? nextCountry : '';
    hideTip();countrySearch?.close();productSearch?.close();
    $('application').setAttribute('aria-busy','true');
    $('live-status').textContent = requested ? 'Loading Alberta → '+countries().find(c => c.code === requested).name+'…' : 'Loading all destinations…';
    try {
      let data;
      if (!requested) data = globalProducts;
      else {
        data = await loadMarketDetail(requested);
      }
      if (token !== requestId) return;
      const next = aggregate(data);
      const expected = requested ? countries().find(c => c.code === requested).total : summary.overall.total;
      if (next.total !== expected) throw new Error('Country detail does not reconcile with summary');
      country = requested;current = next;
      hs4 = aggregate(globalProducts).map.has(next4) ? next4 : '';hs6 = next.map.has(hs4) && next.map.get(hs4).children.some(p => p.code === next6) ? next6 : '';
      productLimit = childLimit = 25;searchMatches = new Set();
      $('product-search').value = hs6 || hs4;renderMarket();renderDestinations();renderProducts();renderMixes();
      renderCountryMark();revealCountry();
      if (push) writeHash();
      $('live-status').textContent = `Alberta → ${scopeName()}: ${money(current.total)}, ${current.four.length} HS4 headings.${hs4 ? ' Selected HS4 '+hs4+'.' : ''}`;
    } catch (error) {
      if (token !== requestId) return;
      cache.delete(requested);$('live-status').textContent = 'Unable to load this selection. '+error.message+'. The previous view is retained; select a market to retry.';
      if (!push) writeHash();
    } finally {if (token === requestId) $('application').setAttribute('aria-busy','false');}
  }
  function selectCountry(code) {return setState(code,hs4,'');}
  function selectProduct(code) {
    const parent = code.slice(0,4);if (!current.map.has(parent)) return;
    // Product selections supersede any older pending geographic request.
    ++requestId;$('application').setAttribute('aria-busy','false');hideTip();productSearch.close();
    hs4 = parent;hs6 = code.length === 6 ? code : '';childLimit = 25;searchMatches = new Set();
    $('product-search').value = code;renderProducts();renderMixes();revealInside($('product-list'),$('product-list').querySelector(`[data-code="${hs4}"]`));writeHash();
    $('live-status').textContent = `Selected HS${code.length} ${code}: ${description(code)}. Alberta → ${scopeName()}.`;
  }
  function restoreHash() {
    const params = new URLSearchParams(location.search);
    new URLSearchParams(location.hash.slice(1)).forEach((value,key) => params.set(key,value));
    const modes = ['10','25','all'];
    $('destination-limit').value = modes.includes(params.get('destinations')) ? params.get('destinations') : 'all';
    $('mix-limit').value = modes.includes(params.get('mix')) ? params.get('mix') : 'all';
    document.querySelector(`input[name="scale"][value="${params.get('scale') === 'log' ? 'log' : 'linear'}"]`).checked = true;
    $('exclude-us').checked = params.get('excludeUS') === '1';
    return setState(params.get('country') || '',params.get('hs4') || params.get('hs6')?.slice(0,4) || '',params.get('hs6') || '',false);
  }
  function clearProduct() {
    ++requestId;$('application').setAttribute('aria-busy','false');hs4 = hs6 = '';productSearch.clear();hideTip();renderProducts();renderMixes();writeHash();
    $('live-status').textContent = 'Product selection cleared. Alberta → '+scopeName()+'.';
  }
  async function init() {
    try {
      [summary,dictionary,globalProducts] = await Promise.all(['summary.json','product-index.json','alberta-products.json'].map(fetchJSON));
      current = aggregate(globalProducts);renderOverview();$('application').hidden = false;
      countrySearch = autocomplete('country-search','country-suggestions',searchCountries,c => `<strong>${escape(c.name)} <small>${c.code}</small></strong><small>Rank ${c.rank} · ${money(c.total)} · ${pct(c.share)} of Alberta exports</small>`,c => selectCountry(c.code));
      productSearch = autocomplete('product-search','product-suggestions',searchProducts,p => `<strong>HS${p.code.length} ${p.code} · ${escape(description(p.code))}</strong><small>${money(p.value)} · ${pct(p.value/current.total*100)} of ${country ? escape(scopeName())+' exports' : 'Alberta exports'}</small>`,p => selectProduct(p.code),query => {
        searchMatches = new Set(searchProducts(query,Infinity).map(p => p.code.slice(0,4)));
        productLimit = 25;renderTreemap();renderProductList();$('clear-product').disabled = !hs4 && !query;
      });
      $('destination-limit').addEventListener('change',() => {renderDestinations();revealCountry();writeHash();});
      document.querySelectorAll('input[name="scale"]').forEach(n => n.addEventListener('change',() => {renderDestinations();writeHash();}));
      document.querySelectorAll('[data-country]').forEach(n => n.addEventListener('click',() => selectCountry(n.dataset.country)));
      $('clear-market').addEventListener('click',() => setState('',hs4,''));
      $('clear-product').addEventListener('click',clearProduct);
      $('retry-trace').addEventListener('click',() => {traceFailed = false;renderMixes();});
      $('more-products').addEventListener('click',() => {productLimit += 25;renderProductList();});
      $('more-hs6').addEventListener('click',() => {childLimit += 25;renderDrill();});
      $('mix-limit').addEventListener('change',() => {renderMixes();revealCountry();writeHash();});$('exclude-us').addEventListener('change',() => {renderMixes();revealCountry();writeHash();});
      window.addEventListener('popstate',restoreHash);window.addEventListener('hashchange',restoreHash);
      window.addEventListener('scroll',hideTip,{passive:true});
      new ResizeObserver(renderTreemap).observe($('treemap'));
      renderCurve();await restoreHash();
    } catch (error) {$('application').hidden = true;$('live-status').textContent = 'Atlas could not load: '+error.message+'. Serve the repository through HTTP; see methodology below.';}
  }
  init();
})();
