/* Uses the Product Explorer's loaded annual arrays; no second fetch or source. */
(() => {
  'use strict';
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const sum = values => values.reduce((a,b) => a + b,0);
  const escape = text => String(text).replace(/[&<>"']/g,c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const money = value => {
    if (!Number.isFinite(value)) return 'Data unavailable';
    for (const [scale,suffix] of [[1e9,'B'],[1e6,'M'],[1e3,'K']]) if (value >= scale) return 'CAD ' + (value / scale).toFixed(1) + suffix;
    return 'CAD ' + Math.round(value).toLocaleString('en-CA');
  };
  const fullMoney = value => Number.isFinite(value) ? 'CAD ' + value.toLocaleString('en-CA',{maximumFractionDigits:0}) : 'Data unavailable';
  const percent = value => Number.isFinite(value) ? value.toFixed(1) + '%' : 'Data unavailable';
  const intensityColor = intensity => {
    check(Number.isFinite(intensity) && intensity >= 0 && intensity <= 100,'Invalid colour intensity');
    const light = [161,199,229], dark = [14,65,126];
    return `rgb(${light.map((v,i) => Math.round(v + (dark[i] - v) * intensity / 100)).join(',')})`;
  };

  function createSummaries({hs4,hs6,products,origins,scope,meta}) {
    check(meta.year === '2025' && meta.destination === 'US' && meta.currency === 'CAD' && meta.measure === 'Domestic exports','Unexpected trade basis');
    check(scope && Array.isArray(scope.hs6),'Section 338 scope unavailable');
    const expected = ['AB','BC','MB','NB','NL','NS','ON','PE','QC','SK','NT','NU','YT'];
    check(origins.length === 13 && origins.every(([code],i) => code === expected[i]),'Unexpected origin order');
    const descriptions = new Map(products.map(p => [p.code,p.description]));
    const policy = new Set(scope.hs6);
    const groups = new Map(Object.keys(hs4).map(code => [code,{total:Array(13).fill(0),exposed:Array(13).fill(0)}]));
    // Aggregate only HS6 observations. Membership is boolean, so duplicate
    // detailed policy lines cannot multiply a Canadian observation.
    for (const [code,values] of Object.entries(hs6)) {
      const group = groups.get(code.slice(0,4));
      check(/^\d{6}$/.test(code) && group,'Invalid HS6 parent');
      check(values.length === 13 && values.every(v => Number.isSafeInteger(v) && v >= 0),'Unavailable or invalid HS6 values: ' + code);
      values.forEach((value,i) => { group.total[i] += value; if (policy.has(code)) group.exposed[i] += value; });
    }
    for (const [code,group] of groups) {
      check(hs4[code].length === 13 && group.total.every((v,i) => Number.isSafeInteger(v) && v === hs4[code][i]),'HS4 child reconciliation failed: ' + code);
    }
    const cache = new Map();
    for (const [geography,name] of [['CANADA','Canada'],...origins]) {
      const index = origins.findIndex(([code]) => code === geography);
      const valueFor = values => geography === 'CANADA' ? sum(values) : values[index];
      const sectors = [...groups].map(([code,group]) => {
        const total = valueFor(group.total), exposed = valueFor(group.exposed);
        check(Number.isSafeInteger(total) && Number.isSafeInteger(exposed) && exposed <= total,'Invalid exposure totals');
        return {code,description:descriptions.get(code) || 'Heading description unavailable in compatible local source',total,exposed,intensity:total > 0 ? exposed / total * 100 : null};
      });
      const total = sum(sectors.map(row => row.total)), exposed = sum(sectors.map(row => row.exposed));
      check(total === (geography === 'CANADA' ? meta.validation.annual_value_total : meta.validation.province_totals[geography]),'Geographic audit mismatch');
      if (geography === 'CANADA') check(exposed === scope.matched_exports,'National exposure audit mismatch');
      const ranking = sectors.filter(row => row.exposed > 0).sort((a,b) => b.exposed - a.exposed || a.code.localeCompare(b.code));
      check(ranking.every(row => Number.isFinite(row.intensity) && row.intensity >= 0 && row.intensity <= 100),'Invalid intensity');
      cache.set(geography,{geography,name,year:meta.year,total,exposed,intensity:total > 0 ? exposed / total * 100 : null,ranking});
    }
    return cache;
  }

  function initialize(data) {
    const $ = id => document.getElementById(id);
    if (!$('geography-explorer')) return;
    let cache;
    try { cache = createSummaries(data); } catch (error) {
      $('geo-status').textContent = 'Geographic exposure data unavailable: ' + error.message;
      $('geo-ranking-note').textContent = 'Data unavailable. No missing observations have been substituted with zero.';
      console.error(error); return;
    }
    if (window.ExposureComparison) window.ExposureComparison.initialize(cache);
    const input = $('geo-search'), list = $('geo-options'), browse = $('geo-browse');
    const choices = [...cache.values()].map(({geography,name}) => ({code:geography,name}));
    let selected = 'CANADA', matches = [], active = -1;
    const tooltip = document.createElement('div'); tooltip.id = 'geo-tooltip'; tooltip.className = 'tooltip'; tooltip.setAttribute('role','tooltip'); tooltip.hidden = true; document.body.append(tooltip);
    let keyboardTooltip = null;
    const hideTooltip = () => { keyboardTooltip = null; tooltip.hidden = true; };
    const close = () => { list.hidden = true; input.setAttribute('aria-expanded','false'); browse.setAttribute('aria-expanded','false'); input.removeAttribute('aria-activedescendant'); active = -1; };
    function open(all = false) {
      const query = all ? '' : input.value.trim().toLowerCase();
      matches = choices.filter(choice => choice.name.toLowerCase().includes(query)); active = -1;
      list.innerHTML = matches.map(choice => `<li id="geo-option-${choice.code}" role="option" aria-selected="${choice.code === selected}" data-geo-code="${choice.code}">${escape(choice.name)}${choice.code === selected ? ' ✓' : ''}</li>`).join('');
      list.hidden = !matches.length; input.setAttribute('aria-expanded',String(matches.length > 0)); browse.setAttribute('aria-expanded',String(matches.length > 0)); input.removeAttribute('aria-activedescendant');
      $('geo-status').textContent = matches.length ? `${matches.length} matching geographies. Currently selected: ${cache.get(selected).name}.` : 'No matching geography. Clear restores Canada.';
    }
    function activate(index) {
      if (!matches.length) return;
      active = (index + matches.length) % matches.length;
      [...list.children].forEach((option,i) => option.classList.toggle('geo-active',i === active));
      const option = list.children[active]; input.setAttribute('aria-activedescendant',option.id); option.scrollIntoView({block:'nearest'});
    }
    function showTooltip(text,event,element) {
      tooltip.textContent = text; tooltip.hidden = false;
      const box = element.getBoundingClientRect();
      const x = event.clientX || box.left + box.width / 2, y = event.clientY || box.bottom;
      tooltip.style.left = Math.max(8,Math.min(window.innerWidth - tooltip.offsetWidth - 8,x + 14)) + 'px';
      tooltip.style.top = Math.max(8,Math.min(window.innerHeight - tooltip.offsetHeight - 8,y + 14)) + 'px';
    }
    function render() {
      hideTooltip(); const result = cache.get(selected);
      $('geo-selected').textContent = 'Selected: ' + result.name;
      $('geo-summary-title').textContent = `${result.name} — ${result.year} domestic exports to the US`;
      $('geo-ranking-title').textContent = `${result.name} — HS4 Exposure Ranking`;
      for (const [id,value,format] of [['geo-exposed',result.exposed,money],['geo-total',result.total,money],['geo-intensity',result.intensity,percent]]) {
        $(id).textContent = format(value); $(id).title = id === 'geo-intensity' ? percent(value) : fullMoney(value);
      }
      const maximum = result.ranking.length ? result.ranking[0].exposed : 0;
      $('geo-axis').innerHTML = maximum ? `<div>${[0,.5,1].map(f => `<span>${money(maximum * f)}</span>`).join('')}</div>` : '';
      const fragment = document.createDocumentFragment();
      for (const row of result.ranking) {
        const item = document.createElement('li'), button = document.createElement('button');
        const tip = `HS4 ${row.code} | ${row.description}\n${result.name} | Reference year: ${result.year}\nTotal US-bound domestic exports: ${fullMoney(row.total)}\nExposed HS6 export value: ${fullMoney(row.exposed)}\nExposure intensity: ${percent(row.intensity)}`;
        button.type = 'button'; button.className = 'geo-row'; button.dataset.hs4 = row.code;
        button.setAttribute('aria-label',tip); button.setAttribute('aria-describedby','geo-tooltip'); button.title = tip;
        button.innerHTML = `<span class="geo-row-label"><strong>${row.code}</strong><span>${escape(row.description)}</span></span><span class="geo-row-track" aria-hidden="true"><span class="geo-row-fill" style="width:${row.exposed / maximum * 100}%;background:${intensityColor(row.intensity)}"></span></span><span class="geo-row-value">${money(row.exposed)}<small>${percent(row.intensity)} intensity</small></span>`;
        ['pointerenter','pointermove','click'].forEach(name => button.addEventListener(name,event => showTooltip(tip,event,button)));
        button.addEventListener('focus',event => { keyboardTooltip = button; showTooltip(tip,event,button); });
        button.addEventListener('pointerleave',() => { if (keyboardTooltip !== button) hideTooltip(); });
        button.addEventListener('blur',hideTooltip);
        button.addEventListener('keydown',event => { if (event.key === 'Escape') hideTooltip(); });
        item.append(button); fragment.append(item);
      }
      $('geo-ranking').replaceChildren(fragment); $('geo-chart-scroll').scrollTop = 0;
      $('geo-ranking-note').textContent = result.ranking.length ? `${result.ranking.length} HS4 sectors with positive exposure, ranked by exposed value. Hover, focus, or tap a row for full descriptions and values. Zero-exposure sectors are excluded.` : 'No positive exposure is recorded for this geography. This is an observed zero, not unavailable data.';
      $('geo-status').textContent = `${result.name} selected. ${money(result.exposed)} exposed of ${money(result.total)} domestic exports to the US; ${percent(result.intensity)} overall intensity. ${result.ranking.length} HS4 sectors ranked.`;
    }
    function select(code) { selected = code; input.value = cache.get(code).name; close(); render(); }
    input.addEventListener('input',() => open());
    input.addEventListener('focus',() => { input.select(); open(true); });
    input.addEventListener('keydown',event => {
      if (event.key === 'Escape') { close(); input.value = cache.get(selected).name; hideTooltip(); }
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault(); if (list.hidden) open(true);
        activate(active < 0 ? (event.key === 'ArrowDown' ? 0 : matches.length - 1) : active + (event.key === 'ArrowDown' ? 1 : -1));
      }
      if (event.key === 'Enter' && !list.hidden && matches.length) { event.preventDefault(); select(matches[Math.max(0,active)].code); }
    });
    browse.addEventListener('pointerdown',event => event.preventDefault());
    browse.addEventListener('click',() => { const wasOpen = !list.hidden; input.focus(); if (wasOpen) close(); else open(true); });
    list.addEventListener('pointerdown',event => { const option = event.target.closest('[data-geo-code]'); if (option) { event.preventDefault(); select(option.dataset.geoCode); } });
    input.addEventListener('blur',() => { close(); input.value = cache.get(selected).name; });
    $('geo-clear').addEventListener('click',() => { input.focus(); select('CANADA'); });
    document.addEventListener('pointerdown',event => { if (!event.target.closest('.geo-search-controls')) close(); if (!event.target.closest('.geo-row')) hideTooltip(); });
    // Keyboard focus can itself scroll the page. Keep its tooltip positioned
    // while the focused row remains visible; pointer tooltips close on scroll.
    const onScroll = () => {
      const focused = document.activeElement;
      if (keyboardTooltip === focused && focused && focused.classList.contains('geo-row')) {
        const box = focused.getBoundingClientRect(), viewport = $('geo-chart-scroll').getBoundingClientRect();
        if (box.bottom > Math.max(0,viewport.top) && box.top < Math.min(window.innerHeight,viewport.bottom)) {
          showTooltip(focused.getAttribute('aria-label'),{},focused); return;
        }
        tooltip.hidden = true; return;
      }
      hideTooltip();
    };
    $('geo-chart-scroll').addEventListener('scroll',onScroll,{passive:true});
    window.addEventListener('scroll',onScroll,{passive:true}); window.addEventListener('resize',hideTooltip);
    for (const id of ['geo-search','geo-browse','geo-clear']) $(id).disabled = false;
    render();
  }
  const api = {createSummaries,money,percent,intensityColor,initialize};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else window.GeographyExposure = api;
})();
