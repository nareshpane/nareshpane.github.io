/* Static geography-first explorer. All values come from validated local JSON.
 * Rank = position in the builder's deterministic value-descending/code-ascending
 * order. Segment widths use exact integer amounts, not rounded stored shares.
 */
(() => {
  'use strict';
  const base = '2b-canada-provinces-trade-by-hs/';
  const $ = id => document.getElementById(id);
  const search = $('geography-search'), list = $('suggestions'), tip = $('tooltip');
  const cache = new Map(), destinationCache = new Map();
  let countryNames, destinationData = null, destinationLimit = 10, destinationRequest = 0;
  let metadata, descriptions, current, data, selected, activeChild = null;
  let suggestions = [], activeOption = -1, limit = 20, request = 0;
  let tipAnchor = null, tipKind = null;
  const integer = new Intl.NumberFormat('en-CA', {maximumFractionDigits:0});
  const exactMoney = n => '$' + integer.format(n);
  function money(n) {
    if (n >= 1e9) return '$' + (n / 1e9).toFixed(1) + 'B';
    if (n >= 1e6) return '$' + integer.format(n / 1e6) + 'M';
    return exactMoney(n);
  }
  const {percent,shade} = window.TradeDisplay;
  const escape = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const normalized = s => s.normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().trim();
  function concise(s) {
    if (s.length <= 155) return s;
    const clause = s.split(';')[0];
    if (clause.length <= 155 && clause.length >= 25) return clause + '…';
    return s.slice(0,155).replace(/\s+\S*$/,'') + '…';
  }
  async function json(file) {
    const response = await fetch(base + file);
    if (!response.ok) throw new Error(file + ': HTTP ' + response.status);
    return response.json();
  }
  function load(geography) {
    if (!cache.has(geography.id)) {
      const pending = json(geography.file).catch(error=>{cache.delete(geography.id); throw error;});
      cache.set(geography.id,pending);
    }
    return cache.get(geography.id);
  }
  function closeSuggestions() {
    list.hidden = true;
    search.setAttribute('aria-expanded','false');
    search.removeAttribute('aria-activedescendant');
    activeOption = -1;
  }
  function geographyMatches(query) {
    const q = normalized(query);
    return metadata.geographies.filter(g => !q || normalized(g.name).includes(q) || normalized(g.id).startsWith(q) || (g.statcan_id && g.statcan_id === q));
  }
  function showSuggestions() {
    if (!metadata) return;
    suggestions = geographyMatches(search.value);
    activeOption = -1;
    list.innerHTML = suggestions.length ? suggestions.map((g,i)=>`<li role="option" id="geo-option-${i}" data-option="${i}" aria-selected="false">${escape(g.name)} <span class="small">${g.id === 'CANADA' ? 'National total' : escape(g.id)}</span></li>`).join('') : '<li role="presentation">No matching geography.</li>';
    list.hidden = false;
    search.setAttribute('aria-expanded','true');
    search.removeAttribute('aria-activedescendant');
  }
  function moveOption(delta) {
    if (list.hidden) showSuggestions();
    if (!suggestions.length) return;
    activeOption = activeOption < 0 ? (delta > 0 ? 0 : suggestions.length - 1) : (activeOption + delta + suggestions.length) % suggestions.length;
    [...list.querySelectorAll('[role=option]')].forEach((el,i)=>el.setAttribute('aria-selected',String(i === activeOption)));
    const el = $('geo-option-' + activeOption);
    search.setAttribute('aria-activedescendant',el.id);
    el.scrollIntoView({block:'nearest'});
  }
  async function selectGeography(id, {keepHeading = true, blankSearch = false} = {}) {
    const g = metadata.geographies.find(g=>g.id === id);
    if (!g) return;
    const ticket = ++request, previousHeading = selected;
    closeSuggestions(); hideTip();
    ++destinationRequest; destinationData = null;
    search.value = blankSearch ? '' : g.name;
    $('results').hidden = true;
    $('live-status').textContent = 'Loading ' + g.name + ' · 2025 domestic exports…';
    try {
      const loaded = await load(g);
      if (ticket !== request) return; // A slow prior request cannot overwrite a new choice.
      if (loaded.geography !== g.id || loaded.year !== 2025 || loaded.total !== g.total) throw new Error('Geography/data mismatch');
      current = g; data = loaded; activeChild = null;
      selected = !keepHeading ? data.headings[0]?.[0] : previousHeading === null ? null : data.headings.some(h=>h[0] === previousHeading) ? previousHeading : data.headings[0]?.[0];
      $('geography-name').textContent = g.name;
      $('geography-id').textContent = g.id === 'CANADA' ? 'Derived from 13 StatCan origins' : 'StatCan ' + g.id + ' · ' + g.statcan_id;
      $('kpi-total').textContent = money(g.total); $('kpi-total').title = exactMoney(g.total) + ' CAD';
      $('kpi-hs4').textContent = integer.format(g.hs4Count);
      $('kpi-hs6').textContent = integer.format(g.hs6Count);
      $('kpi-share').textContent = percent(g.canadaShare);
      $('share-caption').textContent = g.id === 'CANADA' ? 'National benchmark · ' + percent(100) : 'Of Canadian domestic exports';
      $('largest-heading').textContent = g.largest[0] + ' · ' + concise(descriptions.hs4[g.largest[0]]);
      $('largest-heading').dataset.heading = g.largest[0];
      $('largest-value').textContent = money(g.largest[1]); $('largest-value').title = exactMoney(g.largest[1]) + ' CAD';
      renderRanking(); renderDetail();
      $('results').hidden = false;
      $('live-status').textContent = g.name + ' · 2025 · ' + exactMoney(g.total) + ' CAD · ' + g.hs4Count + ' HS4 headings and ' + g.hs6Count + ' HS6 products with positive exports.';
    } catch (error) {
      if (ticket !== request) return;
      $('live-status').textContent = 'Could not load ' + g.name + ': ' + error.message + '. Serve this page over HTTP; see the local testing instructions.';
      console.error(error);
    }
  }
  function segment(h, child, i, count) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'hs6-segment';
    button.dataset.heading = h[0]; button.dataset.child = child[0];
    button.style.width = child[1] / h[1] * 100 + '%';
    button.style.backgroundColor = shade(i + 1,count);
    button.setAttribute('aria-label',current.name + ', HS6 ' + child[0] + ', ' + descriptions.hs6[child[0]] + ', ' + exactMoney(child[1]) + ' CAD, ' + percent(child[1]/h[1]*100) + ' of HS4 ' + h[0] + ', rank ' + (i + 1));
    button.setAttribute('aria-describedby','tooltip');
    if (activeChild === child[0]) button.classList.add('active');
    return button;
  }
  function renderRanking() {
    const rows = limit === 'all' ? data.headings : data.headings.slice(0,limit);
    const max = data.headings[0]?.[1] || 1;
    const fragment = document.createDocumentFragment();
    rows.forEach((h,i)=>{
      const row = document.createElement('li');
      row.className = 'hs4-row' + (h[0] === selected ? ' selected' : '');
      row.dataset.hs4 = h[0];
      row.innerHTML = `<span class="rank">${i + 1}</span><button type="button" class="hs4-label" data-heading="${h[0]}" aria-describedby="tooltip" aria-controls="hs4-detail" aria-label="Inspect HS4 ${h[0]}, ${escape(descriptions.hs4[h[0]])}, rank ${i+1}, ${exactMoney(h[1])} CAD"><strong>${h[0]}</strong>${h[0].startsWith('98') || h[0].startsWith('99') ? '<span class="special-label">Special provision</span>' : ''}<span>${escape(concise(descriptions.hs4[h[0]]))}</span></button><div class="hs4-track"><div class="hs4-stack" role="group" aria-label="HS6 products in HS4 ${h[0]}" style="width:${h[1]/max*100}%"></div></div><button type="button" class="hs4-value" data-heading="${h[0]}" aria-describedby="tooltip" aria-label="HS4 ${h[0]}: ${exactMoney(h[1])} CAD">${money(h[1])}</button>`;
      h[3].forEach((child,j)=>row.querySelector('.hs4-stack').append(segment(h,child,j,h[3].length)));
      fragment.append(row);
    });
    $('ranked-bars').replaceChildren(fragment);
    $('rank-scroll').scrollTop = 0;
    $('bar-title').textContent = current.name + ' · ranked HS4 domestic exports';
    $('rank-scroll').setAttribute('aria-label',current.name + ': scrollable ranked HS4 chart');
    $('chart-axis').innerHTML = [0,.25,.5,.75,1].map(u=>`<span>${money(max*u)}</span>`).join('');
    const visibleTotal = rows.reduce((sum,h)=>sum+h[1],0);
    $('chart-summary').textContent = 'Showing ' + rows.length + ' of ' + current.hs4Count + ' positive-export headings · ' + percent(visibleTotal/data.total*100) + ' of ' + current.name + "'s domestic exports. HS4 rank is by value; HS6 colours restart within each bar.";
  }
  function renderDetail() {
    const i = data.headings.findIndex(h=>h[0] === selected), h = data.headings[i];
    if (!h) {
      $('detail-body').hidden = true; $('detail-empty').hidden = false;
      $('detail-summary').textContent = '';
      ['detail-title','detail-code','detail-description','description-source','detail-total','detail-rank','detail-share','composition-example','table-caption'].forEach(id=>$(id).textContent='');
      $('detail-stack').replaceChildren(); $('hs6-rows').replaceChildren();
      renderComparison(); renderDestinations(); return;
    }
    $('detail-body').hidden = false; $('detail-empty').hidden = true;
    $('detail-summary').textContent = current.name + ' · HS4 ' + h[0];
    $('detail-code').textContent = 'HS2 ' + h[0].slice(0,2) + ' → HS4 ' + h[0];
    $('detail-title').textContent = current.name + ' · HS4 ' + h[0];
    $('detail-description').textContent = descriptions.hs4[h[0]];
    const provenance = descriptions.hs4_sources[h[0]];
    $('description-source').innerHTML = 'Heading description: ' + (provenance.url ? `<a href="${escape(provenance.url)}">${escape(provenance.source)}</a>` : escape(provenance.source)) + '. ' + escape(provenance.method) + '. HS6 labels: StatCan CIMT, valid during 2025.';
    $('detail-total').textContent = money(h[1]); $('detail-total').title = exactMoney(h[1]) + ' CAD';
    $('detail-rank').textContent = '#' + (i+1) + ' of ' + current.hs4Count;
    $('detail-share').textContent = percent(h[1]/data.total*100);
    const stack = document.createDocumentFragment();
    h[3].forEach((child,j)=>stack.append(segment(h,child,j,h[3].length)));
    $('detail-stack').replaceChildren(stack);
    const first = h[3][0];
    $('composition-example').textContent = 'For example, in ' + current.name + "'s observed 2025 exports, HS6 " + first[0] + ' (' + descriptions.hs6[first[0]] + ') contributes ' + money(first[1]) + ', or ' + percent(first[1]/h[1]*100) + ', of HS4 ' + h[0] + '. ' + (h[3].length === 1 ? 'This heading has one positive-export subheading here.' : 'The remaining ' + (h[3].length-1) + ' positive-export subheadings complete the bar.');
    $('table-caption').textContent = current.name + ' · HS4 ' + h[0] + ' · all ' + h[3].length + ' positive-export HS6 products · 2025';
    $('hs6-rows').innerHTML = h[3].map((child,j)=>`<tr data-hs6="${child[0]}"><td>${j+1}</td><th scope="row"><button type="button" class="hs6-product" data-heading="${h[0]}" data-child="${child[0]}" aria-describedby="tooltip"><strong>${child[0]}</strong>${escape(descriptions.hs6[child[0]])}</button></th><td>${exactMoney(child[1])}</td><td>${percent(child[1]/h[1]*100)}</td><td><div class="mini-track" aria-hidden="true"><i style="width:${child[1]/h[1]*100}%;background:${shade(j+1,h[3].length)}"></i></div></td></tr>`).join('');
    renderComparison(); renderDestinations();
  }
  function renderComparison() {
    $('comparison-body').hidden = !selected;
    if (!selected) { $('comparison-title').textContent = 'No HS4 category selected.'; $('comparison-rows').replaceChildren(); return; }
    $('comparison-title').textContent = 'HS4 ' + selected + ' · ' + concise(descriptions.hs4[selected]);
    const values = metadata.comparisons[selected];
    $('comparison-rows').innerHTML = metadata.geographies.map(g=>{
      const record = values[g.id];
      return `<tr class="${g.id === current.id ? 'current' : ''}"><th scope="row"><button type="button" data-compare="${g.id}">${escape(g.name)}</button></th><td>${record ? exactMoney(record[0]) : 'No positive recorded exports'}</td><td>${record ? '#' + record[1] : '—'}</td><td>${record ? percent(record[0]/g.total*100) : '—'}</td></tr>`;
    }).join('');
  }
  function selectHeading(code, child, scroll = false) {
    if (!data.headings.some(h=>h[0] === code)) return;
    hideTip(); selected = code; activeChild = child || null;
    document.querySelectorAll('.hs4-row').forEach(row=>row.classList.toggle('selected',row.dataset.hs4 === code));
    document.querySelectorAll('#ranked-bars .hs6-segment').forEach(el=>el.classList.toggle('active',el.dataset.child === activeChild));
    $('hs4-detail').open = true;
    renderDetail();
    if (scroll) $('hs4-detail').scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'auto':'smooth'});
    $('live-status').textContent = current.name + ' · HS4 ' + code + ' selected · ' + money(data.headings.find(h=>h[0]===code)[1]) + ' in 2025 domestic exports.';
  }

  function clearHeading() {
    selected = null; activeChild = null; hideTip();
    document.querySelectorAll('.hs4-row.selected,.hs6-segment.active').forEach(el=>el.classList.remove('selected','active'));
    renderDetail();
    $('live-status').textContent = current.name + ' · HS4 selection cleared.';
  }
  function loadDestinations(g) {
    if (!destinationCache.has(g.id)) {
      destinationCache.set(g.id,json('destinations-' + g.id + '-2025.json').catch(error=>{destinationCache.delete(g.id);throw error;}));
    }
    return destinationCache.get(g.id);
  }
  async function renderDestinations() {
    const ticket = ++destinationRequest, code = selected, g = current;
    destinationData = null;
    $('destination-bars').replaceChildren();
    $('destination-other').hidden = true; $('destination-other').textContent = '';
    $('destination-summary').textContent = ''; $('destination-reconciliation').hidden = true;
    $('destination-content').hidden = true; $('destination-controls').disabled = true;
    if (!code) {
      $('destination-subtitle').textContent = 'Select an HS4 category in Section 02 to see its export destinations.';
      return;
    }
    const h = data.headings.find(h=>h[0] === code);
    $('destination-subtitle').textContent = '2025 domestic exports from ' + g.name + ' of HS4 ' + code + ' · ' + descriptions.hs4[code] + ', ranked by destination country.';
    $('destination-summary').textContent = 'Loading official destination observations…';
    try {
      const loaded = await loadDestinations(g);
      if (ticket !== destinationRequest || current.id !== g.id || selected !== code) return;
      if (loaded.geography !== g.id || loaded.year !== 2025) throw new Error('Destination/geography mismatch');
      destinationData = loaded.headings[code] || [];
      if (!destinationData.length) {
        $('destination-subtitle').textContent += ' No eligible destination-country observations are recorded for this selection.';
        return;
      }
      $('destination-controls').disabled = false;
      $('destination-content').hidden = false;
      $('destination-scroll').setAttribute('aria-label',g.name + ' HS4 ' + code + ': scrollable destination-country ranking');
      drawDestinations(h);
    } catch(error) {
      if (ticket !== destinationRequest) return;
      $('destination-subtitle').textContent += ' Destination data could not be loaded: ' + error.message + '.';
      console.error(error);
    }
  }
  function drawDestinations(h = data.headings.find(h=>h[0] === selected)) {
    if (!h || !destinationData) return;
    if (tipAnchor?.dataset.destination) hideTip();
    const rows = destinationLimit === 'all' ? destinationData : destinationData.slice(0,destinationLimit);
    const max = destinationData[0]?.[1] || 1;
    $('destination-bars').innerHTML = rows.map(([code,value],i)=>'<li><button type="button" class="destination-row" data-destination="'+code+'" aria-describedby="tooltip" aria-label="'+escape(countryNames[code])+', rank '+(i+1)+', '+exactMoney(value)+' CAD, '+percent(value/h[1]*100)+' of selected HS4 exports"><span class="rank">'+(i+1)+'</span><span class="destination-name">'+escape(countryNames[code])+'</span><span class="destination-track" aria-hidden="true"><i class="destination-fill" style="width:'+value/max*100+'%"></i></span><span class="destination-value">'+money(value)+'</span><span class="destination-share">'+percent(value/h[1]*100)+'</span></button></li>').join('');
    $('destination-scroll').scrollTop = 0;
    const shown = rows.reduce((s,r)=>s+r[1],0), all = destinationData.reduce((s,r)=>s+r[1],0), other = all-shown;
    $('destination-other').hidden = !other;
    $('destination-other').textContent = other ? 'Other destinations ('+(destinationData.length-rows.length)+', unranked): '+exactMoney(other)+' CAD · '+percent(other/h[1]*100)+' of selected HS4 exports.' : '';
    $('destination-summary').textContent = 'Showing '+rows.length+' of '+destinationData.length+' destinations · shares use the complete '+money(h[1])+' origin-HS4 total, including destinations outside the displayed Top N.';
    const difference = h[1]-all;
    $('destination-reconciliation').hidden = difference === 0;
    $('destination-reconciliation').textContent = 'Individual international destinations sum to '+exactMoney(all)+' CAD; '+exactMoney(difference)+' CAD of the recorded HS4 total belongs to excluded destination categories. Country values have not been rescaled.';
  }
  function wireDestinations() {
    const container = $('destination-bars');
    container.addEventListener('pointerover',event=>{const el=event.target.closest('[data-destination]');if(el)showTip(el,event);});
    container.addEventListener('pointermove',event=>{if(!tip.hidden)positionTip(event.clientX,event.clientY);});
    container.addEventListener('pointerout',hideTip);
    container.addEventListener('focusin',event=>{const el=event.target.closest('[data-destination]');if(el)showTip(el);});
    container.addEventListener('focusout',hideTip);
    container.addEventListener('click',event=>{const el=event.target.closest('[data-destination]');if(el)showTip(el,event);});
    container.addEventListener('keydown',event=>{if(event.key==='Escape')hideTip();});
    $('destination-scroll').addEventListener('scroll',viewportTip,{passive:true});
  }
  function tooltipFor(el) {
    if (el.dataset.destination) {
      const h = data.headings.find(h=>h[0] === selected);
      const i = destinationData?.findIndex(r=>r[0] === el.dataset.destination);
      const record = destinationData?.[i];
      if (!h || !record) return '';
      return '<strong>'+escape(current.name)+' · 2025</strong><br>HS4 '+h[0]+' · '+escape(descriptions.hs4[h[0]])+'<hr><strong>'+escape(countryNames[record[0]])+'</strong><span class="tip-value">'+exactMoney(record[1])+' CAD</span>'+percent(record[1]/h[1]*100)+' of selected HS4 exports · Destination rank '+(i+1)+' of '+destinationData.length;
    }
    const index = data.headings.findIndex(h=>h[0] === el.dataset.heading), h = data.headings[index];
    if (!h) return '';
    const name = `<strong>${escape(current.name)} · 2025</strong><br>HS4 ${h[0]} · ${escape(descriptions.hs4[h[0]])}`;
    if (el.dataset.child) {
      const j = h[3].findIndex(c=>c[0] === el.dataset.child), child = h[3][j];
      return name + `<hr>HS6 <strong>${child[0]}</strong> · ${escape(descriptions.hs6[child[0]])}<span class="tip-value">${exactMoney(child[1])} CAD</span>${percent(child[1]/h[1]*100)} of parent HS4 · HS6 rank ${j+1} of ${h[3].length}`;
    }
    return name + `<span class="tip-value">${exactMoney(h[1])} CAD</span>HS4 rank ${index+1} · ${percent(h[1]/data.total*100)} of ${escape(current.name)}'s domestic exports`;
  }
  function positionTip(x,y) {
    const rect = tip.getBoundingClientRect();
    tip.style.left = Math.max(8,Math.min(x+12,innerWidth-rect.width-8)) + 'px';
    tip.style.top = Math.max(8,Math.min(y+14,innerHeight-rect.height-8)) + 'px';
  }
  function showTip(el,event) {
    tip.innerHTML = tooltipFor(el);
    if (!tip.innerHTML) return;
    tipAnchor = el;
    tipKind = !event || event.type === 'click' && event.detail === 0 ? 'focus' : 'pointer';
    tip.hidden = false;
    const r = el.getBoundingClientRect();
    positionTip(event?.clientX ?? r.left + Math.min(r.width/2,80), event?.clientY ?? r.bottom);
  }
  function hideTip() { tip.hidden = true; tipAnchor = null; tipKind = null; }
  function viewportTip() {
    // Keyboard focus can scroll a newly focused segment into view. Keep its
    // tooltip available and anchored after that scroll; pointer tips dismiss.
    if (tipKind === 'focus' && tipAnchor?.isConnected && document.activeElement === tipAnchor) {
      const r = tipAnchor.getBoundingClientRect();
      positionTip(r.left + Math.min(r.width/2,80),r.bottom);
    } else hideTip();
  }
  // Delegation keeps the All view inexpensive even with thousands of segments.
  function wireChart(container) {
    container.addEventListener('pointerover',event=>{
      const el = event.target.closest('[data-heading]');
      if (el && data) showTip(el,event);
    });
    container.addEventListener('pointermove',event=>{if (!tip.hidden) positionTip(event.clientX,event.clientY);});
    container.addEventListener('pointerout',event=>{if (!container.contains(event.relatedTarget) || event.target.closest('[data-heading]')) hideTip();});
    container.addEventListener('focusin',event=>{
      const el = event.target.closest('[data-heading]');
      if (el && data) showTip(el);
    });
    container.addEventListener('focusout',hideTip);
    container.addEventListener('click',event=>{
      const el = event.target.closest('[data-heading]');
      if (el) {
        // Keep keyboard focus intact when selecting inside the detail section.
        if (container.id === 'ranked-bars') selectHeading(el.dataset.heading,el.dataset.child,true);
        else {
          activeChild = el.dataset.child || null;
          document.querySelectorAll('.hs6-segment').forEach(s=>s.classList.toggle('active',s.dataset.child === activeChild));
        }
        showTip(el,event);
      }
    });
    container.addEventListener('keydown',event=>{
      if (event.key === 'Escape') hideTip();
      if (!['ArrowRight','ArrowLeft'].includes(event.key)) return;
      const el = event.target.closest('.hs6-segment');
      if (!el) return;
      event.preventDefault();
      const sibling = event.key === 'ArrowRight' ? el.nextElementSibling : el.previousElementSibling;
      sibling?.focus();
    });
  }
  async function resetGeographySearch() {
    await selectGeography('CANADA',{keepHeading:false,blankSearch:true});
    search.focus(); showSuggestions();
  }
  async function init() {
    try {
      [metadata,descriptions,countryNames] = await Promise.all([json('geography-summary-2025.json'),json('product-descriptions.json'),json('destination-countries-2025.json').then(d=>d.countries)]);
      await selectGeography('CANADA');
      search.disabled = false; $('clear-search').disabled = false; $('clear-geography').disabled = false;
      document.querySelectorAll('[data-geography]').forEach(el=>{el.disabled=false;el.addEventListener('click',()=>selectGeography(el.dataset.geography));});
      search.addEventListener('input',showSuggestions);
      search.addEventListener('focus',showSuggestions);
      search.addEventListener('keydown',event=>{
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); moveOption(event.key === 'ArrowDown'?1:-1); }
        else if (event.key === 'Enter') {
          event.preventDefault();
          const g = suggestions[activeOption] || geographyMatches(search.value)[0];
          if (g) selectGeography(g.id);
        } else if (event.key === 'Escape') closeSuggestions();
        else if (event.key === 'Tab') closeSuggestions();
      });
      list.addEventListener('pointerdown',event=>event.preventDefault());
      list.addEventListener('click',event=>{const el=event.target.closest('[data-option]');if(el)selectGeography(suggestions[Number(el.dataset.option)].id);});
      document.addEventListener('pointerdown',event=>{if(!event.target.closest('.search-shell'))closeSuggestions();if(!event.target.closest('[data-heading],[data-destination]'))hideTip();});
      $('clear-search').addEventListener('click',resetGeographySearch);
      $('clear-geography').addEventListener('click',resetGeographySearch);
      document.querySelectorAll('[name=top-limit]').forEach(el=>el.addEventListener('change',()=>{limit=el.value === 'all'?'all':Number(el.value);hideTip();if(data)renderRanking();}));
      $('clear-heading').addEventListener('click',clearHeading);
      document.querySelectorAll('[name=destination-limit]').forEach(el=>el.addEventListener('change',()=>{destinationLimit=el.value==='all'?'all':Number(el.value);drawDestinations();}));
      wireDestinations();
      $('largest-heading').addEventListener('click',()=>selectHeading($('largest-heading').dataset.heading,null,true));
      $('comparison-rows').addEventListener('click',event=>{const el=event.target.closest('[data-compare]');if(el)selectGeography(el.dataset.compare,{keepHeading:true});});
      wireChart($('ranked-bars')); wireChart($('detail-stack')); wireChart($('hs6-rows'));
      $('rank-scroll').addEventListener('scroll',viewportTip,{passive:true});
      window.addEventListener('resize',viewportTip,{passive:true});
      window.addEventListener('scroll',viewportTip,{passive:true});
      // Small per-geography files are prefetched after Canada is usable. A cache
      // miss hides prior results while loading; rapid choices use a request token.
      metadata.geographies.filter(g=>g.id!=='CANADA').forEach(g=>load(g).catch(()=>{}));
      json('validation-2025.json').then(a=>{
        $('validation-summary').textContent = integer.format(a.source_rows) + ' monthly HS6 records · 12 months · ' + a.destination_categories + ' destination categories · zero duplicate records or keys, zero missing/suppressed values, and CAD $0 reconciliation differences. ' + a.later_cbsa_heading_labels.length + ' HS4 labels use the separately identified later CBSA reference. Canada is ' + exactMoney(a.annual_total) + ' CAD.';
      }).catch(()=>{$('validation-summary').textContent='Read the linked validation audit for all checks.';});
    } catch (error) {
      $('live-status').textContent = 'Unable to load the local JSON assets: ' + error.message + '. Open this page through the HTTP preview URL in README.md, not file://.';
      console.error(error);
    }
  }
  init();
})();
