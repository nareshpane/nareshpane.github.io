/* Annual HS4 / HS6 explorer. No runtime dependencies or monthly browser data. */
(() => {
  'use strict';
  const DATA = '2-section-338-hs4-hs6-exposure-canada/data/';
  const $ = id => document.getElementById(id);
  const sum = values => values.reduce((a, b) => a + b, 0);
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const colors = ['#38765f','#487593','#815b31','#356e74','#6b5c80','#8c5c45','#37678a','#765d37','#536c49','#486d8c','#685a75','#716039','#516e60'];
  const state = {code:null, measure:'value', heatMeasure:'value', more:false, policy:false, pinned:null};
  let meta, hs4, hs6, products, productMap, origins, scope = null, scopeSet = new Set();
  let suggestions = [], activeSuggestion = -1;
  const tariffCache = new Map(), rows = new Map(), segments = new Map();
  const numberFrames = new WeakMap();

  // Formatting and transitions ------------------------------------------------
  function money(value) {
    for (const [scale, suffix] of [[1e9,'B'],[1e6,'M'],[1e3,'K']]) {
      if (value >= scale) return '$' + (value / scale).toFixed(1) + suffix;
    }
    return '$' + Math.round(value).toLocaleString('en-CA');
  }
  const fullMoney = value => new Intl.NumberFormat('en-CA', {style:'currency',currency:'CAD',maximumFractionDigits:0}).format(value) + ' CAD';
  const percent = value => value === null ? 'n/a' : (value > 0 && value < .1 ? '<0.1%' : value.toFixed(1) + '%');
  const share = (value, total) => total > 0 ? value / total * 100 : null;
  const values = code => (code.length === 4 ? hs4 : hs6)[code];
  const children = code => products.filter(p => p.code.length === 6 && p.code.startsWith(code));
  function animateNumber(element, target, format) {
    cancelAnimationFrame(numberFrames.get(element));
    const initial = Number(element.dataset.value || 0);
    if (reduced.matches || initial === target) {
      element.textContent = format(target); element.dataset.value = target; return;
    }
    const start = performance.now();
    function frame(now) {
      const progress = reduced.matches ? 1 : Math.min(1, (now - start) / 400);
      const current = initial + (target - initial) * (1 - Math.pow(1 - progress, 3));
      element.dataset.value = current; element.textContent = format(current);
      if (progress < 1) numberFrames.set(element, requestAnimationFrame(frame));
    }
    numberFrames.set(element, requestAnimationFrame(frame));
  }
  async function loadJSON(name) {
    const response = await fetch(DATA + name);
    if (!response.ok) throw new Error(name + ': HTTP ' + response.status);
    return response.json();
  }

  // Search: code and description ranking; one combobox for both levels --------
  function normalizedText(text) {
    return text.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, ' ').trim();
  }
  function searchProducts(query) {
    const raw = query.trim();
    if (!raw) return [];
    const isCode = /^[\d.\s_\-/]+$/.test(raw);
    const codeQuery = raw.replace(/[.\s_\-/]/g, '');
    const words = normalizedText(raw).split(' ').filter(Boolean).map(w => w === 'electrical' ? 'electric' : w.length > 3 && w.endsWith('s') ? w.slice(0,-1) : w);
    const phrase = words.join(' ');
    return products.map(p => {
      let rank = 99;
      if (isCode && p.code === codeQuery) rank = 0;
      else if (isCode && p.code.startsWith(codeQuery)) rank = 1;
      else if (!isCode && p.search.replace(/^other /,'').startsWith(phrase)) rank = 2;
      else if (!isCode && words.every(w => new RegExp('\\b' + w + '(?:s|es)?\\b').test(p.search))) rank = 3;
      else if (!isCode && (words.every(w => p.search.includes(w)) || p.search.replace(/ /g,'').includes(phrase.replace(/ /g,'')))) rank = 4;
      return {p, rank};
    }).filter(r => r.rank < 99).sort((a,b) => a.rank - b.rank || a.p.code.length - b.p.code.length || a.p.code.localeCompare(b.p.code)).slice(0,10).map(r => r.p);
  }
  function highlighted(text, query) {
    const words = normalizedText(query).split(' ').filter(Boolean);
    const codeQuery = query.replace(/[.\s_\-/]/g, '');
    if (/^\d+$/.test(codeQuery)) words.push(codeQuery);
    if (!words.length) return escape(text);
    const pattern = new RegExp('(' + words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|') + ')', 'ig');
    return text.split(pattern).map((part, i) => i % 2 ? '<mark>' + escape(part) + '</mark>' : escape(part)).join('');
  }
  function closeSearch() {
    $('suggestions').hidden = true; $('product-search').setAttribute('aria-expanded','false');
    $('product-search').removeAttribute('aria-activedescendant'); activeSuggestion = -1;
  }
  function updateSearch() {
    const query = $('product-search').value;
    suggestions = searchProducts(query); activeSuggestion = -1;
    $('suggestions').innerHTML = suggestions.map((p, i) => `<li id="suggestion-${i}" role="option" aria-selected="false" data-index="${i}"><div><span class="suggestion-level">HS${p.code.length}</span><strong>${highlighted(p.code,query)}</strong></div><span>${highlighted(p.description,query)}</span></li>`).join('');
    $('suggestions').hidden = !suggestions.length;
    $('product-search').setAttribute('aria-expanded',String(!!suggestions.length));
    $('product-search').removeAttribute('aria-activedescendant');
    $('live-status').textContent = query.trim() ? suggestions.length ? `${suggestions.length} suggestions shown. Use the arrow keys and Enter to select.` : 'No matching product. Try a shorter code or another description.' : 'Search an HS4 or HS6 product, or try an example.';
  }
  function activateSuggestion(index) {
    if (!suggestions.length) return;
    activeSuggestion = (index + suggestions.length) % suggestions.length;
    $('suggestions').querySelectorAll('[role=option]').forEach((option,i) => option.setAttribute('aria-selected',String(i === activeSuggestion)));
    const id = 'suggestion-' + activeSuggestion;
    $('product-search').setAttribute('aria-activedescendant',id);
    $(id).scrollIntoView({block:'nearest'});
  }
  function setupSearch() {
    $('product-search').addEventListener('input', updateSearch);
    $('product-search').addEventListener('focus', () => { if ($('product-search').value) updateSearch(); });
    $('product-search').addEventListener('keydown', event => {
      if (event.key === 'Escape') { closeSearch(); hideTooltip(); return; }
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault(); if ($('suggestions').hidden) updateSearch();
        activateSuggestion(activeSuggestion < 0 ? (event.key === 'ArrowDown' ? 0 : suggestions.length - 1) : activeSuggestion + (event.key === 'ArrowDown' ? 1 : -1));
      }
      if (event.key === 'Enter' && !$('suggestions').hidden && suggestions.length) {
        event.preventDefault(); selectProduct(suggestions[Math.max(0,activeSuggestion)].code);
      }
    });
    $('suggestions').addEventListener('pointerdown', event => {
      const option = event.target.closest('[data-index]');
      if (option) { event.preventDefault(); selectProduct(suggestions[Number(option.dataset.index)].code); }
    });
    document.addEventListener('pointerdown', event => { if (!event.target.closest('.search-shell')) closeSearch(); });
    $('product-search').addEventListener('blur', () => setTimeout(closeSearch, 150));
    $('clear-search').addEventListener('click', () => { $('product-search').value = ''; closeSearch(); $('product-search').focus(); $('live-status').textContent = 'Search cleared. The selected chart remains visible.'; });
    document.querySelectorAll('[data-product]').forEach(button => button.addEventListener('click', () => selectProduct(button.dataset.product)));
  }

  // Tooltips and linked origin highlighting ----------------------------------
  function showTooltip(text, event, element) {
    const tooltip = $('tooltip'); tooltip.textContent = text; tooltip.hidden = false;
    const box = element.getBoundingClientRect();
    const x = event && event.clientX || box.left + box.width / 2;
    const y = event && event.clientY || box.bottom;
    tooltip.style.left = Math.max(8,Math.min(window.innerWidth - tooltip.offsetWidth - 8,x + 14)) + 'px';
    tooltip.style.top = Math.max(8,Math.min(window.innerHeight - tooltip.offsetHeight - 8,y + 14)) + 'px';
  }
  const hideTooltip = () => { $('tooltip').hidden = true; };
  function highlightOrigin(code) {
    document.querySelectorAll('[data-origin]').forEach(el => {
      el.classList.toggle('highlighted', el.dataset.origin === code);
      el.classList.toggle('dimmed', !!code && el.dataset.origin !== code && el.dataset.origin !== 'CANADA');
    });
    if (code && code !== 'CANADA' && state.code) {
      const i = origins.findIndex(o => o[0] === code), v = values(state.code), total = sum(v);
      $('composition-readout').textContent = `${origins[i][1]} · ${fullMoney(v[i])} · ${percent(share(v[i],total))} of Canada's selected-product exports.`;
    } else $('composition-readout').textContent = 'Hover, focus or tap an origin to connect the two views. Zero-value origins have no segment.';
  }
  function bindOrigin(element, code, valueFn) {
    const inspect = event => { highlightOrigin(code); showTooltip(valueFn(),event,element); };
    element.addEventListener('pointerenter',inspect); element.addEventListener('pointermove',inspect);
    element.addEventListener('focus',inspect);
    const clear = () => { highlightOrigin(state.pinned); hideTooltip(); };
    element.addEventListener('pointerleave',clear); element.addEventListener('blur',clear);
    element.addEventListener('click', event => { state.pinned = state.pinned === code ? null : code; highlightOrigin(state.pinned); showTooltip(valueFn(),event,element); });
    element.addEventListener('keydown', event => { if (event.key === 'Escape') { state.pinned = null; clear(); } });
  }
  function originText(code) {
    const v = values(state.code), total = sum(v);
    if (code === 'CANADA') return `Canada · ${fullMoney(total)} · Sum of all thirteen origins`;
    const i = origins.findIndex(o => o[0] === code);
    return `${origins[i][1]} · ${fullMoney(v[i])} · ${percent(share(v[i],total))} of Canada`;
  }

  // Persistent bar elements let CSS animate length and ranking changes --------
  function setupCharts() {
    [['CANADA','CANADA'], ...origins].forEach(([code,name],i) => {
      const row = document.createElement('button'); row.type = 'button'; row.className = 'bar-row'; row.dataset.origin = code;
      row.style.setProperty('--origin-color',code === 'CANADA' ? '#234e50' : colors[i-1]);
      row.innerHTML = `<span class="bar-label">${escape(name)}</span><span class="bar-track" aria-hidden="true"><span class="bar-fill"></span></span><span class="bar-value" aria-hidden="true"></span>`;
      $('bars').append(row); rows.set(code,row); bindOrigin(row,code,() => originText(code));
      if (code !== 'CANADA') {
        const segment = document.createElement('button'); segment.type = 'button'; segment.className = 'segment'; segment.dataset.origin = code;
        segment.style.backgroundColor = colors[i-1]; segment.textContent = code;
        $('composition').append(segment); segments.set(code,segment); bindOrigin(segment,code,() => originText(code));
      }
    });
    document.querySelectorAll('[name=bar-measure]').forEach(input => input.addEventListener('change', () => {
      state.measure = input.value;
      // Unit changes fade labels rather than tweening dollars into absurd percentages.
      const v = values(state.code), total = sum(v);
      [['CANADA',total],...origins.map((o,i) => [o[0],v[i]])].forEach(([code,value]) => {
        const element = rows.get(code).querySelector('.bar-value');
        element.dataset.value = state.measure === 'share' ? (share(value,total) || 0) : value;
      });
      $('bars').classList.remove('unit-change'); renderBars();
      requestAnimationFrame(() => $('bars').classList.add('unit-change'));
    }));
    document.querySelectorAll('[name=heat-measure]').forEach(input => input.addEventListener('change', () => { state.heatMeasure = input.value; renderHeatmap(); }));
    $('heat-more').addEventListener('click', () => { state.more = !state.more; renderHeatmap(); });
    $('show-policy').addEventListener('change', () => { state.policy = $('show-policy').checked; renderPolicy(); });
  }
  function renderBars() {
    const v = values(state.code), total = sum(v);
    const order = origins.map((o,i) => ({code:o[0],value:v[i]})).sort((a,b) => b.value - a.value || a.code.localeCompare(b.code));
    const chartRows = [{code:'CANADA',value:total},...order];
    const rowHeight = window.innerWidth <= 600 ? 64 : 43;
    $('bars').style.height = chartRows.length * rowHeight + 'px';
    chartRows.forEach(({code,value},rank) => {
      const row = rows.get(code); row.style.transform = `translateY(${rank * rowHeight}px)`;
      const proportion = total ? value / total * 100 : 0;
      // Both modes share Canada as their maximum, so bar geometry is deliberately
      // comparable; the axis and labels change units without a misleading rescale.
      row.querySelector('.bar-fill').style.width = proportion + '%';
      const displayed = state.measure === 'share' ? proportion : value;
      animateNumber(row.querySelector('.bar-value'), displayed, state.measure === 'share' ? percent : money);
      row.setAttribute('aria-label',originText(code)); row.style.zIndex = String(15-rank);
      // DOM order follows visible rank as well, preserving keyboard reading order.
      $('bars').append(row);
    });
    $('chart-axis').innerHTML = [0,.25,.5,.75,1].map(f => `<span>${escape(state.measure === 'share' ? percent(f * 100) : money(total * f))}</span>`).join('');
    $('chart-summary').textContent = total ? `${fullMoney(total)} for HS${state.code.length} ${state.code}. Canada equals the sum of all thirteen origins; the benchmark is not an additional origin.` : 'No positive annual exports are recorded for this product. Province shares are unavailable.';
    if (!total && state.measure === 'share') chartRows.forEach(r => { const e = rows.get(r.code).querySelector('.bar-value'); cancelAnimationFrame(numberFrames.get(e)); e.textContent = 'n/a'; });
  }
  function renderComposition() {
    const v = values(state.code), total = sum(v);
    origins.forEach(([code],i) => {
      const segment = segments.get(code), pct = total ? v[i] / total * 100 : 0;
      segment.style.width = pct + '%'; segment.classList.toggle('small-segment',pct < 6);
      segment.disabled = v[i] === 0; segment.setAttribute('aria-label',originText(code));
      segment.setAttribute('title',originText(code));
    });
    $('origin-legend').innerHTML = origins.map(([code,name],i) => `<button type="button" data-origin="${code}" style="--origin-color:${colors[i]}"><span class="swatch" aria-hidden="true"></span><span>${escape(name)}</span><strong>${money(v[i])} · ${percent(share(v[i],total))}</strong></button>`).join('');
    $('origin-legend').querySelectorAll('button').forEach(button => bindOrigin(button,button.dataset.origin,() => originText(button.dataset.origin)));
    highlightOrigin(state.pinned);
  }

  // HS4 drilldown: values are explicit, with a sequential square-root scale ----
  function heatColor(value, maximum) {
    const t = maximum > 0 ? Math.sqrt(value / maximum) : 0;
    const light = [242,246,242], dark = [37,105,109];
    return {color:`rgb(${light.map((c,i) => Math.round(c + (dark[i]-c)*t)).join(',')})`, dark:t > .65};
  }
  function renderHeatmap() {
    hideTooltip();
    const isHeading = state.code.length === 4;
    $('heat-title').textContent = isHeading ? 'Inside this HS4 heading' : 'At the international detail level';
    $('heat-controls').hidden = !isHeading; $('heat-more').hidden = true;
    if (!isHeading) {
      $('heat-note').textContent = 'HS6 is the most detailed international level used in this explorer. Select its HS4 parent to compare sibling products.';
      $('heat-content').innerHTML = `<button class="control" type="button" data-child="${state.code.slice(0,4)}">Explore HS4 ${state.code.slice(0,4)} →</button>`;
    } else {
      const all = children(state.code).sort((a,b) => sum(hs6[b.code]) - sum(hs6[a.code]) || a.code.localeCompare(b.code));
      const chosen = all.slice(0,state.more ? 24 : 10), parent = values(state.code);
      const cellValue = (p,i) => state.heatMeasure === 'share' ? (share(hs6[p.code][i],parent[i]) || 0) : hs6[p.code][i];
      const maximum = Math.max(0,...all.flatMap(p => origins.map((_,i) => cellValue(p,i))));
      $('heat-note').textContent = `${chosen.length} of ${all.length} HS6 children recorded in this U.S.-bound trade dataset, ranked by Canada value. ${state.heatMeasure === 'share' ? 'Each column divides by that origin’s exports within this HS4 heading; it does not divide by all provincial exports.' : 'Cell values are annual CAD. The same scale covers all children in this heading.'} Select a code to explore it. Swipe the table on narrow screens.`;
      $('heat-content').innerHTML = `<div class="heat-legend small"><span class="heat-gradient" aria-hidden="true"></span><span>0 → ${state.heatMeasure === 'share' ? percent(maximum) : money(maximum)} · square-root color scale</span></div><div class="table-wrap" tabindex="0" aria-label="Scrollable HS6 heatmap"><table class="heatmap"><caption class="sr-only">HS6 child exports by province and territory</caption><thead><tr><th scope="col">HS6 product</th>${origins.map(([code,name]) => `<th scope="col"><abbr title="${escape(name)}">${code}</abbr></th>`).join('')}</tr></thead><tbody>${chosen.map(p => `<tr><th scope="row"><button type="button" class="child-product" data-child="${p.code}" title="${escape(p.description)}"><strong>${p.code}</strong><span>${escape(p.description)}</span></button></th>${origins.map(([code,name],i) => {
        const dollars = hs6[p.code][i], pct = share(dollars,parent[i]), color = heatColor(cellValue(p,i),maximum);
        const text = state.heatMeasure === 'share' ? percent(pct) : money(dollars);
        const tip = `${name} · HS6 ${p.code} · ${p.description} · ${fullMoney(dollars)} · ${percent(pct)} of this origin's selected HS4 exports`;
        return `<td><button type="button" class="heat-cell" style="background:${color.color};color:${color.dark ? '#fff' : '#263d3b'}" data-tooltip="${escape(tip)}" aria-label="${escape(tip)}">${text}</button></td>`;
      }).join('')}</tr>`).join('')}</tbody></table></div>`;
      $('heat-more').hidden = all.length <= 10;
      $('heat-more').textContent = state.more ? 'Show less' : `Show more (up to ${Math.min(24,all.length)})`;
    }
    $('heat-content').querySelectorAll('[data-child]').forEach(button => button.addEventListener('click', () => selectProduct(button.dataset.child)));
    $('heat-content').querySelectorAll('[data-tooltip]').forEach(button => {
      ['pointerenter','pointermove','focus','click'].forEach(event => button.addEventListener(event,e => showTooltip(button.dataset.tooltip,e,button)));
      ['pointerleave','blur'].forEach(event => button.addEventListener(event,hideTooltip));
      button.addEventListener('keydown',event => { if (event.key === 'Escape') hideTooltip(); });
    });
  }

  // Canadian import context: lazy bundles preserve multiple rates and rows ----
  async function renderTariffs() {
    const code = state.code, content = $('tariff-content');
    if (code.length === 4) {
      content.innerHTML = `<p>This heading has ${children(code).length} HS6 children in the trade dataset. Canadian tariff treatment is defined below HS6, at national tariff items.</p><p class="small">Select an HS6 child for associated HS8 items, MFN treatment and preferential tariff text.</p>`; return;
    }
    content.textContent = 'Loading the associated Canadian tariff items…';
    try {
      if (!tariffCache.has(code[0])) tariffCache.set(code[0],loadJSON('tariffs-' + code[0] + '.json'));
      const bundle = await tariffCache.get(code[0]);
      if (state.code !== code) return; // An earlier selection must not overwrite a newer one.
      const product = bundle.products[code];
      if (!product || !product.rows.length) { content.textContent = 'No compatible Canadian tariff detail is available in the local Page 1 snapshot for this code.'; return; }
      const items = [...new Set(product.rows.map(row => row[0]))];
      const rates = [...new Set(product.rows.map(row => bundle.strings[row[4]] || 'Blank source cell'))];
      content.innerHTML = `<p><strong>${items.length} associated Canadian HS8 tariff ${items.length === 1 ? 'item' : 'items'}</strong> · ${product.rows.length} source rows including statistical suffixes.</p><p class="small"><strong>MFN treatments in the source:</strong> ${rates.map(escape).join('; ')}. Read each item separately; this is not a single HS6 rate.</p>${product.warning ? `<p class="source-warning">${escape(product.warning)}</p>` : ''}<details><summary>View all tariff items &amp; preferential text</summary><div class="table-wrap" tabindex="0" aria-label="Scrollable Canadian tariff table"><table class="tariff-table"><caption>CBSA T2026-2 · ${code}</caption><thead><tr>${['HS8 item','SS','Description','Unit','MFN','Preferential tariffs'].map(t => `<th scope="col">${t}</th>`).join('')}</tr></thead><tbody>${product.rows.map(row => `<tr>${[row[0],row[1],...row.slice(2).map(i => bundle.strings[i])].map(cell => `<td>${escape(cell || '—')}</td>`).join('')}</tr>`).join('')}</tbody></table></div><p class="small">“—” denotes a blank source cell. Descriptions include ancestor wording.</p></details><p class="small"><a href="${escape(product.url)}">Read the full CBSA chapter ↗</a></p><p class="small">Preferences depend on origin, rules and evidence. Canadian import treatments are separate from the U.S. policy match.</p>`;
    } catch (error) {
      tariffCache.delete(code[0]);
      if (state.code === code) content.textContent = 'Tariff context could not load. See Page 1 and the official CBSA source; the annual export explorer remains available.';
      console.error(error);
    }
  }

  // Optional policy membership, with no multiplication of trade values --------
  function renderPolicy() {
    $('policy-content').hidden = !state.policy; $('policy-badge').hidden = !state.policy;
    if (!state.policy) return;
    if (!scope) { $('policy-content').textContent = 'The supplied policy data are unavailable. Export charts remain independent of this layer.'; $('policy-badge').hidden = true; return; }
    const code = state.code, reconstruction = scope.mode === 'supplied-september-reconstruction';
    let intro;
    if (code.length === 6) {
      const matched = scopeSet.has(code);
      $('policy-badge').textContent = matched ? 'Section 338 · HS6 scope match' : 'Section 338 · No HS6 scope match';
      intro = `<p><strong>${matched ? 'Matched at HS6' : 'No HS6 match'} in the ${reconstruction ? 'reconstructed supplied September scope' : 'supplied July product list'}.</strong></p>`;
    } else {
      const all = children(code), matched = all.filter(p => scopeSet.has(p.code));
      const exposed = sum(matched.map(p => sum(hs6[p.code]))), total = sum(values(code));
      $('policy-badge').textContent = `${matched.length} of ${all.length} observed HS6 children matched`;
      intro = `<dl class="policy-stats"><div><dt>HS6 children in this trade dataset</dt><dd>${all.length}</dd></div><div><dt>Children matched to policy scope</dt><dd>${matched.length}</dd></div><div><dt>Exports in matched children</dt><dd>${money(exposed)}</dd></div><div><dt>Share of this heading’s Canada → U.S. exports</dt><dd>${percent(share(exposed,total))}</dd></div></dl>`;
    }
    $('policy-content').innerHTML = intro + `<p class="small">This matches U.S. policy lines to the first six digits of Canadian product codes. It does not mean every shipment or Canadian HS8 item is covered, that these exports paid a tariff, or that the policy caused an equivalent economic loss.</p><p class="small">${escape(scope.note)}</p><p class="small">Across all products: ${scope.hs6.length} unique scope codes; ${scope.positive_export_matches} positive-export matches; ${money(scope.matched_exports)} (${scope.exposure_share.toFixed(2)}%) of the 2025 baseline. <a href="#methodology">Methodology ↓</a></p>`;
  }

  // State and audit ----------------------------------------------------------
  function selectProduct(code) {
    const product = productMap.get(code); if (!product) return;
    closeSearch(); hideTooltip(); state.code = code; state.more = false; state.pinned = null;
    $('product-search').value = code; $('results').hidden = false;
    $('selected-level').textContent = 'HS' + code.length; $('selected-code').textContent = code;
    $('selected-description').textContent = product.description;
    $('description-source').textContent = 'Description: ' + meta.description_sources[product.source];
    const v = values(code), total = sum(v), largest = Math.max(...v), index = v.indexOf(largest);
    animateNumber($('kpi-total'),total,money); $('kpi-origin').textContent = total ? origins[index][1] : 'No positive exports';
    $('kpi-share').textContent = total ? percent(share(largest,total)) + ' of Canada’s selected-product exports' : 'Share unavailable';
    animateNumber($('kpi-count'),v.filter(value => value > 0).length,n => Math.round(n) + ' / 13');
    renderBars(); renderComposition(); renderHeatmap(); renderTariffs(); renderPolicy();
    if (window.HS4Exposure) window.HS4Exposure.select(code);
    $('live-status').textContent = `Selected HS${code.length} ${code}. ${product.description}. Canada: ${fullMoney(total)}. Charts updated.`;
  }
  function renderAudit() {
    const a = meta.validation;
    $('validation-summary').innerHTML = `<p>${a.raw_rows.toLocaleString()} raw observations; ${a.selected_rows.toLocaleString()} filtered U.S. observations; ${a.hs6_annual_rows.toLocaleString()} positive product-origin HS6 rows and ${a.hs4_annual_rows.toLocaleString()} HS4 rows. Dense arrays include explicit zeroes for all 13 origins.</p><p>${a.hs4_products.toLocaleString()} HS4 products; ${a.hs6_products.toLocaleString()} HS6 products. Raw all-destination total: ${fullMoney(a.raw_value_total)}. U.S. total before and after aggregation: ${fullMoney(a.annual_value_total)}. Exact duplicate records: ${a.duplicate_rows}. ${escape(a.reconciliation)}.</p><p>${scope ? `${scope.hs6.length} reconstructed policy HS6 codes; ${scope.positive_export_matches} positive matches; ${fullMoney(scope.matched_exports)}; ${scope.exposure_share.toFixed(4)}%.` : 'Policy context unavailable.'} Rounded reference targets are comparison checks, never inputs to aggregation.</p>`;
    if (scope) $('policy-sources').innerHTML = scope.sources.filter(source => source.urls).map(source => `<li>${escape(source.file)}: ${source.urls.map(url => `<a href="${escape(url)}">Recorded source ↗</a>`).join(' · ')}</li>`).join('');
  }
  function setupIntro() {
    const svg = $('intro-visual');
    const observer = new IntersectionObserver(entries => { if (entries.some(e => e.isIntersecting)) { svg.classList.add('played'); observer.disconnect(); } },{threshold:.2});
    observer.observe(svg);
  }
  async function init() {
    setupIntro();
    try {
      [meta, hs4, hs6, products] = await Promise.all(['metadata.json','exports-hs4-2025-us.json','exports-hs6-2025-us.json','search-index.json'].map(loadJSON));
      origins = meta.origins;
      if (origins.length !== 13 || meta.year !== '2025' || meta.destination !== 'US') throw new Error('Unexpected annual dataset schema');
      products = products.map(([code,description,source]) => ({code,description,source,search:normalizedText(description)}));
      productMap = new Map(products.map(p => [p.code,p]));
      for (const data of [hs4,hs6]) for (const [code,v] of Object.entries(data)) {
        if (!productMap.has(code) || v.length !== 13 || v.some(n => !Number.isSafeInteger(n) || n < 0)) throw new Error('Invalid product array: ' + code);
      }
      try { scope = await loadJSON('section338-hs6.json'); scopeSet = new Set(scope.hs6); } catch (error) { console.error(error); }
      if (window.HS4Exposure) window.HS4Exposure.initialize({hs4,hs6,products,origins,scope},
        {money,fullMoney,percent,animateNumber,showTooltip,hideTooltip,onSelect:selectProduct});
      else $('exposure-live').textContent = 'The exposure module could not load. The existing export explorer is available.';
      setupSearch(); setupCharts(); renderAudit();
      document.querySelectorAll('#product-search,#clear-search,[data-product]').forEach(el => { el.disabled = false; });
      selectProduct('8414');
      window.addEventListener('resize', () => { hideTooltip(); if (state.code) renderBars(); });
      window.addEventListener('scroll',hideTooltip,{passive:true});
      reduced.addEventListener('change', () => { if (state.code) { renderBars(); renderComposition(); } });
    } catch (error) {
      $('live-status').textContent = 'The explorer could not load its local annual data. Use the Python HTTP preview command in the project README, or check the JSON asset paths.';
      $('live-status').classList.add('error'); console.error(error);
    }
  }
  init();
})();
