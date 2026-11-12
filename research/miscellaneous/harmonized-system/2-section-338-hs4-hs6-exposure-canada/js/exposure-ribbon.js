/* HS4 exposure composition. References the existing annual arrays and policy
 * membership; never fetches or reconstructs a second analytical dataset. */
(() => {
  'use strict';
  const sum = list => list.reduce((a,b) => a + b,0);
  const check = (condition,message) => { if (!condition) throw new Error(message); };
  const escape = value => String(value).replace(/[&<>"']/g,c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  function createModel({hs4,hs6,products,origins,scope}) {
    check(scope && Array.isArray(scope.hs6),'Section 338 scope is unavailable');
    check(origins.length === 13,'Expected thirteen origins');
    const policy = new Set(scope.hs6), descriptions = new Map(products.map(p => [p.code,p.description]));
    const geographyIndex = new Map(origins.map(([code],i) => [code,i]));
    const groups = new Map(Object.keys(hs4).map(code => [code,[]]));
    for (const [code,values] of Object.entries(hs6)) {
      check(/^\d{6}$/.test(code) && groups.has(code.slice(0,4)),'Invalid HS6 parent: ' + code);
      check(values.length === 13 && values.every(v => Number.isSafeInteger(v) && v >= 0),'Invalid HS6 values: ' + code);
      groups.get(code.slice(0,4)).push({code,description:descriptions.get(code) || 'Description unavailable in supplied lookup',values,exposed:policy.has(code)});
    }
    function calculate(heading,geography = 'CANADA') {
      check(groups.has(heading),'Select a four-digit heading');
      const index = geographyIndex.get(geography);
      check(geography === 'CANADA' || index !== undefined,'Unknown geography: ' + geography);
      const valueFor = values => geography === 'CANADA' ? sum(values) : values[index];
      const children = groups.get(heading).map(child => ({code:child.code,description:child.description,
        exposed:child.exposed,value:valueFor(child.values)})).sort((a,b) => b.value - a.value || a.code.localeCompare(b.code));
      const total = sum(children.map(child => child.value));
      const exposed = sum(children.filter(child => child.exposed).map(child => child.value));
      const unexposed = sum(children.filter(child => !child.exposed).map(child => child.value));
      check(total === valueFor(hs4[heading]),'HS4 does not equal all HS6 children: ' + heading + '/' + geography);
      check(exposed + unexposed === total && unexposed === total - exposed,'Exposure reconciliation failed');
      const intensity = total > 0 ? exposed / total * 100 : null;
      check(intensity === null || intensity >= 0 && intensity <= 100,'Intensity outside 0–100');
      return {heading,geography,total,exposed,unexposed,intensity,children,
        childCount:children.length,exposedChildCount:children.filter(child => child.exposed).length};
    }
    // Validate the new measure for every heading × Canada / origin combination.
    // Integer CAD gives exact equality, without needing a floating tolerance.
    for (const heading of groups.keys()) for (const geography of ['CANADA',...geographyIndex.keys()]) calculate(heading,geography);
    return {calculate,headings:[...groups.keys()]};
  }

  function groupSegments(result) {
    if (!result.total) return [];
    const positive = result.children.filter(child => child.value > 0), kept = [];
    let covered = 0;
    // Keep the largest children until 95% is covered, capped at 12 individual
    // segments. Both remaining status groups still contribute in full.
    for (const child of positive) {
      if (kept.length === 12 || covered >= result.total * .95) break;
      kept.push(child); covered += child.value;
    }
    const segments = kept.map(child => ({key:child.code,label:child.code,value:child.value,exposed:child.exposed,children:[child]}));
    const rest = positive.slice(kept.length);
    for (const exposed of [true,false]) {
      const children = rest.filter(child => child.exposed === exposed);
      if (children.length) segments.push({key:'other-' + (exposed ? 'exposed' : 'unexposed'),
        label:'Other ' + (exposed ? 'exposed' : 'unexposed') + ' HS6',exposed,
        value:sum(children.map(child => child.value)),children});
    }
    check(sum(segments.map(segment => segment.value)) === result.total,'Ribbon lost trade value');
    check(sum(segments.filter(segment => segment.exposed).map(segment => segment.value)) === result.exposed,'Ribbon changed exposure');
    return segments;
  }

  const $ = id => document.getElementById(id);
  let model, helpers, origins, heading, geography = 'CANADA', filter = 'all', result;
  let initialized = false, selectedCode = null, pinned = null, generation = 0;
  const pieces = new Map(), removalTimers = new Map();
  const geographyName = () => geography === 'CANADA' ? 'Canada' : origins.find(o => o[0] === geography)[1];
  const ratio = (value,total) => total > 0 ? value / total * 100 : null;
  function tooltipFor(entry) {
    const first = entry.children[0], grouped = entry.children.length > 1;
    const identity = grouped ? `${entry.label} · ${entry.children.length} smaller children; see the ranked list` : `HS6 ${first.code} · ${first.description}`;
    const contribution = entry.exposed ? ratio(entry.value,result.exposed) : 0;
    return `${identity}\n${geographyName()} · 2025 exports: ${helpers.fullMoney(entry.value)}\nShare of selected HS4: ${helpers.percent(ratio(entry.value,result.total))}\nSection 338: ${entry.exposed ? 'Exposed (HS6 scope match)' : 'Not exposed (no HS6 scope match)'}\nContribution to exposed value: ${helpers.percent(contribution)}\nContribution to HS4 intensity: ${entry.exposed ? helpers.percent(ratio(entry.value,result.total)).replace('%',' percentage points') : '0.0 percentage points'}`;
  }
  function highlight(codes) {
    const selected = new Set(codes || []);
    $('hs4-exposure').querySelectorAll('[data-exposure-codes]').forEach(element => {
      const matches = element.dataset.exposureCodes.split(',').some(code => selected.has(code));
      element.classList.toggle('exposure-highlighted',matches);
      element.classList.toggle('exposure-muted',selected.size > 0 && !matches);
    });
  }
  function bindInspection(element) {
    function inspect(event) {
      highlight(element._exposure.children.map(child => child.code));
      helpers.showTooltip(tooltipFor(element._exposure),event,element);
    }
    const clear = () => { highlight(pinned); helpers.hideTooltip(); };
    ['pointerenter','pointermove','focus'].forEach(name => element.addEventListener(name,inspect));
    ['pointerleave','blur'].forEach(name => element.addEventListener(name,clear));
    element.addEventListener('click',event => {
      const codes = element._exposure.children.map(child => child.code);
      pinned = pinned && pinned.join(',') === codes.join(',') ? null : codes;
      highlight(pinned); helpers.showTooltip(tooltipFor(element._exposure),event,element);
    });
    element.addEventListener('keydown',event => { if (event.key === 'Escape') { pinned = null; clear(); } });
  }
  function renderRibbon() {
    const ribbon = $('exposure-ribbon'), visible = groupSegments(result);
    const keys = new Set(visible.map(segment => heading + ':' + segment.key));
    const currentGeneration = ++generation;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    for (const [key,element] of pieces) if (!keys.has(key)) {
      element.style.width = '0%'; element.disabled = true; element.setAttribute('aria-hidden','true');
      clearTimeout(removalTimers.get(key));
      if (reduced) { element.remove(); pieces.delete(key); }
      else removalTimers.set(key,setTimeout(() => { element.remove(); pieces.delete(key); removalTimers.delete(key); },420));
    }
    ribbon.classList.toggle('exposure-empty',result.total === 0);
    ribbon.setAttribute('aria-label',`${geographyName()} HS4 ${heading}: ${result.total ? helpers.percent(result.intensity) + ' exposure intensity' : 'no exports; intensity unavailable'}`);
    for (const segment of visible) {
      const key = heading + ':' + segment.key;
      let element = pieces.get(key);
      if (!element) {
        element = document.createElement('button'); element.type = 'button'; element.style.width = '0%';
        element.className = 'exposure-piece'; pieces.set(key,element); bindInspection(element);
      }
      clearTimeout(removalTimers.get(key)); removalTimers.delete(key);
      element.disabled = false; element.removeAttribute('aria-hidden'); element._exposure = segment;
      element.dataset.exposureCodes = segment.children.map(child => child.code).join(',');
      element.classList.toggle('is-exposed',segment.exposed);
      const pct = ratio(segment.value,result.total);
      element.classList.toggle('exposure-small-piece',pct < (segment.children.length > 1 ? 18 : 8));
      element.textContent = segment.children.length > 1 ? 'Other' : segment.children[0].code;
      element.setAttribute('aria-label',tooltipFor(segment)); ribbon.append(element);
      const settle = () => { if (generation === currentGeneration) element.style.width = pct + '%'; };
      if (reduced) settle(); else requestAnimationFrame(() => requestAnimationFrame(settle));
    }
    const other = visible.filter(segment => segment.key.startsWith('other-'));
    $('exposure-grouping').textContent = result.total === 0 ? 'No exports recorded for this heading and geography. Exposure intensity is undefined, not 0%.' :
      `All ${result.childCount} HS6 children contribute to the measure. ${other.length ? 'The largest children are kept until 95% of value is covered, with a cap of 12 individual segments; the remainder is grouped separately by exposure status.' : 'The ribbon retains the positive-value children individually.'} Zero-value children have no ribbon width and remain in the breakdown.`;
  }
  function renderLadder() {
    const children = result.children.filter(child => filter === 'all' || child.exposed);
    const ladder = $('exposure-ladder'); ladder.replaceChildren();
    if (!children.length) { ladder.innerHTML = '<p class="small">No Section 338-matched HS6 children in this heading.</p>'; return; }
    const list = document.createElement('ol'); list.setAttribute('aria-label','HS6 children ranked by export value');
    children.forEach((child,index) => {
      const item = document.createElement('li'), button = document.createElement('button');
      button.type = 'button'; button.className = 'exposure-row';
      button._exposure = {label:child.code,value:child.value,exposed:child.exposed,children:[child]};
      button.dataset.exposureCodes = child.code;
      button.setAttribute('aria-label',tooltipFor(button._exposure));
      button.innerHTML = `<span class="exposure-rank" aria-hidden="true">${index + 1}</span><span class="exposure-child"><strong>${child.code}</strong><span>${escape(child.description)}</span></span><span class="exposure-row-value">${helpers.money(child.value)}<small>${helpers.percent(ratio(child.value,result.total))} of HS4</small></span><span class="exposure-row-status ${child.exposed ? 'is-exposed' : ''}">${child.exposed ? 'Exposed' : 'Not exposed'}</span>`;
      bindInspection(button); item.append(button); list.append(item);
    });
    ladder.append(list); highlight(pinned);
  }
  function render() {
    pinned = null; helpers.hideTooltip(); highlight(null);
    result = model.calculate(heading,geography);
    helpers.animateNumber($('exposure-intensity'),result.intensity || 0,n => result.intensity === null ? 'n/a' : helpers.percent(n));
    for (const [id,value] of [['exposure-value',result.exposed],['exposure-total',result.total],['exposure-unexposed',result.unexposed]]) {
      helpers.animateNumber($(id),value,helpers.money); $(id).title = helpers.fullMoney(value);
    }
    $('exposure-caption').textContent = `${geographyName()} · HS4 ${heading} · 2025 → United States`;
    $('exposure-intensity').title = result.intensity === null ? 'No exports: intensity unavailable' : helpers.percent(result.intensity);
    renderRibbon(); renderLadder();
    $('exposure-live').textContent = `${geographyName()}, HS4 ${heading}: ${helpers.percent(result.intensity)} intensity. Exposed ${helpers.fullMoney(result.exposed)} of ${helpers.fullMoney(result.total)}; not exposed ${helpers.fullMoney(result.unexposed)}. ${result.exposedChildCount} of ${result.childCount} HS6 children matched in the trade dataset. Supplied policy scope, not a certified current legal schedule.`;
  }
  function initialize(data,callbacks) {
    helpers = callbacks; origins = data.origins;
    try { model = createModel(data); } catch (error) {
      $('exposure-live').textContent = 'Exposure composition is unavailable: ' + error.message;
      console.error(error); return;
    }
    initialized = true;
    const selector = $('exposure-geography');
    selector.innerHTML = [['CANADA','Canada'],...origins].map(([code,name]) => `<option value="${code}">${escape(name)}</option>`).join('');
    selector.disabled = false;
    $('exposure-ladder').addEventListener('scroll',helpers.hideTooltip,{passive:true});
    selector.addEventListener('change',() => { geography = selector.value; render(); });
    document.querySelectorAll('[name=exposure-filter]').forEach(input => input.addEventListener('change',() => {
      filter = input.value; pinned = null; helpers.hideTooltip(); renderLadder();
      $('exposure-live').textContent = `${filter === 'all' ? 'All HS6 children' : 'Section 338-matched children only'} shown in the breakdown. The ribbon and intensity still use all HS6 children.`;
    }));
    const info = $('exposure-info'), infoText = $('exposure-info-text').textContent;
    ['pointerenter','focus','click'].forEach(name => info.addEventListener(name,event => helpers.showTooltip(infoText,event,info)));
    ['pointerleave','blur'].forEach(name => info.addEventListener(name,helpers.hideTooltip));
    info.addEventListener('keydown',event => { if (event.key === 'Escape') helpers.hideTooltip(); });
  }
  function select(code) {
    selectedCode = code;
    if (!initialized) return;
    const isHeading = code.length === 4;
    $('exposure-body').hidden = !isHeading; $('exposure-geography-control').hidden = !isHeading;
    $('exposure-message').hidden = isHeading;
    if (!isHeading) {
      generation++; helpers.hideTooltip();
      $('exposure-message').innerHTML = `HS4 exposure composition is available when an HS4 heading is selected. <button type="button" class="control" id="exposure-parent">View parent HS4 ${code.slice(0,4)}</button>`;
      $('exposure-parent').addEventListener('click',() => helpers.onSelect(selectedCode.slice(0,4)));
      $('exposure-live').textContent = 'Select an HS4 heading to inspect its exposure composition.';
      return;
    }
    heading = code; render();
  }
  const api = {createModel,groupSegments,initialize,select};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else window.HS4Exposure = api;
})();
