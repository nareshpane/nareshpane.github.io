/* Section 3 consumes Section 1's cached HS4 summaries. No data fetch, trade
 * aggregation, policy reconstruction, or dependency on explorer selections. */
(() => {
  'use strict';
  const check = (condition,message) => { if (!condition) throw new Error(message); };
  const sum = entries => entries.reduce((total,entry) => total + entry.exposed,0);
  const colors = ['#164b7c','#3777ab','#6ba3ce','#a1c7e5','#d6dfce'];

  function createComparison(cache) {
    check(cache instanceof Map && cache.size === 14,'Expected fourteen cached geographies');
    // Copy rather than sort or annotate Section 1's cache in place.
    const rows = [...cache.values()].map(result => {
      const {geography,name,year,exposed,ranking} = result;
      if (!Number.isSafeInteger(exposed) || exposed < 0 || !Array.isArray(ranking)) {
        return {geography,name,year,exposed:null,available:false,components:[],topThree:[]};
      }
      check(ranking.every(row => Number.isSafeInteger(row.exposed) && row.exposed > 0),'Invalid sector value');
      check(new Set(ranking.map(row => row.code)).size === ranking.length,'Duplicated HS4 sector');
      const sectors = [...ranking].sort((a,b) => b.exposed - a.exposed || a.code.localeCompare(b.code));
      check(sum(sectors) === exposed,'HS4 exposure does not reconcile: ' + geography);
      const leading = sectors.slice(0,4).map((row,index) => ({code:row.code,description:row.description,
        exposed:row.exposed,rank:index+1,share:exposed > 0 ? row.exposed / exposed * 100 : null}));
      const remainder = sum(sectors.slice(4));
      const components = [...leading];
      if (remainder > 0) components.push({code:null,description:'Other HS4 sectors',exposed:remainder,rank:5,share:remainder / exposed * 100});
      check(sum(components) === exposed,'Stack reconciliation failed: ' + geography);
      return {geography,name,year,exposed,available:true,components,topThree:leading.slice(0,3)};
    }).sort((a,b) => Number(b.available) - Number(a.available) || (a.available ? b.exposed - a.exposed : 0) || a.geography.localeCompare(b.geography));
    const maximum = Math.max(0,...rows.filter(row => row.available).map(row => row.exposed));
    return {rows,maximum};
  }

  function tooltipText(row,format) {
    if (!row.available) return `${row.name}\nData unavailable; not ranked.`;
    const lines = [`${row.name} | ${row.year} domestic exports to the US`, `Total exposed exports: ${format.money(row.exposed)}`];
    row.topThree.forEach((sector,i) => lines.push(`\n${i+1}. HS4 ${sector.code} — ${sector.description}\n${format.money(sector.exposed)} | ${format.percent(sector.share)} of ${row.name}'s exposure`));
    if (!row.exposed) lines.push('\nNo positive exposed HS4 exports recorded; percentage shares are unavailable.');
    // Supplement the three-sector list so a tiny fourth segment is also
    // inspectable at its real width, without imposing any minimum geometry.
    const fourth = row.components.find(component => component.rank === 4);
    if (fourth) lines.push(`\n4th bar segment: HS4 ${fourth.code} — ${fourth.description}\n${format.money(fourth.exposed)} | ${format.percent(fourth.share)} of ${row.name}'s exposure`);
    const other = row.components.find(component => component.rank === 5);
    if (other) lines.push(`\nOther exposed HS4 sectors: ${format.money(other.exposed)} | ${format.percent(other.share)} of ${row.name}'s exposure`);
    lines.push('\nSupplied September scope reconstruction; potential HS6 exposure, not verified tariff liability.');
    return lines.join('\n');
  }

  function initialize(cache) {
    const $ = id => document.getElementById(id), list = $('comparison-bars');
    if (!list) return;
    let model;
    try { model = createComparison(cache); } catch (error) {
      $('comparison-status').textContent = 'Data unavailable: ' + error.message; return;
    }
    const format = window.GeographyExposure;
    const tooltip = document.createElement('div'); tooltip.id = 'comparison-tooltip'; tooltip.className = 'tooltip'; tooltip.setAttribute('role','tooltip'); tooltip.hidden = true; document.body.append(tooltip);
    let owner = null, keyboard = null, pinned = null, hideTimer;
    const hide = () => {
      clearTimeout(hideTimer); tooltip.hidden = true;
      if (owner) owner.setAttribute('aria-expanded','false');
      owner = null; keyboard = null; pinned = null;
    };
    function show(element,event = {}) {
      clearTimeout(hideTimer);
      if (owner && owner !== element) owner.setAttribute('aria-expanded','false');
      owner = element; element.setAttribute('aria-expanded','true'); tooltip.textContent = element._comparisonTip; tooltip.hidden = false;
      const box = element.getBoundingClientRect();
      const x = event.clientX || box.left + box.width / 2, y = event.clientY || box.bottom;
      tooltip.style.left = Math.max(8,Math.min(window.innerWidth - tooltip.offsetWidth - 8,x+14)) + 'px';
      tooltip.style.top = Math.max(8,Math.min(window.innerHeight - tooltip.offsetHeight - 8,y+14)) + 'px';
    }
    $('comparison-axis').replaceChildren(...[0,.25,.5,.75,1].map(f => {
      const tick = document.createElement('span'); tick.textContent = format.money(model.maximum * f); return tick;
    }));
    const fragment = document.createDocumentFragment();
    for (const row of model.rows) {
      const item = document.createElement('li'), button = document.createElement('button'), label = document.createElement('span');
      button.type = 'button'; button.className = 'comparison-row'; button.dataset.geography = row.geography;
      button._comparisonTip = tooltipText(row,format); button.setAttribute('aria-label',button._comparisonTip);
      button.setAttribute('aria-controls','comparison-tooltip'); button.setAttribute('aria-expanded','false');
      label.className = 'comparison-label'; label.textContent = row.name; button.append(label);
      if (row.available) {
        const track = document.createElement('span'), stack = document.createElement('span'), value = document.createElement('span');
        track.className = 'comparison-track'; stack.className = 'comparison-stack';
        track.style.setProperty('--comparison-width',(model.maximum > 0 ? row.exposed / model.maximum * 100 : 0) + '%');
        stack.setAttribute('aria-hidden','true');
        row.components.forEach(component => {
          const segment = document.createElement('span'); segment.className = 'comparison-segment ' + (component.rank < 5 ? 'comparison-rank-'+component.rank : 'comparison-other');
          segment.style.width = component.share + '%'; segment.dataset.rank = component.rank;
          segment.dataset.value = component.exposed; segment.dataset.hs4 = component.code || 'other'; stack.append(segment);
        });
        value.className = 'comparison-total'; value.textContent = format.money(row.exposed);
        track.append(stack,value); button.append(track);
      } else {
        const message = document.createElement('span'); message.className = 'comparison-unavailable'; message.textContent = 'Data unavailable — not ranked'; button.append(message);
      }
      ['pointerenter','pointermove'].forEach(name => button.addEventListener(name,event => show(button,event)));
      button.addEventListener('focus',event => { keyboard = button; show(button,event); });
      button.addEventListener('pointerleave',event => {
        if (keyboard === button || pinned === button || tooltip.contains(event.relatedTarget)) return;
        hideTimer = setTimeout(hide,180);
      });
      button.addEventListener('blur',event => { keyboard = null; if (pinned !== button && !tooltip.contains(event.relatedTarget)) hide(); });
      button.addEventListener('click',event => { if (pinned === button) hide(); else { pinned = button; show(button,event); } });
      button.addEventListener('keydown',event => { if (event.key === 'Escape') { event.preventDefault(); hide(); } });
      item.append(button); fragment.append(item);
    }
    list.replaceChildren(fragment);
    tooltip.addEventListener('pointerenter',() => clearTimeout(hideTimer));
    tooltip.addEventListener('pointerleave',() => { if (!keyboard && !pinned) hide(); });
    tooltip.addEventListener('keydown',event => { if (event.key === 'Escape') hide(); });
    document.addEventListener('pointerdown',event => { if (!event.target.closest('.comparison-row') && !tooltip.contains(event.target)) hide(); });
    const reposition = () => {
      if (keyboard && document.activeElement === keyboard) {
        const box = keyboard.getBoundingClientRect();
        if (box.bottom > 0 && box.top < window.innerHeight) show(keyboard);
        else tooltip.hidden = true;
      } else hide();
    };
    window.addEventListener('scroll',reposition,{passive:true}); window.addEventListener('resize',reposition);
    const missing = model.rows.filter(row => !row.available).length;
    $('comparison-status').textContent = `${model.rows.length} geographies · 2025 domestic exports to the US · CAD · ${missing ? missing + ' unavailable and unranked.' : 'All stacked HS4 contributions reconcile to Section 1 exposure totals.'}`;
  }
  const api = {createComparison,tooltipText,colors,initialize};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else window.ExposureComparison = api;
})();
