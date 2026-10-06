'use strict';
(() => {
  const base = '1-harmonized-system-canada/';
  const $ = id => document.getElementById(id);
  const state = {nodes: [], sections: [], mode: 'all', lens: null};
  const el = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  };
  function link(url, label) {
    const a = el('a', '', label); a.href = url; return a;
  }
  function setOpen(record, open) {
    record.button.setAttribute('aria-expanded', String(open));
    record.panel.hidden = !open;
  }
  function canadianTariffDetails(data) {
    const details = el('details', 'canadian-tariff-details');
    const lines = data.canadian_tariff_lines || [];
    const summary = el('summary', '', 'Canadian tariff details');
    summary.append(el('span', 'tariff-row-count', ` · ${lines.length} source ${lines.length === 1 ? 'row' : 'rows'}`));
    details.append(summary);
    // Do not auto-open the secondary accordion during search or Expand all.
    // Build its table only on first opening, avoiding thousands of hidden tables.
    let rendered = false;
    details.addEventListener('toggle', () => {
      if (!details.open || rendered) return;
      rendered = true;
      details.append(el('p', '', `Canadian tariff lines beneath HS6 ${data.display_code}. Units and tariff treatments belong to these national rows, not to the international HS6 itself.`));
      if (!lines.length) {
        details.append(el('p', '', 'No Canadian tariff rows are available in this snapshot. Consult the official chapter schedule.'));
        return;
      }
      const note = el('p', 'tariff-table-note', 'Source rows are shown separately, including base tariff-item rows with blank SS and their statistical suffix rows. Repeated values are retained as printed; rates are not inherited or combined. An em dash (—) marks a blank source cell, not a zero rate or suffix. Scroll within the table to see all rows and columns.');
      note.id = `tariff-note-${data.code}`;
      details.append(note);
      const wrap = el('div', 'table-wrap canadian-tariff-scroll');
      wrap.tabIndex = 0;
      wrap.setAttribute('role', 'region');
      wrap.setAttribute('aria-label', `Canadian tariff table for HS6 ${data.display_code}; scroll horizontally for all columns`);
      const table = el('table', 'canadian-tariff-table');
      table.setAttribute('aria-describedby', note.id);
      table.append(el('caption', '', `Canadian source rows beneath HS6 ${data.display_code}`));
      const columns = [['Tariff Item', 'tariff_item'], ['SS', 'statistical_suffix'],
        ['Description', 'description'], ['Unit', 'unit'], ['MFN Tariff', 'mfn_tariff'],
        ['Applicable Preferential Tariffs', 'preferential_tariffs']];
      const head = el('thead'), headerRow = el('tr');
      for (const [label] of columns) {
        const th = el('th', '', label); th.scope = 'col'; headerRow.append(th);
      }
      head.append(headerRow); table.append(head);
      const body = el('tbody');
      for (const line of lines) {
        const row = el('tr');
        for (const [, key] of columns) {
          const cell = el('td', '', line[key] || '—');
          if (!line[key]) cell.setAttribute('aria-label', 'Blank in source');
          row.append(cell);
        }
        body.append(row);
      }
      table.append(body); wrap.append(table); details.append(wrap);
    });
    return details;
  }
  function makeNode(data, level, parent, section) {
    const root = el('div', `hs-node level-${level}`);
    root.dataset.code = data.code;
    const button = el('button', 'node-toggle');
    button.type = 'button';
    const panel = el('div', 'node-panel');
    panel.id = `panel-${data.code}`; panel.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    button.setAttribute('aria-controls', panel.id);
    const arrow = el('span', 'chevron', '▸'); arrow.setAttribute('aria-hidden', 'true');
    button.append(arrow, el('span', 'code-badge', `HS${level} ${level === 4 ? data.code : data.display_code}`),
      el('span', 'node-label', data.description + (data.source_warning ? ' · Source needs review' : '')));
    if (level === 6) {
      const badge = el('span', 'section338-badge', 'U.S. Section 338 HS6 match');
      badge.hidden = true;
      button.append(badge);
    }
    const record = {root, button, panel, data, parent, section, children: [], level};
    button.addEventListener('click', () => setOpen(record, button.getAttribute('aria-expanded') !== 'true'));
    const detail = el('div', 'node-detail');
    if (level === 2) {
      detail.append(el('p', '', data.reserved ? 'Reserved for possible future use in the Harmonized System. No headings or subheadings.' : `${data.headings.length} headings · Chapter schedule effective ${data.effective_date}.`));
    } else if (level === 4) {
      detail.append(el('p', '', `Heading ${data.display_code} · ${data.subheadings.length} six-digit prefixes.`));
      if (data.extraction === 'inferred_from_child_row') detail.append(el('p', '', `This heading is reconstructed from source row ${data.inferred_from}; the HTML omits a separate HS4 row.`));
    } else {
      detail.append(el('p', '', 'Full source description: ' + data.source_description));
      const list = el('dl');
      for (const [label, value] of [['Normalized code', data.code], ['Chapter → heading', `${data.chapter_code} → ${data.heading_code}`], ['Extraction', data.extraction === 'explicit_hs6_row' ? 'Explicit six-digit source row' : 'Inferred six-digit prefix of Canadian tariff item(s): ' + data.inferred_from.map(c => c.slice(0,4)+'.'+c.slice(4,6)+'.'+c.slice(6)).join(', ')]]) {
        list.append(el('dt', '', label), el('dd', '', value));
      }
      detail.append(list);
      if (data.source_warning) {
        detail.append(el('p', 'error', data.source_warning));
        const raw = el('details'); raw.append(el('summary', '', 'Inspect source row descriptions'));
        data.source_row_descriptions.forEach(text => raw.append(el('p', '', text)));
        detail.append(raw);
      }
    }
    if (level === 6) detail.append(canadianTariffDetails(data));
    panel.append(detail);
    for (const child of data.headings || data.subheadings || []) {
      const childRecord = makeNode(child, level + 2, record, section);
      record.children.push(childRecord); panel.append(childRecord.root);
    }
    state.nodes.push(record); root.append(button, panel); return record;
  }
  function updateMode() {
    const active = state.mode === 'section338';
    $('explorer').classList.toggle('section338-active', active);
    $('section338-notice').hidden = !active;
    for (const node of state.nodes) {
      // Children precede parents. Only HS6 participates in the bridge;
      // Canadian national digits are never tested against U.S. tariff lines.
      node.inMode = !active || (node.level === 6 ? state.lensCodes.has(node.data.code) :
        node.children.some(child => child.inMode));
      if (node.level === 6) node.button.querySelector('.section338-badge').hidden = !active || !node.inMode;
      if (node.level < 6) {
        const count = node.children.filter(child => child.inMode).length;
        node.panel.querySelector('.node-detail > p').textContent = active ?
          (node.level === 2 ? `${count} Canadian headings connected to U.S. Section 338 HS6 matches · Chapter schedule effective ${node.data.effective_date}.` :
            `Heading ${node.data.display_code} · ${count} U.S. Section 338 HS6 matches.`) :
          (node.level === 2 ? (node.data.reserved ? 'Reserved for possible future use in the Harmonized System. No headings or subheadings.' : `${node.data.headings.length} headings · Chapter schedule effective ${node.data.effective_date}.`) :
            `Heading ${node.data.display_code} · ${node.data.subheadings.length} six-digit prefixes.`);
      }
    }
    for (const section of state.sections) section.jumpLink.hidden = !section.chapters.some(node => node.inMode);
    const stats = state.data.statistics;
    const values = active ? [state.sections.filter(s => s.chapters.some(n => n.inMode)).length,
      ...[2, 4, 6].map(level => state.nodes.filter(n => n.level === level && n.inMode).length)] :
      [stats.sections, stats.hs2_chapters, stats.hs4_headings, stats.hs6_subheadings];
    const labels = active ? ['Connected HS sections', 'Connected HS2 chapters', 'Connected HS4 headings', 'Matched HS6 categories'] :
      ['HS sections', 'HS2 chapters, including reserved 77', 'HS4 headings', 'HS6 prefixes in the source HTML'];
    ['stat-sections', 'stat-hs2', 'stat-hs4', 'stat-hs6'].forEach((id, i) => {
      $(id).textContent = values[i].toLocaleString();
      $(id).nextElementSibling.textContent = labels[i];
    });
    $('data-note').textContent = active ? state.lensSummary : state.normalDataNote;
    search();
  }
  function search() {
    const query = $('hs-search').value.trim().toLocaleLowerCase();
    const isCode = /^[\d.\s-]+$/.test(query);
    const digits = query.replace(/\D/g, '');
    const terms = query.split(/\s+/);
    let matches = 0;
    // Children precede parents in state.nodes, allowing one bottom-up pass.
    for (const node of state.nodes) {
      node.directMatch = node.inMode && Boolean(query) && (isCode && digits ? node.data.code.startsWith(digits) :
        terms.every(term => (node.data.description + ' ' + (node.data.source_description || '')).toLocaleLowerCase().includes(term)));
      if (node.directMatch) matches++;
      node.hasMatch = node.directMatch || node.children.some(child => child.hasMatch);
    }
    // A direct parent match exposes its descendants as context. Direct-match
    // count excludes those extra context rows and the containing section titles.
    for (const node of [...state.nodes].reverse()) {
      node.context = Boolean(node.parent && (node.parent.directMatch || node.parent.context));
      node.root.hidden = !node.inMode || (Boolean(query) && !(node.hasMatch || node.context));
      node.root.classList.toggle('match', node.directMatch);
      setOpen(node, Boolean(query) && !node.root.hidden);
    }
    for (const section of state.sections) section.root.hidden = !section.chapters.some(c => !c.root.hidden);
    $('no-results').hidden = !query || matches > 0;
    $('result-count').textContent = query ? `${matches.toLocaleString()} matching classifications · ancestors shown for context` : `${state.nodes.filter(n => n.inMode).length.toLocaleString()} classifications · all collapsed`;
  }
  function render(data) {
    state.data = data;
    const stats = data.statistics;
    for (const [id, value] of [['stat-sections', stats.sections], ['stat-hs2', stats.hs2_chapters], ['stat-hs4', stats.hs4_headings], ['stat-hs6', stats.hs6_subheadings]]) $(id).textContent = value.toLocaleString();
    $('snapshot-meta').textContent = `${data.source.edition} · Effective ${data.source.chapter_effective_dates.join(', ')} · Source modified ${data.source.date_modified}.`;
    $('data-note').textContent = `${stats.active_hs2_chapters} active chapters + reserved Chapter 77. ${stats.inferred_hs6.toLocaleString()} HS6 prefixes inferred from Canadian tariff rows. ${stats.validation_warnings} source warnings; see collection method below. Chapters 98–99 are separate.`;
    state.normalDataNote = $('data-note').textContent;
    const fragment = document.createDocumentFragment();
    for (const section of data.sections) {
      const root = el('section', 'hs-section'); root.id = `section-${section.code}`;
      const heading = el('header', 'section-heading');
      heading.append(el('span', '', `SECTION ${section.code}`), el('h3', '', section.description));
      root.append(heading);
      const record = {root, chapters: []};
      for (const chapter of section.chapters) {
        const node = makeNode(chapter, 2, null, record); record.chapters.push(node); root.append(node.root);
      }
      state.sections.push(record); fragment.append(root);
      const li = el('li'); li.append(link(`#${root.id}`, `${section.code} — ${section.description}`)); $('section-links').append(li);
      record.jumpLink = li;
    }
    $('explorer-tree').replaceChildren(fragment);
    for (const chapter of data.special_chapters) {
      const card = el('article'); card.append(el('h3', '', `${chapter.code} — ${chapter.description}`),
        el('p', 'small', 'Canadian national provisions. Excluded from the international HS2 → HS4 → HS6 explorer. Official schedules are linked in References & Sources.'));
      $('special-chapters').append(card);
      const li = el('li'); li.append(link(chapter.source_url, `CBSA Chapter ${chapter.code}`), document.createTextNode(' · '), link(chapter.pdf_url, 'PDF'));
      $('national-source-links').append(li);
    }
    const example = data.chapter6_example;
    if (example) {
      const row = el('tr');
      for (const key of ['tariff_item', 'ss', 'description', 'unit', 'mfn', 'preferences']) row.append(el('td', '', example[key]));
      $('tariff-example').replaceChildren(row);
    }
    $('build-summary').textContent = `${stats.html_chapter_pages_parsed} chapter HTML pages parsed; ${stats.canadian_tariff_lines.toLocaleString()} Canadian tariff rows captured; ${stats.pdf_links_discovered} PDF links discovered; ${stats.validation_errors} validation errors; ${stats.validation_warnings} source warnings. Chapter 6 regression: ${data.validation.chapter6_regression}. Chapter 15 regression: ${data.validation.chapter15_regression}. Multiple-row regression: ${data.validation.multiple_tariff_rows_regression}.`;
    $('warning-list').replaceChildren(...data.validation.warnings.map(w => el('li', '', w)));
    document.querySelectorAll('.explorer-control').forEach(control => { control.disabled = false; });
    $('hs-search').addEventListener('input', search);
    $('clear-search').addEventListener('click', () => { $('hs-search').value = ''; search(); $('hs-search').focus(); });
    $('expand-all').addEventListener('click', () => { state.nodes.filter(n => !n.root.hidden).forEach(n => setOpen(n, true)); $('result-count').textContent = $('hs-search').value.trim() ? $('result-count').textContent.replace(/ · (expanded|collapsed)/g, '') + ' · expanded' : `${state.nodes.filter(n => n.inMode).length.toLocaleString()} classifications · all expanded`; });
    $('collapse-all').addEventListener('click', () => { state.nodes.forEach(n => setOpen(n, false)); $('result-count').textContent = $('hs-search').value.trim() ? $('result-count').textContent.replace(' · expanded', '').replace(' · collapsed', '') + ' · collapsed' : `${state.nodes.filter(n => n.inMode).length.toLocaleString()} classifications · all collapsed`; });
    document.querySelectorAll('input[name="exploration-mode"]').forEach(control => {
      control.addEventListener('change', () => { state.mode = control.value; updateMode(); });
    });
    $('loading').hidden = true; updateMode();
    // Failure of the optional lens does not disable the Canadian explorer.
    fetch(base + 'data/section338-hs6.json').then(response => {
      if (!response.ok) throw new Error(`Section 338 lookup request failed (${response.status})`);
      return response.json();
    }).then(lens => {
      if (!Array.isArray(lens.hs6) || !lens.hs6.length || lens.hs6.some(code => !/^[0-9]{6}$/.test(code))) throw new Error('Invalid Section 338 HS6 lookup');
      state.lens = lens;
      state.lensCodes = new Set(lens.hs6);
      const canadianCodes = new Set(state.nodes.filter(n => n.level === 6).map(n => n.data.code));
      const unmatched = [...state.lensCodes].filter(code => !canadianCodes.has(code));
      const matched = state.lensCodes.size - unmatched.length;
      state.lensSummary = `${lens.source_tariff_code_count.toLocaleString()} unique U.S. HTS8 product codes → ${state.lensCodes.size.toLocaleString()} unique HS6 prefixes → ${matched.toLocaleString()} matched in Canada; ${unmatched.length} unmatched${unmatched.length ? ': ' + unmatched.join(', ') : ''}.`;
      $('section338-build-summary').textContent = state.lensSummary;
      $('mode-section338').disabled = false;
      if (unmatched.length) console.warn('Unmatched Section 338 HS6 values:', unmatched);
    }).catch(error => {
      $('section338-build-summary').textContent = 'The optional Section 338 lookup could not be loaded. ' + error.message;
      $('lens-load-status').hidden = false;
      $('lens-load-status').textContent = 'U.S. Section 338 lens unavailable; the Canadian hierarchy remains available.';
      console.error(error);
    });
  }
  fetch(base + 'data/hs-t2026-2.json').then(response => {
    if (!response.ok) throw new Error(`Dataset request failed (${response.status})`);
    return response.json();
  }).then(render).catch(error => {
    $('loading').className = 'status error';
    $('loading').textContent = 'The local dataset could not be loaded. Serve this repository over HTTP and check that hs-t2026-2.json is present. ' + error.message;
    console.error(error);
  });
})();
