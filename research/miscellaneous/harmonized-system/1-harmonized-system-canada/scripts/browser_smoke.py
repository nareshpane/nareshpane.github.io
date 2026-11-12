#!/usr/bin/env python3
"""Optional browser smoke test. Requires Playwright, Edge/Chromium and localhost:8000.

No runtime dependency is added to the website. Screenshots go to the OS temp folder.
"""
import json
from pathlib import Path
import tempfile
import sys
from playwright.sync_api import sync_playwright

BASE = 'http://localhost:8000/research/harmonized-system/'
PAGES = ['harmonized-system-canada.html']
shots = Path(tempfile.gettempdir()) / 'hs-review'; shots.mkdir(exist_ok=True)
asset_base = Path(__file__).resolve().parent.parent
canadian = json.loads((asset_base / 'data/hs-t2026-2.json').read_text(encoding='utf-8'))
lens = json.loads((asset_base / 'data/section338-hs6.json').read_text(encoding='utf-8'))
hs6_codes = set(lens['hs6'])
expected_hs4 = {code[:4] for code in hs6_codes}
expected_hs2 = {code[:2] for code in hs6_codes}
expected_sections = {s['code'] for s in canadian['sections'] if any(c['code'] in expected_hs2 for c in s['chapters'])}
excluded_code = next(n['code'] for s in canadian['sections'] for c in s['chapters']
                     for h in c['headings'] for n in h['subheadings'] if n['code'] not in hs6_codes)
errors, console_errors, external_requests = [], [], []
results = {}
with sync_playwright() as p:
    browser = p.chromium.launch(channel='msedge' if sys.platform == 'win32' else None, headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    page.route('**/favicon.ico', lambda route: route.fulfill(status=204))
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('console', lambda message: console_errors.append(message.text) if message.type == 'error' else None)
    page.on('request', lambda request: external_requests.append(request.url) if not request.url.startswith('http://localhost:8000') else None)
    for file in PAGES:
        response = page.goto(BASE + file)
        assert response.status == 200, file
        assert page.locator('h1').count() == 1
        nav = page.locator('nav[aria-label="Collection navigation"]').first
        assert nav.locator('a').count() == 6
        for anchor in nav.locator('a').all():
            response = page.request.get(anchor.evaluate('(a) => a.href'))
            assert response.status == 200
        assert nav.locator('a[aria-current="page"]').get_attribute('href') == file
    page.goto(BASE + PAGES[0])
    page.wait_for_selector('#hs-search:enabled')
    assert page.locator('.hs-section').count() == 21
    assert page.locator('.hs-node.level-2').count() == 97
    assert page.locator('.hs-node.level-4').count() == 1228
    assert page.locator('.hs-node.level-6').count() == 5614
    assert page.locator('.node-toggle[aria-expanded="true"]').count() == 0
    assert page.locator('[data-code="98"], [data-code="99"]').count() == 0
    assert page.locator('.canadian-tariff-details').count() == 5614
    assert page.locator('.canadian-tariff-details[open]').count() == 0
    assert page.locator('.canadian-tariff-table').count() == 0  # render only on opening
    # Real button keyboard activation at all three hierarchy levels.
    for code, key in [('06', 'Enter'), ('0601', 'Space'), ('060110', 'Enter')]:
        b = page.locator(f'[data-code="{code}"] > button')
        b.focus(); page.keyboard.press(key)
        assert b.get_attribute('aria-expanded') == 'true'
        assert page.locator('#' + b.get_attribute('aria-controls')).is_visible()
    for query in ['060110', '0601.10', '06.01', 'bulbs', 'petroleum']:
        page.locator('#hs-search').fill(query)
        assert not page.locator('#no-results').is_visible()
        results[query] = page.locator('#result-count').inner_text()
        if query in ['060110', '0601.10']:
            assert page.locator('.match').count() == 1
            for code in ['06', '0601', '060110']:
                assert page.locator(f'[data-code="{code}"] > button').get_attribute('aria-expanded') == 'true'
                assert page.locator(f'[data-code="{code}"]').is_visible()
            assert page.locator('#section-II').is_visible()
        if query == '06.01':
            assert page.locator('[data-code="0601"]').is_visible()
            assert page.locator('[data-code="0602"]').is_hidden()
    page.locator('#hs-search').fill('no-such-commodity-xyz')
    assert page.locator('#no-results').is_visible()
    page.locator('#clear-search').click()
    assert page.locator('#hs-search').input_value() == ''
    assert page.locator('.node-toggle[aria-expanded="true"]').count() == 0
    page.locator('#expand-all').click()
    assert page.locator('.node-toggle[aria-expanded="true"]').count() == 6939
    assert page.locator('.canadian-tariff-details[open]').count() == 0
    page.locator('#collapse-all').click()
    assert page.locator('.node-toggle[aria-expanded="true"]').count() == 0
    page.locator('#hs-search').fill('7104.20')
    assert page.locator('[data-code="710420"] .error').is_visible()
    page.locator('#clear-search').click()
    # National rows are a separate, initially collapsed control under HS6.
    page.locator('#hs-search').fill('1501.10')
    lard = page.locator('[data-code="150110"] .canadian-tariff-details')
    assert not lard.evaluate('(e) => e.open')
    lard.locator('summary').focus(); page.keyboard.press('Enter')
    lard.locator('tbody tr').first.wait_for()
    cells = lard.locator('tbody tr').first.locator('td').all_text_contents()
    assert cells[:2] == ['1501.10.00', '00'] and cells[3:5] == ['KGM', 'Free'], cells
    assert cells[5] == 'CCCT, LDCT, GPT, UST, MXT, CIAT, CT, CRT, IT, PT, COLT, JT, PAT, HNT, KRT, CEUT, UAT, CPTPT, UKT, CPUKT: Free'
    lard.locator('summary').press('Space')
    assert not lard.evaluate('(e) => e.open')
    page.locator('#hs-search').fill('0601.10')
    multiple = page.locator('[data-code="060110"] .canadian-tariff-details')
    multiple.locator('summary').click()
    multiple.locator('tbody tr').first.wait_for()
    assert multiple.locator('tbody tr').count() == 13
    assert set(multiple.locator('tbody tr td:nth-child(5)').all_text_contents()) == {'6%', 'Free'}
    assert multiple.locator('tbody tr td:nth-child(2)').all_text_contents()[:4] == ['00', '—', '10', '90']
    # Closing/reopening does not duplicate the lazily rendered rows.
    multiple.locator('summary').click(); multiple.locator('summary').click()
    assert multiple.locator('tbody tr').count() == 13
    page.locator('#clear-search').click()
    # The lens filters through HS6 only, with all Canadian descendants retained.
    page.wait_for_selector('#mode-section338:enabled')
    assert page.locator('#mode-all').is_checked()
    assert page.locator('#section338-notice').is_hidden()
    assert page.locator('#explorer-tree a').count() == 0
    assert page.locator('.source-panel').count() == 0
    assert page.locator('#explorer h2').inner_text() == 'Explore Canada’s HS Hierarchy'
    page.locator('#mode-section338').focus(); page.keyboard.press('Space')
    assert page.locator('#mode-section338').is_checked()
    assert page.locator('#section338-notice').is_visible()
    eligible = lambda level: page.locator(f'.level-{level}:not([hidden])')
    assert set(eligible(6).evaluate_all('(nodes) => nodes.map(n => n.dataset.code)')) == hs6_codes
    assert set(eligible(4).evaluate_all('(nodes) => nodes.map(n => n.dataset.code)')) == expected_hs4
    assert set(eligible(2).evaluate_all('(nodes) => nodes.map(n => n.dataset.code)')) == expected_hs2
    assert page.locator('.hs-section:not([hidden])').count() == len(expected_sections)
    for element_id, value in [('stat-sections',len(expected_sections)),('stat-hs2',len(expected_hs2)),('stat-hs4',len(expected_hs4)),('stat-hs6',len(hs6_codes))]:
        assert page.locator('#'+element_id).inner_text() == f'{value:,}'
    page.locator('#expand-all').click()
    assert page.locator('.section338-badge:visible').count() == len(hs6_codes)
    assert page.locator('.node-toggle[aria-expanded="true"]').count() == len(hs6_codes)+len(expected_hs4)+len(expected_hs2)
    page.locator('#collapse-all').click()
    assert page.locator('.node-toggle[aria-expanded="true"]').count() == 0
    for query in ['2203.00', '22.03', 'beer']:
        page.locator('#hs-search').fill(query)
        assert not page.locator('#no-results').is_visible(), query
        for code in ['22','2203','220300']:
            assert page.locator(f'[data-code="{code}"]').is_visible(), (query,code)
        assert set(eligible(6).evaluate_all('(nodes) => nodes.map(n => n.dataset.code)')) <= hs6_codes
    beer = page.locator('[data-code="220300"] .canadian-tariff-details')
    beer.locator('summary').press('Enter')
    beer.locator('tbody tr').first.wait_for()
    assert 'Canadian' in beer.locator('caption').inner_text()
    beer_data = next(n for s in canadian['sections'] for c in s['chapters'] for h in c['headings'] for n in h['subheadings'] if n['code']=='220300')
    assert beer.locator('tbody tr').count() == len(beer_data['canadian_tariff_lines'])
    page.locator('#hs-search').fill(excluded_code)
    assert page.locator('#no-results').is_visible()  # excluded Canadian branch
    page.locator('#mode-all').check()  # preserve query, restore its Canadian result
    assert page.locator(f'[data-code="{excluded_code}"]').is_visible()
    assert page.locator('#stat-hs6').inner_text() == '5,614'
    page.locator('#clear-search').click()
    assert eligible(6).count() == 5614
    print('Normal and Section 338 behavior checks passed.', flush=True)
    # Width checks include the tall explorer and horizontally scrollable example table.
    for width, height, label in [(1440,1000,'desktop'),(768,1024,'tablet'),(390,844,'mobile')]:
        page.set_viewport_size({'width':width, 'height':height})
        for file in PAGES:
            page.goto(BASE+file)
            page.wait_for_selector('#hs-search:enabled')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (file,width)
            page.screenshot(path=str(shots / f'{file[:-5]}-{label}.png'))
        page.goto(BASE+PAGES[0]); page.wait_for_selector('#hs-search:enabled')
        page.locator('#hs-search').fill('0601.10')
        page.evaluate("window.scrollTo(0, document.getElementById('section-II').getBoundingClientRect().top + scrollY - 190)")
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(shots / f'explorer-{label}.png'))
        multiple = page.locator('[data-code="060110"] .canadian-tariff-details')
        multiple.locator('summary').click()
        multiple.locator('tbody tr').first.wait_for()
        assert multiple.locator('tbody tr').count() == 13
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
        assert multiple.locator('td').first.evaluate('(e) => getComputedStyle(e).fontSize') == '16px'
        for selector in ['.node-detail', '.node-detail dt', '.node-detail dd']:
            assert page.locator(selector).first.evaluate('(e) => parseFloat(getComputedStyle(e).fontSize)') >= 15
        scroller = multiple.locator('.canadian-tariff-scroll')
        scroller.focus(); scroller.press('ArrowRight')
        scroller.evaluate('(e) => e.scrollLeft = e.scrollWidth')
        assert scroller.evaluate('(e) => e.scrollLeft > 0')
        scroller.evaluate('(e) => e.scrollLeft = 0')
        assert scroller.evaluate('(e) => e.scrollHeight > e.clientHeight')
        scroller.evaluate('(e) => { e.scrollTop = e.scrollHeight; }')
        assert scroller.evaluate('(e) => e.scrollTop > 0')
        scroller.evaluate('(e) => e.scrollTop = 0')
        multiple.evaluate('(e) => window.scrollTo(0, e.getBoundingClientRect().top + scrollY - 190)')
        page.screenshot(path=str(shots / f'canadian-tariffs-{label}.png'))
        page.wait_for_selector('#mode-section338:enabled')
        page.locator('#mode-section338').check()
        page.locator('#clear-search').click()
        page.locator('#explorer h2').evaluate("(e) => window.scrollTo({top:e.getBoundingClientRect().top + scrollY - 24,behavior:'instant'})")
        page.screenshot(path=str(shots / f'section338-selector-{label}.png'))
        page.locator('#hs-search').fill('2203.00')
        beer = page.locator('[data-code="220300"] .canadian-tariff-details')
        beer.locator('summary').click(); beer.locator('tbody tr').first.wait_for()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), ('lens',width)
        page.locator('[data-code="220300"]').evaluate("(e) => window.scrollTo({top:e.getBoundingClientRect().top + scrollY - 210,behavior:'instant'})")
        page.screenshot(path=str(shots / f'section338-{label}.png'))
        beer.scroll_into_view_if_needed()
        page.screenshot(path=str(shots / f'section338-tariffs-{label}.png'))
        page.locator('#references').evaluate("(e) => window.scrollTo({top:e.getBoundingClientRect().top + scrollY - 24,behavior:'instant'})")
        page.screenshot(path=str(shots / f'references-{label}.png'))
        print(f'{label}: both modes and Canadian tables fit at {width}px.', flush=True)
    # The established site typography loads Google Fonts; data remains local.
    assert all(url.startswith(('https://fonts.googleapis.com/', 'https://fonts.gstatic.com/')) for url in external_requests), external_requests
    collection_external_requests = list(external_requests)
    # This revision does not change or make assumptions about research-list order.
    assert not errors, errors
    assert not console_errors, console_errors
    # Missing optional data leaves normal exploration available.
    page.route('**/section338-hs6.json', lambda route: route.fulfill(status=404, body='Missing'))
    page.goto(BASE+PAGES[0]); page.wait_for_selector('#lens-load-status:visible')
    assert page.locator('#hs-search').is_enabled()
    assert page.locator('#mode-section338').is_disabled()
    page.locator('#hs-search').fill('060110')
    assert page.locator('[data-code="060110"]').is_visible()
    page.unroute('**/section338-hs6.json')
    # An unavailable Canadian JSON produces useful feedback.
    page.route('**/hs-t2026-2.json', lambda route: route.fulfill(status=404, body='Missing'))
    page.goto(BASE+PAGES[0])
    page.wait_for_selector('#loading.error')
    assert 'could not be loaded' in page.locator('#loading').inner_text()
    browser.close()
print(json.dumps({'searches':results,'widths':[1440,768,390], 'pages':len(PAGES),
                  'section338':{'source_codes':lens['source_tariff_code_count'],'hs6':len(hs6_codes),'matched':lens['matched_hs6_count'],'unmatched':lens['unmatched_hs6'], 'sections':len(expected_sections),'hs2':len(expected_hs2),'hs4':len(expected_hs4)},
                  'javascript_errors':errors,'console_errors_before_expected_failure':0,
                  'external_collection_requests':collection_external_requests,'screenshots':str(shots),
                  'result':'passed'},indent=2))
