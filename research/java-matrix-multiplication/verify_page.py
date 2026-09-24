"""Optional browser verification; requires Playwright outside the static site.

Usage: python3 verify_page.py http://127.0.0.1:8001
Screenshots are saved in /tmp/matrix-redesign-review.
"""
import json
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('/tmp/matrix-redesign-review')
OUT.mkdir(exist_ok=True)
BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8001'
URL = BASE + '/research/java-matrix-multiplication.html'

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
    errors, failed = [], []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('response', lambda r: failed.append(r.url) if r.status >= 400 else None)
    page.goto(URL, wait_until='networkidle')
    page.wait_for_function("document.querySelector('#theatreStatus').textContent.startsWith('Step 0/')")
    page.evaluate("document.fonts.ready")
    page.screenshot(path=str(OUT / 'desktop-opening.png'))
    ids = page.locator('[id]').evaluate_all('(nodes) => nodes.map(n => n.id)')
    assert len(ids) == len(set(ids)), 'duplicate ids'
    for href in page.locator('[href],[src],[poster]').evaluate_all("nodes => nodes.flatMap(n => ['href','src','poster'].map(a => n.getAttribute(a)).filter(Boolean))"):
        if href.startswith('#'):
            assert href[1:] in ids, href
        elif not urlparse(href).scheme:
            path = unquote(urlparse(urljoin(URL, href)).path).lstrip('/')
            assert (ROOT / path).is_file(), path
    assert page.locator('mjx-merror').count() == 0, 'MathJax error'
    assert page.locator('mjx-container').count() > 15, 'formulas did not render'
    index_page = browser.new_page()
    index_page.goto(BASE + '/research.html', wait_until='domcontentloaded')
    entry = index_page.locator('ul.research-list > li').first
    assert entry.locator('a.project-title').get_attribute('href') == 'research/java-matrix-multiplication.html'
    assert 'Inside Matrix Multiplication' in entry.inner_text()
    index_page.close()

    def button(scene, action):
        return page.locator('[data-player="' + scene + '"]').get_by_role('button', name=scene+' · '+action, exact=True)

    def finish(scene):
        for _ in range(100):
            next_button = button(scene, 'Next step')
            if next_button.is_disabled():
                return
            next_button.evaluate('(b) => b.click()')
        raise AssertionError(scene + ' never finished')

    assert page.locator('#theatreC .empty').count() == 4
    for _ in range(7): button('theatre', 'Next step').evaluate('(b) => b.click()')
    page.locator('#theatre').screenshot(path=str(OUT / 'desktop-theatre.png'))
    finish('theatre')
    assert page.locator('#theatreC .matrix-cell').all_text_contents() == ['0','21','4','23']
    button('theatre','Previous step').click()
    assert page.locator('#theatreC .empty').count() == 1
    button('theatre','Reset').click()
    assert page.locator('#theatreC .empty').count() == 4
    page.locator('#theatreSpeed').fill('3')
    button('theatre','Play').click()
    page.wait_for_function("!document.querySelector('#theatreStatus').textContent.startsWith('Step 0/')", timeout=5000)
    button('theatre','Pause').click()
    paused = page.locator('#theatreStatus').inner_text()
    page.wait_for_timeout(550)
    assert page.locator('#theatreStatus').inner_text() == paused
    assert 'Step 0/' not in paused

    page.locator('#representationToggle').click()
    page.locator('#representationCells button').nth(4).click()
    assert 'A[1][1] = 0' in page.locator('#representationStatus').inner_text()
    finish('loops')
    assert page.locator('#loopC .matrix-cell').all_text_contents() == ['0','21','4','23']
    assert page.locator('#loopCounts').inner_text() == '12 products · 12 additions'
    page.locator('#classical').screenshot(path=str(OUT / 'desktop-loops.png'))
    page.locator('#geometrySteps [data-step="2"]').click()
    assert '(0, 1)' in page.locator('#geometryStatus').inner_text()
    page.locator('#geometryOrder [data-order="ba"]').click()
    assert '(−1, 2)' in page.locator('#geometryStatus').inner_text()
    page.locator('#geometrySteps [data-step="3"]').click()
    assert 'multiply(multiply(B, A), x)' == page.locator('#geometryCode').inner_text()
    page.wait_for_timeout(600)
    page.locator('#transformations').screenshot(path=str(OUT / 'desktop-geometry.png'))
    page.locator('#scaleSizes button').last.click()
    assert page.locator('#scaleProducts').inner_text() == '262,144'
    finish('scale')
    page.locator('#scale').screenshot(path=str(OUT / 'desktop-scale.png'))
    finish('tiles')
    expected = json.loads((ROOT/'research/java-matrix-multiplication/traces/blocked.json').read_text())['result']
    assert page.locator('#tileC span').all_text_contents() == [str(v) for row in expected for v in row]
    page.locator('#tiles').screenshot(path=str(OUT / 'desktop-tiles.png'))
    page.locator('#sparseToggle').click()
    assert page.locator('#sparseGrid.fade .zero').count() == 29
    finish('parallel')
    assert page.locator('#parallelC span').all_text_contents() == [str(v) for row in expected for v in row]
    page.locator('#parallel').screenshot(path=str(OUT / 'desktop-parallel.png'))
    finish('strassen')
    assert page.locator('#sevenProducts b').all_text_contents() == ['49','12','-15','-4','20','-2','-18']
    assert '343 leaves' in page.locator('#strassenStatus').inner_text()
    page.locator('.strassen-scene').screenshot(path=str(OUT / 'desktop-strassen.png'))
    page.locator('#precisionToggle').click()
    assert page.locator('#precisionValue').inner_text() == '0.6'
    page.locator('#tensor summary').click()
    page.locator('#tensor').screenshot(path=str(OUT / 'desktop-tensor.png'))
    finish('recap')
    page.wait_for_timeout(550)
    page.locator('#recap').screenshot(path=str(OUT / 'desktop-recap.png'))
    page.locator('.formula-details summary').click()
    page.locator('.media-details summary').click()
    page.locator('video').evaluate('(v) => v.load()')
    page.wait_for_function('document.querySelector("video").readyState >= 1')
    assert round(page.locator('video').evaluate('(v) => v.duration')) == 30
    assert page.locator('img').evaluate_all('(ns) => ns.every(n => n.complete && n.naturalWidth > 0)')

    for width in [768,390,360]:
        page.set_viewport_size({'width':width,'height':900})
        page.wait_for_timeout(100)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'horizontal overflow at '+str(width)
        for scene in ['theatre','representation','classical','scale','tiles','sparse','parallel','strassen','transformations','numerical','tensor','recap']:
            page.locator('#'+scene).screenshot(path=str(OUT/(str(width)+'-'+scene+'.png')))
        assert page.locator('.mini-tile span').first.evaluate('(n) => getComputedStyle(n).position') == 'static'
        assert page.locator('.mini-tile').first.evaluate('(n) => n.scrollWidth <= n.clientWidth + 1')
    # Every player can restart and advance through the shared control implementation.
    for name in ['theatre','loops','scale','tiles','parallel','strassen','recap']:
        button(name,'Reset').click()
        assert button(name,'Previous step').is_disabled()
        button(name,'Next step').click()
        assert not button(name,'Previous step').is_disabled()
        button(name,'Previous step').click()
        assert button(name,'Previous step').is_disabled()
    reduced = browser.new_page(viewport={'width':390,'height':844}, reduced_motion='reduce')
    reduced.goto(URL,wait_until='networkidle')
    reduced.wait_for_function("document.querySelector('#theatreStatus').textContent.startsWith('Step 0/')")
    reduced.locator('[data-player="theatre"] button').nth(3).click()
    assert reduced.evaluate('document.getAnimations().length') == 0
    reduced.locator('#representationToggle').click()
    assert reduced.evaluate('document.getAnimations().length') == 0
    reduced.locator('#tileLayout').click()
    assert reduced.evaluate('document.getAnimations().length') == 0
    reduced.screenshot(path=str(OUT/'reduced-motion.png'))
    assert not errors, errors
    assert not failed, failed
    print('PASS: browser controls, exact trace results, geometry, media, formulas, links/assets, research entry, no JS errors, 1440/768/390/360 layouts, reduced motion.')
    print('Screenshots:', OUT)
    browser.close()
