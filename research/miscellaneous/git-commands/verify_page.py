#!/usr/bin/env python3
"""Static checks plus optional Chromium audit over HTTP.

python3 research/git-commands/verify_page.py
python3 research/git-commands/verify_page.py --browser http://127.0.0.1:8765
The browser mode needs Playwright installed outside the site. Screenshots go to /tmp.
"""
import argparse
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ASSETS = Path(__file__).resolve().parent
PAGE = ASSETS.parent / 'git-commands.html'
ROOT = ASSETS.parent.parent


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.refs = []
        self.stack = []
        self.errors = []
        self.void = set('area base br col embed hr img input link meta param source track wbr'.split())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        for key in ('src', 'href'):
            if key in attrs:
                self.refs.append(attrs[key])
        if tag not in self.void:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.void:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in self.void:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append((tag, self.stack[-5:]))
        else:
            self.stack.pop()


def static_checks():
    source = PAGE.read_text()
    doc = Document()
    doc.feed(source)
    assert not doc.errors and not doc.stack, (doc.errors, doc.stack)
    assert not [key for key, count in Counter(doc.ids).items() if count > 1]
    for ref in doc.refs:
        url = urlsplit(ref)
        if url.scheme or url.netloc:
            continue
        if url.path:
            assert (PAGE.parent / unquote(url.path)).exists(), ref
        elif url.fragment:
            assert url.fragment in doc.ids, ref
    assert 'math-behind-poker/' not in source
    for script in ASSETS.glob('*.js'):
        subprocess.run(['node', '--check', str(script)], check=True)
    current = (ROOT / 'research.html').read_bytes()
    original = subprocess.check_output(['git', 'show', 'HEAD:research.html'], cwd=ROOT)
    assert current == original, 'research.html must remain unchanged in this revision'
    for asset in ('cormorant-garamond-hero.ttf', 'Cormorant-Garamond-OFL.txt'):
        assert (ASSETS / asset).is_file(), asset
    print(f'PASS: HTML nesting, {len(doc.ids)} unique IDs, assets/anchors, JS syntax; research.html unchanged')


def browser_checks(base):
    from playwright.sync_api import sync_playwright
    errors = []
    failures = []
    out = Path('/tmp/git-guide-audit')
    out.mkdir(exist_ok=True)
    url = base.rstrip('/') + '/research/git-commands.html'
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1440, 'height': 1000})
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(error.stack))
        page.on('console', lambda msg: errors.append(msg.text) if msg.type == 'error' else None)
        page.on('requestfailed', lambda req: failures.append(req.url))
        page.on('response', lambda res: failures.append(f'{res.status} {res.url}') if res.status >= 400 else None)
        def slide(selector, value):
            page.locator(selector).evaluate('(el,v)=>{el.value=v;el.dispatchEvent(new Event("input",{bubbles:true}));}', value)

        # Fresh loads must run immediately, without scrolling to the movie.
        for reload_index in range(3):
            page.goto(url, wait_until='domcontentloaded')
            assert page.title() == 'The Enigma of Git Commands'
            assert page.locator('#film-play').is_disabled(), ('Autoplay did not start on landing', errors)
            assert float(page.locator('#film-stage').get_attribute('data-elapsed')) < 2
            assert page.evaluate('scrollY') == 0
            page.wait_for_timeout(150)
            assert float(page.locator('#film-stage').get_attribute('data-elapsed')) > .05
        page.locator('#film-pause').click()
        assert page.locator('#film-pause').is_disabled()
        slide('#film-position', 20.4)
        assert float(page.locator('#film-stage').get_attribute('data-elapsed')) == 20.4
        assert page.locator('#file-html').is_visible()
        assert float(page.locator('#capture-ghost').evaluate('(e)=>e.style.opacity')) > .5
        page.locator('#film-play').click()
        page.wait_for_timeout(250)
        slide('#film-position', 74.8)
        assert page.locator('#film-play').is_disabled(), 'Playing scrub paused playback'
        page.wait_for_timeout(250)
        assert float(page.locator('#film-stage').get_attribute('data-elapsed')) > 74.95
        page.locator('#film-pause').click()
        for speed in [.5, 2]:
            slide('#film-speed', speed)
            slide('#film-position', 10)
            page.locator('#film-play').click()
            page.wait_for_timeout(400)
            page.locator('#film-pause').click()
            delta = float(page.locator('#film-stage').get_attribute('data-elapsed')) - 10
            assert .3 * speed <= delta <= 1.4 * speed, (speed, delta)
        page.locator('#film-replay').click()
        assert float(page.locator('#film-stage').get_attribute('data-elapsed')) < 1
        page.locator('#film-pause').click()
        slide('#film-speed', 1)

        # Reconstruct the same complete world after both forward and reverse seeks.
        def fingerprint(t):
            slide('#film-position', t)
            return page.locator('#git-world').evaluate('(e)=>e.outerHTML') + page.locator('.cinema-terminal').inner_text()
        for t in [2.3, 20.4, 35.8, 44, 56, 68.3, 76.9, 83.5, 89]:
            first = fingerprint(t)
            fingerprint(89 if t < 45 else 0)
            assert first == fingerprint(t), f'Non-deterministic seek at {t}'
        slide('#film-position', 44)
        assert page.locator('#film-stage').get_attribute('data-head') == 'experiment'
        assert page.locator('#film-stage').get_attribute('data-main') == 'B'
        slide('#film-position', 76.9)
        assert page.locator('#film-stage').get_attribute('data-tracking') == 'N'
        assert page.locator('#film-stage').get_attribute('data-main') == 'M'
        slide('#film-position', 78.9)
        assert page.locator('#film-stage').get_attribute('data-main') == 'N'
        slide('#film-position', 83.5)
        assert page.locator('#developer').get_attribute('data-expression') == 'concerned'
        slide('#film-position', 85.8)
        assert page.locator('#developer').get_attribute('data-expression') == 'relieved'
        assert float(page.locator('#rescue-ref').evaluate('(e)=>e.style.opacity')) > .9
        slide('#film-position', 56)
        assert page.locator('#cinema-conflict').is_visible()
        slide('#film-position', 59)
        assert 'resolved' in page.locator('#cinema-conflict').get_attribute('class')

        for width in [1440, 1024, 768, 390]:
            page.set_viewport_size({'width': width, 'height': 1000})
            page.wait_for_timeout(100)
            page.evaluate('document.fonts.ready')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
            title = page.locator('h1')
            assert title.inner_text().replace('\n', ' ') == 'The Enigma of Git Commands'
            style = title.evaluate('(e)=>{const s=getComputedStyle(e);return [s.fontFamily,s.fontWeight,s.color,s.letterSpacing,s.fontSize]}')
            assert 'Astra Hero Cormorant' in style[0] and style[1] == '700', style
            assert style[2] == 'rgb(24, 51, 56)', style
            assert page.locator('h1 span').evaluate('(e)=>getComputedStyle(e).color') == 'rgb(40, 108, 112)'
            assert title.evaluate('(e)=>e.scrollWidth <= e.clientWidth'), width
            assert title.bounding_box()['height'] < (140 if width == 390 else 180)
            assert page.locator('#first-repository pre').first.evaluate('(e)=>parseFloat(getComputedStyle(e).fontSize)') >= (15.2 if width == 390 else 16)
            assert page.locator('.cinema-terminal pre').first.evaluate('(e)=>parseFloat(getComputedStyle(e).fontSize)') >= 15.2
            for t in [0, 7, 14, 20.4, 28, 36, 44, 51, 53, 56, 59, 62, 68, 76.9, 83.5, 90]:
                slide('#film-position', t)
                assert page.locator('#git-world').is_visible()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, t)
            for t, name in [(20.4,'stage'), (56,'conflict'), (62,'merge'), (68,'push'), (90,'final')]:
                slide('#film-position',t)
                page.locator('#git-film').screenshot(path=str(out / f'film-{name}-{width}.png'))
            page.evaluate('scrollTo(0,0)')
            page.screenshot(path=str(out / f'opening-{width}.png'))
        page.set_viewport_size({'width': 1440, 'height': 1000})
        for button in page.locator('[data-map]').all():
            button.click()
            assert button.get_attribute('aria-pressed') == 'true'
            assert page.locator('#map-description').inner_text()
        count = 0
        for category in page.locator('#command-category option').evaluate_all('(els)=>els.map(e=>e.value)'):
            page.locator('#command-category').select_option(category)
            for option in page.locator('#command-choice option').evaluate_all('(els)=>els.map(e=>e.value)'):
                page.locator('#command-choice').select_option(option)
                assert page.locator('#command-details .mechanics > div').count() == 6
                count += 1
        for action in ['edit', 'add', 'edit-again']:
            page.locator(f'[data-status-action="{action}"]').click()
        assert 'MM index.html' in page.locator('#status-output').inner_text()
        page.locator('[data-status-action="commit"]').click()
        assert ' M index.html' in page.locator('#status-output').inner_text()
        page.locator('[data-status-action="restore"]').click()
        assert '(clean)' in page.locator('#status-output').inner_text()
        page.locator('[data-status-action="create"]').click()
        assert '?? notes.txt' in page.locator('#status-output').inner_text()
        for action in ['add', 'unstage']:
            page.locator(f'[data-status-action="{action}"]').click()
        assert '?? notes.txt' in page.locator('#status-output').inner_text()
        page.locator('[data-status-action="reset"]').click()
        for action in ['branch', 'commit', 'switch', 'commit', 'merge']:
            page.locator(f'[data-graph-action="{action}"]').click()
        assert 'Three-way' in page.locator('#graph-description').inner_text()
        page.locator('[data-graph-action="reset"]').click()
        assert 'stays visible' in page.locator('#graph-description').inner_text()
        page.locator('[data-graph-action="restart"]').click()
        for action in ['branch', 'switch', 'commit', 'switch', 'merge']:
            page.locator(f'[data-graph-action="{action}"]').click()
        assert 'Fast-forward' in page.locator('#graph-description').inner_text()
        # Repeated reset/commit must not put retained nodes on top of each other.
        for action in ['reset', 'commit', 'reset', 'commit']:
            page.locator(f'[data-graph-action="{action}"]').click()
        positions = page.locator('#graph-canvas circle').evaluate_all(
            '(els)=>els.map(e=>[+e.getAttribute("cx"),+e.getAttribute("cy")])')
        assert len(positions) == len(set(map(tuple, positions)))
        for width in [1440, 390]:
            page.set_viewport_size({'width': width, 'height': 1000})
            page.wait_for_timeout(100)
            for section in ['relationship-map', 'status-lab', 'graph-lab', 'synthesis']:
                page.locator('#' + section).screenshot(path=str(out / f'{section}-{width}.png'))
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.set_viewport_size({'width': 1440, 'height': 1000})

        for choice in ['merge', 'rebase', 'before']:
            page.locator(f'[data-integrate="{choice}"]').click()
            assert page.locator(f'[data-integrate="{choice}"]').get_attribute('aria-pressed') == 'true'
        page.locator('#hash-edit').click()
        page.wait_for_function('document.querySelectorAll(".hash-step.active").length===5')
        page.locator('#hash-reset').click()
        assert page.locator('.hash-step.active').count() == 0
        page.locator('[data-synth="index"]').focus()
        assert 'proposed next snapshot' in page.locator('#synth-info').inner_text()
        page.locator('a[href="#places"]').first.click()
        assert page.url.endswith('#places')
        # Manual pause/scroll must not restart the one-shot autoplay.
        page.locator('#film-stage').scroll_into_view_if_needed()
        assert not page.locator('#film-play').is_disabled()
        slide('#film-position', 89.8)
        slide('#film-speed', 2)
        page.locator('#film-play').click()
        page.wait_for_timeout(300)
        assert page.locator('#film-time').inner_text() == '1:30 / 1:30'
        assert page.locator('#film-pause').is_disabled()
        page.locator('#film-replay').click()
        page.wait_for_function('document.querySelector("#film-stage").dataset.elapsed === "90.000"', timeout=55000)
        assert page.locator('#film-pause').is_disabled()
        assert 'THE COMPLETE SYSTEM' in page.locator('#cinema-chapter').inner_text().upper()
        assert not errors, errors
        assert not failures, failures
        context.close()
        reduced = browser.new_context(reduced_motion='reduce', viewport={'width':390,'height':844})
        page = reduced.new_page()
        page.goto(url, wait_until='networkidle')
        page.locator('#film-stage').scroll_into_view_if_needed()
        page.wait_for_timeout(150)
        assert float(page.locator('#film-stage').get_attribute('data-elapsed')) == 0
        assert page.locator('#film-pause').is_disabled()
        assert 'Reduced motion' in page.locator('#film-mode').inner_text()
        page.locator('#film-position').focus()
        page.keyboard.press('ArrowRight')
        assert float(page.locator('#film-position').input_value()) > 0
        page.locator('#hash-edit').click()
        assert page.locator('.hash-step.active').count() == 5
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        reduced.close()
        nojs = browser.new_context(java_script_enabled=False, viewport={'width':390,'height':844})
        page = nojs.new_page()
        page.goto(url, wait_until='networkidle')
        assert page.locator('#film-fallback').is_visible()
        assert not page.locator('#film-controls').is_visible()
        assert page.locator('#objects').is_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        nojs.close()
        browser.close()
        print(f'PASS: browser widths 1440/1024/768/390; repeated landing autoplay, deterministic seeking, typography and controls; {count} explorer forms; labs; keyboard; reduced motion; no-JS; no console/network errors')
        print(f'Screenshots: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', metavar='HTTP_ORIGIN')
    args = parser.parse_args()
    static_checks()
    if args.browser:
        browser_checks(args.browser)
