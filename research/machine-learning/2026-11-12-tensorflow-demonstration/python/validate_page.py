"""Static monograph checks addressing anchors, assets and copied Python validity.

No third-party dependencies. Browser visual/interaction QA is a separate check.
"""
from __future__ import annotations
import ast
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
from build_monograph import ROOT, PAGE, canonical


class Article(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.refs, self.images, self.headings = [], [], [], []
        self.panels, self.current, self.in_code, self.pre_depth = [], None, False, 0
        self.skip, self.words = 0, []
        self.heading = None
        self.stack = []

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if 'id' in attr:
            self.ids.append(attr['id'])
        for name in ('src', 'href'):
            if attr.get(name):
                self.refs.append(attr[name])
        if tag == 'img':
            self.images.append(attr)
        if tag in ('script', 'style', 'pre'):
            self.skip += 1
        if tag == 'details':
            self.stack.append(self.current)
            if 'code-panel' in attr.get('class', ''):
                self.current = {'source': attr.get('data-source'), 'symbol': attr.get('data-symbol'), 'text': ''}
                self.panels.append(self.current)
        if tag == 'pre':
            self.pre_depth += 1
        if tag == 'code' and self.pre_depth:
            self.in_code = True
        if re.fullmatch('h[1-6]', tag):
            self.heading = [int(tag[1]), '']

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'pre'):
            self.skip -= 1
        if tag == 'pre':
            self.pre_depth -= 1
        if tag == 'code':
            self.in_code = False
        if tag == 'details':
            self.current = self.stack.pop()
        if self.heading and tag == 'h' + str(self.heading[0]):
            self.headings.append(self.heading)
            self.heading = None

    def handle_data(self, data):
        if not self.skip:
            self.words.extend(re.findall(r"[\w]+(?:[’'−-][\w]+)*", data))
        if self.heading:
            self.heading[1] += data
        if self.in_code and self.current:
            self.current['text'] += data


def validate():
    text = PAGE.read_text(encoding='utf-8')
    article = Article()
    article.feed(text)
    errors = []
    duplicates = [name for name, count in Counter(article.ids).items() if count > 1]
    if duplicates:
        errors.append(f'Duplicate ids: {duplicates}')
    for ref in article.refs:
        if ref.startswith('#'):
            if ref[1:] not in article.ids:
                errors.append('Broken anchor: ' + ref)
        elif not re.match(r'\w+:|//', ref):
            target = PAGE.parent / ref.split('#')[0].split('?')[0]
            if not target.exists():
                errors.append('Missing relative asset: ' + ref)
    if len([h for h in article.headings if h[0] == 1]) != 1:
        errors.append('Expected one h1')
    previous = 0
    for level, title in article.headings:
        if level > previous + 1 and previous:
            errors.append('Heading level jump: ' + title)
        previous = level
    for image in article.images:
        if not image.get('alt'):
            errors.append('Missing image alt: ' + image.get('src', ''))
    for panel in article.panels:
        if panel['source']:
            expected = canonical(panel['source'], panel['symbol'] or None)
            if panel['text'] != expected:
                errors.append('Canonical copy mismatch: ' + panel['source'])
            ast.parse(panel['text'])
    for path in (ROOT / 'images').glob('*.svg'):
        tree = ET.parse(path)
        if not tree.getroot().get('viewBox'):
            errors.append('SVG missing viewBox: ' + path.name)
    for path in (ROOT / 'data').glob('*.json'):
        json.loads(path.read_text(encoding='utf-8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if '<!-- CHAPTERS_CONTINUE -->' in text:
        errors.append('Unfinished chapter marker')
    if len([p for p in article.panels if p['source'] and not p['symbol']]) != 6:
        errors.append('Expected six complete source panels')
    panel_count = text.count('<details class="code-panel"')
    for component in ('code-icon', 'code-expand-label', 'code-collapse-label', 'code-chevron'):
        if text.count(f'class="{component}"') != panel_count:
            errors.append('Missing shared code disclosure component: ' + component)
    if panel_count != 27:
        errors.append('Expected 26 canonical Python disclosures and one PowerShell disclosure')
    if '<!-- NEW-COMPARISON -->' in text:
        errors.append('Unfinished comparison marker')
    if 'class="masthead"' in text:
        errors.append('Removed masthead is still present')
    chapter_matches = list(re.finditer(r'<section class="chapter" id="([^"]+)"', text))
    expected_counts = {'foundations': 13, 'vision': 13, 'language': 13,
                       'forecasting': 14, 'physics': 15, 'synthesis': 7}
    for index, chapter in enumerate(chapter_matches):
        end = chapter_matches[index + 1].start() if index + 1 < len(chapter_matches) else len(text)
        count = len(re.findall(r'<section class="subsection"', text[chapter.start():end]))
        if count != expected_counts.get(chapter.group(1)):
            errors.append(f'Unexpected subsection count in {chapter.group(1)}: {count}')
    if re.search(r'<div class="equation(?: [^"]*)?">(?:(?!</div>).)*<p', text, re.S):
        errors.append('Explanatory paragraph is inside an equation panel')
    report = {'page': PAGE.name, 'prose_words_approximate': len(article.words),
              'headings': len(article.headings), 'anchors': len(article.ids),
              'canonical_code_panels': len([p for p in article.panels if p['source']]),
              'images': len(article.images), 'errors': errors,
              'checked': ['relative assets and anchors', 'unique ids and heading hierarchy',
                          'image alternatives and SVG view boxes', 'finite JSON syntax',
                          'full-program and fragment byte-for-text parity', 'embedded Python syntax'],
              'browser_qa': 'See qa-report.md; static checks do not establish visual or interactive behavior.'}
    (ROOT / 'data' / 'page-validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    validate()
