"""Static HTML, asset, canonical-code, JSON-provenance, and structural checks."""
from __future__ import annotations
from collections import Counter
from html.parser import HTMLParser
import ast
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
sys.dont_write_bytecode=True
from build_monograph import ROOT,PAGE,canonical


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids=[];self.refs=[];self.images=[];self.panels=[];self.panel=None
        self.code=False;self.pre=False;self.skip=0;self.words=[];self.sections=[];self.heading=None;self.headings=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        for key in ('src','href'):
            if a.get(key):self.refs.append(a[key])
        if tag=='img':self.images.append(a)
        if tag=='section':self.sections.append(a)
        if tag=='details' and 'code-panel' in a.get('class',''):
            self.panel=dict(source=a['data-source'],symbol=a.get('data-symbol',''),opened='open' in a,text='')
            self.panels.append(self.panel)
        if tag=='pre':self.pre=True
        if tag=='code' and self.pre:self.code=True
        if tag in ['pre','script','style']:self.skip+=1
        if re.fullmatch(r'h[1-6]',tag):self.heading=[int(tag[1]),'']
    def handle_endtag(self,tag):
        if tag=='code':self.code=False
        if tag=='pre':self.pre=False
        if tag=='details':self.panel=None
        if tag in ['pre','script','style']:self.skip-=1
        if self.heading and tag=='h'+str(self.heading[0]):
            self.headings.append(self.heading);self.heading=None
    def handle_data(self,data):
        if self.code and self.panel:self.panel['text']+=data
        if not self.skip:self.words.extend(re.findall(r'\b[\w]+\b',data))
        if self.heading:self.heading[1]+=data


def validate():
    page=Page();page.feed(PAGE.read_text(encoding='utf-8'))
    errors=[]
    duplicates=[name for name,count in Counter(page.ids).items() if count>1]
    if duplicates:errors.append('Duplicate IDs: '+str(duplicates))
    for ref in page.refs:
        if ref.startswith('#'):
            if ref[1:] not in page.ids:errors.append('Missing anchor '+ref)
        elif not re.match(r'^(https?:|mailto:|data:)',ref):
            path=PAGE.parent/ref.split('#')[0].split('?')[0]
            if not path.exists():errors.append('Missing asset '+ref)
    for image in page.images:
        if not image.get('alt'):errors.append('Image without meaningful alt: '+str(image))
        if not image.get('width') or not image.get('height'):errors.append('Image missing reserved dimensions')
    for panel in page.panels:
        expected=canonical(panel['source'],panel['symbol'])
        if expected!=panel['text']:errors.append('Canonical source mismatch '+panel['source']+':'+panel['symbol'])
        if panel['source'].endswith('.py'):
            ast.parse(expected)
    opened=[p['source'] for p in page.panels if p['opened']]
    if opened!=['explore_api.py','comparison_tensorflow_gradient.py','comparison_pytorch_gradient.py','comparison_tensorflow_model.py','comparison_pytorch_model.py','comparison_keras_backend.py']:errors.append('Incorrect default expanded code panels '+str(opened))
    structure=json.loads((ROOT/'data'/'structure.json').read_text())
    expected=[entry['id'] for chapter in structure for entry in chapter['subsections']]
    actual=[section['id'] for section in page.sections if 'subsection' in section.get('class','')]
    if expected!=actual:errors.append('Structure specification and article subsections differ')
    if len(structure)!=7:errors.append('Expected seven chapters')
    previous=1
    for level,title in page.headings:
        if level>previous+1:errors.append('Skipped heading level: '+title)
        previous=level
    for path in (ROOT/'images').glob('*.svg'):
        root=ET.parse(path).getroot()
        if 'viewBox' not in root.attrib:errors.append('Missing SVG viewBox '+path.name)
        if not root.find('{http://www.w3.org/2000/svg}title') is not None:errors.append('Missing SVG title '+path.name)
    # Browser exports are exact selections from canonical files, with explicit thinning.
    for name,keys in [('survival',['profiles','timelines','gradient_max_error']),
                       ('financial-network',['gallery','aggregation_max_error']),
                       ('river',['observations','truth','gradient']),('allocation',['policies','evaluation'])]:
        data=json.loads((ROOT/'data'/f'{name}-demo.json').read_text())
        source=json.loads((ROOT/'outputs'/'seed-42'/f'{name}.json').read_text())
        for key in keys:
            if data[key]!=source[key]:errors.append('Export drift '+name+':'+key)
    manifest=json.loads((ROOT/'data'/'generation-manifest.json').read_text())
    for filename,expected_hash in manifest['asset_sha256'].items():
        path=ROOT/filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash:
            errors.append('Manifest hash drift: '+filename)
    report=dict(status='passed' if not errors else 'failed',errors=errors,chapters=len(structure),
                subsections=len(actual),chapter_subsection_counts=[len(x['subsections']) for x in structure],
                ids=len(page.ids),code_panels=len(page.panels),canonical_open_by_default=opened,
                images=len(page.images),equations=PAGE.read_text(encoding='utf-8').count('class="eq-number"'),
                prose_words_including_navigation=len(page.words),html_bytes=PAGE.stat().st_size)
    (ROOT/'data'/'page-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    assert not errors, errors
    return report


if __name__=='__main__':validate()
