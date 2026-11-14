"""One TeX source, existing MathJax SVG renderer, and precompiled native MathML fallback."""
from __future__ import annotations
import hashlib
import html
import json
import os
import unicodedata
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE_PATH=ROOT/'data/math-render-cache.json'
CACHE=json.loads(CACHE_PATH.read_text(encoding='utf-8'))['expressions'] if CACHE_PATH.exists() else {}
EXPRESSIONS={}


@lru_cache(maxsize=None)
def native_mathml(markup):
    """Map legacy mathvariant styles to MathML Core mathematical alphabet characters."""
    ET.register_namespace('','http://www.w3.org/1998/Math/MathML')
    root=ET.fromstring(markup)
    def styled(character,variant):
        style=variant.replace('double-struck','double_struck').replace('sans-serif','sans_serif').replace('-',' ').replace('_','-').upper()
        if character.isascii() and character.isalpha():
            suffix=('CAPITAL ' if character.isupper() else 'SMALL ')+character.upper()
        elif 'GREEK ' in unicodedata.name(character,''):
            suffix=unicodedata.name(character).replace('GREEK ','').replace('LETTER ','')
        elif character.isascii() and character.isdigit():
            suffix=unicodedata.name(character)
        else:return character
        for name in ['MATHEMATICAL '+style+' '+suffix,style+' '+suffix]:
            try:return unicodedata.lookup(name)
            except KeyError:pass
        return character
    def visit(node,inherited='normal'):
        variant=node.attrib.get('mathvariant',inherited)
        if node.tag.rsplit('}',1)[-1] in {'mi','mn'} and variant!='normal' and node.text:
            transformed=''.join(styled(c,variant) for c in node.text)
            if transformed!=node.text:
                node.text=transformed;node.attrib.pop('mathvariant',None)
        for child in node:visit(child,variant)
    visit(root)
    return ET.tostring(root,encoding='unicode')


def m(tex,display=False,scrollable=False):
    key=hashlib.sha256((('display:' if display else 'inline:')+tex).encode()).hexdigest()[:20]
    EXPRESSIONS[key]=dict(key=key,tex=tex,display=display)
    cached=CACHE.get(key)
    if cached is None and os.environ.get('MONOGRAPH_COLLECT_MATH')!='1':
        raise ValueError('Missing compiled mathematical fallback. Collect math sources, run js/render_math.mjs, then rebuild: '+tex)
    fallback=native_mathml(cached['mathml']) if cached else '<math xmlns="http://www.w3.org/1998/Math/MathML"><mtext>Compiling mathematical source</mtext></math>'
    opening,closing=('\\[','\\]') if display else ('\\(','\\)')
    cls='math-display' if display else 'math-inline'
    region=' tabindex="0" role="region" aria-label="Mathematical expression; scroll horizontally if needed"' if scrollable else ''
    return (f'<span class="math-fragment {cls}" data-math-key="{key}"{region}>'
            f'<span class="math-tex">{opening}{html.escape(tex)}{closing}</span>'
            f'<span class="math-fallback">{fallback}</span></span>')


def save_sources():
    (ROOT/'data/math-source.json').write_text(json.dumps(dict(engine='MathJax 3.2.2',expressions=sorted(EXPRESSIONS.values(),key=lambda row:row['key'])),indent=2)+'\n',encoding='utf-8')
