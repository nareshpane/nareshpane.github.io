"""Embed canonical Python in the monograph; preserve author-written HTML outside markers.

Run after changing a program: python build_monograph.py
Markers remain in the generated HTML, so rebuilding is deterministic/idempotent.
Python's tokenizer adds presentation spans; textContent remains exact source text.
"""
from __future__ import annotations
import ast
import html
import io
import keyword
from pathlib import Path
import re
import tokenize
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT.parent / (ROOT.name + '.html')


def canonical(filename, symbol=None):
    source = (ROOT / 'python' / filename).read_text(encoding='utf-8')
    example = bool(symbol and symbol.startswith('example_'))
    if example:
        symbol = symbol.removeprefix('example_')
    if symbol:
        node = next((n for n in ast.walk(ast.parse(source))
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == symbol), None)
        if node is None:
            raise ValueError(f'{filename}: no symbol {symbol}')
        source = ''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    if example:
        source = ('import numpy as np\n\n\n' + source.rstrip() +
                  '\n\n\nif __name__ == "__main__":\n' +
                  f'    print({symbol}()["final_parameters"])\n')
    return source


def highlighted(source):
    # Token positions are line/column pairs; copy untouched gaps to preserve spaces.
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    def position(p):
        return offsets[min(p[0] - 1, len(offsets) - 1)] + p[1]
    result, cursor = [], 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        style = {tokenize.COMMENT: 'comment', tokenize.STRING: 'string', tokenize.NUMBER: 'number'}.get(token.type)
        if token.type == tokenize.NAME and keyword.iskeyword(token.string):
            style = 'keyword'
        if not style:
            continue
        start, end = position(token.start), position(token.end)
        result.append(html.escape(source[cursor:start]))
        result.append(f'<span class="code-{style}">{html.escape(source[start:end])}</span>')
        cursor = end
    result.append(html.escape(source[cursor:]))
    return ''.join(result)


def panel(filename, symbol, title):
    full = symbol is None
    source = canonical(filename, symbol)
    path = f'{ROOT.name}/python/{filename}'
    note = ('Complete standalone program. Save the copied text as ' + filename +
            '; install the documented dependencies before running.') if full else (
            'Complete short example, including imports and an executable entry point. '
            'Extracted from the canonical comparison source.') if symbol.startswith('example_') else (
            'Canonical function excerpt, not a standalone program. ' +
            'The complete source below supplies imports and supporting functions.')
    return (f'<details class="code-panel" data-source="{filename}" data-symbol="{symbol or ""}">'
            '<summary><svg class="code-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
            '<path d="m8 6-6 6 6 6m8-12 6 6-6 6M14 3l-4 18"/></svg>'
            f'<span class="code-title">{html.escape(title)}</span><span class="language">Python</span>'
            '<span class="code-action"><span class="code-expand-label">Expand Code</span>'
            '<span class="code-collapse-label">Collapse Code</span>'
            '<svg class="code-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4"/></svg></span></summary>' 
            f'<p class="code-note">{note}</p><div class="code-toolbar">'
            f'<button class="copy-code">{"Copy Complete Python Program" if full else "Copy Code"}</button>'
            '<button class="expand-code" aria-expanded="false">Expand code fully</button>'
            f'<a class="download" href="{path}" download>Download {filename}</a></div>'
            f'<pre><code>{highlighted(source)}</code></pre></details>')


def build():
    text = PAGE.read_text(encoding='utf-8')
    section_ids = re.findall(r'<section class="subsection" id="([^"]+)"', text)
    # Existing generated panels always begin with details immediately after the
    # marker. A fresh marker must never consume prose or a later source marker.
    pattern = re.compile(r'<!-- (CODE|FULL):([^>]+) -->(?:\n<details class="code-panel"(?:(?!<!-- (?:CODE|FULL):).)*?<!-- END-EMBED -->)?', re.S)
    def replace(match):
        kind, spec = match.group(1, 2)
        if kind == 'CODE':
            filename, symbol, title = spec.split(':', 2)
        else:
            filename, symbol, title = spec, None, 'Complete program · ' + spec
        return match.group(0).split('-->')[0] + '-->\n' + panel(filename, symbol, title) + '\n<!-- END-EMBED -->'
    text, count = pattern.subn(replace, text)
    def inline_svg(match):
        name = match.group(1)
        source = (ROOT / 'images' / name).read_text(encoding='utf-8')
        source = re.sub(r'<\?xml[^>]*>\s*', '', source)
        return f'<!-- SVG:{name} -->\n{source.strip()}\n<!-- END-SVG -->'
    text = re.sub(r'<!-- SVG:([^>]+) -->(?:\n<svg\b(?:(?!<!-- SVG:).)*?<!-- END-SVG -->)?', inline_svg, text, flags=re.S)
    def svg_size(match):
        tag = match.group(0)
        source = re.search(r'src="([^"]+\.svg)"', tag)
        if not source:
            return tag
        width, height = ET.parse(PAGE.parent / source.group(1)).getroot().attrib['viewBox'].split()[2:]
        tag = re.sub(r' (?:width|height)="[^"]*"', '', tag)
        return tag[:-1] + f' width="{width}" height="{height}">'
    text = re.sub(r'<img\b[^>]+>', svg_size, text)
    if re.findall(r'<section class="subsection" id="([^"]+)"', text) != section_ids:
        raise ValueError('Embedding changed article sections; refusing to write')
    PAGE.write_text(text, encoding='utf-8', newline='\n')
    print(f'Embedded {count} canonical source panels into {PAGE.name}')


if __name__ == '__main__':
    build()
