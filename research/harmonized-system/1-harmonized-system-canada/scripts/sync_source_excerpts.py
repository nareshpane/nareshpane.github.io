#!/usr/bin/env python3
"""Refresh HTML code excerpts from the actual builder; --check verifies equality."""
import argparse
from html import escape
from pathlib import Path

base = Path(__file__).resolve().parents[1]
source = (base / 'scripts/build_cbsa_hs.py').read_text()
page = base.parent / 'harmonized-system-canada.html'
parts = []
for key, title in [('discovery', 'Discover the source links'), ('parser', 'Reconstruct the HS hierarchy'), ('pdf', 'Download optional reference PDFs')]:
    excerpt = source.split(f'# excerpt:{key}:start\n', 1)[1].split(f'# excerpt:{key}:end', 1)[0].strip()
    parts.append(f'<h3>{title}</h3>\n<pre><code>{escape(excerpt)}</code></pre>')
start, end = '<!-- SOURCE_EXCERPTS_START -->', '<!-- SOURCE_EXCERPTS_END -->'
old = page.read_text()
updated = old.split(start)[0] + start + '\n' + '\n'.join(parts) + '\n' + end + old.split(end)[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check', action='store_true')
if parser.parse_args().check:
    if old != updated:
        raise SystemExit('Python excerpts are stale; run sync_source_excerpts.py')
    print('Visible Python excerpts match the actual builder.')
else:
    page.write_text(updated)
    print('Updated visible Python excerpts.')
