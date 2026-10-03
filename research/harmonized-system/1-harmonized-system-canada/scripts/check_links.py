#!/usr/bin/env python3
"""Check collection paths/fragments; --official additionally checks CBSA URLs.

External checks use paced, sequential HEAD requests. No PDFs are downloaded.
"""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import unquote, urlsplit
from bs4 import BeautifulSoup
from build_cbsa_hs import BASE, Fetcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official', action='store_true')
    args = parser.parse_args()
    root = BASE.parent
    errors, external = [], set()
    checked = 0
    for page in sorted(root.glob('*.html')):
        soup = BeautifulSoup(page.read_text(), 'lxml')
        ids = [tag['id'] for tag in soup.select('[id]')]
        if len(ids) != len(set(ids)):
            errors.append(f'{page.name}: duplicate element IDs')
        for tag in soup.select('[href], [src]'):
            url = tag.get('href', tag.get('src'))
            parts = urlsplit(url)
            if parts.scheme in {'http', 'https'}:
                external.add(url); continue
            target = (page.parent / unquote(parts.path)).resolve() if parts.path else page
            if not target.is_file():
                errors.append(f'{page.name}: missing {url}')
            elif parts.fragment and target.suffix == '.html':
                target_soup = soup if target == page else BeautifulSoup(target.read_text(), 'lxml')
                if not target_soup.find(id=unquote(parts.fragment)):
                    errors.append(f'{page.name}: missing fragment {url}')
            checked += 1
    main_page = BeautifulSoup((root.parents[1] / 'research.html').read_text(), 'lxml')
    entries = main_page.select('ul.research-list > li')
    expected = 'research/harmonized-system/harmonized-system-index.html'
    if entries[0].find('a')['href'] != expected:
        errors.append('Collection is not the first main research entry')
    data = json.loads((BASE/'data/hs-t2026-2.json').read_text())
    chapters = [c for s in data['sections'] for c in s['chapters']] + data['special_chapters']
    external.update(c['source_url'] for c in chapters)
    external.update(data['source']['pdf_links'])
    print(f'Local link/asset/fragment references checked: {checked}', flush=True)
    if args.official:
        session = Fetcher().session
        for index, url in enumerate(sorted(external), 1):
            if urlsplit(url).hostname != 'www.cbsa-asfc.gc.ca':
                errors.append(f'Unexpected external host: {url}'); continue
            time.sleep(0.15)
            try:
                response = session.head(url, allow_redirects=True, timeout=(15, 45))
                response.raise_for_status()
                print(f'Official link {index}/{len(external)}: {response.status_code} {url}', flush=True)
            except Exception as exc:
                errors.append(f'{url}: {exc}')
    print(f'Official URLs {"checked" if args.official else "discovered"}: {len(external)}')
    print(f'Link errors: {len(errors)}')
    for error in errors:
        print(error)
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
