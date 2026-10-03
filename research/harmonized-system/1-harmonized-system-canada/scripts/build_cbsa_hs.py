#!/usr/bin/env python3
"""Build an HS2 → HS4 → HS6 research snapshot from CBSA HTML (never PDFs).

Run from any directory. HTML responses are cached under data/.cache; the master
is fetched on each build. --force refreshes chapter HTML and optional PDFs.
--validate-only checks the existing JSON without network access.
Nothing is published. A failed build never replaces the last validated JSON.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

MASTER_URL = 'https://www.cbsa-asfc.gc.ca/trade-commerce/tariff-tarif/2026/html/tblmod-2-eng.html'
BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / 'data/hs-t2026-2.json'
COLUMNS = ['Tariff Item', 'SS', 'Description of Goods', 'Unit of Meas.',
           'MFN Tariff', 'Applicable Preferential Tariffs']


def clean(value):
    return re.sub(r'\s+', ' ', value).strip()


class Fetcher:
    def __init__(self, force=False):
        self.force = force
        self.session = requests.Session()
        self.session.headers['User-Agent'] = (
            'NareshNeupane-HSResearch/1.0 (educational static dataset; '
            'https://nareshpane.github.io; sequential requests)')
        self.session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))

    def get(self, url, target=None, refresh=False, pdf=False):
        if target and target.exists() and not (self.force or refresh):
            return target.read_bytes(), False
        time.sleep(0.2)
        response = self.session.get(url, timeout=(15, 90))
        response.raise_for_status()
        payload = response.content
        if pdf and not payload.startswith(b'%PDF-'):
            raise ValueError(f'Expected a PDF: {url}')
        if not pdf and b'<html' not in payload.lower():
            raise ValueError(f'Expected HTML: {url}')
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.tmp')
            temporary.write_bytes(payload)
            temporary.replace(target)
        return payload, True

    def html(self, url, refresh=False):
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        payload, fetched = self.get(url, BASE / 'data/.cache' / f'{key}.html', refresh)
        return BeautifulSoup(payload, 'lxml'), fetched, hashlib.sha256(payload).hexdigest()


# excerpt:discovery:start

def discover(master, master_url):
    """Use section headers, chapter rows, and their actual HTML/PDF links."""
    sections, special = [], []
    for table in master.select('main table'):
        header = table.find_previous('h3')
        match = re.match(r'Section ([IVX]+):\s*(.*)', clean(header.get_text(' '))) if header else None
        if not match:
            continue
        section = {'code': match[1], 'description': match[2], 'chapters': []}
        for row in table.select('tr'):
            cells = row.find_all(['th', 'td'], recursive=False)
            if len(cells) != 3:
                continue
            label = BeautifulSoup(str(cells[0]), 'lxml')
            for extra in label.select('ol, ul'):
                extra.decompose()  # subchapter summaries are not chapter titles
            chapter_match = re.match(r'Chapter (\d+)\s*[-–]\s*(.*)', clean(label.get_text(' ')))
            if not chapter_match:
                continue
            code = chapter_match[1].zfill(2)
            links = {clean(a.get_text(' ')): urljoin(master_url, a['href'])
                     for a in cells[2].select('a[href]')}
            html_url = links.get('HTML')
            if not html_url:
                raise ValueError(f'No chapter HTML link for {code}')
            chapter = {'code': code, 'display_code': code,
                       'description': chapter_match[2], 'source_url': html_url,
                       'pdf_url': next((u for u in links.values() if urlparse(u).path.endswith('.pdf')), None),
                       'effective_date': clean(cells[1].get_text()),
                       'reserved': code == '77', 'section_code': section['code'],
                       'headings': []}
            if code in {'98', '99'}:
                chapter['section_code'] = None
                chapter['national_provision'] = True
                special.append(chapter)
            else:
                section['chapters'].append(chapter)
        if section['chapters']:
            sections.append(section)
    pdf_links = sorted({urljoin(master_url, a['href']) for a in master.select('main a[href]')
                        if urlparse(urljoin(master_url, a['href'])).path.lower().endswith('.pdf')})
    return sections, special, pdf_links
# excerpt:discovery:end


def tariff_rows(soup):
    """Select tables by their six semantic columns, then inspect cell values."""
    for table in soup.select('table'):
        rows = table.select('tr')
        if not rows:
            continue
        labels = [clean(c.get_text(' ')) for c in rows[0].find_all(['th', 'td'], recursive=False)]
        if labels != COLUMNS:
            continue
        for row in rows[1:]:
            cells = row.find_all(['th', 'td'], recursive=False)
            if len(cells) != 6:
                raise ValueError('Unexpected tariff table column count')
            values = [clean(c.get_text(' ')) for c in cells]
            if values == COLUMNS:
                continue  # CBSA repeats the column labels inside tbody
            yield values


def node(code, description, source, **extra):
    display = code[:2] + '.' + code[2:] if len(code) == 4 else code[:4] + '.' + code[4:]
    return dict(code=code, display_code=display, description=description, source_url=source, **extra)


def relative_description(full, heading):
    # CBSA repeats the entire ancestor text in the Description of Goods cell.
    # Strip only an exact heading prefix; preserve intermediate group descriptions.
    if full.startswith(heading):
        return full[len(heading):].lstrip(' -–') or full
    return full


# excerpt:parser:start

def parse_chapter(chapter, soup):
    headings, explicit, candidates = {}, {}, {}
    tariff_lines = {}
    current = None
    example = None
    for values in tariff_rows(soup):
        display, suffix, description, unit, mfn, preferences = values
        if not display:
            continue  # unnumbered explanatory/group rows are not HS nodes
        if re.fullmatch(r'\d{2}\.\d{2}', display):
            code = display.replace('.', '')
            if code in headings:
                raise ValueError(f'Duplicate HS4 row: {code}')
            current = node(code, description, chapter['source_url'],
                           chapter_code=chapter['code'], extraction='explicit_hs4_row', subheadings=[])
            headings[code] = current
        elif re.fullmatch(r'\d{4}\.\d{2}(?:\.\d{2})?', display):
            code = display.replace('.', '')
            if current is None or current['code'] != code[:4]:
                if code[:4] in headings:
                    raise ValueError(f'Out-of-order HS4 parent for {display}')
                # Unsplit headings (e.g. 0205.00.00) can omit HS4 as well.
                # The flattened description starts with the heading before
                # CBSA's space-hyphen-space hierarchy separators.
                current = node(code[:4], description.split(' - ')[0], chapter['source_url'],
                               chapter_code=chapter['code'], extraction='inferred_from_child_row',
                               inferred_from=display, subheadings=[])
                headings[code[:4]] = current
            hs6 = code[:6]  # strings throughout: preserve leading zeros
            if len(code) == 6:
                if hs6 in explicit:
                    raise ValueError(f'Duplicate HS6 row: {display}')
                explicit[hs6] = description
            else:
                # SS distinguishes statistical detail from the base tariff item.
                candidates.setdefault(hs6, []).append((code, suffix, description))
                # Retain every Canadian source row, including blank-SS base
                # rows and each statistical suffix. Never inherit or merge rates.
                tariff_lines.setdefault(hs6, []).append(dict(zip(
                    ['tariff_item', 'statistical_suffix', 'description', 'unit',
                     'mfn_tariff', 'preferential_tariffs'], values)))
                if display == '0601.10.11' and suffix == '00':
                    example = dict(zip(['tariff_item', 'ss', 'description', 'unit', 'mfn', 'preferences'], values))
        else:
            raise ValueError(f'Unrecognized tariff code cell: {display!r}')
    for hs6 in sorted(set(explicit) | set(candidates)):
        parent = headings[hs6[:4]]
        inferred = hs6 not in explicit
        evidence = []
        source_warning = None
        if not inferred:
            full = explicit[hs6]
        else:
            rows = candidates[hs6]
            evidence = sorted({r[0] for r in rows})
            # An unsplit .00 tariff item carries the international description.
            # Prefer the blank-SS base row over more detailed statistical rows.
            base_rows = [r for r in rows if r[0].endswith('00') and r[1] in {'', '00'}]
            if not base_rows:
                if not all(r[0].endswith('00') for r in rows):
                    raise ValueError(f'Cannot safely infer HS6 description for {hs6}; inspect source')
                # Some source rows have only statistical suffixes and literal
                # "null" text. Retain the common ancestor path, flag it visibly,
                # and retain all verbatim descriptions for manual verification.
                paths = [r[2].split(' - ') for r in rows]
                common = []
                for parts in zip(*paths):
                    if len(set(parts)) != 1 or parts[0] == 'null':
                        break
                    common.append(parts[0])
                if len(common) < 2:
                    raise ValueError(f'No common HS6 description path: {hs6}')
                full = ' - '.join(common)
                source_warning = ('Source HTML has only statistical rows for this prefix, '
                                  'with no base tariff-item row. The common description path '
                                  'is retained; verify classification against the official schedule.')
            else:
                full = min((r[2] for r in base_rows), key=len)
        parent['subheadings'].append(node(
            hs6, relative_description(full, parent['description']), chapter['source_url'],
            heading_code=parent['code'], chapter_code=chapter['code'],
            source_description=full,
            extraction='inferred_from_tariff_item' if inferred else 'explicit_hs6_row',
            inferred_from=evidence, source_warning=source_warning,
            source_row_descriptions=[r[2] for r in candidates.get(hs6, [])] if source_warning else [],
            canadian_tariff_lines=tariff_lines.get(hs6, [])))
    chapter['headings'] = list(headings.values())
    if not chapter['reserved'] and not headings:
        raise ValueError(f'No tariff headings parsed for chapter {chapter["code"]}')
    return example
# excerpt:parser:end


def validate(data):
    errors, warnings = [], []
    def check(ok, message):
        if not ok:
            errors.append(message)
    sections = data['sections']
    chapters = [c for s in sections for c in s['chapters']]
    check(len(sections) == 21, 'Expected 21 international sections')
    check(len({s['code'] for s in sections}) == 21, 'Duplicate section codes')
    check({c['code'] for c in chapters} == {f'{i:02}' for i in range(1, 98)}, 'Expected chapters 01–97 including reserved 77')
    check(len(chapters) == 97, 'Duplicate/missing HS2 nodes')
    check({c['code'] for c in data['special_chapters']} == {'98', '99'}, 'Separate national chapters 98/99 required')
    check(len(data['special_chapters']) == 2, 'Duplicate national chapters')
    check(data['source'].get('edition') == 'T2026-2', 'Unexpected source edition')
    check(bool(data['source'].get('date_modified')), 'Missing master modification date')
    for section in sections:
        for chapter in section['chapters']:
            check(chapter['section_code'] == section['code'], f'Chapter section mismatch: {chapter["code"]}')
    for c in data['special_chapters']:
        check(c.get('national_provision') and c.get('section_code') is None and not c['headings'], 'National chapters must not enter international hierarchy')
    seen4, seen6 = set(), set()
    for c in chapters + data['special_chapters']:
        check(isinstance(c['code'], str) and bool(re.fullmatch(r'\d{2}', c['code'])), f'Invalid HS2: {c["code"]}')
        if not isinstance(c['code'], str):
            continue
        check(bool(c.get('source_url', '').startswith('https://www.cbsa-asfc.gc.ca/')), f'Missing chapter URL: {c["code"]}')
        check(bool(c['description']), f'Missing chapter description: {c["code"]}')
        if c['code'] == '77':
            check(c['reserved'] and 'reserved' in c['description'].lower() and not c['headings'], 'Chapter 77 must be reserved and empty')
        for h in c['headings']:
            code = h['code']
            check(isinstance(code, str) and bool(re.fullmatch(r'\d{4}', code)), f'Invalid HS4: {code}')
            if not isinstance(code, str):
                continue
            check(code.startswith(c['code']) and h['chapter_code'] == c['code'], f'HS4 parent mismatch: {code}')
            check(code not in seen4, f'Duplicate HS4 node: {code}')
            seen4.add(code)
            check(h['display_code'] == code[:2]+'.'+code[2:], f'Invalid HS4 display: {code}')
            check(bool(h['description']) and h['source_url'] == c['source_url'], f'HS4 description/source missing: {code}')
            check(bool(h['subheadings']), f'Heading without subheadings: {code}')
            for sub in h['subheadings']:
                sc = sub['code']
                check(isinstance(sc, str) and bool(re.fullmatch(r'\d{6}', sc)), f'Invalid HS6: {sc}')
                if not isinstance(sc, str):
                    continue
                check(sc.startswith(code) and sub['heading_code'] == code and sub['chapter_code'] == c['code'], f'HS6 parent mismatch: {sc}')
                check(sc not in seen6, f'Duplicate HS6 node: {sc}')
                seen6.add(sc)
                check(sub['display_code'] == sc[:4]+'.'+sc[4:], f'Invalid HS6 display: {sc}')
                check(bool(sub['description']) and bool(sub['source_description']) and sub['source_url'] == c['source_url'], f'HS6 description/source missing: {sc}')
                if sub.get('source_warning'):
                    warnings.append(f'{sub["display_code"]}: {sub["source_warning"]}')
                lines = sub.get('canadian_tariff_lines')
                check(isinstance(lines, list) and bool(lines), f'Missing Canadian tariff lines: {sc}')
                if not isinstance(lines, list):
                    continue
                for line in lines:
                    fields = ['tariff_item', 'statistical_suffix', 'description',
                              'unit', 'mfn_tariff', 'preferential_tariffs']
                    check(all(isinstance(line.get(field), str) for field in fields),
                          f'Tariff-line fields must be strings: {sc}')
                    item, suffix = line.get('tariff_item'), line.get('statistical_suffix')
                    check(isinstance(item, str) and bool(re.fullmatch(r'\d{4}\.\d{2}\.\d{2}', item))
                          and item.replace('.', '').startswith(sc), f'Tariff-line parent/code mismatch: {sc}')
                    check(isinstance(suffix, str) and (suffix == '' or bool(re.fullmatch(r'\d{2}', suffix))),
                          f'Invalid statistical suffix: {sc}')
                    check(bool(line.get('description')), f'Missing tariff-line description: {sc}')
    c6 = next((c for c in chapters if c['code'] == '06'), None)
    check(c6 is not None, 'Chapter 6 regression: missing 06')
    if c6:
        h = next((h for h in c6['headings'] if h['code'] == '0601'), None)
        check(h is not None and any(s['code'] == '060110' and s['display_code'] == '0601.10' for s in h['subheadings']), 'Chapter 6 regression: 06 → 0601 → 0601.10')
        inferred = next((s for h in c6['headings'] for s in h['subheadings'] if s['code'] == '060210'), {})
        check(inferred.get('extraction') == 'inferred_from_tariff_item' and '06021000' in inferred.get('inferred_from', []), 'Chapter 6 regression: infer 0602.10 from 0602.10.00')
        check(inferred.get('description') == 'Unrooted cuttings and slips', 'Chapter 6 inferred description regression')
        multiple = next((s for h in c6['headings'] for s in h['subheadings'] if s['code'] == '060110'), {})
        lines = multiple.get('canadian_tariff_lines', [])
        check(len(lines) == 13, 'Chapter 6 multiple-row regression: expected all 13 rows under 0601.10')
        check({r.get('statistical_suffix') for r in lines if r.get('tariff_item') == '0601.10.19'} == {'', '10', '90'},
              'Chapter 6 multiple-row regression: preserve base row and separate statistical suffixes')
    c15 = next((c for c in chapters if c['code'] == '15'), None)
    check(c15 is not None, 'Chapter 15 regression: missing 15')
    if c15:
        h15 = next((h for h in c15['headings'] if h['code'] == '1501'), None)
        check(h15 is not None, 'Chapter 15 regression: missing 1501')
        lard = next((s for s in (h15 or {}).get('subheadings', []) if s['code'] == '150110'), {})
        check(lard.get('display_code') == '1501.10', 'Chapter 15 regression: missing 1501.10')
        lard_line = next((r for r in lard.get('canadian_tariff_lines', [])
                          if r.get('tariff_item') == '1501.10.00' and r.get('statistical_suffix') == '00'), {})
        check(lard_line.get('unit') == 'KGM' and lard_line.get('mfn_tariff') == 'Free',
              'Chapter 15 regression: lard must retain unit KGM and MFN Free')
        check(lard_line.get('preferential_tariffs') == (
            'CCCT, LDCT, GPT, UST, MXT, CIAT, CT, CRT, IT, PT, COLT, JT, PAT, HNT, KRT, '
            'CEUT, UAT, CPTPT, UKT, CPUKT: Free'), 'Chapter 15 regression: lard preference text differs')
    check('0101' in seen4 and '010121' in seen6, 'Leading-zero regression failed')
    return errors, warnings


# excerpt:pdf:start

def download_pdfs(fetcher, pdf_links):
    downloaded = 0
    for index, url in enumerate(pdf_links, 1):
        # URLs come from anchors in the master page, never a hand-written list.
        destination = BASE / 'source-pdfs' / Path(urlparse(url).path).name
        _, fetched = fetcher.get(url, destination, pdf=True)
        downloaded += int(fetched)
        print(f'PDF {index}/{len(pdf_links)}: {destination.name} ({"downloaded" if fetched else "cached"})', flush=True)
    return downloaded
# excerpt:pdf:end


def summarize(data):
    chapters = [c for s in data['sections'] for c in s['chapters']]
    headings = [h for c in chapters for h in c['headings']]
    subheadings = [s for h in headings for s in h['subheadings']]
    return dict(sections=len(data['sections']), hs2_chapters=len(chapters),
                active_hs2_chapters=sum(not c['reserved'] for c in chapters),
                hs4_headings=len(headings), hs6_subheadings=len(subheadings),
                inferred_hs6=sum(s['extraction'] == 'inferred_from_tariff_item' for s in subheadings),
                canadian_tariff_lines=sum(len(s['canadian_tariff_lines']) for s in subheadings),
                special_chapters=len(data['special_chapters']),
                html_chapter_pages_parsed=data['source']['html_chapter_pages_parsed'],
                pdf_links_discovered=len(data['source']['pdf_links']),
                pdfs_downloaded=data['source']['pdfs_downloaded'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-pdfs', action='store_true', help='Download discovered official PDFs for local reference')
    parser.add_argument('--force', action='store_true', help='Refresh cached HTML and, with --download-pdfs, PDFs')
    parser.add_argument('--validate-only', action='store_true', help='Validate existing JSON offline')
    args = parser.parse_args()
    if args.validate_only:
        data = json.loads(OUTPUT.read_text())
        errors, warnings = validate(data)
    else:
        fetcher = Fetcher(args.force)
        print(f'Fetching master: {MASTER_URL}', flush=True)
        master, _, master_hash = fetcher.html(MASTER_URL, refresh=True)
        sections, special, pdf_links = discover(master, MASTER_URL)
        title = clean(master.select_one('h1').get_text(' '))
        edition = re.search(r'T\d{4}-\d+', title)
        modified = master.select_one('time[property="dateModified"]')
        data = {'schema_version': 2, 'source': {
            'organization': 'Canada Border Services Agency', 'title': title,
            'url': MASTER_URL, 'edition': edition[0] if edition else None,
            'date_modified': clean(modified.get_text()) if modified else None,
            'retrieved_at': datetime.now(timezone.utc).isoformat(),
            'master_sha256': master_hash,
            'method': 'CBSA HTML tariff tables; missing HS4 rows reconstructed from child descriptions; missing HS6 rows inferred from unsplit .00 tariff items. Statistical-only prefixes retain common description paths with explicit source warnings.',
            'pdf_links': pdf_links, 'pdfs_downloaded': 0,
            'html_chapter_pages_parsed': 0}, 'sections': sections,
            'special_chapters': special, 'chapter6_example': None}
        warnings = []
        chapters = [c for s in sections for c in s['chapters']] + special
        if len(chapters) != 99:
            raise ValueError(f'Expected 99 chapter links; discovered {len(chapters)}')
        for index, chapter in enumerate(chapters, 1):
            soup, fetched, digest = fetcher.html(chapter['source_url'])
            chapter['source_sha256'] = digest
            if chapter.get('national_provision'):
                if not list(tariff_rows(soup)):
                    raise ValueError(f'Missing national schedule: {chapter["code"]}')
            else:
                example = parse_chapter(chapter, soup)
                if example:
                    data['chapter6_example'] = example
            data['source']['html_chapter_pages_parsed'] += 1
            print(f'Chapter {chapter["code"]} ({index}/99): {len(chapter["headings"])} HS4 ({"fetched" if fetched else "cached"})', flush=True)
        data['source']['chapter_effective_dates'] = sorted({c['effective_date'] for c in chapters})
        errors, validation_warnings = validate(data)
        warnings += validation_warnings
        if args.download_pdfs and not errors:
            data['source']['pdfs_downloaded'] = download_pdfs(fetcher, pdf_links)
    summary = summarize(data)
    summary.update(validation_errors=len(errors), validation_warnings=len(warnings))
    print(json.dumps(summary, indent=2))
    for message in errors + warnings:
        print(message, file=sys.stderr)
    if errors:
        return 1
    if not args.validate_only:
        data['validation'] = dict(errors=errors, warnings=warnings, chapter6_regression='passed',
                                  chapter15_regression='passed', multiple_tariff_rows_regression='passed')
        data['statistics'] = summary
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUTPUT.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(OUTPUT)
        print(f'Wrote {OUTPUT}')
    print('Validation passed, including Chapters 6 and 15, multiple tariff rows and leading-zero regressions.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (requests.RequestException, ValueError, OSError, KeyError) as exc:
        print(f'Build failed; existing JSON was not replaced: {exc}', file=sys.stderr)
        sys.exit(1)
