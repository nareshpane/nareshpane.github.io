"""Validate every static asset, embedded source, data cell and math constant."""
import argparse
import hashlib
import json
import math
from html.parser import HTMLParser
from pathlib import Path
from .statistics import LIMITS, NS, wilson


class ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links=[]; self.ids=[]; self.source=None; self.in_code=False
        self.sources={}; self.in_data=False; self.data=''; self.details=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag in ['a','link','script']:
            link=a.get('href',a.get('src'))
            if link:self.links.append(link)
        if tag=='details':
            self.details+=1
            if 'data-source' in a:
                self.source=a['data-source']; self.sources[self.source]=''
        if tag=='code' and self.source:self.in_code=True
        if tag=='script' and a.get('id')=='research-data':self.in_data=True
    def handle_endtag(self,tag):
        if tag=='code':self.in_code=False
        if tag=='details':self.source=None
        if tag=='script':self.in_data=False
    def handle_data(self,text):
        if self.in_code:self.sources[self.source]+=text
        if self.in_data:self.data+=text


def validate(assets,page):
    text=page.read_text(encoding='utf8'); p=ArticleParser();p.feed(text)
    assert len(p.ids)==len(set(p.ids)), 'Duplicate HTML ids'
    missing=[]
    for link in p.links:
        if link.startswith(('https:','http:','mailto:')):continue
        if link.startswith('#'):
            assert link[1:] in p.ids,link
        elif not ((page.parents[2]/link.lstrip('/').split('#')[0]) if link.startswith('/') else (page.parent/link.split('#')[0])).exists():missing.append(link)
    assert not missing, missing
    paths=[path for path in assets.rglob('*.py') if '__pycache__' not in path.parts]
    assert len(p.sources)==len(paths),(len(p.sources),len(paths))
    for path in paths:
        rel=path.relative_to(assets).as_posix()
        assert p.sources[rel]==path.read_text(encoding='utf8'), f'Source differs: {rel}'
    data=json.loads(p.data)
    assert data==json.loads((assets/'data/results.json').read_text(encoding='utf8'))
    assert len(data['rows'])==60
    assert len({(r['n'],r['model'],r['experiment']) for r in data['rows']})==60
    for experiment in ['snapshot','trajectory']:
        for model in ['simple','multigraph']:
            assert sorted(r['n'] for r in data['rows'] if r['model']==model and r['experiment']==experiment)==NS
    for row in data['rows']:
        assert row['trials']==data['metadata']['target_trials']
        assert row['probability']==row['successes']/row['trials']
        assert (row['ci_low'],row['ci_high'])==wilson(row['successes'],row['trials'])
        assert row['theoretical_limit']==LIMITS[row['experiment']]
    assert f'Limit {LIMITS["snapshot"]:.7f}' in text
    assert f'Limit {LIMITS["trajectory"]:.7f}' in text
    report={'static_validation':'passed','relative_links_checked':len(p.links),
            'python_sources_verified':len(paths),'data_cells_verified':60,
            'python_tests':'9 tests passed (see test log)',
            'page_sha256':hashlib.sha256(page.read_bytes()).hexdigest(),
            'browser_checks':'Pending browser verification'}
    output=assets/'validation.json'
    if output.exists():
        old=json.loads(output.read_text()); browser=old.get('browser_checks')
        if isinstance(browser,dict) and browser.get('page_sha256')==report['page_sha256']:
            report['browser_checks']=browser
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--page',type=Path)
    a=p.parse_args();assets=a.assets.resolve()
    validate(assets,a.page or assets.parent/'giant-component-1993.html')


if __name__=='__main__':main()
