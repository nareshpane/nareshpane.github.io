"""Meaningful structural, numerical, source-integrity and cross-language checks."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
from collections import Counter
import hashlib, json, math, subprocess
import numpy as np
import pandas as pd
from scipy.stats import norm
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
PAGE=ROOT.parent/(ROOT.name+'.html')

class Inventory(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.links=[];self.images=[];self.buttons=0;self.ranges=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag in ['a','link'] and 'href' in a:self.links.append(a['href'])
        if tag in ['script','img'] and 'src' in a:self.links.append(a['src'])
        if tag=='img':self.images.append(a)
        if tag=='button':self.buttons+=1
        if tag=='input' and a.get('type')=='range':self.ranges+=1

def main():
    checks={};html=PAGE.read_text(encoding='utf-8');inv=Inventory();inv.feed(html)
    assert not [x for x,n in Counter(inv.ids).items() if n>1]
    checks['unique_ids']=len(inv.ids)
    for link in inv.links:
        u=urlsplit(link)
        if u.scheme or u.netloc:continue
        if u.path:
            target=(PAGE.parent/unquote(u.path)).resolve()
            assert target.exists(),f'Missing local resource: {link}'
        if u.fragment and not u.path:assert u.fragment in inv.ids,link
    checks['local_links_and_anchors']=len(inv.links)
    assert all(x.get('alt') for x in inv.images);checks['images_with_alt']=len(inv.images)
    for p in (ROOT/'figures').glob('*.svg'):ET.parse(p)
    checks['valid_svg_files']=len(list((ROOT/'figures').glob('*.svg')))
    metadata=json.loads((ROOT/'sources/dryad-files.json').read_text())
    for f in metadata['_embedded']['stash:files']:
        p=ROOT/('data' if f['path'].endswith('CSV') else 'sources')/f['path']
        assert hashlib.md5(p.read_bytes()).hexdigest()==f['digest']
    checks['source_md5_matches']=3
    d=pd.read_csv(ROOT/'data/Panigrahi et al.SEPSSIF.CSV',na_values=['.'])
    summary=json.loads((ROOT/'data/derived_summary.json').read_text())
    for key,r in summary['outcomes'].items():
        for z in [0,1]:
            assert int(d.loc[d.TX.eq(z),key].sum())==r['events'+str(z)]
        assert [r['events0'],r['events1']]==summary['published'][key][:2]
    checks['published_outcome_numerators_matched']=len(summary['outcomes'])
    p=summary['primary'];a=123;c=206;n=2278
    independently_rr=(a/n)/(c/n); independently_rd=(a-c)/n
    independently_se=math.sqrt((1-a/n)/a+(1-c/n)/c)
    assert math.isclose(independently_rr,p['rr'],abs_tol=1e-12)
    assert math.isclose(independently_rd,p['rd'],abs_tol=1e-12)
    assert np.allclose(np.exp(np.log(independently_rr)+np.array([-1,1])*norm.ppf(.975)*independently_se),p['rr_ci'])
    assert '0.597087' in html and '0.481261–0.740790' in html
    checks['primary_independent_formula_check']=True
    assert 7089-2506-27==4556
    assert 2185+13+76+4==2278 and 2141+19+112+6==2278
    checks['published_flow_arithmetic']=True
    assert sum(b['n'] for b in summary['event_age_bins'])==329-summary['quality']['primary_time_missing']
    checks['event_age_histogram_reconciled']=True
    assert '<th scope="col"ead' not in html
    assert not [c for c in html if ord(c)<32 and c not in '\n\r\t']
    checks['table_headers_and_text_controls_valid']=True
    assert '<b class="model">Model:</b><span>GPT-6.1 Sol</span>' in html
    assert '<b class="reasoning">Reasoning:</b><span>High</span>' in html
    assert 'not recorded' not in html.lower()
    assert 'Ira H. Gewolb' in html and 'paper-title' in inv.ids
    checks['requested_metadata_and_prominent_citation']=True
    second=json.loads((ROOT/'data/independent_verification.json').read_text(encoding='utf-8'))
    for key,r in summary['outcomes'].items():
        for field in ['events0','events1','rr','rd','pearson_p','fisher_p']:
            assert math.isclose(r[field],second['outcomes'][key][field],abs_tol=2e-10)
    checks['separate_implementation_outcomes_matched']=len(summary['outcomes'])
    assert '.7536' in html and '0.526635' in html
    checks['mortality_test_discrepancy_documented']=True
    for filename in ['legacy-labs.js','monograph.js','refinement-labs.js','statistical-math.js','data/derived_summary.js']:
        subprocess.run(['node','--check',str(ROOT/filename)],check=True,capture_output=True)
    legacy=(ROOT/'legacy-labs.js').read_text()
    powercode='function erf'+legacy.split('function erf',1)[1].split(' function drawPower',1)[0]
    scenarios=[[100,.036,.08,.05],[4221,.016,.08,.05],[6000,.005,.2,.01],[300,.1,.5,.1],[100,.025,.03,.01],[6000,.595,.6,.1]]
    jscode='const M=require('+json.dumps(str(ROOT/'statistical-math.js'))+');console.log(JSON.stringify('+json.dumps(scenarios)+'.map(x=>M.planning(...x).power)));'
    actual=json.loads(subprocess.run(['node','-e',jscode],check=True,capture_output=True,text=True).stdout)
    expected=[]
    for n,delta,p0,alpha in scenarios:
        p1=p0-delta;avg=(p0+p1)/2;cut=norm.ppf(1-alpha/2)*math.sqrt(2*avg*(1-avg)/n);se=math.sqrt((p0*(1-p0)+p1*(1-p1))/n)
        expected.append(norm.cdf((-cut+delta)/se)+norm.sf((cut+delta)/se))
    assert np.allclose(actual,expected,atol=3e-7,rtol=0)
    checks['javascript_power_vs_scipy']={'scenarios':len(scenarios),'max_absolute_error':max(abs(a-b) for a,b in zip(actual,expected))}
    assert '(extreme+1)/(fisher.length+1)' in legacy
    checks['fisher_plus_one_correction']=True
    for key in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','ab','ac','ad']:
        assert key in inv.ids
    checks['original_lettered_sections_retained']=30
    checks['buttons']=inv.buttons;checks['range_controls']=inv.ranges
    (ROOT/'documentation/validation-results.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
