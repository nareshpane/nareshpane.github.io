"""Independent HTML-release comparisons and preservation checks for the refinements.

These are additional source matches against the same official release, not 70
new independent price estimates. The original 242-comparison audit is retained.
"""
from pathlib import Path
import hashlib, html, json, re

ROOT = Path(__file__).resolve().parents[1]

def main():
    data = json.loads((ROOT/'data/processed/cpi.json').read_text(encoding='utf-8'))
    extra = json.loads((ROOT/'data/processed/refinement-data.json').read_text(encoding='utf-8'))
    baseline = json.loads((ROOT/'validation/refinement-baseline.json').read_text(encoding='utf-8'))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['data/processed/cpi.json', 'validation/replication.json']:
        assert digest(ROOT/name) == baseline[name], 'Validated baseline changed: '+name
    source = (ROOT/'references/geographical-release-2026-09.html').read_text(encoding='utf-8')
    clean = lambda text: html.unescape(re.sub('<[^>]+>', '', text)).strip()
    source_rows = []
    for tr in re.findall(r'<tr\b[^>]*>(.*?)</tr>', source, re.S):
        label = re.search(r'<th\b[^>]*id="t001stub\d+"[^>]*>(.*?)</th>', tr, re.S)
        values = re.findall(r'<td\b[^>]*>(.*?)</td>', tr, re.S)
        if label and len(values)==5:
            source_rows.append((clean(label[1]), [float(clean(v)) for v in values]))
    assert len(source_rows)==14
    aliases = {'Whitehorse':'Whitehorse, Yukon', 'Yellowknife':'Yellowknife, Northwest Territories', 'Iqaluit (Dec. 2002=100)':'Iqaluit, Nunavut'}
    rows = []
    for label, official in source_rows:
        geo = aliases.get(label,label)
        series = next(s for s in data['series'].values() if s['label']=='All-items' and s['geography']==geo)
        v = series['values']
        calculated = [v['2025-09'],v['2026-08'],v['2026-09'],100*(v['2026-09']/v['2026-08']-1),100*(v['2026-09']/v['2025-09']-1)]
        for i, (published, computed) in enumerate(zip(official,calculated)):
            tolerance = 1e-8 if i<3 else .05+1e-12
            row = dict(geography=geo,metric=['Sep 2025 index','Aug 2026 index','Sep 2026 index','MoM','YoY'][i],official=published,calculated=computed,difference=computed-published,tolerance=tolerance,vector=series['vector'],passed=abs(computed-published)<=tolerance)
            rows.append(row)
    assert all(r['passed'] for r in rows), 'Geographical release mismatch'
    for image in extra['illustrations']:
        assert digest(ROOT/image['file'])==image['sha256']
        source_path=Path(image['source'])
        if source_path.exists(): assert digest(source_path)==image['sha256']
    assert len(extra['illustrations'])==9
    # The displayed national hierarchy follows the source parent edges.
    assert all(data['products'][str(i)]['parent']==str(i-1) for i in range(3,8))
    # Rounded published geographical shares need not add to exactly 100.
    regional = {k:v['link']['value'] for k,v in extra['regional_weights'].items() if k!='Canada'}
    assert len(regional)==13 and abs(sum(regional.values())-100)<.1
    report={'source':'references/geographical-release-2026-09.html','source_comparisons':len(rows),'failures':[],'illustrations_hash_verified':9,'baseline_data_and_replication_unchanged':True,'regional_share_sum_before_normalization':sum(regional.values()),'geographical_rows':rows}
    (ROOT/'validation/refinement-checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='geographical_rows'},indent=2))

if __name__=='__main__':main()
