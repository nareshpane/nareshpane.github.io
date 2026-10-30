"""Audit known generated QA artifacts; retain research provenance and runtime inputs.

This script only reports candidates. Removal is a separate explicit, path-checked
operation. Unknown files are never candidates. Previous deletion records persist.
"""
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'validation/asset-audit.json'

def main():
    old=json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.exists() else {}
    known=[]
    for chapter in range(1,19):
        for width in (1440,390):known.append(f'validation/chapter-s{chapter:02}-{width}.jpg')
    for group in ('chapter','mobile','detail'):
        for part in range(1,4):known.append(f'validation/{group}-review-{part}.jpg')
    for name in ('classification','continuous-branch','geo-multi','mini-price-progress','official-walkthrough','province-ranking','quality-value','region-model-select','regional-components-chart','set-view','weighted-milk-price'):
        known.append(f'validation/detail-{name}.jpg')
    for width in (1024,1280,1600,1920,768):known.append(f'validation/refined-hero-{width}.jpg')
    known.append('validation/mobile-study.jpg')
    for path in (ROOT/'scripts/__pycache__').glob('*.pyc'):
        source=ROOT/'scripts'/(path.name.split('.')[0]+'.py')
        if source.is_file():known.append(path.relative_to(ROOT).as_posix())
    # Inspect runtime, build scripts and documentation, not just literal HTML src.
    files=[ROOT.parent/(ROOT.name+'.html')]+[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.js','.css','.py','.html','.md') and p.resolve()!=Path(__file__).resolve() and p.name!='ASSET_AUDIT.md']
    texts=[(p,p.read_text(encoding='utf-8',errors='replace').lower()) for p in files]
    candidates=[];retained=[]
    for relative in sorted(set(known)):
        path=ROOT/relative
        if not path.is_file():continue
        references=[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else '../'+p.name for p,t in texts if path.name.lower() in t or relative.lower() in t]
        dynamic=[]
        if path.suffix=='.jpg':
            prefix='chapter-' if path.name.startswith('chapter-') else 'detail-' if path.name.startswith('detail-') else 'refined-hero-' if path.name.startswith('refined-hero-') else None
            # "chapter-header" / "chapter-jump" are CSS/DOM identifiers, not image loaders.
            if prefix:dynamic=[str(p.relative_to(ROOT)) for p,t in texts if p.suffix in ('.js','.py') and '.jpg' in t and prefix in t]
        entry={'file':relative,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'references':sorted(set(references+dynamic)),'reason':'Regenerable Python import cache; corresponding .py source retained.' if path.suffix=='.pyc' else 'Superseded browser QA screenshot or contact sheet; final representative views and numerical reports retained.'}
        (retained if entry['references'] else candidates).append(entry)
    report={'scope':str(ROOT),'source_and_script_files_searched':len(files),'original_inputs_protected':30,'original_supplied_images_protected':9,'retained_roles':['All PDFs, SDMX/XML, downloaded ZIPs, data wrappers, processed source datasets, scripts, manifests and numerical validation reports.','release-table1.png is retained as source-table inspection evidence.','refined-hero-1440.jpg and refined-hero-390.jpg are retained and refreshed; desktop-weights.jpg and detail-sample-frame.jpg likewise.'],'candidates':candidates,'retained_due_to_references':retained,'removed_files':old.get('removed_files',[])}
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    removed=report['removed_files'];size=sum(r['bytes'] for r in removed)
    lines=['# CPI asset audit','','Only explicitly identified generated QA screenshots/contact sheets and regenerable import caches were removed. Runtime references, build scripts and documentation were searched; dynamic screenshot-prefix loading was also checked.','','Original research downloads, all three PDFs, all nine original PNGs, scripts, manifests and numerical reports remain. Source-table inspection evidence and five representative final screenshots remain.','',f'Removed {len(removed)} files ({size:,} bytes).', '', '| Removed file | Reason |','|---|---|']
    lines += [f"| `{r['file']}` | {r['reason']} |" for r in removed]
    if not removed:lines.append('| None yet | Audit only. |')
    (ROOT/'ASSET_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'candidates':len(candidates),'candidate_bytes':sum(r['bytes'] for r in candidates),'retained_due_to_references':len(retained),'previously_removed':len(removed)}))

if __name__=='__main__':main()
