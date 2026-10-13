"""Check the revision against the pre-edit snapshot without touching git's index."""
from pathlib import Path
import hashlib,json,re,tempfile,subprocess,difflib,sys
sys.stdout.reconfigure(encoding='utf-8')
from html.parser import HTMLParser
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[1]
PAGE=ROOT/'research/labour-force-survey-uncertainty-measurement.html'
before=Path(tempfile.gettempdir())/'lfs-redesign-before.html'
manifest=json.loads((BASE/'figures/redesign-preservation-baseline.json').read_text())
for name,digest in manifest.items():
    assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest,name

class Inventory(HTMLParser):
    def __init__(self,text):
        super().__init__();self.sections=[];self.paths=[];self.forms=[];self.feed(text)
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='section' and 'id' in a:self.sections.append(a['id'])
        if tag=='form':self.forms.append(a.get('id'))
        for key in ['href','src']:
            value=a.get(key,'')
            if value and not value.startswith(('https:','http:','#','data:')):self.paths.append(value)

old=Inventory(before.read_text(encoding='utf-8'));new=Inventory(PAGE.read_text(encoding='utf-8'))
assert old.sections==new.sections,'Article sections or order changed'
assert old.forms==new.forms,'A statistical form was removed'
for path in new.paths:assert (PAGE.parent/path.split('#')[0]).exists(),path
new_text=PAGE.read_text(encoding='utf-8')
assert 'Full contents' not in new_text and 'Current design:' not in new_text
assert 'These labels are part of the analysis.' not in new_text
assert all(f'<b>{title}</b>' in new_text for title in ['Measure','Represent','Quantify','Interpret','Compare','Look beyond the interval'])
assert new_text.count('class="chapter"')==before.read_text(encoding='utf-8').count('class="chapter"')
assert not list((BASE/'figures').glob('.browser-profile-*')),'A test browser profile remains'
report={'preserved_files_verified':len(manifest),'article_sections_preserved':new.sections,'statistical_forms_preserved':len(new.forms),'local_asset_paths_verified':len(new.paths),'mathematical_corrections':'No numerical or substantive formula corrections required; equivalent multiline layouts and explicit auxiliary notation only.'}
(BASE/'figures/redesign-preservation-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
diff=subprocess.run(['git','diff','--no-index','--stat','--',str(before),str(PAGE)],capture_output=True,text=True)
print('Review against pre-edit page (untracked files do not appear in ordinary git diff):')
print(diff.stdout)
print('Relevant changed-line review (long standalone HTML lines abbreviated):')
for line in difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(),new_text.splitlines(),n=0):
    if line.startswith(('+','-','@@')):
        print(line[:320]+(' …' if len(line)>320 else ''))
