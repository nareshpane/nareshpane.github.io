"""Check HTML structure, local links, anchors, source integrity and index integration."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import hashlib,json,re,subprocess

BASE=Path(__file__).resolve().parents[1]
PAGE=BASE.parent/(BASE.name+'.html')
ROOT=BASE.parents[2]
class Audit(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids=[];self.links=[];self.stack=[];self.errors=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        for k in ('href','src'):
            if k in a:self.links.append(a[k])
        if tag not in {'meta','link','img','input','br','hr','source','wbr','area','base','col','embed','param','track'}:
            self.stack.append(tag)
    def handle_endtag(self,tag):
        if not self.stack or self.stack[-1]!=tag:self.errors.append('Misnested '+tag)
        else:self.stack.pop()
    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)

def run():
    report={}
    for file in [PAGE,ROOT/'research.html']:
        p=Audit();p.feed(file.read_text(encoding='utf-8-sig'))
        assert not p.errors,(file,p.errors)
        assert not p.stack,(file,p.stack)
        assert len(p.ids)==len(set(p.ids)),'Duplicate IDs'
        checked=0
        for link in p.links:
            u=urlsplit(link)
            if u.scheme or u.netloc:continue
            dest=(file.parent/unquote(u.path)).resolve() if u.path else file
            assert dest.exists(),(file,link)
            if u.fragment and dest==file:assert unquote(u.fragment) in p.ids,link
            checked+=1
        report[file.name]=dict(localLinks=checked,ids=len(p.ids),structure='pass')
    saved=json.loads((BASE/'data/validation-results.json').read_text())
    for name,digest in saved['source_sha256'].items():
        assert hashlib.sha256((BASE/'Verbatim'/name).read_bytes()).hexdigest()==digest,name
    # This refinement must not modify any part of the existing landing page.
    current=(ROOT/'research.html').read_text(encoding='utf-8-sig')
    before=subprocess.check_output(['git','show','HEAD:research.html'],cwd=ROOT).decode('utf-8')
    assert current==before
    report['sourceIntegrity']='13 original PNG hashes unchanged'
    report['existingResearchIndex']='unchanged from HEAD'
    (BASE/'data/site-check-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
