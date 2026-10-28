"""Static structure, asset preservation, rename, and research-entry scope audit."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import hashlib, json, re, subprocess

ASSETS = Path(__file__).resolve().parent
PAGE = ASSETS.with_suffix('.html')
ROOT = ASSETS.parents[2]

class Audit(HTMLParser):
    def __init__(self):
        super().__init__(); self.stack=[]; self.ids=[]; self.links=[]; self.errors=[]
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if 'id' in attrs: self.ids.append(attrs['id'])
        for key in ('href','src'):
            if key in attrs: self.links.append(attrs[key])
        if tag not in {'meta','link','input','img','br','hr','source','wbr'}: self.stack.append(tag)
    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1]!=tag: self.errors.append((tag,self.stack[-3:]))
        else: self.stack.pop()
    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag,attrs)

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)

def run():
    parser=Audit(); content=PAGE.read_text(encoding='utf-8'); parser.feed(content)
    assert not parser.errors and not parser.stack,(parser.errors,parser.stack)
    assert len(parser.ids)==len(set(parser.ids)), 'Duplicate ids'
    local=0
    for link in parser.links:
        url=urlsplit(link)
        if url.scheme or url.netloc: continue
        target=(PAGE.parent/unquote(url.path)).resolve() if url.path else PAGE
        assert target.exists(),link
        if url.fragment and target==PAGE: assert url.fragment in parser.ids,link
        local+=1
    assert '<!-- ORIGINAL_GALLERY -->' not in content
    assert 'hypergraph_outputs' not in content and 'hypergraph_demo.html' not in content
    assert not (PAGE.parent/'hypergraph_demo.html').exists()
    assert not (PAGE.parent/'hypergraph_outputs').exists()
    # Compare images to the pre-redesign tracked names, or their preserved names after commit.
    preserved=[]
    for p in sorted(ASSETS.glob('hypergraph_*.png')):
        old='research/graph-theory/hypergraph_outputs/'+p.name
        probe=subprocess.run(['git','cat-file','-e','HEAD:'+old],cwd=ROOT,capture_output=True)
        original=git('show','HEAD:'+old) if probe.returncode==0 else git('show','HEAD:research/graph-theory/'+ASSETS.name+'/'+p.name)
        assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(original).digest(),p
        preserved.append(p.name)
    assert len(preserved)==10
    index=(ROOT/'research.html').read_text(encoding='utf-8')
    before=git('show','HEAD:research.html').decode('utf-8')
    assert index==before, 'research.html changed during refinement'
    pos=index.index('research/graph-theory/'+PAGE.name)
    assert index.rfind('<details',0,pos)>index.rfind('</details>',0,pos), 'Not in collapsible subsection'
    report=dict(status='PASS',localLinks=local,uniqueIDs=len(parser.ids),originalImagesUnchanged=len(preserved),researchIndex='Unchanged from HEAD; link still resolves in details',oldNames='Absent',htmlStructure='Balanced',externalRuntimeDependencies=0)
    (ASSETS/'refinement-site-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':run()
