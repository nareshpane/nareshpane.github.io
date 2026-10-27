"""Compare the continuous panels with every original transcription token."""
from html.parser import HTMLParser
from pathlib import Path
import json,re
BASE=Path(__file__).resolve().parents[1]
class Plain(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.text=[]
    def handle_data(self,s):self.text.append(s)
def normalize(s):
    p=Plain();p.feed(s);s=''.join(p.text)
    s=re.sub(r'\\text\{([^{}]*)\}',r'\1',s)
    s=re.sub(r'\\(?:qquad|quad|,|;|!| )','',s).replace('&','')
    s=re.sub(r'\\(?:begin|end)\{(?:gathered|aligned)\}','',s)
    s=re.sub(r'\\[\[\]()]','',s).replace('\\\\','')
    return re.sub(r'\s+','',s)
class Panels(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.depth=0;self.panels=[]
    def handle_starttag(self,tag,attrs):
        if tag=='div':
            if dict(attrs).get('class')=='transcript':self.depth=1;self.panels.append('')
            elif self.depth:self.depth+=1
    def handle_endtag(self,tag):
        if tag=='div' and self.depth:self.depth-=1
    def handle_data(self,s):
        if self.depth:self.panels[-1]+=s

source=json.loads((BASE/'validation/transcriptions.json').read_text(encoding='utf-8'))
page=(BASE.parent/(BASE.name+'.html')).read_text(encoding='utf-8')
p=Panels();p.feed(page);assert len(p.panels)==4
groups=[['1a','1b','1c','1d','1f'],['2a','2b'],['3a','3b','3c','3d'],['4a','4b']]
for i,names in enumerate(groups):
    parts=[part for name in names for part in source[name]]
    if i==3:
        caption=parts.pop(len(source['4a'])-1)
        caption=re.sub(r"<span class='editorial'>.*?</span> ",'',caption)
        parts.append(caption)
    expected=normalize(' '.join(parts));actual=normalize(p.panels[i])
    if i==2:assert actual.startswith(expected),i+1
    else:assert actual==expected,(i+1,len(actual),len(expected))
    assert '.png' not in p.panels[i]
assert 'View Original Scanned Excerpts' not in page
assert '<span class="source-tag">' not in page
assert page.count('Original Research — Verbatim')==4
assert 'Source excerpts:' not in page
assert 'Role / first appearance' not in page
report={'sourceGroups':4,'sourceFiles':13,'originalTranscriptionTokens':'unchanged in order','paragraphJoins':['1a–1b','1c–1d','2a–2b','4a–4b'],'equationJoin':'3c–3d','supplement':'Final paragraph completed from supplied PDF p.555; disclosed outside verbatim','visibleSourceFilenames':0,'scanGalleries':0,'failures':0}
(BASE/'data/transcription-check-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
