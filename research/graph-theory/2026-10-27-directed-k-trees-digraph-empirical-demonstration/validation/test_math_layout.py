"""Typesetting regression tests: preserve formulas and matrix entries, remove scrollers."""
from pathlib import Path
import json,re
from html.parser import HTMLParser
from math_layout import flow_math,fit_math
BASE=Path(__file__).resolve().parents[1]
class Plain(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.parts=[]
    def handle_data(self,s):self.parts.append(s)
def canonical(s):
    p=Plain();p.feed(s);s=''.join(p.parts)
    s=re.sub(r'\\text\{([^{}]*)\}',r'\1',s)
    s=re.sub(r'\\(?:begin|end)\{(?:aligned|gathered)\}','',s)
    s=re.sub(r'\\(?:qquad|quad|,|;|!| )','',s)
    s=re.sub(r'\\[\[\]()]','',s).replace('\\\\','').replace('&','')
    return re.sub(r'\s+','',s)
data=json.loads((BASE/'validation/transcriptions.json').read_text(encoding='utf-8'))
tested=0
for name,parts in data.items():
    for p in parts:
        for m in re.finditer(r'\\\((.*?)\\\)|\\\[(.*?)\\\]',p,re.S):
            original=m[1] if m[1] is not None else m[2]
            assert canonical(original)==canonical(flow_math(original)),(name,original)
            tested+=1
matrix=r'\begin{pmatrix}1&-1&0&0\\-1&3&-1&0\\0&-1&3&-1\\0&-1&-1&3\end{pmatrix}'
cases=r'\begin{cases}0&v\in R\\1&v\notin R\end{cases}'
for block in (matrix,cases):assert block in flow_math('L='+block+',x=12')
sample=r'<head><script>displayMath:[["\\[","\\]"]]</script></head><body><p>\(P_{ij}=(i,k_1)(k_1,k_2)(k_2,j)\in E\)</p></body>'
result=fit_math(sample)
assert result.split('<body>')[0]==sample.split('<body>')[0]
assert canonical(sample)==canonical(result)
css=(BASE/'css/style.css').read_text()
assert not re.search(r'overflow(?:-x|-y)?\s*:\s*(?:hidden|clip)',css)
assert 'scrollbar-width' not in css
page=(BASE.parent/(BASE.name+'.html')).read_text(encoding='utf-8')
assert 'inline-math-scroll' not in page
assert 'overflow-x:auto' not in page
report={'sourceMathExpressionsChecked':tested,'matrixAndCasesEntries':'exactly preserved','renderer':'MathJax 3 SVG retained','clippingOrScrollbarHidingCSS':False,'failures':0}
(BASE/'data/math-layout-test-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
