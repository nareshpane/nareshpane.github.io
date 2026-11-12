"""Validate frozen official inputs, worked examples and synthetic bootstrap."""
from pathlib import Path
import json, math, numpy as np
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'data'/'september-2026.json').read_text())
report=[]
for g in data['geographies']:
 a=g['indicators'];E,U,L,P=[a[k]['value'] for k in ['Employment','Unemployment','Labour force','Population']]
 assert abs(E+U-L)<=.151,(g['geography'],E,U,L)
 for key,expected in [('Unemployment rate',100*U/L),('Employment rate',100*E/P),('Participation rate',100*L/P)]:
  assert abs(a[key]['value']-expected)<=.065,(g['geography'],key,expected)
 for key,x in a.items():
  assert math.isclose(x['change'],x['value']-x['previous'],abs_tol=1e-7)
  for name in ['se','se_change']:
   assert x[name] is None or x[name]>=0
 for indicator in ['Employment','Unemployment rate']:
  x=a[indicator];se=x['se_change'];delta=x['change']
  report.append(dict(geography=g['geography'],indicator=indicator,change=delta,se_change=se,lo=delta-1.95996398454*se,hi=delta+1.95996398454*se,z=delta/se))
assert sum(g['target_households_2025'] for g in data['geographies'][1:])==63595
# Published Table 4.1 reports Canada 63,596; province rows sum to 63,595.
assert math.isclose(math.sqrt(.96*.0736/400),.013290598,abs_tol=1e-8)
assert math.isclose(.05*math.sqrt(.04/.96)*math.sqrt(.955/.045),.0470185,abs_tol=1e-6)
assert math.isclose(1e6*.9*.09/100+4e6*.9*.16/200,3690)
assert math.isclose(25000**2*2*(1-.7),375000000,abs_tol=1e-6)
rng=np.random.default_rng(338);U=np.array([[2,4,3,7],[6,10,8,12]]);L=np.full((2,4),50);rep=[]
for b in range(1000):
 m=np.array([np.bincount(rng.integers(0,4,3),minlength=4) for h in range(2)])
 weights=4/3*m;rep.append(float((weights*U).sum()/(weights*L).sum()))
variance=float(np.var(rep));assert variance>0
report.append(dict(synthetic_bootstrap=True,seed=338,original_rate=52/400,replicate_mean=float(np.mean(rep)),variance=variance,se=math.sqrt(variance)))
class Links(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.refs=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id'in a:self.ids.append(a['id'])
  for key in ['src','href']:
   if key in a:self.refs.append(a[key])
page=ROOT.parent/(ROOT.name+'.html');parser=Links();parser.feed(page.read_text(encoding='utf-8'))
assert len(parser.ids)==len(set(parser.ids)),'Duplicate IDs'
for ref in parser.refs:
 if ref.startswith('#'):assert ref[1:] in parser.ids,ref
 elif not urlparse(ref).scheme:assert (page.parent/unquote(ref.split('#')[0])).exists(),ref
(ROOT/'data'/'validation-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('PASS: 11 geographies; count identities, rate denominators, changes, SEs, worked examples, bootstrap, IDs and local paths.')
for item in report[:3]: print(item)
