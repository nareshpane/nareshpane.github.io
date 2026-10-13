"""Insert official tables as progressive enhancement fallbacks; rerunnable."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1];page=ROOT.parent/(ROOT.name+'.html')
d=json.loads((ROOT/'data'/'september-2026.json').read_text());geos=d['geographies']
def table(caption,headers,rows):
 return '<table><caption>'+caption+'</caption><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(('<th>' if i==0 else '<td>')+str(c)+('</th>' if i==0 else '</td>') for i,c in enumerate(row))+'</tr>' for row in rows)+'</tbody></table>'
alloc=table('Table 1. 2025 target household allocation (Guide Table 4.1)',['Geography','Household target'],[[g['geography'],f"{g['target_households_2025']:,}"] for g in geos])
province=table('Table 2. September 2026 unemployment rates and published SEs (percentage points)',['Geography','Unemployment rate (%)','SE(level)','Change (points)','SE(change)'],[[g['geography']]+[g['indicators']['Unemployment rate'][k] for k in ['value','se','change','se_change']] for g in geos])
rows=[]
for g in geos:
 a=g['indicators']['Employment'];lo=a['change']-1.95996398454*a['se_change'];hi=a['change']+1.95996398454*a['se_change']
 rows.append([g['geography'],a['value'],a['change'],a['se_change'],f'[{lo:.2f}, {hi:.2f}]',f"{a['change']/a['se_change']:.2f}",'Excludes zero' if lo>0 or hi<0 else 'Includes zero'])
case=table('Table 3. Employment, thousands; official inputs, author-calculated 95% normal change intervals',['Geography','September','Change','SE(change)','95% interval','Z','Interpretation'],rows)
text=page.read_text(encoding='utf-8')
for id,content in [('allocation-table',alloc),('province-table',province),('case-table',case)]:
 text=re.sub(r'(<div class="table-wrap" id="'+id+r'">).*?(</div>)',lambda m:m[1]+content+m[2],text,flags=re.S)
page.write_text(text,encoding='utf-8')
print('Static official table fallbacks inserted.')
