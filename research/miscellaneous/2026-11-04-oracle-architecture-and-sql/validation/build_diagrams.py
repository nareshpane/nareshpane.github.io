"""Rebuild original self-contained printable architecture and workload SVGs."""
from pathlib import Path
from html import escape
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]/'svg'
def icon(name,x,y,w=100,h=71):
    raw=(ROOT/'illustrations'/f'{name}.svg').read_text(encoding='utf-8')
    geometry=raw[raw.index('<g '):raw.rindex('</svg>')]
    return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="0 0 180 128" aria-hidden="true">{geometry}</svg>'
def rect(x,y,w,h,fill):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#c2cbbd" stroke-width="1.3"/>'
def text(x,y,value,size=15,anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(value)}</text>'
def lines(x,y,values,size=14,dy=21):return ''.join(text(x,y+i*dy,v,size) for i,v in enumerate(values))
def arrow(path):return f'<path d="{path}" fill="none" stroke="#8b9e84" stroke-width="1.6" stroke-linejoin="round" marker-end="url(#arrow)"/>'
def svg(w,h,title,body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img"><title>{escape(title)}</title><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 1 L7 4 L0 7" fill="none" stroke="#8b9e84" stroke-width="1.3"/></marker></defs><g font-family="Segoe UI,Arial,sans-serif" fill="#283835">{body}</g></svg>'
b=text(20,32,'Source Systems',23)+text(20,59,'Care settings create different records at their own event grains.',15)
sources=[('Pharma-Nepal','pharmacy','One product dispensing line'),('Hospital systems','hospital','One inpatient episode'),('Physician systems','physician','One completed encounter'),('Emergency departments','emergency','One ED attendance')]
for i,(name,image,grain) in enumerate(sources):
    x=20+i*295;b+=rect(x,80,275,150,'#fcf3eb')+icon(image,x+85,88,105,75)+text(x+137.5,187,name,18,'middle')+text(x+137.5,211,grain,13,'middle')
# Four distinct source branches meet in the inter-row gap. Only the shared
# trunk has an arrowhead, ending on the first processing card's top border.
for source_center in [157.5,452.5,747.5,1042.5]:
    b+=f'<path class="source-branch" d="M{source_center} 230 C{source_center} 245 600 244 600 250" fill="none" stroke="#8b9e84" stroke-width="1.6" stroke-linecap="round"/>'
b+='<circle cx="600" cy="250" r="2.5" fill="#8b9e84"/>'
b+=arrow('M600 250 C600 265 200 259 200 280')
nodes=[('Secure Transmission','secure-transmission','#f2f7ee',['Authenticated senders deliver records','securely and receive an acknowledgment.'],['Encrypted interfaces · Files / APIs','Replay-safe delivery']),('Staging','staging','#eff5f8',['Incoming records are held and checked','before being trusted as warehouse data.'],['Immutable source payloads','Batch metadata · Quarantine']),('Integration','integration','#f5f1fa',['Records are standardized to refer to','consistent clients, products and services.'],['Person linkage · Provider identities','Code crosswalks · Canonical events'])]
for i,(name,image,fill,plain,technical) in enumerate(nodes):
    x=20+i*400;b+=rect(x,280,360,180,fill)+icon(image,x+270,286,75,54)+text(x+16,316,name,19)+lines(x+16,355,plain,14)+lines(x+16,415,technical,12,18)
    if i<2:b+=arrow(f'M{x+360} 370 H{x+400}')
b+=arrow('M1000 460 V480 H600 V500')
b+=rect(250,500,700,132,'#fcf3eb')+icon('warehouse',267,508,140,100)+text(425,532,'Data Warehouse',23)+lines(425,560,['Validated historical records are organized','into consistent structures for analysis.'],15)+text(425,612,'Event facts · Conformed dimensions · Bridges · History',12)
b+=arrow('M600 632 V654 H295 V680')
for x,name,image,fill,plain,tech in [(20,'Data Marts and Semantic Layer','marts','#f2f7ee',['Approved subject datasets and shared definitions','make reports comparable.'],'Population health · Medicines · Service utilization'),(630,'Approved Analytical Access','analyst','#eff5f8',['Authorized users ask questions through controlled','tools and reporting services.'],'SQL Developer · Python · R · Power BI · Reports')]:
    b+=rect(x,680,550,130,fill)+icon(image,x+450,684,84,60)+text(x+16,712,name,19)+lines(x+16,743,plain,14,20)+text(x+16,797,tech,12)
b+=arrow('M570 746 H630')+text(600,842,'Across every layer: governance · security · lineage · quality · monitoring',13,'middle')
(ROOT/'national-architecture.svg').write_text(svg(1200,862,'HealthDataNepal: source systems to approved analytical access',b),encoding='utf-8')
b=rect(20,15,450,220,'#faeee8')+rect(490,74,220,110,'#f8f4e7')+rect(730,15,450,220,'#e8f0f5')
b+=text(38,48,'OLTP',22)+text(38,74,'Online Transaction Processing',16)+icon('pharmacy',345,88,110,79)+lines(38,113,['Record one dispensing reliably.','Many concurrent short transactions.','Selective reads and writes.','Current operational state.'],14,25)
b+=icon('integration',558,78,84,60)+text(600,166,'Governed integration',15,'middle')
b+=text(748,48,'OLAP',22)+text(748,74,'Online Analytical Processing',16)+icon('analyst',1055,88,110,79)+lines(748,113,['Examine many historical records.','Broad scans, joins and summaries.','Conformed dimensions and history.','Controlled analytical access.'],14,25)
b+=arrow('M470 129 H490')+arrow('M710 129 H730')+text(20,256,'Conceptual comparison: workload and configuration determine performance; no benchmark is asserted.',12)
(ROOT/'workloads.svg').write_text(svg(1200,270,'Operational transactions feed a historical warehouse through governed integration',b),encoding='utf-8')
for name in ['national-architecture.svg','workloads.svg']:ET.parse(ROOT/name)
print('Rebuilt two self-contained original overview SVGs.')
