"""Inventory untouched inputs, extract PDF reference text and summarize SDMX vintages."""
from pathlib import Path
import hashlib, json, xml.etree.ElementTree as ET
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
NS = {'s': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure',
      'c': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common'}
def main():
    for d in ('data/processed', 'data/metadata', 'validation', 'references', 'charts', 'illustrations', 'animations', 'vendor'):
        (ROOT/d).mkdir(parents=True, exist_ok=True)
    inventory=[]
    for f in sorted(ROOT.glob('*.pdf')):
        reader=PdfReader(f)
        pages=[p.extract_text() or '' for p in reader.pages]
        (ROOT/'references'/f'{f.stem}.txt').write_text('\n'.join(f'\n--- PDF PAGE {i+1} ---\n{t}' for i,t in enumerate(pages)),encoding='utf-8')
        inventory.append(dict(file=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),pages=len(pages)))
    structure=ROOT/'cpi-stat-canada-18100004-SDMX/18100004_Structure.xml'
    tree=ET.parse(structure)
    inventory.append(dict(file=str(structure.relative_to(ROOT)),bytes=structure.stat().st_size,sha256=hashlib.sha256(structure.read_bytes()).hexdigest(),format='SDMX 2.1 structure and codelists'))
    codebooks={}
    for cl in tree.findall('.//s:Codelist',NS):
        codebooks[cl.attrib['id']]={}
        for c in cl.findall('s:Code',NS):
            name=c.find("c:Name[@{http://www.w3.org/XML/1998/namespace}lang='en']",NS)
            par=c.find('s:Parent/Ref',NS)
            codebooks[cl.attrib['id']][c.attrib['id']]={'label':name.text if name is not None else c.attrib['id'], 'parent':par.attrib.get('id') if par is not None else None}
    (ROOT/'data/metadata/codebooks.json').write_text(json.dumps(codebooks,indent=2,ensure_ascii=False),encoding='utf-8')
    for f in sorted(structure.parent.glob('*.xml')):
        if f==structure: continue
        count=0; series=0; lo='9999'; hi='0000'; prepared=''; geos=set(); attrs={}
        for event,e in ET.iterparse(f,events=('end',)):
            tag=e.tag.split('}')[-1]
            if tag=='Prepared': prepared=e.text
            if tag=='Obs':
                count+=1; t=e.attrib.get('TIME_PERIOD','');lo=min(lo,t);hi=max(hi,t)
            if tag=='Series':
                series+=1;geos.add(e.attrib.get('Geography'));attrs=e.attrib.copy()
            e.clear()
        inventory.append(dict(file=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),prepared=prepared,observations=count,series=series,first=lo,last=hi,geographies=sorted(geos),example_series_attributes=attrs))
    (ROOT/'data/metadata/input-inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print(json.dumps(inventory,indent=2))
if __name__=='__main__':main()
