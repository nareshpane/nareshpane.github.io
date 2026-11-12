"""Targeted refinement of existing SVG scenes and explanatory figures."""
from pathlib import Path
import json,hashlib,math,re,sys
ROOT=Path(__file__).resolve().parents[3]
ASSETS=ROOT/'research/labour-force-survey-uncertainty-measurement'
PAGE=ROOT/'research/labour-force-survey-uncertainty-measurement.html'
SNAP=ASSETS/'figures/refinement-baseline.json'
def baseline():
    if not SNAP.exists():
        files=[PAGE,*[p for directory in ['js','css','data','sources'] for p in (ASSETS/directory).glob('*') if p.is_file()]]
        SNAP.write_text(json.dumps({str(p.relative_to(ROOT)):{'sha':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text(encoding='utf-8') if p.suffix in ['.html','.js','.css'] and 'sources' not in p.parts else None} for p in files},ensure_ascii=False),encoding='utf-8')
def project(lon,lat):
    def lam(lon,lat):
        a,b=map(math.radians,[49,77]);n=math.log(math.cos(a)/math.cos(b))/math.log(math.tan(math.pi/4+b/2)/math.tan(math.pi/4+a/2));f=math.cos(a)*math.tan(math.pi/4+a/2)**n/n;r=f/math.tan(math.pi/4+math.radians(lat)/2)**n;q=n*math.radians(lon+96);return r*math.sin(q),-r*math.cos(q)
    bc=lam(-124,55);ab=lam(-114.5,55);s=(154.9-108.5)/(ab[0]-bc[0]);x,y=lam(lon,lat)
    return [round(108.5+(x-bc[0])*s,2),round(253.6-(y-bc[1])*s,2)]

def implement():
    h=PAGE.read_text(encoding='utf-8')
    if 'js/visual-refinements.js' in h:
        print('Targeted visual revision already applied.');return
    region='<script defer src="labour-force-survey-uncertainty-measurement/js/regional-scenes.js"></script>'
    visuals='<script defer src="labour-force-survey-uncertainty-measurement/js/visual-refinements.js"></script>'
    h=h.replace('<script defer src="labour-force-survey-uncertainty-measurement/js/documentary.js">',region+visuals+'<script defer src="labour-force-survey-uncertainty-measurement/js/documentary.js">')
    h=h.replace('</head>','<link rel="stylesheet" href="labour-force-survey-uncertainty-measurement/css/visual-refinements.css"></head>')
    h=h.replace('108-second conceptual documentary','133-second conceptual documentary').replace('0 / 108 s','0 / 133 s').replace('max="108" step="0.1"','max="133" step="0.1"')
    pattern=r'<div class="grid"><div class="card"><b>Dwelling</b>.*?</div></div>'
    old=re.search(pattern,h,re.S).group(0)
    definitions=re.findall(r'<b>(.*?)</b>(.*?)</div>',old,re.S)
    cards='<div class="grid conceptual-boxes">'+''.join('<div class="card"><b>'+title+'</b></div>' for title,desc in definitions)+'</div>'
    entries='<div class="concept-definitions">'+''.join('<p><strong>'+title+'.</strong> '+desc+'</p>' for title,desc in definitions)+'</div>'
    h=h.replace(old,cards+entries)
    h=h.replace('<div class="classification">','<div id="scenario-tabs" class="scenario-tabs" role="tablist" aria-label="Labour force classification examples"></div><div class="classification">',1)
    h=h.replace('id="classification-result" aria-live="polite"','id="classification-result" role="tabpanel" tabindex="0" aria-live="polite"')
    population_label='Heterogeneous dwelling frame: 48 illustrative dwellings, with six sampled dwellings highlighted'
    h=h.replace(population_label,'Target population, covered frame, strata, geographic PSUs, selected dwelling SSUs and eligible people')
    h=h.replace('Different building types locate illustrative dwellings in a heterogeneous frame. An amber window marks one selected dwelling, not an entire selected building. Visible household attributes do not determine selection and these numbers are not the LFS sampling fraction.', 'The target population defines whose labour market quantities are estimated; the sampling frame and units define how eligible people are reached. Boundaries and selections are conceptual. Visible household characteristics do not determine selection. Nine provinces generally use geographic clusters before dwellings; PEI uses one-stage dwelling selection, and the territories have a separate design.')
    start=h.index('<figure class="widget"><h3>A synthetic design map');end=h.index('</figure>',start)+len('</figure>')
    h=h[:start]+'''<figure class="widget synthetic-design"><h3>A synthetic design map</h3><div id="design-map" aria-label="Original geography, three strata, PSU formation and selection, then dwelling selection"></div><figcaption>Figure 3. Illustrative two-stage probability sampling. Actual LFS strata, cluster construction, and selection follow documented survey-design procedures. The three synthetic strata illustrate geographic or socioeconomic grouping, not the production algorithm. Compare the repeated-sampling designs in <a href="#lab-a">Laboratory A</a>.</figcaption></figure>'''+h[end:]
    note='''<aside class="note coverage-note"><strong>Settings do not establish coverage.</strong> The northern scene is an illustrative winter landscape. Provincial coverage excludes people on reserves and other Indigenous settlements, institutional residents and full-time Armed Forces members. Some very low-density remote dwellings are excluded from sampling, although the guide includes those remote populations in weighting control totals. A frame gap is distinct from a target-population exclusion, and a picture cannot quantify its bias. <a href="https://www150.statcan.gc.ca/n1/pub/71-543-g/71-543-g2025001-eng.htm#a4">2025 Guide §4</a>.</aside>'''
    pos=h.index('<aside class="note"><strong>Territories require')
    h=h[:pos]+note+h[pos:]
    PAGE.write_text(h,encoding='utf-8')
    # Retire only the superseded static generators, not their statistical logic.
    p=ASSETS/'js/app.js';js=p.read_text(encoding='utf-8')
    start=js.index("$('population').innerHTML=Array.from");end=js.index('const scenarios=',start)
    js=js[:start]+"$('population-play').onclick=()=>{const run=$('population').classList.toggle('running');$('population-play').textContent=run?'Clear selection emphasis':'Emphasize selected sampling units';$('population-play').setAttribute('aria-pressed',run);};\n"+js[end:]
    start=js.index("let map='';");end=js.index('let month=',start)
    js=js[:start]+js[end:]
    p.write_text(js,encoding='utf-8')
    print('Applied targeted HTML and illustration hooks; classifications and numerical functions unchanged.')

def prepare_browser():
    p=ASSETS/'scripts/verify_documentary.js';s=p.read_text(encoding='utf-8')
    s=s.replace('107.7','132.7').replace('),108);','),133);').replace('t<=108','t<=133').replace('at 108 seconds','at 133 seconds').replace('433 quarter','533 quarter').replace('433 deterministic','533 deterministic')
    s=s.replace("[[0,'canada'],[7.5,'ontario'],[10.5,'bc'],[13.5,'prairie'],[17.5,'pei'],[22.5,'yukon'],[31,'zoom'],[38,'selection'],[47,'rotation'],[55,'households'],[66,'interview'],[81,'imputation'],[90,'analysis'],[102,'release'],[108,'reliability']]", "[[0,'canada'],[7.5,'toronto'],[12.5,'north-ontario'],[17.5,'vancouver'],[22.5,'kelowna'],[27.5,'regina'],[32.5,'rural-saskatchewan'],[37.5,'edmonton'],[42.5,'pei'],[47.5,'yukon'],[56,'zoom'],[63,'selection'],[72,'rotation'],[80,'households'],[91,'interview'],[106,'imputation'],[115,'analysis'],[127,'release'],[133,'reliability']]")
    s=s.replace("'redesign-","'refinement-").replace('`redesign-','`refinement-').replace('redesign-browser-results.json','refinement-browser-results.json')
    extra=r'''
assert.equal(await ev("document.querySelectorAll('#scenario-tabs [role=tab]').length"),7);
assert(await ev("document.getElementById('scenario').closest('.controls').hidden"));
assert.equal(await ev("getComputedStyle(document.getElementById('scenario').closest('.controls')).display"),"none");
assert.equal(await ev("getComputedStyle(document.getElementById('scenario').closest('.controls')).display"),"none");
const classificationRules={hour:'Employed',absent:'Employed',search:'Unemployed',layoff:'Unemployed',future:'Unemployed',none:'Not in the labour force',student:'Not in the labour force'};
for(const [key,outcome]of Object.entries(classificationRules)){await ev(`document.getElementById('scenario-tab-${key}').click()`);assert.match(await ev("document.getElementById('classification-result').textContent"),new RegExp(outcome));assert.equal(await ev("document.querySelector('#scenario-tabs [aria-selected=true]').dataset.scenario"),key);}
await ev("document.getElementById('scenario-tab-hour').focus()");await cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight'});await cdp('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowRight',code:'ArrowRight'});assert.equal(await ev("document.getElementById('scenario').value"),'absent');await cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'End',code:'End'});assert.equal(await ev("document.getElementById('scenario').value"),'student');await cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'Home',code:'Home'});assert.equal(await ev("document.getElementById('scenario').value"),'hour');
for(const r of await ev("LFSRegions.regions.slice(0,7)"))assert(await ev(`document.getElementById('film-geo-${r.geo}').isPointInFill(new DOMPoint(${r.point[0]},${r.point[1]}))`),'Geographic anchor outside province: '+r.id);
assert.equal(await ev("document.querySelectorAll('.conceptual-boxes svg').length"),3);
assert.equal(await ev("document.querySelectorAll('.classification svg').length"),3);
assert.equal(await ev("document.querySelectorAll('.unit-flow .unit-step').length"),6);
assert.equal(await ev("document.querySelectorAll('.design-flow .unit-step').length"),5);
assert.equal(await ev("document.querySelectorAll('.support-figure').length"),3);
assert.equal(await ev("document.querySelectorAll('.weighting-figure [marker-end]').length"),0);
assert.match(await ev("document.querySelector('.weighting-figure').textContent"),/135/);
report.interactions.push('seven original classification scenarios; click and Arrow/Home/End keyboard access; seven city anchors within provincial outlines; all targeted figures present');
await ev("LFSJourney.setTime(70)");assert.match(await ev("document.getElementById('film-panel-0').textContent"),/Interview 6\/6/);
await ev("LFSJourney.setTime(75)");assert.match(await ev("document.getElementById('film-panel-6').textContent"),/Interview 1\/6/);assert.match(await ev("document.getElementById('film-panel-1').textContent"),/Interview 6\/6/);
'''
    s=s.replace("const nodes=await ev",extra+"\nconst nodes=await ev")
    s=s.replace("['examples','examples-math']", "['examples','examples-math'],['population','units'],['methodology','concepts'],['classification','tabs'],['design-map','synthetic'],['geographic-guide','weights'],['clustering-visual','clustering'],['replicate-visual','replication'],['error-visual','error']")
    # Wait for the three dynamically inserted weighting equations as well.
    s=s.replace("assert(await ev(\"!!window.LFSJourney\"));", "assert(await ev(\"!!window.LFSJourney\"));await pause(150);")
    (ASSETS/'scripts/verify_visual_refinements.js').write_text(s,encoding='utf-8')
    print('Prepared regional, keyboard, rotation and mathematical preservation browser tests.')

def audit_preservation():
    old=json.loads(SNAP.read_text(encoding='utf-8'));a=old[str(PAGE.relative_to(ROOT))]['text'];b=PAGE.read_text(encoding='utf-8')
    equations=lambda s:re.findall(r'\\[\[(](.*?)\\[\])]',s,re.S)
    assert equations(a)==equations(b),'Existing source equations changed'
    oldsections=re.findall(r'<section[^>]*id="([^"]+)"',a);newsections=re.findall(r'<section[^>]*id="([^"]+)"',b);assert oldsections==newsections
    assert re.findall(r'<form[^>]*id="([^"]+)"',a)==re.findall(r'<form[^>]*id="([^"]+)"',b)
    checked=0
    for name,entry in old.items():
        normalized=name.replace('\\','/')
        if any('/'+p+'/' in normalized for p in ['data','sources']) or name.endswith(('statistics.js','official-data.js','map-data.js','enhancements.js','illustrations.js')):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==entry['sha'],name;checked+=1
    for target in re.findall(r'(?:href|src)="([^"]+)"',b):
        if not target.startswith(('https:','http:','#','data:')):assert (PAGE.parent/target.split('#')[0]).exists(),target
    changed=[name.replace('\\','/') for name,entry in old.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=entry['sha']]
    print(json.dumps({'source_equations_preserved':len(equations(a)),'sections_preserved':len(oldsections),'protected_files_verified':checked,'mathematical_changes':'None','existing_files_modified':changed},indent=2))
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    baseline()
    if '--apply' in sys.argv:
        implement();sys.exit()
    if '--prepare-browser' in sys.argv:
        prepare_browser();sys.exit()
    if '--audit' in sys.argv:
        audit_preservation();sys.exit()
    for name,lon,lat in [('Toronto',-79.38,43.70),('Northern Ontario',-84,50),('Vancouver metro',-122.9,49.28),('Kelowna',-119.5,49.9),('Regina',-104.62,50.45),('Rural Saskatchewan',-106,51.5),('Edmonton',-113.5,53.55)]:print(name,project(lon,lat))
