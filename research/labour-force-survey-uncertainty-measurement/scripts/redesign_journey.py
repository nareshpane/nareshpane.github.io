"""Reproducible presentation revision; empirical inputs are never written."""
from pathlib import Path
import re, json, hashlib
import tempfile
import sys
ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / 'research/labour-force-survey-uncertainty-measurement'
PAGE = ROOT / 'research/labour-force-survey-uncertainty-measurement.html'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    original = PAGE.read_text(encoding='utf-8')
    baseline = Path(tempfile.gettempdir()) / 'lfs-redesign-before.html'
    if not baseline.exists():
        baseline.write_text(original, encoding='utf-8')
        manifest = {str(p.relative_to(ASSETS)): sha(p) for folder in ['data', 'sources'] for p in (ASSETS/folder).glob('*') if p.is_file()}
        manifest['js/statistics.js'] = sha(ASSETS/'js/statistics.js')
        (ASSETS/'figures/redesign-preservation-baseline.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    html = original
    if 'js/documentary.js' in html:
        print('Presentation revision already applied; no changes.')
        return
    html = html.replace('</head>', '<link rel="stylesheet" href="labour-force-survey-uncertainty-measurement/css/documentary.css"><script defer src="labour-force-survey-uncertainty-measurement/js/illustrations.js"></script><script defer src="labour-force-survey-uncertainty-measurement/js/documentary.js"></script></head>')
    # Keep the six exact titles, replacing prose only inside the opening grid.
    start=html.index('<div class="grid">'); end=html.index('</div></div>', start)+len('</div></div>')
    intro=html[start:end]
    intro=re.sub(r'(<b>[^<]+</b>)[^<]+', r'\1', intro)
    html=html[:start]+intro+html[end:]
    html=re.sub(r'<p class="source">Current design:.*?</p>\s*', '', html, count=1)
    html=re.sub(r'<div class="legend"><span class="tag">Official published inputs</span>.*?</div>', '', html, count=1)
    html=html.replace('<p>These labels are part of the analysis. A calculated interval is never described here as an officially published confidence interval. The data are a frozen extract; the live Statistics Canada table can later change.</p>', '')
    html=re.sub(r'<details id="contents">.*?</details>', '', html, count=1, flags=re.S)
    html=html.replace('<a href="#contents">All sections</a>', '<a href="#sources">Sources</a>')
    start=html.index('<figure class="journey widget"');end=html.index('</figure>',start)+len('</figure>')
    film='''<figure class="documentary widget" id="lfs-journey">
<div class="film-heading"><h3 id="journey-heading">Canada: a geography of households</h3><span>108-second conceptual documentary</span></div>
<svg class="film-stage" id="journey-scene" viewBox="0 0 1000 560" role="img" aria-labelledby="journey-scene-title journey-scene-desc"><title id="journey-scene-title">From Canada to a labour market estimate</title><desc id="journey-scene-desc">Canada, selected dwellings, households, interviews, processing, analysis and publication.</desc><text x="80" y="240">Canada → dwellings → interviews → estimates</text></svg>
<p class="film-caption" id="journey-caption" aria-live="polite">The ten provinces contribute to national monthly totals. Territorial results are separate.</p>
<div class="film-controls" role="group" aria-label="Survey documentary playback"><button type="button" id="journey-play">Play</button><button type="button" id="journey-pause">Pause</button><button type="button" id="journey-replay">Replay</button><button type="button" id="journey-prev">Previous</button><button type="button" id="journey-next">Next</button><output id="journey-time">0 / 108 s</output><label for="journey-progress">Timeline</label><input id="journey-progress" type="range" min="0" max="108" step="0.1" value="0" aria-label="Survey journey progress, in seconds"><a href="#sampling">Explore the uncertainty →</a></div>
<figcaption>Figure 1a. Real provincial outlines anchor a conceptual journey; neighbourhoods, strata, PSUs, dwellings, people and records are illustrative. Selection does not depend on the visible household characteristics. Panel colours identify selected dwelling groups, not population fractions or strata. Ten provinces contribute to national monthly totals; territorial estimates are separate three-month moving averages. Reduced-motion users can explore still scenes with Previous, Next and the timeline.</figcaption><p class="source"><a href="https://www150.statcan.gc.ca/n1/pub/71-543-g/71-543-g2025001-eng.htm#a4">2025 Guide §§4–7, 9</a>. Boundary geometry: Natural Earth, as in the reference research animation.</p></figure>
<div class="geographic-guide" id="geographic-guide" aria-label="Geographic design and dwelling-to-estimate illustrations"></div>'''
    html=html[:start]+film+html[end:]
    html=html.replace('Explore the monthly survey · seven stages','Explore the monthly survey · a continuous journey')
    # RHC prose now introduces one equation at a time; probabilities are explicitly conditional.
    html=html.replace(r'let \(M_g=\sum_{i\in g}M_i\). Then \(\pi_{i\mid G}=M_i/M_g\).', r'let the group size total and conditional selection probability be</p><div class="equation">\[M_g=\sum_{i\in g}M_i.\]</div><div class="equation">\[\pi_{i\mid G}=\frac{M_i}{M_g},\qquad i\in g\in G.\]</div><p>')
    # Move substantial inline mathematics out of paragraphs, including expanded details.
    def paragraph(match):
        p=match.group(0)
        def promote(m):
            tex=m.group(1)
            if len(tex)>55:
                return '</p><div class="equation">\\['+tex+'\\]</div><p>'
            return m.group(0)
        return re.sub(r'\\\((.*?)\\\)', promote, p, flags=re.S)
    html=re.sub(r'<p(?:\s[^>]*)?>.*?</p>', paragraph, html, flags=re.S)
    # Split equation lists at outermost spacing commands, never inside sums/cases.
    def multiline(m):
        tex=m.group(1); chunks=[]; depth=0; last=0;i=0
        while i<len(tex):
            if tex[i]=='{':depth+=1
            elif tex[i]=='}':depth-=1
            elif depth==0:
                token=next((s for s in [r'\qquad',r'\quad'] if tex.startswith(s,i)),None)
                if token:
                    chunks.append(tex[last:i].rstrip(' ,'));i+=len(token);last=i;continue
            i+=1
        if chunks:
            chunks.append(tex[last:]);tex=r'\begin{gathered}'+r'\\[5pt]'.join(chunks)+r'\end{gathered}'
        return r'\['+tex+r'\]'
    html=re.sub(r'\\\[(.*?)\\\]', multiline, html, flags=re.S)
    html=re.sub(r'<p>\s*</p>', '', html)
    PAGE.write_text(html, encoding='utf-8')
    # Retain all educational APIs and imputation/bootstrap interactions; retire only the old film.
    jsfile=ASSETS/'js/enhancements.js';js=jsfile.read_text(encoding='utf-8')
    start=js.index('const geos=');end=js.index('let donorTimer;',start)
    js=js[:start]+"const colours=['#28685d','#a34925','#496e41','#79588b','#315e85','#82641f'];\nconst reduced=matchMedia('(prefers-reduced-motion: reduce)');\n"+js[end:]
    jsfile.write_text(js,encoding='utf-8')
    # Replace identical person emoji with heterogeneous dwelling frame units.
    appfile=ASSETS/'js/app.js';app=appfile.read_text(encoding='utf-8')
    app=app.replace("$('population').innerHTML=Array.from({length:120},(_,i)=>`<span class=\"person ${i%10===0?'sampled':''}\" aria-hidden=\"true\">🧍</span>`).join('');", "$('population').innerHTML=Array.from({length:48},(_,i)=>`<svg class=\"dwelling-unit ${i%8===0?'sampled':''}\" viewBox=\"0 0 120 150\" aria-hidden=\"true\">${LFSArt.building(10,140,['house','duplex','midrise','townhouse'][i%4],i%4===3?.65:.9)}</svg>`).join('');")
    # A precision diagram uses information marks, not pretend sample members.
    app=app.replace('<span class="person ${i<Math.min(100,100/d)?\'sampled\':\'\'}" aria-hidden="true">🧍</span>', '<svg class="information-unit ${i<Math.min(100,100/d)?\'sampled\':\'\'}" viewBox="0 0 18 28" aria-hidden="true"><path d="M9 2V26M3 2H15M3 26H15" stroke="${i<Math.min(100,100/d)?\'#a54f24\':\'#a9bcb2\'}" stroke-width="2"/><circle cx="9" cy="14" r="3" fill="#28685d"/></svg>')
    appfile.write_text(app,encoding='utf-8')
    # Artwork must load before app.js; the film needs the map loaded first.
    html=PAGE.read_text(encoding='utf-8')
    tag='<script defer src="labour-force-survey-uncertainty-measurement/js/illustrations.js"></script>'
    html=html.replace(tag,'').replace('<script defer src="labour-force-survey-uncertainty-measurement/js/app.js">',tag+'<script defer src="labour-force-survey-uncertainty-measurement/js/app.js">')
    html=html.replace('Conceptual population of 120 person pictograms, with 12 sampled people highlighted','Heterogeneous dwelling frame: 48 illustrative dwellings, with six sampled dwellings highlighted')
    html=html.replace('Animate sampled people','Animate selected dwellings').replace('Pictograms illustrate selection through geographic structure; their number is not the LFS sampling fraction.', 'Different building types illustrate a dwelling frame. Highlighted dwellings represent a small conceptual sample; visible household attributes do not determine selection and the numbers are not the LFS sampling fraction.')
    PAGE.write_text(html,encoding='utf-8')
    print('Revised page, film, illustrations and mathematical typography; baseline:',baseline)

def prepare_tests():
    source=(ASSETS/'scripts/verify_browser.js').read_text(encoding='utf-8')
    prefix=source.split('const report={errors,failed,mathjaxRendered:')[0]
    prefix=prefix.replace("'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe','C:/Program Files/Google/Chrome/Application/chrome.exe'", "'C:/Program Files/Google/Chrome/Application/chrome.exe','C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'")
    prefix=prefix.replace("i<100&&!fs.existsSync(portFile)","i<200&&!fs.existsSync(portFile)")
    prefix=prefix.replace("stdio:'ignore'", "stdio:['ignore','ignore','pipe']")
    prefix=prefix.replace('let socket;',"browser.stderr.on('data',data=>{if(/ERROR|FATAL/.test(String(data)))console.error(String(data));});let socket;")
    prefix=prefix.replace("path.join(os.tmpdir(),'lfs-browser-')", "path.join(out,'.browser-profile-')")
    prefix=prefix.replace('/ERROR|FATAL/','/FATAL/')
    tests=r'''
assert(await ev("!!window.LFSJourney"));
const report={errors,failed,mathjaxRendered:await ev("document.querySelectorAll('mjx-container').length"),layouts:[],scenes:[],interactions:[]};
assert(report.mathjaxRendered>100,'MathJax must actually render for the equation audit');
await ev("document.getElementById('journey-scene').scrollIntoView({behavior:'instant',block:'center'})");await pause(400);
assert(await ev("LFSJourney.state().playing"),'Playback must start when entering viewport');
let initial=await ev("LFSJourney.state().elapsed");await pause(300);assert(await ev("LFSJourney.state().elapsed")>initial);
await ev("document.getElementById('journey-pause').click()");initial=await ev("LFSJourney.state().elapsed");await pause(150);assert.equal(await ev("LFSJourney.state().elapsed"),initial);
await ev("document.getElementById('journey-play').click()");await pause(150);assert(await ev("LFSJourney.state().elapsed")>initial);
await ev("document.getElementById('journey-next').click()");assert(await ev("LFSJourney.state().playing"),'Next must preserve automatic playback');
await ev("document.getElementById('top').scrollIntoView({behavior:'instant'})");await pause(200);assert.equal(await ev("LFSJourney.state().playing"),false);
report.interactions.push('viewport autoplay; offscreen suspension; pause/resume; Next while playing');
const nodes=await ev("document.querySelectorAll('#journey-scene *').length");
await ev("for(let i=0;i<50;i++){document.getElementById('journey-replay').click();document.getElementById('journey-next').click();document.getElementById('journey-prev').click();}LFSJourney.pause()");
assert.equal(await ev("document.querySelectorAll('#journey-scene *').length"),nodes);
await ev("document.getElementById('journey-replay').click();LFSJourney.pause()");assert.equal(await ev("LFSJourney.state().elapsed"),0);
report.interactions.push('50 rapid replay/next/previous cycles; stable node count; replay reset');
for(const [name,width,height,mobile]of[['desktop',1440,1000,false],['tablet',768,1024,true],['mobile',390,844,true],['narrow',320,800,true]]){
  await cdp('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile});
  await ev("document.querySelectorAll('details').forEach(d=>d.open=true);LFSJourney.pause()");await pause(120);
  const layout=await ev(`({width:innerWidth,clientWidth:document.documentElement.clientWidth,documentWidth:document.documentElement.scrollWidth,mathOverflow:[...document.querySelectorAll('mjx-container')].flatMap((e,i)=>{const b=e.getBoundingClientRect(),m=e.querySelector('mjx-math')?.getBoundingClientRect(),p=e.closest('.equation')?.getBoundingClientRect();if(!b.width||!m)return [];return ((p&&m.width>p.width+2)||(!p&&m.right>${width}-10))?[{index:i,width:m.width,available:p?.width,section:e.closest('section')?.id,tex:e.querySelector('mjx-assistive-mml')?.textContent}]:[]}),wideElements:[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>${width}+1&&!e.closest('.chart-frame,.table-wrap,mjx-assistive-mml,pre')).slice(0,12).map(e=>({tag:e.tagName,id:e.id,cls:String(e.className),width:e.getBoundingClientRect().width,right:e.getBoundingClientRect().right})),mathScrollbars:[...document.querySelectorAll('.equation,mjx-container')].filter(e=>['auto','scroll'].includes(getComputedStyle(e).overflowX)).length,navRows:new Set([...document.querySelectorAll('.toc a')].map(a=>Math.round(a.getBoundingClientRect().top))).size,navScroll:document.querySelector('.toc').scrollWidth>document.querySelector('.toc').clientWidth,duplicateIds:[...document.querySelectorAll('[id]')].map(e=>e.id).filter((id,i,a)=>a.indexOf(id)!==i),missingAnchors:[...document.querySelectorAll('a[href^="#"]')].map(a=>a.getAttribute('href')).filter(s=>s.length>1&&!document.getElementById(s.slice(1)))})`);
  report.layouts.push({name,...layout});assert.equal(layout.navScroll,false);assert.deepEqual(layout.duplicateIds,[]);assert.deepEqual(layout.missingAnchors,[]);assert.equal(await ev("!!document.getElementById('contents')"),false);
  assert(layout.documentWidth<=width+1,JSON.stringify(layout));assert.deepEqual(layout.mathOverflow,[],JSON.stringify(layout));assert.equal(layout.mathScrollbars,0);
  if(name==='desktop')assert.equal(layout.navRows,2);
  for(const [t,label]of [[0,'canada'],[7.5,'ontario'],[10.5,'bc'],[13.5,'prairie'],[17.5,'pei'],[22.5,'yukon'],[31,'zoom'],[38,'selection'],[47,'rotation'],[55,'households'],[66,'interview'],[81,'imputation'],[90,'analysis'],[102,'release'],[108,'reliability']]){
    await ev(`LFSJourney.setTime(${t});document.getElementById('journey-scene').scrollIntoView({behavior:'instant',block:'center'})`);await pause(30);
    const state=await ev("LFSJourney.state()");report.scenes.push({name,label,...state});
    if(['desktop','mobile'].includes(name)){const shot=await cdp('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(out,`redesign-${name}-${label}.png`),Buffer.from(shot.data,'base64'));}
  }
  for(const [id,label] of [['top','intro'],['sampling-units','math'],['imputation','imputation-math'],['weights','weights-math'],['variance','variance-math'],['bootstrap','bootstrap-math'],['examples','examples-math']]){await ev(`document.getElementById('${id}').scrollIntoView({behavior:'instant',block:'start'})`);const shot=await cdp('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(out,`redesign-${name}-${label}.png`),Buffer.from(shot.data,'base64'));}
  await ev("[...document.querySelectorAll('summary')].find(e=>e.textContent.includes('Rao–Hartley')).scrollIntoView({behavior:'instant',block:'start'})");const rhcShot=await cdp('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(out,`redesign-${name}-rhc.png`),Buffer.from(rhcShot.data,'base64'));
}
await ev("document.getElementById('journey-scene').scrollIntoView({behavior:'instant',block:'center'});LFSJourney.setTime(107.7);LFSJourney.play()");await pause(650);assert.equal(await ev("LFSJourney.state().elapsed"),108);assert.equal(await ev("LFSJourney.state().playing"),false);report.interactions.push('automatic completion at 108 seconds');
assert(await ev("document.getElementById('film-geo-ON').isPointInFill(new DOMPoint(350,388))"),'Neighbourhood anchor must lie inside Ontario');
await ev("for(let t=0;t<=108;t+=.25){LFSJourney.setTime(t);for(const el of document.querySelectorAll('#journey-scene [transform]'))if(/NaN|Infinity/.test(el.getAttribute('transform')))throw Error('Invalid transform at '+t);}LFSJourney.setTime(0)");
report.interactions.push('Ontario anchor inside actual outline; 433 deterministic timeline samples have finite transformations');
await cdp('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
await ev("document.getElementById('journey-replay').click();document.getElementById('journey-play').click()");await pause(200);assert.equal(await ev("LFSJourney.state().playing"),false);assert.equal(await ev("LFSJourney.state().elapsed"),0);
await ev("document.getElementById('journey-next').click()");assert.equal(await ev("LFSJourney.state().stage"),1);assert.equal(await ev("LFSJourney.state().playing"),false);report.interactions.push('reduced motion disables autoplay; still-scene navigation works');
await ev("document.getElementById('donor-run').click()");await pause(100);assert.match(await ev("document.getElementById('donor-field').textContent"),/imputed/);await ev("document.getElementById('donor-reset').click()");assert.match(await ev("document.getElementById('donor-field').textContent"),/missing/);
await ev("document.getElementById('replicate-sim').requestSubmit();document.getElementById('imputation-sim').elements.mechanism.value='MNAR';document.getElementById('imputation-sim').requestSubmit()");assert.match(await ev("document.getElementById('replicate-result').textContent"),/1000 replicates/);assert.match(await ev("document.getElementById('imputation-result').textContent"),/Truth=20%/);
report.interactions.push('donor transfer/reset; weighted replicate laboratory; MNAR imputation experiment; all original laboratory and provincial controls');
await cdp('Emulation.setScriptExecutionDisabled',{value:true});await cdp('Page.reload');await pause(500);assert.equal(await ev("document.querySelectorAll('#case-table tbody tr').length"),11);report.noScriptTables=true;
await cdp('Emulation.setScriptExecutionDisabled',{value:false});await cdp('Page.navigate',{url:require('node:url').pathToFileURL(path.join(root,'research/labour-force-survey-uncertainty-measurement.html')).href});for(let i=0;i<50;i++){if(await ev("!!window.LFSJourney && !!document.getElementById('province-result')?.textContent.includes('calculated')"))break;await pause(100);}assert.match(await ev("document.getElementById('province-result').textContent"),/calculated/);assert(await ev("!!window.LFSJourney"));report.directFileOpening=true;
assert.equal(errors.length,0,JSON.stringify(errors));
fs.writeFileSync(path.join(out,'redesign-browser-results.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify({errors,failed,mathjaxRendered:report.mathjaxRendered,layouts:report.layouts,interactions:report.interactions},null,2));
}finally{if(socket)socket.close();browser.kill();server.close();await pause(800);if(temp.startsWith(path.join(out,'.browser-profile-')))try{fs.rmSync(temp,{recursive:true,force:true});}catch(e){console.error('Temporary test profile still locked:',temp);}}
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    (ASSETS/'scripts/verify_documentary.js').write_text(prefix+tests,encoding='utf-8')
    print('Prepared focused documentary, equation and preservation browser checks.')

def clean_legacy_css():
    p=ASSETS/'css/enhancements.css';css=p.read_text(encoding='utf-8')
    start=css.find('.journey{');end=css.find('.donor-demo{',start)
    if start>=0 and end>start:css=css[:start]+css[end:]
    css=re.sub(r'/\* Long inline expressions.*?mjx-container:not\(\[display="true"\]\)\{[^}]*\}', '', css,flags=re.S)
    p.write_text(css,encoding='utf-8')
    print('Removed retired film CSS and obsolete inline-math scrolling rule.')

def align_geography():
    p=ASSETS/'js/documentary.js';js=p.read_text(encoding='utf-8')
    for a,b in [('cx="344" cy="388"','cx="350" cy="388"'),('translate(314 372)','translate(320 372)'),('cityBox=[310,368,68,38]','cityBox=[316,368,68,38]'),('[314,372,36,48]','[320,372,36,48]'),('[313,371,39,46]','[319,371,39,46]'),('M344 388L275 385','M350 388L275 385')]:js=js.replace(a,b)
    p.write_text(js,encoding='utf-8')
    print('Aligned the conceptual neighbourhood anchor with inland southern Ontario.')

if __name__=='__main__':
    if '--prepare-tests' in sys.argv:prepare_tests()
    elif '--clean-css' in sys.argv:clean_legacy_css()
    elif '--align-geography' in sys.argv:align_geography()
    else:main()

