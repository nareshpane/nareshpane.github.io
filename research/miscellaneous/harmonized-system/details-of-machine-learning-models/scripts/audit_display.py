"""Inspect rendered precision and measured layout without changing calculation files.

Start `python -m http.server 8000` at the repository root, then run this script.
Use --before to save the pre-audit evidence. Chrome uses a disposable D: profile.
Decimal matches are diagnostic candidates, never automatic text replacements.
"""
from pathlib import Path
import json, re, tempfile, shutil, sys
import xml.etree.ElementTree as ET
from browser_checks import ROOT, BROWSER, BASE, HARNESS
from chrome_audit_session import ChromeSession

OUT=ROOT/'results/browser'
OUT.mkdir(exist_ok=True)
AUDIT=r'''
 $('expand-all').click();
 const decimalCandidates=[];
 const walk=d.createTreeWalker(d.body,w.NodeFilter.SHOW_TEXT);
 let text;
 while(text=walk.nextNode()){
   const parent=text.parentElement;
   if(!parent||parent.closest('script,style,pre,code,mjx-assistive-mml')||!parent.getClientRects().length)continue;
   const matches=text.textContent.match(/[-+]?\d[\d,]*\.\d{3,}(?:e[-+]?\d+)?/gi);
   if(matches)decimalCandidates.push({values:matches,context:text.textContent.slice(0,280),section:parent.closest('details')?.id||parent.closest('section')?.id,math:!!parent.closest('mjx-container')});
 }
 const tooltips=[...d.querySelectorAll('[title]')].filter(x=>/\d\.\d{3,}/.test(x.title)).map(x=>({text:x.textContent,title:x.title}));
 const mathCandidates=[...w.MathJax.startup.document.math].filter(x=>/\d\.\d{3,}/.test(x.math)).map(x=>x.math);
 const tables=[...d.querySelectorAll('.table-scroll')].map(x=>{
   const t=x.querySelector('table'),r=t?.querySelector('tbody tr');
   return {caption:t?.caption?.textContent,width:x.clientWidth,tableWidth:t?.getBoundingClientRect().width,parentWidth:x.parentElement.clientWidth,classes:x.className,columns:r?[...r.cells].map(c=>({text:c.textContent.slice(0,70),width:Math.round(c.getBoundingClientRect().width),whiteSpace:w.getComputedStyle(c).whiteSpace})):[]};
 });
 const paragraphs=models.map(m=>({model:m.id,panel:m.querySelector('.model-body').clientWidth,paragraph:[...m.querySelectorAll('.model-body>p')].map(x=>({width:x.clientWidth,maxWidth:w.getComputedStyle(x).maxWidth})).slice(0,2)}));
 const issues=[...d.querySelectorAll('.equation,.table-scroll,.charts,.model-body')].filter(x=>x.getBoundingClientRect().right>d.documentElement.clientWidth+1).map(x=>x.className);
 const wideShortCells=tables.flatMap(t=>t.columns.filter(c=>c.text.length<16&&c.width>300).map(c=>({caption:t.caption,...c})));
 for(const el of d.querySelectorAll('script[src],link[rel="stylesheet"]')){const url=el.src||el.href;check((await fetch(url)).ok,'script/stylesheet resolves: '+url);}
 check(w.getComputedStyle(d.body).backgroundColor==='rgb(247, 244, 238)','established theme stylesheet applied');
 const measured={decimalCandidates,mathCandidates,tooltips,tables,paragraphs,issues,wideShortCells,inspectedModels:models.map(m=>m.id)};
 check(!decimalCandidates.length&&!mathCandidates.length&&!tooltips.length,'no unformatted long decimals in prose, equations or tooltips');
 '''

def run():
    before='--before' in sys.argv
    tag='before' if before else 'after'
    # SVG coordinates and metadata are source numbers, not user-visible labels.
    # Read text elements separately without modifying canonical plot files.
    labels=[''.join(e.itertext()) for f in (ROOT/'results/figures').glob('*.svg') for e in ET.parse(f).getroot().iter() if e.tag.endswith('}text')]
    assert not [s for s in labels if re.search(r'\d\.\d{2,}',s)],'Review chart-label precision'
    audit=AUDIT
    if before:audit=audit.replace(" check(!decimalCandidates.length&&!mathCandidates.length&&!tooltips.length,'no unformatted long decimals in prose, equations or tooltips');",'')
    harness=HARNESS.replace(' const report={',audit+' const report={').replace("checks:['combined", "measurements:measured,checks:['combined")
    # Force responsive layout synchronously: headless Chrome can throttle the
    # iframe's requestAnimationFrame indefinitely before its report is written.
    harness=harness.replace(' await new Promise(resolve=>w.requestAnimationFrame(resolve));',' void d.documentElement.offsetWidth;')
    # Keep real DOM and styles; isolate the requested component for screenshots only.
    extra=r'''if(view.startsWith('model-')){const parts=view.split('-'),n=Number(parts[1])-1,m=models[n],body=m.querySelector('.model-body');if(parts[2]==='start'){body.replaceChildren(...[...body.children].slice(0,5));}else{const prefix=parts[2]==='metrics'?'H /':'E /',stop=parts[2]==='metrics'?'I /':'F /';const h=[...body.querySelectorAll('h3')].find(x=>x.textContent.startsWith(prefix));const keep=[];let node=h;while(node&&!node.textContent.startsWith(stop)){keep.push(node);node=node.nextElementSibling;}body.replaceChildren(...keep);}$('main').replaceChildren(m);}
 if(view==='features'){$('main').replaceChildren($('inputs'));}
 if(view==='comparison'){$('main').replaceChildren($('comparison'));}
 if(view==='follow'){$('main').replaceChildren($('follow'));}
 '''
    harness=harness.replace(" document.getElementById('qa').textContent=JSON.stringify(report);",extra+" document.getElementById('qa').textContent=JSON.stringify(report);")
    (ROOT/'results/browser-audit.html').write_text(harness,encoding='utf-8')
    views=[(1440,'model-'+str(k)) for k in range(1,7)]+[(390,'model-'+str(k)) for k in range(1,7)]+[(1440,'model-'+str(k)+'-start') for k in range(1,7)]+[(1440,'model-'+str(k)+'-metrics') for k in range(1,7)]+[(960,'features'),(390,'dataset'),(1440,'comparison'),(390,'follow')]
    if before:views=[(1440,'model-2'),(390,'model-6'),(960,'features')]
    if '--focus-record' in sys.argv:
        tag='record';views=[(390,'follow'),(1440,'follow')]
    reports=[]
    profiles=OUT/'profiles';profiles.mkdir(exist_ok=True)
    profile=Path(tempfile.mkdtemp(prefix='chapter-qa-',dir=profiles)).resolve()
    browser=ChromeSession(BROWSER,profile)
    try:
        for width,view in views:
            name=f'audit-{tag}-{width}-{view}'
            report=browser.capture(BASE+f'browser-audit.html?view={view}&width={width}',OUT/(name+'.png'),width)
            report.update(view=view,requested_width=width)
            reports.append(report)
            print(name,report['status'],flush=True)
            (OUT/f'audit-{tag}.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
            assert report['status']=='PASS',report
    finally:
        browser.close()
        assert profile.parent==profiles.resolve() and profile.name.startswith('chapter-qa-')
        shutil.rmtree(profile)
    print('All six sections expanded and measured; screenshots saved for the requested views.')

if __name__=='__main__':run()
