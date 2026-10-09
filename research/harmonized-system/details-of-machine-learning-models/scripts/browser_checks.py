"""Localhost headless Chrome checks, using an isolated disposable profile.

Run after starting python -m http.server 8000 at the repository root.
No real Chrome profile or original page is opened or modified.
"""
from pathlib import Path
import json, re, subprocess, tempfile, shutil, os, urllib.request
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/browser';OUT.mkdir(exist_ok=True)
BROWSER=Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe')
BASE='http://localhost:8000/research/harmonized-system/details-of-machine-learning-models/results/'
HARNESS=r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}iframe{width:100%;height:100vh;border:0}</style></head><body><iframe id="page" src="../../details-of-machine-learning-models.html"></iframe><pre id="qa" hidden></pre><script>
const check=(a,s)=>{if(!a)throw Error(s)};
document.getElementById('page').addEventListener('load',async()=>{
try {
 const frame=document.getElementById('page'),w=frame.contentWindow,d=w.document;
 const requested=Number(new URLSearchParams(location.search).get('width'));
 frame.style.width=requested+'px';
 await new Promise(resolve=>w.requestAnimationFrame(resolve));
 await w.MathJax.startup.promise;
 const $=id=>d.getElementById(id);const dispatch=(el,type)=>el.dispatchEvent(new w.Event(type,{bubbles:true}));
 const visible=body=>[...body.rows].filter(r=>!r.hidden).length;
 const models=[...d.querySelectorAll('details.model')];
 check(models.length===6&&models.every(x=>!x.open),'six initially collapsed models');
 $('expand-all').click();check(models.every(x=>x.open),'Expand All');
 check(d.documentElement.scrollWidth<=d.documentElement.clientWidth+1,'expanded algorithms have no page-wide overflow');
 $('collapse-all').click();check(models.every(x=>!x.open),'Collapse All');
 models[0].open=true;models[1].open=true;check(models.filter(x=>x.open).length===2,'independent expansion');$('collapse-all').click();
 const rows=$('dataset-body');check(rows.rows.length===336&&visible(rows)===336,'336 static accessible rows');
 $('dataset-search').value='Canada 1001';dispatch($('dataset-search'),'input');check(visible(rows)===48,'country and HS4 search');
 $('filter-year').value='2024';dispatch($('filter-year'),'change');check(visible(rows)===12,'search plus year');
 $('filter-exporter').value='Canada';dispatch($('filter-exporter'),'change');check(visible(rows)===6,'exporter filter');
 $('filter-destination').value='China';dispatch($('filter-destination'),'change');check(visible(rows)===1,'destination filter');
 $('filter-sector').value='8703';dispatch($('filter-sector'),'change');check(visible(rows)===0,'conflicting HS search/filter gives zero');
 $('clear-filters').click();check(visible(rows)===336&&$('dataset-search').value==='','Clear Filters restores 336');
 for(let k=1;k<=6;k++){
   const control=d.querySelector('[data-body="pred-body-'+k+'"]');const sel=control.querySelector('select'),input=control.querySelector('input'),body=$('pred-body-'+k);
   check(visible(body)===168,'168 evaluation predictions');sel.value='2024';dispatch(sel,'change');check(visible(body)===84,'84 validation predictions');
   input.value='Canada 1001';dispatch(input,'input');check(visible(body)===12,'prediction country/sector search');
   control.querySelector('button').click();check(visible(body)===168,'prediction clear');
   sel.value='2025';dispatch(sel,'change');check(visible(body)===84,'84 forecast predictions');
   check([...body.rows].filter(r=>!r.hidden).every(r=>r.cells[5].textContent==='N/A'&&r.cells[7].textContent==='N/A'),'2025 observed outcomes/errors missing');
   control.querySelector('button').click();
 }
 const scroll=d.querySelector('.table-scroll.dataset');check(scroll.scrollHeight>scroll.clientHeight&&scroll.scrollWidth>scroll.clientWidth,'vertical and horizontal dataset scrolling');
 scroll.scrollTop=160;scroll.scrollLeft=120;check(scroll.scrollTop>0&&scroll.scrollLeft>0,'both scrolling offsets');scroll.scrollTop=0;scroll.scrollLeft=0;
 check(w.getComputedStyle(rows.rows[0].cells[0]).position==='sticky','sticky ID');check(w.getComputedStyle(d.querySelector('.dataset th')).position==='sticky','sticky header');
 const images=[...d.images];images.forEach(i=>i.loading='eager');await Promise.all(images.map(i=>i.decode()));check(images.every(i=>i.naturalWidth>0),'all images decoded');
 const response=await fetch('../toy_trade_dataset.csv');const csv=await response.text();check(response.ok&&csv.includes('observation_id'),'CSV download resolves');
 const width=d.documentElement.clientWidth;check(d.documentElement.scrollWidth<=width+1,'no page-wide mobile overflow');
 check(d.querySelectorAll('mjx-container').length>100,'math rendering');check(d.querySelectorAll('mjx-merror').length===0,'no equation errors');
 check(w.chapterUI.errors.length===0,'no captured JavaScript errors');
 check(w.innerWidth===requested,'exact responsive frame viewport');
 const report={status:'PASS',viewport:w.innerWidth,documentWidth:width,models:6,datasetRows:336,predictionRowsEach:168,mathExpressions:d.querySelectorAll('mjx-container').length,mathErrors:0,images:images.length,consoleErrors:w.chapterUI.errors,checks:['combined search and all filters','clear restores all rows','six independent collapses and all-controls','prediction table year/search/clear','2025 observed N/A','sticky headers/ID','both table scroll axes','all images','local MathJax','CSV response','responsive no overflow']};
 // Component-focused views preserve real page DOM/CSS, with no browser scrolling dependency.
 const view=new URLSearchParams(location.search).get('view');
 if(view==='algorithm'){models[0].open=true;const m=models[0];const keep=[...m.querySelector('.model-body').children].slice(0,5);m.querySelector('.model-body').replaceChildren(...keep);$('main').replaceChildren(m);}
 if(view==='worked'){models[5].open=true;const m=models[5],body=m.querySelector('.model-body');const h=[...body.querySelectorAll('h3')].find(x=>x.textContent.startsWith('E /'));const keep=[];let node=h;while(node&&!node.textContent.startsWith('F /')){keep.push(node);node=node.nextElementSibling;}body.replaceChildren(...keep);$('main').replaceChildren(m);}
 if(view==='dataset'){$('main').replaceChildren($('dataset'));}
 document.getElementById('qa').textContent=JSON.stringify(report);
}catch(e){document.getElementById('qa').textContent=JSON.stringify({status:'FAIL',message:e.message,stack:e.stack});}
});</script></body></html>'''

def run():
    # Verify local serving before launching the browser.
    with urllib.request.urlopen(BASE+'../toy_trade_dataset.csv') as r:assert r.status==200
    harness=ROOT/'results/browser-harness.html';harness.write_text(HARNESS,encoding='utf-8')
    profiles=OUT/'profiles';profiles.mkdir(exist_ok=True)
    # Remove only disposable profiles created by this exact harness on previous runs.
    for owned in profiles.iterdir():
        assert owned.resolve().parent==profiles.resolve() and owned.name.startswith('chapter-qa-')
        shutil.rmtree(owned.resolve())
    reports=[]
    for width,view in [(1440,'top'),(390,'top'),(1440,'algorithm'),(390,'algorithm'),(1440,'worked'),(390,'worked'),(390,'dataset')]:
        profile=Path(tempfile.mkdtemp(prefix='chapter-qa-',dir=profiles)).resolve();name=f'{width}-{view}'
        args=[str(BROWSER),'--headless','--disable-gpu','--no-first-run','--no-default-browser-check',f'--user-data-dir={profile}',f'--window-size={max(width,500)},1000','--force-device-scale-factor=1',f'--screenshot={OUT/(name+".png")}', '--virtual-time-budget=30000','--dump-dom',BASE+f'browser-harness.html?view={view}&width={width}']
        result=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
        (OUT/(name+'.log')).write_text(result.stderr,encoding='utf-8')
        match=re.search(r'<pre id="qa" hidden="">(.*?)</pre>',result.stdout,re.S)
        if not match or not match.group(1):
            (OUT/(name+'-failure.html')).write_text(result.stdout,encoding='utf-8');raise RuntimeError('No browser report: '+name+' '+result.stderr[-1000:])
        from html import unescape
        report=json.loads(unescape(match.group(1)));report['requested_width']=width;report['view']=view;reports.append(report)
        print(json.dumps(report),flush=True)
        # Recursive cleanup is restricted to this exact, verified newly-created profile.
        assert profile.parent==profiles.resolve() and profile.name.startswith('chapter-qa-')
        shutil.rmtree(profile)
        if report['status']!='PASS':raise RuntimeError(report)
    (ROOT/'results/browser_verification.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
if __name__=='__main__':run()
