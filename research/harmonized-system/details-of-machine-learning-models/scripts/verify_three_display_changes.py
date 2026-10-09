"""Focused localhost checks for column removal, centering and page provenance."""
from pathlib import Path
import json, tempfile, shutil
from browser_checks import ROOT, BROWSER, BASE, HARNESS
from chrome_audit_session import ChromeSession

CHECKS=r'''
 $('expand-all').click();
 const headers=[...d.querySelectorAll('.dataset thead th')].map(x=>x.textContent);
 check(headers.length===29&&!headers.includes('Source & vintage'),'only requested source column removed');
 check([...$('dataset-body').rows].every(x=>x.cells.length===29),'all 336 rows align with headers');
 check(csv.split('\n')[0].includes('data_source'),'CSV source field retained');
 check([...d.querySelectorAll('#follow td')].some(x=>x.textContent==='data_source'),'source field retained in full-record table');
 const centered=[];
 function measureTables(){
  for(const el of d.querySelectorAll('.table-scroll')){
   const box=el.getBoundingClientRect(),parent=el.parentElement.getBoundingClientRect();
   check(Math.abs((box.left+box.right-parent.left-parent.right)/2)<1,'centered wrapper: '+el.querySelector('caption')?.textContent);
   centered.push(el.querySelector('caption')?.textContent||'Table');
  }
 }
 measureTables();
 $('dataset-search').value='Canada 1001';dispatch($('dataset-search'),'input');measureTables();$('clear-filters').click();
 for(const control of d.querySelectorAll('.prediction-controls')){control.querySelector('input').value='Canada 1001';dispatch(control.querySelector('input'),'input');measureTables();control.querySelector('button').click();}
 check([...d.querySelectorAll('td.numeric')].every(x=>w.getComputedStyle(x).textAlign==='right'),'numeric alignment preserved');
 const box=d.querySelector('.page-provenance');check(d.querySelectorAll('.page-provenance').length===1,'one provenance box');
 check(box.textContent.includes('October 8, 2026')&&box.textContent.includes('Codex on Windows'),'verified creation date and harness');
 check(box.querySelectorAll('.provenance-item').length===4,'four provenance fields');
 check(box.querySelectorAll('.provenance-icon svg').length===1,'reference SVG model icon');
 const title=d.querySelector('.hero>div'),a=box.getBoundingClientRect(),b=title.getBoundingClientRect();
 check(a.right<=width&&a.left>=0,'provenance does not overflow');
 check(requested>800?a.left>=b.right:a.top>=b.bottom,'provenance does not overlap title');
 // Compare the reference component's computed styles without starting its app.
 const reference=await (await fetch('../../canada-and-provinces-trade-by-hs.html')).text();
 const parsed=new w.DOMParser().parseFromString(reference,'text/html');
 const refbox=parsed.querySelector('.page-provenance');
 const refFrame=d.createElement('iframe');refFrame.style.cssText='position:absolute;left:-10000px;width:500px;height:300px';
 const loaded=new Promise(resolve=>refFrame.onload=resolve);
 refFrame.srcdoc='<base href="'+new URL('../../canada-and-provinces-trade-by-hs.html',location.href)+'"><link rel="stylesheet" href="2b-canada-provinces-trade-by-hs/css/style.css"><link rel="stylesheet" href="2b-canada-provinces-trade-by-hs/css/explorer.css">'+refbox.outerHTML;
 d.body.append(refFrame);await loaded;
 const referenceStyle=refFrame.contentWindow.getComputedStyle(refFrame.contentDocument.querySelector('.page-provenance')),ours=w.getComputedStyle(box);
 const properties=['backgroundColor','borderTopWidth','borderTopColor','borderRightWidth','borderRightColor','borderRadius','paddingTop','paddingRight','fontSize','lineHeight','fontFamily','color','boxShadow'];
 const matched={};for(const key of properties){check(ours[key]===referenceStyle[key],'reference provenance '+key);matched[key]=ours[key];}
 check([...box.querySelectorAll('.provenance-icon')].map(x=>x.innerHTML).join('|')===[...refbox.querySelectorAll('.provenance-icon')].map(x=>x.innerHTML).join('|'),'exact reference icons');
 refFrame.remove();
 const displayChanges={datasetColumns:headers.length,centeredTables:new Set(centered).size,expandedModels:models.map(x=>x.id),provenance:box.textContent,referenceStyles:matched};
 '''

def run():
    out=ROOT/'results/browser';out.mkdir(exist_ok=True)
    harness=HARNESS.replace(' await new Promise(resolve=>w.requestAnimationFrame(resolve));',' void d.documentElement.offsetWidth;')
    harness=harness.replace(' const report={',CHECKS+' const report={displayChanges,')
    views=r'''if(view==='features'){$('main').replaceChildren($('inputs'));}
    if(view==='elastic'){const m=models[1],body=m.querySelector('.model-body'),h=[...body.querySelectorAll('h3')].find(x=>x.textContent.startsWith('E /')),keep=[];let node=h;while(node&&!node.textContent.startsWith('F /')){keep.push(node);node=node.nextElementSibling;}body.replaceChildren(...keep);$('main').replaceChildren(m);}
    '''
    harness=harness.replace(' document.getElementById(\'qa\').textContent=JSON.stringify(report);',views+' document.getElementById(\'qa\').textContent=JSON.stringify(report);')
    (ROOT/'results/browser-three-changes.html').write_text(harness,encoding='utf-8')
    profile=Path(tempfile.mkdtemp(prefix='chapter-qa-',dir=out/'profiles')).resolve()
    browser=ChromeSession(BROWSER,profile);reports=[]
    try:
        for width,view in [(1440,'top'),(390,'top'),(1440,'features'),(1440,'elastic'),(1440,'algorithm'),(1440,'worked'),(390,'worked'),(1440,'dataset')]:
            report=browser.capture(BASE+f'browser-three-changes.html?width={width}&view={view}',out/f'three-changes-{width}-{view}.png',width)
            report.update(requested_width=width,view=view);reports.append(report)
            (ROOT/'results/browser_three_changes.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
            print(width,view,report['status'],flush=True)
            assert report['status']=='PASS',report
    finally:
        browser.close()
        assert profile.parent==(out/'profiles').resolve() and profile.name.startswith('chapter-qa-')
        shutil.rmtree(profile)

if __name__=='__main__':run()
