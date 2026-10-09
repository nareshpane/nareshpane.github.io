"""Local browser and content-integrity checks for the collection navigation.

Serve the repository root on port 8000, then run this script. Results and
screenshots go in shared/navigation-checks; no datasets or models are changed.
"""
from pathlib import Path
import hashlib,json,re,sys,tempfile,shutil
from urllib.request import urlopen
from collection_navigation import COLLECTION,NAV_ITEMS,NAV_PATTERN,STYLESHEET

sys.path.append(str(COLLECTION/'details-of-machine-learning-models/scripts'))
from chrome_audit_session import ChromeSession

BASE='http://localhost:8000/research/harmonized-system/'
OUT=COLLECTION/'shared/navigation-checks'
BROWSER=Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe')
HARNESS=r'''<!doctype html><meta charset="utf-8"><style>body{margin:0}iframe{border:0;height:1100px}</style><iframe id="page"></iframe><pre id="qa" hidden></pre><script>
const params=new URLSearchParams(location.search),width=Number(params.get('width')),name=params.get('page');
const frame=document.getElementById('page');frame.style.width=width+'px';
const expected=ITEMS;
frame.addEventListener('load',async()=>{try{
 const w=frame.contentWindow,d=w.document,nav=d.querySelector('nav.collection-top-nav');
 const check=(x,msg)=>{if(!x)throw Error(msg)};
 check(nav,'shared top navigation');
 const links=[...nav.querySelectorAll('a')];check(links.length===7,'seven navigation links');
 check(!links.some(a=>a.textContent==='Main Research Page'),'Main Research Page absent from top navigation');
 check(links.every((a,i)=>a.getAttribute('href')===expected[i][0]&&a.textContent===expected[i][1]),'consistent order, URLs and descriptors');
 const selected=links.filter(a=>a.getAttribute('aria-current')==='page');
 check(selected.length===1&&selected[0].getAttribute('href')===name,'one correct current-page link');
 const style=w.getComputedStyle(selected[0]);
 check(style.backgroundColor==='rgb(228, 236, 238)'&&style.fontWeight==='700','pale blue stronger active item');
 check(style.borderRadius==='7px','rounded active item');
 check(parseFloat(style.paddingLeft)>=10&&parseFloat(style.paddingTop)>=7,'comfortable link padding');
 const rect=nav.getBoundingClientRect();check(rect.right<=width+1&&rect.left>=0,'navigation fits the viewport');
 const positions=links.map(a=>a.getBoundingClientRect());
 check(positions.every(r=>Math.abs(r.top-positions[0].top)<1),'single horizontal row');
 check(positions.slice(1).every((r,i)=>r.left>=positions[i].right),'no overlapping labels');
 check(w.getComputedStyle(nav).overflowX==='auto','horizontal scrolling enabled');
 if(nav.scrollWidth>nav.clientWidth){nav.scrollLeft=100;check(nav.scrollLeft>0,'horizontal navigation scroll works');}
 // Each active link must remain readable when brought into the scroll viewport.
 nav.scrollLeft=selected[0].offsetLeft-nav.offsetLeft;
 const active=selected[0].getBoundingClientRect();check(active.left>=rect.left-1&&active.right<=rect.right+1,'active item is visible within scroll area');
 nav.scrollLeft=0;
 check(parseFloat(w.getComputedStyle(nav).fontSize)>=13,'readable font size');
 const result={status:'PASS',page:name,width,links:links.map(a=>({label:a.textContent,href:a.getAttribute('href'),active:a.getAttribute('aria-current')==='page'})),navigationWidth:nav.clientWidth,contentWidth:nav.scrollWidth,activeBackground:style.backgroundColor,pageWidth:d.documentElement.scrollWidth};
 document.getElementById('qa').textContent=JSON.stringify(result);
}catch(e){document.getElementById('qa').textContent=JSON.stringify({status:'FAIL',page:name,width,message:e.message});}});
frame.src='../../'+name;
</script>'''

def integrity():
    baseline=json.loads((COLLECTION/'details-of-machine-learning-models/results/browser_navigation_baseline.json').read_text(encoding='utf-8'))
    for name,snapshot in baseline['pages'].items():
        text=(COLLECTION/name).read_bytes().decode('utf-8')
        rest=re.sub(NAV_PATTERN,'',text,count=1,flags=re.S).replace(STYLESHEET,'',1)
        assert hashlib.sha256(rest.encode()).hexdigest()==snapshot['remaining_sha256'],'Non-navigation edit: '+name
    repo=COLLECTION.parents[1]
    for name,value in baseline['tracked'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==value,'Unrelated edit: '+name
    # The preceding presentation baseline also protects canonical calculations.
    calculation_baseline=json.loads((COLLECTION/'details-of-machine-learning-models/results/presentation-baseline.json').read_text())
    allowed={str((COLLECTION/name).relative_to(repo)).replace('\\','/') for name in baseline['pages']}
    for name,value in calculation_baseline.items():
        if name.replace('\\','/') not in allowed:
            assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==value,'Protected calculation changed: '+name
    return {'seven_pages_unchanged_outside_top_navigation_and_stylesheet_link':True,'unrelated_tracked_files_unchanged':len(baseline['tracked']),'calculations_datasets_and_models_unchanged':True}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    links=[]
    for href,label in NAV_ITEMS:
        with urlopen(BASE+href,timeout=15) as response:assert response.status==200;links.append({'href':href,'status':response.status})
    report={'integrity':integrity(),'link_statuses':links,'browser':[]}
    (OUT/'harness.html').write_text(HARNESS.replace('ITEMS',json.dumps(NAV_ITEMS)),encoding='utf-8')
    profiles=OUT/'profiles';profiles.mkdir(exist_ok=True)
    profile=Path(tempfile.mkdtemp(prefix='navigation-qa-',dir=profiles)).resolve()
    browser=ChromeSession(BROWSER,profile)
    try:
        for width in [1440,390]:
            for name,_ in NAV_ITEMS:
                result=browser.capture(BASE+f'shared/navigation-checks/harness.html?width={width}&page={name}',OUT/f'{width}-{Path(name).stem}.png',width)
                report['browser'].append(result)
                (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(width,name,result['status'],flush=True)
                assert result['status']=='PASS',result
    finally:
        browser.close()
        assert profile.parent==profiles.resolve() and profile.name.startswith('navigation-qa-')
        shutil.rmtree(profile)
    print('All seven URLs and fourteen page/viewport checks passed; research content unchanged.')

if __name__=='__main__':main()
