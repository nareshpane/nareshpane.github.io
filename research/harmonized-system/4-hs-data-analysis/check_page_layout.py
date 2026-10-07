"""Headless rendering QA using installed Chrome; no website runtime dependency.

Run with a localhost server on port 8000. --smoke checks renderer availability.
Screenshots/profiles are temporary; compact reports stay beside the analysis.
No browser security protections are disabled and no real browser profile is used.
"""
from pathlib import Path
import os
import re
import json
import argparse
import subprocess
import tempfile
import shutil

ROOT = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8000/research/harmonized-system/machine-learning-trade-sector-prediction.html'
BASE = 'http://127.0.0.1:8000/research/harmonized-system/'
BROWSER = Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe')
REVIEW = ROOT / 'cache/browser-review'
PROFILE_ROOT = ROOT / 'cache/browser-profiles'
REVIEW.mkdir(parents=True,exist_ok=True)
PROFILE_ROOT.mkdir(parents=True,exist_ok=True)

def render(url, width, name, dump=False):
    profile = tempfile.mkdtemp(prefix='trade-render-profile-',dir=PROFILE_ROOT)
    args=[str(BROWSER),'--headless','--disable-gpu','--no-first-run','--no-default-browser-check',
          '--force-device-scale-factor=1',f'--user-data-dir={profile}',f'--window-size={width},1000',
          f'--screenshot={REVIEW/(name+".png")}', '--virtual-time-budget=12000',url]
    if dump:args.insert(-1,'--dump-dom')
    result=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace',
          timeout=55,creationflags=subprocess.CREATE_NO_WINDOW,env={k.upper():v for k,v in os.environ.items()})
    (REVIEW/(name+'-browser.log')).write_text(result.stderr,encoding='utf-8')
    if result.returncode or not (REVIEW/(name+'.png')).is_file():
        raise RuntimeError(f'Headless renderer failed ({result.returncode}): {result.stderr[-1800:]}')
    owned=Path(profile).resolve()
    assert owned.parent==PROFILE_ROOT.resolve() and owned.name.startswith('trade-render-profile-')
    shutil.rmtree(owned)
    return result.stdout

HARNESS = r'''<script>
window.addEventListener('load', async () => {
  if (window.MathJax && MathJax.startup) await MathJax.startup.promise;
  const models = [...document.querySelectorAll('details.model')];
  const initial = models.filter(x => x.open).length;
  document.getElementById('expand-all').click();
  const expanded = models.filter(x => x.open).length;
  document.getElementById('collapse-all').click();
  const collapsed = models.filter(x => x.open).length;
  const target = document.getElementById('model-06');
  target.open = true;
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  const content = target.querySelector('TARGET');
  const image=content.querySelector('img');
  if(image) await image.decode().catch(() => {});
  const result = {
    width:innerWidth, details:models.length, initialOpen:initial,
    expanded:expanded, collapsed:collapsed,
    documentWidth:document.documentElement.scrollWidth,
    viewportWidth:document.documentElement.clientWidth,
    mathRendered:document.querySelectorAll('mjx-container').length,
    mathErrors:document.querySelectorAll('mjx-merror').length,
    responsiveTableCount:document.querySelectorAll('.table-scroll').length,
    contentText:content.textContent.slice(0,120),
    figureNaturalWidth:target.querySelector('img').naturalWidth,
    bodyFont:getComputedStyle(document.body).fontSize
  };
  const report=document.createElement('pre');report.id='layout-result';report.hidden=true;
  report.textContent=JSON.stringify(result);document.body.appendChild(report);
  if(window.parent!==window) window.parent.postMessage({type:'trade-layout-result',result},location.origin);
  // Retain the real model/CSS layout but focus its component for screenshot QA.
  // This avoids Windows headless compositor blank captures after long-page scrolling.
  target.querySelector('.model-body').replaceChildren(content);
  document.getElementById('main').replaceChildren(target);
  window.scrollTo(0,0);
  await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
});
</script>'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke:
        render(URL,1440,'renderer-check')
        print('Installed browser rendered localhost successfully')
        return
    source=(ROOT.parent/'machine-learning-trade-sector-prediction.html').read_text(encoding='utf-8')
    cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    reports=[]
    for width in [1440,390]:
        def mobile_frame(url,name):
            # Windows Chrome has a 500px native window minimum. An iframe gives
            # the child page a real 390px CSS viewport without browser automation.
            frame=cache/(name+'-frame.html')
            frame.write_text('<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;background:#f7f4ee}iframe{display:block;border:0;width:390px;height:1000px}</style></head><body><iframe title="390 pixel page review" src="'+url+'"></iframe><script>addEventListener("message",e=>{if(e.origin!==location.origin||e.data.type!=="trade-layout-result")return;const p=document.createElement("pre");p.id="layout-result";p.hidden=true;p.textContent=JSON.stringify(e.data.result);document.body.appendChild(p)})</script></body></html>',encoding='utf-8')
            return BASE+'4-hs-data-analysis/cache/'+frame.name
        closed_url=mobile_frame(URL,'closed-390') if width==390 else URL
        render(closed_url,width,f'page-{width}-closed')
        for view,target in [('equations','.equation'),('chart','figure')]:
            fixture=source.replace('<head>','<head><base href="'+BASE+'">',1)
            fixture=fixture.replace('</body>',HARNESS.replace('TARGET',target)+'</body>')
            path=cache/f'layout-{width}-{view}.html';path.write_text(fixture,encoding='utf-8')
            fixture_url=BASE+f'4-hs-data-analysis/cache/{path.name}'
            if width==390:fixture_url=mobile_frame(fixture_url,f'{view}-390')
            dom=render(fixture_url,width,f'page-{width}-{view}',dump=True)
            match=re.search(r'<pre id="layout-result"[^>]*>(.*?)</pre>',dom,re.S)
            if not match:raise RuntimeError('Browser layout harness did not finish: '+view)
            from html import unescape
            record=json.loads(unescape(match.group(1)))
            record['view']=view;reports.append(record)
            assert record['details']==6 and record['initialOpen']==0
            assert record['expanded']==6 and record['collapsed']==0
            assert record['documentWidth']<=record['viewportWidth']+1,record
            assert record['mathRendered']>0 and record['mathErrors']==0,record
            if width==390:assert record['width']==390,record
    (ROOT/'browser_layout_checks.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2))
    print('Screenshots:',REVIEW)

if __name__=='__main__':main()
