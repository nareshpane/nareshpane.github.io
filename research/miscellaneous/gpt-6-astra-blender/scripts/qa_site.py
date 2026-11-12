"""Inspect local HTTP, exact asset paths, WebGL, maths, video and mobile layout.

Start a repository-root HTTP server first; install Playwright and its Chromium.
Run: python scripts/qa_site.py --url http://127.0.0.1:8000
"""
import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urljoin,urlparse,unquote
from urllib.request import urlopen
from urllib.error import HTTPError
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
A=json.loads((ROOT/'data/analysis.json').read_text())
# Use a consistent software graphics/media stack for headless QA. On this
# machine VA-API selected alongside SwiftShader disconnected its decoder
# (PIPELINE_ERROR_DISCONNECTED); FFmpegVideoDecoder plays the same H.264 file.
# This does not bypass Chromium's autoplay policy or alter the public page.
BROWSER_ARGS=['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader','--disable-accelerated-video-decode']
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8000');p.add_argument('--preflight',action='store_true');args=p.parse_args()
URL=args.url+'/research/gpt-6-astra-blender.html'
report={'status':'RUNNING','url':URL,'headless_video_decode':'software; VA-API disabled for SwiftShader compatibility'}
output=ROOT/'data'/('preflight_qa.json' if args.preflight else 'local_qa.json')
output.write_text(json.dumps(report)+'\n')
class PageParser(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        self.links.extend(a[k] for k in ['src','href','poster'] if k in a)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='img':self.images.append(a)
html=(ROOT.parent/'gpt-6-astra-blender.html').read_text()
assert not re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',html),'Unexpected control character (check Python LaTeX escapes)'
parser=PageParser();parser.feed(html)
assert len(parser.ids)==len(set(parser.ids)),'Duplicate HTML IDs'
assert all(i.get('alt') for i in parser.images)
checked=[]
for href in sorted(set(parser.links)):
    if href.startswith(('https:','http:','mailto:')):continue
    if href.startswith('#'):
        assert href[1:] in parser.ids,href;continue
    url=urljoin(URL,href)
    try:
        with urlopen(url,timeout=20) as response:assert response.status==200,url
    except HTTPError as e:
        if args.preflight and e.code==404:continue
        raise
    checked.append(unquote(urlparse(url).path))
report['http_asset_paths_checked']=checked

def release_hero_decoder(page):
    # Test-only isolation: the newly autoplaying reel must not occupy a second
    # decoder during an unrelated clip or software-WebGL test. Its real,
    # unmodified autoplay behaviour is tested separately below.
    page.locator('#city-cinematic').evaluate('''v => {
        v.pause(); v.removeAttribute('autoplay'); v.preload='none';
        v.querySelectorAll('source').forEach(s=>s.removeAttribute('src'));
        v.load();
    }''')

with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=BROWSER_ARGS)
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    errors=[];failed=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
    page.on('requestfailed',lambda r:failed.append({'url':r.url,'failure':r.failure}))
    page.goto(URL,wait_until='networkidle',timeout=60000)
    release_hero_decoder(page)
    page.wait_for_function('window.MathJax && MathJax.startup && MathJax.startup.document && document.querySelector("mjx-container svg")')
    assert page.locator('mjx-merror').count()==0
    assert page.locator('[data-mjx-error]').count()==0
    page.wait_for_function('(n)=>+document.getElementById("graph-lab").dataset.routeSeconds === n',arg=A['route']['seconds'])
    assert abs(float(page.locator('#graph-lab').get_attribute('data-route-metres'))-A['route']['metres'])<1e-6
    page.locator('#close-bridge').check()
    page.wait_for_function('(n)=>+document.getElementById("graph-lab").dataset.routeSeconds === n',arg=A['closure_route']['seconds'])
    assert abs(float(page.locator('#graph-lab').get_attribute('data-route-metres'))-A['closure_route']['metres'])<1e-6
    page.locator('#origin').select_option('35');page.locator('#destination').select_option('36')
    page.wait_for_function('document.getElementById("graph-lab").dataset.routeSeconds === "252"')
    page.locator('#close-bridge').uncheck()
    page.wait_for_function('document.getElementById("graph-lab").dataset.routeSeconds === "36"')
    page.locator('#origin').select_option(str(A['source']));page.locator('#destination').select_option(str(A['target']))
    page.locator('[data-access-view="after"]').click()
    assert 'access_after' in page.locator('#access-image').get_attribute('src')
    page.locator('[data-access-view="before"]').click()
    # Release the main page before testing media. Chromium's software WebGL and
    # media decoders can contend in headless mode; use one fresh browser per
    # embedded clip so each assertion reflects page/media compatibility.
    browser.close()
    hero_checks=[]
    for width in [1440,390]:
        hero_browser=pw.chromium.launch(headless=True,args=BROWSER_ARGS)
        hero_page=hero_browser.new_page(viewport={'width':width,'height':1000 if width==1440 else 844},
                                        device_scale_factor=1,is_mobile=width==390,has_touch=width==390)
        hero_page.on('pageerror',lambda e:errors.append(str(e)))
        hero_page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
        hero_page.on('requestfailed',lambda r:failed.append({'url':r.url,'failure':r.failure}))
        hero_page.goto(URL,wait_until='networkidle',timeout=60000)
        hero=hero_page.locator('#city-cinematic')
        hero.scroll_into_view_if_needed()
        # No play() call and no relaxed autoplay launch policy: this is native
        # muted autoplay, including Chromium's mobile viewport behaviour.
        hero_page.wait_for_function('(v)=>!v.paused && v.currentTime>.15 && v.readyState>=2',arg=hero.element_handle(),timeout=30000)
        info=hero.evaluate('''v=>({duration:v.duration,width:v.videoWidth,height:v.videoHeight,
            muted:v.muted,defaultMuted:v.defaultMuted,autoplay:v.autoplay,loop:v.loop,
            controls:v.controls,playsInline:v.playsInline,played:v.currentTime,
            decodedFrames:v.getVideoPlaybackQuality().totalVideoFrames})''')
        assert [info['width'],info['height']]==[1280,720] and abs(info['duration']-24)<.01
        assert all(info[k] for k in ['muted','defaultMuted','autoplay','loop','controls','playsInline'])
        assert info['decodedFrames']>1
        poster=hero.evaluate('''async v=>{const i=new Image();i.src=v.poster;await i.decode();
            return {width:i.naturalWidth,height:i.naturalHeight};}''')
        assert poster=={'width':1280,'height':720}
        hero.evaluate('v=>v.currentTime=v.duration-.35')
        hero_page.wait_for_function('(v)=>v.currentTime<2 && !v.paused && !v.seeking',arg=hero.element_handle(),timeout=15000)
        info['loop_observed']=True
        hero.evaluate('v=>v.pause()')
        paused=hero.evaluate('v=>v.currentTime')
        hero_page.wait_for_timeout(300)
        assert hero.evaluate('v=>v.paused && v.currentTime')==paused
        hero.evaluate('v=>v.play()')
        hero_page.wait_for_function('(p)=>document.getElementById("city-cinematic").currentTime>p+.1',arg=paused)
        info.update(viewport_width=width,poster_decoded=poster,pause_resume=True)
        assert hero_page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        hero_checks.append(info)
        hero_browser.close()
    report['cinematic_hero']=hero_checks
    videos=[]
    for index in range(3):
        media_browser=pw.chromium.launch(headless=True,args=BROWSER_ARGS)
        media_page=media_browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
        media_page.on('pageerror',lambda e:errors.append(str(e)))
        media_page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
        media_page.on('requestfailed',lambda r:failed.append({'url':r.url,'failure':r.failure}))
        media_page.goto(URL,wait_until='networkidle',timeout=60000)
        release_hero_decoder(media_page)
        video=media_page.locator('video:not(#city-cinematic)').nth(index)
        source=video.locator('source').get_attribute('src')
        if args.preflight and not (ROOT.parent/source).exists():
            media_browser.close();continue
        video.scroll_into_view_if_needed()
        video.evaluate('(v)=>{v.muted=true;v.preload="auto";v.load();}')
        media_page.wait_for_function('(v)=>v.readyState>=2',arg=video.element_handle(),timeout=30000)
        video.evaluate('(v)=>v.play()')
        media_page.wait_for_function('(v)=>v.currentTime>0.1',arg=video.element_handle(),timeout=15000)
        info=video.evaluate('(v)=>({src:v.currentSrc,duration:v.duration,width:v.videoWidth,height:v.videoHeight,played:v.currentTime})')
        assert info['width']==960 and info['height']==540 and abs(info['duration']-14)<.1
        video.evaluate('(v)=>v.pause()')
        videos.append(info)
        media_browser.close()
    report['browser_videos']=videos
    viewer_browser=pw.chromium.launch(headless=True,args=BROWSER_ARGS)
    page=viewer_browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
    page.on('requestfailed',lambda r:failed.append({'url':r.url,'failure':r.failure}))
    page.goto(URL,wait_until='networkidle',timeout=60000)
    release_hero_decoder(page)
    page.locator('[data-model="city"]').click()
    page.wait_for_function('document.querySelector("model-viewer")?.dataset.verifiedLoad === "true"',timeout=60000)
    page.locator('model-viewer').screenshot(path=str(ROOT/'renders/qa_city_viewer.png'))
    # Exercise actual camera control without depending on pixel-identical lighting.
    orbit_before=page.locator('model-viewer').evaluate('(m)=>m.getCameraOrbit().theta')
    page.locator('model-viewer').locator('.userInput').focus()
    page.keyboard.press('ArrowRight');page.wait_for_timeout(500)
    orbit_after=page.locator('model-viewer').evaluate('(m)=>m.getCameraOrbit().theta')
    assert orbit_before!=orbit_after,'Keyboard camera control did not move'
    page.locator('[data-model="graph"]').click()
    page.wait_for_function('document.querySelector("model-viewer")?.dataset.verifiedLoad === "true"',timeout=60000)
    report['browser_models']={'city':'loaded and camera rotated','graph':'loaded'}
    layouts=[]
    for width in [1440,768,390,320]:
        page.set_viewport_size({'width':width,'height':950})
        page.wait_for_timeout(200)
        for state in [False,True]:
            page.locator('details#prompt').evaluate('(e,opened)=>e.open=opened',state)
            dimensions=page.evaluate('({viewport:innerWidth,document:document.documentElement.scrollWidth,body:document.body.scrollWidth})')
            assert dimensions['document']<=width+1,(width,state,dimensions)
        layouts.append({'width':width,'horizontal_overflow':False,'prompt_open_and_closed':True})
        if width in [1440,390]:
            page.locator('details#prompt').evaluate('(e)=>e.open=false')
            # A long smooth scroll from the viewer can outlast 200 ms and
            # capture an intermediate, sometimes blank compositor frame.
            # Settle the test viewport without changing public-page behavior.
            page.evaluate('scrollTo({top:0,left:0,behavior:"instant"})')
            page.wait_for_function('scrollY===0')
            page.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
            page.screenshot(path=str(ROOT/'renders'/f'qa_page_{width}.png'))
    report['responsive_layouts']=layouts
    report['mathjax_equations']=page.locator('mjx-container').count()
    # Browser media fetches can be cancelled after the test deliberately pauses
    # a verified clip. They are not load failures: readyState, dimensions,
    # duration and playback advancement were already asserted above.
    expected_media_aborts=[f for f in failed if f['url'].endswith('.mp4') and f['failure'] and 'ERR_ABORTED' in f['failure']]
    failed=[f for f in failed if f not in expected_media_aborts]
    report['console_errors']=errors;report['request_failures']=failed
    report['expected_media_request_aborts']=expected_media_aborts
    assert not errors,errors
    assert not failed,failed
    viewer_browser.close()
report['status']='PREFLIGHT_PASS' if args.preflight else 'PASS'
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='http_asset_paths_checked'},indent=2))
