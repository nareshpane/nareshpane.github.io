"""Focused Chromium check of revised geometry/materials and graph numerics."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
A=json.loads((ROOT/'data/analysis.json').read_text())
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader','--disable-accelerated-video-decode'])
    page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
    page.goto('http://127.0.0.1:8000/research/gpt-6-astra-blender.html',wait_until='networkidle')
    page.locator('#city-cinematic').evaluate("v=>{v.pause();v.removeAttribute('autoplay');v.querySelector('source').removeAttribute('src');v.load();}")
    page.wait_for_function('(n)=>+document.querySelector("#graph-lab").dataset.routeSeconds===n',arg=A['route']['seconds'])
    assert page.locator('#graph-lab circle').count()==64
    for closed,key in [(False,'route'),(True,'closure_route')]:
        page.locator('#close-bridge').set_checked(closed)
        page.wait_for_function('(n)=>+document.querySelector("#graph-lab").dataset.routeSeconds===n',arg=A[key]['seconds'])
        assert abs(float(page.locator('#graph-lab').get_attribute('data-route-metres'))-A[key]['metres'])<1e-6
    page.locator('[data-model="city"]').click()
    page.wait_for_function('document.querySelector("model-viewer")?.dataset.verifiedLoad==="true"',timeout=60000)
    viewer=page.locator('model-viewer');viewer.scroll_into_view_if_needed()
    materials=viewer.evaluate('v=>v.model.materials.map(m=>m.name)')
    assert any('Foliage' in m for m in materials) and any('Skin' in m for m in materials)
    assert any('Clothing' in m for m in materials)
    # A close export-only view: no master-only animation or extra source model.
    viewer.evaluate('''v=>{v.setAttribute('camera-target','114.1m 1.2m 113m');
      v.setAttribute('min-camera-orbit','auto auto 1m');v.setAttribute('camera-orbit','50deg 78deg 6m');v.jumpCameraToGoal();}''')
    page.wait_for_timeout(1200)
    viewer.screenshot(path=str(ROOT/'renders/qa_world_person.png'))
    assert not errors,errors
    report=dict(status='PASS',errors=errors,graph_nodes=64,route_seconds=A['route']['seconds'],
        closure_seconds=A['closure_route']['seconds'],city_loaded=True,tree_and_human_materials_present=True,materials=materials)
    (ROOT/'data/world_browser_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    browser.close();print('WORLD_BROWSER_PASS')
