"""Exercise the rendered local research site in Chromium and save representative views."""
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = f'http://127.0.0.1:8765/research/miscellaneous/{ROOT.name}.html'
CAPTURES = ROOT / 'tests' / 'screenshots'
WIDTHS = [1920, 1440, 1280, 1024, 768, 390]


def ready(page, name):
    panel = page.locator(f'[data-lab="{name}"]')
    panel.scroll_into_view_if_needed(timeout=30000)
    panel.locator('.leaflet-container').wait_for(timeout=30000)
    page.wait_for_function("""name => document.querySelector(
        '[data-lab="'+name+'"] .legend').children.length > 0""", arg=name, timeout=60000)
    return panel


def control(panel, name):
    return panel.locator(f'[data-control="{name}"]')


def status(panel, text, timeout=30000):
    expect(panel.locator('.map-status')).to_contain_text(text, timeout=timeout)


def province_coverage(panel):
    return panel.locator('.map').evaluate("""map => {
        const canvas = map.querySelector('canvas');
        const outer = map.getBoundingClientRect(), inner = canvas.getBoundingClientRect();
        const scale = canvas.width / inner.width;
        const x = Math.max(0, Math.round((outer.left - inner.left)*scale));
        const y = Math.max(0, Math.round((outer.top - inner.top)*scale));
        const width = Math.min(canvas.width-x, Math.round(outer.width*scale));
        const height = Math.min(canvas.height-y, Math.round(outer.height*scale));
        const context = canvas.getContext('2d');
        const {data} = context.getImageData(x,y,width,height);
        let lowX=width, highX=0, lowY=height, highY=0;
        for (let row=0;row<height;row+=4) for(let col=0;col<width;col+=4) {
            const i=(row*width+col)*4;
            if(data[i+3]>190){
                lowX=Math.min(lowX,col);highX=Math.max(highX,col);
                lowY=Math.min(lowY,row);highY=Math.max(highY,row);
            }
        }
        return {width:outer.width,height:outer.height,outlineWidth:(highX-lowX)/scale,outlineHeight:(highY-lowY)/scale};
    }""")


def verify_shared_layout(page):
    sizes = page.locator('.intro-example-stack > :not(p)').evaluate_all(
        'panels => panels.map(panel => [panel.getBoundingClientRect().left, panel.offsetWidth])')
    assert len(sizes) == 4 and all(size == sizes[0] for size in sizes), sizes
    groups = page.locator('.code-group')
    assert groups.count() == 4
    for group in groups.all():
        panels = group.locator('.code')
        bounds = """panels => panels.map(panel => {
            const summary = panel.querySelector('summary');
            const {left, right, height} = summary.getBoundingClientRect();
            return [left, right, height];
        })"""
        before = panels.evaluate_all(bounds)
        assert len(before) > 1 and all(box == before[0] for box in before), before
        first = panels.first
        first.locator('summary').click()
        first.locator('.tok-key').first.wait_for()
        expect(first.locator('.collapse-label')).to_be_visible()
        assert panels.evaluate_all(bounds) == before
        first.locator('summary').click()
        expect(first.locator('.expand-label')).to_be_visible()


def browser_check():
    CAPTURES.mkdir(parents=True, exist_ok=True)
    summary = {'viewports': [], 'map_framing': {}, 'tested': [], 'local_asset_failures': [], 'page_errors': []}
    with sync_playwright() as pw:
        executable = Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
        kwargs = {'executable_path': str(executable)} if executable.exists() else {}
        browser = pw.chromium.launch(headless=True, **kwargs)
        context = browser.new_context(viewport={'width': 1440, 'height': 1000}, reduced_motion='reduce',
                                      accept_downloads=True)
        context.grant_permissions(['clipboard-read', 'clipboard-write'], origin='http://127.0.0.1:8765')
        page = context.new_page()
        page.on('pageerror', lambda error: summary['page_errors'].append(str(error)))
        page.on('response', lambda response: summary['local_asset_failures'].append(
            f'{response.status} {urlsplit(response.url).path}') if
            response.status >= 400 and urlsplit(response.url).hostname == '127.0.0.1' else None)
        page.on('console', lambda msg: summary['page_errors'].append(msg.text) if msg.type == 'error' else None)
        page.goto(URL, wait_until='domcontentloaded')
        page.locator('[data-metrics="wildfire"] article').first.wait_for(timeout=60000)
        assert page.title() == 'GeoPandas: Four Applied Geospatial Analyses'
        expect(page.locator('.crumb')).to_have_count(0)
        assert page.locator('.meta dd').all_inner_texts() == [
            'November 26, 2026', 'Codex on Windows', 'GPT Sol-6.2', 'High']
        assert page.locator('.meta dt').evaluate_all(
            "fields => fields.map(field => field.lastChild.textContent.trim())") == [
                'Created', 'Harness', 'Model', 'Reasoning']
        assert page.locator('.cards').evaluate(
            "cards => cards.getBoundingClientRect().bottom <= document.querySelector('#atlas').getBoundingClientRect().top")
        page.locator('.hero').screenshot(path=CAPTURES / 'title-metadata.png')
        page.locator('.cards').screenshot(path=CAPTURES / 'application-cards.png')
        for target in ('healthcare','wildfire','flood','socioeconomic'):
            page.locator(f'.app-card[href="#{target}"]').click()
            page.wait_for_function('target => location.hash === "#"+target', arg=target)
        page.evaluate("history.replaceState(null,'',location.pathname)")
        opening = ready(page, 'opening')
        page.locator('.open-text').screenshot(path=CAPTURES / 'atlas-workflow.png')
        initial = province_coverage(opening)
        assert initial['outlineHeight'] >= .82*initial['height'],initial
        summary['map_framing']['opening'] = initial
        outer = opening.bounding_box()
        assert math.isclose(outer['width'] / 1078.59375, .65, abs_tol=.002)
        assert math.isclose(initial['height'] / 850, .80, abs_tol=.002)
        assert opening.evaluate("""panel => {
            const {left,right} = panel.getBoundingClientRect();
            const parent = panel.parentElement.getBoundingClientRect();
            return Math.abs(left - parent.left - (parent.right - right)) < 1;
        }""")
        summary['opening_comparison'] = {'before_outer_width': 1078.59375, 'after_outer_width': outer['width'],
                                         'before_canvas_height': 850, 'after_canvas_height': initial['height']}
        page.evaluate('scrollTo(0,0)')
        page.screenshot(path=CAPTURES / 'opening-1440.png')
        opening.screenshot(path=CAPTURES / 'layered-atlas.png')
        stages = [([], 'Start with'), (['Hospital'], 'Facility records'), (['Hospital','Calgary roads'], 'Points locate'),
                  (['DA boundary','Population density'], 'Each dissemination area'), (['Calgary roads'], 'Clip to a study'),
                  (['Hospital','Population density'], 'Compare facility locations')]
        for index,(layers,description) in enumerate(stages):
            step=page.locator(f'[data-step="{index}"]')
            step.click()
            expect(step).to_have_attribute('aria-pressed','true')
            expect(page.locator('.steps button[aria-pressed=true]')).to_have_count(1)
            expect(step).to_have_css('background-color','rgb(252, 232, 216)')
            for other in page.locator('.steps button[aria-pressed=false]').all():
                expect(other).to_have_css('background-color','rgb(255, 255, 255)')
            expect(page.locator('#step-description')).to_contain_text(description)
            for layer in layers:
                expect(opening.locator('.legend')).to_contain_text(layer, timeout=30000)
        unselected = page.locator('[data-step="0"]')
        unselected.hover()
        expect(unselected).to_have_css('background-color','rgb(255, 255, 255)')
        expect(unselected).to_have_css('border-top-color','rgb(167, 121, 91)')
        unselected.focus()
        page.keyboard.press('Tab')
        expect(page.locator('[data-step="1"]')).to_be_focused()
        expect(page.locator('[data-step="1"]')).to_have_css('outline-width','3px')
        control(opening, 'das').check()
        opening.locator('.legend').get_by_text('DA boundary').wait_for()
        baseline_scale = opening.locator('.leaflet-control-scale-line').inner_text()
        opening.locator('.leaflet-control-zoom-in').click()
        expect(opening.locator('.leaflet-control-scale-line')).not_to_have_text(baseline_scale)
        canvas_map=opening.locator('.map')
        canvas_map.scroll_into_view_if_needed()
        box=canvas_map.bounding_box()
        pane=canvas_map.locator('.leaflet-map-pane')
        before=pane.evaluate('node => node.style.transform')
        page.mouse.move(box['x']+box['width']*.52,box['y']+box['height']*.46)
        page.mouse.down()
        page.mouse.move(box['x']+box['width']*.58,box['y']+box['height']*.46,steps=8)
        page.mouse.up()
        assert pane.evaluate('node => node.style.transform')!=before
        opening.locator('[data-reset]').click()
        expect(opening.locator('.leaflet-control-scale-line')).to_have_text(baseline_scale)
        summary['tested'].append('Card navigation, all six atlas stages, layer toggles, zoom, pan and reset')

        example=page.locator('.boundary-example')
        expect(example.locator('[data-output-file]')).to_contain_text('EPSG:3400')
        expect(example.locator('[data-boundary=geometry_type]')).to_have_text('Polygon')
        stack=page.locator('.intro-example-stack')
        stack.screenshot(path=CAPTURES / 'intro-python-example.png')
        example.locator('.boundary-outputs').screenshot(path=CAPTURES / 'intro-python-output.png')
        stack.locator('[data-copy]').click()
        expect(stack.locator('[data-copy]')).to_have_text('Copied')
        for pre in (stack.locator('.script-block pre'), example.locator('.output-panel pre')):
            expect(pre).to_have_css('font-variant-ligatures','none')
            expect(pre).to_have_css('font-size','15.5px')
            assert pre.evaluate('pre => getComputedStyle(pre).fontFamily').startswith('Consolas')
        assert 'alberta.plot()' in page.evaluate('navigator.clipboard.readText()')
        geometry=page.locator('#geometry-gallery')
        page.locator('p:has(+ #geometry-gallery)').screenshot(path=CAPTURES / 'geometry-introduction.png')
        for shape in ('Point','MultiPoint','LineString','MultiLineString','Polygon','MultiPolygon','GeometryCollection'):
            geometry.locator('select').select_option(shape)
            expect(geometry.get_by_role('img',name=shape,exact=True)).to_be_visible()
            assert geometry.locator('.geometry-notation').inner_text()
            if shape in ('MultiLineString','Polygon'):
                geometry.screenshot(path=CAPTURES / f'geometry-{shape.lower()}.png')
        projection_table=page.locator('[data-table=projections]')
        expect(projection_table.locator('td.numeric').first).to_have_css('text-align','right')
        projection_table.screenshot(path=CAPTURES / 'compact-projection-table.png')
        page.locator('#crs + p').screenshot(path=CAPTURES / 'foundations-paragraph.png')
        for chapter in ('health-model','fire-model','flood-model','weights'):
            page.locator(f'#{chapter} + p').screenshot(path=CAPTURES / f'{chapter}-paragraph.png')
        assert page.locator('.chapter .equation').count() >= 10

        predicate = page.locator('#predicate-lab')
        page.locator('#b-position').fill('46')
        assert 'touches true' in predicate.locator('p').inner_text()
        page.locator('#b-operation').select_option('Intersection')
        predicate.screenshot(path=CAPTURES / 'geometric-foundations.png')
        page.locator('figure:has(img[src$="projections.png"])').screenshot(path=CAPTURES / 'crs-comparison.png')
        index = page.locator('#index-lab input')
        index.fill('190')
        assert 'candidates' in page.locator('#index-lab p').inner_text()
        prob = page.locator('#probability-lab input')
        prob.fill('30')
        assert '26.0%' in page.locator('#probability-lab p').inner_text()
        path = page.locator('#network-lab')
        for _ in range(6):
            button=path.locator('[data-step-path]')
            if button.is_enabled():
                button.click()
        assert 'Shortest cost 8' in path.locator('p').inner_text()
        path.locator('input').check()
        for _ in range(6):
            button=path.locator('[data-step-path]')
            if button.is_enabled():
                button.click()
        assert 'Shortest cost 12' in path.locator('p').inner_text()
        summary['tested'].append('Geometry predicates, bounding-box diagram, exceedance calculator and graph')

        healthcare = ready(page, 'health')
        framed=province_coverage(healthcare)
        assert framed['height']>framed['width'] and framed['outlineHeight']>.82*framed['height'],framed
        summary['map_framing']['healthcare'] = framed
        control(healthcare, 'threshold').select_option('50')
        status(healthcare, 'within 50 km')
        control(healthcare, 'view').select_option('catchments')
        healthcare.locator('.legend').get_by_text('Nearest catchments').wait_for(timeout=30000)
        control(healthcare, 'view').select_option('scenario')
        healthcare.locator('.legend').get_by_text('Hypothetical location').wait_for(timeout=30000)
        control(healthcare, 'candidate').select_option('cover')
        status(healthcare, 'reduces')
        control(healthcare, 'category').select_option('hospital_v11')
        status(healthcare, 'reduces')
        control(healthcare, 'weight').select_option('age65')
        control(healthcare, 'candidate').select_option(index=4)
        assert healthcare.locator('.map-status').inner_text()
        healthcare.screenshot(path=CAPTURES / 'healthcare-placement.png')
        control(healthcare, 'category').select_option('clinic')
        control(healthcare, 'weight').select_option('population')
        control(healthcare, 'region').select_option('4806')
        control(healthcare, 'threshold').select_option('10')
        control(healthcare, 'candidate').select_option(label='Calgary')
        status(healthcare, '546,310 of 1,590,639 represented residents')
        control(healthcare, 'threshold').select_option('100')
        control(healthcare, 'candidate').select_option(label='Drumheller')
        status(healthcare, '448,129 of 1,590,639 represented residents')
        page.evaluate("""() => {
            const geoJSON = L.geoJSON;
            L.geoJSON = (data, ...options) => {
                const layer = geoJSON(data, ...options);
                if (data?.features?.[0]?.properties?.candidate_index != null) {
                    window.scenarioMarkerIndices = layer.getLayers().map(
                        marker => marker.feature.properties.candidate_index);
                }
                return layer;
            };
        }""")
        for part in (1, 2):
            label = f'Lloydminster (mapped part {part})'
            index = int(control(healthcare, 'candidate').select_option(label=label)[0])
            status(healthcare, f'{label} reduces')
            assert page.evaluate('window.scenarioMarkerIndices') == [index]
        summary['tested'].append('Healthcare threshold, age weights, catchments and candidate scenario')

        wildfire = ready(page, 'wildfire')
        annual=page.locator('[data-chart=fire]')
        annual.locator('[data-year="2023"] rect').click()
        expect(annual.locator('[data-year="2023"]')).to_have_attribute('aria-pressed','true')
        assert control(wildfire,'year').input_value()=='2023'
        annual.locator('[data-year="2012"] rect').click()
        status(wildfire,'No mapped perimeter layer was acquired')
        expect(annual.locator('[data-year="2011"] rect')).to_have_css('fill','rgb(191, 227, 181)')
        expect(page.locator('[data-chart=fire-fraction] rect').first).to_have_css('fill','rgb(191, 227, 181)')
        page.evaluate('document.activeElement.blur()')
        page.mouse.move(0,0)
        annual.locator('..').screenshot(path=CAPTURES / 'wildfire-annual-chart.png')
        page.locator('[data-chart=fire-fraction]').locator('..').screenshot(path=CAPTURES / 'wildfire-distribution-chart.png')
        wildfire.locator('[data-action="restart"]').click()
        assert control(wildfire,'year').input_value() == control(wildfire,'year').get_attribute('min')
        wildfire.locator('[data-action="play"]').click()
        first=control(wildfire,'year').input_value()
        page.wait_for_function("year => document.querySelector('[data-lab=wildfire] [data-control=year]').value!==year", arg=first)
        wildfire.locator('[data-action="pause"]').click()
        control(wildfire,'view').select_option('recurrence')
        status(wildfire, 'cells cross')
        control(wildfire,'grid').select_option('50')
        control(wildfire,'tau').select_option('5')
        status(wildfire, '5% burned')
        wildfire.screenshot(path=CAPTURES / 'wildfire-recurrence.png')
        control(wildfire,'view').select_option('intensity')
        control(wildfire,'h').select_option('50')
        status(wildfire, '50 km bandwidth')
        summary['tested'].append('Wildfire chronology playback, grid and bandwidth controls')

        flood = ready(page, 'flood')
        assert flood.get_attribute('data-map-layout')=='urban'
        expect(flood.locator('.map')).to_have_css('background-color','rgb(255, 248, 242)')
        assert flood.locator('.map').evaluate("""map => {
            const canvas = map.querySelector('canvas');
            const viewport = map.getBoundingClientRect(), pixels = canvas.getBoundingClientRect();
            const scale = canvas.width / pixels.width;
            const x = Math.round((viewport.left - pixels.left + 12)*scale);
            const y = Math.round((viewport.top - pixels.top + 12)*scale);
            return canvas.getContext('2d').getImageData(x,y,1,1).data[3] === 0;
        }""")
        control(flood,'roofs').check()
        control(flood,'minimum').select_option('0.25')
        status(flood, '25% asset')
        flood.screenshot(path=CAPTURES / 'flood-exposure.png')
        page.locator('[data-table=flood]').screenshot(path=CAPTURES / 'flood-table.png')
        control(flood,'view').select_option('coverage')
        flood.locator('.legend').get_by_text('Mapped flood hazard',exact=True).wait_for(timeout=30000)
        assert flood.get_attribute('data-map-layout')=='province'
        control(flood,'view').select_option('scenario')
        flood.locator('.legend').get_by_text('Hypothetical closure fragments').wait_for(timeout=30000)
        assert flood.get_attribute('data-map-layout')=='urban'
        summary['tested'].append('Flood coverage, asset visibility, threshold and disruption view')

        social = ready(page,'social')
        control(social,'indicator').select_option('low_income')
        control(social,'mode').select_option('clusters')
        control(social,'weights').select_option('knn')
        control(social,'inference').select_option('raw')
        social.locator('.legend').get_by_text('High–high',exact=True).wait_for(timeout=30000)
        social.locator('input[name=uid]').fill('48060101')
        social.get_by_role('button',name='Locate').click()
        status(social, '48060101')
        rows = json.loads((ROOT/'maps'/'socioeconomic'/'low_income_knn.json').read_text(encoding='utf-8'))
        chosen, point = max(rows.items(), key=lambda item: item[1][0])
        x_min = math.floor(min(-1, *(r[0] for r in rows.values())))
        x_max = math.ceil(max(1, *(r[0] for r in rows.values())))
        y_min = math.floor(min(-1, *(r[1] for r in rows.values())))
        y_max = math.ceil(max(1, *(r[1] for r in rows.values())))
        canvas = page.locator('[data-chart=moran] canvas')
        size = canvas.bounding_box()
        assert 1.35 < size['width']/size['height'] < 1.55, size
        canvas.click(position={'x': 58+(size['width']-76)*(point[0]-x_min)/(x_max-x_min),
                               'y': size['height']-47-(size['height']-65)*(point[1]-y_min)/(y_max-y_min)})
        status(social, chosen)
        control(social,'region').select_option('calgary')
        social.screenshot(path=CAPTURES / 'socioeconomic-clusters.png')
        page.locator('[data-chart=moran]').screenshot(path=CAPTURES / 'moran-scatterplot.png')
        control(social,'mode').select_option('composite')
        social.locator('[data-weight]').first.fill('9')
        status(social, 'Complete-case')
        sliders = social.locator('[data-weight]')
        for slider in sliders.all():
            slider.fill('0')
        expect(social.locator('[data-weight-label]')).to_have_text(['33.3%'] * 3)
        sliders.last.fill('8')
        expect(social.locator('[data-weight-label]')).to_have_text(['0.0%', '0.0%', '100.0%'])
        summary['tested'].append('Census indicators, weights, cluster inference, DA selection, scatterplot and composite')

        code=page.locator('details[data-script]').first
        code.scroll_into_view_if_needed()
        collapsed_width=code.bounding_box()['width']
        expect(code.locator('.expand-label')).to_be_visible()
        code.screenshot(path=CAPTURES / 'code-collapsed.png')
        code.locator('summary').click()
        code.locator('.tok-key').first.wait_for()
        expect(code.locator('.collapse-label')).to_be_visible()
        assert abs(code.bounding_box()['width']-collapsed_width)<1
        code.locator('[data-copy]').click()
        code.get_by_text('Copied',exact=True).wait_for()
        assert 'def prepare' in page.evaluate('navigator.clipboard.readText()')
        code.screenshot(path=CAPTURES / 'code-expanded.png')
        code.locator('summary').click()
        expect(code.locator('.expand-label')).to_be_visible()
        verify_shared_layout(page)
        group=page.locator('#wildfire .code-group')
        group.screenshot(path=CAPTURES/'code-group-collapsed.png')
        group.locator('summary').first.click()
        group.locator('.tok-key').first.wait_for()
        group.screenshot(path=CAPTURES/'code-group-expanded.png')
        group.locator('summary').first.click()
        for panel in page.locator('.map-panel, .boundary-figure, .projection-figure').all():
            expect(panel).to_have_css('border-top-color','rgb(189, 98, 104)')
            expect(panel).to_have_css('border-top-width','1px')
        for canvas in page.locator('.map-panel .map').all():
            expect(canvas).to_have_css('background-color','rgb(255, 248, 242)')
        for topic in ('healthcare','wildfire','flood','socioeconomic'):
            page.locator(f'[data-interpretation="{topic}"]').screenshot(path=CAPTURES/f'{topic}-interpretation.png')
        resource=page.locator('#reproducibility')
        resource.locator('.resource-grid').screenshot(path=CAPTURES / 'download-resources.png')
        assert resource.locator('.resource-heading img').evaluate_all(
            'icons => icons.length === 4 && icons.every(icon => icon.complete && icon.naturalWidth > 0)')
        for a in resource.locator('a[href]').all():
            href=a.get_attribute('href')
            assert page.request.head(URL.rsplit('/',1)[0]+'/'+href).ok, href
        with page.expect_download() as received:
            resource.get_by_role('link',name='Download CSV').first.click()
        assert received.value.suggested_filename == 'distances.csv'
        summary['tested'].append('Code expansion, highlighting, clipboard, local resource links and CSV download')

        for width in WIDTHS:
            page.set_viewport_size({'width':width,'height':900})
            page.goto(URL, wait_until='domcontentloaded')
            atlas = ready(page, 'opening')
            framing = province_coverage(atlas)
            assert framing['outlineHeight'] >= .80 * framing['height'], framing
            page.evaluate('scrollTo(0,0)')
            page.wait_for_timeout(250)
            dimensions=page.evaluate('({document:document.documentElement.scrollWidth, viewport:innerWidth})')
            assert dimensions['document']<=dimensions['viewport'],dimensions
            if width in (1920, 1024, 768, 390):
                verify_shared_layout(page)
                page.evaluate('scrollTo(0,0)')
            assert page.locator('.chapter > p').evaluate_all(
                'rows => rows.every(row => row.scrollWidth <= row.clientWidth + 1)')
            dimensions['card_columns']=page.locator('.cards').evaluate(
                'node => getComputedStyle(node).gridTemplateColumns.split(" ").length')
            assert dimensions['card_columns'] == (4 if width>850 else 2 if width>560 else 1),dimensions
            if width<=560:
                assert page.locator('.app-card').evaluate_all(
                    'cards => cards.every(card => card.getBoundingClientRect().height < 155)')
            dimensions['title_lines']=page.locator('h1').evaluate(
                'node => Math.round(node.offsetHeight/parseFloat(getComputedStyle(node).lineHeight))')
            if width >= 1280:
                assert dimensions['title_lines']==1,dimensions
            summary['viewports'].append(dimensions)
            if width != 1440:
                page.screenshot(path=CAPTURES/f'opening-{width}.png')
        toggle=page.get_by_role('button',name='Contents')
        toggle.click()
        page.locator('.toc a[href="#performance"]').click()
        page.wait_for_function("""() => document.querySelector('.toc a.active')?.hash === '#performance'""")
        assert toggle.get_attribute('aria-expanded')=='false'
        summary['tested'].append('Six responsive widths, desktop title wrapping, mobile contents and active-section highlighting')
        ready(page, 'health').screenshot(path=CAPTURES / 'healthcare-mobile.png')
        ready(page, 'opening').screenshot(path=CAPTURES / 'layered-atlas-mobile.png')
        page.locator('.intro-example-stack').screenshot(path=CAPTURES / 'intro-stack-mobile.png')
        page.locator('#wildfire .code-group').screenshot(path=CAPTURES / 'code-group-mobile.png')
        page.locator('.resource-grid').screenshot(path=CAPTURES / 'resource-grid-mobile.png')
        summary['chromium_version']=browser.version
        browser.close()
    assert not summary['local_asset_failures'],summary['local_asset_failures']
    assert not summary['page_errors'],summary['page_errors']
    (ROOT/'metadata'/'browser_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    browser_check()


