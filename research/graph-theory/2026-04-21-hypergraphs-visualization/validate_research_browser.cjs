/* Existing Playwright/Edge; serve repository root on :8765. No new dependency. */
const {chromium}=require('playwright'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const R=require('./research-math.js'),M=require('./math.js');
const target=process.env.HYPERGRAPH_TEST_URL||'http://127.0.0.1:8765/research/graph-theory/2026-04-21-hypergraphs-visualization.html',out=path.join(__dirname,'qa');
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'}),page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],missing=[],checks=[],layout=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('response',r=>{if(r.status()>=400)missing.push(r.url());});
 await page.goto(target);const click=async id=>page.locator('#'+id).click(),contains=async(id,s)=>assert.ok((await page.locator('#'+id).innerText()).includes(s),id+' contains '+s);
 assert.equal(await page.locator('.rep-panel').count(),12);assert.equal(await page.locator('#definitions #four-representations').count(),1);
 assert.ok(await page.evaluate(()=>!!(document.querySelector('#four-representations').compareDocumentPosition(document.querySelector('#incidence'))&Node.DOCUMENT_POSITION_FOLLOWING)));
 for(let row=0;row<3;row++){
   const host=page.locator('#rep-row-'+row),h=R.gallery[row],matrix=M.incidence(h);
   const payloads=await host.locator('svg').evaluateAll(es=>es.map(e=>({vertices:JSON.parse(e.dataset.vertices),incidence:JSON.parse(e.dataset.incidence)})));
   assert.deepEqual(payloads,Array.from({length:4},()=>({vertices:h.vertices,incidence:matrix})));
   const marks=await host.locator('.membership-mark').evaluateAll(es=>es.map(e=>({v:e.dataset.vertex,j:+e.dataset.edge})));
   assert.deepEqual(marks,h.edges.flatMap((e,j)=>e.map(v=>({v,j}))));
   const enclosed=await host.locator('.rep-type-0 svg').evaluate(svg=>{
     const inside=([x,y],ps)=>{let c=false;for(let i=0,j=ps.length-1;i<ps.length;j=i++){const [xi,yi]=ps[i],[xj,yj]=ps[j];if((yi>y)!==(yj>y)&&x<(xj-xi)*(y-yi)/(yj-yi)+xi)c=!c;}return c;};
     const vertices=[...svg.querySelectorAll('.rep-vertex')].map(g=>({v:g.dataset.vertex,p:[+g.querySelector('circle').getAttribute('cx'),+g.querySelector('circle').getAttribute('cy')]}));
     return [...svg.querySelectorAll('[data-polygon]')].map(e=>{const ps=e.dataset.polygon.split(' ').map(p=>p.split(',').map(Number));return vertices.filter(v=>inside(v.p,ps)).map(v=>v.v);});
   });assert.deepEqual(enclosed,h.edges,'Actual enclosure geometry '+row);
   const atoms=await host.locator('.rep-type-2 [data-mask]').evaluateAll(es=>es.map(e=>({mask:+e.dataset.mask,members:e.dataset.members?e.dataset.members.split(','):[]})));
   for(const atom of atoms)assert.deepEqual(atom.members,R.atoms(h).find(a=>a.mask===atom.mask)?.members||[]);
   if(row<2){const positions=await host.locator('.rep-type-2 svg').evaluate(svg=>{const circles=[...svg.querySelectorAll('.rep-edge circle')];return [...svg.querySelectorAll('[data-mask]')].map(g=>{const t=g.querySelector('text'),x=+t.getAttribute('x'),y=+t.getAttribute('y')-3;return {expected:+g.dataset.mask,actual:circles.reduce((m,c,i)=>m+(Math.hypot(x-+c.getAttribute('cx'),y-+c.getAttribute('cy'))<+c.getAttribute('r')?1<<i:0),0)};});});assert.ok(positions.every(p=>p.actual===p.expected),'Actual Venn region positions');}
   assert.deepEqual(await host.locator('.projected-pair').evaluateAll(es=>es.map(e=>e.dataset.pair.split(','))),M.twoSection(h));
 }
 await page.locator('#rep-row-2 .rep-type-1 [role=button][data-edge="1"]').focus();await page.keyboard.press('Enter');await contains('rep-row-2','already occur in another edge');assert.equal(await page.locator('#rep-row-2 .projected-pair.is-selected').count(),1);assert.equal(await page.locator('#rep-row-2 .atom-block.is-selected').count(),2);assert.equal(await page.evaluate(()=>document.activeElement.getAttribute('data-edge')),'1');await page.locator('#rep-row-2 [data-reset]').click();assert.equal(await page.locator('#rep-row-2 .projected-pair.is-selected').count(),0);
 checks.push('Exactly twelve panels inside Section 03; shared incidence payloads; geometric enclosure membership; Venn region positions; PAOH marks; atom contents; deduplicated pairs; keyboard-linked e2 selection and reset');
 assert.deepEqual(errors,[],'Initial script errors');assert.deepEqual(missing,[],'Initial resource errors');await page.locator('#bitcoin-fig1-incidence svg').waitFor();
 assert.equal(await page.locator('.bitcoin-verbatim').count(),4);assert.equal(await page.locator('#bitcoin img').count(),0);
 for(const a of ['Stephen Ranshous','Cliff A. Joslyn','Sean Kreyling','Kathleen Nowak','Nagiza F. Samatova','Curtis L. West','Samuel Winters'])assert.ok((await page.locator('.source-attribution').allTextContents()).every(t=>t.includes(a)));
 assert.equal(await page.locator('#bitcoin-fig1-incidence .directed-link').count(),18);assert.equal(await page.locator('#bitcoin-fig1-tx .directed-link').count(),12);assert.equal(await page.locator('#bitcoin-fig2 .directed-link').count(),14);await click('bitcoin-consolidation');assert.equal(await page.locator('#bitcoin-fig2 .directed-link').count(),18);await click('bitcoin-consolidation');
 for(const [id,circular]of [['bitcoin-fig4-linear',false],['bitcoin-fig4-circular',true]]){
   const data=await page.locator('#'+id).evaluate(el=>{const arcs=['E1','E2'].map(id=>({id,tail:[],head:[]}));el.querySelectorAll('.directed-link').forEach(e=>{for(const a of arcs){if(e.dataset.to===a.id)a.tail.push(e.dataset.from);if(e.dataset.from===a.id)a.head.push(e.dataset.to);}});return {arcs,exchanges:[...el.querySelectorAll('.directed-node circle[fill="#253a32"]')].map(e=>e.parentElement.dataset.node)};});
   const test=R.stb(...data.arcs,data.exchanges);assert.equal(test.circular,circular);assert.equal(test.linear,!circular);assert.equal(data.arcs.reduce((s,a)=>s+a.tail.length+a.head.length,0),10);
 }
 const generic=await page.locator('#bitcoin-fig4-generic').evaluate(el=>({vertices:[...el.querySelectorAll('.directed-node[data-type=address]')].map(e=>e.dataset.node).sort(),links:[...el.querySelectorAll('.directed-link')].map(e=>[e.dataset.from,e.dataset.to]).sort()}));
 assert.deepEqual(generic.vertices,[...R.sourceFigure4.addresses].sort());assert.deepEqual(generic.links,R.sourceFigure4.arcs.flatMap(a=>a.tail.map(v=>[v,a.id]).concat(a.head.map(v=>[a.id,v]))).sort());assert.equal(generic.links.length,19);
 for(const value of ['alpha','alphaPrime','beta','gamma','L1','L2','mixed']){await page.selectOption('#bitcoin-motif',value);await contains('bitcoin-motif-output',value==='L1'||value==='L2'?'not connected':'Connected');}
 checks.push('Four short attributed source excerpts; no screenshot embeds; Fig. 1–2 arc counts and consolidation; Fig. 4 source topologies satisfy Definition 1; every motif preset tested');
 for(const key of Object.keys(R.examples)){
   await page.selectOption('#btc-example',key);const n=R.getExample(key);
   assert.deepEqual(await page.locator('#btc-in-matrix tbody tr').evaluateAll(es=>es.slice(0,-1).map(tr=>[...tr.querySelectorAll('td button')].map(b=>+b.textContent))),R.incidence(n,'tail'));
   assert.deepEqual(await page.locator('#btc-out-matrix tbody tr').evaluateAll(es=>es.slice(0,-1).map(tr=>[...tr.querySelectorAll('td button')].map(b=>+b.textContent))),R.incidence(n,'head'));
   assert.equal(await page.locator('#btc-graph .directed-node[data-type=address]').count(),n.addresses.length);
   assert.equal(await page.locator('#btc-graph .directed-node[data-type=transaction]').count(),n.arcs.length);
   if(key==='linear')await contains('btc-detection','Linear STB: yes');if(key==='circular')await contains('btc-detection','Circular STB: yes');
 }
 await page.selectOption('#btc-example','linear');const before=await page.locator('#btc-sets').innerText();await page.locator('#btc-exchanges [data-address=A2]').click();await contains('btc-detection','Linear STB: no');assert.equal(await page.locator('#btc-sets').innerText(),before);
 await page.selectOption('#btc-example','one');await page.locator('#btc-out-matrix [data-address=A3]').click();assert.equal(await page.locator('#btc-graph .selected-incidence[data-from=E1][data-to=A3]').count(),1);await page.locator('#btc-out-matrix [data-address=A1]').click();assert.equal(await page.locator('#btc-graph .selected-incidence').count(),0);await contains('btc-incidence-selection','No such incidence');
 await page.locator('#btc-inputs [data-address=A1]').click();await page.locator('#btc-inputs [data-address=A2]').click();await click('btc-save');await contains('btc-message','at least one input');await click('btc-reset');
 await page.locator('#btc-inputs [data-address=A3]').click();await page.locator('#btc-allow-reuse').uncheck();await click('btc-save');await contains('btc-message','reuse is disabled');await page.locator('#btc-allow-reuse').check();await click('btc-save');await contains('btc-message','Transaction saved');assert.equal(await page.locator('#btc-in-matrix [data-address=A3]').innerText(),'1');
 for(let i=0;i<3;i++)await click('btc-add');await click('btc-add');await contains('btc-message','Four-transaction display cap');for(let i=0;i<5;i++)await click('btc-address');await contains('btc-message','Eight-address display cap');assert.equal(await page.locator('#btc-inputs button').count(),8);await click('btc-reset');
 await click('btc-enumerate');await contains('btc-experiment','2401 ordered pairs; 2353 connected; 24 connected pure patterns; 4 linear STBs; 22 circular STBs');
 checks.push('Five lab levels; exact input/output matrix cells; transaction/address identity; editable exchange exclusions; exact arrow selection; invalid empty side; reuse policy; save/add; transaction/address caps; exhaustive experiment output');
 const clockPage=await browser.newPage({viewport:{width:1440,height:1000}});await clockPage.clock.install();await clockPage.goto(target);await clockPage.clock.pauseAt(new Date(Date.now()+10000));await clockPage.selectOption('#btc-example','chain');await clockPage.locator('#btc-play').scrollIntoViewIfNeeded();await clockPage.waitForTimeout(200);await clockPage.locator('#btc-play').click();assert.equal(await clockPage.locator('#btc-play').getAttribute('aria-pressed'),'true');await clockPage.clock.runFor(4300);assert.equal(await clockPage.locator('#btc-play').getAttribute('aria-pressed'),'false');assert.ok((await clockPage.locator('#btc-phase').innerText()).includes('Phase 4/4'));await clockPage.close();
 await page.selectOption('#btc-example','chain');await click('btc-animation-reset');await click('btc-play');await page.waitForTimeout(100);await page.evaluate(()=>document.querySelector('#random').scrollIntoView({behavior:'instant'}));await page.waitForTimeout(200);assert.equal(await page.locator('#btc-play').getAttribute('aria-pressed'),'false');
 await page.emulateMedia({reducedMotion:'reduce'});await click('btc-animation-reset');await click('btc-play');await contains('btc-phase','Phase 1/4');assert.equal(await page.locator('#btc-play').getAttribute('aria-pressed'),'false');await page.emulateMedia({reducedMotion:'no-preference'});
 checks.push('Four-phase transaction animation finishes and pauses offscreen; reduced motion advances one readable phase');
 await page.locator('[data-figure="bitcoin-fig1-incidence"]').click();assert.equal(await page.locator('#research-figure-dialog').evaluate(e=>e.open),true);assert.equal(await page.locator('#figure-zoom-body .directed-link').count(),18);await page.locator('#figure-zoom-scale').fill('140');await page.locator('#figure-zoom-scale').dispatchEvent('input');assert.ok((await page.locator('#figure-zoom-body svg').boundingBox()).width>=980);assert.deepEqual(await page.evaluate(()=>[...document.querySelectorAll('[id]')].map(e=>e.id).filter((id,i,a)=>a.indexOf(id)!==i)),[]);await page.locator('#research-figure-dialog button').click();assert.equal(await page.locator('#figure-zoom-body svg').count(),0);
 checks.push('Vector magnifier enlarges source labels without raster images or duplicate SVG marker IDs');
 const duplicateIds=await page.evaluate(()=>[...document.querySelectorAll('[id]')].map(e=>e.id).filter((id,i,a)=>a.indexOf(id)!==i));assert.deepEqual(duplicateIds,[]);
 for(const width of [1920,1440,1366,1024,768,390]){
   await page.setViewportSize({width,height:1000});
   const metrics=await page.evaluate(()=>{
     const panels=[...document.querySelectorAll('.rep-panel')],rects=panels.map(e=>{const b=e.getBoundingClientRect();return {width:b.width,height:b.height};});
     const clipped=[...document.querySelectorAll('.rep-panel,.rep-panel figcaption,.bitcoin-verbatim,.bitcoin-study p,.bitcoin-study .math-lines>span')].filter(e=>e.scrollWidth>e.clientWidth+2).map(e=>e.textContent.slice(0,90));
     const svgText=[...document.querySelectorAll('#four-representations svg text,#bitcoin svg text')].filter(e=>{const b=e.getBBox(),v=e.ownerSVGElement.viewBox.baseVal;return b.x<v.x-1||b.y<v.y-1||b.x+b.width>v.x+v.width+1||b.y+b.height>v.y+v.height+1;}).map(e=>e.textContent);
     return {width:innerWidth,document:document.documentElement.scrollWidth,cardDimensions:rects,clipped,svgText,columns:getComputedStyle(document.querySelector('.rep-panels')).gridTemplateColumns.split(' ').length};
   });
   assert.ok(metrics.document<=width,'Page overflow '+width);assert.deepEqual(metrics.clipped,[],'Clipped content '+width);assert.deepEqual(metrics.svgText,[],'SVG text outside viewBox '+width);assert.ok(metrics.cardDimensions.every(r=>r.height===420&&Math.abs(r.width-metrics.cardDimensions[0].width)<1),'Matching twelve panel dimensions');assert.equal(metrics.columns,width>1100?4:width>560?2:1);layout.push(metrics);
   if([1440,768,390].includes(width)){
     for(const [id,name]of [['rep-row-2','gallery-complex'],['bitcoin-source-1','source-1'],['bitcoin-source-3','source-3'],['bitcoin-source-4','source-4'],['bitcoin-lab','lab']])await page.locator('#'+id).screenshot({path:path.join(out,`${name}-${width}.png`)});
   }
 }
 assert.deepEqual(errors,[]);assert.deepEqual(missing,[]);
 checks.push('All six requested widths; equal twelve-card dimensions; 4/2/1 columns; no page overflow, clipping, duplicate IDs, console errors, or missing resources');
 const report={status:'PASS',browser:'Installed Microsoft Edge',checks,layout,errors,missing,limitations:['No cross-browser matrix','Hidden-document handler remains covered by existing simulated visibility-event test']};fs.writeFileSync(path.join(out,'research-browser-audit.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
