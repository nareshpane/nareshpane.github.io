"""Small editorial and responsive enhancements, safe to rerun."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1];page=ROOT.parent/(ROOT.name+'.html')
text=page.read_text(encoding='utf-8').replace('2017001/references-eng.htm','2017001/ref-biblio-eng.htm').replace('2025001-eng.htm#h2_4','2025001-eng.htm#a4')
if 'one-household discrepancy' not in text:
 text=text.replace('<div class="table-wrap" id="allocation-table">','<p class="small">Table 4.1 reports Canada at 63,596, while its displayed provincial rows sum to 63,595. This one-household discrepancy is retained as published; it has not been silently corrected.</p><div class="table-wrap" id="allocation-table">')
if 'case-reading' not in text:
 text=text.replace('<p>The table prioritizes Canada,', '''<details id="case-reading" open><summary>Read the five principal employment-change examples</summary><div>
<p><strong>Canada.</strong> The precise table change is −68.3 thousand, rather than the rounded headline −68 thousand. SE(change)=33.2 thousand gives Z=−2.057 and a calculated 95% normal interval [−133.37,−3.23] thousand. It excludes zero, but its upper endpoint is relatively close to zero. For the separate unemployment-rate change of +0.1 points, SE(change)=0.1 points gives Z=1 and an interval [−0.096,+0.296] points. These two indicators answer different questions and yield different tests.</p>
<p><strong>Alberta.</strong> Employment rose 23.1 thousand, with SE(change)=13.0 thousand. Z=1.777 and the interval [−2.38,+48.58] thousand includes zero. The reported increase remains the point estimate; this normal test does not distinguish it from zero at 5%. The −0.4-point unemployment-rate change also needs its own published SE, not the employment SE.</p>
<p><strong>British Columbia.</strong> Employment fell 20.2 thousand with SE(change)=12.1 thousand. Z=−1.669 and the interval [−43.92,+3.52] thousand includes zero. Neither the national significant decline nor the sign of the provincial movement substitutes for a province-specific test.</p>
<p><strong>Ontario.</strong> Employment fell 20.0 thousand, with SE(change)=21.3 thousand. Z=−0.939 and the interval [−61.75,+21.75] thousand is wide relative to the movement. A large employment level does not make every month-to-month movement precise.</p>
<p><strong>Quebec.</strong> Employment fell 49.4 thousand with SE(change)=17.1 thousand. Z=−2.889 and the interval [−82.92,−15.88] thousand excludes zero. The +0.4-point unemployment-rate change is a different contrast: with SE(change)=0.3 points it has Z=1.333 and a zero-containing interval. A falling employment level can coincide with an imprecisely estimated unemployment-rate movement because participation and denominators also change.</p>
<p class="small">All five examples use published inputs in Table 14-10-0287-01 and author-calculated normal intervals. Select the other indicators in Table 3 for all eleven geographies. Provincial employment movements need not add exactly to the national movement because published series are rounded and seasonally adjusted/reconciled.</p></div></details>
<p>The table prioritizes Canada,''')
# Keep mobile charts readable inside explicit horizontal scroll containers.
text=re.sub(r'(?<!<div class="chart-frame">)(<svg\b[^>]*class="chart"[^>]*>.*?</svg>)',r'<div class="chart-frame" tabindex="0" role="region" aria-label="Scrollable figure">\1</div>',text,flags=re.S)
text=text.replace('Figure 9. Official point estimates','Figure 9. On small screens, scroll the chart horizontally to read the labels. Official point estimates')
page.write_text(text,encoding='utf-8')
css=ROOT/'css'/'style.css';style=css.read_text(encoding='utf-8')
extra='\n.chart-frame{overflow-x:auto;max-width:100%;border-radius:9px}.chart-frame .chart text{font-size:15px}@media(max-width:600px){.chart-frame .chart{min-width:720px}.chart-frame #province-chart{min-width:900px}}\n'
if '.chart-frame{overflow-x:auto' not in style:style+=extra
css.write_text(style,encoding='utf-8')
print('Source links, case interpretation and contained mobile charts updated.')
test=ROOT/'scripts'/'verify_browser.js'
code=test.read_text(encoding='utf-8').replace("querySelectorAll('svg.chart')].filter(e=>e.getBoundingClientRect().right>innerWidth+1)","querySelectorAll('.chart-frame')].filter(e=>e.getBoundingClientRect().right>innerWidth+1)")
code=code.replace("!e.closest('.table-wrap,.equation,mjx-container,.toc,pre')","!e.closest('.chart-frame,.table-wrap,.equation,mjx-container,.toc,pre')")
test.write_text(code,encoding='utf-8')
