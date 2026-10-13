"""Small presentation pass: defaults, neutral rotation and redundant code removal."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[3]
assets = root / 'research/labour-force-survey-uncertainty-measurement'
page = root / 'research/labour-force-survey-uncertainty-measurement.html'
p = assets / 'js/app.js'
s = p.read_text(encoding='utf-8')
s = s.replace('function bindForm(id,fn,auto=true)', 'function bindForm(id,fn,auto=true,initialize=auto)').replace('if(auto)fn();return form;', 'if(initialize)fn();return form;')
s = s.replace('},false);', '},false,true);')
s = s.replace("const colors=['#16646b','#a54f24','#416d4f','#705775','#315e85','#7e6132'];", '')
s = s.replace('class="panel" style="background:${colors[((born%6)+6)%6]}"', 'class="panel${age===1?\' entering\':\'\'}"')
p.write_text(s, encoding='utf-8')
h = page.read_text(encoding='utf-8')
h = re.sub(r'<pre aria-label="Replicate-weight pseudocode">.*?</pre>\s*', '', h, flags=re.S)
h = re.sub(r'<details><summary>Python: reproduce the unequal-weight teaching scheme</summary>.*?</details>', '<p class="small"><a href="labour-force-survey-uncertainty-measurement/scripts/verify_enhancements.js">Numerical checks for the teaching simulation</a> compare browser variance with exact enumeration of its small synthetic distribution.</p>', h, flags=re.S)
h = h.replace('Simulations run when requested and use seeded pseudorandom numbers.', 'The sampling and bootstrap figures initialize with the displayed defaults; rerun them after changing inputs. Simulations use seeded pseudorandom numbers.')
h = h.replace('id="bootstrap-chart" class="chart" viewBox="0 0 800 280"', 'id="bootstrap-chart" class="chart" viewBox="0 0 800 300"')
page.write_text(h, encoding='utf-8')
print('Initialized two seeded figures, neutralized rotation panels and removed redundant bootstrap code blocks.')
