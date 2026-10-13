"""Apply the small fixes identified during the visual browser review."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = root / 'css/visual-refinements.css'
s = p.read_text(encoding='utf-8')
rule = '#classification .controls[hidden]{display:none}\n'
if rule not in s:
    p.write_text(rule+s, encoding='utf-8')
p = root / 'js/regional-scenes.js'
s = p.read_text(encoding='utf-8').replace('x="203" y="130" width="16" height="15"', 'x="203.1" y="138.3" width="15.3" height="13.5"').replace('M192 105H300V280H192Z', 'M40 62H438V305H40Z').replace("+T(55,480,'Illustrative winter setting',22)", '').replace('x="843" y="130" width="110" height="150"', 'x="843" y="85" width="110" height="70"').replace('[896,213,.8]', '[896,151,.5]')
p.write_text(s, encoding='utf-8')
p = root / 'js/documentary.js'
s = p.read_text(encoding='utf-8')
old_figure = '<figure><svg viewBox="0 0 570 440" role="img" aria-label="A selected dwelling supplies linked person observations;'
if old_figure in s:
    start = s.index(old_figure)
    end = s.index('</figure>', start)+len('</figure>')
    s = s[:start]+'<figure class="weighting-figure"></figure>'+s[end:]
    p.write_text(s, encoding='utf-8')
p = root / 'js/visual-refinements.js'
s = p.read_text(encoding='utf-8')
if '    const formulas=' in s:
    start = s.index('    const formulas=')
    end = s.index("    document.getElementById('variance')", start)
    s = s[:start]+s[end:]
# Keep TeX escapes explicit when inserted dynamically.
start = s.index('  function weighting()')
end = s.index('  function clustering()', start)
block = s[start:end]
import re
formulas = ["slash+'pi='+slash+'frac{1}{45}'", "'w='+slash+'frac{1}{'+slash+'pi}=45'", "'3'+slash+'times45=135'"]
for formula in formulas:
    block = re.sub(r'<div class="equation">.*?</div>', lambda _: '${math('+formula+')}', block, count=1)
s = s[:start]+block+s[end:]
s = s.replace("'Observed PSU sample',19", "'Observed PSUs',19").replace("'fill=\"#fffaf0\"'", "'style=\"fill:#fffaf0\"'")
s = s.replace('35+i%4*26+Math.floor(i/4)*65', '36+Math.floor(i/4)*112+i%2*27').replace('62+Math.floor(i/4)*27', '78+Math.floor(i%4/2)*35')
s = s.replace("'<path d=\"M17 42H306V156H17Z\" fill=\"none\" stroke=\"#739b76\" stroke-dasharray=\"6 4\"/>'", "[0,1,2].map(j=>`<rect x=\"${19+j*112}\" y=\"51\" width=\"83\" height=\"91\" rx=\"8\" fill=\"none\" stroke=\"#739b76\" stroke-dasharray=\"6 4\"/>`).join('')")
p.write_text(s, encoding='utf-8')
p = root / 'scripts/refine_visuals.py'
s = p.read_text(encoding='utf-8').replace('assert(await ev("document.getElementById(\'scenario\').closest(\'.controls\').hidden"));', 'assert(await ev("document.getElementById(\'scenario\').closest(\'.controls\').hidden"));\nassert.equal(await ev("getComputedStyle(document.getElementById(\'scenario\').closest(\'.controls\')).display"),"none");')
s = s.replace("['methodology','units']", "['population','units'],['conceptual-boxes','concepts']").replace("['design','synthetic']", "['design-map','synthetic']").replace("['geographic-guide','weights']", "['weighting-figure','weights']")
# Screenshot helper addresses elements by ID, so retain classes separately below.
s = s.replace("['conceptual-boxes','concepts']", "['methodology','concepts']").replace("['weighting-figure','weights']", "['geographic-guide','weights']")
p.write_text(s, encoding='utf-8')
