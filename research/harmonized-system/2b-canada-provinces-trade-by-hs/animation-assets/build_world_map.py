"""Copy the validated world land geometry from the local reference study.
Natural Earth v5.1.2 50m land, public domain; Robinson projection, excluding
Antarctica. No network access and no runtime dependency on the reference page.
Run: python research/harmonized-system/2b-canada-provinces-trade-by-hs/animation-assets/build_world_map.py
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
BASE = Path(__file__).resolve().parent
reference = BASE.parent.parent/'machine-learning-trade-sector-prediction.html'
content = reference.read_text(encoding='utf-8')
match = re.search(r'<g class="ta-land"[^>]*>.*?</g>', content, re.S)
assert match, 'Reference world land geometry missing'
land = match.group().replace('class="ta-land"', 'id="world-land"')
svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 490"><title>World land, Natural Earth v5.1.2, Robinson projection</title>'+land+'</svg>\n'
ET.fromstring(svg)
(BASE/'world-map.svg').write_text(svg, encoding='utf-8')
print('World SVG bytes:', len(svg.encode('utf-8')))
