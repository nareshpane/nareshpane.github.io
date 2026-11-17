"""Retain BC's public-domain Natural Earth v5.1.2 geometry from the local design reference."""
from pathlib import Path
import re
root = Path(__file__).resolve().parents[1]
source = root.parent / 'harmonized-system/2b-canada-provinces-trade-by-hs/animation-assets/animation-maps.svg'
group = re.search(r'<g id="BC".*?</g>', source.read_text(encoding='utf-8'), re.S).group(0)
(root / 'figures/bc-outline.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="55 170 95 160" role="img" aria-label="British Columbia outline, Natural Earth v5.1.2"><g fill="#d6e5bd" stroke="#52795f" stroke-width=".7">' + group + '</g></svg>', encoding='utf-8')
