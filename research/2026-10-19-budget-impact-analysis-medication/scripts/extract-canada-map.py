"""Extract 13 unaltered Natural Earth province/territory geometries from the local design reference."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[1]
source=root.parent/'harmonized-system/2b-canada-provinces-trade-by-hs/animation-assets/animation-maps.svg'
text=source.read_text(encoding='utf-8')
codes=['BC','AB','SK','MB','ON','QC','NB','NS','PE','NL','YT','NT','NU']
groups=[re.search(r'<g id="'+code+r'".*?</g>',text,re.S).group(0) for code in codes]
(root/'figures/canada-provinces.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 570 410"><title>Canada: provinces and territories</title><!-- Natural Earth v5.1.2, public domain; original Lambert conformal conic coordinates. -->'+''.join(groups)+'</svg>',encoding='utf-8')
