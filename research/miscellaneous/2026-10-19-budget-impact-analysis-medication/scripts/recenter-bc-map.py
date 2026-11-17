"""Recenter existing Lambert conic BC geometry at 126 W, without changing its shape.

The source SVG uses standard parallels 49/77 and central meridian 96 W.
Changing only the central meridian is a rigid rotation on a conic map.
Natural Earth v5.1.2 geometry is public domain; no network access is needed.
"""
from pathlib import Path
import math, re
root=Path(__file__).resolve().parents[1]
source=(root/'figures/canada-provinces.svg').read_text(encoding='utf-8')
group=re.search(r'<g id="BC".*?</g>',source,re.S).group(0)
a,b=map(math.radians,(49,77))
n=math.log(math.cos(a)/math.cos(b))/math.log(math.tan(math.pi/4+b/2)/math.tan(math.pi/4+a/2))
angle=-n*30
c,s=math.cos(math.radians(angle)),math.sin(math.radians(angle))
points=[]
def rotate(match):
    x,y=map(float,match.groups());xx=x*c-y*s;yy=x*s+y*c
    points.append((xx,yy));return f'{xx:.3f},{yy:.3f}'
group=re.sub(r'(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)',rotate,group)
left=min(x for x,y in points)-3;top=min(y for x,y in points)-3
width=max(x for x,y in points)-left+3;height=max(y for x,y in points)-top+3
svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{left:.3f} {top:.3f} {width:.3f} {height:.3f}" role="img" aria-label="British Columbia, Lambert conformal conic centred at 126 degrees west"><title>British Columbia, Natural Earth v5.1.2</title><!-- Standard parallels 49/77; central meridian -126. Rigid recentering from the existing Canada map. --><g fill="#98c97e" stroke="#527661" stroke-width=".65">{group}</g></svg>'
(root/'figures/bc-centred.svg').write_text(svg,encoding='utf-8')
print('BC conic recentering angle:',round(angle,4),'degrees')
