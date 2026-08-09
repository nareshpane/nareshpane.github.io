"""Locate small K3[k] witnesses in reproducible G(n,1/2) samples; exact only through k=3."""
from pathlib import Path
from itertools import combinations
import random
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).parent; FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def f(n): return ImageFont.truetype(FONT,n)
def graph(n,seed):
    r=random.Random(seed); a=[[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n): a[i][j]=a[j][i]=r.random()<.5
    return a
def has(a,k):
    ss=[s for s in combinations(range(len(a)),k) if all(not a[i][j] for i,j in combinations(s,2))]
    for x,y,z in combinations(ss,3):
        if set(x)&set(y) or set(x)&set(z) or set(y)&set(z): continue
        if all(a[i][j] for u,v in ((x,y),(x,z),(y,z)) for i in u for j in v): return True
    return False

records=[]
for n,seed in ((12,810),(16,811),(20,812),(24,813)):
    a=graph(n,seed); k=3 if has(a,3) else (2 if has(a,2) else 1)
    records.append((n,k,seed))
with open(OUT/"random_graph_search_results.txt","w") as out:
    out.write("G(n, 1/2), one reproducible sample each; exact search capped at k=3\n")
    for n,k,seed in records: out.write(f"n={n}, seed={seed}, witnessed k={k}\n")

W,H=1300,680; L,R,T,B=130,80,105,120
im=Image.new("RGB",(W,H),"#fffdf9"); d=ImageDraw.Draw(im)
d.rounded_rectangle((15,15,W-15,H-15),24,fill="#fffdf9",outline="#ddd5c7",width=2)
d.text((L,55),"Random-graph intuition: small witnessed blow-ups",font=f(37),fill="#1b1b1b")
d.text((L,108),"One G(n, 1/2) sample per n; exact search is deliberately capped at k = 3.",font=f(19),fill="#555555")
x=lambda n:L+(n-12)/(24-12)*(W-L-R); y=lambda k:H-B-(k-1)/3*(H-T-B)
for k in range(1,5):
    yy=y(k); d.line((L,yy,W-R,yy),fill="#e8e1d6"); d.text((85,yy-11),str(k),font=f(19),fill="#555555")
d.line((L,H-B,W-R,H-B),fill="#555555",width=2); d.line((L,T,L,H-B),fill="#555555",width=2)
points=[(x(n),y(k)) for n,k,_ in records]; d.line(points,fill="#1d4f91",width=4)
for (n,k,seed),(px,py) in zip(records,points):
    d.ellipse((px-9,py-9,px+9,py+9),fill="#c58a2b",outline="#1b1b1b",width=2)
    d.text((px-15,H-B+17),str(n),font=f(18),fill="#555555"); d.text((px-33,py-42),f"k={k}",font=f(18),fill="#1d4f91")
d.text((L,H-58),"n",font=f(21),fill="#555555"); d.text((L+22,T+18),"witnessed k",font=f(20),fill="#555555")
d.text((L, H-28),"These finite samples illustrate a slow scale only; they neither estimate the asymptotic optimum nor rule out larger witnesses.",font=f(17),fill="#555555")
im.save(OUT/"random_graph_blowup_scale.png")
