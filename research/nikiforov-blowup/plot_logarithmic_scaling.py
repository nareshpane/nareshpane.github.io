"""Plot a log-scale reference and deliberately planted comparison values."""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).parent; FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def f(n): return ImageFont.truetype(FONT,n)
nvals=[16,24,32,48,64,96,128,192,256]
planted=[2,2,3,3,3,4,4,4,4]  # deliberately constructed reference instances
W,H=1450,760; L,R,T,B=130,80,95,135
im=Image.new("RGB",(W,H),"#fffdf9"); d=ImageDraw.Draw(im)
d.text((L,T-55),"The logarithmic scale: a visual reference",font=f(40),fill="#1b1b1b")
d.text((L,T-7),"The blue curve is log(n), rescaled only for visual comparison. Gold points are deliberately planted K₃[k] instances.",font=f(18),fill="#555555")
x=lambda n:L+(math.log(n)-math.log(16))/(math.log(256)-math.log(16))*(W-L-R)
y=lambda v:H-B-v/6*(H-T-B)
for tick in range(7):
    yy=y(tick); d.line((L,yy,W-R,yy),fill="#e8e1d6",width=1); d.text((80,yy-11),str(tick),font=f(18),fill="#555555")
d.line((L,H-B,W-R,H-B),fill="#555555",width=2); d.line((L,T,L,H-B),fill="#555555",width=2)
for n in nvals:
    xx=x(n); d.line((xx,H-B,xx,H-B+8),fill="#555555",width=2); d.text((xx-16,H-B+16),str(n),font=f(16),fill="#555555")
blue=[(x(n),y(math.log(n))) for n in nvals]; gold=[(x(n),y(k)) for n,k in zip(nvals,planted)]
d.line(blue,fill="#1d4f91",width=5)
for p in blue: d.ellipse((p[0]-6,p[1]-6,p[0]+6,p[1]+6),fill="#1d4f91")
d.line(gold,fill="#c58a2b",width=4)
for p in gold: d.ellipse((p[0]-8,p[1]-8,p[0]+8,p[1]+8),fill="#c58a2b",outline="#1b1b1b")
d.text((L,H-68),"n (logarithmic horizontal spacing)",font=f(20),fill="#555555")
d.text((L+25,T+25),"vertical axis: log(n) or planted k",font=f(18),fill="#555555")
d.rectangle((W-420,T+35,W-95,T+110),fill="#fffdf9",outline="#ddd5c7")
d.line((W-395,T+60,W-340,T+60),fill="#1d4f91",width=5); d.text((W-325,T+46),"log(n) reference",font=f(18),fill="#1b1b1b")
d.line((W-395,T+90,W-340,T+90),fill="#c58a2b",width=4); d.text((W-325,T+76),"planted k (not a fit)",font=f(18),fill="#1b1b1b")
im.save(OUT/"logarithmic_blowup_scale.png")
