"""Generate frames and a meaningful MP4 of the three complete bipartite joins."""
from pathlib import Path
import subprocess
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).parent; FRAMES=OUT/"blowup_animation_frames"; FRAMES.mkdir(exist_ok=True)
FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def f(n): return ImageFont.truetype(FONT,n)
clusters=[[(610,410),(650,410),(630,370)],[(390,650),(430,650),(410,610)],[(850,650),(890,650),(870,610)]]
for frame in range(36):
    im=Image.new("RGB",(1280,820),"#f7f4ee"); d=ImageDraw.Draw(im)
    d.rounded_rectangle((25,25,1255,795),26,fill="#fffdf9",outline="#ddd5c7",width=2)
    d.text((65,65),"Building K₃[3] from K₃",font=f(43),fill="#1b1b1b")
    stage=min(3,frame//9)
    d.text((65,126),["Three independent clusters", "Join clusters A and B", "Also join A and C", "Finally join B and C"][stage],font=f(24),fill="#1d4f91")
    for pi,(a,b) in enumerate(((0,1),(0,2),(1,2))):
        if pi < stage:
            for u in clusters[a]:
                for v in clusters[b]: d.line((u,v),fill="#7ea6d6",width=3)
    for ci,pts in enumerate(clusters):
        cx=sum(p[0] for p in pts)/3; cy=sum(p[1] for p in pts)/3
        d.ellipse((cx-78,cy-78,cx+78,cy+78),outline="#9db6d3",width=3)
        d.text((cx-15,cy+86),"ABC"[ci],font=f(22),fill="#555555")
        for x,y in pts: d.ellipse((x-19,y-19,x+19,y+19),fill="#c58a2b",outline="#1b1b1b",width=2)
    d.text((65,735),"No edges within a cluster; every K₃ edge becomes all 3 × 3 cross-cluster edges.",font=f(19),fill="#555555")
    im.save(FRAMES/f"frame_{frame:03d}.png")
subprocess.run(["ffmpeg","-y","-framerate","6","-i",str(FRAMES/"frame_%03d.png"),"-c:v","libx264","-pix_fmt","yuv420p",str(OUT/"blowup_construction.mp4")],check=True)
