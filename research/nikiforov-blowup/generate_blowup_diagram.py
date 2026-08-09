"""Draw a small K_3 and its balanced 3-blow-up using Pillow."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent
W, H = 1500, 760
BLUE, GOLD, INK, PALE, LINE = "#1d4f91", "#c58a2b", "#1b1b1b", "#f7f4ee", "#d8cfbf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

def f(size, serif=False):
    return ImageFont.truetype(SERIF if serif else FONT, size)

def edge(draw, a, b, fill=BLUE, width=4):
    draw.line([a, b], fill=fill, width=width)

def node(draw, p, label, fill=GOLD):
    r = 22
    draw.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill=fill, outline=INK, width=2)
    if label:
        box = draw.textbbox((0,0), label, font=f(22))
        draw.text((p[0]-(box[2]-box[0])/2, p[1]-(box[3]-box[1])/2-2), label, fill="white", font=f(22))

im = Image.new("RGB", (W, H), PALE)
d = ImageDraw.Draw(im)
d.rounded_rectangle((25, 25, W-25, H-25), radius=28, fill="#fffdf9", outline=LINE, width=2)
d.text((80, 72), "A graph and a balanced blow-up", font=f(48, True), fill=INK)
d.text((85, 136), "Every original vertex becomes an independent cluster; every original edge becomes a complete bipartite connection.", font=f(22), fill="#555555")
left = [(260, 330), (120, 590), (400, 590)]
for i in range(3): edge(d, left[i], left[(i+1)%3], width=5)
for p, label in zip(left, ("a", "b", "c")): node(d, p, label)
d.text((170, 230), "H = K₃", font=f(35, True), fill=BLUE)
d.text((118, 650), "3 vertices; 3 edges", font=f(20), fill="#555555")
d.text((528, 447), "→", font=f(76, True), fill=BLUE)
centers = [(1000, 285), (745, 600), (1255, 600)]
clusters = []
for ci, (cx, cy) in enumerate(centers):
    pts = [(cx-28, cy+17), (cx+28, cy+17), (cx, cy-31)]
    clusters.append(pts)
    d.ellipse((cx-78, cy-78, cx+78, cy+78), outline="#9db6d3", width=3)
    d.text((cx-56, cy+82), f"cluster {chr(65+ci)}", font=f(16), fill="#555555")
for i in range(3):
    for j in range(i+1,3):
        for a in clusters[i]:
            for b in clusters[j]: edge(d, a, b, fill="#7ea6d6", width=2)
for pts in clusters:
    for p in pts: node(d, p, "", fill=GOLD)
d.text((875, 190), "H[3] = K₃[3]", font=f(35, True), fill=BLUE)
d.text((790, 705), "3 independent clusters of size 3; every pair of clusters is completely joined", font=f(18), fill="#555555")
im.save(OUT / "triangle_to_balanced_blowup.png", quality=95)
im.save(OUT / "triangle_to_balanced_blowup.jpeg", quality=93)
