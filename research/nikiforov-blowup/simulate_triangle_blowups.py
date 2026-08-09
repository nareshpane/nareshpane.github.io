"""Reproducible finite triangle/blow-up experiments (Pillow + standard library)."""
from pathlib import Path
from itertools import combinations
import csv, random
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def font(n): return ImageFont.truetype(FONT, n)

def random_graph(n, p, seed):
    rng = random.Random(seed); adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n): adj[i][j] = adj[j][i] = rng.random() < p
    return adj

def plant_blowup(n, k, p, seed):
    adj = random_graph(n, p, seed); groups = [list(range(i*k, (i+1)*k)) for i in range(3)]
    for group in groups:
        for i,j in combinations(group,2): adj[i][j] = adj[j][i] = False
    for a,b in combinations(groups,2):
        for i in a:
            for j in b: adj[i][j] = adj[j][i] = True
    return adj

def triangle_count(a):
    return sum(a[i][j] and a[i][k] and a[j][k] for i in range(len(a)) for j in range(i+1,len(a)) for k in range(j+1,len(a)))

def independent_sets(a, k):
    return [s for s in combinations(range(len(a)),k) if all(not a[i][j] for i,j in combinations(s,2))]

def contains_k3_blowup(a, k):
    """Exact exhaustive search for k=2,3 only; return a witness or None."""
    sets = independent_sets(a,k)
    for x,y,z in combinations(sets,3):
        if set(x)&set(y) or set(x)&set(z) or set(y)&set(z): continue
        if all(a[i][j] for u,v in ((x,y),(x,z),(y,z)) for i in u for j in v): return (x,y,z)
    return None

raw = [
    ("random G(18, 0.50)", random_graph(18,.50,701), 0),
    ("random G(24, 0.50)", random_graph(24,.50,702), 0),
    ("planted K₃[3] + noise", plant_blowup(24,3,.22,703), 3),
    ("planted K₃[4] + noise", plant_blowup(30,4,.16,704), 4),
]
experiments = []
for label, a, guaranteed in raw:
    # Exhaustive three-cluster enumeration grows very quickly; cap it at n=24.
    found3 = contains_k3_blowup(a,3) if len(a) <= 24 else None
    found2 = contains_k3_blowup(a,2) if len(a) <= 24 else None
    reported = max(guaranteed, 3 if found3 else (2 if found2 else 1))
    method = "planted witness; exhaustive search through k=3" if guaranteed else "exhaustive search through k=3"
    if len(a) > 24: method = "planted witness; no exhaustive search above n=24"
    experiments.append((label, len(a), triangle_count(a), reported, method))

with open(OUT / "triangle_blowup_experiments.csv", "w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n"); writer.writerow(["host graph","n","triangles","reported k","method"]); writer.writerows(experiments)

im = Image.new("RGB", (1450, 620), "#fffdf9"); d = ImageDraw.Draw(im)
d.rounded_rectangle((15,15,1435,605), 24, fill="#fffdf9", outline="#ddd5c7", width=2)
d.text((55,50), "Finite triangle-copy and blow-up experiments", font=font(38), fill="#1b1b1b")
d.text((55,104), "Random seeds: 701–704. Random rows use exact search only through k = 3; planted rows include their construction witness.", font=font(18), fill="#555555")
cols=[55,470,650,895,1050]; headers=["host graph", "n", "triangle copies", "reported k", "search / witness"]
for x,h in zip(cols,headers): d.text((x,170),h,font=font(20),fill="#1d4f91")
for r,row in enumerate(experiments):
    y=230+r*78; d.line((48,y-14,1400,y-14),fill="#ddd5c7",width=1)
    for x,val in zip(cols,row): d.text((x,y),str(val),font=font(19),fill="#1b1b1b")
d.text((55,550), "A reported value is a witnessed lower bound, not necessarily the largest blow-up present in the host graph.", font=font(18),fill="#555555")
im.save(OUT / "triangle_copy_and_blowup_experiments.png")
