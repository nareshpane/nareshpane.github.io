"""Create the figures, animations, and reproducible finite experiments for the
Nikiforov blow-up page.  Uses only Pillow, the standard library, and ffmpeg.

Containment convention: a K3[k] witness is three *disjoint* k-sets whose
cross pairs are all edges.  We deliberately do not reject extra edges within
one chosen set: the page discusses ordinary (not induced) subgraph containment.
"""
from pathlib import Path
from itertools import combinations
from tempfile import TemporaryDirectory
import csv, math, random, subprocess
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
W, H = 1500, 820
INK, MUTED, PAPER, CARD, LINE, BLUE = "#1b1b1b", "#5d6268", "#f7f4ee", "#fffdf9", "#ddd5c7", "#1d4f91"
CLRS = ["#d26a5d", "#4d8b75", "#5e88c9", "#b58a42"]  # A, B, C, D throughout

def f(n, serif=False): return ImageFont.truetype(SERIF if serif else FONT, n)
def txt(d, xy, s, size=22, fill=INK, serif=False, anchor=None): d.text(xy, s, font=f(size, serif), fill=fill, anchor=anchor)
def canvas(title, subtitle=""):
    im=Image.new("RGB",(W,H),PAPER); d=ImageDraw.Draw(im)
    d.rounded_rectangle((24,24,W-24,H-24),28,fill=CARD,outline=LINE,width=2)
    txt(d,(68,62),title,43,INK,True)
    if subtitle: txt(d,(70,119),subtitle,19,MUTED)
    return im,d
def node(d,p,label,color=BLUE,r=21):
    x,y=p; d.ellipse((x-r,y-r,x+r,y+r),fill=color,outline=INK,width=2); txt(d,(x,y),label,17,"#ffffff",anchor="mm")
def edge(d,a,b,color="#49515a",width=3): d.line((a,b),fill=color,width=width)
def labelbox(d, box, title, body, accent=BLUE):
    d.rounded_rectangle(box,18,fill="#fcfaf6",outline=LINE,width=2); x,y=box[:2]
    txt(d,(x+18,y+16),title,20,accent); txt(d,(x+18,y+49),body,17,MUTED)
def cluster(d, center, letter, k, color, radius=84, spread=30):
    cx,cy=center; d.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline=color,width=3)
    pts=[]
    for i in range(k):
        angle=-math.pi/2 + 2*math.pi*i/k
        pts.append((cx+spread*math.cos(angle),cy+spread*math.sin(angle)))
    for i,p in enumerate(pts,1): node(d,p,f"{letter}{i}",color,17)
    txt(d,(cx,cy+radius+24),f"cluster {letter}",17,color,anchor="mm")
    return pts
def joins(d, groups, pairs, color="#59616a", width=2):
    for i,j in pairs:
        for a in groups[i]:
            for b in groups[j]: edge(d,a,b,color,width)
def save(im,name): im.save(OUT/name)

def fig01():
    im,d=canvas("Figure 1. The simplest graph: one edge", "A graph is a collection of dots (vertices) and lines (edges).")
    a,b=(480,410),(940,410); edge(d,a,b,"#424a52",6); node(d,a,"A",CLRS[0],34); node(d,b,"B",CLRS[1],34)
    txt(d,(710,370),"edge",21,MUTED,anchor="mm")
    labelbox(d,(250,570,680,700),"Vertex","A circle labelled A or B.",CLRS[0])
    labelbox(d,(820,570,1250,700),"Edge","The line joining A and B.",BLUE)
    txt(d,(750,760),"This two-vertex graph is called K₂.",22,INK,True,anchor="mm")
    save(im,"figure_01_simplest_graph_k2.png")

def fig02():
    im,d=canvas("Figure 2. Blow up one edge: K₂ → K₂[3]", "One vertex becomes a group; one edge becomes every cross-connection between the groups.")
    a,b=(230,380),(455,380); edge(d,a,b,"#424a52",5); node(d,a,"A",CLRS[0],27); node(d,b,"B",CLRS[1],27); txt(d,(342,510),"original K₂",25,BLUE,True,anchor="mm")
    txt(d,(575,390),"→",64,BLUE,True,anchor="mm")
    groups=[cluster(d,(840,405),"A",3,CLRS[0]),cluster(d,(1190,405),"B",3,CLRS[1])]; joins(d,groups,[(0,1)],"#48515a",3)
    # redraw nodes over joins
    for gi,g in enumerate(groups):
        for i,p in enumerate(g,1): node(d,p,f"{'AB'[gi]}{i}",CLRS[gi],17)
    txt(d,(1015,555),"K₂[3]: 9 required cross edges = 3 × 3",20,INK,anchor="mm")
    labelbox(d,(150,640,1350,742),"Read the picture","Every Aᵢ is joined to every Bⱼ. Only after seeing this pattern do we name it K₃,₃.")
    save(im,"figure_02_k2_to_k2_3.png")

def fig03():
    im,d=canvas("Figure 3. A non-edge matters: P₃ → P₃[3]", "The base path has edges A–B and B–C, but no edge A–C.")
    base=[(160,380),(330,380),(500,380)]; edge(d,base[0],base[1],"#414951",5); edge(d,base[1],base[2],"#414951",5)
    for p,l,c in zip(base,"ABC",CLRS): node(d,p,l,c,24)
    txt(d,(330,500),"P₃",26,BLUE,True,anchor="mm"); txt(d,(590,382),"→",60,BLUE,True,anchor="mm")
    groups=[cluster(d,(800,410),"A",3,CLRS[0]),cluster(d,(1060,255),"B",3,CLRS[1]),cluster(d,(1250,520),"C",3,CLRS[2])]
    joins(d,groups,[(0,1),(1,2)],"#4c555e",2)
    for gi,g in enumerate(groups):
        for i,p in enumerate(g,1): node(d,p,f"{'ABC'[gi]}{i}",CLRS[gi],17)
    labelbox(d,(105,650,705,754),"edge in H → complete cross-connection","A–B and B–C create all corresponding cross edges.",CLRS[1])
    labelbox(d,(790,650,1390,754),"non-edge in H → no corresponding edge family","A and C have no required A–C cross connections.",CLRS[2])
    save(im,"figure_03_p3_nonedge_matters.png")

def fig04():
    im,d=canvas("Figure 4. Triangle blow-up: K₃ → K₃[3]", "Every pair in the base triangle is an edge, so every pair of clusters is completely joined.")
    base=[(270,240),(150,480),(390,480)]
    for x,y in combinations(base,2): edge(d,x,y,"#414951",4)
    for p,l,c in zip(base,"ABC",CLRS): node(d,p,l,c,25)
    txt(d,(270,575),"K₃",28,BLUE,True,anchor="mm"); txt(d,(545,390),"→",60,BLUE,True,anchor="mm")
    groups=[cluster(d,(1030,230),"A",3,CLRS[0]),cluster(d,(800,550),"B",3,CLRS[1]),cluster(d,(1260,550),"C",3,CLRS[2])]
    joins(d,groups,[(0,1),(0,2),(1,2)],"#4b535c",2)
    for gi,g in enumerate(groups):
        for i,p in enumerate(g,1): node(d,p,f"{'ABC'[gi]}{i}",CLRS[gi],17)
    labelbox(d,(140,660,1360,755),"New notation","K₃[k] is the complete tripartite graph Kₖ,ₖ,ₖ: three groups of k, with all cross-group edges. The name follows the visual rule above.")
    save(im,"figure_04_k3_to_k3_3.png")

def fig05():
    im,d=canvas("Figure 5. A four-vertex example: C₄ → C₄[2]", "Blow-ups work for any fixed base graph, not only edges and triangles.")
    base=[(270,250),(470,250),(470,450),(270,450)]
    for i in range(4): edge(d,base[i],base[(i+1)%4],"#414951",4)
    for p,l,c in zip(base,"ABCD",CLRS): node(d,p,l,c,23)
    txt(d,(370,555),"C₄ (a four-cycle)",25,BLUE,True,anchor="mm"); txt(d,(590,360),"→",60,BLUE,True,anchor="mm")
    groups=[cluster(d,(940,230),"A",2,CLRS[0],64,23),cluster(d,(1200,230),"B",2,CLRS[1],64,23),cluster(d,(1200,510),"C",2,CLRS[2],64,23),cluster(d,(940,510),"D",2,CLRS[3],64,23)]
    joins(d,groups,[(0,1),(1,2),(2,3),(3,0)],"#4b535c",2)
    for gi,g in enumerate(groups):
        for i,p in enumerate(g,1): node(d,p,f"{'ABCD'[gi]}{i}",CLRS[gi],15)
    labelbox(d,(130,655,1370,755),"Follow the cycle","Required families are A–B, B–C, C–D, and D–A. A–C and B–D are non-edges of C₄, so no corresponding cross-edge family is required.")
    save(im,"figure_05_c4_to_c4_2.png")

def host_layout():
    rng=random.Random(2309); bg=[(120+rng.randrange(1260),180+rng.randrange(500)) for _ in range(15)]
    centers=[(820,230),(670,520),(1080,520)]; groups=[]
    for ci,c in enumerate(centers):
        groups.append([(c[0]-24,c[1]+16),(c[0]+24,c[1]+16),(c[0],c[1]-28)])
    return bg,groups
def draw_host(d,highlight=False):
    bg,groups=host_layout(); rng=random.Random(771)
    allp=bg+[p for g in groups for p in g]
    for i,j in combinations(range(len(allp)),2):
        if rng.random()<.105: edge(d,allp[i],allp[j],"#c9c7c1",2)
    if highlight: joins(d,groups,[(0,1),(0,2),(1,2)],"#33424f",4)
    for p in bg: node(d,p,"", "#aeb3b6",11)
    for gi,g in enumerate(groups):
        if highlight:
            cx=sum(x for x,y in g)/3; cy=sum(y for x,y in g)/3; d.ellipse((cx-70,cy-70,cx+70,cy+70),outline=CLRS[gi],width=3)
        for i,p in enumerate(g,1): node(d,p,f"{'ABC'[gi]}{i}",CLRS[gi] if highlight else "#aeb3b6",16)
def fig06():
    im,d=canvas("Figure 6. Finding a blow-up inside a larger host graph G", "The same host is shown twice: first as a whole, then with one K₃[3] witness revealed.")
    d.line((750,160,750,680),fill=LINE,width=2); txt(d,(375,160),"(a) host graph G",25,BLUE,True,anchor="mm"); txt(d,(1125,160),"(b) a highlighted K₃[3] subgraph",25,BLUE,True,anchor="mm")
    # translate draw calls via temporary image halves by deliberately use layouts shifted
    # draw left custom
    bg,groups=host_layout();
    def shifted(items,dx): return [(x+dx,y) for x,y in items]
    rng=random.Random(771); allp=bg+[p for g in groups for p in g]
    for dx,hi in ((-610,False),(140,True)):
        for i,j in combinations(range(len(allp)),2):
            if rng.random()<.105: edge(d,(allp[i][0]+dx,allp[i][1]),(allp[j][0]+dx,allp[j][1]),"#c9c7c1",2)
        if hi:
            gs=[shifted(g,dx) for g in groups]; joins(d,gs,[(0,1),(0,2),(1,2)],"#33424f",4)
        for p in bg: node(d,(p[0]+dx,p[1]),"","#aeb3b6",10)
        for gi,g in enumerate(groups):
            gg=shifted(g,dx)
            if hi:
                cx=sum(x for x,y in gg)/3; cy=sum(y for x,y in gg)/3; d.ellipse((cx-70,cy-70,cx+70,cy+70),outline=CLRS[gi],width=3)
            for i,p in enumerate(gg,1): node(d,p,f"{'ABC'[gi]}{i}" if hi else "",CLRS[gi] if hi else "#aeb3b6",15 if hi else 10)
    labelbox(d,(115,695,1385,775),"Meaning of “G contains H[k]”","Select distinct vertices in G and use the required edges among them. Extra edges elsewhere—or even inside selected clusters—do not invalidate ordinary subgraph containment.")
    save(im,"figure_06_host_graph_witness.png")

def fig07():
    im,d=canvas("Figure 7. Many triangles versus one organized triangle blow-up", "Both sides contain triangles, but only the right side displays a deliberately coordinated K₃[3].")
    d.line((750,160,750,685),fill=LINE,width=2)
    # scattered triangle components
    tri_centers=[(220,310),(490,315),(355,520)]
    for c in tri_centers:
        pts=[(c[0],c[1]-45),(c[0]-48,c[1]+35),(c[0]+48,c[1]+35)]
        for x,y in combinations(pts,2): edge(d,x,y,"#6c747b",3)
        for p in pts: node(d,p,"","#9aa0a4",13)
    txt(d,(375,195),"scattered triangles",26,BLUE,True,anchor="mm"); txt(d,(375,645),"n = 9 · edges = 9 · triangles = 3 · known k = 1",18,MUTED,anchor="mm")
    groups=[cluster(d,(1125,250),"A",3,CLRS[0]),cluster(d,(920,535),"B",3,CLRS[1]),cluster(d,(1330,535),"C",3,CLRS[2])]
    joins(d,groups,[(0,1),(0,2),(1,2)],"#46505a",2)
    for gi,g in enumerate(groups):
        for i,p in enumerate(g,1): node(d,p,f"{'ABC'[gi]}{i}",CLRS[gi],16)
    txt(d,(1125,195),"organized K₃[3]",26,BLUE,True,anchor="mm"); txt(d,(1125,645),"n = 9 · edges = 27 · triangles = 27 · known k = 3",18,MUTED,anchor="mm")
    labelbox(d,(155,700,1345,775),"Finite-example caution","This is a comparison of two deliberately chosen graphs, not a general inference from triangle count alone.")
    save(im,"figure_07_scattered_vs_organized.png")

def random_graph(n,p,seed):
    r=random.Random(seed); a=[[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n): a[i][j]=a[j][i]=r.random()<p
    return a
def plant(n,k,p,seed):
    a=random_graph(n,p,seed); groups=[list(range(i*k,(i+1)*k)) for i in range(3)]
    for x,y in combinations(groups,2):
        for i in x:
            for j in y:a[i][j]=a[j][i]=True
    return a
def counts(a):
    n=len(a); e=sum(a[i][j] for i in range(n) for j in range(i+1,n)); t=sum(a[i][j] and a[i][z] and a[j][z] for i in range(n) for j in range(i+1,n) for z in range(j+1,n)); return e,t
def k2_exact(a):
    pairs=list(combinations(range(len(a)),2))
    for x,y,z in combinations(pairs,3):
        if len(set(x+y+z))<6: continue
        if all(a[i][j] for u,v in ((x,y),(x,z),(y,z)) for i in u for j in v): return (x,y,z)
    return None
def k3_heuristic(a,tries=1800,seed=91):
    # random disjoint triples; validates a witness but does not prove maximum
    r=random.Random(seed); n=len(a)
    for _ in range(tries):
        v=list(range(n)); r.shuffle(v); g=(tuple(v[:3]),tuple(v[3:6]),tuple(v[6:9]))
        if all(a[i][j] for x,y in combinations(g,2) for i in x for j in y): return g
    return None
def experiments():
    rows=[]
    specs=[("A: deterministic K3[3]",9,3,0,101,"planted witness"),("B: planted + light noise",24,3,.10,202,"planted witness; heuristic checked"),("B: planted + dense noise",24,3,.35,203,"planted witness; heuristic checked"),
           ("C: random G(18,0.50)",18,0,.50,301,"exact k=2; k=3 heuristic"),("C: random G(24,0.50)",24,0,.50,302,"exact k=2; k=3 heuristic"),
           ("D: same n, p=0.25",20,0,.25,401,"exact k=2; k=3 heuristic"),("D: same n, p=0.50",20,0,.50,402,"exact k=2; k=3 heuristic"),("D: same n, p=0.75",20,0,.75,403,"exact k=2; k=3 heuristic")]
    for name,n,k,p,seed,method in specs:
        a=plant(n,k,p,seed) if k else random_graph(n,p,seed); e,t=counts(a)
        found= k if k else (3 if k3_heuristic(a,seed=seed) else (2 if k2_exact(a) else 1))
        rows.append([name,n,e,t,f"{t/n**3:.5f}",str(k) if k else "-",found,method])
    # E uses planted instances: known sizes, explicit rather than a claimed fit.
    for n,k,seed in [(12,2,501),(18,2,502),(27,3,503),(36,3,504),(48,4,505)]:
        a=plant(n,k,.16,seed); e,t=counts(a); rows.append(["E: increasing n (planted)",n,e,t,f"{t/n**3:.5f}",k,k,"planted witness; no maximum claim"])
    return rows
def fig08(rows):
    im,d=canvas("Figure 8. Finite scaling experiment", "Planted values are known witnesses; random values are lower bounds found by the stated limited search.")
    data=[r for r in rows if r[0].startswith("E:")]; L,R,T,B=130,100,190,145; xmax=50
    x=lambda n:L+(n-12)/(xmax-12)*(W-L-R); y=lambda k:H-B-(k-1)/4*(H-T-B)
    for k in range(1,6):
        yy=y(k); d.line((L,yy,W-R,yy),fill="#e8e1d6"); txt(d,(90,yy),str(k),18,MUTED,anchor="mm")
    d.line((L,H-B,W-R,H-B),fill=MUTED,width=2); d.line((L,T,L,H-B),fill=MUTED,width=2)
    pts=[]
    for r in data:
        n,k=int(r[1]),int(r[6]); pts.append((x(n),y(k))); txt(d,(x(n),H-B+25),str(n),17,MUTED,anchor="mm")
    d.line(pts,fill=BLUE,width=4)
    for (px,py),r in zip(pts,data): d.ellipse((px-9,py-9,px+9,py+9),fill="#c58a2b",outline=INK,width=2); txt(d,(px,py-27),f"k={r[6]}",16,BLUE,anchor="mm")
    # reference log curve scaled only to plotting range
    curve=[(x(n),y(1+1.05*math.log(n/12+1))) for n in range(12,49)]
    d.line(curve,fill="#7ea6d6",width=3)
    txt(d,(L,H-72),"n (number of host vertices)",20,MUTED); txt(d,(L+20,T+20),"known / reported k",20,MUTED)
    labelbox(d,(810,205,1365,300),"Blue curve: log-shaped visual reference","It is not fitted to the gold planted data and is not c_H(γ).")
    labelbox(d,(810,325,1365,420),"Gold points: planted witnesses","They make the slowly increasing scale tangible; finite computation does not prove the theorem.","#b58a42")
    save(im,"figure_08_scaling_experiment.png")

def make_animation(name,frames):
    with TemporaryDirectory(prefix="nikiforov_frames_") as tmp:
        td=Path(tmp)
        for i,im in enumerate(frames): im.save(td/f"frame_{i:03d}.png")
        subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate","2","-i",str(td/"frame_%03d.png"),"-c:v","libx264","-pix_fmt","yuv420p",str(OUT/name)],check=True)
def tracker(d,stage,clusters,verts,edges,status):
    d.rounded_rectangle((75,650,1425,760),18,fill="#f4f0e8",outline=LINE,width=2)
    for x,label,val in [(110,"Stage",stage),(380,"Clusters",clusters),(650,"Vertices",verts),(910,"Required cross edges",edges),(1220,"Status",status)]: txt(d,(x,680),label,15,MUTED); txt(d,(x,712),str(val),20,BLUE)
def anim_k2():
    frames=[]
    for st in range(4):
        im,d=canvas("Animation 1. Constructing K₂[3]");
        if st==0:
            a,b=(600,390),(900,390); edge(d,a,b,"#434c55",5); node(d,a,"A",CLRS[0],28); node(d,b,"B",CLRS[1],28); tracker(d,"start: K₂",2,2,"1 base edge","starting")
        else:
            gs=[cluster(d,(570,390),"A",3,CLRS[0]),cluster(d,(930,390),"B",3,CLRS[1])]
            shown=[0,3,9][st-1]
            pairs=[(a,b) for a in gs[0] for b in gs[1]]
            for a,b in pairs[:shown]: edge(d,a,b,"#46515b",3)
            for gi,g in enumerate(gs):
                for i,p in enumerate(g,1):node(d,p,f"{'AB'[gi]}{i}",CLRS[gi],17)
            tracker(d,["separate vertices","reveal cross edges","final K₂[3]"][st-1],2,6,f"{shown} (A–B family)","witness found" if st==3 else "constructing")
        frames += [im,im]
    make_animation("animation_01_k2_construction.mp4",frames)
def anim_k3():
    frames=[]; gscent=[(750,255),(540,530),(960,530)]
    for st in range(5):
        im,d=canvas("Animation 2. Constructing K₃[3]"); gs=[cluster(d,c,"ABC"[i],3,CLRS[i]) for i,c in enumerate(gscent)]
        pairs=[(0,1),(1,2),(0,2)][:max(0,st-1)]; joins(d,gs,pairs,"#46515b",3)
        for gi,g in enumerate(gs):
            for i,p in enumerate(g,1):node(d,p,f"{'ABC'[gi]}{i}",CLRS[gi],17)
        names=["clusters ready","join A–B","join B–C","join A–C","final K₃[3]"]; tracker(d,names[st],3,9,9*min(3,max(0,st-1)),"complete" if st==4 else "constructing")
        frames += [im,im]
    make_animation("animation_02_k3_construction.mp4",frames)
def anim_search():
    frames=[]
    for st in range(4):
        im,d=canvas("Animation 3. Looking for K₃[3] inside G", "An educational search reveal: valid witnesses need all required cross-cluster edges.")
        bg,gs=host_layout(); rng=random.Random(771); allp=bg+[p for g in gs for p in g]
        for i,j in combinations(range(len(allp)),2):
            if rng.random()<.105: edge(d,allp[i],allp[j],"#c9c7c1",2)
        if st>=2: joins(d,gs,[(0,1),(0,2),(1,2)],"#33424f",4)
        for p in bg: node(d,p,"","#aeb3b6",10)
        for gi,g in enumerate(gs):
            for i,p in enumerate(g,1): node(d,p,f"{'ABC'[gi]}{i}" if st>=1 else "",CLRS[gi] if st>=1 else "#aeb3b6",16)
        if st==1: txt(d,(750,185),"candidate clusters",24,BLUE,True,anchor="mm")
        if st==2: txt(d,(750,185),"check every cross-cluster pair",24,BLUE,True,anchor="mm")
        if st==3: txt(d,(750,185),"all 27 required cross edges present",24,BLUE,True,anchor="mm")
        tracker(d,["host graph","candidates","checking","witness found"][st],3,24,27,"witness found" if st==3 else "searching")
        frames += [im,im]
    make_animation("animation_03_host_search.mp4",frames)

if __name__ == "__main__":
    fig01(); fig02(); fig03(); fig04(); fig05(); fig06(); fig07()
    rows=experiments()
    with open(OUT/"experiment_tracker.csv","w",newline="") as fh:
        w=csv.writer(fh, lineterminator="\n"); w.writerow(["scenario","n","edges","triangles","gamma_hat=triangles/n^3","planted_k","largest_k_reported","search_type"]); w.writerows(rows)
    fig08(rows); anim_k2(); anim_k3(); anim_search()
    print("Generated 8 PNG figures, 3 MP4 animations, and experiment_tracker.csv.")
