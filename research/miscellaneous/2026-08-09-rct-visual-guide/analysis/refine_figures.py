"""Original editable SVG compositions and an aggregate-data forest plot.
No reference artwork is copied, traced, embedded or transformed.
"""
from pathlib import Path
import json, math
from html import escape

ROOT=Path(__file__).resolve().parents[1]
INK='#344750';BLUE='#5b8fae';GREEN='#78a783';PEACH='#d8a687'
def text(x,y,label,size=18,anchor='start',color=INK,weight='400'):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{escape(str(label))}</text>'
def box(x,y,w,h,fill='#eff7fc',stroke='#c5d8e2',rx=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'
def path(d,stroke=INK,fill='none',width=2,dash=''):
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
def circle(x,y,r,fill,stroke=INK):return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}"/>'
def person(x,y,shirt=GREEN,skin='#bf886e',scale=1,clipboard=False):
    b=path('M-18 94L-22 143H-8L0 103L8 143H22L18 94',fill='#425a65')
    b+=path('M-22 143L-30 150H-7V142M8 142V150H32L22 143',fill=INK)
    b+=path('M-18 43Q-35 49-33 83L-20 85L-15 65M18 43Q36 49 35 78L22 84L15 64',fill=skin)
    b+=path('M-13 39Q0 33 13 39L23 96H-23Z',fill=shirt)
    b+=path('M-7 35V44L0 52L7 44V35',fill=skin)
    b+=circle(0,20,18,skin)+path('M-18 20Q-23-2-1-2Q23-1 18 20L12 10Q-3 14-12 9L-16 23',fill='#344750')
    b+=path('M-8 23H-6M6 23H8M-5 31Q0 35 6 30',width=1.3)
    if clipboard:
        b+=box(-18,59,41,47,'#fffefa','#6d8189',4)+path('M-9 56H12V63H-9Z',fill='#c8d8d7')+path('M-10 73H13M-10 82H13M-10 91H5',stroke='#8da1a8',width=1.3)
        b+=path('M-31 77Q-14 92-6 83M34 76Q22 91 14 86',stroke=skin,width=8)
    return f'<g transform="translate({x} {y}) scale({scale})">{b}</g>'
def baby(x,y,scale=1,blanket='#c6dcdf'):
    b=path('M-31 0Q0-21 32 1L44 34Q10 57-27 32Z',fill=blanket)
    b+=circle(9,-3,15,'#e0b59b')+path('M-3-9Q9-21 21-8',stroke='#7b5b50',width=2)
    b+=path('M3-3H5M14-3H16M7 5Q11 8 15 5',width=1.1)+path('M-20 8L27 27M-16 27L7 12',stroke='#849fa7',width=1.2)
    return f'<g transform="translate({x} {y}) scale({scale})">{b}</g>'
def microscope(x,y,s=1):
    b=path('M0 75H100L90 62H15Z',fill='#bdd7c2')+path('M40 63Q94 48 65 8L51 17Q76 42 42 49Z',fill='#7da78b')
    b+=path('M36 0L52-9L69 17L53 28Z',fill='#d4e1e5')+path('M24 45H72M31 48V56',width=4)+circle(69,33,9,'#f9f7f1')
    return f'<g transform="translate({x} {y}) scale({s})">{b}</g>'
def write(name,title,desc,body,w=1000,h=340):
    s=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><rect width="100%" height="100%" fill="#fffefa"/><g font-family="Inter,Arial,sans-serif" stroke-width="1.5">{body}</g></svg>'
    (ROOT/'figures'/name).write_text(s,encoding='utf-8')

def main():
    data=json.loads((ROOT/'data/derived_summary.json').read_text(encoding='utf-8'))
    # A new clinical arrangement: parent, infant scale and screening clipboard.
    b=path('M25 242Q-8 81 120 57Q271-4 366 134Q421 296 234 285H70Z',fill='#edf7ef',stroke='none')
    b+=person(84,88,shirt='#9bb5d1',skin='#bd826b',scale=1.1)
    b+=person(300,63,shirt='#f6f5ed',skin='#d8ab8b',scale=1.15,clipboard=True)
    b+=box(130,185,121,23,'#dce9ec','#7d9ca6',5)+path('M144 208V259M237 208V259M133 259H249',stroke='#718f98',width=3)
    b+=path('M142 182Q184 201 240 180L228 153H154Z',fill='#e6eef2',stroke='#8ca5b3')+baby(185,150,.76)
    b+=box(170,210,43,24,'#fffefa','#a4bac1',4)+text(191,227,'g',14,'middle')
    b+=text(36,313,'Caregiver consent + clinical screening',17)
    b+=text(435,43,'Eligibility is a boundary of the evidence',24,weight='600')
    for y,label,detail,fill,stroke in [(70,'Birth weight ≥ 2,000 g','Published threshold; cleaned BWT used in analysis','#eff7fc','#c5d8e2'),(147,'Gestation ≥ 35 weeks','The result does not automatically transport to very preterm infants','#fff5ed','#e2cdbb'),(224,'Feeding established; no signs of illness','Enrollment window extended to days 1–4','#f0f9f1','#c5dac8')]:
        b+=box(435,y,535,66,fill,stroke)+text(454,y+26,label,19,weight='600')+text(454,y+49,detail,14)
    write('neonatal-screening.svg','Eligibility assessment in the infant synbiotic trial','Original conceptual illustration of caregiver, infant scale and clinician; published eligibility boundaries are labeled.',b,h=340)
    # Balanced oral arms, then an age-scaled follow-up timeline.
    b=text(28,30,'Matched allocation procedures; different assigned content',23,weight='600')
    for x,name,content,fill,stroke,color in [(25,'Synbiotic','L. plantarum + FOS','#eff7fc','#bfd6e4',BLUE),(515,'Placebo','Maltodextrin','#fff5ed','#e2cdbb',PEACH)]:
        b+=box(x,50,460,144,fill,stroke)+text(x+22,80,name,21,weight='600')+text(x+22,106,content,17)
        b+=box(x+23,122,66,46,'#fffefa','#9aafb6',6)+text(x+56,150,'CODE',12,'middle')
        b+=baby(x+348,129,.9,blanket='#d7e8dc')
        b+=path(f'M{x+141} 134H{x+231}M{x+148} 121V147M{x+162} 124H{x+216}V144H{x+162}Z M{x+218} 133H{x+252}',stroke=color,width=3)
        b+=path(f'M{x+254} 133L{x+272} 133M{x+266} 127L{x+273} 133L{x+266} 139',stroke=color)
        b+=text(x+145,174,'Oral, needle-free delivery',14)
    x=lambda age:65+age/60*870
    b+=text(28,227,'Age scale: treatment starts after enrollment, surveillance continues to day 60',17)
    b+=path(f'M{x(0)} 282H{x(60)}',stroke='#91a4a2',width=3)
    b+=box(x(1),265,x(10)-x(1),30,'#d7e9d8','#a5c6aa',5)
    for age in [0,10,30,60]:b+=path(f'M{x(age)} 277V291',width=1.5)+text(x(age),314,str(age)+' d',16,'middle')
    b+=text(65,252,'Enroll 1–4 d → seven daily doses',15)+text(580,252,'Daily monitoring / clinical referral',15)
    b+=text(28,347,'Schematic kits reveal content for teaching; actual allocation was masked. Dosing ends around age 7–10 days.',14)
    write('oral-trial-timeline.svg','Two oral treatment strategies and the day-60 follow-up horizon','Balanced panels depict oral delivery without infusion. A common age scale shows an enrollment window and seven doses, followed by surveillance.',b,h=370)
    # Microbiology illustration is deliberately not an invented diagnostic algorithm.
    b=path('M23 239Q0 72 155 54Q331 17 338 224Q330 304 178 290Z',fill='#f0f7f1',stroke='none')+person(100,94,shirt='#f6f5ed',skin='#a96f59',scale=1.1,clipboard=True)
    b+=box(162,218,182,16,'#e5e8e5','#acbbb9',5)+path('M177 234V291M329 234V291',stroke='#7b9299',width=3)+microscope(204,137,.9)
    b+=text(26,322,'Clinical assessment + microbiology where indicated',15)
    b+=text(385,42,'Endpoint categories in the released data',23,weight='600')
    for y,k,title,detail,fill in [(65,'CULTPOS','A · Culture-positive','27 placebo / 6 synbiotic','#eff7fc'),(140,'SEPSISB','B · Culture-negative clinical sepsis','36 placebo / 19 synbiotic','#fff5ed'),(215,'SEPSISC','C · pSBI–LRTI','139 placebo / 92 synbiotic','#f6f2fb')]:
        total=data['outcomes'][k]['events0']+data['outcomes'][k]['events1']
        b+=box(385,y,585,64,fill,'#d3dcd5')+text(405,y+25,title,18,weight='600')+text(405,y+49,detail,15)+text(941,y+42,total,27,'end',color='#4d6a55',weight='600')
    b+=text(385,316,'The additional 94 culture-negative evaluations have no dedicated release flag.',14)
    write('clinical-classification.svg','Clinical and laboratory context for corrected sepsis classifications','Original scientist illustration beside the three mutually exclusive released sepsis indicators. This is a classification summary, not a diagnostic decision algorithm.',b,h=345)
    # Infant-specific counterfactual, without claiming an individual effect was observed.
    b=box(20,40,300,175,'#eff7fc','#c5d8e2')+box(580,40,300,175,'#fff5ed','#e2cdbb')
    b+=baby(450,111,1.1)+text(450,190,'One synthetic infant',18,'middle',weight='600')
    b+=path('M408 110H334M491 110H568',dash='6 5')
    b+=text(45,73,'Assigned synbiotic: Z = 1',18,weight='600')+text(45,110,'Yᵢ(1) = adverse event by day 60?',18)+text(45,152,'Observed only if assigned this strategy',14)
    b+=text(605,73,'Assigned placebo: Z = 0',18,weight='600')+text(605,110,'Yᵢ(0) = adverse event by day 60?',18)+text(605,152,'The alternative outcome stays unobserved',14)
    b+=text(450,248,'Randomization compares groups. It never reveals both outcomes for one infant.',17,'middle')
    write('infant-counterfactual.svg','One infant, two potential assignment outcomes','A conceptual, synthetic infant links two assignment worlds. The two possible outcomes cannot both be observed for that infant.',b,w=900,h=275)
    # Reproducibility composition with factual estimate read from aggregate output.
    b=text(25,30,'A traceable path from original records to a scientific claim',23,weight='600')
    labels=[('Original CSV','Preserve bytes','#f5f6f5'),('Dictionary','Define endpoints','#fff5ed'),('Statistical code','Deterministic calculation','#eff7fc'),('Aggregate result',f'RR {data["primary"]["rr"]:.4f}','#f0f9f1'),('Publication check','Keep discrepancies visible','#f6f2fb')]
    for i,(title,detail,color) in enumerate(labels):
        xx=25+i*195;b+=box(xx,65,170,115,color,'#d5dcd4')+text(xx+85,96,title,17,'middle',weight='600')+text(xx+85,159,detail,12,'middle')
        if i<4:b+=path(f'M{xx+174} 120H{xx+188}M{xx+183} 115L{xx+189} 120L{xx+183} 125',stroke='#6f8a7b')
        if i==0:
            for yy in [112,122,132]:b+=path(f'M{xx+35} {yy}H{xx+136}',stroke='#9eacb1',width=1.5)
        elif i==1:b+=path(f'M{xx+42} 113H{xx+128}M{xx+42} 128H{xx+112}',stroke='#c19c7a')
        elif i==2:b+=text(xx+85,130,'{ a, c, n₁, n₀ }',17,'middle',color='#4e778b')
        elif i==3:b+=path(f'M{xx+31} 121H{xx+133}',stroke='#7ca087',width=3)+circle(xx+76,121,5,'#527f5e')
        else:b+=path(f'M{xx+51} 113L{xx+63} 125L{xx+86} 105M{xx+98} 112H{xx+135}M{xx+98} 126H{xx+127}',stroke='#8d79a0')
    b+=person(220,214,shirt='#a4cbb1',skin='#c89174',scale=.65,clipboard=True)+person(765,214,shirt='#b8cddd',skin='#946750',scale=.65,clipboard=True)
    b+=text(490,256,'Recheck the computation',20,'middle',weight='600')+text(490,286,'Preserve disagreement as evidence, rather than adjust records to match.',15,'middle')
    write('reproducibility-chain.svg','Reproducible analytical workflow','Original data, dictionary, statistical code, aggregate estimate and publication comparison are connected; two conceptual analysts inspect the result.',b,h=325)
    # Refined empirical forest: event columns, endcaps and explicit effect direction.
    keys=['SEPSISDEATH','SEPSIS','CULTPOS','SEPSISB','SEPSISC','DEATH','DIARRHEA','LOCINF','OMPHA']
    labels={'SEPSISDEATH':'Sepsis or death','SEPSIS':'Sepsis A+B+C','CULTPOS':'Culture-positive A','SEPSISB':'Culture-negative B','SEPSISC':'pSBI–LRTI C','DEATH':'Death','DIARRHEA':'Diarrhoea','LOCINF':'Local infection','OMPHA':'Omphalitis'}
    x=lambda rr:510+(math.log(rr)-math.log(.03))/(math.log(8)-math.log(.03))*330
    b=text(18,27,'Endpoint',16,weight='600')+text(270,27,'Events / 2,278',15,weight='600')+text(270,49,'Placebo  |  Synbiotic',13)+text(860,27,'RR [95% CI]',16,weight='600')
    for i in range(len(keys)):
        if i%2==0:b+=box(8,61+i*43,1050,39,'#f5f7f3','none',5)
    b+=path(f'M{x(1)} 53V477',stroke='#8e9d98',dash='4 4')
    for i,key in enumerate(keys):
        r=data['outcomes'][key];yy=83+i*43
        lo,hi=r['rr_ci'];b+=text(20,yy+5,labels[key],17)+text(292,yy+5,f'{r["events0"]:>3}   |   {r["events1"]:>3}',17)
        color='#668d74' if key=='SEPSISDEATH' else '#5d89a1'
        b+=path(f'M{x(lo)} {yy}H{x(hi)}M{x(lo)} {yy-5}V{yy+5}M{x(hi)} {yy-5}V{yy+5}',stroke=color,width=2.5)+circle(x(r['rr']),yy,5.2,color,color)+text(860,yy+5,f'{r["rr"]:.2f} [{lo:.2f}, {hi:.2f}]',16)
    for v in [.05,.1,.25,.5,1,2,4,8]:b+=text(x(v),495,str(v),14,'middle')
    b+=text(510,524,'← Lower adverse-event risk',15)+text(846,549,'Higher adverse-event risk →',15,'end')+text(18,549,'Unadjusted, fixed-sample, pointwise intervals; no multiplicity correction.',14)
    write('secondary-forest.svg','Reproduced outcome risk ratios, event counts and pointwise 95% intervals','Common logarithmic RR scale, reference at one, interval endcaps and placebo/synbiotic event counts. Mortality uncertainty is wide.',b,w=1080,h=575)

if __name__=='__main__':main()
